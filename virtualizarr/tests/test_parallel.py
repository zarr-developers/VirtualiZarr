import gc
import multiprocessing as mp
import time
import tracemalloc

import pytest

from virtualizarr.parallel import (
    DaskDelayedExecutor,
    LithopsEagerFunctionExecutor,
    SerialExecutor,
    get_executor,
)
from virtualizarr.tests import requires_dask, requires_lithops


@pytest.mark.flaky
@requires_lithops
def test_lithops_executor_with_multiple_args():
    with LithopsEagerFunctionExecutor() as exec:
        results = exec.map(lambda x, y: x + y, (1, 2, 3), (4, 5, 6))

    assert tuple(results) == (5, 7, 9)


@pytest.mark.flaky
@requires_lithops
def test_lithops_executor_with_partial():
    from functools import partial

    inc = partial(lambda x, y: x + y, 1)

    with LithopsEagerFunctionExecutor() as exec:
        results = exec.map(inc, (1, 2, 3))

    assert tuple(results) == (2, 3, 4)


@pytest.mark.skipif(
    mp.get_start_method() != "fork",
    reason="Default multiprocessing start method is not 'fork'",
)
def test_get_executor_process_pool_mode():
    from concurrent.futures import ProcessPoolExecutor

    executor = get_executor(ProcessPoolExecutor)()

    assert isinstance(executor, ProcessPoolExecutor), "Expected a ProcessPoolExecutor"

    ctx = executor._mp_context

    assert ctx is not None, "Expected executor to have a multiprocessing context"
    assert ctx.get_start_method() == "forkserver"


def _make_executor(executor_cls):
    """Create a pytest param for an executor class with appropriate marks."""
    marks = {
        "DaskDelayedExecutor": [requires_dask],
        "LithopsEagerFunctionExecutor": [requires_lithops],
    }
    return pytest.param(
        executor_cls,
        id=executor_cls.__name__,
        marks=marks.get(executor_cls.__name__, []),
    )


ALL_CUSTOM_EXECUTORS = [
    _make_executor(SerialExecutor),
    _make_executor(DaskDelayedExecutor),
    _make_executor(LithopsEagerFunctionExecutor),
]


@pytest.mark.parametrize("executor_cls", ALL_CUSTOM_EXECUTORS)
class TestExecutorShutdown:
    def test_shutdown_clears_futures(self, executor_cls):
        """Internal _futures list should be empty after shutdown."""
        with executor_cls() as executor:
            executor.submit(lambda x: x * 2, 1)
            executor.submit(lambda x: x + 1, 2)
            assert len(executor._futures) == 2
            if executor_cls is LithopsEagerFunctionExecutor:
                # grab refs before they get cleared
                lithops_futures = list(executor.lithops_client.futures)
                assert len(lithops_futures) == 2

        assert len(executor._futures) == 0

        # Lithops-specific: verify lithops internal futures are also cleared
        if executor_cls is LithopsEagerFunctionExecutor:
            assert len(executor.lithops_client.futures) == 0
            assert all(f._call_output is None for f in lithops_futures)

        # Testing idempotency
        executor.shutdown()
        assert len(executor._futures) == 0


@requires_lithops
def test_lithops_executor_data_cleaner_disabled():
    """data_cleaner must be False to prevent atexit registration of lithops' clean method."""
    with LithopsEagerFunctionExecutor() as executor:
        assert executor.lithops_client.data_cleaner is False
