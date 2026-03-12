import multiprocessing as mp

import pytest

from virtualizarr.parallel import (
    DaskDelayedExecutor,
    LithopsEagerFunctionExecutor,
    SerialExecutor,
    get_executor,
)
from virtualizarr.tests import requires_dask, requires_lithops


@requires_lithops
def test_lithops_executor_with_multiple_args():
    with LithopsEagerFunctionExecutor() as exec:
        results = exec.map(lambda x, y: x + y, (1, 2, 3), (4, 5, 6))

    assert tuple(results) == (5, 7, 9)


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


ALL_EXECUTORS = [
    _make_executor(SerialExecutor),
    _make_executor(DaskDelayedExecutor),
    _make_executor(LithopsEagerFunctionExecutor),
]


@pytest.mark.parametrize("executor_cls", ALL_EXECUTORS)
class TestExecutorShutdown:
    def test_shutdown_clears_futures(self, executor_cls):
        with executor_cls() as executor:
            executor.submit(lambda: 42)
            executor.submit(lambda: 99)
            assert len(executor._futures) == 2

        assert len(executor._futures) == 0

    def test_shutdown_via_context_manager(self, executor_cls):
        with executor_cls() as executor:
            executor.submit(lambda: 42)
            assert len(executor._futures) == 1

        assert len(executor._futures) == 0

    def test_shutdown_idempotent(self, executor_cls):
        executor = executor_cls()
        executor.submit(lambda: 1)
        executor.shutdown()
        executor.shutdown()
        assert len(executor._futures) == 0


@requires_lithops
class TestLithopsExecutorShutdownSpecific:
    def test_shutdown_clears_lithops_client_futures(self):
        executor = LithopsEagerFunctionExecutor()
        executor.submit(lambda: 42)

        executor.shutdown()
        assert len(executor.lithops_client.futures) == 0

    def test_shutdown_clears_lithops_cached_results(self):
        """Verify that shutdown clears _call_output on lithops ResponseFutures."""
        with LithopsEagerFunctionExecutor() as executor:
            executor.map(lambda x: x * 2, (1, 2, 3))
            lithops_futures = list(executor.lithops_client.futures)
            assert len(lithops_futures) > 0

        # After shutdown, lithops futures list should be cleared
        assert len(executor.lithops_client.futures) == 0
