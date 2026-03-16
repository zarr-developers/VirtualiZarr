import multiprocessing as mp

import pytest

from virtualizarr.parallel import LithopsEagerFunctionExecutor, get_executor
from virtualizarr.tests import requires_lithops


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
