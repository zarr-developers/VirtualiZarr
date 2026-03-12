import gc
import multiprocessing as mp
import weakref

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


@requires_lithops
class TestLithopsExecutorShutdown:
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
class TestExecutorMemory:
    def test_executor_does_not_leak_after_context_manager(self, executor_cls):
        """Executor and its futures should be GC-collectable after the with block."""

        with executor_cls() as executor:
            # Use map() since lithops call_async requires a data argument
            list(executor.map(lambda x: x * 2, range(5)))
            ref = weakref.ref(executor)

        # Drop the only local reference to the executor
        del executor
        gc.collect()

        assert ref() is None, (
            f"{executor_cls.__name__} was not garbage collected after shutdown"
        )

    def test_repeated_executor_use_does_not_grow_memory(self, executor_cls):
        """Memory should not grow when creating and destroying executors repeatedly."""
        import tracemalloc

        def _run_once():
            with executor_cls() as executor:
                # Use map() to produce non-trivial results
                return list(executor.map(lambda x: list(range(10_000)), range(5)))

        # Warm up (first run may allocate caches, import modules, etc.)
        _run_once()
        gc.collect()

        # Measure baseline: peak memory from a single run
        tracemalloc.start()
        _run_once()
        gc.collect()
        _, baseline_peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        # Now run many iterations and check peak doesn't grow
        tracemalloc.start()
        n_iterations = 10
        for _ in range(n_iterations):
            _run_once()
            gc.collect()
        _, multi_peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        # If memory leaks, peak will scale with n_iterations.
        # Allow 1.2x the single-run peak to account for GC timing jitter.
        assert multi_peak < 1.2 * baseline_peak, (
            f"{executor_cls.__name__} leaked memory: single run peak "
            f"{baseline_peak / 1024:.0f} KB, {n_iterations} runs peak "
            f"{multi_peak / 1024:.0f} KB"
        )
