from concurrent.futures import Executor, Future
from typing import Any, Callable, Literal, Optional

# TODO this entire module could ideally be upstreamed into xarray as part of https://github.com/pydata/xarray/pull/9932


def get_executor(parallel: Literal[False] | Executor) -> Executor:
    """Choose from a range of parallel executors, or pass your own."""

    executor: Executor
    if parallel == "dask":
        from dask.distributed import Client

        executor = Client
    elif parallel == "lithops":
        import lithops

        # TODO use RetryingFunctionExecutor instead?
        # TODO what's the easiest way to pass the lithops config in?
        executor = lithops.FunctionExecutor
    elif isinstance(parallel, Executor):
        executor = parallel
    elif parallel is False:
        executor = SerialExecutor
        # TODO change the default to use a ThreadPoolExecutor instead?
    else:
        raise ValueError(f"Unrecognized option for ``parallel`` kwarg: {parallel}.")

    return executor


class SerialExecutor(Executor):
    """
    A custom Executor that runs tasks sequentially, mimicking the
    concurrent.futures.Executor interface. Useful as a default and for debugging.
    """

    def __init__(self):
        # Track submitted futures to maintain interface compatibility
        self._futures = []

    def submit(self, fn: Callable, *args: Any, **kwargs: Any) -> Future:
        """
        Submit a callable to be executed.

        Unlike parallel executors, this runs the task immediately and sequentially.

        Parameters
        ----------
        fn
            The callable to execute
        args
            Positional arguments for the callable
        kwargs
            Keyword arguments for the callable

        Returns
        -------
        A Future representing the result of the execution
        """
        # Create a future to maintain interface compatibility
        future = Future()

        try:
            # Execute the function immediately
            result = fn(*args, **kwargs)

            # Set the result of the future
            future.set_result(result)
        except Exception as e:
            # If an exception occurs, set it on the future
            future.set_exception(e)

        # Keep track of futures for potential cleanup
        self._futures.append(future)

        return future

    def map(
        self, fn: Callable, *iterables: Any, timeout: Optional[float] = None
    ) -> Any:
        """
        Execute a function over an iterable sequentially.

        Parameters
        ----------
        fn
            Function to apply to each item
        iterables
            Iterables to process
        timeout
            Optional timeout (ignored in serial execution)

        Returns
        -------
        Generator of results
        """
        return map(fn, *iterables)

    def shutdown(self, wait: bool = True) -> None:
        """
        Shutdown the executor.

        Parameters
        ----------
        wait
            Whether to wait for pending futures (always True for serial executor)
        """
        # In a serial executor, shutdown is a no-op
        pass
