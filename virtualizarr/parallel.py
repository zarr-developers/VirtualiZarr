import warnings
from concurrent.futures import Executor, Future
from typing import Any, Callable, Optional

# TODO this entire module could ideally be upstreamed into xarray as part of https://github.com/pydata/xarray/pull/9932
# TODO the DaskDelayedExecutor class could ideally be upstreamed into dask


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


class DaskDelayedExecutor(Executor):
    """
    An Executor that uses dask.delayed for parallel computation.

    This executor mimics the concurrent.futures.Executor interface but uses Dask's delayed computation model.
    """

    def __init__(self):
        """Initialize the Dask Delayed Executor."""

        # Track submitted futures
        self._futures = []

    def submit(self, fn: Callable, *args: Any, **kwargs: Any) -> Future:
        """
        Submit a task to be computed with dask.delayed.

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
        import dask

        # Create a delayed computation
        delayed_task = dask.delayed(fn)(*args, **kwargs)

        # Create a concurrent.futures Future to maintain interface compatibility
        future = Future()

        try:
            # Compute the result
            result = delayed_task.compute()

            # Set the result on the future
            future.set_result(result)
        except Exception as e:
            # Set any exception on the future
            future.set_exception(e)

        # Track the future
        self._futures.append(future)

        return future

    def map(
        self, fn: Callable, *iterables: Any, timeout: Optional[float] = None
    ) -> Any:
        """
        Apply a function to an iterable using dask.delayed.

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
        import dask

        if timeout is not None:
            warnings.warn("Timeout parameter is not directly supported by Dask delayed")

        # Create delayed computations for each item
        delayed_tasks = [dask.delayed(fn)(*items) for items in zip(*iterables)]

        # Compute all tasks
        return list(dask.compute(*delayed_tasks))

    def shutdown(self, wait: bool = True) -> None:
        """
        Shutdown the executor

        Parameters
        ----------
        wait
            Whether to wait for pending futures (always True for serial executor))
        """
        # For Dask.delayed, shutdown is essentially a no-op
        pass
