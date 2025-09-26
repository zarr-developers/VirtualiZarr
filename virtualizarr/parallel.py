import inspect
import warnings
from concurrent.futures import Executor, Future
from typing import Any, Callable, Iterable, Iterator, Literal, TypeVar

__all__ = [
    "SerialExecutor",
    "DaskDelayedExecutor",
    "LithopsEagerFunctionExecutor",
]


# TODO this entire module could ideally be upstreamed into xarray as part of https://github.com/pydata/xarray/pull/9932
# TODO the DaskDelayedExecutor class could ideally be upstreamed into dask
# TODO lithops should just not require a special wrapper class, see https://github.com/lithops-cloud/lithops/issues/1427


# Type variable for return type
T = TypeVar("T")


def get_executor(
    parallel: Literal["dask", "lithops"] | type[Executor] | Literal[False],
) -> type[Executor]:
    """Get an executor that follows the concurrent.futures.Executor ABC API."""

    if parallel == "dask":
        return DaskDelayedExecutor
    elif parallel == "lithops":
        return LithopsEagerFunctionExecutor
    elif parallel is False:
        return SerialExecutor
    elif inspect.isclass(parallel) and issubclass(parallel, Executor):
        return parallel
    else:
        raise ValueError(
            f"Unrecognized argument to ``parallel``: {parallel}"
            "Please supply either ``'dask'``, ``'lithops'``, ``False``, or a concrete subclass of ``concurrent.futures.Executor``."
        )


class SerialExecutor(Executor):
    """
    A custom Executor that runs tasks sequentially, mimicking the
    concurrent.futures.Executor interface. Useful as a default and for debugging.
    """

    def __init__(self) -> None:
        # Track submitted futures to maintain interface compatibility
        self._futures: list[Future] = []

    def submit(self, fn: Callable[..., T], /, *args: Any, **kwargs: Any) -> Future[T]:
        """
        Submit a callable to be executed.

        Unlike parallel executors, this runs the task immediately and sequentially.

        Parameters
        ----------
        fn
            The callable to execute
        *args
            Positional arguments for the callable
        **kwargs
            Keyword arguments for the callable

        Returns
        -------
        A Future representing the result of the execution
        """
        # Create a future to maintain interface compatibility
        future: Future = Future()

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
        self,
        fn: Callable[..., T],
        *iterables: Iterable[Any],
        timeout: float | None = None,
        chunksize: int = 1,
    ) -> Iterator[T]:
        """
        Execute a function over an iterable sequentially.

        Parameters
        ----------
        fn
            Function to apply to each item
        *iterables
            Iterables to process
        timeout
            Optional timeout (ignored in serial execution)

        Returns
        -------
        Generator of results
        """
        return map(fn, *iterables)

    def shutdown(self, wait: bool = True, *, cancel_futures: bool = False) -> None:
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
    An Executor that uses [dask.delayed][dask.delayed.delayed] for parallel computation.

    This executor mimics the concurrent.futures.Executor interface but uses Dask's delayed computation model.
    """

    def __init__(self) -> None:
        """Initialize the Dask Delayed Executor."""

        # Track submitted futures
        self._futures: list[Future] = []

    def submit(self, fn: Callable[..., T], /, *args: Any, **kwargs: Any) -> Future[T]:
        """
        Submit a task to be computed with [dask.delayed][dask.delayed.delayed].

        Parameters
        ----------
        fn
            The callable to execute
        *args
            Positional arguments for the callable
        **kwargs
            Keyword arguments for the callable

        Returns
        -------
        A Future representing the result of the execution
        """
        import dask  # type: ignore[import-untyped]

        # Create a delayed computation
        delayed_task = dask.delayed(fn)(*args, **kwargs)

        # Create a concurrent.futures Future to maintain interface compatibility
        future: Future = Future()

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
        self,
        fn: Callable[..., T],
        *iterables: Iterable[Any],
        timeout: float | None = None,
        chunksize: int = 1,
    ) -> Iterator[T]:
        """
        Apply a function to an iterable using [dask.delayed][dask.delayed.delayed].

        Parameters
        ----------
        fn
            Function to apply to each item
        *iterables
            Iterables to process
        timeout
            Optional timeout (ignored in serial execution)

        Returns
        -------
        Generator of results
        """
        import dask  # type: ignore[import-untyped]

        if timeout is not None:
            warnings.warn("Timeout parameter is not directly supported by Dask delayed")

        # Create delayed computations for each item
        delayed_tasks = [dask.delayed(fn)(*items) for items in zip(*iterables)]

        # Compute all tasks
        return iter(dask.compute(*delayed_tasks))

    def shutdown(self, wait: bool = True, *, cancel_futures: bool = False) -> None:
        """
        Shutdown the executor

        Parameters
        ----------
        wait
            Whether to wait for pending futures (always True for serial executor))
        """
        # For Dask.delayed, shutdown is essentially a no-op
        pass


class LithopsEagerFunctionExecutor(Executor):
    """
    Lithops-based function executor which follows the [concurrent.futures.Executor][] API.

    Only required because lithops doesn't follow the [concurrent.futures.Executor][] API, see https://github.com/lithops-cloud/lithops/issues/1427.
    """

    def __init__(self, **kwargs) -> None:
        import lithops  # type: ignore[import-untyped]

        # Create Lithops client with optional configuration
        self.lithops_client = lithops.FunctionExecutor(**kwargs)

        # Track submitted futures
        self._futures: list[Future] = []

    def submit(self, fn: Callable[..., T], /, *args: Any, **kwargs: Any) -> Future[T]:
        """
        Submit a task to be computed using lithops.

        Parameters
        ----------
        fn
            The callable to execute
        *args
            Positional arguments for the callable
        **kwargs
            Keyword arguments for the callable

        Returns
        -------
        A concurrent.futures.Future representing the result of the execution
        """

        # Create a concurrent.futures Future to maintain interface compatibility
        future: Future = Future()

        try:
            # Submit to Lithops
            lithops_future = self.lithops_client.call_async(fn, *args, **kwargs)

            # Add a callback to set the result or exception
            def _on_done(lithops_result):
                try:
                    result = lithops_result.result()
                    future.set_result(result)
                except Exception as e:
                    future.set_exception(e)

            # Register the callback
            lithops_future.add_done_callback(_on_done)
        except Exception as e:
            # If submission fails, set exception immediately
            future.set_exception(e)

        # Track the future
        self._futures.append(future)

        return future

    def map(
        self,
        fn: Callable[..., T],
        *iterables: Iterable[Any],
        timeout: float | None = None,
        chunksize: int = 1,
    ) -> Iterator[T]:
        """
        Apply a function to an iterable using lithops.

        Only needed because [lithops.executors.FunctionExecutor.map][lithops.executors.FunctionExecutor.map] returns futures, unlike [concurrent.futures.Executor.map][].

        Parameters
        ----------
        fn
            Function to apply to each item
        *iterables
            Iterables to process
        timeout
            Optional timeout (ignored in serial execution)

        Returns
        -------
        Generator of results
        """
        import lithops  # type: ignore[import-untyped]

        fexec = lithops.FunctionExecutor()

        futures = fexec.map(fn, *iterables)
        results = fexec.get_result(futures)

        return results

    def shutdown(self, wait: bool = True, *, cancel_futures: bool = False) -> None:
        """
        Shutdown the executor.

        Parameters
        ----------
        wait
            Whether to wait for pending futures.
        """
        # Should this call lithops .clean() method?
        pass
