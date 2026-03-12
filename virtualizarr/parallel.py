import inspect
import multiprocessing as mp
import warnings
from concurrent.futures import Executor, Future, ProcessPoolExecutor
from functools import partial
from typing import (
    Any,
    Callable,
    Generic,
    Iterable,
    Iterator,
    Literal,
    ParamSpec,
    TypeVar,
)

__all__ = [
    "SerialExecutor",
    "DaskDelayedExecutor",
    "LithopsEagerFunctionExecutor",
]


# TODO this entire module could ideally be upstreamed into xarray as part of https://github.com/pydata/xarray/pull/9932
# TODO the DaskDelayedExecutor class could ideally be upstreamed into dask
# TODO lithops should just not require a special wrapper class, see https://github.com/lithops-cloud/lithops/issues/1427


P = ParamSpec("P")
# Type variable for return type
T = TypeVar("T")


def get_executor(
    parallel: Literal["dask", "lithops", False] | type[Executor],
) -> Callable[..., Executor]:
    """Get a callable with a return type that follows the concurrent.futures.Executor ABC API."""

    if parallel == "dask":
        return DaskDelayedExecutor
    if parallel == "lithops":
        return LithopsEagerFunctionExecutor
    if parallel is False:
        return SerialExecutor
    if parallel is ProcessPoolExecutor:
        # TODO Once we drop support for python <3.14, we can remove this context
        # dance because from 3.14 onward, POSIX defaults to "forkserver" rather
        # than "fork".
        method = mp.get_context().get_start_method()
        context = mp.get_context("forkserver" if method == "fork" else method)
        return partial(ProcessPoolExecutor, mp_context=context)
    if inspect.isclass(parallel) and issubclass(parallel, Executor):
        return parallel

    raise ValueError(
        f"Invalid value for `parallel`: {parallel}.  Please supply "
        "either the string 'dask' or 'lithops', or a concrete subclass of "
        "concurrent.futures.Executor.  To obtain a serial executor, specify "
        "the boolean value `False`."
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
        self._futures.clear()


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
        self._futures.clear()


class LithopsEagerFunctionExecutor(Executor):
    """
    Lithops-based function executor that follows the [concurrent.futures.Executor][] API.

    Only required because lithops doesn't follow the [concurrent.futures.Executor][] API.
    See https://github.com/lithops-cloud/lithops/issues/1427.
    """

    class compatible_callable(Generic[P, T]):
        """Wraps a callable to make it fully compatible with Lithops.

        This wrapper deals with 2 oddities in Lithops:

        1. Use of `functools.partial`, which Lithops fails to recognize as being
           callable.  This is likely due to the builtin `partial` class using
           slots, which causes Lithops to not recognize the `__call__` method as
           a method.  See https://github.com/lithops-cloud/lithops/issues/1428.
        2. Use of generic function wrappers that define generic `args` and
           `kwargs` parameters.  In this case, because of the way Lithops
           inspects function signatures to determine how to pass arguments, it
           does not properly "spread" arguments as normally expected.  Instead,
           it collects all positional arguments and associates them with the
           keyword argument `"args"`, which is utterly unhelpful.
        """

        def __init__(self, f: Callable[P, T]):
            self.f = f

        def __call__(self, *args: P.args, **kwargs: P.kwargs) -> T:
            if not args and "args" in kwargs:
                # Lithops collected all positional args into an "args" kwarg,
                # so we're undoing that nonsense here.
                args = kwargs.pop("args", ())  # type: ignore

            return self.f(*args, **kwargs)

    def __init__(self, **kwargs) -> None:
        import lithops  # type: ignore[import-untyped]

        # Create Lithops client with optional configuration
        self.lithops_client = lithops.FunctionExecutor(**kwargs).__enter__()

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
            lithops_future = self.lithops_client.call_async(
                LithopsEagerFunctionExecutor.compatible_callable(fn),
                *args,
                **kwargs,
            )

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

        Only needed because [lithops.executors.FunctionExecutor.map][lithops.executors.FunctionExecutor.map]
        returns futures, unlike [concurrent.futures.Executor.map][].

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
        fexec = self.lithops_client
        futures = fexec.map(
            LithopsEagerFunctionExecutor.compatible_callable(fn),
            list(zip(*iterables)),
        )

        return fexec.get_result(futures)  # type: ignore

    def shutdown(self, wait: bool = True, *, cancel_futures: bool = False) -> None:
        """
        Shutdown the executor.

        Parameters
        ----------
        wait
            Whether to wait for pending futures.
        """
        # Free cached results from lithops ResponseFuture objects before shutdown.
        # lithops.FunctionExecutor.futures is never cleared internally — each map()
        # call extends it with new ResponseFutures that cache deserialized results
        # in _call_output. Without this, memory accumulates across repeated calls.
        for f in self.lithops_client.futures:
            f._call_output = None
        self.lithops_client.futures.clear()
        self._futures.clear()
        self.lithops_client.__exit__(None, None, None)
