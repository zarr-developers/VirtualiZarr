from concurrent.futures import Executor, Future
from typing import Any, Callable, Literal, Optional

import xarray as xr

# TODO this entire module could ideally be upstreamed into xarray as part of https://github.com/pydata/xarray/pull/9932


def execute(
    func: Callable[[str], xr.Dataset],
    paths: list[str],
    parallel: Literal["lithops", "dask", False] | Executor,
) -> list[xr.Dataset]:
    """
    Map a function over a set of filepaths and execute it with one of a range of parallel executors.

    Parameters
    ----------
    paths
    func
    parallel
        Choice of how to parallelize execution. Passing False
    """
    executor: Executor = select_executor(parallel=parallel)

    # wait for all the serverless workers to finish, and send their resulting virtual datasets back to the client
    with executor() as exec:
        results = list(exec.map(func, paths))

    # if parallel == "dask":
    #     virtual_datasets = [open_(p, **kwargs) for p in paths1d]
    #     closers = [getattr_(ds, "_close") for ds in virtual_datasets]
    #     if preprocess is not None:
    #         virtual_datasets = [preprocess(ds) for ds in virtual_datasets]

    #     # calling compute here will return the datasets/file_objs lists,
    #     # the underlying datasets will still be stored as dask arrays
    #     virtual_datasets, closers = dask.compute(virtual_datasets, closers)
    # elif parallel == "lithops":

    # TODO add file closers

    return results


def select_executor(parallel: Literal["lithops", "dask", False] | Executor) -> Executor:
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
        raise ValueError(
            f"Unrecognized option for ``parallel`` kwarg to ``open_virtual_mfdataset``: {parallel}."
            "Expected one of ``'dask'``, ``'lithops'``, ``False``, or an instance of a subclass of ``concurrent.futures.Executor``."
        )

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
