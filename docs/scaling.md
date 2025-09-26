# Scaling

This page explains how to scale up your usage of VirtualiZarr to cloud-optimize large numbers of files.

## Pre-requisites

Before you attempt to use VirtualiZarr on a large number of files at once, you should check that you can successfully use the library on a small subset of your data.

In particular, you should check that:

- You can call [`open_virtual_dataset`][virtualizarr.open_virtual_dataset] on one of your files, which requires there to be a parser which can interpret that file format.
- After calling [`open_virtual_dataset`][virtualizarr.open_virtual_dataset] on a few files making up a representative subset of your data, you can concatenate them into one logical datacube without errors (see the [FAQ](faq.md#can-my-specific-data-be-virtualized) for possible reasons for errors at this stage).
- You can serialize those virtual references to some format (e.g. Kerchunk/Icechunk) and read the data back.
- The data you read back is exactly what you would have expected to get if you read the data from the original files.

If you don't do these checks now, you might find that you deploy a large amount of resources to run VirtualiZarr on many files, only to hit a problem that you could have found much earlier.

## Strategy

### The need for parallelization

VirtualiZarr is a tool designed for taking a large number of slow-to-access files (i.e. non-cloud-optimized data) and creating a way to make all subsequent accesses much faster (i.e. a cloud-optimized datacube).

Running [`open_virtual_dataset`][virtualizarr.open_virtual_dataset] on just one file can take a while (seconds to minutes), because for data in object storage, fetching just the metadata can be almost as time-consuming as fetching the actual data.
(For a full explanation as to why [see this article](https://earthmover.io/blog/fundamentals-what-is-cloud-optimized-scientific-data)).
In some cases we may find it's easiest to load basically the entire contents of the file in order to virtualize it.

Therefore we should expect that running VirtualiZarr on all our data files will take a long time - we are paying this cost once up front so that our users do not have to pay it again on subsequent data accesses.

However, the [`open_virtual_dataset`][virtualizarr.open_virtual_dataset] calls for each file are completely independent, meaning that part of the computation is "embarrassingly parallelizable".

### Map-reduce

The problem of scaling VirtualiZarr is an example of a classic map-reduce problem, with two parts:

1. We first must apply the [`open_virtual_dataset`][virtualizarr.open_virtual_dataset] function over every file we want to virtualize. This is the map step, and can be parallelized.
2. Then we must take all the resultant virtual datasets (one per file), and combine them together into one final virtual dataset. This is the reduce step.

Finally we write this single virtual dataset to some persistent format.
We have already reduced the data, so this step is a third step, the serialization step.

In our case the amount of data being reduced is fairly small - each virtual dataset is hopefully only a few kBs in memory, small enough to send over the network.
Even a million such virtual datasets together would only require a few GB of RAM in total to hold in memory at once.
This means that as long as we can get all the virtual datasets to be sent back successfully, the reduce step can generally be performed in memory on a single small machine, such as a laptop.
This avoids the need for more complicated parallelization strategies such as a tree-reduce.

## Parallelization Approaches

There are two ways you can implement a map-reduce approach to virtualization in your code.
The first is to write it yourself, and the second is to use [`open_virtual_mfdataset`][virtualizarr.open_virtual_mfdataset].

### Manual parallelism

You are free to call [`open_virtual_dataset`][virtualizarr.open_virtual_dataset] on your various files however you like, using any method to apply them, including applying them in parallel.

For example you may want to parallelize using the [dask library](https://www.dask.org/), which you can do by wrapping each call using `dask.delayed` like this:

```python
import virtualizarr as vz
import dask

tasks = [dask.delayed(vz.open_virtual_dataset)(url,registry) for url in urls]
virtual_datasets = dask.compute(tasks)
```

This returns a list of virtual `xr.Dataset` objects, which you can then combine:

```python
import xarray as xr

combined_vds = xr.combine_by_coords(virtual_datasets)
```

### The `parallel` kwarg to `open_virtual_mfdataset`

Alternatively, you can use [virtualizarr.open_virtual_mfdataset][]'s `parallel` keyword argument.

This argument allows you to conveniently choose from a range of pre-defined parallel execution frameworks, or even pass your own executor.

The resulting code only takes one function call to generate virtual references in parallel and combine them into one virtual dataset.

```python
combined_vds = vz.open_virtual_mfdataset(urls, parallel=<choice_of_executor>)
```

VirtualiZarr's [`open_virtual_mfdataset`][virtualizarr.open_virtual_mfdataset] is designed to mimic the API of xarray's [`open_mfdataset`][xarray.open_mfdataset], and so accepts all the same keyword argument options for combining.

## Executors

VirtualiZarr comes with a small selection of executors you can choose from when using [`open_virtual_mfdataset`][virtualizarr.open_virtual_mfdataset], provided under the `virtualizarr.parallel` namespace.

!!! note
    If you prefer to do manual parallelism but would like to use one of these executors you can - just import the executor directly from the `virtualizarr.parallel` namespace and use its `.map` method.

### Serial

The simplest executor is the [`SerialExecutor`][virtualizarr.parallel.SerialExecutor], which executes all the [`open_virtual_dataset`][virtualizarr.open_virtual_dataset] calls in serial, not in parallel.
It is the default executor.

### Threads or Processes

One way to parallelize creating virtual references from a single machine is to across multiple threads or processes.
For this you can use the [`ThreadPoolExecutor`][concurrent.futures.ThreadPoolExecutor] or [`ProcessPoolExecutor`][concurrent.futures.ProcessPoolExecutor] class from the [`concurrent.futures`][] module in the python standard library.
You simply pass the executor class directly via the `parallel` kwarg to [`open_virtual_mfdataset`][virtualizarr.open_virtual_mfdataset].

```python
from concurrent.futures import ThreadPoolExecutor

combined_vds = vz.open_virtual_mfdataset(urls, registry=registry, parallel=ThreadPoolExecutor)
```

This can work well when virtualizing files in remote object storage because it parallelizes the issuing of HTTP GET requests for each file.

### Dask Delayed

You can parallelize using `dask.delayed` automatically by passing `parallel='dask'`.
This will select the [`DaskDelayedExecutor`][virtualizarr.parallel.DaskDelayedExecutor].

```python
combined_vds = vz.open_virtual_mfdataset(urls, registry=registry, parallel='dask')
```

This uses the same approach that [`open_mfdataset`][xarray.open_mfdataset] does when `parallel=True` is passed to it.
Using `dask.delayed` allows for parallelizing with any type of dask cluster, included a managed [Coiled](http://www.coiled.io) cluster.

### Lithops

As the map step is totally embarrassingly parallel, it can be performed entirely using serverless functions.
This approach allows for virtualizing N files in the same time it takes to virtualize 1 file, (assuming you can provision N concurrent serverless functions), avoiding the need to configure, scale, and shutdown a cluster.

You can parallelize VirtualiZarr serverlessly by using the [lithops](http://lithops-cloud.github.io) library.
Lithops can run on all the main cloud provider's serverless FaaS platforms.

To run on lithops you need to configure lithops for the relevant compute backend (e.g. AWS Lambda), build a runtime using Docker ([example Dockerfile](https://github.com/zarr-developers/VirtualiZarr/tree/develop/examples/oae/Dockerfile) with the required dependencies), and ensure the necessary cloud permissions to run are available.
Then you can use the [`LithopsEagerFunctionExecutor`][virtualizarr.parallel.LithopsEagerFunctionExecutor] simply via:

```python
combined_vds = vz.open_virtual_mfdataset(urls, registry=registry, parallel='lithops')
```

### Custom Executors

You can also define your own executor to run in some other way, for example on a different serverless platform such as [Modal](https://modal.com).

Your custom executor must inherit from the [`concurrent.futures.Executor`][] ABC, and must implement the `.map` method.

```python
from concurrent.futures import Executor

class CustomExecutor(Executor):
    def map(
        self,
        fn: Callable,
        *iterables: Iterable,
    ) -> Iterator:
        ...

combined_vds = vz.open_virtual_mfdataset(urls, registry=registry, parallel=CustomExecutor)
```

## Memory usage

For the virtualization to succeed you need to ensure that your available memory is not exceeded at any point.
There are 3 points at which this might happen:

1. While generating references
2. While combining references
3. While writing references

While generating references each worker calling [`open_virtual_dataset`][virtualizarr.open_virtual_dataset] needs to avoid running out of memory.
This primarily depends on how the file is read - see the section on [caching](#caching-remote-files) below.

The combine step happens back on the machine on which [`open_virtual_mfdataset`][virtualizarr.open_virtual_mfdataset] was called, so while combining references that machine must have enough memory to hold all the virtual references at once.
You can find the in-memory size of the references for a single virtual dataset by calling the [`.nbytes`][virtualizarr.accessor.VirtualiZarrDatasetAccessor.nbytes] accessor method on it (not to be confused with the [`.nbytes`][xarray.Dataset.nbytes] xarray method, which returns the total size if all that data were actually loaded into memory).
Do this for one file, and multiply by the number of files you have to estimate the total memory required for this step.

Writing the combined virtual references out requires converting them to a different references format, which may have different memory requirements.

## Scalability of references formats

After the map-reduce operation is complete, you will likely still want to persist the virtual references in some format.
Depending on the format, this step may also have scalability concerns.

### Kerchunk

The Kerchunk references specification supports 3 formats: an in-memory (nested) `dict`, JSON, and Parquet.

Both the in-memory Kerchunk `dict` and Kerchunk JSON formats are extremely inefficient ways to represent virtual references.
You may well find that a virtual dataset object that easily fits in memory suddenly uses up many times more memory or space on disk when converted to one of these formats.
Persisting large numbers of references in these formats is therefore not recommended.

The Kerchunk Parquet format is more scalable, but you may want to experiment with the  `record_size` and `categorical_threshold` arguments to the virtualizarr [`.to_kerchunk`][virtualizarr.accessor.VirtualiZarrDatasetAccessor.to_kerchunk] accessor method.

### Icechunk

[Icechunk](https://icechunk.io/) uses it's own [open format](https://icechunk.io/en/latest/spec/) for persisting virtual references.

Icechunk's format stores the virtual references in dedicated binary files, and can use "manifest splitting", together meaning that it should be a scalable way to store large numbers of references.

## Tips for success

Here are some assorted tips for successfully scaling VirtualiZarr.

### Caching remote files

When you call [`open_virtual_dataset`][virtualizarr.open_virtual_dataset] on a remote file, it  needs to extract the metadata and store it in memory (the returned virtual dataset).

One way to do this is to issue HTTP range requests only for each piece of metadata.
This will download the absolute minimum amount of data in total, but issue a lot of HTTP requests, each of which can take a long time to be returned from high-latency object storage.
This approach therefore uses the minimum amount of memory on the worker but takes more time.

The other extreme is to download the entire file up front.
This downloads all the metadata by definition, but also all the actual data, which is likely millions of times more than you need for virtualization.
This approach usually takes a lot less time on the worker but requires the maximum amount of memory - using this approach on every file in the dataset entails downloading the entire dataset across all workers!

There are various tricks one can use when fetching metadata, such as pre-fetching, minimum fetch sizes, or read-ahead caching strategies.
All of these approaches will put your memory requirements somewhere in between the two extremes described above, and are not necessary for successful execution.

Generally if you have access only to a limited amount of RAM you want to avoid caching to avoid running out of memory, whereas if you are able to scale out across many workers (e.g. serverlessly using lithops) your job will complete faster if you cache the files.
Caching a file onto a worker requires that the memory available on that worker is greater than the size of the file.

### Batching

You don't need to create and write virtual references for all your files in one go.

Creating virtual references for subsets of files in batches means the memory requirements for combining and serializing each batch are lower.

Batching also allows you to pick up where you left off.
This works particularly well with Icechunk, as you can durably commit each batch of references in a separate transaction.

```python
import icehunk as ic

repo = ic.open(<repo_url>)

for i, batch in enumerate(file_batches):
    session = repo.writable_session("main")

    combined_batch_vds = vz.open_virtual_mfdataset(batch, registry=registry)

    combined_batch_vds.vz.to_icechunk(session.store, append_dim=...)

    session.commit(f"wrote virtual references for batch {i}")
```

Notice this workflow could also be used for appending data only as it becomes available, e.g. by replacing the for loop with a cron job.

### Retries

Sometimes an [`open_virtual_dataset`][virtualizarr.open_virtual_dataset] call might fail for a transient reason, such as a failed HTTP response from a server.
In such a scenario automatically retrying the failed call might be enough to obtain success and keep the computation proceeding.

If you are batching your computation then you could retry each loop iteration if any [`open_virtual_dataset`][virtualizarr.open_virtual_dataset] calls fail, but that's potentially very inefficient, because that would also retry the successful calls.

Instead what is more efficient is to use per-task retries at te executor level.


In the future, we plan to add support for automatic retries to the Lithops and Dask executors (see Github PR #575)
