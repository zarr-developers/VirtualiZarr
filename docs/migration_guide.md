# V2 Migration Guide


With the release of VirtualiZarr 2.0, there are some core changes to how VirtualiZarr works. The goal of this guide is to provide some context around the core changes and demonstrate the updated usage. 

## `open_virtual_dataset` in V1

In V1 there was a lot of auto-magic guesswork of filetypes and urls that was happening under the hood. 
While this made it easy to get started, it could lead to a lot of footguns and unexpected behavior.

```python
from virtualizarr import open_virtual_dataset
vds = open_virtual_dataset(filepath)
```

## V2
In V2, virtualizing a dataset requires a bit more input, but is much more explicit.
You now must pass in a `Parser` (formally called a reader) and an [Obstore Store](https://developmentseed.org/obstore/latest/getting-started/#constructing-a-store) into `open_virtual_dataset`. 
This change adds a bit more verbosity, but is intended to make virtualizing datasets more robust. 

=== "S3 Store"

    ```python
    from virtualizarr import open_virtual_dataset
    from virtualizarr.parsers import HDFParser
    from obstore.store import S3Store

    file_url = "data1.nc"
    store = S3Store("bucket-name", region="us-east-1", skip_signature=True)
    parser = HDFParser()

    vds = open_virtual_dataset(file_url = file_url, object_store = store, parser=parser)
    ```

=== "Local Store"

    ```python
    from virtualizarr import open_virtual_dataset
    from virtualizarr.parsers import HDFParser
    from obstore.store import LocalStore
    from pathlib import Path

    file_url = 'path/data1.nc'
    path = Path(file_url)
    store = LocalStore(prefix=path.parent)
    parser = HDFParser()

    vds = open_virtual_dataset(file_url = file_url, object_store = store, parser=parser)
    ```



### Reading chunks without writing to disk
In Virtualizarr V1 if you wanted to access the underlying chunks of a dataset, you first had to write the reference to disk. From there you could read those references back into Xarray and access the chunks like you would with a normal Xarray dataset.

In V2 you can now **directly read the chunks from a Parser into Xarray without writing them to disk first**.
Since each `Parser` is now responsible for creating a `ManifestStore` and the `ManifestStore` chunks are backed by `Obstore`, you should be able to pass a `ManifestStore` directly into Xarray.

`Parser -> ManifestStore -> Xarray`

```python
from virtualizarr import open_virtual_dataset
from virtualizarr.parsers import HDFParser
from obstore.store import S3Store

file_url = "data1.nc"
store = S3Store("bucket-name", region="us-east-1", skip_signature=True)
parser = HDFParser()

manifest_store = parser(file_url = file_url, object_store = store)
ds = xr.open_zarr(manifest_store)
```