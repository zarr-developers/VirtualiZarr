# V2 Migration Guide

VirtualiZarr V2 includes breaking changes and other conceptual differences relative to V1. The goal of this guide
is to provide some context around the core changes and demonstrate the updated usage.

## Breaking API changes in `open_virtual_dataset`

### Filetype identification, parsers, and stores

In V1 there was a lot of auto-magic guesswork of filetypes, urls, and types of remote storage happening under the hood.
While this made it easy to get started, it could lead to a lot of foot-guns and unexpected behavior.

For example, the following V1-style usage would guess that your data is in a NetCDF file format and that your data
is stored in a local file. However, this did not provide a way for people to develop their own utilities
for data formats or specific datasets. This guess work also made it more challenging for developers to avoid bugs and
users to understand VirtualiZarr's behavior.

```python
from virtualizarr import open_virtual_dataset
vds = open_virtual_dataset("data1.nc")
```

To provide a more extensible and reliable API, VirtualiZarr V2 requires more explicit configuration by the user.
You now must pass in a valid [Parser][virtualizarr.parsers.typing.Parser] and a [virtualizarr.registry.ObjectStoreRegistry][] to [virtualizarr.open_virtual_dataset][].
This change adds a bit more verbosity, but is intended to make virtualizing datasets more robust. It is most common for the
[ObjectStoreRegistry][virtualizarr.registry.ObjectStoreRegistry] to contain one or more [ObjectStores][obstore.store.ObjectStore]
for reading the original data, but some parsers may accept an empty [ObjectStoreRegistry][virtualizarr.registry.ObjectStoreRegistry].

=== "S3 Store"

    ```python exec="on" source="material-block" session="migration" result="code"
    from obstore.store import S3Store

    from virtualizarr import open_virtual_dataset
    from virtualizarr.parsers import HDFParser
    from virtualizarr.registry import ObjectStoreRegistry

    bucket = "nex-gddp-cmip6"
    store = S3Store(
        bucket=bucket,
        region="us-west-2",
        skip_signature=True # required for this specific example data because the data is in a public bucket, so the S3Store shouldn't fetch and use credentials.
    )
    registry = ObjectStoreRegistry({f"s3://{bucket}": store})
    parser = HDFParser()
    vds = open_virtual_dataset(
        url=f"s3://{bucket}/NEX-GDDP-CMIP6/ACCESS-CM2/ssp126/r1i1p1f1/tasmax/tasmax_day_ACCESS-CM2_ssp126_r1i1p1f1_gn_2015_v2.0.nc",
        registry=registry,
        parser=parser
    )
    print(vds)
    ```

=== "Local Store"

```python


from obstore.store import LocalStore

from virtualizarr import open_virtual_dataset
from virtualizarr.parsers import HDFParser
from virtualizarr.registry import ObjectStoreRegistry

from pathlib import Path

store_path = Path.cwd()
file_path = str(store_path / "tasmax_day_ACCESS-CM2_ssp126_r1i1p1f1_gn_2015_v2.0.nc")
file_url = f"file://{file_path}"

store = LocalStore(prefix=store_path)
registry = ObjectStoreRegistry({file_url: store})
parser = HDFParser()

vds = open_virtual_dataset(
    url=file_url,
    registry=registry,
    parser=parser
)
    print(vds)

```

### Deprecation of other kwargs

We have removed some keyword arguments to `open_virtual_dataset` that were deprecated, saw little use, or are now redundant. Specifically:

- `indexes` - there is little need to control this separately from `loadable_variables`,
- `cftime_variables` - this argument is deprecated upstream in favor of `decode_times`,
- `backend` - replaced by the `parser` kwarg,
- `virtual_backend_kwargs` - replaced by arguments to the `parser` instance,
- `reader_options` - replaced by arguments to the ObjectStore instance.
- `virtual_array_class` - so far has not been needed,

## Missing features

We have worked hard to ensure that nearly all features from VirtualiZarr V1 are available in V2. To our knowledge,
the only functionality regression is the ability to "glob" in [virtualizarr.open_virtual_mfdataset][]. We aim to support
this in the future. Please see [issue #569](https://github.com/zarr-developers/VirtualiZarr/issues/569) for progress
towards this feature.

### Xarray accessor name

In VirtualiZarr V2 you should use the shorthand `.vz` accessor for Xarray operations. The previous accessor name
`virtualize` is available but will yield a `DeprecationWarning`. It may be remove entirely in the future. Here
is an example of using the new accessor name:

```python
vds.vz.to_icechunk(icechunk_store)
```

## New functionality

### Reading chunks without writing to disk

In Virtualizarr V1 if you wanted to access the underlying chunks of a dataset, you first had to write the reference to disk. From there you could read those references back into Xarray and access the chunks like you would with a normal Xarray dataset.

In V2 you can now **directly read the chunks from a Parser into Xarray without writing them to disk first**. 🤯
Since each `Parser` is now responsible for creating a [ManifestStore][virtualizarr.manifests.ManifestStore] and the [ManifestStore][virtualizarr.manifests.ManifestStore] has the ability to fetch data through any [ObjectStore][obstore.store.ObjectStore] in the [ObjectStoreRegistry][virtualizarr.registry.ObjectStoreRegistry]. You
can load data using the [ManifestStore][virtualizarr.manifests.ManifestStore] via either Zarr or Xarray. Here's an example using Xarray:

```python exec="on" source="material-block" session="migration" result="code"
import xarray as xr
from obstore.store import S3Store

from virtualizarr.parsers import HDFParser
from virtualizarr.registry import ObjectStoreRegistry

bucket = "nex-gddp-cmip6"
store = S3Store(
    bucket=bucket,
    region="us-west-2",
    skip_signature=True
)
registry = ObjectStoreRegistry({f"s3://{bucket}": store})
parser = HDFParser()
manifest_store = parser(
    url=f"s3://{bucket}/NEX-GDDP-CMIP6/ACCESS-CM2/ssp126/r1i1p1f1/tasmax/tasmax_day_ACCESS-CM2_ssp126_r1i1p1f1_gn_2015_v2.0.nc",
    registry=registry
)
loadable_ds = xr.open_zarr(
    manifest_store,
    consolidated=False,
    zarr_format=3,
)
print(loadable_ds)
```

Note how the Xarray dataset contains loadable Dask arrays rather than manifest arrays.

### Bring your own parser

The V2 API means that you can use VirtualiZarr's data structure and xarray's functionality merging and combining datasets
completely independently from the VirtualiZarr library! [Virtual-Tiff](https://github.com/virtual-zarr/virtual-tiff) and
the [hrrr-parser](https://github.com/virtual-zarr/hrrr-parser) are examples of this pattern. Read some instructions
on how to write a parser in the [Custom Parsers](custom_parsers.md) page.
