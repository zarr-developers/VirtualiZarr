# The object store registry

This page explains the `registry` argument that [`open_virtual_dataset`][virtualizarr.open_virtual_dataset] requires.

Virtualizing a file doesn't copy its data, but VirtualiZarr still has to read from the file in order to virtualize it, and again whenever you load data from it.

- **Building the virtual dataset.** The parser reads the file's header and chunk index to find where each chunk lives.
- **Loading data.** After parsing, a separate loading step fetches the bytes of any variable you load. By default, `open_virtual_dataset` loads the dimension coordinates (such as `time`, `lat` and `lon`) so that xarray can index the dataset.

Both steps need an object that can do the I/O, fetching just the byte ranges they need from wherever the file lives, whether that's S3, an HTTP server, or your own disk.
VirtualiZarr uses an [obstore](https://developmentseed.org/obstore/latest/) [`ObjectStore`][obstore.store.ObjectStore] object for this, called a store on the rest of this page (see [section 1](#1-what-a-store-holds)).

A store covers a single bucket or host.
When your files are spread across more than one, you need several stores, and VirtualiZarr needs to know which store to use for each file.
The [`ObjectStoreRegistry`][obspec_utils.registry.ObjectStoreRegistry] organizes them by mapping URL prefixes to the stores you configured, so it can match each file's URL to its store for you.
`open_virtual_dataset` always takes a registry, so with a single store you pass a registry with one entry.

For files on your own disk, `ObjectStoreRegistry({"file:///": LocalStore()})` works for every file (see [section 4](#4-local-files)).

## 1. What a store holds

This page builds a time series of sea surface temperature from GOES-East, the NOAA weather satellite that views the Americas and the Atlantic.
On 7 April 2025, GOES-19 replaced GOES-16 as GOES-East.
NOAA publishes each satellite's data in its own public bucket on AWS S3, so the time series reads from `noaa-goes16` before the handover and from `noaa-goes19` after it.

To virtualize a file, the parser has to read it, but the file's URL alone is not enough information.
You also need to provide three settings.

- **The service.** `s3://` URLs are also used for S3-compatible services such as Cloudflare R2 or a Ceph cluster, which need a custom endpoint (see the R2 and CEPH tabs in the [usage guide](../how_to/usage.md#opening-files-as-virtual-datasets)).
- **The region.** Each S3 bucket lives in one AWS region, and the store needs to know which.
- **The credentials.** Private data needs your access keys. Public data can be read with anonymous (unsigned) requests, which obstore calls `skip_signature=True`.

A store holds exactly these settings.
Both GOES buckets are public and in `us-east-1`, so both stores use that region and anonymous requests.
Each [`S3Store`][obstore.store.S3Store] reads from one bucket, so the two buckets still need two stores.
VirtualiZarr can't guess the settings, so you create the stores yourself.

```python exec="on" session="registry" source="above"
from pprint import pformat

from obstore.store import S3Store
from obspec_utils.registry import ObjectStoreRegistry

from virtualizarr import (
    open_virtual_dataset,
    open_virtual_mfdataset,
)
from virtualizarr.parsers import HDFParser

goes16_bucket = "s3://noaa-goes16"
goes19_bucket = "s3://noaa-goes19"

goes16_store = S3Store.from_url(
    goes16_bucket,
    region="us-east-1",
    # public bucket, so send anonymous (unsigned) requests
    skip_signature=True,
)
goes19_store = S3Store.from_url(
    goes19_bucket,
    region="us-east-1",
    # public bucket, so send anonymous (unsigned) requests
    skip_signature=True,
)
```

## 2. Building the time series

A time series that spans the handover needs at least one file from each side of it.
This example uses two files, taken at 12:00 UTC on 6 April and 8 April 2025, but the same code works for any number of files.

```python exec="on" session="registry" source="above"
goes16_url = (
    f"{goes16_bucket}/ABI-L2-SSTF/2025/096/12/"
    "OR_ABI-L2-SSTF-M6_G16_s20250961200208_"
    "e20250961259516_c20250961304427.nc"
)
goes19_url = (
    f"{goes19_bucket}/ABI-L2-SSTF/2025/098/12/"
    "OR_ABI-L2-SSTF-M6_G19_s20250981200209_"
    "e20250981259517_c20250981304588.nc"
)
```

Each file has to be read through the store for its own bucket.
If you opened the files one at a time in a loop, you could pick the store for each URL yourself.
A registry does that matching for you, and it's the only way to do it when a single call opens many files.
Register each store under the URL prefix it serves, usually its bucket, then pass the registry to [`open_virtual_mfdataset`][virtualizarr.open_virtual_mfdataset], which opens both files and joins them along time.

```python exec="on" session="registry" source="above" result="code"
registry = ObjectStoreRegistry(
    {goes16_bucket: goes16_store, goes19_bucket: goes19_store}
)

vds = open_virtual_mfdataset(
    [goes16_url, goes19_url],
    registry=registry,
    parser=HDFParser(),
    combine="nested",
    concat_dim="t",
    data_vars="minimal",
    coords="minimal",
    compat="override",
    loadable_variables=["t", "x", "y"],
)
print(vds["SST"].sizes)
```

The arguments after `parser` are xarray's options for combining datasets, and have nothing to do with the registry.
They stop xarray from comparing values that exist only as virtual references (see [Combining virtual datasets](../how_to/usage.md#combining-virtual-datasets)).

For each file, the parser read the header through the store the registry matched to it, and created a virtual chunk reference for each chunk.
Each reference records the URL of the file the chunk lives in, the `offset` where the chunk starts, and its `length` in bytes.
The chunk keys of `SST` start with the time index, so here is one reference from each side of the handover.

```python exec="on" session="registry" source="above" result="code"
chunks = vds["SST"].data.manifest.dict()

before = next(v for k, v in chunks.items() if k.startswith("0."))
after = next(v for k, v in chunks.items() if k.startswith("1."))

print(pformat(before))
print(pformat(after))
```

One variable now holds chunks from both buckets.
To load a chunk, the loading step asks the registry for the store that matches its `path`, then reads `length` bytes starting at `offset` through that store.

## 3. How the registry picks a store

Whenever the parser or the loading step needs to read a URL, it asks the registry for the store that can read it.
The registry returns the store whose prefix matches the URL, along with the object's path inside that store.

```python exec="on" session="registry" source="above" result="code"
goes16_match, goes16_path = registry.resolve(goes16_url)
goes19_match, goes19_path = registry.resolve(goes19_url)

print(goes16_match)
print(f"  {goes16_path}")
print(goes19_match)
print(f"  {goes19_path}")
```

The registry matches the scheme and bucket (or host) exactly, then picks the longest registered path that is a prefix of the URL.

`open_virtual_mfdataset` gives the registry to the parser for each file, and each parser passes it on to its loading step (a [`ManifestStore`][virtualizarr.manifests.ManifestStore]).
See [Data structures](data_structures.md) for what a `ManifestStore` holds, and [Custom parsers](custom_parsers.md) if you are writing a parser.

## 4. Local files

Local files have nothing to configure, but the parser and the loading step only read through stores, so you still need one.
[`LocalStore`][obstore.store.LocalStore] is the store for your disk.
Registered under `"file:///"`, it covers every file on the machine.
To show this, first write a small netCDF file to a temporary directory.

```python exec="on" session="registry" source="above"
import tempfile
from pathlib import Path

import numpy as np
import xarray as xr
from obstore.store import LocalStore

local_file = str(Path(tempfile.mkdtemp()) / "air.nc")
ds = xr.Dataset({"air": ("time", np.arange(4.0))})
ds.to_netcdf(local_file, engine="h5netcdf")
```

Create the registry, then ask it which store it would use for that file.

```python exec="on" session="registry" source="above" result="code"
local_registry = ObjectStoreRegistry({"file:///": LocalStore()})

matched_store, path_in_store = local_registry.resolve(
    f"file://{local_file}"
)
print(matched_store)
print(path_in_store)
```

The registry returns the `LocalStore`, with the file's path relative to the filesystem root.
With the registry in place, you can virtualize the file.

```python exec="on" session="registry" source="above"
local_vds = open_virtual_dataset(
    local_file, registry=local_registry, parser=HDFParser()
)
```

The `air` variable in `local_vds` holds virtual chunk references rather than data.
Here is the reference for its first chunk.

```python exec="on" session="registry" source="above" result="code"
first_local_chunk = local_vds["air"].data.manifest.dict()["0"]
print(pformat(first_local_chunk))
```

`open_virtual_dataset` accepted a plain path, and the parser recorded it in the virtual chunk reference as a `file://` URL.
The loading step asks the registry for a store for that URL, so the registry key must be a `file://` prefix of it.
`"file:///"` is a prefix of every local file URL.

!!! warning
    Registering the URL of a single file, such as `{"file:///home/me/data/air.nc": LocalStore(prefix="/home/me/data")}`, works for that one file only.
    Opening a second file from the same directory then fails with `Could not find an ObjectStore matching the url`, because the first file's URL is not a prefix of the second's.
    Register a directory, or `"file:///"`, instead.

## 5. Reading the data later

Whoever reads your virtual dataset later, including you in a new Python session, has to fetch the same chunks, so they need stores with the same settings.
The registry can't give them those stores, because it exists only in your Python session.
Writing a virtual dataset with [`to_icechunk`][virtualizarr.accessor.VirtualiZarrDatasetAccessor.to_icechunk] saves the chunk URLs, not the registry.

Icechunk solves this with its own mapping.
Its virtual chunk containers ([`icechunk.VirtualChunkContainer`][icechunk.VirtualChunkContainer]) map URL prefixes to storage configurations, just as the registry does, but they are saved in the repository's config.
See [Writing to an Icechunk Store](../how_to/usage.md#writing-to-an-icechunk-store).

## Troubleshooting

### "Could not find an ObjectStore matching the url"

This error means the parser or the loading step asked the registry for a store for a URL, and no registry key was a prefix of it.
The URL in the message is the one that failed.
Compare it with your registry keys, checking each of these.

- **Scheme.** `https://my-bucket.s3.amazonaws.com/...` and `s3://my-bucket/...` refer to the same object but need different keys.
- **Bucket or host.** Must match exactly.
- **Path.** For local files, the key must be a `file://` prefix of the file's absolute path (see [section 4](#4-local-files)).

`registry.resolve(url)` raises the same error, so you can check a registry against a URL before opening anything.

### Registry keys and bucket names

Registry keys are URLs, so they include the scheme, but `S3Store(bucket=...)` takes the bare bucket name.
Mixing up the two forms fails in different ways.

- **A registry key without a scheme**, such as `"noaa-goes16"`, raises `ValueError: Urls are expected to contain a scheme` when you create the registry.
- **A bucket name with a scheme**, such as `S3Store(bucket="s3://noaa-goes16")`, is accepted when you create the store, but every request then fails with an S3 error, because the store sends `s3://noaa-goes16` as the bucket name.

[`S3Store.from_url`][obstore.store.S3Store.from_url] takes the URL form, so you can pass the same string to the store and use it as the registry key, as the examples on this page do.
