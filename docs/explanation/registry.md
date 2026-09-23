# The object store registry

This page explains the `registry` argument that [`open_virtual_dataset`][virtualizarr.open_virtual_dataset] requires.

Virtualizing a file doesn't copy its data, but VirtualiZarr still has to read from the file in order to virtualize it, and again whenever you load data from it:

- **Building the virtual dataset.** The parser reads the file's header and chunk index to find where each chunk lives.
- **Loading data.** After parsing, a separate loading step fetches the bytes of any variable you load. By default, `open_virtual_dataset` loads the dimension coordinates (such as `time`, `lat` and `lon`) so that xarray can index the dataset.

Both steps need an object that can do the I/O: fetch just the byte ranges they need from wherever the file lives, whether that's S3, an HTTP server, or your own disk.
VirtualiZarr uses an [obstore](https://developmentseed.org/obstore/latest/) [`ObjectStore`][obstore.store.ObjectStore] for this (see [section 1](#1-what-a-store-holds)).

A store covers a single bucket or host.
When your files are spread across more than one, you need several stores, and VirtualiZarr needs to know which store to use for each file.
The [`ObjectStoreRegistry`][obspec_utils.registry.ObjectStoreRegistry] organizes them: a map from URL prefixes to the stores you configured.
`open_virtual_dataset` always takes a registry, so with a single store you pass a registry with one entry.

For files on your own disk, `ObjectStoreRegistry({"file:///": LocalStore()})` works for every file (see [section 5](#5-local-files)).

```python exec="on" session="registry"
import warnings
warnings.filterwarnings(
  "ignore",
  message="Numcodecs codecs are not in the Zarr version 3 specification*",
  category=UserWarning
)
```

## 1. What a store holds

Take one file from the NASA NEX-GDDP-CMIP6 dataset, which is stored in the public `nex-gddp-cmip6` bucket on AWS S3.
To virtualize it, the parser has to read it, but the URL is not enough information on its own.
You also need to provide:

- **The service.** `s3://` URLs are also used for S3-compatible services such as Cloudflare R2 or a Ceph cluster, which need a custom endpoint (see the R2 and CEPH tabs in the [usage guide](../how_to/usage.md#opening-files-as-virtual-datasets)).
- **The region.** The `nex-gddp-cmip6` bucket is in `us-west-2`.
- **The credentials.** `nex-gddp-cmip6` is public, so requests go unsigned (`skip_signature=True`). A private bucket needs your keys.

A store holds exactly these settings.
Below, `S3Store.from_url(...)` creates a store that can read any object in the `nex-gddp-cmip6` bucket, and the registry wraps it so `open_virtual_dataset` can use it:

```python exec="on" session="registry" source="above" result="code"
from pprint import pformat

from obstore.store import S3Store
from obspec_utils.registry import ObjectStoreRegistry

from virtualizarr import open_virtual_dataset
from virtualizarr.parsers import HDFParser

bucket = "s3://nex-gddp-cmip6"
url = f"{bucket}/NEX-GDDP-CMIP6/ACCESS-CM2/ssp126/r1i1p1f1/tasmax/tasmax_day_ACCESS-CM2_ssp126_r1i1p1f1_gn_2015_v2.0.nc"

store = S3Store.from_url(bucket, region="us-west-2", skip_signature=True)
registry = ObjectStoreRegistry({bucket: store})

vds = open_virtual_dataset(url, registry=registry, parser=HDFParser())

first_chunk = vds["tasmax"].data.manifest.dict()["0.0.0"]
print(pformat(first_chunk))
```

The parser read the file's header through that store and recorded where each chunk lives.
The output is the record for the first chunk of `tasmax`.
To load that chunk, the loading step asks for `length` bytes starting at `offset` from the object at `path`.
That object is in the same bucket, so the loading step reads through the same store.

## 2. Why VirtualiZarr doesn't create the stores itself

VirtualiZarr can't create a store for you, because it doesn't have the settings from section 1: which service to connect to, which region, and which credentials to use.
Only you know those, so you create the stores and pass them in, once, as a registry.
`open_virtual_dataset` gives the registry to the parser, and the parser passes it on to the loading step (a [`ManifestStore`][virtualizarr.manifests.ManifestStore]).
Both steps then read through the stores you configured.
See [Data structures](data_structures.md) for what a `ManifestStore` holds, and [Custom parsers](custom_parsers.md) if you are writing a parser.

## 3. How the registry picks a store

With more than one store registered, each step needs to know which store to use for a given URL.
When the parser or the loading step needs to read a URL, it asks the registry for the store that can read it.
You register each store under the URL prefix it serves, usually its bucket.
The registry then returns the store whose prefix matches the URL, along with the object's path inside that store:

```python exec="on" session="registry" source="above" result="code"
matched_store, path_in_store = registry.resolve(url)
print(matched_store)
print(path_in_store)
```

The registry matches the scheme and bucket (or host) exactly, then picks the longest registered path that is a prefix of the URL.

## 4. Multiple buckets in one registry

Suppose you combine the NEX-GDDP-CMIP6 file above with files from a private bucket of your own.
The chunks now live in two buckets with different settings, so loading them needs two stores.
Register a store for the second bucket in the same registry, and the registry returns the right store for each chunk:

```python exec="on" session="registry" source="above" result="code"
private_store = S3Store(bucket="my-private-bucket", region="eu-west-1")
registry.register("s3://my-private-bucket", private_store)

nex_store, _ = registry.resolve("s3://nex-gddp-cmip6/NEX-GDDP-CMIP6/some-file.nc")
private_store_match, _ = registry.resolve("s3://my-private-bucket/model-output/file.nc")

print(nex_store)
print(private_store_match)
```

## 5. Local files

Local files have nothing to configure, but the parser and the loading step only read through stores, so you still need one: [`LocalStore`][obstore.store.LocalStore], the store for your disk.
Registered under `"file:///"`, it covers every file on the machine.
This example writes a small netCDF file to a temporary directory and virtualizes it:

```python exec="on" session="registry" source="above" result="code"
import tempfile
from pathlib import Path

import numpy as np
import xarray as xr
from obstore.store import LocalStore

local_file = str(Path(tempfile.mkdtemp()) / "air.nc")
xr.Dataset({"air": ("time", np.arange(4.0))}).to_netcdf(local_file, engine="h5netcdf")

local_registry = ObjectStoreRegistry({"file:///": LocalStore()})

local_vds = open_virtual_dataset(local_file, registry=local_registry, parser=HDFParser())
print(pformat(local_vds["air"].data.manifest.dict()["0"]))
```

`open_virtual_dataset` accepted a plain path, and the parser recorded it in the chunk records as a `file://` URL.
The loading step asks the registry for a store for that URL, so the registry key must be a `file://` prefix of it.
`"file:///"` is a prefix of every local file URL.

!!! warning
    Registering the URL of a single file, such as `{"file:///home/me/data/air.nc": LocalStore(prefix="/home/me/data")}`, works for that one file only.
    Opening a second file from the same directory then fails with `Could not find an ObjectStore matching the url`, because the first file's URL is not a prefix of the second's.
    Register a directory, or `"file:///"`, instead.

## 6. Reading the data later

Whoever reads your virtual dataset later, including you in a new Python session, has to fetch the same chunks, so they need stores with the same settings.
The registry can't give them those stores: it exists only in your Python session, and writing a virtual dataset with [`to_icechunk`][virtualizarr.accessor.VirtualiZarrDatasetAccessor.to_icechunk] saves the chunk URLs, not the registry.

Icechunk solves this with its own mapping.
Its virtual chunk containers ([`icechunk.VirtualChunkContainer`][icechunk.VirtualChunkContainer]) map a URL prefix to a storage configuration, just as the registry does, but they are saved in the repository's config.
Writing to Icechunk therefore means describing the same locations twice: once as registry entries for VirtualiZarr, and once as virtual chunk containers for Icechunk (see [Writing to an Icechunk Store](../how_to/usage.md#writing-to-an-icechunk-store)).

## Troubleshooting

### "Could not find an ObjectStore matching the url"

This error means the parser or the loading step asked the registry for a store for a URL, and no registry key was a prefix of it.
The URL in the message is the one that failed; compare it with your registry keys:

- **Scheme.** `https://my-bucket.s3.amazonaws.com/...` and `s3://my-bucket/...` refer to the same object but need different keys.
- **Bucket or host.** Must match exactly.
- **Path.** For local files, the key must be a `file://` prefix of the file's absolute path (see [section 5](#5-local-files)).

`registry.resolve(url)` raises the same error, so you can check a registry against a URL before opening anything.
