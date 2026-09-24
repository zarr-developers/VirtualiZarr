# The object store registry

This page explains the `registry` argument that [`open_virtual_dataset`][virtualizarr.open_virtual_dataset] requires.

Virtualizing a file doesn't copy its data, but VirtualiZarr still has to read from the file in order to virtualize it, and again whenever you load data from it.

- **Building the virtual dataset.** The parser reads the file's header and chunk index to find where each chunk lives. Some parsers also fetch the bytes of certain chunks and store them inline in the virtual dataset.
- **Loading data.** After parsing, a separate loading step fetches the bytes of any variable you load.

Both steps need an object that can do the I/O, fetching just the byte ranges they need from wherever the file lives, whether that's S3, an HTTP server, or your own disk.
VirtualiZarr uses an [obstore](https://developmentseed.org/obstore/latest/) [`ObjectStore`][obstore.store.ObjectStore] object for this, called a store on the rest of this page.

The URL of the file(s) is not enough on it's own to perform the I/O. You also need to provide three settings.

- **The service.** `s3://` URLs are also used for S3-compatible services such as Cloudflare R2 or a Ceph cluster, which need a custom endpoint (see the R2 and CEPH tabs in the [usage guide](../how_to/usage.md#opening-files-as-virtual-datasets)).
- **The region.** Each S3 bucket lives in one AWS region, and the store needs to know which.
- **The credentials.** Private data needs your access keys. Public data can be read with anonymous (unsigned) requests, which obstore calls `skip_signature=True`.

A store only works for a single bucket or host. If your files are spread across more than one, you need several stores, and VirtualiZarr needs to know which store to use for each file.

That mapping is the job of the [`ObjectStoreRegistry`][obspec_utils.registry.ObjectStoreRegistry].

`open_virtual_dataset` always takes a registry, even when you only use a single store it must be passed in inside of a registry. Similarlly even though local files don't require bucket config or credentials they get read via a `LocalStore` and must be passed in via a `ObjectStoreRegistry({"file:///": LocalStore()})`.

## Example on Two Buckets

GOES-East is a weather satellite over the Americas and the Atlantic. On 7 April 2025, GOES-19 replaced GOES-16 as GOES-East. NOAA publishes each satellite's data in its own public bucket on AWS S3, so the time series of GOES-East has to read from `noaa-goes16` before the handover and from `noaa-goes19` after it.

Both GOES buckets are public and in `us-east-1`, so both stores use that region and anonymous requests. But we still need two stores and a registry to let the parser distinguish them.

```python exec="on" session="registry" source="above"
from obstore.store import S3Store
from obspec_utils.registry import ObjectStoreRegistry

from virtualizarr import open_virtual_mfdataset
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
    skip_signature=True,
)
```

This example uses two files, taken at 12:00 UTC on 6 April and 8 April 2025, but the same code works for any number of files.
These files are chosen as examples of spanning the handover between the two data sources.

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

With the file URLs and the stores set up all that remains is to construct the registry and pass it to [`open_virtual_mfdataset`][virtualizarr.open_virtual_mfdataset], which opens both files and joins them along time.

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

Each virtual chunk reference records the URL of the file its chunk lives in, so `SST` now points into both buckets.
Loading any chunk goes back through the registry to find the store for that URL.

## Reading the data later

Whoever reads your virtual dataset later, including you in a new Python session, has to fetch the same chunks, so they need stores with the same settings.
The registry can't give them those stores, because it exists only in your Python session.
Writing a virtual dataset with [`to_icechunk`][virtualizarr.accessor.VirtualiZarrDatasetAccessor.to_icechunk] saves the chunk URLs, not the registry.

Icechunk solves this with its own mapping.
Its virtual chunk containers ([`icechunk.VirtualChunkContainer`][icechunk.VirtualChunkContainer]) map URL prefixes to storage configurations, just as the registry does, but they are saved in the repository's config.
See [Writing to an Icechunk Store](../how_to/usage.md#writing-to-an-icechunk-store).

## Troubleshooting

### "Could not find an ObjectStore matching the url"

This error means the parser or the loading step asked the registry for a store for a URL, and no registry key was a prefix of it.
The registry matches the scheme and bucket (or host) exactly, then picks the longest registered path that is a prefix of the URL.
The URL in the message is the one that failed.
Compare it with your registry keys, checking each of these.

- **Scheme.** `https://my-bucket.s3.amazonaws.com/...` and `s3://my-bucket/...` refer to the same object but need different keys.
- **Bucket or host.** Must match exactly.
- **Path.** For local files, the key must be a `file://` prefix of the file's absolute path (see [Local files](#local-files)).

`registry.resolve(url)` raises the same error, so you can check a registry against a URL before opening anything.
