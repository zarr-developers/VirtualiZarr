# Loading inlined Kerchunk references

Kerchunk reference files can contain two kinds of chunk references:

- **Virtual references** point to byte ranges in external files (e.g., `["s3://bucket/data.nc", 1024, 512]`)
- **Inlined references** embed the raw chunk data directly in the JSON as base64-encoded strings (e.g., `"base64:AAAB..."`)

Inlined references are common for small variables like coordinate arrays, dimension labels, and scalar metadata. Kerchunk inlines data below a configurable `inline_threshold`.

VirtualiZarr can read both kinds of references. Inlined data is stored as **native chunks** directly in the [ChunkManifest][virtualizarr.manifests.ChunkManifest], so it travels with the manifest through concatenation, serialization, and pickling without needing access to any external file.

## Roundtrip example

This example demonstrates that the full pipeline---NetCDF to kerchunk JSON (with inlined coordinates) back to an xarray Dataset---produces results identical to loading the NetCDF directly.

### 1. Create a sample NetCDF file

```python
import tempfile, os, json
import numpy as np
import xarray as xr

tmpdir = tempfile.mkdtemp()
nc_path = os.path.join(tmpdir, "example.nc")

ds = xr.Dataset(
    {"temperature": xr.DataArray(
        np.arange(12, dtype="float32").reshape(3, 4),
        dims=["time", "x"],
    )},
    coords={
        "time": np.array([0, 1, 2], dtype="int64"),
        "x": np.array([10, 20, 30, 40], dtype="int64"),
    },
)
ds.to_netcdf(nc_path, format="NETCDF4")
```

### 2. Virtualize and write to kerchunk JSON

Use the HDF parser to read the NetCDF file. Specify `loadable_variables` for the
coordinate arrays so they are loaded into memory as numpy arrays. When serialized
to kerchunk format, these loaded variables are automatically base64-encoded as
inlined references.

```python
from virtualizarr import open_virtual_dataset
from virtualizarr.parsers import HDFParser
from obspec_utils.registry import ObjectStoreRegistry
from obstore.store import LocalStore

store = LocalStore(prefix="/")
registry = ObjectStoreRegistry({"file://": store})

with open_virtual_dataset(
    url=f"file://{nc_path}",
    registry=registry,
    parser=HDFParser(),
    loadable_variables=["time", "x"],
) as vds:
    refs = vds.vz.to_kerchunk(format="dict")

# Write to disk
ref_path = os.path.join(tmpdir, "refs.json")
with open(ref_path, "w") as f:
    json.dump(refs, f)
```

The resulting JSON has a mix of virtual and inlined references:

```python
for key, value in refs["refs"].items():
    if isinstance(value, str) and value.startswith("base64:"):
        print(f"  Inlined: {key}")
    elif isinstance(value, list):
        print(f"  Virtual: {key} -> {value[0]}")
```

```
  Inlined: time/0
  Inlined: x/0
  Virtual: temperature/0.0 -> /tmp/.../example.nc
```

### 3. Load the kerchunk JSON back

Use the `KerchunkJSONParser` to read the reference file. Inlined data is decoded
from base64 and stored as native chunks in the manifest.

```python
from virtualizarr.parsers import KerchunkJSONParser

parser = KerchunkJSONParser()
manifest_store = parser(url=f"file://{ref_path}", registry=registry)
```

Open the manifest store as an xarray Dataset via the Zarr engine:

```python
loaded = xr.open_dataset(
    manifest_store, engine="zarr", consolidated=False, zarr_format=3
).load()
```

### 4. Verify the roundtrip

```python
direct = xr.open_dataset(nc_path).load()
xr.testing.assert_identical(direct, loaded)
```

The two datasets are identical: coordinate values, data values, attributes, and dtypes all match.

## How it works

When the kerchunk parser encounters a base64-encoded inlined reference, it decodes the bytes and stores them as a **native chunk** on the `ChunkManifest`. Native chunks are held in a sparse dictionary keyed by chunk grid index:

```python
# After parsing, the manifest for 'time' has one native chunk:
time_manifest = manifest_store._group.arrays["time"].manifest
print(time_manifest._native)
# {(0,): b'\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00...'}
```

Native chunks participate in all manifest operations:

- **Concatenation and stacking**: indices are shifted to their new positions
- **Serialization**: included when writing back to kerchunk (re-encoded as base64) or Icechunk (written as real data)
- **Pickling**: travel with the manifest for distributed workflows (Dask, multiprocessing)
- **ManifestStore reads**: returned directly from memory without any network or disk I/O
