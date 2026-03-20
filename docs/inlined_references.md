# Inlined chunks

VirtualiZarr's [ChunkManifest][virtualizarr.manifests.ChunkManifest] can hold two kinds of chunks:

- **Virtual chunks** point to byte ranges in external files (e.g., a range inside a NetCDF file on S3).
- **Inlined chunks** store the raw chunk data directly in memory, inside the manifest itself.

Inlined chunks are useful for small variables like coordinate arrays, dimension labels, and scalar metadata, where the overhead of a remote read would exceed the cost of simply carrying the bytes along.

## How it works

Inlined chunks are stored in a sparse dictionary (`_inlined`) on the `ChunkManifest`, keyed by chunk grid index:

```python
manifest._inlined
# {(0,): b'\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00...'}
```

In the paths array, inlined chunks are distinguished from missing chunks by a sentinel value (`INLINED_CHUNK_PATH`), while missing chunks use an empty string (`MISSING_CHUNK_PATH`).

## Manifest operations

Inlined chunks participate in all manifest operations:

- **Concatenation and stacking**: indices are shifted to their new positions
- **Broadcasting**: singleton dimensions are prepended to keys
- **Equality**: two manifests are equal only if their inlined data also matches
- **Pickling**: inlined data travels with the manifest for distributed workflows (Dask, multiprocessing)
- **ManifestStore reads**: returned directly from memory without any network or disk I/O
- **`nbytes`**: includes the size of inlined data

## Creating a manifest with inlined chunks

Pass entries with a `data` key to the `ChunkManifest` constructor:

```python
from virtualizarr.manifests import ChunkManifest

manifest = ChunkManifest(
    entries={
        "0.0": {"path": "s3://bucket/foo.nc", "offset": 100, "length": 100},
        "0.1": {"path": "", "offset": 0, "length": 4, "data": b"\x00\x01\x02\x03"},
    }
)
```

Or use `from_arrays` with the `inlined` parameter:

```python
import numpy as np

manifest = ChunkManifest.from_arrays(
    paths=np.asarray(["s3://bucket/foo.nc", ""], dtype=np.dtypes.StringDType),
    offsets=np.asarray([100, 0], dtype=np.uint64),
    lengths=np.asarray([100, 4], dtype=np.uint64),
    inlined={(1,): b"\x00\x01\x02\x03"},
)
```
