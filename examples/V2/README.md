# V2 Examples

These examples use the VirtualiZarr 2.x API with `obstore` for cloud storage access.

## Examples

### [goes_basic.py](goes_basic.py)

Virtualizes a GOES-16 satellite file using a standard obstore without any optimizations. Use this as a baseline for performance comparison.

```bash
uv run --script examples/V2/goes_basic.py
```

### [goes_with_caching_stores.py](goes_with_caching_stores.py)

Virtualizes the same GOES-16 file using `CachingReadableStore` and `SplittingReadableStore` from `obspec-utils` to optimize metadata access.

```bash
uv run --script examples/V2/goes_with_caching_stores.py
```

### [its_live.ipynb](its_live.ipynb)

Mosaics two [ITS_LIVE](https://its-live.jpl.nasa.gov/) glacier-velocity granules that share a global grid but each cover only part of it. Uses native xarray `concat(..., join="outer")` to align them onto the shared grid (sparse, with fill where a granule has no data) and stack them along `time`, keeping the data variables virtual (`ManifestArray`) throughout, then writes the result to Icechunk without copying any pixel data.

## Performance Comparison

Run both scripts and compare the "Virtualization time" printed at the end of each:

```bash
uv run --script examples/V2/goes_basic.py
uv run --script examples/V2/goes_with_caching_stores.py
```

The timing is measured internally to exclude dependency installation overhead. Running on a laptop in North Carolina, the basic approach takes 47s while the caching + splitting approach takes 9s.

## Requirements

These examples use [uv](https://docs.astral.sh/uv/) inline script metadata for dependency management. Each script specifies its own dependencies and can be run directly with `uv run`.
