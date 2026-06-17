# GRIB

The [gribberish](https://github.com/mpiannucci/gribberish) library (a Rust-backed GRIB reader) provides a `GribberishParser` for VirtualiZarr. **Both GRIB1 and GRIB2 are supported.** Each GRIB message becomes a single chunk, and reading a chunk back decodes the referenced message bytes through gribberish's registered zarr codec.

!!! note "Packing templates"

    gribberish decodes GRIB2 grid-point fields (including complex/second-order packing) and GRIB1 *simple* packing. GRIB1 messages using complex or spherical-harmonic packing are not yet supported and raise an error on read. See the [gribberish](https://github.com/mpiannucci/gribberish) repository for the current coverage.

Install it alongside VirtualiZarr with:

```shell
pip install "virtualizarr[grib]"
```

Then point [virtualizarr.open_virtual_dataset][] (or [virtualizarr.open_virtual_datatree][], for files whose messages don't fit a single hypercube) at a GRIB file:

```python
from obspec_utils.registry import ObjectStoreRegistry
from obstore.store import LocalStore

from gribberish.virtualizarr import GribberishParser
from virtualizarr import open_virtual_dataset

registry = ObjectStoreRegistry({"file://": LocalStore()})
parser = GribberishParser()
vds = open_virtual_dataset(
    url="file:///path/to/file.grib2", parser=parser, registry=registry
)
```

See the [gribberish documentation](https://github.com/mpiannucci/gribberish/tree/main/python) for the full parser API and its filtering options (`only_variables`, `drop_variables`, etc.).
