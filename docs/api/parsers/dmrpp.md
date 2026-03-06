# DMR++

The DMRPP parser reads DMR++ (Data Model Response Protocol Plus) XML files, which contain metadata describing datasets and their chunk locations. It creates virtual Zarr arrays backed by the original data files.

::: virtualizarr.parsers.DMRPPParser

## Usage

```python
import virtualizarr
from virtualizarr.parsers import DMRPPParser

# Open a DMR++ file as a virtual dataset
ds = virtualizarr.open_virtual_dataset(
    "file:///path/to/dataset.dmrpp",
    parser=DMRPPParser(),
)
```

## Validation

The DMRPP parser includes validation for missing attributes in the DMR++ XML. When required attributes are missing, the parser raises a `ValueError`. When optional attributes are missing, validation issues are accumulated and can be accessed via the `_validation_issues` attribute of the parser instance.
