from importlib.metadata import version as _version

from virtualizarr.accessor import (
    VirtualiZarrDatasetAccessor,
    VirtualiZarrDataTreeAccessor,
)
from virtualizarr.xarray import open_virtual_dataset, open_virtual_mfdataset

try:
    __version__ = _version("virtualizarr")
except Exception:
    # Local copy or not installed with setuptools.
    # Disable minimum version checks on downstream libraries.
    __version__ = "9999"

__all__ = [
    "VirtualiZarrDatasetAccessor",
    "VirtualiZarrDataTreeAccessor",
    "open_virtual_dataset",
    "open_virtual_mfdataset",
]
