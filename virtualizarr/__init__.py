from .manifests import ChunkManifest, ManifestArray  # type: ignore # noqa
from .xarray import VirtualiZarrDatasetAccessor  # type: ignore # noqa
from .xarray import open_virtual_dataset  # noqa: F401

from importlib.metadata import version as _version

try:
    __version__ = _version("virtualizarr")
except Exception:
    # Local copy or not installed with setuptools.
    # Disable minimum version checks on downstream libraries.
    __version__ = "9999"
