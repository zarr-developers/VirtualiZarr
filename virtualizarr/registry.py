import warnings

from obspec_utils import ObjectStoreRegistry

warnings.warn(
    "Importing ObjectStoreRegistry from VirtualiZarr is deprecated. "
    "Please use 'from obspec_utils import ObjectStoreRegistry instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["ObjectStoreRegistry"]
