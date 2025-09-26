import importlib

import pytest
from packaging.version import Version

requires_network = pytest.mark.network
requires_minio = pytest.mark.minio
slow_test = pytest.mark.slow


def _importorskip(
    modname: str, minversion: str | None = None
) -> tuple[bool, pytest.MarkDecorator]:
    try:
        mod = importlib.import_module(modname)
        has = True
        if minversion is not None:
            v = getattr(mod, "__version__", "999")
            if Version(v) < Version(minversion):
                raise ImportError("Minimum version not satisfied")
    except ImportError:
        has = False

    reason = f"requires {modname}"
    if minversion is not None:
        reason += f">={minversion}"
    func = pytest.mark.skipif(not has, reason=reason)
    return has, func


has_astropy, requires_astropy = _importorskip("astropy")
has_icechunk, requires_icechunk = _importorskip("icechunk")
has_kerchunk, requires_kerchunk = _importorskip("kerchunk")
has_fastparquet, requires_fastparquet = _importorskip("fastparquet")
has_s3fs, requires_s3fs = _importorskip("s3fs")
has_lithops, requires_lithops = _importorskip("lithops")
has_scipy, requires_scipy = _importorskip("scipy")
has_tifffile, requires_tifffile = _importorskip("tifffile")
has_imagecodecs, requires_imagecodecs = _importorskip("imagecodecs")
has_hdf5plugin, requires_hdf5plugin = _importorskip("hdf5plugin")
has_zarr_python, requires_zarr_python = _importorskip("zarr")
has_dask, requires_dask = _importorskip("dask")
has_obstore, requires_obstore = _importorskip("obstore")
