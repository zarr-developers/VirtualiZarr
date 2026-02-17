import numpy as np
import pytest
import xarray as xr
from obspec_utils.registry import ObjectStoreRegistry
from obstore.store import LocalStore, S3Store
from xarray import Dataset, DataTree

from virtualizarr import open_virtual_dataset, open_virtual_datatree
from virtualizarr.tests import requires_network, requires_tiff, requires_tifffile

virtual_tiff = pytest.importorskip("virtual_tiff")


@requires_tiff
@requires_network
def test_virtual_tiff_datatree() -> None:
    store = S3Store("sentinel-cogs", region="us-west-2", skip_signature=True)
    registry = ObjectStoreRegistry({"s3://sentinel-cogs/": store})
    url = "s3://sentinel-cogs/sentinel-s2-l2a-cogs/12/S/UF/2022/6/S2B_12SUF_20220609_0_L2A/B04.tif"
    parser = virtual_tiff.VirtualTIFF(ifd_layout="nested")
    with open_virtual_datatree(url=url, parser=parser, registry=registry) as vdt:
        assert isinstance(vdt, DataTree)
        assert list(vdt["0"].ds.variables) == ["0"]
        var = vdt["0"].ds["0"].variable
        assert var.sizes == {"y": 10980, "x": 10980}
        assert var.dtype == "<u2"
        var = vdt["1"].ds["1"].variable
        assert var.sizes == {"y": 5490, "x": 5490}
        assert var.dtype == "<u2"


@requires_tiff
@requires_network
def test_virtual_tiff_dataset() -> None:
    store = S3Store("sentinel-cogs", region="us-west-2", skip_signature=True)
    registry = ObjectStoreRegistry({"s3://sentinel-cogs/": store})
    url = "s3://sentinel-cogs/sentinel-s2-l2a-cogs/12/S/UF/2022/6/S2B_12SUF_20220609_0_L2A/B04.tif"
    parser = virtual_tiff.VirtualTIFF(ifd=0)
    with open_virtual_dataset(url=url, parser=parser, registry=registry) as vds:
        assert isinstance(vds, Dataset)
        assert list(vds.variables) == ["0"]
        var = vds["0"].variable
        assert var.sizes == {"y": 10980, "x": 10980}
        assert var.dtype == "<u2"


@requires_tiff
@requires_tifffile
def test_concat_rectilinear_tiff_datasets(tmp_path) -> None:
    """Test concatenating two virtual TIFF datasets with rectilinear chunk grids.

    Creates stripped TIFFs where image_height is not evenly divisible by rows_per_strip,
    producing rectilinear chunks, then verifies they can be concatenated.
    """
    import tifffile

    # Create two stripped TIFFs where image_height (100) is not evenly divisible
    # by rows_per_strip (30), creating rectilinear chunks: [[30, 30, 30, 10], [50]]
    shape = (100, 50)
    rows_per_strip = 30

    filepath1 = tmp_path / "test1.tif"
    filepath2 = tmp_path / "test2.tif"

    tifffile.imwrite(
        str(filepath1), np.ones(shape, dtype=np.uint8), rowsperstrip=rows_per_strip
    )
    tifffile.imwrite(
        str(filepath2), np.ones(shape, dtype=np.uint8) * 2, rowsperstrip=rows_per_strip
    )

    parser = virtual_tiff.VirtualTIFF(ifd=0)
    registry = ObjectStoreRegistry({"file://": LocalStore()})

    with (
        open_virtual_dataset(
            url=f"file://{filepath1}", parser=parser, registry=registry
        ) as vds1,
        open_virtual_dataset(
            url=f"file://{filepath2}", parser=parser, registry=registry
        ) as vds2,
    ):
        # Verify both datasets have the expected shape
        assert vds1["0"].sizes == {"y": 100, "x": 50}
        assert vds2["0"].sizes == {"y": 100, "x": 50}

        # Verify both datasets have rectilinear chunk grids
        from zarr.core.chunk_grids import RectilinearChunkGrid

        assert isinstance(
            vds1["0"].variable.data.metadata.chunk_grid, RectilinearChunkGrid
        )
        assert isinstance(
            vds2["0"].variable.data.metadata.chunk_grid, RectilinearChunkGrid
        )

        # Concatenate along a new dimension
        combined = xr.concat([vds1, vds2], dim="time")

        assert isinstance(combined, Dataset)
        assert combined["0"].sizes == {"time": 2, "y": 100, "x": 50}
