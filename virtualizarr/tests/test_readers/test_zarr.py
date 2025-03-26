import numpy as np
import pytest

from virtualizarr import open_virtual_dataset
from virtualizarr.manifests import ManifestArray
from virtualizarr.tests import requires_network


@requires_network
@pytest.mark.parametrize(
    "zarr_store",
    [
        pytest.param(
            2,
            id="Zarr V2",
            marks=pytest.mark.skip(reason="Zarr V2 not currently supported."),
        ),
        pytest.param(3, id="Zarr V3"),
    ],
    indirect=True,
)
class TestOpenVirtualDatasetZarr:
    def test_loadable_variables(self, zarr_store, loadable_variables=["time", "air"]):
        # check that loadable variables works
        vds = open_virtual_dataset(
            filepath=zarr_store, loadable_variables=loadable_variables
        )
        assert isinstance(vds["time"].data, np.ndarray)
        assert isinstance(vds["air"].data, np.ndarray), type(vds["air"].data)

    def test_drop_variables(self, zarr_store, drop_variables=["air"]):
        # check variable is dropped
        vds = open_virtual_dataset(filepath=zarr_store, drop_variables=drop_variables)
        assert len(vds.data_vars) == 0

    def test_manifest_indexing(self, zarr_store):
        vds = open_virtual_dataset(filepath=zarr_store)
        assert "0.0.0" in vds["air"].data.manifest.dict().keys()

    def test_virtual_dataset_zarr_attrs(self, zarr_store):
        import zarr

        zg = zarr.open_group(zarr_store)
        vds = open_virtual_dataset(filepath=zarr_store)

        non_var_arrays = ["time", "lat", "lon"]

        # check dims and coords are present
        assert set(vds.coords) == set(non_var_arrays)
        assert set(vds.dims) == set(non_var_arrays)
        # check vars match
        assert set(vds.keys()) == set(["air"])

        # check top level attrs
        assert zg.attrs.asdict() == vds.attrs

        arrays = [val for val in zg.keys()]

        # arrays are ManifestArrays
        for array in arrays:
            # check manifest array ArrayV3Metadata dtype
            assert isinstance(vds[array].data, ManifestArray)

            # compare manifest array ArrayV3Metadata
            assert zg[array].metadata == vds[array].data.metadata
