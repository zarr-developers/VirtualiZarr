import numpy as np
import pytest
import zarr

from virtualizarr import open_virtual_dataset
from virtualizarr.manifests import ManifestArray
from virtualizarr.tests import network, requires_zarrV3


@requires_zarrV3
@network
@pytest.mark.parametrize(
    "zarr_store",
    [
        pytest.param(2, id="Zarr V2"),
        pytest.param(
            3,
            id="Zarr V3",
            marks=pytest.mark.skip(
                reason="ToDo/WIP: Need to translate metadata naming conventions/transforms"
            ),
        ),
    ],
    indirect=True,
)
class TestOpenVirtualDatasetZarr:
    def test_loadable_variables(self, zarr_store, loadable_variables=["time", "air"]):
        # check that loadable variables works
        vds = open_virtual_dataset(
            filepath=zarr_store, loadable_variables=loadable_variables, indexes={}
        )
        assert isinstance(vds["time"].data, np.ndarray)
        assert isinstance(vds["air"].data, np.ndarray)

    def test_drop_variables(self, zarr_store, drop_variables=["air"]):
        # check variable is dropped
        vds = open_virtual_dataset(
            filepath=zarr_store, drop_variables=drop_variables, indexes={}
        )
        assert len(vds.data_vars) == 0

    def test_virtual_dataset_from_zarr_group(self, zarr_store):
        # check that loadable variables works

        zg = zarr.open_group(zarr_store)
        vds = open_virtual_dataset(filepath=zarr_store, indexes={})

        zg_metadata_dict = zg.metadata.to_dict()
        non_var_arrays = ["time", "lat", "lon"]
        # check dims and coords are present
        assert set(vds.coords) == set(non_var_arrays)
        assert set(vds.dims) == set(non_var_arrays)
        # check vars match
        assert set(vds.keys()) == set(["air"])

        # arrays are ManifestArrays
        for array in list(vds):
            assert isinstance(vds[array].data, ManifestArray)

        # check top level attrs
        assert zg.attrs.asdict() == vds.attrs

        # check ZArray values
        arrays = [val for val in zg.keys()]
        zarray_checks = [
            "shape",
            "chunks",
            "dtype",
            "order",
            "compressor",
            "filters",
            "zarr_format",
            "dtype",
        ]
        for array in arrays:
            for attr in zarray_checks:
                assert (
                    getattr(vds[array].data.zarray, attr)
                    == zg_metadata_dict["consolidated_metadata"]["metadata"][array][
                        attr
                    ]
                )
