import numpy as np
import pytest

from virtualizarr import open_virtual_dataset
from virtualizarr.manifests import ManifestArray
from virtualizarr.tests import requires_network, requires_zarr_python_v3

# It seems like this PR: https://github.com/zarr-developers/zarr-python/pull/2533
# might fix this issue: https://github.com/zarr-developers/zarr-python/issues/2554


@requires_zarr_python_v3
@requires_network
@pytest.mark.parametrize(
    "zarr_store",
    [
        pytest.param(
            2,
            id="Zarr V2",
        ),
        pytest.param(
            3,
            id="Zarr V3",
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

    def test_virtual_dataset_zarr_attrs(self, zarr_store):
        import zarr

        zg = zarr.open_group(zarr_store)
        vds = open_virtual_dataset(filepath=zarr_store, indexes={})
        zg_metadata_dict = zg.metadata.to_dict()
        zarr_format = zg_metadata_dict["zarr_format"]

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

        shared_v2_v3_attrs = [
            "shape",
            "zarr_format",
        ]
        v2_attrs = ["chunks", "dtype", "order", "compressor", "filters"]

        def _validate_v2(attrs: list[str]):
            for array in arrays:
                for attr in attrs:
                    vds_attr = getattr(vds[array].data.zarray, attr)
                    zarr_metadata_attr = zg_metadata_dict["consolidated_metadata"][
                        "metadata"
                    ][array][attr]
                    import ipdb

                    ipdb.set_trace()
                    assert vds_attr == zarr_metadata_attr

        def _validate_v3(attrs: list[str]):
            # check v2, v3 shared attrs
            for array in arrays:
                for attr in attrs:
                    zarr_metadata_attr = zg_metadata_dict["consolidated_metadata"][
                        "metadata"
                    ][array][attr]
                    vds_attr = getattr(vds[array].data.zarray, attr)
                    assert vds_attr == zarr_metadata_attr

            # Cases where v2 and v3 attr keys differ: order, compressor, filters, dtype & chunks

            # chunks vs chunk_grid.configuration.chunk_shape
            assert (
                getattr(vds[array].data.zarray, "chunks")
                == zg_metadata_dict["consolidated_metadata"]["metadata"][array][
                    "chunk_grid"
                ]["configuration"]["chunk_shape"]
            )

            # dtype vs datatype
            assert (
                getattr(vds[array].data.zarray, "dtype")
                == zg_metadata_dict["consolidated_metadata"]["metadata"][array][
                    "data_type"
                ].to_numpy()
            )

            # order: In zarr v3, it seems like order was replaced with the transpose codec.

        if zarr_format == 2:
            _validate_v2(shared_v2_v3_attrs + v2_attrs)

        elif zarr_format == 3:
            _validate_v3(shared_v2_v3_attrs)

        else:
            raise NotImplementedError(f"Zarr format {zarr_format} not in [2,3]")
