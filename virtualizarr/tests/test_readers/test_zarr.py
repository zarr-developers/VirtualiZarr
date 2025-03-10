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

        from virtualizarr.zarr import ZARR_DEFAULT_FILL_VALUE, ZArray

        zg = zarr.open_group(zarr_store)
        vds = open_virtual_dataset(filepath=zarr_store, indexes={})

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

        def _validate_attr_match(
            array: str, zarr_array: zarr.Array | zarr.AsyncArray, zarray: ZArray
        ):
            zarr_array_fill_value = zarr_array.fill_value  # type: ignore[union-attr]

            if zarr_array_fill_value:
                zarr_array_fill_value = ZARR_DEFAULT_FILL_VALUE[
                    zarr_array_fill_value.dtype.kind
                ]
            else:
                zarr_array_fill_value = 0
            zarr_array_filters = (
                zarr_array.filters if zarr_array.filters else None
            )  # if tuple is empty, assign filters to None to match ZArray def

            assert zarr_array.shape == zarray.shape, (
                f"Mismatch in [shape] for {array} between Zarr Array: {zarr_array.shape} and ZArray: {zarray.shape}"
            )
            assert zarr_array.chunks == zarray.chunks, (
                f"Mismatch in [chunks] for {array} between Zarr Array: {zarr_array.chunks} and ZArray: {zarray.chunks}"
            )
            assert zarr_array.dtype == zarray.dtype, (
                f"Mismatch in [dtype] between Zarr Array: {zarr_array.dtype} and ZArray: {(zarray.dtype,)}"
            )
            assert zarr_array_fill_value == zarray.fill_value, (
                f"Mismatch in [fill_value] for {array} between Zarr Array: {zarr_array_fill_value} and ZArray: {zarray.fill_value}"
            )
            assert zarr_array.order == zarray.order, (
                f"Mismatch in [order] for {array} between Zarr Array: {zarr_array.order} and ZArray: {zarray.order}"
            )
            assert zarr_array_filters == zarray.filters, (
                f"Mismatch in [filters] for {array} between Zarr Array: {zarr_array_filters} and ZArray: {(zarray.filters,)}"
            )
            assert zarr_array.metadata.zarr_format == zarray.zarr_format, (
                f"Mismatch in [zarr_format] for {array} between Zarr Array: {zarr_array.metadata.zarr_format} and ZArray: {(zarray.zarr_format,)}"
            )

            if zarr_array.metadata.zarr_format == 2:
                zarr_array_compressor = zarr_array.compressor.get_config()  # type: ignore[union-attr]
            elif zarr_array.metadata.zarr_format == 3:
                zarr_array_compressor = zarr_array.compressors[0].to_dict()
            else:
                raise NotImplementedError(
                    f"Zarr format {zarr_array.metadata.zarr_format} not in [2,3]"
                )

            assert zarr_array_compressor == zarray.compressor, (
                f"Mismatch in [compressor] for {array} between Zarr Array: {zarr_array_compressor} and ZArray: {zarray.compressor}"
            )

        [
            _validate_attr_match(
                array=array, zarr_array=zg[array], zarray=vds[array].data.zarray
            )
            for array in arrays
        ]
