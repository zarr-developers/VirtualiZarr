import numpy as np
import pytest

from virtualizarr import open_virtual_dataset
from virtualizarr.manifests import ManifestArray
from virtualizarr.parsers import ZarrParser
from virtualizarr.parsers.zarr import get_chunk_mapping_prefix
from virtualizarr.tests.utils import obstore_local


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
        # check loadable variables
        print(zarr_store)
        store = obstore_local(zarr_store)
        parser = ZarrParser()
        vds = open_virtual_dataset(
            file_url=zarr_store,
            object_store=store,
            parser=parser,
            loadable_variables=loadable_variables,
        )
        assert isinstance(vds["time"].data, np.ndarray)
        assert isinstance(vds["air"].data, np.ndarray), type(vds["air"].data)

    def test_drop_variables(self, zarr_store, drop_variables=["air"]):
        store = obstore_local(zarr_store)
        parser = ZarrParser(drop_variables=drop_variables)
        # check variable is dropped
        vds = open_virtual_dataset(
            file_url=zarr_store,
            object_store=store,
            parser=parser,
        )
        assert len(vds.data_vars) == 0

    def test_manifest_indexing(self, zarr_store):
        store = obstore_local(zarr_store)
        parser = ZarrParser()
        vds = open_virtual_dataset(
            file_url=zarr_store,
            object_store=store,
            parser=parser,
        )
        assert "0.0.0" in vds["air"].data.manifest.dict().keys()

    def test_virtual_dataset_zarr_attrs(self, zarr_store):
        import zarr

        zg = zarr.open_group(zarr_store)
        store = obstore_local(zarr_store)
        parser = ZarrParser()
        vds = open_virtual_dataset(
            file_url=zarr_store,
            object_store=store,
            parser=parser,
            loadable_variables=[],
        )

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
            expected = zg[array].metadata.to_dict()
            # Check attributes
            assert expected["attributes"] == vds[array].attrs
            assert expected["dimension_names"] == vds[array].dims
            expected.pop(
                "dimension_names"
            )  # dimension_names are removed in conversion to virtual variable
            expected[
                "attributes"
            ] = {}  # attributes are removed in conversion to virtual variable
            actual = vds[array].data.metadata.to_dict()
            assert expected == actual


def test_scalar_get_chunk_mapping_prefix(zarr_store_scalar):
    # Use a scalar zarr store with a /c/ representing the scalar:
    # https://zarr-specs.readthedocs.io/en/latest/v3/chunk-key-encodings/default/index.html#description

    import asyncio

    chunk_map = asyncio.run(
        get_chunk_mapping_prefix(
            zarr_array=zarr_store_scalar, filepath=str(zarr_store_scalar.store_path)
        )
    )
    assert chunk_map["c"]["offset"] == 0
    assert chunk_map["c"]["length"] == 10
