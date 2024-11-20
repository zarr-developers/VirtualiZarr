import pytest
import zarr

from virtualizarr.manifests import ManifestArray
from virtualizarr.readers.zarr import virtual_dataset_from_zarr_group
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
def test_dataset_from_zarr(zarr_store):
    zg = zarr.open_group(zarr_store)
    vds = virtual_dataset_from_zarr_group(filepath=zarr_store, indexes={})

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
    ]  # "dtype"

    # Failure! fill value from zarr is None, None: ipdb> np.dtype(None): dtype('float64') is coerced in zarr.py L21 to 0.0.
    # assert vds[array].data.zarray.fill_value == zg_metadata_dict['consolidated_metadata']['metadata'][array]['fill_value']

    # loop through each array and check ZArray info
    for array in arrays:
        for attr in zarray_checks:
            assert (
                getattr(vds[array].data.zarray, attr)
                == zg_metadata_dict["consolidated_metadata"]["metadata"][array][attr]
            )
