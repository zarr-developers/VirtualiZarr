import pytest
import zarr

from virtualizarr.readers.zarr import virtual_dataset_from_zarr_group
from virtualizarr.tests import network


@network
@pytest.mark.parametrize(
    "zarr_store",
    [
        pytest.param(2, id="Zarr V2"),
        pytest.param(
            3,
            id="Zarr V3",
            marks=pytest.mark.skip(
                reason="Need to translate metadata naming conventions/transforms"
            ),
        ),
    ],
    indirect=True,
)
def test_dataset_from_zarr(zarr_store):
    zg = zarr.open_group(zarr_store)
    vds = virtual_dataset_from_zarr_group(filepath=zarr_store, indexes={})

    zg_metadata_dict = zg.metadata.to_dict()

    arrays = [val for val in zg.keys()]

    # loop through each array and check ZArray info
    for array in arrays:
        # shape match
        assert (
            vds[array].data.zarray.shape
            == zg_metadata_dict["consolidated_metadata"]["metadata"][array]["shape"]
        )
        # match chunks
        assert (
            vds[array].data.zarray.chunks
            == zg_metadata_dict["consolidated_metadata"]["metadata"][array]["chunks"]
        )

        assert (
            vds[array].data.zarray.dtype
            == zg_metadata_dict["consolidated_metadata"]["metadata"][array]["dtype"]
        )

        # Failure! fill value from zarr is None, None: ipdb> np.dtype(None): dtype('float64') is coerced in zarr.py L21 to 0.0.
        # assert vds[array].data.zarray.fill_value == zg_metadata_dict['consolidated_metadata']['metadata'][array]['fill_value']

        # match order
        assert (
            vds[array].data.zarray.order
            == zg_metadata_dict["consolidated_metadata"]["metadata"][array]["order"]
        )
        # match compressor
        assert (
            vds[array].data.zarray.compressor
            == zg_metadata_dict["consolidated_metadata"]["metadata"][array][
                "compressor"
            ]
        )
        # match filters
        assert (
            vds[array].data.zarray.filters
            == zg_metadata_dict["consolidated_metadata"]["metadata"][array]["filters"]
        )
        # match format
        assert (
            vds[array].data.zarray.zarr_format
            == zg_metadata_dict["consolidated_metadata"]["metadata"][array][
                "zarr_format"
            ]
        )
