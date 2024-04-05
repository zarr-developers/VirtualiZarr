from virtualizarr import open_virtual_dataset

from pprint import pprint


def test_numpy_arrays_to_inlined_kerchunk_refs(netcdf4_file):
    from kerchunk.hdf import SingleHdf5ToZarr

    # test inlining all the variables
    expected = SingleHdf5ToZarr(netcdf4_file, spec=1, inline_threshold=1e9).translate()

    pprint(expected)

    # loading all the variables should produce same result as inlining them all using kerchunk
    vds = open_virtual_dataset(netcdf4_file, loadable_variables=['air', 'time', 'lat', 'lon'], indexes={})
    refs = vds.virtualize.to_kerchunk(format='dict')

    assert refs == expected
