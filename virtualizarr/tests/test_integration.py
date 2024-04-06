from virtualizarr import open_virtual_dataset


def test_numpy_arrays_to_inlined_kerchunk_refs(netcdf4_file):
    from kerchunk.hdf import SingleHdf5ToZarr

    # test inlining all the variables
    expected = SingleHdf5ToZarr(netcdf4_file, spec=1, inline_threshold=500).translate()

    # loading all the variables should produce same result as inlining them all using kerchunk

    # TODO test the time variable too
    vds = open_virtual_dataset(netcdf4_file, loadable_variables=['lat', 'lon'], indexes={})
    refs = vds.virtualize.to_kerchunk(format='dict')

    # TODO I would just compare the entire dicts but kerchunk returns inconsistent results - see https://github.com/TomNicholas/VirtualiZarr/pull/73#issuecomment-2040931202
    #assert refs == expected
    assert refs['refs']['air/0.0.0'] == expected['refs']['air/0.0.0']
    assert refs['refs']['lon/0'] == expected['refs']['lon/0']
    assert refs['refs']['lat/0'] == expected['refs']['lat/0']
    assert refs['refs']['time/0'] == expected['refs']['time/0']
