import pytest

from virtualizarr import open_virtual_dataset


@pytest.mark.parametrize("inline_threshold, vars_to_inline", [
    (5e2, ['lat', 'lon']),
    pytest.param(5e4, ['lat', 'lon', 'time'], marks=pytest.mark.xfail(reason='time encoding')),
    pytest.param(5e7, ['lat', 'lon', 'time', 'air'], marks=pytest.mark.xfail(reason='scale factor encoding')),
])
def test_numpy_arrays_to_inlined_kerchunk_refs(netcdf4_file, inline_threshold, vars_to_inline):
    from kerchunk.hdf import SingleHdf5ToZarr

    # inline_threshold is chosen to test inlining only the variables listed in vars_to_inline
    expected = SingleHdf5ToZarr(netcdf4_file, spec=1, inline_threshold=inline_threshold).translate()

    # loading the variables should produce same result as inlining them using kerchunk
    vds = open_virtual_dataset(netcdf4_file, loadable_variables=vars_to_inline, indexes={})
    refs = vds.virtualize.to_kerchunk(format='dict')

    # TODO I would just compare the entire dicts but kerchunk returns inconsistent results - see https://github.com/TomNicholas/VirtualiZarr/pull/73#issuecomment-2040931202
    #assert refs == expected
    assert refs['refs']['air/0.0.0'] == expected['refs']['air/0.0.0']
    assert refs['refs']['lon/0'] == expected['refs']['lon/0']
    assert refs['refs']['lat/0'] == expected['refs']['lat/0']
    assert refs['refs']['time/0'] == expected['refs']['time/0']
