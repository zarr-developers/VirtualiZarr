from virtualizarr.testing.utils import fake_parser, put_fake_data
from virtualizarr.v2.api import open_virtual_dataset

from . import requires_obstore


@requires_obstore
def test_open_virtual_dataset(tmpdir):
    import obstore as obs

    store = obs.store.LocalStore()

    filepath = f"{tmpdir}/data.tmp"
    put_fake_data(store, filepath=filepath)
    assert open_virtual_dataset(
        filepath=filepath, object_reader=store, parser=fake_parser
    )
