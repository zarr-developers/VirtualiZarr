import numpy as np
from obstore.store import ObjectStore

from virtualizarr.manifests import (
    ChunkManifest,
    ManifestArray,
    ManifestGroup,
    ManifestStore,
)
from virtualizarr.manifests.utils import create_v3_array_metadata


def put_fake_data(store: ObjectStore, filepath: str):
    """
    Puts a sequence of 16 bytes in a file, which can simulate storing
    4 contiguous uncompressed 4-byte chunks (or 8 2-byte chunks, etc). This
    provides an easily understandable structure for testing ManifestStore's
    ability to redirect Zarr chunk key requests and extract subsets of the file.

    Parameters:
    -----------
    store : ObjectStore
        ObjectStore instance for holding the file
    filepath : str
        Filepath for storing temporary testing file
    """
    import obstore as obs

    obs.put(
        store,
        filepath,
        b"\x01\x02\x03\x04\x05\x06\x07\x08\x09\x10\x11\x12\x13\x14\x15\x16",
    )


def fake_parser(filepath: str, object_reader: ObjectStore) -> ManifestStore:
    """
    Generate a ManifestStore for testing.

    This puts a sequence of 16 bytes in a file, which can simulate storing
    4 contiguous uncompressed 4-byte chunks (or 8 2-byte chunks, etc). This
    provides an easily understandable structure for testing ManifestStore's
    ability to redirect Zarr chunk key requests and extract subsets of the file.

    Parameters:
    -----------
    filepath : str
        Filepath for storing temporary testing file
    store : ObjectStore
        ObjectStore instance for holding the file
    Returns:
    --------
    ManifestStore
    """
    chunk_dict = {
        "0.0": {"path": filepath, "offset": 0, "length": 4},
        "0.1": {"path": filepath, "offset": 4, "length": 4},
        "1.0": {"path": filepath, "offset": 8, "length": 4},
        "1.1": {"path": filepath, "offset": 12, "length": 4},
    }
    manifest = ChunkManifest(entries=chunk_dict)
    codecs = [{"configuration": {"endian": "little"}, "name": "bytes"}]
    array_metadata = create_v3_array_metadata(
        shape=(4, 4),
        chunk_shape=(2, 2),
        data_type=np.dtype("int32"),
        codecs=codecs,
        chunk_key_encoding={"name": "default", "separator": "."},
        fill_value=0,
        dimension_names=("x", "y"),
    )
    manifest_array = ManifestArray(metadata=array_metadata, chunkmanifest=manifest)
    manifest_group = ManifestGroup(
        arrays={"foo": manifest_array, "bar": manifest_array},
        attributes={"Zarr": "Hooray!"},
    )
    return ManifestStore(store=object_reader, group=manifest_group)
