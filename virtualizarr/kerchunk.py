from typing import NewType, cast

import ujson  # type: ignore

from virtualizarr.zarr import ZArray, ZAttrs

# Distinguishing these via type hints makes it a lot easier to mentally keep track of what the opaque kerchunk "reference dicts" actually mean
# (idea from https://kobzol.github.io/rust/python/2023/05/20/writing-python-like-its-rust.html)
# TODO I would prefer to be more specific about these types
KerchunkStoreRefs = NewType(
    "KerchunkStoreRefs", dict
)  # top-level dict with keys for 'version', 'refs'
KerchunkArrRefs = NewType(
    "KerchunkArrRefs",
    dict,
)  # lower-level dict containing just the information for one zarr array


def find_var_names(ds_reference_dict: KerchunkStoreRefs) -> list[str]:
    """Find the names of zarr variables in this store/group."""

    refs = ds_reference_dict["refs"]
    found_var_names = {key.split("/")[0] for key in refs.keys() if "/" in key}
    return list(found_var_names)


def extract_array_refs(
    ds_reference_dict: KerchunkStoreRefs, var_name: str
) -> KerchunkArrRefs:
    """Extract only the part of the kerchunk reference dict that is relevant to this one zarr array"""

    found_var_names = find_var_names(ds_reference_dict)

    refs = ds_reference_dict["refs"]
    if var_name in found_var_names:
        # TODO these function probably have more loops in them than they need to...

        arr_refs = {
            key.split("/")[1]: refs[key]
            for key in refs.keys()
            if var_name == key.split("/")[0]
        }

        return fully_decode_arr_refs(arr_refs)
    else:
        raise KeyError(
            f"Could not find zarr array variable name {var_name}, only {found_var_names}"
        )


def parse_array_refs(
    arr_refs: KerchunkArrRefs,
) -> tuple[dict, ZArray, ZAttrs]:
    zarray = ZArray.from_kerchunk_refs(arr_refs.pop(".zarray"))
    zattrs = arr_refs.pop(".zattrs", {})
    chunk_dict = arr_refs

    return chunk_dict, zarray, zattrs


def fully_decode_arr_refs(d: dict) -> KerchunkArrRefs:
    """
    Only have to do this because kerchunk.SingleHdf5ToZarr apparently doesn't bother converting .zarray and .zattrs contents to dicts, see https://github.com/fsspec/kerchunk/issues/415 .
    """
    sanitized = d.copy()
    for k, v in d.items():
        if k.startswith("."):
            # ensure contents of .zattrs and .zarray are python dictionaries
            sanitized[k] = ujson.loads(v)

    return cast(KerchunkArrRefs, sanitized)
