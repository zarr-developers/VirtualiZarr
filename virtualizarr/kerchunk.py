from typing import NewType, Literal, Dict
import json


from virtualizarr.types import ZArray, ZAttrs


# Distinguishing these via type hints makes it a lot easier to keep track of what the opaque kerchunk "reference dicts" actually mean
# (idea from https://kobzol.github.io/rust/python/2023/05/20/writing-python-like-its-rust.html)
KerchunkStoreRefs = NewType(
    "KerchunkStoreRefs", dict[Literal["version"] | Literal["refs"], int | dict]
)  # top-level dict with keys for 'version', 'refs'
KerchunkArrRefs = NewType(
    "KerchunkArrRefs",
    dict[Literal[".zattrs"], ZAttrs] | dict[Literal[".zarray"], ZArray] | dict[str, str],
)  # lower-level dict containing just the information for one zarr array


def find_var_names(ds_reference_dict: KerchunkStoreRefs) -> list[str]:
    """Find the names of zarr variables in this store/group."""
    
    refs = ds_reference_dict['refs']
    found_var_names = [key.split('/')[0] for key in refs.keys() if '/' in key]
    return found_var_names


def extract_array_refs(ds_reference_dict: KerchunkStoreRefs, var_name: str) -> tuple[KerchunkArrRefs, ZAttrs]:
    """Extract only the part of the kerchunk reference dict that is relevant to this one zarr array"""
    
    found_var_names = find_var_names(ds_reference_dict)

    refs = ds_reference_dict['refs']
    if var_name in found_var_names:
        var_refs = {key.split('/')[1]: refs[key] for key in refs.keys() if var_name == key.split('/')[0]}

        zattrs = var_refs.pop('.zattrs')  # we are going to store these separately later
        
        return var_refs, zattrs
    else:
        raise KeyError(f"Could not find zarr array variable name {var_name}, only {found_var_names}")
    

def fully_decode_arr_refs(d: KerchunkArrRefs) -> KerchunkArrRefs:
    """
    Only have to do this because kerchunk.SingleHdf5ToZarr apparently doesn't bother converting .zarray and .zattrs contents to dicts, see https://github.com/fsspec/kerchunk/issues/415 .
    """
    sanitized = d.copy()
    for k, v in d.items():
        
        if k.startswith('.'):
            # ensure contents of .zattrs and .zarray are python dictionaries
            sanitized[k] = json.loads(v)
        # TODO should we also convert the byte range values stored under chunk keys to python lists? e.g. 'time/0': ['air.nc', 7757515, 11680]
    
    return sanitized
