from typing import NewType

# Distinguishing these via type hints makes it a lot easier to mentally keep track of what the opaque kerchunk "reference dicts" actually mean
# (idea from https://kobzol.github.io/rust/python/2023/05/20/writing-python-like-its-rust.html)
# TODO I would prefer to be more specific about these types
KerchunkStoreRefs = NewType(
    "KerchunkStoreRefs",
    dict,  # dict_keys(['version', 'refs'])
)  # top-level dict containing kerchunk version and 'refs' dictionary which assumes single '.zgroup' key and multiple KerchunkArrRefs
KerchunkArrRefs = NewType(
    "KerchunkArrRefs",
    dict,  # dict_keys(['.zarray', '.zattrs', '0.0', '0.1', ...)
)  # lower-level dict defining a single Zarr Array, with keys for '.zarray', '.zattrs', and every chunk
