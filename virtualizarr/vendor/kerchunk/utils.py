import base64

import fsspec
import ujson
import zarr


def _encode_for_JSON(store):
    """Make store JSON encodable"""
    for k, v in store.copy().items():
        if isinstance(v, list):
            continue
        else:
            try:
                # minify JSON
                v = ujson.dumps(ujson.loads(v))
            except (ValueError, TypeError):
                pass
            try:
                store[k] = v.decode() if isinstance(v, bytes) else v
            except UnicodeDecodeError:
                store[k] = "base64:" + base64.b64encode(v).decode()
    return store


def _inline_array(group, threshold, names, prefix=""):
    for name, thing in group.items():
        if prefix:
            prefix1 = f"{prefix}.{name}"
        else:
            prefix1 = name
        if isinstance(thing, zarr.Group):
            _inline_array(thing, threshold=threshold, prefix=prefix1, names=names)
        else:
            cond1 = threshold and thing.nbytes < threshold
            cond2 = prefix1 in names
            if cond1 or cond2:
                original_attrs = dict(thing.attrs)
                arr = group.create_dataset(
                    name=name,
                    dtype=thing.dtype,
                    shape=thing.shape,
                    data=thing[:],
                    chunks=thing.shape,
                    compression=None,
                    overwrite=True,
                    fill_value=thing.fill_value,
                )
                arr.attrs.update(original_attrs)


def inline_array(store, threshold=1000, names=None, remote_options=None):
    """Inline whole arrays by threshold or name, replace with a single metadata chunk

    Inlining whole arrays results in fewer keys. If the constituent keys were
    already inlined, this also results in a smaller file overall. No action is taken
    for arrays that are already of one chunk (they should be in

    Parameters
    ----------
    store: dict/JSON file
        reference set
    threshold: int
        Size in bytes below which to inline. Set to 0 to prevent inlining by size
    names: list[str] | None
        It the array name (as a dotted full path) appears in this list, it will
        be inlined irrespective of the threshold size. Useful for coordinates.
    remote_options: dict | None
        Needed to fetch data, if the required keys are not already individually inlined
        in the data.

    Returns
    -------
    amended references set (simple style)
    """
    fs = fsspec.filesystem(
        "reference", fo=store, **(remote_options or {}), skip_instance_cache=True
    )
    g = zarr.open_group(fs.get_mapper(), mode="r+")
    _inline_array(g, threshold, names=names or [])
    return fs.references
