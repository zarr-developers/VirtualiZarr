# HDF5 links and object references

HDF5 files can connect their objects in ways that Zarr has no built-in equivalent for:

- **Object references**: a value that points at another group or dataset. They can be stored as a dataset, as an attribute, or as a field of a compound dataset.
- **Soft links**: a second name for an object, given as a path, which may or may not exist.
- **External links**: a name that points at an object in another HDF5 file.
- **Hard links**: one object stored under more than one name.

This page shows how to keep these connections when you virtualize an HDF5 file, and how to follow them when you read the result.

## What happens by default

By default, [`HDFParser`][virtualizarr.parsers.HDFParser] doesn't record any of these connections:

- A dataset of object references raises `NotImplementedError`.
- An attribute that holds an object reference can't be written to JSON.
- A soft link or a second hard link is followed and its target copied, so it becomes a separate copy of the array or group, with nothing to say it was a link.
- An external link can't be followed, because the parser reads only the one file, so parsing fails.

Pass `links_and_references=True` to record these connections instead. They are written in the attributes of the Zarr hierarchy, following two conventions:

- the [`ref` convention](https://github.com/R-CF/zarr_convention_ref), whose objects identify a node of a Zarr hierarchy;
- the draft [Links and Object References convention](https://github.com/catalystneuro/zarr-links-references), which builds on `ref` to describe HDF5 links and reference datasets.

The flag is off by default while the Links and Object References convention is still a draft.

## An example file

This file has each kind of connection. It holds an array of measurements and a table of the sites they were taken at, and has an external link to another file.

```python exec="on" session="hdf5_links" source="above"
import tempfile
from pathlib import Path

import h5py
import numpy as np

directory = Path(tempfile.mkdtemp())

with h5py.File(directory / "source.h5", "w") as f:
    f["readings/values"] = np.zeros((100, 4), dtype="int16")

with h5py.File(directory / "file.h5", "w") as f:
    sites = f.create_group("metadata/sites")
    sites["name"] = ["A", "B", "C"]
    f["measurements"] = np.arange(3.0)

    # An object reference in an attribute
    f["measurements"].attrs["sites"] = sites.ref
    # A dataset of object references, with a null reference at the end
    f["sources"] = np.array(
        [sites.ref, f["measurements"].ref, h5py.Reference()], dtype=h5py.ref_dtype
    )
    # A soft link, an external link, and a second hard link
    f["derived/measurements"] = h5py.SoftLink("/measurements")
    f["derived/source"] = h5py.ExternalLink("source.h5", "/readings")
    f["views/sites"] = sites
```

Without the flag, the file can't be parsed. Leaving out the group with the external link, the dataset of references still raises an error:

```python exec="on" session="hdf5_links" source="above" result="code"
from obspec_utils.registry import ObjectStoreRegistry
from obstore.store import LocalStore

from virtualizarr.parsers import HDFParser

url = f"file://{directory}/file.h5"
registry = ObjectStoreRegistry({url: LocalStore(prefix=directory)})

try:
    HDFParser(drop_variables=["derived"])(url=url, registry=registry)
except NotImplementedError as error:
    print(error)
```

## Recording links and references

With `links_and_references=True` the file can be parsed, and the result can be opened like any other Zarr hierarchy:

```python exec="on" session="hdf5_links" source="above"
import json

import zarr

store = HDFParser(links_and_references=True)(url=url, registry=registry)
root = zarr.open_group(store, mode="r", zarr_format=3)
```

### Links

A group's links are recorded in its attributes under `"links_references"`, and are not copied:

```python exec="on" session="hdf5_links" source="above" result="code"
print(json.dumps(root["derived"].attrs["links_references"], indent=2))
print(sorted(root["derived"].keys()))
```

Each link is a `ref` object:

- A `node` with no `uri` is in the same hierarchy. Its path is relative to the group that holds the link.
- An external link also has a `uri`, the location of the other file. A file name in the external link is resolved against the location of the file being parsed, so the `uri` here is an absolute path. The `node` of an external link is the absolute path of its target in that file.

A soft link's target doesn't have to exist. A link whose target is missing is still recorded, as HDF5 allows.

When an object has more than one hard link, its data is stored under the first name the parser reaches. Each other name becomes a link to that one:

```python exec="on" session="hdf5_links" source="above" result="code"
print(json.dumps(root["views"].attrs["links_references"], indent=2))
```

The group also lists the conventions it follows in `zarr_conventions`:

```python exec="on" session="hdf5_links" source="above" result="code"
print([convention["name"] for convention in root["derived"].attrs["zarr_conventions"]])
```

### Datasets of object references

A dataset of object references becomes an array of strings. Each string is the path of a reference's target, relative to the array. A null reference is the empty string:

```python exec="on" session="hdf5_links" source="above" result="code"
print(root["sources"][:])
print(root["sources"].attrs["links_references"])
```

The paths are found when the file is parsed, and stored with the rest of the metadata as inlined chunks. Reading them later doesn't need h5py or the HDF5 file.

### Attributes that hold object references

An attribute that holds an object reference becomes a `ref` object. Its path is relative to the array or group that has the attribute. An attribute that holds several references becomes a list of `ref` objects, with `null` for a null reference.

```python exec="on" session="hdf5_links" source="above" result="code"
print(root["measurements"].attrs["sites"])
```

Region references, which point at part of a dataset, are not converted.

### Compound datasets

A compound dataset whose fields hold object references is read into a structured array. The reference fields hold the paths of their targets, like a dataset of object references. The array's `"links_references"` attribute lists which fields are references.

## Following a link or reference

A `node` path is resolved like a relative path in a file system, starting from the node that holds it. A link or attribute starts from its group or array, and a value of a reference dataset starts from its array. This function finds the absolute path of a target within the same hierarchy:

```python exec="on" session="hdf5_links" source="above" result="code"
import posixpath


def resolve(holder: str, node: str) -> str:
    """The absolute path of a target, given the path of the node that refers to it."""
    return posixpath.normpath(posixpath.join("/", holder, node))


link = root["derived"].attrs["links_references"]["links"]["measurements"]["ref"]
print(resolve("derived", link["node"]))
print(root[resolve("derived", link["node"])][:])

reference = root["measurements"].attrs["sites"]["ref"]
print(resolve("measurements", reference["node"]))
print(root[resolve("measurements", reference["node"])]["name"][:])
```

For a link with a `uri`, open the file at that `uri` first, then look up the absolute `node` path in it.

## Writing and reading the result

The links and references are ordinary JSON attributes and inlined chunks. They are kept when the virtual dataset is written to Kerchunk references or to Icechunk.

Readers that don't know the conventions still read every array and attribute. They just won't follow the links. A reader in any language can follow them using only the JSON metadata, the same way `resolve` does above.
