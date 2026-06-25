# Validation and Cleaning

This page contains tips and best practices for handling the messy inconsistencies typical of real-world datasets during virtual ingestion.

## Data inconsistencies

Assembling a single virtual store from many files requires exploiting some common structure across those files.
Sometimes you know that structure a priori, and sometimes you infer it by looking at the contents of the files.

However sometimes you _think_ you know the structure, only to find upon closer examination that the files aren't as orderly as you assumed.
This can be frustrating, but some best practices can help make it less painful.

## Declarative schema validation

One very helpful approach is to declare your expectations about the files up-front, then validate their actual contents against these expectations as you ingest.

### Validation with Pandera

One good tool for defining and validating schemas is `pandera`, whose `pandera.xarray` module ([docs](https://pandera.readthedocs.io/en/v0.31.1/xarray_guide/index.html)) can be applied to virtual datasets.

As pandera [separates](https://pandera.readthedocs.io/en/v0.31.1/xarray_guide/error_reporting.html#validation-depth-and-what-appears-in-the-report) schema validation from data validation, its default behaviour avoids trying to load virtual chunks, and so it works with virtual datasets out of the box, for example:

```python exec="on" session="usage" source="material-block" result="code"
import numpy as np
import xarray as xr
import pandera.xarray as pa

schema = pa.DatasetSchema(
    data_vars={
        "temperature": pa.DataVar(dtype=np.float64, dims=("x", "y")),
        "pressure": pa.DataVar(dtype=np.float64, dims=("x", "y")),
    },
    coords={"x": pa.Coordinate(dtype=np.float64)},
)

# TODO this should be a virtual dataset
ds = xr.Dataset(
    {
        "temperature": (("x", "y"), np.random.rand(3, 4)),
        "pressure": (("x", "y"), np.random.rand(3, 4)),
    },
    coords={"x": np.arange(3, dtype=np.float64)},
)
schema.validate(ds)
```

This is a very powerful way to double-check your assumptions about your data files.
The best time to check these assumptions is generally _before_ you attempt to combine them.

::: note

    While VirtualiZarr should prevent you combining virtual datasets in ways that cannot form a single valid Zarr store (e.g. by erroring upon concatenation), the reason it's a good idea to perform these kinds of checks _before_ combining is that you get more immediate and informative errors.

    It also makes it easier to iteratively develop a processing pipeline - adjusting expectations of your schema is clearer and more robust than adding more if-else statements to your code.

::: tip

    If you pass `lazy=True` to `schema.validate(ds)`, then instead of `pandera` raising on the first violation of your schema it detects, instead you will get a detailed list of every way in which the dataset fails to conform to your schema.

Note that `pandera` does allow you to check [assertions about data values](https://pandera.readthedocs.io/en/v0.31.1/xarray_guide/checks_and_parsers.html) if you want.
This can be very useful for loadable variables especially - perhaps you want to check that all your files share the same latitude / longitude grid values, or have a datatime coordinate that falls within the expected year.

```python
# TODO example of validating coordinate values
```

One simple but effective approach is to set strict expectations and then scan every single file in your dataset to check for inconsistencies.

### Validation during preprocessing

If you're using `open_virtual_mfdataset`, a neat pattern is to use the `preprocess` function kwarg as your opportunity to do validation.

```python
# TODO example of passing validation function to `preprocess` kwarg
```

### Wrangling to consistency

Virtual ingestion cannot alter the contents of virtual chunks (at least not without duplicating the data), but it can alter almost anything else, such as array metadata or chunk layouts.
Therefore some types of inconsistencies can be corrected during the virtual ingestion process.
Correcting data so that it is more suitable for use is known as data munging or data wrangling.

Another useful pattern is to parse each file, validate its contents against a somewhat permissive schema, then wrangle it until it perfectly fits a more rigid schema, all before combining.
Again this can be done as part of the pre-processing step.

```python
# TODO
```

## Example inconsistencies

### Missing files/variables/chunks

Real-world datasets have missing data.
Zarr's convention for representing missing data elements is to set their value to a specified `fill_value`.

But sometimes there are not just one or two elements missing - entire chunks/variables/files are not present.
In other words the chunks of an array across various files do form a logical grid, but only a sparse grid, not a dense grid.

::: note

    Missing chunks are handled very efficiently by Icechunk - arguably much more efficiently than with the Native Zarr format.

    In Native Zarr, while a chunk key is allowed to be uninitialized, a reader only discovers the absence at read-time, when the store attempts to fetch the chunk key from storage and finds nothing at that path.

    In contrast, since Icechunk records in the manifest whether or not each chunk was initialized, it doesn't need to attempt to fetch a chunk to know whether it exists.

These sparse chunk grids can arise in real datasets for a few reasons:

- Perhaps the data model of the archival files gives fewer guarantees about the chunk grid, and allows chunks to be effectively missing, such as is the case with GRIB.
- Perhaps some variables were not sampled at certain timesteps and ended up being omitted from certain files entirely.
- Perhaps the data is inherently sparse at a global level, whilst still being dense at a regional level (e.g. tree canopy height data would not exist over the oceans). In that case it might make sense for the data files to all be individually complete, and all be tilable onto a global domain, but some files still not be present.

If the chunks do truly align onto a single grid, it is possible to represent the absence of entire virtual chunks, since **the chunk manifest is allowed to be sparse**.

VirtualiZarr provides some utilities for creating virtual datasets with such sparse chunk grids.

The `ManifestArray.with_fill_value_only` method returns a new ManifestArray with the same schema (shape, chunks, codecs, dimension names, attributes) as a given ManifestArray, but with an empty chunk manifest and the given `fill_value`.
This is useful for filling in missing files/variables, by creating virtual datasets containing manifestarrays which have no chunk references.
These empty manifestarrays can then be concatenated with the manifestarrays containing chunk references, to create a virtual dataset which a logical grid spanning regions with both present and missing data.

```python
# TODO example
```

### Inhomogenous Codecs

### Inhomogenenous encoding
