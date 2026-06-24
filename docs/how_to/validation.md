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

### Missing variables

Sometimes entire variables are missing from files.
This might break the model of a regular timeseries, but can still be represented in a Zarr store as the array having missing values (i.e. a default `fill_value` such as NaN).

### Missing chunks

### Inhomogenous Codecs

### Inhomogenenous encoding
