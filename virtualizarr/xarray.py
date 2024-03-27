from typing import List, Literal, Optional, Union, overload

import ujson  # type: ignore
import xarray as xr
from xarray import register_dataset_accessor
from xarray.backends import BackendArray

import virtualizarr.kerchunk as kerchunk
from virtualizarr.kerchunk import KerchunkStoreRefs
from virtualizarr.manifests import ChunkManifest, ManifestArray


class ManifestBackendArray(ManifestArray, BackendArray):
    """Using this prevents xarray from wrapping the KerchunkArray in ExplicitIndexingAdapter etc."""

    ...


def open_virtual_dataset(
    filepath: str,
    filetype: Optional[str] = None,
    drop_variables: Optional[List[str]] = None,
    virtual_array_class=ManifestArray,
    indexes={},
) -> xr.Dataset:
    """
    Open a file or store as an xarray Dataset wrapping virtualized zarr arrays.

    It's important that we avoid creating any IndexVariables, as our virtualized zarr array objects don't actually contain a collection that can be turned into a pandas.Index.

    Parameters
    ----------
    filepath : str, default None
        File path to open as a set of virtualized zarr arrays.
    filetype : str, default None
        Type of file to be opened. Used to determine which kerchunk file format backend to use.
        Can be one of {'netCDF3', 'netCDF4'}.
        If not provided will attempt to automatically infer the correct filetype from the the filepath's extension.
    drop_variables: list[str], default is None
        Variables in the file to drop before returning.
    virtual_array_class
        Virtual array class to use to represent the references to the chunks in each on-disk array.
        Currently can only be ManifestArray, but once VirtualZarrArray is implemented the default should be changed to that.
    """

    # this is the only place we actually always need to use kerchunk directly
    ds_refs = kerchunk.read_kerchunk_references_from_file(
        filepath=filepath,
        filetype=filetype,
    )

    ds = dataset_from_kerchunk_refs(
        ds_refs,
        drop_variables=drop_variables,
        virtual_array_class=virtual_array_class,
        indexes=indexes,
    )

    # TODO we should probably also use ds.set_close() to tell xarray how to close the file we opened

    return ds


def dataset_from_kerchunk_refs(
    refs: KerchunkStoreRefs,
    drop_variables: Optional[List[str]] = None,
    virtual_array_class=ManifestArray,
    indexes={},
) -> xr.Dataset:
    """
    Translate a store-level kerchunk reference dict into an xarray Dataset containing virtualized arrays.

    drop_variables: list[str], default is None
        Variables in the file to drop before returning.
    virtual_array_class
        Virtual array class to use to represent the references to the chunks in each on-disk array.
        Currently can only be ManifestArray, but once VirtualZarrArray is implemented the default should be changed to that.
    """

    var_names = kerchunk.find_var_names(refs)
    if drop_variables is None:
        drop_variables = []
    var_names_to_keep = [
        var_name for var_name in var_names if var_name not in drop_variables
    ]

    vars = {}
    for var_name in var_names_to_keep:
        # TODO abstract all this parsing into a function/method?
        arr_refs = kerchunk.extract_array_refs(refs, var_name)
        chunk_dict, zarray, zattrs = kerchunk.parse_array_refs(arr_refs)
        manifest = ChunkManifest.from_kerchunk_chunk_dict(chunk_dict)
        dims = zattrs["_ARRAY_DIMENSIONS"]

        varr = virtual_array_class(zarray=zarray, chunkmanifest=manifest)
        vars[var_name] = xr.Variable(data=varr, dims=dims, attrs=zattrs)

    data_vars, coords = separate_coords(vars, indexes)

    ds_attrs = kerchunk.fully_decode_arr_refs(refs["refs"]).get(".zattrs", {})

    ds = xr.Dataset(
        data_vars,
        coords=coords,
        # indexes={},  # TODO should be added in a later version of xarray
        attrs=ds_attrs,
    )

    return ds


def separate_coords(
    vars: dict[str, xr.Variable],
    indexes={},
) -> tuple[dict[str, xr.Variable], xr.Coordinates]:
    """
    Try to generate a set of coordinates that won't cause xarray to automatically build a pandas.Index for the 1D coordinates.

    I thought this should be easy but it was actually really hard - in the end I had to checkout xarray v2023.08.0, the last one before #8107 was merged.
    """

    # this would normally come from CF decoding, let's hope the fact we're skipping that doesn't cause any problems...
    coord_names: List[str] = []
    # split data and coordinate variables (promote dimension coordinates)
    data_vars = {}
    coord_vars = {}
    for name, var in vars.items():
        if name in coord_names or var.dims == (name,):
            # use workaround to avoid creating IndexVariables described here https://github.com/pydata/xarray/pull/8107#discussion_r1311214263
            if len(var.dims) == 1:
                dim1d, *_ = var.dims
                coord_vars[name] = (dim1d, var.data)
            else:
                coord_vars[name] = var
        else:
            data_vars[name] = var

    # this is stolen from https://github.com/pydata/xarray/pull/8051
    # needed otherwise xarray errors whilst trying to turn the KerchunkArrays for the 1D coordinate variables into indexes
    # but it doesn't appear to work with `main` since #8107, which is why the workaround above is needed
    # EDIT: actually even the workaround doesn't work - to avoid creating indexes I had to checkout xarray v2023.08.0, the last one before #8107 was merged
    set_indexes = False
    if set_indexes:
        coords = coord_vars
    else:
        # explict Coordinates object with no index passed
        coords = xr.Coordinates(coord_vars, indexes=indexes)

    return data_vars, coords


@register_dataset_accessor("virtualize")
class VirtualiZarrDatasetAccessor:
    """
    Xarray accessor for writing out virtual datasets to disk.

    Methods on this object are called via `ds.virtualize.{method}`.
    """

    def __init__(self, ds):
        self.ds = ds

    def to_zarr(self, storepath: str) -> None:
        """
        Serialize all virtualized arrays in this xarray dataset as a Zarr store.

        Parameters
        ----------
        storepath : str
        """
        raise NotImplementedError(
            "No point in writing out these virtual arrays to Zarr until at least one Zarr reader can actually read them."
        )

    @overload
    def to_kerchunk(self, filepath: None, format: Literal["dict"]) -> KerchunkStoreRefs:
        ...

    @overload
    def to_kerchunk(self, filepath: str, format: Literal["json"]) -> None:
        ...

    @overload
    def to_kerchunk(self, filepath: str, format: Literal["parquet"]) -> None:
        ...

    def to_kerchunk(
        self,
        filepath: Optional[str] = None,
        format: Union[Literal["dict"], Literal["json"], Literal["parquet"]] = "dict",
    ) -> Union[KerchunkStoreRefs, None]:
        """
        Serialize all virtualized arrays in this xarray dataset into the kerchunk references format.

        Parameters
        ----------
        filepath : str, default: None
            File path to write kerchunk references into. Not required if format is 'dict'.
        format : 'dict', 'json', or 'parquet'
            Format to serialize the kerchunk references as.
            If 'json' or 'parquet' then the 'filepath' argument is required.

        References
        ----------
        https://fsspec.github.io/kerchunk/spec.html
        """
        refs = kerchunk.dataset_to_kerchunk_refs(self.ds)

        if format == "dict":
            return refs
        elif format == "json":
            if filepath is None:
                raise ValueError("Filepath must be provided when format is 'json'")

            with open(filepath, "w") as json_file:
                ujson.dump(refs, json_file)

            return None
        elif format == "parquet":
            raise NotImplementedError()
        else:
            raise ValueError(f"Unrecognized output format: {format}")
