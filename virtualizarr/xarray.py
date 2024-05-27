from collections.abc import Iterable, Mapping, MutableMapping
from pathlib import Path
from typing import (
    Literal,
    Optional,
    overload,
)

import ujson  # type: ignore
import xarray as xr
from xarray import register_dataset_accessor
from xarray.backends import BackendArray
from xarray.core.indexes import Index, PandasIndex
from xarray.core.variable import IndexVariable

import virtualizarr.kerchunk as kerchunk
from virtualizarr.kerchunk import FileType, KerchunkStoreRefs
from virtualizarr.manifests import ChunkManifest, ManifestArray
from virtualizarr.utils import _fsspec_openfile_from_filepath
from virtualizarr.zarr import (
    attrs_from_zarr_group_json,
    dataset_to_zarr,
    metadata_from_zarr_json,
)


class ManifestBackendArray(ManifestArray, BackendArray):
    """Using this prevents xarray from wrapping the KerchunkArray in ExplicitIndexingAdapter etc."""

    ...


def open_virtual_dataset(
    filepath: str,
    filetype: FileType | None = None,
    drop_variables: Iterable[str] | None = None,
    loadable_variables: Iterable[str] | None = None,
    indexes: Mapping[str, Index] | None = None,
    virtual_array_class=ManifestArray,
    reader_options: Optional[dict] = {
        "storage_options": {"key": "", "secret": "", "anon": True}
    },
) -> xr.Dataset:
    """
    Open a file or store as an xarray Dataset wrapping virtualized zarr arrays.

    No data variables will be loaded.

    Xarray indexes can optionally be created (the default behaviour). To avoid creating any xarray indexes pass indexes={}.

    Parameters
    ----------
    filepath : str, default None
        File path to open as a set of virtualized zarr arrays.
    filetype : FileType, default None
        Type of file to be opened. Used to determine which kerchunk file format backend to use.
        Can be one of {'netCDF3', 'netCDF4', 'zarr_v3'}.
        If not provided will attempt to automatically infer the correct filetype from the the filepath's extension.
    drop_variables: list[str], default is None
        Variables in the file to drop before returning.
    loadable_variables: list[str], default is None
        Variables in the file to open as lazy numpy/dask arrays instead of instances of virtual_array_class.
        Default is to open all variables as virtual arrays (i.e. ManifestArray).
    indexes : Mapping[str, Index], default is None
        Indexes to use on the returned xarray Dataset.
        Default is None, which will read any 1D coordinate data to create in-memory Pandas indexes.
        To avoid creating any indexes, pass indexes={}.
    virtual_array_class
        Virtual array class to use to represent the references to the chunks in each on-disk array.
        Currently can only be ManifestArray, but once VirtualZarrArray is implemented the default should be changed to that.
    reader_options: dict, default {'storage_options':{'key':'', 'secret':'', 'anon':True}}
        Dict passed into Kerchunk file readers. Note: Each Kerchunk file reader has distinct arguments,
        so ensure reader_options match selected Kerchunk reader arguments.

    Returns
    -------
    vds
        An xarray Dataset containing instances of virtual_array_cls for each variable, or normal lazily indexed arrays for each variable in loadable_variables.
    """

    if drop_variables is None:
        drop_variables = []
    elif isinstance(drop_variables, str):
        drop_variables = [drop_variables]
    else:
        drop_variables = list(drop_variables)
    if loadable_variables is None:
        loadable_variables = []
    elif isinstance(loadable_variables, str):
        loadable_variables = [loadable_variables]
    else:
        loadable_variables = list(loadable_variables)
    common = set(drop_variables).intersection(set(loadable_variables))
    if common:
        raise ValueError(f"Cannot both load and drop variables {common}")

    if virtual_array_class is not ManifestArray:
        raise NotImplementedError()

    if filetype == "zarr_v3":
        # TODO is there a neat way of auto-detecting this?
        return open_virtual_dataset_from_v3_store(
            storepath=filepath, drop_variables=drop_variables, indexes=indexes
        )
    else:
        # this is the only place we actually always need to use kerchunk directly
        # TODO avoid even reading byte ranges for variables that will be dropped later anyway?
        vds_refs = kerchunk.read_kerchunk_references_from_file(
            filepath=filepath,
            filetype=filetype,
            reader_options=reader_options,
        )
        virtual_vars = virtual_vars_from_kerchunk_refs(
            vds_refs,
            drop_variables=drop_variables + loadable_variables,
            virtual_array_class=virtual_array_class,
        )
        ds_attrs = kerchunk.fully_decode_arr_refs(vds_refs["refs"]).get(".zattrs", {})

        if indexes is None or len(loadable_variables) > 0:
            # TODO we are reading a bunch of stuff we know we won't need here, e.g. all of the data variables...
            # TODO it would also be nice if we could somehow consolidate this with the reading of the kerchunk references
            # TODO really we probably want a dedicated xarray backend that iterates over all variables only once
            fpath = _fsspec_openfile_from_filepath(
                filepath=filepath, reader_options=reader_options
            )

            ds = xr.open_dataset(fpath, drop_variables=drop_variables)

            if indexes is None:
                # add default indexes by reading data from file
                indexes = {name: index for name, index in ds.xindexes.items()}
            elif indexes != {}:
                # TODO allow manual specification of index objects
                raise NotImplementedError()
            else:
                indexes = dict(**indexes)  # for type hinting: to allow mutation

            loadable_vars = {
                name: var
                for name, var in ds.variables.items()
                if name in loadable_variables
            }

            # if we only read the indexes we can just close the file right away as nothing is lazy
            if loadable_vars == {}:
                ds.close()
        else:
            loadable_vars = {}
            indexes = {}

        vars = {**virtual_vars, **loadable_vars}

        data_vars, coords = separate_coords(vars, indexes)

        vds = xr.Dataset(
            data_vars,
            coords=coords,
            # indexes={},  # TODO should be added in a later version of xarray
            attrs=ds_attrs,
        )

        # TODO we should probably also use vds.set_close() to tell xarray how to close the file we opened

        return vds


def open_virtual_dataset_from_v3_store(
    storepath: str,
    drop_variables: list[str],
    indexes: Mapping[str, Index] | None,
) -> xr.Dataset:
    """
    Read a Zarr v3 store and return an xarray Dataset containing virtualized arrays.
    """
    _storepath = Path(storepath)

    ds_attrs = attrs_from_zarr_group_json(_storepath / "zarr.json")

    # TODO recursive glob to create a datatree
    # Note: this .is_file() check should not be necessary according to the pathlib docs, but tests fail on github CI without it
    # see https://github.com/TomNicholas/VirtualiZarr/pull/45#discussion_r1547833166
    all_paths = _storepath.glob("*/")
    directory_paths = [p for p in all_paths if not p.is_file()]

    vars = {}
    for array_dir in directory_paths:
        var_name = array_dir.name
        if var_name in drop_variables:
            break

        zarray, dim_names, attrs = metadata_from_zarr_json(array_dir / "zarr.json")
        manifest = ChunkManifest.from_zarr_json(str(array_dir / "manifest.json"))

        marr = ManifestArray(chunkmanifest=manifest, zarray=zarray)
        var = xr.Variable(data=marr, dims=dim_names, attrs=attrs)
        vars[var_name] = var

    if indexes is None:
        raise NotImplementedError()
    elif indexes != {}:
        # TODO allow manual specification of index objects
        raise NotImplementedError()
    else:
        indexes = dict(**indexes)  # for type hinting: to allow mutation

    data_vars, coords = separate_coords(vars, indexes)

    ds = xr.Dataset(
        data_vars,
        coords=coords,
        # indexes={},  # TODO should be added in a later version of xarray
        attrs=ds_attrs,
    )

    return ds


def virtual_vars_from_kerchunk_refs(
    refs: KerchunkStoreRefs,
    drop_variables: list[str] | None = None,
    virtual_array_class=ManifestArray,
) -> Mapping[str, xr.Variable]:
    """
    Translate a store-level kerchunk reference dict into aa set of xarray Variables containing virtualized arrays.

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

    vars = {
        var_name: variable_from_kerchunk_refs(refs, var_name, virtual_array_class)
        for var_name in var_names_to_keep
    }

    return vars


def dataset_from_kerchunk_refs(
    refs: KerchunkStoreRefs,
    drop_variables: list[str] = [],
    virtual_array_class: type = ManifestArray,
    indexes: MutableMapping[str, Index] | None = None,
) -> xr.Dataset:
    """
    Translate a store-level kerchunk reference dict into an xarray Dataset containing virtualized arrays.

    drop_variables: list[str], default is None
        Variables in the file to drop before returning.
    virtual_array_class
        Virtual array class to use to represent the references to the chunks in each on-disk array.
        Currently can only be ManifestArray, but once VirtualZarrArray is implemented the default should be changed to that.
    """

    vars = virtual_vars_from_kerchunk_refs(refs, drop_variables, virtual_array_class)

    if indexes is None:
        indexes = {}
    data_vars, coords = separate_coords(vars, indexes)

    ds_attrs = kerchunk.fully_decode_arr_refs(refs["refs"]).get(".zattrs", {})

    vds = xr.Dataset(
        data_vars,
        coords=coords,
        # indexes={},  # TODO should be added in a later version of xarray
        attrs=ds_attrs,
    )

    return vds


def variable_from_kerchunk_refs(
    refs: KerchunkStoreRefs, var_name: str, virtual_array_class
) -> xr.Variable:
    """Create a single xarray Variable by reading specific keys of a kerchunk references dict."""

    arr_refs = kerchunk.extract_array_refs(refs, var_name)
    chunk_dict, zarray, zattrs = kerchunk.parse_array_refs(arr_refs)
    manifest = ChunkManifest._from_kerchunk_chunk_dict(chunk_dict)
    dims = zattrs["_ARRAY_DIMENSIONS"]
    varr = virtual_array_class(zarray=zarray, chunkmanifest=manifest)

    return xr.Variable(data=varr, dims=dims, attrs=zattrs)


def separate_coords(
    vars: Mapping[str, xr.Variable],
    indexes: MutableMapping[str, Index],
) -> tuple[Mapping[str, xr.Variable], xr.Coordinates]:
    """
    Try to generate a set of coordinates that won't cause xarray to automatically build a pandas.Index for the 1D coordinates.

    Currently requires a workaround unless xarray 8107 is merged.

    Will also preserve any loaded variables and indexes it is passed.
    """

    # this would normally come from CF decoding, let's hope the fact we're skipping that doesn't cause any problems...
    coord_names: list[str] = []

    # split data and coordinate variables (promote dimension coordinates)
    data_vars = {}
    coord_vars = {}
    for name, var in vars.items():
        if name in coord_names or var.dims == (name,):
            # use workaround to avoid creating IndexVariables described here https://github.com/pydata/xarray/pull/8107#discussion_r1311214263
            if len(var.dims) == 1:
                dim1d, *_ = var.dims
                coord_vars[name] = (dim1d, var.data)

                if isinstance(var, IndexVariable):
                    # unless variable actually already is a loaded IndexVariable,
                    # in which case we need to keep it and add the corresponding indexes explicitly
                    coord_vars[name] = var
                    # TODO this seems suspect - will it handle datetimes?
                    indexes[name] = PandasIndex(var, dim1d)
            else:
                coord_vars[name] = var
        else:
            data_vars[name] = var

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

        Currently requires all variables to be backed by ManifestArray objects.

        Not very useful until some implementation of a Zarr reader can actually read these manifest.json files.
        See https://github.com/zarr-developers/zarr-specs/issues/287

        Parameters
        ----------
        storepath : str
        """
        dataset_to_zarr(self.ds, storepath)

    @overload
    def to_kerchunk(
        self, filepath: None, format: Literal["dict"]
    ) -> KerchunkStoreRefs: ...

    @overload
    def to_kerchunk(self, filepath: str | Path, format: Literal["json"]) -> None: ...

    @overload
    def to_kerchunk(
        self,
        filepath: str | Path,
        format: Literal["parquet"],
        record_size: int = 100_000,
        categorical_threshold: int = 10,
    ) -> None: ...

    def to_kerchunk(
        self,
        filepath: str | Path | None = None,
        format: Literal["dict", "json", "parquet"] = "dict",
        record_size: int = 100_000,
        categorical_threshold: int = 10,
    ) -> KerchunkStoreRefs | None:
        """
        Serialize all virtualized arrays in this xarray dataset into the kerchunk references format.

        Parameters
        ----------
        filepath : str, default: None
            File path to write kerchunk references into. Not required if format is 'dict'.
        format : 'dict', 'json', or 'parquet'
            Format to serialize the kerchunk references as.
            If 'json' or 'parquet' then the 'filepath' argument is required.
        record_size (parquet only): int
            Number of references to store in each reference file (default 100,000). Bigger values
            mean fewer read requests but larger memory footprint.
        categorical_threshold (parquet only) : int
            Encode urls as pandas.Categorical to reduce memory footprint if the ratio
            of the number of unique urls to total number of refs for each variable
            is greater than or equal to this number. (default 10)

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
            from kerchunk.df import refs_to_dataframe

            if isinstance(filepath, Path):
                url = str(filepath)
            elif isinstance(filepath, str):
                url = filepath

            # refs_to_dataframe is responsible for writing to parquet.
            # at no point does it create a full in-memory dataframe.
            refs_to_dataframe(
                refs,
                url=url,
                record_size=record_size,
                categorical_threshold=categorical_threshold,
            )
            return None
        else:
            raise ValueError(f"Unrecognized output format: {format}")
