import warnings
from collections import deque
from datetime import datetime
from functools import wraps
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Callable,
    Generator,
    Iterable,
    Literal,
    overload,
)

import xarray as xr

from virtualizarr.manifests import ManifestArray
from virtualizarr.types.kerchunk import KerchunkStoreRefs
from virtualizarr.writers.kerchunk import dataset_to_kerchunk_refs

if TYPE_CHECKING:
    from icechunk import IcechunkStore  # type: ignore[import-not-found]


def warn_if_not_virtual(cls_name: Literal["Dataset", "DataTree"]):
    """Decorator for methods which only make sense for fully virtual xarray objects."""

    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            all_vars: Iterable[xr.Variable]
            match cls_name:
                case "Dataset":
                    all_vars = self.ds.variables.values()
                case "DataTree":
                    all_vars = all_datatree_variables(self.dt)

            if not any(isinstance(var.data, ManifestArray) for var in all_vars):
                warnings.warn(
                    f"Attempting to write an entirely non-virtual {cls_name} to a virtual references format - i.e. your `xarray.{cls_name}` contains zero `ManifestArray` objects. "
                    "This is almost certainly not intended, as the entire data contents will be duplicated rather than referenced. "
                    f"This may have happened because you used `xarray.open_{cls_name}` instead of `virtualizarr.open_virtual_{cls_name}`, or you set all variables to be `loadable_variables`."
                    "Please read the usage docs.",
                    UserWarning,
                )

            return func(self, *args, **kwargs)

        return wrapper

    return decorator


def all_datatree_variables(root: xr.DataTree) -> Generator[xr.Variable, None, None]:
    """Flat iterable over all variables in a DataTree"""
    queue = deque([root])
    while queue:
        node = queue.popleft()
        yield from node.variables.values()
        queue.extend(node.children.values())


class _VirtualiZarrDatasetAccessor:
    """
    Xarray accessor for writing out virtual datasets to disk.

    Methods on this object are called via `ds.vz.{method}`.
    """

    def __init__(self, ds: xr.Dataset):
        self.ds: xr.Dataset = ds

    @warn_if_not_virtual("Dataset")
    def to_icechunk(
        self,
        store: "IcechunkStore",
        *,
        group: str | None = None,
        append_dim: str | None = None,
        validate_containers: bool = True,
        last_updated_at: datetime | None = None,
    ) -> None:
        """
        Write an xarray dataset to an Icechunk store.

        Any variables backed by ManifestArray objects will be be written as virtual
        references. Any other variables will be loaded into memory before their binary
        chunk data is written into the store.

        If `append_dim` is provided, the virtual dataset will be appended to the
        existing IcechunkStore along the `append_dim` dimension.

        If `last_updated_at` is provided, it will be used as a checksum for any virtual
        chunks written to the store with this operation.  At read time, if any of the
        virtual chunks have been updated since this provided datetime, an error will be
        raised.  This protects against reading outdated virtual chunks that have been
        updated since the last read.  When not provided, the current time is used.  This
        value is stored in Icechunk with seconds precision, so be sure to take that into
        account when providing this value.

        Parameters
        ----------
        store
            Store to write dataset into.
        group
            Path of the group to write the dataset into (default: the root group).
        append_dim
            Dimension along which to append the virtual dataset.
        validate_containers
            If ``True``, raise if any virtual chunks have a refer to locations that don't
            match any existing virtual chunk container set on this Icechunk repository.

            It is not generally recommended to set this to ``False``, because it can lead to
            confusing runtime results and errors when reading data back.
        last_updated_at
            Datetime to use as a checksum for any virtual chunks written to the store
            with this operation. When not provided, the current time is used.

        Raises
        ------
        ValueError
            If the store is read-only.
        """
        from virtualizarr.writers.icechunk import virtual_dataset_to_icechunk

        virtual_dataset_to_icechunk(
            self.ds,
            store,
            group=group,
            append_dim=append_dim,
            validate_containers=validate_containers,
            last_updated_at=last_updated_at,
        )

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

    @warn_if_not_virtual("Dataset")
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
        filepath
            File path to write kerchunk references into. Not required if format is 'dict'.
        format
            Format to serialize the kerchunk references as.
            If 'json' or 'parquet' then the 'filepath' argument is required.
        record_size
            Number of references to store in each reference file (default 100,000). Bigger values
            mean fewer read requests but larger memory footprint. Only available when `format` is 'parquet'.
        categorical_threshold
            Encode urls as pandas.Categorical to reduce memory footprint if the ratio
            of the number of unique urls to total number of refs for each variable
            is greater than or equal to this number (default 10). Only available when `format` is 'parquet'.

        References
        ----------
        https://fsspec.github.io/kerchunk/spec.html
        """
        refs = dataset_to_kerchunk_refs(self.ds)

        if format == "dict":
            return refs
        elif format == "json":
            import ujson

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

    def rename_paths(
        self,
        new: str | Callable[[str], str],
    ) -> xr.Dataset:
        """
        Rename paths to chunks in every ManifestArray in this dataset.

        Accepts either a string, in which case this new path will be used for all chunks, or
        a function which accepts the old path and returns the new path.

        Parameters
        ----------
        new
            New path to use for all chunks, either as a string, or as a function which accepts and returns strings.

        Returns
        -------
        Dataset

        See Also
        --------

        virtualizarr.ManifestArray.rename_paths

        virtualizarr.ChunkManifest.rename_paths

        Examples
        --------
        Rename paths to reflect moving the referenced files from local storage to an S3 bucket.

        >>> def local_to_s3_url(old_local_path: str) -> str:
        ...     from pathlib import Path
        ...
        ...     new_s3_bucket_url = "http://s3.amazonaws.com/my_bucket/"
        ...
        ...     filename = Path(old_local_path).name
        ...     return str(new_s3_bucket_url / filename)
        >>>
        >>> ds.vz.rename_paths(local_to_s3_url)
        """

        new_ds = self.ds.copy()
        for var_name in new_ds.variables:
            data = new_ds[var_name].data
            if isinstance(data, ManifestArray):
                new_ds[var_name].data = data.rename_paths(new=new)

        return new_ds

    @property
    def nbytes(self) -> int:
        """
        Size required to hold these references in memory in bytes.

        Note this is not the size of the referenced chunks if they were actually loaded into memory,
        this is only the size of the pointers to the chunk locations.
        If you were to load the data into memory it would be ~1e6x larger for 1MB chunks.

        In-memory (loadable) variables are included in the total using xarray's normal ``.nbytes`` method.
        """
        return sum(
            var.data.nbytes_virtual
            if isinstance(var.data, ManifestArray)
            else var.nbytes
            for var in self.ds.variables.values()
        )


@xr.register_dataset_accessor("vz")
class VirtualiZarrDatasetAccessor(_VirtualiZarrDatasetAccessor):
    pass


@xr.register_dataset_accessor("virtualize")
class DeprecatedVirtualiZarrDatasetAccessor(_VirtualiZarrDatasetAccessor):
    def __init__(self, ds: xr.Dataset):
        super().__init__(ds)
        warnings.warn(
            "VirtualiZarr's accessor has been renamed from `virtualize` to `vz`. Please use `.vz`",
            DeprecationWarning,
        )


class _VirtualiZarrDataTreeAccessor:
    """
    Xarray accessor for writing out virtual datatrees to disk.

    Methods on this object are called via `dt.vz.{method}`.
    """

    def __init__(self, dt: xr.DataTree):
        self.dt = dt

    @warn_if_not_virtual("DataTree")
    def to_icechunk(
        self,
        store: "IcechunkStore",
        *,
        write_inherited_coords: bool = False,
        validate_containers: bool = True,
        last_updated_at: datetime | None = None,
    ) -> None:
        """
        Write an xarray DataTree to an Icechunk store.

        Any variables backed by ManifestArray objects will be be written as virtual
        references. Any other variables will be loaded into memory before their binary
        chunk data is written into the store.

        If ``last_updated_at`` is provided, it will be used as a checksum for any
        virtual chunks written to the store with this operation.  At read time, if any
        of the virtual chunks have been updated since this provided datetime, an error
        will be raised.  This protects against reading outdated virtual chunks that have
        been updated since the last read.  When not provided, no check is performed.
        This value is stored in Icechunk with seconds precision, so be sure to take that
        into account when providing this value.

        Parameters
        ----------
        store
            Store to write dataset into.
        write_inherited_coords
            If ``True``, replicate inherited coordinates on all descendant nodes.
            Otherwise, only write coordinates at the level at which they are
            originally defined. This saves disk space, but requires opening the
            full tree to load inherited coordinates.
        validate_containers
            If ``True``, raise if any virtual chunks have a refer to locations that don't
            match any existing virtual chunk container set on this Icechunk repository.

            It is not generally recommended to set this to ``False``, because it can lead to
            confusing runtime results and errors when reading data back.
        last_updated_at
            Datetime to use as a checksum for any virtual chunks written to the store
            with this operation.  When not provided, no check is performed.

        Raises
        ------
        ValueError
            If the store is read-only.

        Examples
        --------
        To ensure an error is raised if the files containing referenced virtual chunks
        are modified at any time from now on, pass the current time to
        ``last_updated_at``.

        >>> from datetime import datetime
        >>> vdt.vz.to_icechunk(  # doctest: +SKIP
        ...     icechunkstore,
        ...     last_updated_at=datetime.now(),
        ... )
        """
        from virtualizarr.writers.icechunk import virtual_datatree_to_icechunk

        virtual_datatree_to_icechunk(
            self.dt,
            store,
            write_inherited_coords=write_inherited_coords,
            validate_containers=validate_containers,
            last_updated_at=last_updated_at,
        )


@xr.register_datatree_accessor("vz")
class VirtualiZarrDataTreeAccessor(_VirtualiZarrDataTreeAccessor):
    pass


@xr.register_datatree_accessor("virtualize")
class DeprecatedVirtualiZarrDataTreeAccessor(_VirtualiZarrDataTreeAccessor):
    def __init__(self, dt: xr.DataTree):
        super().__init__(dt)
        warnings.warn(
            "VirtualiZarr's accessor has been renamed from `virtualize` to `vz`. Please use `.vz`",
            DeprecationWarning,
        )
