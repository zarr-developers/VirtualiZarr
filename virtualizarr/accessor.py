from pathlib import Path
from typing import TYPE_CHECKING, Callable, Literal, Optional, overload

from xarray import Dataset, register_dataset_accessor

from virtualizarr.manifests import ManifestArray
from virtualizarr.types.kerchunk import KerchunkStoreRefs
from virtualizarr.writers.kerchunk import dataset_to_kerchunk_refs
from virtualizarr.writers.zarr import dataset_to_zarr

if TYPE_CHECKING:
    from icechunk import IcechunkStore  # type: ignore[import-not-found]


@register_dataset_accessor("virtualize")
class VirtualiZarrDatasetAccessor:
    """
    Xarray accessor for writing out virtual datasets to disk.

    Methods on this object are called via `ds.virtualize.{method}`.
    """

    def __init__(self, ds: Dataset):
        self.ds: Dataset = ds

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

    def to_icechunk(
        self, store: "IcechunkStore", append_dim: Optional[str] = None
    ) -> None:
        """
        Write an xarray dataset to an Icechunk store.

        Any variables backed by ManifestArray objects will be be written as virtual references, any other variables will be loaded into memory before their binary chunk data is written into the store.

        If `append_dim` is provided, the virtual dataset will be appended to the existing IcechunkStore along the `append_dim` dimension.

        Parameters
        ----------
        store: IcechunkStore
        append_dim: str, optional
        """
        from virtualizarr.writers.icechunk import dataset_to_icechunk

        dataset_to_icechunk(self.ds, store, append_dim=append_dim)

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
    ) -> Dataset:
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

        >>> ds.virtualize.rename_paths(local_to_s3_url)

        See Also
        --------
        ManifestArray.rename_paths
        ChunkManifest.rename_paths
        """

        new_ds = self.ds.copy()
        for var_name in new_ds.variables:
            data = new_ds[var_name].data
            if isinstance(data, ManifestArray):
                new_ds[var_name].data = data.rename_paths(new=new)

        return new_ds
