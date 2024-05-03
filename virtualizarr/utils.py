from typing import Optional, Union

# TODO: importing fsspec and s3fs to get typing. Is there a better way incase these are optional deps?
from s3fs.core import S3File
from fsspec.implementations.local import LocalFileOpener


def _fsspec_openfile_from_filepath(*, filepath: str, reader_options: Optional[dict]) -> Union[S3File, LocalFileOpener]:
    """Utility function to facilitate reading remote file paths using fsspec.

    :param filepath: Input filepath
    :type filepath: str
    :param reader_options: Dict containing options to pass to fsspec file reader
    :type reader_options: Optional[dict]
    :rtype: Union[S3File, LocalFileOpener]
    """
    import fsspec
    from upath import UPath

    universal_filepath = UPath(filepath)
    protocol = universal_filepath.protocol

    # why does UPath give an empty string for a local file protocol :(
    if protocol == '':

        fpath = fsspec.open(filepath, 'rb').open()

    
    elif protocol in ["s3"]:
        if not bool(reader_options):
            storage_options = {} # type: dict
        else:
            storage_options = reader_options.get('storage_options') #type: ignore
        target_options = {"anon":True}
        fpath = fsspec.filesystem(protocol).open(filepath,target_options = {"anon":True})

    else:
        raise NotImplementedError("Only local and s3 file protocols are currently supported")

    return fpath
