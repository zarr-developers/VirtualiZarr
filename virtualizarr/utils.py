from typing import Optional, Union

# TODO: importing fsspec and s3fs to get typing. Is there a better way incase these are optional deps?
from s3fs.core import S3File
from fsspec.implementations.local import LocalFileOpener


def _fsspec_openfile_from_filepath(*, filepath: str, reader_options: Optional[dict] = {'storage_options':{'key':'', 'secret':'', 'anon':True}}) -> Union[S3File, LocalFileOpener]:
    """Converts input filepath to fsspec openfile object.

    Parameters
    ----------
    filepath : str
        Input filepath
    reader_options : _type_, optional
        Dict containing kwargs to pass to file opener, by default {'storage_options':{'key':'', 'secret':'', 'anon':True}}

    Returns
    -------
    Union[S3File, LocalFileOpener]
        Either S3File or LocalFileOpener, depending on which protocol was supplied.

    Raises
    ------
    NotImplementedError
        Raises a Not Implemented Error if filepath protocol is not supported.
    """


    import fsspec
    from upath import UPath

    universal_filepath = UPath(filepath)
    protocol = universal_filepath.protocol

    if protocol == '':

        fpath = fsspec.open(filepath, 'rb').open()

    elif protocol in ["s3"]:
        s3_anon_defaults = {'key':'', 'secret':'', 'anon':True}
        if not bool(reader_options):
            storage_options = s3_anon_defaults

        else:
            storage_options = reader_options.get('storage_options') #type: ignore

            # using dict merge operator to add in defaults if keys are not specified
            storage_options = s3_anon_defaults | storage_options

        fpath = fsspec.filesystem(protocol, **storage_options).open(filepath)

    else:
        raise NotImplementedError("Only local and s3 file protocols are currently supported")

    return fpath
