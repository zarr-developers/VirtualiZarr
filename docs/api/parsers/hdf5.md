# HDF5/NetCDF4

!!! note "Chunk-dense files over the network"

    Local files are walked with HDF5's own index-aware driver, so building a
    manifest reads only the chunk index. For **remote** chunk-dense files (many
    small chunks — e.g. sparse single-cell `.h5ad`), the default block reader
    can instead fetch a large fraction of the whole file just to recover that
    index.

    The robust fix is to **download the file once and parse it locally**, which
    takes the native fast path automatically.

    If downloading isn't practical, tune the block size via the `reader_factory`
    parameter (documented below). The default 1 MiB block over-reads the ~2 KiB
    chunk-index nodes, so a smaller block cuts the bytes fetched — at the cost
    of more requests:

    ```python
    import functools
    from obspec_utils.readers import BlockStoreReader

    parser = HDFParser(
        reader_factory=functools.partial(BlockStoreReader, block_size=64 * 1024),
    )
    ```

    This only mitigates the over-read: the ideal block size depends on the
    file's chunk-index layout, so downloading and parsing locally remains the
    reliable option.

::: virtualizarr.parsers.HDFParser
