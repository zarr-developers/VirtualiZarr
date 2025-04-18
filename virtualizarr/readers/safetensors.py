import json
import struct
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.parse import urlparse

import numpy as np
from obstore.store import (
    HTTPStore,
    LocalStore,
    ObjectStore,  # type: ignore[import-not-found]
)
from xarray import Dataset, Index

from virtualizarr.manifests import (
    ChunkEntry,
    ChunkManifest,
    ManifestArray,
    ManifestGroup,
    ManifestStore,
    ObjectStoreRegistry,
)
from virtualizarr.manifests.store import default_object_store
from virtualizarr.manifests.utils import create_v3_array_metadata
from virtualizarr.readers.api import VirtualBackend
from virtualizarr.types import ChunkKey


class SafeTensorsVirtualBackend(VirtualBackend):
    """
    Backend for reading SafeTensors files as virtual datasets.

    SafeTensors is a format for safely storing tensors (multidimensional arrays),
    without using pickle, and with zero-copy access. It is commonly used for storing
    model weights in the fields of ML and AI.

    The format consists of:
    - 8 bytes (header size): unsigned little-endian 64-bit integer containing the size of the header
    - N bytes (header): a JSON UTF-8 string containing tensor metadata
    - Rest of the file: byte-buffer containing tensor data

    Examples
    --------
    Open a local SafeTensors file with default settings:

    >>> vds = open_virtual_dataset("model_weights.safetensors")

    Open a SafeTensors file with custom dimension names:

    >>> custom_dims = {"weight": ["input_dims", "output_dims"], "bias": ["output_dims"]}
    >>> vds = open_virtual_dataset(
    ...     "model_weights.safetensors",
    ...     virtual_backend_kwargs={"dimension_names": custom_dims}
    ... )

    Open a GPT-2 model from Hugging Face Hub:

    >>> vds = open_virtual_dataset(
    ...     "https://huggingface.co/openai-community/gpt2/resolve/main/model.safetensors"
    ... )
    >>> # Access various GPT-2 tensors
    >>> print(vds["wte.weight"].shape)  # Word Token Embeddings: (50257, 768)
    >>> print(vds["wpe.weight"].shape)  # Word Position Embeddings: (1024, 768)
    >>> print(vds["ln_f.weight"].shape)  # Final layer norm weight
    """

    @staticmethod
    def _parse_safetensors_header(
        filepath: str, store: ObjectStore
    ) -> tuple[dict[str, Any], int]:
        """
        Parse the header of a SafeTensors file to extract metadata.

        This method reads the header of a SafeTensors file which contains:
        1. Header size (8 bytes): uint64 little-endian indicating header length
        2. Header content (variable length): JSON-encoded metadata describing tensors

        The header metadata includes tensor names, data types, shapes, and byte offsets.

        Parameters
        ----------
        filepath : str
            Path to the SafeTensors file. Can be a local path or a URL.
        store : ObjectStore
            Object store to use for reading the file. Should be compatible with
            the filepath type (LocalStore for local files, HTTPStore for URLs, etc.).

        Returns
        -------
        tuple[dict[str, Any], int]
            A tuple containing:
            - header (dict): Parsed JSON header containing tensor metadata
              including names, dtypes, shapes, and data_offsets
            - header_size (int): Size of the header in bytes

        Examples
        --------
        The returned header might look like:
        {
            "weight": {
                "dtype": "F32",
                "shape": [10, 20],
                "data_offsets": [0, 800]
            },
            "bias": {
                "dtype": "F32",
                "shape": [20],
                "data_offsets": [800, 880]
            },
            "__metadata__": {
                "framework": "pytorch",
                "version": "2.0"
            }
        }
        """
        from virtualizarr.utils import ObstoreReader

        reader = ObstoreReader(store, filepath)

        #  8 bytes, uint64 little-endian
        header_size_bytes = reader.read(8)
        header_size = struct.unpack("<Q", header_size_bytes)[0]

        header_bytes = reader.read(header_size)
        header = json.loads(header_bytes.decode("utf-8"))

        return header, header_size

    @staticmethod
    def _create_manifest_group(
        filepath: str,
        drop_variables: list,
        store: ObjectStore,
        dimension_names: Optional[Dict[str, list[str]]] = None,
    ) -> ManifestGroup:
        """
        Create a ManifestGroup from a SafeTensors file.

        This method reads the SafeTensors header, parses tensor metadata, and creates
        ManifestArrays for each tensor. Each tensor is treated as a single chunk for
        efficient memory-mapped access.

        Parameters
        ----------
        filepath : str
            Path to the SafeTensors file. Can be a local path or a URL.
        drop_variables : list
            List of tensor names to exclude from the dataset.
        store: ObjectStore
            Object store used for reading the file. Should match the filepath type.
        dimension_names : Dict[str, list[str]], optional
            Custom dimension names for specific tensors. The keys should be tensor names,
            and the values should be lists of dimension names matching the tensor's shape.
            If not provided, default names are generated as "{tensor_name}_dim_{i}".

        Returns
        -------
        ManifestGroup
            A group containing ManifestArrays for each tensor, with appropriate
            metadata and attributes.

        Notes
        -----
        - Each tensor is represented as a single chunk for direct memory access
        - The __metadata__ field from the SafeTensors header is preserved as attributes
        - Tensor metadata includes:
          - Original SafeTensors dtype (e.g., "F32", "I64", "BF16")
          - Storage information indicating contiguous layout
          - Any additional tensor-specific metadata from the header

        Examples
        --------
        >>> store = default_object_store("model.safetensors")
        >>> dimension_names = {
        ...     "weight": ["input_channels", "output_channels"],
        ...     "bias": ["features"]
        ... }
        >>> manifest_group = _create_manifest_group(
        ...     "model.safetensors",
        ...     drop_variables=[],
        ...     store=store,
        ...     dimension_names=dimension_names
        ... )
        """
        header, header_size = SafeTensorsVirtualBackend._parse_safetensors_header(
            filepath, store
        )

        manifest_dict = {}

        attrs = {}
        if "__metadata__" in header:
            metadata_content = header["__metadata__"]
            if isinstance(metadata_content, dict):
                for key, value in metadata_content.items():
                    # safetensors spec only allows text-to-text map in __metadata__
                    if not isinstance(value, str):
                        value = json.dumps(value)
                    attrs[key] = value
            else:
                attrs["__metadata__"] = (
                    metadata_content
                    if isinstance(metadata_content, str)
                    else json.dumps(metadata_content)
                )

        data_start = 8 + header_size

        for tensor_name, tensor_info in header.items():
            if tensor_name == "__metadata__" or tensor_name in drop_variables:
                continue

            dtype_str = tensor_info["dtype"]
            shape = tuple(tensor_info["shape"])
            # data offsets relative to end of header
            start_offset, end_offset = tensor_info["data_offsets"]

            dtype = SafeTensorsVirtualBackend._map_dtype(dtype_str)

            abs_start = data_start + start_offset
            abs_end = data_start + end_offset

            chunk_manifest = SafeTensorsVirtualBackend._create_chunk_manifest(
                filepath=filepath,
                offset=abs_start,
                length=abs_end - abs_start,
                shape=shape,
            )

            if dimension_names and tensor_name in dimension_names:
                custom_names = dimension_names[tensor_name]
                if len(custom_names) != len(shape):
                    raise ValueError(
                        f"Provided dimension names for '{tensor_name}' has {len(custom_names)} "
                        f"names, but tensor has {len(shape)} dimensions."
                    )
                dim_names = custom_names
            else:
                dim_names = [f"{tensor_name}_dim_{i}" for i in range(len(shape))]

            tensor_attrs = {}
            #  not clear to me from the spec if additional keys allowed, parse just in case
            for key, value in tensor_info.items():
                if key not in {"dtype", "shape", "data_offsets"}:
                    tensor_attrs[key] = (
                        value if isinstance(value, str) else json.dumps(value)
                    )

            tensor_attrs["original_safetensors_dtype"] = dtype_str

            tensor_attrs["safetensors_storage_info"] = json.dumps(
                {
                    "chunked": False,
                    "contiguous": True,
                }
            )

            metadata = create_v3_array_metadata(
                shape=shape,
                data_type=dtype,
                chunk_shape=shape,  # Treat the whole tensor as a single chunk
                dimension_names=dim_names,
                attributes=tensor_attrs,
            )

            manifest_array = ManifestArray(
                metadata=metadata,
                chunkmanifest=chunk_manifest,
            )

            manifest_dict[tensor_name] = manifest_array

        return ManifestGroup(arrays=manifest_dict, attributes=attrs)

    @staticmethod
    def _create_manifest_store(
        filepath: str,
        drop_variables: list,
        dimension_names: Optional[Dict[str, list[str]]] = None,
        revision: Optional[str] = None,
    ) -> ManifestStore:
        """
        Create a ManifestStore for a SafeTensors file.

        This method handles the complete workflow of reading a SafeTensors file and
        creating a virtual store. It automatically determines the appropriate store type
        based on the filepath and handles Hugging Face Hub URLs specially.

        Parameters
        ----------
        filepath : str
            Path to the SafeTensors file. Can be:
            - Local filesystem path (e.g., "/path/to/file.safetensors")
            - HTTP/HTTPS URL (e.g., "https://huggingface.co/.../file.safetensors")
        drop_variables : list
            List of tensor names to exclude from the dataset.
        dimension_names : Dict[str, list[str]], optional
            Custom dimension names for specific tensors. Keys are tensor names,
            values are lists of dimension names matching the tensor's shape.
        revision : str, optional
            Repository revision (branch, tag, or commit hash) for Hugging Face Hub.
            Defaults to "main" if not specified.

        Returns
        -------
        ManifestStore
            A store containing virtual references to tensors with metadata,
            optimized for the detected storage backend.

        Notes
        -----
        For Hugging Face Hub URLs, this method:
        - Automatically inserts the correct API format with "/resolve/{revision}/"

        Examples
        --------
        Local file:
        >>> store = _create_manifest_store(
        ...     "model.safetensors",
        ...     drop_variables=["optimizer_state"]
        ... )

        Hugging Face Hub file:
        >>> store = _create_manifest_store(
        ...     "https://huggingface.co/openai-community/gpt2/model.safetensors",
        ...     drop_variables=[],
        ...     revision="v2.0"
        ... )
        """
        store_registry = ObjectStoreRegistry()
        store = default_object_store(filepath)

        if not revision:
            revision = "main"

        # file is on the hub and not local
        if isinstance(store, HTTPStore):
            path = urlparse(filepath).path

            # Check if path already contains '/resolve/' - if so, don't modify it
            if "/resolve/" not in path:
                # HF API requires insertion of 'resolve' + '{revision}' after repo name
                filepath = "/".join(
                    path.split("/")[0:3]
                    + ["resolve", f"{revision}"]
                    + path.split("/")[3:]
                )

        # obstore and virtualizarr require absolute paths
        if isinstance(store, LocalStore):
            filepath = str(Path(urlparse(filepath).path).resolve())

        store_registry.register_store(filepath, store)

        manifest_group = SafeTensorsVirtualBackend._create_manifest_group(
            filepath=filepath,
            drop_variables=drop_variables,
            dimension_names=dimension_names,
            store=store,
        )

        return ManifestStore(group=manifest_group, store_registry=store_registry)

    @staticmethod
    def open_virtual_dataset(
        filepath: str,
        group: str | None = None,
        drop_variables: Iterable[str] | None = None,
        loadable_variables: Iterable[str] | None = None,
        decode_times: bool | None = None,
        indexes: Mapping[str, Index] | None = None,
        virtual_backend_kwargs: Optional[dict] = None,
        reader_options: Optional[dict] = None,
    ) -> Dataset:
        """
        Open a SafeTensors file as a virtual dataset.

        SafeTensors is a format used primarily for storing ML and AI model weights
        and parameters. This method creates a virtual xarray Dataset where each tensor
        becomes a separate variable, without loading the data into memory.

        Parameters
        ----------
        filepath : str
            Path to the SafeTensors file. Can be a local path or a Hugging Face Hub URL.
            For Hugging Face Hub, use format:
            "https://huggingface.co/{username}/{repo_name}/{filename}"
        group : str, optional
            Not used for SafeTensors files as they don't have hierarchical structure.
        drop_variables : Iterable[str], optional
            Names of tensors to exclude from the dataset. Useful for skipping large
            or unnecessary tensors.
        loadable_variables : Iterable[str], optional
            Variables to load as lazy numpy/dask arrays instead of ManifestArrays.
            These will be loaded on-demand when accessed.
        decode_times : bool, optional
            Not applicable for SafeTensors files (no time encoding).
        indexes : Mapping[str, Index], optional
            Custom indexes to attach to the returned Dataset.
        virtual_backend_kwargs : dict, optional
            Additional keyword arguments for the SafeTensors backend:

            - dimension_names : Dict[str, list[str]], optional
              Custom dimension names for specific tensors. The keys should be tensor names,
              and the values should be lists of dimension names matching the tensor's shape.
              Example: {"weight": ["input_dims", "output_dims"]} for a 2D weight tensor.
              If not provided, defaults to "{tensor_name}_dim_{i}".

            - revision : str, optional
              Repository revision for Hugging Face Hub (branch/tag/commit).
              Defaults to "main" if not specified.

        reader_options : dict, optional
            Not supported for SafeTensors files.

        Returns
        -------
        xr.Dataset
            A virtual dataset where:
            - Each tensor becomes a separate variable
            - Metadata from the SafeTensors header becomes dataset attributes

        Raises
        ------
        ValueError
            If group parameter is provided (not supported for SafeTensors)
        NotImplementedError
            If unsupported virtual_backend_kwargs or reader_options are provided

        Examples
        --------
        Open a local SafeTensors file:
        >>> vds = open_virtual_dataset("model.safetensors")
        >>> print(vds.variables.keys())
        ['weight', 'bias', 'embedding.weight']

        Open with custom dimension names:
        >>> dims = {
        ...     "weight": ["hidden_size", "output_size"],
        ...     "bias": ["output_size"]
        ... }
        >>> vds = open_virtual_dataset(
        ...     "model.safetensors",
        ...     virtual_backend_kwargs={"dimension_names": dims}
        ... )
        >>> print(vds["weight"].dims)
        ('hidden_size', 'output_size')

        Open from Hugging Face Hub:
        >>> vds = open_virtual_dataset(
        ...     "https://huggingface.co/openai-community/gpt2/model.safetensors"
        ... )
        >>> print(vds["wte.weight"].shape)  # Word token embeddings
        (50257, 768)

        Load specific tensors as lazy arrays:
        >>> vds = open_virtual_dataset(
        ...     "model_weights.safetensors",
        ...     loadable_variables=["small_tensor"],
        ...     drop_variables=["large_optimizer_state"]
        ... )

        Notes
        -----
        - Each tensor is treated as a single chunk for optimal access patterns
        """
        if group is not None:
            raise ValueError("group parameter is not supported for SafeTensors files")

        dimension_names = None
        revision = "main"  # Default to main branch

        if virtual_backend_kwargs:
            if "dimension_names" in virtual_backend_kwargs:
                dimension_names = virtual_backend_kwargs.pop("dimension_names")

            if "revision" in virtual_backend_kwargs:
                revision = virtual_backend_kwargs.pop("revision")

            if virtual_backend_kwargs:
                raise NotImplementedError(
                    f"SafeTensors reader does not support the following virtual_backend_kwargs: {list(virtual_backend_kwargs.keys())}"
                )

        if reader_options:
            raise NotImplementedError(
                "SafeTensors reader does not support non-empty reader_options."
            )

        _drop_vars = [] if drop_variables is None else list(drop_variables)

        manifest_store = SafeTensorsVirtualBackend._create_manifest_store(
            filepath=filepath,
            drop_variables=_drop_vars,
            dimension_names=dimension_names,
            revision=revision,
        )

        ds = manifest_store.to_virtual_dataset(
            loadable_variables=loadable_variables,
            decode_times=decode_times,
            indexes=indexes,
        )
        return ds

    @staticmethod
    def _create_chunk_manifest(
        filepath: str,
        offset: int,
        length: int,
        shape: tuple[int, ...],
    ) -> ChunkManifest:
        """
        Create a ChunkManifest for a tensor in a SafeTensors file.

        SafeTensors files store tensors as contiguous binary data. This method creates
        a chunk manifest that points to the exact location of a tensor within the file,
        treating the entire tensor as a single chunk for efficient memory mapping.

        Parameters
        ----------
        filepath : str
            Path to the SafeTensors file where the tensor data is stored.
        offset : int
            Byte offset from the start of the file where the tensor data begins.
            This is calculated as: header_size + 8 bytes + tensor_data_offset.
        length : int
            Length of the tensor data in bytes. Calculated from data_offsets
            as: end_offset - start_offset.
        shape : tuple[int, ...]
            Shape of the tensor. Used to determine the dimensionality for
            creating chunk keys.

        Returns
        -------
        ChunkManifest
            A ChunkManifest object containing a single chunk entry that references
            the tensor's location within the file.

        Notes
        -----
        - Each tensor is represented as a single chunk, regardless of size
        - Chunk keys are generated as: "0" for 1D, "0.0" for 2D, "0.0.0" for 3D, etc.

        Examples
        --------
        For a 3D tensor (shape=(10, 20, 30)) starting at byte 1024 with length 24000:
        >>> manifest = _create_chunk_manifest("model.safetensors", 1024, 24000, (10, 20, 30))
        >>> print(manifest.entries.keys())
        dict_keys(['0.0.0'])
        """
        # Create a single chunk key (e.g., "0" for a 1D tensor, "0.0" for a 2D tensor)
        key_parts = ["0"] * (len(shape) or 1)
        chunk_key = ChunkKey(".".join(key_parts))

        chunk_entry = ChunkEntry.with_validation(  # type: ignore[attr-defined]
            path=filepath,
            offset=offset,
            length=length,
        )

        chunk_entries = {chunk_key: chunk_entry}

        return ChunkManifest(entries=chunk_entries)

    @staticmethod
    def _map_dtype(dtype_str: str) -> np.dtype:
        """
        Map SafeTensors dtype string to NumPy dtype.

        SafeTensors uses its own dtype naming convention that needs to be mapped
        to NumPy dtypes for use in xarray. This method performs that mapping,
        supporting all standard dtypes including ML-specific types like BF16.

        Parameters
        ----------
        dtype_str : str
            SafeTensors dtype string. Valid values are:
            - Integer types: "I8", "I16", "I32", "I64"
            - Unsigned types: "U8", "U16", "U32", "U64"
            - Float types: "F16", "F32", "F64"
            - ML types: "BF16" (bfloat16), "F8_E5M2", "F8_E4M3" (float8 variants)
            - Boolean: "BOOL"

        Returns
        -------
        np.dtype
            Corresponding NumPy dtype object that can be used with xarray.

        Raises
        ------
        ValueError
            If the provided dtype_str is not recognized or supported.

        Notes
        -----
        - BF16, F8_E5M2, and F8_E4M3 require the ml_dtypes package

        Examples
        --------
        >>> dtype = _map_dtype("F32")
        >>> print(dtype)
        float32

        >>> dtype = _map_dtype("BOOL")
        >>> print(dtype)
        bool

        >>> dtype = _map_dtype("BF16")
        >>> print(dtype)
        bfloat16
        """
        try:
            # this import will register new numpy dtypes as a side-effect e.g. bfloat16
            import ml_dtypes  # noqa: F401
        except ImportError:
            raise ImportError(
                "The ml_dtypes package is required to read safetensors files. Please install it with pip install virtualizarr[safetensors]."
            )

        dtype_map = {
            "BOOL": np.dtype("bool"),
            "U8": np.dtype("uint8"),
            "I8": np.dtype("int8"),
            "I16": np.dtype("int16"),
            "U16": np.dtype("uint16"),
            "I32": np.dtype("int32"),
            "U32": np.dtype("uint32"),
            "I64": np.dtype("int64"),
            "U64": np.dtype("uint64"),
            "F16": np.dtype("float16"),
            "BF16": np.dtype(
                "bfloat16"
            ),  # TO DO: broken until zarr supports dtype extensions
            "F32": np.dtype("float32"),
            "F64": np.dtype("float64"),
            "F8_E5M2": np.dtype(
                "float8_e5m2"
            ),  # TO DO: broken until zarr supports dtype extensions
            "F8_E4M3": np.dtype(
                "float8_e4m3"
            ),  # TO DO: broken until zarr supports dtype extensions
        }

        if dtype_str not in dtype_map:
            raise ValueError(f"Unsupported SafeTensors dtype: {dtype_str}")

        return dtype_map[dtype_str]
