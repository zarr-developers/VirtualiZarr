# SafeTensors Reader User Guide

The SafeTensors reader in VirtualiZarr allows you to reference tensors stored in SafeTensors files. This guide explains how to use the reader effectively.

## What is SafeTensors Format?

SafeTensors is a file format developed by HuggingFace for storing tensors (multidimensional arrays)
that offers several advantages:
- Safe: No use of pickle, eliminating security concerns
- Efficient: Zero-copy access for fast loading
- Simple: Straightforward binary format with JSON header
- Language-agnostic: Available across Python, Rust, C++, and JavaScript

The format consists of:
- 8 bytes (header size): little-endian uint64 containing the size of the header
- JSON header: Contains metadata for all tensors (shapes, dtypes, offsets)
- Binary data: Contiguous tensor data

## How VirtualiZarr's SafeTensors Reader Works

VirtualiZarr's SafeTensors reader allows you to:
- Create "virtual" Zarr stores pointing to chunks of data inside SafeTensors files
- Open the virtual zarr stores as xarray DataArrays with named dimensions
- Access specific slices of tensors from cloud storage
- Preserve metadata from the original SafeTensors file

## Basic Usage

Opening a SafeTensors file is straightforward:

```python
import virtualizarr as vz

# Open a SafeTensors file
vds = vz.open_virtual_dataset("model.safetensors")

# Access tensors as xarray variables
weight = vds["weight"]
bias = vds["bias"]
```

## Custom Dimension Names

By default, dimensions are named generically (e.g., "weight_dim_0", "weight_dim_1"). You can provide custom dimension names for better semantics:

```python
# Define custom dimension names
custom_dims = {
    "weight": ["input_dims", "output_dims"],
    "bias": ["output_dims"]
}

# Open with custom dimension names
vds = vz.open_virtual_dataset(
    "model.safetensors",
    virtual_backend_kwargs={"dimension_names": custom_dims}
)

# Now dimensions have meaningful names
print(vds["weight"].dims)  # ('input_dims', 'output_dims')
print(vds["bias"].dims)    # ('output_dims',)
```

## Loading Specific Variables

You can specify which variables to load as eager arrays instead of virtual references:

```python
# Load specific variables as eager arrays
vds = vz.open_virtual_dataset(
    "model_weights.safetensors",
    loadable_variables=["small_tensor1", "small_tensor2"]
)

# These will be loaded as regular numpy arrays
small_tensor1 = vds["small_tensor1"]
# Large tensors remain virtual references
large_tensor = vds["large_tensor"]
```

## Working with Remote Files

The SafeTensors reader supports reading from the HuggingFace Hub:
```python
# HuggingFace Hub
vds = vz.open_virtual_dataset(
    "https://huggingface.co/openai-community/gpt2/model.safetensors",
    virtual_backend_kwargs={"revision": "main"}
)
```

It supports reading from object storage:

```python
# S3
vds = vz.open_virtual_dataset(
    "s3://my-bucket/model.safetensors",
    reader_options={
        "storage_options": {
            "key": "ACCESS_KEY",
            "secret": "SECRET_KEY",
            "region_name": "us-west-2"
        }
    }
)
```

## Accessing Metadata

SafeTensors files can contain metadata at the file level and tensor level:

```python
# Access file-level metadata
print(vds.attrs)  # File-level metadata

# Access tensor-specific metadata
print(vds["weight"].attrs)  # Tensor-specific metadata

# Access original SafeTensors dtype information
original_dtype = vds["weight"].attrs["original_safetensors_dtype"]
print(f"Original dtype: {original_dtype}")
```

## Known Limitations

### Performance Considerations
- Very large tensors (>1GB) are treated as a single chunk, which may impact memory usage when accessing small slices
- Files with thousands of tiny tensors may have overhead due to metadata handling

## Best Practices

- **For large tensors**: Use slicing to access only the portions you need
- **For remote files**: Use appropriate credentials and optimize access patterns
- **For many small tensors**: Consider loading them eagerly using `loadable_variables`
