"""
Tests for the SafeTensors reader in VirtualiZarr.
"""

import json
import os
import tempfile

import numpy as np
import pytest

from virtualizarr import open_virtual_dataset
from virtualizarr.backend import FileType
from virtualizarr.readers.safetensors import SafeTensorsVirtualBackend
from virtualizarr.tests import requires_network

try:
    from safetensors.numpy import save_file

    HAS_SAFETENSORS = True
except ImportError:
    HAS_SAFETENSORS = False


@pytest.fixture
def sample_safetensors_file():
    """Create a sample SafeTensors file for testing."""
    if not HAS_SAFETENSORS:
        pytest.skip("safetensors not installed")

    with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as tmpfile:
        filepath = tmpfile.name

        # TO DO: after zarr supports dtype extensions, test bfloat16 here
        tensors = {
            "tensor1": np.ones((10, 10), dtype=np.float32),
            "tensor2": np.zeros((5, 5), dtype=np.float32),
            "tensor3": np.arange(100, dtype=np.float32).reshape(10, 10),
        }

        metadata = {
            "framework": "numpy",
            "version": "1.0",
            "created_by": "virtualizarr_test",
        }

        save_file(tensors, filepath, metadata=metadata)

    yield filepath

    os.unlink(filepath)


@pytest.fixture
def sample_safetensors_file_with_complex_metadata():
    """Create a sample SafeTensors file with complex metadata for testing."""
    if not HAS_SAFETENSORS:
        pytest.skip("safetensors not installed")

    with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as tmpfile:
        filepath = tmpfile.name

        tensors = {
            "weight": np.random.randn(10, 20).astype(np.float32),
            "bias": np.random.randn(20).astype(np.float32),
        }

        # Convert complex nested metadata to strings as required by safetensors
        model_info = {
            "name": "test_model",
            "version": "1.0",
            "parameters": 220,
            "layers": [
                {"name": "layer1", "type": "linear"},
                {"name": "layer2", "type": "activation"},
            ],
        }
        training = {"epochs": 10, "optimizer": "adam", "learning_rate": 0.001}

        metadata = {
            "model_info": json.dumps(model_info),
            "training": json.dumps(training),
        }

        save_file(tensors, filepath, metadata=metadata)

    yield filepath

    os.unlink(filepath)


@pytest.mark.skipif(not HAS_SAFETENSORS, reason="safetensors not installed")
def test_reader_registration():
    """Test that the SafeTensors reader is properly registered in VirtualiZarr."""
    from virtualizarr.backend import VIRTUAL_BACKENDS

    assert "safetensors" in VIRTUAL_BACKENDS
    assert VIRTUAL_BACKENDS["safetensors"] == SafeTensorsVirtualBackend


@pytest.mark.skipif(not HAS_SAFETENSORS, reason="safetensors not installed")
def test_file_detection(sample_safetensors_file):
    """Test that VirtualiZarr correctly identifies the file as a SafeTensors file."""
    from virtualizarr.backend import automatically_determine_filetype

    filetype = automatically_determine_filetype(filepath=sample_safetensors_file)
    assert filetype == FileType.safetensors


@pytest.mark.skipif(not HAS_SAFETENSORS, reason="safetensors not installed")
def test_open_virtual_dataset(sample_safetensors_file):
    """Test opening a SafeTensors file as a virtual dataset."""
    vds = open_virtual_dataset(sample_safetensors_file)

    assert set(vds.variables) == {"tensor1", "tensor2", "tensor3"}

    assert vds["tensor1"].shape == (10, 10)
    assert vds["tensor2"].shape == (5, 5)
    assert vds["tensor3"].shape == (10, 10)


@pytest.mark.skipif(not HAS_SAFETENSORS, reason="safetensors not installed")
def test_tensor_values(sample_safetensors_file):
    """Test that the tensor values are correctly read."""

    loadable_variables = ["tensor1", "tensor2", "tensor3"]
    vds = open_virtual_dataset(
        sample_safetensors_file, loadable_variables=loadable_variables
    )

    tensor1 = vds["tensor1"].values
    tensor2 = vds["tensor2"].values
    tensor3 = vds["tensor3"].values

    expected_tensor1 = np.ones((10, 10))
    expected_tensor2 = np.zeros((5, 5))
    expected_tensor3 = np.arange(100).reshape(10, 10)

    np.testing.assert_array_equal(tensor1, expected_tensor1)
    np.testing.assert_array_equal(tensor2, expected_tensor2)
    np.testing.assert_array_equal(tensor3, expected_tensor3)


@pytest.mark.skipif(not HAS_SAFETENSORS, reason="safetensors not installed")
def test_custom_dimension_names(sample_safetensors_file):
    """Test specifying custom dimension names when opening a SafeTensors file."""

    custom_dims = {"tensor1": ["dim_x", "dim_y"], "tensor2": ["height", "width"]}

    vds = open_virtual_dataset(
        sample_safetensors_file, virtual_backend_kwargs={"dimension_names": custom_dims}
    )

    assert vds["tensor1"].dims == ("dim_x", "dim_y")
    assert vds["tensor2"].dims == ("height", "width")
    # tensor3 should use default dimension names
    assert vds["tensor3"].dims == ("tensor3_dim_0", "tensor3_dim_1")


@pytest.mark.skipif(not HAS_SAFETENSORS, reason="safetensors not installed")
def test_invalid_dimension_names(sample_safetensors_file):
    """Test error handling for invalid dimension names."""

    invalid_dims = {
        "tensor1": ["dim_x", "dim_y", "dim_z"]  # tensor1 is 2D but we provide 3 names
    }

    with pytest.raises(ValueError) as excinfo:
        _ = open_virtual_dataset(
            sample_safetensors_file,
            virtual_backend_kwargs={"dimension_names": invalid_dims},
        )

    assert "has 3 names, but tensor has 2 dimensions" in str(excinfo.value)


@pytest.mark.skipif(not HAS_SAFETENSORS, reason="safetensors not installed")
def test_metadata_preservation(sample_safetensors_file):
    """Test preservation of metadata from SafeTensors file."""

    vds = open_virtual_dataset(sample_safetensors_file)

    assert vds.attrs["framework"] == "numpy"
    assert vds.attrs["version"] == "1.0"
    assert vds.attrs["created_by"] == "virtualizarr_test"


@pytest.mark.skipif(not HAS_SAFETENSORS, reason="safetensors not installed")
def test_complex_metadata(sample_safetensors_file_with_complex_metadata):
    """Test handling of complex nested metadata."""

    vds = open_virtual_dataset(sample_safetensors_file_with_complex_metadata)

    assert "model_info" in vds.attrs
    assert "training" in vds.attrs

    model_info = json.loads(vds.attrs["model_info"])
    training = json.loads(vds.attrs["training"])

    assert model_info["name"] == "test_model"
    assert model_info["parameters"] == 220
    assert training["epochs"] == 10
    assert training["optimizer"] == "adam"

    assert vds["weight"].shape == (10, 20)
    assert vds["bias"].shape == (20,)

    assert vds["weight"].attrs["original_safetensors_dtype"] == "F32"
    assert vds["bias"].attrs["original_safetensors_dtype"] == "F32"


@pytest.mark.skipif(not HAS_SAFETENSORS, reason="safetensors not installed")
def test_enhanced_metadata(sample_safetensors_file):
    """Test that enhanced metadata is correctly created and preserved."""

    vds = open_virtual_dataset(sample_safetensors_file)

    assert vds["tensor1"].attrs["original_safetensors_dtype"] == "F32"
    assert vds["tensor2"].attrs["original_safetensors_dtype"] == "F32"
    assert vds["tensor3"].attrs["original_safetensors_dtype"] == "F32"


@pytest.mark.skipif(not HAS_SAFETENSORS, reason="safetensors not installed")
def test_relative_path_handling(sample_safetensors_file):
    """Test that relative paths to SafeTensors files are handled correctly."""
    import os

    # Get the full path and then compute a relative path from current directory
    full_path = sample_safetensors_file
    current_dir = os.getcwd()

    # Make the relative path by removing the common prefix
    relative_path = os.path.relpath(full_path, current_dir)

    # First test without loading data
    vds_metadata = open_virtual_dataset(relative_path)

    # Verify that the content is accessible
    assert set(vds_metadata.variables) == {"tensor1", "tensor2", "tensor3"}
    assert vds_metadata["tensor1"].shape == (10, 10)
    assert vds_metadata["tensor2"].shape == (5, 5)
    assert vds_metadata["tensor3"].shape == (10, 10)

    # Now test with loadable_variables to actually access the data
    vds = open_virtual_dataset(relative_path, loadable_variables=["tensor1"])

    # Verify we can access the data
    tensor1 = vds["tensor1"].values
    np.testing.assert_array_equal(tensor1, np.ones((10, 10)))


@pytest.fixture
def sample_safetensors_file_with_many_small_tensors():
    """Create a sample SafeTensors file with many small tensors for testing."""
    if not HAS_SAFETENSORS:
        pytest.skip("safetensors not installed")

    with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as tmpfile:
        filepath = tmpfile.name

        # Create 1000 small tensors
        tensors = {
            f"tensor_{i}": np.random.randn(10, 10).astype(np.float32)
            for i in range(1000)
        }

        metadata = {
            "description": "many_small_tensors",
            "tensor_count": "1000",
            "shape": "(10, 10)",
            "dtype": "float32",
        }

        save_file(tensors, filepath, metadata=metadata)

    yield filepath

    os.unlink(filepath)


@pytest.mark.skipif(not HAS_SAFETENSORS, reason="safetensors not installed")
def test_many_small_tensors(sample_safetensors_file_with_many_small_tensors):
    """Test handling a SafeTensors file with many small tensors."""

    vds = open_virtual_dataset(sample_safetensors_file_with_many_small_tensors)

    # Check that all 1000 tensors are present
    assert len(vds.variables) == 1000

    # Check that all tensors have the expected names
    expected_names = {f"tensor_{i}" for i in range(1000)}
    assert set(vds.variables) == expected_names

    # Check metadata preservation
    assert vds.attrs["description"] == "many_small_tensors"
    assert vds.attrs["tensor_count"] == "1000"
    assert vds.attrs["shape"] == "(10, 10)"
    assert vds.attrs["dtype"] == "float32"

    # Check shape of several tensors
    for i in [0, 500, 999]:  # First, middle, and last tensor
        tensor_name = f"tensor_{i}"
        assert vds[tensor_name].shape == (10, 10)
        assert vds[tensor_name].attrs["original_safetensors_dtype"] == "F32"

    # Load a few tensors to check values
    loadable_variables = [f"tensor_{i}" for i in range(3)]  # Load first 3 tensors
    vds_with_data = open_virtual_dataset(
        sample_safetensors_file_with_many_small_tensors,
        loadable_variables=loadable_variables,
    )

    for i in range(3):
        tensor_name = f"tensor_{i}"
        tensor_data = vds_with_data[tensor_name].values
        assert tensor_data.shape == (10, 10)
        assert tensor_data.dtype == np.float32
        # Check that data is not all zeros
        assert not np.all(tensor_data == 0)


@requires_network
@pytest.mark.skipif(not HAS_SAFETENSORS, reason="safetensors not installed")
def test_open_huggingface_safetensors():
    """Test opening a SafeTensors file from Hugging Face Hub with metadata inspection only."""

    # This is a widely-downloaded GPT-2 model (500mb) file from Hugging Face Hub with standard dtypes
    vds = open_virtual_dataset(
        "https://huggingface.co/openai-community/gpt2/resolve/main/model.safetensors"
    )

    assert len(vds.variables) > 0

    # Check that some expected GPT-2 parameter tensors are present
    # These are common parameter names in GPT-2 models
    expected_tensors = [
        "wte.weight",  # Word Token Embeddings
        "wpe.weight",  # Word Position Embeddings
        "ln_f.weight",  # Final layer norm weight
    ]

    for tensor_name in expected_tensors:
        assert tensor_name in vds.variables

    for tensor_name in expected_tensors:
        var = vds[tensor_name]
        assert hasattr(var, "shape")
        assert hasattr(var, "dtype")
        assert "original_safetensors_dtype" in var.attrs

    # Check wte.weight shape - should be (vocab_size, embedding_dim)
    wte_shape = vds["wte.weight"].shape
    assert len(wte_shape) == 2
    assert wte_shape[0] == 50257  # GPT-2 vocabulary size
    assert wte_shape[1] == 768  # GPT-2 embedding dimension

    # Check wpe.weight shape - should be (context_length, embedding_dim)
    wpe_shape = vds["wpe.weight"].shape
    assert len(wpe_shape) == 2
    assert wpe_shape[0] == 1024  # GPT-2 context length
    assert wpe_shape[1] == 768  # GPT-2 embedding dimension

    # Test loading a small subset of data from a large tensor
    # Only load a small slice to avoid downloading the entire 500MB file
    loadable_vds = open_virtual_dataset(
        "https://huggingface.co/openai-community/gpt2/model.safetensors",
        loadable_variables=["wpe.weight"],  # shape (1024, 768)
    )

    # Get a small subset from the word positin embeddings: rows 1000-1100, columns 100-200
    subset = loadable_vds["wpe.weight"][1000:1100, 100:200].values
    assert subset.shape == (24, 100)
    assert subset.dtype == np.float32

    # Verify that we actually got data (not all zeros or NaNs)
    assert not np.all(subset == 0)
    assert not np.any(np.isnan(subset))
