import warnings

import h5py  # type: ignore
import numpy as np
import pytest
import xarray as xr
import zarr
from obspec_utils.registry import ObjectStoreRegistry
from obstore.store import from_url

from virtualizarr import open_virtual_dataset
from virtualizarr.parsers import HDFParser
from virtualizarr.parsers.hdf.hdf import LINKS_REFERENCES_CONVENTION, REF_CONVENTION
from virtualizarr.tests import (
    requires_hdf5plugin,
    requires_imagecodecs,
)
from virtualizarr.tests.utils import manifest_store_from_hdf_url, open_hdf5


@requires_hdf5plugin
@requires_imagecodecs
class TestDatasetChunkManifest:
    @pytest.mark.xfail(
        reason="Tutorial data non coord dimensions are serialized with big endidan types and internally dropped"
    )
    def test_empty_chunks(self, empty_chunks_hdf5_url):
        manifest_store = manifest_store_from_hdf_url(empty_chunks_hdf5_url)
        assert manifest_store._group.arrays["data"].shape == (0,)

    def test_empty_dataset(self, empty_dataset_hdf5_url):
        manifest_store = manifest_store_from_hdf_url(empty_dataset_hdf5_url)
        assert manifest_store._group.arrays["data"].shape == (0,)

    def test_no_chunking(self, no_chunks_hdf5_url):
        manifest_store = manifest_store_from_hdf_url(no_chunks_hdf5_url)
        assert manifest_store._group.arrays["data"].manifest.shape_chunk_grid == (1, 1)

    def test_chunked(self, chunked_hdf5_url):
        manifest_store = manifest_store_from_hdf_url(chunked_hdf5_url)
        assert manifest_store._group.arrays["data"].manifest.shape_chunk_grid == (2, 2)

    def test_chunked_roundtrip(self, chunked_roundtrip_hdf5_url):
        manifest_store = manifest_store_from_hdf_url(chunked_roundtrip_hdf5_url)
        assert manifest_store._group.arrays["var2"].manifest.shape_chunk_grid == (2, 8)


@requires_hdf5plugin
@requires_imagecodecs
class TestDatasetDims:
    def test_single_dimension_scale(self, single_dimension_scale_hdf5_url):
        manifest_store = manifest_store_from_hdf_url(single_dimension_scale_hdf5_url)
        assert manifest_store._group.arrays["data"].metadata.dimension_names == ("x",)

    def test_is_dimension_scale(self, is_scale_hdf5_url):
        manifest_store = manifest_store_from_hdf_url(is_scale_hdf5_url)
        assert manifest_store._group.arrays["data"].metadata.dimension_names == (
            "data",
        )

    def test_multiple_dimension_scales(self, multiple_dimension_scales_hdf5_url):
        with pytest.raises(ValueError, match="dimension scales attached"):
            manifest_store_from_hdf_url(multiple_dimension_scales_hdf5_url)

    def test_no_dimension_scales(self, no_chunks_hdf5_url):
        manifest_store = manifest_store_from_hdf_url(no_chunks_hdf5_url)
        assert manifest_store._group.arrays["data"].metadata.dimension_names == (
            "phony_dim_0",
            "phony_dim_1",
        )


@requires_hdf5plugin
@requires_imagecodecs
class TestDatasetToManifestArray:
    def test_chunked_dataset(self, chunked_dimensions_netcdf4_url):
        manifest_store = manifest_store_from_hdf_url(chunked_dimensions_netcdf4_url)
        assert manifest_store._group.arrays["data"].metadata.chunks == (50, 50)

    def test_not_chunked_dataset(self, single_dimension_scale_hdf5_url):
        manifest_store = manifest_store_from_hdf_url(single_dimension_scale_hdf5_url)
        assert manifest_store._group.arrays["data"].metadata.chunks == (2,)

    def test_unlimited_dimension_chunks(self, unlimited_dimension_netcdf4_url):
        # see https://github.com/zarr-developers/VirtualiZarr/issues/803
        # HDF5 reports chunks=(512,) for an unlimited dim holding only 5 values;
        # the reported chunk shape is trimmed to the actual array shape
        manifest_store = manifest_store_from_hdf_url(unlimited_dimension_netcdf4_url)
        time = manifest_store._group.arrays["time"]
        assert time.shape == (5,)
        assert time.metadata.chunks == (5,)
        # the trimmed manifest must still read back the written values
        result = zarr.open(manifest_store, mode="r")["time"][:]
        np.testing.assert_array_equal(result, np.full(5, 10, dtype="i8"))

    def test_unlimited_dimension_compressed_chunks_not_trimmed(
        self, unlimited_dimension_compressed_hdf5_url
    ):
        # a compressed oversized chunk cannot be byte-trimmed to the array shape,
        # so the parser leaves it as-is (reading still works by cropping); the
        # user is warned later, at to_virtual_dataset time, to load the variable
        manifest_store = manifest_store_from_hdf_url(
            unlimited_dimension_compressed_hdf5_url
        )
        data = manifest_store._group.arrays["data"]
        assert data.shape == (5,)
        assert data.metadata.chunks == (512,)

    def test_unlimited_dimension_compressed_chunks_warn_unless_loaded(
        self, unlimited_dimension_compressed_hdf5_url
    ):
        manifest_store = manifest_store_from_hdf_url(
            unlimited_dimension_compressed_hdf5_url
        )
        # left virtual, the oversized chunk warns and points at loadable_variables
        with pytest.warns(UserWarning, match="loadable_variables"):
            manifest_store.to_virtual_dataset()
        # loading the variable silences the warning and reads back correctly
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            vds = manifest_store.to_virtual_dataset(loadable_variables=["data"])
        assert not [w for w in record if "loadable_variables" in str(w.message)]
        np.testing.assert_array_equal(vds["data"].values, np.full(5, 10, dtype="i8"))

    def test_dataset_attributes(self, string_attributes_hdf5_url):
        manifest_store = manifest_store_from_hdf_url(string_attributes_hdf5_url)
        metadata = manifest_store._group.arrays["data"].metadata
        assert metadata.attributes["attribute_name"] == "attribute_name"

    def test_bytes_array_attributes(self, bytes_attributes_hdf5_url):
        manifest_store = manifest_store_from_hdf_url(bytes_attributes_hdf5_url)
        attributes = manifest_store._group.arrays["data"].metadata.attributes
        assert attributes["character_array"] == ["2", "0", "4", "2"]
        assert attributes["array"] == ["first", "second"]
        assert attributes["array_2d"] == [["a", "b"], ["c", "d"]]

    def test_scalar_fill_value(self, scalar_fill_value_hdf5_url):
        manifest_store = manifest_store_from_hdf_url(scalar_fill_value_hdf5_url)
        metadata = manifest_store._group.arrays["data"].metadata
        assert metadata.fill_value == 42

    def test_cf_fill_value(self, cf_fill_value_hdf5_file):
        cf_fill_value_hdf5_url = f"file://{cf_fill_value_hdf5_file}"
        with open_hdf5(cf_fill_value_hdf5_file) as f:
            dtype = f["data"].dtype
        if dtype.kind in "S":
            pytest.xfail("Investigate fixed-length binary encoding in Zarr v3")
        if dtype.names:
            pytest.xfail("To fix, structured dtype fill value encoding for Zarr parser")
        manifest_store = manifest_store_from_hdf_url(cf_fill_value_hdf5_url)
        metadata = manifest_store._group.arrays["data"].metadata
        assert "_FillValue" in metadata.attributes

    def test_string_dtype_fill_value(self, string_dtype_hdf5_url):
        manifest_store = manifest_store_from_hdf_url(string_dtype_hdf5_url)
        metadata = manifest_store._group.arrays["data"].metadata
        assert isinstance(metadata.fill_value, (str, bytes, np.bytes_))

    def test_fixed_length_bytes_roundtrip(self, fixed_length_bytes_hdf5_url):
        with (
            manifest_store_from_hdf_url(fixed_length_bytes_hdf5_url) as ms,
            xr.open_zarr(ms, zarr_format=3, consolidated=False).load() as ds,
        ):
            np.testing.assert_array_equal(
                ds["data"].to_numpy(), np.array([b"hello", b"world"], dtype="S10")
            )

    def test_fixed_length_bytes_fill_value_preserved(
        self, non_utf8_fill_value_hdf5_url
    ):
        manifest_store = manifest_store_from_hdf_url(non_utf8_fill_value_hdf5_url)
        metadata = manifest_store._group.arrays["data"].metadata
        assert metadata.fill_value == b"\xff\xfe\xff\xfe\xff"

    def test_variable_length_string_roundtrip(self, vlen_string_hdf5_url):
        # the strings live in the HDF5 global heap, so they are read at parse
        # time and inlined, and no warning is needed
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            manifest_store = manifest_store_from_hdf_url(vlen_string_hdf5_url)
        marr = manifest_store._group.arrays["data"]
        assert len(marr.manifest._inlined) == 1
        arr = zarr.open_array(manifest_store, path="data", mode="r", zarr_format=3)
        np.testing.assert_array_equal(arr[...], np.array(["hello", "world"]))

    def test_ascii_variable_length_string_roundtrip(self, ascii_vlen_string_hdf5_url):
        # an ascii-cset vlen string is still a vlen string, so it should be read
        # like the utf-8 case rather than fail the whole file as an unsupported
        # object dtype
        manifest_store = manifest_store_from_hdf_url(ascii_vlen_string_hdf5_url)
        metadata = manifest_store._group.arrays["data"].metadata
        assert metadata.data_type.to_native_dtype() == np.dtypes.StringDType()
        arr = zarr.open_array(manifest_store, path="data", mode="r", zarr_format=3)
        np.testing.assert_array_equal(arr[...], np.array(["hello", "world"]))

    @pytest.mark.parametrize(
        "kwargs",
        [
            pytest.param(
                dict(
                    data=np.array([f"s{i}" * (i % 4) for i in range(10)], dtype=object),
                    chunks=(4,),
                    compression="gzip",
                ),
                id="chunked-gzip-partial-edge-chunk",
            ),
            pytest.param(
                dict(
                    data=np.array([["α", "b", "c"], ["d", "", "ffff"]], dtype=object),
                    chunks=(1, 2),
                ),
                id="2d",
            ),
            pytest.param(dict(data="hello"), id="scalar"),
            pytest.param(dict(shape=(3,), chunks=(2,)), id="unwritten"),
            pytest.param(dict(shape=(0,)), id="empty"),
        ],
    )
    def test_variable_length_string_layouts(self, tmp_path, kwargs):
        filepath = str(tmp_path / "vlen_strings.h5")
        with h5py.File(filepath, "w") as f:
            f.create_dataset("data", dtype=h5py.string_dtype(), **kwargs)
        with h5py.File(filepath, "r") as f:
            expected = np.asarray(f["data"].asstr()[()], dtype=object)

        manifest_store = manifest_store_from_hdf_url(f"file://{filepath}")
        arr = zarr.open_array(manifest_store, path="data", mode="r", zarr_format=3)
        actual = arr[...] if arr.shape else arr[()]
        assert np.shape(actual) == expected.shape
        np.testing.assert_array_equal(np.asarray(actual, dtype=object), expected)

    def test_variable_length_string_loadable_variable(self, vlen_string_hdf5_url):
        registry = ObjectStoreRegistry()
        registry.register(vlen_string_hdf5_url, from_url("file:///"))
        with open_virtual_dataset(
            url=vlen_string_hdf5_url,
            registry=registry,
            parser=HDFParser(),
            loadable_variables=["data"],
        ) as ds:
            np.testing.assert_array_equal(
                ds["data"].values, np.array(["hello", "world"])
            )

    def test_string_dtype_cf_fill_value(self, string_dtype_with_fillvalue_hdf5_url):
        manifest_store = manifest_store_from_hdf_url(
            string_dtype_with_fillvalue_hdf5_url
        )
        metadata = manifest_store._group.arrays["data"].metadata
        assert "_FillValue" in metadata.attributes
        assert isinstance(metadata.attributes["_FillValue"], str)

    def test_cf_array_fill_value(self, cf_array_fill_value_hdf5_file):
        cf_array_fill_value_hdf5_url = f"file://{cf_array_fill_value_hdf5_file}"
        manifest_store = manifest_store_from_hdf_url(cf_array_fill_value_hdf5_url)
        metadata = manifest_store._group.arrays["data"].metadata
        assert not isinstance(metadata.attributes["_FillValue"], np.ndarray)


def _parse(url, **kwargs):
    registry = ObjectStoreRegistry()
    registry.register(url, from_url("file:///"))
    return HDFParser(**kwargs)(url=url, registry=registry)


@pytest.fixture
def object_references_hdf5_url(tmp_path) -> str:
    filepath = str(tmp_path / "object_references.h5")
    with h5py.File(filepath, "w") as f:
        f["values"] = np.arange(3.0)
        f.create_group("devices/probe")
        electrodes = f.create_group("general/extracellular_ephys/electrodes")
        electrodes.create_dataset(
            "group",
            data=[f["devices/probe"].ref] * 5 + [h5py.Reference(), f["values"].ref],
            dtype=h5py.ref_dtype,
            chunks=(3,),
        )
        f.create_dataset("scalar", data=f["values"].ref, dtype=h5py.ref_dtype)
        f.create_dataset(
            "grid",
            data=np.array([[f["values"].ref, f["devices/probe"].ref]] * 2),
            dtype=h5py.ref_dtype,
        )
    return f"file://{filepath}"


@requires_hdf5plugin
@requires_imagecodecs
class TestObjectReferences:
    def test_off_by_default(self, object_references_hdf5_url):
        with pytest.raises(NotImplementedError, match="links_and_references=True"):
            _parse(object_references_hdf5_url)

    def test_paths_relative_to_the_array(self, object_references_hdf5_url):
        store = _parse(object_references_hdf5_url, links_and_references=True)
        root = zarr.open_group(store, mode="r", zarr_format=3)
        arr = root["general/extracellular_ephys/electrodes/group"]
        # ".." is the array's parent group, as in the ref convention
        assert arr[...].tolist() == ["../../../../devices/probe"] * 5 + [
            "",
            "../../../../values",
        ]
        assert root["scalar"][()] == "../values"
        assert root["grid"][...].tolist() == [["../values", "../devices/probe"]] * 2

    def test_marked_with_the_convention(self, object_references_hdf5_url):
        store = _parse(object_references_hdf5_url, links_and_references=True)
        marr = store._group["general"]["extracellular_ephys"]["electrodes"].arrays[
            "group"
        ]
        attributes = marr.metadata.attributes
        assert attributes["links_references"] == {"references": True}
        assert attributes["zarr_conventions"] == [LINKS_REFERENCES_CONVENTION]
        assert marr.metadata.fill_value == ""
        assert len(marr.manifest._inlined) == 3  # 7 references in chunks of 3

    def test_each_target_name_is_found_once(
        self, object_references_hdf5_url, monkeypatch
    ):
        """The name of an object opened by reference makes HDF5 search the file."""
        from virtualizarr.parsers.hdf.hdf import _reference_paths

        path = object_references_hdf5_url.removeprefix("file://")
        with h5py.File(path, "r") as f:
            read = _reference_paths(f["general/extracellular_ephys/electrodes/group"])
            calls = []
            name = h5py.Group.name

            def counting(self):
                calls.append(1)
                return name.fget(self)

            monkeypatch.setattr(h5py.Group, "name", property(counting))
            assert (
                read((slice(0, 7),)).tolist()[:5] == ["../../../../devices/probe"] * 5
            )
        # /devices/probe is the target of five references, and its name is found once
        assert len(calls) == 1

    def test_region_references_still_raise(self, tmp_path):
        filepath = str(tmp_path / "region_references.h5")
        with h5py.File(filepath, "w") as f:
            f["values"] = np.arange(3.0)
            f.create_dataset(
                "regions", data=[f["values"].regionref[0:2]], dtype=h5py.regionref_dtype
            )
        with pytest.raises(NotImplementedError) as error:
            _parse(f"file://{filepath}", links_and_references=True)
        assert "links_and_references" not in str(error.value)


@pytest.fixture
def links_hdf5_url(tmp_path) -> str:
    with h5py.File(tmp_path / "other.h5", "w") as other:
        other["acquisition/raw"] = np.arange(4)
    filepath = str(tmp_path / "links.h5")
    with h5py.File(filepath, "w") as f:
        f["values"] = np.arange(3.0)
        f.create_group("general/devices/probe")
        g = f.create_group("general/extracellular_ephys/g")
        g["device"] = h5py.SoftLink("/general/devices/probe")
        g["data"] = h5py.SoftLink("/values")
        g.create_group("child")["inner"] = np.arange(2)
        g["inner_alias"] = h5py.SoftLink("child/inner")  # relative to g
        g["dangling"] = h5py.SoftLink("/nowhere")
        f["raw"] = h5py.ExternalLink("other.h5", "/acquisition/raw")
        f["values_again"] = f["values"]  # a second hard link
    return f"file://{filepath}"


@pytest.fixture
def compound_hdf5_url(tmp_path) -> str:
    filepath = str(tmp_path / "compound.h5")
    with h5py.File(filepath, "w") as f:
        f["values"] = np.arange(3.0)
        with_references = np.dtype(
            [
                ("idx_start", "<i4"),
                ("count", "<i4"),
                ("label", h5py.string_dtype()),
                ("timeseries", h5py.ref_dtype),
            ]
        )
        f.create_dataset(
            "rows",
            data=np.array(
                [
                    (0, 3, "é one", f["values"].ref),
                    (3, 2, "two", h5py.Reference()),
                    (5, 1, "", f["values"].ref),
                ],
                dtype=with_references,
            ),
            chunks=(2,),
        )
        strings_only = np.dtype([("id", "<i8"), ("name", h5py.string_dtype())])
        f.create_dataset(
            "names", data=np.array([(1, "a"), (2, "bcd")], dtype=strings_only)
        )
        f.create_dataset(
            "scalar", data=np.array((7, "seven"), dtype=strings_only), shape=()
        )
        numeric = np.dtype([("a", "<i4"), ("b", "<f8")])
        f.create_dataset("numeric", data=np.array([(1, 2.0), (3, 4.0)], dtype=numeric))
    return f"file://{filepath}"


@requires_hdf5plugin
@requires_imagecodecs
class TestLinks:
    def test_links_are_copied_by_default(self, tmp_path):
        filepath = str(tmp_path / "soft.h5")
        with h5py.File(filepath, "w") as f:
            f["values"] = np.arange(3.0)
            f["alias"] = h5py.SoftLink("/values")
        group = _parse(f"file://{filepath}")._group
        assert sorted(group.arrays) == ["alias", "values"]
        assert "links_references" not in group.metadata.attributes

    def test_links_are_recorded(self, links_hdf5_url):
        group = _parse(links_hdf5_url, links_and_references=True)._group
        g = group.groups["general"].groups["extracellular_ephys"].groups["g"]
        assert sorted(g.arrays) == []
        assert sorted(g.groups) == ["child"]
        assert g.metadata.attributes["links_references"] == {
            "links": {
                "device": {"ref": {"node": "../../devices/probe"}},
                "data": {"ref": {"node": "../../../values"}},
                "inner_alias": {"ref": {"node": "child/inner"}},
                "dangling": {"ref": {"node": "../../../nowhere"}},
            }
        }
        assert g.metadata.attributes["zarr_conventions"] == [
            LINKS_REFERENCES_CONVENTION,
            REF_CONVENTION,
        ]

    def test_external_and_hard_links(self, links_hdf5_url):
        group = _parse(links_hdf5_url, links_and_references=True)._group
        assert sorted(group.arrays) == ["values"]
        directory = links_hdf5_url.rpartition("/")[0]
        assert group.metadata.attributes["links_references"]["links"] == {
            "raw": {
                "ref": {"uri": f"{directory}/other.h5", "node": "/acquisition/raw"}
            },
            "values_again": {"ref": {"node": "values"}},
        }

    def test_hard_links_in_other_groups(self, tmp_path):
        filepath = str(tmp_path / "hard.h5")
        with h5py.File(filepath, "w") as f:
            electrodes = f.create_group("general/electrodes")
            electrodes["location"] = np.arange(2)
            f["probe/electrodes"] = electrodes
            f["electrodes"] = electrodes
        group = _parse(f"file://{filepath}", links_and_references=True)._group
        # The group is stored under the first of its names, /electrodes
        assert sorted(group.groups["electrodes"].arrays) == ["location"]
        for name in ["general", "probe"]:
            member = group.groups[name]
            assert sorted(member.groups) == []
            assert member.metadata.attributes["links_references"]["links"] == {
                "electrodes": {"ref": {"node": "../electrodes"}}
            }

    def test_drop_variables_applies_to_links(self, links_hdf5_url):
        store = _parse(
            links_hdf5_url, links_and_references=True, drop_variables=["raw"]
        )
        links = store._group.metadata.attributes["links_references"]["links"]
        assert "raw" not in links

    def test_links_read_through_zarr(self, links_hdf5_url):
        store = _parse(links_hdf5_url, links_and_references=True)
        root = zarr.open_group(store, mode="r", zarr_format=3)
        attrs = root["general/extracellular_ephys/g"].attrs
        assert attrs["links_references"]["links"]["device"] == {
            "ref": {"node": "../../devices/probe"}
        }
        np.testing.assert_array_equal(root["values"][...], np.arange(3.0))


@pytest.fixture
def reference_attributes_hdf5_url(tmp_path) -> str:
    filepath = str(tmp_path / "reference_attributes.h5")
    with h5py.File(filepath, "w") as f:
        f["values"] = np.arange(3.0)
        table = f.create_group("general/extracellular_ephys/electrodes")
        f["values"].attrs["table"] = table.ref
        f["values"].attrs["many"] = np.array(
            [table.ref, f["values"].ref, h5py.Reference()], dtype=h5py.ref_dtype
        )
        f["values"].attrs["units"] = "volts"
        g = f.create_group("general/devices/probe")
        g.attrs["source"] = f["values"].ref
        g["alias"] = h5py.SoftLink("/values")
    return f"file://{filepath}"


@requires_hdf5plugin
@requires_imagecodecs
class TestReferenceAttributes:
    def test_unchanged_by_default(self, reference_attributes_hdf5_url):
        attributes = (
            _parse(reference_attributes_hdf5_url)
            ._group.arrays["values"]
            .metadata.attributes
        )
        assert isinstance(attributes["table"], h5py.Reference)

    def test_ref_objects(self, reference_attributes_hdf5_url):
        group = _parse(reference_attributes_hdf5_url, links_and_references=True)._group
        attributes = group.arrays["values"].metadata.attributes
        assert attributes["table"] == {
            "ref": {"node": "../general/extracellular_ephys/electrodes"}
        }
        assert attributes["many"] == [
            {"ref": {"node": "../general/extracellular_ephys/electrodes"}},
            {"ref": {"node": "."}},
            None,
        ]
        assert attributes["units"] == "volts"
        assert attributes["zarr_conventions"] == [REF_CONVENTION]

    def test_group_with_links_registers_ref_once(self, reference_attributes_hdf5_url):
        group = _parse(reference_attributes_hdf5_url, links_and_references=True)._group
        probe = group.groups["general"].groups["devices"].groups["probe"]
        attributes = probe.metadata.attributes
        assert attributes["source"] == {"ref": {"node": "../../../values"}}
        assert attributes["links_references"] == {
            "links": {"alias": {"ref": {"node": "../../../values"}}}
        }
        assert sorted(c["name"] for c in attributes["zarr_conventions"]) == [
            "links_references",
            "ref",
        ]

    def test_read_through_zarr(self, reference_attributes_hdf5_url):
        store = _parse(reference_attributes_hdf5_url, links_and_references=True)
        values = zarr.open_group(store, mode="r", zarr_format=3)["values"]
        assert values.attrs["table"] == {
            "ref": {"node": "../general/extracellular_ephys/electrodes"}
        }

    def test_netcdf_dimension_attributes_stay_hidden(
        self, chunked_dimensions_netcdf4_url
    ):
        store = _parse(chunked_dimensions_netcdf4_url, links_and_references=True)
        for marr in store._group.arrays.values():
            assert "DIMENSION_LIST" not in marr.metadata.attributes
            assert "REFERENCE_LIST" not in marr.metadata.attributes


@pytest.mark.parametrize(
    "filepath, filename, expected",
    [
        ("file:///data/a.h5", "b.h5", "file:///data/b.h5"),
        ("file:///data/x/a.h5", "../b.h5", "file:///data/b.h5"),
        ("s3://bucket/dir/a.nwb", "sub/b.nwb", "s3://bucket/dir/sub/b.nwb"),
        ("s3://bucket/dir/a.nwb", "/abs/b.nwb", "file:///abs/b.nwb"),
        ("file:///data/a.h5", "https://example.org/b.h5", "https://example.org/b.h5"),
    ],
)
def test_resolve_external(filepath, filename, expected):
    from virtualizarr.parsers.hdf.hdf import _resolve_external

    assert _resolve_external(filepath, filename) == expected


@requires_hdf5plugin
@requires_imagecodecs
class TestCompoundObjectFields:
    def test_reference_fields_need_the_flag(self, compound_hdf5_url):
        with pytest.raises(NotImplementedError, match="links_and_references=True"):
            _parse(compound_hdf5_url)

    def test_strings_and_references(self, compound_hdf5_url):
        store = _parse(compound_hdf5_url, links_and_references=True)
        rows = zarr.open_group(store, mode="r", zarr_format=3)["rows"]
        assert rows.dtype == np.dtype(
            [
                ("idx_start", "<i4"),
                ("count", "<i4"),
                ("label", "<U5"),
                ("timeseries", "<U9"),
            ]
        )
        assert rows[...].tolist() == [
            (0, 3, "é one", "../values"),
            (3, 2, "two", ""),
            (5, 1, "", "../values"),
        ]
        marr = store._group.arrays["rows"]
        attributes = marr.metadata.attributes
        assert attributes["links_references"] == {"references": ["timeseries"]}
        assert attributes["zarr_conventions"] == [LINKS_REFERENCES_CONVENTION]
        assert len(marr.manifest._inlined) == 2  # 3 rows in chunks of 2

    def test_string_fields_without_the_flag(self, tmp_path):
        filepath = str(tmp_path / "names.h5")
        with h5py.File(filepath, "w") as f:
            dtype = np.dtype([("id", "<i8"), ("name", h5py.string_dtype())])
            f.create_dataset(
                "names", data=np.array([(1, "a"), (2, "bcd")], dtype=dtype)
            )
            f.create_dataset(
                "scalar", data=np.array((7, "seven"), dtype=dtype), shape=()
            )
        store = _parse(f"file://{filepath}")
        root = zarr.open_group(store, mode="r", zarr_format=3)
        assert root["names"][...].tolist() == [(1, "a"), (2, "bcd")]
        assert root["scalar"][()].tolist() == (7, "seven")
        assert (
            "links_references" not in store._group.arrays["names"].metadata.attributes
        )

    def test_numeric_compound_is_still_referenced(self, compound_hdf5_url):
        store = _parse(compound_hdf5_url, links_and_references=True)
        marr = store._group.arrays["numeric"]
        assert not marr.manifest._inlined
        numeric = zarr.open_group(store, mode="r", zarr_format=3)["numeric"]
        assert numeric[...].tolist() == [(1, 2.0), (3, 4.0)]


@requires_hdf5plugin
@requires_imagecodecs
class TestExtractAttributes:
    def test_root_attribute(self, root_attributes_hdf5_url):
        manifest_store = manifest_store_from_hdf_url(root_attributes_hdf5_url)
        assert (
            manifest_store._group.metadata.attributes["attribute_name"]
            == "attribute_name"
        )

    def test_multiple_attributes(self, string_attributes_hdf5_url):
        manifest_store = manifest_store_from_hdf_url(string_attributes_hdf5_url)
        metadata = manifest_store._group.arrays["data"].metadata
        assert len(metadata.attributes.keys()) == 2


@requires_hdf5plugin
@requires_imagecodecs
class TestManifestGroupFromHDF:
    def test_variable_with_dimensions(self, chunked_dimensions_netcdf4_url):
        manifest_store = manifest_store_from_hdf_url(chunked_dimensions_netcdf4_url)
        assert len(manifest_store._group.arrays) == 3

    def test_nested_groups_are_ignored_when_group_is_specificed(
        self, nested_group_hdf5_url
    ):
        manifest_store = manifest_store_from_hdf_url(
            nested_group_hdf5_url, group="group"
        )
        assert len(manifest_store._group.arrays) == 1

    def test_nested_groups_are_detected(self, nested_group_hdf5_url):
        manifest_store = manifest_store_from_hdf_url(nested_group_hdf5_url)
        assert len(manifest_store._group["group"]["nested_group"].arrays) == 1

    def test_nested_data(self, nested_group_hdf5_url):
        manifest_store = manifest_store_from_hdf_url(nested_group_hdf5_url)
        z = zarr.open_group(manifest_store, mode="r", zarr_format=3)

        with h5py.File(nested_group_hdf5_url.removeprefix("file://"), mode="r") as f:
            np.testing.assert_array_equal(f["group"]["data"], z["group"]["data"][...])
            np.testing.assert_array_equal(
                f["group"]["nested_group"]["data"][...],
                z["group"]["nested_group"]["data"][...],
            )

    def test_drop_variables(self, multiple_datasets_hdf5_url, local_registry):
        parser = HDFParser(drop_variables=["data2"])
        manifest_store = parser(url=multiple_datasets_hdf5_url, registry=local_registry)
        assert "data2" not in manifest_store._group.arrays.keys()

    def test_dataset_in_group(self, group_hdf5_url):
        manifest_store = manifest_store_from_hdf_url(group_hdf5_url, group="group")
        assert len(manifest_store._group.arrays) == 1

    def test_non_group_error(self, group_hdf5_url):
        with pytest.raises(ValueError):
            manifest_store_from_hdf_url(group_hdf5_url, group="group/data")


@requires_hdf5plugin
@requires_imagecodecs
class TestOpenVirtualDataset:
    def test_coord_names(self, root_coordinates_hdf5_file, local_registry):
        root_coordinates_hdf5_url = f"file://{root_coordinates_hdf5_file}"
        parser = HDFParser()
        with open_virtual_dataset(
            url=root_coordinates_hdf5_url,
            registry=local_registry,
            parser=parser,
        ) as vds:
            assert set(vds.coords) == {"lat", "lon"}

    def test_big_endian(self, big_endian_dtype_hdf5_file, local_registry):
        big_endian_dtype_hdf5_url = f"file://{big_endian_dtype_hdf5_file}"
        parser = HDFParser()
        with (
            parser(
                url=big_endian_dtype_hdf5_url, registry=local_registry
            ) as manifest_store,
            xr.open_dataset(big_endian_dtype_hdf5_file) as expected,
        ):
            observed = xr.open_dataset(
                manifest_store, engine="zarr", consolidated=False, zarr_format=3
            )
            assert isinstance(observed, xr.Dataset)
            xr.testing.assert_identical(observed.load(), expected.load())


@requires_hdf5plugin
@requires_imagecodecs
@pytest.mark.parametrize("group", [None, "/", "subgroup", "subgroup/", "/subgroup/"])
def test_subgroup_variable_names(
    netcdf4_file_with_data_in_multiple_groups, group, local_registry
):
    # regression test for GH issue #364
    netcdf4_url_with_data_in_multiple_groups = (
        f"file://{netcdf4_file_with_data_in_multiple_groups}"
    )
    parser = HDFParser(group=group)
    with open_virtual_dataset(
        url=netcdf4_url_with_data_in_multiple_groups,
        registry=local_registry,
        parser=parser,
    ) as vds:
        assert list(vds.dims) == ["dim_0"]


@pytest.mark.network
@pytest.mark.flaky
def test_netcdf_over_https():
    url = "https://www.earthbyte.org/webdav/gmt_mirror/gmt/data/cache/topo_32.nc"
    store = from_url(url)
    registry = ObjectStoreRegistry({url: store})
    parser = HDFParser()
    with (
        parser(url=url, registry=registry) as ms,
        xr.open_zarr(ms, zarr_format=3, consolidated=False).load() as ds,
    ):
        np.testing.assert_allclose(ds["z"].min().to_numpy(), -6)
        np.testing.assert_allclose(ds["z"].max().to_numpy(), 817)


def test_fillvalue_runtime_error():
    from virtualizarr.parsers.hdf.hdf import _get_fill_value

    dtype = np.dtype("float32")

    class _RuntimeErrorDataset:
        @property
        def fillvalue(self):
            raise RuntimeError("Unable to get fill value")

    dataset = _RuntimeErrorDataset()
    dataset.dtype = dtype  # type: ignore[attr-defined]

    result = _get_fill_value(dataset)
    assert result == np.ma.default_fill_value(dtype)
