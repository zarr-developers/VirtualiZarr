from __future__ import annotations

import textwrap
from time import datetime
from typing import TYPE_CHECKING, Iterator, Mapping, Optional

import xarray as xr
from zarr.core.group import GroupMetadata

from virtualizarr.manifests import ManifestArray

if TYPE_CHECKING:
    from icechunk import IcechunkStore  # type: ignore[import-not-found]


class ManifestGroup(
    Mapping[str, "ManifestArray | ManifestGroup"],
):
    """
    Immutable representation of a single virtual zarr group.
    """

    _members: Mapping[str, "ManifestArray | ManifestGroup"]
    _metadata: GroupMetadata

    def __init__(
        self,
        arrays: Mapping[str, ManifestArray] | None = None,
        groups: Mapping[str, "ManifestGroup"] | None = None,
        attributes: dict | None = None,
    ) -> None:
        """
        Create a ManifestGroup containing ManifestArrays and/or sub-groups, as well as any group-level metadata.

        Parameters
        ----------
        arrays : Mapping[str, ManifestArray], optional
            ManifestArray objects to represent virtual zarr arrays.
        groups : Mapping[str, ManifestGroup], optional
            ManifestGroup objects to represent virtual zarr subgroups.
        attributes : dict, optional
            Zarr attributes to add as zarr group metadata.
        """
        self._metadata = GroupMetadata(attributes=attributes)

        _arrays: Mapping[str, ManifestArray] = {} if arrays is None else arrays

        if groups:
            # TODO add support for nested groups
            raise NotImplementedError
        else:
            _groups: Mapping[str, ManifestGroup] = {} if groups is None else groups

        for name, arr in _arrays.items():
            if not isinstance(arr, ManifestArray):
                raise TypeError(
                    f"ManifestGroup can only wrap ManifestArray objects, but array {name} passed is of type {type(arr)}"
                )

        # TODO type check groups passed

        # TODO check that all arrays have the same shapes or dimensions?
        # Technically that's allowed by the zarr model, so we should theoretically only check that upon converting to xarray

        colliding_names = set(_arrays.keys()).intersection(set(_groups.keys()))
        if colliding_names:
            raise ValueError(
                f"Some names collide as they are present in both the array and group keys: {colliding_names}"
            )

        self._members = {**_arrays, **_groups}

    @property
    def metadata(self) -> GroupMetadata:
        """Zarr group metadata."""
        return self._metadata

    @property
    def arrays(self) -> dict[str, ManifestArray]:
        """ManifestArrays contained in this group."""
        return {k: v for k, v in self._members.items() if isinstance(v, ManifestArray)}

    @property
    def groups(self) -> dict[str, "ManifestGroup"]:
        """Subgroups contained in this group."""
        return {k: v for k, v in self._members.items() if isinstance(v, ManifestGroup)}

    def __getitem__(self, path: str) -> "ManifestArray | ManifestGroup":
        """Obtain a group member."""
        if "/" in path:
            raise ValueError(
                f"ManifestGroup.__getitem__ can only be used to get immediate subgroups and subarrays, but received multi-part path {path}"
            )

        return self._members[path]

    def __iter__(self) -> Iterator[str]:
        return iter(self._members.keys())

    def __len__(self) -> int:
        return len(self._members)

    def __repr__(self) -> str:
        return textwrap.dedent(
            f"""
            ManifestGroup(
                arrays={self.arrays},
                groups={self.groups},
                metadata={self.metadata},
            )
            """
        )

    def to_virtual_dataset(self) -> xr.Dataset:
        """
        Create a "virtual" xarray.Dataset containing the contents of one zarr group.

        All variables in the returned Dataset will be "virtual", i.e. they will wrap ManifestArray objects.
        """

        from virtualizarr.xarray import construct_fully_virtual_dataset

        # The xarray data model stores coordinate names outside of the arbitrary extra metadata it can store on a Dataset,
        # so to avoid that information being duplicated we strip it from the zarr group attributes before storing it.
        metadata_dict = self.metadata.to_dict()
        attributes = metadata_dict["attributes"]
        coord_names = attributes.pop("coordinates", [])

        virtual_vars = {
            name: marr.to_virtual_variable() for name, marr in self.arrays.items()
        }

        return construct_fully_virtual_dataset(
            virtual_vars=virtual_vars,
            coord_names=coord_names,
            attrs=attributes,
        )

    @classmethod
    def from_virtual_dataset(
        cls,
        vds: xr.Dataset,
    ) -> "ManifestGroup":
        """
        Create a new ManifestGroup from a virtual dataset object.

        The virtual dataset should contain only virtual variables, i.e. those backed by ManifestArrays.

        Parameters
        ----------
        vds: xr.Dataset
            Virtual dataset, containing only virtual variables.
        """

        for name, var in vds.variables.items():
            if not isinstance(var.data, ManifestArray):
                raise TypeError(
                    f"Cannot convert a dataset containing a loadable variable directly to a ManifestGroup, but found variable {name} has type {type(var.data)}"
                )

        manifestarrays = {name: var.data for name, var in vds.variables.items()}

        attributes = vds.attrs
        # TODO test this is correct
        attributes["dimension_names"] = " ".join(list(vds.dims))
        # TODO test this constructor round-trips coordinates
        attributes["coordinates"] = " ".join(list(vds.coords))

        return cls(arrays=manifestarrays, attributes=attributes)

    def to_icechunk(
        store: "IcechunkStore",
        group: Optional[str] = None,
        append_dim: Optional[str] = None,
        last_updated_at: Optional[datetime] = None,
    ) -> None:
        """
        Write the contents of this ManifestGroup into an Icechunk Store as virtual chunks.

        Parameters
        ----------
        store: IcechunkStore
            Store to write dataset into.
        group: str, optional
            Path of the group to write the dataset into (default: the root group).
        append_dim: str, optional
            Dimension along which to append the virtual dataset.
        last_updated_at: datetime, optional
            Datetime to use as a checksum for any virtual chunks written to the store
            with this operation.  When not provided, no check is performed.

        Raises
        ------
        ValueError
            If the store is read-only.
        """
        from virtualizarr.writers.icechunk import write_manifestgroup_to_icechunk

        write_manifestgroup_to_icechunk(
            store=store,
            group=group,
            append_dim=append_dim,
            last_updated_at=last_updated_at,
        )
