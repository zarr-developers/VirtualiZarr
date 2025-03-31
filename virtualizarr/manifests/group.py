from __future__ import annotations

import textwrap
from typing import Iterator, Mapping

from zarr.core.group import GroupMetadata
import xarray as xr

from virtualizarr.manifests import ManifestArray


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

    def as_virtual_dataset(self) -> xr.Dataset:
        """
        Create a "virtual" xarray dataset containing the contents of one zarr group.

        All variables in the returned Dataset will be "virtual", i.e. they will wrap ManifestArray objects.
        """

        virtual_vars = {name: marr.as_virtual_var() for name, marr in self.arrays}
        raise NotImplementedError()
