from __future__ import annotations

from typing import Iterator, Mapping

from zarr.core.group import GroupMetadata

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
        # TODO rename attributes to metadata
        attributes: dict = None,
    ) -> None:
        """
        Create a ManifestGroup from the dictionary of ManifestArrays and the group / dataset level metadata

        Parameters
        ----------
        arrays : Mapping[str, ManifestArray]
        groups : Mapping[str, ManifestGroup]
        attributes : dict
            Zarr attributes to use as zarr group metadata.
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
        # Technically that's allowed by the zarr model, so we should theoretically only check that upon converting to

        # TODO check for name collisions

        self._members = {**_arrays, **_groups}

    @property
    def metadata(self) -> GroupMetadata:
        """Zarr group metadata."""
        return self._metadata

    @property
    def arrays(self) -> dict[str, ManifestArray]:
        return {k: v for k, v in self._members.items() if isinstance(v, ManifestArray)}

    @property
    def groups(self) -> dict[str, "ManifestGroup"]:
        return {k: v for k, v in self._members.items() if isinstance(v, ManifestGroup)}

    def __getitem__(self, path: str) -> "ManifestArray | ManifestGroup":
        """Obtain a group member."""
        if path.contains("/"):
            raise ValueError(
                f"ManifestGroup.__getitem__ can only be used to get immediate subgroups and subarrays, but received multi-part path {path}"
            )

        return self._members[path]

    def __iter__(self) -> Iterator[str]:
        return iter(self._members.keys())

    def __len__(self) -> int:
        return len(self._members)

    def __str__(self) -> str:
        return f"ManifestGroup(arrays={self.arrays}, groups={self.groups}, metadata={self.metadata})"
