from typing import Mapping

from zarr.core.group import GroupMetadata

from virtualizarr.manifests import ManifestArray


class ManifestGroup:
    """
    Virtualized representation of multiple ManifestArrays as a Zarr Group.
    """

    # TODO: Consider refactoring according to https://github.com/zarr-developers/VirtualiZarr/pull/490#discussion_r2007805272
    _arrays: Mapping[str, ManifestArray]
    _metadata: GroupMetadata

    def __init__(
        self,
        arrays: Mapping[str, ManifestArray],
        attributes: dict,
    ) -> None:
        """
        Create a ManifestGroup from the dictionary of ManifestArrays and the group / dataset level metadata

        Parameters
        ----------
        arrays : Mapping[str, ManifestArray]
        attributes : dict
            Zarr attributes to use as zarr group metadata.
        """

        self._metadata = GroupMetadata(attributes=attributes)

        for name, arr in arrays.items():
            if not isinstance(arr, ManifestArray):
                raise TypeError(
                    f"ManifestGroup can only wrap ManifestArray objects, but array {name} passed is of type {type(arr)}"
                )

        # TODO check that all arrays have the same shapes or dimensions?
        # Technically that's allowed by the zarr model, so we should theoretically only check that upon converting to

        self._arrays = arrays

    @property
    def metadata(self) -> GroupMetadata:
        """Zarr group metadata."""
        return self._metadata

    def __str__(self) -> str:
        return f"ManifestGroup(arrays={self._arrays}, metadata={self.metadata})"
