from typing import TypeAlias

from zarr.core.group import GroupMetadata

from virtualizarr.manifests import ManifestArray

ManifestArrayVariableMapping: TypeAlias = dict[str, ManifestArray]


class ManifestGroup:
    """
    Virtualized representation of multiple ManifestArrays as a Zarr Group.
    """

    # TODO: Consider refactoring according to https://github.com/zarr-developers/VirtualiZarr/pull/490#discussion_r2007805272
    _manifest_arrays: ManifestArrayVariableMapping
    _metadata: GroupMetadata

    def __init__(
        self,
        manifest_arrays: ManifestArrayVariableMapping,
        attributes: dict,
    ) -> None:
        """
        Create a ManifestGroup from the dictionary of ManifestArrays and the group / dataset level metadata

        Parameters
        ----------
        attributes : attributes to include in Group metadata
        manifest_dict : ManifestArrayVariableMapping
        """

        self._metadata = GroupMetadata(attributes=attributes)
        self._manifest_arrays = manifest_arrays

    def __str__(self) -> str:
        return (
            f"ManifestArrayVariableMapping({self._manifest_arrays}, {self._metadata})"
        )
