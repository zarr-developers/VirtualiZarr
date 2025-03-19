from typing import TypeAlias

from zarr.core.group import GroupMetadata

from virtualizarr.manifests import ManifestArray

ManifestDict: TypeAlias = dict[str, ManifestArray]


class ManifestGroup:
    """
    Virtualized representation of multiple ManifestArrays as a Zarr Group.
    """

    _manifest_dict: ManifestDict
    _metadata: GroupMetadata

    def __init__(
        self,
        manifest_dict: ManifestDict,
        attributes: dict,
    ) -> None:
        """
        Create a ManifestGroup from the dictionary of ManifestArrays and the group / dataset level metadata

        Parameters
        ----------
        attributes : attributes to include in Group metadata
        manifest_dict : ManifestDict
        """

        self._metadata = GroupMetadata(attributes=attributes)
        self._manifest_dict = manifest_dict

    def __str__(self) -> str:
        return f"ManifestGroup({self._manifest_dict}, {self._metadata})"
