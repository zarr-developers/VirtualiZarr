from dataclasses import dataclass
from typing import cast, TYPE_CHECKING, Iterable

import xarray as xr

from virtualizarr.manifests import ChunkManifest
from virtualizarr.manifests import ManifestArray

if TYPE_CHECKING:
    import pyarrow as pa


@dataclass(frozen=True)
class ArrowChunkManifest:
    """Arrow-backed chunk manifest for efficient validation and writing to icechunk."""

    locations: "pa.StringArray"
    offsets: "pa.UInt64Array"
    lengths: "pa.UInt64Array"
    shape_chunk_grid: tuple[int, ...]

    @classmethod
    def from_manifest(cls, manifest: ChunkManifest) -> "ArrowChunkManifest":
        """Convert a ChunkManifest to Arrow arrays.

        Empty paths (representing missing chunks) are converted to nulls.
        """
        import pyarrow as pa

        n_chunks = len(manifest)
        paths_flat = manifest._paths.ravel()

        # Create null mask from empty strings (True = null)
        null_mask = paths_flat == ""

        # Create arrays with mask applied during construction (no extra copies)
        return cls(
            locations=pa.array(
                paths_flat.tolist(), type=pa.string(), size=n_chunks, mask=null_mask
            ),
            offsets=pa.array(
                manifest._offsets.ravel(), type=pa.uint64(), size=n_chunks, mask=null_mask
            ),
            lengths=pa.array(
                manifest._lengths.ravel(), type=pa.uint64(), size=n_chunks, mask=null_mask
            ),
            shape_chunk_grid=manifest.shape_chunk_grid,
        )
    

def extract_arrow_manifests(vds: xr.Dataset) -> dict[str, ArrowChunkManifest]:
    """Extract all manifests from a dataset and convert to Arrow format."""
    return {
        name: ArrowChunkManifest.from_manifest(cast(ManifestArray, var.data).manifest)
        for name, var in vds.variables.items()
        if isinstance(var.data, ManifestArray)
    }


def validate_location_prefixes(
    arrow_manifests: Iterable[ArrowChunkManifest],
    valid_prefixes: list[str],
) -> None:
    """
    Validate that all chunk locations start with one of the valid prefixes.

    Uses PyArrow compute for efficient validation of large manifests.

    Parameters
    ----------
    arrow_manifests
        Manifests to validate.
    valid_prefixes
        List of allowed location prefixes. If empty, validation is skipped.
    """
    import pyarrow.compute as pc

    for manifest in arrow_manifests:
        locations = manifest.locations

        # Build a mask of locations that match at least one prefix
        # Nulls (missing chunks) become null in the result and are skipped
        matches = pc.starts_with(locations, valid_prefixes[0])
        for prefix in valid_prefixes[1:]:
            matches = pc.or_(matches, pc.starts_with(locations, prefix))

        # Check if all non-null locations match at least one prefix
        all_match = pc.all(matches, skip_nulls=True)

        # If any don't match then do more work to find the first offender
        if all_match.is_valid and not all_match.as_py():
            # Find first invalid location to report in error
            invalid = pc.invert(pc.fill_null(matches, True))
            invalid_indices = pc.indices_nonzero(invalid)
            first_invalid_idx = invalid_indices[0].as_py()
            invalid_location = locations[first_invalid_idx].as_py()
            raise ValueError(
                f"Location {invalid_location!r} does not start with any supported prefix: {valid_prefixes}"
            )
