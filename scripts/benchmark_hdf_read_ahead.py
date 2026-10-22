"""Time HDFParser on remote HDF5 files with and without read_ahead, and check the results match.

    python scripts/benchmark_hdf_read_ahead.py [URL ...] [--repeat N]

With no URLs, it uses NWB files from the DANDI Archive whose datasets have
many chunks, read over HTTPS from its public S3 bucket. Each file is parsed
with read_ahead=True and read_ahead=False, alternating which goes first, and
the manifests and metadata from the two are compared. Prints a Markdown table.
"""

from __future__ import annotations

import argparse
import json
import time
import warnings
from urllib.parse import urlparse

import numpy as np
from obspec_utils.registry import ObjectStoreRegistry
from obstore.store import HTTPStore

from virtualizarr.parsers import HDFParser

DANDI = "https://dandiarchive.s3.amazonaws.com/blobs"
# DANDI blob ID and the dandiset it belongs to
DEFAULT_FILES = {
    "c48080db-d89b-4c08-be5f-455f1a173919": "000296",
    "1bda1058-91c0-4900-9080-fc449ee27cf5": "001205",
    "195b50fb-7272-4544-ae0a-14e3e712c800": "001469",
    "61063c84-e7c7-4714-b2c7-4d4a07b88479": "000296",
    "49d7be07-2633-4765-b1ce-653482eefc11": "001276",
    "0f05d7e8-5de9-4113-85de-9b41c2d81267": "000253",
    "33212d66-fb93-4aca-8072-3801924c4e44": "000168",
}


def dandi_url(blob_id: str) -> str:
    return f"{DANDI}/{blob_id[:3]}/{blob_id[3:6]}/{blob_id}"


def parse(url: str, read_ahead: bool):
    """Parse url with a new store, so nothing is cached between runs; return the seconds taken and the group."""
    parsed = urlparse(url)
    base = f"{parsed.scheme}://{parsed.netloc}"
    registry = ObjectStoreRegistry({base: HTTPStore.from_url(base)})
    start = time.perf_counter()
    store = HDFParser(read_ahead=read_ahead)(url=url, registry=registry)
    return time.perf_counter() - start, store._group


def arrays(group, prefix: str = ""):
    for name, array in group.arrays.items():
        yield prefix + name, array
    for name, subgroup in group.groups.items():
        yield from arrays(subgroup, f"{prefix}{name}/")


def largest_chunk_count(group) -> int:
    return max((array.manifest._offsets.size for _, array in arrays(group)), default=0)


def differences(first, second) -> list[str]:
    """The arrays whose manifests or metadata differ between two parses."""
    a, b = dict(arrays(first)), dict(arrays(second))
    if a.keys() != b.keys():
        return sorted(a.keys() ^ b.keys())
    found = []
    for name in a:
        x, y = a[name], b[name]
        same = all(
            np.array_equal(getattr(x.manifest, field), getattr(y.manifest, field))
            for field in ("_paths", "_offsets", "_lengths")
        )
        # Compared as JSON, so that NaN attributes count as equal
        same = same and json.dumps(x.metadata.to_dict(), default=str) == json.dumps(
            y.metadata.to_dict(), default=str
        )
        if not same:
            found.append(name)
    return found


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "urls", nargs="*", help="HDF5 files over HTTP(S); default: DANDI files"
    )
    parser.add_argument(
        "--repeat", type=int, default=1, help="times to parse each file each way"
    )
    args = parser.parse_args()
    warnings.filterwarnings("ignore")

    files = (
        {url: url for url in args.urls}
        if args.urls
        else {
            dandi_url(
                blob
            ): f"[{dandiset}](https://dandiarchive.org/dandiset/{dandiset}) `{blob[:8]}`"
            for blob, dandiset in DEFAULT_FILES.items()
        }
    )
    print(
        "| file | chunks in largest array | `read_ahead=False` | `read_ahead=True` | speed-up | same result |"
    )
    print("|---|---|---|---|---|---|")
    for url, label in files.items():
        times: dict[bool, list[float]] = {True: [], False: []}
        groups = {}
        for i in range(args.repeat):
            for read_ahead in (True, False) if i % 2 == 0 else (False, True):
                seconds, groups[read_ahead] = parse(url, read_ahead)
                times[read_ahead].append(seconds)
        off, on = min(times[False]), min(times[True])
        differ = differences(groups[True], groups[False])
        print(
            f"| {label} | {largest_chunk_count(groups[True]):,} | {off:.1f} s | {on:.1f} s "
            f"| {off / on:.1f}x | {'yes' if not differ else 'NO: ' + ', '.join(differ[:3])} |",
            flush=True,
        )


if __name__ == "__main__":
    main()
