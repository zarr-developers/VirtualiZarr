import asyncio
from itertools import starmap
from typing import (
    Any,
    Awaitable,
    Callable,
    Iterable,
    TypeVar,
)

# Vendored directly from Zarr-python V3's private API
# https://github.com/zarr-developers/zarr-python/blob/458299857141a5470ba3956d8a1607f52ac33857/src/zarr/core/common.py#L53
T = TypeVar("T", bound=tuple[Any, ...])
V = TypeVar("V")


async def _concurrent_map(
    items: Iterable[T],
    func: Callable[..., Awaitable[V]],
    limit: int | None = None,
) -> list[V]:
    if limit is None:
        return await asyncio.gather(*list(starmap(func, items)))

    else:
        sem = asyncio.Semaphore(limit)

        async def run(item: tuple[Any]) -> V:
            async with sem:
                return await func(*item)

        return await asyncio.gather(
            *[asyncio.ensure_future(run(item)) for item in items]
        )
