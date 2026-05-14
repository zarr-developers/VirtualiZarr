"""Parser for converting an icechunk repository into a VirtualiZarr ManifestStore.

This package splits the parser into two pieces:

- :mod:`virtualizarr.parsers.icechunk.parser` contains the user-facing
  :class:`IcechunkParser` class and the async helpers that walk an icechunk
  store + assemble manifest arrays.
- :mod:`virtualizarr.parsers.icechunk._obstore_storage` contains the
  obstore-object → :class:`icechunk.Storage` translation used by
  :meth:`IcechunkParser.__call__` to open the repo from a URL.

Most users only need :class:`IcechunkParser`.
"""

from virtualizarr.parsers.icechunk.parser import IcechunkParser

__all__ = ["IcechunkParser"]
