import json
import numbers
from typing import Any


class NumberEncoder(json.JSONEncoder):
    def default(self, o):
        # See json.JSONEncoder.default docstring for explanation
        # This is necessary to encode numpy dtype
        if isinstance(o, numbers.Integral):
            return int(o)
        if isinstance(o, numbers.Real):
            return float(o)
        return json.JSONEncoder.default(self, o)


def json_dumps(o: Any) -> bytes:
    """Write JSON in a consistent, human-readable way."""
    return json.dumps(
        o,
        indent=4,
        sort_keys=True,
        ensure_ascii=True,
        separators=(",", ": "),
        cls=NumberEncoder,
    ).encode("ascii")
