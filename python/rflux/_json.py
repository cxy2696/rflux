"""Convert report values to plain JSON-safe Python objects."""

from __future__ import annotations

import math
from typing import Any

import numpy as np


def package_metadata() -> dict[str, Any]:
    """Version block shared by the JSON research reports."""
    from ._ffi import abi_version, native_version
    from ._version import __version__

    return {
        "version": __version__,
        "native_version": native_version(),
        "abi_version": abi_version(),
    }


def jsonable(value: Any) -> Any:
    """Replace non-finite floats with None and unwrap NumPy scalars/arrays."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, np.generic):
        return jsonable(value.item())
    if isinstance(value, np.ndarray):
        return [jsonable(item) for item in value.tolist()]
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return str(value)
