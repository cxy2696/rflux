"""Small conversion, validation and JSON helpers shared by public modules."""

from __future__ import annotations

import math
import operator
from typing import Any, Sequence

import numpy as np


def asset_names(names: Sequence[str] | str | None, count: int) -> list[str]:
    if names is None:
        result = [f"asset_{index}" for index in range(count)]
    elif isinstance(names, str):
        result = [names]
    else:
        result = [str(name) for name in names]
    if len(result) != count:
        raise ValueError(f"expected {count} asset names, got {len(result)}")
    if any(not name for name in result):
        raise ValueError("asset names must not be empty")
    if len(set(result)) != len(result):
        raise ValueError("asset names must be unique")
    return result


def as_size(value: Any, name: str, *, minimum: int = 1) -> int:
    """Validate a non-negative/positive C size_t argument without wrapping."""
    try:
        integer = operator.index(value)
    except TypeError as exc:
        raise TypeError(f"{name} must be an integer") from exc
    if integer < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    if integer > np.iinfo(np.uintp).max:
        raise ValueError(f"{name} is too large for size_t")
    return int(integer)


def as_u64(value: Any, name: str) -> int:
    """Validate an unsigned 64-bit seed/index argument."""
    try:
        integer = operator.index(value)
    except TypeError as exc:
        raise TypeError(f"{name} must be an integer") from exc
    if integer < 0 or integer > np.iinfo(np.uint64).max:
        raise ValueError(f"{name} must be in the uint64 range")
    return int(integer)


def as_f64_1d(values: Any, name: str = "values") -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got shape {array.shape}")
    return np.ascontiguousarray(array)


def as_f64_2d(values: Any, name: str = "values") -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError(f"{name} must be two-dimensional, got shape {array.shape}")
    return np.ascontiguousarray(array)


def _as_integer_1d(values: Any, name: str) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got shape {array.shape}")
    if not np.issubdtype(array.dtype, np.integer):
        raise TypeError(f"{name} must contain integers")
    return array


def as_i32_1d(values: Any, name: str = "values") -> np.ndarray:
    array = _as_integer_1d(values, name)
    info = np.iinfo(np.int32)
    if np.any(array < info.min) or np.any(array > info.max):
        raise ValueError(f"{name} must fit in the int32 range")
    return np.ascontiguousarray(array, dtype=np.int32)


def as_u32_1d(values: Any, name: str = "values") -> np.ndarray:
    array = _as_integer_1d(values, name)
    if np.any(array < 0) or np.any(array > np.iinfo(np.uint32).max):
        raise ValueError(f"{name} must be in the uint32 range")
    return np.ascontiguousarray(array, dtype=np.uint32)


def as_f64_panel(
    values: Any,
    name: str = "panel",
    *,
    min_rows: int = 1,
    min_columns: int = 1,
) -> np.ndarray:
    """Normalize a 1D series or 2D panel to a contiguous observation panel."""
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 1:
        array = array.reshape(-1, 1)
    if array.ndim != 2:
        raise ValueError(f"{name} must be 1D or 2D, got shape {array.shape}")
    if array.shape[0] < min_rows or array.shape[1] < min_columns:
        raise ValueError(
            f"{name} must have at least {min_rows} rows and "
            f"{min_columns} columns"
        )
    return np.ascontiguousarray(array)


def as_finite_float(value: Any, name: str) -> float:
    number = float(value)
    if not np.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def as_positive_float(value: Any, name: str) -> float:
    number = as_finite_float(value, name)
    if number <= 0.0:
        raise ValueError(f"{name} must be positive")
    return number


def as_open_unit_float(value: Any, name: str) -> float:
    number = as_finite_float(value, name)
    if not 0.0 < number < 1.0:
        raise ValueError(f"{name} must be in (0, 1)")
    return number


def as_square(values: Any, name: str = "matrix") -> np.ndarray:
    array = as_f64_2d(values, name)
    if array.shape[0] != array.shape[1]:
        raise ValueError(f"{name} must be square, got shape {array.shape}")
    if array.shape[0] == 0:
        raise ValueError(f"{name} must not be empty")
    return array


def package_metadata() -> dict[str, Any]:
    """Version block shared by the JSON research reports."""
    from ._native_backend import abi_version, native_version
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
