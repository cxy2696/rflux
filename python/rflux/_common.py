"""Small ctypes/numpy conversion helpers shared by the public modules."""

from __future__ import annotations

import operator
from typing import Any

import numpy as np

from ._ffi import c_double_p, c_int32_p, c_uint32_p


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


def as_i32_1d(values: Any, name: str = "values") -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got shape {array.shape}")
    if not np.issubdtype(array.dtype, np.integer):
        raise TypeError(f"{name} must contain integers")
    info = np.iinfo(np.int32)
    if np.any(array < info.min) or np.any(array > info.max):
        raise ValueError(f"{name} must fit in the int32 range")
    return np.ascontiguousarray(array, dtype=np.int32)


def as_u32_1d(values: Any, name: str = "values") -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got shape {array.shape}")
    if not np.issubdtype(array.dtype, np.integer):
        raise TypeError(f"{name} must contain integers")
    if np.any(array < 0) or np.any(array > np.iinfo(np.uint32).max):
        raise ValueError(f"{name} must be in the uint32 range")
    return np.ascontiguousarray(array, dtype=np.uint32)


def as_square(values: Any, name: str = "matrix") -> np.ndarray:
    array = as_f64_2d(values, name)
    if array.shape[0] != array.shape[1]:
        raise ValueError(f"{name} must be square, got shape {array.shape}")
    if array.shape[0] == 0:
        raise ValueError(f"{name} must not be empty")
    return array


def empty_f64(n: int) -> np.ndarray:
    return np.empty(int(n), dtype=np.float64)


def empty_i32(n: int) -> np.ndarray:
    return np.empty(int(n), dtype=np.int32)


def double_ptr(array: np.ndarray):
    return array.ctypes.data_as(c_double_p)


def int32_ptr(array: np.ndarray):
    return array.ctypes.data_as(c_int32_p)


def uint32_ptr(array: np.ndarray):
    return array.ctypes.data_as(c_uint32_p)


def double_matrix(flat: np.ndarray, n: int) -> np.ndarray:
    return np.asarray(flat, dtype=np.float64).reshape(int(n), int(n))
