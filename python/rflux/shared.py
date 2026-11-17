"""Zero-copy NumPy arrays in OS shared memory.

A :class:`SharedArray` owns a POSIX/Windows shared-memory segment and exposes a
regular ``numpy.ndarray`` view over it.  The array can be handed directly to the
native extension or to another process without pickling its contents; only the
small segment descriptor is transferred.

The creator is responsible for calling :meth:`SharedArray.unlink` once every
consumer has detached.  ``close()`` only releases the local mapping.
"""

from __future__ import annotations

import multiprocessing.shared_memory as _shm
from typing import Any

import numpy as np

__all__ = ["SharedArray"]


class SharedArray:
    """A NumPy array backed by a named shared-memory segment."""

    __slots__ = (
        "_shm",
        "_array",
        "_owner",
        "_unlink_on_exit",
        "_closed",
        "_unlinked",
    )

    def __init__(
        self,
        shm: _shm.SharedMemory,
        array: np.ndarray,
        *,
        owner: bool,
        unlink_on_exit: bool = False,
    ):
        self._shm = shm
        self._array = array
        self._owner = owner
        self._unlink_on_exit = bool(unlink_on_exit)
        self._closed = False
        self._unlinked = False

    @classmethod
    def create(
        cls,
        shape: int | tuple[int, ...],
        dtype: Any = np.float64,
        *,
        name: str | None = None,
        unlink_on_exit: bool = False,
    ) -> "SharedArray":
        """Allocate a contiguous array in a new shared-memory segment."""
        shape_tuple = _normalize_shape(shape)
        dtype_object = np.dtype(dtype)
        nbytes = _byte_count(shape_tuple, dtype_object)
        shm = _shm.SharedMemory(name=name, create=True, size=max(nbytes, 1))
        try:
            array = np.ndarray(shape_tuple, dtype=dtype_object, buffer=shm.buf)
        except Exception:
            shm.close()
            shm.unlink()
            raise
        return cls(shm, array, owner=True, unlink_on_exit=unlink_on_exit)

    @classmethod
    def attach(
        cls,
        name: str,
        shape: int | tuple[int, ...],
        dtype: Any = np.float64,
    ) -> "SharedArray":
        """Attach to an existing segment created by another process."""
        shape_tuple = _normalize_shape(shape)
        dtype_object = np.dtype(dtype)
        nbytes = _byte_count(shape_tuple, dtype_object)
        shm = _shm.SharedMemory(name=name)
        if shm.size < nbytes:
            shm.close()
            raise ValueError(
                f"shared segment {name!r} is {shm.size} bytes, "
                f"expected at least {nbytes}"
            )
        try:
            array = np.ndarray(shape_tuple, dtype=dtype_object, buffer=shm.buf)
        except Exception:
            shm.close()
            raise
        return cls(shm, array, owner=False)

    @property
    def name(self) -> str:
        return self._shm.name

    @property
    def array(self) -> np.ndarray:
        """The live NumPy view; writes are visible to every attached process."""
        if self._closed:
            raise RuntimeError("SharedArray is closed")
        return self._array

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(self._array.shape)

    @property
    def dtype(self) -> np.dtype[Any]:
        return self._array.dtype

    @property
    def nbytes(self) -> int:
        return int(self._array.nbytes)

    @property
    def owner(self) -> bool:
        return self._owner

    @property
    def closed(self) -> bool:
        return self._closed

    def view(self, dtype: Any = None) -> np.ndarray:
        """Return the shared view, optionally reinterpreted as another dtype."""
        array = self.array
        if dtype is None or np.dtype(dtype) == array.dtype:
            return array
        return array.view(np.dtype(dtype))

    def close(self) -> None:
        """Release this process's mapping; other attached processes are unaffected."""
        if self._closed:
            return
        self._closed = True
        self._shm.close()

    def unlink(self) -> None:
        """Request deletion of the underlying segment when all users detach."""
        if self._unlinked:
            return
        self._unlinked = True
        try:
            self._shm.unlink()
        finally:
            self.close()

    def __enter__(self) -> "SharedArray":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        if self._unlink_on_exit:
            self.unlink()
        else:
            self.close()

    def __reduce__(self):
        # Transfer the descriptor, not the data.  The receiving process must
        # attach while the creator still owns the segment.
        return (
            _attach_from_descriptor,
            (self.name, self.shape, self.dtype.str),
        )

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def __repr__(self) -> str:
        state = "closed" if self._closed else "open"
        role = "owner" if self._owner else "attached"
        return (
            f"SharedArray(name={self.name!r}, shape={self.shape}, "
            f"dtype={self.dtype}, {role}, {state})"
        )


def _normalize_shape(shape: int | tuple[int, ...]) -> tuple[int, ...]:
    if isinstance(shape, (int, np.integer)):
        result = (int(shape),)
    else:
        try:
            result = tuple(int(item) for item in shape)
        except TypeError as exc:
            raise TypeError("shape must be an int or a tuple of ints") from exc
    if any(item < 0 for item in result):
        raise ValueError("shape entries must be non-negative")
    return result


def _byte_count(shape: tuple[int, ...], dtype: np.dtype[Any]) -> int:
    count = 1
    for item in shape:
        count *= item
    return int(count * dtype.itemsize)


def _attach_from_descriptor(
    name: str, shape: tuple[int, ...], dtype: str
) -> SharedArray:
    return SharedArray.attach(name, shape, np.dtype(dtype))
