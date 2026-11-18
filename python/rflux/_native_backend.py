"""Optional access to the pybind11 native bridge.

The extension is built by CMake/scikit-build and shipped inside the wheel.  A
source checkout that has not built it still works through the stable C ABI; set
``RFLUX_DISABLE_NATIVE=1`` to exercise that path explicitly.
"""

from __future__ import annotations

import os
from typing import Any, Callable, NoReturn, TypeVar

from ._ffi import RFluxError
from ._version import __version__

try:
    from . import _native as native_module
    _NATIVE_ERROR: str | None = None
except ImportError as exc:  # pragma: no cover - exercised by the ctypes fallback
    native_module = None
    _NATIVE_ERROR = str(exc)

if native_module is not None:
    _extension_version = getattr(native_module, "__version__", None)
    if _extension_version != __version__:
        _NATIVE_ERROR = (
            f"native extension version {_extension_version!r} does not match "
            f"Python package {__version__}"
        )
        native_module = None

if os.environ.get("RFLUX_DISABLE_NATIVE") == "1":
    native_module = None
    _NATIVE_ERROR = "disabled by RFLUX_DISABLE_NATIVE=1"

NATIVE_AVAILABLE = native_module is not None

T = TypeVar("T")


def raise_as_rflux(function: str, exc: BaseException) -> NoReturn:
    """Translate a native bridge error into the public exception type."""
    status = int(getattr(exc, "status", 6))
    raise RFluxError(function, status, str(exc)) from None


def native_call(module: Any, function: str, call: Callable[[], T]) -> T:
    """Run one native call and translate ``NativeError`` to ``RFluxError``."""
    if module is None:  # pragma: no cover - callers check first
        raise RuntimeError("native extension is not available")
    try:
        return call()
    except module.NativeError as exc:
        raise_as_rflux(function, exc)


def native_or(
    module: Any,
    function: str,
    native: Callable[[], T],
    fallback: Callable[[], T],
) -> T:
    """Use the native bridge when present, otherwise the stable C ABI."""
    if module is None:
        return fallback()
    return native_call(module, function, native)


def native_info() -> dict[str, object]:
    if native_module is None:
        if _NATIVE_ERROR:
            return {"available": False, "error": _NATIVE_ERROR}
        return {"available": False}
    return {
        "available": True,
        "version": native_module.__version__,
        "stable_abi": bool(native_module.STABLE_ABI),
        "gil_released": bool(native_module.GIL_RELEASED),
        "module": str(native_module.__file__),
    }


__all__ = [
    "NATIVE_AVAILABLE",
    "native_call",
    "native_info",
    "native_module",
    "native_or",
    "raise_as_rflux",
]
