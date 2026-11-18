"""Optional low-level access to the pybind11 bridge.

Most users should call :mod:`rflux.qflux` and :mod:`rflux.regimelab`; they use
the native bridge automatically.  This module exposes the raw zero-copy entry
points for applications that need to reuse an output buffer or inspect native
availability.  Every attribute is ``None``/absent when the extension was not
built into the installation.
"""

from __future__ import annotations

from ._native_backend import (
    NATIVE_AVAILABLE as available,
    native_info as info,
    native_module as _native,
)

__all__ = ["available", "info"]

if _native is not None:
    # Mirror the public extension surface without keeping a second hand-written
    # export list in sync with bindings.cpp.
    exported = [name for name in dir(_native) if not name.startswith("_")]
    globals().update({name: getattr(_native, name) for name in exported})
    __all__ += exported
