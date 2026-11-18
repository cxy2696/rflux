"""Low-level access to the required pybind11 extension.

Most users should call :mod:`rflux.qflux`, :mod:`rflux.percolab` and the other
public modules; they use the native bridge automatically and translate errors
to :class:`rflux.RFluxError`.  This module exposes the raw extension surface,
including zero-copy buffer entry points, for applications that need it.  Raw
functions raise ``rflux._native.NativeError`` with a ``status`` attribute.
"""

from __future__ import annotations

from ._native_backend import native_info as info
from ._native_backend import native_module as _native

available = True
__all__ = ["available", "info"]

# Mirror the extension surface without keeping a hand-written list in sync with
# bindings.cpp.
exported = [name for name in dir(_native) if not name.startswith("_")]
globals().update({name: getattr(_native, name) for name in exported})
__all__ += exported
