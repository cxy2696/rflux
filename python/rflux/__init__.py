"""rflux — a portable quantitative-algorithm package.

Layers
------
- C ABI: ``include/rflux/rflux.h`` (stable, opaque handles, no exceptions).
- C++17 core: ``rflux::qflux``, ``rflux::regimelab``, ``rflux::percolab``.
- Python: NumPy wrappers, data adapters and research tools.

The native library is loaded lazily.  Set ``RFLUX_LIBRARY`` to an explicit
shared-library path, or build it into ``build/`` / ``python/rflux/_lib/``.
"""

from __future__ import annotations

__version__ = "0.2.0"

from . import backtest, data, metrics, percolab, qflux, regimelab  # noqa: E402
from ._ffi import (  # noqa: E402
    ABI_VERSION,
    ABIError,
    LibraryNotFoundError,
    RFluxError,
    abi_version,
    native_version,
)

__all__ = [
    "ABIError",
    "ABI_VERSION",
    "LibraryNotFoundError",
    "RFluxError",
    "__version__",
    "abi_version",
    "backtest",
    "data",
    "metrics",
    "native_version",
    "percolab",
    "qflux",
    "regimelab",
]
