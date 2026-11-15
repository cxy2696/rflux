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

from ._version import __version__

from . import backtest, data, metrics, percolab, qflux, regimelab  # noqa: E402
from .analysis import analyze_prices  # noqa: E402
from ._ffi import (  # noqa: E402
    ABI_VERSION,
    ABIError,
    LibraryNotFoundError,
    RFluxError,
    abi_version,
    native_version,
)
from .metrics import performance_summary  # noqa: E402
from .percolab import correlation_network  # noqa: E402
from .qflux import returns, sma  # noqa: E402
from .regimelab import fit_hmm  # noqa: E402

__all__ = [
    "ABIError",
    "ABI_VERSION",
    "LibraryNotFoundError",
    "RFluxError",
    "__version__",
    "abi_version",
    "analyze_prices",
    "backtest",
    "correlation_network",
    "data",
    "fit_hmm",
    "metrics",
    "native_version",
    "performance_summary",
    "percolab",
    "qflux",
    "regimelab",
    "returns",
    "sma",
]
