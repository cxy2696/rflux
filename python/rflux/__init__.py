"""rflux — a portable quantitative-algorithm package.

Layers
------
- C ABI: ``include/rflux/rflux.h`` (stable, opaque handles, no exceptions).
- C++17 core: qflux, regimelab, percolab, factorlab, portfoliolab, execution and mllab.
- Python: NumPy wrappers, data adapters and research tools.

The pybind11 extension is optional and is preferred when present; the stable C
ABI remains a lazy, portable fallback.  Set ``RFLUX_LIBRARY`` to an explicit
shared-library path, or ``RFLUX_DISABLE_NATIVE=1`` to force ctypes.
"""

from __future__ import annotations

from ._version import __version__

from . import (  # noqa: E402
    advice,
    backtest,
    config,
    data,
    execution,
    factorlab,
    metrics,
    mllab,
    native,
    percolab,
    portfoliolab,
    qflux,
    regimelab,
    risk,
    shared,
    symbols,
)
from .advice import trade_advice  # noqa: E402
from .analysis import analyze_prices  # noqa: E402
from .apps import portfolio_risk, regime_attribution  # noqa: E402
from ._ffi import (  # noqa: E402
    ABI_VERSION,
    ABIError,
    LibraryNotFoundError,
    RFluxError,
    abi_version,
    native_version,
)
from ._native_backend import native_info  # noqa: E402
from .config import (  # noqa: E402
    ConfigError,
    FitOptions,
    HMMConfig,
    ModelConfig,
    load as load_config,
    loads as load_configs,
    save as save_config,
)
from .shared import SharedArray  # noqa: E402
from .metrics import performance_summary  # noqa: E402
from .percolab import correlation_network  # noqa: E402
from .qflux import returns, sma  # noqa: E402
from .regimelab import fit_hmm  # noqa: E402

__all__ = [
    "ABIError",
    "ABI_VERSION",
    "ConfigError",
    "FitOptions",
    "HMMConfig",
    "LibraryNotFoundError",
    "ModelConfig",
    "RFluxError",
    "SharedArray",
    "__version__",
    "abi_version",
    "advice",
    "analyze_prices",
    "backtest",
    "config",
    "correlation_network",
    "data",
    "execution",
    "factorlab",
    "fit_hmm",
    "load_config",
    "load_configs",
    "metrics",
    "mllab",
    "native",
    "native_info",
    "native_version",
    "performance_summary",
    "percolab",
    "portfolio_risk",
    "portfoliolab",
    "qflux",
    "regime_attribution",
    "regimelab",
    "returns",
    "risk",
    "save_config",
    "shared",
    "sma",
    "symbols",
    "trade_advice",
]
