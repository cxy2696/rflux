"""Low-level ctypes bridge to the stable rflux C ABI.

The rest of the Python package uses this module; user code should normally use
:mod:`rflux.qflux`, :mod:`rflux.regimelab` and :mod:`rflux.percolab`.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import os
import sys
from pathlib import Path
from typing import Iterable

ABI_VERSION = 1
STATUS_OK = 0
STATUS_INVALID_ARGUMENT = 1
STATUS_OUT_OF_MEMORY = 2
STATUS_NONFINITE = 3
STATUS_MODEL_NOT_READY = 4
STATUS_NUMERICAL = 5
STATUS_INTERNAL = 6

STATUS_NAMES = {
    STATUS_OK: "RFLUX_OK",
    STATUS_INVALID_ARGUMENT: "RFLUX_ERR_INVALID_ARGUMENT",
    STATUS_OUT_OF_MEMORY: "RFLUX_ERR_OUT_OF_MEMORY",
    STATUS_NONFINITE: "RFLUX_ERR_NONFINITE",
    STATUS_MODEL_NOT_READY: "RFLUX_ERR_MODEL_NOT_READY",
    STATUS_NUMERICAL: "RFLUX_ERR_NUMERICAL",
    STATUS_INTERNAL: "RFLUX_ERR_INTERNAL",
}

c_double_p = ctypes.POINTER(ctypes.c_double)
c_int32_p = ctypes.POINTER(ctypes.c_int32)
c_uint32_p = ctypes.POINTER(ctypes.c_uint32)
c_size_p = ctypes.POINTER(ctypes.c_size_t)


class RFluxError(RuntimeError):
    """A native call returned a non-zero rflux_status."""

    def __init__(self, function: str, status: int, message: str):
        self.function = function
        self.status = int(status)
        prefix = f"{function}: "
        if message.startswith(prefix):
            message = message[len(prefix):]
        self.message = message
        name = STATUS_NAMES.get(self.status, f"RFLUX_STATUS_{self.status}")
        detail = f": {message}" if message else ""
        super().__init__(f"{function} failed with {name}{detail}")


class LibraryNotFoundError(ImportError):
    """The native rflux shared library could not be located."""


class ABIError(ImportError):
    """The shared library and Python binding do not use the same ABI."""


class HMMOptions(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_size_t),
        ("max_iter", ctypes.c_uint32),
        ("n_init", ctypes.c_uint32),
        ("tol", ctypes.c_double),
        ("variance_floor", ctypes.c_double),
        ("seed", ctypes.c_uint64),
        ("sort_states", ctypes.c_int32),
        ("reserved0", ctypes.c_int32),
    ]


_LIBRARY: ctypes.CDLL | None = None


def _library_names() -> list[str]:
    if sys.platform == "darwin":
        return ["librflux.dylib", "librflux.1.dylib"]
    if os.name == "nt":
        return ["rflux.dll", "librflux.dll"]
    return ["librflux.so", "librflux.so.1"]


def _candidate_paths() -> Iterable[Path]:
    env_path = os.environ.get("RFLUX_LIBRARY")
    if env_path:
        yield Path(env_path).expanduser()
        return

    package_dir = Path(__file__).resolve().parent
    lib_dir = package_dir / "_lib"
    for name in _library_names():
        yield lib_dir / name

    repository = package_dir.parents[1]
    for subdir in ("build", "build/Release", "build/Debug", "build/lib"):
        for name in _library_names():
            yield repository / subdir / name


def _configure(lib: ctypes.CDLL) -> None:
    def bind(name: str, restype, argtypes: list) -> None:
        try:
            function = getattr(lib, name)
        except AttributeError as exc:
            raise ABIError(f"native library is missing symbol {name}") from exc
        function.restype = restype
        function.argtypes = argtypes

    uint32 = ctypes.c_uint32
    uint64 = ctypes.c_uint64
    size_t = ctypes.c_size_t
    int32 = ctypes.c_int32
    double = ctypes.c_double
    void = None

    bind("rflux_abi_version", uint32, [])
    bind("rflux_version_string", ctypes.c_char_p, [])
    bind("rflux_status_name", ctypes.c_char_p, [ctypes.c_int])
    bind("rflux_last_error", ctypes.c_char_p, [])
    bind("rflux_clear_error", void, [])

    # qflux indicators
    bind("rflux_returns", ctypes.c_int,
         [c_double_p, size_t, int32, c_double_p])
    bind("rflux_sma", ctypes.c_int, [c_double_p, size_t, size_t, c_double_p])
    bind("rflux_rolling_sum", ctypes.c_int,
         [c_double_p, size_t, size_t, c_double_p])
    bind("rflux_rolling_std", ctypes.c_int,
         [c_double_p, size_t, size_t, size_t, c_double_p])
    bind("rflux_rolling_min", ctypes.c_int,
         [c_double_p, size_t, size_t, c_double_p])
    bind("rflux_rolling_max", ctypes.c_int,
         [c_double_p, size_t, size_t, c_double_p])
    bind("rflux_rolling_zscore", ctypes.c_int,
         [c_double_p, size_t, size_t, size_t, c_double_p])
    bind("rflux_rolling_quantile", ctypes.c_int,
         [c_double_p, size_t, size_t, ctypes.c_double, c_double_p])
    bind("rflux_ema", ctypes.c_int,
         [c_double_p, size_t, size_t, c_double_p])
    bind("rflux_rsi", ctypes.c_int,
         [c_double_p, size_t, size_t, c_double_p])
    bind("rflux_macd", ctypes.c_int,
         [c_double_p, size_t, size_t, size_t, size_t, c_double_p, c_double_p,
          c_double_p])
    bind("rflux_bollinger", ctypes.c_int,
         [c_double_p, size_t, size_t, double, c_double_p, c_double_p,
          c_double_p])
    bind("rflux_atr", ctypes.c_int,
         [c_double_p, c_double_p, c_double_p, size_t, size_t, c_double_p])
    bind("rflux_drawdown", ctypes.c_int, [c_double_p, size_t, c_double_p])
    bind("rflux_rolling_corr", ctypes.c_int,
         [c_double_p, c_double_p, size_t, size_t, c_double_p])
    bind("rflux_rolling_cov", ctypes.c_int,
         [c_double_p, c_double_p, size_t, size_t, size_t, c_double_p])
    bind("rflux_rolling_beta", ctypes.c_int,
         [c_double_p, c_double_p, size_t, size_t, c_double_p])
    bind("rflux_rolling_slope", ctypes.c_int,
         [c_double_p, size_t, size_t, c_double_p])
    bind("rflux_rolling_r2", ctypes.c_int,
         [c_double_p, size_t, size_t, c_double_p])

    # qflux statistics
    bind("rflux_describe", ctypes.c_int,
         [c_double_p, size_t, c_double_p, c_double_p, c_double_p, c_double_p,
          c_double_p, c_size_p])
    bind("rflux_quantile", ctypes.c_int,
         [c_double_p, size_t, double, c_double_p])
    bind("rflux_pearson", ctypes.c_int,
         [c_double_p, c_double_p, size_t, c_double_p])
    bind("rflux_corr_matrix", ctypes.c_int,
         [c_double_p, size_t, size_t, c_double_p])
    bind("rflux_sharpe", ctypes.c_int,
         [c_double_p, size_t, double, double, c_double_p])
    bind("rflux_sortino", ctypes.c_int,
         [c_double_p, size_t, double, double, c_double_p])
    bind("rflux_var_cvar", ctypes.c_int,
         [c_double_p, size_t, double, c_double_p, c_double_p])

    # qflux cross-sectional ranking
    bind("rflux_rank", ctypes.c_int, [c_double_p, size_t, c_double_p])
    bind("rflux_percentile_rank", ctypes.c_int,
         [c_double_p, size_t, c_double_p])
    bind("rflux_zscore", ctypes.c_int,
         [c_double_p, size_t, size_t, c_double_p])
    bind("rflux_winsorize", ctypes.c_int,
         [c_double_p, size_t, double, double, c_double_p])
    bind("rflux_neutralize", ctypes.c_int,
         [c_double_p, size_t, c_int32_p, size_t, c_double_p])

    # regimelab
    bind("rflux_hmm_options_init", ctypes.c_int,
         [ctypes.POINTER(HMMOptions), size_t])
    bind("rflux_hmm_create", ctypes.c_int,
         [size_t, size_t, ctypes.POINTER(ctypes.c_void_p)])
    bind("rflux_hmm_destroy", void, [ctypes.c_void_p])
    bind("rflux_hmm_dimensions", ctypes.c_int,
         [ctypes.c_void_p, c_size_p, c_size_p])
    bind("rflux_hmm_fit", ctypes.c_int,
         [ctypes.c_void_p, c_double_p, size_t, ctypes.POINTER(HMMOptions)])
    bind("rflux_hmm_fit_summary", ctypes.c_int,
         [ctypes.c_void_p, c_size_p, c_int32_p, c_double_p, c_double_p,
          c_double_p])
    bind("rflux_hmm_score", ctypes.c_int,
         [ctypes.c_void_p, c_double_p, size_t, c_double_p])
    bind("rflux_hmm_get_parameters", ctypes.c_int,
         [ctypes.c_void_p, c_double_p, c_double_p, c_double_p, c_double_p])
    bind("rflux_hmm_set_parameters", ctypes.c_int,
         [ctypes.c_void_p, c_double_p, c_double_p, c_double_p, c_double_p])
    bind("rflux_hmm_viterbi", ctypes.c_int,
         [ctypes.c_void_p, c_double_p, size_t, c_int32_p])
    bind("rflux_hmm_filtered", ctypes.c_int,
         [ctypes.c_void_p, c_double_p, size_t, c_double_p])
    bind("rflux_hmm_smoothed", ctypes.c_int,
         [ctypes.c_void_p, c_double_p, size_t, c_double_p])
    bind("rflux_hmm_forecast", ctypes.c_int,
         [ctypes.c_void_p, c_double_p, size_t, c_double_p])
    bind("rflux_hmm_filter_step", ctypes.c_int,
         [ctypes.c_void_p, c_double_p, c_double_p, c_double_p])
    bind("rflux_hmm_stationary", ctypes.c_int,
         [ctypes.c_void_p, c_double_p])
    bind("rflux_hmm_expected_durations", ctypes.c_int,
         [ctypes.c_void_p, c_double_p])

    # rule-based regimes
    bind("rflux_volatility_regime", ctypes.c_int,
         [c_double_p, size_t, size_t, size_t, double, double, c_int32_p,
          c_double_p])
    bind("rflux_trend_regime", ctypes.c_int,
         [c_double_p, size_t, size_t, size_t, double, c_int32_p, c_double_p,
          c_double_p])
    bind("rflux_composite_regime", ctypes.c_int,
         [c_int32_p, c_int32_p, size_t, c_int32_p])

    # percolab
    bind("rflux_corr_to_distance", ctypes.c_int,
         [c_double_p, size_t, c_double_p])
    bind("rflux_graph_components", ctypes.c_int,
         [c_double_p, size_t, double, c_int32_p, c_size_p, c_size_p])
    bind("rflux_network_strength", ctypes.c_int,
         [c_double_p, size_t, c_double_p])
    bind("rflux_network_centrality", ctypes.c_int,
         [c_double_p, size_t, uint32, c_double_p])
    bind("rflux_network_clustering", ctypes.c_int,
         [c_double_p, size_t, c_double_p])
    bind("rflux_network_transitivity", ctypes.c_int,
         [c_double_p, size_t, c_double_p])
    bind("rflux_network_path_stats", ctypes.c_int,
         [c_double_p, size_t, c_double_p, c_double_p, c_size_p, c_size_p,
          c_int32_p])
    bind("rflux_network_core_numbers", ctypes.c_int,
         [c_double_p, size_t, c_int32_p])
    bind("rflux_network_assortativity", ctypes.c_int,
         [c_double_p, size_t, c_double_p])
    bind("rflux_network_modularity", ctypes.c_int,
         [c_double_p, size_t, c_int32_p, size_t, c_double_p])
    bind("rflux_pagerank", ctypes.c_int,
         [c_double_p, size_t, double, size_t, double, c_double_p])
    bind("rflux_network_betweenness", ctypes.c_int,
         [c_double_p, size_t, ctypes.c_int, c_double_p])
    bind("rflux_network_bridges", ctypes.c_int,
         [c_double_p, size_t, c_uint32_p, c_uint32_p, c_size_p])
    bind("rflux_network_articulation_points", ctypes.c_int,
         [c_double_p, size_t, c_int32_p])
    bind("rflux_network_truss_numbers", ctypes.c_int,
         [c_double_p, size_t, c_int32_p])
    bind("rflux_network_mst", ctypes.c_int,
         [c_double_p, size_t, c_uint32_p, c_uint32_p, c_double_p, c_size_p])
    bind("rflux_mst_clusters", ctypes.c_int,
         [c_double_p, size_t, double, c_int32_p, c_size_p, c_size_p])
    bind("rflux_bond_percolation", ctypes.c_int,
         [c_double_p, size_t, c_double_p, size_t, size_t, uint64, c_double_p,
          c_double_p, c_double_p])
    bind("rflux_site_percolation", ctypes.c_int,
         [c_double_p, size_t, c_double_p, size_t, size_t, uint64, c_double_p,
          c_double_p, c_double_p])
    bind("rflux_bond_percolation_weighted", ctypes.c_int,
         [c_double_p, size_t, c_double_p, size_t, size_t, uint64, c_double_p,
          c_double_p, c_double_p])
    bind("rflux_site_percolation_weighted", ctypes.c_int,
         [c_double_p, size_t, c_double_p, c_double_p, size_t, size_t, uint64,
          c_double_p, c_double_p, c_double_p])
    bind("rflux_percolation_scan", ctypes.c_int,
         [c_double_p, size_t, c_double_p, c_double_p, int32, c_double_p,
          size_t, size_t, uint64, c_double_p, c_double_p, c_double_p])
    bind("rflux_cluster_size_distribution", ctypes.c_int,
         [c_double_p, size_t, c_double_p, c_double_p, int32, double, size_t,
          uint64, c_double_p])
    bind("rflux_targeted_removal", ctypes.c_int,
         [c_double_p, size_t, c_uint32_p, size_t, c_size_p, c_double_p,
          c_double_p, c_size_p])
    bind("rflux_risk_diffusion", ctypes.c_int,
         [c_double_p, size_t, c_uint32_p, size_t, double, double, size_t,
          size_t, uint64, c_double_p, c_double_p])

    # factorlab
    bind("rflux_factor_momentum", ctypes.c_int,
         [c_double_p, size_t, size_t, size_t, c_double_p])
    bind("rflux_factor_volume_reversal", ctypes.c_int,
         [c_double_p, c_double_p, size_t, size_t, size_t, c_double_p])
    bind("rflux_factor_vwap_deviation", ctypes.c_int,
         [c_double_p, c_double_p, size_t, size_t, size_t, c_double_p])
    bind("rflux_factor_amihud", ctypes.c_int,
         [c_double_p, c_double_p, size_t, size_t, size_t, double, c_double_p])

    # portfolio construction
    bind("rflux_hrp_weights", ctypes.c_int,
         [c_double_p, size_t, c_double_p])
    bind("rflux_hrp_order", ctypes.c_int,
         [c_double_p, size_t, c_size_p])

    # execution and robust prediction
    bind("rflux_almgren_chriss_plan", ctypes.c_int,
         [double, size_t, double, double, double, double, double, c_double_p,
          c_double_p, c_double_p, c_double_p])
    bind("rflux_huber_regression", ctypes.c_int,
         [c_double_p, c_double_p, size_t, size_t, double, size_t, double,
          ctypes.c_int32, c_double_p, c_double_p, c_double_p, c_size_p,
          c_int32_p])

    # percolation detector and pre-trade gate
    bind("rflux_site_percolation_analyze", ctypes.c_int,
         [c_double_p, size_t, double, size_t, uint64, double, size_t,
          c_double_p, c_double_p, c_double_p, c_size_p, c_int32_p])
    bind("rflux_site_percolation_threshold", ctypes.c_int,
         [c_double_p, size_t, size_t, uint64, double, size_t, c_double_p])
    bind("rflux_pretrade_check", ctypes.c_int,
         [ctypes.c_int64, double, ctypes.c_int32, double, double, double,
          double, double, double, ctypes.c_int32, c_int32_p, ctypes.c_char_p,
          ctypes.c_size_t])


def _load_library() -> ctypes.CDLL:
    global _LIBRARY
    if _LIBRARY is not None:
        return _LIBRARY

    attempts: list[str] = []
    explicit_env = os.environ.get("RFLUX_LIBRARY")
    candidates = list(_candidate_paths())
    for candidate in candidates:
        try:
            if not candidate.exists():
                attempts.append(f"{candidate} (missing)")
                continue
            lib = ctypes.CDLL(str(candidate))
            _configure(lib)
            abi = int(lib.rflux_abi_version())
            if abi != ABI_VERSION:
                raise ABIError(
                    f"native library {candidate} exposes ABI {abi}, "
                    f"but this Python package expects ABI {ABI_VERSION}")
            _LIBRARY = lib
            return lib
        except OSError as exc:
            attempts.append(f"{candidate} ({exc})")
        except ABIError:
            raise
    if explicit_env:
        details = "\n  ".join(attempts) if attempts else explicit_env
        raise LibraryNotFoundError(
            "RFLUX_LIBRARY was set but the shared library could not be used.\n"
            f"Tried:\n  {details}"
        )

    found = ctypes.util.find_library("rflux")
    if found:
        try:
            lib = ctypes.CDLL(found)
            _configure(lib)
            abi = int(lib.rflux_abi_version())
            if abi != ABI_VERSION:
                raise ABIError(
                    f"system rflux exposes ABI {abi}, expected {ABI_VERSION}")
            _LIBRARY = lib
            return lib
        except OSError as exc:
            attempts.append(f"{found} ({exc})")

    details = "\n  ".join(attempts) if attempts else "no candidates"
    raise LibraryNotFoundError(
        "Could not load the rflux shared library. Build it with `make` or "
        "CMake, then set RFLUX_LIBRARY to the full library path.\n"
        f"Tried:\n  {details}")


def get_lib() -> ctypes.CDLL:
    return _load_library()


def native_version() -> str:
    value = get_lib().rflux_version_string()
    return value.decode("utf-8", "replace") if value else ""


def abi_version() -> int:
    return int(get_lib().rflux_abi_version())


def last_error() -> str:
    value = get_lib().rflux_last_error()
    return value.decode("utf-8", "replace") if value else ""


def check(status: int, function: str) -> None:
    if int(status) == STATUS_OK:
        return
    raise RFluxError(function, int(status), last_error())


__all__ = [
    "ABIError",
    "ABI_VERSION",
    "HMMOptions",
    "LibraryNotFoundError",
    "RFluxError",
    "STATUS_NAMES",
    "abi_version",
    "check",
    "get_lib",
    "last_error",
    "native_version",
]
