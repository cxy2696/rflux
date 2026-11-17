#!/usr/bin/env python3
"""Small ctypes-vs-pybind11 benchmark for the hottest rflux entry points."""

from __future__ import annotations

import argparse
import ctypes
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from rflux import qflux  # noqa: E402
from rflux._native_backend import native_module as _native  # noqa: E402
from rflux._common import double_ptr  # noqa: E402
from rflux._ffi import HMMOptions, get_lib  # noqa: E402


def _time(function, repeats: int) -> float:
    best = float("inf")
    for _ in range(repeats):
        start = time.perf_counter()
        function()
        best = min(best, time.perf_counter() - start)
    return best


def bench_rolling_quantile(values: np.ndarray, window: int, repeats: int) -> None:
    if _native is None:
        print("native extension unavailable; nothing to compare")
        return
    lib = get_lib()
    out = np.empty_like(values)

    def ctypes_call():
        lib.rflux_rolling_quantile(
            double_ptr(values), values.size, window, 0.5, double_ptr(out)
        )

    def native_call():
        _native.series("rflux_rolling_quantile", values, window, 0.5)

    ctypes_time = _time(ctypes_call, repeats)
    native_time = _time(native_call, repeats)
    print(
        f"rolling_quantile n={values.size:,} window={window}: "
        f"ctypes={ctypes_time * 1e3:.1f} ms "
        f"native={native_time * 1e3:.1f} ms "
        f"speedup={ctypes_time / native_time:.2f}x"
    )


def bench_hmm(observations: np.ndarray, repeats: int) -> None:
    if _native is None:
        return
    lib = get_lib()
    options = HMMOptions()
    lib.rflux_hmm_options_init(ctypes.byref(options), ctypes.sizeof(options))
    options.n_init = 2
    options.max_iter = 50

    def ctypes_fit():
        handle = ctypes.c_void_p()
        lib.rflux_hmm_create(2, 1, ctypes.byref(handle))
        try:
            lib.rflux_hmm_fit(
                handle, double_ptr(observations), observations.shape[0],
                ctypes.byref(options),
            )
        finally:
            lib.rflux_hmm_destroy(handle)

    def native_fit():
        model = _native.GaussianHMM(2, 1)
        model.fit(observations, n_init=2, max_iter=50, seed=42)

    ctypes_time = _time(ctypes_fit, repeats)
    native_time = _time(native_fit, repeats)
    print(
        f"hmm_fit n={observations.shape[0]:,} k=2: "
        f"ctypes={ctypes_time * 1e3:.1f} ms "
        f"native={native_time * 1e3:.1f} ms "
        f"speedup={ctypes_time / native_time:.2f}x"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=1_000_000)
    parser.add_argument("--window", type=int, default=512)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    rng = np.random.default_rng(0)
    values = rng.normal(size=args.rows)
    bench_rolling_quantile(values, args.window, args.repeats)
    observations = rng.normal(size=(min(args.rows, 200_000), 1))
    bench_hmm(observations, max(1, args.repeats // 2))
    print("qflux rolling results are identical:",
          bool(np.allclose(
              qflux.rolling_quantile(values[:10_000], args.window, 0.5),
              _native.series("rflux_rolling_quantile",
                             values[:10_000], args.window, 0.5),
              equal_nan=True,
          )) if _native is not None else "n/a")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
