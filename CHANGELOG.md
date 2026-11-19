# Changelog

## 0.9.0 - 2026-11-19

The Python layer is now a thin binding: every algorithm and computation runs in
C++, and the ctypes bridge was removed.  The C ABI stays `1`; the new report
kernels are exposed through pybind11 only.

- Removed the ctypes fallback (`python/rflux/_ffi.py`, pointer helpers,
  `RFLUX_DISABLE_NATIVE`, `RFLUX_LIBRARY` and the `_lib` wheel payload).  The
  Python package imports the compiled pybind11 extension directly and reports
  a clear `ImportError` when it has not been built.  The public
  `LibraryNotFoundError` export is removed with the ctypes loader.
- Added C++ research kernels: advice signals/scores/allocation/regime/sizing
  (`src/portfoliolab/advice.cpp`), regime attribution
  (`src/regimelab/attribution.cpp`), sticky regimes, aligned HMM parameter
  reordering, correlation adjacency and component sizes, percolation summaries,
  removal ordering and core curves, equal-weight market returns, complete-case
  covariance and portfolio series.
- Added small numeric helpers used by the report layer: finite-value statistics,
  last finite value, top-k indices, argmax/row-argmax, int32 maximum and linear
  prediction.  The backtest IC kernel accepts a minimum pair count.
- Rewrote `qflux.py`, `metrics.py`, `execution.py`, `factorlab.py`, `mllab.py`,
  `portfoliolab.py`, `risk.py`, `percolab.py`, `regimelab.py`, `advice.py`,
  `analysis.py`, `apps.py` and `backtest.py` as validation/formatting wrappers
  over the native extension.  The Python package drops from about 8.3k to
  6.4k lines and no longer contains NumPy compute loops.
- `make check`, `make algo-check`, the VS Code verification script and CI now
  run a single native test path; the wheel ships only the extension, while the
  C ABI library remains a separate CMake install target.
