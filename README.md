# rflux

[![CI](https://github.com/cxy2696/rflux/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/cxy2696/rflux/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

**English** · [中文](README_cn.md)

rflux is a C++17 + Python quantitative research library. The calculation core
is C++; a stable C ABI and a pybind11 extension expose it to Python. Data
adapters, backtests and command-line tools live in the Python layer.

Current version: `0.7.0`. C ABI version: `1`.

## Install

From source:

```bash
git clone https://github.com/cxy2696/rflux.git
cd rflux
python -m pip install .
```

This needs CMake 3.26+, Ninja, a C++17 compiler and Python 3.10+. The runtime
dependency is NumPy. Data adapters need pandas and one provider SDK:

```bash
python -m pip install ".[all]"   # pandas + AKShare + yfinance
```

There is no PyPI publication. Install a locally built wheel with:

```bash
python -m pip install dist/rflux-*.whl
```

CPython 3.12+ uses an abi3 wheel; 3.10 and 3.11 use version-specific wheels.

## Quick start

```python
import numpy as np
import rflux

prices = np.array([100, 101, 99, 103, 105, 104, 108, 110], dtype=float)
print(rflux.qflux.returns(prices))
print(rflux.qflux.macd(prices).histogram[-1])

rng = np.random.default_rng(0)
returns = np.concatenate([
    rng.normal(0.0005, 0.008, 300),
    rng.normal(-0.0005, 0.02, 300),
]).reshape(-1, 1)

model = rflux.fit_hmm(returns, n_states=2, seed=42)
try:
    print(model.filtered(returns)[-1])
finally:
    model.close()
```

Command line:

```bash
rflux doctor
rflux analyze examples/data/sample_prices.csv --no-hmm
rflux risk examples/data/sample_prices.csv --weights 0.2,0.2,0.2,0.2,0.2
rflux advice examples/data/sample_prices.csv --no-hmm
```

`rflux advice` is a transparent, long-only research rebalance proposal:
HRP base weights, 20/60-day momentum, trend and RSI adjustments, a volatility
target, an optional HMM risk state, 2-sigma stop references and position
sizing. It is not personalised investment advice and does not promise returns.

## Features

- **qflux**: returns, moving averages, RSI, MACD, Bollinger bands, ATR,
  drawdown, rolling statistics, quantiles, covariance, beta, ranking,
  winsorisation, neutralisation, correlation matrices, Sharpe/Sortino and
  historical VaR/CVaR.
- **regimelab**: Gaussian HMM fitting, filtered/smoothed probabilities,
  Viterbi paths, state forecasts, state alignment, volatility/trend regimes and
  sticky regimes.
- **percolab**: correlation networks, components, centrality, PageRank,
  clustering, k-core, bridges, articulation points, k-truss, MST, bond/site
  percolation, cluster-size distributions and risk diffusion.
- **Algorithms**: momentum, volume reversal, VWAP, Amihud, HRP/NCO-style
  allocation, Huber regression, graph smoothing, Almgren-Chriss execution and
  top-of-book metrics.
- **Data**: local CSV, AKShare and yfinance normalised to one long OHLCV
  contract, with symbol normalisation, validation, SQLite cache and retries.
- **C ABI**: factor, HRP, Almgren-Chriss, Huber, percolation and pre-trade risk
  kernels are callable from C and from the Python ctypes fallback.

## Data example

```python
from rflux import data

loader = data.DataLoaderFactory.create("csv", root="./data/daily")
frame = loader.load(["AAPL.US", "MSFT.US"], start="2024-01-01", end="2024-12-31")
close = data.pivot_ohlcv(frame, "close")
```

## Development

```bash
python -m pip install -e ".[test]"
python -m pytest tests/python -q

cmake --preset dev
cmake --build --preset dev
ctest --preset dev --output-on-failure
```

Set `RFLUX_DISABLE_NATIVE=1` to force the ctypes C ABI path.

## Documentation

- [中文说明](README_cn.md)
- [C ABI](docs/ABI.md): ownership, status codes and compatibility
- [Data layer](docs/DATA_en.md) · [中文](docs/DATA_cn.md)
- [Algorithm matrix](docs/ALGORITHMS_en.md) · [中文](docs/ALGORITHMS_cn.md)
- [C++ core contracts](docs/CORE_en.md) · [中文](docs/CORE_cn.md)
- [Native bridge](docs/NATIVE_en.md) · [中文](docs/NATIVE_cn.md)
- [Methods and formulas](docs/METHODS_en.md) · [中文](docs/METHODS_cn.md)
- [Applications](docs/APPLICATIONS_en.md) · [中文](docs/APPLICATIONS_cn.md)
- [Extension guide](docs/EXTENDING_en.md) · [中文](docs/EXTENDING_cn.md)
- [Examples](examples/README.md), [changelog](CHANGELOG.md), [release checklist](docs/RELEASE.md)

## License

MIT. See [LICENSE](LICENSE).
