# rflux

[![CI](https://github.com/cxy2696/rflux/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/cxy2696/rflux/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
![C](https://img.shields.io/badge/C-11-555)
![C++](https://img.shields.io/badge/C%2B%2B-17-00599C)
![Python](https://img.shields.io/badge/python-3.10%2B-3776AB)

**Language:** [中文说明](README_cn.md) · [English guide](README_en.md)

rflux is a quantitative research library. The calculations live in C++17, a stable C ABI keeps them callable from other languages, and the Python package adds data loading, backtests and a command-line interface for quick inspection.

Version `0.4.2` keeps ABI version 1 and adds a direct analysis API plus the `rflux` command.

```python
import numpy as np
import rflux

prices = np.array([100, 101, 99, 103, 105, 104, 108, 110], dtype=float)
returns = rflux.qflux.returns(prices)
macd = rflux.qflux.macd(prices)

rng = np.random.default_rng(0)
regime_returns = np.concatenate([
    rng.normal(0.0005, 0.008, 300),
    rng.normal(-0.0005, 0.02, 300),
]).reshape(-1, 1)

model = rflux.regimelab.fit_hmm(regime_returns, n_states=2, seed=42)
filtered = model.filtered(regime_returns)
print(filtered[-1])
model.close()
```

Build, test and try it directly:

```bash
git clone https://github.com/cxy2696/rflux.git
cd rflux
make -j
PYTHONPATH=python python -m rflux version
PYTHONPATH=python python -m rflux analyze examples/data/sample_prices.csv
PYTHONPATH=python python -m rflux risk examples/data/sample_prices.csv \
  --weights 0.2,0.2,0.2,0.2,0.2
```

Once the package is installed (`make python-lib && pip install -e python`), the same command is available as `rflux`.

```python
import numpy as np
from rflux import analyze_prices

prices = np.array([[100, 50], [101, 49], [99, 51], [102, 52]], dtype=float)
report = analyze_prices(prices, names=["AAA", "BBB"])
print(report["market"]["sharpe"])
```

The shared library is written to `build/`. Python finds it automatically when run from the repository, or you can set `RFLUX_LIBRARY=/path/to/librflux.dylib` / `.so`.

## Documentation

- [中文说明](README_cn.md): build, Python/C/C++ usage, data adapters and HMM guidance.
- [English guide](README_en.md): same scope in English.
- [Examples and use cases](examples/README.md): factor screening, walk-forward HMM, network risk, real data and command-line analysis.
- Applications: [portfolio risk and regime attribution](docs/APPLICATIONS_en.md) ([中文](docs/APPLICATIONS_cn.md)).
- [C ABI rules](docs/ABI.md): ownership, status codes, buffer lengths and compatibility (Chinese; the public header is documented in English).
- Methods and conventions: [中文](docs/METHODS_cn.md) · [English](docs/METHODS_en.md).
- Extension guide: [中文](docs/EXTENDING_cn.md) · [English](docs/EXTENDING_en.md).
- [Changelog](CHANGELOG.md) and [contributing notes](CONTRIBUTING.md).

## Scope

The library covers:

- **qflux**: returns, moving averages, RSI, MACD, Bollinger bands, ATR, drawdown, rolling statistics, covariance, beta, cross-sectional ranking, winsorization, z-scores, neutralization, correlation matrices, Sharpe/Sortino and historical VaR/CVaR.
- **regimelab**: diagonal Gaussian HMMs with multiple restarts, filtered/smoothed state probabilities, Viterbi paths, state forecasts, out-of-sample scores, state alignment, causal volatility/trend regimes.
- **percolab**: correlation distance, threshold graphs, connected components, centrality, PageRank, clustering, minimum spanning trees, bond/site percolation and SIR diffusion.
- **Python tools**: optional akshare/yfinance/CSV adapters, performance summaries, IC, quantile portfolios, a cost-aware long-short backtest, portfolio risk contributions and regime attribution.

The code is deterministic for a given input and seed. Random procedures use explicit seeds; nothing is downloaded at import time.

## License

MIT. See [LICENSE](LICENSE).
