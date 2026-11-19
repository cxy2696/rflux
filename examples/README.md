# rflux examples

**English** · [中文](README_cn.md)

Run these from the repository root. Python examples need NumPy;
`python_real_data.py` also needs pandas and one provider SDK. Methods are
documented in [METHODS_en.md](../docs/METHODS_en.md), the algorithm matrix in
[ALGORITHMS_en.md](../docs/ALGORITHMS_en.md).

```bash
make -j
python -m pip install -e .   # builds and installs the native extension
```

The Python examples import the installed package; there is no ctypes or
pure-Python execution path. `make native` is equivalent to the editable
install when only the extension needs rebuilding.

## 0. Command line

```bash
python -m rflux analyze examples/data/sample_prices.csv
python -m rflux analyze examples/data/sample_prices.csv --format json
python -m rflux advice examples/data/sample_prices.csv --no-hmm
```

`analyze` reads a wide CSV, detects the date column, and reports asset
statistics, an equal-weight market, a correlation network and a 3-state HMM.
`--no-hmm` and `--no-network` skip the slower sections. `advice` produces a
risk-aware rebalance proposal. After installation the prefix becomes
`rflux ...`.

## 1. Factor screen

```bash
python examples/python_factor_screen.py
```

Synthetic multi-asset prices drive a 20-day momentum, low-volatility and
short-term reversal screen. The pipeline applies winsorisation, cross-sectional
z-scores and industry neutralisation, then combines the factors and runs rank IC,
quantile portfolios and a cost-aware long-short backtest.

Factors at close `t` are aligned only with returns from `t` to `t+1`; future
prices are never read. The output checks the code path; it is not an investment result. Replace `simulate_prices()` with a real wide price panel to reuse the
pipeline:

```python
from rflux import data

frame = data.load_ohlcv(
    "yfinance", ["AAPL", "MSFT", "NVDA", "AMZN"], start="2021-01-01"
)
prices = data.pivot_ohlcv(frame, "close").ffill().to_numpy()
```

## 2. HMM walk-forward

```bash
python examples/python_hmm_walkforward.py
```

Refits a 3-state Gaussian HMM on rolling windows, scores the next block
out-of-sample, and aligns state labels to a reference model with the Hungarian
algorithm. It answers two questions: is the out-of-sample likelihood stable,
and do labels remain comparable after refitting?

## 3. Correlation network, percolation and diffusion

```bash
python examples/python_end_to_end.py
```

Builds a synthetic panel, computes features, an HMM state, a 120-day
correlation network, components, eigenvector centrality, path efficiency,
k-core layers, bond percolation, targeted removal and SIR diffusion, then runs
a quantile long-short backtest.

Minimal network use:

```python
from rflux import percolab

network = percolab.correlation_network(returns_panel, threshold=0.45)
print(network.components.largest_size, network.max_core)
print(network.eigenvector, network.path_stats.efficiency)

scan = percolab.bond_percolation(
    network.adjacency, [0.0, 0.25, 0.5, 0.75, 1.0],
    repetitions=300, seed=42,
)
print(scan.giant_mean, scan.susceptibility)

removal = percolab.targeted_removal(
    network.adjacency, strategy="eigenvector", checkpoints=10
)
print(removal.giant_fraction)
```

## 4. Real data

```bash
python -m pip install pandas yfinance
python examples/python_real_data.py \
  --source yfinance --symbols AAPL MSFT NVDA --start 2021-01-01
```

A-share data:

```bash
python -m pip install pandas akshare
python examples/python_real_data.py \
  --source akshare --symbols 600519.SH 000001.SZ --start 2021-01-01
```

Custom CSV:

```bash
python examples/python_real_data.py \
  --source csv --path my_prices.csv --date-column date \
  --symbols AAPL MSFT
```

Adapters download data, set the date index and normalise column names.
Adjustments, suspensions, point-in-time constituents and survivorship remain
the caller's responsibility. Provider schemas can change; the script turns
empty responses and schema errors into `DataSourceError`.

## 5. Portfolio risk and regime attribution

```bash
python examples/python_applications.py
python -m rflux risk examples/data/sample_prices.csv \
  --weights 0.30,0.25,0.20,0.15,0.10
```

`python_applications.py` computes weighted portfolio volatility and Euler risk
contributions, then labels equal-weight market returns with a causal
volatility regime and reports return, volatility, share and transition matrix
per state. See [APPLICATIONS_en.md](../docs/APPLICATIONS_en.md) for estimator
and interpretation details.

## 6. C and C++ integration

```bash
make examples
./build/example_c_basic
./build/example_cpp_research
```

- `c_basic.c`: C ABI only — SMA/RSI plus a threshold graph.
- `cpp_research.cpp`: C++ core — features, HMM, network, percolation and diffusion.

C users only need `include/rflux/rflux.h`; C++ users can include
`include/rflux/rflux.hpp` for all public headers.

## File index

| File | Purpose | Main API |
| --- | --- | --- |
| [data/sample_prices.csv](data/sample_prices.csv) | sample wide price table | CLI input |
| [python_factor_screen.py](python_factor_screen.py) | factor screen, neutralisation, backtest | `qflux`, `backtest` |
| [python_hmm_walkforward.py](python_hmm_walkforward.py) | rolling HMM, OOS score, state alignment | `regimelab` |
| [python_end_to_end.py](python_end_to_end.py) | regime + network + percolation + backtest | `regimelab`, `percolab`, `backtest` |
| [python_real_data.py](python_real_data.py) | AKShare/yfinance/long CSV | `data.DataLoaderFactory`, `load_ohlcv`, `pivot_ohlcv` |
| [python_applications.py](python_applications.py) | portfolio risk + regime attribution | `portfolio_risk`, `regime_attribution` |
| [c_basic.c](c_basic.c) | minimal C ABI | `rflux_sma`, `rflux_rsi`, `rflux_graph_components` |
| [cpp_research.cpp](cpp_research.cpp) | C++ research pipeline | `rflux::qflux`, `rflux::regimelab`, `rflux::percolab` |

All examples use explicit random seeds. Synthetic returns only exercise code
paths; they are not market conclusions.
