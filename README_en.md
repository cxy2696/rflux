# rflux — English guide

[中文说明](README_cn.md) · [Project home](README.md) · [CI](https://github.com/cxy2696/rflux/actions/workflows/ci.yml)

rflux is a quantitative research library with a stable C ABI, a C++17 calculation core and a Python research layer. Python calls the shared library through `ctypes`, so the same binary serves every supported Python version. Other languages can use the C header without dealing with C++ types or exceptions.

The current release is `0.4.2`; the C ABI version is `1`.

## What is included

| Area | Contents |
| --- | --- |
| `qflux` | Returns, SMA, EMA, RSI, MACD, Bollinger, ATR, drawdown; rolling mean, volatility, min/max, z-score, correlation, beta, slope, R²; moments, quantiles, Sharpe, Sortino, VaR/CVaR; ranking, percentile rank, winsorization, group neutralization and pairwise correlation matrices |
| `regimelab` | Diagonal Gaussian HMM with multiple restarts, filtered/smoothed probabilities, Viterbi, state forecasting, out-of-sample scoring and state alignment; causal volatility, trend and composite regimes |
| `percolab` | Correlation distance, threshold graphs, connected components, strength, eigenvector centrality, clustering, minimum spanning trees, bond/site percolation and SIR diffusion |
| Python tools | Optional akshare/yfinance/CSV adapters, performance and risk summaries, factor IC, quantile portfolios, a simple cost-aware long-short backtest, portfolio risk contributions and regime attribution |

Rolling volatility, z-scores, correlation, beta, slope and R² use mergeable Welford block moments, so a large price level does not erase the small variance inside a window. Random procedures take explicit seeds and are reproducible on the same platform and version.

## Direct use

The repository includes a small price panel for a first run. After building:

```bash
make -j
PYTHONPATH=python python -m rflux version
PYTHONPATH=python python -m rflux analyze examples/data/sample_prices.csv
```

After an editable install, the command is just `rflux`:

```bash
make python-lib
python -m pip install -e python
rflux doctor
rflux analyze examples/data/sample_prices.csv --no-hmm --format json
rflux risk examples/data/sample_prices.csv --weights 0.2,0.2,0.2,0.2,0.2
```

`analyze` reads a wide CSV with a header, detects a `date`/`日期` column, and reports per-asset annual return, volatility, Sharpe, maximum drawdown and RSI, plus the equal-weight market, correlation summary, network and a 3-state HMM. `risk` uses the same price table and `--weights` to compute portfolio volatility and Euler risk contributions. Non-numeric columns are dropped, dates are sorted, and duplicate dates or mixed text/number columns are errors rather than silent gaps.

From Python:

```python
import numpy as np
import rflux
from rflux import analyze_prices

prices = np.array([[100, 50], [101, 49], [99, 51], [102, 52]], dtype=float)
report = analyze_prices(prices, names=["AAA", "BBB"])
print(report["market"]["annualized_return"])

risk = rflux.portfolio_risk(
    np.array([[0.01, 0.02], [0.02, -0.01], [-0.01, 0.01]]),
    [0.6, 0.4],
    names=["AAA", "BBB"],
)
print(risk["portfolio"]["annualized_volatility"])
```

See `python -m rflux analyze --help` and `python -m rflux risk --help` for all options. The CLI uses only NumPy and the standard library; pandas remains optional for the provider adapters.

## Use cases

Runnable scripts and commands are indexed in [examples/README.md](examples/README.md).

| Scenario | Entry point |
| --- | --- |
| Cross-sectional factor screening, sector neutralization, IC and long-short backtest | [python_factor_screen.py](examples/python_factor_screen.py) |
| Rolling HMM fitting, out-of-sample scoring and state alignment | [python_hmm_walkforward.py](examples/python_hmm_walkforward.py) |
| Correlation network, percolation scan and risk diffusion | [python_end_to_end.py](examples/python_end_to_end.py) |
| akshare / yfinance data | [python_real_data.py](examples/python_real_data.py) |
| C ABI / C++ research pipeline | [c_basic.c](examples/c_basic.c) · [cpp_research.cpp](examples/cpp_research.cpp) |

## Build and test

You need a C11 compiler, a C++17 compiler and GNU Make. The Python bindings need Python 3.10+ and NumPy.

```bash
git clone https://github.com/cxy2696/rflux.git
cd rflux
make -j
make check
```

Outputs:

- `build/librflux.dylib` on macOS, `build/librflux.so` on Linux
- `build/librflux_core.a`
- `build/test_capi`, `build/test_cpp`
- `build/example_c_basic`, `build/example_cpp_research`

CMake is also supported:

```bash
cmake -S . -B build/cmake -DCMAKE_BUILD_TYPE=Release
cmake --build build/cmake -j
ctest --test-dir build/cmake --output-on-failure
cmake --install build/cmake --prefix "$PWD/dist"
```

The installed CMake targets are `rflux::rflux` for the C ABI library and `rflux::rflux_core` for the C++ core.

## Python setup

For development from the repository:

```bash
export PYTHONPATH="$PWD/python"
export RFLUX_LIBRARY="$PWD/build/librflux.dylib"   # Linux: build/librflux.so
python -c "import rflux; print(rflux.native_version())"
```

Without `RFLUX_LIBRARY`, the loader checks `python/rflux/_lib/`, the repository `build/` directory and then the system library path.

To install the Python package:

```bash
make python-lib
python -m pip install -e python
```

Optional data dependencies:

```bash
python -m pip install pandas
python -m pip install akshare
python -m pip install yfinance
```

There is no standalone wheel yet. The Python package needs a native shared library for the same platform, so build it first. `_lib/` binaries are not committed. C++ users can include `rflux/rflux.hpp` for the full public interface.

## Factors and statistics

```python
import numpy as np
import rflux

prices = np.array([100, 101, 99, 103, 105, 104, 108, 110], dtype=float)
returns = rflux.qflux.returns(prices)
vol = rflux.qflux.rolling_std(returns, 5, ddof=1)
macd = rflux.qflux.macd(prices)
beta = rflux.qflux.rolling_beta(returns[1:], returns[1:] * 0.8, 5)

values = np.array([3.0, 1.0, 4.0, np.nan])
print(rflux.qflux.rank(values))
print(rflux.qflux.zscore(np.array([1.0, 2.0, 3.0])))
print(rflux.qflux.neutralize(
    np.array([1.0, 2.0, 10.0, 12.0]),
    np.array([0, 0, 1, 1], dtype=np.int32),
))
```

`rolling_beta(asset, benchmark, window)` is the window OLS slope `Cov(asset, benchmark) / Var(benchmark)`. Windows with missing values or zero benchmark variance return NaN.

## HMM regimes

```python
import numpy as np
from rflux import regimelab

rng = np.random.default_rng(7)
returns = np.concatenate([
    rng.normal(-0.001, 0.008, 300),
    rng.normal(0.001, 0.018, 300),
])

model = regimelab.fit_hmm(
    returns.reshape(-1, 1),
    n_states=2,
    n_init=3,
    max_iter=200,
    seed=42,
)
try:
    print(model.fit_info)
    filtered = model.filtered(returns)          # P(s_t | x_0..x_t)
    labels = model.filtered_states(returns)     # causal labels
    forecast = model.forecast(filtered[-1], horizon=5)
    expected = model.conditional_mean(filtered[-1], feature=0)
    print(labels[-10:], forecast, expected)
finally:
    model.close()
```

Use `filtered()` for point-in-time work. `smoothed()` and `viterbi()` use the entire sample and are suitable for retrospective analysis, not as signals known at time t.

State labels are not identified by the likelihood. The default fit sorts states by the first feature mean, which makes a single fit deterministic but does not keep labels stable across refits. `score()` evaluates the current parameters on new data for walk-forward model selection. `conditional_mean()` is a conditional emission mean, not a price forecast.

## Keeping state labels stable

HMM states can swap without changing the likelihood. For multi-window studies, align each new model to a reference model:

```python
from rflux import regimelab

# reference_parameters: HMMParameters from an earlier fit
# later_model:           GaussianHMM fitted on the later window
order = regimelab.match_states(reference_parameters, later_model.parameters)
aligned = regimelab.align_parameters(
    reference_parameters, later_model.parameters
)
labels = regimelab.align_labels(later_model.filtered_states(new_data), order)
probabilities = regimelab.align_probabilities(
    later_model.filtered(new_data), order
)
```

`order[k]` is the new-model state that corresponds to reference state `k`. The assignment minimises standardized mean differences plus log-variance differences using the Hungarian algorithm. Alignment fixes labels, not model drift.

A complete rolling example is included:

```bash
python examples/python_hmm_walkforward.py
```

## Real data

```python
from rflux import data

prices = data.load_yfinance(
    ["AAPL", "MSFT", "NVDA"], start="2021-01-01", end="2024-12-31"
)
returns = data.to_returns(prices, method="simple")

# China A-shares:
# prices = data.load_akshare(["000001", "600519"], start="2021-01-01")
```

akshare and yfinance are imported only when called. Empty responses and schema changes are reported as `DataSourceError`. Adjustments, suspensions, time zones and constituent-history bias remain the user's responsibility.

## Graphs, percolation and backtests

```python
import numpy as np
from rflux import backtest, percolab

network = percolab.correlation_network(returns.tail(252).to_numpy(), 0.45)
print(network.n_edges, network.components.n_components, network.eigenvector)

scan = percolab.bond_percolation(
    network.adjacency,
    np.linspace(0.0, 1.0, 11),
    repetitions=200,
    seed=7,
)
print(scan.giant_mean)

price_window = prices.tail(201).to_numpy()
factor = returns.tail(201).to_numpy()[:-1]
forward = backtest.forward_returns(price_window, periods=1)[:-1]
ic = backtest.factor_ic(factor, forward)
result = backtest.long_short_backtest(
    factor, forward, n_quantiles=3, cost_bps=5.0
)
print(backtest.summarize_ic(ic), result.metrics)
```

`factor[t]` must be known at the start of the holding period and `forward_returns[t]` is realised afterwards. The library does not repair misalignment. Turnover and cost conventions are documented in [METHODS_en.md](docs/METHODS_en.md), section 7.

By default `performance_summary()` annualises over the full input length, so internal NaN gaps do not shorten the clock; pass `periods=` to choose another horizon. A long-short book can lose more than its initial capital. In that case geometric annual return and Calmar are NaN, while the remaining risk statistics are still reported.

## C usage

```c
#include "rflux/rflux.h"
#include <stdio.h>

int main(void) {
    double prices[] = {1, 2, 3, 4, 5};
    double out[5];
    if (rflux_sma(prices, 5, 3, out) != RFLUX_OK) {
        fprintf(stderr, "%s\n", rflux_last_error());
        return 1;
    }
    printf("%.3f\n", out[4]);
    return 0;
}
```

```bash
cc -std=c11 -Iinclude main.c -Lbuild -lrflux \
   -Wl,-rpath,"$PWD/build" -o main && ./main
```

The C boundary exposes only C types and status codes. Error strings are thread-local. Callers own every output buffer, and each declaration states the required length. See [C ABI rules](docs/ABI.md) for the full contract.

## C++ usage

```cpp
#include "rflux/rflux.hpp"
#include <cstdio>
#include <vector>

int main() {
    std::vector<double> prices = {100, 101, 99, 103, 105};
    auto returns = rflux::qflux::returns(prices.data(), prices.size());

    rflux::regimelab::GaussianHMM model(2, 1);
    rflux::regimelab::HmmOptions options;
    options.seed = 42;
    options.n_init = 3;
    model.fit(returns.data() + 1, returns.size() - 1, options);

    std::printf("%.3f\n", model.score(returns.data() + 1, returns.size() - 1));
    return 0;
}
```

C++ uses exceptions for invalid input and numerical failures; the C wrapper converts them to status codes. The C++ ABI is a source interface, not a cross-compiler binary contract. Use the C ABI when that matters.

## Layout

```text
include/rflux/       C ABI and C++ headers
src/                 C++ implementation and C wrapper
python/rflux/        ctypes bindings, adapters and research tools
tests/               C, C++ and Python tests
examples/            factors, walk-forward HMM, networks, real data and C/C++ examples
scripts/             Markdown link, fence and math-delimiter checks
docs/                ABI, methods and extension notes
```

## Documentation

- Methods and boundary conditions: [中文](docs/METHODS_cn.md) · [English](docs/METHODS_en.md)
- Extension steps and release checks: [中文](docs/EXTENDING_cn.md) · [English](docs/EXTENDING_en.md)
- Portfolio risk and regime attribution: [English](docs/APPLICATIONS_en.md) · [中文](docs/APPLICATIONS_cn.md)
- C ABI ownership, buffer and compatibility rules: [docs/ABI.md](docs/ABI.md) (Chinese)

## Tests and CI

```bash
make check
```

The tests cover C ABI buffer/error handling, known indicator values, pairwise-complete correlations with missing data, HMM fitting and causality, state alignment, out-of-sample scoring, graph algorithms, percolation endpoints, diffusion bounds and the Python backtest helpers. `make check` also verifies Markdown links, code fences and GitHub math delimiters.

GitHub Actions runs Make and CMake/CTest tests on Linux and macOS, and executes the factor, walk-forward HMM, network and C/C++ examples on Linux.

## Limits

- HMM emissions are diagonal Gaussian; there is no full-covariance or Student-t variant.
- HMM fitting is batch EM, not an online learner.
- State alignment only reconciles labels; it does not resolve structural model changes.
- Percolation and SIR diffusion are scenario tools, not market probability forecasts.
- The Python backtest does not model borrow availability, suspensions, limit moves, market impact or capacity.
- C++ headers are a source interface, not a stable ABI.

## License

MIT. See [LICENSE](LICENSE).
