# rflux 中文说明

[English guide](README_en.md) · [项目首页](README.md) · [CI](https://github.com/cxy2696/rflux/actions/workflows/ci.yml)

rflux 是一个用于量化研究的 C / C++ / Python 算法库。C 层提供稳定的 ABI，C++ 层实现计算，Python 层负责数据接入和研究流程。

这样分层的目的很实际：Python 调用不需要为每个解释器版本重新编译扩展；C++ 代码可以被单独复用；其他语言只依赖 `rflux.h`，不需要理解 C++ 类型。

当前版本 `0.4.2`，C ABI 版本为 1。

## 内容

| 模块 | 功能 |
| --- | --- |
| `qflux` | 收益率、SMA、EMA、RSI、MACD、Bollinger、ATR、回撤；滚动均值、波动率、最小/最大值、z-score、相关系数、协方差、beta、斜率、R²；描述统计、分位数、Sharpe、Sortino、VaR/CVaR；排序、percentile、z-score、缩尾、分组中性化、成对相关矩阵 |
| `regimelab` | 对角高斯 HMM：多起点 EM、filtered/smoothed 状态概率、Viterbi、状态预测、单步过滤、平稳分布、期望持续期、样本外评分、状态对齐；因果波动率、趋势和复合 regime |
| `percolab` | 相关距离、阈值图、连通分量、加权度、特征向量中心性、PageRank、介数中心性、聚类系数、最小生成树、bond/site 渗流及加权版本、SIR 风险扩散 |
| Python 工具 | akshare / yfinance / tushare / CSV 适配，绩效与风险摘要，IC，分位数组合，带成本的多空回测，组合风险贡献和状态归因 |

滚动波动、z-score、相关、beta、斜率和 R² 使用可合并的 Welford 块矩，不会因为价格水平很大而丢掉窗口内的微小波动。所有随机过程接收显式 seed，同一版本、同一平台的结果可复现。

## 直接使用

仓库里已经包含一个可以直接分析的示例价格表。构建后从终端运行：

```bash
make -j
PYTHONPATH=python python -m rflux version
PYTHONPATH=python python -m rflux analyze examples/data/sample_prices.csv
```

安装编辑版之后，命令简化为 `rflux analyze ...`：

```bash
make python-lib
python -m pip install -e python
rflux doctor
rflux analyze examples/data/sample_prices.csv --no-hmm --format json
rflux risk examples/data/sample_prices.csv --weights 0.2,0.2,0.2,0.2,0.2
```

`analyze` 会读取带表头的宽表 CSV，自动识别 `date`/`日期` 列，计算每个资产的年度收益、波动、Sharpe、最大回撤、RSI，以及等权市场组合、相关矩阵摘要、相关网络和 3 状态 HMM。`risk` 使用同一份价格表，按 `--weights` 计算组合波动和 Euler 风险贡献。非数值列会被丢弃；日期会排序；重复日期和某列中混合文本/数字会直接报错，不会静默处理。

也可以直接从 Python 调用：

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

完整参数见 `python -m rflux analyze --help` 和 `python -m rflux risk --help`。命令行只依赖 NumPy 和标准库；需要 pandas 的数据适配器是另一条路径。

## 使用案例

可以直接运行的脚本和命令汇总在 [examples/README.md](examples/README.md)。

| 场景 | 入口 |
| --- | --- |
| 多因子选股、行业中性化、IC 和多空回测 | [python_factor_screen.py](examples/python_factor_screen.py) |
| HMM 滚动拟合、样本外评分、状态对齐 | [python_hmm_walkforward.py](examples/python_hmm_walkforward.py) |
| 相关网络、渗流扫描和风险扩散 | [python_end_to_end.py](examples/python_end_to_end.py) |
| akshare / yfinance 真实数据 | [python_real_data.py](examples/python_real_data.py) |
| C ABI / C++ 研究管线 | [c_basic.c](examples/c_basic.c) · [cpp_research.cpp](examples/cpp_research.cpp) |

## 构建

需要 C11 编译器、C++17 编译器和 GNU Make。Python 绑定需要 Python 3.10+ 与 NumPy。

```bash
git clone https://github.com/cxy2696/rflux.git
cd rflux
make -j
make check
```

产物：

- `build/librflux.dylib`（macOS）或 `build/librflux.so`（Linux）
- `build/librflux_core.a`
- `build/test_capi`、`build/test_cpp`
- `build/example_c_basic`、`build/example_cpp_research`

`make check` 运行 C ABI 测试、C++ 测试、Python 测试和 Markdown 文档检查。

CMake：

```bash
cmake -S . -B build/cmake -DCMAKE_BUILD_TYPE=Release
cmake --build build/cmake -j
ctest --test-dir build/cmake --output-on-failure
cmake --install build/cmake --prefix "$PWD/dist"
```

安装后 CMake 提供 `rflux::rflux`（C ABI 共享库）和 `rflux::rflux_core`（C++ 核心）两个 target。

## Python

最简单的开发方式是在仓库根目录运行：

```bash
export PYTHONPATH="$PWD/python"
export RFLUX_LIBRARY="$PWD/build/librflux.dylib"   # Linux: build/librflux.so
python -c "import rflux; print(rflux.native_version())"
```

不设置 `RFLUX_LIBRARY` 时，加载器会依次查找 `python/rflux/_lib/`、仓库 `build/` 和系统库。

需要安装成 Python 包时：

```bash
make python-lib
python -m pip install -e python
```

可选数据依赖：

```bash
python -m pip install pandas
python -m pip install akshare
python -m pip install yfinance
python -m pip install tushare
```

CSV 和 tushare 的宽表入口：

```python
from rflux import data

custom = data.load_csv(
    "my_prices.csv",
    date_column="trade_date",
    columns=["AAA", "BBB"],
    sep=",",
)
ts_prices = data.load_tushare(
    ["000001.SZ", "600519.SH"],
    start="2024-01-01",
    token="YOUR_TUSHARE_TOKEN",   # 也可设置 TUSHARE_TOKEN
)
```

`load_csv` 支持自定义日期列和列选择，也接受 `pandas.read_csv` 的参数。`rflux analyze`/`rflux risk` 的 CSV 读取不依赖 pandas，适合直接分析自有宽表。tushare 返回原始日收盘价，复权处理由调用方负责。

当前没有发布独立 wheel。Python 包需要同平台的 native 共享库，安装前先构建；`_lib/` 中的二进制不提交到仓库。C++ 用户可以直接包含 `rflux/rflux.hpp` 获取全部公共头。

### 指标和因子

```python
import numpy as np
import rflux

prices = np.array([100, 101, 99, 103, 105, 104, 108, 110], dtype=float)
returns = rflux.qflux.returns(prices)
vol = rflux.qflux.rolling_std(returns, 5, ddof=1)
macd = rflux.qflux.macd(prices)
beta = rflux.qflux.rolling_beta(returns[1:], returns[1:] * 0.8, 5)

rank = rflux.qflux.rank(np.array([3.0, 1.0, 4.0, np.nan]))
z = rflux.qflux.zscore(np.array([1.0, 2.0, 3.0]))
neutral = rflux.qflux.neutralize(
    np.array([1.0, 2.0, 10.0, 12.0]),
    np.array([0, 0, 1, 1], dtype=np.int32),
)
```

`rolling_beta(asset, benchmark, window)` 返回窗口 OLS 斜率 `Cov(asset, benchmark) / Var(benchmark)`；缺失窗口和基准零方差窗口为 NaN。

### HMM regime

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
    filtered = model.filtered(returns)       # P(s_t | x_0..x_t)
    labels = model.filtered_states(returns)  # 因果状态标签
    forecast = model.forecast(filtered[-1], horizon=5)
    expected = model.conditional_mean(filtered[-1], feature=0)
    print(filtered[-1], labels[-10:], forecast, expected)
finally:
    model.close()
```

注意事项：

- `filtered()` 只使用当前及过去观测，可以作为时点信号。
- `smoothed()` 和 `viterbi()` 使用全样本，适合回溯归因，不适合作为 t 时刻可交易标签。
- 状态编号是任意的。默认按第一特征均值排序，只能保证单次拟合的确定性，不能保证跨窗口一致。
- `score()` 可以对训练集或新数据计算对数似然，用于 walk-forward 模型选择。
- `conditional_mean()` 是当前参数下的条件发射均值，不是价格预测。

### 跨窗口状态对齐

HMM 的状态可以交换而不改变似然。多窗口研究里，推荐把新模型对齐到参考模型：

```python
from rflux import regimelab

# reference_parameters: 早期模型的 HMMParameters
# later_model:          新数据上拟合的 GaussianHMM
order = regimelab.match_states(reference_parameters, later_model.parameters)
aligned = regimelab.align_parameters(
    reference_parameters, later_model.parameters
)
labels = regimelab.align_labels(later_model.filtered_states(new_data), order)
probs = regimelab.align_probabilities(later_model.filtered(new_data), order)
```

`order[k]` 表示参考状态 k 对应新模型的状态。对齐用 Hungarian 算法最小化标准化均值差和对数方差差。它只统一标签，不保证两个模型完全相同。

完整滚动示例：

```bash
python examples/python_hmm_walkforward.py
```

### 真实数据

```python
from rflux import data

prices = data.load_yfinance(
    ["AAPL", "MSFT", "NVDA"], start="2021-01-01", end="2024-12-31"
)
returns = data.to_returns(prices, method="simple")

# A 股
# prices = data.load_akshare(["000001", "600519"], start="2021-01-01")
```

akshare 和 yfinance 只在函数调用时导入。供应商接口变化或返回空数据时，会抛出 `DataSourceError`。复权、停牌、时区和成分股偏差仍由使用者处理。

### 网络、渗流和回测

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

`factor[t]` 必须是 t 时刻已知的值，`forward_returns[t]` 是随后一个持有期的收益。库不会自动修正错位。回测换手率口径和成本假设写在 [METHODS_cn.md](docs/METHODS_cn.md) 第 7 节。

`performance_summary()` 默认按输入总长度年化，NaN 缺失不会缩短年化时钟；可用 `periods=` 显式指定时间格数。多空组合可以亏损超过 100%，此时几何年化收益和 Calmar 为 NaN，其他风险指标仍会返回。

## C

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

C 接口只暴露 C 类型。返回状态码，错误信息是线程局部的。输出缓冲区由调用方分配，长度在每个函数声明上方注释。详细规则见 [C ABI 文档](docs/ABI.md)。

## C++

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
    std::printf("%.3f\n",
                model.score(returns.data() + 1, returns.size() - 1));
    return 0;
}
```

C++ 接口用异常报告参数和数值错误；C ABI 会捕获并转成状态码。C++ ABI 不承诺跨编译器稳定，需要稳定边界时使用 C。

## 项目结构

```text
include/rflux/         C ABI 与 C++ 公共头
src/                   C++ 实现和 C ABI 包装
python/rflux/          ctypes 绑定、数据适配和上层工具
tests/                 C、C++、Python 测试
examples/              因子、HMM 走步、网络、真实数据和 C/C++ 示例
scripts/               Markdown 链接、代码围栏和公式分隔符检查
docs/                  ABI、方法和扩展说明
```

## 文档

- 计算口径和边界条件：[中文](docs/METHODS_cn.md) · [English](docs/METHODS_en.md)
- 扩展步骤和发布检查：[中文](docs/EXTENDING_cn.md) · [English](docs/EXTENDING_en.md)
- 组合风险与状态归因：[中文](docs/APPLICATIONS_cn.md) · [English](docs/APPLICATIONS_en.md)
- C ABI 的缓冲区、所有权和兼容性规则：[docs/ABI.md](docs/ABI.md)

## 测试与 CI

```bash
make check
```

测试覆盖 C ABI 的缓冲区/错误码、指标的已知值、缺失数据下的相关矩阵、HMM 拟合与因果性、状态对齐、样本外评分、图算法、渗流端点、扩散边界和 Python 回测。`make check` 还会检查 Markdown 链接、代码围栏和 GitHub 公式分隔符。

GitHub Actions 在 Linux 和 macOS 上运行 Make、CMake/CTest 测试，并在 Linux 上实际运行因子、HMM walk-forward、网络和 C/C++ 示例。

## 限制

- HMM 使用对角高斯发射，不支持全协方差或 Student-t 发射。
- HMM 是一次性批量拟合，不是在线学习器。
- 状态对齐是标签匹配，不解决模型结构变化。
- 渗流和 SIR 是 Monte Carlo 情景工具，不是市场概率预测。
- Python 回测不处理融券、停牌、涨跌停、冲击成本和容量。
- C++ 头文件是源码接口，不是稳定 ABI。

## License

MIT，见 [LICENSE](LICENSE)。
