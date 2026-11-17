# rflux

[![CI](https://github.com/cxy2696/rflux/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/cxy2696/rflux/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

rflux 是一个 C++17 + Python 量化算法库。核心计算用 C++ 实现，通过稳定的 C ABI 和 pybind11 扩展供 Python 调用；数据适配、回测和命令行工具放在 Python 层。

当前版本 `0.7.0`，C ABI 版本为 `1`。中文详细说明见 [README_cn.md](README_cn.md)，英文说明见 [README_en.md](README_en.md)。

## 安装

从源码安装：

```bash
git clone https://github.com/cxy2696/rflux.git
cd rflux
python -m pip install .
```

需要 CMake 3.26+、Ninja、C++17 编译器。Python 3.10+，运行时依赖 NumPy。

本项目不发布到 PyPI。可安装本地构建的 wheel：

```bash
pip install dist/rflux-*.whl
pip install "rflux[all]"    # 可选：pandas 与 AKShare/yfinance
```

CPython 3.12+ 使用 abi3 wheel；3.10/3.11 使用对应版本的 wheel。因子、HRP、Almgren-Chriss、Huber、渗流检测和下单前风控都有 C ABI 入口。

## 快速开始

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

命令行：

```bash
rflux doctor
rflux analyze examples/data/sample_prices.csv --no-hmm
rflux risk examples/data/sample_prices.csv --weights 0.2,0.2,0.2,0.2,0.2
rflux advice examples/data/sample_prices.csv --no-hmm
```

## 功能

- **qflux**：收益率、均线、RSI、MACD、布林带、ATR、回撤、滚动统计、分位数、协方差、beta、排序、缩尾、中性化、相关矩阵、Sharpe/Sortino、VaR/CVaR。
- **regimelab**：高斯 HMM 拟合、filtered/smoothed 概率、Viterbi、状态预测、状态对齐、波动率/趋势 regime、sticky regime。
- **percolab**：相关网络、连通分量、中心性、PageRank、聚类、k-core、割边/割点、k-truss、MST、bond/site 渗流、风险扩散。
- **算法层**：动量、量价反转、VWAP、Amihud、HRP、NCO 风格配置、Huber 回归、图平滑、Almgren-Chriss 执行。
- **数据层**：本地 CSV、AKShare、yfinance 统一为长表 OHLCV；包含代码归一、校验、SQLite 缓存和重试。使用前安装 `rflux[all]` 或对应 provider extra。
- **研究建议**：`rflux advice` 把动量、趋势、RSI、HRP、波动率目标和 HMM 风险状态组合成透明的多头再平衡提案，输出每项资产的目标权重、变动理由、换手率和风险估计。

## 数据示例

```python
from rflux import data

loader = data.DataLoaderFactory.create("csv", root="./data/daily")
frame = loader.load(["600519.SH", "AAPL.US"], start="2024-01-01", end="2024-12-31")
close = data.pivot_ohlcv(frame, "close")
```

## 开发

```bash
python -m pip install -e ".[test]"
python -m pytest tests/python -q

cmake --preset dev
cmake --build --preset dev
ctest --preset dev --output-on-failure
```

`RFLUX_DISABLE_NATIVE=1` 可以强制使用 ctypes C ABI。

## 文档

- [中文说明](README_cn.md) / [English guide](README_en.md)
- [统一数据层](docs/DATA_cn.md) · [算法矩阵](docs/ALGORITHMS_cn.md) · [C++ 核心契约](docs/CORE_cn.md) · [原生桥](docs/NATIVE_cn.md)
- [方法说明](docs/METHODS_cn.md) · [C ABI](docs/ABI.md) · [扩展示例](docs/EXTENDING_cn.md)
- [示例](examples/README.md) · [变更记录](CHANGELOG.md) · [发布清单](docs/RELEASE.md)

## 许可

MIT，见 [LICENSE](LICENSE)。
