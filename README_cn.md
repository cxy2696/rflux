# rflux

[![CI](https://github.com/cxy2696/rflux/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/cxy2696/rflux/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

[English](README.md) · **中文**

rflux 是一个 C++17 + Python 量化研究库。计算核心用 C++ 实现，通过稳定的 C ABI
和 pybind11 扩展供 Python 调用；数据适配、回测和命令行工具放在 Python 层。

当前版本 `0.7.0`，C ABI 版本 `1`。

## 安装

从源码安装：

```bash
git clone https://github.com/cxy2696/rflux.git
cd rflux
python -m pip install .
```

需要 CMake 3.26+、Ninja、C++17 编译器和 Python 3.10+。运行时依赖 NumPy。
数据适配需要 pandas 和一个数据源 SDK：

```bash
python -m pip install ".[all]"   # pandas + AKShare + yfinance
```

项目不发布到 PyPI。安装本地 wheel：

```bash
python -m pip install dist/rflux-*.whl
```

CPython 3.12+ 使用 abi3 wheel；3.10/3.11 使用版本相关 wheel。

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

`rflux advice` 生成透明、只做多的研究再平衡提案：HRP 基础权重、20/60 日动量、
趋势与 RSI 调整、波动率目标、可选 HMM 风险状态、2 倍日标准差参考止损和
仓位计算。它不是个性化投资建议，也不承诺收益。

## 功能

- **qflux**：收益率、均线、RSI、MACD、布林带、ATR、回撤、滚动统计、分位数、
  协方差、beta、排序、缩尾、中性化、相关矩阵、Sharpe/Sortino、历史 VaR/CVaR。
- **regimelab**：高斯 HMM 拟合、filtered/smoothed 概率、Viterbi、状态预测、
  状态对齐、波动率/趋势 regime、sticky regime。
- **percolab**：相关网络、连通分量、中心性、PageRank、聚类、k-core、割边/割点、
  k-truss、MST、bond/site 渗流、簇大小分布和风险扩散。
- **算法层**：动量、量价反转、VWAP、Amihud、HRP/NCO 风格配置、Huber 回归、
  图平滑、Almgren-Chriss 执行和盘口指标。
- **数据层**：本地 CSV、AKShare、yfinance 统一为长表 OHLCV，包含代码归一、
  校验、SQLite 缓存和重试。
- **C ABI**：因子、HRP、Almgren-Chriss、Huber、渗流和下单前风控均可从 C 调用，
  也可作为 Python ctypes 回退。

## 数据示例

```python
from rflux import data

loader = data.DataLoaderFactory.create("csv", root="./data/daily")
frame = loader.load(["AAPL.US", "MSFT.US"], start="2024-01-01", end="2024-12-31")
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

设置 `RFLUX_DISABLE_NATIVE=1` 可强制使用 ctypes C ABI。

## 文档

- [English README](README.md)
- [C ABI](docs/ABI_cn.md)：所有权、状态码和兼容性
- [数据层](docs/DATA_cn.md) · [English](docs/DATA_en.md)
- [算法矩阵](docs/ALGORITHMS_cn.md) · [English](docs/ALGORITHMS_en.md)
- [C++ 核心契约](docs/CORE_cn.md) · [English](docs/CORE_en.md)
- [原生桥](docs/NATIVE_cn.md) · [English](docs/NATIVE_en.md)
- [方法与公式](docs/METHODS_cn.md) · [English](docs/METHODS_en.md)
- [应用](docs/APPLICATIONS_cn.md) · [English](docs/APPLICATIONS_en.md)
- [扩展指南](docs/EXTENDING_cn.md) · [English](docs/EXTENDING_en.md)
- [示例](examples/README_cn.md)、[更新记录](CHANGELOG_cn.md)、[发布清单](docs/RELEASE_cn.md)

## 许可

MIT，见 [LICENSE](LICENSE)。
