# rflux 使用案例

这些示例都可以从仓库根目录直接运行。Python 示例只依赖 NumPy；`python_real_data.py` 还需要 pandas 和一个可选数据源。计算口径见 [METHODS_cn.md](../docs/METHODS_cn.md)，英文版见 [METHODS_en.md](../docs/METHODS_en.md)。

```bash
make -j
make python-lib            # 可选：把共享库放进 python/rflux/_lib/
export PYTHONPATH="$PWD/python"
```

在仓库内运行时，Python 加载器会自动查找 `build/librflux.dylib` 或 `build/librflux.so`；也可以显式设置 `RFLUX_LIBRARY`。`make python-lib` 只在需要把包安装到别处时使用。

## 0. 命令行直接分析

```bash
PYTHONPATH=python python -m rflux analyze examples/data/sample_prices.csv
PYTHONPATH=python python -m rflux analyze examples/data/sample_prices.csv --format json
```

`analyze` 读取宽表 CSV，自动识别日期列，计算资产统计、等权市场、相关网络和 3 状态 HMM。`--no-hmm`、`--no-network` 可以关掉较慢的部分；`--format json` 适合接入脚本。安装为包之后命令简写为 `rflux ...`。

Python 里可以直接调用同一套分析：

```python
from rflux import analyze_prices

report = analyze_prices(prices, names=["AAA", "BBB"])
```

## 1. 选股因子截面筛选

```bash
python examples/python_factor_screen.py
```

演示一条完整的选股流水线：合成多资产价格、计算 20 日动量/低波动/短期反转、缩尾、横截面 z-score、行业中性化、合成因子、计算 rank IC、分位数组合和带成本多空回测。

关键点：因子在 t 日收盘后计算，只与 t → t+1 的收益对齐，不读取未来价格。这个例子使用合成数据，输出的正负收益都不是投资结论。

可替换部分在 `simulate_prices()`：接入真实数据时，把返回的 `prices` 换成宽表价格矩阵即可。例如：

```python
from rflux import data

close = data.load_yfinance(
    ["AAPL", "MSFT", "NVDA", "AMZN"], start="2021-01-01"
)
prices = close.ffill().to_numpy()
```

## 2. HMM 市场状态识别与跨窗口对齐

```bash
python examples/python_hmm_walkforward.py
```

按滚动窗口重新拟合 3 状态高斯 HMM，对下一块数据计算样本外对数似然，并用 Hungarian 状态匹配把每次新拟合的标签对齐到参考模型。输出包括每块 OOS log-likelihood、对齐顺序、最终状态分布和条件均值。

这个例子适合回答两个问题：

- 模型在未见数据上的似然是否稳定；
- 重拟合后状态标签是否还能对应上。

## 3. 相关网络、渗流与风险扩散

```bash
python examples/python_end_to_end.py
```

从合成市场面板开始，计算特征、HMM 状态、最近 120 日相关网络、连通分量、特征向量中心性、bond percolation 扫描和 SIR 风险扩散，最后跑一个分位数多空回测。

如果需要把网络部分单独接进研究流程，核心代码可以压缩成：

```python
from rflux import percolab

network = percolab.correlation_network(returns_panel, threshold=0.45)
print(network.components.largest_size)
print(network.eigenvector)

scan = percolab.bond_percolation(
    network.adjacency, [0.0, 0.25, 0.5, 0.75, 1.0],
    repetitions=300, seed=42
)
print(scan.giant_mean, scan.susceptibility)
```

## 4. 真实数据接入

```bash
pip install pandas yfinance
python examples/python_real_data.py \
  --source yfinance --symbols AAPL MSFT NVDA --start 2021-01-01
```

A 股示例：

```bash
pip install pandas akshare
python examples/python_real_data.py \
  --source akshare --symbols 000001 600519 --start 2021-01-01
```

适配器只负责下载、日期索引和列名归一化。复权、停牌、点时成分股和生存偏差由使用者处理。远程供应商接口可能变化；脚本把空响应和 schema 错误转成 `DataSourceError`。

## 5. 组合风险与状态归因

```bash
python examples/python_applications.py
PYTHONPATH=python python -m rflux risk examples/data/sample_prices.csv \
  --weights 0.30,0.25,0.20,0.15,0.10
```

`python_applications.py` 先计算带权组合的年化波动和 Euler 风险贡献，再用因果波动率 regime 给等权市场收益打标签，输出各状态的收益、波动、占比和转移矩阵。方法、缺失值处理和解释边界见 [APPLICATIONS_cn.md](../docs/APPLICATIONS_cn.md)。

## 6. C 和 C++ 集成

```bash
make examples
./build/example_c_basic
./build/example_cpp_research
```

- `c_basic.c`：只用 C ABI 计算 SMA/RSI 并构建阈值图。
- `cpp_research.cpp`：直接用 C++ 核心完成特征、HMM、网络、渗流和扩散。

C 用户只需要 `include/rflux/rflux.h`；C++ 用户可包含 `include/rflux/rflux.hpp` 获取全部公共头。

## 文件索引

| 文件 | 场景 | 主要 API |
| --- | --- | --- |
| [data/sample_prices.csv](data/sample_prices.csv) | 命令行和直接分析的示例价格表 | CLI 示例输入 |
| [python_factor_screen.py](python_factor_screen.py) | 选股因子、行业中性和多空回测 | `qflux.winsorize/zscore/neutralize`, `backtest.factor_ic/long_short_backtest` |
| [python_hmm_walkforward.py](python_hmm_walkforward.py) | HMM 滚动拟合、OOS 评分、状态对齐 | `regimelab.fit_hmm/score/match_states/align_labels` |
| [python_end_to_end.py](python_end_to_end.py) | 状态 + 网络 + 渗流 + 回测 | `regimelab`, `percolab`, `backtest` |
| [python_real_data.py](python_real_data.py) | akshare/yfinance 真实数据 | `data.load_yfinance/load_akshare/to_returns` |
| [python_applications.py](python_applications.py) | 组合风险贡献和 regime 归因 | `portfolio_risk`, `regime_attribution` |
| [c_basic.c](c_basic.c) | C ABI 最小调用 | `rflux_sma`, `rflux_rsi`, `rflux_graph_components` |
| [cpp_research.cpp](cpp_research.cpp) | C++ 研究管线 | `rflux::qflux`, `rflux::regimelab`, `rflux::percolab` |

所有示例都使用显式随机种子。合成数据上的收益数字只用于验证代码路径，不代表任何市场结论。
