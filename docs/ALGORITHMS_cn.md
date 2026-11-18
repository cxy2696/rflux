# 算法矩阵与扩展契约

[English](ALGORITHMS_en.md) · **中文**

rflux 0.7 在数值核心上增加 C++ 算法层。每个算法都有类型化接口、输入输出契约和原生实现；Python 层只做输入校验和报告组装，不计算算法结果。

## 扩展契约

底层行情 view、预测/执行接口、注册表和渗流契约见 [CORE_cn.md](CORE_cn.md)。

公开契约位于
[`include/rflux/core/algorithm.hpp`](../include/rflux/core/algorithm.hpp)：

| 接口 | 职责 |
| --- | --- |
| `BaseAlgorithm` | 稳定名称/版本与虚析构 |
| `BaseFactor` | 将 `FactorContext` 转换为行主序因子面板 |
| `BaseRegimeDetector` | 拟合状态模型并输出状态/概率 |
| `BaseAllocator` | 将协方差矩阵映射为 long-only 权重 |
| `BasePredictor` | 拟合并预测特征矩阵 |
| `BaseExecutionPlanner` | 将订单映射为确定性执行计划 |

所有输入输出都是调用方拥有的连续缓冲区。C++ 核心不解析文件、不加载行情、不依赖 Python；TOML/JSON/YAML、数据源和语言绑定都留在组合层。

## 模块一：因子工程

四个因子都继承 `PanelFactor`，由基类负责面板校验和 asset/time 遍历；因子本身只提供
单格计算和 trailing-return helper。新增因子不再重复输出分配和预热处理。

C++ 实现位于 [`include/rflux/factorlab/factors.hpp`](../include/rflux/factorlab/factors.hpp)：

| 因子 | 定义 | 预热期 |
| --- | --- | --- |
| `MomentumFactor(window)` | `close[t] / close[t-window] - 1` | `t >= window` |
| `VolumeReversalFactor(window)` | `-ret(window) * volume[t] / mean(volume)` | `t >= window` |
| `VwapDeviationFactor(window)` | `(close[t] - VWAP_window) / VWAP_window` | `t >= window-1` |
| `AmihudFactor(window, scale)` | `scale * mean(abs(ret) / dollar_volume)` | `t >= window` |

面板形状为 `(n_obs, n_assets)`；缺失或预热单元为 NaN。量价因子要求成交量非负。

Python 入口 `rflux.factorlab` 提供 `momentum`、`volume_reversal`、`vwap_deviation`、`amihud`、`information_coefficient`（Pearson 或 rank IC）和 `factor_turnover`。

**后续工作：** 分析师预期修正、PE/PB/ROE 基本面、LLM 因子假设和进化式公式搜索需要数据适配。接口已经就绪，数据契约不随核心打包。

## 模块二：状态识别与尾部风险

- 现有对角高斯 HMM：filtered/smoothed、Viterbi、状态预测、状态对齐、滚动评分；`fit_hmm_snapshot` 统一一次性拟合并关闭有状态模型，返回脱离句柄的最新滤波/预测数组供报告使用。
- `rflux.regimelab.sticky_regime`：在 HMM filtered 概率上增加滞回和最短驻留时间。挑战状态必须达到 entry 概率，现任状态必须跌破 exit 概率，且已持续 `min_duration` 个 bar 才切换，降低单 bar 抖动带来的换手。
- `rflux.risk.tail_dependence`：经验下尾/上尾两两尾部相依系数，捕捉 Pearson 相关之外的极端联动；每个资产对取 `P(i|j)` 与 `P(j|i)` 两个有限样本条件概率的平均，保证矩阵对称。
- 现有 `percolab` 网络/渗流工具覆盖风险传染路径。

C++ 侧 `HmmRegimeDetector` 将现有 HMM 适配到 `BaseRegimeDetector`。

## 模块三：组合构建

`rflux.portfoliolab` 提供：

- `hrp_weights`：相关距离、单链接层次树、准对角排序、递归二分；不需求逆协方差，long-only、权重和为 1。
- `nco_weights`：嵌套聚类风险平价。按相关距离分簇，簇内 HRP，簇间按簇方差的倒数分配预算。完整带约束凸优化的 NCO 属于后续工作。
- `risk_parity_weights`：对角逆方差基线。
- `rebalance_targets`：按每个组合（2D 输入按行）独立门控的漂移阈值调仓，与 sticky regime 配合可减少不必要的换手。

## 模块四：鲁棒机器学习

`rflux.mllab` 提供：

- `huber_regression`：IRLS + MAD 鲁棒尺度。C++ 实现位于
  [`include/rflux/mllab/huber.hpp`](../include/rflux/mllab/huber.hpp)，通过原生桥暴露；NumPy 回退。
- `cvar_loss`：历史 CVaR（损失为正），可作为模型选择或组合目标诊断。
- `graph_smooth`：度归一化图信号扩散，是轻量图正则原语，不是已训练 GNN。

**后续工作：** 时序 GNN、图注意力、产业链消息传递和 CVaR 正则训练应放在 `BasePredictor` 之后。

## 模块五：执行与微观结构

`rflux.execution` 提供：

- `almgren_chriss`：线性临时/永久冲击、风险厌恶、指定切片数的确定性最优清算轨迹；返回交易序列、期望成本、成本方差和 `kappa`。
- `order_book_metrics`：基于调用方提供的一档盘口快照计算 spread、mid、microprice 和 sizes imbalance。

**后续工作：** 队列位置、最小变动单位约束、暗池、venue routing 和量子定价依赖交易场所或硬件，本版本不包含。

## 可追溯性与验证

每个公开算法族都有唯一的实现入口、可选的稳定 C ABI、Python 包装和具名回归测试。下表列出公式、代码与测试的对应关系：

| 算法族 | C++ 实现 | C ABI | Python API | 主要测试 |
| --- | --- | --- | --- | --- |
| 序列与滚动窗口 | [`src/qflux/window.cpp`](../src/qflux/window.cpp) | `rflux_rolling_*`、`rflux_sma`、`rflux_ema` | `qflux.rolling_*`、`qflux.sma`、`qflux.ema` | [`test_qflux.py`](../tests/python/test_qflux.py)、[`test_core_gtest.cpp`](../tests/cpp/test_core_gtest.cpp) |
| 指标 | [`src/qflux/indicators.cpp`](../src/qflux/indicators.cpp) | `rflux_returns`、`rflux_rsi`、`rflux_atr`、`rflux_macd`、`rflux_bollinger`、`rflux_drawdown` | `qflux.returns`、`qflux.rsi`、`qflux.atr`、`qflux.macd`、`qflux.bollinger`、`qflux.drawdown` | [`test_qflux.py`](../tests/python/test_qflux.py)、[`test_core_gtest.cpp`](../tests/cpp/test_core_gtest.cpp) |
| 统计与排序 | [`src/qflux/stats.cpp`](../src/qflux/stats.cpp)、[`ranking.cpp`](../src/qflux/ranking.cpp)、[`performance.cpp`](../src/qflux/performance.cpp) | `rflux_describe`、`rflux_quantile`、`rflux_pearson`、`rflux_sharpe`、`rflux_sortino`、`rflux_var_cvar`、`rflux_rank`、`rflux_zscore`、`rflux_winsorize`、`rflux_neutralize`、`rflux_equity_curve`、`rflux_performance_summary` | `qflux.describe`、`quantile`、`pearson`、`sharpe`、`sortino`、`var_cvar`、`rank`、`zscore`、`winsorize`、`neutralize` | [`test_qflux.py`](../tests/python/test_qflux.py)、[`test_core_gtest.cpp`](../tests/cpp/test_core_gtest.cpp) |
| 高斯 HMM | [`src/regimelab/hmm.cpp`](../src/regimelab/hmm.cpp)、[`src/regimelab/alignment.cpp`](../src/regimelab/alignment.cpp) | `rflux_hmm_*`、`rflux_regime_match_states` | `regimelab.GaussianHMM`、`fit_hmm`、`fit_hmm_snapshot`、`match_states`、`align_parameters` | [`test_hmm_regime.py`](../tests/python/test_hmm_regime.py)、[`test_native.py`](../tests/python/test_native.py)、[`test_core_gtest.cpp`](../tests/cpp/test_core_gtest.cpp) |
| 规则状态 | [`src/regimelab/regime.cpp`](../src/regimelab/regime.cpp) | `rflux_volatility_regime`、`rflux_trend_regime`、`rflux_composite_regime` | `regimelab.volatility_regime`、`trend_regime`、`composite_regime`、`sticky_regime` | [`test_hmm_regime.py`](../tests/python/test_hmm_regime.py) |
| 网络指标 | [`src/percolab/network.cpp`](../src/percolab/network.cpp) | `rflux_network_*`、`rflux_pagerank`、`rflux_corr_to_distance`、`rflux_mst_clusters` | `percolab.centrality`、`pagerank`、`betweenness`、`clustering`、`path_statistics`、`core_numbers`、`truss_numbers`、`mst_*` | [`test_percolab.py`](../tests/python/test_percolab.py)、[`test_core_gtest.cpp`](../tests/cpp/test_core_gtest.cpp) |
| 渗流与传染 | [`src/percolab/percolation.cpp`](../src/percolab/percolation.cpp)、[`detector.cpp`](../src/percolab/detector.cpp) | `rflux_*percolation*`、`rflux_targeted_removal`、`rflux_risk_diffusion` | `percolab.site_percolation`、`bond_percolation`、`percolation_scan`、`cluster_size_distribution`、`targeted_removal`、`core_percolation`、`risk_diffusion` | [`test_percolab.py`](../tests/python/test_percolab.py)、[`test_core_gtest.cpp`](../tests/cpp/test_core_gtest.cpp) |
| 因子与回测 | [`src/factorlab/factors.cpp`](../src/factorlab/factors.cpp)、[`src/factorlab/evaluation.cpp`](../src/factorlab/evaluation.cpp)、[`src/factorlab/backtest.cpp`](../src/factorlab/backtest.cpp) | `rflux_factor_*`、`rflux_factor_ic`、`rflux_factor_turnover`、`rflux_factor_forward_returns`、`rflux_factor_quantile_returns`、`rflux_factor_long_short`、`rflux_factor_ic_summary` | `factorlab.momentum`、`volume_reversal`、`vwap_deviation`、`amihud`、`information_coefficient`、`factor_turnover`、`backtest.*` | [`test_algorithms.py`](../tests/python/test_algorithms.py)、[`test_tools.py`](../tests/python/test_tools.py)、[`test_core_gtest.cpp`](../tests/cpp/test_core_gtest.cpp) |
| 组合构建 | [`src/portfoliolab/hrp.cpp`](../src/portfoliolab/hrp.cpp)、[`src/portfoliolab/risk.cpp`](../src/portfoliolab/risk.cpp) | `rflux_hrp_weights`、`rflux_hrp_order`、`rflux_risk_parity_weights`、`rflux_nco_weights`、`rflux_rebalance_targets`、`rflux_portfolio_risk` | `portfoliolab.hrp_weights`、`nco_weights`、`risk_parity_weights`、`rebalance_targets` | [`test_algorithms.py`](../tests/python/test_algorithms.py)、[`test_core_gtest.cpp`](../tests/cpp/test_core_gtest.cpp) |
| 鲁棒机器学习 | [`src/mllab/huber.cpp`](../src/mllab/huber.cpp)、[`src/mllab/robust.cpp`](../src/mllab/robust.cpp) | `rflux_huber_regression`、`rflux_cvar_loss`、`rflux_graph_smooth` | `mllab.huber_regression`、`cvar_loss`、`graph_smooth` | [`test_algorithms.py`](../tests/python/test_algorithms.py)、[`test_core_gtest.cpp`](../tests/cpp/test_core_gtest.cpp) |
| 执行与风险 | [`src/execution/almgren_chriss.cpp`](../src/execution/almgren_chriss.cpp)、[`src/execution/microstructure.cpp`](../src/execution/microstructure.cpp)、[`src/risk/pretrade.cpp`](../src/risk/pretrade.cpp)、[`src/risk/tail_dependence.cpp`](../src/risk/tail_dependence.cpp) | `rflux_almgren_chriss_plan`、`rflux_order_book_metrics`、`rflux_pretrade_check`、`rflux_tail_dependence` | `execution.almgren_chriss`、`execution.order_book_metrics`、`risk.tail_dependence` | [`test_algorithms.py`](../tests/python/test_algorithms.py)、[`test_core_gtest.cpp`](../tests/cpp/test_core_gtest.cpp) |
| 建议与归因 | [`src/portfoliolab/advice.cpp`](../src/portfoliolab/advice.cpp)、[`src/regimelab/attribution.cpp`](../src/regimelab/attribution.cpp) | — | `trade_advice`、`regime_attribution` | [`test_advice.py`](../tests/python/test_advice.py)、[`test_apps.py`](../tests/python/test_apps.py) |
| 研究报告 | C++ 内核，Python 组装（`analysis.py`、`advice.py`、`apps.py`） | — | `analyze_prices`、`trade_advice`、`portfolio_risk`、`regime_attribution`、`performance_summary`、`backtest.*` | [`test_advice.py`](../tests/python/test_advice.py)、[`test_tools.py`](../tests/python/test_tools.py)、[`test_apps.py`](../tests/python/test_apps.py) |

`scripts/verify_algorithms.py` 是可执行检查：遍历所有公开算法族，检查命名、形状、预热和范围，在可独立对照处给出参考值，记录热路径耗时，并把带时间戳的 JSON 报告写入 `artifacts/validation/`。`make algo-check` 通过原生扩展运行一次。新增公开算法族时，表格和测试必须一起更新。

## 新算法工程规则

1. 数学实现放在 `src/` 的 C++ 中，并在 `include/rflux/` 暴露契约。
2. 增加 GoogleTest：解析解或独立暴力参考；覆盖 NaN、零成交量、零方差、空面板。
3. 增加 pybind11 原生绑定，耗时部分释放 GIL，输入输出使用连续 float64 数组。
4. Python 包装只做形状校验并返回原生结果；不增加 NumPy 回退或 Python 计算路径。
5. 如果进入跨语言契约，C ABI 只能追加，不能修改现有布局；C++ 核心不允许文件 I/O 或 Python 类型。
6. 文档明确预热、因果性、缩放和估计量定义。
