# 算法矩阵与扩展契约

rflux 0.7 在数值核心之上增加了 C++ 算法层。目标不是继续堆砌指标，而是让每个算法都有明确接口、输入输出契约、原生快速路径，以及在可行时提供研究层回退。

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

**后续层：** 分析师预期修正、PE/PB/ROE 基本面、LLM 因子假设和进化式公式搜索属于数据/适配问题。接口已经准备好，但数据契约不随核心打包。

## 模块二：状态识别与尾部风险

- 现有对角高斯 HMM：filtered/smoothed、Viterbi、状态预测、状态对齐、滚动评分。
- `rflux.regimelab.sticky_regime`：在 HMM filtered 概率上增加滞回和最短驻留时间。挑战状态必须达到 entry 概率，现任状态必须跌破 exit 概率，且已持续 `min_duration` 个 bar 才切换，降低单 bar 抖动带来的换手。
- `rflux.risk.tail_dependence`：经验下尾/上尾两两尾部相依系数，捕捉 Pearson 相关之外的极端联动。
- 现有 `percolab` 网络/渗流工具覆盖风险传染路径。

C++ 侧 `HmmRegimeDetector` 将现有 HMM 适配到 `BaseRegimeDetector`。

## 模块三：组合构建

`rflux.portfoliolab` 提供：

- `hrp_weights`：相关距离、单链接层次树、准对角排序、递归二分；不需求逆协方差，long-only、权重和为 1。
- `nco_weights`：嵌套聚类风险平价。按相关距离分簇，簇内 HRP，簇间按簇方差的倒数分配预算。这是 NCO 风格的聚类层；完整带约束凸优化的 NCO 是明确的后续步骤。
- `risk_parity_weights`：对角逆方差基线。
- `rebalance_targets`：带漂移阈值的调仓，与 sticky regime 配合可减少不必要的换手。

## 模块四：鲁棒机器学习

`rflux.mllab` 提供：

- `huber_regression`：IRLS + MAD 鲁棒尺度。C++ 实现位于
  [`include/rflux/mllab/huber.hpp`](../include/rflux/mllab/huber.hpp)，通过原生桥暴露；NumPy 回退。
- `cvar_loss`：历史 CVaR（损失为正），可作为模型选择或组合目标诊断。
- `graph_smooth`：度归一化图信号扩散，是轻量图正则原语，不是已训练 GNN。

**后续层：** 时序 GNN、图注意力、产业链消息传递、CVaR 正则训练应放在 `BasePredictor` 之后。

## 模块五：执行与微观结构

`rflux.execution` 提供：

- `almgren_chriss`：线性临时/永久冲击、风险厌恶、指定切片数的确定性最优清算轨迹；返回交易序列、期望成本、成本方差和 `kappa`。
- `order_book_metrics`：基于调用方提供的一档盘口快照计算 spread、mid、microprice 和 sizes imbalance。

**后续层：** 队列位置、最小变动单位约束、暗池、venue routing 和 QML/量子定价依赖具体交易场所或硬件，本版本不包含。

## 新算法工程规则

1. 数学实现放在 `src/` 的 C++ 中，并在 `include/rflux/` 暴露契约。
2. 增加 GoogleTest：解析解或独立暴力参考；覆盖 NaN、零成交量、零方差、空面板。
3. 增加 pybind11 原生绑定，耗时部分释放 GIL，输入输出使用连续 float64 数组。
4. 如果进入公开 Python API，提供 NumPy 回退，并测试 native/fallback 一致性。
5. 如果进入跨语言契约，C ABI 只能追加，不能修改现有布局；C++ 核心不允许文件 I/O 或 Python 类型。
6. 文档明确预热、因果性、缩放和估计量定义。
