# C++ 核心契约：行情、预测、执行与渗流

本页说明 0.7 的 C++ 扩展层。与直接使用 Eigen 的头文件相比，有三点不同：

1. **公共边界不引入 Eigen。** rflux 已对 C ABI、pybind11 暴露连续缓冲区；
   Eigen 会带来重头文件依赖、对齐规则和 ABI/生命周期问题。核心改用
   `MarketSnapshotView`、`MatrixView`、`OrderBookView` 和标准所有权容器。
2. **预测/执行契约携带足够上下文。** Predictor 有 `fit` 和 `fitted`；
   ExecutionRequest 携带 symbols、目标/当前权重、组合价值和 timestamp，
   因为仅凭权重向量和一个匿名订单簿无法推导订单。
3. **工厂是参数化且线程安全的。** 内置因子需要 window/scale，零参数
   `unique_ptr<BaseFactor>()` creator 不够。`FactorFactory` 接收
   `FactorSpec`，拒绝重名，并在 mutex 下复制 creator 后在锁外构造。

## 跨模块结构

[`include/rflux/core/market.hpp`](../include/rflux/core/market.hpp)：

- `MarketData` / `MarketSnapshotView`：拥有与零拷贝 OHLCV 快照。
- `OrderBook` / `OrderBookView`：盘口档位。
- `MatrixView`：行主序特征矩阵。
- `PredictionResult`：预期收益 + 置信度。
- `RiskConstraints`、`Order`、`Side`、`ExecutionRequest`。

所有 view 由调用方拥有，必须覆盖算法调用生命周期；校验函数拒绝空指针、空维度和
非法盘口。

## 预测与执行接口

- `mllab::BaseForecastModel`：`fit(MatrixView, target)` +
  `predict(MatrixView, regime) -> PredictionResult`，`fitted()` 明确状态。
- `execution::BaseExecutionAlgo`：`execute(ExecutionRequest, OrderBookView)`
  返回子订单；request 是单标的，多标的父订单引擎需先拆分。

更底层的 `core::BasePredictor` 保留为窄接口的特征→目标回归器，供
`HuberRegressor` 使用，不需要伪造 regime 参数。

## 因子注册

[`include/rflux/factorlab/factory.hpp`](../include/rflux/factorlab/factory.hpp)：

```cpp
rflux::factorlab::register_builtin_factors();
rflux::factorlab::FactorSpec spec;
spec.window = 20;
auto factor = rflux::factorlab::FactorFactory::instance()
                  .create("momentum", spec);
```

`AlgorithmRegistry<Interface, Args...>` 现在覆盖所有算法族：

| 工厂 | 参数 | 内置算法 |
| --- | --- | --- |
| `FactorFactory` | `FactorSpec` | momentum、volume_reversal、vwap_deviation、amihud |
| `AllocatorFactory` | 无 | hrp |
| `PredictorFactory` | `PredictorSpec` | huber |
| `ExecutionPlannerFactory` | `ExecutionPlannerSpec` | almgren_chriss |
| `PercolationDetectorFactory` | `PercolationSpec` | site_percolation |
| `RegimeDetectorFactory` | `RegimeSpec` | gaussian_hmm |

注册线程安全；重名默认失败（除非 `replace=true`）；构造期间不持有注册表锁。

## 金融网络与渗流

[`include/rflux/percolab/detector.hpp`](../include/rflux/percolab/detector.hpp)：

- `FinancialNetwork` 使用 CSR（`row_offsets`、`column_indices`、
  `edge_weights`）和可选 symbol→index 映射，避免 5000×5000 稠密矩阵。
- `make_network(dense, n)` 从稠密邻接矩阵转换。
- `SitePercolationDetector` 用固定 seed 做随机 site percolation，并查集求
  连通分量；输出巨连通分量占比、标准差、有限簇 susceptibility、最大分量和
  重复次数。
- `estimate_threshold()` 在配置的概率网格上校准 `p_c`；未校准前
  `percolation_threshold` 为 NaN。避免假装每次 `analyze(p)` 都能在热路径里
  稳定推导临界阈值。
- `rflux::risk::PreTradeRiskControl` 可用 position、drawdown、turnover 和
  系统性渗流标志做下单前拦截。

Monte Carlo 当前是串行固定网格；OpenMP/TBB 加速和自适应阈值搜索是明确的
后续步骤。并查集单次 trial 复杂度为 O(n + m α(n))。

## C ABI 覆盖

0.7 的核心算法都有 C ABI 入口：因子、HRP 权重与叶序、Almgren-Chriss 计划、Huber 回归、site percolation 分析和阈值、下单前风控。Python 在原生扩展不可用时使用这些 C 函数，不再维护第二套 NumPy 算法实现。

## 冗余治理

- 渗流检测复用 `internal/graph.hpp` 的 `UnionFind` 与 `mix_seed`；
- HMM k-medoid 的最近中心选择提取为单一 helper；
- Huber 每轮只计算一次残差，并复用线性预测 helper；
- 因子 pybind11 绑定共享 `make_factor_context` / `compute_factor_panel`；
- SQLite cache 写入改为数组 zip，而不是 `iterrows()`。
