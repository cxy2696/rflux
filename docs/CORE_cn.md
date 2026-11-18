# C++ 核心契约：行情、预测、执行与渗流

[English](CORE_en.md) · **中文**

0.7 的 C++ 扩展层遵循三项设计规则：

1. **公共边界不引入 Eigen。** C ABI 和 pybind11 接收连续缓冲区；Eigen 会增加
   头文件依赖、对齐规则和 ABI/生命周期问题。核心使用 `MarketSnapshotView`、
   `MatrixView`、`OrderBookView` 和标准所有权容器。
2. **预测/执行契约携带上下文。** Predictor 有 `fit` 和 `fitted`；
   ExecutionRequest 携带 symbols、目标/当前权重、组合价值和 timestamp。
   仅凭权重向量和一个匿名订单簿无法推导订单。
3. **工厂接收显式参数并保证线程安全。** 内置因子需要 window/scale，零参数
   `unique_ptr<BaseFactor>()` creator 无法携带这些参数。`FactorFactory` 接收
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

更底层的 `core::BasePredictor` 保留为窄接口的特征到目标的回归器，供
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
  `edge_weights`）和可选 symbol 到 index 的映射，避免 5000×5000 稠密矩阵。
- `make_network(dense, n)` 从稠密邻接矩阵转换。
- `SitePercolationDetector` 用固定 seed 做随机 site percolation，并查集求
  连通分量；输出巨连通分量占比、标准差、有限簇 susceptibility、最大分量和
  重复次数。
- `estimate_threshold()` 在配置的概率网格上校准 `p_c`，并对第一个巨组分
  交点做线性插值；未校准前 `percolation_threshold` 为 NaN。避免假装每次
  `analyze(p)` 都能在热路径里稳定推导临界阈值。
- `rflux::risk::PreTradeRiskControl` 可用 position、drawdown、turnover 和
  系统性渗流标志做下单前拦截。

Monte Carlo 现在使用 `src/internal/parallel.hpp` 的确定性 fork-join：概率点
和重复种子按静态下标分块，每块保持原有的随机流与累加顺序，因此任意线程数下
结果逐位一致。`RFLUX_THREADS=N` 限制工作线程数，`RFLUX_THREADS=1` 强制串行。
阈值校准会对第一个网格交点做线性插值；完整的自适应二分搜索仍是后续步骤。并查集单次 trial 复杂度为
O(n + m α(n))。

## Buffer-first 数值内核

qflux 的每个内核在公开头文件里都有两个重载：`foo_into(..., double* out)`
精确写入 `n` 个值且不分配内存；`foo(...)` 分配结果向量并转发给 buffer 版本。
C ABI 和 Python 桥直接调用 buffer 版本，因此调用过程中不会产生临时结果向量：
C 端直接写入调用方数组，pybind11 直接写入 NumPy 输出（`series_into` 因此是
零拷贝）。`out` 不得与输入别名，且必须能容纳 `n` 个值；参数在第一次
写入之前完成校验。

`rolling_quantile` 根据窗口长度选择顺序统计结构：窗口不超过 1024 时维护一个
有序缓冲区（二分查找加分块搬移，不需要全局坐标压缩）；更长的窗口仍使用压缩
坐标上的 Fenwick 树。两条路径给出相同的 Type-7 分位数。

## 确定性并行

`detail::parallel_for(count, fn, max_threads)` 是核心中唯一的线程原语。它按
静态下标划分 `[0, count)`，不依赖调度状态，并重新抛出第一个工作线程异常。
它用于渗流扫描（按概率点并行）、簇大小采样（每线程私有计数，再按 worker
顺序归约）、HMM Baum-Welch 重启（按 `n_init` 并行，最多 4 个并发重启以控制
内存）以及相关矩阵（完整面板和 pairwise 路径都按资产并行）。小规模输入会
跳过 fork-join，因为线程创建比计算本身更贵；嵌套调用自动退化为串行，避免
过度订阅 CPU。

## 构建优化

CMake 在编译器支持时为 core、C ABI、原生扩展、测试和示例目标启用 IPO/LTO。
Release 和 RelWithDebInfo 默认开启，Debug 和 sanitizer 构建自动跳过。需要对比
构建或缩短链接时间时，传 `-DRFLUX_ENABLE_IPO=OFF`。

## C ABI 覆盖

0.7 的核心算法都有 C ABI 入口：因子、HRP 权重与叶序、Almgren-Chriss 计划、Huber 回归、site percolation 分析和阈值、下单前风控。Python 在原生扩展不可用时使用这些 C 函数，不再维护第二套 NumPy 算法实现。

## 共享代码

- 渗流检测复用 `internal/graph.hpp` 的 `UnionFind` 与 `mix_seed`；
- HMM k-medoid 的最近中心选择提取为单一 helper；
- Huber 每轮只计算一次残差，并复用线性预测 helper；
- 因子 pybind11 绑定共享 `make_factor_context` / `compute_factor_panel`；
- SQLite cache 写入改为数组 zip，而不是 `iterrows()`；
- `SlidingBlocks<Ops>` 是唯一的两栈可合并队列，`Moments1Ops` /
  `Moments2Ops` 只提供各自的一元/二元合并操作；
- `for_each_complete_window` 统一管理所有滚动内核的预热、过期和缺失值规则，
  `rolling_quantile` 的两条后端路径都复用它；
- 滚动相关、协方差和 beta 合并为一个 `rolling_pair_stat_into` 实现（带统计量
  选择器）；Sharpe 与 Sortino 共用 `performance_ratio`；
- C ABI 的 HMM 入口共享 `guarded_model_vector` / `guarded_model_i32`，C++ 模型
  只有一个 `validate_inference` / `require_fit_diagnostics` 守卫；Python 桥
  只有一套 `run_released`、`filled_array`、`filled_triple` 和 `array_from`；
- Python 层通过 `native_or` / `native_call` 和
  `GaussianHMM._native_or_abi` 选择后端；报告元数据、CLI 报告收尾、渗流参数
  编组和概率矩阵读取各自只有一个 helper；
- `jscpd` 在 `src/`、`include/`、`python/rflux/` 和 `tests/` 上，默认
  5 行/50 token 与更严格的 5 行/40 token 阈值下都报告 0 个完全克隆；中英双语
  文档镜像是有意保留的唯一重复块。
