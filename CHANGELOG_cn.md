# Changelog

## 0.8.0 — 2026-11-19

原生核心的性能与架构优化；C ABI 仍为 `1`，既有签名、布局和结果语义不变。
`0.8.0` 新增可选 C 入口，按 ABI 规则属于 minor 版本。

- 新增 C++ 回测内核（`src/factorlab/backtest.cpp`）：前瞻简单/对数收益、
  与 `np.array_split` 一致的分位组合、带成本等权多空组合，以及逐期 IC 汇总。
  C ABI 新增 `rflux_factor_forward_returns`、`rflux_factor_quantile_returns`、
  `rflux_factor_long_short`、`rflux_factor_ic_summary`，pybind11 暴露同名函数。
  `python/rflux/backtest.py` 只做形状校验并调用内核，不再用 NumPy 循环计算。
- HMM 状态匹配从 Python 移到 C++（`src/regimelab/alignment.cpp`）。C ABI
  新增 `rflux_regime_match_states`，pybind11 暴露 `regime_match_states`，使用
  O(n^3) Hungarian 算法；`regimelab.match_states` 与 `align_parameters` 改为调用
  该实现，返回契约不变。
- CI 与 wheel 工作流只在 `ubuntu-latest` 上运行。移除 macOS runner 矩阵、
  Windows wheel 矩阵和示例步骤中的平台条件；仍保留 C ABI 库、GoogleTest、
  sanitizer 构建和 pybind11 扩展的检查。
- 删除未使用的 `SharedArray` 共享内存工具及其测试，包顶层不再导出。
- `rflux.native` 改为自动镜像扩展导出，不再手工维护与 `bindings.cpp`
  同步的第二份清单。

- qflux 内核新增 buffer-first 的 `*_into` 重载。C ABI 直接写入调用方数组，
  pybind11 直接写入 NumPy 结果，调用不再分配临时 `std::vector` 再拷贝
  （`series_into` 因此是零拷贝）。
- `rolling_quantile` 在有序窗口（不超过 1024）和 Fenwick 树（更长窗口）之间
  选择；短窗口不再对所有坐标做全局排序，在常见因子场景下快数倍。
- 新增 `internal/parallel.hpp` 确定性静态分块 fork-join：支持异常传播、嵌套
  调用退化为串行，并可用 `RFLUX_THREADS` 限制线程数。它并行化渗流扫描、
  簇大小采样、HMM 重启和完整面板相关矩阵，同时保证任意线程数下结果一致。
- HMM 发射分布的对数归一化常数和方差倒数每次 M 步只计算一次，而不是逐观测
  重复；Baum-Welch 重启并行执行（最多 4 个并发以控制内存）。
- 滑动矩的单元追加使用专用 merge（每个观测一次倒数而非两次除法），通用块
  合并也复用同一个倒数。
- 标量分位数、历史 VaR/CVaR 和缩尾改用 `std::nth_element` 选取顺序统计量，
  不再对样本整体排序（100 万数值约快 2.6–2.7 倍，Type-7 结果不变）。
- 最小生成树由 Kruskal 改为稠密 Prim：不再构造并排序全部 `n(n-1)/2` 条候选
  边，时间复杂度从 `O(n^2 log n)` 降到 `O(n^2)`，辅助空间从 `O(n^2)` 降到
  `O(n)`。本机随机完全图 n=200 时 1.23 ms -> 0.27 ms，n=400 时
  6.40 ms -> 0.42 ms，n=800 时 20.83 ms -> 1.90 ms；输出边仍按
  `(weight, u, v)` 排序。
- CMake 在工具链支持时为 core、C ABI、原生扩展、测试和示例目标启用 IPO/LTO；
  `-DRFLUX_ENABLE_IPO=OFF` 可关闭，Debug 构建自动跳过。内部异常类使用默认
  符号可见性，保证 core 以 LTO 链接进 pybind11 模块后错误翻译器仍能匹配同一
  个 RTTI 类型。
- 完整面板相关矩阵预计算各资产均值与方差，每个资产对只需一趟协方差，并按
  资产并行；本地基准中 100 资产 × 2000 行从约 30 ms 降到 2.6 ms，含缺失值
  的 pairwise 路径保持不变。
- 清理剩余的原生重复块：两种矩共用一套 `SlidingBlocks` 两栈队列；所有滚动
  内核和分位数后端共用一次完整窗口遍历；相关/协方差/beta 合并为单一成对统计
  内核；Sharpe/Sortino 共用同一函数体；相关矩阵单元写入与 HMM 守卫也各自
  只保留一份。
- Python 层把所有“原生扩展或 C ABI”的选择收敛到 `native_or` /
  `native_call` 和唯一的 `GaussianHMM._native_or_abi`，替换原先逐函数的
  `try/except`；报告元数据、CLI 报告收尾、渗流参数编组和概率矩阵读取各只有
  一个 helper。
- 测试共用图 fixture、HMM 拟合 helper 和合成面板。`jscpd` 在 `src/`、
  `include/`、`python/rflux/`、`tests/` 上以 5 行/40 token 阈值报告 0 个完全
  克隆，仅中英双语文档镜像保留重复；新增可选的 `make dup-check` 目标。
- Python 层：因子 IC/换手率、NCO 权重、风险平价、调仓门控、尾部相依、CVaR、
  图平滑、盘口指标、绩效摘要、权益曲线和组合风险改由 C++ 实现
  （`portfoliolab/hrp.cpp`、`risk/tail_dependence.cpp`），新增 C ABI 函数和
  pybind11 入口；Python 包装保留 ctypes 回退并通过 `native_or` 调度。
- 文件清理：删除独立的 `tests/cpp/test_cpp.cpp` smoke 套件，其独有的
  describe/ranking/尾部风险断言并入 GoogleTest。CTest 现在有两个 target，
  GoogleTest 共 30 个用例。
- 文档：README、算法、核心、方法、应用、发布和示例页改为直接说明输入、输出、
  估计量和限制的工程笔记；`scripts/check_docs.py` 会拒绝 `not just`、
  `deliberately`、`Next layer` 等模板化标记。
- 算法逻辑：`trade_advice` 的总敞口按提案组合由协方差得到的事前波动率缩放，
  不再使用等权市场波动率；换手率计入现金腿。动量与日波动率窗口按日历位置
  计算，不再跳过缺失观测。
- 算法逻辑：`risk.tail_dependence` 对每个资产对平均两个有限样本条件概率，
  保证矩阵对称；`rebalance_targets` 对 2D 输入逐行独立门控。
- 算法逻辑：渗流阈值校准对第一个巨组分交点做线性插值，不再吸附到网格；
  检测器改用共享的 Welford 递推方差。波动率状态分类维护有序参考窗口，
  结果标签不变且无需每步全量排序。
- Python 公共逻辑去重：等权市场收益和 complete-case 协方差收敛到
  `_common`，`analysis`/`advice`/`apps` 共用；绩效参数校验收敛到
  `metrics._validated_options`；`fit_hmm_snapshot` 统一拟合并关闭模型，
  返回脱离句柄的最新状态/预测快照；`native.py` 的公开导出改为单一清单驱动；
  共享内存布局校验合并为一个 helper；`execution` 复用公共正数校验器。
- 新增算法可追溯矩阵（`docs/ALGORITHMS_en.md` / `_cn.md`）和
  `scripts/verify_algorithms.py`；`make algo-check` 分别通过原生扩展和 ctypes
  C ABI 跑全部公开算法族，校验形状/预热/范围/参考值并记录热路径耗时，报告写入
  `artifacts/validation/`。
- 打包卫生：sdist 的根级 include 加锚点，`.pytest_cache/README.md` 不再误匹配
  `README.md`；缓存、日志、`.DS_Store`、artifacts 和 build 目录均被排除。
  已用最终工作树重新生成 wheel/sdist。
- 构建：CMake 链接 `Threads::Threads`；开发 Makefile 在非 macOS 平台传入
  `-pthread`。
- 测试：覆盖所有 qflux 内核的 buffer/vector 等价性、两条分位数算法路径的
  暴力 Type-7 校验、输出越界哨兵、相关矩阵等价性、HMM 重启确定性、Monte
  Carlo 可重复性；Python 测试覆盖原生 `series_into` 路径。

[English](CHANGELOG.md) · **中文**

## 0.7.1 — 2026-11-18

C ABI 仍为 `1`。本版本新增 C++ 算法接口层、统一数据层和原生绑定，并保持既有 C 函数签名与结构布局不变。

- 新增 `core/algorithm.hpp`、`core/market.hpp`、`core/registry.hpp`：因子、状态、配置、预测、执行和渗流的 C++ 契约。
- 新增 `factorlab`、`portfoliolab`、`mllab`、`execution`、`risk`：动量/量价反转/VWAP/Amihud、HRP/NCO 风格配置、Huber 回归、Almgren-Chriss 和下单前风控。
- 新增 `percolab` CSR 网络与并查集 site percolation，支持 `p_c` 校准。
- 新增 `rflux.data` 统一长表 OHLCV 契约、CSV/AKShare/yfinance 适配器、代码归一、SQLite 缓存和重试。
- 新增 pybind11 原生桥、零拷贝数组、GIL 释放、CPython 3.12+ abi3 wheel、配置驱动 HMM 和共享内存数组。
- 改用 scikit-build-core + CMake/Ninja，加入 GoogleTest、pytest、Sanitizer 和 cibuildwheel 工作流。
- 清理了并查集、HMM 最近中心、Huber 残差、因子绑定和 SQLite 写入中的重复实现。
- 因子层增加 `PanelFactor` 基类：统一面板校验、输出分配和 asset/time 遍历；四个内置因子只保留单格计算和公共 trailing-return helper。
- `jscpd` 检查中 C++ 重复片段为 0。
- 适配 AKShare 1.20.46+ 的 `stock_zh_a_hist_em` 重命名，并保留旧 SDK 回退。
- yfinance 增加 `download()` 回退和 MultiIndex 列兼容；新增 opt-in 真实数据测试。
- 新增因子、HRP、Almgren-Chriss、Huber、site percolation 检测和下单前风控的 C ABI 入口；Python 在无原生扩展时统一走 C ABI。
- 移除 Tushare 适配器和相关依赖；项目不配置 PyPI 发布。
- 修正 `largest_component` 对多次 trial 均值做截断的问题，改为四舍五入。
- 本地验证把 CMake 的 Python 固定到当前虚拟环境；`make check` 优先使用 `.venv/bin/python`。
- 合并 CLI 参数定义和 AKShare 端点调用路径；清理工作区里的 `python/build` 旧产物。
- AKShare 增加 ETF 路由（`fund_etf_hist_em`），支持股票/ETF 自动识别。
- 新增 `PercolationScan.p_half_giant()`，处理稀疏相关网络上 susceptibility 在 `p = 0` 端点退化的问题。
- 修复 `trade_advice` 在首行收益为 NaN 时的协方差估计。
- 修正 Almgren-Chriss 临时成本为连续时间口径 $\eta\sum(n_j/\tau)^2\tau$；固定系数时风险中性成本不再随切片数增加。
- 新增 `rflux advice` 研究再平衡提案：HRP 基础权重、动量/趋势/RSI 调整、波动率目标与 HMM 风险状态、逐资产动作和风险估计。
- 补齐算法族工厂：`AllocatorFactory`、`PredictorFactory`、`ExecutionPlannerFactory`、`PercolationDetectorFactory`、`RegimeDetectorFactory`，统一使用参数化线程安全注册表。

- README 默认改为英文；所有 Markdown 文档增加中英文跳转；补充 HMM、HRP、Almgren-Chriss、Huber、渗流和建议公式的记号与推导。

更早版本请查看 git 标签。
