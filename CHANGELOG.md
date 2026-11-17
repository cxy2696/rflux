# Changelog

## 0.7.0 — 2026-11-17

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
- 新增 `rflux advice` 研究再平衡提案：HRP 基础权重、动量/趋势/RSI 调整、波动率目标与 HMM 风险状态、逐资产动作和风险估计。
- 补齐算法族工厂：`AllocatorFactory`、`PredictorFactory`、`ExecutionPlannerFactory`、`PercolationDetectorFactory`、`RegimeDetectorFactory`，统一使用参数化线程安全注册表。

更早版本请查看 git 标签。
