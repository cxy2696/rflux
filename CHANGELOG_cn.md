# Changelog

## 0.9.0 — 2026-11-19

Python 层改为薄绑定：所有算法和计算都在 C++ 中完成，ctypes 桥已移除。C ABI
仍为 `1`；新增报告内核只通过 pybind11 暴露。

- 移除 ctypes 回退（`python/rflux/_ffi.py`、指针辅助函数、
  `RFLUX_DISABLE_NATIVE`、`RFLUX_LIBRARY` 和 wheel 里的 `_lib` 载荷）。Python
  包直接导入编译好的 pybind11 扩展；未构建时给出明确的 `ImportError`。公开的
  `LibraryNotFoundError` 导出随 ctypes 加载器一起移除。
- 新增 C++ 研究内核：建议信号/评分/配置/状态/仓位
  （`src/portfoliolab/advice.cpp`）、状态归因
  （`src/regimelab/attribution.cpp`）、sticky regime、HMM 参数对齐重排、
  相关邻接矩阵与连通分量大小、渗流扫描摘要、移除顺序与 k-core 曲线、
  等权市场收益、完整样本协方差和组合收益序列。
- 新增报告层使用的小型数值函数：有限值统计、最后一个有限值、top-k 索引、
  argmax/row-argmax、int32 最大值和线性预测。回测 IC 内核支持最小配对数量。
- 将 `qflux.py`、`metrics.py`、`execution.py`、`factorlab.py`、`mllab.py`、
  `portfoliolab.py`、`risk.py`、`percolab.py`、`regimelab.py`、`advice.py`、
  `analysis.py`、`apps.py`、`backtest.py` 重写为原生扩展之上的校验/格式化
  包装。Python 包从约 8.3k 行降到 6.4k 行，不再包含 NumPy 计算循环。
- `make check`、`make algo-check`、VS Code 验证脚本和 CI 改为单条原生测试
  路径；wheel 只包含扩展，C ABI 库仍由独立 CMake 安装目标提供。
