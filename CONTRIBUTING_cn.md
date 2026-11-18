# 参与贡献

[English](CONTRIBUTING.md) · **中文**

欢迎提交 bug 报告和聚焦的 pull request。

## 开发

```bash
make -j
make check
```

本地完整验证（构建、CTest、pytest、CLI、示例和文档）可运行
`bash scripts/vscode_verify.sh`；日志写到 `artifacts/validation/`，失败时返回
非零。

`make check` 包含 Markdown 检查（`scripts/check_docs.py`，检查相对链接、
代码围栏和 GitHub 数学分隔符）、`scripts/check_version.py`（保证 CMake、C 头
和 Python 元数据版本一致）以及 `make algo-check`：分别通过原生扩展和
ctypes C ABI 运行 `scripts/verify_algorithms.py`。算法可追溯矩阵见
`docs/ALGORITHMS_cn.md`。

发布代码和测试保持 0 个完全克隆。大型重构前运行 `make dup-check`（需要
Node/`npx`，因此 CI 不执行）；该命令以更严格的 5 行/40 token 阈值检查
`src/`、`include/`、`python/rflux/` 和 `tests/`。中英双语文档镜像是有意保留的
唯一重复块。

## 文档风格

文档按工程笔记来写，面向需要复现结果的人。直接写清输入、输出、估计量和测试；
少用营销形容词、重复免责声明和模板化过渡句，限制说明在影响使用的位置写一次。
中英文页面保持同步。`scripts/check_docs.py` 会拒绝 `not just`、`deliberately`、
`Next layer` 等一小串模板化标记。

C/C++ 改动最快的循环是 Make。需要原生 Python 扩展和可复现的 CMake/Ninja
构建时：

```bash
python -m pip install -e ".[test]"
python -m pytest tests/python -q

cmake --preset dev
cmake --build --preset dev
ctest --preset dev --output-on-failure

# ASan + UBSan；Linux CI 开启 leak detection，macOS 不支持
cmake --preset sanitize
cmake --build --preset sanitize
ctest --preset sanitize --output-on-failure
```

CMake 优先使用系统 GoogleTest；未安装时可传
`-DRFLUX_FETCH_GTEST=ON` 下载构建，或用 `-DRFLUX_BUILD_GTEST=OFF` 跳过该
测试套件。

`RFLUX_DISABLE_NATIVE=1` 会强制走 ctypes 回退。添加原生绑定时，耗时的 C++
计算要释放 GIL，所有 Python API 调用必须放在释放区之外。新的公开行为需要
在 `tests/python/` 增加 pytest；新的 C++ 内核还应增加 GoogleTest 用例。

## 添加内核

1. 在对应的 `include/rflux/**/*.hpp` 中声明。
2. 在 `src/` 中实现。
3. 用已知输入增加数值测试，覆盖 NaN、零方差等边界。
4. 如果需要跨语言使用，在 `include/rflux/rflux.h` 和 `src/rflux_capi.cpp`
   增加 C 函数，再增加 ctypes 绑定和 C ABI 测试。
5. 确定性图算法要用已知图和小规模暴力实现对拍。Monte Carlo 算法要用独立
   单次 replicate 验证均值和总体标准差。

## 添加 C ABI 函数

- 返回 `rflux_status`，在 `src/rflux_capi.cpp` 中使用 `api_guard`。
- 在声明上方写清输出长度和缺失值约定。
- 不要暴露 C++ 类型，也不要在边界上让异常逃逸。
- ABI-v1 的现有签名和 options 布局不能变；破坏性变更必须提升 ABI 版本。

## Python 改动

可选数据依赖保持惰性。`import rflux` 不得访问网络或强制依赖 pandas。新的
公开函数需要在 `tests/python/` 增加测试。

## 风格

跟随周边代码。C++ 使用 C++17，Python 支持 3.10+。优先使用清晰命名和显式
边界处理，不要用抽象隐藏数值行为。
