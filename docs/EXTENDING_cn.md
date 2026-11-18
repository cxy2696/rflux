# 扩展指南

[English](EXTENDING_en.md) · **中文**

rflux 的边界清晰：C++ 负责计算，C ABI 负责跨语言，Python 负责输入输出和组合。扩展前先确定新功能属于哪一层。

## 1. 新增 C++ 算法

以 `qflux` 为例：

1. 在 `include/rflux/qflux/<module>.hpp` 声明函数。
2. 在 `src/qflux/<module>.cpp` 实现。参数检查使用 `src/internal/common.hpp` 的 `require`、`require_pointer`、`require_finite`，抛出 `rflux::InvalidArgument`、`rflux::NonFinite` 或 `rflux::NumericalError`。
3. 在头文件和实现里写清楚：
   - 输入是否允许 NaN；
   - 输出长度；
   - 预热期；
   - 零方差、样本不足、完全缺失时返回什么；
   - 是否使用随机数，seed 从哪里来。
4. 在 `tests/cpp/test_cpp.cpp` 增加数值测试。至少包含正常值、边界值和退化输入。
5. 如果需要跨语言，再按下一节增加 C ABI。不要把 `std::vector`、异常或内部指针直接暴露给 C。

`Makefile` 用通配符收集三个模块目录下的 `.cpp`。新增子目录时，同时修改 `Makefile` 和 `CMakeLists.txt` 的源文件列表。

## 2. 新增 C ABI 函数

### 头文件

在 `include/rflux/rflux.h` 增加：

```c
RFLUX_API rflux_status rflux_my_function(
    const double* input, size_t n,
    double* output);
```

规则：

- 返回 `rflux_status`，使用 `RFLUX_API`。
- 不用 C++ 类型，不抛异常。
- 每个输出指针的长度写清楚；除明确允许的情况外，不接受 `NULL`。
- 已有结构体布局不能原地改变。需要新选项时，新建带 `struct_size` 的版本化 options 类型。
- 新枚举值放在末尾，并说明旧客户端如何忽略它。

### 实现

在 `src/rflux_capi.cpp` 中包一层。单输出数组优先用现成的 guard：

```cpp
rflux_status rflux_my_function(const double* input, size_t n, double* output) {
    return guarded_vector("rflux_my_function", output, "output",
                          [&] { return rflux::qflux::my_function(input, n); });
}
```

`guarded_scalar` 用于单个 double 输出，`guarded_i32` 用于标签数组。多输出函数继续使用 `api_guard`，逐项校验后再计算；HMM 这类需要“先校验模型、再校验输出”的函数使用带 prepare 回调的重载。

- 所有逻辑放进 `api_guard`，异常才能变成状态码。
- 先校验输出指针，再计算；失败时不要返回部分结果。
- `size_t` 乘法用 `checked_mul`。
- 不返回内部静态容器或 C++ 对象。
- 同一个句柄的拟合和推断不要并发使用；纯函数之间没有共享状态。

### 测试

在 `tests/c/test_capi.c` 覆盖：

- 正常输入；
- 空指针、零长度；
- 非有限值；
- 边界窗口和预热；
- 失败后成功调用是否清空 `rflux_last_error()`；
- 输出数组尾部哨兵，确认没有越界写。

同时在 `tests/python/` 加绑定测试。`_ffi.py` 的 `argtypes` 和 `restype` 必须写完整，否则在 64 位或 Windows 上可能截断指针。

## 3. 新增 Python 绑定

`python/rflux/_ffi.py` 的 `_configure()` 是 ctypes 签名登记处。新增：

```python
bind("rflux_my_function", ctypes.c_int,
     [c_double_p, size_t, c_double_p])
```

然后在对应模块中做 NumPy 转换、分配输出并调用 `check(...)`。约定如下：

- 输入转成连续数组，必要时复制；
- 形状和明显越界的参数在 Python 层先报错，但 C 层仍要独立校验；
- 失败不要返回部分数组；
- 结构化返回值用 `dataclass(frozen=True)`；
- 不把 ctypes 指针泄漏给调用方。

Python 层只处理接口和组合，不要把重计算搬到 Python。

## 4. 新增数据源

在 `python/rflux/data.py` 中加 loader：

1. 依赖惰性导入，保证没装 akshare/yfinance 的用户仍能 `import rflux`。
2. 缺依赖时抛 `MissingDependencyError`，并给出安装命令。
3. 输出统一为：日期升序、日期唯一、每列一个价格的宽 `DataFrame`。
4. 网络、供应商 schema、空数据问题包装成 `DataSourceError`，保留原始原因。
5. 不把 API key、cookie 或 token 写进日志、异常或仓库。
6. 默认测试不联网；真实接口测试必须可跳过。

现有的 CSV、akshare 和 yfinance loader 可以作为模板。

## 5. 新增 HMM 变体

当前 `GaussianHMM` 使用对角高斯发射。增加全协方差、Student-t 或其他发射时：

- 不要改变 `GaussianHMM` 的现有参数布局。
- 新建类型，或扩展带版本的 options，并同步 C ABI。
- 保持 `filtered()` 因果、`smoothed()`/`viterbi()` 全样本的语义。
- 补充长序列、极端值、零方差特征的数值测试。
- 重新检查状态排序、参数校验和状态对齐代价。
- 不要静默改变 Python `GaussianHMM` 的含义；需要新类或显式模型类型。

## 6. 新增图或渗流算法

图函数遵循统一输入约定：

- 行主序、n×n、有限、对称；
- 忽略对角线；
- 是否要求非负，头文件、实现和文档必须一致；
- 随机过程显式接收 seed，同平台同版本可复现；
- 输出 `int32_t` 标签时，检查 n 不超过 `INT32_MAX`；
- Monte Carlo 结果说明样本量、标准差，以及是否排除巨组分；
- 确定性算法（最短路径、core 数、模块度、MST 聚类、定向移除）不得消耗随机数；
- 相似度图用 `weight >= threshold`，距离图用 `distance <= threshold`，不要在一个 helper 中混用；
- 可选的边/节点占用权重必须校验有限、对称（适用时）且非负；属于另一种占用类型的权重应报错；
- 划分标签为 `int32_t` 且在 `[0, n_labels)` 内，`n_labels <= n`；文档要说明是否忽略对角线和自环；
- 路径统计要明确平均是对可达点对还是全部点对，以及单节点输入的返回值。

如果新度量依赖图的方向，另开类型或参数，不要偷偷把对称矩阵当有向图用。

新增确定性度量时，在小规模随机图上与 brute-force 实现对比；新增蒙特卡洛内核时，先用独立的单次调用核对均值和总体标准差，再加入端点测试。

## 7. 提交前检查

```bash
make clean
make -j
make check
```

`make check` 包含 C ABI、C++、Python 和 Markdown 检查。也可以单独运行：

```bash
make docs-check
cc -std=c11 -fsyntax-only -Iinclude -x c include/rflux/rflux.h
```

导出符号：

```bash
nm -gU build/librflux.dylib   # macOS
nm -D build/librflux.so       # Linux
```

发布前确认：

1. `include/rflux/rflux.h` 的版本宏、README 和 changelog 同步。
2. ABI 没有在旧版本号下发生破坏性变化。
3. 旧版 C 程序还能链接新库。
4. 新随机过程有固定 seed 测试。
5. Python import 不联网；可选依赖缺失时错误信息清楚。
6. 新增公式使用 GitHub 支持的 `$...$` / `$$...$$`，并运行 `make docs-check`。

CI 会在 Ubuntu 和 macOS 上验证 Make、CMake、Python 测试和示例。修改文档后，也建议用 GitHub 的预览或 Markdown API 再看一遍渲染。
