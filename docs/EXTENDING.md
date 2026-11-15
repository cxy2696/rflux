# 扩展指南

这份文档说明如何在不破坏现有边界的前提下增加算法、C ABI 函数、Python 绑定和数据源。

## 1. 新增 C++ 算法

以 `qflux` 为例：

1. 在 `include/rflux/qflux/<module>.hpp` 中声明函数。公共头只依赖 C++ 标准库和已有公共类型。
2. 在 `src/qflux/<module>.cpp` 中实现。输入校验使用 `src/internal/common.hpp` 的
   `require` / `require_pointer` / `require_finite` 等工具，抛出
   `rflux::InvalidArgument`、`rflux::NonFinite` 或 `rflux::NumericalError`。
3. 明确：
   - 输入是否允许 NaN；
   - 输出长度；
   - 预热期；
   - 除零、零方差、样本不足时的返回值；
   - 是否使用随机数，随机数种子如何传入。
4. 增加 C++ 测试 `tests/cpp/test_cpp.cpp`。
5. 若需要跨语言，再按下一节增加 C ABI；不要直接把 `std::vector` 或 C++ 异常暴露给 C。

`Makefile` 用通配符收集 `src/qflux/*.cpp`、`src/regimelab/*.cpp`、`src/percolab/*.cpp`；如果新增子目录，需要同步修改 `Makefile` 和 `CMakeLists.txt` 的源文件列表。

## 2. 新增 C ABI 函数

### 2.1 头文件

在 `include/rflux/rflux.h` 中增加：

```c
RFLUX_API rflux_status rflux_my_function(
    const double* input, size_t n,
    double* output);
```

要求：

- 使用 `RFLUX_API`。
- 函数返回 `rflux_status`，不要返回 C++ 类型或抛出异常。
- 明确每个输出指针的长度和是否允许 `NULL`。
- 不改变已有 struct 布局。需要输入选项时，优先新增一个 `struct_size` 版本化的 options 类型。
- 新的枚举值放在末尾；旧 C 客户端可能不认识新值，需要在文档中说明。

### 2.2 实现

在 `src/rflux_capi.cpp` 中：

```cpp
rflux_status rflux_my_function(const double* input, size_t n, double* output) {
    return api_guard("rflux_my_function", [&] {
        require_output(output, "output");
        copy_out(rflux::qflux::my_function(input, n), output);
    });
}
```

- 所有逻辑放在 `api_guard` 内，确保异常转换为状态码。
- `api_guard` 成功时自动清空线程局部错误。
- 不捕获错误后继续返回成功；错误必须反映在 status 里。
- 小心 `size_t` 乘法和输出长度；必要时用 `checked_mul`。
- 不把内部指针、静态容器或 C++ 对象返回给 C。

### 2.3 测试

在 `tests/c/test_capi.c` 中增加：

- 正常输入；
- 空指针/零长度；
- 非有限值；
- 边界长度和预热门槛；
- 失败后再次成功时线程局部错误是否清空；
- 输出长度不被越界写（用足够大的哨兵数组检查尾部）。

在 `tests/python/` 中增加对应绑定测试。若新增 `_ffi.py` prototype，务必写完整 `argtypes`/`restype`，否则 64 位或 Windows 上可能截断指针。

## 3. 新增 Python 绑定函数

`python/rflux/_ffi.py` 中的 `_configure()` 是唯一的 ctypes 签名登记处。新增：

```python
bind("rflux_my_function", ctypes.c_int,
     [c_double_p, size_t, c_double_p])
```

然后在 `qflux.py` / `regimelab.py` / `percolab.py` 中做 NumPy 输入转换、输出数组分配和
`check(...)`。约定：

- 输入用 `np.ascontiguousarray`；
- 1D/2D 形状在 Python 层先校验，给出更早的错误；
- C 层仍要独立校验，不能依赖 Python；
- 失败不要返回部分数组；
- 结构化结果优先用 `dataclass(frozen=True)`；
- 不把 ctypes 指针泄漏到公共 API。

## 4. 新增数据源

1. 在 `python/rflux/data.py` 中惰性导入依赖，不要放到模块顶层；这样没装 akshare 的用户仍可
   `import rflux`。
2. 在依赖缺失时抛 `MissingDependencyError`，并给出安装命令。
3. 把供应商 schema 归一化为：日期索引（升序、去重）+ 每列一个价格的宽 `DataFrame`。
4. 网络/供应商失败包装成 `DataSourceError`，但保留原始原因。
5. 不把 API key、cookie 或私有 token 放进日志、异常或仓库。
6. 如果加真实调用测试，必须在没有网络时能跳过；默认测试不应该联网。

## 5. 新增 HMM 变体

当前 C++ HMM 强制对角高斯协方差。若增加全协方差或 Student-t 发射：

- 不要改变 `GaussianHMM` 的现有参数布局；
- 新建类型或扩展 options，并同步 C ABI 的版本化 options；
- 保持 `filtered()` 因果、`smoothed()` 全样本的语义；
- 增加数值稳定性测试（长序列、极端值、零方差特征）；
- 重新评估状态排序和 `set_parameters` 的校验逻辑；
- 不要把 Python 的 `GaussianHMM` 类静默变成另一个模型；需要新的类或显式模型类型字段。

## 6. 新增图/渗流算法

图函数的输入约定：

- 矩阵行主序、n×n、有限、对称；
- 图算法忽略对角线；
- 非负性要求要在文档和实现里一致；
- 随机算法显式接收 seed，并保证同平台同版本可复现；
- 输出标签使用 `int32_t` 时，检查 n 不超过 `INT32_MAX`；
- Monte Carlo 输出要说明样本量、标准差、是否排除巨组分。

## 7. 测试与发布检查

```bash
make clean
make -j
make check
```

还建议：

```bash
# 检查 C 头文件能独立作为 C 编译
cc -std=c11 -fsyntax-only -Iinclude -x c include/rflux/rflux.h

# 检查导出符号
nm -gU build/librflux.dylib        # macOS
nm -D build/librflux.so            # Linux

# Python 包错误路径
RFLUX_LIBRARY=/nonexistent PYTHONPATH=python python -c "import rflux; rflux.abi_version()"
```

发布前确认：

1. `README`、`docs/ABI.md` 的版本号和函数表同步。
2. 选项结构没有未提升 ABI 的破坏性变化。
3. 旧的 C 测试程序可以链接新库。
4. 新随机过程有固定 seed 测试。
5. Python 数据适配器没有在 import 时联网。
6. 变更日志记录行为口径变化，而不仅是源码变化。
