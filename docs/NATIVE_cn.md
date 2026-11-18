# 原生 Python 桥、零拷贝与 abi3

[English](NATIVE_en.md) · **中文**

rflux 保持三层独立边界：

```
C ABI (rflux.h)        其他语言 / 长期二进制契约
C++17 核心             纯数学，不解析文件、不依赖 Python 类型
pybind11 扩展          NumPy buffer、GIL 释放、稳定 ABI
Python 研究层          参数校验、数据适配、CLI、配置
```

原生扩展是可选的。存在扩展时，公开的 `qflux` 函数和
`regimelab.GaussianHMM` 自动走原生路径；不存在时回退到 ctypes 的稳定 C ABI。
设置 `RFLUX_DISABLE_NATIVE=1` 可强制回退，用于测试和诊断。

```python
import rflux
print(rflux.native_info())
# {'available': True, 'version': '0.7.1', 'stable_abi': True,
#  'gil_released': True, 'module': '.../_native.abi3.so'}
```

## 零拷贝

原生函数通过 pybind11 的 buffer-aware array caster 接收 C 连续 `float64`
NumPy 数组。连续数组直接进入 C++，不做数据拷贝；非连续或非 `float64`
输入会先由 NumPy 转换。两条输出路径都直接调用 C++ 的 `*_into` 内核：
`series` 先分配 NumPy 数组再由内核写入（一次分配，无结果拷贝），
`series_into` 直接写入调用方缓冲区：

```python
import numpy as np
from rflux import native, qflux

values = np.arange(20.0)
out = np.empty_like(values)
result = native.series_into("rflux_sma", values, out, 3)
assert result is out
np.testing.assert_allclose(out, qflux.sma(values, 3), equal_nan=True)
```

`series_into` 会拒绝非连续、dtype 不正确或只读的输出缓冲区，避免“看起来
零拷贝、实际悄悄复制”。HMM 的 `fit` 同样把观测矩阵直接交给 C++；形状、
有限性和特征维数在进入 EM 之前完成校验。

0.7 的算法模块复用同一机制：因子面板、HRP 权重、Almgren-Chriss 执行计划和
Huber 回归通过原生模块暴露，C++ 计算期间释放 GIL。

跨进程传输使用 `rflux.SharedArray`：

```python
from rflux.shared import SharedArray
import numpy as np

with SharedArray.create((1_000_000,), unlink_on_exit=True) as panel:
    panel.array[:] = np.random.default_rng(0).normal(size=panel.array.size)
    # panel.array 可直接传给原生桥；另一个进程用
    # SharedArray.attach(panel.name, (1_000_000,)) 附加同一段内存。
    print(panel.name, panel.array.nbytes)
```

只 pickle 描述符，不序列化数据。创建者在所有消费者 detach 后调用
`unlink()`。操作系统会按页向上取整共享内存大小，因此 `attach` 允许 payload
不超过已分配段的 shape。

## GIL 释放

所有耗时入口都在 C++ 计算期间释放 GIL：HMM 拟合、评分、filtered/smoothed、
Viterbi，以及全部 qflux 序列/统计核。参数校验和数组分配留在 Python 侧。
因此当 Baum-Welch 在大样本上运行时，数据接入或网络线程可以继续工作。

## 稳定 ABI（abi3）

CPython 3.12+ 的 wheel 使用 pybind11 的 `STABLE_ABI` 构建，一个
`_native.abi3.so` 可服务 CPython 3.12、3.13 及以后版本，wheel tag 为
`cp312-abi3-...`，不再按每个 Python 小版本分别编译。

CPython 3.10/3.11 没有 pybind11 稳定 ABI 所需的子集，因此这两个版本的
wheel 仍是版本相关的 `cp310`、`cp311`。Python 包本身支持 3.10+，只是二进制
tag 不同。

abi3 能力目前来自 vendored 的 pybind11 commit
`33e4318d64e795cb32a13836da8830e9cd42ab5f`（`3.2.0a0`），位于
`third_party/pybind11`，因为该特性尚未进入 PyPI 正式版。commit 固定、目录
自包含，离线 wheel 构建不依赖网络。pybind11 3.2 正式发布后可整体替换。

## 配置驱动模型

`rflux.config` 在组合层读取 TOML/JSON/YAML，然后通过纯参数接口实例化 C++
模型；核心不解析文件、不依赖配置库。

```toml
[model]
type = "hmm"
n_states = 2
n_features = 1

[model.initial]
start_prob = [0.6, 0.4]
transition = [[0.9, 0.1], [0.2, 0.8]]
means = [[-1.0], [1.0]]
variances = [[1.0], [1.0]]

[model.fit]
max_iter = 200
n_init = 3
seed = 42
```

```python
from rflux import config

model_config = config.load("model.toml").model
model = model_config.build()          # 只加载参数，不拟合
fitted = model_config.fit_model(returns)
```

存在 `model.initial` 时可以省略 `n_states`、`n_features`，解析器会从参数表
推断并与显式值交叉校验。未知字段、非有限值、负方差和形状不匹配都会在创建
原生句柄之前抛出 `rflux.ConfigError`。命令行也可以校验：
`rflux config model.toml --json --build`。

## 构建与验证

原生路径需要 CMake 3.26+ 与 Ninja。系统存在 GoogleTest 时会直接使用；也可以传
`-DRFLUX_FETCH_GTEST=ON` 下载构建。

```bash
python -m pip install ".[test]"
python -m pytest tests/python -q

# 可选的 CMake/Ninja 开发循环
cmake --preset dev
cmake --build --preset dev
ctest --preset dev --output-on-failure

# 本地 wheel
python -m pip wheel . --no-deps -w dist
```

`python -m rflux doctor --json` 同时报告 C ABI 共享库和 pybind11 扩展，
包括是否使用稳定 ABI、是否启用 GIL 释放。

## 设计要点

- 原生扩展链接与 C ABI 相同的 `rflux_core` 静态库，不复制任何数值逻辑。
- wheel 内同时包含 `rflux/_lib/` 下的 C ABI 共享库，因此干净安装中的
  ctypes 回退和 CLI 仍然可用。
- 原生异常携带与 C ABI 相同的状态码（`1` 参数错误、`3` 非有限值、
  `4` 模型未就绪、`5` 数值错误），Python 层转换为 `rflux.RFluxError`。
- 所有输出在进入 C++ 前完成形状检查；GIL 释放期间不调用任何 Python API。
