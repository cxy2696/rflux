# rflux C ABI 约定

本文只讨论 `include/rflux/rflux.h`。`include/rflux/**/*.hpp` 是 C++ 源码接口，不承诺跨编译器/跨版本的 C++ ABI。

## 1. 版本

```c
#define RFLUX_ABI_VERSION 1u
#define RFLUX_VERSION_MAJOR 0
#define RFLUX_VERSION_MINOR 5
#define RFLUX_VERSION_PATCH 0
```

运行时：

```c
uint32_t rflux_abi_version(void);
const char* rflux_version_string(void);
```

动态绑定必须同时检查 ABI 与版本。当前 Python 绑定在装载库时检查 `rflux_abi_version() == 1`，否则抛出 `ABIError`。

版本策略：

- 同一 `RFLUX_ABI_VERSION` 内，已有函数签名、结构布局和语义保持兼容。
- 增加函数、增加新枚举值属于向后兼容，发布时应提升次版本号；未发布的分支可以先累积在 Unreleased 中。
- 删除函数、改变参数/返回类型、改变错误语义、扩张 ABI-v1 options 结构，必须提升 ABI 版本。
- 头文件可以在同一 ABI 内新增可选的函数；旧客户端不需要重新编译。
- 0.2.0 新增 `rflux_rolling_beta` 与 `rflux_hmm_score`，ABI 仍为 1。
- 0.4.2 仍使用 ABI 1；新增的 Python 高层分析和 CLI 不改变 C 接口。
- 0.5.0 继续使用 ABI 1；新增算法都是向后兼容的函数，没有修改已有布局或签名。
- 未发布的 `main` 分支继续使用 ABI 1；新增 percolab 函数（路径统计、k-core、同配性、模块度、MST 聚类、通用渗流扫描、簇大小分布、定向移除）都是向后兼容的追加函数，没有修改已有布局或签名，也尚未提升版本号。

## 2. 状态码与异常

```c
typedef enum rflux_status {
    RFLUX_OK = 0,
    RFLUX_ERR_INVALID_ARGUMENT = 1,
    RFLUX_ERR_OUT_OF_MEMORY = 2,
    RFLUX_ERR_NONFINITE = 3,
    RFLUX_ERR_MODEL_NOT_READY = 4,
    RFLUX_ERR_NUMERICAL = 5,
    RFLUX_ERR_INTERNAL = 6
} rflux_status;
```

实现中的 C++ 异常映射：

| C++ | C |
| --- | --- |
| 正常返回 | `RFLUX_OK` |
| `rflux::InvalidArgument` | `RFLUX_ERR_INVALID_ARGUMENT` |
| `rflux::NonFinite` | `RFLUX_ERR_NONFINITE` |
| `rflux::NotReady` | `RFLUX_ERR_MODEL_NOT_READY` |
| `rflux::NumericalError` | `RFLUX_ERR_NUMERICAL` |
| `std::bad_alloc` | `RFLUX_ERR_OUT_OF_MEMORY` |
| 其他 `std::exception` | `RFLUX_ERR_INTERNAL` |
| 未知异常 | `RFLUX_ERR_INTERNAL` |

没有任何异常会穿越 C 边界。

`rflux_status_name(int)` 接受任意整数；未知值返回 `"RFLUX_ERR_UNKNOWN"`。它不改变线程错误串。

每个线程有独立的错误字符串：

```c
const char* rflux_last_error(void);
void rflux_clear_error(void);
```

- 成功调用会把当前线程的错误串清空。
- `rflux_last_error()` 返回线程局部缓冲区指针，直到该线程下一次 rflux 调用前有效；不要释放或长期持有。
- 失败时输出缓冲区的内容未定义，调用方不应读取。

## 3. 内存与所有权

- 输入数组由调用方拥有，库在调用期间只读。
- 输出数组由调用方预分配，长度必须满足函数注释；库不分配、不接管、不释放输出缓冲区。
- `rflux_hmm*` 由 `rflux_hmm_create` 创建，由 `rflux_hmm_destroy` 释放；`destroy(NULL)` 安全。
- 返回的 `const char*` 是静态字符串（版本、状态名）或线程局部错误串，调用方不能 `free`。
- 输出数组之间不要互相别名。输入与输出别名当前实现可用（先算后拷贝），但不要依赖这一点；推荐完全分离。
- 失败时可能已经写入部分输出；把整个调用视为失败，不要部分采用结果。

## 4. 数组输出长度总表

`K = n_states`，`D = n_features`，`T = n_obs`。

### qflux 指标

| 函数 | 输出长度 |
| --- | --- |
| `rflux_returns` | n |
| `rflux_sma` / `rflux_rolling_sum` | n |
| `rflux_rolling_std` / `rflux_rolling_min` / `rflux_rolling_max` / `rflux_rolling_zscore` / `rflux_rolling_quantile` | n |
| `rflux_ema` / `rflux_rsi` / `rflux_atr` / `rflux_drawdown` | n |
| `rflux_rolling_corr` / `rflux_rolling_cov` / `rflux_rolling_beta` / `rflux_rolling_slope` / `rflux_rolling_r2` | n |
| `rflux_macd` | 三个长度为 n 的数组（macd、signal、histogram） |
| `rflux_bollinger` | 三个长度为 n 的数组（middle、upper、lower） |

### qflux 统计与横截面

| 函数 | 输出 |
| --- | --- |
| `rflux_describe` | 5 个标量 + 1 个 `size_t` |
| `rflux_quantile` / `rflux_pearson` / `rflux_sharpe` / `rflux_sortino` | 1 个标量 |
| `rflux_var_cvar` | var、cvar 两个标量 |
| `rflux_corr_matrix` | `n_assets * n_assets`，行主序 |
| `rflux_rank` / `rflux_percentile_rank` / `rflux_zscore` / `rflux_winsorize` / `rflux_neutralize` | n |

### regimelab

| 函数 | 输出 |
| --- | --- |
| `rflux_hmm_get_parameters` | start K、transition K²、means K·D、variances K·D |
| `rflux_hmm_set_parameters` | 输入同上 |
| `rflux_hmm_viterbi` | T 个 `int32_t` |
| `rflux_hmm_filtered` / `rflux_hmm_smoothed` | T·K，行主序 |
| `rflux_hmm_score` | 1 个标量（当前参数下的对数似然） |
| `rflux_hmm_forecast` / `rflux_hmm_filter_step` / `rflux_hmm_stationary` / `rflux_hmm_expected_durations` | K |
| `rflux_volatility_regime` | labels n、volatility n |
| `rflux_trend_regime` | labels n、fast_ma n、slow_ma n |
| `rflux_composite_regime` | n |

### percolab

| 函数 | 输出 |
| --- | --- |
| `rflux_corr_to_distance` | n² |
| `rflux_graph_components` | labels n + components + largest |
| `rflux_network_strength` / `rflux_network_centrality` / `rflux_network_clustering` / `rflux_network_betweenness` / `rflux_pagerank` | n |
| `rflux_network_mst` | 至少 n-1 的三组边数组 + n_edges |
| `rflux_bond_percolation` / `rflux_site_percolation` / `rflux_bond_percolation_weighted` / `rflux_site_percolation_weighted` | 每组 p 的 giant_mean、giant_std、susceptibility，共 n_probabilities |
| `rflux_risk_diffusion` | mean_affected 长度 steps+1，infection_frequency 长度 n |

## 5. HMM options 布局

```c
typedef struct rflux_hmm_options {
    size_t struct_size;
    uint32_t max_iter;
    uint32_t n_init;
    double tol;
    double variance_floor;
    uint64_t seed;
    int32_t sort_states;
    int32_t reserved0;
} rflux_hmm_options;
```

- 必须通过 `rflux_hmm_options_init(&options, sizeof(options))` 初始化。
- 当前实现要求 `struct_size >= sizeof(rflux_hmm_options)`。
- 当前 ABI-v1 字段顺序和大小冻结；不要在不同 ABI 版本中原地增加字段。
- `rflux_hmm_fit` 传 `NULL` 使用默认值：`max_iter=200`、`n_init=3`、`tol=1e-6`、`variance_floor=1e-6`、`seed=42`、`sort_states=1`。

## 6. 线程安全

- 不同 `rflux_hmm` 句柄之间可以并行使用。
- 同一个 `rflux_hmm` 句柄不是线程安全的：拟合、set_parameters 和推断不应并发。
- 纯函数（指标、统计、网络、渗流扫描、扩散）没有共享可变状态，可以并发调用。
- 错误信息是线程局部的。
- 随机数只出现在显式接收 seed 的 percolation / diffusion 函数中；每次调用使用独立 RNG，不共享全局状态。

## 7. 平台导出

- Linux/macOS：`RFLUX_API` 展开为 `visibility("default")`，库以 `-fvisibility=hidden` 编译，只有标注的函数被导出。
- Windows：构建共享库时定义 `RFLUX_BUILDING_DLL`，得到 `__declspec(dllexport)`；消费者不定义该宏，得到 `__declspec(dllimport)`。
- 静态链接消费者可定义 `RFLUX_STATIC` 以去掉 dllimport 标注。

## 8. 兼容性检查清单

发布新版本前至少检查：

1. `nm -D` / `dumpbin` 中的导出符号集合是否只增加、未删除。
2. 旧版 C 测试程序能否直接链接新库运行。
3. `RFLUX_ABI_VERSION` 是否在破坏性变更时提升。
4. `rflux_hmm_options` 的 `struct_size` 和字段偏移是否未变。
5. 错误码是否保持语义，特别是 `MODEL_NOT_READY` 和 `NONFINITE`。
6. 所有输出长度和 NaN 约定是否有文档和测试。
