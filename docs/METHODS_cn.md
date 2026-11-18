# 方法与统计口径

[English](METHODS_en.md) · **中文**

这份文档说明 rflux 代码实际采用的计算口径。不同库经常在预热、缺失值、方差自由度、分位数和年化方式上使用不同默认值；遇到这些细节时，以这里和代码为准。

## 1. qflux 时间序列

输入序列记为 $x_0,\dots,x_{n-1}$ ，窗口长度为 $w$ 。

### 1.1 收益率

简单收益率：

$$
r_t = \frac{x_t}{x_{t-1}} - 1
$$

对数收益率：

$$
r_t = \log(x_t / x_{t-1}), \quad x_{t-1}>0,\ x_t>0
$$

首项和无法计算的位置返回 NaN。简单收益率在前值为 0 时为 NaN；对数收益率在任一端非正时为 NaN。

### 1.2 滚动统计

SMA、rolling sum、min、max 都要求窗口内所有值有限，否则该位置为 NaN。滚动标准差：

$$
s_t = \sqrt{\frac{\sum_{i=t-w+1}^{t}(x_i-\bar x_t)^2}{w-d}}
$$

其中 $d$ 是 `ddof` ，取值必须小于 $w$ 。`ddof=0` 为总体标准差，`ddof=1` 为样本标准差。滚动 z-score 为：

$$
z_t = \frac{x_t-\bar x_t}{s_t}
$$

当 $s_t=0$ 时输出 NaN，因为没有可定义的标准化。

滚动分位数使用 Type-7 定义：把窗口内有限值排序后，位置为 $h=(w-1)q$，在相邻顺序统计量之间线性插值。实现根据窗口长度选择顺序统计结构：窗口不超过 1024 时维护有序缓冲区（二分查找加分块搬移，不做全局坐标压缩），更长的窗口仍用压缩坐标上的 Fenwick 树。两条路径给出相同的 Type-7 分位数；短窗口路径避免了全局排序，对因子计算中常见的窗口快数倍。`rolling_median` 等价于 $q=0.5$，包含缺失值的窗口仍返回 NaN。

实现没有使用“平方和减去和的平方”这种滑动公式。它用两个栈维护可合并的 Welford 块矩，每个窗口从中心矩合并得到。这样即使价格在 $10^9$ 量级、窗口内只波动 0.1，也不会把有效数字抵消掉。push/pop 均摊 O(1)，额外内存 O(window)。“滚动相关、beta、斜率、R²”使用同一思路的二元块矩。

### 1.3 EMA 与 MACD

EMA 的平滑系数：

$$
\alpha = \frac{2}{\text{span}+1}, \qquad
e_t = \alpha x_t + (1-\alpha)e_{t-1}
$$

每个连续有限段的第一个值作为递归种子。遇到 NaN 时输出 NaN 并重置；下一个有限值重新作为种子。因此 EMA 从第一个有限值起就有输出，不额外保留 `span-1` 的显式预热。研究使用时应自行丢弃起步段，或把起步期标记为不可用。

MACD：

$$
\mathrm{MACD}_t = \mathrm{EMA}_{fast}(x)_t - \mathrm{EMA}_{slow}(x)_t
$$

再对 MACD 线做同口径 EMA 得到 signal，histogram 为两者之差。参数要求 `fast < slow`。

### 1.4 RSI

价格差分记为 $\Delta_t=x_t-x_{t-1}$ 。以周期 $P$ 做 Wilder 平滑。首次平均：

$$
\overline G = \frac{1}{P}\sum_{i=1}^{P}\max(\Delta_i,0), \qquad
\overline L = \frac{1}{P}\sum_{i=1}^{P}\max(-\Delta_i,0)
$$

之后：

$$
\overline G_t = \frac{(P-1)\overline G_{t-1}+\max(\Delta_t,0)}{P}
$$

损失项同理。RSI 为：

$$
RSI = 100-\frac{100}{1+\overline G/\overline L}
$$

边界处理如下：

- $\overline L=0$ 且 $\overline G=0$ ：RSI = 50，表示完全平坦；
- $\overline L=0$ 且 $\overline G>0$ ：RSI = 100；
- $\overline G=0$ 且 $\overline L>0$ ：RSI = 0。

出现 NaN 时重新积累 `period` 个差分。

### 1.5 Bollinger

$$
\mathrm{mid}_t=\mathrm{SMA}_w(x)_t,\quad
\mathrm{upper}_t=\mathrm{mid}_t+k\sigma^{pop}_{w,t},\quad
\mathrm{lower}_t=\mathrm{mid}_t-k\sigma^{pop}_{w,t}
$$

其中 $\sigma^{pop}$ 使用 `ddof=0`。

### 1.6 ATR

真实波幅：

$$
TR_0=H_0-L_0,\qquad
TR_t=\max(H_t-L_t,\ |H_t-C_{t-1}|,\ |L_t-C_{t-1}|)
$$

上一收盘不可用时退化为 $H_t-L_t$ 。ATR 用 Wilder 平滑，第一个值取前 `period` 个有效 TR 的均值，之后：

$$
ATR_t=\frac{(P-1)ATR_{t-1}+TR_t}{P}
$$

高低价缺失会重置累计。首个有效 TR 的位置以及缺失后的重新开始，都以实际数据为准。

### 1.7 回撤、回归与相关

运行峰值 $M_t=\max_{i\le t}x_i$ ，回撤为：

$$
DD_t = \frac{x_t}{M_t}-1
$$

滚动 OLS 斜率使用窗口内的时间索引：

$$
\hat\beta = \frac{\sum (t-\bar t)(y_t-\bar y)}{\sum (t-\bar t)^2}
$$

R² 是同一窗口线性拟合的决定系数。常数 y 的 R² 没有定义，返回 NaN。

滚动相关系数是窗口 Pearson 相关。任一侧缺失的窗口为 NaN，任一方差为 0 的窗口也为 NaN。

滚动 beta 是资产 y 对基准 x 的窗口 OLS 斜率：

$$
\beta_t=\frac{\operatorname{Cov}_w(x,y)}{\operatorname{Var}_w(x)}
$$

缺失窗口或基准零方差窗口返回 NaN。beta 使用原始收益，不会自动去无风险利率，也不做滞后调整。

滚动协方差使用同一个窗口二元矩：

$$
\operatorname{Cov}_{w,d}(x,y)=
rac{\sum_{i=t-w+1}^{t}(x_i-ar x_t)(y_i-ar y_t)}{w-d}
$$

`ddof=d` 必须小于窗口长度。任一侧缺失的窗口为 NaN。`ddof=0` 为总体协方差，`ddof=1` 为样本协方差。

## 2. qflux 统计

### 2.1 描述统计

对有限样本 $x_1,\dots,x_m$ ：

$$
\bar x=\frac1m\sum x_i,\qquad
s^2=\frac1{m-1}\sum(x_i-\bar x)^2
$$

偏度使用调整 Fisher-Pearson $G_1$ ：

$$
g_1=\frac{m_3}{m_2^{3/2}},\qquad
G_1=\frac{\sqrt{m(m-1)}}{m-2}g_1
$$

其中 $m_k=\frac1m\sum(x_i-\bar x)^k$ 。超额峰度使用无偏 $G_2$ ：

$$
g_2=\frac{m_4}{m_2^2}-3,\qquad
G_2=\frac{m-1}{(m-2)(m-3)}\left[(m+1)g_2+6\right]
$$

样本不足时对应的矩为 NaN，而不是强行计算。完全没有有限值时才报参数错误。

### 2.2 分位数

采用 Type-7 线性插值。位置为 $h=(m-1)q$ ，在相邻顺序统计量之间插值。`quantile`、`var_cvar` 和 `winsorize` 用 `std::nth_element` 选出所需顺序统计量，无需整段排序；大样本约快三倍，返回值不变。

### 2.3 相关矩阵

每个资产对只使用两个资产同时有限的观测。均值和协方差都在这份交叠样本上重新计算。当整个面板都是有限值时，每个资产的均值和方差只计算一次，每个资产对只需一趟协方差；资产行并行处理。

这一点很重要：先对每个资产使用全样本均值、再套相关公式，会在缺失模式不同的时候得到错误结果。输出矩阵对称；交叠样本不足、任一方差为 0 的位置为 NaN，包括对角线上的退化序列。

### 2.4 Sharpe、Sortino、VaR 与 CVaR

周期收益先减去 `risk_free / periods_per_year`。Sharpe：

$$
\frac{\bar r_e}{s(r_e)}\sqrt{PY}
$$

$s$ 是样本标准差。少于两个观测或零离散度时返回 NaN。

Sortino 的下行目标为 0：

$$
DD=\sqrt{\frac1m\sum\min(r_{e,t},0)^2}
$$

$$
\mathrm{Sortino}=\frac{\bar r_e}{DD}\sqrt{PY}
$$

没有下行离散度时返回 NaN。

历史 VaR/CVaR 使用收益率的下尾：

- $q=\mathrm{Quantile}_{1-c}(r)$ ；
- VaR $=-q$ ；
- CVaR $=-\mathrm{mean}(r\mid r\le q)$ 。

两者都按“正损失”报告。如果整个尾部仍是收益，结果可以为负。

## 3. 横截面处理

有效值个数记为 $m$ 。

### 3.1 平均秩

升序排到第 1 位；并列值取位置平均值。NaN 保持 NaN，不占用秩。

### 3.2 Percentile

$$
pct_i=\frac{\mathrm{rank}_i-0.5}{m}
$$

### 3.3 z-score

$$
z_i=\frac{x_i-\bar x}{s}
$$

`ddof` 只允许 0 或 1。零标准差返回 NaN。

### 3.4 缩尾与中性化

缩尾把每个有限值裁到经验分位数 $q_{low}$ 和 $q_{high}$ 之间，NaN 保留。

分组中性化对每组计算有限值均值，返回 $x_i-\bar x_{g(i)}$ 。`group=-1` 表示排除，输出 NaN。

## 4. 高斯 HMM

### 4.1 模型

状态数为 $K$ ，特征数为 $D$ ：

$$
s_1\sim\pi,\qquad
s_t\mid s_{t-1}\sim A_{s_{t-1},\cdot},\qquad
x_t\mid s_t=k\sim\mathcal N(\mu_k,\operatorname{diag}(\sigma_k^2))
$$

参数包括初始分布 $\pi\in\mathbb R^K$ 、行随机转移矩阵 $A\in\mathbb R^{K\times K}$ ，以及均值、方差 $\mu,\sigma^2\in\mathbb R^{K\times D}$ 。

### 4.2 前向、后向与 EM

发射对数密度：

$$
\log b_k(x_t)=-\frac12\sum_{d=1}^D\left[
\log(2\pi)+\log\sigma_{k,d}^2+
\frac{(x_{td}-\mu_{k,d})^2}{\sigma_{k,d}^2}\right]
$$

前向递推，初值 $\log\alpha_1(k)=\log\pi_k+\log b_k(x_1)$，使用 log-sum-exp：

$$
\log\alpha_t(k)=\log b_k(x_t)+
\log\sum_j\exp\left(\log\alpha_{t-1}(j)+\log A_{jk}\right)
$$

后向递推，终值 $\beta_T(k)=1$：

$$
\log\beta_t(j)=\log\sum_k\exp\left(
\log A_{jk}+\log b_k(x_{t+1})+\log\beta_{t+1}(k)\right)
$$

状态边缘概率和相邻状态后验：

$$
\gamma_t(k)=\exp\left(\log\alpha_t(k)+\log\beta_t(k)-\log p(x)\right),
$$

$$
\xi_t(j,k)=\frac{\alpha_t(j)A_{jk}b_k(x_{t+1})\beta_{t+1}(k)}{p(x)}.
$$

M 步：

$$
A_{jk}=\frac{\sum_{t=1}^{T-1}\xi_t(j,k)}
{\sum_{t=1}^{T-1}\gamma_t(j)},
\qquad
\mu_{k,d}=\frac{\sum_{t=1}^{T}\gamma_t(k)x_{td}}
{\sum_{t=1}^{T}\gamma_t(k)},
$$

$$
\sigma_{k,d}^2=\frac{\sum_{t=1}^{T}\gamma_t(k)(x_{td}-\mu_{k,d})^2}
{\sum_{t=1}^{T}\gamma_t(k)}.
$$

实现用 log-sum-exp 在对数空间维护 $\alpha$、 $\beta$ 和 $\xi$，避免普通缩放前向在长序列或极端观测下下溢。三个推断函数的口径不同：

- `filtered()` 只使用截至 t 的观测，是因果的；
- `smoothed()` 和 `viterbi()` 使用全样本，不能直接当作 t 时刻信号；
- `forecast()` 只传播隐状态分布，不预测价格。
- `filter_step(prior, observation)` 做一次因果更新：先乘转移矩阵和发射密度，再归一化；适合流式数据。
- `stationary_distribution()` 求解转移矩阵的左特征向量，返回长期状态分布；多解或奇异时抛数值错误。

单步过滤：

$$
P(s_t=k\mid \text{prior},x_t)
=\frac{\left(\sum_j \pi_j A_{jk}\right)b_k(x_t)}
{\sum_l\left(\sum_j \pi_j A_{jl}\right)b_l(x_t)}
$$

平稳分布满足 $\pi A=\pi$ 且 $\sum_k\pi_k=1$。状态期望持续期为：

$$
\mathbb E[T_k]=\frac{1}{1-A_{kk}}
$$

吸收态（ $A_{kk}=1$ ）返回正无穷。

### 4.3 初始化与拟合

- 每个 restart 用 k-means++ 式距离抽样选初始中心，以 k-medoids 方式更新；EM 随后用发射均值替换中心。
- 方差下限为：

$$
\text{floor}_d=\max\left(10^{-12},\ \text{variance\_floor}\cdot
\max(\widehat{\operatorname{Var}}(x_d),10^{-12})\right)
$$

- 初始分布和转移行加极小概率下限后归一化，避免 $\log(0)$ 。
- 最多运行 `n_init` 次 restart，保留训练对数似然最高的一组参数。
- 达到 `tol` 后停止；达到 `max_iter` 仍未收敛不算错误，`converged=0`。
- `set_parameters()` 只校验并保存参数，不额外加概率 floor；显式零会保留。`forecast()` 按公式传播，也不做 floor。

### 4.4 状态编号与可识别性

HMM 对状态置换不可识别。默认 `sort_states=1` 按第一特征均值升序重排，单次拟合结果确定，但跨窗口仍可能交换。多特征时，第一特征均值不必然等于风险高低；不要直接拿状态编号跨资产比较。

`conditional_mean(state_probs)` 计算：

$$
\sum_k P(s=k)\,\mu_{k,d}
$$

它是当前参数下的条件发射均值，属于样本内描述。外推到未来还依赖“状态均值保持不变”的假设。

### 4.5 样本外评分

`GaussianHMM.score()` 、`rflux_hmm_score()` 和 Python 的 `GaussianHMM.score()` 都只运行前向算法，返回当前参数下序列的对数似然。它不要求训练诊断，因此可以在 `set_parameters` 之后或新数据上使用。训练集上的 score 应与 `log_likelihood()` 一致（浮点误差内）。

### 4.6 状态对齐

状态置换不会改变似然，因此不同窗口的拟合可能把同一状态编成不同编号。Python 提供：

- `match_states(reference, target)` ：返回 `order` ，reference 状态 k 对应 target 状态 `order[k]` ；
- `align_parameters()` ：同步重排初始分布、转移矩阵、均值和方差；
- `align_probabilities()` ：重排概率列；
- `align_labels()` ：把 target 标签映射回 reference 标签。

代价函数使用标准化均值差平方和对数方差差平方，由 Hungarian 算法求全局最小分配。对齐只解决标签对应，不消除模型漂移。跨窗口研究仍应检查对齐代价、状态占用率和转移结构。

### 4.7 AIC 与 BIC

自由参数个数：

$$
p=(K-1)+K(K-1)+2KD
$$

$$
AIC=-2\log L+2p,\qquad BIC=-2\log L+p\log T
$$

## 5. 规则 regime

### 5.1 波动率 regime

- 计算收益的滚动样本标准差（`ddof=1`）。
- 历史有效波动率最多保留 `lookback` 个。
- t 时刻只使用 t 之前的波动率经验分布，不包含当前值；至少需要 20 个历史值。
- 当前值高于历史 `high_q` 分位标为 2；低于 `low_q` 标为 0；否则为 1。

### 5.2 趋势 regime

$$
spread_t=\frac{MA_{fast,t}-MA_{slow,t}}{|MA_{slow,t}|}
$$

`spread > threshold` 标为 2，`spread < -threshold` 标为 0，其余为 1。预热为 -1。

### 5.3 复合 regime

$$
risk_t=\max(vol_t,2-trend_t)
$$

0 = 风险偏好，1 = 中性，2 = 风险规避，-1 = 未定义。

## 6. 图、渗流与扩散

### 6.1 相关距离与 MST

相关距离采用 Mantegna 距离：

$$
d_{ij}=\sqrt{2(1-\rho_{ij})}
$$

MST 在完整距离矩阵上使用稠密 Prim，输出 n-1 条边；辅助空间 O(n)，比较次数 O(n^2)。边按 (weight, u, v) 排序，保证输出顺序确定。距离矩阵必须有限、对称、非负。

### 6.2 阈值图与中心性

- 连通分量：边 $(i,j)$ 存在当且仅当 $w_{ij}\ge threshold$ 。
- 强度： $\sum_{j\ne i}w_{ij}$ ，要求非负对称邻接。
- 特征向量中心性：对 $A+I$ 做 L2 归一化幂迭代。加单位阵不改变特征向量，但把谱整体平移，避免二部图上的周期 2 振荡；全零图返回全零，未收敛时抛数值错误。
- 局部聚类系数：邻居之间实际边数除以可能边数；度小于 2 时为 NaN。
- 全局聚类系数（传递性）：设三角形数为 $t$ ，连通三元组数为 $q=\sum_i \binom{d_i}{2}$ ，则 $T=3t/q$ ； $q=0$ 时为 NaN。
- PageRank：对非负加权有向邻接做幂迭代，阻尼 $d\in[0,1)$ 。悬挂行（出边权重和为 0）把当前质量均匀分给所有节点，每轮加入 $(1-d)/n$ 的传送项；未收敛时抛数值错误。
- Brandes 介数中心性：在无向、无权图上对每个源点做 BFS，累计最短路径依赖；`normalized=1` 时按无向图最大可能介数 $(n-1)(n-2)/2$ 归一化。

$$
p^{(t+1)}=\frac{1-d}{n}\mathbf 1
+d\left(P^\top p^{(t)}+\frac{\sum_{j\in D}p_j^{(t)}}{n}\mathbf 1\right),
\qquad P_{ij}=\frac{A_{ij}}{\sum_k A_{ik}}
$$

其中 $D$ 是悬挂行集合；行和为 0 时 $P$ 对应行不参与。最终向量归一化到和为 1。

除 PageRank 外，图函数要求对称矩阵，忽略对角线。`connected_components()` 只把权重与阈值比较，因此接受任意有限对称矩阵；`correlation_to_distance()` 接受 $[-1,1]$ 内的相关系数；其余无向图函数要求非负元素。PageRank 使用有向邻接，行表示出边。

### 6.3 最短路径、k-core、同配性与模块度

路径统计、k-core 数和度同配性把 $A_{ij}>0$ 视为无权边并忽略对角线；模块度使用边权本身。

**路径统计。** 从每个源点做 BFS。对有序点对 $i\ne j$ ，记最短路径长度为 $d_{ij}$ ，不可达时令 $1/d_{ij}=0$ 。全局效率与平均路径长度为

$$
E=\frac{1}{n(n-1)}\sum_{i\ne j}\frac{1}{d_{ij}},
\qquad
\bar d=\frac{1}{|\mathcal P|}\sum_{\{i,j\}\in\mathcal P} d_{ij}
$$

其中 $\mathcal P$ 是可达的无序点对集合。`diameter` 是最大的有限 $d_{ij}$ ；当所有无序点对都可达时 `connected` 为真。 $n=1$ 时 $E=0$ ， $\bar d$ 为 NaN，`diameter` 为 0，`connected` 为真。平均路径长度只对可达点对取平均，避免在非连通图上无定义。

**k-core 与 core 渗流。** 节点 v 的 core 数 $c_v$ 是使 v 属于 k-core 的最大 k；k-core 是最大诱导子图，其中每个节点的度数至少为 k。孤立节点的 $c_v=0$ 。实现使用 Batagelj-Zaveršnik 度桶剥离，在稠密矩阵上为 $O(n^2)$ 。删除所有 $c_v<k$ 的节点后恰好剩下 k-core。这是确定性 k-core 剥离过程（常称 core percolation）的不动点，不是随机的 bootstrap 渗流过程。 $\#\{v:c_v\ge k\}/n$ 就是 k-core 的节点占比；`core_percolation()` 无需蒙特卡洛即可给出该曲线。该比例统计所有存活节点，不要求 k-core 连通。

**度同配性。** 对无向无权图，Newman 的边端点度相关为（M 为边数）：

$$
r=\frac{M^{-1}\sum_e j_e k_e-\left[M^{-1}\sum_e (j_e+k_e)/2\right]^2}
{M^{-1}\sum_e (j_e^2+k_e^2)/2-\left[M^{-1}\sum_e (j_e+k_e)/2\right]^2}
$$

$j_e$ 和 $k_e$ 是边两端节点的度。 $M=0$ 或所有度相同时分母为 0，结果为 NaN。

**模块度。** 对划分为社区 c 的加权邻接 $A_{ij}$ ，令强度 $k_i=\sum_{j\ne i}A_{ij}$ 、 $2m=\sum_i k_i$ ：

$$
Q=\sum_c\left(\frac{\sum_{i,j\in c}A_{ij}}{2m}
-\left(\frac{\sum_{i\in c}k_i}{2m}\right)^2\right)
$$

忽略对角线。C/C++ 入口显式接收 $n\_labels$ （需在 $[1,n]$ 内）并要求每个标签位于 $[0,n\_labels)$ ；Python 封装接受任意非负标签并自动重映射为连续社区编号。 $2m=0$ 时 $Q$ 为 NaN。结果不做截断，因此浮点舍入可能让它在 $[-1,1]$ 外差几个 ulp。

**MST 单链聚类。** 将 MST 中距离大于 `threshold` 的边切断，剩余连通分量即为单链聚类。因为 MST 在每个距离阈值下都保持连通性，它等价于所有距离不超过 `threshold` 的边构成的图的连通分量。标签按每个分量中的最小节点编号。

### 6.4 割边、割点与 k-truss

三个函数都把 $A_{ij}>0$ 视为无权边，忽略对角线，结果是确定性的。

**割边与割点。** 从每个未访问节点做迭代 DFS，维护发现时间 $disc[v]$ 、低链 $low[v]$ 、父节点和子节点数。对树边 $(p,v)$ ：

- $(p,v)$ 是割边当且仅当 $low[v] > disc[p]$ ；
- 非根节点 p 是割点当且仅当存在子节点 v 满足 $low[v] \ge disc[p]$ ；
- DFS 根节点是割点当且仅当它有多于一个 DFS 子节点。

这是标准 Tarjan 割边/割点条件。无向图中每条非树边连接祖先和后裔，因此除父节点外对每个已访问邻居更新 low 即可。孤立点不是割点。割边按 $u<v$ 返回并按 $(u,v)$ 排序；割点返回 0/1 标签。

**k-truss 与边 trussness。** 子图中边 $e=(u,v)$ 的支撑度是同时与 u、v 相邻的公共邻居 w 数量。k-truss 是每条边支撑度至少 $k-2$ 的极大子图；边 trussness $t(e)$ 是使 e 属于 k-truss 的最大 k。完全不在三角形中的边 $t(e)=2$ 。

实现按 $k=3,\dots,s_{max}+2$ 分阶段， $s_{max}$ 为初始最大支撑度。阶段 k 反复删除当前支撑度小于 $k-2$ 的存活边，赋 trussness $k-1$ ；每删除一条边，对仍然存在的每个三角形把另外两条边的支撑度减一。最后仍存活的边 trussness 为 $s_{max}+2$ 。这是标准 k-truss 剥离论证：在阶段 k 被删除的边通过了 $3,\dots,k-1$ 阶段但在 k 阶段失败，所以 $k-1$ 正是其最大值；支撑度只减不增，不会漏掉更早的阶段。输出为 $n\times n$ 对称矩阵，对角线和不存在边为 0。

复杂度：稠密矩阵上支撑度构造为 $O(n^3)$ ；剥离每条边删除一次、每个仍存活三角形处理一次，堆操作 $O(\log m)$ 。总时间为 $O(n^3 + T\log m)$ ，空间为 $O(n^2)$ ，其中 T 为三角形数，m 为边数。

### 6.5 定向节点移除

移除顺序是节点下标的一个排列 $\pi$ 。移除 r 个节点后只剩 $\{\pi_{r+1},\dots,\pi_n\}$ ，边取诱导子图。`targeted_removal()` 在 $r_k=\lfloor k n / K\rfloor$ （ $k=0,\dots,K$ ）处报告：

- `giant_fraction` $s_{max}/n$ ；
- `susceptibility` $\sum_{C\ne C_{max}}|C|^2/n$ ；
- `n_components`。

`checkpoints` K 必须为正；大于 n 时部分报告点会重复。整个过程是确定性的。`removal_order()` 提供 strength、degree、eigenvector、betweenness 或 core 数排序；相同分数按节点下标升序。该曲线是静态攻击曲线，不会在每次移除后重算中心性。

### 6.6 Bond 与 site 渗流

Bond：

- 候选边为 $A_{ij}>0$ ；
- 每条边独立以概率 p 保留；
- 每次重复重新计算连通分量。

Site：

- 每个节点独立以概率 p 存活；
- 只有两端都存活且 $A_{ij}>0$ 时才存在边；
- 巨组分比例的分母始终是原始节点数 n。
加权版本把概率乘上权重并截到 1：bond 的边 $i,j$ 以 $\min(1,p w_{ij})$ 保留；site 的节点 i 以 $\min(1,p v_i)$ 存活。权重必须有限非负；这允许将相关强度、暴露或自定义 CSV 的边权接入同一套扫描。

通用 `percolation_scan()` 把候选图与占用权重分开：候选边仍然要求 $A_{ij}>0$ ，但 bond 保留概率可以使用独立的对称边权矩阵，site 存活概率可以使用独立的节点权重向量；便捷封装分别把邻接作为边权或把传入向量作为节点权重。传入与所选类型无关的权重会报非法参数，避免拼错参数时静默改变模型。

在单个概率下，`cluster_size_distribution()` 报告大小为 $1..n$ 的簇的平均数量；下标 0 恒为 0，最大簇也计入。Python 扫描结果还提供 `critical_probability()`（susceptibility 最大处的概率，并列时取最小 p）、`max_susceptibility()` 和 `giant_area()`（巨组分曲线对扫描概率的梯形积分）。稀疏图上 `p = 0` 可能因所有节点孤立而具有最大 susceptibility，该端点不能当作阈值；应使用 `p_half_giant()`，即巨组分达到终值一半的第一个扫描概率。这些都是有限样本的描述性汇总，不是外推得到的临界指数。

对每个 p 做 R 次重复：

$$
\overline S(p)=\frac1R\sum_{r=1}^{R}\frac{s_{max,r}}n
$$

$$
\chi_{finite}(p)=\frac1R\sum_{r=1}^{R}
\frac{\sum_{C\ne C_{max}}|C|^2}{n}
$$

`giant_std` 是巨组分比例的总体标准差，用 Welford 递推累计。`susceptibility` 排除了最大簇，只描述有限簇在临界区附近的增长，不是无限系统的严格理论 susceptibility。每个重复编号在所有概率上复用同一个随机种子（common random numbers），因此同一次扫描内的点彼此相关；这能降低差值噪声，但它们不是独立样本。`cluster_size_distribution()` 使用同样的按重复编号播种约定。

### 6.7 SIR 风险扩散

每一步：

- 感染节点 i 对每个易感邻居 j 以 $1-\exp(-\beta w_{ij})$ 独立尝试感染；
- 同时，每个感染节点以 $\gamma$ 恢复；
- 先结算恢复，再应用新感染；当天新感染的节点不会当天恢复；
- 恢复后不再感染，这是 SIR，不是 SIS。

`mean_affected[t]` 是第 t 步 I+R 占 n 的均值，包含 t=0。`infection_frequency[i]` 是跨重复“曾经感染”的频率。它是情景模拟，不是统计估计；beta 和 gamma 需要使用者校准并做敏感性分析。

## 7. Python 回测与绩效

1. `factor[t, i]` 必须是 t 时刻已知的因子；`forward_returns[t, i]` 是随后一个持有期的收益。对齐由调用方负责。
2. 每期只在因子和远期收益都有限的资产中分组。
3. 因子升序分 q 组；最高非空组等权做多，总多头暴露 1；最低非空组等权做空，总空头暴露 1。
4. 多空毛收益 = 高组等权收益 − 低组等权收益。
5. 换手率按两腿权重计算：

$$
turnover_t=\frac12\sum_i|w_{t,i}-w_{t-1,i}|
$$

6. 成本：

$$
cost_t=turnover_t\times\frac{cost\_bps}{10000}
$$

7. 净收益 = 毛收益 − 成本。没有有效横截面时仓位为零，但清仓仍计换手和成本。
8. 年化频率由 `periods_per_year` 控制，默认 252。

`factor_ic()` 逐期计算横截面 Pearson 或 Spearman 相关。`summarize_ic()` 的 `information_ratio` 是逐期 IC 均值除以标准差，不做年化。

`performance_summary()` 的 `periods` 参数控制几何年化使用的时间格数，默认等于输入长度。因此内部 NaN 不会自动缩短年化时钟；`n_periods` 是有限收益个数，`elapsed_periods` 是年化采用的时间格数。杠杆多空可能亏损超过初始资金，此时总损失、波动、Sharpe/Sortino 和尾部风险仍会报告，几何年化收益与 Calmar 为 NaN。

这套回测没有融券成本、借券可得性、涨跌停、冲击成本、容量和税。它适合做可解释的研究基线，不是生产撮合系统。

## 8. 算法层公式与推导

本节把组合和执行公式写成可逐项复核的形式。向量均为列向量，收益率默认是简单收益率。代码是最终口径，本节说明公式背后的假设。

### 8.1 记号与假设

设 $r_t\in\mathbb{R}^n$ 为 $t$ 时刻的收益率向量， $\Sigma=\operatorname{Cov}(r_t)$ 为 $n\times n$ 协方差矩阵。记 $\sigma_i^2=\Sigma_{ii}$， $\rho_{ij}=\Sigma_{ij}/(\sigma_i\sigma_j)$。除非调用方另行处理，假设 $\Sigma$ 对称半正定且对角线为正。权重 $w$ 为多头当 $w_i\ge 0$，无杠杆当 $\mathbf{1}^\top w\le 1$；剩余部分 $1-\mathbf{1}^\top w$ 是现金，在本节公式中收益为零。对簇 $C$， $\Sigma_{CC}$ 是其资产对应的主子矩阵。

### 8.2 因子定义

对收盘价 $p_t$ 和成交量 $v_t$，四个内置因子为

$$
m_t=\frac{p_t}{p_{t-w}}-1,
$$

$$
v_t^{\mathrm{rev}}=-\left(\frac{p_t}{p_{t-w}}-1\right)
\frac{v_t}{\frac{1}{w}\sum_{j=t-w+1}^{t}v_j},
$$

$$
\mathrm{vwap}_t=\frac{\sum_{j=t-w+1}^{t}p_jv_j}
{\sum_{j=t-w+1}^{t}v_j},\qquad
d_t=\frac{p_t}{\mathrm{vwap}_t}-1,
$$

$$
a_t=\frac{1}{w}\sum_{j=t-w+1}^{t}
\frac{|p_j/p_{j-1}-1|}{p_jv_j}\cdot 10^6.
$$

当窗口未满、价格/成交量非有限，或分母要求正数而输入不满足时，实现返回 NaN。动量和量价反转共用同一个 trailing-return helper，因此预热和基准价格检查不会出现两套口径。

### 8.3 层次风险平价

相关距离为

$$
d_{ij}=\sqrt{\frac{1-\rho_{ij}}{2}}.
$$

单链接聚类把簇 $A$ 与 $B$ 的距离定义为

$$
d(A,B)=\min_{i\in A,\,j\in B}d_{ij}.
$$

对簇 $C$，逆方差预算为

$$
\tilde w_C=\frac{\left(\operatorname{diag}\Sigma_{CC}\right)^{-1}}
{\mathbf{1}^\top\left(\operatorname{diag}\Sigma_{CC}\right)^{-1}},
\qquad
V(C)=\tilde w_C^\top\Sigma_{CC}\tilde w_C.
$$

在总预算为 $b$ 的切分 $C=L\cup R$ 上，HRP 分配

$$
b_L=b\,\frac{V(R)}{V(L)+V(R)},\qquad b_R=b-b_L.
$$

递归到单资产时，叶节点预算就是 HRP 权重。若 $V(L)+V(R)=0$，实现等分预算。逆方差权重非负，因此结果天然多头。朴素单链接构造为 $O(n^3)$；当前实现面向常见研究组合，上千资产需要稀疏或近似方法。

### 8.4 Almgren-Chriss 执行

设数量为 $Q$，切片数为 $N$，时间跨度为 $T$， $\tau=T/N$。 $x_j$ 是第 $j$ 片后的持仓， $x_0=Q$、 $x_N=0$；第 $j$ 片交易 $n_j=x_{j-1}-x_j$。规划器采用

$$
\mathrm{TC}=\eta\sum_{j=1}^{N}\left(\frac{n_j}{\tau}\right)^2\tau,
\qquad
\mathrm{PC}=\frac{1}{2}\gamma Q^2,
$$

$$
\mathrm{EC}=\frac{1}{2}\gamma Q^2+
\eta\sum_{j=1}^{N}\left(\frac{n_j}{\tau}\right)^2\tau,
$$

$$
\mathrm{Var}=\sigma^2\sum_{j=1}^{N}\tau x_j^2.
$$

其中 $\mathrm{TC}$、 $\mathrm{PC}$、 $\mathrm{EC}$、 $\mathrm{Var}$ 分别是临时成本、永久成本、期望成本和方差。

对风险厌恶 $\lambda\ge 0$，定义

$$
\kappa=\sqrt{\frac{\lambda\sigma^2}{\eta}}.
$$

连续时间最优持仓为

$$
x(t)=Q\,\frac{\sinh\left(\kappa(T-t)\right)}{\sinh(\kappa T)}.
$$

实现按离散切片计算同一比值，并把最后一片目标持仓设为零。当 $\kappa T$ 数值上可忽略时，采用风险中性线性调度 $x_j=Q(1-j/N)$，此时 $n_j=Q/N$、 $\sigma=0$ 时方差为零，临时成本为 $\eta Q^2/T$，与切片数无关。`expected_cost` 和 `expected_variance` 直接使用上面的离散公式，不再做年化处理。

### 8.5 Huber 回归

对残差 $r_i=y_i-\beta_0-x_i^\top\beta$，Huber 损失为

$$
\rho_\epsilon(r)=
\begin{cases}
\frac{1}{2}r^2, & |r|<\epsilon s,\\
\epsilon s|r|-\frac{1}{2}(\epsilon s)^2, & |r|\ge\epsilon s,
\end{cases}
$$

其中 $s$ 是稳健尺度。实现取

$$
s=1.4826\cdot\operatorname{median}\left(
\left|r_i-\operatorname{median}(r)\right|\right),
\qquad s\ge 10^{-12}.
$$

每轮求解加权最小二乘

$$
\min_{\beta_0,\beta}\sum_i w_i\left(y_i-\beta_0-x_i^\top\beta\right)^2,
\qquad
w_i=
\begin{cases}
1, & |r_i|\le \epsilon s,\\
\dfrac{\epsilon s}{|r_i|}, & |r_i|>\epsilon s.
\end{cases}
$$

当最大系数变化不超过 `tol` 时停止；最终尺度由最终残差重新计算。尺度使用中位数绝对偏差（MAD）；普通标准差会被单个异常值放大，进而影响整组权重。

### 8.6 Site percolation

对 $n$ 个节点的图，令 $X_i\sim\operatorname{Bernoulli}(p)$ 表示节点是否存活。节点 $i$ 与 $j$ 相连当且仅当 $X_i=X_j=1$ 且邻接项为正。若连通分量为 $C_1,\dots,C_m$，定义

$$
G(p)=\mathbb{E}\left[\frac{\max_k |C_k|}{n}\right],
$$

$$
\chi(p)=\mathbb{E}\left[
\frac{1}{n}\sum_{k:\,|C_k|<\max_\ell |C_\ell|}|C_k|^2
\right].
$$

有限簇磁化率不包含最大连通分量。对 $R$ 次独立 trial，实现报告样本均值和总体标准差：

$$
\bar G_R(p)=\frac{1}{R}\sum_{a=1}^{R}G_a(p),
\qquad
\widehat{\operatorname{sd}}(G)=
\sqrt{\frac{1}{R}\sum_{a=1}^{R}\left(G_a-\bar G_R\right)^2}.
$$

阈值是 $\bar G_R(p)\ge c$ 的第一个交点（ $c$ 为 `crisis_fraction`），在相邻两个网格点之间线性插值，因此不会退化为网格分辨率。这些是 Monte Carlo 估计，不是精确图概率。`largest_component` 取
$\operatorname{round}(\frac{1}{R}\sum_a \max_k |C_{k,a}|)$。

### 8.7 研究再平衡提案

设 $b$ 为封顶后的 HRP 基础权重， $s_i\in[-1,1]$ 为截面信号。信号由 20 日和 60 日动量 $m_i^{(20)}$、 $m_i^{(60)}$、趋势 $t_i\in[-1,1]$ 与 RSI 调整 $a_i$ 组成：

$$
s_i=\operatorname{clip}\left(
0.5\,z\left(m_i^{(20)}\right)+
0.3\,z\left(m_i^{(60)}\right)+
0.2\,t_i+a_i,\,-1,\,1
\right),
$$

其中 $z$ 是对有限资产的截面 z-score。倾斜权重为

$$
\tilde b_i=\operatorname{cap}\left(
b_i(1+0.5s_i)\right),
$$

`cap` 是向 $0\le \tilde b_i\le w_{\max}$、 $\sum_i\tilde b_i=1$ 的封顶单纯形投影。总敞口为

$$
e=\min\left(1,\frac{\sigma^\star}{\sigma_P}\right)e_r,
\qquad
w_i=e\tilde b_i,
\qquad
\mathrm{cash}=1-\sum_i w_i,
$$

其中 $\sigma^\star$ 是目标波动率， $\sigma_P=\sqrt{\tilde b^{\top}\Sigma\tilde b}$ 是提案组合的事前波动率， $e_r\in\{0.5,0.7,1\}$ 是 HMM 风险状态乘数；只有协方差不可用时才退回等权市场波动率。换手率包含现金腿：

$$
\tau_{\mathrm{turn}}=\frac{1}{2}\left(
\sum_i|w_i-w_i^{0}|+
\left|\mathrm{cash}-\mathrm{cash}^{0}\right|
\right).
$$

参考止损为 $p_i(1-2\hat\sigma_i)$，其中 $\hat\sigma_i$ 是最近 20 个日收益率的标准差。给定组合市值 $V$ 和单笔风险预算 $\rho$，按权重和按风险计算的股数为

$$
q_i^{w}=\left\lfloor\frac{Vw_i}{p_i}\right\rfloor,
\qquad
q_i^{\mathrm{risk}}=\left\lfloor
\frac{V\rho}{p_i-\mathrm{stop}_i}\right\rfloor,
\qquad
q_i=\min(q_i^{w},q_i^{\mathrm{risk}}).
$$

这些是研究提案：不预测收益，不完整建模成本/最小交易单位，也不构成个性化投资建议。
