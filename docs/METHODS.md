# 方法、公式与边界条件

本文记录实现中真正采用的定义，避免“名字一样但口径不同”。

## 1. qflux 时间序列

记输入为  $x_0,\dots,x_{n-1}$ ，窗口为  $w$ 。

### 1.1 收益率

简单收益率：

$$
r_t = \frac{x_t}{x_{t-1}} - 1
$$

对数收益率：

$$
r_t = \log(x_t / x_{t-1}), \quad x_{t-1}>0, x_t>0
$$

`r_0` 为 NaN；任一端非有限、简单收益前值为 0、或对数收益非正时，该位置为 NaN。

### 1.2 SMA / rolling sum / std / min / max / z-score

窗口内所有值必须有限，否则输出 NaN。滚动标准差：

$$
s_t = \sqrt{\frac{\sum_{i=t-w+1}^{t}(x_i-\bar x_t)^2}{w-d}}, \quad d \in \{0,\dots,w-1\}
$$

`ddof=0` 是总体标准差，`ddof=1` 是样本标准差。滚动 z-score：

$$
z_t = \frac{x_t-\bar x_t}{s_t}
$$

当  $s_t=0$  时输出 NaN，而不是 0。

数值实现使用两个栈维护可合并的 Welford 块矩：每个窗口的均值/中心二阶矩按块合并，而不是从原始平方和相减。这样即使价格水平在 1e9 量级、窗口波动只有 0.1，也不会把有效数字全部抵消掉。push/pop 均摊 O(1)，内存 O(window)。`rolling_corr`、`rolling_beta`、`rolling_slope` 和 `rolling_r2` 使用相同思想的二元块矩（含协矩）。

### 1.3 EMA / MACD

$$
\alpha = \frac{2}{\text{span}+1}, \qquad
e_t = \alpha x_t + (1-\alpha)e_{t-1}
$$

- 每个连续有限段的第一个值作为递归种子： $e_t=x_t$ 。
- 遇到 NaN 输出 NaN 并重置状态；下一个有限值重新作为种子。
- 因此 EMA 的输出从第一个有限值起就存在，不做 `span-1` 的显式预热。研究使用时应自行丢弃至少 `slow+signal-1` 条起步数据，或在组合中把预热期标为不可用。

MACD：

$$
\mathrm{MACD}_t = \mathrm{EMA}_{fast}(x)_t-\mathrm{EMA}_{slow}(x)_t
$$

再对 MACD 线做同口径 EMA 得到 signal，histogram = MACD − signal。要求 `fast < slow`。

### 1.4 RSI

价格差分  $\Delta_t=x_t-x_{t-1}$ 。以 `period=P` 做 Wilder 平滑：

首次：

$$
\overline G = \frac{1}{P}\sum_{i=1}^{P}\max(\Delta_i,0), \qquad
\overline L = \frac{1}{P}\sum_{i=1}^{P}\max(-\Delta_i,0)
$$

之后：

$$
\overline G_t = \frac{(P-1)\overline G_{t-1}+\max(\Delta_t,0)}{P}
$$

损失项同理。RSI：

$$
RSI = 100-\frac{100}{1+\overline G/\overline L}
$$

边界：

-  $\overline L=0,\overline G=0 \Rightarrow RSI=50$ （完全平坦）；
-  $\overline L=0,\overline G>0 \Rightarrow RSI=100$ ；
-  $\overline G=0,\overline L>0 \Rightarrow RSI=0$ 。

出现 NaN 时重置并重新积累 `P` 个差分。

### 1.5 Bollinger

$$
\mathrm{mid}_t=\mathrm{SMA}_w(x)_t,\quad
\mathrm{upper}_t = \mathrm{mid}_t + k\,\sigma^{pop}_{w,t},\quad
\mathrm{lower}_t = \mathrm{mid}_t - k\,\sigma^{pop}_{w,t}
$$

其中  $\sigma^{pop}$  使用 `ddof=0`。

### 1.6 ATR

真实波幅：

$$
TR_0 = H_0-L_0,\qquad
TR_t=\max(H_t-L_t,\ |H_t-C_{t-1}|,\ |L_t-C_{t-1}|)
$$

若上一收盘不可用，则退化为  $H_t-L_t$ 。ATR 用 Wilder 平滑，第一值在第 `period` 个有效 TR 的平均，此后：

$$
ATR_t = \frac{(P-1)ATR_{t-1}+TR_t}{P}
$$

高/低缺失会重置累计。

### 1.7 回撤与滚动回归

运行峰值  $M_t=\max_{i\le t}x_i$ ，回撤：

$$
DD_t = x_t/M_t - 1
$$

滚动 OLS 斜率把窗口内时间索引设为  $0,\dots,w-1$ ：

$$
\hat\beta = \frac{\sum (t-\bar t)(y_t-\bar y)}{\sum (t-\bar t)^2}
$$

R² 是窗口内线性拟合的决定系数；常数 y 的 R² 定义为 NaN。

滚动相关系数使用窗口内 Pearson 相关；任一侧有 NaN 的窗口为 NaN，零方差的窗口为 NaN。

滚动 beta 是资产 y 对基准 x 的窗口 OLS 斜率：

$$
\beta_t = \frac{\operatorname{Cov}_w(x,y)}{\operatorname{Var}_w(x)}
$$

窗口内任一值缺失或基准方差为 0 时输出 NaN。它是原始收益的回归斜率，不会自动剔除无风险利率，也不做滞后调整。

## 2. qflux 统计

### 2.1 描述统计

对有限样本  $x_1,\dots,x_m$ ：

$$
\bar x = \frac{1}{m}\sum x_i,\qquad
s^2 = \frac{1}{m-1}\sum (x_i-\bar x)^2
$$

偏度使用调整 Fisher-Pearson  $G_1$ ：

$$
g_1 = \frac{m_3}{m_2^{3/2}},\qquad
G_1 = \frac{\sqrt{m(m-1)}}{m-2}g_1
$$

其中  $m_k=\frac1m\sum(x_i-\bar x)^k$ 。超额峰度使用无偏  $G_2$ ：

$$
g_2 = \frac{m_4}{m_2^2}-3,\qquad
G_2 = \frac{m-1}{(m-2)(m-3)}\left[(m+1)g_2+6\right]
$$

样本不足时对应矩为 NaN，而不是抛错；完全没有有限值时抛参数错误。

### 2.2 分位数

Type-7 / 线性插值：位置  $h=(m-1)q$ ，取相邻顺序统计量插值。

### 2.3 相关矩阵

每个资产对使用**成对完整**样本：只保留两个资产同时有限的观测。均值和协方差都在这个交叠子样本上重算。

这不是“先对每个资产用全样本均值再套公式”——在缺失模式下，这样会得到错误的相关系数。矩阵对称；样本不足或任一方差为 0 的位置为 NaN，包括对角线上退化序列。

### 2.4 Sharpe / Sortino / VaR

周期收益首先减去年化无风险利率 / `periods_per_year`。

Sharpe：

$$
\frac{\bar r_e}{s(r_e)}\sqrt{PY}
$$

其中  $s$  是样本标准差。少于两个观测或零离散度时为 NaN。

Sortino 的下行偏差以下行目标 0 计算：

$$
DD = \sqrt{\frac1m\sum \min(r_{e,t},0)^2}
$$

$$
\mathrm{Sortino} = \frac{\bar r_e}{DD}\sqrt{PY}
$$

没有下行离散度时为 NaN。

历史 VaR/CVaR：

-  $q=\mathrm{Quantile}_{1-c}(r)$ ；
- VaR =  $-q$ ；
- CVaR =  $-\mathrm{mean}(r\mid r\le q)$ 。

两者是“正损失”口径，因此可能在尾部全为收益时为负。

## 3. 横截面

设有效值为  $m$  个。

### 3.1 平均秩

升序排序，1 起；并列值取它们位置的平均。NaN 保持 NaN 且不占秩。

### 3.2 Percentile

$$
pct_i = \frac{\mathrm{rank}_i-0.5}{m}
$$

### 3.3 z-score

$$
z_i = \frac{x_i-\bar x}{s}
$$

`ddof` 只允许 0 或 1。零标准差返回 NaN。

### 3.4 缩尾

把每个有限值裁到经验分位数  $q_{low},q_{high}$  之间；NaN 保留。

### 3.5 分组中性化

对每个组计算组内有限值均值，返回  $x_i-\bar x_{g(i)}$ 。`group=-1` 表示排除，输出 NaN。

## 4. 高斯 HMM

### 4.1 模型

状态数 K，特征数 D：

$$
s_1\sim \pi,\qquad s_t\mid s_{t-1}\sim A_{s_{t-1},\cdot},\qquad
x_t\mid s_t=k \sim \mathcal N(\mu_k,\operatorname{diag}(\sigma^2_k))
$$

参数为  $\pi\in\mathbb R^K$ 、行随机  $A\in\mathbb R^{K\times K}$ 、 $\mu,\sigma^2\in\mathbb R^{K\times D}$ 。

### 4.2 对数域前向/后向

发射对数密度：

$$
\log b_k(x_t)=-\frac12\sum_{d=1}^D
\left[\log(2\pi)+\log\sigma^2_{k,d}+
\frac{(x_{td}-\mu_{k,d})^2}{\sigma^2_{k,d}}\right]
$$

前向：

$$
\log\alpha_t(k)=\log b_k(x_t)+
\log\sum_j \exp(\log\alpha_{t-1}(j)+\log A_{jk})
$$

后向类似。边缘状态概率：

$$
\gamma_t(k)=\exp(\log\alpha_t(k)+\log\beta_t(k)-\log p(x))
$$

EM 的转移计数在 log 域计算  $\xi_t(j,k)$  后求和。这样避免了普通缩放前向在长序列或极端观测下的下溢。

### 4.3 初始化与拟合

- 每个 restart 用 k-means++ 式的距离抽样选择初始中心，再以 k-medoids 更新中心；EM 随后用发射均值替换这些中心。
- 硬分配得到状态均值、方差和转移计数。
- 方差下限：

$$
\text{floor}_d=\max(10^{-12},\ \text{variance\_floor}\cdot \max(\widehat{\mathrm{Var}}(x_d),10^{-12}))
$$

- 对初始分布和每行转移概率加极小的概率下限再归一化，避免 `log(0)`。
- 最多 `n_init` 次重启；选择训练对数似然最高的参数。
- 达到 `tol` 的收敛条件后停止；达到 `max_iter` 仍未收敛不是错误，`converged=0`。

### 4.4 状态排序与可识别性

HMM 本身不可识别：状态置换不改变似然。默认 `sort_states=1` 按第一特征均值升序排列。这给出确定性的重排，但：

- 多特征时“第一特征均值”未必等于风险高低；
- 跨窗口重新拟合仍可能发生状态交换；
- 不要用状态编号直接跨资产比较。

`smoothed()` / `viterbi()` 用全样本信息；`filtered()` 才是因果的。`forecast()` 只传播隐状态分布：

$$
p_{t+h}=p_t A^h
$$

不是价格或收益预测。`conditional_mean(state_probs)` 计算：

$$
\sum_k P(s=k)\,\mu_{k,d}
$$

它是当前模型参数下的条件发射均值，属于样本内描述；外推到未来仍依赖“状态均值保持不变”的假设。

### 4.5 样本外评分

`GaussianHMM::score()` / `rflux_hmm_score()` / `Python GaussianHMM.score()` 只运行前向算法，返回当前参数下观测序列的对数似然。它不依赖训练诊断，可在 `set_parameters` 后或新数据上使用，适合 walk-forward 模型比较。训练集上的 `score()` 应与 `log_likelihood()` 一致（浮点误差内）。

### 4.6 状态对齐

状态置换不改变 HMM 的似然，因此不同窗口的拟合结果可能把同一市场状态编成不同编号。Python 层提供 `match_states()` / `align_parameters()` / `align_probabilities()` / `align_labels()`：

- `match_states(reference, target)` 返回 `order`，含义是 reference 状态 k 对应 target 状态 `order[k]`；
- 代价函数为标准化均值差平方与对数方差差平方的加权和，使用 Hungarian 算法求全局最小代价分配；
- `align_parameters()` 同时重排初始分布、转移矩阵、均值和方差；
- `align_probabilities()` 重排概率列；`align_labels()` 把 target 标签映射回 reference 标签。

状态对齐只解决标签对应关系，不会让两个模型变成同一个概率模型，也不会消除模型漂移。跨窗口研究仍应检查对齐代价、状态占用和转移结构。

### 4.7 AIC / BIC

自由参数数：

$$
p=(K-1)+K(K-1)+2KD
$$

$$
AIC=-2\log L+2p,\qquad BIC=-2\log L+p\log T
$$

## 5. 规则 regime

### 5.1 波动率 regime

- 计算收益的滚动样本标准差（`ddof=1`）。
- 只保留历史有效波动率值，最多 `lookback` 个。
- t 时刻标签使用 **t 之前**的波动率经验分布，不含当前值；至少需要 20 个历史值。
- 当前值高于历史 `high_q` 分位 → 2；低于 `low_q` → 0；否则 1。

### 5.2 趋势 regime

$$
spread_t=\frac{MA_{fast,t}-MA_{slow,t}}{|MA_{slow,t}|}
$$

`spread > threshold` → 2；`spread < -threshold` → 0；否则 1。预热为 -1。

### 5.3 复合

$$
risk_t=\max(vol_t, 2-trend_t)
$$

0 = 风险偏好，1 = 中性，2 = 风险规避，-1 = 未定义。

## 6. 图与渗流

### 6.1 相关距离与 MST

Mantegna 距离：

$$
d_{ij}=\sqrt{2(1-\rho_{ij})}
$$

MST 使用 Kruskal；输出 n-1 条边。距离矩阵必须有限、对称、非负。

### 6.2 阈值图与指标

- 连通分量：边  $(i,j)$  存在当  $w_{ij}\ge threshold$ 。
- 强度： $\sum_{j\ne i}w_{ij}$ ，要求非负对称邻接。
- 特征向量中心性：对非负邻接加单位阵（A + I）后做 L2 归一化幂迭代。加单位阵不改变特征向量，只把谱整体平移，避免二部图上的周期 2 振荡；全零图返回全零，未收敛时抛数值错误。
- 局部聚类系数：邻居之间实际边数 / 可能边数；度 < 2 为 NaN。
- 图函数要求对称矩阵；对角线被忽略。

### 6.3 Bond/site 渗流

Bond：

- 候选边为  $A_{ij}>0$ ；
- 每条候选边独立以 p 保留；
- 每次重复重新计算连通分量。

Site：

- 每个节点独立以 p 存活；
- 只有两端都存活且  $A_{ij}>0$  才有边；
- 巨组分比例以原始节点数 n 为分母。

对每个 p 和 R 次重复：

$$
\overline{S}(p)=\frac1R\sum_{r=1}^{R}\frac{s_{max,r}}n
$$

$$
\chi_{finite}(p)=\frac1R\sum_{r=1}^{R}
\frac{\sum_{C\ne C_{max}}|C|^2}{n}
$$

`giant_std` 是巨组分比例的总体标准差。`susceptibility` 明确排除最大簇，用于观察有限簇在临界区附近的增长；它不是无限系统的严格理论 susceptibility。

### 6.4 SIR 风险扩散

每步：

- 每个感染节点 i 对每个易感邻居 j 以  $1-\exp(-\beta w_{ij})$  独立尝试感染；
- 同时，每个感染节点以  $\gamma$  恢复；
- 先应用恢复，再应用新感染；新感染当天不恢复；
- 一个节点恢复后不再感染（SIR，不是 SIS）。

`mean_affected[t]` 是第 t 步 I+R 占 n 的均值，包含 t=0；`infection_frequency[i]` 是跨重复“曾经感染”的频率。

这是情景模拟，不是统计估计；beta/gamma 应由使用者校准并做敏感性分析。

## 7. Python 回测口径

1. 输入 `factor[t, i]` 是 t 时刻已知的因子；`forward_returns[t, i]` 是随后一个持有期的实现收益。对齐由调用方负责。
2. 每期只在“因子和远期收益都有限”的资产中分组。
3. 因子升序排序后分成 q 组；最高非空组多头等权（总多头暴露 1），最低非空组空头等权（总空头暴露 1）。
4. 多空毛收益 = 高组等权收益 − 低组等权收益。
5. 权重  $w_t$  包含多头和空头两腿：

$$
turnover_t = \frac12\sum_i |w_{t,i}-w_{t-1,i}|
$$

6. 成本：

$$
cost_t = turnover_t \times \frac{cost\_bps}{10000}
$$

7. 净收益 = 毛收益 − 成本；没有有效横截面时仓位为零，但清仓仍计换手和成本。
8. 年化使用 `periods_per_year`，默认 252。

`factor_ic()` 逐期计算横截面 Pearson 或 Spearman 相关；`summarize_ic()` 的 `information_ratio` 是逐期 IC 的均值/标准差，不做年化。

`performance_summary()` 的 `periods` 参数控制几何年化所用的时间格数，默认等于输入长度；这样内部 NaN 缺失不会缩短年化时钟。`n_periods` 记录有限收益个数，`elapsed_periods` 记录年化采用的时间格数。杠杆多空组合可能亏损超过初始资金的 100%，此时仍返回总损失、波动、Sharpe/Sortino、VaR/CVaR，但几何年化收益与 Calmar 为 NaN。

这不包含融券成本、借券可得性、涨跌停、冲击成本、容量和税。它是一个可解释的研究基线，不是生产回测器。
