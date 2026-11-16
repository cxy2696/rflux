# 方法与统计口径

[English](METHODS_en.md) · [返回中文说明](../README_cn.md)

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
\operatorname{Cov}_{w,d}(x,y)=rac{\sum_{i=t-w+1}^{t}(x_i-ar x_t)(y_i-ar y_t)}{w-d}
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

采用 Type-7 线性插值。位置为 $h=(m-1)q$ ，在相邻顺序统计量之间插值。

### 2.3 相关矩阵

每个资产对只使用两个资产同时有限的观测。均值和协方差都在这份交叠样本上重新计算。

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

前向：

$$
\log\alpha_t(k)=\log b_k(x_t)+
\log\sum_j\exp\left(\log\alpha_{t-1}(j)+\log A_{jk}\right)
$$

后向同理。状态边缘概率：

$$
\gamma_t(k)=\exp\left(\log\alpha_t(k)+\log\beta_t(k)-\log p(x)\right)
$$

EM 的转移计数在 log 域计算 $\xi_t(j,k)$ 后求和，避免普通缩放前向在长序列或极端观测下下溢。三个推断函数的口径不同：

- `filtered()` 只使用截至 t 的观测，是因果的；
- `smoothed()` 和 `viterbi()` 使用全样本，不能直接当作 t 时刻信号；
- `forecast()` 只传播隐状态分布，不预测价格。

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

MST 使用 Kruskal 算法，输出 n-1 条边。距离矩阵必须有限、对称、非负。

### 6.2 阈值图与中心性

- 连通分量：边 $(i,j)$ 存在当且仅当 $w_{ij}\ge threshold$ 。
- 强度： $\sum_{j\ne i}w_{ij}$ ，要求非负对称邻接。
- 特征向量中心性：对 $A+I$ 做 L2 归一化幂迭代。加单位阵不改变特征向量，但把谱整体平移，避免二部图上的周期 2 振荡；全零图返回全零，未收敛时抛数值错误。
- 局部聚类系数：邻居之间实际边数除以可能边数；度小于 2 时为 NaN。
- PageRank：对非负加权有向邻接做幂迭代，阻尼 $d\in[0,1)$ 。悬挂行（出边权重和为 0）把当前质量均匀分给所有节点，每轮加入 $(1-d)/n$ 的传送项；未收敛时抛数值错误。

$$
p^{(t+1)}=\frac{1-d}{n}\mathbf 1
+d\left(P^\top p^{(t)}+\frac{\sum_{j\in D}p_j^{(t)}}{n}\mathbf 1\right),
\qquad P_{ij}=\frac{A_{ij}}{\sum_k A_{ik}}
$$

其中 $D$ 是悬挂行集合；行和为 0 时 $P$ 对应行不参与。最终向量归一化到和为 1。

除 PageRank 外，图函数要求对称矩阵，忽略对角线。PageRank 使用有向邻接，行表示出边。

### 6.3 Bond 与 site 渗流

Bond：

- 候选边为 $A_{ij}>0$ ；
- 每条边独立以概率 p 保留；
- 每次重复重新计算连通分量。

Site：

- 每个节点独立以概率 p 存活；
- 只有两端都存活且 $A_{ij}>0$ 时才存在边；
- 巨组分比例的分母始终是原始节点数 n。

对每个 p 做 R 次重复：

$$
\overline S(p)=\frac1R\sum_{r=1}^{R}\frac{s_{max,r}}n
$$

$$
\chi_{finite}(p)=\frac1R\sum_{r=1}^{R}
\frac{\sum_{C\ne C_{max}}|C|^2}{n}
$$

`giant_std` 是巨组分比例的总体标准差。`susceptibility` 排除了最大簇，只描述有限簇在临界区附近的增长，不是无限系统的严格理论 susceptibility。

### 6.4 SIR 风险扩散

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
