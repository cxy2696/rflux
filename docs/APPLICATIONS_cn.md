# 应用：组合风险与状态归因

[English](APPLICATIONS_en.md) · [返回中文说明](../README_cn.md)

这里记录两个直接可调用的应用层函数。它们不引入新的数值内核，而是用 `qflux` 和 `metrics` 中已经测试过的口径组合出研究结论。结果字典带有估计量、年化方式和输入样本信息，便于复查。

## 1. 组合风险

`rflux.portfolio_risk(returns, weights)` 接收周期性算术收益矩阵，行是时间，列是资产。权重是每单位资本的暴露，可以为负；不要求权重和等于 1。

### 样本与协方差

先保留所有资产都有限的完整行。丢掉的行数写在 `input.dropped_observations`。记保留样本为 $R_c$ ，样本协方差为：

$$
\hat\Sigma = \frac{1}{N-d} R_c^\top R_c
$$

其中 $d$ 是 `ddof` （默认 1）。使用同一批完整观测而不是逐对完整样本，目的是保证协方差矩阵半正定，组合方差不会因为缺失模式不同而失去意义。

组合每期收益和方差：

$$
r_p = R_c w,\qquad
\sigma_p^2 = w^\top \hat\Sigma w
$$

年化波动为 $\sigma_p\sqrt{PY}$ ， $PY$ 默认 252。

### 风险贡献

组合波动为正时，使用 Euler 分解：

$$
\mathrm{MCTR}_i=\frac{(\hat\Sigma w)_i}{\sigma_p},\qquad
\mathrm{CTR}_i=w_i\mathrm{MCTR}_i
$$

报告中的 `component_volatility` 是年化后的 $\mathrm{CTR}_i\sqrt{PY}$ ，因此各项之和等于组合年化波动。`percent_contribution` 是各项占组合波动的比例。对冲资产的贡献可以为负。

另外报告：

- `diversification_ratio = sum(|w_i| * asset_vol_i) / portfolio_vol`；
- `gross_exposure = sum(|w_i|)` 和 `net_exposure = sum(w_i)`；
- `gross_hhi = sum((w_i / gross_exposure)^2)` ，描述总敞口的集中度。

组合的另一组指标由 `metrics.performance_summary` 计算，包括总收益、几何年化收益、波动、Sharpe、Sortino、最大回撤和 VaR/CVaR。权重没有归一化约束，因此 `gross_hhi` 只描述权重形状，不代表净敞口。`ddof` 只影响协方差和风险贡献；收益指标里的波动始终使用样本标准差（`ddof=1`），报告中的 `metrics_ddof` 明确记录这一点。

```python
import numpy as np
import rflux

returns = np.array([
    [0.01, 0.02],
    [0.02, -0.01],
    [-0.01, 0.01],
    [0.03, 0.00],
])
report = rflux.portfolio_risk(returns, [0.6, 0.4], names=["A", "B"])
print(report["portfolio"]["annualized_volatility"])
print(report["asset_risk"]["A"]["component_volatility"])
```

### 缺失值和零波动

- 只有所有资产都有限的收益行参与估计；这不是逐对相关矩阵的口径。
- `ddof` 必须小于完整样本数。
- 组合完全对冲、波动为零时，资产风险贡献和分散化比率返回 `None`；收益指标仍会计算。
- 输入超过 2 个资产时，权重顺序必须与收益列和 `names` 顺序一致。

## 2. 状态归因

`rflux.regime_attribution(returns, labels)` 把已有的状态标签映射到收益序列。标签可以是 HMM 的 `filtered_states()` ，也可以是规则 regime；`-1` 表示未定义或预热，不参与状态统计，也不参与转移。

每个状态报告：

- `periods` 和 `period_share`：标签出现次数和占比；
- `n_returns`：该状态内有限收益的个数；
- `mean_return` 和 `annualized_mean_return`：算术平均，后者是均值乘 `periods_per_year`，不是几何年化；
- `annualized_volatility`：样本标准差乘 $\sqrt{periods_per_year}$ ；
- `sharpe`、`var`、`cvar`、`hit_rate`；
- `sum_return` 和 `share_of_total_sum`：算术收益之和及其占全部有限收益和的比例。

转移计数只统计相邻且两边都有效的标签。行和为 0 时，该行的转移概率写成 `None`，而不是编造一组概率。

```python
labels = rflux.regimelab.volatility_regime(market_returns).labels
report = rflux.regime_attribution(market_returns, labels)
print(report["states"]["0"]["annualized_volatility"])
print(report["transitions"]["counts"])
```

### 解释边界

状态归因是描述性统计，不是因果分解。不同状态的样本数可能差别很大；`share_of_total_sum` 对复利路径没有可加性，只适合检查状态在算术收益中的占比。状态标签若来自 HMM，必须使用因果的 `filtered()` 或 `filtered_states()`；`smoothed()` 和 Viterbi 使用全样本，不能直接当作实时标签。

## 3. 命令行

`rflux risk` 从价格 CSV 计算收益并调用 `portfolio_risk`：

```bash
PYTHONPATH=python python -m rflux risk examples/data/sample_prices.csv \
  --weights 0.2,0.2,0.2,0.2,0.2

PYTHONPATH=python python -m rflux risk examples/data/sample_prices.csv \
  --weights 1,-1,0,0,0 --format json
```

权重顺序与 CSV 中除日期列外的数据列顺序一致。输出文本会列出每个资产的权重、年化波动、风险贡献和贡献占比；JSON 输出包含协方差矩阵、方法元数据和组合指标。

`rflux analyze` 仍负责资产统计、相关网络和 HMM；`risk` 只处理带权组合。两者都不需要 pandas。

## 4. 可复查性

返回结果顶层的 `method` 和 `input` 记录了估计量、`ddof`、年化方式、风险无风险利率、置信水平、完整样本数和丢弃行数。复现实验时保存完整结果即可；不要只保存四舍五入后的收益率。

这些应用不处理借券成本、交易成本、税费、容量、涨跌停或流动性。组合收益是给定权重下的事后算术组合，不是可交易组合的净值。
