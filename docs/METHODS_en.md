# Methods and conventions

[中文](METHODS_cn.md) · [Back to English guide](../README_en.md)

This page records what the code actually computes. Libraries often use the same indicator names with different defaults for warm-up, missing values, degrees of freedom, quantiles and annualisation. Where those details matter, this document and the implementation are the reference.

## 1. qflux time series

The input series is $x_0,\dots,x_{n-1}$ and the window length is $w$ .

### 1.1 Returns

Simple returns:

$$
r_t = \frac{x_t}{x_{t-1}} - 1
$$

Log returns:

$$
r_t = \log(x_t / x_{t-1}), \quad x_{t-1}>0,\ x_t>0
$$

The first value and any undefined position are NaN. A simple return is NaN when the previous value is zero; a log return is NaN when either price is non-positive.

### 1.2 Rolling statistics

SMA, rolling sum, min and max require every value in the window to be finite. Rolling standard deviation is:

$$
s_t = \sqrt{\frac{\sum_{i=t-w+1}^{t}(x_i-\bar x_t)^2}{w-d}}
$$

Here $d$ is `ddof` and must be smaller than $w$ . `ddof=0` gives the population standard deviation and `ddof=1` the sample standard deviation. Rolling z-score:

$$
z_t = \frac{x_t-\bar x_t}{s_t}
$$

A zero denominator produces NaN because the z-score is undefined.

The implementation does not use the sliding “sum of squares minus square of sum” formula. It keeps mergeable Welford moments in a two-stack queue, so a price level around $10^9$ with a window range of 0.1 does not wipe out the significant digits. Push and pop are amortised O(1), with O(window) memory. Rolling correlation, beta, slope and R² use the same idea with a second cross-moment.

### 1.3 EMA and MACD

EMA smoothing:

$$
\alpha = \frac{2}{\text{span}+1}, \qquad
e_t = \alpha x_t + (1-\alpha)e_{t-1}
$$

The first finite value in each contiguous run seeds the recursion. A NaN breaks the run; the next finite value starts a new seed. EMA therefore has an output from the first finite value and does not impose an explicit `span-1` warm-up. For research, discard the early values or mark them as unavailable.

MACD:

$$
\mathrm{MACD}_t = \mathrm{EMA}_{fast}(x)_t - \mathrm{EMA}_{slow}(x)_t
$$

Signal is an EMA of the MACD line with the same convention, and histogram is MACD minus signal. The implementation requires `fast < slow`.

### 1.4 RSI

For price changes $\Delta_t=x_t-x_{t-1}$ , use Wilder smoothing with period $P$ . The first averages are:

$$
\overline G = \frac{1}{P}\sum_{i=1}^{P}\max(\Delta_i,0), \qquad
\overline L = \frac{1}{P}\sum_{i=1}^{P}\max(-\Delta_i,0)
$$

After that:

$$
\overline G_t = \frac{(P-1)\overline G_{t-1}+\max(\Delta_t,0)}{P}
$$

The loss term is analogous. RSI is:

$$
RSI = 100-\frac{100}{1+\overline G/\overline L}
$$

Edge cases:

- $\overline L=0,\overline G=0$ : RSI = 50 (a perfectly flat window);
- $\overline L=0,\overline G>0$ : RSI = 100;
- $\overline G=0,\overline L>0$ : RSI = 0.

A NaN resets the calculation; it then needs a fresh `period` of changes.

### 1.5 Bollinger bands

$$
\mathrm{mid}_t=\mathrm{SMA}_w(x)_t,\quad
\mathrm{upper}_t=\mathrm{mid}_t+k\sigma^{pop}_{w,t},\quad
\mathrm{lower}_t=\mathrm{mid}_t-k\sigma^{pop}_{w,t}
$$

The width uses the population standard deviation, `ddof=0`.

### 1.6 ATR

True range:

$$
TR_0=H_0-L_0,\qquad
TR_t=\max(H_t-L_t,\ |H_t-C_{t-1}|,\ |L_t-C_{t-1}|)
$$

When the previous close is unavailable, the value falls back to $H_t-L_t$ . ATR applies Wilder smoothing. The first ATR is the mean of the first `period` valid true ranges; afterwards:

$$
ATR_t=\frac{(P-1)ATR_{t-1}+TR_t}{P}
$$

A missing high or low resets the accumulation. Restart positions are based on the actual valid true ranges.

### 1.7 Drawdown, regression and correlation

With running peak $M_t=\max_{i\le t}x_i$ :

$$
DD_t = \frac{x_t}{M_t} - 1
$$

Rolling OLS slope uses time inside the window:

$$
\hat\beta = \frac{\sum (t-\bar t)(y_t-\bar y)}{\sum (t-\bar t)^2}
$$

R² is the coefficient of determination for the same window. A constant y has no defined R² and returns NaN.

Rolling correlation is Pearson correlation over the window. A window with a missing value or zero variance returns NaN.

Rolling beta is the window OLS slope of asset y on benchmark x:

$$
\beta_t=\frac{\operatorname{Cov}_w(x,y)}{\operatorname{Var}_w(x)}
$$

Missing windows and zero benchmark variance return NaN. The calculation uses raw returns; it does not subtract a risk-free rate or add a lag.

## 2. qflux statistics

### 2.1 Descriptive statistics

For finite observations $x_1,\dots,x_m$ :

$$
\bar x=\frac1m\sum x_i,\qquad
s^2=\frac1{m-1}\sum(x_i-\bar x)^2
$$

Skewness is the adjusted Fisher-Pearson $G_1$ :

$$
g_1=\frac{m_3}{m_2^{3/2}},\qquad
G_1=\frac{\sqrt{m(m-1)}}{m-2}g_1
$$

where $m_k=\frac1m\sum(x_i-\bar x)^k$ . Excess kurtosis is the unbiased $G_2$ :

$$
g_2=\frac{m_4}{m_2^2}-3,\qquad
G_2=\frac{m-1}{(m-2)(m-3)}\left[(m+1)g_2+6\right]
$$

When the sample is too short for a moment, that moment is NaN rather than a fabricated number. A completely empty finite sample is an argument error.

### 2.2 Quantiles

Type-7 linear interpolation. The position is $h=(m-1)q$ and the result is interpolated between adjacent order statistics.

### 2.3 Correlation matrix

Each asset pair uses only observations where both assets are finite. The means and covariance are recomputed on that pairwise-complete sample.

This matters: using full-sample means for each asset and then applying the usual formula produces incorrect correlations when the missing patterns differ. The result is symmetric. Pairs with fewer than two observations or a zero variance are NaN, including a degenerate diagonal.

### 2.4 Sharpe, Sortino, VaR and CVaR

Periodic returns first have `risk_free / periods_per_year` subtracted. Sharpe:

$$
\frac{\bar r_e}{s(r_e)}\sqrt{PY}
$$

$s$ is the sample standard deviation. Fewer than two observations or zero dispersion gives NaN.

Sortino uses a downside target of zero:

$$
DD=\sqrt{\frac1m\sum\min(r_{e,t},0)^2}
$$

$$
\mathrm{Sortino}=\frac{\bar r_e}{DD}\sqrt{PY}
$$

No downside dispersion gives NaN.

Historical VaR and CVaR use the lower tail of returns:

- $q=\mathrm{Quantile}_{1-c}(r)$ ;
- VaR $=-q$ ;
- CVaR $=-\mathrm{mean}(r\mid r\le q)$ .

Both are reported as positive losses. If the whole tail is positive, the result can be negative.

## 3. Cross-sectional operations

Let $m$ be the number of finite values.

### 3.1 Average rank

Ascending, starting at 1. Ties share the average of their positions. NaN stays NaN and does not consume a rank.

### 3.2 Percentile rank

$$
pct_i=\frac{\mathrm{rank}_i-0.5}{m}
$$

### 3.3 z-score

$$
z_i=\frac{x_i-\bar x}{s}
$$

`ddof` must be 0 or 1. Zero standard deviation returns NaN.

### 3.4 Winsorisation and neutralisation

Winsorisation clips every finite value to the empirical quantiles $q_{low}$ and $q_{high}$ . NaN stays NaN.

Group neutralisation computes the finite mean within each group and returns $x_i-\bar x_{g(i)}$ . `group=-1` excludes the item and produces NaN.

## 4. Gaussian HMM

### 4.1 Model

For $K$ states and $D$ features:

$$
s_1\sim\pi,\qquad
s_t\mid s_{t-1}\sim A_{s_{t-1},\cdot},\qquad
x_t\mid s_t=k\sim\mathcal N(\mu_k,\operatorname{diag}(\sigma_k^2))
$$

The parameters are a start distribution $\pi\in\mathbb R^K$ , a row-stochastic transition matrix $A\in\mathbb R^{K\times K}$ , and means and variances $\mu,\sigma^2\in\mathbb R^{K\times D}$ .

### 4.2 Forward, backward and EM

Emission log density:

$$
\log b_k(x_t)=-\frac12\sum_{d=1}^D\left[
\log(2\pi)+\log\sigma_{k,d}^2+
\frac{(x_{td}-\mu_{k,d})^2}{\sigma_{k,d}^2}\right]
$$

Forward:

$$
\log\alpha_t(k)=\log b_k(x_t)+
\log\sum_j\exp\left(\log\alpha_{t-1}(j)+\log A_{jk}\right)
$$

Backward follows the same pattern. The marginal state probability is:

$$
\gamma_t(k)=\exp\left(\log\alpha_t(k)+\log\beta_t(k)-\log p(x)\right)
$$

EM sums the transition counts in log space through $\xi_t(j,k)$ . This avoids the underflow that a naively scaled forward pass can hit on long series or extreme observations. The three inference paths are not interchangeable:

- `filtered()` uses observations through t and is causal;
- `smoothed()` and `viterbi()` use the full sample and are not point-in-time signals;
- `forecast()` propagates the hidden-state distribution, not prices.

### 4.3 Initialisation and fitting

- Each restart seeds centres with k-means++-style distance sampling, updates them as k-medoids, and lets EM replace them with emission means.
- The variance floor is:

$$
\text{floor}_d=\max\left(10^{-12},\ \text{variance\_floor}\cdot
\max(\widehat{\operatorname{Var}}(x_d),10^{-12})\right)
$$

- Start probabilities and each transition row receive a tiny floor before normalisation to avoid $\log(0)$ .
- Up to `n_init` restarts run; the parameters with the highest training log-likelihood are kept.
- Fitting stops at `tol`. Reaching `max_iter` without convergence is not an error; `converged=0` reports it.
- `set_parameters()` validates and stores parameters without adding a probability floor; explicit zeros stay zero. `forecast()` also follows the product without flooring.

### 4.4 State labels and identifiability

An HMM is not identifiable under state permutation. The default `sort_states=1` reorders states by the first-feature mean. That makes one fit deterministic, but labels can still swap across windows. With multiple features, the first-feature mean is not necessarily a risk ordering; do not compare state ids across assets as if they had a global meaning.

`conditional_mean(state_probs)` computes:

$$
\sum_k P(s=k)\,\mu_{k,d}
$$

This is the conditional emission mean under the current parameters. Extending it forward still assumes the state mean stays fixed.

### 4.5 Out-of-sample scoring

`GaussianHMM.score()`, `rflux_hmm_score()` and the Python `GaussianHMM.score()` run only the forward pass and return the log-likelihood under the current parameters. They do not require fit diagnostics, so they work after `set_parameters` and on new data. On the training set, score should match `log_likelihood()` to floating-point tolerance.

### 4.6 State alignment

A state permutation does not change the likelihood, so separate fits can assign different ids to the same market state. Python exposes:

- `match_states(reference, target)` returns `order`, where reference state k corresponds to target state `order[k]`;
- `align_parameters()` reorders the start distribution, transition matrix, means and variances;
- `align_probabilities()` reorders probability columns;
- `align_labels()` maps target labels back to reference labels.

The assignment cost combines squared standardised mean differences and squared log-variance differences, minimised globally with the Hungarian algorithm. Alignment fixes labels, not model drift. Check the alignment cost, occupancy and transition structure when comparing windows.

### 4.7 AIC and BIC

The number of free parameters is:

$$
p=(K-1)+K(K-1)+2KD
$$

$$
AIC=-2\log L+2p,\qquad BIC=-2\log L+p\log T
$$

## 5. Rule-based regimes

### 5.1 Volatility regime

- Compute rolling sample standard deviation of returns (`ddof=1`).
- Keep at most `lookback` previous valid volatility values.
- At time t, use only the volatility distribution before t; at least 20 history values are required.
- Label 2 when current volatility is above the historical `high_q` quantile, 0 when below `low_q`, otherwise 1.

### 5.2 Trend regime

$$
spread_t=\frac{MA_{fast,t}-MA_{slow,t}}{|MA_{slow,t}|}
$$

Label 2 for `spread > threshold`, 0 for `spread < -threshold`, otherwise 1. Warm-up is -1.

### 5.3 Composite regime

$$
risk_t=\max(vol_t,2-trend_t)
$$

0 = risk-on, 1 = neutral, 2 = risk-off, -1 = undefined.

## 6. Graphs, percolation and diffusion

### 6.1 Correlation distance and MST

Mantegna distance:

$$
d_{ij}=\sqrt{2(1-\rho_{ij})}
$$

The MST uses Kruskal's algorithm and returns n-1 edges. The distance matrix must be finite, symmetric and non-negative.

### 6.2 Threshold graphs and centrality

- Components: an edge $(i,j)$ exists when $w_{ij}\ge threshold$ .
- Strength: $\sum_{j\ne i}w_{ij}$ on a non-negative symmetric adjacency.
- Eigenvector centrality: L2-normalised power iteration on $A+I$ . Adding the identity does not change eigenvectors; it shifts the spectrum and prevents period-2 oscillation on bipartite graphs. An all-zero graph returns zeros. Non-convergence raises a numerical error.
- Local clustering coefficient: actual neighbour-neighbour edges divided by possible edges. Degree below 2 gives NaN.

Graph functions require a symmetric matrix and ignore the diagonal.

### 6.3 Bond and site percolation

Bond:

- Candidate edges are $A_{ij}>0$ ;
- Each edge is retained independently with probability p;
- Components are recomputed for every repetition.

Site:

- Each node survives independently with probability p;
- An edge exists only when both ends survive and $A_{ij}>0$ ;
- Giant fractions always use the original node count n as denominator.

For each p and R repetitions:

$$
\overline S(p)=\frac1R\sum_{r=1}^{R}\frac{s_{max,r}}n
$$

$$
\chi_{finite}(p)=\frac1R\sum_{r=1}^{R}
\frac{\sum_{C\ne C_{max}}|C|^2}{n}
$$

`giant_std` is the population standard deviation of the giant fraction. `susceptibility` excludes the largest cluster and describes the growth of finite clusters near the critical region; it is not the strict infinite-system susceptibility.

### 6.4 SIR risk diffusion

Each step:

- infected node i attempts to infect each susceptible neighbour j with probability $1-\exp(-\beta w_{ij})$ ;
- each infected node recovers with probability $\gamma$ ;
- recoveries are applied before new infections; newly infected nodes cannot recover on the same step;
- recovered nodes stay recovered. This is SIR, not SIS.

`mean_affected[t]` is the mean fraction of I+R at step t, including t=0. `infection_frequency[i]` is the across-run frequency of ever being infected. This is a scenario simulator, not a statistical estimate; beta and gamma need calibration and sensitivity analysis.

## 7. Python backtest and performance

1. `factor[t, i]` must be known at time t; `forward_returns[t, i]` is the next holding-period return. The caller owns the alignment.
2. Each period uses assets where both the factor and forward return are finite.
3. Sort by factor into q groups. The highest non-empty group is equal-weight long with gross exposure 1; the lowest is equal-weight short with gross exposure 1.
4. Gross long-short return = high-group equal-weight return − low-group equal-weight return.
5. Turnover over both legs:

$$
turnover_t=\frac12\sum_i|w_{t,i}-w_{t-1,i}|
$$

6. Cost:

$$
cost_t=turnover_t\times\frac{cost\_bps}{10000}
$$

7. Net return = gross return − cost. With no valid cross-section the position is flat, but closing an existing position still counts turnover and cost.
8. Annualisation uses `periods_per_year`, default 252.

`factor_ic()` computes cross-sectional Pearson or Spearman IC per period. The `information_ratio` in `summarize_ic()` is the per-period IC mean divided by its standard deviation; it is not annualised.

The `periods` argument to `performance_summary()` sets the number of elapsed periods used for geometric annualisation. It defaults to the input length, so internal NaN gaps do not shorten the clock. `n_periods` is the count of finite returns and `elapsed_periods` is the annualisation horizon. A levered long-short book can lose more than its initial capital; total loss, volatility, Sharpe/Sortino and tail risk are still reported, while geometric annual return and Calmar become NaN.

The backtest does not include borrow fees, short availability, limit moves, market impact, capacity or taxes. It is a transparent research baseline, not an execution engine.
