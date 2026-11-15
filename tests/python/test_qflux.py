import unittest

import numpy as np

import rflux
from rflux import qflux
from rflux._ffi import RFluxError


class TestLibraryAndIndicators(unittest.TestCase):
    def test_native_identity(self):
        self.assertEqual(rflux.abi_version(), 1)
        self.assertEqual(rflux.native_version(), "0.2.0")
        self.assertEqual(rflux.__version__, "0.2.0")

    def test_returns_and_sma(self):
        prices = np.arange(1.0, 11.0)
        simple = qflux.returns(prices)
        self.assertTrue(np.isnan(simple[0]))
        np.testing.assert_allclose(simple[1:], 1.0 / np.arange(1.0, 10.0))
        log = qflux.returns(prices, log=True)
        self.assertAlmostEqual(log[1], np.log(2.0))
        sma = qflux.sma(prices, 3)
        self.assertTrue(np.isnan(sma[:2]).all())
        np.testing.assert_allclose(sma[2:], np.arange(2.0, 10.0))

    def test_ema_rsi_macd_bollinger_atr(self):
        prices = np.arange(1.0, 21.0)
        ema = qflux.ema(prices, 3)
        self.assertAlmostEqual(ema[0], 1.0)
        self.assertAlmostEqual(ema[1], 1.5)
        self.assertAlmostEqual(ema[2], 2.25)
        rsi = qflux.rsi(prices, 14)
        self.assertTrue(np.isnan(rsi[13]))
        self.assertAlmostEqual(rsi[14], 100.0)

        macd = qflux.macd(prices, fast=3, slow=5, signal=2)
        np.testing.assert_allclose(
            macd.histogram[-1], macd.macd[-1] - macd.signal[-1]
        )
        bands = qflux.bollinger(prices, window=20, n_std=2.0)
        expected_sd = np.sqrt(((20.0**2) - 1.0) / 12.0)
        self.assertAlmostEqual(bands.middle[-1], 10.5)
        self.assertAlmostEqual(bands.upper[-1], 10.5 + 2.0 * expected_sd)

        high = prices + 1.0
        low = prices - 1.0
        atr = qflux.atr(high, low, prices, period=3)
        self.assertTrue(np.isnan(atr[1]))
        np.testing.assert_allclose(atr[2:], 2.0)

    def test_rolling_and_missing_values(self):
        values = np.array([1.0, 2.0, np.nan, 4.0, 5.0, 6.0])
        sma = qflux.sma(values, 2)
        self.assertTrue(np.isnan(sma[2]) and np.isnan(sma[3]))
        self.assertAlmostEqual(sma[4], 4.5)
        self.assertAlmostEqual(qflux.rolling_sum(values, 2)[5], 11.0)
        self.assertAlmostEqual(qflux.rolling_min(values, 2)[5], 5.0)
        self.assertAlmostEqual(qflux.rolling_max(values, 2)[5], 6.0)
        self.assertAlmostEqual(qflux.rolling_std(values, 2)[5], 0.5)

        prices = np.arange(1.0, 11.0)
        correlation = qflux.rolling_corr(prices, 2.0 * prices, 3)
        np.testing.assert_allclose(correlation[2:], 1.0)
        slope = qflux.rolling_slope(prices, 5)
        np.testing.assert_allclose(slope[4:], 1.0)
        r2 = qflux.rolling_r2(prices, 5)
        np.testing.assert_allclose(r2[4:], 1.0)
        beta = qflux.rolling_beta(2.0 * prices + 3.0, prices, 5)
        np.testing.assert_allclose(beta[4:], 2.0)
        beta_missing = qflux.rolling_beta(
            np.array([1.0, 2.0, np.nan, 4.0, 5.0]),
            np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
            2,
        )
        self.assertTrue(np.isnan(beta_missing[2]) and np.isnan(beta_missing[3]))
        stable_gap = np.array([1.0, 2.0, np.nan, 4.0, 5.0, 6.0])
        stable_std = qflux.rolling_std(stable_gap, 2)
        self.assertTrue(np.isnan(stable_std[2]) and np.isnan(stable_std[3]))
        np.testing.assert_allclose(stable_std[4:], 0.5)
        stable_corr = qflux.rolling_corr(
            stable_gap, 2.0 * stable_gap + 1.0, 2
        )
        self.assertTrue(np.isnan(stable_corr[2]) and np.isnan(stable_corr[3]))
        np.testing.assert_allclose(stable_corr[4:], 1.0)
        stable_slope = qflux.rolling_slope(stable_gap, 2)
        self.assertTrue(np.isnan(stable_slope[2]) and np.isnan(stable_slope[3]))
        np.testing.assert_allclose(stable_slope[4:], 1.0)
        drawdown = qflux.drawdown(np.array([100.0, 110.0, 99.0, 120.0]))
        np.testing.assert_allclose(drawdown, [0.0, 0.0, -0.1, 0.0], atol=1e-12)

    def test_zscore_single_observation_is_undefined(self):
        self.assertTrue(np.isnan(qflux.rolling_zscore([5.0], 1)[0]))
        self.assertEqual(float(qflux.rolling_std([5.0], 1)[0]), 0.0)

    def test_large_offset_rolling_moments(self):
        t = np.arange(100.0)
        x = 1e9 + np.sin(t)
        y = 2.0 * x + 3.0
        window = 20
        for end in (40, 99):
            xw = x[end - window + 1:end + 1]
            yw = y[end - window + 1:end + 1]
            tw = t[end - window + 1:end + 1]
            xc = xw - xw.mean()
            yc = yw - yw.mean()
            tc = tw - tw.mean()
            expected_std = xw.std(ddof=0)
            expected_beta = np.dot(xc, yc) / np.dot(xc, xc)
            expected_corr = np.dot(xc, yc) / np.sqrt(
                np.dot(xc, xc) * np.dot(yc, yc)
            )
            expected_z = (x[end] - xw.mean()) / expected_std
            self.assertAlmostEqual(
                qflux.rolling_std(x, window)[end], expected_std, delta=1e-6
            )
            self.assertAlmostEqual(
                qflux.rolling_beta(y, x, window)[end], expected_beta, delta=1e-6
            )
            self.assertAlmostEqual(
                qflux.rolling_corr(y, x, window)[end], expected_corr, delta=1e-6
            )
            self.assertAlmostEqual(
                qflux.rolling_zscore(x, window)[end], expected_z, delta=1e-6
            )
            slope = qflux.rolling_slope(x, window)[end]
            r2 = qflux.rolling_r2(x, window)[end]
            expected_slope = np.dot(tc, xc) / np.dot(tc, tc)
            expected_r2 = np.dot(tc, xc) ** 2 / (
                np.dot(tc, tc) * np.dot(xc, xc)
            )
            self.assertAlmostEqual(slope, expected_slope, delta=1e-6)
            self.assertAlmostEqual(r2, expected_r2, delta=1e-6)

        linear = 2.0 * t + 1e9 + 3.0
        self.assertAlmostEqual(qflux.rolling_slope(linear, window)[-1], 2.0)
        self.assertAlmostEqual(qflux.rolling_r2(linear, window)[-1], 1.0)

    def test_invalid_arguments_raise_native_error(self):
        with self.assertRaises(ValueError):
            qflux.sma(np.arange(5.0), 0)
        with self.assertRaises(RFluxError) as caught:
            qflux.sma(np.arange(5.0), 6)
        self.assertEqual(caught.exception.status, 1)
        with self.assertRaises(ValueError):
            qflux.sma(np.arange(5.0).reshape(1, 5), 2)


class TestStatsAndRanking(unittest.TestCase):
    def test_describe_quantile_pearson(self):
        values = np.arange(1.0, 11.0)
        summary = qflux.describe(values)
        self.assertEqual(summary.n_valid, 10)
        self.assertAlmostEqual(summary.mean, 5.5)
        self.assertAlmostEqual(summary.variance, 55.0 / 6.0)
        self.assertAlmostEqual(summary.skewness, 0.0)
        self.assertAlmostEqual(summary.excess_kurtosis, -1.2)
        self.assertAlmostEqual(qflux.quantile(values, 0.25), 3.25)
        self.assertAlmostEqual(qflux.pearson(values, 3.0 * values + 2.0), 1.0)

    def test_correlation_matrix_pairwise_complete(self):
        panel = np.array(
            [
                [1.0, 2.0, 5.0],
                [2.0, 4.0, 4.0],
                [3.0, np.nan, 3.0],
                [np.nan, 8.0, 2.0],
                [5.0, 10.0, 1.0],
            ]
        )
        corr = qflux.correlation_matrix(panel)
        self.assertAlmostEqual(corr[0, 0], 1.0)
        self.assertAlmostEqual(corr[0, 1], 1.0)
        self.assertAlmostEqual(corr[1, 0], 1.0)
        self.assertAlmostEqual(corr[2, 2], 1.0)
        np.testing.assert_allclose(corr, corr.T)

    def test_sharpe_sortino_var_cvar(self):
        self.assertAlmostEqual(
            qflux.sharpe(np.array([0.02, -0.01]), 252.0, 0.0),
            3.7416573868,
            places=8,
        )
        self.assertAlmostEqual(
            qflux.sortino(np.array([0.02, -0.01]), 252.0, 0.0),
            11.2249721603,
            places=8,
        )
        risk = qflux.var_cvar(np.array([-0.1, -0.05, 0.0, 0.05, 0.1]), 0.8)
        self.assertAlmostEqual(risk.var, 0.06)
        self.assertAlmostEqual(risk.cvar, 0.1)

    def test_rank_zscore_neutralize(self):
        values = np.array([3.0, 1.0, 1.0, np.nan, 5.0])
        np.testing.assert_allclose(
            qflux.rank(values), [3.0, 1.5, 1.5, np.nan, 4.0]
        )
        percentile = qflux.percentile_rank(values)
        np.testing.assert_allclose(
            percentile, [(3.0 - 0.5) / 4, (1.5 - 0.5) / 4,
                         (1.5 - 0.5) / 4, np.nan, (4.0 - 0.5) / 4]
        )
        z = qflux.zscore(np.array([1.0, 2.0, 3.0]))
        self.assertAlmostEqual(z[0], -1.22474487139)
        winsorized = qflux.winsorize(np.arange(1.0, 11.0), 0.1, 0.9)
        self.assertAlmostEqual(winsorized[0], 1.9)
        self.assertAlmostEqual(winsorized[-1], 9.1)
        neutral = qflux.neutralize(
            np.array([1.0, 2.0, 10.0, 12.0]), np.array([0, 0, 1, 1])
        )
        np.testing.assert_allclose(neutral, [-0.5, 0.5, -1.0, 1.0])
        with self.assertRaises(ValueError):
            qflux.neutralize(
                np.array([1.0, 2.0]), np.array([0, 2**32], dtype=np.int64)
            )


if __name__ == "__main__":
    unittest.main()
