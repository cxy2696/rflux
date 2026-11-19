import unittest

import numpy as np

from rflux import backtest, metrics
from rflux import RFluxError


class TestMetricsAndBacktest(unittest.TestCase):
    def test_performance_summary(self):
        returns = np.array([0.01, 0.02, -0.005, 0.0, 0.015])
        summary = metrics.performance_summary(returns, periods_per_year=252)
        self.assertEqual(summary.n_periods, 5)
        self.assertEqual(summary.elapsed_periods, 5)
        self.assertTrue(np.isfinite(summary.total_return))
        self.assertTrue(np.isfinite(summary.annualized_volatility))
        self.assertLessEqual(summary.max_drawdown, 0.0)
        self.assertTrue(0.0 <= summary.hit_rate <= 1.0)
        self.assertTrue(np.isfinite(summary.var))
        self.assertTrue(np.isfinite(summary.cvar))
        with self.assertRaises(ValueError):
            metrics.performance_summary([0.01])
        with self.assertRaises(ValueError):
            metrics.performance_summary([0.01, 0.02], periods_per_year=0.0)
        with self.assertRaises(ValueError):
            metrics.performance_summary([0.01, 0.02], confidence=1.0)
        leveraged = metrics.performance_summary([0.01, -1.0])
        self.assertAlmostEqual(leveraged.total_return, -1.0)
        self.assertTrue(np.isnan(leveraged.annualized_return))
        with self.assertRaises(ValueError):
            metrics.equity_curve([0.01, -1.0])
        first_loss = metrics.performance_summary([-0.2, 0.1, 0.1])
        self.assertAlmostEqual(first_loss.max_drawdown, -0.2)
        gapped = metrics.performance_summary(
            [0.01, np.nan, 0.01, np.nan, np.nan, np.nan, np.nan]
        )
        self.assertEqual(gapped.n_periods, 2)
        self.assertEqual(gapped.elapsed_periods, 7)
        expected = (1.0201 ** (252.0 / 7.0)) - 1.0
        self.assertAlmostEqual(gapped.annualized_return, expected, places=12)

    def test_equity_and_forward_returns(self):
        curve = metrics.equity_curve(np.array([0.1, -0.1]))
        np.testing.assert_allclose(curve, [1.1, 0.99], atol=1e-12)
        prices = np.array(
            [[100.0, 50.0], [110.0, 55.0], [121.0, 60.5]]
        )
        forward = backtest.forward_returns(prices, periods=1)
        np.testing.assert_allclose(forward[:2], 0.1, atol=1e-12)
        self.assertTrue(np.isnan(forward[2]).all())
        zero_prices = np.array([[0.0], [1.0], [2.0]])
        zero_forward = backtest.forward_returns(zero_prices, periods=1)
        self.assertTrue(np.isnan(zero_forward[0, 0]))
        self.assertAlmostEqual(zero_forward[1, 0], 1.0)
        inf_prices = np.array([[1.0], [np.inf], [3.0], [4.0]])
        inf_forward = backtest.forward_returns(inf_prices, periods=1)
        self.assertTrue(np.isnan(inf_forward[0, 0]))
        self.assertTrue(np.isnan(inf_forward[1, 0]))
        self.assertAlmostEqual(inf_forward[2, 0], 1.0 / 3.0)
        self.assertTrue(np.isnan(inf_forward[3, 0]))
        for empty in (np.empty((0, 3)), np.empty((3, 0)), np.empty((0,))):
            result = backtest.forward_returns(empty, periods=1)
            self.assertEqual(result.shape, empty.shape)
            self.assertTrue(np.isnan(result).all())

    def test_factor_ic_and_quantiles(self):
        factor = np.array(
            [[1.0, 2.0, 3.0, 4.0], [4.0, 3.0, 2.0, 1.0]]
        )
        forward = np.array(
            [[0.01, 0.02, 0.03, 0.04], [0.04, 0.03, 0.02, 0.01]]
        )
        ic = backtest.factor_ic(factor, forward)
        np.testing.assert_allclose(ic, [1.0, 1.0])
        summary = backtest.summarize_ic(ic)
        self.assertAlmostEqual(summary.mean, 1.0)
        self.assertEqual(summary.n_periods, 2)

        quantiles = backtest.quantile_portfolio_returns(factor, forward, 2)
        self.assertEqual(quantiles.shape, (2, 2))
        np.testing.assert_allclose(quantiles[0], [0.015, 0.015])
        np.testing.assert_allclose(quantiles[1], [0.035, 0.035])

        # More buckets than eligible assets leaves the trailing buckets empty.
        sparse = backtest.quantile_portfolio_returns(
            np.array([[1.0, 2.0]]), np.array([[0.1, 0.2]]), 4
        )
        self.assertEqual(sparse.shape, (4, 1))
        np.testing.assert_allclose(sparse[:2, 0], [0.1, 0.2])
        self.assertTrue(np.isnan(sparse[2:, 0]).all())

    def test_long_short_backtest(self):
        factor = np.array(
            [[1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0]]
        )
        forward = np.array(
            [[0.01, 0.02, 0.03, 0.04], [0.01, 0.02, 0.03, 0.04]]
        )
        result = backtest.long_short_backtest(
            factor, forward, n_quantiles=2, cost_bps=10.0
        )
        np.testing.assert_allclose(result.gross_returns, [0.02, 0.02])
        np.testing.assert_allclose(result.turnover, [1.0, 0.0])
        np.testing.assert_allclose(result.costs, [0.001, 0.0])
        np.testing.assert_allclose(result.net_returns, [0.019, 0.02])
        self.assertEqual(result.equity.shape, (3,))
        self.assertIsNotNone(result.metrics)

        leveraged_factor = np.array(
            [[1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0]]
        )
        leveraged_forward = np.array(
            [[3.0, 3.0, -0.2, -0.2], [3.0, 3.0, -0.2, -0.2]]
        )
        leveraged = backtest.long_short_backtest(
            leveraged_factor, leveraged_forward, n_quantiles=2
        )
        np.testing.assert_allclose(leveraged.gross_returns, [-3.2, -3.2])
        self.assertIsNotNone(leveraged.metrics)
        self.assertTrue(np.isnan(leveraged.metrics.annualized_return))

    def test_integer_arguments_are_not_truncated(self):
        prices = np.array([100.0, 101.0, 102.0, 103.0])
        with self.assertRaises(TypeError):
            backtest.forward_returns(prices, periods=1.5)
        factor = np.array([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]])
        forward = np.array([[0.1, 0.2, 0.3], [0.3, 0.2, 0.1]])
        with self.assertRaises(TypeError):
            backtest.quantile_portfolio_returns(factor, forward, n_quantiles=2.5)
        with self.assertRaises(TypeError):
            backtest.long_short_backtest(factor, forward, n_quantiles=2.5)

    def test_invalid_panels(self):
        with self.assertRaises(ValueError):
            backtest.factor_ic(np.zeros((2, 2)), np.zeros((3, 2)))
        with self.assertRaises(ValueError):
            backtest.long_short_backtest(
                np.zeros((2, 2)), np.zeros((2, 2)), n_quantiles=1
            )



if __name__ == "__main__":
    unittest.main()
