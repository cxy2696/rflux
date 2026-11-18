import unittest

import numpy as np

from rflux import execution, factorlab, mllab, portfoliolab, regimelab
from rflux.risk import tail_dependence


class TestFactorLab(unittest.TestCase):
    def setUp(self):
        self.close = np.array(
            [[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [4.0, 40.0]]
        )
        self.volume = np.array(
            [[1.0, 2.0], [1.0, 2.0], [1.0, 2.0], [2.0, 4.0]]
        )

    def test_momentum_vwap_amihud(self):
        momentum = factorlab.momentum(self.close, 2)
        self.assertTrue(np.isnan(momentum[:2]).all())
        np.testing.assert_allclose(momentum[2:], [[2.0, 2.0], [1.0, 1.0]])

        vwap = factorlab.vwap_deviation(self.close, self.volume, 2)
        self.assertTrue(np.isnan(vwap[0]).all())
        np.testing.assert_allclose(
            vwap[1:], [[1 / 3, 1 / 3], [0.2, 0.2], [1 / 11, 1 / 11]]
        )
        amihud = factorlab.amihud(self.close, self.volume, 2)
        self.assertTrue(np.isnan(amihud[:2]).all())
        self.assertTrue(np.all(amihud[2:] > 0.0))

    def test_volume_reversal(self):
        values = factorlab.volume_reversal(self.close, self.volume, 2)
        self.assertTrue(np.isnan(values[:2]).all())
        np.testing.assert_allclose(values[2:], [[-2.0, -2.0], [-4 / 3, -4 / 3]])

    def test_information_coefficient_and_turnover(self):
        forward = self.close * 1.1
        ic = factorlab.information_coefficient(self.close, forward)
        np.testing.assert_allclose(ic, 1.0)
        turnover = factorlab.factor_turnover(
            np.array([[0.5, 0.5], [0.6, 0.4], [0.6, 0.4]])
        )
        np.testing.assert_allclose(turnover, [0.0, 0.1, 0.0])





class TestPortfolioLab(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(7)
        values = rng.normal(size=(300, 6))
        self.covariance = np.cov(values, rowvar=False)

    def test_hrp_two_asset_matches_inverse_variance(self):
        covariance = np.array([[0.04, 0.006], [0.006, 0.09]])
        expected = (1.0 / np.diag(covariance))
        expected /= expected.sum()
        np.testing.assert_allclose(
            portfoliolab.hrp_weights(covariance), expected, atol=1e-12
        )

    def test_hrp_and_nco_weights(self):
        weights = portfoliolab.hrp_weights(self.covariance)
        self.assertEqual(weights.shape, (6,))
        self.assertGreaterEqual(float(weights.min()), 0.0)
        self.assertAlmostEqual(float(weights.sum()), 1.0, places=10)
        nco = portfoliolab.nco_weights(self.covariance)
        self.assertAlmostEqual(float(nco.sum()), 1.0, places=10)
        self.assertTrue(np.all(nco >= 0.0))
        parity = portfoliolab.risk_parity_weights(self.covariance)
        inverse = 1.0 / np.diag(self.covariance)
        np.testing.assert_allclose(parity, inverse / inverse.sum(), atol=1e-12)

    def test_rebalance_threshold(self):
        current = np.array([0.5, 0.5])
        self.assertTrue(
            np.array_equal(
                portfoliolab.rebalance_targets(current, [0.51, 0.49], threshold=0.05),
                current,
            )
        )
        np.testing.assert_allclose(
            portfoliolab.rebalance_targets(current, [0.7, 0.3], threshold=0.05),
            [0.7, 0.3],
        )


    def test_rebalance_targets_gates_each_period_independently(self):
        current = np.array([[0.5, 0.5], [0.5, 0.5]])
        target = np.array([[0.51, 0.49], [0.7, 0.3]])
        gated = portfoliolab.rebalance_targets(current, target, threshold=0.05)
        np.testing.assert_allclose(gated[0], current[0])
        np.testing.assert_allclose(gated[1], target[1])


class TestExecution(unittest.TestCase):
    def test_linear_and_risk_averse_schedules(self):
        linear = execution.almgren_chriss(
            10.0, slices=4, horizon=1.0, temporary_impact=1.0
        )
        np.testing.assert_allclose(linear.trades, [2.5] * 4)
        self.assertAlmostEqual(linear.expected_cost, 100.0)
        curved = execution.almgren_chriss(
            1000.0,
            slices=10,
            horizon=1.0,
            temporary_impact=1.0,
            permanent_impact=0.1,
            volatility=0.02,
            risk_aversion=2.0,
        )
        self.assertAlmostEqual(float(curved.trades.sum()), 1000.0)
        self.assertGreater(curved.trades[0], curved.trades[-1])
        self.assertGreater(curved.kappa, 0.0)

    def test_order_book_metrics(self):
        metrics = execution.order_book_metrics([99.9], [100.0], [100.1], [200.0])
        np.testing.assert_allclose(metrics["spread"], [0.2])
        np.testing.assert_allclose(metrics["mid"], [100.0])
        np.testing.assert_allclose(metrics["microprice"], [99.9666666667])
        np.testing.assert_allclose(metrics["imbalance"], [-1 / 3])


class TestMLLab(unittest.TestCase):
    def test_huber_resists_outlier(self):
        rng = np.random.default_rng(0)
        features = rng.normal(size=(100, 3))
        target = 1.0 + 2.0 * features[:, 0] - features[:, 1] + rng.normal(
            scale=0.1, size=100
        )
        target[0] = 100.0
        fit = mllab.huber_regression(features, target)
        self.assertTrue(fit.converged)
        self.assertAlmostEqual(fit.intercept, 1.0, delta=0.1)
        self.assertAlmostEqual(fit.coefficients[0], 2.0, delta=0.1)
        self.assertAlmostEqual(fit.coefficients[1], -1.0, delta=0.1)

    def test_cvar_and_graph_smooth(self):
        self.assertAlmostEqual(mllab.cvar_loss([0.01, -0.02, -0.05, 0.03]), 0.05)
        smoothed = mllab.graph_smooth(
            [1.0, 0.0, 1.0], [[0, 1, 1], [1, 0, 1], [1, 1, 0]]
        )
        self.assertEqual(smoothed.shape, (3,))
        self.assertAlmostEqual(float(smoothed[0]), float(smoothed[2]))


class TestRegimeAndRisk(unittest.TestCase):
    def test_sticky_regime_suppresses_flicker(self):
        class FakeModel:
            def filtered(self, observations):
                n = len(observations)
                probabilities = np.tile([0.5, 0.5], (n, 1))
                probabilities[10:20] = [0.7, 0.3]
                probabilities[20:30] = [0.5, 0.5]
                probabilities[30:] = [0.2, 0.8]
                return probabilities

        result = regimelab.sticky_regime(
            FakeModel(), np.zeros(40), min_duration=10,
            entry_threshold=0.6, exit_threshold=0.4,
        )
        self.assertEqual(result.labels.shape, (40,))
        self.assertEqual(result.labels[0], 0)
        self.assertEqual(result.labels[39], 1)
        self.assertEqual(result.n_switches, 1)

    def test_tail_dependence_symmetrises_conditionals(self):
        rng = np.random.default_rng(9)
        x = rng.normal(size=800)
        y = 0.8 * x + 0.6 * rng.normal(size=800)
        y[y < -0.6] = np.nan  # asymmetric missingness in the lower tail
        values = np.column_stack([x, y])
        q = 0.1
        lower = tail_dependence(values, q=q)
        np.testing.assert_allclose(lower, lower.T, equal_nan=True)
        # Reconstruct the two conditionals exactly as the estimator does.
        events = np.zeros_like(values, dtype=bool)
        for column in range(values.shape[1]):
            valid = np.isfinite(values[:, column])
            cutoff = float(np.quantile(values[valid, column], q))
            events[:, column] = valid & (values[:, column] <= cutoff)
        joint = int((events[:, 0] & events[:, 1]).sum())
        count_x = int(events[:, 0].sum())
        count_y = int(events[:, 1].sum())
        expected = 0.5 * (joint / count_x + joint / count_y)
        self.assertAlmostEqual(lower[0, 1], expected, places=12)

    def test_tail_dependence(self):
        rng = np.random.default_rng(1)
        values = rng.normal(size=(1000, 3))
        values[:, 1] = 0.8 * values[:, 0] + 0.2 * values[:, 1]
        lower = tail_dependence(values)
        upper = tail_dependence(values, tail="upper")
        self.assertEqual(lower.shape, (3, 3))
        np.testing.assert_allclose(lower, lower.T)
        np.testing.assert_allclose(np.diag(lower), 1.0)
        self.assertGreater(lower[0, 1], 0.5)
        np.testing.assert_allclose(upper, upper.T)


if __name__ == "__main__":
    unittest.main()
