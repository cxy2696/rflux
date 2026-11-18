import json
import unittest

import numpy as np

from rflux import advice, portfoliolab, qflux


class TestTradeAdvice(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(5)
        self.prices = 100.0 * np.cumprod(
            1.0 + rng.normal(0.0004, 0.01, size=(260, 4)), axis=0
        )
        self.names = ["AAA", "BBB", "CCC", "DDD"]

    def _high_vol_prices(self, seed: int, n: int = 500) -> np.ndarray:
        rng = np.random.default_rng(seed)
        returns = rng.normal(0.0, 0.35 / np.sqrt(252.0), size=(n, 3))
        return 100.0 * np.cumprod(1.0 + returns, axis=0)

    def test_allocation_is_long_only_and_capped(self):
        report = advice.trade_advice(
            self.prices,
            names=self.names,
            include_hmm=False,
            max_weight=0.40,
        )
        allocation = report["allocation"]
        target = np.asarray(allocation["target_weights"])
        self.assertTrue(np.all(target >= 0.0))
        self.assertLessEqual(float(target.max()), 0.40 + 1e-12)
        self.assertLessEqual(float(target.sum()), 1.0 + 1e-12)
        self.assertAlmostEqual(
            float(target.sum() + allocation["cash_weight"]), 1.0, places=12
        )
        self.assertEqual(
            [item["asset"] for item in allocation["actions"]], self.names
        )

    def test_base_weights_use_hrp_on_complete_returns(self):
        report = advice.trade_advice(
            self.prices, names=self.names, include_hmm=False, max_weight=1.0
        )
        returns = qflux.returns_panel(self.prices)
        complete = returns[np.all(np.isfinite(returns), axis=1)]
        expected = portfoliolab.hrp_weights(np.cov(complete, rowvar=False))
        np.testing.assert_allclose(
            report["allocation"]["base_weights"], expected, rtol=1e-10, atol=1e-12
        )

    def test_current_weights_drive_actions(self):
        report = advice.trade_advice(
            self.prices,
            names=self.names,
            current_weights=[0.25, 0.25, 0.25, 0.25],
            include_hmm=False,
        )
        actions = report["allocation"]["actions"]
        self.assertEqual(len(actions), 4)
        self.assertTrue(all("delta" in item for item in actions))
        self.assertTrue(all("reason" in item for item in actions))
        self.assertEqual(report["input"]["current_weights_source"], "provided")

    def test_invalid_cap_is_rejected(self):
        with self.assertRaises(ValueError):
            advice.trade_advice(
                self.prices, names=self.names, max_weight=0.10, include_hmm=False
            )

    def test_position_sizing_and_stop_reference(self):
        report = advice.trade_advice(
            self.prices,
            names=self.names,
            include_hmm=False,
            portfolio_value=1_000_000.0,
        )
        for item in report["allocation"]["actions"]:
            self.assertIsNotNone(item["last_price"])
            self.assertIsNotNone(item["reference_stop"])
            self.assertLess(item["reference_stop"], item["last_price"])
            self.assertGreaterEqual(item["recommended_shares"], 0)
            if item["risk_limited_shares"] is not None:
                self.assertLessEqual(
                    item["recommended_shares"], item["target_shares"]
                )

    def test_volatility_scale_uses_portfolio_not_market_vol(self):
        # One quiet and one volatile uncorrelated asset.  HRP concentrates on
        # the quiet asset, so the proposed portfolio is below the 15% target
        # even though the equal-weight market volatility is above it.
        rng = np.random.default_rng(11)
        n = 500
        returns = np.column_stack([
            rng.normal(0.0002, 0.05 / np.sqrt(252.0), n),
            rng.normal(0.0002, 0.40 / np.sqrt(252.0), n),
        ])
        prices = 100.0 * np.cumprod(1.0 + returns, axis=0)
        report = advice.trade_advice(
            prices, names=["QUIET", "VOLATILE"], include_hmm=False,
            max_weight=1.0, target_volatility=0.15,
        )
        self.assertGreater(report["risk"]["market_annualized_volatility"], 0.15)
        self.assertLessEqual(
            report["risk"]["portfolio_annualized_volatility"], 0.15 + 1e-12
        )
        self.assertEqual(report["allocation"]["volatility_scale"], 1.0)
        self.assertEqual(report["allocation"]["exposure"], 1.0)

    def test_volatility_scale_hits_target_on_high_vol_portfolio(self):
        prices = self._high_vol_prices(12)
        target = 0.12
        report = advice.trade_advice(
            prices, names=["A", "B", "C"], include_hmm=False,
            max_weight=1.0, target_volatility=target,
        )
        exposure = report["allocation"]["exposure"]
        portfolio_vol = report["risk"]["portfolio_annualized_volatility"]
        self.assertLessEqual(exposure, 1.0)
        self.assertAlmostEqual(
            exposure * portfolio_vol, target, places=6
        )

    def test_expected_turnover_includes_cash(self):
        prices = self._high_vol_prices(13)
        report = advice.trade_advice(
            prices, names=["A", "B", "C"], include_hmm=False,
            max_weight=1.0, target_volatility=0.05,
        )
        allocation = report["allocation"]
        target = np.asarray(allocation["target_weights"])
        current = np.full(3, 1.0 / 3.0)
        expected = 0.5 * (
            np.abs(target - current).sum()
            + abs(allocation["cash_weight"] - 0.0)
        )
        self.assertAlmostEqual(
            allocation["expected_turnover"], float(expected), places=12
        )

    def test_momentum_requires_the_exact_calendar_base(self):
        rng = np.random.default_rng(14)
        prices = 100.0 * np.cumprod(
            1.0 + rng.normal(0.0002, 0.01, size=(120, 2)), axis=0
        )
        prices[-21, 0] = np.nan
        report = advice.trade_advice(
            prices, names=["A", "B"], include_hmm=False, max_weight=1.0
        )
        self.assertIsNone(report["signals"]["A"]["momentum_20"])
        self.assertIsNotNone(report["signals"]["B"]["momentum_20"])

    def test_regime_section_is_json_safe(self):
        report = advice.trade_advice(
            self.prices,
            names=self.names,
            include_hmm=True,
            hmm_states=2,
            hmm_max_iter=50,
            hmm_n_init=1,
            seed=3,
        )
        self.assertIsNotNone(report["regime"])
        self.assertTrue(report["regime"]["latest_filtered"])
        json.dumps(report, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
