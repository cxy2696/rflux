import json
import unittest

import numpy as np

from rflux import advice


class TestTradeAdvice(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(5)
        self.prices = 100.0 * np.cumprod(
            1.0 + rng.normal(0.0004, 0.01, size=(260, 4)), axis=0
        )
        self.names = ["AAA", "BBB", "CCC", "DDD"]

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
