import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from rflux import apps, cli


class TestPortfolioRisk(unittest.TestCase):
    def test_known_two_asset_contributions(self):
        returns = np.array(
            [
                [0.01, 0.02],
                [0.02, -0.01],
                [-0.01, 0.01],
                [0.03, 0.00],
            ]
        )
        weights = np.array([0.6, 0.4])
        report = apps.portfolio_risk(
            returns,
            weights,
            names=["A", "B"],
            periods_per_year=252.0,
        )
        covariance = np.cov(returns, rowvar=False, ddof=1)
        variance = float(weights @ covariance @ weights)
        annual_vol = float(np.sqrt(variance * 252.0))
        marginal = covariance @ weights / np.sqrt(variance)
        components = weights * marginal * np.sqrt(252.0)
        self.assertAlmostEqual(
            report["portfolio"]["annualized_volatility"], annual_vol, places=12
        )
        self.assertAlmostEqual(
            report["portfolio"]["metrics"]["annualized_volatility"],
            annual_vol,
            places=12,
        )
        for name, expected in zip(("A", "B"), components):
            self.assertAlmostEqual(
                report["asset_risk"][name]["component_volatility"],
                expected,
                places=12,
            )
        self.assertAlmostEqual(
            sum(
                item["component_volatility"]
                for item in report["asset_risk"].values()
            ),
            annual_vol,
            places=12,
        )
        self.assertAlmostEqual(
            sum(
                item["percent_contribution"]
                for item in report["asset_risk"].values()
            ),
            1.0,
            places=12,
        )
        self.assertEqual(report["input"]["complete_observations"], 4)
        self.assertEqual(report["input"]["dropped_observations"], 0)
        json.dumps(report, allow_nan=False)

    def test_complete_cases_and_long_short(self):
        returns = np.array(
            [
                [0.01, 0.02],
                [0.02, np.nan],
                [-0.01, 0.01],
                [0.03, 0.00],
                [0.00, 0.02],
            ]
        )
        report = apps.portfolio_risk(
            returns, [1.0, -1.0], names=["A", "B"]
        )
        self.assertEqual(report["input"]["complete_observations"], 4)
        self.assertEqual(report["input"]["dropped_observations"], 1)
        self.assertAlmostEqual(report["input"]["net_exposure"], 0.0)
        self.assertAlmostEqual(report["input"]["gross_exposure"], 2.0)
        components = [
            item["component_volatility"]
            for item in report["asset_risk"].values()
        ]
        self.assertAlmostEqual(
            sum(components),
            report["portfolio"]["annualized_volatility"],
            places=12,
        )
        json.dumps(report, allow_nan=False)

    def test_invalid_inputs(self):
        with self.assertRaises(ValueError):
            apps.portfolio_risk([[1.0, 2.0]], [0.5, 0.5])
        with self.assertRaises(ValueError):
            apps.portfolio_risk([[1.0, 2.0], [2.0, 1.0]], [0.5])
        with self.assertRaises(ValueError):
            apps.portfolio_risk(
                [[1.0, 2.0], [2.0, 1.0]], [0.0, 0.0]
            )
        with self.assertRaises(ValueError):
            apps.portfolio_risk(
                [[np.nan, 2.0], [np.nan, 1.0]], [0.5, 0.5]
            )


class TestRegimeAttribution(unittest.TestCase):
    def test_known_states_and_transitions(self):
        returns = np.array([0.01, -0.02, 0.03, -0.01, 0.02])
        labels = np.array([0, 0, 1, 1, 0], dtype=np.int32)
        report = apps.regime_attribution(returns, labels)
        self.assertEqual(report["input"]["states"], 2)
        self.assertAlmostEqual(report["states"]["0"]["mean_return"], 0.01 / 3)
        self.assertAlmostEqual(report["states"]["1"]["mean_return"], 0.01)
        self.assertAlmostEqual(report["states"]["0"]["period_share"], 0.6)
        self.assertAlmostEqual(report["states"]["1"]["period_share"], 0.4)
        self.assertAlmostEqual(
            report["states"]["0"]["share_of_total_sum"], 1.0 / 3.0
        )
        self.assertAlmostEqual(
            report["states"]["1"]["share_of_total_sum"], 2.0 / 3.0
        )
        self.assertEqual(
            report["transitions"]["counts"], [[1, 1], [1, 1]]
        )
        self.assertEqual(
            report["transitions"]["probabilities"], [[0.5, 0.5], [0.5, 0.5]]
        )
        json.dumps(report, allow_nan=False)

    def test_missing_returns_and_labels(self):
        returns = np.array([0.01, np.nan, 0.03, -0.01, 0.02])
        labels = np.array([0, 0, -1, 1, 1], dtype=np.int32)
        report = apps.regime_attribution(returns, labels)
        self.assertEqual(report["input"]["labelled_observations"], 4)
        self.assertEqual(report["states"]["0"]["n_returns"], 1)
        self.assertIsNone(report["states"]["0"]["annualized_volatility"])
        self.assertEqual(report["transitions"]["counts"], [[1, 0], [0, 1]])
        json.dumps(report, allow_nan=False)

    def test_invalid_labels(self):
        with self.assertRaises(TypeError):
            apps.regime_attribution([0.1, 0.2], [0.0, 1.0])
        with self.assertRaises(ValueError):
            apps.regime_attribution([0.1, 0.2], [0, -2])
        with self.assertRaises(ValueError):
            apps.regime_attribution([0.1, 0.2], [0, 1, 2])


class TestRiskCommand(unittest.TestCase):
    def test_json_risk_command(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "prices.csv"
            rows = ["date,A,B"]
            for index in range(20):
                rows.append(
                    f"2024-01-{index + 1:02d},{100 + index},{50 + index % 3}"
                )
            path.write_text("\n".join(rows) + "\n", encoding="utf-8")
            stdout = io.StringIO()
            stderr = io.StringIO()
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                status = cli.main(
                    [
                        "risk",
                        str(path),
                        "--weights",
                        "0.6,0.4",
                        "--format",
                        "json",
                    ]
                )
        self.assertEqual(status, 0, stderr.getvalue())
        report = json.loads(stdout.getvalue())
        self.assertEqual(report["input"]["asset_names"], ["A", "B"])
        self.assertEqual(report["input"]["observations"], 20)
        self.assertEqual(report["input"]["complete_observations"], 19)
        self.assertEqual(report["input"]["weights"], [0.6, 0.4])
        json.dumps(report, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
