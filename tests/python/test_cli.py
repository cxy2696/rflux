import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

import rflux
from rflux import analysis, cli


class TestDirectAPI(unittest.TestCase):
    def test_top_level_shortcuts(self):
        np.testing.assert_allclose(rflux.sma([1.0, 2.0, 3.0], 2), [np.nan, 1.5, 2.5])
        report = rflux.analyze_prices(
            np.array([1.0, 2.0, 1.5, 2.2, 2.4]),
            names="X",
            include_hmm=False,
            include_network=False,
        )
        self.assertEqual(report["input"]["assets"], 1)
        self.assertAlmostEqual(report["market"]["total_return"], 1.4)

    def test_analysis_optional_sections(self):
        rng = np.random.default_rng(11)
        n = 240
        market = np.concatenate(
            [rng.normal(0.0005, 0.008, n // 2), rng.normal(-0.0005, 0.025, n // 2)]
        )
        common = market[:, None]
        idio = rng.normal(0.0, 0.003, (n, 4))
        returns = common * np.array([0.7, 0.9, 1.0, 1.2])[None, :] + idio
        prices = 100.0 * np.cumprod(1.0 + returns, axis=0)
        report = analysis.analyze_prices(
            prices,
            names=["A", "B", "C", "D"],
            include_hmm=True,
            hmm_states=2,
            include_network=True,
            network_threshold=0.4,
            include_correlation_matrix=True,
        )
        self.assertIn("matrix", report["correlation"])
        self.assertIsNotNone(report["network"])
        self.assertIsNotNone(report["hmm"])
        self.assertEqual(report["errors"], [])
        json.dumps(report, allow_nan=False)

    def test_invalid_panel(self):
        with self.assertRaises(ValueError):
            analysis.analyze_prices(np.array([1.0]))
        with self.assertRaises(ValueError):
            analysis.analyze_prices(np.ones((10, 2)), names=["A"])


class TestCSVReader(unittest.TestCase):
    def test_sort_drop_metadata_and_missing(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "prices.csv"
            path.write_text(
                "date,AAA,BBB,note\n"
                "2024-01-03,102,50,x\n"
                "2024-01-02,100,,y\n"
                "2024-01-04,103,52,z\n",
                encoding="utf-8",
            )
            parsed = cli.read_price_csv(path)
            self.assertEqual(parsed.names, ["AAA", "BBB"])
            self.assertEqual(parsed.dates, ["2024-01-02", "2024-01-03", "2024-01-04"])
            self.assertTrue(np.isnan(parsed.prices[0, 1]))
            self.assertIn("note", parsed.dropped_columns)
            self.assertTrue(parsed.warnings)

    def test_duplicate_dates_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "prices.csv"
            path.write_text(
                "date,AAA\n2024-01-02,100\n2024-01-02,101\n",
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                cli.read_price_csv(path)

    def test_non_numeric_data_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "prices.csv"
            path.write_text(
                "date,AAA\n2024-01-02,100\n2024-01-03,bad\n",
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                cli.read_price_csv(path)


class TestCommandLine(unittest.TestCase):
    def _run(self, argv):
        stdout = io.StringIO()
        stderr = io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            status = cli.main(argv)
        return status, stdout.getvalue(), stderr.getvalue()

    def test_version_and_doctor(self):
        status, output, _ = self._run(["version", "--json"])
        self.assertEqual(status, 0)
        payload = json.loads(output)
        self.assertEqual(payload["version"], "0.7.0")
        self.assertEqual(payload["native_version"], "0.7.0")

        status, output, _ = self._run(["doctor", "--json"])
        self.assertEqual(status, 0)
        payload = json.loads(output)
        self.assertTrue(payload["native"]["loaded"])
        self.assertIn("native_extension", payload)

    def test_config_validate(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.toml"
            path.write_text(
                "[model]\n"
                "n_states = 2\n"
                "n_features = 1\n"
                "\n[model.initial]\n"
                "start_prob = [0.6, 0.4]\n"
                "transition = [[0.9, 0.1], [0.2, 0.8]]\n"
                "means = [[-1.0], [1.0]]\n"
                "variances = [[1.0], [1.0]]\n",
                encoding="utf-8",
            )
            status, output, error = self._run(
                ["config", str(path), "--build", "--json"]
            )
        self.assertEqual(status, 0, error)
        payload = json.loads(output)
        self.assertTrue(payload["valid"])
        self.assertTrue(payload["constructed"])
        self.assertEqual(payload["n_states"], 2)

    def test_analyze_json(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "prices.csv"
            path.write_text(
                "date,AAA,BBB\n"
                "2024-01-02,100,50\n"
                "2024-01-03,101,49\n"
                "2024-01-04,99,51\n"
                "2024-01-05,102,52\n"
                "2024-01-08,103,53\n",
                encoding="utf-8",
            )
            status, output, error = self._run(
                [
                    "analyze",
                    str(path),
                    "--no-hmm",
                    "--no-network",
                    "--format",
                    "json",
                ]
            )
        self.assertEqual(status, 0, error)
        report = json.loads(output)
        self.assertEqual(report["input"]["assets"], 2)
        self.assertEqual(report["input"]["asset_names"], ["AAA", "BBB"])
        self.assertIsNone(report["hmm"])
        self.assertIsNone(report["network"])

    def test_analyze_sample_text(self):
        sample = Path(__file__).resolve().parents[2] / "examples/data/sample_prices.csv"
        status, output, error = self._run(
            ["analyze", str(sample), "--no-hmm", "--format", "text"]
        )
        self.assertEqual(status, 0, error)
        self.assertIn("asset", output)
        self.assertIn("network:", output)

    def test_missing_file(self):
        status, _, error = self._run(["analyze", "/does/not/exist.csv"])
        self.assertEqual(status, 1)
        self.assertIn("error:", error)

    def test_output_file_and_invalid_arguments(self):
        sample = (
            Path(__file__).resolve().parents[2]
            / "examples/data/sample_prices.csv"
        )
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "report.txt"
            status, output, error = self._run(
                [
                    "analyze",
                    str(sample),
                    "--no-hmm",
                    "--no-network",
                    "--output",
                    str(destination),
                ]
            )
            self.assertEqual(status, 0, error)
            self.assertIn("wrote", output)
            self.assertIn("rflux", destination.read_text(encoding="utf-8"))
        status, _, error = self._run(
            ["analyze", str(sample), "--periods-per-year", "0"]
        )
        self.assertEqual(status, 1)
        self.assertIn("error:", error)


if __name__ == "__main__":
    unittest.main()
