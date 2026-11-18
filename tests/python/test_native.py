import importlib.util
import threading
import time
import os
import unittest
from pathlib import Path

import numpy as np

import rflux
from rflux import qflux
from rflux._native_backend import native_module

@unittest.skipIf(native_module is None, "native extension is not built")
class TestNativeBridge(unittest.TestCase):
    def test_identity_and_abi(self):
        from rflux import native

        self.assertTrue(native.available)
        self.assertEqual(native.info(), rflux.native_info())
        info = rflux.native_info()
        self.assertTrue(info["available"])
        self.assertEqual(info["version"], rflux.__version__)
        self.assertTrue(info["gil_released"])
        module_path = Path(info["module"])
        if native_module.STABLE_ABI:
            # CPython stable-ABI wheels use the .abi3 suffix on POSIX.  Windows
            # CPython still loads the stable-ABI module from a plain .pyd.
            if os.name == "nt":
                self.assertTrue(module_path.name.endswith(".pyd"))
            else:
                self.assertIn(".abi3.", module_path.name)

    def test_public_facade_covers_every_native_export(self):
        from rflux import native

        for name in native.__all__:
            self.assertTrue(hasattr(native, name), name)
        self.assertIs(native.series, native_module.series)
        self.assertIs(native.GaussianHMM, native_module.GaussianHMM)

    def test_series_into_is_zero_copy(self):
        values = np.arange(20.0)
        out = np.empty_like(values)
        returned = native_module.series_into("rflux_sma", values, out, 3)
        self.assertIs(returned, out)
        expected = qflux.sma(values, 3)
        np.testing.assert_allclose(out, expected, equal_nan=True)

        for wrong_dtype in (np.float32, np.int64):
            output = np.empty(values.size, dtype=wrong_dtype)
            with self.assertRaises(native_module.NativeError):
                native_module.series_into("rflux_sma", values, output, 3)

    def test_native_errors_carry_status(self):
        with self.assertRaises(native_module.NativeError) as caught:
            native_module.series("rflux_sma", np.arange(3.0), 10)
        self.assertEqual(caught.exception.status, 1)

    def test_hmm_native_fit_and_export(self):
        rng = np.random.default_rng(21)
        observations = np.concatenate(
            [rng.normal(-1.0, 0.2, 120), rng.normal(1.0, 0.25, 120)]
        ).reshape(-1, 1)
        model = native_module.GaussianHMM(2, 1)
        model.fit(observations, seed=5, n_init=2)
        self.assertTrue(model.ready)
        info = model.fit_info
        self.assertGreater(info[0], 0)
        filtered = model.filtered(observations)
        self.assertEqual(filtered.shape, (240, 2))
        np.testing.assert_allclose(filtered.sum(axis=1), 1.0, atol=1e-9)
        with self.assertRaises(native_module.NativeError) as caught:
            native_module.GaussianHMM(2, 1).fit_info
        self.assertEqual(caught.exception.status, 4)

    def test_gil_is_released_during_native_kernel(self):
        # A long-running quantile scan gives the concurrent Python thread a
        # chance to make progress.  The assertion is inherently scheduling
        # based, so skip instead of failing on an unusually fast machine.
        values = np.sin(np.linspace(0.0, 200.0, 2_000_000))
        worker_done = threading.Event()
        started = threading.Event()
        result: dict[str, object] = {}
        timing: dict[str, float] = {}

        def run_native():
            started.set()
            begin = time.perf_counter()
            result["value"] = native_module.series(
                "rflux_rolling_quantile", values, 512, 0.5
            )
            timing["elapsed"] = time.perf_counter() - begin
            worker_done.set()

        thread = threading.Thread(target=run_native)
        thread.start()
        self.assertTrue(started.wait(timeout=2.0))
        counter = 0
        deadline = time.perf_counter() + 0.08
        while time.perf_counter() < deadline:
            if worker_done.is_set():
                break
            counter += 1
        thread.join(timeout=10.0)
        self.assertFalse(thread.is_alive(), "native kernel did not finish")
        self.assertIn("value", result)
        elapsed = timing.get("elapsed", 0.0)
        if elapsed < 0.05:
            self.skipTest("native kernel completed before concurrency could be measured")
        self.assertGreater(
            counter, 0,
            "main thread made no progress while the native kernel held the GIL",
        )


if __name__ == "__main__":
    unittest.main()


@unittest.skipIf(native_module is None, "native extension is not built")
class TestNativeBufferKernels(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(1234)
        n = 4096
        values = rng.normal(size=n) * 10.0
        values[rng.integers(0, n, 15)] = np.nan
        self.values = values
        self.other = rng.normal(size=n)
        self.other[rng.integers(0, n, 9)] = np.nan

    def test_series_into_matches_series_exactly(self):
        cases = [
            ("rflux_returns", (1,)),
            ("rflux_sma", (32,)),
            ("rflux_rolling_sum", (17,)),
            ("rflux_rolling_std", (33, 1)),
            ("rflux_rolling_min", (9,)),
            ("rflux_rolling_max", (9,)),
            ("rflux_rolling_zscore", (21, 0)),
            ("rflux_rolling_quantile", (64, 0.3)),
            ("rflux_rolling_quantile", (1536, 0.6)),
            ("rflux_ema", (12,)),
            ("rflux_rsi", (14,)),
            ("rflux_drawdown", ()),
            ("rflux_rolling_slope", (25,)),
            ("rflux_rolling_r2", (25,)),
            ("rflux_rank", ()),
            ("rflux_percentile_rank", ()),
            ("rflux_zscore", (1,)),
            ("rflux_winsorize", (0.02, 0.98)),
        ]
        for name, args in cases:
            with self.subTest(name=name, args=args):
                expected = native_module.series(name, self.values, *args)
                out = np.empty_like(self.values)
                returned = native_module.series_into(
                    name, self.values, out, *args
                )
                self.assertIs(returned, out)
                np.testing.assert_array_equal(out, expected)

    def test_pair_series_matches_python_wrappers(self):
        for name, window, extra in (
            ("rflux_rolling_corr", 31, ()),
            ("rflux_rolling_cov", 31, (1,)),
            ("rflux_rolling_beta", 31, ()),
        ):
            with self.subTest(name=name):
                expected = native_module.pair_series(
                    name, self.values, self.other, window, *extra
                )
                wrapper = {
                    "rflux_rolling_corr": qflux.rolling_corr,
                    "rflux_rolling_cov": qflux.rolling_cov,
                    "rflux_rolling_beta": qflux.rolling_beta,
                }[name](self.values, self.other, window, *extra)
                np.testing.assert_array_equal(wrapper, expected)

    def test_multi_output_kernels_match_wrappers(self):
        macd = native_module.macd(self.values)
        reference = qflux.macd(self.values)
        np.testing.assert_array_equal(macd[0], reference.macd)
        np.testing.assert_array_equal(macd[1], reference.signal)
        np.testing.assert_array_equal(macd[2], reference.histogram)

        bands = native_module.bollinger(self.values)
        expected = qflux.bollinger(self.values)
        np.testing.assert_array_equal(bands[0], expected.middle)
        np.testing.assert_array_equal(bands[1], expected.upper)
        np.testing.assert_array_equal(bands[2], expected.lower)

    def test_correlation_matrix_matches_numpy(self):
        panel = np.column_stack(
            [self.values, self.other, np.sin(np.arange(self.values.size))]
        )
        complete = panel[np.all(np.isfinite(panel), axis=1)]
        expected = np.corrcoef(complete, rowvar=False)
        actual = native_module.correlation_matrix(complete)
        np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-12)

    def test_native_hmm_fit_is_deterministic(self):
        rng = np.random.default_rng(77)
        observations = np.concatenate(
            [rng.normal(-0.5, 0.4, 400), rng.normal(0.8, 0.6, 400)]
        ).reshape(-1, 1)
        first = native_module.GaussianHMM(2, 1)
        second = native_module.GaussianHMM(2, 1)
        first.fit(observations, n_init=3, max_iter=50, seed=31)
        second.fit(observations, n_init=3, max_iter=50, seed=31)
        np.testing.assert_array_equal(
            first.parameters[2], second.parameters[2]
        )
        np.testing.assert_array_equal(
            first.parameters[3], second.parameters[3]
        )
        self.assertEqual(first.fit_info[2], second.fit_info[2])
