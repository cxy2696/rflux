import importlib.util
import threading
import time
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
            self.assertIn(".abi3.", module_path.name)

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
