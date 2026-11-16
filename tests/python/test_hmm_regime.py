import copy
import ctypes
import tempfile
import unittest
from pathlib import Path

import numpy as np

from rflux import regimelab
from rflux._ffi import HMMOptions, RFluxError, check, get_lib


def make_observations(seed=4, n=500):
    rng = np.random.default_rng(seed)
    return np.concatenate(
        [
            rng.normal(-1.0, 0.25, n // 2),
            rng.normal(1.0, 0.3, n // 2),
        ]
    )


class TestGaussianHMM(unittest.TestCase):
    def test_options_layout_matches_native(self):
        options = HMMOptions()
        check(
            get_lib().rflux_hmm_options_init(
                ctypes.byref(options), ctypes.sizeof(options)
            ),
            "rflux_hmm_options_init",
        )
        self.assertEqual(int(options.struct_size), ctypes.sizeof(options))
        self.assertEqual(int(options.max_iter), 200)
        self.assertEqual(int(options.n_init), 3)
        self.assertEqual(options.sort_states, 1)

    def test_fit_decode_forecast_and_roundtrip(self):
        observations = make_observations()
        with regimelab.GaussianHMM(2, 1) as model:
            with self.assertRaises(RFluxError) as before_fit:
                _ = model.fit_info
            self.assertEqual(before_fit.exception.status, 4)
            model.fit(
                observations, n_init=3, max_iter=200, seed=7, sort_states=True
            )
            info = model.fit_info
            self.assertGreater(info.iterations, 0)
            self.assertTrue(np.isfinite(info.log_likelihood))
            self.assertTrue(np.isfinite(info.aic))
            self.assertTrue(np.isfinite(info.bic))
            self.assertAlmostEqual(
                model.score(observations), info.log_likelihood, places=8
            )

            parameters = model.parameters
            self.assertEqual(parameters.start_prob.shape, (2,))
            self.assertEqual(parameters.transition.shape, (2, 2))
            self.assertEqual(parameters.means.shape, (2, 1))
            np.testing.assert_allclose(parameters.start_prob.sum(), 1.0)
            np.testing.assert_allclose(parameters.transition.sum(axis=1), 1.0)
            self.assertLess(parameters.means[0, 0], parameters.means[1, 0])
            self.assertTrue(np.all(parameters.variances > 0.0))

            path = model.viterbi(observations)
            self.assertGreater(np.count_nonzero(path[:250] == 0), 245)
            self.assertGreater(np.count_nonzero(path[250:] == 1), 245)

            filtered = model.filtered(observations)
            np.testing.assert_allclose(filtered.sum(axis=1), 1.0, atol=1e-9)
            np.testing.assert_array_equal(
                model.filtered_states(observations), np.argmax(filtered, axis=1)
            )
            prefix = filtered[:200]
            np.testing.assert_allclose(
                model.filtered(observations[:200]), prefix, atol=1e-12
            )
            smoothed = model.smoothed(observations)
            np.testing.assert_allclose(smoothed.sum(axis=1), 1.0, atol=1e-9)

            forecast = model.forecast(filtered[-1], horizon=5)
            np.testing.assert_allclose(forecast.sum(), 1.0, atol=1e-9)

            with tempfile.TemporaryDirectory() as directory:
                path_file = Path(directory) / "model.npz"
                model.save(path_file)
                reloaded = regimelab.GaussianHMM.load(path_file)
                try:
                    np.testing.assert_allclose(
                        reloaded.parameters.means, parameters.means, atol=1e-12
                    )
                    np.testing.assert_allclose(
                        reloaded.parameters.transition,
                        parameters.transition,
                        atol=1e-12,
                    )
                    np.testing.assert_array_equal(
                        reloaded.viterbi(observations), path
                    )
                    self.assertAlmostEqual(
                        reloaded.score(observations),
                        model.score(observations),
                        places=10,
                    )
                finally:
                    reloaded.close()

    def test_parameter_setter_and_known_forecast(self):
        parameters = regimelab.HMMParameters(
            start_prob=np.array([0.6, 0.4]),
            transition=np.array([[0.9, 0.1], [0.2, 0.8]]),
            means=np.array([[-1.0], [1.0]]),
            variances=np.array([[1.0], [1.0]]),
        )
        model = regimelab.GaussianHMM(2, 1)
        try:
            model.set_parameters(parameters)
            forecast = model.forecast([0.6, 0.4], horizon=1)
            np.testing.assert_allclose(forecast, [0.62, 0.38], atol=1e-12)
            path = model.viterbi(np.array([-1.0, -0.9, 1.0, 1.1]))
            np.testing.assert_array_equal(path, [0, 0, 1, 1])
            self.assertAlmostEqual(
                model.conditional_mean([0.25, 0.75]), 0.5
            )
            with self.assertRaises(RFluxError):
                _ = model.fit_info
        finally:
            model.close()
        with self.assertRaises(RuntimeError):
            model.filtered(np.array([0.0]))

    def test_handle_copy_is_rejected_and_clone_is_independent(self):
        parameters = regimelab.HMMParameters(
            start_prob=np.array([0.6, 0.4]),
            transition=np.array([[0.9, 0.1], [0.2, 0.8]]),
            means=np.array([[-1.0], [1.0]]),
            variances=np.array([[1.0], [1.0]]),
        )
        model = regimelab.GaussianHMM(2, 1)
        model.set_parameters(parameters)
        try:
            with self.assertRaises(TypeError):
                copy.copy(model)
            with self.assertRaises(TypeError):
                copy.deepcopy(model)
            duplicate = model.clone()
            try:
                duplicate.close()
                self.assertEqual(model.filtered([0.0]).shape, (1, 2))
            finally:
                duplicate.close()
        finally:
            model.close()
        with self.assertRaises(TypeError):
            regimelab.GaussianHMM(2.9)
        with self.assertRaises(TypeError):
            regimelab.fit_hmm(np.zeros((4, 1)), n_states=2.9)

    def test_structural_zeros_survive_set_and_forecast(self):
        parameters = regimelab.HMMParameters(
            start_prob=np.array([1.0, 0.0]),
            transition=np.array([[1.0, 0.0], [0.0, 1.0]]),
            means=np.array([[-1.0], [1.0]]),
            variances=np.array([[1.0], [1.0]]),
        )
        model = regimelab.GaussianHMM(2, 1)
        try:
            model.set_parameters(parameters)
            stored = model.parameters
            np.testing.assert_array_equal(stored.start_prob, [1.0, 0.0])
            np.testing.assert_array_equal(
                stored.transition, [[1.0, 0.0], [0.0, 1.0]]
            )
            np.testing.assert_array_equal(
                model.forecast([1.0, 0.0], horizon=5), [1.0, 0.0]
            )
            np.testing.assert_array_equal(
                model.filtered(np.array([[-1.0], [1.0]])),
                [[1.0, 0.0], [1.0, 0.0]],
            )
            transition = np.array([[0.9, 0.1], [0.2, 0.8]])
            model.set_parameters(
                regimelab.HMMParameters(
                    start_prob=np.array([0.6, 0.4]),
                    transition=transition,
                    means=np.array([[-1.0], [1.0]]),
                    variances=np.array([[1.0], [1.0]]),
                )
            )
            np.testing.assert_allclose(
                model.forecast([0.6, 0.4], horizon=3),
                np.array([0.6, 0.4]) @ np.linalg.matrix_power(transition, 3),
                atol=1e-12,
            )
        finally:
            model.close()

    def test_invalid_parameter_tables_are_rejected(self):
        model = regimelab.GaussianHMM(2, 1)
        good = regimelab.HMMParameters(
            start_prob=np.array([0.6, 0.4]),
            transition=np.array([[0.9, 0.1], [0.2, 0.8]]),
            means=np.array([[-1.0], [1.0]]),
            variances=np.array([[1.0], [1.0]]),
        )
        try:
            for bad in (
                regimelab.HMMParameters(
                    np.array([0.6, 0.5]), good.transition, good.means,
                    good.variances
                ),
                regimelab.HMMParameters(
                    good.start_prob, np.array([[0.9, 0.2], [0.2, 0.8]]),
                    good.means, good.variances
                ),
                regimelab.HMMParameters(
                    good.start_prob, good.transition, good.means,
                    np.array([[1.0], [0.0]])
                ),
            ):
                with self.assertRaises(RFluxError) as caught:
                    model.set_parameters(bad)
                self.assertEqual(caught.exception.status, 1)
        finally:
            model.close()

    def test_invalid_input(self):
        model = regimelab.GaussianHMM(2, 1)
        try:
            with self.assertRaises(RFluxError) as caught:
                model.fit(np.array([0.1, np.nan, 0.2, 0.3]))
            self.assertEqual(caught.exception.status, 3)
            with self.assertRaises(RFluxError):
                model.fit(np.zeros((1, 1)))
        finally:
            model.close()


class TestStateAlignment(unittest.TestCase):
    def test_permuted_states_are_recovered(self):
        reference = regimelab.HMMParameters(
            start_prob=np.array([0.6, 0.4]),
            transition=np.array([[0.9, 0.1], [0.2, 0.8]]),
            means=np.array([[-1.0], [1.0]]),
            variances=np.array([[1.0], [0.25]]),
        )
        # Swap target state ids 0 and 1.
        target = regimelab.HMMParameters(
            start_prob=np.array([0.4, 0.6]),
            transition=np.array([[0.8, 0.2], [0.1, 0.9]]),
            means=np.array([[1.0], [-1.0]]),
            variances=np.array([[0.25], [1.0]]),
        )
        order = regimelab.match_states(reference, target)
        np.testing.assert_array_equal(order, [1, 0])
        alignment = regimelab.align_parameters(reference, target)
        np.testing.assert_allclose(alignment.order, [1, 0])
        np.testing.assert_allclose(
            alignment.parameters.start_prob, reference.start_prob
        )
        np.testing.assert_allclose(
            alignment.parameters.transition, reference.transition
        )
        np.testing.assert_allclose(
            alignment.parameters.means, reference.means
        )
        np.testing.assert_allclose(
            alignment.parameters.variances, reference.variances
        )
        probabilities = np.array([[0.8, 0.2], [0.1, 0.9]])
        np.testing.assert_allclose(
            regimelab.align_probabilities(probabilities, order),
            probabilities[:, [1, 0]],
        )
        np.testing.assert_array_equal(
            regimelab.align_labels([0, 1, -1], order), [1, 0, -1]
        )

    def test_three_state_assignment_is_exact(self):
        reference = regimelab.HMMParameters(
            start_prob=np.array([0.2, 0.5, 0.3]),
            transition=np.array(
                [[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]]
            ),
            means=np.array([[-2.0], [0.0], [2.0]]),
            variances=np.array([[1.0], [0.5], [2.0]]),
        )
        permutation = np.array([2, 0, 1])
        target = regimelab.HMMParameters(
            start_prob=reference.start_prob[permutation],
            transition=reference.transition[np.ix_(permutation, permutation)],
            means=reference.means[permutation],
            variances=reference.variances[permutation],
        )
        order = regimelab.match_states(reference, target)
        np.testing.assert_array_equal(order, [1, 2, 0])
        alignment = regimelab.align_parameters(reference, target)
        np.testing.assert_allclose(
            alignment.parameters.means, reference.means, atol=1e-12
        )
        np.testing.assert_allclose(
            alignment.parameters.transition, reference.transition, atol=1e-12
        )

    def test_invalid_alignment_inputs(self):
        parameters = regimelab.HMMParameters(
            start_prob=np.array([1.0]),
            transition=np.array([[1.0]]),
            means=np.array([[0.0]]),
            variances=np.array([[1.0]]),
        )
        with self.assertRaises(ValueError):
            regimelab.align_probabilities(np.zeros((3, 2)), [0])
        with self.assertRaises(ValueError):
            regimelab.align_labels([0], [0, 0])
        with self.assertRaises(ValueError):
            regimelab.match_states(parameters, parameters, variance_weight=-1)


class TestRuleBasedRegimes(unittest.TestCase):
    def test_volatility_trend_composite(self):
        rng = np.random.default_rng(10)
        returns = np.concatenate(
            [rng.normal(0.0, 0.005, 160), rng.normal(0.0, 0.025, 240)]
        )
        volatility = regimelab.volatility_regime(
            returns, vol_window=20, lookback=60
        )
        self.assertTrue(np.all(volatility.labels[:38] == -1))
        self.assertTrue(np.all((volatility.labels >= -1) & (volatility.labels <= 2)))
        self.assertGreater(np.count_nonzero(volatility.labels == 2), 0)

        prices = 100.0 + 0.2 * np.arange(400.0) + np.sin(np.arange(400.0) * 0.1)
        trend = regimelab.trend_regime(prices, 20, 60)
        self.assertEqual(int(trend.labels[-1]), 2)
        self.assertTrue(np.all(trend.labels[:59] == -1))

        composite = regimelab.composite_regime(
            volatility.labels, trend.labels
        )
        valid = (volatility.labels >= 0) & (trend.labels >= 0)
        expected = np.maximum(
            volatility.labels[valid], 2 - trend.labels[valid]
        )
        np.testing.assert_array_equal(composite[valid], expected)
        self.assertTrue(np.all(composite[~valid] == -1))

    def test_invalid_regime_arguments(self):
        with self.assertRaises(ValueError):
            regimelab.volatility_regime(np.zeros(10), vol_window=1, lookback=20)
        with self.assertRaises(RFluxError):
            regimelab.volatility_regime(
                np.zeros(100), vol_window=20, lookback=30,
                low_q=0.8, high_q=0.2
            )
        prices = np.linspace(1.0, 100.0, 100)
        with self.assertRaises(RFluxError):
            regimelab.trend_regime(prices, fast_window=60, slow_window=20)
        with self.assertRaises(ValueError):
            regimelab.composite_regime(
                np.zeros(3, dtype=np.int32), np.zeros(4, dtype=np.int32)
            )


if __name__ == "__main__":
    unittest.main()
