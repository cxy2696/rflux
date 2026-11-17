import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

import rflux
from rflux import config, regimelab

TOML_TEXT = """
[model]
type = "hmm"
n_states = 2
n_features = 1

[model.initial]
start_prob = [0.6, 0.4]
transition = [[0.9, 0.1], [0.2, 0.8]]
means = [[-1.0], [1.0]]
variances = [[1.0], [1.0]]

[model.fit]
max_iter = 20
n_init = 2
seed = 99
"""


class TestHMMConfig(unittest.TestCase):
    def _write(self, suffix, text):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / f"model{suffix}"
        path.write_text(text, encoding="utf-8")
        return path

    def test_toml_build_and_roundtrip(self):
        path = self._write(".toml", TOML_TEXT)
        loaded = config.load(path)
        self.assertEqual(loaded.type, "hmm")
        model_config = loaded.model
        self.assertEqual(model_config.n_states, 2)
        self.assertEqual(model_config.n_features, 1)
        self.assertEqual(model_config.fit.seed, 99)

        model = model_config.build()
        try:
            np.testing.assert_allclose(
                model.parameters.start_prob, [0.6, 0.4]
            )
            np.testing.assert_allclose(
                model.forecast([0.6, 0.4], 1), [0.62, 0.38], atol=1e-12
            )
        finally:
            model.close()

        output = path.with_name("roundtrip.toml")
        config.save(output, loaded)
        reloaded = config.load(output)
        self.assertEqual(reloaded.model.n_states, 2)
        np.testing.assert_allclose(
            reloaded.model.initial.transition,
            [[0.9, 0.1], [0.2, 0.8]],
        )

    def test_json_and_inferred_dimensions(self):
        payload = {
            "model": {
                "initial": {
                    "start_prob": [0.5, 0.5],
                    "transition": [[0.8, 0.2], [0.3, 0.7]],
                    "means": [[0.0, 1.0], [2.0, 3.0]],
                    "variances": [[1.0, 1.0], [2.0, 2.0]],
                }
            }
        }
        loaded = config.loads(json.dumps(payload), format="json")
        self.assertEqual(loaded.model.n_states, 2)
        self.assertEqual(loaded.model.n_features, 2)
        model = loaded.model.build()
        try:
            self.assertEqual(model.n_states, 2)
            self.assertEqual(model.n_features, 2)
        finally:
            model.close()

    def test_fit_model_uses_configured_options(self):
        loaded = config.loads(TOML_TEXT, format="toml")
        rng = np.random.default_rng(3)
        observations = np.concatenate(
            [rng.normal(-1.0, 0.2, 80), rng.normal(1.0, 0.2, 80)]
        )
        model = loaded.model.fit_model(observations)
        try:
            self.assertTrue(model.fit_info.converged)
            self.assertEqual(model.filtered(observations).shape, (160, 2))
        finally:
            model.close()

    def test_invalid_configurations_are_rejected(self):
        with self.assertRaises(config.ConfigError):
            config.loads("[model]\nn_features = 1\n", format="toml")
        with self.assertRaises(config.ConfigError):
            config.loads(
                TOML_TEXT.replace("seed = 99", "unknown = 1"), format="toml"
            )
        with self.assertRaises(config.ConfigError):
            config.loads(
                TOML_TEXT.replace("n_states = 2", "n_states = 3"),
                format="toml",
            )
        with self.assertRaises(config.ConfigError):
            config.loads(
                TOML_TEXT.replace("variances = [[1.0], [1.0]]",
                                  "variances = [[1.0], [-1.0]]"),
                format="toml",
            )
        with self.assertRaises(config.ConfigError):
            config.loads(TOML_TEXT, format="ini")

    def test_yaml_is_optional(self):
        try:
            import yaml  # noqa: F401
        except ImportError:
            with self.assertRaises(config.ConfigError):
                config.loads("model: {}", format="yaml")
        else:
            loaded = config.loads(
                "model:\n  n_states: 2\n  n_features: 1\n", format="yaml"
            )
            self.assertEqual(loaded.model.n_states, 2)

    def test_top_level_exports(self):
        self.assertIs(rflux.HMMConfig, config.HMMConfig)
        self.assertIs(rflux.ConfigError, config.ConfigError)
        self.assertIs(rflux.load_config, config.load)


if __name__ == "__main__":
    unittest.main()
