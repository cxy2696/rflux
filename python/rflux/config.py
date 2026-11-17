"""Configuration-driven model construction.

The C++ core does not read files. It consumes plain parameter arrays and
options.  This module is the composition edge that validates TOML,
JSON or YAML documents and instantiates a model through the public Python API.
Keeping parsing here preserves the "pure math core" boundary while making HMM
initialisation, covariance parameters and fit options reproducible.

Canonical TOML layout::

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
    max_iter = 200
    n_init = 3
    tol = 1e-6
    variance_floor = 1e-6
    seed = 42
    sort_states = true

``n_states`` and ``n_features`` may be omitted when ``model.initial`` is
present; they are inferred from the parameter tables.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from . import regimelab
from ._common import as_size, as_u64

try:  # Python 3.11+
    import tomllib
except ImportError:  # pragma: no cover - Python 3.10 fallback
    try:
        import tomli as tomllib  # type: ignore[no-redef]
    except ImportError:  # pragma: no cover
        tomllib = None  # type: ignore[assignment]


class ConfigError(ValueError):
    """A model configuration is malformed or fails validation."""


@dataclass(frozen=True, slots=True)
class FitOptions:
    """Validated HMM fitting options."""

    max_iter: int = 200
    n_init: int = 3
    tol: float = 1e-6
    variance_floor: float = 1e-6
    seed: int = 42
    sort_states: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "max_iter", as_size(self.max_iter, "max_iter"))
        object.__setattr__(self, "n_init", as_size(self.n_init, "n_init"))
        object.__setattr__(self, "tol", _positive_float(self.tol, "tol"))
        object.__setattr__(
            self,
            "variance_floor",
            _positive_float(self.variance_floor, "variance_floor"),
        )
        object.__setattr__(self, "seed", as_u64(self.seed, "seed"))
        object.__setattr__(self, "sort_states", bool(self.sort_states))

    def as_kwargs(self) -> dict[str, Any]:
        return {
            "max_iter": self.max_iter,
            "n_init": self.n_init,
            "tol": self.tol,
            "variance_floor": self.variance_floor,
            "seed": self.seed,
            "sort_states": self.sort_states,
        }


@dataclass(frozen=True, slots=True)
class HMMConfig:
    """A validated, reproducible Gaussian HMM specification."""

    n_states: int
    n_features: int
    initial: regimelab.HMMParameters | None = None
    fit: FitOptions = field(default_factory=FitOptions)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        k = as_size(self.n_states, "n_states")
        d = as_size(self.n_features, "n_features")
        object.__setattr__(self, "n_states", k)
        object.__setattr__(self, "n_features", d)
        if self.initial is not None:
            _validate_initial(self.initial, k, d)
        object.__setattr__(self, "metadata", dict(self.metadata))

    def build(self) -> regimelab.GaussianHMM:
        """Instantiate a model and load initial parameters when supplied."""
        model = regimelab.GaussianHMM(self.n_states, self.n_features)
        if self.initial is not None:
            try:
                model.set_parameters(self.initial)
            except Exception:
                model.close()
                raise
        return model

    def fit_model(self, observations: Any) -> regimelab.GaussianHMM:
        """Build a model and fit it with this configuration's options."""
        model = self.build()
        try:
            model.fit(observations, **self.fit.as_kwargs())
        except Exception:
            model.close()
            raise
        return model

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": {
                "type": "hmm",
                "n_states": self.n_states,
                "n_features": self.n_features,
                "fit": {
                    "max_iter": self.fit.max_iter,
                    "n_init": self.fit.n_init,
                    "tol": self.fit.tol,
                    "variance_floor": self.fit.variance_floor,
                    "seed": self.fit.seed,
                    "sort_states": self.fit.sort_states,
                },
            }
        }
        if self.initial is not None:
            payload["model"]["initial"] = {
                "start_prob": self.initial.start_prob.tolist(),
                "transition": self.initial.transition.tolist(),
                "means": self.initial.means.tolist(),
                "variances": self.initial.variances.tolist(),
            }
        if self.metadata:
            payload["metadata"] = dict(self.metadata)
        return payload

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True) + "\n"


@dataclass(frozen=True, slots=True)
class ModelConfig:
    """Container that leaves room for additional model families."""

    model: HMMConfig
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def type(self) -> str:
        return "hmm"

    def to_dict(self) -> dict[str, Any]:
        payload = self.model.to_dict()
        if self.metadata:
            payload["metadata"] = dict(self.metadata)
        return payload


def _positive_float(value: Any, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{name} must be a number") from exc
    if not math.isfinite(number) or number <= 0.0:
        raise ConfigError(f"{name} must be finite and positive")
    return number


def _validate_initial(
    parameters: regimelab.HMMParameters, n_states: int, n_features: int
) -> None:
    try:
        arrays = {
            "start_prob": np.asarray(parameters.start_prob, dtype=np.float64),
            "transition": np.asarray(parameters.transition, dtype=np.float64),
            "means": np.asarray(parameters.means, dtype=np.float64),
            "variances": np.asarray(parameters.variances, dtype=np.float64),
        }
    except (TypeError, ValueError) as exc:
        raise ConfigError("initial parameters must be numeric arrays") from exc
    if arrays["start_prob"].shape != (n_states,):
        raise ConfigError(f"start_prob must have shape ({n_states},)")
    if arrays["transition"].shape != (n_states, n_states):
        raise ConfigError(
            f"transition must have shape ({n_states}, {n_states})"
        )
    expected = (n_states, n_features)
    if arrays["means"].shape != expected:
        raise ConfigError(f"means must have shape {expected}")
    if arrays["variances"].shape != expected:
        raise ConfigError(f"variances must have shape {expected}")
    for name, array in arrays.items():
        if not np.all(np.isfinite(array)):
            raise ConfigError(f"{name} must contain only finite values")
    if np.any(arrays["variances"] <= 0.0):
        raise ConfigError("variances must be positive")
    if np.any(arrays["start_prob"] < 0.0):
        raise ConfigError("start_prob must be non-negative")
    if np.any(arrays["transition"] < 0.0):
        raise ConfigError("transition must be non-negative")
    if abs(float(arrays["start_prob"].sum()) - 1.0) > 1e-8:
        raise ConfigError("start_prob must sum to one")
    row_sums = arrays["transition"].sum(axis=1)
    if np.any(np.abs(row_sums - 1.0) > 1e-8):
        raise ConfigError("each transition row must sum to one")


def _parameter_table(payload: Any, name: str) -> np.ndarray:
    try:
        array = np.asarray(payload, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{name} must be a rectangular numeric table") from exc
    if array.ndim == 0 or array.size == 0:
        raise ConfigError(f"{name} must be a non-empty numeric table")
    return array


def _parse_initial(
    payload: Mapping[str, Any],
) -> tuple[regimelab.HMMParameters, int, int]:
    required = {"start_prob", "transition", "means", "variances"}
    missing = required - set(payload)
    if missing:
        raise ConfigError(
            "model.initial is missing: " + ", ".join(sorted(missing))
        )
    start = _parameter_table(payload["start_prob"], "start_prob")
    transition = _parameter_table(payload["transition"], "transition")
    means = _parameter_table(payload["means"], "means")
    variances = _parameter_table(payload["variances"], "variances")
    if start.ndim != 1:
        raise ConfigError("start_prob must be one-dimensional")
    k = int(start.size)
    if transition.shape != (k, k):
        raise ConfigError(f"transition must have shape ({k}, {k})")
    if means.ndim != 2 or means.shape[0] != k:
        raise ConfigError(f"means must have shape ({k}, n_features)")
    if variances.shape != means.shape:
        raise ConfigError("variances must have the same shape as means")
    parameters = regimelab.HMMParameters(
        start_prob=start.copy(),
        transition=transition.copy(),
        means=means.copy(),
        variances=variances.copy(),
    )
    _validate_initial(parameters, k, int(means.shape[1]))
    return parameters, k, int(means.shape[1])


def _parse_fit(payload: Any) -> FitOptions:
    if payload is None:
        return FitOptions()
    if not isinstance(payload, Mapping):
        raise ConfigError("model.fit must be a table")
    allowed = {
        "max_iter",
        "n_init",
        "tol",
        "variance_floor",
        "seed",
        "sort_states",
    }
    unknown = set(payload) - allowed
    if unknown:
        raise ConfigError(
            "unknown model.fit option(s): " + ", ".join(sorted(unknown))
        )
    return FitOptions(**dict(payload))


def _parse_hmm(payload: Any) -> HMMConfig:
    if not isinstance(payload, Mapping):
        raise ConfigError("model must be a table/object")
    model_type = str(payload.get("type", "hmm")).lower()
    if model_type != "hmm":
        raise ConfigError(f"unsupported model.type {model_type!r}")
    allowed = {"type", "n_states", "n_features", "initial", "fit"}
    unknown = set(payload) - allowed
    if unknown:
        raise ConfigError(
            "unknown model option(s): " + ", ".join(sorted(unknown))
        )

    initial = None
    inferred_states = None
    inferred_features = None
    if "initial" in payload:
        initial, inferred_states, inferred_features = _parse_initial(
            payload["initial"]
        )

    if "n_states" in payload:
        n_states = as_size(payload["n_states"], "n_states")
        if inferred_states is not None and n_states != inferred_states:
            raise ConfigError(
                "model.n_states disagrees with model.initial arrays"
            )
    elif inferred_states is not None:
        n_states = inferred_states
    else:
        raise ConfigError("model.n_states is required when there is no initial state")

    if "n_features" in payload:
        n_features = as_size(payload["n_features"], "n_features")
        if inferred_features is not None and n_features != inferred_features:
            raise ConfigError(
                "model.n_features disagrees with model.initial arrays"
            )
    elif inferred_features is not None:
        n_features = inferred_features
    else:
        n_features = 1

    return HMMConfig(
        n_states=n_states,
        n_features=n_features,
        initial=initial,
        fit=_parse_fit(payload.get("fit")),
    )


def from_dict(payload: Mapping[str, Any]) -> ModelConfig:
    """Validate a parsed configuration mapping."""
    if not isinstance(payload, Mapping):
        raise ConfigError("configuration root must be a table/object")
    if "model" not in payload:
        raise ConfigError("configuration must contain a [model] table")
    unknown = set(payload) - {"model", "metadata"}
    if unknown:
        raise ConfigError(
            "unknown top-level key(s): " + ", ".join(sorted(unknown))
        )
    model = _parse_hmm(payload["model"])
    metadata = payload.get("metadata", {})
    if not isinstance(metadata, Mapping):
        raise ConfigError("metadata must be a table/object")
    return ModelConfig(model=model, metadata=dict(metadata))


def loads(text: str, *, format: str = "toml") -> ModelConfig:
    """Parse configuration text in ``toml``, ``json`` or ``yaml`` format."""
    normalized = format.lower().lstrip(".")
    if normalized in {"toml", "tml"}:
        if tomllib is None:
            raise ConfigError(
                "TOML parsing requires Python 3.11+ or the 'tomli' package"
            )
        try:
            payload = tomllib.loads(text)
        except Exception as exc:
            raise ConfigError(f"invalid TOML: {exc}") from exc
    elif normalized == "json":
        try:
            payload = json.loads(text)
        except Exception as exc:
            raise ConfigError(f"invalid JSON: {exc}") from exc
    elif normalized in {"yaml", "yml"}:
        try:
            import yaml  # type: ignore[import-not-found]
        except ImportError as exc:
            raise ConfigError(
                "YAML parsing requires the optional 'PyYAML' package"
            ) from exc
        try:
            payload = yaml.safe_load(text)
        except Exception as exc:
            raise ConfigError(f"invalid YAML: {exc}") from exc
    else:
        raise ConfigError(
            f"unsupported configuration format {format!r}; use toml, json or yaml"
        )
    return from_dict(payload)


def load(path: str | Path) -> ModelConfig:
    """Load a model configuration from ``.toml``, ``.json``, ``.yaml``/``.yml``."""
    config_path = Path(path).expanduser()
    suffix = config_path.suffix.lower()
    if suffix == ".toml":
        format = "toml"
    elif suffix == ".json":
        format = "json"
    elif suffix in {".yaml", ".yml"}:
        format = "yaml"
    else:
        raise ConfigError(
            f"unsupported configuration suffix {suffix!r}; use .toml, .json, "
            ".yaml or .yml"
        )
    try:
        text = config_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigError(f"cannot read {config_path}: {exc}") from exc
    return loads(text, format=format)


def save(path: str | Path, config: ModelConfig | HMMConfig) -> None:
    """Write a configuration as TOML or JSON, selected by the suffix."""
    config_path = Path(path).expanduser()
    if isinstance(config, HMMConfig):
        config = ModelConfig(config)
    payload = config.to_dict()
    suffix = config_path.suffix.lower()
    if suffix == ".json":
        config_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return
    if suffix != ".toml":
        raise ConfigError("save() supports .toml and .json paths")
    config_path.write_text(_to_toml(payload), encoding="utf-8")


def _toml_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, list):
        if value and all(isinstance(item, list) for item in value):
            return "[" + ", ".join(
                "[" + ", ".join(_toml_value(item) for item in row) + "]"
                for row in value
            ) + "]"
        return "[" + ", ".join(_toml_value(item) for item in value) + "]"
    return json.dumps(value)


def _to_toml(payload: Mapping[str, Any]) -> str:
    lines: list[str] = []
    model = payload["model"]
    lines.append("[model]")
    lines.append(f'type = "{model["type"]}"')
    lines.append(f"n_states = {model['n_states']}")
    lines.append(f"n_features = {model['n_features']}")
    if "initial" in model:
        lines.append("")
        lines.append("[model.initial]")
        for key in ("start_prob", "transition", "means", "variances"):
            lines.append(f"{key} = {_toml_value(model['initial'][key])}")
    fit = model.get("fit")
    if fit:
        lines.append("")
        lines.append("[model.fit]")
        for key, value in fit.items():
            lines.append(f"{key} = {_toml_value(value)}")
    metadata = payload.get("metadata")
    if metadata:
        lines.append("")
        lines.append("[metadata]")
        for key, value in metadata.items():
            lines.append(f"{key} = {_toml_value(value)}")
    return "\n".join(lines) + "\n"


__all__ = [
    "ConfigError",
    "FitOptions",
    "HMMConfig",
    "ModelConfig",
    "from_dict",
    "load",
    "loads",
    "save",
]
