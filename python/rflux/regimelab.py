"""regimelab: Gaussian HMMs and strictly causal rule-based regime labels.

Label conventions
-----------------
- Volatility: 0 = low, 1 = normal, 2 = high, -1 = warm-up.
- Trend: 0 = down, 1 = sideways, 2 = up, -1 = warm-up.
- Composite risk: 0 = risk-on, 1 = neutral, 2 = risk-off, -1 = warm-up.

The model, regime and alignment kernels live in the C++ core; this module
validates inputs and exposes the Python research API.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ._common import as_f64_1d, as_f64_2d, as_i32_1d, as_size, as_u64
from ._native_backend import RFluxError, native_call, native_api as native_module


@dataclass(frozen=True, slots=True)
class HMMParameters:
    start_prob: np.ndarray
    transition: np.ndarray
    means: np.ndarray
    variances: np.ndarray


@dataclass(frozen=True, slots=True)
class HMMFitInfo:
    iterations: int
    converged: bool
    log_likelihood: float
    aic: float
    bic: float


@dataclass(frozen=True, slots=True)
class HMMSnapshot:
    """Detached fitted HMM state and latest filtered/forecast information."""

    n_states: int
    n_features: int
    iterations: int
    converged: bool
    log_likelihood: float
    aic: float
    bic: float
    latest_filtered: np.ndarray
    forecast: np.ndarray
    transition: np.ndarray
    means: np.ndarray
    variances: np.ndarray


@dataclass(frozen=True, slots=True)
class StateAlignment:
    """``order[k]`` is the target state placed at reference state ``k``."""

    order: np.ndarray
    parameters: HMMParameters
    cost: float


@dataclass(frozen=True, slots=True)
class VolatilityRegimeResult:
    volatility: np.ndarray
    labels: np.ndarray


@dataclass(frozen=True, slots=True)
class TrendRegimeResult:
    fast_ma: np.ndarray
    slow_ma: np.ndarray
    labels: np.ndarray


@dataclass(frozen=True, slots=True)
class StickyRegimeResult:
    """Filtered HMM probabilities with duration and hysteresis constraints."""

    probabilities: np.ndarray
    labels: np.ndarray
    switches: np.ndarray

    @property
    def n_switches(self) -> int:
        return int(self.switches.size)


def _as_observations(observations: Any, n_features: int) -> np.ndarray:
    array = np.asarray(observations, dtype=np.float64)
    if array.ndim == 1:
        if n_features != 1:
            raise ValueError(
                f"model expects {n_features} features; pass an (n_obs, "
                f"{n_features}) array"
            )
        array = array.reshape(-1, 1)
    elif array.ndim != 2:
        raise ValueError(
            f"observations must be 1D or 2D, got shape {array.shape}"
        )
    if array.shape[1] != n_features:
        raise ValueError(
            f"observations have {array.shape[1]} features, model expects "
            f"{n_features}"
        )
    return np.ascontiguousarray(array)


class GaussianHMM:
    """Diagonal-covariance Gaussian HMM backed by the C++ core.

    ``filtered()`` is causal; ``smoothed()`` and ``viterbi()`` use the whole
    sample and should not be treated as point-in-time signals.
    """

    def __init__(self, n_states: int, n_features: int = 1):
        self._n_states = as_size(n_states, "n_states")
        self._n_features = as_size(n_features, "n_features")
        self._model: Any = native_call(
            native_module, "GaussianHMM",
            lambda: native_module.GaussianHMM(
                self._n_states, self._n_features),
        )

    @property
    def n_states(self) -> int:
        return self._n_states

    @property
    def n_features(self) -> int:
        return self._n_features

    def _require_model(self) -> Any:
        if self._model is None:
            raise RuntimeError("GaussianHMM handle is closed")
        return self._model

    def _state_probabilities(self, state_probs: Any) -> np.ndarray:
        probabilities = as_f64_1d(state_probs, "state_probs")
        if probabilities.size != self._n_states:
            raise ValueError(f"state_probs must have length {self._n_states}")
        return probabilities

    def close(self) -> None:
        self._model = None

    def __enter__(self) -> "GaussianHMM":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def __repr__(self) -> str:
        state = "open" if self._model is not None else "closed"
        return (
            f"GaussianHMM(n_states={self._n_states}, "
            f"n_features={self._n_features}, {state}, backend=native)"
        )

    def __copy__(self) -> "GaussianHMM":
        raise TypeError(
            "GaussianHMM owns a native handle and cannot be shallow-copied; "
            "use clone()"
        )

    def __deepcopy__(self, memo) -> "GaussianHMM":
        raise TypeError(
            "GaussianHMM owns a native handle and cannot be deep-copied; "
            "use clone()"
        )

    def clone(self) -> "GaussianHMM":
        """Return an independent model with the same parameters."""
        parameters = self.parameters
        duplicate = GaussianHMM(self._n_states, self._n_features)
        try:
            duplicate.set_parameters(parameters)
        except Exception:
            duplicate.close()
            raise
        return duplicate

    def fit(
        self,
        observations: Any,
        *,
        max_iter: int = 200,
        n_init: int = 3,
        tol: float = 1e-6,
        variance_floor: float = 1e-6,
        seed: int = 42,
        sort_states: bool = True,
    ) -> "GaussianHMM":
        data = _as_observations(observations, self._n_features)
        max_iter = as_size(max_iter, "max_iter")
        n_init = as_size(n_init, "n_init")
        if max_iter > 2**32 - 1:
            raise ValueError("max_iter must fit in uint32")
        if n_init > 2**32 - 1:
            raise ValueError("n_init must fit in uint32")
        model = self._require_model()
        native_call(
            native_module, "rflux_hmm_fit",
            lambda: model.fit(
                data,
                max_iter=max_iter,
                n_init=n_init,
                tol=float(tol),
                variance_floor=float(variance_floor),
                seed=as_u64(seed, "seed"),
                sort_states=bool(sort_states),
            ),
        )
        return self

    @property
    def fit_info(self) -> HMMFitInfo:
        values = native_call(
            native_module, "rflux_hmm_fit_summary",
            lambda: self._require_model().fit_info,
        )
        iterations, converged, log_likelihood, aic, bic = values
        return HMMFitInfo(
            int(iterations), bool(converged), float(log_likelihood),
            float(aic), float(bic),
        )

    def score(self, observations: Any) -> float:
        """Log-likelihood of observations under the current parameters."""
        data = _as_observations(observations, self._n_features)
        return float(native_call(
            native_module, "rflux_hmm_score",
            lambda: self._require_model().score(data),
        ))

    def filtered_states(self, observations: Any) -> np.ndarray:
        """Causal state labels from the filtered distribution (argmax)."""
        return native_call(
            native_module, "row_argmax",
            lambda: native_module.row_argmax(self.filtered(observations)),
        )

    def conditional_mean(self, state_probs: Any, feature: int = 0) -> float:
        """Expected emission mean under a state distribution."""
        probabilities = self._state_probabilities(state_probs)
        feature_index = as_size(feature, "feature", minimum=0)
        if feature_index >= self._n_features:
            raise ValueError(f"feature must be in [0, {self._n_features})")
        return float(native_call(
            native_module, "conditional_mean",
            lambda: native_module.conditional_mean(
                probabilities, self.parameters.means, feature_index
            ),
        ))

    @property
    def parameters(self) -> HMMParameters:
        values = native_call(
            native_module, "rflux_hmm_get_parameters",
            lambda: self._require_model().parameters,
        )
        return HMMParameters(*values)

    def set_parameters(self, parameters: HMMParameters) -> "GaussianHMM":
        k = self._n_states
        d = self._n_features
        start = np.ascontiguousarray(parameters.start_prob, dtype=np.float64)
        transition = np.ascontiguousarray(
            parameters.transition, dtype=np.float64
        )
        means = np.ascontiguousarray(parameters.means, dtype=np.float64)
        variances = np.ascontiguousarray(parameters.variances, dtype=np.float64)
        if start.shape != (k,):
            raise ValueError(f"start_prob must have shape ({k},)")
        if transition.shape != (k, k):
            raise ValueError(f"transition must have shape ({k}, {k})")
        if means.shape != (k, d):
            raise ValueError(f"means must have shape ({k}, {d})")
        if variances.shape != (k, d):
            raise ValueError(f"variances must have shape ({k}, {d})")
        native_call(
            native_module, "rflux_hmm_set_parameters",
            lambda: self._require_model().set_parameters(
                start, transition, means, variances
            ),
        )
        return self

    def viterbi(self, observations: Any) -> np.ndarray:
        """Full-sample Viterbi path (non-causal)."""
        data = _as_observations(observations, self._n_features)
        return native_call(
            native_module, "rflux_hmm_viterbi",
            lambda: self._require_model().viterbi(data),
        )

    def filtered(self, observations: Any) -> np.ndarray:
        """Causal filtered probabilities, shape (n_obs, n_states)."""
        data = _as_observations(observations, self._n_features)
        return native_call(
            native_module, "rflux_hmm_filtered",
            lambda: self._require_model().filtered(data),
        )

    def smoothed(self, observations: Any) -> np.ndarray:
        """Full-sample smoothed probabilities, shape (n_obs, n_states)."""
        data = _as_observations(observations, self._n_features)
        return native_call(
            native_module, "rflux_hmm_smoothed",
            lambda: self._require_model().smoothed(data),
        )

    def forecast(self, state_probs: Any, horizon: int = 1) -> np.ndarray:
        """Prior hidden-state distribution ``state_probs @ transition**horizon``."""
        probabilities = self._state_probabilities(state_probs)
        horizon = as_size(horizon, "horizon", minimum=0)
        return native_call(
            native_module, "rflux_hmm_forecast",
            lambda: self._require_model().forecast(probabilities, horizon),
        )

    def filter_step(self, state_probs: Any, observation: Any) -> np.ndarray:
        """Causal one-step update from a prior state distribution."""
        probabilities = self._state_probabilities(state_probs)
        single = np.asarray(observation, dtype=np.float64)
        if single.ndim == 0:
            if self._n_features != 1:
                raise ValueError(
                    f"observation must have shape (1, {self._n_features})"
                )
            single = single.reshape(1, 1)
        elif single.ndim == 1:
            if single.size != self._n_features:
                raise ValueError(
                    f"observation must have length {self._n_features}"
                )
            single = single.reshape(1, -1)
        elif single.shape != (1, self._n_features):
            raise ValueError(
                f"observation must have shape (1, {self._n_features})"
            )
        return native_call(
            native_module, "rflux_hmm_filter_step",
            lambda: self._require_model().filter_step(
                probabilities, np.ascontiguousarray(single)
            ),
        )

    def stationary_distribution(self) -> np.ndarray:
        """Long-run distribution implied by the current transition matrix."""
        return native_call(
            native_module, "rflux_hmm_stationary",
            lambda: self._require_model().stationary_distribution(),
        )

    def expected_durations(self) -> np.ndarray:
        """Expected sojourn length of each state from the transition diagonal."""
        return native_call(
            native_module, "rflux_hmm_expected_durations",
            lambda: self._require_model().expected_durations(),
        )

    def save(self, path: Any) -> None:
        """Save parameters to a compressed NumPy ``.npz`` file."""
        parameters = self.parameters
        np.savez_compressed(
            path,
            format_version=np.int32(1),
            n_states=np.int32(self._n_states),
            n_features=np.int32(self._n_features),
            start_prob=parameters.start_prob,
            transition=parameters.transition,
            means=parameters.means,
            variances=parameters.variances,
        )

    @classmethod
    def load(cls, path: Any) -> "GaussianHMM":
        with np.load(path, allow_pickle=False) as data:
            version = int(data["format_version"])
            if version != 1:
                raise ValueError(f"unsupported rflux HMM format {version}")
            n_states = int(data["n_states"])
            n_features = int(data["n_features"])
            model = cls(n_states, n_features)
            try:
                model.set_parameters(
                    HMMParameters(
                        np.asarray(data["start_prob"], dtype=np.float64),
                        np.asarray(data["transition"], dtype=np.float64),
                        np.asarray(data["means"], dtype=np.float64),
                        np.asarray(data["variances"], dtype=np.float64),
                    )
                )
            except Exception:
                model.close()
                raise
        return model


def sticky_regime(
    model: GaussianHMM,
    observations: Any,
    *,
    min_duration: int = 5,
    entry_threshold: float = 0.6,
    exit_threshold: float = 0.4,
) -> StickyRegimeResult:
    """Hysteresis-smoothed causal regime labels."""
    if not hasattr(model, "filtered"):
        raise TypeError("model must provide filtered(observations)")
    min_duration = as_size(min_duration, "min_duration")
    entry = float(entry_threshold)
    exit_ = float(exit_threshold)
    if not (np.isfinite(entry) and np.isfinite(exit_)):
        raise ValueError("thresholds must be finite")
    if not 0.0 < exit_ < entry <= 1.0:
        raise ValueError("require 0 < exit_threshold < entry_threshold <= 1")
    probabilities = model.filtered(observations)
    if probabilities.shape[0] == 0:
        return StickyRegimeResult(
            probabilities,
            np.empty(0, dtype=np.int32),
            np.empty(0, dtype=np.int32),
        )
    labels, switches = native_call(
        native_module, "sticky_regime",
        lambda: native_module.sticky_regime(
            probabilities, min_duration, entry, exit_
        ),
    )
    return StickyRegimeResult(
        probabilities,
        np.asarray(labels, dtype=np.int32),
        np.asarray(switches, dtype=np.int32),
    )


def fit_hmm(observations: Any, n_states: int = 2, **kwargs: Any) -> GaussianHMM:
    """Convenience wrapper: infer features, fit, and return the model."""
    array = np.asarray(observations, dtype=np.float64)
    if array.ndim == 1:
        n_features = 1
    elif array.ndim == 2:
        n_features = array.shape[1]
    else:
        raise ValueError("observations must be 1D or 2D")
    model = GaussianHMM(n_states, n_features)
    try:
        return model.fit(array, **kwargs)
    except Exception:
        model.close()
        raise


def fit_hmm_snapshot(
    observations: Any,
    n_states: int = 2,
    *,
    horizon: int = 1,
    **kwargs: Any,
) -> HMMSnapshot:
    """Fit an HMM and return a detached latest-filtered snapshot."""
    array = np.asarray(observations, dtype=np.float64)
    if array.ndim == 1:
        array = array.reshape(-1, 1)
    elif array.ndim != 2:
        raise ValueError("observations must be 1D or 2D")
    model = fit_hmm(array, n_states=n_states, **kwargs)
    try:
        probabilities = np.asarray(model.filtered(array), dtype=np.float64)
        if probabilities.shape[0] == 0:
            raise ValueError("observations must not be empty")
        latest = probabilities[-1].copy()
        forecast = np.asarray(model.forecast(latest, horizon), dtype=np.float64)
        fit = model.fit_info
        parameters = model.parameters
        return HMMSnapshot(
            n_states=int(model.n_states),
            n_features=int(model.n_features),
            iterations=int(fit.iterations),
            converged=bool(fit.converged),
            log_likelihood=float(fit.log_likelihood),
            aic=float(fit.aic),
            bic=float(fit.bic),
            latest_filtered=latest,
            forecast=forecast.copy(),
            transition=np.array(parameters.transition, copy=True),
            means=np.array(parameters.means, copy=True),
            variances=np.array(parameters.variances, copy=True),
        )
    finally:
        model.close()


def volatility_regime(
    returns_: Any,
    vol_window: int = 20,
    lookback: int = 252,
    low_q: float = 0.25,
    high_q: float = 0.75,
) -> VolatilityRegimeResult:
    """Causal volatility regime from trailing realized volatility."""
    data = as_f64_1d(returns_, "returns")
    volatility, labels = native_call(
        native_module, "volatility_regime",
        lambda: native_module.volatility_regime(
            data,
            as_size(vol_window, "vol_window", minimum=2),
            as_size(lookback, "lookback", minimum=20),
            float(low_q), float(high_q),
        ),
    )
    return VolatilityRegimeResult(
        np.asarray(volatility, dtype=np.float64),
        np.asarray(labels, dtype=np.int32),
    )


def trend_regime(
    prices: Any,
    fast_window: int = 20,
    slow_window: int = 60,
    threshold: float = 0.0,
) -> TrendRegimeResult:
    """Causal SMA-spread trend regime."""
    data = as_f64_1d(prices, "prices")
    fast_ma, slow_ma, labels = native_call(
        native_module, "trend_regime",
        lambda: native_module.trend_regime(
            data,
            as_size(fast_window, "fast_window"),
            as_size(slow_window, "slow_window"),
            float(threshold),
        ),
    )
    return TrendRegimeResult(
        np.asarray(fast_ma, dtype=np.float64),
        np.asarray(slow_ma, dtype=np.float64),
        np.asarray(labels, dtype=np.int32),
    )


def composite_regime(volatility_labels: Any, trend_labels: Any) -> np.ndarray:
    """Risk score ``max(volatility, 2 - trend)`` with -1 for warm-up."""
    volatility = as_i32_1d(volatility_labels, "volatility_labels")
    trend = as_i32_1d(trend_labels, "trend_labels")
    if volatility.size != trend.size:
        raise ValueError("label arrays must have equal lengths")
    return native_call(
        native_module, "composite_regime",
        lambda: native_module.composite_regime(volatility, trend),
    )


def _parameter_arrays(parameters: HMMParameters) -> tuple[np.ndarray, ...]:
    start = np.ascontiguousarray(parameters.start_prob, dtype=np.float64)
    transition = np.ascontiguousarray(parameters.transition, dtype=np.float64)
    means = np.ascontiguousarray(parameters.means, dtype=np.float64)
    variances = np.ascontiguousarray(parameters.variances, dtype=np.float64)
    if start.ndim != 1 or start.size == 0:
        raise ValueError("start_prob must be a non-empty 1D array")
    k = start.size
    if transition.shape != (k, k):
        raise ValueError(f"transition must have shape ({k}, {k})")
    if means.ndim != 2 or means.shape[0] != k:
        raise ValueError(f"means must have shape ({k}, n_features)")
    if variances.shape != means.shape:
        raise ValueError("variances must have the same shape as means")
    for name, array in (
        ("start_prob", start),
        ("transition", transition),
        ("means", means),
        ("variances", variances),
    ):
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{name} must be finite")
    if np.any(variances <= 0.0):
        raise ValueError("variances must be positive")
    return start, transition, means, variances


def _validate_variance_weight(value: float) -> float:
    if not np.isfinite(value) or value < 0.0:
        raise ValueError("variance_weight must be finite and non-negative")
    return float(value)


def _matched_order(
    reference_means: np.ndarray,
    reference_variances: np.ndarray,
    target_means: np.ndarray,
    target_variances: np.ndarray,
    variance_weight: float,
) -> np.ndarray:
    values = native_call(
        native_module, "regime_match_states",
        lambda: native_module.regime_match_states(
            reference_means, reference_variances, target_means,
            target_variances, float(variance_weight),
        ),
    )
    return np.asarray(values, dtype=np.int64)


def match_states(
    reference: HMMParameters,
    target: HMMParameters,
    *,
    variance_weight: float = 1.0,
) -> np.ndarray:
    """Match target HMM states to reference states.

    Returns ``order`` such that reference state ``k`` corresponds to target
    state ``order[k]``.
    """
    _, _, reference_means, reference_variances = _parameter_arrays(reference)
    _, _, target_means, target_variances = _parameter_arrays(target)
    if reference_means.shape != target_means.shape:
        raise ValueError(
            "reference and target must have the same state/feature shape"
        )
    variance_weight = _validate_variance_weight(variance_weight)
    return _matched_order(
        reference_means, reference_variances, target_means, target_variances,
        variance_weight,
    )


def align_parameters(
    reference: HMMParameters,
    target: HMMParameters,
    *,
    variance_weight: float = 1.0,
) -> StateAlignment:
    """Return target parameters reordered to the reference state labels."""
    start, transition, means, variances = _parameter_arrays(target)
    reference_start, _, reference_means, reference_variances = _parameter_arrays(
        reference
    )
    if reference_start.size != start.size:
        raise ValueError("reference and target state counts must match")
    if reference_means.shape != means.shape:
        raise ValueError(
            "reference and target must have the same state/feature shape"
        )
    variance_weight = _validate_variance_weight(variance_weight)
    payload = native_call(
        native_module, "align_parameters",
        lambda: native_module.align_parameters(
            reference_means, reference_variances, start, transition, means,
            variances, float(variance_weight),
        ),
    )
    order = np.asarray(payload["order"], dtype=np.int64)
    aligned = HMMParameters(
        start_prob=np.array(payload["start_prob"], dtype=np.float64, copy=True),
        transition=np.array(payload["transition"], dtype=np.float64, copy=True),
        means=np.array(payload["means"], dtype=np.float64, copy=True),
        variances=np.array(payload["variances"], dtype=np.float64, copy=True),
    )
    if order.size != start.size:
        raise ValueError("state matching returned an invalid assignment")
    return StateAlignment(
        order=order, parameters=aligned, cost=float(payload["cost"])
    )


def _as_permutation(order: Any, *, allow_empty: bool) -> np.ndarray:
    mapping = np.asarray(order, dtype=np.int64)
    if mapping.ndim != 1 or (mapping.size == 0 and not allow_empty):
        raise ValueError(
            "order must be a non-empty permutation" if not allow_empty
            else "order must be one-dimensional"
        )
    if not np.array_equal(np.sort(mapping), np.arange(mapping.size)):
        raise ValueError("order must be a permutation")
    return mapping


def align_probabilities(probabilities: Any, order: Any) -> np.ndarray:
    """Reorder state-probability columns from target to reference labels."""
    values = as_f64_2d(probabilities, "probabilities")
    mapping = _as_permutation(order, allow_empty=True)
    if mapping.size != values.shape[1]:
        raise ValueError("order must have one entry per probability column")
    return np.ascontiguousarray(values[:, mapping])


def align_labels(labels: Any, order: Any) -> np.ndarray:
    """Reorder integer state labels from target to reference labels."""
    mapping = _as_permutation(order, allow_empty=False)
    values = np.asarray(labels)
    if values.ndim != 1:
        raise ValueError("labels must be one-dimensional")
    if not np.issubdtype(values.dtype, np.integer):
        raise TypeError("labels must contain integers")
    if np.any(values < -1) or np.any(values >= mapping.size):
        raise ValueError("labels must be -1 or valid target state ids")
    inverse = np.empty(mapping.size, dtype=np.int32)
    inverse[mapping] = np.arange(mapping.size, dtype=np.int32)
    result = np.full(values.shape, -1, dtype=np.int32)
    valid = values >= 0
    result[valid] = inverse[values[valid]]
    return result


__all__ = [
    "GaussianHMM",
    "HMMFitInfo",
    "HMMParameters",
    "HMMSnapshot",
    "RFluxError",
    "StateAlignment",
    "StickyRegimeResult",
    "TrendRegimeResult",
    "VolatilityRegimeResult",
    "align_labels",
    "align_parameters",
    "align_probabilities",
    "composite_regime",
    "fit_hmm",
    "fit_hmm_snapshot",
    "match_states",
    "sticky_regime",
    "trend_regime",
    "volatility_regime",
]
