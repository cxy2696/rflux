"""regimelab: Gaussian HMMs and strictly causal rule-based regime labels.

Label conventions
-----------------
- Volatility: 0 = low, 1 = normal, 2 = high, -1 = warm-up.
- Trend: 0 = down, 1 = sideways, 2 = up, -1 = warm-up.
- Composite risk: 0 = risk-on, 1 = neutral, 2 = risk-off, -1 = warm-up.
"""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
from typing import Any

import numpy as np

from ._common import (
    as_f64_1d,
    as_f64_2d,
    as_i32_1d,
    as_size,
    as_u64,
    call_i32,
    call_scalar,
    call_vector,
    double_ptr,
    empty_f64,
    empty_i32,
    int32_ptr,
)
from ._ffi import HMMOptions, RFluxError, check, get_lib


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
class StateAlignment:
    """Result of matching target HMM states to a reference model.

    ``order[k]`` is the target state that should be placed at reference state
    ``k``.  ``parameters`` is the reordered target parameter set.
    """

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
    sample and should not be treated as point-in-time signals.  A model handle
    is stateful and must be used by one thread at a time.
    """

    def __init__(self, n_states: int, n_features: int = 1):
        self._n_states = as_size(n_states, "n_states")
        self._n_features = as_size(n_features, "n_features")
        self._handle: ctypes.c_void_p | None = ctypes.c_void_p()
        handle = ctypes.c_void_p()
        check(
            get_lib().rflux_hmm_create(
                self._n_states, self._n_features, ctypes.byref(handle)
            ),
            "rflux_hmm_create",
        )
        self._handle = handle

    @property
    def n_states(self) -> int:
        return self._n_states

    @property
    def n_features(self) -> int:
        return self._n_features

    def _require_handle(self) -> ctypes.c_void_p:
        if self._handle is None:
            raise RuntimeError("GaussianHMM handle is closed")
        return self._handle

    def close(self) -> None:
        if self._handle is not None:
            get_lib().rflux_hmm_destroy(self._handle)
            self._handle = None

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
        state = "closed" if self._handle is None else "open"
        return (
            f"GaussianHMM(n_states={self._n_states}, "
            f"n_features={self._n_features}, {state})"
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
        """Return an independent model with the same parameters.

        Only fitted or parameterised models can be cloned.
        """
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
        options = HMMOptions()
        options_pointer = ctypes.byref(options)
        check(
            get_lib().rflux_hmm_options_init(
                options_pointer, ctypes.sizeof(options)
            ),
            "rflux_hmm_options_init",
        )
        options.max_iter = max_iter
        options.n_init = n_init
        options.tol = float(tol)
        options.variance_floor = float(variance_floor)
        options.seed = as_u64(seed, "seed")
        options.sort_states = 1 if sort_states else 0
        check(
            get_lib().rflux_hmm_fit(
                self._require_handle(),
                double_ptr(data),
                data.shape[0],
                options_pointer,
            ),
            "rflux_hmm_fit",
        )
        return self

    @property
    def fit_info(self) -> HMMFitInfo:
        iterations = ctypes.c_size_t(0)
        converged = ctypes.c_int32(0)
        log_likelihood = np.empty(1, dtype=np.float64)
        aic = np.empty(1, dtype=np.float64)
        bic = np.empty(1, dtype=np.float64)
        check(
            get_lib().rflux_hmm_fit_summary(
                self._require_handle(),
                ctypes.byref(iterations),
                ctypes.byref(converged),
                double_ptr(log_likelihood),
                double_ptr(aic),
                double_ptr(bic),
            ),
            "rflux_hmm_fit_summary",
        )
        return HMMFitInfo(
            int(iterations.value),
            bool(converged.value),
            float(log_likelihood[0]),
            float(aic[0]),
            float(bic[0]),
        )

    def score(self, observations: Any) -> float:
        """Log-likelihood of observations under the current parameters.

        Works after ``fit`` and after ``set_parameters``; useful for
        walk-forward or holdout model selection.
        """
        data = _as_observations(observations, self._n_features)
        return call_scalar(
            "rflux_hmm_score", get_lib().rflux_hmm_score,
            self._require_handle(), double_ptr(data), data.shape[0])

    def filtered_states(self, observations: Any) -> np.ndarray:
        """Causal state labels from the filtered distribution (argmax)."""
        return np.argmax(self.filtered(observations), axis=1).astype(np.int32)

    def conditional_mean(self, state_probs: Any, feature: int = 0) -> float:
        """Expected emission mean under a state distribution.

        This is ``sum_k P(s=k) * mu[k, feature]`` under the model's current
        parameters.  It is an in-sample conditional mean, not a price forecast.
        """
        probabilities = as_f64_1d(state_probs, "state_probs")
        if probabilities.size != self._n_states:
            raise ValueError(f"state_probs must have length {self._n_states}")
        if np.any(~np.isfinite(probabilities)) or np.any(probabilities < 0.0):
            raise ValueError("state_probs must be finite and non-negative")
        total = float(probabilities.sum())
        if total <= 0.0:
            raise ValueError("state_probs must have positive mass")
        feature_index = as_size(feature, "feature", minimum=0)
        if feature_index >= self._n_features:
            raise ValueError(f"feature must be in [0, {self._n_features})")
        parameters = self.parameters
        weights = probabilities / total
        return float(weights @ parameters.means[:, feature_index])

    @property
    def parameters(self) -> HMMParameters:
        k = self._n_states
        d = self._n_features
        start = empty_f64(k)
        transition = empty_f64(k * k)
        means = empty_f64(k * d)
        variances = empty_f64(k * d)
        check(
            get_lib().rflux_hmm_get_parameters(
                self._require_handle(),
                double_ptr(start),
                double_ptr(transition),
                double_ptr(means),
                double_ptr(variances),
            ),
            "rflux_hmm_get_parameters",
        )
        return HMMParameters(
            start.copy(),
            transition.reshape(k, k).copy(),
            means.reshape(k, d).copy(),
            variances.reshape(k, d).copy(),
        )

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
        check(
            get_lib().rflux_hmm_set_parameters(
                self._require_handle(),
                double_ptr(start),
                double_ptr(transition.reshape(-1)),
                double_ptr(means.reshape(-1)),
                double_ptr(variances.reshape(-1)),
            ),
            "rflux_hmm_set_parameters",
        )
        return self

    def viterbi(self, observations: Any) -> np.ndarray:
        """Full-sample Viterbi path (non-causal)."""
        data = _as_observations(observations, self._n_features)
        return call_i32(
            "rflux_hmm_viterbi", data.shape[0], get_lib().rflux_hmm_viterbi,
            self._require_handle(), double_ptr(data), data.shape[0])

    def filtered(self, observations: Any) -> np.ndarray:
        """Causal filtered probabilities, shape (n_obs, n_states)."""
        data = _as_observations(observations, self._n_features)
        out = call_vector(
            "rflux_hmm_filtered", data.shape[0] * self._n_states,
            get_lib().rflux_hmm_filtered,
            self._require_handle(), double_ptr(data), data.shape[0])
        return out.reshape(data.shape[0], self._n_states)

    def smoothed(self, observations: Any) -> np.ndarray:
        """Full-sample smoothed probabilities, shape (n_obs, n_states)."""
        data = _as_observations(observations, self._n_features)
        out = call_vector(
            "rflux_hmm_smoothed", data.shape[0] * self._n_states,
            get_lib().rflux_hmm_smoothed,
            self._require_handle(), double_ptr(data), data.shape[0])
        return out.reshape(data.shape[0], self._n_states)

    def forecast(self, state_probs: Any, horizon: int = 1) -> np.ndarray:
        """Prior hidden-state distribution ``state_probs @ transition**horizon``."""
        probabilities = as_f64_1d(state_probs, "state_probs")
        if probabilities.size != self._n_states:
            raise ValueError(
                f"state_probs must have length {self._n_states}"
            )
        horizon = as_size(horizon, "horizon", minimum=0)
        return call_vector(
            "rflux_hmm_forecast", self._n_states, get_lib().rflux_hmm_forecast,
            self._require_handle(), double_ptr(probabilities), horizon)

    def filter_step(self, state_probs: Any, observation: Any) -> np.ndarray:
        """Causal one-step update from a prior state distribution."""
        probabilities = as_f64_1d(state_probs, "state_probs")
        if probabilities.size != self._n_states:
            raise ValueError(f"state_probs must have length {self._n_states}")
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
        single = np.ascontiguousarray(single)
        return call_vector(
            "rflux_hmm_filter_step", self._n_states,
            get_lib().rflux_hmm_filter_step, self._require_handle(),
            double_ptr(probabilities), double_ptr(single))

    def stationary_distribution(self) -> np.ndarray:
        """Long-run distribution implied by the current transition matrix."""
        return call_vector(
            "rflux_hmm_stationary", self._n_states,
            get_lib().rflux_hmm_stationary, self._require_handle())

    def expected_durations(self) -> np.ndarray:
        """Expected sojourn length of each state from the transition diagonal."""
        return call_vector(
            "rflux_hmm_expected_durations", self._n_states,
            get_lib().rflux_hmm_expected_durations, self._require_handle())

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


def volatility_regime(
    returns_: Any,
    vol_window: int = 20,
    lookback: int = 252,
    low_q: float = 0.25,
    high_q: float = 0.75,
) -> VolatilityRegimeResult:
    """Causal volatility regime from trailing realized volatility."""
    data = as_f64_1d(returns_, "returns")
    labels = empty_i32(data.size)
    volatility = empty_f64(data.size)
    check(
        get_lib().rflux_volatility_regime(
            double_ptr(data),
            data.size,
            as_size(vol_window, "vol_window", minimum=2),
            as_size(lookback, "lookback", minimum=20),
            float(low_q),
            float(high_q),
            int32_ptr(labels),
            double_ptr(volatility),
        ),
        "rflux_volatility_regime",
    )
    return VolatilityRegimeResult(volatility, labels)


def trend_regime(
    prices: Any,
    fast_window: int = 20,
    slow_window: int = 60,
    threshold: float = 0.0,
) -> TrendRegimeResult:
    """Causal SMA-spread trend regime."""
    data = as_f64_1d(prices, "prices")
    labels = empty_i32(data.size)
    fast_ma = empty_f64(data.size)
    slow_ma = empty_f64(data.size)
    check(
        get_lib().rflux_trend_regime(
            double_ptr(data),
            data.size,
            as_size(fast_window, "fast_window"),
            as_size(slow_window, "slow_window"),
            float(threshold),
            int32_ptr(labels),
            double_ptr(fast_ma),
            double_ptr(slow_ma),
        ),
        "rflux_trend_regime",
    )
    return TrendRegimeResult(fast_ma, slow_ma, labels)


def composite_regime(volatility_labels: Any, trend_labels: Any) -> np.ndarray:
    """Risk score ``max(volatility, 2 - trend)`` with -1 for warm-up."""
    volatility = as_i32_1d(volatility_labels, "volatility_labels")
    trend = as_i32_1d(trend_labels, "trend_labels")
    if volatility.size != trend.size:
        raise ValueError("label arrays must have equal lengths")
    return call_i32(
        "rflux_composite_regime", volatility.size,
        get_lib().rflux_composite_regime,
        int32_ptr(volatility), int32_ptr(trend), volatility.size)


def _parameter_arrays(parameters: HMMParameters) -> tuple[np.ndarray, ...]:
    start = np.asarray(parameters.start_prob, dtype=np.float64)
    transition = np.asarray(parameters.transition, dtype=np.float64)
    means = np.asarray(parameters.means, dtype=np.float64)
    variances = np.asarray(parameters.variances, dtype=np.float64)
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


def _hungarian(cost: np.ndarray) -> np.ndarray:
    """Minimum-cost assignment for a square finite cost matrix.

    Returns ``assignment[i] = j``, the target column assigned to reference row i.
    This is the O(n^3) shortest augmenting path implementation with potentials.
    """
    matrix = np.asarray(cost, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("cost must be square")
    n = matrix.shape[0]
    if n == 0:
        raise ValueError("cost must not be empty")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("cost must be finite")
    u = np.zeros(n + 1, dtype=np.float64)
    v = np.zeros(n + 1, dtype=np.float64)
    p = np.zeros(n + 1, dtype=np.int64)
    way = np.zeros(n + 1, dtype=np.int64)
    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = np.full(n + 1, np.inf, dtype=np.float64)
        used = np.zeros(n + 1, dtype=bool)
        while True:
            used[j0] = True
            i0 = int(p[j0])
            delta = np.inf
            j1 = 0
            for j in range(1, n + 1):
                if used[j]:
                    continue
                current = matrix[i0 - 1, j - 1] - u[i0] - v[j]
                if current < minv[j]:
                    minv[j] = current
                    way[j] = j0
                if minv[j] < delta:
                    delta = minv[j]
                    j1 = j
            for j in range(n + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = int(way[j0])
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break
    assignment = np.empty(n, dtype=np.int64)
    for j in range(1, n + 1):
        assignment[int(p[j]) - 1] = j - 1
    return assignment


def match_states(
    reference: HMMParameters,
    target: HMMParameters,
    *,
    variance_weight: float = 1.0,
) -> np.ndarray:
    """Match target HMM states to reference states.

    The cost combines standardized mean differences and log-variance
    differences.  Returns ``order`` such that reference state ``k`` corresponds
    to target state ``order[k]``.  Use :func:`align_probabilities` or
    :func:`align_labels` to apply the mapping without mutating a model.
    """
    _, _, reference_means, reference_variances = _parameter_arrays(reference)
    _, _, target_means, target_variances = _parameter_arrays(target)
    if reference_means.shape != target_means.shape:
        raise ValueError("reference and target must have the same state/feature shape")
    if not np.isfinite(variance_weight) or variance_weight < 0.0:
        raise ValueError("variance_weight must be finite and non-negative")
    scale = np.sqrt(reference_variances)
    mean_cost = (target_means[None, :, :] - reference_means[:, None, :]) ** 2
    mean_cost = np.sum(mean_cost / (scale[:, None, :] ** 2), axis=2)
    log_ratio = np.log(target_variances[None, :, :]) - np.log(
        reference_variances[:, None, :]
    )
    variance_cost = variance_weight * np.sum(log_ratio**2, axis=2)
    cost = mean_cost + variance_cost
    return _hungarian(cost)


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
    order = match_states(reference, target, variance_weight=variance_weight)
    k = start.size
    aligned = HMMParameters(
        start_prob=start[order].copy(),
        transition=transition[np.ix_(order, order)].copy(),
        means=means[order].copy(),
        variances=variances[order].copy(),
    )
    scale = np.sqrt(reference_variances)
    mean_cost = np.sum(
        (aligned.means - reference_means) ** 2 / (scale**2)
    )
    log_ratio = np.log(aligned.variances) - np.log(reference_variances)
    cost = float(mean_cost + variance_weight * np.sum(log_ratio**2))
    if order.size != k:
        raise ValueError("state matching returned an invalid assignment")
    return StateAlignment(order=order, parameters=aligned, cost=cost)


def align_probabilities(probabilities: Any, order: Any) -> np.ndarray:
    """Reorder state-probability columns from target to reference labels."""
    values = as_f64_2d(probabilities, "probabilities")
    mapping = np.asarray(order, dtype=np.int64)
    if mapping.ndim != 1 or mapping.size != values.shape[1]:
        raise ValueError("order must have one entry per probability column")
    if not np.array_equal(np.sort(mapping), np.arange(mapping.size)):
        raise ValueError("order must be a permutation")
    return np.ascontiguousarray(values[:, mapping])


def align_labels(labels: Any, order: Any) -> np.ndarray:
    """Reorder integer state labels from target to reference labels."""
    mapping = np.asarray(order, dtype=np.int64)
    if mapping.ndim != 1 or mapping.size == 0:
        raise ValueError("order must be a non-empty permutation")
    if not np.array_equal(np.sort(mapping), np.arange(mapping.size)):
        raise ValueError("order must be a permutation")
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
    "RFluxError",
    "StateAlignment",
    "TrendRegimeResult",
    "VolatilityRegimeResult",
    "align_labels",
    "align_parameters",
    "align_probabilities",
    "composite_regime",
    "fit_hmm",
    "match_states",
    "trend_regime",
    "volatility_regime",
]
