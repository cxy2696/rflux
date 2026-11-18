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
from typing import Any, Callable

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
from ._native_backend import native_call, native_module


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
    """Fitted HMM state and latest filtered/forecast information.

    This is a detached snapshot: the underlying native model is closed before
    the object is returned, so callers cannot accidentally keep a stateful
    handle alive after a one-shot report.
    """

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
    sample and should not be treated as point-in-time signals.  A model handle
    is stateful and must be used by one thread at a time.
    """

    def __init__(self, n_states: int, n_features: int = 1):
        self._n_states = as_size(n_states, "n_states")
        self._n_features = as_size(n_features, "n_features")
        self._native_model: Any = None
        self._backend = "ctypes"
        self._handle: ctypes.c_void_p | None = ctypes.c_void_p()
        if native_module is not None:
            self._native_model = native_call(
                native_module,
                "rflux_hmm_create",
                lambda: native_module.GaussianHMM(
                    self._n_states, self._n_features),
            )
            self._backend = "native"
            self._handle = None
            return
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
        if self._native_model is not None:
            raise RuntimeError("native GaussianHMM has no ctypes handle")
        if self._handle is None:
            raise RuntimeError("GaussianHMM handle is closed")
        return self._handle

    def _state_probabilities(self, state_probs: Any) -> np.ndarray:
        probabilities = as_f64_1d(state_probs, "state_probs")
        if probabilities.size != self._n_states:
            raise ValueError(f"state_probs must have length {self._n_states}")
        return probabilities

    def _native_or_abi(
        self,
        function: str,
        native: Callable[[Any], Any],
        fallback: Callable[[], Any],
    ) -> Any:
        """Run one model operation on the active backend.

        Callers pass the native projection and the C ABI fallback; this helper
        owns the backend choice and the NativeError translation so each method
        contains only its own arithmetic and argument marshalling.
        """
        if self._native_model is None:
            return fallback()
        return native_call(
            native_module, function,
            lambda: native(self._native_model),
        )

    def close(self) -> None:
        if self._native_model is not None:
            self._native_model = None
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
        open_model = self._native_model is not None or self._handle is not None
        state = "open" if open_model else "closed"
        return (
            f"GaussianHMM(n_states={self._n_states}, "
            f"n_features={self._n_features}, {state}, "
            f"backend={self._backend})"
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
        def native(model: Any) -> "GaussianHMM":
            model.fit(
                data,
                max_iter=max_iter,
                n_init=n_init,
                tol=float(tol),
                variance_floor=float(variance_floor),
                seed=as_u64(seed, "seed"),
                sort_states=bool(sort_states),
            )
            return self

        def fallback() -> "GaussianHMM":
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

        return self._native_or_abi("rflux_hmm_fit", native, fallback)

    @property
    def fit_info(self) -> HMMFitInfo:
        def native(model: Any) -> HMMFitInfo:
            iterations, converged, log_likelihood, aic, bic = model.fit_info
            return HMMFitInfo(
                int(iterations),
                bool(converged),
                float(log_likelihood),
                float(aic),
                float(bic),
            )

        def fallback() -> HMMFitInfo:
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

        return self._native_or_abi("rflux_hmm_fit_summary", native, fallback)

    def score(self, observations: Any) -> float:
        """Log-likelihood of observations under the current parameters.

        Works after ``fit`` and after ``set_parameters``; useful for
        walk-forward or holdout model selection.
        """
        data = _as_observations(observations, self._n_features)
        return self._native_or_abi(
            "rflux_hmm_score",
            lambda model: float(model.score(data)),
            lambda: call_scalar(
                "rflux_hmm_score", get_lib().rflux_hmm_score,
                self._require_handle(), double_ptr(data), data.shape[0]),
        )

    def filtered_states(self, observations: Any) -> np.ndarray:
        """Causal state labels from the filtered distribution (argmax)."""
        return np.argmax(self.filtered(observations), axis=1).astype(np.int32)

    def conditional_mean(self, state_probs: Any, feature: int = 0) -> float:
        """Expected emission mean under a state distribution.

        This is ``sum_k P(s=k) * mu[k, feature]`` under the model's current
        parameters.  It is an in-sample conditional mean, not a price forecast.
        """
        probabilities = self._state_probabilities(state_probs)
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
        def fallback() -> HMMParameters:
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

        return self._native_or_abi(
            "rflux_hmm_get_parameters",
            lambda model: HMMParameters(*model.parameters),
            fallback,
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
        def native(model: Any) -> "GaussianHMM":
            model.set_parameters(start, transition, means, variances)
            return self

        def fallback() -> "GaussianHMM":
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

        return self._native_or_abi(
            "rflux_hmm_set_parameters", native, fallback)

    def viterbi(self, observations: Any) -> np.ndarray:
        """Full-sample Viterbi path (non-causal)."""
        data = _as_observations(observations, self._n_features)
        return self._native_or_abi(
            "rflux_hmm_viterbi",
            lambda model: model.viterbi(data),
            lambda: call_i32(
                "rflux_hmm_viterbi", data.shape[0],
                get_lib().rflux_hmm_viterbi, self._require_handle(),
                double_ptr(data), data.shape[0]),
        )

    def _probability_matrix(
        self, name: str, observations: Any, method: str
    ) -> np.ndarray:
        data = _as_observations(observations, self._n_features)

        def fallback() -> np.ndarray:
            out = call_vector(
                name, data.shape[0] * self._n_states, getattr(get_lib(), name),
                self._require_handle(), double_ptr(data), data.shape[0])
            return out.reshape(data.shape[0], self._n_states)

        return self._native_or_abi(
            name, lambda model: getattr(model, method)(data), fallback)

    def filtered(self, observations: Any) -> np.ndarray:
        """Causal filtered probabilities, shape (n_obs, n_states)."""
        return self._probability_matrix(
            "rflux_hmm_filtered", observations, "filtered")

    def smoothed(self, observations: Any) -> np.ndarray:
        """Full-sample smoothed probabilities, shape (n_obs, n_states)."""
        return self._probability_matrix(
            "rflux_hmm_smoothed", observations, "smoothed")

    def forecast(self, state_probs: Any, horizon: int = 1) -> np.ndarray:
        """Prior hidden-state distribution ``state_probs @ transition**horizon``."""
        probabilities = self._state_probabilities(state_probs)
        horizon = as_size(horizon, "horizon", minimum=0)
        return self._native_or_abi(
            "rflux_hmm_forecast",
            lambda model: model.forecast(probabilities, horizon),
            lambda: call_vector(
                "rflux_hmm_forecast", self._n_states,
                get_lib().rflux_hmm_forecast, self._require_handle(),
                double_ptr(probabilities), horizon),
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
        single = np.ascontiguousarray(single)
        return self._native_or_abi(
            "rflux_hmm_filter_step",
            lambda model: model.filter_step(probabilities, single),
            lambda: call_vector(
                "rflux_hmm_filter_step", self._n_states,
                get_lib().rflux_hmm_filter_step, self._require_handle(),
                double_ptr(probabilities), double_ptr(single)),
        )

    def stationary_distribution(self) -> np.ndarray:
        """Long-run distribution implied by the current transition matrix."""
        return self._native_or_abi(
            "rflux_hmm_stationary",
            lambda model: model.stationary_distribution(),
            lambda: call_vector(
                "rflux_hmm_stationary", self._n_states,
                get_lib().rflux_hmm_stationary, self._require_handle()),
        )

    def expected_durations(self) -> np.ndarray:
        """Expected sojourn length of each state from the transition diagonal."""
        return self._native_or_abi(
            "rflux_hmm_expected_durations",
            lambda model: model.expected_durations(),
            lambda: call_vector(
                "rflux_hmm_expected_durations", self._n_states,
                get_lib().rflux_hmm_expected_durations,
                self._require_handle()),
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
    """Hysteresis-smoothed causal regime labels.

    A challenger state must reach ``entry_threshold`` while the incumbent falls
    to ``exit_threshold`` and the incumbent has lasted ``min_duration`` bars.
    This suppresses one-bar HMM flicker and the turnover it would cause.
    """
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
    labels = np.empty(probabilities.shape[0], dtype=np.int32)
    current = int(np.argmax(probabilities[0]))
    labels[0] = current
    last_switch = 0
    switches: list[int] = []
    for t in range(1, probabilities.shape[0]):
        challenger = int(np.argmax(probabilities[t]))
        if (
            challenger != current
            and probabilities[t, challenger] >= entry
            and probabilities[t, current] <= exit_
            and t - last_switch >= min_duration
        ):
            current = challenger
            last_switch = t
            switches.append(t)
        labels[t] = current
    return StickyRegimeResult(
        probabilities,
        labels,
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
    """Fit an HMM and return a detached latest-filtered snapshot.

    The model handle is always closed, including on failure.  ``kwargs`` are
    forwarded to :meth:`GaussianHMM.fit`.
    """
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


def _validate_variance_weight(value: float) -> float:
    if not np.isfinite(value) or value < 0.0:
        raise ValueError("variance_weight must be finite and non-negative")
    return float(value)


def _alignment_cost(
    reference_means: np.ndarray,
    reference_variances: np.ndarray,
    target_means: np.ndarray,
    target_variances: np.ndarray,
    variance_weight: float,
) -> np.ndarray:
    scale = np.sqrt(reference_variances)
    mean_cost = np.sum(
        (target_means[None, :, :] - reference_means[:, None, :]) ** 2
        / (scale[:, None, :] ** 2),
        axis=2,
    )
    log_ratio = np.log(target_variances[None, :, :]) - np.log(
        reference_variances[:, None, :]
    )
    return mean_cost + variance_weight * np.sum(log_ratio**2, axis=2)


def _matched_order(
    reference_means: np.ndarray,
    reference_variances: np.ndarray,
    target_means: np.ndarray,
    target_variances: np.ndarray,
    variance_weight: float,
) -> np.ndarray:
    cost = _alignment_cost(
        reference_means,
        reference_variances,
        target_means,
        target_variances,
        variance_weight,
    )
    return _hungarian(cost)


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
        raise ValueError(
            "reference and target must have the same state/feature shape"
        )
    variance_weight = _validate_variance_weight(variance_weight)
    return _matched_order(
        reference_means,
        reference_variances,
        target_means,
        target_variances,
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
    variance_weight = _validate_variance_weight(variance_weight)
    order = _matched_order(
        reference_means,
        reference_variances,
        means,
        variances,
        variance_weight,
    )
    k = start.size
    aligned = HMMParameters(
        start_prob=start[order].copy(),
        transition=transition[np.ix_(order, order)].copy(),
        means=means[order].copy(),
        variances=variances[order].copy(),
    )
    cost = float(
        np.trace(
            _alignment_cost(
                reference_means,
                reference_variances,
                aligned.means,
                aligned.variances,
                variance_weight,
            )
        )
    )
    if order.size != k:
        raise ValueError("state matching returned an invalid assignment")
    return StateAlignment(order=order, parameters=aligned, cost=cost)


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
