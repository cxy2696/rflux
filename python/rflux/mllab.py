"""Robust supervised learning and graph signal propagation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ._common import as_f64_1d, as_f64_2d, as_size
from ._native_backend import native_api as native_module

__all__ = [
    "HuberFit",
    "cvar_loss",
    "graph_smooth",
    "huber_regression",
]


@dataclass(frozen=True, slots=True)
class HuberFit:
    coefficients: np.ndarray
    intercept: float
    scale: float
    iterations: int
    converged: bool

    def predict(self, features: Any) -> np.ndarray:
        matrix = as_f64_2d(features, "features")
        if matrix.shape[1] != self.coefficients.size:
            raise ValueError("features have the wrong number of columns")
        return native_module.linear_predict(
            matrix, self.coefficients, float(self.intercept)
        )


def huber_regression(
    features: Any,
    target: Any,
    *,
    epsilon: float = 1.35,
    max_iter: int = 100,
    tol: float = 1e-8,
    fit_intercept: bool = True,
) -> HuberFit:
    """Huber M-estimator fitted by iteratively reweighted least squares."""
    matrix = as_f64_2d(features, "features")
    labels = as_f64_1d(target, "target")
    if matrix.shape[0] != labels.size:
        raise ValueError("features and target must have the same row count")
    if matrix.shape[0] < 2:
        raise ValueError("at least two observations are required")
    if not np.all(np.isfinite(matrix)) or not np.all(np.isfinite(labels)):
        raise ValueError("features and target must be finite")
    epsilon = float(epsilon)
    if not np.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError("epsilon must be finite and positive")
    max_iter = as_size(max_iter, "max_iter")
    tol = float(tol)
    if not np.isfinite(tol) or tol <= 0.0:
        raise ValueError("tol must be finite and positive")
    coefficients, intercept, scale, iterations, converged = (
        native_module.huber_regression(
            matrix, labels, epsilon, max_iter, tol, bool(fit_intercept))
    )
    return HuberFit(
        np.asarray(coefficients, dtype=np.float64),
        float(intercept),
        float(scale),
        int(iterations),
        bool(converged),
    )


def cvar_loss(returns: Any, *, alpha: float = 0.95) -> float:
    """Historical CVaR of losses at confidence ``alpha``, implemented in C++."""
    values = as_f64_1d(returns, "returns")
    if values.size == 0:
        raise ValueError("returns must not be empty")
    if not np.all(np.isfinite(values)):
        raise ValueError("returns must be finite")
    alpha = float(alpha)
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    return float(native_module.cvar_loss(values, alpha))


def graph_smooth(
    signal: Any, adjacency: Any, *, alpha: float = 0.5, steps: int = 3
) -> np.ndarray:
    """Diffuse a per-node signal over a graph, implemented in C++."""
    values = as_f64_1d(signal, "signal")
    matrix = as_f64_2d(adjacency, "adjacency")
    if matrix.shape[0] != matrix.shape[1] or matrix.shape[0] != values.size:
        raise ValueError("adjacency must be square and match the signal length")
    if not np.all(np.isfinite(matrix)) or not np.all(np.isfinite(values)):
        raise ValueError("signal and adjacency must be finite")
    if np.any(matrix < 0.0):
        raise ValueError("adjacency weights must be non-negative")
    alpha = float(alpha)
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1]")
    return native_module.graph_smooth(
        values, matrix, alpha, as_size(steps, "steps"))
