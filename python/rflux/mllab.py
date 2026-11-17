"""Robust supervised learning and graph signal propagation.

These are small, dependency-free NumPy implementations for the research
layer.  Heavy GNN training belongs in a separate optional backend; the
C++ core exposes the stable algorithm interfaces in
``include/rflux/core/algorithm.hpp``.
"""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
from typing import Any

import numpy as np

from ._common import as_f64_1d, as_f64_2d, as_size, double_ptr, empty_f64
from ._ffi import check, get_lib
from ._native_backend import native_module, raise_as_rflux

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
        return matrix @ self.coefficients + self.intercept


def _c_huber(
    matrix: np.ndarray,
    labels: np.ndarray,
    epsilon: float,
    max_iter: int,
    tol: float,
    fit_intercept: bool,
) -> HuberFit:
    coefficients = empty_f64(matrix.shape[1])
    intercept = ctypes.c_double()
    scale = ctypes.c_double()
    iterations = ctypes.c_size_t()
    converged = ctypes.c_int32()
    check(
        get_lib().rflux_huber_regression(
            double_ptr(matrix),
            double_ptr(labels),
            matrix.shape[0],
            matrix.shape[1],
            float(epsilon),
            max_iter,
            float(tol),
            1 if fit_intercept else 0,
            double_ptr(coefficients),
            ctypes.byref(intercept),
            ctypes.byref(scale),
            ctypes.byref(iterations),
            ctypes.byref(converged),
        ),
        "rflux_huber_regression",
    )
    return HuberFit(
        coefficients,
        float(intercept.value),
        float(scale.value),
        int(iterations.value),
        bool(converged.value),
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
    """Huber M-estimator fitted by iteratively reweighted least squares.

    ``epsilon`` is measured in robust scale units.  The returned coefficients
    exclude the intercept; ``predict`` adds it back.
    """
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

    if native_module is not None:
        try:
            coefficients, intercept, scale, iterations, converged = (
                native_module.huber_regression(
                    matrix,
                    labels,
                    epsilon,
                    max_iter,
                    tol,
                    bool(fit_intercept),
                )
            )
        except native_module.NativeError as exc:
            raise_as_rflux("rflux_huber_regression", exc)
        return HuberFit(
            np.asarray(coefficients, dtype=np.float64),
            float(intercept),
            float(scale),
            int(iterations),
            bool(converged),
        )
    return _c_huber(matrix, labels, epsilon, max_iter, tol, fit_intercept)


def cvar_loss(returns: Any, *, alpha: float = 0.95) -> float:
    """Historical CVaR of losses at confidence ``alpha`` (positive is a loss)."""
    values = as_f64_1d(returns, "returns")
    if values.size == 0:
        raise ValueError("returns must not be empty")
    if not np.all(np.isfinite(values)):
        raise ValueError("returns must be finite")
    alpha = float(alpha)
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    losses = -values
    tail = np.quantile(losses, alpha)
    selected = losses[losses >= tail]
    return float(selected.mean())


def graph_smooth(
    signal: Any, adjacency: Any, *, alpha: float = 0.5, steps: int = 3
) -> np.ndarray:
    """Diffuse a per-node signal over a graph.

    This is a lightweight graph-regularisation primitive, not a trained GNN:
    ``s <- (1-alpha) s + alpha D^-1 A s`` for ``steps`` iterations.
    """
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
    steps = as_size(steps, "steps")
    degree = matrix.sum(axis=1)
    normalized = np.divide(
        matrix,
        degree[:, None],
        out=np.zeros_like(matrix),
        where=degree[:, None] > 0.0,
    )
    result = values.astype(np.float64, copy=True)
    for _ in range(steps):
        result = (1.0 - alpha) * result + alpha * (normalized @ result)
    return result
