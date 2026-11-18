"""Execution scheduling and market-microstructure diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ._common import (
    as_f64_1d,
    as_finite_float,
    as_positive_float,
    as_size,
)
from ._native_backend import native_api as native_module

__all__ = [
    "ExecutionPlan",
    "almgren_chriss",
    "order_book_metrics",
]


@dataclass(frozen=True, slots=True)
class ExecutionPlan:
    trades: np.ndarray
    expected_cost: float
    expected_variance: float
    kappa: float


def _nonnegative(value: Any, name: str) -> float:
    number = as_finite_float(value, name)
    if number < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return number


def almgren_chriss(
    quantity: float,
    *,
    slices: int,
    horizon: float,
    temporary_impact: float,
    permanent_impact: float = 0.0,
    volatility: float = 0.0,
    risk_aversion: float = 0.0,
) -> ExecutionPlan:
    """Optimal liquidation schedule under linear Almgren-Chriss impact."""
    quantity = as_positive_float(quantity, "quantity")
    slices = as_size(slices, "slices")
    horizon = as_positive_float(horizon, "horizon")
    temporary_impact = as_positive_float(temporary_impact, "temporary_impact")
    permanent_impact = _nonnegative(permanent_impact, "permanent_impact")
    volatility = _nonnegative(volatility, "volatility")
    risk_aversion = _nonnegative(risk_aversion, "risk_aversion")

    trades, cost, variance, kappa = native_module.almgren_chriss(
        quantity, slices, horizon, temporary_impact, permanent_impact,
        volatility, risk_aversion,
    )
    return ExecutionPlan(trades, float(cost), float(variance), float(kappa))


def order_book_metrics(
    bid_price: Any,
    bid_size: Any,
    ask_price: Any,
    ask_size: Any,
) -> dict[str, np.ndarray]:
    """Level-1 order-book metrics, implemented in C++."""
    bid = as_f64_1d(bid_price, "bid_price")
    ask = as_f64_1d(ask_price, "ask_price")
    bid_qty = as_f64_1d(bid_size, "bid_size")
    ask_qty = as_f64_1d(ask_size, "ask_size")
    if not (bid.size == ask.size == bid_qty.size == ask_qty.size):
        raise ValueError("order-book arrays must have equal lengths")
    if np.any(~np.isfinite(bid)) or np.any(~np.isfinite(ask)):
        raise ValueError("prices must be finite")
    if np.any(~np.isfinite(bid_qty)) or np.any(~np.isfinite(ask_qty)):
        raise ValueError("sizes must be finite")
    if np.any(bid <= 0.0) or np.any(ask <= 0.0):
        raise ValueError("prices must be positive")
    if np.any(bid_qty < 0.0) or np.any(ask_qty < 0.0):
        raise ValueError("sizes must be non-negative")
    if np.any(ask < bid):
        raise ValueError("ask price must not be below bid price")
    spread, mid, microprice, imbalance = native_module.order_book_metrics(
        bid, bid_qty, ask, ask_qty
    )

    return {
        "spread": spread,
        "mid": mid,
        "microprice": microprice,
        "imbalance": imbalance,
    }
