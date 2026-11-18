"""Execution scheduling and market-microstructure diagnostics."""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
from typing import Any

import numpy as np

from ._common import (
    as_f64_1d,
    as_finite_float,
    as_positive_float,
    as_size,
    double_ptr,
    empty_f64,
)
from ._ffi import check, get_lib
from ._native_backend import native_module, native_or

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

    def native() -> ExecutionPlan:
        trades, cost, variance, kappa = native_module.almgren_chriss(
            quantity,
            slices,
            horizon,
            temporary_impact,
            permanent_impact,
            volatility,
            risk_aversion,
        )
        return ExecutionPlan(
            trades, float(cost), float(variance), float(kappa))

    def fallback() -> ExecutionPlan:
        trades = empty_f64(slices)
        cost = ctypes.c_double()
        variance = ctypes.c_double()
        kappa = ctypes.c_double()
        check(
            get_lib().rflux_almgren_chriss_plan(
                float(quantity),
                slices,
                float(horizon),
                float(temporary_impact),
                float(permanent_impact),
                float(volatility),
                float(risk_aversion),
                double_ptr(trades),
                ctypes.byref(cost),
                ctypes.byref(variance),
                ctypes.byref(kappa),
            ),
            "rflux_almgren_chriss_plan",
        )
        return ExecutionPlan(
            trades, float(cost.value), float(variance.value),
            float(kappa.value)
        )

    return native_or(
        native_module, "rflux_almgren_chriss", native, fallback
    )


def order_book_metrics(
    bid_price: Any,
    bid_size: Any,
    ask_price: Any,
    ask_size: Any,
) -> dict[str, np.ndarray]:
    """Level-1 spread, mid, microprice and size imbalance.

    ``imbalance`` is ``(bid_size - ask_size) / (bid_size + ask_size)`` and
    ``microprice`` weights the opposite side by its available size.
    """
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
    total = bid_qty + ask_qty
    with np.errstate(divide="ignore", invalid="ignore"):
        microprice = np.where(
            total > 0.0,
            (ask * bid_qty + bid * ask_qty) / np.where(total > 0.0, total, 1.0),
            (bid + ask) * 0.5,
        )
        imbalance = np.where(
            total > 0.0, (bid_qty - ask_qty) / np.where(total > 0.0, total, 1.0), 0.0
        )
    return {
        "spread": ask - bid,
        "mid": (ask + bid) * 0.5,
        "microprice": microprice,
        "imbalance": imbalance,
    }
