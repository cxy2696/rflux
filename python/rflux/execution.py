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
    n = bid.size
    def fallback() -> tuple[np.ndarray, ...]:
        spread = empty_f64(n)
        mid = empty_f64(n)
        microprice = empty_f64(n)
        imbalance = empty_f64(n)
        check(
            get_lib().rflux_order_book_metrics(
                double_ptr(bid), double_ptr(bid_qty), double_ptr(ask),
                double_ptr(ask_qty), n, double_ptr(spread), double_ptr(mid),
                double_ptr(microprice), double_ptr(imbalance)
            ),
            "rflux_order_book_metrics",
        )
        return spread, mid, microprice, imbalance

    spread, mid, microprice, imbalance = native_or(
        native_module,
        "rflux_order_book_metrics",
        lambda: native_module.order_book_metrics(bid, bid_qty, ask, ask_qty),
        fallback,
    )
    return {
        "spread": spread,
        "mid": mid,
        "microprice": microprice,
        "imbalance": imbalance,
    }
