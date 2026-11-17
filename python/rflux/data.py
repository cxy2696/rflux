"""Optional data adapters for CSV, akshare, yfinance and tushare.

Pandas, akshare, yfinance and tushare are optional dependencies.  The module itself can
be imported without them; only the corresponding loader raises a helpful
``MissingDependencyError`` when a dependency is absent.

All loaders return a wide pandas DataFrame indexed by date with one price
column per instrument, sorted ascending.
"""

from __future__ import annotations

import os
from datetime import date, datetime
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np

from . import qflux


class MissingDependencyError(ImportError):
    """An optional data dependency is not installed."""


class DataSourceError(RuntimeError):
    """A data source returned no usable rows or an unexpected schema."""


def _pandas():
    try:
        import pandas as pd  # type: ignore
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise MissingDependencyError(
            "pandas is required for the data adapters: "
            "pip install 'rflux[data]'"
        ) from exc
    return pd


def _akshare():
    try:
        import akshare as ak  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise MissingDependencyError(
            "akshare is required for load_akshare: pip install 'rflux[akshare]'"
        ) from exc
    return ak


def _tushare():
    try:
        import tushare as ts  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise MissingDependencyError(
            "tushare is required for load_tushare: "
            "pip install 'rflux[tushare]'"
        ) from exc
    return ts


def _yfinance():
    try:
        import yfinance as yf  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise MissingDependencyError(
            "yfinance is required for load_yfinance: "
            "pip install 'rflux[yfinance]'"
        ) from exc
    return yf


def _as_sequence(value: Any, name: str) -> list[str]:
    if isinstance(value, str):
        items = [value]
    elif isinstance(value, Sequence):
        items = [str(item) for item in value]
    else:
        raise TypeError(f"{name} must be a string or a sequence of strings")
    if not items or any(not item for item in items):
        raise ValueError(f"{name} must not be empty")
    return items


def _compact_date(value: Any, default: str, what: str = "date") -> str:
    if value is None:
        return default
    if isinstance(value, (datetime, date)):
        return value.strftime("%Y%m%d")
    text = str(value).strip().replace("-", "").replace("/", "")
    if len(text) != 8 or not text.isdigit():
        raise ValueError(f"invalid {what} {value!r}; use YYYY-MM-DD")
    return text


def _frame_to_returns(
    prices: Any, method: str, fill_method: str | None
) -> Any:
    normalized = str(method).lower()
    if normalized not in {"simple", "log", "logarithmic"}:
        raise ValueError("method must be 'simple' or 'log'")
    log = normalized in {"log", "logarithmic"}
    if fill_method not in (None, "none", "ffill"):
        raise ValueError("fill_method must be None, 'none' or 'ffill'")

    try:
        import pandas as pd  # type: ignore
    except ImportError:
        pd = None

    if pd is not None and isinstance(prices, (pd.DataFrame, pd.Series)):
        frame = prices.ffill() if fill_method == "ffill" else prices
        if isinstance(frame, pd.DataFrame):
            return pd.DataFrame(
                qflux.returns_panel(frame.to_numpy(), log=log),
                index=frame.index,
                columns=frame.columns,
            )
        return pd.Series(
            qflux.returns(frame.to_numpy(), log=log),
            index=frame.index,
            name=frame.name,
        )
    if fill_method not in (None, "none"):
        raise ValueError("fill_method applies only to pandas objects")
    array = np.asarray(prices)
    if array.ndim == 1:
        return qflux.returns(array, log=log)
    if array.ndim == 2:
        return qflux.returns_panel(array, log=log)
    raise ValueError("prices must be 1D or 2D")


def to_returns(
    prices: Any, *, method: str = "simple", fill_method: str | None = None
) -> Any:
    """Convert prices to simple/log returns, preserving pandas labels."""
    return _frame_to_returns(prices, method, fill_method)


def load_csv(
    path: str | Path,
    *,
    date_column: str | None = "date",
    columns: Iterable[str] | None = None,
    **read_csv_kwargs: Any,
) -> Any:
    """Load a wide price panel from CSV.

    If ``date_column`` exists it is parsed and used as the index.  Otherwise the
    existing index is kept.  ``columns`` optionally selects the price columns.
    """
    pd = _pandas()
    frame = pd.read_csv(path, **read_csv_kwargs)
    if frame.empty:
        raise DataSourceError(f"CSV {path!s} contains no rows")
    if date_column is not None and date_column in frame.columns:
        frame[date_column] = pd.to_datetime(frame[date_column], errors="raise")
        frame = frame.set_index(date_column)
    if columns is not None:
        selected = list(columns)
        missing = [column for column in selected if column not in frame.columns]
        if missing:
            raise KeyError(f"missing CSV columns: {missing}")
        frame = frame[selected]
    frame = frame.apply(pd.to_numeric, errors="coerce").sort_index()
    frame = frame[~frame.index.duplicated(keep="last")]
    frame = frame.dropna(how="all")
    if frame.empty:
        raise DataSourceError(f"CSV {path!s} has no numeric rows")
    return frame


def _provider_series(
    raw: Any,
    symbol: str,
    source: str,
    date_column: str,
    price_column: str,
    pd: Any,
) -> Any:
    if raw is None or len(raw) == 0:
        raise DataSourceError(f"{source} returned no rows for {symbol}")
    if date_column not in raw.columns:
        raise DataSourceError(
            f"{source} schema for {symbol} is missing {date_column!r}: "
            f"{list(raw.columns)!r}"
        )
    if price_column not in raw.columns:
        raise DataSourceError(
            f"{source} schema for {symbol} is missing {price_column!r}: "
            f"{list(raw.columns)!r}"
        )
    dates = pd.to_datetime(raw[date_column].astype(str), errors="raise")
    values = pd.to_numeric(raw[price_column], errors="coerce")
    item = pd.Series(values.to_numpy(), index=dates, name=symbol)
    return item[~item.index.duplicated(keep="last")].sort_index()


def _assemble_provider_frame(series: list[Any], source: str, pd: Any) -> Any:
    frame = pd.concat(series, axis=1).sort_index()
    frame.index.name = "date"
    frame = frame.dropna(how="all")
    if frame.empty:
        raise DataSourceError(f"{source} returned only empty price rows")
    return frame


def _load_provider_series(
    requested: list[str],
    source: str,
    date_column: str,
    price_column: str,
    fetch: Callable[[str], Any],
    pd: Any,
) -> Any:
    series = []
    for symbol in requested:
        try:
            raw = fetch(symbol)
        except Exception as exc:  # pragma: no cover - network/provider path
            raise DataSourceError(f"{source} failed for {symbol}: {exc}") from exc
        series.append(
            _provider_series(
                raw, symbol, source, date_column, price_column, pd
            )
        )
    return _assemble_provider_frame(series, source, pd)


def load_akshare(
    symbols: str | Sequence[str],
    *,
    start: Any = None,
    end: Any = None,
    adjust: str = "qfq",
    price_column: str = "收盘",
    period: str = "daily",
) -> Any:
    """Load A-share daily closes from akshare.

    ``symbols`` are akshare symbols such as ``"000001"``.  akshare's Chinese
    schema is normalized to a date-indexed wide frame.
    """
    pd = _pandas()
    ak = _akshare()
    requested = _as_sequence(symbols, "symbols")
    start_date = _compact_date(start, "19700101", "akshare start")
    end_date = _compact_date(
        end, datetime.now().strftime("%Y%m%d"), "akshare end"
    )
    return _load_provider_series(
        requested,
        "akshare",
        "日期",
        price_column,
        lambda symbol: ak.stock_zh_a_hist(
            symbol=symbol,
            period=period,
            start_date=start_date,
            end_date=end_date,
            adjust=adjust,
        ),
        pd,
    )


def load_tushare(
    symbols: str | Sequence[str],
    *,
    start: Any = None,
    end: Any = None,
    token: str | None = None,
    price_column: str = "close",
    **daily_kwargs: Any,
) -> Any:
    """Load A-share daily closes from tushare's ``pro.daily`` endpoint.

    A token can be supplied explicitly or through ``TUSHARE_TOKEN``.  The
    returned prices are raw daily closes; adjustment factors are the caller's
    responsibility.  tushare is an optional dependency.
    """
    pd = _pandas()
    ts = _tushare()
    requested = _as_sequence(symbols, "symbols")
    start_date = _compact_date(start, "19700101", "tushare start")
    end_date = _compact_date(
        end, datetime.now().strftime("%Y%m%d"), "tushare end"
    )
    token = token or os.environ.get("TUSHARE_TOKEN")
    try:
        pro = ts.pro_api(token) if token else ts.pro_api()
    except Exception as exc:  # pragma: no cover - provider path
        raise DataSourceError(f"tushare client setup failed: {exc}") from exc
    return _load_provider_series(
        requested,
        "tushare",
        "trade_date",
        price_column,
        lambda symbol: pro.daily(
            ts_code=symbol,
            start_date=start_date,
            end_date=end_date,
            **daily_kwargs,
        ),
        pd,
    )


def load_yfinance(
    tickers: str | Sequence[str],
    *,
    start: Any = None,
    end: Any = None,
    field: str = "Close",
    auto_adjust: bool = True,
    progress: bool = False,
    **download_kwargs: Any,
) -> Any:
    """Load daily closes from yfinance, handling single and MultiIndex schemas."""
    pd = _pandas()
    yf = _yfinance()
    requested = _as_sequence(tickers, "tickers")
    kwargs = dict(download_kwargs)
    kwargs.update(
        start=start,
        end=end,
        auto_adjust=auto_adjust,
        progress=progress,
        group_by="column",
        actions=False,
        threads=True,
    )
    try:
        raw = yf.download(requested, **kwargs)
    except Exception as exc:  # pragma: no cover - network/provider path
        raise DataSourceError(f"yfinance failed: {exc}") from exc
    if raw is None or len(raw) == 0:
        raise DataSourceError("yfinance returned no rows")
    if not isinstance(raw, pd.DataFrame):
        raise DataSourceError("yfinance returned an unexpected object")

    if isinstance(raw.columns, pd.MultiIndex):
        level = None
        for index in range(raw.columns.nlevels):
            if field in raw.columns.get_level_values(index):
                level = index
                break
        if level is None:
            raise DataSourceError(
                f"yfinance columns do not contain field {field!r}: "
                f"{raw.columns.tolist()!r}"
            )
        frame = raw.xs(field, axis=1, level=level)
        if isinstance(frame, pd.Series):
            frame = frame.to_frame(name=requested[0])
    elif field in raw.columns:
        frame = raw[field]
        if isinstance(frame, pd.Series):
            frame = frame.to_frame(name=requested[0])
    else:
        raise DataSourceError(
            f"yfinance columns do not contain field {field!r}: "
            f"{list(raw.columns)!r}"
        )
    frame = frame.apply(pd.to_numeric, errors="coerce")
    frame = frame.sort_index()
    frame.index.name = "date"
    if len(requested) == 1 and frame.shape[1] == 1:
        frame.columns = [requested[0]]
    frame = frame.dropna(how="all")
    if frame.empty:
        raise DataSourceError("yfinance returned only empty price rows")
    return frame


def load_prices(source: str, **kwargs: Any) -> Any:
    """Unified loader: ``csv``, ``akshare``, ``yfinance`` or ``tushare``."""
    normalized = str(source).lower()
    if normalized == "csv":
        return load_csv(**kwargs)
    if normalized == "akshare":
        return load_akshare(**kwargs)
    if normalized in {"yfinance", "yahoo"}:
        return load_yfinance(**kwargs)
    if normalized == "tushare":
        return load_tushare(**kwargs)
    raise ValueError("source must be 'csv', 'akshare', 'yfinance' or 'tushare'")


__all__ = [
    "DataSourceError",
    "MissingDependencyError",
    "load_akshare",
    "load_csv",
    "load_prices",
    "load_tushare",
    "load_yfinance",
    "to_returns",
]
