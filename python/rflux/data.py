"""Data adapters for local CSV, akshare and yfinance.

The legacy helpers (``load_csv``, ``load_akshare``, ``load_yfinance``) return
wide close-price panels for the existing rflux tools.

The unified layer returns a canonical long OHLCV DataFrame with a ``symbol``
column, a tz-naive exchange-local ``datetime`` index and float32 OHLC / int64
share volume.  It is created through ``DataLoaderFactory``.

Pandas and provider SDKs are optional.  Importing :mod:`rflux.data` never
imports a provider; missing packages fail only when the matching loader is used.
"""

from __future__ import annotations

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


def _akshare_etf_endpoint(ak: Any) -> Any:
    endpoint = getattr(ak, "fund_etf_hist_em", None)
    if endpoint is None:
        raise DataSourceError(
            "akshare does not provide fund_etf_hist_em; upgrade akshare"
        )
    return endpoint


def _akshare_asset_type(asset_type: str, symbol: str) -> str:
    normalized = str(asset_type).strip().lower()
    if normalized == "auto":
        return "etf" if _akshare_is_etf(symbol) else "stock"
    if normalized not in {"etf", "stock"}:
        raise ValueError("asset_type must be 'auto', 'etf' or 'stock'")
    return normalized


def _akshare_is_etf(symbol: str) -> bool:
    code = normalize_symbol(symbol).split(".", 1)[0]
    return code.startswith("5") or code.startswith(("15", "16"))


def _akshare_hist_endpoint(ak: Any) -> Any:
    endpoint = getattr(ak, "stock_zh_a_hist_em", None)
    if endpoint is None:
        endpoint = getattr(ak, "stock_zh_a_hist", None)
    if endpoint is None:
        raise DataSourceError(
            "akshare has neither stock_zh_a_hist_em nor stock_zh_a_hist"
        )
    return endpoint


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
    asset_type: str = "auto",
) -> Any:
    """Load A-share stock or ETF daily closes from akshare.

    ``symbols`` are canonical or akshare symbols such as ``"600519"`` or
    ``"510300.SH"``.  ``asset_type="auto"`` uses ``fund_etf_hist_em`` for ETF
    codes and the stock endpoint otherwise.  akshare's Chinese schema is
    normalized to a date-indexed wide frame.
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
        lambda symbol: (
            _akshare_etf_endpoint(ak)
            if _akshare_asset_type(asset_type, symbol) == "etf"
            else _akshare_hist_endpoint(ak)
        )(
            symbol=symbol,
            period=period,
            start_date=start_date,
            end_date=end_date,
            adjust=adjust,
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
    """Unified loader: ``csv``, ``akshare`` or ``yfinance``."""
    normalized = str(source).lower()
    if normalized == "csv":
        return load_csv(**kwargs)
    if normalized == "akshare":
        return load_akshare(**kwargs)
    if normalized in {"yfinance", "yahoo"}:
        return load_yfinance(**kwargs)
    raise ValueError("source must be 'csv', 'akshare' or 'yfinance'")


__all__ = [
    "DataSourceError",
    "MissingDependencyError",
    "load_akshare",
    "load_csv",
    "load_prices",
    "load_yfinance",
    "to_returns",
]


# ---------------------------------------------------------------------------
# Unified long OHLCV contract and provider adapters (0.7 data layer)
# ---------------------------------------------------------------------------

from abc import ABC, abstractmethod

from .symbols import SymbolError, normalize_symbol, provider_symbol

OHLCV_COLUMNS = ("symbol", "open", "high", "low", "close", "volume")
OHLCV_DTYPES = {
    "symbol": "object",
    "open": "float32",
    "high": "float32",
    "low": "float32",
    "close": "float32",
    "volume": "int64",
}

_DEFAULT_COLUMN_MAP = {
    "date": "datetime",
    "datetime": "datetime",
    "trade_date": "datetime",
    "time": "datetime",
    "日期": "datetime",
    "时间": "datetime",
    "open": "open",
    "开盘": "open",
    "high": "high",
    "最高": "high",
    "low": "low",
    "最低": "low",
    "close": "close",
    "收盘": "close",
    "volume": "volume",
    "vol": "volume",
    "成交量": "volume",
    "symbol": "symbol",
    "code": "symbol",
    "证券代码": "symbol",
}


class ContractError(DataSourceError):
    """A frame does not satisfy the unified OHLCV contract."""


def _contract_columns() -> list[str]:
    return list(OHLCV_COLUMNS)


def _normalize_columns(columns: Iterable[Any]) -> list[str]:
    return [str(column).strip().lower() for column in columns]


def _canonical_symbols(symbols: str | Sequence[str]) -> list[str]:
    requested = _as_sequence(symbols, "symbols")
    result: list[str] = []
    seen: set[str] = set()
    for item in requested:
        try:
            canonical = normalize_symbol(item)
        except SymbolError as exc:
            raise ValueError(str(exc)) from exc
        if canonical not in seen:
            result.append(canonical)
            seen.add(canonical)
    return result


def _naive_datetime_index(values: Any, pd: Any) -> Any:
    if isinstance(values, pd.Series):
        source = values
        if pd.api.types.is_integer_dtype(source.dtype):
            source = source.astype("Int64").astype("string")
        else:
            strings = source.astype("string").str.strip()
            non_null = strings.dropna()
            if len(non_null) and bool(
                non_null.str.fullmatch(r"\d{8}").all()
            ):
                source = strings
        is_compact = (
            source.dtype == "string"
            and len(source.dropna())
            and bool(source.dropna().str.fullmatch(r"\d{8}").all())
        )
        parsed = pd.to_datetime(
            source, errors="raise", format="%Y%m%d" if is_compact else None
        )
        if getattr(parsed.dt, "tz", None) is not None:
            parsed = parsed.dt.tz_localize(None)
        return parsed
    if isinstance(values, pd.DatetimeIndex):
        index = values
    elif pd.api.types.is_integer_dtype(getattr(values, "dtype", None)):
        index = pd.DatetimeIndex(
            pd.to_datetime(
                pd.Index(values).astype("Int64").astype("string"),
                errors="raise",
                format="%Y%m%d",
            )
        )
    else:
        index = pd.DatetimeIndex(pd.to_datetime(values, errors="raise"))
    if index.tz is not None:
        index = index.tz_localize(None)
    return index


def _date_bound(value: Any, pd: Any) -> Any:
    if value is None:
        return None
    timestamp = pd.Timestamp(value)
    if timestamp.tz is not None:
        timestamp = timestamp.tz_localize(None)
    return timestamp


def _filter_dates(frame: Any, start: Any, end: Any, pd: Any) -> Any:
    start_bound = _date_bound(start, pd)
    end_bound = _date_bound(end, pd)
    if start_bound is not None:
        frame = frame.loc[frame.index >= start_bound]
    if end_bound is not None:
        frame = frame.loc[frame.index <= end_bound]
    return frame


def validate_ohlcv(frame: Any, *, source: str = "data") -> Any:
    """Validate and normalize the canonical long OHLCV frame.

    Contract:
    - DatetimeIndex named ``datetime`` (tz-naive exchange date/time)
    - columns ``symbol, open, high, low, close, volume``
    - OHLC float32, volume int64, symbol str
    - unique ``(datetime, symbol)`` rows, sorted by datetime then symbol
    - finite positive prices, non-negative integer volume, high/low bounds
    """
    pd = _pandas()
    if not isinstance(frame, pd.DataFrame):
        raise ContractError(f"{source} must return a pandas DataFrame")
    if frame.empty:
        raise ContractError(f"{source} returned no rows")
    if "symbol" not in frame.columns:
        raise ContractError(f"{source} is missing the symbol column")
    if "datetime" in frame.columns:
        frame = frame.set_index("datetime")
    if not isinstance(frame.index, pd.DatetimeIndex):
        try:
            frame.index = _naive_datetime_index(frame.index, pd)
        except Exception as exc:
            raise ContractError(f"{source} has an invalid datetime index") from exc
    frame.index.name = "datetime"
    if frame.index.tz is not None:
        frame.index = frame.index.tz_localize(None)
    if frame.index.isna().any():
        raise ContractError(f"{source} contains a missing datetime")

    missing = [column for column in OHLCV_COLUMNS if column not in frame.columns]
    if missing:
        raise ContractError(f"{source} is missing columns: {missing}")
    frame = frame.loc[:, _contract_columns()].copy()
    if frame["symbol"].isna().any():
        raise ContractError(f"{source} contains missing symbols")
    frame["symbol"] = frame["symbol"].astype(str).str.strip()
    if (frame["symbol"] == "").any():
        raise ContractError(f"{source} contains empty symbols")

    for column in ("open", "high", "low", "close"):
        converted = pd.to_numeric(frame[column], errors="coerce")
        if converted.isna().any():
            raise ContractError(f"{source} contains non-numeric {column} values")
        if (converted <= 0.0).any():
            raise ContractError(f"{source} contains non-positive {column} values")
        frame[column] = converted
    volume = pd.to_numeric(frame["volume"], errors="coerce")
    if volume.isna().any():
        raise ContractError(f"{source} contains non-numeric volume values")
    if (volume < 0.0).any():
        raise ContractError(f"{source} contains negative volume values")
    rounded = np.rint(volume.to_numpy(dtype=np.float64))
    if not np.allclose(volume.to_numpy(dtype=np.float64), rounded, atol=1e-8):
        raise ContractError(f"{source} volume must be an integer share count")
    frame["volume"] = rounded.astype(np.int64)

    high = frame["high"].to_numpy(dtype=np.float64)
    low = frame["low"].to_numpy(dtype=np.float64)
    opening = frame["open"].to_numpy(dtype=np.float64)
    close = frame["close"].to_numpy(dtype=np.float64)
    tolerance = 1e-6 * np.maximum(1.0, np.abs(high))
    if np.any(high + tolerance < np.maximum.reduce([opening, close, low])):
        raise ContractError(f"{source} has high below open/close/low")
    if np.any(low - tolerance > np.minimum.reduce([opening, close, high])):
        raise ContractError(f"{source} has low above open/close/high")

    pairs = pd.MultiIndex.from_frame(frame.reset_index()[["datetime", "symbol"]])
    if pairs.has_duplicates:
        raise ContractError(
            f"{source} has duplicate (datetime, symbol) rows"
        )

    frame = frame.set_index("symbol", append=True).sort_index(
        level=[0, 1], sort_remaining=False
    )
    frame = frame.reset_index("symbol")
    frame.index.name = "datetime"
    frame = frame.loc[:, _contract_columns()]
    for column, dtype in OHLCV_DTYPES.items():
        if column == "symbol":
            frame[column] = frame[column].astype("string").astype("object")
        else:
            frame[column] = frame[column].astype(dtype)
    return frame


def _canonicalize_ohlcv(
    raw: Any,
    symbol: str,
    source: str,
    *,
    column_map: dict[str, str] | None = None,
    volume_multiplier: float = 1.0,
) -> Any:
    pd = _pandas()
    if not isinstance(raw, pd.DataFrame):
        raise ContractError(f"{source} returned an unexpected object for {symbol}")
    if raw.empty:
        raise ContractError(f"{source} returned no rows for {symbol}")
    frame = raw.copy()
    if isinstance(frame.columns, pd.MultiIndex):
        known = set(_DEFAULT_COLUMN_MAP)
        flattened: list[str] = []
        for column in frame.columns:
            parts = [str(part).strip().lower() for part in column]
            field = next((part for part in parts if part in known), parts[0])
            flattened.append(field)
        frame.columns = flattened
    if "datetime" not in _normalize_columns(frame.columns):
        if isinstance(frame.index, pd.DatetimeIndex):
            frame = frame.reset_index()
        elif frame.index.name is not None:
            frame = frame.reset_index()
    frame.columns = _normalize_columns(frame.columns)
    mapping = dict(_DEFAULT_COLUMN_MAP)
    if column_map:
        mapping.update(
            {str(key).strip().lower(): str(value).strip().lower()
             for key, value in column_map.items()}
        )
    frame = frame.rename(
        columns={
            key: value for key, value in mapping.items() if key in frame.columns
        }
    )
    if "datetime" not in frame.columns:
        raise ContractError(f"{source} schema for {symbol} has no datetime column")
    if "close" not in frame.columns:
        if "adj close" in frame.columns:
            frame["close"] = frame["adj close"]
        else:
            raise ContractError(f"{source} schema for {symbol} has no close column")
    if "volume" not in frame.columns:
        raise ContractError(f"{source} schema for {symbol} has no volume column")
    for column in ("open", "high", "low"):
        if column not in frame.columns:
            raise ContractError(f"{source} schema for {symbol} has no {column} column")
    frame["symbol"] = symbol
    frame["datetime"] = _naive_datetime_index(frame["datetime"], pd)
    frame = frame.set_index("datetime")
    frame["volume"] = pd.to_numeric(frame["volume"], errors="coerce") * float(volume_multiplier)
    frame = frame.loc[:, _contract_columns()]
    return validate_ohlcv(frame, source=f"{source}[{symbol}]")


class DataLoader(ABC):
    """Unified long OHLCV loader interface."""

    source_name = "base"

    @abstractmethod
    def _load_symbol(self, symbol: str, start: Any, end: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    def load(
        self,
        symbols: str | Sequence[str],
        start: Any = None,
        end: Any = None,
        **kwargs: Any,
    ) -> Any:
        pd = _pandas()
        requested = _canonical_symbols(symbols)
        frames = [
            _filter_dates(
                self._load_symbol(symbol, start, end, **kwargs),
                start,
                end,
                pd,
            )
            for symbol in requested
        ]
        frame = pd.concat(frames, axis=0)
        return validate_ohlcv(frame, source=self.source_name)


class LocalCsvLoader(DataLoader):
    """Load local per-symbol CSV files or one long CSV with a symbol column."""

    source_name = "csv"

    def __init__(
        self,
        root: str | Path | None = None,
        *,
        path: str | Path | None = None,
        date_column: str = "date",
        symbol_column: str = "symbol",
        column_map: dict[str, str] | None = None,
        volume_multiplier: float = 1.0,
        read_csv_kwargs: dict[str, Any] | None = None,
    ):
        if (root is None) == (path is None):
            raise ValueError("provide exactly one of root or path")
        self.root = Path(root).expanduser() if root is not None else None
        self.path = Path(path).expanduser() if path is not None else None
        self.date_column = date_column
        self.symbol_column = symbol_column
        self.column_map = dict(column_map or {})
        self.column_map.setdefault(date_column.lower(), "datetime")
        self.column_map.setdefault(symbol_column.lower(), "symbol")
        self.volume_multiplier = float(volume_multiplier)
        self.read_csv_kwargs = dict(read_csv_kwargs or {})
        self._path_frame = None

    def _read(self, path: Path) -> Any:
        pd = _pandas()
        if not path.exists():
            raise DataSourceError(f"CSV file not found: {path}")
        try:
            return pd.read_csv(path, **self.read_csv_kwargs)
        except Exception as exc:
            raise DataSourceError(f"cannot read CSV {path}: {exc}") from exc

    def _candidate_paths(self, symbol: str) -> list[Path]:
        assert self.root is not None
        base = symbol.split(".", 1)[0]
        return [
            self.root / f"{symbol}.csv",
            self.root / f"{base}.csv",
            self.root / f"{symbol.lower()}.csv",
            self.root / f"{base.lower()}.csv",
        ]

    def _load_symbol(self, symbol: str, start: Any, end: Any, **kwargs: Any) -> Any:
        pd = _pandas()
        if self.path is not None:
            if self._path_frame is None:
                self._path_frame = self._read(self.path)
            raw = self._path_frame
            columns = _normalize_columns(raw.columns)
            if self.symbol_column.lower() not in columns:
                raise ContractError(
                    f"long CSV {self.path} needs a {self.symbol_column!r} column"
                )
            symbol_values = raw.iloc[
                :, columns.index(self.symbol_column.lower())
            ].astype(str)
            mask = symbol_values == symbol
            if not mask.any():
                base = symbol.split(".", 1)[0]
                mask = symbol_values == base
            raw = raw.loc[mask]
            if raw.empty:
                raise DataSourceError(f"CSV {self.path} has no rows for {symbol}")
        else:
            raw = None
            for candidate in self._candidate_paths(symbol):
                if candidate.exists():
                    raw = self._read(candidate)
                    break
            if raw is None:
                raise DataSourceError(f"no CSV file for {symbol} under {self.root}")
        result = _canonicalize_ohlcv(
            raw,
            symbol,
            "csv",
            column_map=self.column_map,
            volume_multiplier=self.volume_multiplier,
        )
        return _filter_dates(result, start, end, pd)


class AkShareLoader(DataLoader):
    """A-share stock or ETF daily OHLCV adapter for akshare.

    ``asset_type="auto"`` selects ``fund_etf_hist_em`` for ETF codes and
    ``stock_zh_a_hist_em`` otherwise.  Pass ``asset_type="etf"`` for a fund
    that does not match the usual code prefixes.
    """

    source_name = "akshare"

    def __init__(
        self,
        *,
        adjust: str = "qfq",
        period: str = "daily",
        volume_multiplier: float = 100.0,
        column_map: dict[str, str] | None = None,
        asset_type: str = "auto",
    ):
        self.adjust = adjust
        self.period = period
        self.asset_type = str(asset_type).strip().lower()
        if self.asset_type not in {"auto", "etf", "stock"}:
            raise ValueError("asset_type must be 'auto', 'etf' or 'stock'")
        self.volume_multiplier = float(volume_multiplier)
        self.column_map = dict(column_map or {})

    def _load_symbol(self, symbol: str, start: Any, end: Any, **kwargs: Any) -> Any:
        ak = _akshare()
        start_date = _compact_date(start, "19700101", "akshare start")
        end_date = _compact_date(
            end, datetime.now().strftime("%Y%m%d"), "akshare end"
        )
        endpoint = (
            _akshare_etf_endpoint(ak)
            if _akshare_asset_type(self.asset_type, symbol) == "etf"
            else _akshare_hist_endpoint(ak)
        )
        try:
            raw = endpoint(
                symbol=provider_symbol(symbol, "akshare"),
                period=self.period,
                start_date=start_date,
                end_date=end_date,
                adjust=kwargs.pop("adjust", self.adjust),
            )
        except Exception as exc:
            raise DataSourceError(f"akshare failed for {symbol}: {exc}") from exc
        return _canonicalize_ohlcv(
            raw, symbol, "akshare", column_map=self.column_map,
            volume_multiplier=self.volume_multiplier,
        )


class YFinanceLoader(DataLoader):
    """Global daily OHLCV adapter for yfinance."""

    source_name = "yfinance"

    def __init__(
        self,
        *,
        auto_adjust: bool = True,
        volume_multiplier: float = 1.0,
        column_map: dict[str, str] | None = None,
    ):
        self.auto_adjust = auto_adjust
        self.volume_multiplier = float(volume_multiplier)
        self.column_map = dict(column_map or {})

    def _load_symbol(self, symbol: str, start: Any, end: Any, **kwargs: Any) -> Any:
        yf = _yfinance()
        options = dict(kwargs)
        options.setdefault("auto_adjust", self.auto_adjust)
        options.setdefault("actions", False)
        try:
            raw = yf.Ticker(provider_symbol(symbol, "yfinance")).history(
                start=start, end=end, **options
            )
        except Exception as primary:
            # yfinance has two public paths; Ticker.history carries the richer
            # timezone metadata, while download() is a useful fallback when a
            # provider-side endpoint changes.
            download = getattr(yf, "download", None)
            if not callable(download):
                raise DataSourceError(
                    f"yfinance failed for {symbol}: {primary}"
                ) from primary
            try:
                raw = download(
                    provider_symbol(symbol, "yfinance"),
                    start=start,
                    end=end,
                    auto_adjust=options.get("auto_adjust", self.auto_adjust),
                    actions=False,
                    progress=False,
                    threads=False,
                )
            except Exception as secondary:
                raise DataSourceError(
                    f"yfinance failed for {symbol}: {primary}; "
                    f"download fallback: {secondary}"
                ) from primary
            if raw is None or len(raw) == 0:
                raise DataSourceError(
                    f"yfinance failed for {symbol}: {primary}; "
                    "download fallback returned no rows"
                ) from primary
        return _canonicalize_ohlcv(
            raw, symbol, "yfinance", column_map=self.column_map,
            volume_multiplier=self.volume_multiplier,
        )


class DataLoaderFactory:
    """Registry/factory for unified data adapters."""

    _registry: dict[str, type[DataLoader]] = {
        "akshare": AkShareLoader,
        "yfinance": YFinanceLoader,
        "yahoo": YFinanceLoader,
        "csv": LocalCsvLoader,
        "local": LocalCsvLoader,
    }

    @classmethod
    def register(cls, name: str, loader_cls: type[DataLoader]) -> None:
        if not issubclass(loader_cls, DataLoader):
            raise TypeError("loader_cls must inherit DataLoader")
        cls._registry[str(name).strip().lower()] = loader_cls

    @classmethod
    def create(cls, source: str, **kwargs: Any) -> DataLoader:
        name = str(source).strip().lower()
        loader_cls = cls._registry.get(name)
        if loader_cls is None:
            raise ValueError(
                f"unsupported data source {source!r}; "
                f"available: {sorted(cls._registry)}"
            )
        return loader_cls(**kwargs)

    @classmethod
    def available(cls) -> list[str]:
        return sorted(cls._registry)


# Backwards-friendly adapter aliases.
LocalCsvAdapter = LocalCsvLoader
AkShareAdapter = AkShareLoader
YFinanceAdapter = YFinanceLoader


def load_ohlcv(
    source: str,
    symbols: str | Sequence[str],
    *,
    start: Any = None,
    end: Any = None,
    **kwargs: Any,
) -> Any:
    """Load canonical long OHLCV from a named source in one call."""
    return DataLoaderFactory.create(source, **kwargs).load(
        symbols, start=start, end=end
    )


def pivot_ohlcv(frame: Any, field: str = "close") -> Any:
    """Pivot canonical long OHLCV into a wide datetime x symbol panel."""
    validated = validate_ohlcv(frame, source="frame")
    column = str(field).strip().lower()
    if column not in OHLCV_COLUMNS or column == "symbol":
        raise ValueError(f"field must be one of {OHLCV_COLUMNS[1:]}")
    wide = validated.reset_index().pivot(
        index="datetime", columns="symbol", values=column
    )
    wide = wide.sort_index()
    wide.index.name = "datetime"
    return wide


def pivot_ohlcv_panel(frame: Any) -> dict[str, Any]:
    """Return one wide panel for each OHLCV field."""
    validated = validate_ohlcv(frame, source="frame")
    return {
        field: validated.reset_index().pivot(
            index="datetime", columns="symbol", values=field
        )
        for field in ("open", "high", "low", "close", "volume")
    }


__all__ += [
    "AkShareAdapter",
    "AkShareLoader",
    "ContractError",
    "DataLoader",
    "DataLoaderFactory",
    "LocalCsvAdapter",
    "LocalCsvLoader",
    "OHLCV_COLUMNS",
    "YFinanceAdapter",
    "YFinanceLoader",
    "load_ohlcv",
    "pivot_ohlcv",
    "pivot_ohlcv_panel",
    "validate_ohlcv",
]


class SQLiteCache:
    """Small dependency-free OHLCV cache backed by SQLite.

    The table is keyed by ``(datetime, symbol)`` and stores the canonical
    float/int columns.  Writes and reads both go through OHLCV validation.
    """

    def __init__(self, path: str | Path, *, table: str = "ohlcv"):
        import sqlite3

        self.path = Path(path).expanduser()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        table = str(table)
        if (
            not table
            or not (table[0].isalpha() or table[0] == "_")
            or not table.replace("_", "").isalnum()
        ):
            raise ValueError(
                "cache table must start with a letter/underscore and contain "
                "only alphanumerics/underscores"
            )
        self.table = table
        self._sqlite = sqlite3
        self._connection = sqlite3.connect(
            self.path, check_same_thread=False
        )
        self._connection.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {self.table} (
                datetime TEXT NOT NULL,
                symbol TEXT NOT NULL,
                open REAL NOT NULL,
                high REAL NOT NULL,
                low REAL NOT NULL,
                close REAL NOT NULL,
                volume INTEGER NOT NULL,
                PRIMARY KEY (datetime, symbol)
            )
            """
        )
        self._connection.commit()

    def put(self, frame: Any) -> None:
        pd = _pandas()
        validated = validate_ohlcv(frame, source="cache")
        rows = [
            (
                timestamp.isoformat(sep=" "),
                str(symbol),
                float(opening),
                float(high),
                float(low),
                float(close),
                int(volume),
            )
            for timestamp, symbol, opening, high, low, close, volume in zip(
                validated.index,
                validated["symbol"].to_numpy(),
                validated["open"].to_numpy(),
                validated["high"].to_numpy(),
                validated["low"].to_numpy(),
                validated["close"].to_numpy(),
                validated["volume"].to_numpy(),
            )
        ]
        with self._connection:
            self._connection.executemany(
                f"""
                INSERT OR REPLACE INTO {self.table}
                (datetime, symbol, open, high, low, close, volume)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )

    def get(
        self, symbols: str | Sequence[str], start: Any = None, end: Any = None
    ) -> Any:
        pd = _pandas()
        requested = _canonical_symbols(symbols)
        placeholders = ",".join("?" for _ in requested)
        query = (
            f"SELECT datetime, symbol, open, high, low, close, volume "
            f"FROM {self.table} WHERE symbol IN ({placeholders})"
        )
        params: list[Any] = list(requested)
        start_bound = _date_bound(start, pd)
        end_bound = _date_bound(end, pd)
        if start_bound is not None:
            query += " AND datetime >= ?"
            params.append(start_bound.isoformat(sep=" "))
        if end_bound is not None:
            query += " AND datetime <= ?"
            params.append(end_bound.isoformat(sep=" "))
        query += " ORDER BY datetime, symbol"
        raw = pd.read_sql_query(query, self._connection, params=params)
        if raw.empty:
            index = pd.DatetimeIndex([], name="datetime")
            return pd.DataFrame(columns=_contract_columns(), index=index)
        raw["datetime"] = pd.to_datetime(raw["datetime"], errors="raise")
        raw = raw.set_index("datetime")
        return validate_ohlcv(raw, source="cache")

    def clear(self) -> None:
        with self._connection:
            self._connection.execute(f"DELETE FROM {self.table}")

    def close(self) -> None:
        self._connection.close()

    def __enter__(self) -> "SQLiteCache":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()


class CachedLoader(DataLoader):
    """Cache-first loader with optional primary-source degradation.

    The cache is symbol-granular: if a requested symbol has any cached rows it
    is served locally unless ``refresh=True``.  Set ``start``/``end`` when
    warming the cache for a backtest; a production service should add a TTL or
    trading-calendar coverage check on top.
    """

    source_name = "cached"

    def __init__(
        self,
        loader: DataLoader,
        cache: SQLiteCache,
        *,
        refresh: bool = False,
        fallback_to_cache: bool = True,
    ):
        if not isinstance(loader, DataLoader):
            raise TypeError("loader must be a DataLoader")
        if not isinstance(cache, SQLiteCache):
            raise TypeError("cache must be a SQLiteCache")
        self.loader = loader
        self.cache = cache
        self.refresh = bool(refresh)
        self.fallback_to_cache = bool(fallback_to_cache)

    def _load_symbol(self, symbol: str, start: Any, end: Any, **kwargs: Any) -> Any:
        raise NotImplementedError("CachedLoader overrides load()")

    def load(
        self,
        symbols: str | Sequence[str],
        start: Any = None,
        end: Any = None,
        **kwargs: Any,
    ) -> Any:
        pd = _pandas()
        requested = _canonical_symbols(symbols)
        cached = self.cache.get(requested, start=start, end=end)
        if self.refresh:
            try:
                fresh = self.loader.load(
                    requested, start=start, end=end, **kwargs
                )
            except Exception:
                if not cached.empty and self.fallback_to_cache:
                    return validate_ohlcv(cached, source="cached")
                raise
            self.cache.put(fresh)
            return validate_ohlcv(fresh, source="cached")

        have = set(cached["symbol"].tolist())
        missing = [symbol for symbol in requested if symbol not in have]
        frames = [cached] if not cached.empty else []
        if missing:
            try:
                fresh = self.loader.load(
                    missing, start=start, end=end, **kwargs
                )
            except Exception:
                if not cached.empty and self.fallback_to_cache:
                    fresh = None
                else:
                    raise
            if fresh is not None:
                self.cache.put(fresh)
                frames.append(fresh)
        frame = pd.concat(frames, axis=0)
        return validate_ohlcv(frame, source="cached")

    def close(self) -> None:
        self.cache.close()


__all__ += [
    "CachedLoader",
    "SQLiteCache",
]


class RetryingLoader(DataLoader):
    """Retry a unified loader with exponential backoff.

    Provider SDKs often fail transiently.  Retry only at the loader boundary;
    contract/schema errors from :func:`validate_ohlcv` are normally deterministic
    and should not be retried.
    """

    source_name = "retry"

    def __init__(
        self,
        loader: DataLoader,
        *,
        attempts: int = 3,
        backoff: float = 0.5,
        exceptions: tuple[type[BaseException], ...] = (DataSourceError,),
    ):
        if not isinstance(loader, DataLoader):
            raise TypeError("loader must be a DataLoader")
        attempts = int(attempts)
        if attempts < 1:
            raise ValueError("attempts must be at least 1")
        backoff = float(backoff)
        if not np.isfinite(backoff) or backoff < 0.0:
            raise ValueError("backoff must be finite and non-negative")
        if not exceptions:
            raise ValueError("exceptions must not be empty")
        self.loader = loader
        self.attempts = attempts
        self.backoff = backoff
        self.exceptions = exceptions
        self.source_name = f"retry({loader.source_name})"

    def _load_symbol(self, symbol: str, start: Any, end: Any, **kwargs: Any) -> Any:
        raise NotImplementedError("RetryingLoader overrides load()")

    def load(
        self,
        symbols: str | Sequence[str],
        start: Any = None,
        end: Any = None,
        **kwargs: Any,
    ) -> Any:
        import time

        last: BaseException | None = None
        for attempt in range(self.attempts):
            try:
                return self.loader.load(
                    symbols, start=start, end=end, **kwargs
                )
            except self.exceptions as exc:
                if isinstance(exc, ContractError):
                    raise
                last = exc
                if attempt + 1 < self.attempts:
                    time.sleep(self.backoff * (2**attempt))
        raise DataSourceError(
            f"retry exhausted after {self.attempts} attempts: {last}"
        ) from last

    def close(self) -> None:
        close = getattr(self.loader, "close", None)
        if callable(close):
            close()


__all__ += ["RetryingLoader"]
