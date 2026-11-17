# Unified data layer

**English** · [中文](DATA_cn.md)

The 0.7 data layer turns three heterogeneous providers into one long OHLCV
contract.  Install pandas and the provider SDKs with `pip install
"rflux[all]"`, or a narrower extra such as `rflux[akshare]`; pandas alone is
available through `rflux[data]`.  Strategies import `rflux.data`, not provider SDKs.

## Contract

Every unified loader returns a pandas DataFrame with:

| column | dtype | meaning |
| --- | --- | --- |
| `symbol` | object/str | canonical rflux symbol |
| `open` | float32 | open |
| `high` | float32 | high |
| `low` | float32 | low |
| `close` | float32 | close |
| `volume` | int64 | traded shares |

The index is a tz-naive `DatetimeIndex` named `datetime`, representing the
exchange-local trading date/time.  This avoids mixing
`2024-01-02 00:00 America/New_York` with `2024-01-02 00:00 Asia/Shanghai` for
daily bars.  Rows are unique on `(datetime, symbol)` and sorted by datetime then
symbol.

Validation rejects missing columns, non-finite prices, non-positive prices,
negative/ fractional volume, duplicate pairs and bars where `high` is below
`open/close/low` or `low` is above `open/close/high`.  A-share provider volume
is converted from lots to shares by multiplying by 100 by default.

## Symbols

```python
from rflux.symbols import normalize_symbol, provider_symbol

normalize_symbol("600519")       # 600519.SH
normalize_symbol("600519.SS")    # 600519.SH
normalize_symbol("SH600519")     # 600519.SH
normalize_symbol("000001")       # 000001.SZ
normalize_symbol("AAPL")         # AAPL.US
normalize_symbol("0700.HK")      # 00700.HK
provider_symbol("600519.SH", "akshare")  # 600519
provider_symbol("AAPL.US", "yfinance")   # AAPL
```

## Factory

```python
from rflux import data

loader = data.DataLoaderFactory.create("akshare", adjust="qfq")
frame = loader.load(["600519.SH", "000001.SZ"], start="2024-01-01", end="2024-12-31")

# Same strategy code, different source:
loader = data.DataLoaderFactory.create("yfinance")
frame = loader.load(["AAPL.US", "MSFT.US"], start="2024-01-01", end="2024-12-31")

loader = data.DataLoaderFactory.create("csv", root="./data/daily")
frame = loader.load(["600519.SH", "AAPL.US"], start="2024-01-01", end="2024-12-31")
```

The adapter aliases `LocalCsvAdapter`, `AkShareAdapter`, `YFinanceAdapter` and
 are provided for code that follows the Adapter Pattern naming.
`DataLoaderFactory.register("ccxt", MyLoader)` adds a new source without touching
strategy code.

## Local CSV

Per-symbol files under `root` may be named `600519.SH.csv`, `600519.csv`,
`AAPL.US.csv` or `AAPL.csv`.  Column names are case-insensitive and common
aliases (`date`, `trade_date`, `vol`, Chinese provider names) are recognized.
A single long CSV with a `symbol` column is also supported:

```python
loader = data.LocalCsvLoader(path="panel.csv")
frame = loader.load(["600519.SH", "AAPL.US"])
```

## Wide panels for rflux algorithms

The canonical frame is long.  Pivot only at the boundary where an algorithm
needs a datetime x asset panel:

```python
close = data.pivot_ohlcv(frame, "close")
panels = data.pivot_ohlcv_panel(frame)  # open/high/low/close/volume
returns = close.pct_change()
```

The legacy `data.load_csv`, `load_akshare` and `load_yfinance` helpers continue to return wide close panels for the existing `analyze`/`risk`
CLI and examples.

## Cache

`SQLiteCache` is dependency-free and stores the validated long contract:

```python
cache = data.SQLiteCache("~/.rflux/prices.sqlite")
loader = data.CachedLoader(
    data.DataLoaderFactory.create("akshare"),
    cache,
)
frame = loader.load(["600519.SH"], start="2024-01-01", end="2024-12-31")
loader.close()
```

The cache is symbol-granular: cached symbols are served locally; missing symbols
are fetched and upserted.  `refresh=True` forces a re-fetch.  With
`fallback_to_cache=True`, a network failure for missing symbols degrades to the
rows already cached instead of failing the whole request.  A production service
should add a TTL/trading-calendar coverage policy on top.

## Provider notes

- `AkShareLoader` supports A-share stocks and ETFs. `asset_type="auto"` (default) uses `fund_etf_hist_em` for ETF codes and `stock_zh_a_hist_em` otherwise; `asset_type="etf"|"stock"` overrides the inference. `adjust="qfq"|"hfq"|""` is supported.
- `YFinanceLoader` uses `Ticker.history`; `auto_adjust=True` is the default.
- `LocalCsvLoader` supports a directory of per-symbol files or one long panel.

`RetryingLoader` adds bounded exponential-backoff retries at the loader
boundary; contract/schema errors are never retried.  Rate limiting and
concurrent fetch workers remain provider-specific and should be added as
wrappers or provider adapters.  The factory registry is the extension point.
