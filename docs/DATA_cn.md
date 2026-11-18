# 统一数据层

[English](DATA_en.md) · **中文**

0.7 的数据层把三种异构数据源统一为一个长表 OHLCV 契约。使用前安装
`pandas` 和对应 provider SDK：`pip install "rflux[all]"`，或 `rflux[akshare]`
等窄依赖；只要 pandas 可安装 `rflux[data]`。策略代码只依赖
`rflux.data`，不直接依赖 provider SDK。

## 契约

统一 loader 返回如下 pandas DataFrame：

| 字段 | dtype | 说明 |
| --- | --- | --- |
| `symbol` | object/str | rflux 标准标的代码 |
| `open` | float32 | 开盘价 |
| `high` | float32 | 最高价 |
| `low` | float32 | 最低价 |
| `close` | float32 | 收盘价 |
| `volume` | int64 | 成交股数 |

索引为 tz-naive 的 `DatetimeIndex`，名称 `datetime`，表示交易所本地交易日期/时间。
日线不使用 UTC 时间戳，避免把纽约 00:00 和上海 00:00 混在一起。行在
`(datetime, symbol)` 上唯一，并按 datetime、symbol 排序。

校验会拒绝：缺列、非有限或非正价格、负数/小数成交量、重复 pair，以及
`high` 低于 `open/close/low`、`low` 高于 `open/close/high` 的 bar。A 股 provider
的成交量默认从手转换为股（乘 100）。

## 标的代码

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

## 工厂

```python
from rflux import data

loader = data.DataLoaderFactory.create("akshare", adjust="qfq")
frame = loader.load(["600519.SH", "000001.SZ"], start="2024-01-01", end="2024-12-31")

loader = data.DataLoaderFactory.create("yfinance")
frame = loader.load(["AAPL.US", "MSFT.US"], start="2024-01-01", end="2024-12-31")

loader = data.DataLoaderFactory.create("csv", root="./data/daily")
frame = loader.load(["600519.SH", "AAPL.US"], start="2024-01-01", end="2024-12-31")
```

同时提供 `LocalCsvAdapter`、`AkShareAdapter`、`YFinanceAdapter`、
 别名，方便按 Adapter Pattern 命名。通过
`DataLoaderFactory.register("ccxt", MyLoader)` 可以增加新数据源，策略代码不变。

## 本地 CSV

目录模式下每个标的一个文件，文件名可以是 `600519.SH.csv`、`600519.csv`、
`AAPL.US.csv`、`AAPL.csv`。列名大小写不敏感，并识别常见别名（`date`、
`trade_date`、`vol`、中文 provider 字段等）。也可以使用一个带 `symbol` 列的
长表 CSV：

```python
loader = data.LocalCsvLoader(path="panel.csv")
frame = loader.load(["600519.SH", "AAPL.US"])
```

## 转成宽表

统一契约是长表；只在算法边界转成 datetime x asset 宽表：

```python
close = data.pivot_ohlcv(frame, "close")
panels = data.pivot_ohlcv_panel(frame)  # open/high/low/close/volume
returns = close.pct_change()
```

旧接口 `data.load_csv`、`load_akshare`、`load_yfinance` 继续返回
宽表收盘价，供现有 `analyze`/`risk` CLI 和示例使用。

## 缓存

`SQLiteCache` 不依赖额外包，按统一契约存储：

```python
cache = data.SQLiteCache("~/.rflux/prices.sqlite")
loader = data.CachedLoader(
    data.DataLoaderFactory.create("akshare"),
    cache,
)
frame = loader.load(["600519.SH"], start="2024-01-01", end="2024-12-31")
loader.close()
```

缓存以 symbol 为粒度：已有 symbol 本地返回；缺失 symbol 拉取后 upsert。
`refresh=True` 强制刷新；`fallback_to_cache=True` 时缺失 symbol 的网络失败会降级到
已有缓存。生产服务应在此基础上增加 TTL 或交易日覆盖检查。

## Provider 说明

- `AkShareLoader` 支持 A 股股票和 ETF。默认 `asset_type="auto"` 对 ETF 代码使用 `fund_etf_hist_em`，其余使用 `stock_zh_a_hist_em`；可用 `asset_type="etf"|"stock"` 强制指定。支持 `adjust="qfq"|"hfq"|""`。
- `YFinanceLoader` 使用 `Ticker.history`，默认 `auto_adjust=True`。
- `LocalCsvLoader` 支持 per-symbol 目录或单个长表。

`RetryingLoader` 在 loader 边界提供有上限的指数退避重试；契约/字段错误不会重试。
限频和并发拉取仍属于 provider 特定逻辑，应通过包装器或 provider 专用 adapter
实现。工厂注册表就是扩展点。
