import importlib.util
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from rflux import data
from rflux.symbols import SymbolError, normalize_symbol, provider_symbol

HAS_PANDAS = importlib.util.find_spec("pandas") is not None


@unittest.skipUnless(HAS_PANDAS, "pandas is not installed")
class TestSymbols(unittest.TestCase):
    def test_canonical_normalization(self):
        self.assertEqual(normalize_symbol("600519"), "600519.SH")
        self.assertEqual(normalize_symbol("600519.SS"), "600519.SH")
        self.assertEqual(normalize_symbol("SH600519"), "600519.SH")
        self.assertEqual(normalize_symbol("000001"), "000001.SZ")
        self.assertEqual(normalize_symbol("430047"), "430047.BJ")
        self.assertEqual(normalize_symbol("AAPL"), "AAPL.US")
        self.assertEqual(normalize_symbol("0700.HK"), "00700.HK")
        self.assertEqual(provider_symbol("600519.SH", "akshare"), "600519")
        self.assertEqual(provider_symbol("AAPL.US", "yfinance"), "AAPL")
        self.assertEqual(provider_symbol("00700.HK", "yfinance"), "0700.HK")
        with self.assertRaises(SymbolError):
            normalize_symbol("")
        with self.assertRaises(SymbolError):
            normalize_symbol("AAPL.LSE")


@unittest.skipUnless(HAS_PANDAS, "pandas is not installed")
class TestUnifiedDataLayer(unittest.TestCase):
    def _write_local(self, root: Path):
        (root / "600519.SH.csv").write_text(
            "date,open,high,low,close,volume\n"
            "2024-01-02,10,11,9,10.5,1000\n"
            "2024-01-03,10.5,12,10,11.5,1200\n",
            encoding="utf-8",
        )
        (root / "AAPL.csv").write_text(
            "Date,Open,High,Low,Close,Volume\n"
            "2024-01-02,100,101,99,100.5,100000\n"
            "2024-01-03,101,102,100,101.5,110000\n",
            encoding="utf-8",
        )

    def test_local_csv_contract_and_pivot(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_local(root)
            frame = data.LocalCsvLoader(root).load(
                ["600519.SH", "AAPL.US"], start="2024-01-02", end="2024-01-03"
            )
        self.assertEqual(list(frame.columns), list(data.OHLCV_COLUMNS))
        self.assertEqual(frame.index.name, "datetime")
        self.assertIsNone(frame.index.tz)
        self.assertEqual(str(frame["open"].dtype), "float32")
        self.assertEqual(str(frame["volume"].dtype), "int64")
        self.assertEqual(frame.shape, (4, 6))
        self.assertTrue(frame.index.is_monotonic_increasing)
        pairs = list(zip(frame.index, frame["symbol"]))
        self.assertEqual(len(pairs), len(set(pairs)))
        wide = data.pivot_ohlcv(frame, "close")
        self.assertEqual(list(wide.columns), ["600519.SH", "AAPL.US"])
        self.assertAlmostEqual(float(wide.loc["2024-01-03", "AAPL.US"]), 101.5)
        panels = data.pivot_ohlcv_panel(frame)
        self.assertEqual(set(panels), {"open", "high", "low", "close", "volume"})

    def test_long_csv_with_symbol_column(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "panel.csv"
            path.write_text(
                "date,symbol,open,high,low,close,volume\n"
                "2024-01-02,600519.SH,10,11,9,10.5,1000\n"
                "2024-01-02,AAPL.US,100,101,99,100.5,100000\n",
                encoding="utf-8",
            )
            frame = data.LocalCsvLoader(path=path).load(
                ["600519.SH", "AAPL.US"]
            )
        self.assertEqual(frame.shape, (2, 6))

    def test_contract_rejects_invalid_bars(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "600519.SH.csv"
            path.write_text(
                "date,open,high,low,close,volume\n"
                "2024-01-02,10,9,8,10,1000\n",
                encoding="utf-8",
            )
            with self.assertRaises(data.ContractError):
                data.LocalCsvLoader(root).load(["600519.SH"])
            path.write_text(
                "date,open,high,low,close,volume\n"
                "2024-01-02,10,11,9,10,-1\n",
                encoding="utf-8",
            )
            with self.assertRaises(data.ContractError):
                data.LocalCsvLoader(root).load(["600519.SH"])

    def test_factory_and_aliases(self):
        self.assertEqual(
            data.DataLoaderFactory.create("csv", root=".").source_name, "csv"
        )
        with self.assertRaises(ValueError):
            data.DataLoaderFactory.create("unknown")
        self.assertIs(data.LocalCsvAdapter, data.LocalCsvLoader)
        self.assertIn("akshare", data.DataLoaderFactory.available())

    def test_akshare_adapter(self):
        import pandas as pd

        raw = pd.DataFrame(
            {
                "日期": ["2024-01-02", "2024-01-03"],
                "开盘": [10.0, 11.0],
                "最高": [11.0, 12.0],
                "最低": [9.0, 10.0],
                "收盘": [10.5, 11.5],
                "成交量": [10.0, 12.0],
            }
        )
        fake = types.ModuleType("akshare")
        fake.stock_zh_a_hist = lambda **kwargs: raw.copy()
        with mock.patch.dict(sys.modules, {"akshare": fake}):
            frame = data.AkShareLoader().load(
                ["600519.SH"], start="2024-01-01", end="2024-01-31"
            )
        self.assertEqual(frame["symbol"].iloc[0], "600519.SH")
        self.assertEqual(int(frame["volume"].iloc[0]), 1000)
        self.assertTrue(str(frame.index.dtype).startswith("datetime64"))

    def test_akshare_etf_routing(self):
        import pandas as pd

        raw = pd.DataFrame(
            {
                "日期": ["2024-01-02", "2024-01-03"],
                "开盘": [3.0, 3.1],
                "最高": [3.2, 3.3],
                "最低": [2.9, 3.0],
                "收盘": [3.1, 3.2],
                "成交量": [100.0, 120.0],
            }
        )
        calls = {"stock": 0, "etf": 0}
        fake = types.ModuleType("akshare")
        fake.stock_zh_a_hist_em = lambda **kwargs: calls.__setitem__(
            "stock", calls["stock"] + 1
        ) or raw.copy()
        fake.fund_etf_hist_em = lambda **kwargs: calls.__setitem__(
            "etf", calls["etf"] + 1
        ) or raw.copy()
        with mock.patch.dict(sys.modules, {"akshare": fake}):
            frame = data.AkShareLoader().load(
                ["510300.SH"], start="2024-01-01", end="2024-01-31"
            )
        self.assertEqual(frame["symbol"].iloc[0], "510300.SH")
        self.assertEqual(calls, {"stock": 0, "etf": 1})
        with mock.patch.dict(sys.modules, {"akshare": fake}):
            frame = data.AkShareLoader(asset_type="stock").load(
                ["600519.SH"], start="2024-01-01", end="2024-01-31"
            )
        self.assertEqual(frame["symbol"].iloc[0], "600519.SH")
        self.assertEqual(calls, {"stock": 1, "etf": 1})

    def _load_with_fake_yfinance(self, ticker, download=None):
        fake = types.ModuleType("yfinance")
        fake.Ticker = lambda _: ticker
        if download is not None:
            fake.download = download
        with mock.patch.dict(sys.modules, {"yfinance": fake}):
            return data.YFinanceLoader().load(["AAPL.US"])

    def _assert_apple_frame(self, frame):
        self.assertEqual(frame["symbol"].iloc[0], "AAPL.US")
        self.assertAlmostEqual(float(frame["close"].iloc[0]), 100.5)

    def test_yfinance_adapter(self):
        import pandas as pd

        class Ticker:
            def history(self, **kwargs):
                return pd.DataFrame(
                    {
                        "Open": [100.0, 101.0],
                        "High": [101.0, 102.0],
                        "Low": [99.0, 100.0],
                        "Close": [100.5, 101.5],
                        "Volume": [100000, 110000],
                    },
                    index=pd.DatetimeIndex(
                        ["2024-01-02", "2024-01-03"], name="Date"
                    ),
                )

        self._assert_apple_frame(self._load_with_fake_yfinance(Ticker()))

    def test_retrying_loader(self):
        import pandas as pd

        class Flaky(data.DataLoader):
            source_name = "flaky"

            def __init__(self):
                self.calls = 0

            def _load_symbol(self, symbol, start, end, **kwargs):
                self.calls += 1
                if self.calls < 2:
                    raise data.DataSourceError("transient")
                index = pd.DatetimeIndex(["2024-01-02"], name="datetime")
                return pd.DataFrame(
                    {
                        "symbol": [symbol],
                        "open": [10.0],
                        "high": [11.0],
                        "low": [9.0],
                        "close": [10.5],
                        "volume": [1000],
                    },
                    index=index,
                )

        flaky = Flaky()
        loader = data.RetryingLoader(flaky, attempts=2, backoff=0.0)
        frame = loader.load(["600519.SH"])
        self.assertEqual(flaky.calls, 2)
        self.assertEqual(frame["symbol"].iloc[0], "600519.SH")

    def test_yfinance_download_fallback_multindex(self):
        import pandas as pd

        class FailingTicker:
            def history(self, **kwargs):
                raise RuntimeError("endpoint unavailable")

        raw = pd.DataFrame(
            {
                ("Open", "AAPL"): [100.0],
                ("High", "AAPL"): [101.0],
                ("Low", "AAPL"): [99.0],
                ("Close", "AAPL"): [100.5],
                ("Volume", "AAPL"): [1000],
            },
            index=pd.DatetimeIndex(["2024-01-02"], name="Date"),
        )
        frame = self._load_with_fake_yfinance(
            FailingTicker(), lambda *args, **kwargs: raw.copy())
        self._assert_apple_frame(frame)

    def test_sqlite_cache_and_cached_loader(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_local(root)
            cache = data.SQLiteCache(root / "prices.sqlite")
            cached = data.CachedLoader(
                data.LocalCsvLoader(root), cache
            )
            first = cached.load(["600519.SH"])
            self.assertFalse(first.empty)
            (root / "600519.SH.csv").unlink()
            second = cached.load(["600519.SH"])
            cached.close()
        self.assertEqual(first.shape, second.shape)
        np.testing.assert_allclose(
            first["close"].to_numpy(), second["close"].to_numpy()
        )


if __name__ == "__main__":
    unittest.main()
