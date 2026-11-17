"""Opt-in live provider checks.

Set RFLUX_LIVE_TESTS=1 to run.  These tests access external data sources and
are not part of the default CI suite.
"""

import os
import unittest

from rflux import data

LIVE = os.environ.get("RFLUX_LIVE_TESTS") == "1"


@unittest.skipUnless(LIVE, "set RFLUX_LIVE_TESTS=1 to run live provider tests")
class TestLiveProviders(unittest.TestCase):
    def test_akshare_contract(self):
        frame = data.AkShareLoader().load(
            ["600519.SH"], start="2025-01-02", end="2025-01-10"
        )
        self.assertFalse(frame.empty)
        self.assertEqual(frame["symbol"].iloc[0], "600519.SH")
        self.assertEqual(str(frame["volume"].dtype), "int64")
        self.assertTrue((frame["high"] >= frame["close"]).all())

    def test_yfinance_contract(self):
        try:
            frame = data.YFinanceLoader().load(
                ["AAPL.US"], start="2025-01-02", end="2025-01-10"
            )
        except (data.DataSourceError, data.ContractError) as exc:
            message = str(exc).lower()
            if (
                "rate limit" in message
                or "too many requests" in message
                or "returned no rows" in message
            ):
                self.skipTest(f"yfinance unavailable from this network: {exc}")
            raise
        self.assertFalse(frame.empty)
        self.assertEqual(frame["symbol"].iloc[0], "AAPL.US")
        self.assertEqual(str(frame["open"].dtype), "float32")



if __name__ == "__main__":
    unittest.main()
