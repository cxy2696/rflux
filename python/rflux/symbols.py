"""Canonical instrument symbols and provider-specific translation.

Internal rflux symbols use an exchange suffix:

``600519.SH``, ``000001.SZ``, ``430047.BJ``, ``AAPL.US``, ``00700.HK``

Provider adapters translate at their boundary.  Strategies therefore never
contain a provider-specific ticker format.
"""

from __future__ import annotations

import re

__all__ = [
    "SymbolError",
    "is_a_share",
    "normalize_symbol",
    "provider_symbol",
]

_SUFFIXES = {"SH", "SZ", "BJ", "US", "HK"}
_ALIASES = {"SS": "SH", "SHA": "SH", "SHE": "SZ", "XSHE": "SZ"}
_SYMBOL_RE = re.compile(r"^[A-Z0-9][A-Z0-9._\-^=]{0,31}$")


class SymbolError(ValueError):
    """An instrument symbol cannot be normalized."""


def _split_suffix(text: str) -> tuple[str, str | None]:
    if "." not in text:
        return text, None
    base, suffix = text.rsplit(".", 1)
    return base, suffix


def normalize_symbol(value: str, *, default_market: str | None = None) -> str:
    """Return the canonical rflux symbol.

    Bare six-digit A-share codes are inferred from their prefix.  Bare US
    tickers receive ``.US``; a bare five-digit code is treated as Hong Kong.
    ``SS`` is accepted as an alias for Shanghai.
    """
    text = str(value).strip().upper().replace(" ", "")
    if not text:
        raise SymbolError("symbol must not be empty")

    if len(text) > 3 and text[:2] in {"SH", "SZ", "BJ", "HK"} and text[2:].isdigit():
        prefix, number = text[:2], text[2:]
        if prefix == "HK":
            number = number.zfill(5)
        return f"{number}.{prefix}"

    base, suffix = _split_suffix(text)
    if suffix is not None:
        canonical_suffix = _ALIASES.get(suffix, suffix)
        if canonical_suffix not in _SUFFIXES:
            raise SymbolError(f"unsupported exchange suffix {suffix!r}")
        if not base:
            raise SymbolError("symbol has an empty base")
        if canonical_suffix in {"SH", "SZ", "BJ"} and not base.isdigit():
            raise SymbolError(f"A-share symbol must be numeric: {value!r}")
        if canonical_suffix == "HK":
            base = base.zfill(5)
        return f"{base}.{canonical_suffix}"

    if text.isdigit():
        if len(text) == 6:
            if text[0] in {"6", "9"}:
                market = "SH"
            elif text[0] in {"0", "3"}:
                market = "SZ"
            elif text[0] in {"4", "8"}:
                market = "BJ"
            else:
                market = default_market
            if market is None:
                raise SymbolError(
                    f"cannot infer the exchange for A-share code {value!r}"
                )
            return f"{text}.{market}"
        if len(text) == 5:
            return f"{text}.HK"
        if default_market is not None:
            return f"{text.zfill(5)}.{default_market}" if default_market == "HK" else f"{text}.{default_market}"
        raise SymbolError(f"cannot infer a market for numeric symbol {value!r}")

    if default_market is not None:
        return f"{text}.{default_market}"
    if not _SYMBOL_RE.match(text):
        raise SymbolError(f"invalid symbol {value!r}")
    return f"{text}.US"


def provider_symbol(symbol: str, provider: str) -> str:
    """Translate a canonical symbol into the provider's expected format."""
    canonical = normalize_symbol(symbol)
    base, suffix = canonical.rsplit(".", 1)
    name = str(provider).strip().lower()
    if name in {"akshare", "ak"}:
        if suffix not in {"SH", "SZ", "BJ"}:
            raise SymbolError(f"akshare adapter only supports A-share symbols: {symbol!r}")
        return base
    if name in {"yfinance", "yahoo", "yf"}:
        if suffix == "US":
            return base
        if suffix == "HK":
            return f"{int(base):04d}.HK" if base.isdigit() else canonical
        return canonical
    if name in {"csv", "local"}:
        return canonical
    return canonical


def is_a_share(symbol: str) -> bool:
    try:
        canonical = normalize_symbol(symbol)
    except SymbolError:
        return False
    return canonical.endswith((".SH", ".SZ", ".BJ"))
