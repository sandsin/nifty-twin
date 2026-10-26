"""1-minute close loader (Yahoo Finance via yfinance), returned in IST."""

from __future__ import annotations

import pandas as pd
import yfinance as yf

from similarity import IST

PERIOD = "7d"  # Yahoo serves 1-minute bars for roughly the last 7 trading days.


def _to_ist(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    if index.tz is None:
        index = index.tz_localize("UTC")
    return index.tz_convert(IST).floor("min")


def _clean(close: pd.Series) -> pd.Series:
    close = close.dropna()
    if close.empty:
        return close.astype(float)
    close.index = _to_ist(close.index)
    close = close[~close.index.duplicated(keep="last")].sort_index()
    return close.astype(float)


def _download(tickers: list[str], period: str, threads: bool) -> dict[str, pd.Series]:
    out = {t: pd.Series(dtype=float) for t in tickers}
    try:
        raw = yf.download(
            tickers, period=period, interval="1m", group_by="ticker",
            auto_adjust=False, progress=False, threads=threads,
        )
    except Exception:
        return out
    if raw is None or raw.empty:
        return out
    for t in tickers:
        try:
            if isinstance(raw.columns, pd.MultiIndex):
                if t not in raw.columns.get_level_values(0):
                    continue
                close = raw[t]["Close"]
            else:
                close = raw["Close"]
            out[t] = _clean(close)
        except KeyError:
            continue
    return out


def fetch_closes(tickers: list[str], period: str = PERIOD) -> dict[str, pd.Series]:
    """Map of ticker -> 1-minute close series in IST. Missing tickers map to an empty series."""
    if not tickers:
        return {}
    out = _download(tickers, period, threads=True)
    # Threaded downloads can trip over yfinance's sqlite cache; retry the misses one by one.
    for t in [t for t, s in out.items() if s.empty]:
        out.update(_download([t], period, threads=False))
    return out


def trading_days(series: pd.Series) -> list:
    """Distinct IST dates with bars, newest first."""
    if series.empty:
        return []
    return sorted({ts.date() for ts in series.index}, reverse=True)
