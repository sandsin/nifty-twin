"""1-minute close loader (Yahoo Finance via yfinance), returned in IST."""

from __future__ import annotations

import threading
import time

import pandas as pd
import yfinance as yf

from similarity import IST

PERIOD = "7d"  # Yahoo serves 1-minute bars for roughly the last 7 trading days.
RECENT = "1d"  # a refresh only needs each ticker's latest session


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


def fetch_daily_closes(tickers: list[str], period: str = "15d") -> dict[str, pd.Series]:
    """Map of ticker -> official daily closes indexed by exchange date.

    The last 1-minute bar can miss the closing auction (KOSPI, Nikkei), so previous closes
    come from daily bars.
    """
    out = {t: pd.Series(dtype=float) for t in tickers}
    if not tickers:
        return out
    try:
        raw = yf.download(tickers, period=period, interval="1d", group_by="ticker",
                          auto_adjust=False, progress=False, threads=False)
    except Exception:
        return out
    if raw is None or raw.empty:
        return out
    for t in tickers:
        try:
            close = raw[t]["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw["Close"]
        except KeyError:
            continue
        close = close.dropna().astype(float)
        close.index = [ts.date() for ts in close.index]
        out[t] = close
    return out


def prev_close(daily: pd.Series, day) -> float | None:
    """Official close of the last session before `day`."""
    before = daily[[d < day for d in daily.index]] if not daily.empty else daily
    return float(before.iloc[-1]) if not before.empty else None


def trading_days(series: pd.Series) -> list:
    """Distinct IST dates with bars, newest first."""
    if series.empty:
        return []
    return sorted({ts.date() for ts in series.index}, reverse=True)


def _merge(old: pd.Series | None, new: pd.Series) -> pd.Series:
    """`old` with `new` bars added or replaced. An empty download keeps what we had."""
    if old is None or old.empty:
        return new
    if new.empty:
        return old
    both = pd.concat([old, new])
    return both[~both.index.duplicated(keep="last")].sort_index()


class LiveData:
    """1-minute and daily closes kept in memory and refreshed in the background.

    A page run reads the latest copy and never waits on Yahoo, except for the very first
    load. When the copy is older than `max_age` seconds, a background thread fetches just
    the latest session (`RECENT`) and merges it in; the full 7 days and the daily closes are
    re-downloaded only every `history_age` seconds.
    """

    def __init__(self, tickers: list[str], max_age: float = 20, history_age: float = 900,
                 fetch=None, fetch_daily=None):
        self.tickers = list(tickers)
        self.max_age, self.history_age = max_age, history_age
        self._fetch = fetch or fetch_closes
        self._fetch_daily = fetch_daily or fetch_daily_closes
        self._lock = threading.Lock()        # guards the fields below
        self._fetching = threading.Lock()    # one download at a time
        self.closes: dict[str, pd.Series] | None = None
        self.daily: dict[str, pd.Series] = {}
        self.updated = 0.0                   # time.time() of the last refresh
        self._history_at = 0.0
        self._busy = False

    def get(self) -> tuple[dict[str, pd.Series], dict[str, pd.Series]]:
        """(1-minute closes, daily closes) for every ticker."""
        with self._lock:
            ready = self.closes is not None
            if ready and time.time() - self.updated > self.max_age and not self._busy:
                self._busy = True
                threading.Thread(target=self.refresh, daemon=True).start()
        if not ready:
            self.refresh()  # nothing to show yet, so this first load has to wait
        with self._lock:
            return self.closes, self.daily

    def refresh(self) -> None:
        with self._fetching:
            try:
                if self.closes is not None and time.time() - self.updated <= self.max_age:
                    return  # another caller refreshed while we waited
                full = time.time() - self._history_at > self.history_age
                new = self._fetch(self.tickers, PERIOD if full else RECENT)
                daily = self._fetch_daily(self.tickers) if full else None
                old = self.closes or {}
                closes = ({t: new[t] if not new[t].empty else old.get(t, new[t]) for t in self.tickers}
                          if full else {t: _merge(old.get(t), new[t]) for t in self.tickers})
                with self._lock:
                    self.closes = closes
                    if daily is not None:
                        self.daily = {t: daily[t] if not daily[t].empty else self.daily.get(t, daily[t])
                                      for t in self.tickers}
                        if any(not s.empty for s in new.values()):
                            self._history_at = time.time()  # else retry the full load next time
                    self.updated = time.time()
            finally:
                with self._lock:
                    self._busy = False
