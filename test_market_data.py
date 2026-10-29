"""Tests for the background data store, with fake downloads. Run: python test_market_data.py"""

import threading
import time

import pandas as pd

from market_data import PERIOD, RECENT, LiveData


def bars(*minutes, value=1.0):
    idx = pd.DatetimeIndex([pd.Timestamp("2026-10-29 09:15", tz="Asia/Kolkata")
                            + pd.Timedelta(minutes=m) for m in minutes])
    return pd.Series(value, index=idx, dtype=float)


def test_first_load_waits_then_refreshes_in_background():
    calls, gate = [], threading.Event()

    def fetch(tickers, period):
        calls.append(period)
        if len(calls) > 1:
            gate.wait(5)  # a slow refresh must not hold up the page
            return {"A": bars(2, 3, value=2.0), "B": pd.Series(dtype=float)}
        return {"A": bars(0, 1, 2), "B": bars(0)}

    store = LiveData(["A", "B"], max_age=0, fetch=fetch,
                     fetch_daily=lambda t: {k: pd.Series([1.0]) for k in t})
    closes, daily = store.get()
    assert calls == [PERIOD] and len(closes["A"]) == 3 and set(daily) == {"A", "B"}
    time.sleep(0.01)
    started = time.time()
    closes, _ = store.get()  # stale: kicks off a refresh but returns the old copy at once
    assert time.time() - started < 0.5 and len(closes["A"]) == 3
    gate.set()
    for _ in range(100):
        if store.updated and len(store.closes["A"]) == 4:
            break
        time.sleep(0.02)
    assert calls[1] == RECENT
    a = store.closes["A"]
    assert len(a) == 4 and a.iloc[2] == 2.0 and a.iloc[0] == 1.0  # merged, newest bar wins
    assert len(store.closes["B"]) == 1  # an empty download keeps what we had


def test_full_reload_after_history_age():
    periods = []
    store = LiveData(["A"], max_age=0, history_age=0,
                     fetch=lambda t, p: periods.append(p) or {"A": bars(0)},
                     fetch_daily=lambda t: {"A": pd.Series([1.0])})
    store.refresh()
    time.sleep(0.01)
    store.refresh()
    assert periods == [PERIOD, PERIOD]


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
    print("OK")
