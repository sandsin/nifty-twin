"""Tests for similarity.py on synthetic data. Run: python test_similarity.py"""

from datetime import date

import numpy as np
import pandas as pd

from similarity import (IST, OWN_OPEN, SAME_CLOCK, align, interval_returns, rank,
                        session_path)

DAY = date(2026, 10, 23)
ELAPSED = 165  # 9:15 -> 12:00


def ts(hhmm: str) -> pd.Timestamp:
    return pd.Timestamp(f"{DAY} {hhmm}", tz=IST)


def minutes(start: str, end: str) -> pd.DatetimeIndex:
    return pd.date_range(ts(start), ts(end), freq="min")


def walk(idx, seed, base=100.0, drift=0.0):
    rng = np.random.default_rng(seed)
    steps = rng.normal(drift, 0.05, len(idx))
    return pd.Series(base * np.exp(np.cumsum(steps) / 100), index=idx)


def drop_lunch(s, start, end):
    return s[(s.index < ts(start)) | (s.index >= ts(end))]


def test_own_open_anchors_at_first_bar():
    s = pd.Series(np.linspace(100, 110, 301), index=minutes("07:00", "12:00"))
    p = session_path(s, DAY, ELAPSED, OWN_OPEN)
    assert p.anchor == ts("07:00")
    assert p.values.index[0] == 0 and abs(p.values.iloc[0]) < 1e-9
    assert p.values.index[-1] == ELAPSED
    assert p.clock()[-1] == ts("09:45")


def test_same_clock_uses_price_at_915():
    s = drop_lunch(pd.Series(np.linspace(100, 110, 301), index=minutes("07:00", "12:00")), "09:00", "10:30")
    p = session_path(s, DAY, ELAPSED, SAME_CLOCK)
    assert p.anchor == ts("09:15")
    base = s[s.index <= ts("09:15")].iloc[-1]
    assert p.values.index[0] == 75  # first bar after lunch, 10:30
    assert abs(p.values.iloc[0] - (s[ts("10:30")] / base - 1) * 100) < 1e-9


def test_align_no_forward_fill():
    nifty = session_path(walk(minutes("09:15", "12:00"), 1), DAY, ELAPSED).values
    other = drop_lunch(walk(minutes("07:00", "12:00"), 2), "09:30", "10:30")
    a = align(nifty, session_path(other, DAY, ELAPSED).values)
    assert not a.index.isin(range(151, 210)).any()  # Hang Seng-style lunch at +2:30..+3:30
    assert not a.isna().any().any()


def test_returns_skip_gaps():
    idx = pd.Index(range(0, 26)).delete(slice(10, 13))
    df = pd.DataFrame({"nifty": np.arange(len(idx), dtype=float), "other": np.arange(len(idx), dtype=float)}, index=idx)
    r = interval_returns(df, 5)
    for m in r.index:
        assert m - 5 in df.index


def test_own_open_matches_shifted_twin():
    # An index that opens 2h15m earlier and traces Nifty's shape from its own open.
    nifty = walk(minutes("09:15", "12:00"), 1, drift=0.01)
    twin = pd.Series(nifty.values * 1.5, index=nifty.index - pd.Timedelta(minutes=135))
    noise = walk(minutes("07:00", "12:00"), 3)
    others = {"Twin": twin, "Noise": noise}
    own = {r.name: r for r in rank(nifty, others, DAY, ELAPSED, OWN_OPEN, tickers={k: k for k in others})}
    assert own["Twin"].status == "ranked" and own["Twin"].score > 80
    assert own["Twin"].score > own["Noise"].score
    clock = {r.name: r for r in rank(nifty, others, DAY, ELAPSED, SAME_CLOCK, tickers={k: k for k in others})}
    assert clock["Twin"].score < own["Twin"].score


def test_ranking_order_and_partial():
    idx = minutes("09:15", "12:00")
    nifty = walk(idx, 1, drift=0.01)
    twin = nifty * 1.5 + np.random.default_rng(9).normal(0, 0.01, len(idx))
    noise = walk(idx, 3)
    inverse = 200 - nifty
    short = walk(minutes("09:15", "10:15"), 4)
    others = {"Twin": twin, "Noise": noise, "Inverse": inverse, "Short": short, "Empty": pd.Series(dtype=float)}
    res = rank(nifty, others, DAY, ELAPSED, interval=5, min_coverage=0.7, tickers={k: k for k in others})
    by = {r.name: r for r in res}
    assert res[0].name == "Twin", [r.name for r in res]
    assert by["Twin"].score > 80
    assert by["Twin"].score > by["Noise"].score > by["Inverse"].score
    assert by["Short"].status == "partial"
    assert by["Empty"].status == "no data"
    assert all(0 <= r.score <= 100 for r in res if r.status != "no data")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
    print("OK")
