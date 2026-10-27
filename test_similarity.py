"""Tests for similarity.py on synthetic data. Run: python test_similarity.py"""

from datetime import date

import numpy as np
import pandas as pd

from similarity import (IST, OWN_OPEN, SAME_CLOCK, align, interval_returns, move_match, rank,
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


def test_own_open_skips_lunch():
    nifty = session_path(walk(minutes("09:15", "12:00"), 1), DAY, ELAPSED).values
    other = drop_lunch(walk(minutes("07:00", "12:00"), 2), "09:30", "10:30")  # Hang Seng-style lunch
    p = session_path(other, DAY, ELAPSED, OWN_OPEN)
    assert list(p.values.index) == list(range(ELAPSED + 1))  # runs as long as Nifty, no gap
    assert p.clock()[149] == ts("09:29") and p.clock()[150] == ts("10:30")
    assert p.clock()[-1] == ts("10:45")
    assert len(align(nifty, p.values)) == ELAPSED + 1


def test_same_clock_no_forward_fill():
    nifty = session_path(walk(minutes("09:15", "12:00"), 1), DAY, ELAPSED, SAME_CLOCK).values
    other = drop_lunch(walk(minutes("07:00", "12:00"), 2), "09:30", "10:30")
    a = align(nifty, session_path(other, DAY, ELAPSED, SAME_CLOCK).values)
    assert not a.index.isin(range(15, 75)).any()  # 9:30..10:29 is lunch
    assert not a.isna().any().any()


def test_flat_padded_lunch_is_a_break():
    s = walk(minutes("07:00", "12:00"), 6)
    lunch = (s.index >= ts("09:00")) & (s.index < ts("10:30"))
    s[lunch] = s[ts("08:59")]  # Yahoo-style flat bars through lunch
    p = session_path(s, DAY, ELAPSED, OWN_OPEN)
    assert p.clock()[119] == ts("08:59") and p.clock()[120] == ts("10:30")


def test_short_gaps_stay_missing():
    s = walk(minutes("07:00", "12:00"), 5)
    s = s.drop(s.index[30:35])  # five missing bars, not a session break
    p = session_path(s, DAY, ELAPSED, OWN_OPEN)
    assert not p.values.index.isin(range(30, 35)).any()
    assert p.values.index[-1] == ELAPSED


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


def test_prev_close_base_and_points():
    s = pd.Series(np.linspace(25012, 24942, 166), index=minutes("09:15", "12:00"))
    p = session_path(s, DAY, ELAPSED, OWN_OPEN, prev_close=25000)
    assert abs(p.gap - 0.048) < 1e-9          # opened +12 pts
    assert abs(p.points.iloc[0] - 12) < 1e-9
    assert abs(p.points.iloc[-1] + 58) < 1e-9  # 24942 is -58 pts from the previous close
    assert abs(p.values.iloc[-1] - (24942 / 25000 - 1) * 100) < 1e-9
    assert abs(p.since_open.iloc[-1] - (24942 / 25012 - 1) * 100) < 1e-9  # gap taken out


def test_move_match():
    df = pd.DataFrame({"nifty": [-0.27, 0.1], "other": [-0.20, 0.1]})
    assert 0.85 < move_match(df) < 0.95
    assert move_match(pd.DataFrame({"nifty": [-0.2], "other": [0.2]})) == 0


def test_similar_points_beat_big_gap_same_shape():
    """The README example: Hang Seng opens +10 pts and falls 50 while Nifty opens +12 and
    falls 70. KOSPI opens +3% and then moves exactly like Nifty. Hang Seng is closer."""
    idx = minutes("09:15", "12:00")
    rng = np.random.default_rng(11)
    wiggle = np.cumsum(rng.normal(0, 3, len(idx)))
    nifty = pd.Series(np.linspace(25012, 24942, len(idx)) + wiggle, index=idx)
    hk_idx = idx - pd.Timedelta(minutes=135)  # opens 7:00 IST
    hang_seng = pd.Series(np.linspace(26010, 25960, len(idx)) + 0.8 * wiggle
                          + rng.normal(0, 4, len(idx)), index=hk_idx)
    kospi = pd.Series(2575 * nifty.values / nifty.iloc[0], index=idx - pd.Timedelta(minutes=225))
    others = {"Hang Seng": hang_seng, "KOSPI": kospi}
    res = rank(nifty, others, DAY, ELAPSED, OWN_OPEN, tickers={k: k for k in others},
               prev_closes={"Hang Seng": 26000, "KOSPI": 2500}, nifty_prev=25000)
    by = {r.name: r for r in res}
    assert by["KOSPI"].shape > 0.99  # same shape...
    assert by["Hang Seng"].score > by["KOSPI"].score + 20, (by["Hang Seng"].score, by["KOSPI"].score)
    assert abs(by["Hang Seng"].gap_points - 10) < 15  # +10 pts plus the first bar's noise
    assert abs(by["KOSPI"].gap - 3) < 1e-9


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
    print("OK")
