"""Session alignment and similarity scoring against Nifty 50.

Each path is the % change from the index's previous official close, so it starts at the
opening gap. Paths are indexed by elapsed minutes from an anchor, so two indices can be
compared either from each one's own session open ("own open", the default) or over the
same IST clock window starting 9:15 ("same clock").

The score rewards matching the size of moves, not just the shape: an index that opens
+3% and then moves like Nifty is further away than one that opens and moves by about
as much as Nifty does.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, time

import numpy as np
import pandas as pd

IST = "Asia/Kolkata"
NIFTY = "^NSEI"
OPEN_IST = time(9, 15)

OWN_OPEN = "own open"
SAME_CLOCK = "same clock"

CANDIDATES: dict[str, str] = {
    "Nikkei 225": "^N225",
    "Hang Seng": "^HSI",
    "Shanghai Composite": "000001.SS",
    "Shenzhen Component": "399001.SZ",
    "KOSPI": "^KS11",
    "Taiwan TAIEX": "^TWII",
    "Straits Times": "^STI",
    "ASX 200": "^AXJO",
    "Jakarta Composite": "^JKSE",
    "KLCI": "^KLSE",
}

WEIGHTS = {
    "level": 0.35,      # how close the two lines are, gap included
    "moves": 0.25,      # how alike each interval's move is in size and sign
    "gap": 0.15,        # how alike the opening gaps are
    "shape": 0.15,      # path correlation
    "direction": 0.10,  # how often they move the same way
}
LABELS = {
    "level": "Level closeness",
    "moves": "Move size match",
    "gap": "Opening gap match",
    "shape": "Shape",
    "direction": "Direction",
}

# RMSE between paths (percentage points) at which level closeness drops to 1/e.
LEVEL_SCALE = 0.5
# Difference in opening gaps (percentage points) at which gap match drops to 1/e.
GAP_SCALE = 0.25
MIN_POINTS = 5
MIN_RETURNS = 3
# A gap in bars longer than this (minutes) is a session break, e.g. a lunch break.
BREAK_GAP = 15


def nifty_open(day: date) -> pd.Timestamp:
    return pd.Timestamp(datetime.combine(day, OPEN_IST)).tz_localize(IST)


def day_bars(series: pd.Series, day: date) -> pd.Series:
    """Bars on one IST date. Every covered Asian market opens and closes within it."""
    s = series.dropna().sort_index()
    if s.empty:
        return s
    start = pd.Timestamp(day).tz_localize(IST)
    s = s[(s.index >= start) & (s.index < start + pd.Timedelta(days=1))]
    # Yahoo pads some lunch breaks (e.g. Shenzhen) with flat bars; a price frozen for longer
    # than BREAK_GAP is a break, so keep its first bar and drop the padding.
    run = (s.diff() != 0).cumsum()
    run_len = run.map(run.value_counts())
    padding = (run_len > BREAK_GAP) & run.duplicated()
    return s[~padding]


@dataclass
class Path:
    """% change from `base`, indexed by whole minutes since `anchor`.

    `levels` holds the index's price at each point and `times` the bar's IST time.
    `base` is the previous official close when known (then `gap` is the opening gap in %),
    else the first bar or the 9:15 price, and `gap` is NaN. `open_level` is the day's first bar.
    """
    values: pd.Series
    anchor: pd.Timestamp | None = None
    times: pd.DatetimeIndex | None = None
    levels: pd.Series | None = None
    base: float = float("nan")
    gap: float = float("nan")
    open_level: float = float("nan")

    def clock(self) -> pd.DatetimeIndex:
        return self.times

    @property
    def points(self) -> pd.Series:
        return self.levels - self.base

    @property
    def since_open(self) -> pd.Series:
        """% move from the day's open, i.e. the intraday move with the opening gap taken out."""
        return (self.levels / self.open_level - 1.0) * 100.0


EMPTY = Path(pd.Series(dtype=float))


def trading_minutes(index: pd.DatetimeIndex, anchor: pd.Timestamp) -> np.ndarray:
    """Minutes since `anchor` with session breaks (gaps over BREAK_GAP) taken out.

    Shorter gaps are missing bars and still count, so they stay missing rather than filled.
    """
    clock = ((index - anchor) / pd.Timedelta(minutes=1)).to_numpy()
    gaps = np.diff(clock, prepend=clock[0])
    breaks = np.where(gaps > BREAK_GAP, gaps - 1, 0)
    return (clock - np.cumsum(breaks)).astype(int)


def session_path(series: pd.Series, day: date, elapsed: int, mode: str = OWN_OPEN,
                 prev_close: float | None = None) -> Path:
    """The index's path over the first `elapsed` minutes after its anchor on `day`.

    OWN_OPEN: anchored at the index's first bar of the day (its own open), in trading
    minutes, so lunch breaks are skipped and every path runs as long as Nifty's.
    SAME_CLOCK: anchored at 9:15 IST, in clock minutes.
    Both measure from `prev_close`. Without it, the base falls back to the first bar
    (own open) or the last bar at or before 9:15 (same clock).
    """
    bars = day_bars(series, day)
    if bars.empty:
        return EMPTY
    has_prev = prev_close is not None and np.isfinite(prev_close) and prev_close > 0
    if mode == OWN_OPEN:
        anchor = bars.index[0]
        mins = trading_minutes(bars.index, anchor)
        fallback = bars.iloc[0]
    else:
        anchor = nifty_open(day)
        if bars[bars.index >= anchor].empty:
            return EMPTY
        before = bars[bars.index <= anchor]
        fallback = before.iloc[-1] if not before.empty else bars[bars.index >= anchor].iloc[0]
        mins = ((bars.index - anchor) / pd.Timedelta(minutes=1)).to_numpy().astype(int)
    base = float(prev_close) if has_prev else float(fallback)
    keep = (mins >= 0) & (mins <= elapsed)
    if not keep.any() or base == 0 or not np.isfinite(base):
        return EMPTY
    window = bars[keep]
    dup = pd.Index(mins[keep]).duplicated(keep="last")
    idx = mins[keep][~dup]
    levels = pd.Series(window.values[~dup], index=idx)
    gap = (bars.iloc[0] / base - 1.0) * 100.0 if has_prev else float("nan")
    return Path((levels / base - 1.0) * 100.0, anchor, window.index[~dup], levels, base, gap,
                float(bars.iloc[0]))


def align(a: pd.Series, b: pd.Series) -> pd.DataFrame:
    """Two paths on the elapsed minutes where both actually traded (no forward fill)."""
    return pd.concat([a.rename("nifty"), b.rename("other")], axis=1, join="inner").dropna()


def interval_returns(aligned: pd.DataFrame, minutes: int) -> pd.DataFrame:
    """Changes over exact `minutes`-minute hops where both ends are aligned bars.

    Hops that straddle a missing bar are dropped rather than bridged.
    """
    if aligned.empty:
        return aligned
    lagged = aligned.copy()
    lagged.index = lagged.index + minutes
    joined = aligned.join(lagged, rsuffix="_prev", how="inner")
    out = pd.DataFrame({
        "nifty": joined["nifty"] - joined["nifty_prev"],
        "other": joined["other"] - joined["other_prev"],
    })
    # Non-overlapping hops so each minute counts once.
    if len(out):
        out = out[(out.index - out.index[0]) % minutes == 0]
    return out


def _corr(x: pd.Series, y: pd.Series) -> float:
    if len(x) < 2 or x.std() == 0 or y.std() == 0:
        return 0.0
    c = float(np.corrcoef(x, y)[0, 1])
    return c if np.isfinite(c) else 0.0


def move_match(rets: pd.DataFrame) -> float:
    """1 when every interval move is the same size and sign, 0 when unrelated or opposite.

    1 - sum|a - b| / sum(|a| + |b|): Nifty -0.27% vs -0.20% scores 0.85; vs +0.20% scores 0.
    """
    a, b = rets["nifty"].to_numpy(), rets["other"].to_numpy()
    total = np.abs(a).sum() + np.abs(b).sum()
    if total == 0:
        return 1.0
    return float(np.clip(1.0 - np.abs(a - b).sum() / total, 0.0, 1.0))


@dataclass
class Score:
    name: str
    ticker: str
    status: str  # "ranked", "partial", "no data"
    score: float = float("nan")
    level_rmse: float = float("nan")
    move_match: float = float("nan")
    gap: float = float("nan")         # this index's opening gap, %
    gap_points: float = float("nan")
    gap_diff: float = float("nan")    # gap minus Nifty's gap, percentage points
    shape: float = float("nan")
    direction: float = float("nan")
    now: float = float("nan")         # latest % from previous close within the window
    since_open: float = float("nan")  # latest % from the day's open
    coverage: float = 0.0
    points: int = 0
    anchor: pd.Timestamp | None = None
    last_bar: pd.Timestamp | None = None
    components: dict = field(default_factory=dict)  # each WEIGHTS key -> 0-1, as scored


def score_pair(aligned: pd.DataFrame, interval: int, nifty_gap: float = float("nan"),
               other_gap: float = float("nan")) -> dict:
    """Component metrics and the 0-100 score for one aligned pair.

    If either opening gap is unknown, the gap component is left out and the other
    weights are scaled up to fill 100.
    """
    shape = _corr(aligned["nifty"], aligned["other"])
    rets = interval_returns(aligned, interval)
    if len(rets) >= MIN_RETURNS:
        moves = move_match(rets)
        sa, sb = np.sign(rets["nifty"]), np.sign(rets["other"])
        moving = (sa != 0) | (sb != 0)
        direction = float((sa[moving] == sb[moving]).mean()) if moving.any() else 0.0
    else:
        moves, direction = 0.0, 0.0
    rmse = float(np.sqrt(((aligned["nifty"] - aligned["other"]) ** 2).mean()))
    gap_diff = other_gap - nifty_gap

    parts = {
        "level": float(np.exp(-rmse / LEVEL_SCALE)),
        "moves": moves,
        "shape": max(shape, 0.0),
        "direction": direction,
    }
    if np.isfinite(gap_diff):
        parts["gap"] = float(np.exp(-abs(gap_diff) / GAP_SCALE))
    weight = sum(WEIGHTS[k] for k in parts)
    total = 100.0 * sum(WEIGHTS[k] * v for k, v in parts.items()) / weight
    return {
        "score": total,
        "level_rmse": rmse,
        "move_match": moves,
        "gap_diff": gap_diff,
        "shape": shape,
        "direction": direction,
        "components": parts,
    }


def rank(
    nifty: pd.Series,
    others: dict[str, pd.Series],
    day: date,
    elapsed: int,
    mode: str = OWN_OPEN,
    interval: int = 5,
    min_coverage: float = 0.7,
    tickers: dict[str, str] | None = None,
    prev_closes: dict[str, float | None] | None = None,
    nifty_prev: float | None = None,
) -> list[Score]:
    """Score every candidate over the first `elapsed` minutes of Nifty's session.

    `prev_closes` maps candidate name -> previous official close. Ranked ones come first
    by score, then partial, then no data.
    """
    tickers = tickers or CANDIDATES
    prev_closes = prev_closes or {}
    npath = session_path(nifty, day, elapsed, mode, nifty_prev)
    n_nifty = len(npath.values)
    results: list[Score] = []
    for name, series in others.items():
        ticker = tickers.get(name, name)
        path = session_path(series, day, elapsed, mode, prev_closes.get(name)) if series is not None else EMPTY
        if path.values.empty or n_nifty == 0:
            results.append(Score(name, ticker, "no data"))
            continue
        info = dict(
            anchor=path.anchor, last_bar=path.clock()[-1], gap=path.gap,
            gap_points=path.gap * path.base / 100.0, now=path.values.iloc[-1],
            since_open=path.since_open.iloc[-1],
        )
        aligned = align(npath.values, path.values)
        coverage = len(aligned) / n_nifty
        if len(aligned) < MIN_POINTS:
            results.append(Score(name, ticker, "no data", points=len(aligned), coverage=coverage, **info))
            continue
        m = score_pair(aligned, interval, npath.gap, path.gap)
        status = "ranked" if coverage >= min_coverage else "partial"
        results.append(Score(name, ticker, status, coverage=coverage, points=len(aligned), **info, **m))

    order = {"ranked": 0, "partial": 1, "no data": 2}
    results.sort(key=lambda r: (order[r.status], -(r.score if np.isfinite(r.score) else -1)))
    return results
