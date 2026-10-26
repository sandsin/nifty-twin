"""Session alignment and similarity scoring against Nifty 50.

Paths are indexed by elapsed minutes from an anchor, so two indices can be compared
either from each one's own session open ("own open", the default) or over the same
IST clock window starting 9:15 ("same clock").
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
    "path_corr": 0.35,
    "return_corr": 0.25,
    "direction": 0.15,
    "closeness": 0.25,
}

# RMSE (in percentage points) at which closeness drops to 1/e.
CLOSENESS_SCALE = 0.5
MIN_POINTS = 5
MIN_RETURNS = 3


def nifty_open(day: date) -> pd.Timestamp:
    return pd.Timestamp(datetime.combine(day, OPEN_IST)).tz_localize(IST)


def day_bars(series: pd.Series, day: date) -> pd.Series:
    """Bars on one IST date. Every covered Asian market opens and closes within it."""
    s = series.dropna().sort_index()
    if s.empty:
        return s
    start = pd.Timestamp(day).tz_localize(IST)
    return s[(s.index >= start) & (s.index < start + pd.Timedelta(days=1))]


@dataclass
class Path:
    """Cumulative % change indexed by whole minutes since `anchor`."""
    values: pd.Series
    anchor: pd.Timestamp | None

    def clock(self) -> pd.DatetimeIndex:
        return self.anchor + pd.to_timedelta(self.values.index, unit="min")


def session_path(series: pd.Series, day: date, elapsed: int, mode: str = OWN_OPEN) -> Path:
    """The index's path over the first `elapsed` minutes after its anchor on `day`.

    OWN_OPEN: anchor and base are the index's first bar of the day (its own open).
    SAME_CLOCK: anchor is 9:15 IST; the base is the last bar at or before 9:15 that day
    (an index already trading or on lunch), else the first bar after it.
    """
    bars = day_bars(series, day)
    empty = Path(pd.Series(dtype=float), None)
    if bars.empty:
        return empty
    if mode == OWN_OPEN:
        anchor = bars.index[0]
        base = bars.iloc[0]
    else:
        anchor = nifty_open(day)
        before = bars[bars.index <= anchor]
        after = bars[bars.index >= anchor]
        if after.empty:
            return empty
        base = before.iloc[-1] if not before.empty else after.iloc[0]
    window = bars[(bars.index >= anchor) & (bars.index <= anchor + pd.Timedelta(minutes=elapsed))]
    if window.empty or base == 0 or not np.isfinite(base):
        return empty
    mins = ((window.index - anchor) / pd.Timedelta(minutes=1)).astype(int)
    values = pd.Series((window.values / base - 1.0) * 100.0, index=mins)
    return Path(values[~values.index.duplicated(keep="last")], anchor)


def align(a: pd.Series, b: pd.Series) -> pd.DataFrame:
    """Two paths on the elapsed minutes where both actually traded (no forward fill)."""
    return pd.concat([a.rename("nifty"), b.rename("other")], axis=1, join="inner").dropna()


def interval_returns(aligned: pd.DataFrame, minutes: int) -> pd.DataFrame:
    """Changes over exact `minutes`-minute hops where both ends are aligned bars.

    Hops that straddle a lunch break or missing bar are dropped rather than bridged.
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


@dataclass
class Score:
    name: str
    ticker: str
    status: str  # "ranked", "partial", "no data"
    score: float = float("nan")
    path_corr: float = float("nan")
    return_corr: float = float("nan")
    direction: float = float("nan")
    rmse: float = float("nan")
    coverage: float = 0.0
    points: int = 0
    anchor: pd.Timestamp | None = None
    last_bar: pd.Timestamp | None = None
    components: dict = field(default_factory=dict)


def score_pair(aligned: pd.DataFrame, interval: int) -> dict:
    """Component metrics and the 0-100 score for one aligned pair."""
    path_corr = _corr(aligned["nifty"], aligned["other"])
    rets = interval_returns(aligned, interval)
    if len(rets) >= MIN_RETURNS:
        return_corr = _corr(rets["nifty"], rets["other"])
        sa, sb = np.sign(rets["nifty"]), np.sign(rets["other"])
        moving = (sa != 0) | (sb != 0)
        direction = float((sa[moving] == sb[moving]).mean()) if moving.any() else 0.0
    else:
        return_corr, direction = 0.0, 0.0
    rmse = float(np.sqrt(((aligned["nifty"] - aligned["other"]) ** 2).mean()))
    closeness = float(np.exp(-rmse / CLOSENESS_SCALE))

    parts = {
        "path_corr": max(path_corr, 0.0),
        "return_corr": max(return_corr, 0.0),
        "direction": direction,
        "closeness": closeness,
    }
    total = 100.0 * sum(WEIGHTS[k] * v for k, v in parts.items())
    return {
        "score": total,
        "path_corr": path_corr,
        "return_corr": return_corr,
        "direction": direction,
        "rmse": rmse,
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
) -> list[Score]:
    """Score every candidate over the first `elapsed` minutes of Nifty's session.

    Ranked ones come first by score, then partial, then no data.
    """
    tickers = tickers or CANDIDATES
    nifty_path = session_path(nifty, day, elapsed, mode).values
    n_nifty = len(nifty_path)
    results: list[Score] = []
    for name, series in others.items():
        ticker = tickers.get(name, name)
        path = session_path(series, day, elapsed, mode) if series is not None else Path(pd.Series(dtype=float), None)
        if path.values.empty or n_nifty == 0:
            results.append(Score(name, ticker, "no data"))
            continue
        last_bar = path.clock()[-1]
        aligned = align(nifty_path, path.values)
        coverage = len(aligned) / n_nifty
        if len(aligned) < MIN_POINTS:
            results.append(Score(name, ticker, "no data", points=len(aligned), coverage=coverage,
                                 anchor=path.anchor, last_bar=last_bar))
            continue
        m = score_pair(aligned, interval)
        status = "ranked" if coverage >= min_coverage else "partial"
        results.append(Score(name, ticker, status, coverage=coverage, points=len(aligned),
                             anchor=path.anchor, last_bar=last_bar, **m))

    order = {"ranked": 0, "partial": 1, "no data": 2}
    results.sort(key=lambda r: (order[r.status], -(r.score if np.isfinite(r.score) else -1)))
    return results
