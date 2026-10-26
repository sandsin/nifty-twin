"""Nifty Twin: which Asian index is moving most like Nifty 50 since its 9:15 IST open."""

from __future__ import annotations

from datetime import datetime, time

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from data import fetch_closes, trading_days
from similarity import (CANDIDATES, IST, NIFTY, OWN_OPEN, SAME_CLOCK, WEIGHTS, Path, nifty_open,
                        rank, session_path)

MARKET_CLOSE = time(15, 30)
EARLY_BARS = 15
AUTO = "Best match (auto)"
NIFTY_COLOR = "#2a78d6"
OTHER_COLOR = "#eb6834"
MODES = {
    OWN_OPEN: "Each index's own open",
    SAME_CLOCK: "Same clock time (from 9:15 IST)",
}
EPOCH = pd.Timestamp("2000-01-01")  # elapsed minutes are plotted as clock offsets from this

st.set_page_config(page_title="Nifty Twin", page_icon="📈", layout="wide")


@st.cache_data(ttl=25, show_spinner=False)
def load(tickers: tuple[str, ...]) -> dict[str, pd.Series]:
    return fetch_closes(list(tickers))


def fmt_elapsed(minutes: int) -> str:
    return f"{minutes // 60}h {minutes % 60:02d}m"


# ---------- Sidebar ----------
with st.sidebar:
    st.header("Settings")
    names = st.multiselect("Indices", list(CANDIDATES), default=list(CANDIDATES))
    mode = st.radio("Align by", list(MODES), format_func=MODES.get,
                    help="Own open: Nifty's first N minutes after 9:15 vs each index's first N "
                         "minutes after its own open (e.g. Hang Seng from 7:00 IST). "
                         "Same clock: both over the same IST minutes.")
    live = st.toggle("Live", value=True, help="Window ends now and the page auto-updates.")
    refresh = st.slider("Auto-update every (seconds)", 10, 120, 60, step=5, disabled=not live,
                        help="Yahoo publishes 1-minute bars about once a minute.")

    tickers = tuple([NIFTY] + [CANDIDATES[n] for n in names])
    with st.spinner("Loading 1-minute data…"):
        closes = load(tickers)
    days = trading_days(closes[NIFTY])

    if live:
        day, end_time = None, None
    else:
        day = st.selectbox("Trading day", days, format_func=lambda d: d.strftime("%a %d %b %Y"),
                           disabled=not days)
        end_time = st.time_input("Nifty window end (IST)", value=MARKET_CLOSE, step=60)

    interval = st.slider("Return interval (minutes)", 1, 15, 5,
                         help="Minutes per hop for return correlation and direction match.")
    min_cov = st.slider("Min data coverage (%)", 0, 100, 70,
                        help="Overlap with Nifty's bars needed to rank an index.") / 100
    st.caption("Score weights: " + ", ".join(f"{k.replace('_', ' ')} {int(v * 100)}%"
                                             for k, v in WEIGHTS.items()))


def path_frame(path: Path) -> tuple[pd.Series, list[str]]:
    """Path on a full-minute elapsed grid (NaN where not trading, so lines break) plus IST clock labels."""
    v = path.values
    if v.empty:
        return v, []
    grid = range(0, int(v.index[-1]) + 1)
    full = v.reindex(grid)
    clock = pd.Series(path.clock().strftime("%H:%M"), index=v.index).reindex(grid).fillna("")
    return full, clock.tolist()


def chart(nifty: Path, other: Path, other_name: str) -> go.Figure:
    fig = go.Figure()
    for path, name, color, width in [
        (nifty, "Nifty 50", NIFTY_COLOR, 2.5),
        (other, other_name, OTHER_COLOR, 2),
    ]:
        values, clock = path_frame(path)
        if values.empty:
            continue
        x = EPOCH + pd.to_timedelta(values.index, unit="min")
        # The latest value rides in the legend rather than an end-of-line label, which
        # would need a wide right margin that squeezes the plot on a phone.
        fig.add_trace(go.Scatter(
            x=x, y=values.values, name=f"{name} {values.dropna().iloc[-1]:+.2f}%", mode="lines",
            customdata=clock, line=dict(color=color, width=width), connectgaps=False,
            hovertemplate="%{y:+.2f}% at %{customdata} IST<extra>" + name + "</extra>",
        ))
    fig.add_hline(y=0, line_width=1, line_color="rgba(128,128,128,0.5)")
    x_title = "Trading time since open (h:mm)" if mode == OWN_OPEN else "Time since 9:15 IST (h:mm)"
    fig.update_layout(
        height=380, margin=dict(l=10, r=10, t=10, b=10), hovermode="x unified",
        dragmode=False,  # so a touch drag scrolls the page instead of zooming
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        yaxis=dict(title="Change since open", ticksuffix="%", zeroline=False, fixedrange=True,
                   gridcolor="rgba(128,128,128,0.15)"),
        xaxis=dict(title=x_title, tickformat="%H:%M", hoverformat="+%H:%M", showgrid=False,
                   fixedrange=True),
    )
    return fig


def results_table(results) -> pd.DataFrame:
    rows = []
    for i, r in enumerate(results, start=1):
        # Score sits next to the name so it shows on a phone without scrolling sideways.
        row = {"Rank": i, "Index": r.name, "Score": r.score}
        if mode == OWN_OPEN:
            row["Opens (IST)"] = r.anchor.strftime("%H:%M") if r.anchor is not None else "—"
        row.update({
            "Path corr": r.path_corr,
            "Return corr": r.return_corr,
            "Direction match": r.direction * 100,
            "Gap RMSE (pp)": r.rmse,
            "Coverage": r.coverage * 100,
            "Compared to (IST)": r.last_bar.strftime("%H:%M") if r.last_bar is not None else "—",
        })
        rows.append(row)
    return pd.DataFrame(rows)


def dashboard() -> None:
    data = load(tickers)
    nifty = data[NIFTY]
    if nifty.empty:
        st.error("No Nifty data came back from Yahoo. It may be rate-limiting this server; "
                 "try again shortly or run the app locally.")
        return

    now = pd.Timestamp.now(tz=IST)
    nifty_days = trading_days(nifty)
    if live:
        sel_day = now.date()
        if sel_day in nifty_days:
            end = now
        else:
            sel_day = nifty_days[0]
            end = pd.Timestamp(datetime.combine(sel_day, MARKET_CLOSE)).tz_localize(IST)
            st.info(f"Nifty has not traded today yet. Showing the latest session, "
                    f"{sel_day:%a %d %b}.")
    else:
        if day is None:
            st.warning("No trading days available.")
            return
        sel_day = day
        end = pd.Timestamp(datetime.combine(sel_day, end_time)).tz_localize(IST)
    elapsed = int((end - nifty_open(sel_day)) / pd.Timedelta(minutes=1))
    if elapsed < 0:
        st.info("The window ends before Nifty's 9:15 IST open.")
        return

    others = {n: data.get(CANDIDATES[n], pd.Series(dtype=float)) for n in names}
    results = rank(nifty, others, sel_day, elapsed, mode=mode, interval=interval, min_coverage=min_cov)
    ranked = [r for r in results if r.status == "ranked"]
    partial = [r for r in results if r.status == "partial"]
    missing = [r for r in results if r.status == "no data"]

    nifty_path = session_path(nifty, sel_day, elapsed, mode)
    nv = nifty_path.values
    last_bar = nifty_path.clock()[-1] if not nv.empty else None

    # ---------- Header ----------
    st.title("Nifty Twin")
    sub = f"{sel_day:%a %d %b %Y} · Nifty 9:15 → {end:%H:%M} IST ({fmt_elapsed(elapsed)})"
    if mode == OWN_OPEN:
        sub += " · each index over the same span from its own open"
    if live:
        sub += f" · updated {now:%H:%M:%S} IST, every {refresh}s"
    st.caption(sub)

    c1, c2, c3 = st.columns(3)
    if ranked:
        c1.metric("Closest match", ranked[0].name, f"score {ranked[0].score:.0f} / 100",
                  delta_color="off")
    else:
        c1.metric("Closest match", "—")
    c2.metric("Nifty since 9:15", f"{nv.iloc[-1]:+.2f}%" if last_bar is not None else "—")
    if last_bar is not None:
        lag = ""
        if live and sel_day == now.date():
            lag = f"{int((now - last_bar) / pd.Timedelta(minutes=1))} min ago"
        c3.metric("Nifty last bar", f"{last_bar:%H:%M} IST", lag, delta_color="off")
    else:
        c3.metric("Nifty last bar", "—")

    if len(nv) < EARLY_BARS:
        st.warning(f"Only {len(nv)} Nifty bars so far. Matches are noisy until about "
                   f"{EARLY_BARS} bars are in.")

    if not ranked and not partial:
        st.info("No index has enough overlapping data with Nifty in this window yet.")
        return

    # ---------- Chart ----------
    with_data = [r.name for r in ranked + partial]
    choice = st.selectbox("Compare Nifty with", [AUTO] + [n for n in names if n in with_data],
                          key="compare")
    other_name = (ranked[0].name if ranked else partial[0].name) if choice == AUTO else choice
    other_path = session_path(others[other_name], sel_day, elapsed, mode)
    st.plotly_chart(chart(nifty_path, other_path, other_name), width="stretch",
                    config={"displayModeBar": False})
    if mode == OWN_OPEN and other_path.anchor is not None:
        st.caption(f"Nifty 9:15 → {end:%H:%M} IST vs {other_name} "
                   f"{other_path.anchor:%H:%M} → {other_path.clock()[-1]:%H:%M} IST, both from their "
                   "own open. Time is trading time: lunch breaks are skipped so both lines run "
                   "side by side. Tap or hover to see each bar's IST time.")
    else:
        st.caption("Lines break where an index was not trading (lunch breaks, closed markets). "
                   "Nothing is forward-filled.")

    # ---------- Tables ----------
    cols = {
        "Index": st.column_config.TextColumn("Index", pinned=True),
        "Score": st.column_config.ProgressColumn("Score", min_value=0, max_value=100, format="%.0f"),
        "Path corr": st.column_config.NumberColumn(format="%.2f"),
        "Return corr": st.column_config.NumberColumn(format="%.2f"),
        "Direction match": st.column_config.NumberColumn(format="%.0f%%"),
        "Gap RMSE (pp)": st.column_config.NumberColumn(format="%.2f"),
        "Coverage": st.column_config.NumberColumn(format="%.0f%%"),
    }
    st.subheader("Ranking")
    if ranked:
        st.dataframe(results_table(ranked), hide_index=True, width="stretch",
                     column_config=cols)
    else:
        st.info(f"No index meets the {min_cov:.0%} coverage minimum yet.")

    if partial:
        st.subheader("Partial data (not ranked)")
        st.dataframe(results_table(partial).drop(columns="Rank"), hide_index=True,
                     width="stretch", column_config=cols)
    if missing:
        st.caption("No data in this window: " + ", ".join(r.name for r in missing))

    st.caption("Similarity of intraday paths does not predict what Nifty does next. Not trading advice.")


st.fragment(run_every=refresh if live else None)(dashboard)()
