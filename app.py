"""Nifty Twin: which Asian index is moving most like Nifty 50 since its 9:15 IST open."""

from __future__ import annotations

from datetime import datetime, time

import anthropic
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import ai_analysis
from market_data import fetch_closes, fetch_daily_closes, prev_close, trading_days
from similarity import (CANDIDATES, IST, LABELS, NIFTY, OWN_OPEN, SAME_CLOCK, WEIGHTS, Path,
                        nifty_open, rank, session_path)

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


@st.cache_data(ttl=600, show_spinner=False)
def load_daily(tickers: tuple[str, ...]) -> dict[str, pd.Series]:
    return fetch_daily_closes(list(tickers))


def fmt_elapsed(minutes: int) -> str:
    return f"{minutes // 60}h {minutes % 60:02d}m"


def fmt_move(pct: float, pts: float) -> str:
    """'+0.05% (+12 pts)', or '—' when unknown."""
    if not np.isfinite(pct):
        return "—"
    return f"{pct:+.2f}% ({pts:+,.0f} pts)" if np.isfinite(pts) else f"{pct:+.2f}%"


def fmt_pct(pct: float) -> str:
    """Up/down move in %: '▲ 0.32%', '▼ 0.20%', or '—' when unknown."""
    if not np.isfinite(pct):
        return "—"
    arrow = "▲" if pct > 0.005 else "▼" if pct < -0.005 else "■"
    return f"{arrow} {abs(pct):.2f}%"


def has_prev(path: Path) -> bool:
    """Whether the path is measured from a known previous close."""
    return bool(np.isfinite(path.gap))


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

    interval = st.slider("Move interval (minutes)", 1, 15, 5,
                         help="Minutes per hop when comparing the size and direction of moves.")
    min_cov = st.slider("Min data coverage (%)", 0, 100, 70,
                        help="Overlap with Nifty's bars needed to rank an index.") / 100
    st.caption("Score weights: " + ", ".join(f"{LABELS[k]} {int(v * 100)}%"
                                             for k, v in WEIGHTS.items()))


def path_frame(path: Path) -> tuple[pd.Series, list[list[str]]]:
    """Path on a full-minute elapsed grid (NaN where not trading, so lines break) plus hover
    labels per minute: IST time, move vs previous close, move since open."""
    v = path.values
    if v.empty:
        return v, []
    grid = range(0, int(v.index[-1]) + 1)
    full = v.reindex(grid)
    labels = pd.DataFrame({
        "clock": path.clock().strftime("%H:%M"),
        "vs_prev": [fmt_pct(x) for x in v],
        "since_open": [fmt_pct(x) for x in path.since_open],
    }, index=v.index).reindex(grid).fillna("")
    return full, labels.values.tolist()


def chart(nifty: Path, other: Path, other_name: str) -> go.Figure:
    fig = go.Figure()
    for path, name, color, width in [
        (nifty, "Nifty 50", NIFTY_COLOR, 2.5),
        (other, other_name, OTHER_COLOR, 2),
    ]:
        values, labels = path_frame(path)
        if values.empty:
            continue
        x = EPOCH + pd.to_timedelta(values.index, unit="min")
        # The latest value rides in the legend rather than an end-of-line label, which
        # would need a wide right margin that squeezes the plot on a phone.
        latest = fmt_pct(path.values.iloc[-1])
        fig.add_trace(go.Scatter(
            x=x, y=values.values, name=f"{name} {latest}", mode="lines",
            customdata=labels, line=dict(color=color, width=width), connectgaps=False,
            hovertemplate=("%{customdata[1]} · since open %{customdata[2]} "
                           "at %{customdata[0]} IST<extra>" + name + "</extra>"),
        ))
    fig.add_hline(y=0, line_width=1, line_color="rgba(128,128,128,0.5)")
    x_title = "Trading time since open (h:mm)" if mode == OWN_OPEN else "Time since 9:15 IST (h:mm)"
    y_title = "Change from previous close" if has_prev(nifty) else "Change since open"
    fig.update_layout(
        height=380, margin=dict(l=10, r=10, t=10, b=10), hovermode="x unified",
        dragmode=False,  # so a touch drag scrolls the page instead of zooming
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        yaxis=dict(title=y_title, ticksuffix="%", zeroline=False, fixedrange=True,
                   gridcolor="rgba(128,128,128,0.15)"),
        xaxis=dict(title=x_title, tickformat="%H:%M", hoverformat="+%H:%M", showgrid=False,
                   fixedrange=True),
    )
    return fig


def results_table(results) -> pd.DataFrame:
    rows = []
    for i, r in enumerate(results, start=1):
        # Score sits next to the name so it shows on a phone without scrolling sideways.
        row = {
            "Rank": i, "Index": r.name, "Score": r.score,
            "Opening gap": fmt_move(r.gap, r.gap_points),
            "Since open": fmt_pct(r.since_open),
            "Vs prev close": fmt_pct(r.now) if np.isfinite(r.gap) else "—",
            "Level gap (pp)": r.level_rmse,
            "Move size match": r.move_match * 100,
            "Shape": r.shape,
            "Direction": r.direction * 100,
        }
        if mode == OWN_OPEN:
            row["Opens (IST)"] = r.anchor.strftime("%H:%M") if r.anchor is not None else "—"
        row.update({
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

    daily = load_daily(tickers)
    nifty_prev = prev_close(daily.get(NIFTY, pd.Series(dtype=float)), sel_day)
    prevs = {n: prev_close(daily.get(CANDIDATES[n], pd.Series(dtype=float)), sel_day) for n in names}
    others = {n: data.get(CANDIDATES[n], pd.Series(dtype=float)) for n in names}
    results = rank(nifty, others, sel_day, elapsed, mode=mode, interval=interval,
                   min_coverage=min_cov, prev_closes=prevs, nifty_prev=nifty_prev)
    ranked = [r for r in results if r.status == "ranked"]
    partial = [r for r in results if r.status == "partial"]
    missing = [r for r in results if r.status == "no data"]

    nifty_path = session_path(nifty, sel_day, elapsed, mode, nifty_prev)
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

    c1, c2, c3, c4 = st.columns(4)
    if ranked:
        c1.metric("Closest match", ranked[0].name, f"score {ranked[0].score:.0f} / 100",
                  delta_color="off")
    else:
        c1.metric("Closest match", "—")
    if last_bar is not None and has_prev(nifty_path):
        c2.metric("Nifty vs previous close", f"{nv.iloc[-1]:+.2f}%",
                  f"{nifty_path.since_open.iloc[-1]:+.2f}% since open")
        c3.metric("Nifty opening gap", f"{nifty_path.gap:+.2f}%",
                  f"{nifty_path.gap * nifty_path.base / 100:+,.0f} pts")
    else:
        c2.metric("Nifty since open", f"{nv.iloc[-1]:+.2f}%" if last_bar is not None else "—")
        c3.metric("Nifty opening gap", "—")
    if last_bar is not None:
        lag = ""
        if live and sel_day == now.date():
            lag = f"{int((now - last_bar) / pd.Timedelta(minutes=1))} min ago"
        c4.metric("Nifty last bar", f"{last_bar:%H:%M} IST", lag, delta_color="off")
    else:
        c4.metric("Nifty last bar", "—")

    if not has_prev(nifty_path):
        st.warning("Nifty's previous close is unavailable, so paths start from the open and "
                   "opening gaps are not scored.")
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
    other_path = session_path(others[other_name], sel_day, elapsed, mode, prevs.get(other_name))
    st.plotly_chart(chart(nifty_path, other_path, other_name), width="stretch",
                    config={"displayModeBar": False})
    base_note = ("Both lines are measured from their previous close, so each starts at its "
                 "opening gap. " if has_prev(nifty_path) else "")
    if mode == OWN_OPEN and other_path.anchor is not None:
        st.caption(f"Nifty 9:15 → {end:%H:%M} IST vs {other_name} "
                   f"{other_path.anchor:%H:%M} → {other_path.clock()[-1]:%H:%M} IST, both from their "
                   f"own open. {base_note}Time is trading time: lunch breaks are skipped so both "
                   "lines run side by side. Tap or hover for % moves and IST time.")
    else:
        st.caption(f"{base_note}Lines break where an index was not trading (lunch breaks, "
                   "closed markets). Nothing is forward-filled.")

    st.session_state["ai_context"] = ai_briefing(
        sel_day, end, elapsed, nifty_path, results,
        {"Nifty 50": nifty_path} | {r.name: session_path(others[r.name], sel_day, elapsed, mode,
                                                          prevs.get(r.name))
                                    for r in (ranked + partial)[:3]}
        | {other_name: other_path},
    )

    # ---------- Tables ----------
    cols = {
        "Index": st.column_config.TextColumn("Index", pinned=True),
        "Score": st.column_config.ProgressColumn("Score", min_value=0, max_value=100, format="%.0f"),
        "Opening gap": st.column_config.TextColumn(help="Open vs previous close"),
        "Since open": st.column_config.TextColumn(help="Up or down since the index opened, in %"),
        "Vs prev close": st.column_config.TextColumn(help="Up or down vs the previous close, in %"),
        "Level gap (pp)": st.column_config.NumberColumn(
            format="%.2f", help="Typical distance between the two lines, in percentage points"),
        "Move size match": st.column_config.NumberColumn(
            format="%.0f%%", help="How alike each interval's move is in size and sign"),
        "Shape": st.column_config.NumberColumn(format="%.2f", help="Path correlation"),
        "Direction": st.column_config.NumberColumn(format="%.0f%%", help="Intervals moving the same way"),
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


def ai_briefing(sel_day, end, elapsed, nifty_path: Path, results, paths: dict[str, Path]) -> str:
    """The dashboard's current state as text for the AI panel."""
    def sampled(p: Path) -> list[tuple[int, str, float]]:
        v = p.values
        if v.empty:
            return []
        keep = [m for m in v.index if m % ai_analysis.SAMPLE_EVERY == 0] + [v.index[-1]]
        clock = pd.Series(p.clock(), index=v.index)
        return [(int(m), f"{clock[m]:%H:%M}", float(v[m])) for m in dict.fromkeys(keep)]

    nv = nifty_path.values
    return ai_analysis.build_context(
        day=f"{sel_day:%a %d %b %Y}",
        mode=MODES[mode],
        window=f"Nifty 9:15 → {end:%H:%M} IST ({fmt_elapsed(elapsed)}), "
               + ("live" if live else "replay"),
        nifty={
            "gap": nifty_path.gap,
            "gap_points": nifty_path.gap * nifty_path.base / 100 if has_prev(nifty_path) else 0.0,
            "since_open": nifty_path.since_open.iloc[-1] if not nv.empty else np.nan,
            "vs_prev": nv.iloc[-1] if not nv.empty and has_prev(nifty_path) else np.nan,
            "last_bar": f"{nifty_path.clock()[-1]:%H:%M}" if not nv.empty else "n/a",
        },
        results=[{
            "name": r.name, "status": r.status, "score": r.score, "gap": r.gap,
            "gap_diff": r.gap_diff, "since_open": r.since_open, "vs_prev": r.now,
            "opens": f"{r.anchor:%H:%M}" if r.anchor is not None else "n/a",
            "coverage": r.coverage, "level_rmse": r.level_rmse, "move_match": r.move_match,
            "shape": r.shape, "direction": r.direction,
        } for r in results],
        paths={name: sampled(p) for name, p in paths.items() if not p.values.empty},
    )


def ai_panel() -> None:
    """On-demand Claude analysis of what the dashboard currently shows."""
    st.subheader("AI analysis")
    context = st.session_state.get("ai_context")
    client = ai_analysis.make_client()
    if client is None:
        st.info("Add an Anthropic API key to turn this on: put `ANTHROPIC_API_KEY = \"sk-ant-...\"` "
                "in `.streamlit/secrets.toml` locally, or under **Manage app → Settings → Secrets** "
                "on Streamlit Cloud.")
        return
    if not context:
        st.caption("The analysis runs once the dashboard has data for this window.")
        return
    question = st.text_input("Ask a question (optional)", key="ai_question",
                             placeholder="e.g. Why is ASX ranked above Hang Seng?")
    if st.button("Analyze with AI", type="primary"):
        try:
            with st.spinner("Claude is reading the dashboard…"):
                stream = ai_analysis.stream_analysis(client, context, question)
                first = next(stream, "")  # hold the spinner until the first words arrive
            answer = st.write_stream(_chain(first, stream))
            history = st.session_state.setdefault("ai_history", [])
            history.insert(0, (f"{pd.Timestamp.now(tz=IST):%H:%M:%S}", question.strip(), answer))
            del history[5:]
            return
        except anthropic.AuthenticationError:
            st.error("The Anthropic API key was rejected. Check `ANTHROPIC_API_KEY`.")
        except anthropic.RateLimitError:
            st.error("Rate limited by the Anthropic API. Try again in a minute.")
        except anthropic.APIStatusError as e:
            st.error(f"Anthropic API error {e.status_code}. Try again shortly.")
        except anthropic.APIConnectionError:
            st.error("Couldn't reach the Anthropic API. Check the network and try again.")
    for when, q, answer in st.session_state.get("ai_history", []):
        with st.container(border=True):
            st.caption(f"{when} IST" + (f" · {q}" if q else ""))
            st.markdown(answer)
    st.caption(f"Each click is one {ai_analysis.MODEL} request (a few US cents). The analysis "
               "only sees the numbers on this page and is not trading advice.")


def _chain(first: str, rest):
    yield first
    yield from rest


st.fragment(run_every=refresh if live else None)(dashboard)()
st.divider()
st.fragment(ai_panel)()
