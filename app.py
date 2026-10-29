"""Nifty Twin: which Asian index is moving most like Nifty 50 since its 9:15 IST open."""

from __future__ import annotations

from datetime import datetime, time

import anthropic
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import ai_analysis
from market_data import LiveData, prev_close, trading_days
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


@st.cache_resource(show_spinner=False)
def live_data() -> LiveData:
    """One shared copy of every index's data for all viewers, refreshed in the background."""
    return LiveData([NIFTY] + list(CANDIDATES.values()))


def load() -> tuple[dict[str, pd.Series], dict[str, pd.Series]]:
    return live_data().get()


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

    with st.spinner("Loading 1-minute data…"):
        closes, _ = load()
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


def _level(x: float) -> str:
    return f"{x:,.2f}" if np.isfinite(x) else "—"


def _facts(p: Path) -> dict[str, str]:
    """One index's numbers for the side-by-side table."""
    if p.values.empty:
        return {}
    v, prev = p.values, has_prev(p)
    return {
        "Window (IST)": f"{p.clock()[0]:%H:%M} → {p.clock()[-1]:%H:%M}",
        "Previous close": _level(p.base) if prev else "—",
        "Open": _level(p.open_level),
        "Opening gap": fmt_move(p.gap, p.gap * p.base / 100) if prev else "—",
        "Latest": _level(p.levels.iloc[-1]),
        "Since open": fmt_pct(p.since_open.iloc[-1]),
        "Vs prev close": fmt_pct(v.iloc[-1]) if prev else "—",
        "High (vs prev close)": fmt_pct(v.max()) if prev else "—",
        "Low (vs prev close)": fmt_pct(v.min()) if prev else "—",
    }


def side_by_side(nifty: Path, other: Path, other_name: str, score, rank_no) -> pd.DataFrame:
    """Nifty's numbers with the compared index's in the next column."""
    a, b = _facts(nifty), _facts(other)
    rows = [{"": k, "Nifty 50": a.get(k, "—"), other_name: b.get(k, "—")} for k in a or b]
    if score.status != "no data" and np.isfinite(score.score):
        where = f"rank {rank_no}" if rank_no else "not ranked"
        rows.append({"": "Match score", "Nifty 50": "",
                     other_name: f"{score.score:.0f} / 100 ({where})"})
    return pd.DataFrame(rows)


def minute_table(nifty: Path, other: Path, other_name: str) -> pd.DataFrame:
    """Both paths minute by minute on the shared elapsed axis, latest first."""
    def cols(p: Path, name: str) -> pd.DataFrame:
        return pd.DataFrame({f"{name} (IST)": p.clock().strftime("%H:%M"), name: p.values.to_numpy()},
                            index=p.values.index)
    both = cols(nifty, "Nifty 50").join(cols(other, other_name), how="outer").sort_index(ascending=False)
    diff = both[other_name] - both["Nifty 50"]
    out = pd.DataFrame({"Minute": [fmt_elapsed(int(m)) for m in both.index]})
    for name in ("Nifty 50", other_name):
        out[f"{name} (IST)"] = both[f"{name} (IST)"].fillna("—").to_numpy()
        out[name] = [fmt_pct(x) for x in both[name]]
    out["Difference"] = [f"{d:+.2f} pp" if np.isfinite(d) else "—" for d in diff]
    return out


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
    data, daily = load()
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

    # ---------- Twin data next to Nifty's ----------
    focus = next(r for r in results if r.name == other_name)
    rank_no = next((i for i, r in enumerate(ranked, start=1) if r.name == other_name), None)
    st.markdown(f"**Nifty 50 vs {other_name}**")
    st.dataframe(side_by_side(nifty_path, other_path, other_name, focus, rank_no),
                 hide_index=True, width="stretch",
                 column_config={"": st.column_config.TextColumn(pinned=True)})
    with st.expander(f"Minute by minute: Nifty 50 next to {other_name}"):
        st.dataframe(minute_table(nifty_path, other_path, other_name), hide_index=True,
                     width="stretch", height=300)
        st.caption(("Latest first, lined up by trading time since each index's open. "
                    if mode == OWN_OPEN else "Latest first, by time since 9:15 IST. ")
                   + ("Moves are from each index's previous close. " if has_prev(nifty_path)
                      else "Moves are from the open. ")
                   + f"Difference is {other_name} minus Nifty, in percentage points.")

    # ---------- AI analysis ----------
    st.divider()
    ai_panel(sel_day, end, elapsed, nifty_path, results, ranked, other_name, other_path,
             others, prevs)
    st.divider()

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


def chart_points(p: Path) -> list[tuple[int, str, float, float]]:
    """Every chart point: (minute, IST time, % vs previous close, % since open)."""
    return list(zip(p.values.index.astype(int), p.clock().strftime("%H:%M"),
                    p.values.to_numpy(), p.since_open.to_numpy()))


def _sampled(p: Path) -> list[tuple[int, str, float]]:
    """Path points every SAMPLE_EVERY trading minutes, plus the latest."""
    v = p.values
    if v.empty:
        return []
    keep = [m for m in v.index if m % ai_analysis.SAMPLE_EVERY == 0] + [v.index[-1]]
    clock = pd.Series(p.clock(), index=v.index)
    return [(int(m), f"{clock[m]:%H:%M}", float(v[m])) for m in dict.fromkeys(keep)]


def _briefing_header(sel_day, end, elapsed, nifty_path: Path) -> dict:
    nv = nifty_path.values
    return dict(
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
    )


def _result_fields(r) -> dict:
    """One index's numbers, as the AI briefings expect them."""
    parts = r.components
    return {
        "name": r.name, "status": r.status, "score": r.score, "level": parts.get("level", np.nan),
        "move_match": r.move_match,
        "gap_match": f"{parts['gap']:.2f}" if "gap" in parts else "n/a (no previous close)",
        "direction": r.direction, "shape": r.shape, "level_rmse": r.level_rmse,
        "opens": f"{r.anchor:%H:%M}" if r.anchor is not None else "n/a",
        "compared_to": f"{r.last_bar:%H:%M}" if r.last_bar is not None else "n/a",
        "coverage": r.coverage, "gap": r.gap, "gap_points": r.gap_points,
        "gap_diff": r.gap_diff, "since_open": r.since_open, "vs_prev": r.now,
    }


def ai_panel(sel_day, end, elapsed, nifty_path: Path, results, ranked, other_name: str,
             other_path: Path, others: dict, prevs: dict) -> None:
    """On-demand Claude analysis of Nifty vs all the Asian indices, or vs the one selected
    in the dropdown."""
    st.subheader("AI analysis")
    scope = st.radio(
        "Analyse", [ai_analysis.ALL, ai_analysis.PAIR], key="ai_scope", horizontal=True,
        format_func=lambda s: ("Nifty vs all Asian indices" if s == ai_analysis.ALL
                               else f"Nifty vs {other_name} (the index selected above)"))
    header = _briefing_header(sel_day, end, elapsed, nifty_path)
    stamp = f"nifty_twin_{sel_day:%Y-%m-%d}_{end:%H%M}"
    if scope == ai_analysis.ALL:
        label = "all Asian indices"
        paths = {"Nifty 50": nifty_path} | {
            r.name: session_path(others[r.name], sel_day, elapsed, mode, prevs.get(r.name))
            for r in results if r.status != "no data"}
        top = [r.name for r in ranked[:3]]
        context = ai_analysis.build_all_context(
            **header, results=[_result_fields(r) for r in results],
            paths={n: _sampled(p) for n, p in paths.items() if n == "Nifty 50" or n in top})
        file_name = f"{stamp}_all_asia.csv"
        placeholder = "e.g. Which markets are moving with Nifty, and which against it?"
    else:
        label = other_name
        focus = next(r for r in results if r.name == other_name)
        paths = {"Nifty 50": nifty_path, other_name: other_path}
        rank_no = next((i for i, r in enumerate(ranked, start=1) if r.name == other_name), None)
        context = ai_analysis.build_context(
            **header, focus=_result_fields(focus) | {"rank": rank_no, "ranked": len(ranked)},
            ranking=[{"name": r.name, "status": r.status, "score": r.score} for r in results],
            paths={n: _sampled(p) for n, p in paths.items() if not p.values.empty})
        file_name = f"{stamp}_nifty_vs_{other_name.lower().replace(' ', '_')}.csv"
        placeholder = f"e.g. When did {other_name} split from Nifty?"
        st.caption("Follows the **Compare Nifty with** dropdown. Pick another index to analyse it.")
    csv = ai_analysis.chart_csv({n: chart_points(p) for n, p in paths.items()
                                 if not p.values.empty})

    question = st.text_input("Ask a question (optional)", key="ai_question",
                             placeholder=placeholder)
    client = ai_analysis.make_client()
    if client is None:
        claude_ai_prompt(context, question, csv, file_name, scope)
        st.caption("Have an Anthropic API key? Add `ANTHROPIC_API_KEY = \"sk-ant-...\"` to "
                   "`.streamlit/secrets.toml`, or under **Manage app → Settings → Secrets** on "
                   "Streamlit Cloud, to get answers right here instead.")
        return
    with st.expander("Or use your Claude.ai plan"):
        claude_ai_prompt(context, question, csv, file_name, scope)
    if st.button("Analyze with AI", type="primary"):
        try:
            with st.spinner(f"Claude is comparing {label} with Nifty…"):
                stream = ai_analysis.stream_analysis(client, context, question, csv, scope)
                first = next(stream, "")  # hold the spinner until the first words arrive
            answer = st.write_stream(_chain(first, stream))
            history = st.session_state.setdefault("ai_history", [])
            history.insert(0, (f"{pd.Timestamp.now(tz=IST):%H:%M:%S}", label, question.strip(),
                               answer))
            del history[5:]
        except anthropic.AuthenticationError:
            st.error("The Anthropic API key was rejected. Check `ANTHROPIC_API_KEY`.")
        except anthropic.RateLimitError:
            st.error("Rate limited by the Anthropic API. Try again in a minute.")
        except anthropic.APIStatusError as e:
            st.error(f"Anthropic API error {e.status_code}. Try again shortly.")
        except anthropic.APIConnectionError:
            st.error("Couldn't reach the Anthropic API. Check the network and try again.")
    for when, what, q, answer in st.session_state.get("ai_history", []):
        with st.container(border=True):
            st.caption(f"{when} IST · Nifty vs {what}" + (f" · {q}" if q else ""))
            st.markdown(answer)
    cost = ("about 10-25 US cents, more late in the day" if scope == ai_analysis.ALL
            else "a few US cents")
    st.caption(f"Each click is one {ai_analysis.MODEL} request with the full chart data ({cost}). "
               "The analysis only sees the numbers on this page and is not trading advice.")


def claude_ai_prompt(context: str, question: str, csv: str, file_name: str, scope: str) -> None:
    """Send the analysis to a Claude.ai chat, so a Pro plan can run it without an API key.

    The full chart data is too big for a link, so it goes as a CSV the user attaches."""
    attachment = file_name if csv else None
    what = ("every Asian index and Nifty" if scope == ai_analysis.ALL
            else "both lines on the chart")
    st.markdown("**Analyze with your Claude.ai plan.** No API key needed.\n\n"
                f"1. **Download the chart data**: every minute of {what}.\n"
                "2. **Ask Claude.ai**: opens a new chat with the instructions, this page's "
                "numbers and your question filled in.\n"
                "3. **Attach the downloaded file** in that chat (paperclip or drag it in), "
                "then press Enter.")
    c1, c2 = st.columns(2)
    c1.download_button("1 · Download chart data", csv, file_name=file_name, mime="text/csv",
                       on_click="ignore", width="stretch", disabled=not csv)
    c2.link_button("2 · Ask Claude.ai",
                   ai_analysis.claude_ai_url(context, question, attachment, scope),
                   type="primary", width="stretch")
    rows = max(csv.count("\n") - 1, 0)
    st.caption(f"Chart data: {rows:,} rows ({len(csv.encode()) / 1024:,.0f} KB), updated with "
               "the page. Type a question above to include it.")
    with st.expander("Message not filled in? Copy and paste instead"):
        st.code(ai_analysis.chat_prompt(context, question, attachment, scope), language=None,
                wrap_lines=True, height=240)
        st.link_button("Open Claude.ai", "https://claude.ai/new")


def _chain(first: str, rest):
    yield first
    yield from rest


st.fragment(run_every=refresh if live else None)(dashboard)()
