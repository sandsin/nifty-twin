"""AI commentary on the live Nifty Twin ranking, written by Claude via the Anthropic API.

The dashboard turns its current state into a plain-text briefing (`build_context`) plus the
full minute-by-minute chart data as CSV (`chart_csv`), and `stream_analysis` streams Claude's
read of both. For a Claude.ai chat the CSV goes along as an attachment instead. Calls happen only when the user asks, never
on the auto-refresh, so cost stays at one request per click.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from urllib.parse import quote

import anthropic

MODEL = "claude-opus-5-5"
MAX_TOKENS = 64000
EFFORT = "medium"  # short commentary on a small table; keeps answers quick
SAMPLE_EVERY = 15  # minutes between trajectory points in the briefing
CHART_COLUMNS = "index,minute,ist_time,vs_prev_close_pct,since_open_pct"

SYSTEM = """You are the analyst inside "Nifty Twin", a live dashboard that finds which \
Asian stock index is moving most like India's Nifty 50 today.

How the dashboard measures similarity:
- Every path is the % change from that index's previous official close, so it starts at \
the opening gap.
- In "own open" mode, Nifty's first N trading minutes after 9:15 IST are compared with each \
index's first N trading minutes after its own open (lunch breaks skipped). In "same clock" \
mode both are compared over the same IST minutes.
- The score (0-100) weights: level closeness 35% (how close the two lines stay, gap \
included), move size match 25% (each interval's move alike in size and sign), opening gap \
match 15%, shape 15% (correlation), direction 10%. Indices below the coverage minimum are \
"partial" and not ranked.

The user has picked one index in the dashboard to compare with Nifty (the "selected \
index"). The briefing covers that pair in detail, plus a short ranking of the others for \
context. Chart data, when given, is CSV with one row per index per trading minute for the \
two lines on the chart: index, minute (trading minutes since that index's own open; in \
"same clock" mode, minutes since 9:15 IST), ist_time, vs_prev_close_pct, since_open_pct.

Your job: analyse how the selected index compares with Nifty today, for a trader watching \
Nifty. Cover the opening gaps; when the two moved together and when and how they diverged \
(cite IST times for both indices, using the chart data where given); which score \
components helped or hurt its score and why; how it stands against the top-ranked index; \
and anything that makes the read unreliable (early session, low coverage, low score, \
lunch breaks, stale bars).

Rules:
- Describe moves in % with up/down wording or ▲/▼, not index points. Opening gaps may \
also be given in points.
- Ground every statement in the briefing. Do not invent prices, news or causes; if you \
mention a possible reason, label it as a possibility.
- A score under about 40 means no close twin; say so plainly rather than overselling.
- This is analysis, not advice: never recommend buying, selling or position sizes, and \
do not predict where Nifty will go. Similar paths so far do not imply similar paths later.
- Keep it short: about 150-250 words in Markdown, with a one-line headline, then short \
bullets. If the user asks a specific question, answer that first."""


def api_key() -> str | None:
    """ANTHROPIC_API_KEY from Streamlit secrets, else the environment."""
    try:
        import streamlit as st
        key = st.secrets.get("ANTHROPIC_API_KEY")
        if key:
            return str(key)
    except Exception:  # no secrets file, or not running under Streamlit
        pass
    return os.environ.get("ANTHROPIC_API_KEY") or None


def make_client() -> anthropic.Anthropic | None:
    """A client when credentials are configured, else None."""
    key = api_key()
    if key:
        return anthropic.Anthropic(api_key=key)
    # The SDK builds a client even with no credentials and only fails on the first request,
    # so look for the other sources it reads: an auth token or an `ant auth login` profile.
    has_profile = os.path.isdir(os.path.expanduser("~/.config/anthropic"))
    if os.environ.get("ANTHROPIC_AUTH_TOKEN") or has_profile:
        try:
            return anthropic.Anthropic()
        except Exception:
            return None
    return None


def _fmt(pct: float) -> str:
    if pct is None or pct != pct:  # NaN
        return "n/a"
    return f"{pct:+.2f}%"


def build_context(*, day: str, mode: str, window: str, nifty: dict, focus: dict,
                  ranking: list[dict], paths: dict[str, list[tuple[int, str, float]]]) -> str:
    """Plain-text briefing on Nifty vs the selected index.

    `nifty` has gap, gap_points, since_open, vs_prev, last_bar. `focus` is the selected
    index's result: name, status, rank, ranked, score, the score components, gap, gap_points,
    since_open, vs_prev, opens, compared_to, coverage. `ranking` lists every index's name,
    status and score. `paths` maps an index name to (trading minute, IST time,
    % vs previous close) samples.
    """
    f = focus
    lines = [
        f"Date: {day}",
        f"Alignment mode: {mode}",
        f"Window: {window}",
        "",
        "Nifty 50:",
        f"  opening gap {_fmt(nifty['gap'])} ({nifty['gap_points']:+,.0f} pts)"
        if nifty["gap"] == nifty["gap"] else "  opening gap unknown (no previous close)",
        f"  since open {_fmt(nifty['since_open'])}, vs previous close {_fmt(nifty['vs_prev'])}",
        f"  last bar {nifty['last_bar']} IST",
        "",
        f"Selected index: {f['name']}",
        f"  rank {f['rank']} of {f['ranked']} ranked" if f["rank"] else
        f"  not ranked ({f['status']}: coverage below the minimum)",
        f"  score {f['score']:.0f}/100; components (0-1): level closeness {f['level']:.2f}, "
        f"move size match {f['move_match']:.2f}, opening gap match {f['gap_match']}, "
        f"direction {f['direction']:.2f}; shape (correlation) {f['shape']:+.2f}",
        f"  level gap (RMSE between the lines) {f['level_rmse']:.2f} pp",
        f"  opens {f['opens']} IST; compared up to {f['compared_to']} IST; "
        f"coverage {f['coverage']:.0%}",
        f"  opening gap {_fmt(f['gap'])}"
        + (f" ({f['gap_points']:+,.0f} pts)" if f["gap"] == f["gap"] else "")
        + (f", {f['gap_diff']:+.2f} pp vs Nifty's" if f["gap_diff"] == f["gap_diff"] else ""),
        f"  since open {_fmt(f['since_open'])}, vs previous close {_fmt(f['vs_prev'])}",
        "",
        "Other indices (score 0-100):",
    ]
    for r in ranking:
        if r["name"] == f["name"]:
            continue
        lines.append(f"- {r['name']}: no data" if r["status"] == "no data"
                     else f"- {r['name']} [{r['status']}] {r['score']:.0f}")
    if paths:
        lines += ["", f"Paths, % vs previous close, every {SAMPLE_EVERY} trading minutes "
                      "(minute from own open @ IST time):"]
        for name, pts in paths.items():
            lines.append(f"- {name}: " + ", ".join(f"{m}m@{t} {v:+.2f}%" for m, t, v in pts))
    return "\n".join(lines)


def chart_csv(paths: dict[str, list[tuple[int, str, float, float]]]) -> str:
    """Every chart point as CSV. `paths` maps index name -> (minute, IST time,
    % vs previous close, % since open) for each trading minute."""
    rows = [CHART_COLUMNS]
    for name, pts in paths.items():
        rows += [f"{name},{m},{t},{v:.3f},{so:.3f}" for m, t, v, so in pts]
    return "\n".join(rows) + "\n"


def _request(context: str, question: str | None, chart_data: str | None = None,
             attachment: str | None = None) -> str:
    """The user message: briefing, then the chart data inline or a pointer to the attached
    file, then the question."""
    ask = question.strip() if question else ""
    parts = [f"<briefing>\n{context}\n</briefing>"]
    if chart_data:
        parts.append(f'<chart_data format="csv">\n{chart_data}</chart_data>')
    if attachment:
        parts.append(f"I have attached {attachment}: the full minute-by-minute chart data as CSV "
                     f"({CHART_COLUMNS}). Use it as well as the briefing; if no file is "
                     "attached, work from the briefing alone.")
    parts.append(f"Question: {ask}" if ask
                 else "Analyse how the selected index compares with Nifty 50 today.")
    return "\n\n".join(parts)


def chat_prompt(context: str, question: str | None = None, attachment: str | None = None) -> str:
    """One paste-ready message for a Claude.ai chat: the analyst instructions plus the
    briefing, so a Pro plan gives the same kind of answer without an API key."""
    return (f"<instructions>\n{SYSTEM}\n</instructions>\n\n"
            + _request(context, question, attachment=attachment))


# Shorter instructions for the Claude.ai link, which has to fit in a URL.
SHORT_SYSTEM = """You are the analyst for "Nifty Twin", a dashboard comparing Asian indices \
with India's Nifty 50 today. Paths are % from each index's previous close (starting at the \
opening gap), compared from each index's own open. Score 0-100 = level closeness 35%, move \
size match 25%, opening gap match 15%, shape 15%, direction 10%.
Analyse the selected index against Nifty using only this data: opening gaps; when they moved \
together and when they diverged (cite IST times); which score components helped or hurt; \
how it compares with the top-ranked index (under ~40 means no close twin); and what makes the \
read unreliable. Use % with up/down, not points. Not advice: no trade calls or Nifty \
predictions. About 150-250 words: a headline, then short bullets. Answer any question first.
An attached CSV, if any, has every chart point for the two indices (one row per index per \
trading minute); use it for detail."""

CLAUDE_AI_NEW = "https://claude.ai/new?q="
URL_LIMIT = 8000  # encoded characters; much longer URLs get rejected by some servers


def claude_ai_url(context: str, question: str | None = None,
                  attachment: str | None = None) -> str:
    """Link that opens a Claude.ai chat with the analysis prompt pre-filled.

    Uses the short instructions; if that is still too long, drops the sampled paths and
    then the lowest rows of the other-indices list.
    """
    def link(ctx: str) -> str:
        text = f"{SHORT_SYSTEM}\n\n{_request(ctx, question, attachment=attachment)}"
        return CLAUDE_AI_NEW + quote(text, safe="")

    url = link(context)
    if len(url) > URL_LIMIT:
        context = context.split("\n\nPaths,")[0]
        url = link(context)
    if len(url) > URL_LIMIT:
        head, _, rows = context.partition("Other indices")
        lines = rows.split("\n")
        while len(url) > URL_LIMIT and len(lines) > 1:
            lines.pop()
            url = link(head + "Other indices" + "\n".join(lines))
    return url


def stream_analysis(client: anthropic.Anthropic, context: str, question: str | None = None,
                    chart_data: str | None = None) -> Iterator[str]:
    """Yield Claude's analysis as it streams. API errors propagate to the caller."""
    prompt = _request(context, question, chart_data=chart_data)
    with client.beta.messages.stream(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        output_config={"effort": EFFORT},
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",  # re-run on Anthropic's recommended model if a request is declined
        system=SYSTEM,
        messages=[{"role": "user", "content": prompt}],
    ) as stream:
        yield from stream.text_stream
        final = stream.get_final_message()
    if final.stop_reason == "refusal":
        yield "\n\n_The model declined to answer this one. Try rephrasing the question._"
    elif final.stop_reason == "max_tokens":
        yield "\n\n_(Answer cut off.)_"
