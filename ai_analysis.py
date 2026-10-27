"""AI commentary on the live Nifty Twin ranking, written by Claude via the Anthropic API.

The dashboard turns its current state into a plain-text briefing (`build_context`), and
`stream_analysis` streams Claude's read of it. Calls happen only when the user asks, never
on the auto-refresh, so cost stays at one request per click.
"""

from __future__ import annotations

import os
from collections.abc import Iterator

import anthropic

MODEL = "claude-opus-5-5"
MAX_TOKENS = 64000
EFFORT = "medium"  # short commentary on a small table; keeps answers quick
SAMPLE_EVERY = 15  # minutes between trajectory points sent to the model

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

Your job: read the briefing and explain, for a trader watching Nifty, which index is \
tracking Nifty most closely and why, what the numbers say about how close the match \
really is, where the leaders diverge from Nifty, and anything that makes today's read \
unreliable (early session, low coverage, low scores, stale bars).

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


def build_context(*, day: str, mode: str, window: str, nifty: dict, results: list[dict],
                  paths: dict[str, list[tuple[int, str, float]]]) -> str:
    """Plain-text briefing of the dashboard's current state.

    `nifty` has gap, gap_points, since_open, vs_prev, last_bar. Each result has name, status,
    score, the score components, gap, since_open, vs_prev, opens, coverage. `paths` maps an
    index name to (trading minute, IST time, % vs previous close) samples.
    """
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
        "Ranking (score 0-100; components 0-1 except where noted):",
    ]
    for r in results:
        if r["status"] == "no data":
            lines.append(f"- {r['name']}: no data in this window")
            continue
        lines.append(
            f"- {r['name']} [{r['status']}] score {r['score']:.0f} | "
            f"opens {r['opens']} IST, coverage {r['coverage']:.0%} | "
            f"gap {_fmt(r['gap'])} (vs Nifty {_fmt(r['gap_diff'])} pp), "
            f"since open {_fmt(r['since_open'])}, vs prev close {_fmt(r['vs_prev'])} | "
            f"level gap {r['level_rmse']:.2f} pp, move match {r['move_match']:.2f}, "
            f"shape {r['shape']:+.2f}, direction {r['direction']:.0%}"
        )
    if paths:
        lines += ["", f"Paths, % vs previous close, every {SAMPLE_EVERY} trading minutes "
                      "(minute from own open @ IST time):"]
        for name, pts in paths.items():
            lines.append(f"- {name}: " + ", ".join(f"{m}m@{t} {v:+.2f}%" for m, t, v in pts))
    return "\n".join(lines)


def _request(context: str, question: str | None) -> str:
    ask = question.strip() if question else ""
    return (f"<briefing>\n{context}\n</briefing>\n\n"
            + (f"Question: {ask}" if ask else "Give your analysis of today's match."))


def chat_prompt(context: str, question: str | None = None) -> str:
    """One paste-ready message for a Claude.ai chat: the analyst instructions plus the
    briefing, so a Pro plan gives the same kind of answer without an API key."""
    return f"<instructions>\n{SYSTEM}\n</instructions>\n\n{_request(context, question)}"


def stream_analysis(client: anthropic.Anthropic, context: str,
                    question: str | None = None) -> Iterator[str]:
    """Yield Claude's analysis as it streams. API errors propagate to the caller."""
    prompt = _request(context, question)
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
