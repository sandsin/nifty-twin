"""Tests for ai_analysis.py without calling the real API. Run: python test_ai_analysis.py"""

import json
import threading
from http.server import BaseHTTPRequestHandler
from socketserver import TCPServer  # HTTPServer does a slow hostname lookup on bind
from types import SimpleNamespace

import anthropic

import ai_analysis

NIFTY = {"gap": -0.34, "gap_points": -77, "since_open": 0.35, "vs_prev": 0.01, "last_bar": "10:19"}
ROW = {"name": "Hang Seng", "status": "ranked", "score": 29.4, "gap": 0.43, "gap_diff": 0.77,
       "since_open": -0.6, "vs_prev": -0.17, "opens": "07:00", "coverage": 1.0, "level_rmse": 0.37,
       "move_match": 0.15, "shape": -0.88, "direction": 0.4}


def test_build_context():
    text = ai_analysis.build_context(
        day="Tue 27 Oct 2026", mode="Each index's own open", window="Nifty 9:15 → 10:19 IST",
        nifty=NIFTY, results=[ROW, {"name": "KOSPI", "status": "no data"}],
        paths={"Nifty 50": [(0, "09:15", -0.34), (15, "09:30", -0.1)]})
    assert "opening gap -0.34% (-77 pts)" in text
    assert "Hang Seng [ranked] score 29" in text and "since open -0.60%" in text
    assert "KOSPI: no data" in text
    assert "0m@09:15 -0.34%, 15m@09:30 -0.10%" in text


class FakeStream:
    def __init__(self, chunks, stop_reason):
        self.text_stream = iter(chunks)
        self._final = SimpleNamespace(stop_reason=stop_reason)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def get_final_message(self):
        return self._final


def fake_client(chunks, stop_reason="end_turn", calls=None):
    def stream(**kwargs):
        if calls is not None:
            calls.append(kwargs)
        return FakeStream(chunks, stop_reason)
    return SimpleNamespace(beta=SimpleNamespace(messages=SimpleNamespace(stream=stream)))


def test_stream_and_request_shape():
    calls = []
    out = "".join(ai_analysis.stream_analysis(fake_client(["Hang ", "Seng"], calls=calls),
                                              "CTX", "Why?"))
    assert out == "Hang Seng"
    kw = calls[0]
    assert kw["model"] == "claude-opus-5-5" and kw["fallbacks"] == "default"
    assert kw["betas"] == ["server-side-fallback-2026-07-01"]
    assert "<briefing>\nCTX\n</briefing>" in kw["messages"][0]["content"]
    assert kw["messages"][0]["content"].endswith("Question: Why?")


def test_refusal_and_cutoff_notes():
    assert "declined" in "".join(ai_analysis.stream_analysis(fake_client([], "refusal"), "C"))
    assert "cut off" in "".join(ai_analysis.stream_analysis(fake_client(["x"], "max_tokens"), "C"))


def test_sdk_sends_expected_request():
    """Run the real SDK against a local stub server to check what goes over the wire."""
    seen = {}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            seen["path"] = self.path
            seen["beta"] = self.headers.get("anthropic-beta")
            seen["body"] = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            self.send_response(401)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"type":"error","error":{"type":"authentication_error","message":"x"}}')

        def log_message(self, *args):
            pass

    server = TCPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.handle_request, daemon=True).start()
    client = anthropic.Anthropic(api_key="test", base_url=f"http://127.0.0.1:{server.server_address[1]}",
                                 max_retries=0)
    try:
        "".join(ai_analysis.stream_analysis(client, "CTX"))
        raise AssertionError("expected an authentication error")
    except anthropic.AuthenticationError:
        pass
    server.server_close()
    body = seen["body"]
    assert seen["path"].startswith("/v1/messages")
    assert "server-side-fallback-2026-07-01" in seen["beta"]
    assert body["model"] == "claude-opus-5-5" and body["fallbacks"] == "default"
    assert body["stream"] is True and body["output_config"] == {"effort": "medium"}
    assert "thinking" not in body and "temperature" not in body


def test_chat_prompt():
    text = ai_analysis.chat_prompt("CTX", "Why ASX?")
    assert text.startswith("<instructions>\n") and ai_analysis.SYSTEM in text
    assert "<briefing>\nCTX\n</briefing>" in text and text.endswith("Question: Why ASX?")
    assert ai_analysis.chat_prompt("CTX").endswith("Give your analysis of today's match.")


def test_claude_ai_url():
    from urllib.parse import parse_qs, urlparse
    ctx = ai_analysis.build_context(
        day="Tue", mode="own open", window="w", nifty=NIFTY, results=[ROW],
        paths={"Nifty 50": [(0, "09:15", -0.34)]})
    url = ai_analysis.claude_ai_url(ctx, "Why ASX?")
    assert url.startswith("https://claude.ai/new?q=")
    text = parse_qs(urlparse(url).query)["q"][0]  # round-trips exactly
    assert text.startswith(ai_analysis.SHORT_SYSTEM) and ctx in text and text.endswith("Why ASX?")
    # A briefing too long for a URL gets its paths, then ranking rows, dropped.
    many = [dict(ROW, name=f"Index {i}") for i in range(40)]
    long_paths = {f"P{i}": [(m, "10:00", 0.1) for m in range(0, 300, 15)] for i in range(10)}
    big = ai_analysis.build_context(day="Tue", mode="m", window="w", nifty=NIFTY,
                                    results=many, paths=long_paths)
    url = ai_analysis.claude_ai_url(big)
    text = parse_qs(urlparse(url).query)["q"][0]
    assert len(url) <= ai_analysis.URL_LIMIT and "Paths," not in text and "Index 0 " in text


def test_chart_csv_and_where_it_goes():
    csv = ai_analysis.chart_csv({"Nifty 50": [(0, "09:15", -0.34, 0.0), (1, "09:16", -0.3, 0.04)],
                                 "Hang Seng": [(0, "07:00", 0.43, 0.0)]})
    assert csv.splitlines() == [ai_analysis.CHART_COLUMNS, "Nifty 50,0,09:15,-0.340,0.000",
                                "Nifty 50,1,09:16,-0.300,0.040", "Hang Seng,0,07:00,0.430,0.000"]
    calls = []
    "".join(ai_analysis.stream_analysis(fake_client(["x"], calls=calls), "CTX", None, csv))
    assert f'<chart_data format="csv">\n{csv}</chart_data>' in calls[0]["messages"][0]["content"]
    for text in (ai_analysis.chat_prompt("CTX", None, "f.csv"),):
        assert "I have attached f.csv" in text and "<chart_data" not in text
    from urllib.parse import parse_qs, urlparse
    text = parse_qs(urlparse(ai_analysis.claude_ai_url("CTX", "Q?", "f.csv")).query)["q"][0]
    assert "I have attached f.csv" in text and text.endswith("Question: Q?")


def test_no_credentials_means_no_client():
    import os
    saved = {k: os.environ.pop(k, None) for k in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN")}
    try:
        if not os.path.isdir(os.path.expanduser("~/.config/anthropic")):
            assert ai_analysis.make_client() is None
        os.environ["ANTHROPIC_API_KEY"] = "sk-ant-test"
        assert ai_analysis.make_client() is not None
    finally:
        os.environ.pop("ANTHROPIC_API_KEY", None)
        os.environ.update({k: v for k, v in saved.items() if v})


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
    print("OK")
