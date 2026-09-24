"""
test_app.py  --  FastAPI app integration tests (no AWS).

All tests use the `fake_env` fixture (DEMO_FAKE=1).
Ingestion-path tests also set DEMO_FAKE_READY=0.
"""

from __future__ import annotations

import json

from fastapi.testclient import TestClient

import agentcore_demo.app as _app_module
from agentcore_demo.app import (
    _backend,  # noqa: F401 -- to reset singleton
    app,
)


def _reset_backend() -> None:
    """Clear the module-level singleton so each test gets a fresh backend."""
    _app_module._backend = None


def _collect(ws, stop_after: int | None = None) -> list[dict]:
    events: list[dict] = []
    while True:
        ev = json.loads(ws.receive_text())
        events.append(ev)
        if ev["type"] == "done":
            break
        if stop_after is not None and len(events) >= stop_after:
            break
    return events


def _collect_ingest(ws) -> list[dict]:
    events: list[dict] = []
    while True:
        ev = json.loads(ws.receive_text())
        events.append(ev)
        if ev["type"] == "kb_ready":
            break
    return events


# ── HTTP routes ───────────────────────────────────────────────────────────────


def test_get_index_returns_html(fake_env):
    _reset_backend()
    with TestClient(app) as client:
        resp = client.get("/")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]
    assert "Inside the Lines" in resp.text


def test_get_static_index_html(fake_env):
    _reset_backend()
    with TestClient(app) as client:
        resp = client.get("/static/index.html")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]


# ── /api/kb-status ────────────────────────────────────────────────────────────


def test_kb_status_ready_when_fake_ready(fake_env):
    _reset_backend()
    with TestClient(app) as client:
        resp = client.get("/api/kb-status")
    assert resp.status_code == 200
    assert resp.json()["ready"] is True


def test_kb_status_not_ready_when_fake_not_ready(fake_env, monkeypatch):
    monkeypatch.setenv("DEMO_FAKE_READY", "0")
    _reset_backend()
    with TestClient(app) as client:
        resp = client.get("/api/kb-status")
    assert resp.status_code == 200
    assert resp.json()["ready"] is False


# ── /ingest WebSocket ─────────────────────────────────────────────────────────


def test_ingest_ws_streams_progress_then_kb_ready(fake_env, monkeypatch):
    monkeypatch.setenv("DEMO_FAKE_READY", "0")
    _reset_backend()
    with TestClient(app) as client, client.websocket_connect("/ingest") as ws:
        events = _collect_ingest(ws)

    types = [e["type"] for e in events]
    assert "ingesting" in types
    assert types[-1] == "kb_ready"


def test_ingest_ws_kb_ready_has_cost_fields(fake_env, monkeypatch):
    monkeypatch.setenv("DEMO_FAKE_READY", "0")
    _reset_backend()
    with TestClient(app) as client, client.websocket_connect("/ingest") as ws:
        events = _collect_ingest(ws)

    kb_ready = next(e for e in events if e["type"] == "kb_ready")
    assert "ingestion_usd" in kb_ready
    assert "storage_usd_per_month" in kb_ready


def test_ingest_progress_events_monotonically_increasing(fake_env, monkeypatch):
    monkeypatch.setenv("DEMO_FAKE_READY", "0")
    _reset_backend()
    with TestClient(app) as client, client.websocket_connect("/ingest") as ws:
        events = _collect_ingest(ws)

    progress = [e["indexed"] for e in events if e["type"] == "ingesting"]
    assert progress, "expected at least one ingesting event"
    for a, b in zip(progress, progress[1:], strict=False):
        assert b >= a, f"progress went backwards: {a} -> {b}"


# ── /ws ───────────────────────────────────────────────────────────────────────


def test_ws_streams_complete_run_ending_in_done(fake_env):
    _reset_backend()
    with TestClient(app) as client, client.websocket_connect("/ws") as ws:
        events = _collect(ws)

    types = [e["type"] for e in events]
    assert types[-1] == "done"
    assert "receipt" in types
    assert types.count("question") == 5  # five beats since 2026-09-22


def test_ws_receipt_total_matches_row_sum(fake_env):
    _reset_backend()
    with TestClient(app) as client, client.websocket_connect("/ws") as ws:
        events = _collect(ws)

    receipt = next(e for e in events if e["type"] == "receipt")
    assert receipt["total"] == round(sum(r["usd"] for r in receipt["rows"]), 6)


def test_ws_includes_chart_event(fake_env):
    _reset_backend()
    with TestClient(app) as client, client.websocket_connect("/ws") as ws:
        events = _collect(ws)

    assert any(e["type"] == "chart" for e in events)


# ── event ORDER ───────────────────────────────────────────────────────────────


def test_ws_question_precedes_its_model_events(fake_env):
    _reset_backend()
    with TestClient(app) as client, client.websocket_connect("/ws") as ws:
        events = _collect(ws)

    blocks: list[list[dict]] = []
    current: list[dict] = []
    in_questions = False
    for ev in events:
        if ev["type"] == "question":
            if current:
                blocks.append(current)
            current = [ev]
            in_questions = True
        elif in_questions:
            current.append(ev)
    if current:
        blocks.append(current)

    # Five beats, Q1 (plain opener) through Q5 (Cedar policy demo).  The Cedar
    # beat was once excluded by default; the plain opener came later.
    assert len(blocks) == 5
    for block in blocks:
        assert block[0]["type"] == "question"
        model_indices = [i for i, e in enumerate(block) if e["type"] == "model"]
        assert all(i > 0 for i in model_indices)


def test_ws_model_start_precedes_done(fake_env):
    _reset_backend()
    with TestClient(app) as client, client.websocket_connect("/ws") as ws:
        events = _collect(ws)

    seen_start: set[str] = set()
    for ev in (e for e in events if e["type"] == "model"):
        key = f"{ev['tier']}:{ev['label']}"
        if ev["state"] == "start":
            seen_start.add(key)
        elif ev["state"] == "done":
            assert key in seen_start


def test_ws_receipt_before_done(fake_env):
    _reset_backend()
    with TestClient(app) as client, client.websocket_connect("/ws") as ws:
        events = _collect(ws)

    types = [e["type"] for e in events]
    done_idx = types.index("done")
    assert types[done_idx - 1] == "receipt"


def test_ws_done_is_last(fake_env):
    _reset_backend()
    with TestClient(app) as client, client.websocket_connect("/ws") as ws:
        events = _collect(ws)

    assert events[-1]["type"] == "done"
    assert [e for e in events if e["type"] == "done"] == [events[-1]]


def test_ws_client_disconnect_mid_run_does_not_raise(fake_env):
    _reset_backend()
    with TestClient(app) as client, client.websocket_connect("/ws") as ws:
        _collect(ws, stop_after=4)


# ── /api/questions + /ws?q=N ─────────────────────────────────────────────────


def test_get_questions_returns_five_texts(fake_env):
    _reset_backend()
    with TestClient(app) as client:
        resp = client.get("/api/questions")
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["questions"]) == 5
    for text in body["questions"]:
        assert isinstance(text, str) and len(text) > 10


def test_ws_single_question_runs_only_that_question(fake_env):
    _reset_backend()
    # q=3 is the chart beat (code generation + Code Interpreter).  It was q=2
    # until a plain opener was inserted ahead of the guardrail beat on 2026-09-22.
    with TestClient(app) as client, client.websocket_connect("/ws?q=3") as ws:
        events = _collect(ws)

    question_events = [e for e in events if e["type"] == "question"]
    assert len(question_events) == 1
    assert question_events[0]["n"] == 3
    assert any(e["type"] == "chart" for e in events)
    assert events[-1]["type"] == "done"


def test_ws_freeform_text_emits_route_and_done(fake_env):
    """?text=... triggers run_freeform: emits a route event then ends with done."""
    _reset_backend()
    question = "What is the established role of PCSK9?"
    with TestClient(app) as client, client.websocket_connect(f"/ws?text={question}") as ws:
        events = _collect(ws)

    assert any(e["type"] == "route" for e in events)
    assert events[-1]["type"] == "done"
    # No setup_cost: freeform is a single question, not a full run
    assert not any(e["type"] == "setup_cost" for e in events)
