"""
app.py  --  FastAPI web application for the Inside the Lines demo.

This is the web layer.  It serves the static page, manages the singleton
backend, and bridges the browser (via WebSocket) to the agent.

Endpoints:
  GET /                 serves static/index.html (the Alpine.js chat page)
  GET /static/*         serves static assets (CSS, JS, images)
  GET /api/kb-status    {"ready": bool} -- does the KB have indexed documents?
  GET /api/kb-costs     KB panel costs: ingestion, storage, vector/corpus stats
  GET /api/questions    the four locked question texts (used by the teletype UI)
  GET /corpus/{pmcid}   serve a local corpus paper as a readable HTML page
  WS  /ingest           run ingestion; stream progress; send kb_ready when done
  WS  /ws               run the agent; forward every event as JSON

Startup flow:
  1. Browser loads index.html.
  2. Page calls GET /api/kb-status.
  3a. If ready:  page shows the question prompt immediately.
  3b. If not ready: page connects to WS /ingest, shows a progress bar,
      waits for the kb_ready event, then shows the question prompt.
  4. User clicks a question chip or types a free-form question.
  5. Page connects to WS /ws?q=N (or WS /ws?text=...).
  6. Agent events flow over the WebSocket; Alpine.js renders them live.

Backend selection (controlled by environment variables):
  DEMO_FAKE=1           use FakeBackend from fakes.py (no AWS, no config.py)
                        Good for: UI rehearsal, CI tests, demo without credentials.
  DEMO_FAKE_READY=0     start with KB not ready (only with DEMO_FAKE=1)
                        Good for: rehearsing the ingestion progress view.
  (neither set)         build AwsBackend from config.py and live pricing rates.

Singleton pattern:
  _backend is created once per process on the first request and reused for
  all subsequent requests.  This avoids re-fetching pricing rates (a few
  seconds) on every WebSocket connection.  A threading.Lock ensures that
  two concurrent connections don't both try to build the backend at once.

Run with:
  python -m agentcore_demo.app        (opens browser at http://127.0.0.1:8000)
  make demo                           (same thing via Makefile)
  make demo-fake                      (DEMO_FAKE=1 version)
"""

from __future__ import annotations

import asyncio
import contextlib
import importlib.util
import json
import os
import threading
import webbrowser
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from agentcore_demo.agent import Agent
from agentcore_demo.backend import Backend
from agentcore_demo.cost import CostMeter

STATIC_DIR = Path(__file__).parent / "static"

# ── auto-open browser on startup ─────────────────────────────────────────────
# _launch_config is populated by __main__ before uvicorn starts.
_launch_config: dict = {}


@asynccontextmanager
async def _lifespan(app: FastAPI):  # noqa: ARG001
    # Open the browser once on startup.  The 0.5s delay lets uvicorn finish
    # binding the port before the browser tries to connect.
    url = _launch_config.get("browser_url")
    if url:
        await asyncio.sleep(0.5)
        await asyncio.to_thread(webbrowser.open, url)
    yield


app = FastAPI(lifespan=_lifespan)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


# ── singleton backend ─────────────────────────────────────────────────────────
# One Backend instance per process.  Initialized lazily on first use.
# Double-checked locking pattern: check without lock first (fast path),
# then acquire lock and check again (safe path).
_backend: Backend | None = None
_backend_lock = threading.Lock()


def _get_backend() -> Backend:
    """Return the process-global backend, building it on first call."""
    global _backend
    if _backend is None:
        with _backend_lock:
            if _backend is None:
                _backend = _build_backend()
    return _backend


def _build_backend() -> Backend:
    """Construct and return the appropriate backend for this environment.

    If DEMO_FAKE=1, returns a FakeBackend (no AWS, no config.py needed).
    Otherwise, imports config.py, fetches live pricing rates, and builds
    an AwsBackend.

    Raises:
        RuntimeError: if config.py is not found and DEMO_FAKE is not set.
    """
    if os.environ.get("DEMO_FAKE") == "1":
        from agentcore_demo.fakes import FakeBackend  # noqa: PLC0415

        return FakeBackend()

    # config.py is gitignored.  If it doesn't exist, give a clear error
    # rather than an obscure ImportError.
    if importlib.util.find_spec("config") is None:
        raise RuntimeError(
            "config.py not found — copy config.example.py to config.py and fill it in, "
            "or set DEMO_FAKE=1 to run against the fake backend."
        )
    import config  # type: ignore[import]  # noqa: PLC0415
    from agentcore_demo.aws import AwsBackend  # noqa: PLC0415
    from agentcore_demo.pricing import fetch_rates  # noqa: PLC0415

    # fetch_rates() calls the AWS Price List API once and caches the result.
    # Takes 1-3 seconds on first call; subsequent calls return the cached value.
    rates = fetch_rates(config.REGION)
    return AwsBackend.from_config(config, rates)


def _build_meter() -> CostMeter:
    """Build a fresh CostMeter for one question run.

    A new meter is created for every /ws connection so that each run
    starts with a clean receipt.  (The backend is a singleton, but the
    meter is not.)
    """
    if os.environ.get("DEMO_FAKE") == "1":
        from agentcore_demo.fakes import TEST_KB_RATES, TEST_PRICING  # noqa: PLC0415

        return CostMeter(pricing=TEST_PRICING, ci_per_second=0.0, **TEST_KB_RATES)

    import config  # type: ignore[import]  # noqa: PLC0415
    from agentcore_demo.pricing import fetch_rates  # noqa: PLC0415

    rates = fetch_rates(config.REGION)
    return CostMeter(
        pricing=config.PRICING,
        ci_per_second=config.CODE_INTERPRETER_PER_SECOND,
        kb_query_usd_per_1k=rates.get("s3v_query_usd_per_1k", 0.0),
    )


# ── HTTP routes ────────────────────────────────────────────────────────────────


@app.get("/")
async def index() -> FileResponse:
    """Serve the main demo page (Alpine.js chat UI)."""
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/kb-status")
async def kb_status() -> JSONResponse:
    """Return whether the Bedrock Knowledge Base has indexed documents.

    The browser calls this on page load to decide whether to show the
    ingestion progress view or the question prompt.

    Response: {"ready": true} or {"ready": false}
    """
    backend = _get_backend()
    # kb_is_ready() may call AWS; run it in a thread so we don't block
    # the async event loop.
    ready = await asyncio.to_thread(backend.kb_is_ready)
    return JSONResponse({"ready": ready})


@app.get("/api/kb-costs")
async def kb_costs() -> JSONResponse:
    """Return measured KB setup costs for the sidebar panel.

    Response keys:
      ingestion_usd               one-time embedding cost (computed from corpus)
      storage_usd_per_month       S3 Vectors storage cost (from vector count)
      corpus_storage_usd_per_month  S3 corpus storage cost (from local dir)
      vector_count                number of vectors in the index
      vector_size_mb              estimated storage in MB
      corpus_files                number of .txt files in corpus/
      corpus_size_mb              total corpus size in MB
    """
    backend = _get_backend()
    costs = await asyncio.to_thread(backend.kb_setup_costs)
    return JSONResponse(costs)


@app.get("/corpus/{pmcid}")
async def serve_corpus_doc(pmcid: str):
    """Serve a local corpus paper as a readable HTML page.

    The guardrail substitution in agent.py generates links like
    /corpus/PMCxxxxxxx when a paper is in the local corpus.  Clicking
    one of those links hits this endpoint.

    The HTML is minimal -- no external fonts or scripts.  Atkinson
    Hyperlegible is specified as a preference but falls back to Georgia
    so the page renders correctly even without internet access (useful
    on a secure cluster or offline demo machine).
    """
    from fastapi import HTTPException  # noqa: PLC0415
    from fastapi.responses import HTMLResponse  # noqa: PLC0415

    path = Path("corpus") / f"{pmcid}.txt"
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"{pmcid} not in local corpus")

    text = path.read_text(encoding="utf-8")

    # The first line is a comment: "# PMCxxxxxxx  licence: CC0".
    # We strip it and use it as a sub-title in the HTML header.
    lines = text.splitlines()
    header = lines[0] if lines and lines[0].startswith("#") else ""
    body = "\n".join(lines[2:] if len(lines) > 2 else lines).strip()

    ncbi_url = f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmcid}/"
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>{pmcid} — Inside the Lines corpus</title>
  <style>
    body {{ max-width: 780px; margin: 2rem auto; padding: 0 1.5rem 4rem;
            font-family: "Atkinson Hyperlegible", Georgia, serif; line-height: 1.75;
            color: #1a1a1a; background: #fafaf8; font-size: 1.05rem; }}
    header {{ border-bottom: 1px solid #ddd; margin-bottom: 1.5rem; padding-bottom: 0.75rem; }}
    h1 {{ font-size: 1.2rem; font-weight: 700; margin-bottom: 0.3rem; }}
    .meta {{ font-size: 0.85rem; color: #666; }}
    .meta a {{ color: #0066cc; }}
    p {{ margin-bottom: 1em; }}
  </style>
</head>
<body>
  <header>
    <h1>{pmcid}</h1>
    <p class="meta">
      {header.lstrip("# ")} &nbsp;·&nbsp;
      <a href="{ncbi_url}" target="_blank" rel="noopener">View on PubMed Central ↗</a>
    </p>
  </header>
  <article>
    {"".join(f"<p>{para.strip()}</p>" for para in body.split("  ") if para.strip())}
  </article>
</body>
</html>"""
    return HTMLResponse(html)


@app.get("/api/questions")
async def get_questions() -> JSONResponse:
    """Return the four locked question texts.

    The browser uses these to populate the canned question chips and to
    teletype-animate the full question text into the input field before
    firing the WebSocket request.

    Response: {"questions": ["Q1 text", "Q2 text", "Q3 text", "Q4 text"]}
    """
    from agentcore_demo import questions as Q  # noqa: PLC0415

    return JSONResponse({"questions": Q.QUESTIONS})


# ── WebSocket: ingestion ───────────────────────────────────────────────────────


@app.websocket("/ingest")
async def websocket_ingest(ws: WebSocket) -> None:
    """Run a Bedrock ingestion job, streaming progress to the browser.

    The browser connects here when kb-status returns {"ready": false}.
    This WebSocket:
      1. Calls backend.kb_ingest(), which polls the ingestion job every 3s.
      2. Sends {"type": "ingesting", "indexed": N, "total": M} at each poll.
      3. Sends {"type": "kb_ready", "ingestion_usd": ..., ...} when done.

    The progress_cb bridge:
      kb_ingest() is synchronous and runs in a thread pool.  The callback
      it calls needs to send WebSocket messages.  We bridge with
      asyncio.run_coroutine_threadsafe() so the thread can safely schedule
      sends on the async event loop.
    """
    await ws.accept()
    backend = _get_backend()
    loop = asyncio.get_running_loop()

    def progress_cb(indexed: int, total: int) -> None:
        """Called from the ingestion polling thread; sends progress to the browser."""
        fut = asyncio.run_coroutine_threadsafe(
            ws.send_text(json.dumps({"type": "ingesting", "indexed": indexed, "total": total})),
            loop,
        )
        with contextlib.suppress(Exception):
            fut.result(timeout=5)  # block until send completes or 5s timeout

    try:
        costs = await asyncio.to_thread(backend.kb_ingest, progress_cb)
        with contextlib.suppress(Exception):
            await ws.send_text(json.dumps({"type": "kb_ready", **costs}))
    except WebSocketDisconnect:
        pass  # browser navigated away during ingestion -- that's fine
    finally:
        with contextlib.suppress(Exception):
            await ws.close()


# ── WebSocket: query run ───────────────────────────────────────────────────────


@app.websocket("/ws")
async def websocket_run(
    ws: WebSocket,
    q: int | None = Query(default=None, ge=1, le=4),
    text: str | None = Query(default=None),
) -> None:
    """Run the agent and forward every event as JSON over the WebSocket.

    Query parameters:
      q=1|2|3|4   run one of the four canned questions.
      text=...    run a free-form question (routed via Haiku).
      (neither)   run all four canned questions (Q1, Q2, Q3, Q4).

    The agent runs in a thread pool (asyncio.to_thread) so the blocking
    Bedrock API calls don't stall the async event loop.

    stop_event is set on WebSocketDisconnect so the agent's guarded_emit
    stops trying to send after the browser disconnects.  Without this,
    a long-running Opus call would keep going after the user closes the
    tab, wasting money.

    Note: agent.run() is synchronous and thread-safe.  Its emit callback
    schedules sends on the event loop via asyncio.run_coroutine_threadsafe().
    """
    await ws.accept()

    backend = _get_backend()
    meter = _build_meter()
    loop = asyncio.get_running_loop()
    stop_event = threading.Event()

    def emit(event: dict) -> None:
        """Send one event to the browser.  Called from the agent thread."""
        if stop_event.is_set():
            return  # browser disconnected -- don't try to send
        fut = asyncio.run_coroutine_threadsafe(ws.send_text(json.dumps(event)), loop)
        with contextlib.suppress(Exception):
            fut.result(timeout=5)

    try:
        if text:
            await asyncio.to_thread(_run_freeform, backend, meter, emit, stop_event, text)
        else:
            which = (q,) if q is not None else (1, 2, 3, 4)
            await asyncio.to_thread(_run_agent, backend, meter, emit, stop_event, which)
    except WebSocketDisconnect:
        stop_event.set()  # signal the agent thread to stop emitting
    finally:
        stop_event.set()
        with contextlib.suppress(Exception):
            await ws.close()


def _run_agent(
    backend: Backend,
    meter: CostMeter,
    emit,
    stop_event: threading.Event,
    which: tuple[int, ...] = (1, 2, 3, 4),
) -> None:
    """Run canned questions through the Agent.  Called in a thread pool."""

    def guarded_emit(event: dict) -> None:
        """Wrapper that skips the emit if the browser has disconnected."""
        if stop_event.is_set():
            return
        emit(event)

    Agent(backend, meter, guarded_emit).run(which=which)


def _run_freeform(
    backend: Backend,
    meter: CostMeter,
    emit,
    stop_event: threading.Event,
    text: str,
) -> None:
    """Run a free-form question through the Agent.  Called in a thread pool."""

    def guarded_emit(event: dict) -> None:
        if stop_event.is_set():
            return
        emit(event)

    Agent(backend, meter, guarded_emit).run_freeform(text)


if __name__ == "__main__":
    import uvicorn

    # Read host/port from config.py if available; fall back to localhost:8000.
    host = "127.0.0.1"
    port = 8000
    if importlib.util.find_spec("config") is not None:
        import config  # type: ignore[import]

        host = getattr(config, "HOST", host)
        port = getattr(config, "PORT", port)

    # Store the URL so the startup handler can open the browser.
    _launch_config["browser_url"] = f"http://{host}:{port}"
    uvicorn.run("agentcore_demo.app:app", host=host, port=port)
