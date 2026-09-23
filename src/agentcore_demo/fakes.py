"""
fakes.py  --  FakeBackend and test pricing for the demo.

This module is intentionally part of the main package (not in tests/) so
it can be used in two different ways:

  1. By tests (tests/conftest.py, tests/test_agent.py, etc.)
     The test suite imports FakeBackend and make_fake_backend_and_meter()
     directly.  No AWS credentials, no config.py, no network calls.

  2. By the web app in fake mode (DEMO_FAKE=1)
     `make demo-fake` sets DEMO_FAKE=1 and app._build_backend() imports
     FakeBackend here.  This lets you rehearse the entire UI flow --
     including the live cost meter and the Cedar policy denial badge --
     without spending any AWS money.

Environment variables that control FakeBackend behaviour:

  DEMO_FAKE=1
    Required for the app to use FakeBackend.  Without this the app tries
    to import config.py and build a real AwsBackend.

  DEMO_FAKE_READY=0
    Start the fake backend with the KB not ready.  The app will show the
    ingestion progress view and animate through fake progress steps before
    revealing the question prompt.  Default: 1 (KB is ready).
    Example: make demo-fake-ingest

  DEMO_FAKE_INGEST_DELAY=0.3
    Seconds to sleep between fake ingestion progress steps.  Set this to
    0.3 (or any positive float) to make the progress bar animate slowly
    enough to see.  Default: 0 (instant, for tests).
    Example: make demo-fake-ingest

What is "canned" in the fake:

  retrieve()       -- returns 6 plausible-looking passages per call.
  converse()       -- returns a tier-labelled placeholder answer, EXCEPT:
    - for the code-generation system prompt (Q2), returns valid Python
      that prints a 1×1 PNG as CHART_B64 so the chart path is exercised.
    - for the routing system prompt, classifies the question by keyword.
  code_interpreter_run() -- echoes the CHART_B64 line from the code.
  kb_setup_costs() -- returns FAKE_KB_SETUP_COSTS (realistic-ish numbers).
  kb_is_ready()    -- controlled by DEMO_FAKE_READY.
  kb_ingest()      -- animates through _INGEST_STEPS, then sets ready=True.
  kb_flush()       -- sets ready=False.
  query_gateway()  -- denies web_fetch (mimics Cedar policy); allows others.
"""

from __future__ import annotations

import os
import re
import time
from collections.abc import Callable

from agentcore_demo.cost import CostMeter

# Realistic-ish placeholder pricing for deterministic cost assertions in tests.
# These are intentionally NOT the real AWS prices -- tests should assert on
# known values, not on real rates that might change.
TEST_PRICING: dict[str, tuple[float, float]] = {
    "haiku": (0.80, 4.00),  # USD per 1M input, per 1M output tokens
    "sonnet": (3.00, 15.00),
    "opus": (15.00, 75.00),
    "openai": (12.00, 60.00),  # deliberately NOT Astra's real $11/$55
}

# KB retrieval rate used by CostMeter in tests.
# Like TEST_PRICING, a round test number and NOT the real rate: S3 Vectors bills
# $0.0025 per 1,000 QueryVectors requests ($2.50/M).  0.40 happens to be the
# value config.py carried until 2026-09-22, when it was found to be ~160x too
# high -- it survives here only because the cost assertions are calibrated to it.
TEST_KB_RATES = {
    "kb_query_usd_per_1k": 0.40,  # test-only; real rate is 0.0025
}

# Canned KB setup costs returned by FakeBackend.kb_setup_costs().
# Illustrative placeholders for rehearsal, not measurements, and smaller than
# reality: the live corpus is 1,000 papers / ~48 MB.  Nothing reads these except
# the UI panel and the tests, so they are left as-is rather than re-derived.
FAKE_KB_SETUP_COSTS = {
    "ingestion_usd": 0.10,  # one-time embedding cost, order of magnitude
    "storage_usd_per_month": 0.025,  # ~$0.025/month for S3 Vectors
    "corpus_storage_usd_per_month": 0.0002,  # ~fractions of a cent for S3
    "vector_count": 5508,  # a plausible chunk count
    # 5508 × 4608 bytes ÷ 1024².  4608 was aws._BYTES_PER_VECTOR before the
    # 2026-09-22 revision to 6656; not worth re-deriving for a canned value.
    "vector_size_mb": 24.2,
    "corpus_files": 200,  # a subset, so a fake run never claims the full corpus
    "corpus_size_mb": 9.8,  # matching size in MB
}

# A valid 1×1 transparent PNG encoded in base64.
# This is the smallest valid PNG that matplotlib could produce.
# It exercises the chart extraction path (agent._extract_chart) without
# needing to actually run matplotlib in tests.
_PNG_1x1 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAAC0lEQVR4nGNgAAIAAAUAAXpeqz8AAAAASUVORK5CYII="
)

# Fake ingestion progress steps: (indexed, total) pairs that progress_cb receives.
# Simulates a job that indexes 1,000 documents in 4 steps.
_INGEST_STEPS = [(0, 1000), (250, 1000), (500, 1000), (750, 1000), (1000, 1000)]


class FakeBackend:
    """Canned implementation of the Backend protocol -- no AWS, no network.

    All public methods record their calls in self.calls so tests can assert
    on what the agent called without inspecting output.
    Example:
        backend, meter = make_fake_backend_and_meter()
        agent = Agent(backend, meter, emit=lambda e: None)
        agent.question_1()
        assert "retrieve:" in backend.calls[0]
    """

    def __init__(self) -> None:
        # Records every method call as a short string for test assertions.
        self.calls: list[str] = []

        # Controlled by DEMO_FAKE_READY env var.
        # "1" (default): KB is ready, app skips ingestion and goes to question prompt.
        # "0": KB is not ready, app shows ingestion progress view.
        self._ready: bool = os.environ.get("DEMO_FAKE_READY", "1") != "0"

        # Seconds to sleep between fake ingestion steps.
        # 0 = instant (for tests); 0.3 = visible animation (for make demo-fake-ingest).
        self._ingest_delay: float = float(os.environ.get("DEMO_FAKE_INGEST_DELAY", "0"))

        # The model IDs a real run would use.  Present so that Agent._label()
        # takes the SAME code path in tests as in production -- labels are derived
        # from this map, so a fake without it would silently exercise the
        # fallback branch instead.  Kept in step with config.example.py.
        self.models: dict[str, str] = {
            "haiku": "us.anthropic.claude-haiku-4-5-20251001-v1:0",
            "sonnet": "us.anthropic.claude-sonnet-5",
            "opus": "us.anthropic.claude-opus-5",
            "openai": "us.openai.gpt-6-astra",
        }

        # Which of query_gateway()'s three outcomes to return for web_fetch.
        # Default "denied" is the rehearsed beat-4 path; tests override it to
        # reach the "error" and "result" branches.  See query_gateway() below.
        self.gateway_outcome: str = "denied"

    def retrieve(self, query: str, n: int = 12) -> list[dict]:
        """Return n plausible-looking passages about PCSK9."""
        self.calls.append(f"retrieve:{query[:20]}")
        return [
            {
                "text": f"Passage {i} about PCSK9 and LDL.",
                "source": f"s3://demo/PMC{1000 + i}.txt",
                "score": 0.9 - i * 0.01,
            }
            for i in range(min(n, 6))  # cap at 6 so tests don't get huge prompts
        ]

    def converse(
        self,
        tier: str,
        system: str,
        prompt: str,
        max_tokens: int = 1600,
        thinking: str | None = None,
    ) -> tuple[str, dict, list[dict]]:
        """Return a canned response appropriate to the tier and system prompt.

        Special cases:
          - Code-generation prompt (Q2): return valid Python that prints a
            CHART_B64 token so the chart extraction path is exercised.
          - Routing prompt: classify by keyword so routing tests are deterministic.
          - Everything else: return a placeholder answer with the tier name.
        """
        self.calls.append(f"converse:{tier}")
        usage = {"inputTokens": 12_000, "outputTokens": 1_000}

        # Q2: the system prompt mentions "self-contained Python".
        # Return a minimal script that prints a valid base64 PNG.
        if tier == "sonnet" and "self-contained Python" in system:
            code = f"import io, base64\nprint('CHART_B64:{_PNG_1x1}')"
            return code, usage, []

        # Routing: the system prompt says "Reply with exactly one word".
        # Classify by keyword so test inputs produce deterministic routes.
        if "Reply with exactly one word" in system:
            p = prompt.lower()
            if any(k in p for k in ["chart", "compare", "trial", "analys", "quantif", "visuali"]):
                return "ANALYSIS", {"inputTokens": 50, "outputTokens": 1}, []
            if any(k in p for k in ["disagree", "controvers", "adverse", "debate", "next"]):
                return "DEBATE", {"inputTokens": 50, "outputTokens": 1}, []
            return "SYNTHESIS", {"inputTokens": 50, "outputTokens": 1}, []

        # Default: a labelled placeholder for any other call.
        return f"[{tier}] answer to: {prompt[:40]}", usage, []

    def code_interpreter_run(self, code: str) -> tuple[str, float]:
        """Simulate running code: echo the CHART_B64 line the script would print.

        Extracts the CHART_B64 token from the code string (which was generated
        by the fake converse() above) and returns it as if the interpreter
        had actually executed the print statement.
        """
        self.calls.append("code_interpreter")
        m = re.search(r"CHART_B64:([A-Za-z0-9+/=]+)", code)
        return (f"CHART_B64:{m.group(1)}" if m else "ran ok"), 2.5

    def kb_setup_costs(self) -> dict:
        """Return canned KB setup costs -- no AWS calls."""
        self.calls.append("kb_setup_costs")
        return dict(FAKE_KB_SETUP_COSTS)

    def kb_is_ready(self) -> bool:
        """Return the current ready state, controlled by DEMO_FAKE_READY."""
        self.calls.append("kb_is_ready")
        return self._ready

    def kb_ingest(self, progress_cb: Callable[[int, int], None]) -> dict:
        """Simulate ingestion by animating through _INGEST_STEPS.

        If DEMO_FAKE_INGEST_DELAY > 0, sleeps between steps so the
        progress bar animation is visible in the browser.
        """
        self.calls.append("kb_ingest")
        for indexed, total in _INGEST_STEPS:
            progress_cb(indexed, total)
            if self._ingest_delay > 0:
                time.sleep(self._ingest_delay)
        self._ready = True  # mark ready after "ingestion" completes
        return dict(FAKE_KB_SETUP_COSTS)

    def kb_flush(self) -> None:
        """Reset the ready state so the next kb_ingest() will run again."""
        self.calls.append("kb_flush")
        self._ready = False

    def query_gateway(self, tool_name: str, arguments: dict) -> dict:
        """Fake Cedar gateway: deny web_fetch, allow everything else.

        This mimics the real Cedar ForbidWeb policy that denies web_fetch
        at the AgentCore Gateway level.  The demo relies on this denial to
        show the "Cedar Policy Denied" badge and the KB fallback.

        Set ``gateway_outcome`` to drive the other two branches the real
        backend can return.  The "error" case matters most: it is the branch
        added so beat 4 cannot silently succeed when the gateway misbehaves,
        and without a way to reach it from a fake it would ship untested.
            "denied"  (default) -- Cedar policy denial, the rehearsed path
            "error"             -- gateway reachable but the call failed
            "result"            -- the web fetch actually succeeded
        """
        self.calls.append(f"query_gateway:{tool_name}")
        if tool_name == "web_fetch":
            if self.gateway_outcome == "error":
                return {
                    "denied": False,
                    "error": True,
                    "reason": "gateway call failed: HTTP 500 from gateway endpoint",
                }
            if self.gateway_outcome == "denied":
                return {
                    "denied": True,
                    "reason": (
                        "Cedar policy denied: web_fetch is not permitted in this environment"
                    ),
                }
        return {"result": {"content": [{"type": "text", "text": f"[fake result for {tool_name}]"}]}}


def make_fake_backend_and_meter() -> tuple[FakeBackend, CostMeter]:
    """Convenience factory: FakeBackend + CostMeter with test pricing + KB rates.

    Used by both tests/conftest.py and run.py (DEMO_FAKE=1 path) so the
    test and fake-app pricing are always in sync.
    """
    return FakeBackend(), CostMeter(pricing=TEST_PRICING, ci_per_second=0.0, **TEST_KB_RATES)
