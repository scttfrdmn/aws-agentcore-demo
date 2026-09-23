"""
run.py  --  headless terminal runner for the Inside the Lines demo.

Builds the same AwsBackend + CostMeter used by the web app, runs the
questions through the Agent, and prints a live transcript to stdout.

Why use this instead of the web app?
  - To verify the demo works end-to-end BEFORE the talk, without opening
    a browser.  If something is misconfigured (wrong KB ID, missing model
    access), you'll see the error immediately in the terminal.
  - To run a single question cheaply during development: `--questions 1`
    runs Q1 only (Claude Haiku, a few cents) without the full three-question run.
  - To check the receipt total before committing to the full demo.

Headless mode means no WebSocket, no browser -- just the same event stream
printed to the terminal with a simple formatter (_emit_to_terminal).  The
event protocol is identical to the WebSocket version, so a passing headless
run confirms the web app will also work.

Usage:
    python -m agentcore_demo.run                # run Q1, Q2, Q3
    python -m agentcore_demo.run --questions 1  # Q1 only
    python -m agentcore_demo.run --questions 1,3  # Q1 and Q3
    python -m agentcore_demo.run --questions 4  # Q4 only (Cedar policy demo)

    DEMO_FAKE=1 python -m agentcore_demo.run    # fake backend (no AWS, no cost)
    make demo-headless                          # same as the default run
"""

from __future__ import annotations

import argparse
import datetime
import sys
import textwrap

from agentcore_demo.agent import Agent


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Headless terminal run of the Inside the Lines demo agent."
    )
    p.add_argument(
        "--questions",
        default="1,2,3",
        metavar="N[,N]",
        help="Comma-separated question numbers to run (default: 1,2,3).",
    )
    return p.parse_args()


def _build() -> tuple:
    """Build backend + meter using the same factory logic as app.py.

    Mirrors app._build_backend() and app._build_meter() so that the
    headless run and the web run always use the same configuration.
    Any difference here would mean a run that passes headless could fail
    in the browser (or vice versa).
    """
    import os

    if os.environ.get("DEMO_FAKE") == "1":
        # Fake mode: no AWS, no config.py needed.  Good for development.
        from agentcore_demo.fakes import make_fake_backend_and_meter  # noqa: PLC0415

        return make_fake_backend_and_meter()

    # Real AWS path -- requires config.py to be present and populated.
    import importlib.util  # noqa: PLC0415

    if importlib.util.find_spec("config") is None:
        sys.exit(
            "config.py not found — copy config.example.py to config.py and fill it in,\n"
            "or set DEMO_FAKE=1 to run against the fake backend."
        )

    import config  # type: ignore[import]  # noqa: PLC0415
    from agentcore_demo.aws import AwsBackend  # noqa: PLC0415
    from agentcore_demo.cost import CostMeter  # noqa: PLC0415
    from agentcore_demo.pricing import fetch_rates  # noqa: PLC0415

    rates = fetch_rates(config.REGION)
    backend = AwsBackend(
        config.REGION,
        config.KB_ID,
        config.DATA_SOURCE_ID,
        config.MODELS,
        rates=rates,
        vector_bucket_name=getattr(config, "VECTOR_BUCKET_NAME", ""),
    )
    meter = CostMeter(
        pricing=config.PRICING,
        ci_per_second=config.CODE_INTERPRETER_PER_SECOND,
        kb_query_usd_per_1k=rates.get("s3v_query_usd_per_1k", 0.0),
    )
    return backend, meter


# ── terminal renderer ─────────────────────────────────────────────────────────
# These constants and functions format the event stream as plain text.
# The event types mirror the WebSocket protocol in agent.py -- if you add
# a new event type there, add a matching elif branch here.

_WIDTH = 78
_HR = "─" * _WIDTH


def _hr(label: str = "") -> None:
    """Print a horizontal rule, optionally with a centred label."""
    if label:
        pad = _WIDTH - len(label) - 4
        print(f"── {label} {'─' * max(0, pad)}")
    else:
        print(_HR)


def _emit_to_terminal(ev: dict) -> None:  # noqa: C901  (switch-like structure, intentional)
    """Print one event to the terminal in a readable format.

    This is the terminal equivalent of the Alpine.js UI renderer in
    static/index.html.  Both consume the same event protocol.
    """
    t = ev["type"]

    if t == "setup_cost":
        # KB panel costs -- printed once at the start of the run.
        print(f"\n{'KB SETUP COSTS (not in run total)':^{_WIDTH}}")
        print(_HR)
        print(f"  Corpus ingestion (computed from corpus size):  ${ev['ingestion_usd']:.4f}")
        mo = ev["storage_usd_per_month"]
        print(f"  S3 Vectors storage (from vector count):        ${mo:.4f}/mo")
        print()

    elif t == "question":
        # Question header -- printed before each question's output.
        print(f"\n{'':=<{_WIDTH}}")
        print(f"  QUESTION {ev['n']}")
        for line in textwrap.wrap(ev["text"], _WIDTH - 4):
            print(f"  {line}")
        print(f"{'':=<{_WIDTH}}")

    elif t == "phase":
        # Status line -- retrieval, code running, etc.
        print(f"  ▸ {ev['label']}")

    elif t == "retrieval":
        print(f"  ▸ retrieved {ev['count']} passages")

    elif t == "model":
        # "start" shows the model name with trailing dots; "done" overwrites
        # with token counts and cost on the same line.
        if ev["state"] == "start":
            print(f"  ○ {ev['label']} …", end="", flush=True)
        else:
            u = ev.get("usage", {})
            cost = ev.get("cost", 0.0)
            print(
                f"\r  ● {ev['label']:<22}"
                f"  in={u.get('inputTokens', 0):>6,}  out={u.get('outputTokens', 0):>5,}"
                f"  ${cost:.6f}"
            )

    elif t == "code":
        # Show the first 12 lines of generated code -- enough to verify it looks right.
        print("\n  ── Generated code ──")
        for line in ev["text"].splitlines()[:12]:
            print(f"    {line}")
        extra = ev["text"].count("\n") - 12
        if extra > 0:
            print(f"    … ({extra} more lines)")
        print()

    elif t == "chart":
        # Charts can't be displayed in the terminal -- just report the size.
        kb = len(ev["data"]) * 3 // 4 // 1024
        print(f"  ▸ chart: {kb} KB PNG (base64, not displayed in terminal)")

    elif t == "answer":
        # Model answer -- show the first 20 wrapped lines.
        print(f"\n  ┌── {ev['title']}")
        wrapped = textwrap.wrap(ev["text"], _WIDTH - 6)
        for line in wrapped[:20]:
            print(f"  │  {line}")
        if len(wrapped) > 20:
            print(f"  │  … ({len(wrapped) - 20} more lines)")
        print(f"  └{'─' * (_WIDTH - 3)}")

    elif t == "cost":
        # Running total is emitted after every model call; suppress here
        # since each model row already shows its own cost.
        pass

    elif t == "receipt":
        # Final itemised receipt -- same data the browser renders as a table.
        print(f"\n{'RECEIPT':^{_WIDTH}}")
        print(_HR)
        print(f"  {'Step':<26} {'Model':<22} {'In':>7} {'Out':>6} {'Cost':>11}")
        print(f"  {'-' * 26} {'-' * 22} {'-' * 7} {'-' * 6} {'-' * 11}")
        for row in ev["rows"]:
            in_t = f"{row['in_tokens']:,}" if row["in_tokens"] else "—"
            out_t = f"{row['out_tokens']:,}" if row["out_tokens"] else "—"
            print(f"  {row['step']:<26} {row['label']:<22} {in_t:>7} {out_t:>6}  ${row['usd']:.6f}")
        print(_HR)
        print(f"  {'TOTAL':>57}  ${ev['total']:.6f}")

    elif t == "done":
        print(f"\n{'✓ Run complete':^{_WIDTH}}")
        print(_HR)


def main() -> None:
    """Entry point: parse arguments, build backend, run agent, print receipt."""
    args = _parse_args()
    try:
        which = tuple(int(n.strip()) for n in args.questions.split(","))
    except ValueError:
        sys.exit(f"--questions: expected comma-separated integers, got {args.questions!r}")

    print(_HR)
    print(f"{'Inside the Lines — PCSK9':^{_WIDTH}}")
    print(f"{'Bedrock Knowledge Base  ·  Claude + OpenAI':^{_WIDTH}}")
    print(_HR)
    print(f"  Started:   {datetime.datetime.now(datetime.UTC).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(f"  Questions: {list(which)}")
    print()

    backend, meter = _build()
    Agent(backend, meter, _emit_to_terminal).run(which=which)

    print(f"\n  Finished:  {datetime.datetime.now(datetime.UTC).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(_HR)


if __name__ == "__main__":
    main()
