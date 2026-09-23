#!/usr/bin/env python3
"""
start.py  --  the one thing a stranger has to run.

    git clone https://github.com/scttfrdmn/aws-agentcore-demo
    cd aws-agentcore-demo
    make start

That is the whole setup.  No config file to copy, no account ID to look up, no
bucket name to invent, no `aws s3 sync`, no IDs to paste back.  The owner's
requirement, verbatim: *"They should NOT have to get anything right -- it should
just work."*

This script runs the whole chain, in order, and every step knows how to skip
itself when it is already done:

    1. config      bootstrap.ensure_config()  -- generate config.py if absent;
                   fill ACCOUNT_ID from STS and derive BUCKET.  Never overwrites
                   a value a human typed.
    2. preflight   read-only checks.  Stops here, with ONE actionable sentence
                   per problem, if credentials, models, region or permissions
                   would make the run fail.  Costs nothing.
    3. corpus      corpus_fetch.main() -- skipped entirely if ./corpus/ already
                   holds 1,000 papers; resumes from a partial download.
    4. provision   build_kb.main() -- creates the bucket, uploads the corpus,
                   builds the KB/guardrail/gateway (all get-or-create), then
                   WRITES the resulting IDs into config.py.
    5. demo        launches the web app and opens the browser.

RESUMABILITY IS THE POINT.  Ctrl-C at any moment, or a failure in the middle,
then run `make start` again: it re-checks each step and does only what is left.
Nothing here deletes anything, and nothing creates a second copy of anything --
every name it uses is deterministic (see bootstrap.derive_bucket_name) precisely
so that a re-run addresses the SAME resources rather than making new ones.

IT DOES NOT ASK BEFORE SPENDING MONEY.  It prints what the run costs (about 60
cents the first day, 32 cents per rehearsal after that) and proceeds.  That is a
deliberate decision by the repo owner: informed, not interrogated.  `--dry-run`
is available for anyone who wants to look first, but it is not required.

Flags:
    --dry-run       print the plan and the cost, touch nothing
    --skip-demo     do the setup, do not launch the web app
    --no-preflight  skip the read-only checks (for offline debugging only)
"""

from __future__ import annotations

import argparse
import sys

import bootstrap


def _rule(text: str) -> None:
    """Print a step heading that is readable on a projector."""
    print(f"\n{'=' * 72}\n{text}\n{'=' * 72}")


def _corpus_status() -> tuple[int, bool]:
    """Return (papers on disk, is the corpus complete)."""
    import corpus_fetch  # noqa: PLC0415 -- imports `requests`; keep it lazy

    return len(corpus_fetch.existing_ids()), corpus_fetch.is_complete()


def step_config() -> None:
    """Step 1: make config.py exist and be correct.  Idempotent."""
    _rule("1/5  config.py")
    filled = bootstrap.ensure_config()
    if not filled:
        print("  config.py is already complete -- nothing to do")


def step_preflight() -> None:
    """Step 2: read-only checks.  Exits non-zero with plain-English problems.

    This runs BEFORE any resource is created, which is the entire value of it:
    the old failure mode was a botocore traceback nine minutes into provisioning,
    after money had already been spent on a build that could not finish.
    """
    _rule("2/5  preflight (read-only, costs nothing)")
    import preflight  # noqa: PLC0415 -- must follow step_config(); imports config

    problems = preflight.run()
    if not problems:
        return
    print(f"\nCannot run the demo yet. {len(problems)} thing(s) to fix:\n")
    for p in problems:
        print(f"  * {p}\n")
    print("Fix the above and run `make start` again -- it resumes where it stopped.")
    sys.exit(1)


def step_corpus() -> None:
    """Step 3: the paper corpus.  Free, slow, resumable, skipped when complete."""
    _rule("3/5  paper corpus (free; 10-15 minutes the first time)")
    have, complete = _corpus_status()
    if complete:
        print(f"  corpus already complete: {have} papers in ./corpus/ -- skipping")
        return
    import corpus_fetch  # noqa: PLC0415

    corpus_fetch.main()


def step_provision() -> None:
    """Step 4: every AWS resource, then the write-back into config.py.

    build_kb.main() is called in-process rather than shelled out so that a
    failure surfaces here, at the top level, with the step it failed in already
    printed above it.
    """
    _rule("4/5  provision AWS resources (~10 minutes the first time)")
    import build_kb  # noqa: PLC0415 -- builds boto3 clients at import

    build_kb.main()


def step_demo() -> None:
    """Step 5: run the web app and open the browser."""
    _rule("5/5  launch the demo")
    print("  opening http://127.0.0.1:8000 -- Ctrl-C to stop\n")
    import runpy  # noqa: PLC0415

    # runpy, not an import, because app.py does its uvicorn setup under
    # `if __name__ == "__main__"` (reading HOST/PORT from config and arranging
    # for the browser to open).  run_module reproduces `python -m
    # agentcore_demo.app` exactly, so `make start` and `make demo` cannot drift.
    runpy.run_module("agentcore_demo.app", run_name="__main__")


def plan() -> None:
    """--dry-run: print what would happen, and what it would cost.  Changes nothing."""
    have, complete = _corpus_status()
    config_state = "exists" if bootstrap.CONFIG_PATH.exists() else "will be generated"
    print("\nPlan (nothing has been changed):")
    print(f"  1. config.py            {config_state}")
    print("  2. preflight            read-only checks, free")
    print(
        f"  3. paper corpus         {have} papers on disk"
        + ("  -> skip" if complete else f"  -> fetch up to {1000 - have} more")
    )
    print("  4. provision AWS        bucket + upload + KB + guardrail + gateway")
    print("  5. demo                 http://127.0.0.1:8000")
    print(bootstrap.cost_notice())
    print("Run `make start` to do it.")


def main(argv: list[str] | None = None) -> int:
    """Run the chain.  Safe to re-run at any point."""
    ap = argparse.ArgumentParser(description="Set up and run the Inside the Lines demo.")
    ap.add_argument(
        "--dry-run", action="store_true", help="print the plan and cost, change nothing"
    )
    ap.add_argument("--skip-demo", action="store_true", help="set up but do not launch the web app")
    ap.add_argument(
        "--no-preflight", action="store_true", help="skip the read-only checks (debugging only)"
    )
    args = ap.parse_args(argv)

    print("Inside the Lines — PCSK9.  One command; nothing to edit by hand.")

    if args.dry_run:
        plan()
        return 0

    # The cost is printed BEFORE anything is created, so it is visible without
    # reading the README.  It is a notice, not a prompt -- see the module
    # docstring.
    print(bootstrap.cost_notice())

    step_config()
    if not args.no_preflight:
        step_preflight()
    step_corpus()
    step_provision()

    if args.skip_demo:
        _rule("Setup complete")
        print("  `make demo` when you are ready.")
        return 0
    step_demo()
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        # Ctrl-C is a normal way to leave this script (it can sit in a 15-minute
        # download or a 10-minute ingestion), so it gets a calm message instead
        # of a KeyboardInterrupt traceback.  Everything is resumable.
        print("\n\nStopped. Nothing was left half-created — run `make start` again to resume.")
        sys.exit(130)
    except RuntimeError as e:
        # RuntimeError is what this repo raises for "a human needs to know this":
        # an empty corpus, a bucket owned by someone else, an ingestion job that
        # indexed nothing.  Those messages are already written for a reader, so
        # print the message and NOT the traceback.  Anything else -- a genuine
        # bug, or an AWS error nobody anticipated -- still gets its full
        # traceback, because hiding those would make this repo undebuggable.
        print(f"\nStopped: {e}\n")
        print("Run `make preflight` for a read-only check, then `make start` to resume.")
        sys.exit(1)
