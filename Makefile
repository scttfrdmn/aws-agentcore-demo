.PHONY: start plan preflight install lint fix test corpus build-kb demo demo-fake demo-fake-ingest demo-headless teardown teardown-dry-run

# `make start` is THE entry point.  From a fresh `git clone` it generates
# config.py, runs read-only preflight checks, fetches the corpus, provisions
# every AWS resource, writes the resulting IDs back into config.py and opens the
# demo -- with nothing for the operator to look up, invent, upload or paste.
# Safe to re-run at any time: every step skips itself when it is already done.
# The granular targets below all still work for anyone who wants them.
start:      ## ONE COMMAND: set up everything and run the demo (resumable)
	uv run python start.py

plan:       ## show what `make start` would do, and what it costs. Changes nothing
	uv run python start.py --dry-run

preflight:  ## read-only checks: credentials, region, models, permissions. Free
	uv run python preflight.py

install:    ## install the package + dev tools
	uv pip install -e ".[dev]"

lint:       ## ruff check + format check
	uv run --extra dev ruff check .
	uv run --extra dev ruff format --check .

fix:        ## auto-fix lint + format
	uv run --extra dev ruff check --fix .
	uv run --extra dev ruff format .

test:       ## run the test suite (no AWS calls; works without `make install`)
	uv run --extra dev pytest

corpus:     ## fetch the PMC paper corpus (one-time, free, resumable)
	uv run python corpus_fetch.py

build-kb:   ## create the bucket, upload the corpus, provision the KB, write config.py
	uv run python build_kb.py

demo:       ## run the live web app (requires config.py)
	uv run python -m agentcore_demo.app

demo-fake:  ## run the web app with FakeBackend (no AWS, no config.py)
	DEMO_FAKE=1 uv run python -m agentcore_demo.app

demo-fake-ingest:  ## run demo with fake backend starting in not-ready (ingestion) state
	DEMO_FAKE=1 DEMO_FAKE_READY=0 DEMO_FAKE_INGEST_DELAY=0.3 uv run python -m agentcore_demo.app

demo-headless:  ## run all questions headless against real AWS (requires config.py)
	uv run python -m agentcore_demo.run

teardown-dry-run:  ## list what teardown WOULD delete, change nothing
	uv run python teardown.py --dry-run

teardown:   ## delete all billable AWS resources (exits non-zero if any survive)
	uv run python teardown.py
