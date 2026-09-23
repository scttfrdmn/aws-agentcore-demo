.PHONY: install lint fix test corpus build-kb demo demo-fake demo-fake-ingest demo-headless teardown teardown-dry-run

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

corpus:     ## fetch the PMC paper corpus (one-time)
	uv run python corpus_fetch.py

build-kb:   ## provision the Bedrock Knowledge Base (one-time)
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
