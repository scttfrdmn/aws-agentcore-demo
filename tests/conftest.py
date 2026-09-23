"""
conftest.py  --  shared test fixtures.

FakeBackend, TEST_PRICING, and TEST_KB_RATES live in agentcore_demo.fakes (so
both the app and the tests share exactly one implementation). This file
re-exports them and provides pytest fixtures.
"""

import sys
from pathlib import Path

import pytest

# corpus_fetch.py, build_kb.py and teardown.py live at the repo ROOT, not under
# src/, so they are not part of the installed package and pytest does not put
# the root on sys.path (tests/ has no __init__.py, so only tests/ is added).
# Added 2026-09-22 so tests/test_corpus_licence.py can import the licence
# classifier -- the gate that keeps non-commercial papers out of the corpus.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agentcore_demo.cost import CostMeter
from agentcore_demo.fakes import TEST_KB_RATES, TEST_PRICING, FakeBackend

__all__ = ["TEST_KB_RATES", "TEST_PRICING", "FakeBackend"]


@pytest.fixture
def backend() -> FakeBackend:
    return FakeBackend()


@pytest.fixture
def meter() -> CostMeter:
    return CostMeter(pricing=TEST_PRICING, ci_per_second=0.0, **TEST_KB_RATES)


@pytest.fixture
def fake_env(monkeypatch):
    """Set DEMO_FAKE=1 so _make_backend_and_meter() returns the fake backend."""
    monkeypatch.setenv("DEMO_FAKE", "1")
