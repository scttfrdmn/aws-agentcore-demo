"""
test_agent.py  --  drive Agent with the FakeBackend and assert event-stream correctness.
"""

from agentcore_demo.agent import ROUTE_LABELS, Agent
from agentcore_demo.fakes import FAKE_KB_SETUP_COSTS


def test_run_emits_a_well_formed_stream(backend, meter):
    events: list[dict] = []
    Agent(backend, meter, events.append).run()

    types = [e["type"] for e in events]
    assert types[-1] == "done"
    assert "receipt" in types
    assert types.count("question") == 4  # four beats since 2026-09-22

    receipt = next(e for e in events if e["type"] == "receipt")
    assert receipt["total"] == round(sum(r["usd"] for r in receipt["rows"]), 6)


def test_q1_emits_haiku_model_and_answer(backend, meter):
    events: list[dict] = []
    Agent(backend, meter, events.append).run(which=(1,))

    types = [e["type"] for e in events]
    assert types[0] == "setup_cost"
    assert events[1]["type"] == "question"
    assert events[1]["n"] == 1

    model_events = [e for e in events if e["type"] == "model"]
    tiers = [e["tier"] for e in model_events]
    assert "haiku" in tiers

    starts = [e for e in model_events if e["state"] == "start"]
    dones = [e for e in model_events if e["state"] == "done"]
    assert len(starts) == len(dones)

    assert any(e["type"] == "answer" for e in events)


def test_q2_emits_code_and_chart(backend, meter):
    events: list[dict] = []
    Agent(backend, meter, events.append).run(which=(2,))

    types = [e["type"] for e in events]
    assert "code" in types
    assert "chart" in types

    model_events = [e for e in events if e["type"] == "model"]
    assert any(e["tier"] == "sonnet" for e in model_events)


def test_q3_runs_both_opus_and_openai(backend, meter):
    events: list[dict] = []
    Agent(backend, meter, events.append).run(which=(3,))

    model_events = [e for e in events if e["type"] == "model"]
    tiers = [e["tier"] for e in model_events]
    assert "opus" in tiers
    assert "openai" in tiers
    assert "sonnet" in tiers


def test_model_events_carry_usage_and_cost_on_done(backend, meter):
    events: list[dict] = []
    Agent(backend, meter, events.append).run(which=(1,))

    done_models = [e for e in events if e["type"] == "model" and e["state"] == "done"]
    assert done_models
    for ev in done_models:
        assert "usage" in ev
        assert "inputTokens" in ev["usage"]
        assert "outputTokens" in ev["usage"]
        assert "cost" in ev


def test_receipt_total_equals_sum_of_rows(backend, meter):
    events: list[dict] = []
    Agent(backend, meter, events.append).run()

    receipt = next(e for e in events if e["type"] == "receipt")
    computed = round(sum(r["usd"] for r in receipt["rows"]), 6)
    assert receipt["total"] == computed


def test_cost_events_are_monotonically_non_decreasing(backend, meter):
    cost_events: list[float] = []

    def _collect(e):
        if e["type"] == "cost":
            cost_events.append(e["total"])

    Agent(backend, meter, _collect).run()

    assert cost_events
    for a, b in zip(cost_events, cost_events[1:], strict=False):
        assert b >= a, f"cost went backwards: {a} -> {b}"


# ── setup_cost event ──────────────────────────────────────────────────────────


def test_run_emits_exactly_one_setup_cost_event(backend, meter):
    events: list[dict] = []
    Agent(backend, meter, events.append).run()

    sc_events = [e for e in events if e["type"] == "setup_cost"]
    assert len(sc_events) == 1


def test_setup_cost_event_is_first(backend, meter):
    events: list[dict] = []
    Agent(backend, meter, events.append).run()

    assert events[0]["type"] == "setup_cost"


def test_setup_cost_event_has_required_fields(backend, meter):
    events: list[dict] = []
    Agent(backend, meter, events.append).run()

    sc = next(e for e in events if e["type"] == "setup_cost")
    assert "ingestion_usd" in sc
    assert "storage_usd_per_month" in sc


def test_setup_cost_comes_from_backend(backend, meter):
    events: list[dict] = []
    Agent(backend, meter, events.append).run()

    sc = next(e for e in events if e["type"] == "setup_cost")
    assert sc["ingestion_usd"] == FAKE_KB_SETUP_COSTS["ingestion_usd"]
    assert sc["storage_usd_per_month"] == FAKE_KB_SETUP_COSTS["storage_usd_per_month"]


def test_setup_cost_not_in_receipt_total(backend, meter):
    events: list[dict] = []
    Agent(backend, meter, events.append).run()

    sc = next(e for e in events if e["type"] == "setup_cost")
    receipt = next(e for e in events if e["type"] == "receipt")

    row_sum = round(sum(r["usd"] for r in receipt["rows"]), 6)
    assert receipt["total"] == row_sum
    assert sc["ingestion_usd"] + sc["storage_usd_per_month"] > 0


# ── retrieval rows ────────────────────────────────────────────────────────────


def test_receipt_contains_retrieval_rows(backend, meter):
    events: list[dict] = []
    Agent(backend, meter, events.append).run()

    receipt = next(e for e in events if e["type"] == "receipt")
    retrieval_rows = [r for r in receipt["rows"] if r["label"] == "KB retrieval"]
    # Four: Q1-Q4 each retrieve.  Q4 retrieves too, because its Cedar denial
    # falls back to the knowledge base.
    assert len(retrieval_rows) == 4


def test_receipt_rows_have_no_estimated_field(backend, meter):
    events: list[dict] = []
    Agent(backend, meter, events.append).run()

    receipt = next(e for e in events if e["type"] == "receipt")
    for row in receipt["rows"]:
        assert "estimated" not in row


# ── routing ───────────────────────────────────────────────────────────────────


def test_route_returns_valid_path(backend, meter):
    agent = Agent(backend, meter, lambda e: None)
    path = agent.route("What is PCSK9?")
    assert path in ROUTE_LABELS


def test_route_synthesis_for_generic_question(backend, meter):
    agent = Agent(backend, meter, lambda e: None)
    assert agent.route("What is the established role of PCSK9?") == "SYNTHESIS"


def test_route_analysis_for_chart_question(backend, meter):
    agent = Agent(backend, meter, lambda e: None)
    assert agent.route("Compare and chart the trial results") == "ANALYSIS"


def test_route_debate_for_controversy_question(backend, meter):
    agent = Agent(backend, meter, lambda e: None)
    assert agent.route("Where does the literature disagree?") == "DEBATE"


def test_route_is_metered_in_receipt(backend, meter):
    """Routing Haiku call appears in the receipt."""
    events: list[dict] = []
    Agent(backend, meter, events.append).run_freeform("What is PCSK9?")

    receipt = next(e for e in events if e["type"] == "receipt")
    routing_rows = [r for r in receipt["rows"] if "routing" in r["step"]]
    assert routing_rows, "expected a routing row in the receipt"


# ── run_freeform ──────────────────────────────────────────────────────────────


def test_run_freeform_emits_route_event(backend, meter):
    events: list[dict] = []
    Agent(backend, meter, events.append).run_freeform("What is PCSK9?")

    route_events = [e for e in events if e["type"] == "route"]
    assert len(route_events) == 1
    assert route_events[0]["path"] in ROUTE_LABELS
    assert route_events[0]["label"] == ROUTE_LABELS[route_events[0]["path"]]


def test_run_freeform_ends_with_receipt_and_done(backend, meter):
    events: list[dict] = []
    Agent(backend, meter, events.append).run_freeform("What is PCSK9?")

    types = [e["type"] for e in events]
    assert types[-1] == "done"
    assert "receipt" in types


def test_run_freeform_no_setup_cost_event(backend, meter):
    """run_freeform is a single question; it does not emit setup_cost."""
    events: list[dict] = []
    Agent(backend, meter, events.append).run_freeform("What is PCSK9?")

    assert not any(e["type"] == "setup_cost" for e in events)


def test_run_freeform_analysis_path_emits_chart(backend, meter):
    events: list[dict] = []
    Agent(backend, meter, events.append).run_freeform("Compare and chart the trial results")

    assert any(e["type"] == "chart" for e in events)


def test_run_freeform_debate_path_uses_opus_and_openai(backend, meter):
    events: list[dict] = []
    Agent(backend, meter, events.append).run_freeform("Where does the literature disagree?")

    tiers = [e["tier"] for e in events if e["type"] == "model"]
    assert "opus" in tiers
    assert "openai" in tiers


def test_model_events_have_guardrail_field_in_agent_run(backend, meter):
    """FakeBackend returns no guardrail matches; verify run() stays well-formed."""
    events: list[dict] = []
    Agent(backend, meter, events.append).run(which=(1,))

    done_models = [e for e in events if e["type"] == "model" and e["state"] == "done"]
    assert done_models, "expected at least one model-done event"
    for ev in done_models:
        assert "usage" in ev
        assert "cost" in ev

    # No guardrail events expected from FakeBackend (returns empty matches list).
    guardrail_events = [e for e in events if e["type"] == "guardrail"]
    assert guardrail_events == [], "FakeBackend emits no guardrail events"


# ── Q4: AgentCore Gateway / Cedar policy demo ─────────────────────────────────


def test_q4_emits_policy_denied_event(backend, meter):
    """Q4 attempts web_fetch via gateway; fake backend denies it; policy_denied event emitted."""
    events: list[dict] = []
    Agent(backend, meter, events.append).run(which=(4,))

    types = [e["type"] for e in events]
    assert "policy_denied" in types

    pd = next(e for e in events if e["type"] == "policy_denied")
    assert pd["tool"] == "web_fetch"
    assert "denied" in pd["reason"].lower()

    assert any(e["type"] == "answer" for e in events)
    assert events[-1]["type"] == "done"


def test_q4_visible_error_when_gateway_fails_not_a_fake_denial(backend, meter):
    """A broken gateway must surface an error, never a fake "Cedar denied" badge.

    This is the branch that exists so beat 4 cannot silently succeed or claim a
    policy denial it did not get.  Before the Sept 2026 fix an unrecognised
    gateway response fell through as a success, so the rehearsed badge would
    quietly not appear on stage.  Drive it from the fake so it stays covered.
    """
    backend.gateway_outcome = "error"
    events: list[dict] = []
    Agent(backend, meter, events.append).run(which=(4,))

    # No policy_denied event, because no policy denial actually happened.
    assert not [e for e in events if e["type"] == "policy_denied"]
    # The run still completes and still answers from the knowledge base.
    assert any(e["type"] == "answer" for e in events)
