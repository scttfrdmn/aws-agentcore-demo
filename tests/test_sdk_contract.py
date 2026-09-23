"""
SDK surface-contract tests -- drift detectors, not behaviour tests.

Every other test in this suite runs against FakeBackend (agentcore_demo.fakes),
which is exactly what we want for logic: CI never touches the cloud.  The gap
that leaves is real: nothing in the suite ever names the `bedrock_agentcore`
SDK or a boto3 service string, so an upstream rename would pass CI green and
then fail at 4pm on a conference stage.

These tests close that gap.  They assert only that the *names* this repo reaches
for still exist in the installed libraries -- no request is made, no credentials
are read, no endpoint is resolved.  A failure here means the dependency moved
under us; the assertion messages name the call site so the fix is obvious.
"""

import inspect

import pytest
from bedrock_agentcore.tools.code_interpreter_client import CodeInterpreter
from botocore.session import get_session

# ── AgentCore Code Interpreter ───────────────────────────────────────────────
#
# Used by aws.py :: Backend.code_interpreter_run(), which is beat 2 of the demo:
#     ci = CodeInterpreter(region); ci.start()
#     resp = ci.invoke("executeCode", {"language": "python", "code": code})
#     ci.stop()
# The import itself is part of the contract -- the module path has moved before.


@pytest.mark.parametrize("method", ["start", "invoke", "stop"])
def test_code_interpreter_exposes_method(method):
    """The three calls code_interpreter_run() makes must all still be callable."""
    attr = getattr(CodeInterpreter, method, None)
    assert callable(attr), (
        f"bedrock_agentcore CodeInterpreter no longer has a callable .{method}(); "
        f"aws.py Backend.code_interpreter_run() calls it (start -> invoke -> stop). "
        f"Check the installed bedrock-agentcore release notes and update aws.py."
    )


def test_code_interpreter_accepts_region_argument():
    """We construct it positionally as CodeInterpreter(self.region); keep that parameter."""
    params = inspect.signature(CodeInterpreter.__init__).parameters
    assert "region" in params, (
        "bedrock_agentcore CodeInterpreter.__init__ no longer takes a 'region' "
        f"parameter (now: {list(params)}); aws.py Backend.code_interpreter_run() "
        "calls CodeInterpreter(self.region)."
    )


# ── boto3 service names ──────────────────────────────────────────────────────
#
# Service strings are validated against botocore's bundled service models rather
# than by calling boto3.client(...): constructing a client can try to resolve
# credentials and endpoints, and this suite must run credential-free.

# Every service name the repo passes to boto3.client(), with its call site, so a
# failure points at the file to edit rather than just a missing string.
SERVICE_CALL_SITES = {
    "bedrock-agent-runtime": "aws.py (retrieve)",
    "bedrock-runtime": "aws.py (converse)",
    "bedrock-agent": "aws.py, build_kb.py, teardown.py (KB + ingestion mgmt)",
    "bedrock": "build_kb.py, teardown.py (model + guardrail mgmt)",
    "bedrock-agentcore-control": "build_kb.py, teardown.py (Gateway + Cedar PolicyEngine mgmt)",
    "s3": "teardown.py (corpus bucket)",
    "s3vectors": "aws.py, build_kb.py, teardown.py (S3 Vectors store)",
    "iam": "build_kb.py, teardown.py (KB execution role)",
    # Added 2026-09-22: teardown.py now queries the Resource Groups Tagging API
    # to find the S3 corpus bucket by tag -- it is the one resource whose name
    # the project does not control.
    "resourcegroupstaggingapi": "teardown.py (tag sweep for the S3 corpus bucket)",
    "pricing": "pricing.py (live AWS Price List API)",
}


@pytest.mark.parametrize("service,call_site", sorted(SERVICE_CALL_SITES.items()))
def test_boto3_service_name_is_known_to_botocore(service, call_site):
    """A renamed or dropped service name would only surface at runtime otherwise."""
    available = get_session().get_available_services()
    assert service in available, (
        f"botocore no longer knows the service name {service!r}, used in {call_site}. "
        f"Either the service was renamed or the installed boto3/botocore is too old "
        f"-- bump boto3 in pyproject.toml, or fix the client name at that call site."
    )


# ── boto3 operation names on the live-demo hot path ──────────────────────────
#
# A valid service name with a renamed operation fails just as hard on stage, so
# check the two calls every question in the demo makes.  botocore's operation
# list comes from bundled JSON service models -- reading it needs no network.
# Scoped deliberately to aws.py's runtime path (the "verified AWS facts" in
# CLAUDE.md); provisioning operations live in build_kb.py and churn more.

HOT_PATH_OPERATIONS = [
    ("bedrock-agent-runtime", "Retrieve", "aws.py Backend.retrieve()"),
    ("bedrock-runtime", "Converse", "aws.py Backend.converse()"),
]


@pytest.mark.parametrize("service,operation,call_site", HOT_PATH_OPERATIONS)
def test_boto3_operation_is_known_to_botocore(service, operation, call_site):
    """Every demo question goes through these two operations; both must still exist."""
    operations = get_session().get_service_model(service).operation_names
    assert operation in operations, (
        f"botocore's {service} model no longer defines the {operation} operation, "
        f"which {call_site} depends on.  The API was renamed or removed -- check the "
        f"current Bedrock docs and update aws.py."
    )
