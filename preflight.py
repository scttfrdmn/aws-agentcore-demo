#!/usr/bin/env python3
"""
preflight.py  --  say what is wrong in one actionable sentence, not a traceback.

    uv run python preflight.py        # or: make preflight

Everything in here is READ-ONLY (`sts get-caller-identity`, `list*`, `get*`) and
costs nothing.  It runs automatically as the second step of `make start`, before
a single resource is created, because the alternative is what this repo used to
do: fail nine minutes into provisioning with a botocore traceback that a
non-specialist cannot act on.

The contract with the reader: every failure is ONE sentence that names the thing
that is wrong and the exact command that fixes it.  A stranger must never have
to interpret a stack trace.

NO CONSOLE.  That is a standing guardrail in CLAUDE.md, and it constrains this
file specifically: the obvious advice for a missing model is "open the Bedrock
console and request access", and it is not allowed here.  Every message below
names a CLI command or a config.py edit instead.

What it checks, in the order a failure would actually bite:

  1. Credentials      -- present, and not expired.
  2. Region services  -- S3 Vectors and AgentCore answer in cfg.REGION.
  3. Models           -- all four MODELS plus the Titan embedder are ACTIVE
                         there, named individually when they are not.
  4. Permissions      -- iam:SimulatePrincipalPolicy asks IAM, read-only,
                         whether this caller may perform the 14 actions the
                         build actually needs.  This is the check that catches
                         "your credentials work but you are not an admin", which
                         is the most common real-world failure and the one that
                         otherwise surfaces as AccessDeniedException half way
                         through provisioning.
"""

from __future__ import annotations

import sys

import boto3
from botocore.exceptions import (
    ClientError,
    EndpointConnectionError,
    NoCredentialsError,
    ProfileNotFound,
    TokenRetrievalError,
    UnauthorizedSSOTokenError,
)

import bootstrap

# The IAM actions build_kb.py genuinely performs.  Kept explicit rather than
# wildcarded so a denial names something the reader can look up.
REQUIRED_ACTIONS = [
    "s3:CreateBucket",
    "s3:PutObject",
    "s3:PutBucketTagging",
    "iam:CreateRole",
    "iam:PutRolePolicy",
    "iam:PassRole",
    "bedrock:CreateKnowledgeBase",
    "bedrock:CreateDataSource",
    "bedrock:StartIngestionJob",
    "bedrock:CreateGuardrail",
    "bedrock:InvokeModel",
    "bedrock:Retrieve",
    "s3vectors:CreateVectorBucket",
    "s3vectors:CreateIndex",
    "bedrock-agentcore:CreateGateway",
]

# Credential failures, mapped to the command that fixes each.  These are the
# botocore exception/error names that mean "your credentials are the problem",
# separated because the fix is different: absent vs expired.
_EXPIRED_CODES = {
    "ExpiredToken",
    "ExpiredTokenException",
    "RequestExpired",
    "InvalidClientTokenId",
    "TokenRefreshRequired",
}


def _iam_principal_arn(caller_arn: str) -> str | None:
    """Convert an STS caller ARN into an ARN simulate_principal_policy accepts.

    SimulatePrincipalPolicy wants an IAM user or role ARN.  For an assumed role,
    `get-caller-identity` returns the SESSION ARN --
    arn:aws:sts::123:assumed-role/RoleName/session-name -- which IAM rejects, so
    it has to be rewritten to arn:aws:iam::123:role/RoleName.  Returns None for
    anything else (root, or a federated user), in which case the permission
    check is skipped rather than reported as a failure.
    """
    parts = caller_arn.split(":")
    if len(parts) < 6:
        return None
    account, resource = parts[4], parts[5]
    if resource.startswith("assumed-role/"):
        role = resource.split("/")[1]
        return f"arn:aws:iam::{account}:role/{role}"
    if resource.startswith(("user/", "role/")):
        return f"arn:aws:iam::{account}:{resource}"
    return None


def check_credentials(session: boto3.Session) -> tuple[dict | None, list[str]]:
    """Return (caller identity, problems).

    `sts get-caller-identity` needs no IAM permission whatsoever, so if it fails
    the credentials themselves are missing or dead -- there is no third
    explanation, which is why this can give such a specific message.
    """
    try:
        return session.client("sts").get_caller_identity(), []
    except (NoCredentialsError, ProfileNotFound):
        return None, [
            "No AWS credentials found. Run `aws configure` (or "
            "`export AWS_PROFILE=<your-profile>`) and try again."
        ]
    except (TokenRetrievalError, UnauthorizedSSOTokenError):
        return None, [
            "Your AWS SSO session has expired. Run `aws sso login` "
            "(add `--profile <your-profile>` if you use one) and try again."
        ]
    except ClientError as e:
        code = e.response.get("Error", {}).get("Code", "")
        if code in _EXPIRED_CODES:
            return None, [
                "Your AWS credentials have expired. Refresh them "
                "(`aws sso login`, or `aws configure` for long-lived keys) and try again."
            ]
        return None, [
            f"AWS rejected your credentials ({code}). Re-run `aws configure`, or "
            f"check that AWS_PROFILE points at a profile that still works."
        ]


def _switch_region_advice(region: str) -> str:
    """Return ", so set REGION = ..." -- or "" when already in the default region.

    Telling a reader to switch to the region they are already in is worse than
    saying nothing: it reads as a bug in the tool and sends them hunting in the
    wrong place.  Every caller appends this, so the advice appears exactly when
    it is useful.
    """
    if region == bootstrap.DEFAULT_REGION:
        return ""
    return (
        f', so set REGION = "{bootstrap.DEFAULT_REGION}" in config.py '
        f"(where this demo is developed and priced)"
    )


def check_region_services(session: boto3.Session, region: str) -> list[str]:
    """Confirm S3 Vectors and AgentCore answer in `region`.

    Both are young services with a shorter region list than Bedrock itself, and
    neither is in botocore's legacy endpoints.json (`get_available_regions`
    returns an EMPTY list for them -- checked 2026-09-22), so there is no offline
    way to ask.  One cheap read-only List each is the only honest check.

    EndpointConnectionError is the signal that matters: it means the regional
    endpoint does not exist, i.e. the service is not in this region at all.
    """
    problems: list[str] = []
    probes = (
        ("S3 Vectors", "s3vectors", lambda c: c.list_vector_buckets(maxResults=1)),
        (
            "AgentCore",
            "bedrock-agentcore-control",
            lambda c: c.list_gateways(maxResults=1),
        ),
    )
    for label, service, call in probes:
        try:
            call(session.client(service, region_name=region))
        except EndpointConnectionError:
            switch = _switch_region_advice(region)
            problems.append(
                f"{label} is not available in {region}{switch or ' -- check your network'}, "
                f"then re-run."
            )
        except ClientError as e:
            code = e.response.get("Error", {}).get("Code", "")
            if code in ("AccessDeniedException", "AccessDenied", "UnauthorizedException"):
                problems.append(
                    f"Your credentials may not call {label} in {region} "
                    f"({code}). Ask whoever owns this AWS account for the "
                    f"`{service}:*` permissions listed in README Prerequisites."
                )
            # Any other ClientError means the endpoint answered, which is all
            # this probe is asking.
        except Exception as e:  # noqa: BLE001 -- unknown failure, still report usefully
            problems.append(
                f"Could not reach {label} in {region} ({type(e).__name__}). "
                f"Check your network, then re-run `make preflight`."
            )
    return problems


def check_models(session: boto3.Session, region: str, models: dict, embed_model: str) -> list[str]:
    """Confirm every model the demo names is ACTIVE in `region`.

    Two different APIs, because the demo uses two different kinds of model:
      * the four reasoning models are cross-Region INFERENCE PROFILES ("us."),
        listed by `bedrock list-inference-profiles`
      * the Titan embedder is a plain FOUNDATION MODEL, listed by
        `bedrock list-foundation-models`
    Asking the wrong one for either returns "missing" for a model that is fine.
    """
    problems: list[str] = []
    # "Set REGION to us-west-2" is not advice if you are already ON us-west-2 --
    # a message that tells you to do what you have already done reads as a bug and
    # sends the reader looking in the wrong place.
    switch = _switch_region_advice(region)
    try:
        active = bootstrap.active_inference_profiles(session, region)
    except Exception as e:  # noqa: BLE001
        return [
            f"Could not list Bedrock inference profiles in {region} "
            f"({type(e).__name__}). Check with: "
            f"`aws bedrock list-inference-profiles --region {region}`."
        ]

    missing = sorted({v for v in models.values() if v not in active})
    if missing:
        problems.append(
            "These Bedrock models are not ACTIVE for this account in "
            f"{region}: {', '.join(missing)}. Confirm with "
            f"`aws bedrock list-inference-profiles --region {region} "
            "--type-equals SYSTEM_DEFINED`. If they are absent entirely the "
            f"region does not offer them{switch}. If they are "
            "listed but not ACTIVE this account has no access yet: check with "
            f"`aws bedrock get-use-case-for-model-access --region {region}`, or "
            "point MODELS in config.py at models you do have."
        )

    try:
        fms = session.client("bedrock", region_name=region).list_foundation_models()
        embed_ok = any(
            m["modelId"] == embed_model
            and m.get("modelLifecycle", {}).get("status", "ACTIVE") == "ACTIVE"
            for m in fms.get("modelSummaries", [])
        )
    except Exception as e:  # noqa: BLE001
        return problems + [
            f"Could not list Bedrock foundation models in {region} "
            f"({type(e).__name__}). Check with "
            f"`aws bedrock list-foundation-models --region {region}`."
        ]
    if not embed_ok:
        problems.append(
            f"The embedding model {embed_model} is not available in {region}. "
            "Nothing can be ingested without it. Confirm with "
            f"`aws bedrock list-foundation-models --region {region} "
            f"--query \"modelSummaries[?modelId=='{embed_model}']\"`{switch}."
        )
    return problems


def check_permissions(session: boto3.Session, caller_arn: str) -> list[str]:
    """Ask IAM, read-only, whether this caller may do what build_kb.py will do.

    `iam:SimulatePrincipalPolicy` evaluates the caller's real policies without
    performing anything -- the only way to answer "will provisioning succeed?"
    without provisioning.  Verified working against a live IAM user 2026-09-22.

    If the simulation itself is denied (many read-only-ish roles cannot call it)
    this returns NO problems and prints a one-line note.  A check we cannot run
    must not block a run that would have worked; the real call will still fail
    loudly with its own AWS error if permissions are genuinely missing.
    """
    principal = _iam_principal_arn(caller_arn)
    if not principal:
        print("  (skipping the permission check: unusual caller identity)")
        return []
    try:
        result = session.client("iam").simulate_principal_policy(
            PolicySourceArn=principal, ActionNames=REQUIRED_ACTIONS
        )
    except ClientError:
        print("  (skipping the permission check: not allowed to call iam:SimulatePrincipalPolicy)")
        return []
    except Exception as e:  # noqa: BLE001
        print(f"  (skipping the permission check: {type(e).__name__})")
        return []

    denied = sorted(
        r["EvalActionName"]
        for r in result.get("EvaluationResults", [])
        if r["EvalDecision"] != "allowed"
    )
    if not denied:
        return []
    return [
        f"Your AWS identity ({principal.split('/')[-1]}) is not allowed to: "
        f"{', '.join(denied)}. Provisioning would fail part-way. Ask whoever "
        f"owns this AWS account to grant those actions (README Prerequisites "
        f"lists the services), or use a profile that already has them."
    ]


def run(*, echo=print) -> list[str]:
    """Run every check and return the list of problems (empty means good to go).

    Called by start.py before anything is created.  Imports config INSIDE the
    function so that bootstrap.ensure_config() has already had its chance to
    generate config.py -- importing at module scope would be an ImportError on a
    fresh clone, which is the exact ugliness this file removes.
    """
    # safe_session(), not boto3.Session(): the constructor raises ProfileNotFound
    # on a typo'd or empty AWS_PROFILE, which would be a traceback in the middle
    # of the one file whose whole job is not producing tracebacks.
    session, session_problem = bootstrap.safe_session()
    if session_problem or session is None:
        return [session_problem or "Could not build an AWS session."]

    identity, problems = check_credentials(session)
    if problems:  # nothing else can be checked without credentials
        return problems
    assert identity is not None
    echo(f"  credentials OK: account {identity['Account']}")

    import config as cfg  # noqa: PLC0415 -- must follow ensure_config(); see docstring

    region = cfg.REGION
    problems += check_region_services(session, region)
    problems += check_models(session, region, cfg.MODELS, cfg.EMBED_MODEL_ID)
    problems += check_permissions(session, identity["Arn"])
    if not problems:
        echo(f"  region {region}: S3 Vectors, AgentCore, all models and permissions OK")
    return problems


def main() -> int:
    """Entry point.  Prints problems one per line; exits non-zero if any."""
    print("Preflight (read-only, costs nothing)")
    bootstrap.ensure_config()
    problems = run()
    if not problems:
        print("\nAll good -- `make start` will work.")
        return 0
    print(f"\nCannot run the demo yet. {len(problems)} thing(s) to fix:\n")
    for p in problems:
        print(f"  * {p}\n")
    return 1


if __name__ == "__main__":
    sys.exit(main())
