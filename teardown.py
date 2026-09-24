#!/usr/bin/env python3
"""
teardown.py  --  delete everything build_kb.py created, including the runs
                 config.py has forgotten about.

Run this after the talk.  Run it with --dry-run any time you want to *check*
whether the account is clean without touching anything.

    python teardown.py --dry-run    # list what would be deleted, delete nothing
    python teardown.py              # delete it

Exit status is meaningful: after deleting, the script re-lists everything that
still matches this project and exits NON-ZERO if anything survived.  "Clean" is
checkable, not hoped-for.

WHY THIS IS NOT JUST "DELETE WHAT CONFIG NAMES" (2026-09-22)
-----------------------------------------------------------
It used to be.  It keyed off cfg.KB_ID, cfg.VECTOR_BUCKET_NAME, cfg.GUARDRAIL_ID,
cfg.GATEWAY_ID, cfg.GATEWAY_ENGINE_ID, cfg.BUCKET and cfg.KB_ROLE_NAME -- so any
rename between a build and a teardown orphaned the previous generation forever.
One demo account was found holding, all of it billable or at least embarrassing:

  * three S3 vector buckets (plain, -v2, -v3), each with an index
  * a Knowledge Base stuck in DELETE_UNSUCCESSFUL since June
  * two stray Cedar policy engines holding five policies between them
  * a guardrail named "inside-the-lines-guardrail-v3", which build_kb.py could
    never adopt because it get-or-creates "inside-the-lines-url-filter"
  * two stray IAM roles ("...-kb-role-v2", "...-kb-role-v3")

This is a public repo and the rule is that setup and teardown must not leave
things in other people's accounts.  So teardown now DISCOVERS instead of
assuming.  For every category it enumerates what actually exists in the account
and region, and claims an item if ANY of these is true:

  config   -- config.py names it by ID (the fast path; also the only signal that
              survives a resource whose name and tag both went missing)
  tag      -- it carries Project=inside-the-lines, which build_kb.py stamps on
              everything it can tag
  name     -- its name starts with "inside-the-lines" after normalising case and
              punctuation, so "InsideTheLinesEngine" and "...-vectors-v3" both
              match.  This is the backstop for the three things AWS will not let
              us tag: gateway targets, Cedar policies and KB data sources (all
              of which are children of something we DO tag, so they are reached
              through their parent anyway).

Enumerating costs one List call per category, which is cheap and buys the
property that matters: there is no rename that can hide a resource from this.
If a List call fails outright (no permission, ancient botocore), the config
values are seeded in unconditionally so behaviour degrades to the old fast path
rather than to silence -- and the failure is printed.

What gets deleted, and what it costs to leave behind:
  - AgentCore Gateway        -- $0 idle, but pay-per-invocation + Cedar eval fees
  - Gateway targets          -- $0 idle; OpenAPI targets, so no compute behind them
  - Cedar policies + engine   -- $0 idle
  - Bedrock Guardrail        -- ~$0.75 per million text units if left in use
  - Bedrock Knowledge Base   -- $0 idle, but see the S3 Vectors index below
  - S3 Vectors index         -- ~$0.06/GB-month (~$0.002/month for a 1,000-paper
                                corpus; negligible but real, and it accumulates
                                once per orphaned generation)
  - S3 Vectors bucket        -- $0 if empty, deleted after its indexes are gone
  - S3 corpus bucket         -- ~$0.023/GB-month standard storage
  - IAM roles (gateway, KB)  -- no cost, but they are the mess people notice
  - local corpus/ directory  -- the downloaded papers on your laptop

Ordering that is load-bearing (do not reshuffle):
  1. gateway targets drained -> policy engine detached -> gateway deleted
  2. Cedar policies deleted  -> policy engine deleted
  3. every data source set to dataDeletionPolicy=RETAIN -> KB deleted -> WAIT ->
     S3 Vectors index -> S3 Vectors bucket
  4. IAM roles after the resources that assume them

Re-running safely:
  Idempotent.  Every deletion goes through _try(), which prints and continues
  when a resource is already gone.

There is deliberately NO Lambda cleanup here (removed 2026-09-22, kept removed).
An earlier design had a Lambda-backed web_fetch, but the beat-4 gateway target is
an OpenAPI target straight to clinicaltrials.gov -- no compute, no execution
role.  A teardown that lists resources the build never creates invites the
opposite mistake: assuming something exists that does not.
"""

import argparse
import os
import shutil
import sys
import time

import boto3

import config as cfg

agent = boto3.client("bedrock-agent", region_name=cfg.REGION)
br = boto3.client("bedrock", region_name=cfg.REGION)
br_ctrl = boto3.client("bedrock-agentcore-control", region_name=cfg.REGION)
iam = boto3.client("iam")
s3v = boto3.client("s3vectors", region_name=cfg.REGION)

# S3 general-purpose buckets are global but their endpoints are regional.  We
# pin the client to cfg.REGION because that is where build_kb.py expects the
# corpus bucket to live (same region as the KB, or ingestion pays cross-Region
# transfer).  A corpus bucket deliberately placed in another Region will show up
# in list_buckets() but its delete may need a client in that Region -- the _try()
# wrapper will report that rather than crash.
s3 = boto3.client("s3", region_name=cfg.REGION)

# Tag-based discovery for S3 buckets only.  Every other category here has a
# cheap account-wide List with names in it; S3 does too, but the corpus bucket's
# name is user-chosen, so a rename leaves the tag as the only handle.  Checking
# 147 buckets with GetBucketTagging one at a time would be 147 calls; the
# Resource Groups Tagging API answers the same question in one.
rgt = boto3.client("resourcegroupstaggingapi", region_name=cfg.REGION)

# MUST match build_kb.py's PROJECT_TAG_KEY / PROJECT_TAG_VALUE.  Duplicated on
# purpose: importing build_kb here would build an "s3vectors" client at import
# time, so a botocore too old for S3 Vectors would stop teardown from running at
# all -- and teardown has to keep working when the build script cannot.
# Two files, one grep: `grep -rn PROJECT_TAG_ *.py`.
PROJECT_TAG_KEY = "Project"
PROJECT_TAG_VALUE = "inside-the-lines"

# The name backstop.  Normalised (lowercased, punctuation stripped) before
# comparison so that every spelling this repo has ever used matches:
#   "inside-the-lines-vectors-v3"  -> insidethelinesvectorsv3   MATCH
#   "InsideTheLinesEngine"         -> insidethelinesengine      MATCH
#   "inside_the_lines_kb_role"     -> insidethelineskbrole      MATCH
# The policy engine is the reason this is normalised rather than a plain
# startswith: build_kb.py names it "InsideTheLinesEngine", which shares no
# literal prefix with anything else the project creates.
NAME_PREFIX = "inside-the-lines"


def _norm(s: str) -> str:
    """Lowercase and drop every non-alphanumeric character."""
    return "".join(ch for ch in (s or "").lower() if ch.isalnum())


_NORM_PREFIX = _norm(NAME_PREFIX)


def _name_matches(name: str) -> bool:
    """True if `name` looks like a resource this project created."""
    return _norm(name).startswith(_NORM_PREFIX)


def _try(label, fn):
    """Call fn(); print success or skip message.  Never raises.

    Teardown is inherently best-effort: if one resource is already gone
    or was never created, we still want to proceed to the next one.
    """
    try:
        fn()
        print(f"  deleted: {label}")
    except Exception as e:  # noqa: BLE001 -- teardown is best-effort
        print(f"  skip ({label}): {type(e).__name__}")


def _paginate(fn, key, **kwargs):
    """Yield every item from a nextToken-paginated AWS list call.

    AWS is inconsistent about the name of the list in each response
    (ListPolicyEngines -> "policyEngines", ListPolicies -> "policies",
    ListGateways / ListGatewayTargets -> "items"), which has already caused two
    "the API returns empty" bugs in this repo.  So `key` is always explicit.
    """
    token = None
    while True:
        page = fn(**kwargs, **({"nextToken": token} if token else {}))
        yield from page.get(key, [])
        token = page.get("nextToken")
        if not token:
            return


def _has_project_tag(fetch) -> bool:
    """True if a ListTagsForResource response carries Project=inside-the-lines.

    `fetch` is a zero-argument thunk that performs the call, e.g.
        lambda: br.list_tags_for_resource(resourceARN=arn)

    It is a thunk rather than (client_method, arn) because the four services
    involved do NOT agree on the parameter name or the response type, and a
    wrapper that guessed got it wrong on the first try:

        bedrock                      resourceARN  -> tags: [{key, value}]
        bedrock-agent                resourceArn  -> tags: {k: v}
        bedrock-agentcore-control    resourceArn  -> tags: {k: v}
        s3vectors                    resourceArn  -> tags: {k: v}

    `bedrock` is the outlier on BOTH counts -- capital "ARN", and a list where
    everything else returns a map.  (Read out of the botocore service models on
    2026-09-22; `resourceArn` on `bedrock` raises ParamValidationError.)  Letting
    each call site spell its own call keeps that asymmetry visible instead of
    hidden in a helper.

    Failures are printed, not swallowed silently.  An earlier draft of this
    function returned False on any exception with no output, which turned the
    `resourceARN` typo above into "the guardrail simply isn't tagged" -- the
    single most dangerous way for a teardown sweep to fail.
    """
    try:
        tags = fetch().get("tags") or {}
    except Exception as e:  # noqa: BLE001 -- unreadable tags must not stop the sweep
        print(f"  warning: tag lookup failed ({type(e).__name__}); relying on name/config")
        return False
    if isinstance(tags, list):  # bedrock: [{"key": ..., "value": ...}]
        return any(
            t.get("key") == PROJECT_TAG_KEY and t.get("value") == PROJECT_TAG_VALUE for t in tags
        )
    return tags.get(PROJECT_TAG_KEY) == PROJECT_TAG_VALUE


def _why(name: str, config_ids: set, *, tagged: bool = False) -> list[str]:
    """Return the list of reasons this resource is claimed ('' if not claimed).

    Reporting every reason rather than the first is deliberate: the dry-run
    output is how a repo user audits the sweep, and "[name] only" on something
    they expected to be tagged is the signal that build_kb.py failed to tag it.

    That only works if the tag is ALWAYS looked up, so every _discover_* does,
    even when config or name has already claimed the resource.  It used to skip
    the lookup in that case to save a call, which meant the tag could never
    appear next to config/name -- the audit could not tell "tagged" from
    "build_kb.py silently failed to tag", and the tag backstop went unverified
    (found 2026-09-24: every tagged resource showed [config, name]).
    """
    reasons = []
    if config_ids:
        reasons.append("config")
    if tagged:
        reasons.append("tag")
    if _name_matches(name):
        reasons.append("name")
    return reasons


# ----------------------------------------------------------------------
# Discovery -- one function per category.  All read-only.
# ----------------------------------------------------------------------
def _discover_gateways() -> list[dict]:
    """AgentCore Gateways, each with its targets (targets cannot be tagged)."""
    found = []
    cfg_id = getattr(cfg, "GATEWAY_ID", "")
    try:
        for gw in _paginate(br_ctrl.list_gateways, "items"):
            gid, name = gw["gatewayId"], gw.get("name", "")
            ids = {gid} & {cfg_id} if cfg_id else set()
            # Checked even when config or name already claims it -- see _why.
            # ListGateways items carry no ARN; GetGateway does.
            try:
                arn = br_ctrl.get_gateway(gatewayIdentifier=gid)["gatewayArn"]
                tagged = _has_project_tag(
                    lambda a=arn: br_ctrl.list_tags_for_resource(resourceArn=a)
                )
            except Exception as e:  # noqa: BLE001
                print(f"  warning: could not read gateway {gid} ({type(e).__name__})")
                tagged = False
            why = _why(name, ids, tagged=tagged)
            if not why:
                continue
            targets = []
            try:
                targets = [
                    {"name": t.get("name", ""), "id": t["targetId"]}
                    for t in _paginate(br_ctrl.list_gateway_targets, "items", gatewayIdentifier=gid)
                ]
            except Exception as e:  # noqa: BLE001
                print(f"  warning: could not list targets of gateway {gid}: {type(e).__name__}")
            found.append({"id": gid, "name": name, "why": why, "targets": targets})
    except Exception as e:  # noqa: BLE001
        print(f"  warning: list_gateways failed ({type(e).__name__}); falling back to config")
        if cfg_id:
            found.append({"id": cfg_id, "name": "?", "why": ["config"], "targets": []})
    return found


def _discover_policy_engines() -> list[dict]:
    """Cedar policy engines, each with its policies (policies cannot be tagged)."""
    found = []
    cfg_id = getattr(cfg, "GATEWAY_ENGINE_ID", "")
    try:
        for pe in _paginate(br_ctrl.list_policy_engines, "policyEngines"):
            eid, name = pe["policyEngineId"], pe.get("name", "")
            ids = {eid} & {cfg_id} if cfg_id else set()
            tagged = _has_project_tag(
                lambda a=pe["policyEngineArn"]: br_ctrl.list_tags_for_resource(resourceArn=a)
            )
            why = _why(name, ids, tagged=tagged)
            if not why:
                continue
            policies = []
            try:
                policies = [
                    {"name": p.get("name", ""), "id": p["policyId"]}
                    for p in _paginate(
                        br_ctrl.list_policy_summaries, "policies", policyEngineId=eid
                    )
                ]
            except Exception as e:  # noqa: BLE001
                print(f"  warning: could not list policies of engine {eid}: {type(e).__name__}")
            found.append({"id": eid, "name": name, "why": why, "policies": policies})
    except Exception as e:  # noqa: BLE001
        print(f"  warning: list_policy_engines failed ({type(e).__name__}); using config")
        if cfg_id:
            found.append({"id": cfg_id, "name": "?", "why": ["config"], "policies": []})
    return found


def _discover_guardrails() -> list[dict]:
    """Bedrock Guardrails.

    ListGuardrails returns one entry per *version*, so dedupe by id --
    delete_guardrail(guardrailIdentifier=<id>) removes every version at once.
    """
    found: dict[str, dict] = {}
    cfg_id = getattr(cfg, "GUARDRAIL_ID", "")
    try:
        for g in _paginate(br.list_guardrails, "guardrails"):
            gid, name = g["id"], g.get("name", "")
            ids = {gid} & {cfg_id} if cfg_id else set()
            # ListGuardrails summaries DO carry `arn` -- no extra Get needed.
            # NOTE the capital "ARN": `bedrock` is the one service here that
            # spells this parameter resourceARN.  See _has_project_tag.
            tagged = _has_project_tag(lambda a=g["arn"]: br.list_tags_for_resource(resourceARN=a))
            why = _why(name, ids, tagged=tagged)
            if why:
                found[gid] = {"id": gid, "name": name, "why": why}
    except Exception as e:  # noqa: BLE001
        print(f"  warning: list_guardrails failed ({type(e).__name__}); using config")
        if cfg_id:
            found[cfg_id] = {"id": cfg_id, "name": "?", "why": ["config"]}
    return list(found.values())


def _discover_knowledge_bases() -> list[dict]:
    """Knowledge Bases, each with its data sources (data sources cannot be tagged).

    A KB in DELETE_UNSUCCESSFUL is still listed, which is how the zombie in this
    account was eventually found -- so status is carried through to the report.
    """
    found = []
    cfg_id = getattr(cfg, "KB_ID", "")
    try:
        for kb in _paginate(agent.list_knowledge_bases, "knowledgeBaseSummaries"):
            kid, name = kb["knowledgeBaseId"], kb.get("name", "")
            ids = {kid} & {cfg_id} if cfg_id else set()
            # KB summaries carry no ARN (verified 2026-09-22), so a tag check
            # costs one Get.
            try:
                arn = agent.get_knowledge_base(knowledgeBaseId=kid)["knowledgeBase"][
                    "knowledgeBaseArn"
                ]
                tagged = _has_project_tag(lambda a=arn: agent.list_tags_for_resource(resourceArn=a))
            except Exception as e:  # noqa: BLE001
                print(f"  warning: could not read knowledge base {kid} ({type(e).__name__})")
                tagged = False
            why = _why(name, ids, tagged=tagged)
            if not why:
                continue
            sources = []
            try:
                sources = [
                    {"name": d.get("name", ""), "id": d["dataSourceId"]}
                    for d in _paginate(
                        agent.list_data_sources, "dataSourceSummaries", knowledgeBaseId=kid
                    )
                ]
            except Exception as e:  # noqa: BLE001
                print(f"  warning: could not list data sources of KB {kid}: {type(e).__name__}")
            found.append(
                {
                    "id": kid,
                    "name": name,
                    "status": kb.get("status", "?"),
                    "why": why,
                    "sources": sources,
                }
            )
    except Exception as e:  # noqa: BLE001
        print(f"  warning: list_knowledge_bases failed ({type(e).__name__}); using config")
        if cfg_id:
            found.append(
                {"id": cfg_id, "name": "?", "status": "?", "why": ["config"], "sources": []}
            )
    return found


def _discover_vector_buckets() -> list[dict]:
    """S3 Vectors buckets, each with its indexes.

    Both take tags at create and both support TagResource, but their names are
    derived from config so the name backstop catches the -v2/-v3 generations too.
    """
    found = []
    cfg_bucket = getattr(cfg, "VECTOR_BUCKET_NAME", "")
    try:
        for vb in _paginate(s3v.list_vector_buckets, "vectorBuckets"):
            name = vb["vectorBucketName"]
            ids = {name} & {cfg_bucket} if cfg_bucket else set()
            tagged = _has_project_tag(
                lambda a=vb["vectorBucketArn"]: s3v.list_tags_for_resource(resourceArn=a)
            )
            why = _why(name, ids, tagged=tagged)
            if not why:
                continue
            indexes = []
            try:
                indexes = [
                    ix["indexName"]
                    for ix in _paginate(s3v.list_indexes, "indexes", vectorBucketName=name)
                ]
            except Exception as e:  # noqa: BLE001
                print(f"  warning: could not list indexes in {name}: {type(e).__name__}")
            found.append({"name": name, "why": why, "indexes": indexes})
    except Exception as e:  # noqa: BLE001
        print(f"  warning: list_vector_buckets failed ({type(e).__name__}); using config")
        if cfg_bucket:
            cfg_index = getattr(cfg, "VECTOR_INDEX_NAME", "")
            found.append(
                {
                    "name": cfg_bucket,
                    "why": ["config"],
                    "indexes": [cfg_index] if cfg_index else [],
                }
            )
    return found


def _discover_iam_roles() -> list[dict]:
    """IAM roles created by build_kb.py.

    Name prefix only, deliberately.  build_kb.py tags both roles, but there is no
    account-wide "roles with this tag" call -- ListRoles does not return tags, so
    a tag sweep would mean one ListRoleTags per role (386 of them in the account
    this was written against).  The roles' names are ours to choose and both
    start with "inside-the-lines", so the prefix is the right handle here; the
    tag is still applied for cost allocation and for a human reading the console.
    """
    found = []
    cfg_role = getattr(cfg, "KB_ROLE_NAME", "")
    wanted = {cfg_role, "inside-the-lines-gateway-role"} - {""}
    try:
        # ListRoles is the existence check too: a config-named role that does not
        # come back simply is not there, so there is nothing to seed separately.
        for r in _paginate_iam_roles():
            name = r["RoleName"]
            why = _why(name, {name} & wanted)
            if why:
                found.append({"name": name, "why": why})
    except Exception as e:  # noqa: BLE001
        print(f"  warning: list_roles failed ({type(e).__name__}); using config")
        found = [{"name": n, "why": ["config"]} for n in sorted(wanted)]
    return found


def _paginate_iam_roles():
    """IAM pages with Marker/IsTruncated, not nextToken -- hence its own helper."""
    paginator = iam.get_paginator("list_roles")
    for page in paginator.paginate():
        yield from page["Roles"]


def _discover_s3_buckets() -> list[dict]:
    """The S3 corpus bucket(s).

    Two signals, because the corpus bucket is the only resource here whose name
    the project does not control:
      - name prefix / config match over list_buckets()
      - Project=inside-the-lines via the Resource Groups Tagging API, which
        answers "which S3 buckets carry this tag" in a single call.  build_kb.py
        merges that tag onto cfg.BUCKET (tag_corpus_bucket).

    Note the blast radius: anything carrying that tag will be emptied and
    deleted.  That is the same contract as the old config path (which deleted
    cfg.BUCKET outright), just no longer blind to a rename.
    """
    found: dict[str, list[str]] = {}
    cfg_bucket = getattr(cfg, "BUCKET", "")
    try:
        for b in s3.list_buckets().get("Buckets", []):
            name = b["Name"]
            ids = {name} & {cfg_bucket} if cfg_bucket else set()
            why = _why(name, ids)
            if why:
                found[name] = why
    except Exception as e:  # noqa: BLE001
        print(f"  warning: list_buckets failed ({type(e).__name__}); using config")
        if cfg_bucket:
            found[cfg_bucket] = ["config"]
    try:
        page_token = None
        while True:
            kwargs = {
                "ResourceTypeFilters": ["s3"],
                "TagFilters": [{"Key": PROJECT_TAG_KEY, "Values": [PROJECT_TAG_VALUE]}],
            }
            if page_token:
                kwargs["PaginationToken"] = page_token
            resp = rgt.get_resources(**kwargs)
            for m in resp.get("ResourceTagMappingList", []):
                arn = m["ResourceARN"]
                # S3 bucket ARNs are arn:aws:s3:::<bucket>.  Access-grant and
                # other s3:* ARNs also come back under the "s3" filter, so skip
                # anything with a path segment -- we only ever delete buckets.
                name = arn.split(":")[-1]
                if "/" in name:
                    continue
                found.setdefault(name, []).append("tag")
            page_token = resp.get("PaginationToken")
            if not page_token:
                break
    except Exception as e:  # noqa: BLE001
        print(f"  warning: tag search for S3 buckets failed ({type(e).__name__})")
    return [{"name": n, "why": w} for n, w in sorted(found.items())]


def discover() -> dict:
    """Return everything in this account+region that belongs to this project.

    Read-only.  Used three times: to print the --dry-run plan, to drive the
    deletions, and to verify afterwards that nothing survived.  One function for
    all three is the point -- a category that the sweep cannot see is a category
    verification cannot see either, so the two can never drift apart.
    """
    return {
        "gateways": _discover_gateways(),
        "policy_engines": _discover_policy_engines(),
        "guardrails": _discover_guardrails(),
        "knowledge_bases": _discover_knowledge_bases(),
        "vector_buckets": _discover_vector_buckets(),
        "iam_roles": _discover_iam_roles(),
        "s3_buckets": _discover_s3_buckets(),
    }


def is_empty(found: dict) -> bool:
    """True if discover() found nothing at all."""
    return not any(found.values())


def print_plan(found: dict) -> None:
    """Print the discovered inventory, grouped, with why each item was claimed."""
    for gw in found["gateways"]:
        print(f"  gateway            {gw['name']} ({gw['id']})  [{', '.join(gw['why'])}]")
        for t in gw["targets"]:
            print(f"    target           {t['name']} ({t['id']})  [child of gateway]")
    for pe in found["policy_engines"]:
        print(f"  policy engine      {pe['name']} ({pe['id']})  [{', '.join(pe['why'])}]")
        for p in pe["policies"]:
            print(f"    Cedar policy     {p['name']} ({p['id']})  [child of engine]")
    for g in found["guardrails"]:
        print(f"  guardrail          {g['name']} ({g['id']})  [{', '.join(g['why'])}]")
    for kb in found["knowledge_bases"]:
        print(
            f"  knowledge base     {kb['name']} ({kb['id']}) status={kb['status']}"
            f"  [{', '.join(kb['why'])}]"
        )
        for d in kb["sources"]:
            print(
                f"    data source      {d['name']} ({d['id']})"
                "  [child of KB; dataDeletionPolicy -> RETAIN first]"
            )
    for vb in found["vector_buckets"]:
        print(f"  vector bucket      {vb['name']}  [{', '.join(vb['why'])}]")
        for ix in vb["indexes"]:
            print(f"    vector index     {ix}  [child of vector bucket]")
    for r in found["iam_roles"]:
        print(f"  IAM role           {r['name']}  [{', '.join(r['why'])}]")
    for b in found["s3_buckets"]:
        print(
            f"  S3 corpus bucket   {b['name']}  (and every object in it)  [{', '.join(b['why'])}]"
        )
    if is_empty(found):
        print("  (nothing found -- this account is already clean)")


# ----------------------------------------------------------------------
# Deletion
# ----------------------------------------------------------------------
def _delete_bucket(bucket: str) -> None:
    """Delete all objects in an S3 bucket, then delete the bucket itself.

    S3 requires a bucket to be empty before it can be deleted.  Versioned
    buckets also need their delete markers and old versions removed, hence the
    second paginator -- corpus_fetch.py does not enable versioning, but a bucket
    you made yourself might have it on.
    """
    for page in s3.get_paginator("list_object_versions").paginate(Bucket=bucket):
        stale = [
            {"Key": o["Key"], "VersionId": o["VersionId"]}
            for o in page.get("Versions", []) + page.get("DeleteMarkers", [])
        ]
        for i in range(0, len(stale), 1000):  # delete_objects caps at 1000 keys
            s3.delete_objects(Bucket=bucket, Delete={"Objects": stale[i : i + 1000]})
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket):
        keys = [{"Key": o["Key"]} for o in page.get("Contents", [])]
        for i in range(0, len(keys), 1000):
            s3.delete_objects(Bucket=bucket, Delete={"Objects": keys[i : i + 1000]})
    s3.delete_bucket(Bucket=bucket)


def delete_gateway(gw: dict) -> None:
    """Drain targets, detach the policy engine, delete the gateway, wait for gone."""
    gid = gw["id"]
    for t in gw["targets"]:
        _try(
            f"gateway target {t['name']} ({t['id']})",
            lambda tid=t["id"]: br_ctrl.delete_gateway_target(gatewayIdentifier=gid, targetId=tid),
        )
    if gw["targets"]:
        # Target deletion is async.  delete_gateway() fails with
        # ConflictException while any target is still DELETING, so wait for the
        # list to drain.  Bounded, and a timeout is not fatal -- the gateway
        # delete below will simply report its own skip.
        for _ in range(20):
            try:
                if not list(
                    _paginate(br_ctrl.list_gateway_targets, "items", gatewayIdentifier=gid)
                ):
                    break
            except Exception:  # noqa: BLE001 -- gateway may already be gone
                break
            time.sleep(3)

    # delete_gateway() returns ValidationException while a policy engine is
    # attached, so detach first.  update_gateway() is a full replacement: omit
    # policyEngineConfiguration to clear it, but repeat everything else.
    try:
        detail = br_ctrl.get_gateway(gatewayIdentifier=gid)
        if detail.get("policyEngineConfiguration"):
            br_ctrl.update_gateway(
                gatewayIdentifier=gid,
                name=detail["name"],
                roleArn=detail["roleArn"],
                authorizerType=detail["authorizerType"],
            )
            for _ in range(10):
                if br_ctrl.get_gateway(gatewayIdentifier=gid)["status"] == "READY":
                    break
                time.sleep(3)
    except Exception as e:  # noqa: BLE001
        print(f"  skip (detach policy engine from gateway {gid}): {type(e).__name__}")

    _try(f"gateway {gw['name']} ({gid})", lambda: br_ctrl.delete_gateway(gatewayIdentifier=gid))

    # The API returns immediately but the gateway lingers in DELETING, and
    # deleting its policy engine while it exists returns ConflictException.
    for _ in range(30):
        try:
            if br_ctrl.get_gateway(gatewayIdentifier=gid)["status"] not in (
                "DELETING",
                "DELETE_UNSUCCESSFUL",
            ):
                break
        except Exception:  # noqa: BLE001
            break  # gone
        time.sleep(5)


def delete_policy_engine(pe: dict) -> None:
    """Delete every Cedar policy in the engine, then the engine."""
    eid = pe["id"]
    for p in pe["policies"]:
        _try(
            f"Cedar policy {p['name']} ({p['id']})",
            lambda pid=p["id"]: br_ctrl.delete_policy(policyEngineId=eid, policyId=pid),
        )
    if pe["policies"]:
        # Policy deletes are async and the engine cannot go while any remain.
        for _ in range(20):
            try:
                if not list(
                    _paginate(br_ctrl.list_policy_summaries, "policies", policyEngineId=eid)
                ):
                    break
            except Exception:  # noqa: BLE001
                break
            time.sleep(3)
    _try(
        f"policy engine {pe['name']} ({eid})",
        lambda: br_ctrl.delete_policy_engine(policyEngineId=eid),
    )


def retain_data_sources(kb: dict) -> None:
    """Set dataDeletionPolicy=RETAIN on every data source of a KB.

    THIS IS THE FIX FOR DELETE_UNSUCCESSFUL (verified 2026-09-22).  With the
    default policy of DELETE, delete_knowledge_base() tries to purge the vectors
    from the underlying store first; if that store is already gone -- which is
    exactly what happens when a previous teardown deleted the S3 Vectors index
    before the KB, or when the whole vector bucket was cleaned up by hand -- the
    KB can never finish deleting and parks in DELETE_UNSUCCESSFUL forever.  One
    such KB sat in this account from June to September.  Flipping to RETAIN
    tells Bedrock not to touch the vector store, and the delete goes through.
    We delete the index ourselves immediately afterwards, so nothing leaks.

    update_data_source() is a FULL REPLACEMENT, not a patch.  `name` and
    `dataSourceConfiguration` are required, and `vectorIngestionConfiguration`
    must be passed back unchanged or the call fails with:
        ValidationException: vectorIngestionConfiguration.chunkingConfiguration
        cannot be updated once created
    So read the current data source and echo everything back.
    """
    for d in kb["sources"]:
        ds_id = d["id"]
        try:
            cur = agent.get_data_source(knowledgeBaseId=kb["id"], dataSourceId=ds_id)["dataSource"]
        except Exception as e:  # noqa: BLE001
            print(f"  skip (read data source {ds_id}): {type(e).__name__}")
            continue
        if cur.get("dataDeletionPolicy") == "RETAIN":
            print(f"  data source {d['name']} ({ds_id}) already RETAIN")
            continue
        kwargs = {
            "knowledgeBaseId": kb["id"],
            "dataSourceId": ds_id,
            "name": cur["name"],
            "dataSourceConfiguration": cur["dataSourceConfiguration"],
            "dataDeletionPolicy": "RETAIN",
        }
        # Echo back the optional members only when present: sending an empty
        # dict for one of these is a validation error, not a no-op.
        for opt in (
            "description",
            "serverSideEncryptionConfiguration",
            "vectorIngestionConfiguration",
        ):
            if cur.get(opt):
                kwargs[opt] = cur[opt]
        _try(
            f"data source {d['name']} ({ds_id}) -> dataDeletionPolicy=RETAIN",
            lambda kw=kwargs: agent.update_data_source(**kw),
        )


def delete_knowledge_base(kb: dict) -> None:
    """RETAIN every data source, delete the KB, then wait for it to disappear.

    The wait matters as much as the ordering: if the S3 Vectors index goes while
    the KB is still cleaning up, the KB fails and sticks in DELETE_UNSUCCESSFUL
    -- recreating the exact zombie this function exists to avoid.
    """
    retain_data_sources(kb)
    kid = kb["id"]
    _try(
        f"knowledge base {kb['name']} ({kid})",
        lambda: agent.delete_knowledge_base(knowledgeBaseId=kid),
    )
    print("  waiting for KB deletion to complete...")
    for _ in range(60):  # up to 5 minutes
        try:
            status = agent.get_knowledge_base(knowledgeBaseId=kid)["knowledgeBase"]["status"]
        except Exception:  # noqa: BLE001 -- ResourceNotFound means it is gone
            print("  KB deletion confirmed")
            return
        if status == "DELETING":
            time.sleep(5)
            continue
        if status == "DELETE_UNSUCCESSFUL":
            print(
                "  warning: KB stuck in DELETE_UNSUCCESSFUL even with RETAIN set.  "
                "Re-run teardown.py; the final verification below will keep failing "
                "until it clears."
            )
        return


def delete_vector_bucket(vb: dict) -> None:
    """Delete every index in the vector bucket, then the bucket.

    A vector bucket cannot be deleted while it holds indexes, and an index
    cannot be deleted while a KB still references it -- hence KBs first.
    """
    for ix in vb["indexes"]:
        _try(
            f"vector index {vb['name']}/{ix}",
            lambda name=ix: s3v.delete_index(vectorBucketName=vb["name"], indexName=name),
        )
    _try(
        f"vector bucket {vb['name']}",
        lambda: s3v.delete_vector_bucket(vectorBucketName=vb["name"]),
    )


def delete_iam_role(name: str) -> None:
    """Detach inline + managed policies, then delete the role."""

    def _drop():
        for p in _paginate_iam_role_policies(name):
            iam.delete_role_policy(RoleName=name, PolicyName=p)
        for arn in _paginate_iam_attached(name):
            iam.detach_role_policy(RoleName=name, PolicyArn=arn)
        iam.delete_role(RoleName=name)

    _try(f"IAM role {name}", _drop)


def _paginate_iam_role_policies(role: str):
    for page in iam.get_paginator("list_role_policies").paginate(RoleName=role):
        yield from page["PolicyNames"]


def _paginate_iam_attached(role: str):
    for page in iam.get_paginator("list_attached_role_policies").paginate(RoleName=role):
        yield from [p["PolicyArn"] for p in page["AttachedPolicies"]]


def delete_all(found: dict) -> None:
    """Delete everything in `found`, in the one order that works.

    Order rationale, in full, because getting it wrong is how the zombie KB and
    the three orphaned vector buckets happened:
      1. Gateways first.  A gateway holds targets and references a policy
         engine, so nothing downstream can go until it is drained and gone.
      2. Policy engines next, now that no gateway references them.
      3. Guardrails -- independent of everything else.
      4. Knowledge bases, each with RETAIN set on its data sources first, and
         each waited out to gone.
      5. Only THEN the S3 Vectors indexes and buckets the KBs were pointing at.
      6. IAM roles, after the resources that assume them.
      7. The S3 corpus bucket, then the local corpus/ copy.
    """
    print("\n--- gateways + targets ---")
    for gw in found["gateways"]:
        delete_gateway(gw)

    print("\n--- Cedar policies + policy engines ---")
    for pe in found["policy_engines"]:
        delete_policy_engine(pe)

    print("\n--- guardrails ---")
    for g in found["guardrails"]:
        _try(
            f"guardrail {g['name']} ({g['id']})",
            lambda gid=g["id"]: br.delete_guardrail(guardrailIdentifier=gid),
        )

    print("\n--- knowledge bases (data sources -> RETAIN first) ---")
    for kb in found["knowledge_bases"]:
        delete_knowledge_base(kb)

    print("\n--- S3 Vectors indexes + buckets ---")
    for vb in found["vector_buckets"]:
        delete_vector_bucket(vb)

    print("\n--- IAM roles ---")
    for r in found["iam_roles"]:
        delete_iam_role(r["name"])

    print("\n--- S3 corpus buckets ---")
    # WARNING: this deletes all the papers you downloaded with corpus_fetch.py.
    # Re-run corpus_fetch.py + aws s3 sync to rebuild if needed.
    for b in found["s3_buckets"]:
        _try(f"S3 corpus bucket {b['name']}", lambda name=b["name"]: _delete_bucket(name))

    print("\n--- local files ---")
    corpus_dir = "corpus"
    if os.path.isdir(corpus_dir):
        shutil.rmtree(corpus_dir)
        print(f"  deleted: local {corpus_dir}/ directory")
    else:
        print(f"  skip (local {corpus_dir}/): not present")


def verify(attempts: int = 6, pause: int = 10) -> bool:
    """Re-discover and report anything that survived.  True means clean.

    Retried rather than checked once: several of these deletes are async, so a
    single immediate pass would flag resources that are merely mid-DELETING and
    fail an otherwise-perfect teardown.  A resource still present after ~1 minute
    is a real survivor and the caller exits non-zero on it.
    """
    for attempt in range(1, attempts + 1):
        found = discover()
        if is_empty(found):
            print(
                f"\nVERIFIED CLEAN: nothing in {cfg.REGION} carries "
                f'{PROJECT_TAG_KEY}={PROJECT_TAG_VALUE} or a "{NAME_PREFIX}" name.'
            )
            return True
        if attempt < attempts:
            print(f"\n  still settling (attempt {attempt}/{attempts}); waiting {pause}s...")
            time.sleep(pause)
            continue
        print("\nNOT CLEAN -- these still exist:")
        print_plan(found)
        print(
            "\nRe-run `python teardown.py` (it is idempotent).  A Knowledge Base in\n"
            "DELETE_UNSUCCESSFUL usually clears on a second pass now that\n"
            "dataDeletionPolicy is set to RETAIN before the delete."
        )
        return False
    return False


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Delete every AWS resource the Inside the Lines demo created."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="list what would be deleted and delete nothing (read-only: only "
        "list/get calls are made)",
    )
    args = parser.parse_args()

    banner = "DRY RUN -- nothing will be deleted" if args.dry_run else "deleting"
    print(f"Inside the Lines teardown -- region {cfg.REGION} -- {banner}\n")
    print("discovered (config = named in config.py, tag = Project tag, name = name prefix):")
    found = discover()
    print_plan(found)

    if args.dry_run:
        print("\nDry run complete.  Nothing was changed.")
        # Exit 0 whether or not anything was found: --dry-run reports, it does
        # not assert.  The post-deletion verify() is what gates the exit code.
        return 0

    delete_all(found)
    return 0 if verify() else 1


if __name__ == "__main__":
    sys.exit(main())
