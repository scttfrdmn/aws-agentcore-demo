#!/usr/bin/env python3
"""
build_kb.py  --  one-time, fully scripted Knowledge Base provisioning.

Run this script ONCE before the talk to create every AWS resource the demo
needs.  It asks you for nothing and it leaves you nothing to edit: when it
finishes it WRITES the resulting IDs straight into config.py (backing the file up
first) and prints them so you can see what changed.

Nothing here needs config.py to exist beforehand either.  The first thing this
module does is bootstrap.ensure_config(), which generates config.py from the
tracked config.example.py, fills ACCOUNT_ID from STS and derives a
globally-unique BUCKET name -- so `git clone && make build-kb` works.

Everything it creates is tagged Project=inside-the-lines (see PROJECT_TAGS
below).  That tag is what lets teardown.py find resources config.py no longer
names -- read the comment above PROJECT_TAG_KEY for the orphan story it prevents.

What this script creates (in order):
  1. S3 corpus bucket + the corpus in it  -- created if absent (idempotently,
                  and with the us-east-1 LocationConstraint special case
                  handled), tagged so teardown can find it after a rename, and
                  filled from the local corpus/ directory by a parallel,
                  resumable upload.  This step used to be three manual actions
                  in the README: invent a unique bucket name, create it, and
                  `aws s3 sync` into it.
  2. IAM role  -- a service role that Bedrock assumes to read S3 and write
                  to the vector store on your behalf.
  3. S3 Vectors bucket + index  -- the serverless vector store.  No hourly
                  charge; you pay only for storage and query calls.
  4. Bedrock Knowledge Base + S3 data source  -- connects the KB to your S3
                  corpus bucket so Bedrock can find the papers.
  5. Ingestion job  -- reads every paper from S3, splits it into 512-token
                  chunks, embeds each chunk with Titan Embed v2, and writes
                  the vectors into the S3 Vectors index.  Takes ~5 minutes.
  6. Bedrock Guardrail  -- a regex policy that intercepts any https:// URL
                  in model output and replaces it with a local corpus link.
  7. AgentCore Gateway + "web-tools" target + Cedar policy engine  --
                  demonstrates policy-based tool access control.  The target is
                  an OpenAPI target over the public ClinicalTrials.gov v2 API
                  that publishes a real ``web_fetch`` tool (no Lambda, nothing
                  to own); the Cedar "ForbidWeb" policy then denies the
                  web_fetch call that Q5 attempts.  The tool being real is the
                  point: lift the policy and the call genuinely succeeds.

Re-running safely (idempotent):
  All seven steps use get-or-create logic -- if a resource with the expected
  name already exists, the script returns its existing ID instead of creating
  a duplicate.  You can run this script multiple times safely; it will print
  "exists" for steps that are already done and only do real work for steps
  that are missing.  This means you can also re-run after a partial failure.
  Every "exists" branch also re-applies the project tag (see _adopt), so a
  resource created by an older, untagged version of this script gets adopted
  rather than silently left undiscoverable by teardown.py.

S3 Vectors API shapes -- VERIFIED 2026-09-22
S3 Vectors is a recent service (GA Dec 2025), so its shapes used to carry
``# VERIFY`` flags.  They are now confirmed against the installed botocore,
which ships the service models locally -- no AWS call or credentials needed:

    python -c "from botocore.session import get_session as g; \\
      print(sorted(g().get_service_model('s3vectors').operation_names))"

That lists CreateVectorBucket / CreateIndex / ListVectors, and
bedrock-agent's CreateKnowledgeBase storageConfiguration accepts
type='S3_VECTORS' with s3VectorsConfiguration.{vectorBucketArn,indexArn,
indexName}.  Re-run that one-liner if you ever see UnknownServiceException
or a validation error; it is the cheapest possible check.
tests/test_sdk_contract.py guards the service and client names in CI.

What to do if something goes wrong:
  - "EntityAlreadyExistsException" on the IAM role: harmless, continuing.
  - "ResourceConflictException" on the vector bucket/index: harmless.
  - Ingestion job status "FAILED": read the failure reasons from the API --
    aws bedrock-agent list-ingestion-jobs --knowledge-base-id <KB_ID> \\
      --data-source-id <DS_ID> --region us-west-2
    then get-ingestion-job for the failed jobId; failureReasons says why
    to see which documents failed (usually large PDFs or XML parse errors).
  - 400 / ValidationException on create_knowledge_base: the storageConfiguration
    shape changed -- re-run the botocore one-liner above.
  - "AccessDeniedException" anywhere: your IAM user/role is missing permissions;
    see the trust policy and inline policy in create_kb_role().

Requires: boto3 (installed via uv pip install -e ".[dev]")
"""

import json
import time

import boto3

import bootstrap

# ORDER IS LOAD-BEARING.  ensure_config() is what CREATES config.py on a fresh
# clone (from the tracked config.example.py, with ACCOUNT_ID from STS and a
# derived BUCKET), so it has to run before `import config` can possibly succeed.
# Without this line a stranger's first `make build-kb` ends in
# `ModuleNotFoundError: No module named 'config'` -- a traceback instead of an
# instruction.  It is idempotent: on an existing config.py it fills only values
# that are blank or still the shipped placeholder, and never overwrites a real
# one.
bootstrap.ensure_config()

import config as cfg  # noqa: E402 -- must follow ensure_config(); see above

# All three clients share the region from config.py.
iam = boto3.client("iam")
agent = boto3.client("bedrock-agent", region_name=cfg.REGION)

# "s3vectors" confirmed as the boto3 service name (2026-09-22); guarded by
# tests/test_sdk_contract.py so a rename fails CI rather than the live demo.
s3v = boto3.client("s3vectors", region_name=cfg.REGION)

# Used by ensure_corpus_bucket() -- create, tag, and fill the corpus bucket.
# Pinned to cfg.REGION on purpose: create_bucket's LocationConstraint and the
# client's own region have to agree, and the KB pays cross-Region transfer on
# ingestion if the corpus lands anywhere other than the KB's region.
s3 = boto3.client("s3", region_name=cfg.REGION)


# ----------------------------------------------------------------------
# 0. Resource tagging -- the thing that makes teardown.py able to find
#    resources that config.py no longer names.
# ----------------------------------------------------------------------
# WHY THIS EXISTS (2026-09-22).  teardown.py used to delete exactly what the
# *current* config.py happened to name: cfg.KB_ID, cfg.VECTOR_BUCKET_NAME,
# cfg.GUARDRAIL_ID and friends.  So any rename between a build and a teardown
# orphaned the previous generation forever.  That was not theoretical: one demo
# account was found holding three S3 vector buckets (plain, -v2, -v3), a
# Knowledge Base stuck in DELETE_UNSUCCESSFUL for months, two stray Cedar policy
# engines with five policies between them, a guardrail named
# "inside-the-lines-guardrail-v3" that build_kb.py could never adopt (it
# get-or-creates "inside-the-lines-url-filter"), and two stray IAM roles.
#
# Fix: every resource created here is stamped with one tag, and teardown.py
# sweeps for that tag in addition to its config fast path.  Name prefixes are
# the backstop for the handful of resources AWS will not let us tag at all.
#
# NOTE: these two constants are duplicated in teardown.py.  That is deliberate.
# Importing them from here would be DRY-er, but it would also make `teardown.py`
# fail at import time on an old botocore (this module builds an "s3vectors"
# client at import), and teardown must keep working even when the build script
# cannot run.  Two files, one grep: `grep -rn PROJECT_TAG_ *.py`.
PROJECT_TAG_KEY = "Project"
PROJECT_TAG_VALUE = "inside-the-lines"

# Most of the services here take tags as a {key: value} MAP: s3vectors,
# bedrock-agent, bedrock-agentcore-control.  Two do not -- see the helpers
# below.  (All shapes read straight out of the installed botocore service
# models on 2026-09-22; no credentials needed:
#   python -c "from botocore.session import get_session as g; \
#     m=g().get_service_model('s3vectors'); \
#     print(m.operation_model('CreateVectorBucket').input_shape.members['tags'])"
# )
PROJECT_TAGS = {PROJECT_TAG_KEY: PROJECT_TAG_VALUE}

# `bedrock` (guardrails) takes a LIST of {"key", "value"} -- lowercase keys.
PROJECT_TAGS_KV = [{"key": PROJECT_TAG_KEY, "value": PROJECT_TAG_VALUE}]

# `iam` takes a LIST of {"Key", "Value"} -- capitalised, because IAM predates
# the modern tagging conventions.
PROJECT_TAGS_IAM = [{"Key": PROJECT_TAG_KEY, "Value": PROJECT_TAG_VALUE}]


def _adopt(label: str, fn) -> None:
    """Best-effort tag of a resource that already existed before this run.

    Every create call below passes tags inline, which covers the normal path.
    But build_kb.py is get-or-create: on a re-run, or against a resource made by
    an older version of this script, the create is skipped and the tag would
    never be applied -- leaving exactly the untaggable orphan we are trying to
    prevent.  So each "exists" branch also calls TagResource through here.

    Failures are printed, not raised: a missing tag degrades teardown to its
    name-prefix backstop, which is not worth aborting provisioning over.
    """
    try:
        fn()
    except Exception as e:  # noqa: BLE001 -- tagging is best-effort by design
        print(f"    (could not tag {label}: {type(e).__name__})")


# ----------------------------------------------------------------------
# 1. IAM role for the knowledge base
# ----------------------------------------------------------------------
def create_kb_role() -> str:
    """Create (or reuse) the IAM role that the Bedrock Knowledge Base assumes.

    The role needs three permissions:
      - s3:GetObject / s3:ListBucket on your corpus bucket (to read papers)
      - bedrock:InvokeModel on the embedding model (to embed chunks)
      - s3vectors:* (to write and read vectors)

    AWS IAM propagation takes ~10 seconds after role creation; we sleep to
    avoid a ConditionFailure when create_knowledge_base is called right after.

    Returns:
        The role ARN (e.g. "arn:aws:iam::123456789012:role/inside-the-lines-kb-role").
    """
    # Trust policy: only the bedrock.amazonaws.com service can assume this role.
    trust = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Principal": {"Service": "bedrock.amazonaws.com"},
                "Action": "sts:AssumeRole",
            }
        ],
    }
    try:
        arn = iam.create_role(
            RoleName=cfg.KB_ROLE_NAME,
            AssumeRolePolicyDocument=json.dumps(trust),
            Description="Inside the Lines demo -- Bedrock KB execution role",
            Tags=PROJECT_TAGS_IAM,
        )["Role"]["Arn"]
    except iam.exceptions.EntityAlreadyExistsException:
        # Role was created on a previous run -- that's fine, reuse it.
        arn = iam.get_role(RoleName=cfg.KB_ROLE_NAME)["Role"]["Arn"]
        # ...but re-stamp the tag, in case the role predates tagging.
        _adopt(
            f"IAM role {cfg.KB_ROLE_NAME}",
            lambda: iam.tag_role(RoleName=cfg.KB_ROLE_NAME, Tags=PROJECT_TAGS_IAM),
        )

    # Inline policy: grant the minimum permissions Bedrock needs.
    # s3vectors:* is broad -- narrow to s3vectors:PutVectors / GetVectors /
    # ListVectors once you confirm the exact action names in the IAM reference.
    policy = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Action": ["s3:GetObject", "s3:ListBucket"],
                "Resource": [f"arn:aws:s3:::{cfg.BUCKET}", f"arn:aws:s3:::{cfg.BUCKET}/*"],
            },
            {
                # Bedrock must call the embedding model to vectorise each chunk.
                "Effect": "Allow",
                "Action": "bedrock:InvokeModel",
                "Resource": (
                    f"arn:aws:bedrock:{cfg.REGION}::foundation-model/{cfg.EMBED_MODEL_ID}"
                ),
            },
            # VERIFY: scope s3vectors actions per current S3 Vectors IAM reference.
            # As of 2026-05-20, "s3vectors:*" is the safe catch-all.
            {"Effect": "Allow", "Action": "s3vectors:*", "Resource": "*"},
        ],
    }
    iam.put_role_policy(
        RoleName=cfg.KB_ROLE_NAME, PolicyName="kb-access", PolicyDocument=json.dumps(policy)
    )
    print(f"  IAM role: {arn}")

    # IAM roles take ~10 seconds to propagate globally.  If create_knowledge_base
    # is called immediately, it may fail with "Role cannot be assumed."
    time.sleep(10)
    return arn


# ----------------------------------------------------------------------
# 2. S3 Vectors bucket + index
# ----------------------------------------------------------------------
def create_vector_store() -> str:
    """Create an S3 Vectors bucket and a vector index inside it.

    S3 Vectors is the vector store backing the Bedrock Knowledge Base.
    It is serverless -- no hourly charge, no OpenSearch collection.
    You pay per GB stored and per 1,000 query calls.

    The ``nonFilterableMetadataKeys`` configuration is critical:
    Bedrock KB attaches several metadata fields to each vector during
    ingestion.  Some of these fields (e.g. AMAZON_BEDROCK_TEXT, which
    holds the raw chunk text) are large enough to push the filterable
    metadata payload over S3 Vectors' 2,048-byte limit, causing ingestion
    to fail with a validation error.  Marking them non-filterable excludes
    them from the index metadata while still storing them on the vector
    object -- so the KB can still retrieve the text.
    (Verified 2026-05-20.)

    Returns:
        The ARN of the created vector index.
    """
    # The two ARNs are deterministic from (region, account, names), so we can
    # build them up front and use them for tag adoption below without an extra
    # Get call.  Verified against the live account 2026-09-22:
    #   arn:aws:s3vectors:us-west-2:<acct>:bucket/<bucket>
    #   arn:aws:s3vectors:us-west-2:<acct>:bucket/<bucket>/index/<index>
    bucket_arn = f"arn:aws:s3vectors:{cfg.REGION}:{cfg.ACCOUNT_ID}:bucket/{cfg.VECTOR_BUCKET_NAME}"
    index_arn = f"{bucket_arn}/index/{cfg.VECTOR_INDEX_NAME}"

    # create_vector_bucket / create_index confirmed present in the installed
    # botocore s3vectors service model (2026-09-22), both accepting a `tags` map.
    try:
        s3v.create_vector_bucket(vectorBucketName=cfg.VECTOR_BUCKET_NAME, tags=PROJECT_TAGS)
        print(f"  vector bucket: {cfg.VECTOR_BUCKET_NAME}")
    except Exception as e:  # noqa: BLE001 -- already-exists is fine
        # ResourceConflictException = bucket already exists from a previous run.
        print(f"  vector bucket: {type(e).__name__} (continuing)")
        _adopt(
            f"vector bucket {cfg.VECTOR_BUCKET_NAME}",
            lambda: s3v.tag_resource(resourceArn=bucket_arn, tags=PROJECT_TAGS),
        )

    # All Bedrock KB metadata keys must be non-filterable.  If ANY of these
    # are left filterable, ingestion will fail with a metadata size error.
    _NON_FILTERABLE = [
        "x-amz-bedrock-kb-source-uri",
        "x-amz-bedrock-kb-chunk-id",
        "x-amz-bedrock-kb-data-source-id",
        "x-amz-bedrock-kb-document-id",
        "x-amz-bedrock-kb-index-id",
        "x-amz-bedrock-kb-knowledge-base-id",
        "AMAZON_BEDROCK_TEXT",  # raw chunk text -- large, must be non-filterable
        "AMAZON_BEDROCK_METADATA",  # source metadata -- also large
    ]
    try:
        s3v.create_index(
            vectorBucketName=cfg.VECTOR_BUCKET_NAME,
            indexName=cfg.VECTOR_INDEX_NAME,
            dataType="float32",
            dimension=cfg.EMBED_DIM,  # 1024 for Titan Embed v2
            distanceMetric="cosine",  # cosine similarity for semantic search
            metadataConfiguration={"nonFilterableMetadataKeys": _NON_FILTERABLE},
            tags=PROJECT_TAGS,
        )
        print(f"  vector index: {cfg.VECTOR_INDEX_NAME}")
    except Exception as e:  # noqa: BLE001
        print(f"  vector index: {type(e).__name__} (continuing)")
        _adopt(
            f"vector index {cfg.VECTOR_INDEX_NAME}",
            lambda: s3v.tag_resource(resourceArn=index_arn, tags=PROJECT_TAGS),
        )

    # The index ARN is what create_knowledge_base needs in storageConfiguration.
    return index_arn


# ----------------------------------------------------------------------
# 3. Knowledge base + data source
# ----------------------------------------------------------------------
def create_kb(role_arn: str, index_arn: str) -> tuple[str, str]:
    """Create the Bedrock Knowledge Base and attach an S3 data source.

    The Knowledge Base is the logical container Bedrock uses for retrieval.
    It links three things together:
      - the IAM role (so it can call S3 and the embed model)
      - the vector index (where it stores and searches embeddings)
      - the data source (the S3 prefix where the papers live)

    Chunking strategy: fixed-size, 512 tokens, 15% overlap.  This gives
    Bedrock enough context per chunk without making chunks so large that
    retrieval dilutes relevance.

    Returns:
        (kb_id, data_source_id) -- main() writes both into config.py.

    Raises:
        ValidationException: most likely a storageConfiguration shape mismatch.
            Re-run the botocore one-liner in the module docstring.
    """
    embed_arn = f"arn:aws:bedrock:{cfg.REGION}::foundation-model/{cfg.EMBED_MODEL_ID}"

    # Get-or-create: if a KB with this name already exists, reuse it.
    existing = [
        kb
        for kb in agent.list_knowledge_bases().get("knowledgeBaseSummaries", [])
        if kb["name"] == cfg.KB_NAME
    ]
    if existing:
        kb_id = existing[0]["knowledgeBaseId"]
        print(f"  knowledge base exists: {kb_id}")
        # ListKnowledgeBases summaries carry no ARN (verified against the
        # botocore model 2026-09-22: description/knowledgeBaseId/name/status/
        # updatedAt only), so the adopt path needs one Get to obtain it.
        _adopt(
            f"knowledge base {kb_id}",
            lambda: agent.tag_resource(
                resourceArn=agent.get_knowledge_base(knowledgeBaseId=kb_id)["knowledgeBase"][
                    "knowledgeBaseArn"
                ],
                tags=PROJECT_TAGS,
            ),
        )
    else:
        # storageConfiguration shape confirmed 2026-09-22: type "S3_VECTORS" with
        # s3VectorsConfiguration.indexArn.  All three s3VectorsConfiguration
        # fields are optional in the API model; indexArn alone is sufficient.
        kb = agent.create_knowledge_base(
            name=cfg.KB_NAME,
            roleArn=role_arn,
            knowledgeBaseConfiguration={
                "type": "VECTOR",
                "vectorKnowledgeBaseConfiguration": {"embeddingModelArn": embed_arn},
            },
            storageConfiguration={
                "type": "S3_VECTORS",
                "s3VectorsConfiguration": {"indexArn": index_arn},
            },
            tags=PROJECT_TAGS,
        )
        kb_id = kb["knowledgeBase"]["knowledgeBaseId"]
        print(f"  knowledge base: {kb_id}")

    # Get-or-create the data source (pmc-corpus) within this KB.
    existing_ds = [
        ds
        for ds in agent.list_data_sources(knowledgeBaseId=kb_id).get("dataSourceSummaries", [])
        if ds["name"] == "pmc-corpus"
    ]
    if existing_ds:
        ds_id = existing_ds[0]["dataSourceId"]
        print(f"  data source exists: {ds_id}")
        return kb_id, ds_id

    # The data source tells Bedrock which S3 prefix to read papers from.
    # inclusionPrefixes limits ingestion to corpus/ -- other S3 objects are ignored.
    #
    # CANNOT BE TAGGED (verified 2026-09-22): CreateDataSource takes no `tags`
    # member, and a data source has no ARN of its own -- GetDataSource returns
    # knowledgeBaseId + dataSourceId and nothing ARN-shaped, so there is nothing
    # to hand bedrock-agent's TagResource either.  This costs us nothing: a data
    # source only exists inside a KB, it is deleted with the KB, and teardown.py
    # reaches it by calling list_data_sources() on each KB it has *already*
    # discovered.  Tagging would add no discovery power here.
    ds = agent.create_data_source(
        knowledgeBaseId=kb_id,
        name="pmc-corpus",
        dataSourceConfiguration={
            "type": "S3",
            "s3Configuration": {
                "bucketArn": f"arn:aws:s3:::{cfg.BUCKET}",
                "inclusionPrefixes": [cfg.CORPUS_PREFIX],
            },
        },
        vectorIngestionConfiguration={
            "chunkingConfiguration": {
                "chunkingStrategy": "FIXED_SIZE",
                "fixedSizeChunkingConfiguration": {
                    "maxTokens": 512,  # tokens per chunk
                    "overlapPercentage": 15,  # 15% overlap so sentences don't split cold
                },
            }
        },
    )
    return kb_id, ds["dataSource"]["dataSourceId"]


# ----------------------------------------------------------------------
# 4. Ingestion
# ----------------------------------------------------------------------
def ingest(kb_id: str, ds_id: str) -> None:
    """Start an ingestion job and poll until it completes or fails.

    The ingestion job reads every .txt file under corpus/ in S3, splits each
    into chunks, calls Titan Embed v2 to produce a 1024-float32 vector per
    chunk, and writes all vectors to the S3 Vectors index.

    For the 1,000-paper corpus this typically takes 5--10 minutes.  The script polls
    every 15 seconds and prints the current status.

    Note: the ingestion statistics only report document counts -- there is
    NO token count in the API response.  Ingestion cost is computed in
    estimate_ingestion_cost() from the local corpus character count instead.
    (Verified 2026-05-20.)

    Args:
        kb_id: the Knowledge Base ID returned by create_kb().
        ds_id: the data source ID returned by create_kb().
    """
    job_id = agent.start_ingestion_job(knowledgeBaseId=kb_id, dataSourceId=ds_id)["ingestionJob"][
        "ingestionJobId"
    ]
    print("  ingestion started -- embedding + indexing every paper...")

    while True:
        st = agent.get_ingestion_job(
            knowledgeBaseId=kb_id, dataSourceId=ds_id, ingestionJobId=job_id
        )["ingestionJob"]
        print(f"    status: {st['status']}")

        if st["status"] == "COMPLETE":
            stats = st.get("statistics", {})
            indexed = stats.get("numberOfNewDocumentsIndexed")
            scanned = stats.get("numberOfDocumentsScanned")
            print(f"    indexed {indexed} of {scanned} documents")
            # A COMPLETE job that indexed nothing is not a success.  Most often
            # the S3 prefix is empty because the corpus was never synced up.
            if not indexed:
                raise RuntimeError(
                    f"Ingestion reported COMPLETE but indexed {indexed} documents. "
                    f"The knowledge base is EMPTY and the demo will retrieve nothing.\n"
                    f"Most likely s3://{cfg.BUCKET}/{cfg.CORPUS_PREFIX} has no "
                    f"objects. Run `make corpus` to download the papers, then "
                    f"`make build-kb` again -- step 1 uploads them for you."
                )
            break

        if st["status"] == "FAILED":
            # Fatal, deliberately (2026-09-22).  This used to `break` and fall
            # through to printing an ingestion COST ESTIMATE and "Done", so a
            # failed job looked like a successful build: you got an empty
            # knowledge base and no indication anything was wrong until the
            # demo retrieved nothing on stage.  Found by a full teardown ->
            # rebuild cycle where the corpus had not been synced to S3, and
            # Bedrock failed with "The specified bucket does not exist".
            reasons = st.get("failureReasons") or ["(no failureReasons returned)"]
            raise RuntimeError(
                "Ingestion job FAILED -- the knowledge base is empty.\n  "
                + "\n  ".join(str(r) for r in reasons)
                + f"\nCheck that s3://{cfg.BUCKET}/{cfg.CORPUS_PREFIX} contains "
                f"the corpus. Run `make corpus` (downloads the papers) then "
                f"`make build-kb` (step 1 uploads them) to repair it."
            )

        time.sleep(15)


# ----------------------------------------------------------------------
# 5. Bedrock Guardrail
# ----------------------------------------------------------------------
def create_guardrail() -> tuple[str, str]:
    """Create a Bedrock Guardrail that anonymises external URLs in model output.

    The demo's system prompts instruct models to cite papers as full
    PubMed Central URLs (https://www.ncbi.nlm.nih.gov/pmc/articles/PMCxxx/).
    The guardrail intercepts those URLs before they reach the browser:
      - it anonymises every https:// URL, replacing it with {EXTERNAL_URL}
      - agent.py inspects the guardrail trace and substitutes a local
        corpus link (/corpus/PMCxxx) for any PMC article we have locally,
        or "[link removed]" for anything else.

    This demonstrates Bedrock Guardrails enforcing a "no external data
    egress" policy -- the model generates rich cited output but no actual
    external URLs reach the audience's browser.

    Returns:
        (guardrail_id, version) -- main() writes both into config.py.
    """
    br = boto3.client("bedrock", region_name=cfg.REGION)

    # Get-or-create: check if a guardrail with this name already exists.
    existing = [
        g
        for g in br.list_guardrails().get("guardrails", [])
        if g["name"] == "inside-the-lines-url-filter"
    ]
    if existing:
        gid = existing[0]["id"]
        ver = existing[0].get("version", "DRAFT")
        print(f"  guardrail exists: {gid}")
        # ListGuardrails summaries DO carry `arn` (verified 2026-09-22), so the
        # adopt path needs no extra Get.
        #
        # NOTE THE CAPITAL "ARN".  `bedrock` is the only service in this file
        # whose TagResource takes "resourceARN"; bedrock-agent,
        # bedrock-agentcore-control and s3vectors all take "resourceArn".
        # Spelling it the common way raises ParamValidationError -- confirmed
        # offline 2026-09-22 with botocore.validate.validate_parameters against
        # the bedrock TagResource input shape.  It is also why PROJECT_TAGS_KV
        # exists: bedrock wants a list of {"key","value"}, not a map.
        _adopt(
            f"guardrail {gid}",
            lambda: br.tag_resource(resourceARN=existing[0]["arn"], tags=PROJECT_TAGS_KV),
        )
        return gid, ver

    resp = br.create_guardrail(
        name="inside-the-lines-url-filter",
        description="Intercept external URLs in model output for the Inside the Lines demo.",
        sensitiveInformationPolicyConfig={
            "regexesConfig": [
                {
                    "name": "EXTERNAL_URL",
                    "description": "Matches any http/https URL in model output",
                    "pattern": r"https?://[^\s\)\]\"']+",
                    # ANONYMIZE replaces the matched text with {EXTERNAL_URL}
                    # in the model response.  agent.py re-processes those tokens.
                    "action": "ANONYMIZE",
                }
            ]
        },
        # These messages appear only if a guardrail *blocks* a request outright.
        # For this demo we only ANONYMIZE (not block), so these are fallbacks.
        blockedInputMessaging="Input blocked by guardrail.",
        blockedOutputsMessaging="Output blocked by guardrail.",
        # `bedrock` is the odd one out: a LIST of {"key","value"}, not a map.
        tags=PROJECT_TAGS_KV,
    )
    guardrail_id = resp["guardrailId"]
    version = resp.get("version", "DRAFT")
    print(f"  guardrail: {guardrail_id}  version: {version}")
    return guardrail_id, version


# ----------------------------------------------------------------------
# 6. AgentCore Gateway with Cedar policy engine
# ----------------------------------------------------------------------
def web_tools_openapi_schema() -> str:
    """Return the inline OpenAPI 3.0 schema that defines the web_fetch tool.

    Why an OpenAPI target and not a Lambda (or an HTTP passthrough target):

      The Cedar ForbidWeb rule keys off the ACTION ``web-tools___web_fetch`` (NOT
      ``context.toolName`` -- see the policy block below), and `aws.py` calls that
      same MCP tool name.  Both only exist if the gateway has an
      **MCP-category** target that publishes a tool
      called ``web_fetch``.  An OpenAPI target does exactly that with no compute
      to own: the Gateway translates the MCP ``tools/call`` into a plain HTTPS
      GET against ``servers[0].url``.  That is why there is no Lambda in this
      repo -- the earlier design implied one, and it was never needed.

      HTTP *passthrough* targets (the other Lambda-free option) are deliberately
      NOT used: see the note in `create_gateway_target()` for why they cannot
      satisfy this beat.

    The single operation is deliberately named ``web_fetch`` rather than
    something like ``search_studies``.  The operationId IS the tool name (see
    the verified notes below), so renaming it would silently break both the
    Cedar rule and `query_gateway()`.  The name is a slight misnomer for a
    ClinicalTrials.gov search; that trade is worth it to keep one rehearsed,
    load-bearing string identical in three places.

    Verified against the AWS docs on 2026-09-22
    (https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/gateway-schema-openapi.html):
      - ``operationId`` is REQUIRED on every operation and becomes the MCP tool
        name.  Combined with the target name it yields ``web-tools___web_fetch``
        (https://.../gateway-tool-naming.html).
      - ``servers[].url`` must be a real, fully-qualified, static URL.  Templated
        hosts are an SSRF risk and are called out as a bad practice.
      - OpenAPI 3.0/3.1 only; ``oneOf``/``anyOf``/``allOf`` are unsupported, as
        are parameter serialisers.  This schema sticks to flat query parameters
        and plain object/array response types for that reason.
      - Only ``application/json`` is fully supported.

    The endpoint itself was exercised live on 2026-09-22:
      GET https://clinicaltrials.gov/api/v2/studies
          ?query.term=PCSK9&filter.overallStatus=RECRUITING&pageSize=2&format=json
      -> 200 with top-level keys ``totalCount``, ``studies``, ``nextPageToken``.
    The v2 API is public and needs no API key, which is what lets the target be
    created with no credential provider at all (see `create_gateway_target()`).

    Returns:
        The schema as a JSON string, ready for ``openApiSchema.inlinePayload``.
    """
    # NOTE ON PARAMETER NAMES: ClinicalTrials.gov v2 uses dotted query parameters
    # ("query.term", "filter.overallStatus").  Those are the real wire names, so
    # they have to appear verbatim here or the request would not filter anything.
    # AWS warns that operation/property names which violate a downstream model's
    # ToolSpec constraints fail at invoke time -- but nothing in this demo hands
    # web_fetch to a model, `agent.py` calls it directly, so ToolSpec never sees
    # them.  If a future AgentCore release rejects dots at *creation* time, the
    # target will land in FAILED and `create_gateway_target()` raises with the
    # service's own statusReasons rather than letting the talk find out on stage.
    schema = {
        "openapi": "3.0.3",
        "info": {
            "title": "ClinicalTrials.gov study search",
            "version": "2.0.0",
            "description": (
                "Minimal wrapper over the public ClinicalTrials.gov v2 REST API, "
                "exposed through AgentCore Gateway as the web_fetch tool."
            ),
        },
        # Static, fully-qualified host: no URL variables, no SSRF surface.
        "servers": [{"url": "https://clinicaltrials.gov/api/v2"}],
        "paths": {
            "/studies": {
                "get": {
                    # This string is the tool name.  Do not rename it.
                    "operationId": "web_fetch",
                    "summary": "Search ClinicalTrials.gov for registered studies",
                    "description": (
                        "Search the public ClinicalTrials.gov registry for studies "
                        "matching a free-text term, optionally filtered by "
                        "recruitment status."
                    ),
                    "parameters": [
                        {
                            "name": "query.term",
                            "in": "query",
                            "required": True,
                            "description": "Free-text search term, e.g. 'PCSK9'.",
                            "schema": {"type": "string"},
                        },
                        {
                            "name": "filter.overallStatus",
                            "in": "query",
                            "required": False,
                            "description": (
                                "Comma-separated recruitment statuses, e.g. "
                                "'RECRUITING,NOT_YET_RECRUITING'."
                            ),
                            "schema": {"type": "string"},
                        },
                        {
                            "name": "pageSize",
                            "in": "query",
                            "required": False,
                            "description": "Number of studies to return (1-1000).",
                            "schema": {"type": "integer", "default": 5},
                        },
                        {
                            "name": "format",
                            "in": "query",
                            "required": False,
                            "description": "Response format; this tool expects 'json'.",
                            "schema": {"type": "string", "default": "json"},
                        },
                        {
                            "name": "countTotal",
                            "in": "query",
                            "required": False,
                            "description": "Include the total match count in the response.",
                            "schema": {"type": "boolean", "default": True},
                        },
                    ],
                    "responses": {
                        "200": {
                            "description": "Matching studies.",
                            "content": {
                                "application/json": {
                                    # Shape confirmed against a live call on
                                    # 2026-09-22.  Only the fields the demo would
                                    # actually read are described; the real payload
                                    # is far larger and that is fine.
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "totalCount": {"type": "integer"},
                                            "nextPageToken": {"type": "string"},
                                            "studies": {
                                                "type": "array",
                                                "items": {
                                                    "type": "object",
                                                    "properties": {
                                                        "protocolSection": {
                                                            "type": "object",
                                                            "properties": {
                                                                "identificationModule": {
                                                                    "type": "object",
                                                                    "properties": {
                                                                        "nctId": {"type": "string"},
                                                                        "briefTitle": {
                                                                            "type": "string"
                                                                        },
                                                                    },
                                                                },
                                                                "statusModule": {
                                                                    "type": "object",
                                                                    "properties": {
                                                                        "overallStatus": {
                                                                            "type": "string"
                                                                        }
                                                                    },
                                                                },
                                                            },
                                                        }
                                                    },
                                                },
                                            },
                                        },
                                    }
                                }
                            },
                        },
                        "400": {"description": "Invalid search parameters."},
                    },
                }
            }
        },
    }
    return json.dumps(schema)


def create_gateway_target(br_ctrl, gateway_id: str) -> str:
    """Get-or-create the "web-tools" OpenAPI target that publishes web_fetch.

    This is the target that makes beat 5 honest.  Before it existed, the Cedar
    ForbidWeb rule denied a tool that was not registered anywhere: the demo
    only worked because Cedar evaluates ``InvokeTool`` before the gateway
    resolves the target, so the forbid fired first.  Had that ordering ever
    changed, the call would have come back "tool not found", `query_gateway()`
    would not have recognised it as a denial, and the "Cedar Policy Denied"
    badge would have quietly failed to appear on stage.

    Why NOT an HTTP passthrough target (verified 2026-09-22):
      Passthrough targets look like the obvious Lambda-free answer, but they are
      in the **HTTP** target category, and AWS is explicit that "unlike MCP
      targets, HTTP targets do not support capability synchronization or semantic
      tool search.  Clients address each target individually through path-based
      routing" (https://.../gateway-targets-http.html).  Concretely:
        - A passthrough target is reached at
          ``https://{gatewayId}.gateway.bedrock-agentcore.{region}.amazonaws.com/{targetName}/{path}``
          -- it never appears in ``tools/list`` and is not callable via
          ``tools/call``, so there is no ``web-tools___web_fetch``.
        - With no tool there is no ``web-tools___web_fetch`` action name in
          existence, so the rehearsed ForbidWeb rule would stop applying.
      Separately, the pinned botocore (1.43.14) models ``targetConfiguration.http``
      as a union containing only ``agentcoreRuntime`` -- there is no
      ``passthrough`` member to send yet, so the call could not even be made
      from this checkout.
      An OpenAPI target gets the same "no Lambda, real public HTTPS endpoint"
      property while keeping every rehearsed string intact.

    Verified ``create_gateway_target`` shape for this target (2026-09-22,
    https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/gateway-add-target-api-target-config.html
    and the CreateGatewayTarget API reference):

        create_gateway_target(
            gatewayIdentifier=<gateway id>,
            name="web-tools",
            targetConfiguration={"mcp": {"openApiSchema": {"inlinePayload": <json str>}}},
        )

      - ``targetConfiguration`` is a *union*: exactly one of ``mcp`` / ``http``.
      - ``openApiSchema`` is itself a union of ``s3`` / ``inlinePayload``.  Inline
        keeps the whole demo in one repo with nothing extra to upload.
      - ``credentialProviderConfigurations`` is **optional** (API reference:
        "Required: No"), and the outbound-auth matrix lists "No authorization:
        Yes" for OpenAPI schema targets.  ClinicalTrials.gov is public, so we
        omit the field entirely.  Passing ``GATEWAY_IAM_ROLE`` instead would make
        the gateway SigV4-sign requests to clinicaltrials.gov, which does not
        verify SigV4 -- AWS specifically warns against that pairing.
      - ``name`` must match ``([0-9a-zA-Z][-]?){1,100}``; "web-tools" does.
      - The call returns HTTP 202 with ``targetId`` and ``status`` CREATING; the
        target is validated asynchronously, so we poll ``get_gateway_target``.

    Args:
        br_ctrl: a ``bedrock-agentcore-control`` boto3 client.
        gateway_id: the gateway to attach the target to.

    Returns:
        The target ID.

    Raises:
        RuntimeError: if the target ends up FAILED.  This is deliberately fatal:
            a broken target is exactly the silent-failure mode this function
            exists to remove, so the human running build_kb.py must see it.
    """
    target_name = getattr(cfg, "GATEWAY_TARGET_NAME", "web-tools")

    # Get-or-create by name.  Unlike list_policy_engines()/list_policies(), which
    # this file documents as returning empty even when resources exist,
    # list_gateway_targets() is reliable -- teardown.py has depended on it since
    # the first run.  We still paginate, because a gateway may hold many targets.
    existing_id = ""
    next_token = ""
    while True:
        kwargs = {"gatewayIdentifier": gateway_id}
        if next_token:
            kwargs["nextToken"] = next_token
        page = br_ctrl.list_gateway_targets(**kwargs)
        for t in page.get("items", []):
            if t.get("name") == target_name:
                existing_id = t["targetId"]
        next_token = page.get("nextToken", "")
        if existing_id or not next_token:
            break

    if existing_id:
        target_id = existing_id
        print(f"  gateway target exists: {target_name} ({target_id})")
    else:
        # CANNOT BE TAGGED (verified 2026-09-22): CreateGatewayTarget has no
        # `tags` member, and neither CreateGatewayTarget nor GetGatewayTarget
        # returns a target ARN -- the only ARN in either response is the parent
        # `gatewayArn`.  So there is no resourceArn to pass to TagResource
        # afterwards.  Harmless: a target lives inside exactly one gateway, and
        # teardown.py drains targets by calling list_gateway_targets() on every
        # gateway it has already discovered, so target cleanup rides on gateway
        # discovery rather than on its own tag.
        resp = br_ctrl.create_gateway_target(
            gatewayIdentifier=gateway_id,
            name=target_name,
            description="Public ClinicalTrials.gov v2 search, published as web_fetch",
            targetConfiguration={
                "mcp": {"openApiSchema": {"inlinePayload": web_tools_openapi_schema()}}
            },
            # credentialProviderConfigurations intentionally omitted -- see docstring.
        )
        target_id = resp["targetId"]
        print(f"  gateway target: {target_name} ({target_id})")

    # Poll to READY.  A FAILED target is worse than no target for this demo, so
    # surface the service's own statusReasons instead of continuing quietly.
    status = ""
    for _ in range(40):
        detail = br_ctrl.get_gateway_target(gatewayIdentifier=gateway_id, targetId=target_id)
        status = detail.get("status", "")
        if status == "READY":
            break
        if status in ("FAILED", "UPDATE_UNSUCCESSFUL", "SYNCHRONIZE_UNSUCCESSFUL"):
            reasons = "; ".join(detail.get("statusReasons", [])) or "(no reason given)"
            raise RuntimeError(
                f"Gateway target {target_name} is {status}: {reasons}\n"
                "Beat 5 needs a READY target -- fix this before the talk.  The most "
                "likely cause is the inline OpenAPI schema in "
                "web_tools_openapi_schema() failing validation."
            )
        time.sleep(3)
    else:
        print(f"  warning: gateway target {target_name} still {status!r} after ~2 min")

    return target_id


def create_gateway() -> dict:
    """Create an AgentCore Gateway with a Cedar policy engine attached.

    The Gateway is the access-control layer for tool calls.  In this demo
    it intercepts Q5's attempt to call web_fetch and denies it, then the
    agent falls back to the knowledge base.  This demonstrates Cedar
    policy-based guardrails at the tool level.

    Steps:
      a) IAM role for the gateway (service principal bedrock-agentcore.amazonaws.com)
      b) PolicyEngine -- the Cedar evaluation service (waits for ACTIVE)
      c) Gateway -- MCP protocol, no user-level authoriser (waits for READY)
      d) Gateway target -- the "web-tools" OpenAPI target that publishes web_fetch
      e) Cedar policies -- created after gateway so the real ARN can be validated
      f) Attach policy engine to gateway in ENFORCE mode (waits for READY)

    Step (d) comes before (e) on purpose: the policy engine derives its Cedar
    schema from the gateway's tool definitions, so the tool should exist before
    the policies that talk about it.

    Cedar policy quirks (verified 2026-05-21, re-checked 2026-09-22):
      - Tool actions use the format "{target-name}___{tool-name}" (three underscores).
        Our gateway target is named "web-tools", so the Cedar action for web_fetch
        is AgentCore::Action::"web-tools___web_fetch", and that PREFIXED action is
        what ForbidWeb matches on.  The short half ("web_fetch") is the OpenAPI
        operationId.  Matching ``context.toolName`` against the short half does
        NOT work -- verified against real AWS 2026-09-22; see the policy block.
      - A Cedar policy denial comes back as HTTP 200 with a JSON-RPC error body
        ("Tool Execution Denied: ..."), NOT as HTTP 403.  query_gateway() in aws.py
        checks for that pattern.
      - The update_gateway() call that attaches the engine must repeat all the
        original create_gateway() fields (name, roleArn, authorizerType) -- it is
        a full replacement, not a patch.

    Returns:
        dict with gateway_id, gateway_url, engine_id -- main() writes them into config.py.
    """
    br_ctrl = boto3.client("bedrock-agentcore-control", region_name=cfg.REGION)

    ROLE_NAME = "inside-the-lines-gateway-role"
    ENGINE_NAME = "InsideTheLinesEngine"
    # Must match the gateway target name and the Cedar action below.
    TARGET_NAME = getattr(cfg, "GATEWAY_TARGET_NAME", "web-tools")
    ROLE_ARN = f"arn:aws:iam::{cfg.ACCOUNT_ID}:role/{ROLE_NAME}"

    # a. IAM role -- lets the Gateway service assume permissions to call the
    #    Cedar evaluation endpoint and read policies.
    trust = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Principal": {"Service": "bedrock-agentcore.amazonaws.com"},
                "Action": "sts:AssumeRole",
            }
        ],
    }
    try:
        iam.create_role(
            RoleName=ROLE_NAME,
            AssumeRolePolicyDocument=json.dumps(trust),
            Description="Inside the Lines demo -- AgentCore Gateway role",
            Tags=PROJECT_TAGS_IAM,
        )
        print(f"  IAM role created: {ROLE_ARN}")
    except iam.exceptions.EntityAlreadyExistsException:
        print(f"  IAM role exists: {ROLE_ARN}")
        _adopt(
            f"IAM role {ROLE_NAME}",
            lambda: iam.tag_role(RoleName=ROLE_NAME, Tags=PROJECT_TAGS_IAM),
        )

    # The gateway role only needs read access to its own policy engine and policies.
    policy = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Action": [
                    # The policy engine's own evaluation permissions.  Added
                    # 2026-09-22: without these, attaching the engine to the
                    # gateway fails at update_gateway() with
                    #   AccessDeniedException ... assumed-role/
                    #   inside-the-lines-gateway-role/GenesisPolicyEngineCheck
                    #   is not authorized to perform: bedrock-agentcore:<X>
                    # ...so the entire Cedar beat cannot be wired up.
                    #
                    # A wildcard, deliberately.  AgentCore reveals this action
                    # family ONE NAME PER FAILED RUN (we hit AuthorizeAction,
                    # then PartiallyAuthorizeActions), the names are IAM-only so
                    # they do not appear in the botocore service model, and the
                    # Service Authorization Reference page would not render.
                    # Enumerating them by trial and error is fine for us but
                    # would make first-run setup fail repeatedly for anyone
                    # cloning this repo.  Scoped to policy engines only.
                    "bedrock-agentcore:*Authorize*",
                    "bedrock-agentcore:GetPolicyEngine",
                    "bedrock-agentcore:GetPolicy",
                    "bedrock-agentcore:ListPolicies",
                    "bedrock-agentcore:ListPolicyEngines",
                    "bedrock-agentcore:GetGatewayTarget",
                    "bedrock-agentcore:ListGatewayTargets",
                ],
                "Resource": "*",
            }
        ],
    }
    iam.put_role_policy(
        RoleName=ROLE_NAME, PolicyName="gateway-access", PolicyDocument=json.dumps(policy)
    )
    # Wait for IAM propagation before creating the gateway.
    time.sleep(8)

    # b. PolicyEngine -- get-or-create by name.
    #    CORRECTED 2026-09-22.  This block used to claim that list_policy_engines()
    #    "returns empty even when an engine exists (a known API quirk)".  That was
    #    never true -- the code was reading the wrong response key.  AWS is simply
    #    inconsistent about what it calls the list in each response:
    #        ListPolicyEngines   -> "policyEngines"
    #        ListPolicies        -> "policies"
    #        ListGateways        -> "items"
    #        ListGatewayTargets  -> "items"
    #    Reading .get("items") off ListPolicyEngines therefore always yielded [],
    #    which looked like an empty API and left build_kb.py NOT idempotent: a
    #    second run hit ConflictException, failed the lookup, and aborted step 6.
    #    Verify these key names offline, no credentials needed, with:
    #        python -c "from botocore.session import get_session as g; \
    #          m=g().get_service_model('bedrock-agentcore-control'); \
    #          print(list(m.operation_model('ListPolicyEngines').output_shape.members))"
    try:
        pe = br_ctrl.create_policy_engine(name=ENGINE_NAME, tags=PROJECT_TAGS)
        engine_id = pe["policyEngineId"]
        engine_arn = pe["policyEngineArn"]
        print(f"  policy engine: {engine_id}")
        for _ in range(20):
            if br_ctrl.get_policy_engine(policyEngineId=engine_id)["status"] == "ACTIVE":
                break
            time.sleep(3)
    except br_ctrl.exceptions.ConflictException:
        # Engine already exists.  Try config.py first (fastest), then list.
        existing_id = getattr(cfg, "GATEWAY_ENGINE_ID", "")
        if existing_id:
            engine_id = existing_id
            ge = br_ctrl.get_policy_engine(policyEngineId=engine_id)
            engine_arn = ge["policyEngineArn"]
        else:
            # Paginated lookup on the CORRECT response key ("policyEngines").
            match = []
            kwargs: dict = {}
            while True:
                page = br_ctrl.list_policy_engines(**kwargs)
                match += [e for e in page.get("policyEngines", []) if e["name"] == ENGINE_NAME]
                token = page.get("nextToken")
                if not token:
                    break
                kwargs["nextToken"] = token
            if not match:
                raise RuntimeError(
                    f"create_policy_engine said {ENGINE_NAME!r} already exists, but it "
                    "is not in list_policy_engines().  That should be impossible -- if "
                    "you see this, the response shape changed again; check\n"
                    "  aws bedrock-agentcore-control list-policy-engines --region "
                    f"{cfg.REGION}\n"
                    "and set GATEWAY_ENGINE_ID in config.py to work around it."
                ) from None
            engine_id = match[0]["policyEngineId"]
            engine_arn = match[0]["policyEngineArn"]
        print(f"  policy engine exists: {engine_id}")
        _adopt(
            f"policy engine {engine_id}",
            lambda: br_ctrl.tag_resource(resourceArn=engine_arn, tags=PROJECT_TAGS),
        )

    # c. Gateway -- get-or-create by name.
    existing_gws = [
        g for g in br_ctrl.list_gateways().get("items", []) if g["name"] == cfg.GATEWAY_NAME
    ]
    if existing_gws:
        gateway_id = existing_gws[0]["gatewayId"]
        gw_detail = br_ctrl.get_gateway(gatewayIdentifier=gateway_id)
        gateway_url = gw_detail["gatewayUrl"]
        print(f"  gateway exists: {gateway_id}")
        # ListGateways items carry no ARN (only gatewayId/name/status/...), but
        # we already have the Get response here, and that does carry gatewayArn.
        _adopt(
            f"gateway {gateway_id}",
            lambda: br_ctrl.tag_resource(resourceArn=gw_detail["gatewayArn"], tags=PROJECT_TAGS),
        )
    else:
        gw = br_ctrl.create_gateway(
            name=cfg.GATEWAY_NAME,
            roleArn=ROLE_ARN,
            protocolType="MCP",
            authorizerType="NONE",
            tags=PROJECT_TAGS,
        )
        gateway_id = gw["gatewayId"]
        gateway_url = gw["gatewayUrl"]
        print(f"  gateway: {gateway_id}")
        for _ in range(20):
            if br_ctrl.get_gateway(gatewayIdentifier=gateway_id)["status"] == "READY":
                break
            time.sleep(5)

    gateway_arn = f"arn:aws:bedrock-agentcore:{cfg.REGION}:{cfg.ACCOUNT_ID}:gateway/{gateway_id}"

    # d. Gateway target -- the thing ForbidWeb actually forbids.  Created before
    #    the policies so the Cedar schema is generated with the tool present.
    create_gateway_target(br_ctrl, gateway_id)

    # e. Cedar policies -- only create if the gateway doesn't already have this
    #    policy engine attached in ENFORCE mode.  If it does, policies are live.
    #    NOTE: list_policies() returns empty even when policies exist (API quirk
    #    as of 2026-05-21), so we cannot use it to detect existing policies.
    #    Instead we use the gateway attachment as the reliable sentinel.
    current_pec_check = br_ctrl.get_gateway(gatewayIdentifier=gateway_id).get(
        "policyEngineConfiguration"
    )
    # NOTE (2026-09-22): this used to short-circuit on "engine already attached
    # to the gateway in ENFORCE mode" and skip the policy block entirely, using
    # attachment as a proxy for "policies exist".  That proxy is wrong in the one
    # case that matters -- an attached engine whose policies were deleted would
    # never get them back, leaving beat 5 silently unguarded.  Now that the
    # policies are get-or-create by stable name, just always reconcile them; it
    # costs one list_policies call and is self-healing.
    if current_pec_check and current_pec_check.get("arn") == engine_arn:
        print("  engine already attached; reconciling policies")
    # Stable names, NOT timestamped.  The original code appended a timestamp
    # "to guarantee unique policy names", which guaranteed the opposite of
    # idempotency: every build_kb.py run added another PermitAll/ForbidWeb
    # pair to the engine (we accumulated three pairs while debugging this).
    # AgentCore appends its own unique suffix to the policy ID anyway, so
    # fixed names are safe -- and they let us skip creation when a policy of
    # that name is already present.  (2026-09-22)
    existing_policy_names = set()
    _kwargs: dict = {}
    while True:
        _page = br_ctrl.list_policies(policyEngineId=engine_id, **_kwargs)
        existing_policy_names |= {p["name"] for p in _page.get("policies", [])}
        _token = _page.get("nextToken")
        if not _token:
            break
        _kwargs["nextToken"] = _token

    # PermitAll: baseline allow for all principals, actions, and resources.
    permit_cedar = f'permit(principal, action, resource == AgentCore::Gateway::"{gateway_arn}");'
    # ForbidWeb: deny the web_fetch tool call.
    #
    # VERIFIED AGAINST REAL AWS 2026-09-22.  The Cedar ACTION is the fully
    # prefixed MCP tool name, "{target}___{tool}".  This statement previously
    # read:
    #     action == AgentCore::Action::"InvokeTool" ... when { context.toolName == "web_fetch" }
    # which NEVER MATCHED.  It went unnoticed for months because there was no
    # real gateway target, so the call failed for unrelated reasons and the UI
    # showed a denial anyway.  Once "web-tools" became a real OpenAPI target the
    # bug surfaced in the worst possible way: beat 5 quietly SUCCEEDED, fetching
    # live ClinicalTrials.gov results and demonstrating the exact opposite of the
    # security point the beat exists to make.
    #
    # With the form below, AgentCore denies the call and names the policy in the
    # error, e.g.:
    #   "Tool Execution Denied: Tool call not allowed due to policy enforcement
    #    [Policy evaluation denied due to ForbidWeb-xxxxxxxx]"
    # That phrasing is what aws.py :: _is_policy_denial() matches on.
    #
    # The action string couples THREE things that must stay in step: the target
    # name ("web-tools"), the operationId in web_tools_openapi_schema()
    # ("web_fetch"), and this policy.  config.GATEWAY_TARGET_NAME documents it.
    forbid_cedar = (
        f'forbid(principal, action == AgentCore::Action::"{TARGET_NAME}___web_fetch", '
        f'resource == AgentCore::Gateway::"{gateway_arn}");'
    )
    for name, stmt in [("PermitAll", permit_cedar), ("ForbidWeb", forbid_cedar)]:
        if name in existing_policy_names:
            print(f"  policy exists: {name}")
            continue
        # NOT TAGGED, on purpose (2026-09-22): CreatePolicy has no `tags`
        # member.  A Cedar policy *does* have a `policyArn` (it is in both
        # GetPolicy and ListPolicySummaries), so bedrock-agentcore-control's
        # TagResource might accept it -- but that is unverified, and tagging
        # would buy nothing: a policy only exists inside a policy engine, and
        # teardown.py enumerates policies per engine it has already discovered.
        # Guessing at an untested write call here would be worse than the
        # comment.
        br_ctrl.create_policy(
            name=name,
            policyEngineId=engine_id,
            definition={"cedar": {"statement": stmt}},
            # IGNORE_ALL_FINDINGS: skip Cedar schema validation.
            # Use FAIL_ON_ANY_FINDINGS in production for stricter checks.
            validationMode="IGNORE_ALL_FINDINGS",
        )
        print(f"  policy: {name}")

    # f. Attach the policy engine to the gateway in ENFORCE mode -- only if not
    #    already attached.  get_gateway() returns policyEngineConfiguration if set.
    current_pec = br_ctrl.get_gateway(gatewayIdentifier=gateway_id).get("policyEngineConfiguration")
    if current_pec and current_pec.get("arn") == engine_arn:
        print("  policy engine already attached to gateway")
    else:
        # update_gateway() is a full replacement -- all original create fields required.
        # ENFORCE means Cedar denials block the call; MONITOR would only log them.
        br_ctrl.update_gateway(
            gatewayIdentifier=gateway_id,
            name=cfg.GATEWAY_NAME,
            roleArn=ROLE_ARN,
            authorizerType="NONE",
            policyEngineConfiguration={"arn": engine_arn, "mode": "ENFORCE"},
        )
        for _ in range(20):
            if br_ctrl.get_gateway(gatewayIdentifier=gateway_id)["status"] == "READY":
                break
            time.sleep(5)
        print("  gateway READY with Cedar policy engine in ENFORCE mode")

    return {"gateway_id": gateway_id, "gateway_url": gateway_url, "engine_id": engine_id}


def ensure_corpus_bucket() -> None:
    """Create the corpus bucket, tag it, and upload the corpus into it.

    Three things that used to be the reader's problem, in the order they must
    happen.  All three are idempotent, so this is cheap on a re-run:

      1. CREATE the bucket.  cfg.BUCKET is derived as
         "inside-the-lines-pcsk9-<account-id>" by bootstrap.ensure_config(), which
         is globally unique by construction -- no inventing a name that turns out
         to be taken.  bootstrap.ensure_bucket() handles the create_bucket
         us-east-1/everywhere-else asymmetry; read its docstring before touching
         it.
      2. TAG it, so teardown.py can still find it after a rename (below).
      3. UPLOAD corpus/ into it, in parallel, skipping what is already there.
         This is deliberately here and not in corpus_fetch.py -- see the long
         comment above bootstrap.upload_corpus() for why.

    The upload must happen BEFORE the ingestion job in step 5, or the job
    "succeeds" against an empty prefix and leaves an empty knowledge base.
    """
    bootstrap.ensure_bucket(cfg.BUCKET, cfg.REGION, s3)
    tag_corpus_bucket()
    bootstrap.upload_corpus(cfg.BUCKET, cfg.CORPUS_PREFIX, s3=s3, region=cfg.REGION)


def tag_corpus_bucket() -> None:
    """Stamp the project tag on the S3 corpus bucket named by config.BUCKET.

    We tag it because it is the one resource teardown.py deletes whose
    name it cannot guess: cfg.KB_NAME, the vector bucket, the guardrail, the
    gateway and the roles all start with "inside-the-lines", but the corpus
    bucket name is yours to choose.  Without a tag, renaming BUCKET between a
    build and a teardown would leave the old corpus bucket (and its per-GB-month
    charge) behind with nothing to find it by.

    PutBucketTagging is a REPLACE, not a merge (S3 has no "add one tag" call), so
    we read the existing tag set first and merge in ours.  Clobbering a bucket's
    cost-allocation tags would be a genuinely bad thing for this script to do.

    Best-effort: a bucket you do not own the tagging permission on just prints a
    warning.  teardown.py still deletes cfg.BUCKET via its config fast path; the
    tag only adds the ability to find it after a rename.
    """
    try:
        existing = s3.get_bucket_tagging(Bucket=cfg.BUCKET).get("TagSet", [])
    except Exception:  # noqa: BLE001 -- NoSuchTagSet on an untagged bucket is normal
        existing = []
    merged = [t for t in existing if t["Key"] != PROJECT_TAG_KEY] + [
        {"Key": PROJECT_TAG_KEY, "Value": PROJECT_TAG_VALUE}
    ]
    try:
        s3.put_bucket_tagging(Bucket=cfg.BUCKET, Tagging={"TagSet": merged})
        print(f"  tagged corpus bucket: {cfg.BUCKET}")
    except Exception as e:  # noqa: BLE001
        print(f"  (could not tag corpus bucket {cfg.BUCKET}: {type(e).__name__})")


def estimate_ingestion_cost() -> None:
    """Print an ingestion cost estimate based on the local corpus directory.

    The Bedrock ingestion API reports document counts but NOT token counts,
    so we cannot use the API response to compute cost.  Instead we walk the
    local corpus/ directory, sum raw character counts, and approximate the
    token count at 4 chars/token (a rough but consistent heuristic for
    English biomedical text).

    Informational only -- nothing to copy anywhere.  The live UI computes the
    same figure itself from the corpus size (pricing.py / the `setup_cost`
    event), and labels it as COMPUTED rather than metered, which is the honest
    description: Bedrock does not bill an itemised "ingestion" line.
    Printing it here is so you see roughly what this one-time step cost.

    If corpus/ is not present (e.g. you ran this on a different machine
    than the one that ran corpus_fetch.py), the estimate is skipped.
    """
    import os

    corpus_dir = "corpus"
    if not os.path.isdir(corpus_dir):
        print("  (corpus/ not found -- skipping ingestion cost estimate)")
        return

    total_chars = sum(
        os.path.getsize(os.path.join(root, f))
        for root, _dirs, files in os.walk(corpus_dir)
        for f in files
    )
    approx_tokens = total_chars / 4  # 4 characters per token is a common approximation
    usd = (approx_tokens / 1_000_000) * cfg.EMBED_USD_PER_1M_TOKENS
    print(f"\n  Ingestion cost estimate ({total_chars:,} chars ≈ {approx_tokens:,.0f} tokens):")
    # Printed for the operator only.  Nothing reads it back: the app derives the
    # same figure itself (AwsBackend.kb_setup_costs), so there is nothing to copy.
    print(f"    about ${usd:.4f}, one-time -- recorded automatically, nothing to copy")


def main() -> None:
    """Provision everything, then write the resulting IDs into config.py.

    Wrapped in a function (rather than sitting under `if __name__`) so that
    start.py -- the single `make start` entry point -- can call it directly
    instead of shelling out to another interpreter.
    """
    print("1/7  corpus bucket: create, tag, upload")
    ensure_corpus_bucket()

    print("2/7  IAM role")
    role_arn = create_kb_role()

    print("3/7  S3 Vectors bucket + index")
    index_arn = create_vector_store()

    print("4/7  knowledge base + data source")
    kb_id, ds_id = create_kb(role_arn, index_arn)

    print("5/7  ingestion")
    # Skip ingestion if a completed job with indexed documents already exists.
    # This avoids re-embedding the entire corpus on every re-run (costs ~$0.15).
    # If you add new papers to S3, re-run teardown.py then build_kb.py to reindex.
    completed = [
        j
        for j in agent.list_ingestion_jobs(knowledgeBaseId=kb_id, dataSourceId=ds_id).get(
            "ingestionJobSummaries", []
        )
        if j.get("status") == "COMPLETE"
        and j.get("statistics", {}).get("numberOfNewDocumentsIndexed", 0) > 0
    ]
    if completed:
        s = completed[0].get("statistics", {})
        print(
            f"  ingestion already complete: "
            f"{s.get('numberOfNewDocumentsIndexed')} documents indexed -- skipping"
        )
    else:
        ingest(kb_id, ds_id)
    estimate_ingestion_cost()

    print("6/7  guardrail")
    guardrail_id, guardrail_version = create_guardrail()

    print("7/7  AgentCore Gateway + Cedar policy engine")
    gw = create_gateway()

    # These seven values used to be printed under "Done. Paste these into
    # config.py:" and hand-copied.  Seven hand-edits is seven chances to put a
    # gateway URL on the KB_ID line and then debug it live on stage, so they are
    # now written back for you.  bootstrap.write_config():
    #   - rewrites ONLY these exact top-level `NAME = "string"` lines, so every
    #     comment and every other setting in the file survives untouched
    #   - parses the result and reads the values back BEFORE writing anything
    #   - copies config.py to config.py.bak first, and says so
    #   - writes nothing at all if the values are already correct, so running
    #     this script twice cannot duplicate a line
    # It still PRINTS everything it wrote: the transcription step is gone, the
    # visibility is not.
    print("\nWriting the resolved IDs into config.py")
    bootstrap.write_config(
        {
            "KB_ID": kb_id,
            "DATA_SOURCE_ID": ds_id,
            "GUARDRAIL_ID": guardrail_id,
            "GUARDRAIL_VERSION": guardrail_version,
            "GATEWAY_ID": gw["gateway_id"],
            "GATEWAY_URL": gw["gateway_url"],
            "GATEWAY_ENGINE_ID": gw["engine_id"],
        }
    )
    print("\nDone -- nothing left to edit. `make demo` will run the demo.")
    print("Run `make teardown` after the talk (or leave it: ~1.3 cents a month).")


if __name__ == "__main__":
    main()
