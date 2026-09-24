"""
aws.py  --  the real AWS backend for the demo.

This module provides AwsBackend, the live implementation of the Backend
protocol.  It wraps every AWS service call the agent needs:

  retrieve()             -> Bedrock Knowledge Bases vector search
  converse()             -> Bedrock model invocation (Claude and OpenAI)
  code_interpreter_run() -> AgentCore Code Interpreter (isolated microVM)
  kb_setup_costs()       -> one-time + monthly KB costs for the sidebar panel
  kb_is_ready()          -> check whether the KB has indexed documents
  kb_ingest()            -> start/poll an ingestion job with a progress callback
  kb_flush()             -> reset local state to force a fresh ingest (rehearsal)
  query_gateway()        -> invoke a tool on the AgentCore Gateway (Cedar policy)

No AWS calls happen at import time -- the boto3 clients are constructed in
__init__ so the module is safe to import in tests (which use FakeBackend).

Verified AWS quirks (do not "fix" these without checking current docs):

  Ingestion statistics (2026-05-20):
    The Bedrock ingestion job statistics block only contains document counts.
    There is NO token count in the API response.  Ingestion cost is therefore
    computed from local corpus character count: total_chars / 4 tokens × embed
    rate.  The cost is labelled "computed from corpus size" (not metered).

  S3 Vectors storage size (2026-05-20):
    GetVectorBucket does NOT return a sizeBytes field.  Storage cost is derived
    by counting vectors via ListVectors and multiplying by the known per-vector
    byte size for Titan Embed V2 (1024 float32 = 4096 bytes, plus ~512 bytes of
    metadata overhead per vector).  Labelled "from vector count" (not metered).

  Claude inference config (2026-05-20, re-checked 2026-09-22):
    temperature and topP are deprecated on the current Claude models.  Passing
    them returns HTTP 400.  The converse() call omits both; only maxTokens is
    set.  (This is why inferenceConfig only ever contains maxTokens in this
    file.)  Omitting them is also what makes the model set swappable: GPT-6
    Astra and Claude Fable 5.1 both reject temperature too.

  Adaptive thinking is ON by default (2026-09-22):
    Claude Opus 5 and Sonnet 5 run adaptive thinking by default INCLUDING when
    the request omits a "thinking" field.  Opus 4.7 behaved the opposite way
    (omitting it meant no thinking), so the Sept 2026 model bump silently turned
    thinking on.
    Consequences, in order of importance for a live demo:
      1. Latency goes up.  Re-time the run before the talk.
      2. Thinking tokens bill as OUTPUT tokens, so the receipt grows by more
         than the headline rate change suggests.
      3. The reasoning text is NEVER displayed, so on this demo it was ~80s of a
         300s budget spent on output nobody sees.
    converse() therefore takes a `thinking` argument: pass thinking="disabled"
    and it sends additionalModelRequestFields={"thinking": {"type": "disabled"}}.
    agent.question_4 does exactly that for both Q4 reviewers, which is what took
    that beat from 141s to 48s, and Agent._model() now does it for every other
    call as well.
    The field is ANTHROPIC-ONLY: sending it to GPT-6 Astra returns
    ValidationException (unknown_parameter), so converse() gates on the model ID
    containing "anthropic" rather than on the tier name.

  Cedar policy denial shape (2026-05-21):
    A Cedar ENFORCE denial comes back as HTTP 200 with a JSON-RPC error body
    containing "Tool Execution Denied: ..." in the message field -- NOT as
    HTTP 403.  query_gateway() checks for this pattern explicitly.

  AgentCore Gateway tool naming (2026-05-21, re-verified 2026-09-22):
    MCP tool names on the gateway use the format "${target_name}___${tool_name}"
    (three underscores) -- see
    https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/gateway-tool-naming.html
    Our target is named "web-tools", so web_fetch becomes "web-tools___web_fetch"
    in the tools/call JSON-RPC payload.  The Cedar rule matches that SAME prefixed
    string as its action -- ``AgentCore::Action::"web-tools___web_fetch"``, verified
    against real AWS 2026-09-22.  Do not go back to matching
    ``context.toolName == "web_fetch"``: that form never fired, and because the
    denial then came from something else entirely, the UI looked right anyway.

  The web_fetch tool is a real OpenAPI target (2026-09-22):
    "web-tools" is an MCP-category **OpenAPI** gateway target over the public
    ClinicalTrials.gov v2 API; build_kb.web_tools_openapi_schema() holds the
    inline schema, whose single operationId is "web_fetch".  There is no Lambda
    anywhere in this repo.  Tool *arguments* are therefore that operation's query
    parameters ("query.term", "filter.overallStatus", "pageSize", ...), not a
    free-form {"url": ...} -- the old shape, which no target ever accepted.

    HTTP *passthrough* targets cannot be used for this beat: they are in the HTTP
    target category, which AWS documents as path-routed
    ("https://{gatewayId}.gateway.bedrock-agentcore.{region}.amazonaws.com/{targetName}/{path}")
    and explicitly NOT aggregated into tools/list.  With no MCP tool there is no
    "web-tools___web_fetch" action name for the rehearsed ForbidWeb rule to match,
    so the rule would simply stop applying.  See
    https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/gateway-targets-http.html

  Gateway failures are NOT denials (2026-09-22):
    query_gateway() returns three distinct outcomes -- denied / result / error.
    Only a message matching _is_policy_denial() may claim a Cedar denial; every
    other failure returns {"error": True} so beat 5 cannot silently succeed or
    silently fake its own badge.
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable

import boto3
from botocore.config import Config as BotoConfig

from agentcore_demo.backend import Backend

__all__ = ["AwsBackend", "Backend"]

# Actual number of CC0/CC BY PCSK9 papers pulled from PMC.
# Used as the denominator in the ingestion progress bar.
# Corrected 2026-09-22: was 657, but the corpus in S3 is 1000 objects (48 MB) --
# the May run was scaled back up to the original 1000-paper target and this
# constant was never updated, so the progress bar over-reported by ~1.5x.
_EXPECTED_CORPUS_SIZE = 1000

# Titan Embed V2 produces 1024-dimensional float32 vectors.
# Each float32 is 4 bytes, so the vector data alone is 4 096 bytes.
#
# Metadata revised 2026-09-22 (was a flat 512-byte budget, which was too low).
# The dominant metadata item is AMAZON_BEDROCK_TEXT, which Bedrock populates
# with the chunk's own text -- not a short reference to it.  build_kb.py uses
# FIXED_SIZE chunking at maxTokens=512, and 512 tokens of English prose is
# roughly 2 KB, so the text alone is ~4x the old budget.  Add a few hundred
# bytes for the source URI and Bedrock's other managed keys.
#
# This is an ESTIMATE, and the KB panel labels it as such ("from vector
# count", not metered): S3 Vectors has no API that reports an index's stored
# size, so there is nothing to measure.  Erring high is the honest direction
# -- it overstates a cost rather than understating it -- and at this scale the
# whole line is still fractions of a cent per month.
_BYTES_PER_VECTOR = 4096 + 2048 + 512


def _gw_mcp_url(base: str) -> str:
    """Return the gateway's MCP endpoint, whether or not `base` already ends in /mcp.

    Defensive because the value is copy-pasted by a human from build_kb.py's
    output into config.py, and the two sides disagreed about the suffix once
    already.  Idempotent: appending /mcp twice is the failure this prevents.
    """
    trimmed = base.rstrip("/")
    return trimmed if trimmed.endswith("/mcp") else trimmed + "/mcp"


def _is_policy_denial(message: str) -> bool:
    """Return True only for messages that are recognisably a Cedar policy denial.

    Kept deliberately narrow and in one place.  Everything that is *not* matched
    here becomes a visible error rather than a "Cedar Policy Denied" badge, so a
    loose match would let an unrelated failure impersonate the security story
    beat 5 is making.  "Tool Execution Denied" is the phrasing AgentCore actually
    returns (verified 2026-05-21); the other two are defensive.

    Args:
        message: the JSON-RPC error message, or an HTTP error body.

    Returns:
        Whether the message indicates an authorization denial.
    """
    lowered = message.lower()
    return (
        "tool execution denied" in lowered
        or "not allowed" in lowered
        # Broad, but only reached alongside an explicit denial vocabulary check
        # -- "policy" alone appearing in an unrelated stack trace is unlikely.
        or ("denied" in lowered and "policy" in lowered)
    )


def _format_source(uri: str) -> str:
    """Convert an S3 URI to a readable PMC article ID.

    The KB returns source locations like:
        s3://inside-the-lines-corpus/corpus/PMC13156736.txt

    We extract just the PMC ID so the UI can render a clean citation link.
    Falls back to the raw URI if the pattern doesn't match.

    Args:
        uri: an S3 URI string from a Bedrock retrieval result.

    Returns:
        A PMC ID string like "PMC13156736", or the original URI.
    """
    import re  # noqa: PLC0415

    m = re.search(r"(PMC\d+)", uri)
    return m.group(1) if m else uri


class AwsBackend:
    """Real AWS implementation of the Backend protocol.

    Constructed once per process by app._build_backend() or run._build().
    All boto3 clients are created here so that tests (which import Backend
    but use FakeBackend) never trigger any AWS credentials check.
    """

    def __init__(
        self,
        region: str,
        kb_id: str,
        ds_id: str,
        models: dict[str, str],
        rates: dict[str, float] | None = None,
        vector_bucket_name: str = "",
        guardrail_id: str = "",
        guardrail_version: str = "DRAFT",
        gateway_url: str = "",
        gateway_target: str = "web-tools",
    ):
        """
        Args:
            region: AWS region (e.g. "us-west-2").
            kb_id: the Bedrock Knowledge Base ID from config.py.
            ds_id: the data source ID from config.py.
            models: dict mapping tier names ("haiku", "sonnet", etc.) to
                Bedrock inference profile IDs.
            rates: dict of pricing rates from pricing.fetch_rates().
            vector_bucket_name: the S3 Vectors bucket name for storage cost estimation.
            guardrail_id: optional Bedrock Guardrail ID; empty string disables it.
            guardrail_version: the guardrail version to use (default "DRAFT").
            gateway_url: optional AgentCore Gateway URL; empty string disables it.
            gateway_target: the Gateway target name that publishes web_fetch.
                Must equal the target name build_kb.py creates, because MCP tool
                names are "{target}___{tool}".  Defaults to the demo's value so
                existing callers (app.py, run.py) need no change.
        """
        self.region = region
        self.kb_id = kb_id
        self.ds_id = ds_id
        self.models = models
        self.rates = rates or {}
        self.vector_bucket_name = vector_bucket_name
        self.guardrail_id = guardrail_id
        self.guardrail_version = guardrail_version
        self.gateway_url = gateway_url
        self.gateway_target = gateway_target

        # Separate boto3 clients for the three different Bedrock service endpoints.
        self._kb = boto3.client("bedrock-agent-runtime", region_name=region)  # retrieval
        # Model calls get a long read timeout.  botocore defaults to 60s, and
        # that is no longer enough: Claude Opus 5 with adaptive thinking on (the
        # default -- see the module docstring) at maxTokens=8192 took longer than
        # that on Q4 (then numbered Q3) and raised ReadTimeoutError mid-run on
        # 2026-09-22; the demo failed at that beat before this was raised.  Q4 now
        # runs with thinking
        # disabled at maxTokens=4096 and finishes in ~48s, so the 600s ceiling is
        # no longer load-bearing -- it stays as free insurance for a slow day.
        # retries are left at botocore's default mode but capped low: a silent
        # retry of a 2-minute Opus call would blow the demo's timing budget.
        self._llm = boto3.client(
            "bedrock-runtime",
            region_name=region,
            config=BotoConfig(
                read_timeout=600,
                connect_timeout=15,
                retries={"max_attempts": 2, "mode": "standard"},
            ),
        )
        self._agent = boto3.client("bedrock-agent", region_name=region)  # ingestion mgmt

        # Set after kb_ingest() completes; used by kb_setup_costs() to avoid
        # re-fetching stats on every sidebar refresh.
        self._ingestion_stats: dict | None = None

        # Set to True by kb_flush() to force a new ingestion on next call.
        self._force_reingest: bool = False

    @classmethod
    def from_config(cls, config, rates: dict[str, float] | None = None) -> AwsBackend:
        """Build an AwsBackend from a config module -- the ONLY supported way.

        Added 2026-09-22 after a real divergence bug.  app.py passed
        guardrail_id / guardrail_version / gateway_url; run.py passed none of
        them.  The headless runner therefore ran with the Bedrock Guardrail and
        the AgentCore Gateway silently DISABLED, so `make demo-headless` -- the
        very command the README offers for verifying the demo -- exercised
        neither beat 2's link interception nor beat 5's Cedar denial, and beat 5
        reported "No gateway configured" instead of a policy denial.

        Two call sites constructing the same object with different arguments is
        the bug; one constructor is the fix.  Add new backend wiring HERE, never
        in app.py or run.py.
        """
        return cls(
            config.REGION,
            config.KB_ID,
            config.DATA_SOURCE_ID,
            config.MODELS,
            rates=rates,
            vector_bucket_name=getattr(config, "VECTOR_BUCKET_NAME", ""),
            guardrail_id=getattr(config, "GUARDRAIL_ID", ""),
            guardrail_version=getattr(config, "GUARDRAIL_VERSION", "DRAFT"),
            gateway_url=getattr(config, "GATEWAY_URL", ""),
            gateway_target=getattr(config, "GATEWAY_TARGET_NAME", "web-tools"),
        )

    def retrieve(self, query: str, n: int = 12) -> list[dict]:
        """Run a semantic search against the Bedrock Knowledge Base.

        Sends the query to the KB's vector search endpoint, which embeds the
        query text and returns the n most similar document chunks.

        The Bedrock retrieval API shape (verified 2026-05-20):
            client.retrieve(
                knowledgeBaseId=...,
                retrievalQuery={"text": query},
                retrievalConfiguration={"vectorSearchConfiguration": {"numberOfResults": n}},
            )
            → retrievalResults[].content.text  (the chunk text)
            → retrievalResults[].location.s3Location.uri  (the source S3 path)
            → retrievalResults[].score  (cosine similarity, 0.0 – 1.0)

        Args:
            query: the question or search phrase to embed.
            n: number of passages to return (default 12; Q3/Q4 use 16).

        Returns:
            A list of dicts with keys "text", "source" (PMC ID), and "score".
        """
        resp = self._kb.retrieve(
            knowledgeBaseId=self.kb_id,
            retrievalQuery={"text": query},
            retrievalConfiguration={"vectorSearchConfiguration": {"numberOfResults": n}},
        )
        return [
            {
                "text": item["content"]["text"],
                "source": _format_source(
                    item.get("location", {}).get("s3Location", {}).get("uri", "?")
                ),
                "score": item.get("score", 0.0),
            }
            for item in resp["retrievalResults"]
        ]

    def converse(
        self,
        tier: str,
        system: str,
        prompt: str,
        max_tokens: int = 1600,
        thinking: str | None = None,
    ) -> tuple[str, dict, list[dict]]:
        """Invoke a Bedrock foundation model and return text, token usage, and guardrail hits.

        Uses the Bedrock Runtime converse() API, which works uniformly across
        vendors -- Claude and OpenAI take the same request shape and return the
        same response shape.  This is the key Bedrock abstraction that makes it
        easy to swap models, and it is why replacing Amazon Nova Pro with
        OpenAI GPT-6 Astra in September 2026 needed no change to this function.

        Converse is also the only API on which Bedrock Guardrails work with
        OpenAI models, so the guardrail integration below depends on staying
        here rather than moving to the Responses or Chat Completions APIs.

        Guardrail integration:
            If guardrail_id is set, the guardrail config is attached to every
            call.  When the guardrail matches (e.g. a URL in the output),
            the trace contains an outputAssessments block.  This function
            extracts those matches and returns them as the third tuple element
            so agent.py can substitute local corpus links.

        Important: temperature is intentionally omitted from inferenceConfig.
            The current Claude models reject temperature in the request body (return
            HTTP 400).  Using only maxTokens works for all models including
            Haiku, Sonnet, Opus, and GPT-6 Astra.  (Verified 2026-05-20;
            re-checked for Astra 2026-09-22.)

        Args:
            tier: a key into self.models, e.g. "haiku", "sonnet", "opus", "openai".
            system: the system prompt text.
            prompt: the user message text.
            max_tokens: the maximum number of output tokens to generate.
            thinking: None (default) leaves the model's own setting alone, which
                on Claude 5 models means adaptive thinking is ON.  "disabled"
                turns it off -- Anthropic models only; see the note below and the
                module docstring.

        Returns:
            (text, usage, matches) where:
              - text is the model's response string.
              - usage is a dict with "inputTokens" and "outputTokens".
              - matches is a list of guardrail hits:
                [{"name": str, "match": str, "action": str}], empty if none.
        """
        kwargs: dict = {
            "modelId": self.models[tier],
            "system": [{"text": system}],
            "messages": [{"role": "user", "content": [{"text": prompt}]}],
            # temperature is deliberately absent -- current models reject it.
            "inferenceConfig": {"maxTokens": max_tokens},
        }
        # Adaptive thinking is ON by default on Claude 5 models, and its output
        # is NEVER shown to the audience (thinking.display defaults to omitted).
        # Measured on Opus 5 with a ~30K-token prompt on 2026-09-22:
        #   thinking on  -> 43.4s, 2,733 output tokens, blocks [reasoningContent, text]
        #   thinking off -> 27.5s, 1,631 output tokens, blocks [text]
        # Same visible answer, 37% faster, 40% fewer billed output tokens.  Callers
        # that care about the demo's timing budget pass thinking="disabled".
        # "thinking" is an ANTHROPIC-specific field.  Sending it to GPT-6 Astra
        # returns ValidationException: unknown_parameter 'thinking' (observed
        # 2026-09-22), so gate on the vendor rather than the tier -- that keeps
        # this correct if a tier is ever repointed at a different vendor.
        # Astra needs no equivalent: it already returns a single text block and
        # answered in 26s, so there is no hidden reasoning to suppress.
        model_id = self.models.get(tier, "")
        if thinking == "disabled" and "anthropic" in model_id:
            kwargs["additionalModelRequestFields"] = {"thinking": {"type": "disabled"}}
        if self.guardrail_id:
            kwargs["guardrailConfig"] = {
                "guardrailIdentifier": self.guardrail_id,
                "guardrailVersion": self.guardrail_version,
                "trace": "enabled",  # needed to see which URLs were intercepted
            }
        resp = self._llm.converse(**kwargs)

        # Join every text block, rather than assuming content[0] is one.
        #
        # This line used to read content[0]["text"] and that broke the demo the
        # first time it ran on Claude 5 (2026-09-22, KeyError: 'text' on the Sonnet
        # code-generation beat, then Q2, now Q3).
        # Opus 5 and Sonnet 5 run adaptive thinking by default even when the
        # request omits a "thinking" field -- which this method does -- so the
        # FIRST content block is now a reasoning block and the visible answer
        # comes later.  Opus 4.7 put text at index 0, which is why this was
        # never wrong before the model refresh.  Haiku 4.5 still puts text
        # first, which is why the Haiku beat passed and only the Sonnet beat blew up.
        #
        # Converse may also legitimately split one answer across several text
        # blocks, so concatenating is more correct than picking the first.
        blocks = resp["output"]["message"]["content"]
        text = "".join(b["text"] for b in blocks if "text" in b)
        if not text:
            # Fail loudly and diagnosably rather than returning "" and letting a
            # blank panel appear on stage with no explanation.
            kinds = [k for b in blocks for k in b]
            raise RuntimeError(
                f"converse() returned no text block for tier {tier!r} "
                f"(model {self.models.get(tier)!r}); block keys were {kinds}. "
                "If this is a refusal, check stop_reason -- Claude Fable/Mythos "
                "models can decline with stop_reason='refusal'."
            )

        # GPT-6 Astra reports prompt tokens in the CACHE fields, not inputTokens
        # (2026-09-22).  Measured on the identical Q4 review prompt:
        #     Claude Opus 5      inputTokens = 13,107  cacheWrite =     0
        #     OpenAI GPT-6 Astra inputTokens =      2  cacheWrite = 3,614
        # So nothing is missing -- Astra runs implicit prompt caching and books the
        # prompt as a cache WRITE, leaving inputTokens at ~0.  Taken at face value
        # that understates Astra's input cost by ~$0.14 per Q4, which on a talk
        # whose whole thesis is honest cost numbers is the worst place to be
        # quietly wrong (it made Q4 look like $0.1587 instead of $0.2556).
        #
        # Fix: when inputTokens is implausible for the prompt we actually sent,
        # fall back to the cache counters, which are REAL metered numbers.  Only
        # if those are absent too do we resort to a chars/4 estimate, and we flag
        # that case so the receipt can say "estimated" rather than "metered".
        #
        # Caveat we accept knowingly: Bedrock prices cache writes at ~1.25x input
        # and cache reads at ~0.1x, so folding them into inputTokens at the plain
        # input rate is approximate.  It is approximate in the third decimal place;
        # the alternative was wrong in the first.
        usage = dict(resp.get("usage", {}))
        est_input_tokens = (len(system) + len(prompt)) // 4
        if est_input_tokens > 100 and usage.get("inputTokens", 0) < est_input_tokens // 10:
            usage["inputTokensReported"] = usage.get("inputTokens", 0)
            cached = usage.get("cacheWriteInputTokens", 0) + usage.get("cacheReadInputTokens", 0)
            if cached > 0:
                usage["inputTokens"] = cached
                usage["inputTokensFromCacheFields"] = True
            else:
                usage["inputTokens"] = est_input_tokens
                usage["inputTokensEstimated"] = True

        # Parse guardrail matches from the response trace.
        # The trace structure: resp.trace.guardrail.outputAssessments.{assessmentId: [...]}.
        matches: list[dict] = []
        if self.guardrail_id:
            assessments = resp.get("trace", {}).get("guardrail", {}).get("outputAssessments", {})
            for assessment_list in assessments.values():
                for assessment in assessment_list if isinstance(assessment_list, list) else []:
                    regexes = assessment.get("sensitiveInformationPolicy", {}).get("regexes", [])
                    for r in regexes:
                        if r.get("detected"):
                            matches.append(
                                {"name": r["name"], "match": r["match"], "action": r["action"]}
                            )

        return text, usage, matches

    def code_interpreter_run(self, code: str) -> tuple[str, float]:
        """Execute Python code in an isolated AgentCore Code Interpreter microVM.

        The Code Interpreter runs in a fresh, ephemeral container for each
        invocation.  It has matplotlib, numpy, and pandas pre-installed.
        The generated code from Q3 ends by printing a base64 PNG string;
        agent._extract_chart() picks that out of the stdout.

        AgentCore Code Interpreter API shape (verified 2026-05-20):
            from bedrock_agentcore.tools.code_interpreter_client import CodeInterpreter
            ci = CodeInterpreter(region)
            ci.start()
            resp = ci.invoke("executeCode", {"language": "python", "code": code})
            # resp["stream"] is a generator of event dicts
            # each event: {"result": {"content": [{"type": "text", "text": "..."}]}}
            ci.stop()

        The start/stop pattern is important: ci.start() provisions the microVM,
        and ci.stop() releases it so you are not charged while idle.

        Args:
            code: a self-contained Python script to run.

        Returns:
            (stdout, wall_clock_seconds) where stdout is all text output
            concatenated, and wall_clock_seconds is the execution time.
        """
        from bedrock_agentcore.tools.code_interpreter_client import CodeInterpreter  # noqa: PLC0415

        ci = CodeInterpreter(self.region)
        ci.start()
        t0 = time.time()
        try:
            resp = ci.invoke("executeCode", {"language": "python", "code": code})
            out: list[str] = []
            for event in resp["stream"]:
                for block in event.get("result", {}).get("content", []):
                    if block.get("type") == "text":
                        out.append(block["text"])
            return "\n".join(out), time.time() - t0
        finally:
            # Always stop the interpreter even if the code raises an exception,
            # so the microVM is released and billing stops.
            ci.stop()

    def kb_is_ready(self) -> bool:
        """Return True if the KB has at least one completed ingestion job with indexed docs.

        The page calls this on load to decide whether to show the ingestion
        progress view or go straight to the question prompt.

        Returns False in two cases:
          - No completed ingestion jobs exist yet (build_kb.py hasn't been run,
            or ingestion is still in progress).
          - kb_flush() was called to force a re-ingest (rehearsal mode).
        """
        if self._force_reingest:
            return False
        try:
            jobs = self._agent.list_ingestion_jobs(
                knowledgeBaseId=self.kb_id,
                dataSourceId=self.ds_id,
            )
            for job in jobs.get("ingestionJobSummaries", []):
                if job.get("status") == "COMPLETE":
                    stats = job.get("statistics", {})
                    # Require at least one indexed document -- an empty COMPLETE
                    # job means ingestion ran but found nothing to index.
                    if stats.get("numberOfNewDocumentsIndexed", 0) > 0:
                        return True
        except Exception:
            # If the KB ID is wrong or AWS is unreachable, treat as not ready.
            pass
        return False

    def kb_ingest(self, progress_cb: Callable[[int, int], None]) -> dict:
        """Start a Bedrock ingestion job and poll until it completes.

        Called by the /ingest WebSocket endpoint.  Sends progress updates via
        progress_cb(indexed, total) so the browser can show a progress bar.

        After ingestion completes, computes the one-time setup costs:
          - ingestion_usd: from local corpus char count (API has no token count)
          - storage_usd_per_month: from vector count × per-vector bytes
          - corpus_storage_usd_per_month: from local corpus dir size × S3 rate

        Args:
            progress_cb: called with (indexed_count, total_expected) at each poll.

        Returns:
            A dict suitable for the kb_ready WebSocket event:
            {"ingestion_usd": float, "storage_usd_per_month": float, ...}
        """
        # Emit a 0/total progress tick immediately so the browser shows the bar.
        progress_cb(0, _EXPECTED_CORPUS_SIZE)

        job_id = self._agent.start_ingestion_job(
            knowledgeBaseId=self.kb_id,
            dataSourceId=self.ds_id,
        )["ingestionJob"]["ingestionJobId"]

        stats: dict = {}
        while True:
            st = self._agent.get_ingestion_job(
                knowledgeBaseId=self.kb_id,
                dataSourceId=self.ds_id,
                ingestionJobId=job_id,
            )["ingestionJob"]
            stats = st.get("statistics", {})
            # Count both new and modified documents for the progress bar.
            indexed = stats.get("numberOfNewDocumentsIndexed", 0) + stats.get(
                "numberOfModifiedDocumentsIndexed", 0
            )
            progress_cb(indexed, _EXPECTED_CORPUS_SIZE)
            if st["status"] in ("COMPLETE", "FAILED"):
                break
            time.sleep(3)  # poll every 3 seconds

        # Ingestion cost: the API reports document counts only, not token counts.
        # We compute from the local corpus directory instead.
        ingestion_usd = self._compute_ingestion_cost_from_corpus()
        storage_usd = self._compute_storage_cost_from_vector_count()

        self._ingestion_stats = {
            "ingestion_usd": round(ingestion_usd, 4),
            "storage_usd_per_month": round(storage_usd, 4),
            "corpus_storage_usd_per_month": round(self._compute_corpus_s3_storage_cost(), 6),
        }
        self._force_reingest = False
        return dict(self._ingestion_stats)

    def kb_flush(self) -> None:
        """Reset local state so the next kb_ingest() triggers a fresh ingestion job.

        This does NOT delete anything from AWS -- the KB stays fully indexed.
        It only clears the local ready flag so the ingestion progress view
        reappears in the browser.  Useful for rehearsal: you can re-watch the
        ingestion animation without actually re-indexing all 1,000 papers.
        """
        self._force_reingest = True
        self._ingestion_stats = None

    def query_gateway(self, tool_name: str, arguments: dict) -> dict:
        """Invoke a tool on the AgentCore Gateway via the MCP HTTP endpoint.

        The Gateway exposes tools over an MCP-compatible JSON-RPC HTTP interface.
        We POST a tools/call request and classify the response into exactly one
        of three outcomes (see Returns).

        Why three outcomes and not two:
            This used to collapse everything that was not obviously a success
            into ``{"denied": True}``, which made two opposite failures invisible
            in opposite directions.  A tool-not-found error -- the shape you get
            if the "web-tools" target is missing or FAILED -- carries none of the
            denial keywords, so it fell through to the SUCCESS branch and the
            "Cedar Policy Denied" badge silently never appeared.  Meanwhile a
            DNS blip or a stale GATEWAY_URL reported itself AS a Cedar denial,
            which is a lie told on stage.  Beat 5's whole claim is "the policy
            stopped this", so only a recognised denial may say so.

        Cedar policy denial quirk (verified 2026-05-21):
            Denials arrive as HTTP 200 with a JSON-RPC error body, NOT HTTP 403.
            The error message contains "Tool Execution Denied" or "not allowed".

        MCP tool naming quirk (verified 2026-05-21, re-verified 2026-09-22 against
        https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/gateway-tool-naming.html):
            Gateway tool names are "${target_name}___${tool_name}" -- three
            underscores.  Our target is "web-tools" and, because it is an OpenAPI
            target, the tool half is the operationId from the inline schema in
            build_kb.py.  Hence "web-tools___web_fetch".

        MCP tool-level errors (verified 2026-09-22):
            MCP distinguishes protocol errors (a top-level "error") from tool
            errors (a "result" carrying ``isError: true``).  Both are treated as
            failures here; only the former can ever be a Cedar denial.

        Args:
            tool_name: the short tool name (e.g. "web_fetch" -- without the prefix).
            arguments: tool arguments.  For web_fetch these are the
                ClinicalTrials.gov v2 query parameters declared in the target's
                OpenAPI schema, e.g. {"query.term": "PCSK9", "pageSize": 5}.

        Returns:
            {"denied": True, "reason": "..."}  -- a *recognised* Cedar denial.
                Only this outcome may drive the "Cedar Policy Denied" badge.
            {"result": response_body}  -- a genuine tool success.
            {"error": True, "reason": "..."}  -- anything else: unreachable
                gateway, unconfigured gateway, HTTP error, unparseable body,
                tool not found, schema error.  Callers must surface this rather
                than treating it as either a success or a denial.
        """
        import json  # noqa: PLC0415
        import ssl  # noqa: PLC0415
        import urllib.error  # noqa: PLC0415
        import urllib.request  # noqa: PLC0415

        if not self.gateway_url:
            # Not a denial: nothing was evaluated, because there is no gateway.
            # Saying "denied" here would fake the badge on a misconfigured box.
            return {"error": True, "reason": "No gateway configured (GATEWAY_URL is empty)"}

        # Prepend the target name to form the full MCP tool name.  "web-tools" is
        # the OpenAPI gateway target created by build_kb.create_gateway_target().
        mcp_tool_name = f"{self.gateway_target}___{tool_name}"

        payload = json.dumps(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {"name": mcp_tool_name, "arguments": arguments},
            }
        ).encode()

        req = urllib.request.Request(
            # Tolerate a gateway URL given either with or without the /mcp
            # suffix.  build_kb.py used to print it WITH /mcp while this line
            # appended a second one, so a pasted value produced ".../mcp/mcp"
            # and every beat-4 call failed.  Accept both spellings (2026-09-22).
            _gw_mcp_url(self.gateway_url),
            data=payload,
            method="POST",
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
        )
        ctx = ssl.create_default_context()
        try:
            with urllib.request.urlopen(req, context=ctx, timeout=10) as resp:
                raw = resp.read()
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")
            if e.code == 403 and _is_policy_denial(body):
                # Some configurations do return HTTP 403 for a denial -- but only
                # call it one if the body actually says so.  A bare 403 is far
                # more likely to be missing IAM permission on InvokeGateway.
                return {"denied": True, "reason": f"Cedar policy denied: {body[:200]}"}
            return {"error": True, "reason": f"HTTP {e.code}: {body[:200]}"}
        except Exception as ex:  # noqa: BLE001 -- urllib raises a wide variety
            return {"error": True, "reason": f"{type(ex).__name__}: {ex}"}

        try:
            body = json.loads(raw)
        except ValueError:
            return {"error": True, "reason": f"Malformed gateway response: {raw[:200]!r}"}
        if not isinstance(body, dict):
            return {"error": True, "reason": f"Unexpected gateway response: {str(body)[:200]}"}

        # JSON-RPC protocol error.  Cedar denials land here; so do tool-not-found
        # and schema-validation errors, which must NOT be mistaken for denials.
        if "error" in body:
            msg = str(body["error"].get("message", "") if isinstance(body["error"], dict) else "")
            if _is_policy_denial(msg):
                return {"denied": True, "reason": f"Cedar policy denied: {msg}"}
            return {"error": True, "reason": f"Gateway error (not a policy denial): {msg[:200]}"}

        # MCP tool-level error: a result that reports its own failure.
        result = body.get("result")
        if isinstance(result, dict) and result.get("isError"):
            return {"error": True, "reason": f"Tool reported an error: {str(result)[:200]}"}

        # A well-formed JSON-RPC success MUST carry "result".  If it does not, we
        # do not know what we are looking at, so we refuse to call it a success.
        if result is None:
            return {"error": True, "reason": f"Gateway response had no result: {str(body)[:200]}"}

        return {"result": body}

    def kb_setup_costs(self) -> dict:
        """Return KB setup costs and quantitative context for the sidebar panel.

        This is called by the /api/kb-costs HTTP endpoint.  The returned dict
        feeds the "Knowledge Base" panel in the UI, which displays:
          - How much it cost to embed and index the corpus (one-time)
          - How much the vector store costs per month (ongoing)
          - How many papers and vectors are in the corpus

        None of these are live metered charges -- they are computed or derived:
          - ingestion_usd: computed from corpus/ dir char count (API has no tokens)
          - storage_usd_per_month: derived from vector count × per-vector bytes
            (GetVectorBucket has no sizeBytes field -- verified 2026-05-20)
          - corpus_storage_usd_per_month: from corpus/ dir size × S3 rate

        These costs are NOT included in the run total shown on the receipt.
        """
        # Measure fresh stats on every call -- cheap local computation.
        vector_count, vector_size_mb = self._measure_vector_stats()
        corpus_files, corpus_size_mb = self._measure_corpus_stats()

        # Use stats from the most recent ingestion job if available;
        # otherwise recompute from scratch.
        base = (
            self._ingestion_stats
            if self._ingestion_stats is not None
            else {
                "ingestion_usd": round(self._compute_ingestion_cost_from_corpus(), 4),
                "storage_usd_per_month": round(self._compute_storage_cost_from_vector_count(), 4),
                "corpus_storage_usd_per_month": round(self._compute_corpus_s3_storage_cost(), 6),
            }
        )
        return {
            **base,
            "vector_count": vector_count,
            "vector_size_mb": round(vector_size_mb, 1),
            "corpus_files": corpus_files,
            "corpus_size_mb": round(corpus_size_mb, 1),
        }

    # -- private helpers --------------------------------------------------

    def _measure_vector_stats(self) -> tuple[int, float]:
        """Count live vectors in S3 Vectors and estimate their storage size.

        GetVectorBucket does not return a sizeBytes field, so we page through
        all vectors with ListVectors and multiply by the known bytes-per-vector
        for Titan Embed V2.  This is only called for the sidebar panel and runs
        quickly at this corpus's scale (1,000 papers, a few thousand vectors).

        Returns:
            (vector_count, size_mb)
        """
        try:
            import boto3  # noqa: PLC0415

            s3v = boto3.client("s3vectors", region_name=self.region)
            import config  # type: ignore[import]  # noqa: PLC0415

            n = 0
            for page in s3v.get_paginator("list_vectors").paginate(
                vectorBucketName=self.vector_bucket_name or config.VECTOR_BUCKET_NAME,
                indexName=config.VECTOR_INDEX_NAME,
            ):
                n += len(page.get("vectors", []))
            size_mb = n * _BYTES_PER_VECTOR / (1024 * 1024)
            return n, size_mb
        except Exception:
            # If the bucket name is wrong or S3 Vectors is unavailable, return zeros.
            return 0, 0.0

    def _measure_corpus_stats(self) -> tuple[int, float]:
        """Count files and total size of the local corpus/ directory.

        Returns:
            (file_count, size_mb)
        """
        corpus_dir = "corpus"
        if not os.path.isdir(corpus_dir):
            return 0, 0.0
        files = list(os.listdir(corpus_dir))
        total_bytes = sum(os.path.getsize(os.path.join(corpus_dir, f)) for f in files)
        return len(files), total_bytes / (1024 * 1024)

    def _compute_ingestion_cost_from_corpus(self) -> float:
        """Estimate embedding cost from local corpus character count.

        The Bedrock ingestion job statistics block only reports document counts,
        not token counts.  As a proxy we walk the local corpus/ directory,
        sum raw byte sizes (which approximately equal character counts for UTF-8
        English text), and divide by 4 to get a rough token estimate.

        The "4 chars per token" heuristic is consistent with common tokenizer
        behavior for English text (GPT-4, Llama-2, Titan all average ~4 chars/token).

        Returns 0.0 if:
          - corpus/ does not exist (script run on a different machine)
          - the embed rate is not known (pricing lookup failed)
        """
        embed_rate = self.rates.get("embed_usd_per_1m_tokens", 0.0)
        if embed_rate <= 0:
            return 0.0
        corpus_dir = "corpus"
        if not os.path.isdir(corpus_dir):
            return 0.0
        total_chars = sum(
            os.path.getsize(os.path.join(root, f))
            for root, _dirs, files in os.walk(corpus_dir)
            for f in files
        )
        approx_tokens = total_chars / 4  # 4 chars per token approximation
        return (approx_tokens / 1_000_000) * embed_rate

    def _compute_storage_cost_from_vector_count(self) -> float:
        """Derive monthly storage cost from the live vector count in S3 Vectors.

        S3 Vectors' GetVectorBucket API does not expose a sizeBytes field
        (verified 2026-05-20), so we cannot read the storage size directly.
        Instead we count vectors via ListVectors and multiply by the known
        per-vector byte count:
            Titan Embed V2: 1024 floats × 4 bytes = 4096 bytes of vector data
            plus ~512 bytes of per-vector metadata overhead
            total: _BYTES_PER_VECTOR = 4608 bytes per vector

        Returns 0.0 if the bucket name or storage rate is unknown.
        """
        storage_rate = self.rates.get("s3v_storage_usd_per_gb_month", 0.0)
        bucket = self.vector_bucket_name
        if not storage_rate or not bucket:
            return 0.0
        try:
            s3v = boto3.client("s3vectors", region_name=self.region)
            import config  # type: ignore[import]  # noqa: PLC0415

            index_name = config.VECTOR_INDEX_NAME
            n_vectors = 0
            paginator = s3v.get_paginator("list_vectors")
            for page in paginator.paginate(
                vectorBucketName=bucket,
                indexName=index_name,
                # We only need the count, not the actual vector data.
            ):
                n_vectors += len(page.get("vectors", []))
            size_bytes = n_vectors * _BYTES_PER_VECTOR
            size_gb = size_bytes / (1024**3)
            return size_gb * storage_rate
        except Exception:
            return 0.0

    def _compute_corpus_s3_storage_cost(self) -> float:
        """Compute monthly S3 standard storage cost for the local corpus directory.

        Uses the S3 standard storage rate (first 50 TB tier, us-west-2: $0.023/GB-month).
        For this corpus -- 1,000 papers, ~48 MB -- that is about $0.001/month:
        included in the sidebar for completeness, not because it is a meaningful
        expense.

        Falls back to the hard-coded $0.023/GB-month rate if the Price List
        lookup in pricing.py was not available at startup.
        """
        s3_rate = self.rates.get("s3_standard_usd_per_gb_month", 0.023)
        corpus_dir = "corpus"
        if not os.path.isdir(corpus_dir):
            return 0.0
        total_bytes = sum(
            os.path.getsize(os.path.join(root, f))
            for root, _dirs, files in os.walk(corpus_dir)
            for f in files
        )
        size_gb = total_bytes / (1024**3)
        return size_gb * s3_rate
