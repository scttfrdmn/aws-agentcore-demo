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
    the request omits a "thinking" field -- which this file does.  Opus 4.7
    behaved the opposite way (omitting it meant no thinking), so the Sept 2026
    model bump silently turned thinking on.
    Consequences to watch, in order of importance for a live demo:
      1. Latency goes up.  Re-time the run before the talk.
      2. Thinking tokens bill as OUTPUT tokens, so the receipt grows by more
         than the headline rate change suggests.
      3. Answer quality on Q3 should improve, which is why it is left on.
    To turn it off if rehearsal timing demands, pass
      additionalModelRequestFields={"thinking": {"type": "disabled"}}
    to converse().  Both models accept that (Opus 5 caps effort at "high"
    when thinking is disabled).  Deliberately NOT done here -- measure first.

  Cedar policy denial shape (2026-05-21):
    A Cedar ENFORCE denial comes back as HTTP 200 with a JSON-RPC error body
    containing "Tool Execution Denied: ..." in the message field -- NOT as
    HTTP 403.  query_gateway() checks for this pattern explicitly.

  AgentCore Gateway tool naming (2026-05-21):
    MCP tool names on the gateway use the format "{target-name}___{tool-name}"
    (three underscores).  Our target is named "web-tools", so web_fetch becomes
    "web-tools___web_fetch" in the tools/call JSON-RPC payload.
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable

import boto3

from agentcore_demo.backend import Backend

__all__ = ["AwsBackend", "Backend"]

# Actual number of CC0/CC BY PCSK9 papers pulled from PMC.
# Used as the denominator in the ingestion progress bar.
_EXPECTED_CORPUS_SIZE = 657

# Titan Embed V2 produces 1024-dimensional float32 vectors.
# Each float32 is 4 bytes, so the vector data alone is 4 096 bytes.
# S3 Vectors also stores per-vector metadata; we budget ~512 bytes for that.
# This constant is used to estimate storage cost when ListVectors is called.
_BYTES_PER_VECTOR = 4096 + 512


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

        # Separate boto3 clients for the three different Bedrock service endpoints.
        self._kb = boto3.client("bedrock-agent-runtime", region_name=region)  # retrieval
        self._llm = boto3.client("bedrock-runtime", region_name=region)  # model calls
        self._agent = boto3.client("bedrock-agent", region_name=region)  # ingestion mgmt

        # Set after kb_ingest() completes; used by kb_setup_costs() to avoid
        # re-fetching stats on every sidebar refresh.
        self._ingestion_stats: dict | None = None

        # Set to True by kb_flush() to force a new ingestion on next call.
        self._force_reingest: bool = False

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
            n: number of passages to return (default 12; Q2/Q3 use 16).

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
        self, tier: str, system: str, prompt: str, max_tokens: int = 1600
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
        if self.guardrail_id:
            kwargs["guardrailConfig"] = {
                "guardrailIdentifier": self.guardrail_id,
                "guardrailVersion": self.guardrail_version,
                "trace": "enabled",  # needed to see which URLs were intercepted
            }
        resp = self._llm.converse(**kwargs)
        text = resp["output"]["message"]["content"][0]["text"]

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

        return text, resp["usage"], matches

    def code_interpreter_run(self, code: str) -> tuple[str, float]:
        """Execute Python code in an isolated AgentCore Code Interpreter microVM.

        The Code Interpreter runs in a fresh, ephemeral container for each
        invocation.  It has matplotlib, numpy, and pandas pre-installed.
        The generated code from Q2 ends by printing a base64 PNG string;
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
        ingestion animation without actually re-indexing all 650 papers.
        """
        self._force_reingest = True
        self._ingestion_stats = None

    def query_gateway(self, tool_name: str, arguments: dict) -> dict:
        """Invoke a tool on the AgentCore Gateway via the MCP HTTP endpoint.

        The Gateway exposes tools over an MCP-compatible JSON-RPC HTTP interface.
        We POST a tools/call request and inspect the response for Cedar policy
        denials.

        Cedar policy denial quirk (verified 2026-05-21):
            Denials arrive as HTTP 200 with a JSON-RPC error body, NOT HTTP 403.
            The error message contains "Tool Execution Denied" or "not allowed".
            This function checks for both patterns and returns {"denied": True}.

        MCP tool naming quirk (verified 2026-05-21):
            The Gateway prefixes tool names with the target name plus three
            underscores.  Our gateway target is "web-tools", so the web_fetch
            tool must be requested as "web-tools___web_fetch".

        Args:
            tool_name: the short tool name (e.g. "web_fetch" -- without the prefix).
            arguments: a dict of tool arguments (e.g. {"url": "https://..."}).

        Returns:
            {"result": response_body}  on success, or
            {"denied": True, "reason": "..."} if Cedar policy denied the call
            or no gateway is configured.
        """
        import json  # noqa: PLC0415
        import ssl  # noqa: PLC0415
        import urllib.error  # noqa: PLC0415
        import urllib.request  # noqa: PLC0415

        if not self.gateway_url:
            return {"denied": True, "reason": "No gateway configured"}

        # Prepend the target name to form the full MCP tool name.
        # "web-tools" is the name of the Lambda-backed gateway target.
        mcp_tool_name = f"web-tools___{tool_name}"

        payload = json.dumps(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {"name": mcp_tool_name, "arguments": arguments},
            }
        ).encode()

        req = urllib.request.Request(
            self.gateway_url.rstrip("/") + "/mcp",
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
                body = json.loads(resp.read())
                # Cedar denials are HTTP 200 with an "error" key in the JSON-RPC body.
                if "error" in body:
                    msg = body["error"].get("message", "")
                    if "Denied" in msg or "policy" in msg.lower() or "not allowed" in msg.lower():
                        return {"denied": True, "reason": f"Cedar policy denied: {msg}"}
                return {"result": body}
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")
            if e.code == 403:
                # Fallback: some configurations do return HTTP 403.
                return {"denied": True, "reason": f"Cedar policy denied: {body[:200]}"}
            return {"denied": True, "reason": f"HTTP {e.code}: {body[:200]}"}
        except Exception as ex:  # noqa: BLE001
            return {"denied": True, "reason": str(ex)}

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
        quickly for a ~650-paper corpus.

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
        For a 650-paper corpus of ~10 MB the cost is fractions of a cent -- included
        in the sidebar for completeness, not because it's a meaningful expense.

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
