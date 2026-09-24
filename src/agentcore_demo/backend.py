"""
backend.py  --  the Backend interface (Protocol) for the demo.

This module defines the contract between the agent and the outside world.
The agent (agent.py) depends only on this interface, not on boto3 or any
AWS service.  This is the standard Python "dependency injection" pattern:
the agent gets its backend passed in at construction time.

Why a Protocol instead of a base class?
  Python's typing.Protocol means "any object with these methods works" --
  no inheritance required.  FakeBackend in fakes.py satisfies the Protocol
  without subclassing Backend.  This keeps the fake simple and avoids any
  risk of accidentally inheriting real behavior.

Why is this in its own file?
  Two reasons:
    1. agent.py imports Backend (and nothing else AWS-related) so the whole
       agent module loads cleanly without boto3 or config.py.  Tests can
       import agent.py directly without any AWS setup.
    2. AwsBackend (in aws.py) also exports Backend as a re-export, but the
       canonical definition lives here to avoid circular imports.

What each method is for:
  retrieve()       -- vector search: find passages relevant to a question.
  converse()       -- model call: invoke a Bedrock foundation model.
  code_interpreter_run() -- run Python code in an isolated AgentCore microVM.
  kb_setup_costs() -- return KB cost context for the sidebar panel.
  kb_is_ready()    -- check whether the KB has indexed documents.
  kb_ingest()      -- start/poll an ingestion job.
  kb_flush()       -- reset ready state for rehearsal.
  query_gateway()  -- invoke a tool on the AgentCore Gateway (Cedar policy demo).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol


class Backend(Protocol):
    """What agent.py needs from the world.

    The real implementation is AwsBackend (aws.py).
    The test/fake implementation is FakeBackend (fakes.py).
    Both satisfy this Protocol without inheriting from it.
    """

    def retrieve(self, query: str, n: int = 12) -> list[dict]:
        """Run a semantic search and return the n most relevant passages.

        Args:
            query: the search text (a question or keyword phrase).
            n: number of passages to return.

        Returns:
            A list of dicts, each with "text" (str), "source" (PMC ID str),
            and "score" (float 0.0-1.0, cosine similarity).
        """
        ...

    def converse(
        self,
        tier: str,
        system: str,
        prompt: str,
        max_tokens: int = 1600,
        thinking: str | None = None,
    ) -> tuple[str, dict, list[dict]]:
        """Invoke a Bedrock foundation model and return its response.

        Works for both Claude (Haiku, Sonnet, Opus) and OpenAI GPT-6 Astra --
        the Bedrock converse() API accepts all of them with the same shape.

        Args:
            tier: model tier key, e.g. "haiku", "sonnet", "opus", "openai".
            thinking: None leaves the model's default (adaptive thinking is ON
                by default on Claude 5 models); "disabled" turns it off.
            system: the system prompt text.
            prompt: the user message text.
            max_tokens: maximum number of output tokens.

        Returns:
            (text, usage, matches) where:
              - text is the model's response string.
              - usage is {"inputTokens": int, "outputTokens": int}.
              - matches is a list of Bedrock Guardrail hits:
                [{"name": str, "match": str, "action": str}]
                Empty list if no guardrail is configured or no URLs matched.
        """
        ...

    def code_interpreter_run(self, code: str) -> tuple[str, float]:
        """Run a self-contained Python script in an isolated microVM.

        Args:
            code: the Python script to execute.

        Returns:
            (stdout_text, wall_clock_seconds) -- the combined stdout from the
            script and the execution duration in seconds.
        """
        ...

    def kb_setup_costs(self) -> dict:
        """Return KB setup costs and quantitative context for the sidebar panel.

        These are NOT live metered charges; they are computed or derived:
          - ingestion_usd: from corpus char count (Bedrock API has no token count)
          - storage_usd_per_month: from S3 Vectors vector count × per-vector bytes
            (GetVectorBucket has no sizeBytes field)
          - corpus_storage_usd_per_month: from corpus/ dir size × S3 rate
          - vector_count, vector_size_mb, corpus_files, corpus_size_mb: quantitative
            context for the sidebar panel

        NOT included in the live run total.
        """
        ...

    def kb_is_ready(self) -> bool:
        """Return True if the KB has indexed documents and is ready to query.

        The page calls this on load.  Returns False if no completed ingestion
        job exists, or if kb_flush() has been called to force a re-ingest.
        """
        ...

    def kb_ingest(self, progress_cb: Callable[[int, int], None]) -> dict:
        """Start (or resume) a Bedrock ingestion job.

        Args:
            progress_cb: called with (indexed_count, total_expected) at each
                poll cycle so the browser can update its progress bar.

        Returns:
            A dict with ingestion_usd, storage_usd_per_month, and other KB
            setup cost fields (same as kb_setup_costs()).
        """
        ...

    def kb_flush(self) -> None:
        """Mark the KB as needing re-ingestion.

        Does NOT delete anything from AWS.  The next call to kb_ingest()
        will start a new ingestion job.  Used during rehearsal to re-watch
        the ingestion progress animation without actually re-indexing.
        """
        ...

    def query_gateway(self, tool_name: str, arguments: dict) -> dict:
        """Invoke a tool on the AgentCore Gateway.

        The Gateway enforces Cedar policies.  In the demo, the ForbidWeb
        policy denies any call where tool_name == "web_fetch".

        Args:
            tool_name: the short tool name (without the target prefix).
            arguments: tool arguments dict (e.g. {"url": "https://..."}).

        Returns exactly one of THREE shapes -- the third exists so beat 5 can
        never silently succeed, and never fake its own badge:
            {"result": response_body}
                the tool ran and returned data.
            {"denied": True, "reason": "..."}
                a Cedar policy denied the call, or no gateway is configured.
                Only a message that is recognisably a policy denial may
                claim this (see aws.py :: _is_policy_denial).
            {"error": True, "reason": "..."}
                anything else -- transport failure, unparseable body, a bare
                HTTP 403 (far more likely a missing InvokeGateway permission
                than a policy decision), or an MCP result.isError.  The UI
                shows this as a visible failure rather than a denial.
        """
        ...
