"""
agent.py  --  the demo orchestration layer.

This is the core of the demo.  It runs the four locked questions (and any
free-form question) against a Backend, and reports every step of the process
by calling an emit(event: dict) callback.

The event callback is the only I/O this module does.  The web app (app.py)
provides an emit that pushes events over a WebSocket to the browser.  A
command-line runner (run.py) provides an emit that prints to the terminal.
Tests provide an emit that collects events into a list for assertions.

This module is the source of truth for the event protocol.  See CLAUDE.md
for the full table; the EVENT TYPES comment below has the quick reference.

How Q3 works (Opus + GPT-6 Astra run in parallel):
  Question 3 asks two model families to read the same evidence independently,
  then a third model adjudicates where they disagree.  Both reviewers read in
  parallel using a ThreadPoolExecutor so the audience sees both models running
  at the same time.  A threading.Lock protects emit() because the two threads
  fire events concurrently.

How free-form questions are routed:
  A single Claude Haiku call (the "routing call") classifies the question into
  one of three paths: SYNTHESIS, ANALYSIS, or DEBATE.  Haiku is instructed to
  reply with exactly one word (max_tokens=5), keeping the routing call cheap
  (fractions of a cent).  The routing cost IS included in the receipt.

What the guardrail substitution does:
  Q1 and Q3 system prompts instruct models to cite papers as full NCBI URLs.
  The Bedrock Guardrail anonymises those URLs and returns "{EXTERNAL_URL}"
  placeholders in the model output.  _model() inspects the guardrail trace,
  matches each placeholder to its original URL, and replaces it with either:
    - A local corpus link (/corpus/PMCxxxxxxx) if we have that paper locally.
    - A bare PMC ID in brackets ([PMCxxxxxxx]) if we do not.
    - "[link removed]" for anything that isn't a PMC article.
  This demonstrates Bedrock Guardrails keeping external URLs out of the
  demo while still providing cited, useful answers.
"""

from __future__ import annotations

import base64
import os
import re
import threading
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed

from agentcore_demo import questions as Q
from agentcore_demo.backend import Backend
from agentcore_demo.cost import CostMeter

# Type alias for the emit callback.  Any callable that accepts a dict works.
Emit = Callable[[dict], None]

# EVENT TYPES (every event is a dict with a "type" key):
#   question      {n, text}                  a new question started
#   phase         {label}                    status line shown in the UI
#   retrieval     {count}                    N passages retrieved from KB
#   model         {tier, label, state, ...}  state="start" | "done" (+usage, cost, elapsed_s)
#   answer        {title, text}              a synthesis / adjudication result
#   code          {text}                     generated Python analysis code
#   chart         {data}                     base64 PNG for an inline <img>
#   cost          {total}                    running cost-meter total (after each step)
#   route         {path, label}              which path was chosen for a free-form question
#                                            path: "SYNTHESIS" | "ANALYSIS" | "DEBATE"
#   setup_cost    {ingestion_usd,            one-time + recurring KB costs (NOT in run total)
#                  storage_usd_per_month,
#                  corpus_storage_usd_per_month,
#                  vector_count, ...}
#   guardrail     {actions}                  URL matches intercepted; each action has
#                                            {original, local, reason}
#   policy_denied {tool, reason}             Cedar Gateway policy denied a tool call
#   receipt       {rows, total}              the final itemised receipt
#   done          {}                         run complete; browser shows receipt

# Human-readable descriptions for each routing path.
# These appear in the "route" event label field and in the UI.
ROUTE_LABELS = {
    "SYNTHESIS": "retrieval + synthesis · Claude Haiku",
    "ANALYSIS": "code generation + chart · Claude Sonnet + Code Interpreter",
    "DEBATE": "dual review + adjudication · Opus + GPT-6 Astra + Sonnet",
}


class Agent:
    """Drives the demo and emits events as each step completes.

    Constructed fresh for every run (one question or all four).
    The backend and meter are passed in so tests can inject fakes without
    any AWS dependency.
    """

    def __init__(self, backend: Backend, meter: CostMeter, emit: Emit):
        self.backend = backend
        self.meter = meter
        self.emit = emit

    # -- helpers ---------------------------------------------------------

    def _context(self, chunks: list[dict]) -> str:
        """Format retrieved passages for inclusion in a model prompt.

        Each chunk is labelled with its source PMC ID.  The system prompt
        instructs the model to expand "PMCxxxxxxx" to a full NCBI URL --
        the guardrail then intercepts those URLs and replaces them with
        local corpus links.

        This is intentional: we want to see the guardrail act in the demo,
        so we feed it real URLs to intercept.
        """
        parts = []
        for c in chunks:
            # Source is already formatted as "PMCxxxxxxx" by _format_source() in aws.py.
            parts.append(f"[source: {c['source']}]\n{c['text']}")
        return "\n\n".join(parts)

    def _retrieve(self, step: str, query: str, n: int = 12) -> list[dict]:
        """Retrieve passages, record retrieval cost, and emit an updated total.

        Args:
            step: a label for this retrieval step (used in the receipt).
            query: the search text to embed.
            n: number of passages to retrieve.

        Returns:
            A list of passage dicts (text, source, score).
        """
        chunks = self.backend.retrieve(query, n)
        self.meter.add_retrieval(step, n_queries=1)
        self.emit({"type": "cost", "total": round(self.meter.total, 6)})
        return chunks

    def _model(
        self, step: str, tier: str, label: str, system: str, prompt: str, max_tokens: int = 4096
    ) -> str:
        """Run one model call, emitting start/done events and recording cost.

        Emits:
          - {"type": "model", "state": "start", ...}  before the call
          - {"type": "model", "state": "done", "usage": ..., "cost": ...}  after
          - {"type": "guardrail", "actions": [...]}  if any URLs were intercepted
          - {"type": "cost", "total": ...}  after each of the above

        Guardrail substitution:
          If the model output contains {EXTERNAL_URL} placeholders (put there
          by the Bedrock Guardrail), this function replaces each one:
            - If the original URL contained a PMC ID and we have that file in
              corpus/, replace with a Markdown link: [PMCxxxxxxx](/corpus/PMCxxxxxxx).
            - If the PMC article is not in our local corpus, use [PMCxxxxxxx].
            - For non-PMC URLs, use [link removed].
          Any leftover {EXTERNAL_URL} tokens (e.g. if the guardrail count
          differs from the number of placeholders) are also cleaned up.

        Args:
            step: a label for this step on the receipt.
            tier: model tier key ("haiku", "sonnet", "opus", "openai").
            label: human-readable model name for the UI.
            system: the system prompt text.
            prompt: the user message text.
            max_tokens: maximum output tokens.

        Returns:
            The (post-substitution) model response text.
        """
        self.emit({"type": "model", "tier": tier, "label": label, "state": "start"})
        t0 = time.monotonic()
        text, usage, matches = self.backend.converse(tier, system, prompt, max_tokens)
        elapsed = round(time.monotonic() - t0, 1)

        cost = self.meter.add_llm(step, tier, label, usage)
        self.emit(
            {
                "type": "model",
                "tier": tier,
                "label": label,
                "state": "done",
                "elapsed_s": elapsed,
                "usage": {
                    "inputTokens": usage.get("inputTokens", 0),
                    "outputTokens": usage.get("outputTokens", 0),
                },
                "cost": round(cost, 6),
            }
        )

        # Process guardrail matches: replace {EXTERNAL_URL} placeholders.
        if matches:
            actions: list[dict] = []
            for match in matches:
                pmcid_m = re.search(r"PMC\d+", match["match"])
                if pmcid_m:
                    pmcid = pmcid_m.group(0)
                    if os.path.exists(f"corpus/{pmcid}.txt"):
                        # We have this paper locally -- link to the corpus viewer.
                        actions.append(
                            {
                                "original": match["match"],
                                "local": pmcid,
                                "reason": "redirected to local corpus",
                            }
                        )
                        text = text.replace("{EXTERNAL_URL}", f"[{pmcid}](/corpus/{pmcid})", 1)
                    else:
                        # PMC article exists but we don't have it locally.
                        actions.append(
                            {
                                "original": match["match"],
                                "local": None,
                                "reason": "PMC article not in local corpus",
                            }
                        )
                        text = text.replace("{EXTERNAL_URL}", f"[{pmcid}]", 1)
                else:
                    # Not a PMC URL at all -- remove it entirely.
                    actions.append(
                        {
                            "original": match["match"],
                            "local": None,
                            "reason": "external link — no local copy",
                        }
                    )
                    text = text.replace("{EXTERNAL_URL}", "[link removed]", 1)

            # Clean up any {EXTERNAL_URL} tokens the loop didn't consume.
            # This can happen if the guardrail count doesn't match the
            # number of placeholders in the text.
            text = text.replace("{EXTERNAL_URL}", "[link removed]")
            self.emit({"type": "guardrail", "actions": actions})

        self.emit({"type": "cost", "total": round(self.meter.total, 6)})
        return text

    # -- routing ---------------------------------------------------------

    def route(self, text: str) -> str:
        """Classify a free-form question via a single cheap Haiku call.

        The routing system prompt (questions.ROUTING_SYSTEM) instructs Haiku
        to reply with exactly one word: SYNTHESIS, ANALYSIS, or DEBATE.
        max_tokens=5 is enough because the expected output is a single word.

        The routing call IS metered and appears in the receipt, but it is NOT
        emitted as a model event (it is infrastructure, not a visible answer step).

        Returns:
            "SYNTHESIS", "ANALYSIS", or "DEBATE".
            Defaults to "SYNTHESIS" if the model returns something unexpected.
        """
        raw, usage, _ = self.backend.converse("haiku", Q.ROUTING_SYSTEM, text, max_tokens=5)
        self.meter.add_llm("routing", "haiku", "Claude Haiku (routing)", usage)
        self.emit({"type": "cost", "total": round(self.meter.total, 6)})

        # Take the first word and upper-case it.  If it's not a valid path,
        # default to SYNTHESIS (the safest, lowest-cost fallback).
        word = raw.strip().upper().split()[0] if raw.strip() else "SYNTHESIS"
        if word not in ROUTE_LABELS:
            word = "SYNTHESIS"
        return word

    # -- Q1: friction gone -- Haiku reads and cites ----------------------

    def question_1(self, text: str = Q.QUESTIONS[0]) -> None:
        """Beat 1: Haiku answers a background question with citations.

        Demo story: "With Bedrock, a researcher can ask a plain question and
        get a cited answer from 650 papers -- no setup, no data leaving AWS."

        Model choice: Haiku 4.5 -- cheapest capable model.  The point of
        beat 1 is to show that even a fast, cheap model gives useful output
        when backed by a good knowledge base.

        Guardrail in action: Haiku is prompted to cite papers as full NCBI
        URLs.  The guardrail intercepts them and the browser shows the
        "Bedrock Guardrail: N links intercepted" badge.
        """
        self.emit({"type": "question", "n": 1, "text": text})
        self.emit({"type": "phase", "label": "retrieving from the knowledge base"})
        chunks = self._retrieve("Q1  retrieval", text)
        self.emit({"type": "retrieval", "count": len(chunks)})
        result = self._model(
            "Q1  synthesis",
            "haiku",
            "Claude Haiku",
            Q.SYNTHESIS_SYSTEM,
            f"Passages:\n{self._context(chunks)}\n\nQuestion: {text}",
        )
        self.emit({"type": "answer", "title": "Cited synthesis  ·  Claude Haiku", "text": result})

    # -- Q2: real work -- Sonnet writes code, Code Interpreter runs it ---

    def question_2(self, text: str = Q.QUESTIONS[1]) -> None:
        """Beat 2: Sonnet writes analysis code; Code Interpreter runs it in a microVM.

        Demo story: "Real analytical work -- Sonnet reads the trial data,
        writes Python, and a Bedrock microVM executes it and returns a chart."

        Model choice: Sonnet 5 -- the right size for code generation.  Haiku
        would write simpler code; Opus would be slower and more expensive.
        Sonnet hits the sweet spot of quality and speed for live demos.

        Code Interpreter: the generated script runs in an isolated AgentCore
        microVM -- not on the demo laptop.  The chart comes back as a base64 PNG
        printed to stdout and extracted by _extract_chart().

        The code fence stripping (re.sub) removes markdown ``` delimiters in
        case Sonnet wraps the code in a code block despite the system prompt
        telling it not to.
        """
        self.emit({"type": "question", "n": 2, "text": text})
        self.emit({"type": "phase", "label": "retrieving trial data"})
        chunks = self._retrieve("Q2  retrieval", text, n=16)  # 16 passages for richer data

        code = self._model(
            "Q2  code generation",
            "sonnet",
            "Claude Sonnet",
            Q.CODEGEN_SYSTEM,
            f"Passages:\n{self._context(chunks)}",
            max_tokens=4096,
        )

        # Strip any markdown code fences -- the system prompt says "output ONLY
        # the code" but models sometimes add fences anyway.
        code = re.sub(r"^```[a-z]*\n?|```$", "", code.strip(), flags=re.M)
        self.emit({"type": "code", "text": code})

        self.emit({"type": "phase", "label": "running in AgentCore Code Interpreter"})
        stdout, seconds = self.backend.code_interpreter_run(code)
        self.meter.add_compute("Q2  analysis run", seconds)
        self.emit({"type": "cost", "total": round(self.meter.total, 6)})

        chart, clean = self._extract_chart(stdout)
        if chart:
            self.emit({"type": "chart", "data": chart})

        # Only emit an answer block if there is non-chart text output to show.
        # Usually the script prints only the CHART_B64 line, so clean is empty.
        if clean:
            self.emit(
                {"type": "answer", "title": "Code Interpreter output  ·  microVM", "text": clean}
            )

    # -- Q3: the hard call -- Opus AND GPT-6 Astra in parallel, then adjudicate

    def question_3(self, text: str = Q.QUESTIONS[2]) -> None:
        """Beat 3: Opus and GPT-6 Astra read evidence in parallel; Sonnet adjudicates.

        Demo story: "For the hardest question -- where experts disagree --
        we run two frontier models from DIFFERENT companies in parallel and
        use a third model to find where they agree and disagree."

        Model choices:
          - Claude Opus 5: Anthropic's most advanced Opus model.
          - OpenAI GPT-6 Astra: OpenAI's most capable model, and a genuinely
            independent second opinion -- a different company's model family,
            reached through the same Bedrock boundary.
          - Claude Sonnet 5: the adjudicator.  Sonnet is fast and accurate
            enough to compare two reviews; Opus would be overkill here.

        Both reviewers get max_tokens=8192 deliberately.  Giving one model a
        smaller budget than the other would rig the comparison -- the shorter
        review would look less thorough for a reason that has nothing to do
        with the model.  (The previous Amazon Nova Pro reviewer was capped at
        4096 because that was Nova's limit; Astra allows 128K, so the two can
        now be held to the same budget.)  max_tokens is a ceiling, not a spend:
        cost follows the tokens actually produced.

        Threading: both reviewers call backend.converse() in parallel via
        ThreadPoolExecutor.  Since both calls emit events, we protect emit()
        with a threading.Lock (safe_emit) so events don't interleave at the
        character level.  CostMeter is also thread-safe (has its own lock).

        After both reviews are done, Sonnet adjudicates: it identifies
        agreements, disagreements, and the highest-value next experiments.
        """
        self.emit({"type": "question", "n": 3, "text": text})
        self.emit({"type": "phase", "label": "retrieving"})
        chunks = self._retrieve("Q3  retrieval", text, n=16)
        prompt = f"Passages:\n{self._context(chunks)}\n\n{text}"

        # Thread-safe emit wrapper -- needed because both reviewers run in parallel.
        emit_lock = threading.Lock()

        def safe_emit(event: dict) -> None:
            with emit_lock:
                self.emit(event)

        results: dict[str, str] = {}

        def run_review(step: str, tier: str, label: str, max_tok: int) -> tuple[str, str]:
            """Run one model review and emit start/done events thread-safely."""
            safe_emit({"type": "model", "tier": tier, "label": label, "state": "start"})
            t0 = time.monotonic()
            txt, usage, _ = self.backend.converse(tier, Q.REVIEW_SYSTEM, prompt, max_tok)
            elapsed = round(time.monotonic() - t0, 1)
            cost = self.meter.add_llm(step, tier, label, usage)
            safe_emit(
                {
                    "type": "model",
                    "tier": tier,
                    "label": label,
                    "state": "done",
                    "elapsed_s": elapsed,
                    "usage": {
                        "inputTokens": usage.get("inputTokens", 0),
                        "outputTokens": usage.get("outputTokens", 0),
                    },
                    "cost": round(cost, 6),
                }
            )
            safe_emit({"type": "cost", "total": round(self.meter.total, 6)})
            return tier, txt

        # Run both reviewers concurrently.  as_completed() yields each future as
        # it finishes, so the adjudication waits for whichever is slower.
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = {
                pool.submit(run_review, "Q3  reading", "opus", "Claude Opus", 8192): "opus",
                pool.submit(
                    run_review, "Q3  reading", "openai", "OpenAI GPT-6 Astra", 8192
                ): "openai",
            }
            for fut in as_completed(futures):
                tier, txt = fut.result()
                results[tier] = txt

        # Adjudication: Sonnet compares the two reviews.
        adjudication = self._model(
            "Q3  adjudication",
            "sonnet",
            "Claude Sonnet",
            Q.ADJUDICATE_SYSTEM,
            "REVIEW A (Claude Opus):\n"
            f"{results['opus']}\n\n"
            f"REVIEW B (OpenAI GPT-6 Astra):\n{results['openai']}",
            4096,
        )
        self.emit(
            {"type": "answer", "title": "Two model families, cross-checked", "text": adjudication}
        )

    # -- Q4: Cedar Gateway demo -- web_fetch blocked, fallback to KB -----

    def question_4(self) -> None:
        """Beat 4: Cedar policy denies web_fetch; agent falls back to the knowledge base.

        Demo story: "Even with an external tool configured, a Cedar policy
        can block specific tool calls.  The agent detects the denial and
        falls back gracefully to the knowledge base."

        The Cedar policy attached to the gateway has a ForbidWeb rule that
        denies any call where context.toolName == "web_fetch".  When the
        denial is returned, the agent emits a policy_denied event (which
        triggers the "Cedar Policy Denied" badge in the UI) and then
        re-runs the question against the knowledge base instead.

        Note: Cedar policy denials arrive as HTTP 200 with a JSON-RPC error
        body, not as HTTP 403.  query_gateway() in aws.py handles this.
        """
        self.emit({"type": "question", "n": 4, "text": Q.QUESTIONS[3]})
        self.emit({"type": "phase", "label": "querying AgentCore Gateway for clinical trial data"})

        # Attempt the web_fetch tool call through the Gateway.
        # With the ForbidWeb Cedar policy, this will be denied.
        result = self.backend.query_gateway(
            "web_fetch",
            {
                "url": (
                    "https://clinicaltrials.gov/api/query/full_studies"
                    "?expr=PCSK9&min_rnk=1&max_rnk=5&fmt=json"
                )
            },
        )

        if result.get("denied"):
            # Cedar policy blocked the request -- show the denial badge.
            self.emit({"type": "policy_denied", "tool": "web_fetch", "reason": result["reason"]})
            self.emit(
                {
                    "type": "phase",
                    "label": "web access denied by Cedar policy — answering from knowledge base",
                }
            )
            # Fall back to the KB for a knowledge-based answer.
            chunks = self._retrieve("Q4  retrieval", Q.QUESTIONS[3], n=12)
            self.emit({"type": "retrieval", "count": len(chunks)})
            text = self._model(
                "Q4  synthesis",
                "haiku",
                "Claude Haiku",
                Q.Q4_GATEWAY_SYSTEM,
                (
                    f"Note: web_fetch was denied by policy.\n\n"
                    f"Passages:\n{self._context(chunks)}\n\n"
                    f"Question: {Q.QUESTIONS[3]}"
                ),
            )
        else:
            # web_fetch succeeded (unexpected with ENFORCE policy, but handled gracefully).
            self.emit({"type": "phase", "label": "processing clinical trial data"})
            trial_data = str(result.get("result", ""))[:2000]  # cap to avoid huge prompts
            chunks = self._retrieve("Q4  retrieval", Q.QUESTIONS[3], n=8)
            text = self._model(
                "Q4  synthesis",
                "haiku",
                "Claude Haiku",
                Q.Q4_GATEWAY_SYSTEM,
                (
                    f"Trial data:\n{trial_data}\n\n"
                    f"Passages:\n{self._context(chunks)}\n\n"
                    f"Question: {Q.QUESTIONS[3]}"
                ),
            )

        self.emit({"type": "answer", "title": "Clinical trials  ·  Claude Haiku", "text": text})

    # -- free-form: route then dispatch ----------------------------------

    def run_freeform(self, text: str) -> None:
        """Route a free-form question via Haiku, then run the appropriate path.

        The flow is:
          1. Classify the question (route()) -- one cheap Haiku call.
          2. Emit a "route" event so the UI can show the chosen path.
          3. Dispatch to question_1, question_2, or question_3 with the
             user's text replacing the canned question.
          4. Emit receipt and done.

        Free-form questions go through the same paths as canned questions --
        they just use the user's text instead of the rehearsed phrasing.
        """
        self.emit({"type": "phase", "label": "classifying question…"})
        path = self.route(text)
        self.emit({"type": "route", "path": path, "label": ROUTE_LABELS[path]})

        dispatch = {
            "SYNTHESIS": self.question_1,
            "ANALYSIS": self.question_2,
            "DEBATE": self.question_3,
        }
        dispatch[path](text)
        self.emit({"type": "receipt", **self.meter.receipt()})
        self.emit({"type": "done"})

    # -- run (canned questions) -----------------------------------------

    def run(self, which: Sequence[int] = (1, 2, 3)) -> None:
        """Run one or more canned questions and emit the final receipt.

        The setup_cost event is emitted first so the browser can populate
        the KB panel before the first question starts.

        Args:
            which: a sequence of question numbers (1-4) to run.
                   Default is (1, 2, 3) -- the three main demo beats.
        """
        # Emit the KB panel costs as the very first event so the sidebar
        # shows measured numbers while the questions are running.
        self.emit({"type": "setup_cost", **self.backend.kb_setup_costs()})

        steps = {
            1: self.question_1,
            2: self.question_2,
            3: self.question_3,
            4: self.question_4,
        }
        for n in which:
            steps[n]()

        self.emit({"type": "receipt", **self.meter.receipt()})
        self.emit({"type": "done"})

    @staticmethod
    def _extract_chart(stdout: str) -> tuple[str | None, str]:
        """Pull a base64 PNG from the Code Interpreter output.

        The generated Q2 script ends with:
            print('CHART_B64:' + base64.b64encode(buf.getvalue()).decode())

        This method finds that token in stdout, validates that it decodes to a
        real PNG (base64.b64decode will raise if the string is corrupt), and
        returns it separately from any other text output.

        Args:
            stdout: the full stdout string from code_interpreter_run().

        Returns:
            (chart_b64, remaining_text) where chart_b64 is the base64 PNG
            string or None if no CHART_B64 token was found.
        """
        m = re.search(r"CHART_B64:([A-Za-z0-9+/=]+)", stdout)
        if not m:
            return None, stdout.strip()
        # Validate before sending to browser -- corrupt base64 would show
        # a broken image placeholder.
        base64.b64decode(m.group(1))
        return m.group(1), stdout.replace(m.group(0), "").strip()
