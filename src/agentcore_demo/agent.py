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
#   policy_denied {tool, reason}             Cedar Gateway policy denied a tool call.
#                                            Emitted ONLY for a recognised Cedar
#                                            denial -- a gateway that simply failed
#                                            reports itself through `phase` instead,
#                                            so the badge never overstates the case.
#   receipt       {rows, total}              the final itemised receipt
#   done          {}                         run complete; browser shows receipt

# Human-readable descriptions for each routing path.
# These appear in the "route" event label field and in the UI.
# Display names for each model tier, versions included.
#
# One dict so the receipt, the model rows and the route labels can never disagree
# with each other.  Versions are shown deliberately (2026-09-22): the OpenAI
# entry always carried one ("GPT-6 Astra") while the Anthropic ones did not, so a
# projected receipt read "Claude Opus" beside "OpenAI GPT-6 Astra" and invited
# the question "which Opus?".  On a slide about frontier models, the generation
# is the interesting part.
#
# These MUST match the model IDs in config.MODELS.  They are not derived from it
# because config.py is git-ignored and absent in CI, so nothing here can import
# it; test_model_labels_look_versioned() in tests/test_agent.py is the guard.
MODEL_LABELS = {
    "haiku": "Claude Haiku 4.5",
    "sonnet": "Claude Sonnet 5",
    "opus": "Claude Opus 5",
    "openai": "OpenAI GPT-6 Astra",
}

ROUTE_LABELS = {
    "SYNTHESIS": f"retrieval + synthesis · {MODEL_LABELS['haiku']}",
    "ANALYSIS": f"code generation + chart · {MODEL_LABELS['sonnet']} + Code Interpreter",
    "DEBATE": (
        f"dual review + adjudication · {MODEL_LABELS['opus']} "
        f"+ {MODEL_LABELS['openai']} + {MODEL_LABELS['sonnet']}"
    ),
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
        # thinking="disabled" on EVERY model call in this demo (2026-09-22).
        #
        # Nothing here ever renders a thinking block -- thinking.display defaults
        # to "omitted" on the Claude 5 models, so the audience sees none of it.
        # Leaving it on therefore buys nothing and costs three ways: latency,
        # output-token charges, and the output budget itself.  That last one is
        # not theoretical:
        #   - Q3, Opus 5, max_tokens=8192: 108s and the review was truncated at
        #     exactly 8192 because thinking ate the budget.
        #   - Q2, Sonnet 5, max_tokens=4096: returned ONLY a reasoningContent
        #     block and NO text at all -- zero lines of analysis code.  Caught
        #     by the guard in aws.py::converse rather than silently charting
        #     nothing, but fatal to the beat either way.
        # Opus 4.7 ran without thinking unless asked, which is why none of this
        # was a problem before the Sept 2026 model refresh.
        text, usage, matches = self.backend.converse(
            tier, system, prompt, max_tokens, thinking="disabled"
        )
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

        # Linkify bare PMC IDs against the local corpus.
        #
        # This is INDEPENDENT of the guardrail (2026-09-22).  Beat 1 asks the
        # model for bare "[PMC13156736]" citations precisely so that no external
        # URL exists and no guardrail badge fires -- but the papers should still
        # be clickable, and until now the only thing that produced a /corpus/
        # link was guardrail substitution above.  Doing it here means citations
        # work with or without interception, which also makes every other beat
        # more robust: if the guardrail ever misses a URL, the bracketed ID it
        # leaves behind still becomes a link.
        #
        # Runs last, and only on IDs that are not already inside a Markdown
        # link, so it cannot double-wrap what the guardrail block just built.
        text = self._linkify_bare_pmc_ids(text)

        self.emit({"type": "cost", "total": round(self.meter.total, 6)})
        return text

    @staticmethod
    def _linkify_bare_pmc_ids(text: str) -> str:
        """Turn "[PMCxxxxxxx]" into a link to the local copy, where we have one.

        Only rewrites IDs we actually hold in corpus/ -- an ID we do not have
        stays plain text rather than becoming a dead link, which matches what
        the guardrail path does for the same case.

        The negative lookahead for "(" is what stops this from double-wrapping
        a citation the guardrail block already turned into
        "[PMCxxxxxxx](/corpus/PMCxxxxxxx)".
        """

        def repl(m: re.Match) -> str:
            pmcid = m.group(1)
            if os.path.exists(f"corpus/{pmcid}.txt"):
                return f"[{pmcid}](/corpus/{pmcid})"
            return m.group(0)

        return re.sub(r"\[(PMC\d+)\](?!\()", repl, text)

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
        """Beat 1: the plain opener -- one question, one cited answer, no security theatre.

        Added 2026-09-22.  Beat 2 used to open the demo while also carrying the
        Guardrail interception badge, which forced a choice in the first thirty
        seconds: explain a security mechanism nobody had been introduced to, or
        leave something visible on screen unexplained.  This beat carries one
        idea -- "ask a question, get a cited answer from your own 1,000 papers,
        for a fraction of a cent" -- and beat 2 then gets to be a reveal.

        It deliberately does NOT fire the Guardrail: PLAIN_SYSTEM asks for bare
        PMC IDs rather than full NCBI URLs, so there is no external URL to
        intercept and no badge appears.  Citations stay clickable because
        _model() linkifies bare PMC IDs against the local corpus directly --
        that path does not depend on guardrail interception.

        Model: Claude Haiku 4.5, the cheapest capable model.  Measured ~19s.
        """
        self.emit({"type": "question", "n": 1, "text": text})
        self.emit({"type": "phase", "label": "retrieving from the knowledge base"})
        chunks = self._retrieve("Q1  retrieval", text)
        self.emit({"type": "retrieval", "count": len(chunks)})
        answer = self._model(
            "Q1  synthesis",
            "haiku",
            MODEL_LABELS["haiku"],
            Q.PLAIN_SYSTEM,
            f"Passages:\n{self._context(chunks)}\n\n{text}",
            max_tokens=1200,
        )
        self.emit({"type": "answer", "title": "Answer  ·  Claude Haiku", "text": answer})

    def question_2(self, text: str = Q.QUESTIONS[1]) -> None:
        """Beat 1: Haiku answers a background question with citations.

        Demo story: "With Bedrock, a researcher can ask a plain question and
        get a cited answer from 1,000 papers -- no setup, no data leaving AWS."

        Model choice: Haiku 4.5 -- cheapest capable model.  The point of
        beat 1 is to show that even a fast, cheap model gives useful output
        when backed by a good knowledge base.

        Guardrail in action: Haiku is prompted to cite papers as full NCBI
        URLs.  The guardrail intercepts them and the browser shows the
        "Bedrock Guardrail: N links intercepted" badge.
        """
        self.emit({"type": "question", "n": 2, "text": text})
        self.emit({"type": "phase", "label": "retrieving from the knowledge base"})
        chunks = self._retrieve("Q2  retrieval", text)
        self.emit({"type": "retrieval", "count": len(chunks)})
        result = self._model(
            "Q2  synthesis",
            "haiku",
            MODEL_LABELS["haiku"],
            Q.SYNTHESIS_SYSTEM,
            f"Passages:\n{self._context(chunks)}\n\nQuestion: {text}",
        )
        self.emit({"type": "answer", "title": "Cited synthesis  ·  Claude Haiku", "text": result})

    # -- Q2: real work -- Sonnet writes code, Code Interpreter runs it ---

    def question_3(self, text: str = Q.QUESTIONS[2]) -> None:
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
        self.emit({"type": "question", "n": 3, "text": text})
        self.emit({"type": "phase", "label": "retrieving trial data"})
        chunks = self._retrieve("Q3  retrieval", text, n=16)  # 16 passages for richer data

        code = self._model(
            "Q3  code generation",
            "sonnet",
            MODEL_LABELS["sonnet"],
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
        self.meter.add_compute("Q3  analysis run", seconds)
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

    def question_4(self, text: str = Q.QUESTIONS[3]) -> None:
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

        Both reviewers get the SAME max_tokens and the same thinking setting,
        deliberately.  Giving one model a smaller budget than the other would rig
        the comparison -- the shorter review would look less thorough for a
        reason that has nothing to do with the model.

        Both are also run with thinking DISABLED, and the budget is 4096 rather
        than 8192.  Measured against real AWS on 2026-09-22, the original
        settings cost the demo dearly:
            Opus 5, thinking on, max_tokens=8192 -> 108.2s, output hit 8192
              exactly (i.e. the review was truncated), Q3 total 141.2s
            Opus 5, thinking off, max_tokens=3000 ->  27.5s, output 1,631
        Astra produced a complete 1,566-token review, so even 3000 is ample; the
        shipped 4096 is that plus headroom, and 8192 was simply letting Opus run
        until it was cut off.  With these settings plus the "max 400 words"
        instruction in REVIEW_SYSTEM, Q3 came in at 48s against 141s before.
        Adaptive thinking is
        on by default on Claude 5 models and its output is NEVER displayed
        (thinking.display defaults to omitted), so leaving it on spent ~80
        seconds of a 300-second demo, plus output-token charges, on text no
        audience member would ever see.

        Threading: both reviewers call backend.converse() in parallel via
        ThreadPoolExecutor.  Since both calls emit events, we protect emit()
        with a threading.Lock (safe_emit) so events don't interleave at the
        character level.  CostMeter is also thread-safe (has its own lock).

        After both reviews are done, Sonnet adjudicates: it identifies
        agreements, disagreements, and the highest-value next experiments.
        """
        self.emit({"type": "question", "n": 4, "text": text})
        # Say WHERE from, like Q1 and Q2 do.  These phase lines are the only
        # explanation the audience gets of what happens between clicking a
        # question and a model timer appearing, so a bare "retrieving" wastes
        # the one chance to say "this is a vector search over your own papers".
        self.emit({"type": "phase", "label": "retrieving from the knowledge base"})
        chunks = self._retrieve("Q4  retrieval", text, n=16)
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
            txt, usage, _ = self.backend.converse(
                tier, Q.REVIEW_SYSTEM, prompt, max_tok, thinking="disabled"
            )
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
                pool.submit(run_review, "Q4  reading", "opus", MODEL_LABELS["opus"], 4096): "opus",
                pool.submit(
                    run_review, "Q4  reading", "openai", MODEL_LABELS["openai"], 4096
                ): "openai",
            }
            for fut in as_completed(futures):
                tier, txt = fut.result()
                results[tier] = txt

        # Adjudication: Sonnet compares the two reviews.
        adjudication = self._model(
            "Q4  adjudication",
            "sonnet",
            MODEL_LABELS["sonnet"],
            Q.ADJUDICATE_SYSTEM,
            f"REVIEW A ({MODEL_LABELS['opus']}):\n"
            f"{results['opus']}\n\n"
            f"REVIEW B ({MODEL_LABELS['openai']}):\n{results['openai']}",
            4096,
        )
        self.emit(
            {"type": "answer", "title": "Two model families, cross-checked", "text": adjudication}
        )

    # -- Q4: Cedar Gateway demo -- web_fetch blocked, fallback to KB -----

    # Arguments for the web_fetch tool.  These are the query parameters of the
    # ClinicalTrials.gov v2 /studies operation as declared in the "web-tools"
    # OpenAPI target (build_kb.web_tools_openapi_schema()) -- NOT a free-form URL.
    # The Gateway validates arguments against that schema, so this dict and the
    # schema have to agree.  Dotted names are the registry's real wire names.
    _Q5_WEB_FETCH_ARGS = {
        "query.term": "PCSK9",
        "filter.overallStatus": "RECRUITING,NOT_YET_RECRUITING",
        "pageSize": 5,
        "format": "json",
        "countTotal": True,
    }

    def question_5(self) -> None:
        """Beat 4: Cedar policy denies web_fetch; agent falls back to the knowledge base.

        Demo story: "Even with an external tool configured, a Cedar policy
        can block specific tool calls.  The agent detects the denial and
        falls back gracefully to the knowledge base."

        The tool is real.  "web-tools" is an OpenAPI gateway target over the
        public ClinicalTrials.gov v2 API, so if you removed the Cedar policy this
        call would genuinely fetch trials.  That matters for the claim being made
        on stage: the policy is what stops it, not the absence of a tool.

        The Cedar policy attached to the gateway has a ForbidWeb rule whose
        action is the fully prefixed MCP tool name:
            forbid(principal, action == AgentCore::Action::"web-tools___web_fetch",
                   resource == AgentCore::Gateway::"<arn>");
        The prefix matters.  An earlier version matched on
        `context.toolName == "web_fetch"` and never fired at all (verified
        against real AWS 2026-09-22) -- which was invisible while no real gateway
        target existed, and became a silent SUCCESS on stage as soon as one did.
        When the denial is returned, the agent emits a policy_denied event (which
        triggers the "Cedar Policy Denied" badge in the UI) and then
        re-runs the question against the knowledge base instead.

        Three outcomes, because conflating them is how this beat breaks quietly:
          denied  -> emit policy_denied (the badge) and answer from the KB.
          result  -> the policy is not in force; answer WITH the trial data.
          error   -> the gateway call failed for a reason that is not a policy
                     decision (target missing, bad URL, network).  We must NOT
                     show the badge: it would credit Cedar for an outage.  We
                     say so in the phase line and still answer from the KB, so
                     the run survives but nobody is misled.

        Note: Cedar policy denials arrive as HTTP 200 with a JSON-RPC error
        body, not as HTTP 403.  query_gateway() in aws.py handles this.
        """
        self.emit({"type": "question", "n": 5, "text": Q.QUESTIONS[4]})
        self.emit({"type": "phase", "label": "querying AgentCore Gateway for clinical trial data"})

        # Attempt the web_fetch tool call through the Gateway.
        # With the ForbidWeb Cedar policy in ENFORCE mode, this will be denied.
        result = self.backend.query_gateway("web_fetch", dict(self._Q5_WEB_FETCH_ARGS))

        if result.get("denied"):
            # Cedar policy blocked the request -- show the denial badge.
            self.emit({"type": "policy_denied", "tool": "web_fetch", "reason": result["reason"]})
            self.emit(
                {
                    "type": "phase",
                    "label": "web access denied by Cedar policy — answering from knowledge base",
                }
            )
            text = self._q5_from_kb("web_fetch was denied by policy.")
        elif result.get("error"):
            # Not a policy decision.  Report it in the open -- a missing or FAILED
            # "web-tools" target lands here, and that is precisely the condition
            # that used to masquerade as a success.  Reusing the `phase` event
            # keeps the page unchanged; a fabricated badge would not be honest.
            reason = result.get("reason", "unknown gateway error")
            self.emit(
                {
                    "type": "phase",
                    "label": (
                        f"gateway call failed (NOT a policy denial): {reason} "
                        "— answering from knowledge base"
                    ),
                }
            )
            text = self._q5_from_kb(f"web_fetch could not be called: {reason}")
        else:
            # web_fetch succeeded -- the policy is not in ENFORCE mode (or was
            # removed).  Use the live trial data; the beat's security point is
            # simply not being made on this run.
            self.emit({"type": "phase", "label": "reading the trial data returned by the Gateway"})
            trial_data = str(result.get("result", ""))[:2000]  # cap to avoid huge prompts
            chunks = self._retrieve("Q5  retrieval", Q.QUESTIONS[4], n=8)
            text = self._model(
                "Q5  synthesis",
                "haiku",
                MODEL_LABELS["haiku"],
                Q.Q4_GATEWAY_SYSTEM,
                (
                    f"Trial data:\n{trial_data}\n\n"
                    f"Passages:\n{self._context(chunks)}\n\n"
                    f"Question: {Q.QUESTIONS[4]}"
                ),
            )

        self.emit({"type": "answer", "title": "Clinical trials  ·  Claude Haiku", "text": text})

    def _q5_from_kb(self, note: str) -> str:
        """Answer Q4 from the knowledge base alone, telling the model why.

        Shared by both no-web-data paths (policy denial and gateway error) so the
        fallback answer is identical apart from the one-line explanation, and so
        neither path can drift from the other.

        Args:
            note: a short reason, shown to the model, for why there is no web data.

        Returns:
            The synthesised answer text.
        """
        chunks = self._retrieve("Q5  retrieval", Q.QUESTIONS[4], n=12)
        self.emit({"type": "retrieval", "count": len(chunks)})
        return self._model(
            "Q5  synthesis",
            "haiku",
            MODEL_LABELS["haiku"],
            Q.Q4_GATEWAY_SYSTEM,
            (f"Note: {note}\n\nPassages:\n{self._context(chunks)}\n\nQuestion: {Q.QUESTIONS[4]}"),
        )

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
            "SYNTHESIS": self.question_2,
            "ANALYSIS": self.question_3,
            "DEBATE": self.question_4,
        }
        dispatch[path](text)
        self.emit({"type": "receipt", **self.meter.receipt()})
        self.emit({"type": "done"})

    # -- run (canned questions) -----------------------------------------

    def run(self, which: Sequence[int] = (1, 2, 3, 4, 5)) -> None:
        """Run one or more canned questions and emit the final receipt.

        The setup_cost event is emitted first so the browser can populate
        the KB panel before the first question starts.

        Args:
            which: a sequence of question numbers (1-4) to run.
                   Default is (1, 2, 3, 4, 5) -- all five demo beats.  Q5 used to be
                   excluded by default, which meant the Cedar beat never ran
                   unless it was asked for by name; "Run complete" belongs after
                   Q4, not after Q3.
        """
        # Emit the KB panel costs as the very first event so the sidebar
        # shows measured numbers while the questions are running.
        self.emit({"type": "setup_cost", **self.backend.kb_setup_costs()})

        steps = {
            1: self.question_1,
            2: self.question_2,
            3: self.question_3,
            4: self.question_4,
            5: self.question_5,
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
