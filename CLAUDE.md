# CLAUDE.md

Guidance for Claude Code working in this repo. Read this fully before writing code.

## What this is

**Inside the Lines** is a five-minute *live demo* for a 30-minute conference talk
on AWS Bedrock, given to a university research-computing audience. The thesis of
the talk: you can give researchers frontier-model AI agents **without the data
leaving a secure environment**, and pay pennies for it.

The demo drives one biomedical-research agent through four escalating questions
about the gene *PCSK9*, against a Bedrock Knowledge Base of ~1,000 open-access
papers, and shows — in a **local web page, live** — what the agent is doing and
what it costs.

This is a demo, not a product. Optimise for: legibility on a projector,
reliability of a live run, and honest cost numbers.

## Current state — complete and rehearsed

**The demo is built and works.** It was written 2026-05-20 → 06-12 and has been run
live against the real account. Treat everything below as working code to be reviewed
and adjusted, not scaffolding to be filled in. `ruff` is clean and 51 tests pass.

| file | role |
|---|---|
| `corpus_fetch.py` | pulls the PMC paper corpus, licence-filtered |
| `build_kb.py` | provisions the KB (S3 Vectors), Guardrail, Gateway + Cedar policy; idempotent |
| `teardown.py` | deletes every billable resource **and** the local `corpus/` dir |
| `src/agentcore_demo/questions.py` | the four locked questions + system prompts; read its docstring first |
| `src/agentcore_demo/cost.py` | the cost meter (pure logic) |
| `src/agentcore_demo/pricing.py` | Bedrock rate tables and setup-cost derivation |
| `src/agentcore_demo/aws.py` | real AWS backend: `retrieve`, `converse`, `code_interpreter_run`, guardrail, gateway |
| `src/agentcore_demo/backend.py` | the backend protocol both implementations satisfy |
| `src/agentcore_demo/fakes.py` | `FakeBackend` — used by tests and `DEMO_FAKE=1` |
| `src/agentcore_demo/agent.py` | orchestration; runs the questions, emits events |
| `src/agentcore_demo/app.py` | FastAPI: `/ws` WebSocket, serves the page, opens the browser |
| `src/agentcore_demo/run.py` | headless runner — all questions, no browser |
| `src/agentcore_demo/static/index.html` | the live page (Alpine.js, no build step) |
| `tests/` | `test_cost.py`, `test_agent.py`, `test_app.py`, `conftest.py` |

There is no `INITIAL_PROMPT.md` — that planning doc was removed in `91fb746`.

## Architecture

```
corpus_fetch.py ─► S3 bucket ─► build_kb.py ─► Bedrock Knowledge Base (S3 Vectors)
                                                        │
  browser ◄── WebSocket ── app.py (FastAPI) ── agent.py ─┼── Bedrock (Claude + OpenAI)
  (Alpine.js, live)         emits events                 ├── AgentCore Code Interpreter
                                                         ├── Bedrock Guardrail (link redaction)
                                                         └── AgentCore Gateway + Cedar policy
```

`agent.py` talks to a **backend protocol** (`backend.py`), satisfied by `aws.py`
(real) and `fakes.py` (`FakeBackend`). That injection is what keeps AWS out of CI.

The **event protocol** is the contract between `agent.py` and `app.py`. The agent
calls an `emit(event: dict)` callback; the web layer pushes each event over the
WebSocket; the page renders it. Event shapes (see `agent.py` for the source of
truth):

| `type` | fields | meaning |
|---|---|---|
| `question` | `n`, `text` | a new question started |
| `phase` | `label` | status line ("retrieving...") |
| `retrieval` | `count` | N passages retrieved |
| `model` | `tier`, `label`, `state` (`start`/`done`), `usage?`, `cost?` | a model call |
| `answer` | `title`, `text` | a synthesis / adjudication result |
| `code` | `text` | generated analysis code |
| `chart` | `data` (base64 PNG) | the chart, for inline `<img>` |
| `cost` | `total` | running cost-meter total |
| `route` | `path` (`SYNTHESIS`\|`ANALYSIS`\|`DEBATE`), `label` | free-form routing result; emitted before the question event |
| `guardrail` | see `agent.py` | Bedrock Guardrail intercepted external links — redirected to the local corpus copy, or BLOCKED when there is no local match |
| `policy_denied` | see `agent.py` | the Cedar `ForbidWeb` policy denied the Q4 Gateway tool call; agent falls back to the KB |
| `setup_cost` | `ingestion_usd`, `storage_usd_per_month` | KB panel costs: ingestion computed from corpus size, storage from vector count (NOT metered, NOT in run total) |
| `receipt` | `rows`, `total` | the final itemised receipt |
| `done` | — | run complete |

Keep this protocol stable. If the page needs more, add a field; don't repurpose.

## Verified AWS facts (do not re-derive — these were checked)

- **Retrieval**: `boto3.client("bedrock-agent-runtime").retrieve(knowledgeBaseId,
  retrievalQuery={"text": q}, retrievalConfiguration={"vectorSearchConfiguration":
  {"numberOfResults": n}})` → `retrievalResults[].content.text` / `.location` / `.score`.
- **Reasoning**: `boto3.client("bedrock-runtime").converse(modelId, system=[{"text":...}],
  messages=[...], inferenceConfig={"maxTokens":..., "temperature":...})`
  → `output.message.content[0].text` and a `usage` block with `inputTokens` /
  `outputTokens`. `converse` works uniformly across vendors — Claude **and** OpenAI
  GPT-6 Astra take the same request/response shape. Staying on `converse` is also
  what keeps Bedrock Guardrails working: on OpenAI models they are Converse-only.
- **Code Interpreter**: the `bedrock_agentcore` SDK —
  `from bedrock_agentcore.tools.code_interpreter_client import CodeInterpreter`;
  `ci = CodeInterpreter(region); ci.start(); ci.invoke("executeCode",
  {"language":"python","code":...}); ci.stop()`. Result events stream in
  `resp["stream"]`, each with `result.content[]` blocks of `{"type":"text",...}`.
- **Vector store**: the KB uses **Amazon S3 Vectors** (GA, serverless, no hourly
  cost) — not OpenSearch. `build_kb.py` creates an S3 vector bucket + index via
  the `s3vectors` boto3 client and connects it.
- **S3 Vectors wiring is now confirmed working** — `build_kb.py` has been run
  successfully against the real account, so the `s3vectors` method names and the
  `create_knowledge_base` `storageConfiguration` shape in that file are correct as
  written. (This entry previously said "VERIFY before running"; that's done.)
- **Guardrail / Gateway**: `build_kb.py` also provisions a Bedrock Guardrail (used
  to intercept external links in model output and swap in the local corpus copy)
  and an AgentCore Gateway with a Cedar `ForbidWeb` policy for beat 4.

## Conventions

- Python 3.11, `src/` layout, package is `agentcore_demo`.
- **Always `uv`, never bare `pip`**: `uv pip install -e ".[dev]"`, `uv run ...`.
- **`make install` first, once.** After that `make lint` and `make test` work
  (51 tests). In a fresh checkout `make test` on its own fails at collection with
  `ModuleNotFoundError: No module named 'agentcore_demo'`, because `uv run pytest`
  doesn't install the package or the dev extra. If you'd rather not install,
  `uv run --extra dev pytest` works standalone.
- **Lint/format: `ruff`** must be clean. **Tests: `pytest`** must pass. CI runs both
  on every push (it does the editable install explicitly, so CI is unaffected).
- **No AWS calls in tests.** `agent.py` takes an AWS backend by dependency
  injection; tests pass a fake (see `tests/conftest.py`). Never hit the cloud in CI.
- WebSocket/async code uses `pytest-asyncio` (`asyncio_mode = "auto"`).
- Keep functions small; type-hint public functions; docstrings on modules and
  non-obvious functions.

## Models (current as of 2026-09-22 — verified against Bedrock model cards)

| tier | Bedrock inference profile | Bedrock rate /1M in→out |
|---|---|---|
| `haiku` | `us.anthropic.claude-haiku-4-5-20251001-v1:0` | $1.00 → $5.00 |
| `sonnet` | `us.anthropic.claude-sonnet-5` | $2.00 → $10.00 |
| `opus` | `us.anthropic.claude-opus-5` | $5.00 → $25.00 |
| `openai` | `us.openai.gpt-6-astra` | $11.00 → $55.00 |

All four are `us.` **Geo** cross-Region profiles: requests stay within US (and,
for the Anthropic ones, Canada) Regions. `global.` is cheaper but AWS documents
it as routing anywhere worldwide — which would contradict the talk's thesis.
Neither the OpenAI models nor Sonnet 5 offer in-Region inference on
`bedrock-runtime` at all, so a cross-Region profile is mandatory, not a choice.

All four support the **Converse** API and **Bedrock Guardrails** on
`bedrock-runtime`. Staying on Converse is load-bearing: on the OpenAI models
Guardrails are Converse-only, and on `bedrock-mantle` Guardrails are not
supported at all. Do not "modernise" this onto Mantle or the Responses API.

**Adaptive thinking is on by default** on Opus 5 and Sonnet 5, including when
the request omits `thinking` — which `aws.py` does. Opus 4.7 behaved the
opposite way, so the Sept 2026 bump turned thinking on silently. It raises
latency and bills thinking as output tokens. Left on deliberately (it helps Q3);
see the note in `aws.py` for how to disable it if rehearsal timing demands.

**Why not Claude Fable 5.1?** It is Anthropic's most capable widely released
model, it is on Bedrock, it works in us-west-2 via `us.anthropic.claude-fable-5-1`,
and it supports Converse and Guardrails — so it would drop straight in, at
~$11/$55 (a neat match for Astra). It is deliberately **not** used because:
1. It ships blocking classifiers for dual-use **life-sciences** content and AWS
   states refusal rates are *materially higher* than previous Claude models,
   with `stop_reason: "refusal"` to be treated as a primary response path.
   This demo is entirely biomedical (PCSK9 inhibition, adverse effects, trials)
   and has **no refusal handling**. A refusal live on stage is unrecoverable.
2. It requires opting the account into `aws_review` data retention — awkward to
   defend in a talk about data staying inside your boundary.
3. Thinking cannot be disabled and it is built for multi-hour agentic jobs —
   wrong latency profile for a rehearsed five-minute demo.
Revisit only with refusal handling in place and a rehearsal that proves timing.

## Guardrails — do not violate

- **No console.** Everything is scripted. Don't add "go click in the AWS console"
  steps anywhere.
- **No real identifiers committed.** `config.py` is git-ignored; only
  `config.example.py` is tracked. Never hard-code account IDs, bucket names, or
  KB IDs.
- **Always provide teardown.** Any AWS resource a script creates, `teardown.py`
  must be able to delete.
- **The four questions are locked** (`questions.py`). They are rehearsed for the
  live talk; do not reword them.
- **Cost numbers must be real.** The cost meter computes from actual `usage`
  tokens × the rates in `config.py`. Never fabricate or hard-code a total.
- **Frontend: lightweight.** Alpine.js via CDN, plain HTML/CSS, no build step,
  no npm, no bundler. The page must open by just loading a file the FastAPI app
  serves.

## The demo's four beats (for context, so the UI tells the right story)

1. *Friction gone* — one plain question; Claude Haiku reads and cites. The Bedrock
   Guardrail intercepts the NCBI URLs and the UI shows the interception badge.
2. *Real work, faster* — Claude Sonnet writes analysis code; AgentCore Code
   Interpreter runs it in an isolated microVM and returns a chart.
3. *A second opinion* — Claude Opus **and** OpenAI GPT-6 Astra read the evidence
   independently; Sonnet adjudicates where the two model families disagree.
   Astra replaced Amazon Nova Pro in Sept 2026 once OpenAI models reached Bedrock;
   both reviewers get the same 8192-token budget so the comparison is not rigged.
   Astra ($11/$55 per 1M, `us.` Geo) is the priciest model here — Q3 dominates
   the receipt, which is honest and worth saying out loud. See **Models** above
   for the full table and for why Claude Fable 5.1 is deliberately not used.
4. *Secure by policy* — the agent tries to reach ClinicalTrials.gov through the
   AgentCore Gateway; a Cedar `ForbidWeb` policy denies it, and the agent falls
   back to the knowledge base.

Then the receipt: a total well under a dollar, billed only for what ran.
"Run complete" belongs **after Q4**, not after Q3.

`questions.py` carries the full per-beat rationale — which model, and why that
model. Read it before changing anything about routing or model choice.

## UI decisions that are settled (do not re-litigate)

These were arrived at by watching the demo on a screen; several took many rounds.

- The chips + prompt **start at the top and are pushed DOWN** by content appearing
  above them. Not sticky-top, not bottom-anchored — it is plain DOM order
  (history before composer in `index.html`). Both "obvious" alternatives were
  explicitly rejected.
- Fades run 0.7–1.0s. Faster was rejected as "popping in".
- Font is **Atkinson Hyperlegible** / **Atkinson Hyperlegible Mono**.
- No pure white on pure black — the contrast was called unreadable.
- Label every number ("Tokens 7,361 in / 322 out"); the `$` must not be smaller
  than the amount; show plenty of significant digits.
- Show a live elapsed timer per running model; timers must not reset when a
  sibling model finishes first.
- The KB panel stays **visible at all times**, including its ingest state — it is
  something the speaker talks about. Never let a status line get overwritten.
- Any of the four questions may also be asked free-form; the canned chips just
  teletype the locked text and fire it, and track which have run.
