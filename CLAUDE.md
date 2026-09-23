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
and adjusted, not scaffolding to be filled in. `ruff` is clean and 88 tests pass.

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
- **The beat-4 tool is real** (verified 2026-09-22). `build_kb.create_gateway_target()`
  creates an MCP-category **OpenAPI** target named `web-tools` whose inline schema
  (`web_tools_openapi_schema()`) wraps the public ClinicalTrials.gov v2 API:
  ```python
  create_gateway_target(
      gatewayIdentifier=gateway_id, name="web-tools",
      targetConfiguration={"mcp": {"openApiSchema": {"inlinePayload": <json str>}}},
  )   # credentialProviderConfigurations OMITTED -- see below
  ```
  Facts this rests on, all from current docs:
  - Tool names are `${target_name}___${tool_name}`, three underscores
    ([gateway-tool-naming](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/gateway-tool-naming.html)),
    and for an OpenAPI target the tool half **is the `operationId`**
    ([gateway-schema-openapi](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/gateway-schema-openapi.html):
    "`operationId` … is used as the tool name"). Hence `web-tools___web_fetch`,
    and the Cedar action `AgentCore::Action::"web-tools___web_fetch"`. Those
    strings are one fact in three files — `build_kb.py`, `aws.py`, the Cedar
    rule. Cedar matches the **prefixed** action, never `context.toolName`.
  - `credentialProviderConfigurations` is **optional** (CreateGatewayTarget API
    reference: "Required: No") and the outbound-auth matrix lists *No
    authorization → Yes* for OpenAPI targets. ClinicalTrials.gov v2 is public, so
    it is omitted. Do **not** substitute `GATEWAY_IAM_ROLE`: AWS states SigV4
    outbound only works against services that verify SigV4 (API Gateway, Lambda
    URLs, AgentCore Gateway) — clinicaltrials.gov does not.
  - There is **no Lambda anywhere in this repo.** Earlier comments and a
    `teardown.py` block implied an `inside-the-lines-web-tool` function; nothing
    ever created it and both were removed on 2026-09-22.
- **HTTP passthrough targets cannot serve beat 4** (verified 2026-09-22). They are
  the obvious Lambda-free choice and they are wrong here. HTTP targets are a
  *different category* from MCP targets: "unlike MCP targets, HTTP targets do not
  support capability synchronization or semantic tool search. Clients address each
  target individually through path-based routing"
  ([gateway-targets-http](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/gateway-targets-http.html)).
  A passthrough target is reached at
  `https://{gatewayId}.gateway.bedrock-agentcore.{region}.amazonaws.com/{targetName}/{path}`,
  never appears in `tools/list`, and is not callable via `tools/call` — so there is
  no `web-tools___web_fetch` action name for `ForbidWeb` to match.
  Separately, the pinned botocore (1.43.14) models `targetConfiguration.http` as a
  union with only `agentcoreRuntime`; `passthrough` is not in the local model yet.
- **`query_gateway()` has three outcomes, not two** (2026-09-22): `{"denied": True,
  reason}` for a *recognised* Cedar denial only, `{"result": ...}` for a genuine
  success, `{"error": True, reason}` for everything else. This exists because a
  tool-not-found error carries none of the denial keywords and used to fall through
  to the success branch — the "Cedar Policy Denied" badge would then silently not
  appear on stage. `agent.py`'s `question_4` surfaces the `error` case through a
  `phase` line and never fakes the badge. `backend.py`'s `query_gateway`
  docstring documents all three outcomes.

## Verified live 2026-09-22 (a full run against the real account)

All four beats were run end to end. Total **$0.317**, and the things that had
never actually been exercised are now exercised. Findings worth keeping:

- **Cedar's action IS the prefixed MCP tool name.** The rule must read
  `action == AgentCore::Action::"web-tools___web_fetch"`. It previously read
  `action == "InvokeTool" ... when { context.toolName == "web_fetch" }`, which
  **never matched**. That went unnoticed for months because no real gateway
  target existed, so the call failed for unrelated reasons and the UI showed a
  denial anyway. Once `web-tools` became a real OpenAPI target, beat 4 silently
  *succeeded* — fetching live ClinicalTrials.gov data and demonstrating the
  exact opposite of its own security point. A denial names the policy:
  `Tool Execution Denied: ... [Policy evaluation denied due to ForbidWeb-xxxx]`.
- **Adaptive thinking is on by default and is never displayed.** Opus 5 on Q3
  took **108s** and ran output to the `8192` ceiling (truncating its review).
  With `thinking: disabled` and a "be concise, 400 words" instruction shared by
  both reviewers, Q3 went **141s → 48s** with no truncation. `thinking` is
  Anthropic-only — sending it to Astra returns `unknown_parameter`.
- **`content[0]["text"]` is wrong.** With thinking on, block 0 is
  `reasoningContent`. Join every block that has a `text` key.
- **botocore's 60s read timeout is too low.** Opus 5 raised `ReadTimeoutError`
  mid-run. The `bedrock-runtime` client now sets `read_timeout=600`. Note this
  is unrelated to `app.py`'s `asyncio.to_thread` — that keeps the *WebSocket*
  responsive, not the HTTP read alive.
- **GPT-6 Astra books prompt tokens as cache tokens.** `inputTokens` comes back
  as ~2 while `cacheWriteInputTokens`/`cacheReadInputTokens` hold the real
  count. Taken at face value it understated Q3 by ~$0.14. `converse()` now
  falls back to the cache fields.
- **AWS list responses are inconsistently keyed.** `ListPolicyEngines` →
  `policyEngines`, `ListPolicies` → `policies`, `ListGateways` /
  `ListGatewayTargets` → `items`. Two comments in this repo blamed "API bugs"
  that were really the wrong key being read; `list_policies()` works fine.
- **The gateway policy engine needs `bedrock-agentcore:*Authorize*`.** Without
  it, `update_gateway()` fails and the Cedar beat cannot be wired up at all.
- **Beat 4 was excluded by default.** `agent.run()`, `app.py::_run_agent` and
  `run.py --questions` all defaulted to `(1, 2, 3)`.
- **`run.py` and `app.py` built the backend differently** — `run.py` passed no
  guardrail and no gateway, so `make demo-headless` verified *neither* security
  beat. Both now go through `AwsBackend.from_config()`. Add backend wiring there,
  never in the two entry points.

## Conventions

- Python 3.11, `src/` layout, package is `agentcore_demo`.
- **Always `uv`, never bare `pip`**: `uv pip install -e ".[dev]"`, `uv run ...`.
- **`make install` first, once.** After that `make lint` and `make test` work
  (88 tests). In a fresh checkout `make test` on its own fails at collection with
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
latency and bills thinking as output tokens. It is left on for Q1/Q2 but
**disabled for both Q3 reviewers** (`converse(..., thinking="disabled")`), which
is what took Q3 from 141s to 48s. `thinking` is Anthropic-only — Astra rejects
it with `unknown_parameter`.

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

## Teardown must leave nothing behind (this is a public repo)

The owner's requirement, verbatim: *"The setup and teardown have to be clean and
not leave stuff in people's AWS accounts."* It previously was not. Teardown keyed
**only** off the IDs and names in the current `config.py`, so every rename
between a build and a teardown orphaned the previous generation. Real damage
found on 2026-09-22 in one account: three S3 Vector buckets (each with an index),
a Knowledge Base stuck in `DELETE_UNSUCCESSFUL` since June, two Cedar policy
engines holding five policies, a guardrail `build_kb.py` could never adopt
(it get-or-creates `inside-the-lines-url-filter`, the orphan was
`inside-the-lines-guardrail-v3`), and two IAM roles.

How it works now:

- **Everything `build_kb.py` creates is tagged `Project=inside-the-lines`**, at
  create *and* re-applied on every get-or-create "exists" branch, so resources
  made by an older untagged version get adopted rather than stay invisible.
- **Teardown discovers rather than assumes.** One `discover()` feeds plan,
  delete and verify — so a category the sweep can't see is one verify can't see
  either. Each item is claimed by `config`, `tag`, or **normalised** name
  (lowercased, punctuation stripped — load-bearing, because the policy engine is
  `InsideTheLinesEngine` and shares no literal prefix with anything else).
- **`--dry-run`** lists everything with its provenance and changes nothing. This
  is the "is my account clean?" check; `make teardown-dry-run`.
- **`verify()` runs after deletion and `main()` exits non-zero** if anything
  survives (retried 6×10s so in-flight `DELETING` resources don't false-positive).
- **`dataDeletionPolicy=RETAIN` before `delete_knowledge_base`.** This is what
  unstuck the June zombie: a KB whose vector store is already gone can otherwise
  never finish deleting. `update_data_source` is a **full replacement** — echo
  `name`, `dataSourceConfiguration` and `vectorIngestionConfiguration` back or it
  returns `ValidationException: chunkingConfiguration cannot be updated`.
  Delete the KB **before** its index, or you recreate the zombie.
- **`bedrock` alone spells it `resourceARN`** (capital ARN) and returns tags as a
  **list** of `{key,value}`; `bedrock-agent`, `bedrock-agentcore-control` and
  `s3vectors` use `resourceArn` and return a map. A generic wrapper that guessed
  `resourceArn` silently swallowed the `ParamValidationError` and reported "the
  guardrail simply isn't tagged" — the most dangerous way for a sweep to fail.
- **Three things cannot be tagged**, because AWS gives them no ARN: KB data
  sources, gateway targets, and Cedar policies. They are reached through their
  parents. Don't "fix" this by inventing an ARN.
- The S3 corpus bucket is the only resource whose name the project doesn't
  control, so it is also found via the Resource Groups Tagging API.

**Still unverified:** no tag has yet been applied to a live resource, and no
real deletion has run through this code path. The cheapest closing check is
`python build_kb.py` on an already-provisioned account (idempotent; the adopt
branches stamp the tags) then `make teardown-dry-run` — every line should read
`[config, tag, name]`.

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
   both reviewers get the same 4096-token budget so the comparison is not rigged.
   Astra ($11/$55 per 1M, `us.` Geo) is the priciest model here — Q3 dominates
   the receipt, which is honest and worth saying out loud. See **Models** above
   for the full table and for why Claude Fable 5.1 is deliberately not used.
4. *Secure by policy* — the agent tries to reach ClinicalTrials.gov through the
   AgentCore Gateway; a Cedar `ForbidWeb` policy denies it, and the agent falls
   back to the knowledge base.
   **The tool it tries to call is real**: `web-tools` is an OpenAPI gateway target
   over the live ClinicalTrials.gov v2 API, publishing `web_fetch`. Lift the
   policy and the call genuinely fetches trials. That is the whole claim of the
   beat — the *policy* is what stops it, not the absence of a tool — so do not
   "simplify" the target away. See **Verified AWS facts** for the API shape and
   for why an HTTP passthrough target cannot be used instead.

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
