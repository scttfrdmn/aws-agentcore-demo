# Inside the Lines — PCSK9

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![AWS Bedrock](https://img.shields.io/badge/AWS-Bedrock-orange.svg)](https://aws.amazon.com/bedrock/)

A five-minute live demo for a research-computing audience showing frontier
AI agents running entirely within a secure AWS boundary, billed only for
what they use.

The agent answers five questions about the gene *PCSK9* against a Bedrock
Knowledge Base of 1,000 open-access PMC papers (~48 MB), with a live cost
meter and a receipt at the end.

## Quickstart

```bash
git clone https://github.com/scttfrdmn/aws-agentcore-demo && cd aws-agentcore-demo
make start
```

That is the whole setup. There is **nothing to fill in and nothing to paste**:
`make start` writes `config.py` for you, reads your AWS account ID from
`sts get-caller-identity`, derives and **creates** the S3 bucket, downloads the
papers, uploads them, provisions every AWS resource, writes the resulting
resource IDs back into `config.py`, and opens the demo in your browser.

It is safe to re-run at any time. Every step skips itself when it is already
done, so if you Ctrl-C in the middle — or something fails — just run `make start`
again and it picks up where it stopped. Nothing is ever created twice.

You need AWS credentials that work (`aws configure`, or `export AWS_PROFILE=...`)
and [`uv`](https://github.com/astral-sh/uv). Nothing else. If something is
missing, `make start` stops **before spending anything** and tells you in one
sentence what to run — or check first, for free, with `make preflight`.

## What it costs

| | |
|---|---|
| Fetch the papers from PubMed Central | **$0.00** — free, but 10–15 minutes |
| Build the knowledge base (one-time) | **$0.283** — embedding ~14.1M tokens |
| Each full five-beat demo run | **$0.32** |
| Leave it provisioned and never run it | **~$0.013/month** (about a penny) |

**First day, soup to nuts: about 60 cents. Every rehearsal after that: about 32
cents.** Leaving it provisioned between rehearsals costs about a penny a month,
so there is no urgency to tear down — but `make teardown` removes everything and
verifies nothing is left behind.

Q4 dominates a run: the two parallel frontier reviews (Claude Opus 5 and OpenAI
GPT-6 Astra) are **~76% of the per-run cost**. That is why a run costs 32 cents
rather than 3, and it is worth saying out loud on stage.

`make start` prints this same summary before it creates anything, so you do not
have to read this file to know what you are spending. It does not stop to ask for
confirmation — informed, not interrogated. Use `make plan` if you want to look
without touching anything.

<details>
<summary>The fine print on those numbers</summary>

These are **us-west-2 Bedrock prices measured on 2026-09-22**, not estimates:
ingestion was 56,599,850 characters ≈ 14.1M tokens × $0.02/1M (Titan Embed V2),
one full run measured **$0.322552**, idle is 30,261 vectors / 192.1 MB of S3
Vectors storage ($0.0113/month) plus 1,000 files / 54.0 MB of S3 ($0.0012/month),
and the AgentCore Code Interpreter's per-second charge is real but rounds to
~$0.00008 per run.

Two honest caveats. **Rates move** — re-check before you rely on these. And
**your numbers will differ slightly**: the corpus is whatever PMC returns on the
day, so character count and vector count vary by a few percent, and the ingestion
figure is *computed* from corpus size × the live embedding rate rather than
metered by AWS (the KB panel in the UI labels it that way too).

</details>

| Beat | Question | Models | Security feature |
|------|----------|--------|-----------------|
| Q1 | What does PCSK9 do, and why do cardiologists care? | Claude Haiku | none — deliberately. A plain question and a cited answer, nothing else on screen |
| Q2 | Role of PCSK9 in LDL regulation | Claude Haiku | **Bedrock Guardrail** intercepts NCBI URLs → redirects to local corpus |
| Q3 | Compare trial LDL-lowering + chart | Claude Sonnet + Code Interpreter | Isolated microVM for code execution |
| Q4 | Where does the literature disagree? | Claude Opus + OpenAI GPT-6 Astra (parallel) + Sonnet adjudicates | **Bedrock Guardrail** (same as Q2) |
| Q5 | Search ClinicalTrials.gov for ongoing trials | Claude Haiku | **AgentCore Gateway Cedar policy** blocks a real web tool |

Q1 exists so that Q2 can be a reveal. Q1 and Q2 ask nearly the same thing; the
only difference is that Q1's system prompt asks for bare `[PMCxxxxxxx]` citations
while Q2's asks for full NCBI URLs. So Q1 has nothing for the Guardrail to
intercept and shows no badge, and Q2 shows the interception on otherwise
identical ground. Q1's citations are still clickable — bare PMC IDs are linked to
the local corpus directly, without needing the guardrail.

Q5 is worth a note: the `web_fetch` tool the agent reaches for is not a prop. It is
an AgentCore Gateway **OpenAPI target** (`web-tools`) pointing at the public
ClinicalTrials.gov v2 API — no Lambda, nothing to maintain. Remove the Cedar
`ForbidWeb` policy and the call really does fetch trials. The demo's point is that
the *policy* stops it.

---

## Prerequisites

- AWS account with Bedrock model access enabled in **us-west-2**
  (check: `aws bedrock list-inference-profiles --region us-west-2`)
- Four reasoning models must be ACTIVE: Haiku 4.5, Sonnet 5, Opus 5, OpenAI
  GPT-6 Astra (OpenAI models have no in-Region option on `bedrock-runtime` —
  the `us.` cross-Region profile below is required, not optional)
- Amazon Titan Text Embeddings V2 (`amazon.titan-embed-text-v2:0`) must also be
  ACTIVE — it is what embeds the corpus into the Knowledge Base
- Python 3.11+ and [`uv`](https://github.com/astral-sh/uv)

- AWS credentials with permissions for `iam`, `s3`, `s3vectors`, `bedrock`,
  `bedrock-agent`, `bedrock-runtime`, `bedrock-agent-runtime`,
  `bedrock-agentcore-control` and `bedrock-agentcore` (Code Interpreter +
  Gateway). Read access to the AWS Price List API (`pricing`, us-east-1) is
  optional — without it the demo falls back to the rates in `config.py`.

You do **not** need to create an S3 bucket, look up your account ID, or install
the AWS CLI (everything here uses boto3 directly). `make preflight` checks every
item above for you, read-only and free, and names anything that is missing.

---

## One-time setup

```bash
AWS_PROFILE=your-profile make start
```

That is it. The rest of this section explains what those minutes are doing, so
you can follow along or run the steps one at a time — but **you do not need to
read it to succeed**.

### What `make start` does, in order

**1. Writes `config.py`.** Generated from the tracked `config.example.py`, with
three values filled in for you:

| value | where it comes from |
|---|---|
| `ACCOUNT_ID` | `sts get-caller-identity` |
| `BUCKET` | derived as `inside-the-lines-pcsk9-<account-id>` — globally unique by construction, so it cannot collide with anyone else's bucket |
| `REGION` | your profile's region if Bedrock there has all four models, otherwise `us-west-2` |

`config.py` is git-ignored and it is yours. **Any real value you put in it wins** —
the automatic fillers only ever replace an empty string or the shipped
placeholder, never a value you chose.

**2. Preflight.** Read-only and free: credentials, region, the four models plus
the Titan embedder, and — via `iam:SimulatePrincipalPolicy` — whether your
identity may actually perform the 15 actions provisioning needs. If anything is
wrong it stops here, **before spending a cent**, with one sentence per problem
naming the command that fixes it. Run it alone any time with `make preflight`.

**3. Fetches the paper corpus** (`make corpus`). Downloads 1,000 CC0 / CC BY
PCSK9 papers (~48 MB) from PubMed Central via the NCBI E-utilities API into
`corpus/` (gitignored, local only). Free, but budget 10–15 minutes: reaching
1,000 keepers takes roughly 1,800 `efetch` calls and NCBI rate-limits anonymous
callers to 3 requests/sec. Setting `NCBI_API_KEY` raises that to 10/sec — see the
header of `corpus_fetch.py` for how to get a key and how the licence filter
works. Resumable, and skipped entirely once complete.

**4. Provisions everything** (`make build-kb`), about 10 minutes:

- **creates the S3 corpus bucket** if it is not already there, and tags it
  `Project=inside-the-lines` so teardown can find it later
- **uploads `corpus/` into it** — in parallel, in pure boto3 (no AWS CLI needed),
  skipping files already present, so a re-run takes about a second
- creates two IAM service roles (one for the Knowledge Base, one for the
  Gateway), the S3 Vectors store, the Bedrock Knowledge Base and its S3 data
  source, runs the ingestion job, creates the Guardrail, and sets up the
  AgentCore Gateway — the `web-tools` OpenAPI target that publishes `web_fetch`,
  plus the Cedar policies that deny it
- **writes the seven resulting resource IDs into `config.py`** (`KB_ID`,
  `DATA_SOURCE_ID`, `GUARDRAIL_ID`, `GUARDRAIL_VERSION`, `GATEWAY_ID`,
  `GATEWAY_URL`, `GATEWAY_ENGINE_ID`) and prints them so you can see what
  changed. It backs the file up to `config.py.bak` first, rewrites only those
  specific lines — every comment and every other setting survives untouched —
  and writes nothing at all if the values are already correct

Every step is get-or-create, so re-running is safe and cheap.

If the gateway target does not reach `READY`, the script **stops with the
target's own `statusReasons`** rather than continuing. That is deliberate: a
missing target is exactly what makes Q5 fail silently, so it has to fail loudly
here instead of on stage.

**5. Opens the demo** at `http://localhost:8000`.

### Running the steps individually

`make start` is the whole chain, but each piece still works on its own, in this
order — and each one is idempotent:

```bash
make preflight   # read-only checks, free
make corpus      # download the papers (resumable)
make build-kb    # bucket + upload + provision + write config.py
make demo        # run the web app
make plan        # show what `make start` would do, and what it costs
```

### Verify with a headless run

```bash
AWS_PROFILE=your-profile make demo-headless
```

Runs all five questions and prints a receipt. Expected cost: **about $0.32** —
a full five-beat run measured **$0.322552** on 2026-09-22. Q4 dominates it
(~76% of the total), because it runs Claude Opus 5 and GPT-6 Astra in parallel
and Astra is the priciest model in the set.

Under Q5, the line to look for is:

```
  ▸ web access denied by Cedar policy — answering from knowledge base
```

If instead you see `gateway call failed (NOT a policy denial): ...`, the Cedar
policy is not what stopped the call — read the reason. Most likely the
`web-tools` target is missing or `GATEWAY_URL` in `config.py` is stale. Do not
go on stage until Q5 reports the Cedar denial.

Note that the headless renderer prints only the `phase` line above; it does not
render the `policy_denied` or `guardrail` events themselves. To *see* the "Cedar
Policy Denied" and "N links intercepted" badges — the two security beats — run
the web app (`make demo`, or `make demo-fake` with no credentials).

---

## Running the demo

```bash
AWS_PROFILE=your-profile make demo
```

Opens `http://localhost:8000` automatically. Click the Q1–Q5 chips in order, or
type any PCSK9 question in the input box.

(`make start` ends by doing exactly this, so after a first-time setup you are
already here.)

### Rehearse without AWS (free, no credentials needed)

```bash
make demo-fake
```

Uses canned responses. The guardrail badge, Cedar denial badge, cost meter,
and receipt all work -- nothing calls AWS.

To also rehearse the ingestion progress screen:

```bash
make demo-fake-ingest
```

---

## Teardown

**Check first, then delete:**

```bash
AWS_PROFILE=your-profile make teardown-dry-run   # lists what WOULD be deleted
AWS_PROFILE=your-profile make teardown           # actually deletes it
```

Deletes the Knowledge Base and its data source, the S3 Vectors bucket and index,
the Guardrail, the AgentCore Gateway and its targets, the Cedar PolicyEngine and
its policies, both IAM roles, and the S3 corpus bucket. It also removes the local
`corpus/` directory — re-run `make corpus` to rebuild it.

**It finds resources it did not create.** Everything `build_kb.py` makes is
tagged `Project=inside-the-lines`, and teardown sweeps by that tag *and* by
normalised name prefix, not just by the IDs currently in your `config.py`. That
matters: this repo previously leaked a whole generation of resources every time a
name in `config.py` changed between a build and a teardown — three S3 Vector
buckets, a Knowledge Base stuck in `DELETE_UNSUCCESSFUL`, two orphaned Cedar
policy engines, an orphaned Guardrail and two orphaned IAM roles. `--dry-run`
prints why each item was claimed (`config`, `tag`, or `name`).

That backstop is why the corpus bucket name is derived rather than random:
`inside-the-lines-pcsk9-<account-id>` normalises to a name starting
`insidethelines`, so teardown claims it by **name** even if the tag is missing
and `config.py` has since been edited.

**It tells you if it failed.** After deleting, teardown re-runs discovery and
**exits non-zero** if anything survives, so "clean" is checkable rather than
assumed. Three resource types (KB data sources, gateway targets and Cedar
policies) cannot be tagged — AWS gives them no ARN — so they are reached through
their parents.

Idempotent: every delete is best-effort, so re-running after a partial teardown
just prints `skip` for what is already gone.

---

## Development

```bash
make install  # do this once: uv pip install -e ".[dev]"
make lint     # ruff check + format check
make test     # pytest (no AWS calls)
make fix      # auto-fix lint and format
```

`make lint` and `make test` work from a cold checkout — they pass `--extra dev`
explicitly, so no `make install` is needed first. `make install` is still there if
you want the editable install.

No test touches AWS or the network: `agent.py` takes its backend by dependency
injection and the suite passes a fake, and the setup logic is tested against
temporary files in `tmp_path`.

All the AWS details, quirks, and cost notes are in the source files:

| file | what to read it for |
|---|---|
| `start.py` | the `make start` chain, and what makes it resumable |
| `bootstrap.py` | config generation and rewriting, bucket derivation/creation, corpus upload |
| `preflight.py` | the read-only checks and the wording of each failure |
| `build_kb.py` | the provisioning steps |
| `src/agentcore_demo/agent.py` | the orchestration logic |
| `src/agentcore_demo/aws.py` | the AWS API calls |
| `teardown.py` | how the sweep discovers resources it did not create |

---

## License

MIT — see [LICENSE](LICENSE).
