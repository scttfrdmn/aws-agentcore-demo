# Inside the Lines — PCSK9

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![AWS Bedrock](https://img.shields.io/badge/AWS-Bedrock-orange.svg)](https://aws.amazon.com/bedrock/)

A five-minute live demo for a research-computing audience showing frontier
AI agents running entirely within a secure AWS boundary, billed only for
what they use.

The agent answers four questions about the gene *PCSK9* against a Bedrock
Knowledge Base of 1,000 open-access PMC papers (~48 MB), with a live cost
meter and a receipt at the end.

| Beat | Question | Models | Security feature |
|------|----------|--------|-----------------|
| Q1 | Role of PCSK9 in LDL regulation | Claude Haiku | **Bedrock Guardrail** intercepts NCBI URLs → redirects to local corpus |
| Q2 | Compare trial LDL-lowering + chart | Claude Sonnet + Code Interpreter | Isolated microVM for code execution |
| Q3 | Where does the literature disagree? | Claude Opus + OpenAI GPT-6 Astra (parallel) + Sonnet adjudicates | **Bedrock Guardrail** (same as Q1) |
| Q4 | Search ClinicalTrials.gov for ongoing trials | Claude Haiku | **AgentCore Gateway Cedar policy** blocks a real web tool |

Q4 is worth a note: the `web_fetch` tool the agent reaches for is not a prop. It is
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
- An S3 bucket you own (e.g. `my-inside-the-lines-corpus`)
- Python 3.11+ and [`uv`](https://github.com/astral-sh/uv)
- AWS CLI configured with permissions for `iam`, `s3`, `s3vectors`, `bedrock`,
  `bedrock-agent`, `bedrock-runtime`, `bedrock-agent-runtime`,
  `bedrock-agentcore-control` and `bedrock-agentcore` (Code Interpreter +
  Gateway). Read access to the AWS Price List API (`pricing`, us-east-1) is
  optional — without it the demo falls back to the rates in `config.py`.

---

## One-time setup

### 1. Clone and install

```bash
git clone https://github.com/scttfrdmn/aws-agentcore-demo ~/src/aws-agentcore-demo
cd ~/src/aws-agentcore-demo
uv venv && uv pip install -e ".[dev]"
```

### 2. Configure

```bash
cp config.example.py config.py
```

Open `config.py` and fill in your account ID and S3 bucket name:

```python
REGION = "us-west-2"
ACCOUNT_ID = "123456789012"  # your 12-digit account ID
BUCKET = "my-corpus-bucket"  # an S3 bucket you own

# IMPORTANT: Use US inference profile IDs (start with "us.")
MODELS = {
    "haiku": "us.anthropic.claude-haiku-4-5-20251001-v1:0",
    "sonnet": "us.anthropic.claude-sonnet-5",
    "opus": "us.anthropic.claude-opus-5",
    "openai": "us.openai.gpt-6-astra",
}
```

### 3. Fetch the paper corpus

```bash
make corpus
```

Downloads 1,000 CC0 / CC BY PCSK9 papers (~48 MB) from PubMed Central via the
NCBI E-utilities API, and stores them in `corpus/` (gitignored, local only).

Budget at least 10–15 minutes: reaching 1,000 keepers takes roughly 1,800
`efetch` calls, and NCBI rate-limits anonymous callers to 3 requests/sec.
Setting `NCBI_API_KEY` raises that to 10/sec — see the header of
`corpus_fetch.py` for how to get a key and for how the licence filter works.

### 4. Upload the corpus to S3

```bash
AWS_PROFILE=your-profile aws s3 sync corpus/ s3://YOUR-BUCKET/corpus/ --region us-west-2
```

### 5. Provision all AWS resources

```bash
AWS_PROFILE=your-profile make build-kb
```

This takes about 10 minutes. It creates two IAM service roles (one for the
Knowledge Base, one for the Gateway), the S3 Vectors store, the Bedrock
Knowledge Base and its S3 data source, runs the ingestion job, creates the
Guardrail, and sets up the AgentCore Gateway — the `web-tools` OpenAPI target
that publishes `web_fetch`, plus the Cedar policies that deny it.

Every step is get-or-create, so re-running is safe and cheap.

If the gateway target does not reach `READY`, the script **stops with the
target's own `statusReasons`** rather than continuing. That is deliberate: a
missing target is exactly what makes Q4 fail silently, so it has to fail loudly
here instead of on stage.

When it finishes, paste the printed IDs into `config.py`.

### 6. Verify with a headless run

```bash
AWS_PROFILE=your-profile make demo-headless
```

Runs all four questions and prints a receipt. Expected cost: **about $0.32** —
a full four-question run measured **$0.317** on 2026-09-22. Q3 dominates it,
because GPT-6 Astra is the priciest model in the set.

Under Q4, the line to look for is:

```
  ▸ web access denied by Cedar policy — answering from knowledge base
```

If instead you see `gateway call failed (NOT a policy denial): ...`, the Cedar
policy is not what stopped the call — read the reason. Most likely the
`web-tools` target is missing or `GATEWAY_URL` in `config.py` is stale. Do not
go on stage until Q4 reports the Cedar denial.

Note that the headless renderer prints only the `phase` line above; it does not
render the `policy_denied` or `guardrail` events themselves. To *see* the "Cedar
Policy Denied" and "N links intercepted" badges — the two security beats — run
the web app (`make demo`, or `make demo-fake` with no credentials).

---

## Running the demo

```bash
AWS_PROFILE=your-profile make demo
```

Opens `http://localhost:8000` automatically. Click the Q1, Q2, Q3, Q4 chips
in order, or type any PCSK9 question in the input box.

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

`make install` has to come first (step 1 of the setup above already does it).
The other targets shell out to bare `uv run`, which does not install the
package or the `dev` extra, so `make test` and `make lint` fail at collection
in an environment where `make install` has never run. If you would rather not
install anything, `uv run --extra dev pytest` and
`uv run --extra dev ruff check .` work standalone.

All the AWS details, quirks, and cost notes are in the source files.
Start with `src/agentcore_demo/agent.py` for the orchestration logic,
`src/agentcore_demo/aws.py` for the AWS API calls, and `build_kb.py`
for the provisioning steps.

---

## License

MIT — see [LICENSE](LICENSE).
