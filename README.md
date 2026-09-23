# Inside the Lines — PCSK9

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![AWS Bedrock](https://img.shields.io/badge/AWS-Bedrock-orange.svg)](https://aws.amazon.com/bedrock/)

A five-minute live demo for a research-computing audience showing frontier
AI agents running entirely within a secure AWS boundary, billed only for
what they use.

The agent answers four questions about the gene *PCSK9* against a Bedrock
Knowledge Base of ~650 open-access PMC papers, with a live cost meter and
a receipt at the end.

| Beat | Question | Models | Security feature |
|------|----------|--------|-----------------|
| Q1 | Role of PCSK9 in LDL regulation | Claude Haiku | **Bedrock Guardrail** intercepts NCBI URLs → redirects to local corpus |
| Q2 | Compare trial LDL-lowering + chart | Claude Sonnet + Code Interpreter | Isolated microVM for code execution |
| Q3 | Where does the literature disagree? | Claude Opus + OpenAI GPT-6 Astra (parallel) + Sonnet adjudicates | **Bedrock Guardrail** (same as Q1) |
| Q4 | Search ClinicalTrials.gov for ongoing trials | Claude Haiku | **AgentCore Gateway Cedar policy** blocks web access |

---

## Prerequisites

- AWS account with Bedrock model access enabled in **us-west-2**
  (check: `aws bedrock list-inference-profiles --region us-west-2`)
- Four models must be ACTIVE: Haiku 4.5, Sonnet 5, Opus 5, OpenAI GPT-6 Astra
  (OpenAI models have no in-Region option on `bedrock-runtime` — the `us.`
  cross-Region profile below is required, not optional)
- An S3 bucket you own (e.g. `my-inside-the-lines-corpus`)
- Python 3.11+ and [`uv`](https://github.com/astral-sh/uv)
- AWS CLI configured with Bedrock, S3, IAM, and bedrock-agentcore-control permissions

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
REGION     = "us-west-2"
ACCOUNT_ID = "123456789012"       # your 12-digit account ID
BUCKET     = "my-corpus-bucket"  # an S3 bucket you own

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

Downloads ~650 CC0 / CC BY PCSK9 papers from PubMed Central.
Takes about 5 minutes. Stores them in `corpus/` (gitignored, local only).

### 4. Upload the corpus to S3

```bash
AWS_PROFILE=your-profile aws s3 sync corpus/ s3://YOUR-BUCKET/corpus/ --region us-west-2
```

### 5. Provision all AWS resources

```bash
AWS_PROFILE=your-profile make build-kb
```

This takes about 10 minutes. It creates the IAM role, S3 Vectors store,
Bedrock Knowledge Base, runs the ingestion job, creates the Guardrail,
and sets up the AgentCore Gateway with Cedar policies.

When it finishes, paste the printed IDs into `config.py`.

### 6. Verify with a headless run

```bash
AWS_PROFILE=your-profile make demo-headless
```

Runs all three main questions and prints a receipt. Expected cost: **$0.20–0.40**.
Q4 should print `Cedar policy denied: web_fetch is not permitted`.

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

```bash
AWS_PROFILE=your-profile make teardown
```

Deletes the Knowledge Base, S3 Vectors store, Guardrail, AgentCore Gateway,
Cedar PolicyEngine, Lambda function, all IAM roles, and the S3 corpus bucket.
The local `corpus/` directory is not deleted (it is on your machine, not in AWS).

---

## Development

```bash
make lint     # ruff check + format check
make test     # pytest (no AWS calls)
make fix      # auto-fix lint and format
```

All the AWS details, quirks, and cost notes are in the source files.
Start with `src/agentcore_demo/agent.py` for the orchestration logic,
`src/agentcore_demo/aws.py` for the AWS API calls, and `build_kb.py`
for the provisioning steps.

---

## License

MIT — see [LICENSE](LICENSE).
