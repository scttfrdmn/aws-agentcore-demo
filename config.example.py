"""
config.example.py  --  copy to config.py and fill in.

    cp config.example.py config.py

config.py is git-ignored. Never commit real account IDs, bucket names,
or knowledge-base IDs.
"""

# --- account / region ---------------------------------------------------
REGION = "us-west-2"  # region where KB, Bedrock models, S3 Vectors,
# and AgentCore Code Interpreter all live
ACCOUNT_ID = "000000000000"
BUCKET = "your-corpus-bucket"  # S3 bucket holding the paper corpus
CORPUS_PREFIX = "corpus/"

# --- filled in by build_kb.py (paste its output back here) --------------
KB_ID = ""
DATA_SOURCE_ID = ""

# --- guardrail (filled in by build_kb.py) ----------------------------
GUARDRAIL_ID = ""
GUARDRAIL_VERSION = "DRAFT"

# --- resource names build_kb.py will CREATE -----------------------------
KB_NAME = "inside-the-lines-pcsk9"
KB_ROLE_NAME = "inside-the-lines-kb-role"
VECTOR_BUCKET_NAME = "inside-the-lines-vectors"  # S3 Vectors bucket
VECTOR_INDEX_NAME = "inside-the-lines-index"

# --- models -------------------------------------------------------------
# IMPORTANT: Use US cross-region inference profile IDs (start with "us.")
# Foundation model IDs (without "us." prefix) will fail with ValidationException.
# Verify available profiles: aws bedrock list-inference-profiles --region us-west-2
# "us." is US Geo cross-Region inference -- requests stay in US Regions.
MODELS = {
    "haiku": "us.anthropic.claude-haiku-4-5-20251001-v1:0",
    "sonnet": "us.anthropic.claude-sonnet-5",
    "opus": "us.anthropic.claude-opus-5",
    # The non-Anthropic cross-check for Q3.  OpenAI models arrived on Bedrock
    # after this demo was first written; GPT-6 Astra is OpenAI's most capable
    # model (launched 2026-09-08).  Same "us." convention as above, but note
    # two OpenAI-specific facts (verified 2026-09-22):
    #   - Unlike the Claude models, there is NO in-Region option on
    #     bedrock-runtime -- a cross-Region profile is mandatory, not a choice.
    #   - Geo and Global are priced DIFFERENTLY for OpenAI ($11/$55 vs
    #     $10/$50 per 1M).  They are not interchangeable in PRICING below.
    "openai": "us.openai.gpt-6-astra",
}
EMBED_MODEL_ID = "amazon.titan-embed-text-v2:0"
EMBED_DIM = 1024  # must match the vector index

# --- pricing ------------------------------------------------------------
# USD per 1,000,000 tokens (input, output).
# Source: AmazonBedrockFoundationModels Price List API (service code used
# by pricing.get_products); OpenAI rates come from the Bedrock model card.
# https://aws.amazon.com/bedrock/pricing/
# NOTE: these are Bedrock prices, NOT Anthropic/OpenAI first-party API prices
# (which differ).  Re-verify before the talk; rates move.
#
# These must match the inference profiles in MODELS above -- all "us." (Geo).
# For OpenAI that distinction is load-bearing: Geo is $11/$55 and Global is
# $10/$50, so copying the wrong row understates the receipt by ~9%.
PRICING = {
    "haiku": (1.00, 5.00),  # Haiku 4.5  — us. cross-region inference
    "sonnet": (2.00, 10.00),  # Sonnet 5   — us. cross-region inference
    "opus": (5.00, 25.00),  # Opus 5     — us. cross-region inference
    # GPT-6 Astra, Standard tier, Geo CRIS, short context (<=272K input).
    # Verified against the model card 2026-09-22.  Two caveats:
    #   - Long context (>272K input tokens) is billed at a HIGHER rate
    #     ($22/$82.50).  This demo retrieves ~16 passages, so it is always in
    #     the short-context band -- but do not reuse this rate for big inputs.
    #   - The in-Region/Geo price includes a 10% Bedrock fee over OpenAI's
    #     own rates; no need to add it yourself.
    "openai": (11.00, 55.00),  # GPT-6 Astra — us. Geo CRIS, short context
}
# AgentCore Code Interpreter: $0.0895/vCPU-hr + $0.00945/GB-hr (us-west-2).
# Assuming 1 vCPU + 2 GB RAM; actual allocation not exposed via API.
# ($0.0895 + 2×$0.00945) / 3600 ≈ $0.0000301/sec  — verify current AgentCore pricing.
CODE_INTERPRETER_PER_SECOND = 0.0000301

# --- KB rate fallbacks (used ONLY if the Price List API is unavailable) --
# The app fetches current rates from the AWS Price List at startup.
# These values are used as a fallback when that call fails.
# Verify them against the live pricing page and keep them roughly current.
S3V_STORAGE_USD_PER_GB_MONTH = 0.05  # fallback -- S3 Vectors storage per GB/month
S3V_QUERY_USD_PER_1K = 0.40  # fallback -- KB retrieval per 1,000 queries
EMBED_USD_PER_1M_TOKENS = 0.02  # fallback -- Titan Embed v2 per 1M tokens

# Alias used by CostMeter (keeps the config key consistent with pricing module):
KB_QUERY_USD_PER_1K = S3V_QUERY_USD_PER_1K

# --- gateway (filled in by build_kb.py) ----------------------------
GATEWAY_NAME = "inside-the-lines-gateway"
GATEWAY_ID = ""
GATEWAY_URL = ""
GATEWAY_ENGINE_ID = ""

# --- web app ------------------------------------------------------------
HOST = "127.0.0.1"
PORT = 8000
