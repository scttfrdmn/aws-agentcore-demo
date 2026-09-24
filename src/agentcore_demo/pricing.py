"""
pricing.py  --  AWS Price List lookup for infrastructure rates.

This module fetches current pricing for two services that the cost meter
needs but whose rates are NOT in config.PRICING:

  1. Titan Embed V2 embedding rate  (used for ingestion cost estimate)
  2. S3 standard storage rate        (used for corpus storage cost estimate)

It is called once at startup by app._build_backend() and caches the result
for the process lifetime.  The fake-backend path (DEMO_FAKE=1) never calls
this module.

Why not just use config.PRICING for everything?
  config.PRICING holds the Bedrock model inference prices (Haiku, Sonnet,
  Opus, GPT-6 Astra).  The embedding and S3 rates are different services and
  change on a different schedule.  Fetching them live at startup means the
  demo always shows current prices without needing a config update.

Fallback chain:
  1. Live AWS Price List API (boto3 pricing client, us-east-1)
  2. config.py fallback values (S3V_STORAGE_USD_PER_GB_MONTH, etc.)
  3. Hard-coded defaults in _HARD_DEFAULTS (last resort)

Verified AWS Price List quirks (do not "fix" without re-checking):

  AmazonS3Vectors service code (2026-05-20):
    The AmazonS3Vectors service EXISTS in the Price List API but returns
    zero items -- it is too new to have pricing data populated.
    S3 Vectors storage and query rates must come from config.py fallbacks.
    This may change as the service matures; try the live fetch periodically.
    STILL UNRESOLVED as of 2026-09-22: worth one manual get_products call
    against BOTH "AmazonS3Vectors" and "AmazonS3" -- S3 Vectors rates are
    published on the S3 pricing page, so the offer may live under AmazonS3.

  S3 Vectors rate corrections (2026-09-22):
    The fallbacks below were wrong, one of them badly.  Re-read from
    https://aws.amazon.com/s3/pricing/ (us-east-1 figures in the worked
    examples; the rate table itself does not render):
      - storage  $0.05 -> $0.06 per GB-month
      - query    $0.40 -> $0.0025 per 1,000 queries
    The query rate was off by ~160x.  The real fee is $2.50 per MILLION
    QueryVectors requests; the old value read as $0.40 per THOUSAND, which
    made every retrieval look ~160x more expensive than it is.  This is why
    the receipt showed a suspiciously round $0.000400 per retrieval.

    Query cost is actually three terms, not one:
      1. request fee        $2.50 / 1M QueryVectors  <- what we meter
      2. data processed     tiered by index size: $0.004/TB up to 100K
                            vectors, $0.002/TB to 10M, $0.0004/TB beyond
                            (the 10M+ tier was cut ~80% on 2026-06-16)
      3. data returned      $0.01/GB, first 512KB per query free,
                            each result billed at a 256-byte floor
    We deliberately meter only term 1.  For this demo's ~5.5K-vector index,
    term 2 is ~$1e-7 per query and term 3 falls entirely inside the free
    allowance -- so the request fee dominates and the other two would add
    noise digits without changing the number the audience sees.

  Claude 4.x/5 pricing location (2026-05-20, re-checked 2026-09-22):
    Current Claude model prices (Haiku 4.5, Sonnet 5, Opus 5) are NOT in
    the "AmazonBedrock" Price List service code.  That service only covers
    Claude 2.x and 3.x.  Everything from Claude 4.x onward is in the separate
    "AmazonBedrockFoundationModels" service code.
    The PRICING dict in config.py is sourced from that service.
    See: https://aws.amazon.com/bedrock/pricing/
    NOTE: these are Bedrock on-demand prices, not Anthropic API prices.
    The two differ -- Bedrock is generally higher due to the managed service layer.

  Titan Embed V2 pricing (2026-05-20):
    Titan Embed V2 pricing IS in the "AmazonBedrock" Price List under the
    usagetype prefix "USW2-TitanEmbeddingV2-Text-input-tokens".
    The API returns USD per 1K tokens; we multiply by 1,000 to get per 1M.
    Live price: $0.02 / 1M tokens (on-demand, us-west-2) -- matches config.py.

  OpenAI GPT-6 Astra pricing (2026-09-22):
    Astra replaced Amazon Nova Pro as the Q4 cross-check.  Its rates are taken
    from the Bedrock model card, NOT the Price List API:
      $11.00 / $55.00 per 1M tokens (in/out), Standard tier, Geo CRIS,
      short context (<= 272K input tokens).
    Three things to know before touching these numbers:
      - Geo ("us." profile) and Global ("global." profile) are priced
        differently ($11/$55 vs $10/$50).  config.MODELS uses "us.", so the
        Geo row is the correct one.
      - Inputs over 272K tokens bill at a higher long-context rate
        ($22.00/$82.50).  This demo retrieves ~16 passages and never gets
        close, but a cost model that assumed one flat rate would be wrong
        for any larger workload.
      - The quoted price already includes Bedrock's 10% fee over OpenAI's
        own rates; do not add it again.
    Astra is the most expensive model in the demo -- Q4 dominates the receipt.

  Region prefix convention:
    The Price List API uses regional usagetype prefixes that do NOT always match
    the AWS region name.  For example, us-west-2 uses "USW2", not "us-west-2".
    The _region_prefix() function maps region names to their Price List prefixes.
    If your region is not in the map, it falls back to "USE1" (us-east-1).
"""

from __future__ import annotations

import json
import logging

log = logging.getLogger(__name__)

# Hard-coded floor values used only if BOTH the Price List API and config.py
# are unavailable.  Re-verified against the AWS S3 pricing page on 2026-09-22.
_HARD_DEFAULTS: dict[str, float] = {
    "s3v_storage_usd_per_gb_month": 0.06,  # S3 Vectors storage, per GB/month
    "s3v_query_usd_per_1k": 0.0025,  # KB retrieval, per 1,000 queries ($2.50/M)
    "embed_usd_per_1m_tokens": 0.02,  # Titan Embed V2, per 1M input tokens
    "s3_standard_usd_per_gb_month": 0.023,  # S3 standard storage, first 50 TB, us-west-2
}

# Module-level cache: populated on the first fetch_rates() call.
_CACHE: dict[str, float] | None = None


def fetch_rates(region: str) -> dict[str, float]:
    """Return current infrastructure rates, fetched once and cached.

    The returned dict has keys:
      s3v_storage_usd_per_gb_month  -- S3 Vectors storage per GB per month
      s3v_query_usd_per_1k          -- KB retrieval per 1,000 queries
      embed_usd_per_1m_tokens       -- Titan Embed V2 embedding per 1M input tokens
      s3_standard_usd_per_gb_month  -- S3 standard storage per GB per month

    Strategy:
      1. Start from config.py fallbacks.
      2. Overlay with live Titan Embed V2 rate from AmazonBedrock Price List (works).
      3. Overlay with live S3 standard rate from AmazonS3 Price List (works).
      4. S3 Vectors rates remain at config.py values (Price List returns no data).

    Args:
        region: AWS region string (e.g. "us-west-2").

    Returns:
        Dict of rate floats, fully populated (never missing keys).
    """
    global _CACHE
    if _CACHE is not None:
        return _CACHE  # fast path: already fetched this process

    # Start with config.py values, then overlay live prices on top.
    base = _from_config()

    live_embed = _fetch_bedrock_embed_rate(region)
    if live_embed:
        base.update(live_embed)
        log.info("Bedrock embed rate fetched from Price List: %s", live_embed)

    live_s3 = _fetch_s3_standard_rate(region)
    if live_s3:
        base.update(live_s3)
        log.info("S3 standard rate fetched from Price List: %s", live_s3)

    _CACHE = base
    return _CACHE


def reset_cache() -> None:
    """Clear the module-level cache.

    Used only in tests.  Production code should never need to call this --
    pricing rates are stable for the lifetime of a single demo run.
    """
    global _CACHE
    _CACHE = None


# ── Price List: Titan Embed V2 ────────────────────────────────────────────────


def _region_prefix(region: str) -> str:
    """Map an AWS region name to its Price List usagetype prefix.

    The AWS Price List API uses region-specific prefixes in usagetype strings
    that do NOT match the region names directly.  For example:
      us-east-1 -> "USE1"
      us-west-2 -> "USW2"
      eu-west-1 -> "EU"   (note: no numeric suffix for the first EU region)

    This mapping was verified against the AmazonBedrock Price List on 2026-05-20.
    If your region is missing, add it or it will fall back to "USE1" pricing.
    """
    _MAP = {
        "us-east-1": "USE1",
        "us-east-2": "USE2",
        "us-west-1": "USW1",
        "us-west-2": "USW2",
        "eu-west-1": "EU",  # intentionally no number suffix
        "eu-west-2": "EUW2",
        "eu-west-3": "EUW3",
        "eu-central-1": "EUC1",
        "ap-southeast-1": "APS1",
        "ap-southeast-2": "APS2",
        "ap-northeast-1": "APN1",
    }
    return _MAP.get(region, "USE1")  # fall back to us-east-1 prefix for unknown regions


def _fetch_bedrock_embed_rate(region: str) -> dict[str, float] | None:
    """Fetch the Titan Embed V2 on-demand rate from the AmazonBedrock Price List.

    This function fetches the EMBEDDING rate only -- NOT model inference prices.
    Claude 4.x inference prices are in AmazonBedrockFoundationModels, not here.

    The Price List client must be in us-east-1 (the only region the Price List
    API is available in), regardless of which region you're deploying to.

    The usagetype string for us-west-2 is: "USW2-TitanEmbeddingV2-Text-input-tokens"
    The price is in USD per 1K tokens; we convert to per 1M for CostMeter.

    Returns:
        {"embed_usd_per_1m_tokens": float} or None if the fetch fails.
    """
    try:
        import boto3

        # Price List API is only available in us-east-1 -- use that region
        # even when deploying everything else to us-west-2.
        client = boto3.client("pricing", region_name="us-east-1")
        prefix = _region_prefix(region)
        usage_type = f"{prefix}-TitanEmbeddingV2-Text-input-tokens"

        resp = client.get_products(
            ServiceCode="AmazonBedrock",
            FormatVersion="aws_v1",
            Filters=[{"Type": "TERM_MATCH", "Field": "usagetype", "Value": usage_type}],
        )
        items = resp.get("PriceList", [])
        if not items:
            return None  # no pricing data for this region/model combination

        # Each item is a JSON string; parse it and navigate to the price.
        item = json.loads(items[0])
        for term in item.get("terms", {}).get("OnDemand", {}).values():
            for dim in term.get("priceDimensions", {}).values():
                usd_per_1k_str = dim.get("pricePerUnit", {}).get("USD", "")
                usd_per_1k = float(usd_per_1k_str)
                if usd_per_1k > 0:
                    # Price List gives USD per 1K tokens.
                    # CostMeter uses USD per 1M tokens -- multiply by 1,000.
                    return {"embed_usd_per_1m_tokens": round(usd_per_1k * 1000, 6)}
    except Exception as exc:
        log.info("Bedrock Price List unavailable (%s) — using config fallback", exc)
    return None


def _fetch_s3_standard_rate(region: str) -> dict[str, float] | None:
    """Fetch the S3 standard storage rate (first 50 TB tier) from the AmazonS3 Price List.

    Filters for:
      - regionCode matching the target region
      - storageClass "General Purpose" (standard storage)
      - volumeType "Standard"
      - description containing "first 50 TB" (the pricing tier for small buckets)

    Returns:
        {"s3_standard_usd_per_gb_month": float} or None if the fetch fails.
    """
    try:
        import boto3

        client = boto3.client("pricing", region_name="us-east-1")
        resp = client.get_products(
            ServiceCode="AmazonS3",
            FormatVersion="aws_v1",
            Filters=[
                {"Type": "TERM_MATCH", "Field": "regionCode", "Value": region},
                {"Type": "TERM_MATCH", "Field": "storageClass", "Value": "General Purpose"},
                {"Type": "TERM_MATCH", "Field": "volumeType", "Value": "Standard"},
            ],
        )
        import json  # noqa: PLC0415 -- already imported above, but ruff is happy

        for item_str in resp.get("PriceList", []):
            item = json.loads(item_str)
            attrs = item.get("product", {}).get("attributes", {})
            # Filter to time-based storage usagetypes (not request or data transfer).
            if "TimedStorage" not in attrs.get("usagetype", ""):
                continue
            for term in item.get("terms", {}).get("OnDemand", {}).values():
                for dim in term.get("priceDimensions", {}).values():
                    desc = dim.get("description", "")
                    # "first 50 TB" is the cheapest tier; a 48 MB corpus easily fits.
                    if "first 50 TB" not in desc:
                        continue
                    usd = float(dim.get("pricePerUnit", {}).get("USD", "0") or "0")
                    if usd > 0:
                        return {"s3_standard_usd_per_gb_month": usd}
    except Exception as exc:
        log.info("S3 Price List unavailable (%s) — using config fallback", exc)
    return None


# ── config fallback ───────────────────────────────────────────────────────────


def _from_config() -> dict[str, float]:
    """Load fallback rates from config.py, filling gaps with _HARD_DEFAULTS.

    The config.py values are sourced from the AWS pricing page and were
    re-verified on 2026-09-22 (the S3 Vectors query rate was wrong by ~160x
    before that -- see the module docstring).  The hard defaults are a last
    resort for when
    config.py is missing entirely (e.g. when running tests without config.py).
    """
    try:
        import config  # type: ignore[import]

        return {
            "s3v_storage_usd_per_gb_month": getattr(
                config,
                "S3V_STORAGE_USD_PER_GB_MONTH",
                _HARD_DEFAULTS["s3v_storage_usd_per_gb_month"],
            ),
            "s3v_query_usd_per_1k": getattr(
                config,
                "S3V_QUERY_USD_PER_1K",
                _HARD_DEFAULTS["s3v_query_usd_per_1k"],
            ),
            "embed_usd_per_1m_tokens": getattr(
                config,
                "EMBED_USD_PER_1M_TOKENS",
                _HARD_DEFAULTS["embed_usd_per_1m_tokens"],
            ),
            "s3_standard_usd_per_gb_month": getattr(
                config,
                "S3_STANDARD_USD_PER_GB_MONTH",
                _HARD_DEFAULTS["s3_standard_usd_per_gb_month"],
            ),
        }
    except ImportError:
        log.info("config.py not found — using hard-coded default rates")
        return dict(_HARD_DEFAULTS)
