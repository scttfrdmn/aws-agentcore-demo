#!/usr/bin/env python3
"""
corpus_fetch.py  --  build the paper corpus for the demo knowledge base.

What this does:
  Searches PubMed Central (PMC) for open-access PCSK9 papers, downloads the
  full text of each one as XML via the NCBI E-utilities API, parses out the
  article body, and saves each as a plain-text .txt file in ./corpus/.

  Nothing to do afterwards.  `make build-kb` creates the S3 bucket and uploads
  ./corpus/ into it as its first step (see bootstrap.upload_corpus, and the
  comment above it for why the upload lives there and not here).  This script
  deliberately makes NO AWS calls at all, so the slow, free, network-bound step
  works with no credentials configured yet.

  Resumable.  This is the slowest step in the whole setup (10-15 minutes), so it
  is safe to interrupt: articles already in ./corpus/ are counted and skipped, so
  re-running picks up where it stopped, and re-running after a COMPLETE fetch
  does nothing at all and costs nothing.  That is what makes `make start` safe
  to re-run at any point.

Why E-utilities and not the bulk datasets (checked 2026-09-22):
  PMC rebuilt its distribution service during 2026 (announced 2026-02-12,
  completed 2026-08).  Two things that older write-ups of this demo relied on
  are gone:
    - The PMC OA Web Service is retired: "the PMC OA Web Service is no longer
      available" (https://pmc.ncbi.nlm.nih.gov/tools/oa-service/).
    - "All legacy PMC Article Dataset files on the FTP and Cloud Services were
      removed the week of August 24, 2026" -- so oa_package/, oa_file_list.csv
      and the oa_comm / oa_noncomm bundles no longer exist
      (https://pmc.ncbi.nlm.nih.gov/tools/ftp/).  All that is left on FTP is
      PMC-ids.csv.gz for ID cross-referencing.
  Bulk access now means the AWS Open Data bucket s3://pmc-oa-opendata
  (us-east-1, public, free: `aws s3 ls --no-sign-request s3://pmc-oa-opendata/`
  -- https://registry.opendata.aws/ncbi-pmc/).
  None of that affects this script.  It uses E-utilities (esearch + efetch),
  which PMC still lists as a sanctioned automated-retrieval route alongside the
  Cloud Service, OAI-PMH and the BioC API: those "are the only services that may
  be used for automated retrieval of PMC content"
  (https://pmc.ncbi.nlm.nih.gov/tools/textmining/).  A thousand articles is well
  inside per-article retrieval; reach for the S3 bucket if you ever want the
  whole subset.

Why we filter by licence:
  PMC groups Open Access Subset licences three ways
  (https://pmc.ncbi.nlm.nih.gov/tools/textmining/):
    - Commercial use allowed:   CC0, CC BY, CC BY-SA, CC BY-ND
    - Non-commercial use only:  CC BY-NC, CC BY-NC-SA, CC BY-NC-ND
    - Other:                    no machine-readable Creative Commons licence
  This script keeps only CC0 and CC BY -- deliberately NARROWER than PMC's
  commercial-use set.  CC BY-SA's share-alike and CC BY-ND's no-derivatives
  terms are not worth arguing about on a conference stage, and a ~1,000-paper
  corpus is plenty without them.  Non-commercial articles are excluded outright.

  KNOWN WEAKNESS in the classifier (fetch_article below, not yet fixed): the
  "cc by" substring test runs BEFORE the non-commercial test, and "cc by-nc" and
  "cc by-sa" both contain "cc by".  An article whose <permissions> text spells
  its licence as "CC BY-NC 4.0" rather than as "Attribution-NonCommercial" is
  therefore labelled CC BY and kept.  Most publishers use the long spelling, so
  this is a narrow case, but it is a licence mislabel and the order of those
  branches should be inverted.

  The licence is read from each article's own <permissions> block rather than
  trusted from a directory split or a search filter, because PMC warns that
  "License terms vary" per article and that "not all articles in the Open Access
  Subset have a CC license".  PMC does document up-front filters --
  '"cc0 license"[filter]' and '"cc by license"[filter]' alongside the
  '"open access"[filter]' used in search_pmc() below -- and adding them would cut
  the number of articles fetched and discarded.  They are publisher-supplied
  metadata, though, so the XML check here stays regardless; treat them as an
  optimisation, not a replacement.

Output:
  ./corpus/PMCxxxxxxx.txt   one plain-text article per file, TARGET files in all
  (1,000 -- about 48 MB, overwhelmingly CC BY with a handful of CC0).
  Each file starts with a comment header:
      # PMCxxxxxxx  licence: CC0

  The corpus/ directory is gitignored.  Files already present are KEPT and
  counted (see is_complete / main below), so a re-run resumes rather than
  refetching.  To force a genuinely fresh corpus, delete ./corpus/ first --
  `make teardown` already does that for you.

What to do if the fetch fails:
  - Rate limit (429 or HTTP errors): add NCBI_API_KEY (see below) to go faster,
    or reduce TARGET, or just wait a few minutes.
  - ParseError on some articles: normal -- PMC XML is inconsistent.  The script
    skips unparseable articles and continues.
  - "connection refused" / timeout: check your internet connection; NCBI is
    occasionally slow under load.  The script will resume from where it left
    off (by design: it fetches IDs first, then articles one by one).

NCBI API key (optional, recommended):
  Without an API key, NCBI limits you to 3 requests/second.
  With an API key (free): 10 requests/second -- cuts fetch time by ~3×.
  Get one at: https://www.ncbi.nlm.nih.gov/account/
  Then set it before running:
      export NCBI_API_KEY=your-key-here

Requires: requests (installed via uv pip install -e ".[dev]")
"""

import os
import time
import xml.etree.ElementTree as ET

import requests

# The gene we're searching for.  All four demo questions are about PCSK9.
GENE = "PCSK9"

# Co-terms to narrow the search to PCSK9's most relevant literature.
# Without this, "PCSK9" alone returns too many tangential papers.
EXTRA = "(LDL OR cholesterol OR cardiovascular)"

# How many articles to keep in the final corpus.
# We fetch more candidates than this (see over-fetch below) because many
# articles will be filtered out (non-commercial licence, no body text, too short).
#
# Budget the time: at ~55% yield this means roughly 1,800 efetch calls, and NCBI
# allows 3 requests/sec without an API key (10 with one), so a full run is tens
# of minutes rather than a couple.  aws.py hard-codes the same 1000 as
# _EXPECTED_CORPUS_SIZE for the ingestion progress bar -- change both together.
TARGET = 1000

# Where to write the .txt files.
OUTDIR = "corpus"

# NCBI API key -- optional, but strongly recommended for faster fetching.
API_KEY = os.environ.get("NCBI_API_KEY", "")

# Base URL for NCBI E-utilities (the API that powers PubMed and PMC).
EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

# The licences this demo will show on stage.  Narrower than PMC's own
# commercial-use set (which also includes CC BY-SA and CC BY-ND) on purpose --
# see "Why we filter by licence" in the module docstring.
COMMERCIAL_LICENCES = {"CC0", "CC BY"}

# Delay between requests -- NCBI rate limit is 3/sec without a key, 10/sec with.
# We leave a small margin (0.12 instead of 0.1) to avoid 429 errors.
DELAY = 0.12 if API_KEY else 0.34


def _key():
    """Return API key parameter dict, or empty dict if no key is set."""
    return {"api_key": API_KEY} if API_KEY else {}


def search_pmc(retmax):
    """Search PMC for open-access PCSK9 articles and return a list of PMC IDs.

    Uses the NCBI ESearch API to retrieve PMC IDs (integers) matching the
    search term.  Paginates in batches of 200 until retmax IDs are collected.

    '"open access"[filter]' is PMC's documented filter for the Open Access
    Subset (PMC user guide, "By Collection").  It does NOT narrow by licence --
    the subset spans CC0 through CC BY-NC-ND plus articles with no
    machine-readable licence at all -- which is why fetch_article() re-checks
    every article's <permissions> block.

    Args:
        retmax: maximum number of IDs to return.

    Returns:
        A list of PMC ID strings (e.g. ["13156736", "5481105", ...]).
    """
    term = f'{GENE} AND {EXTRA} AND "open access"[filter]'
    ids, retstart = [], 0
    while len(ids) < retmax:
        r = requests.get(
            f"{EUTILS}/esearch.fcgi",
            params={
                "db": "pmc",
                "term": term,
                "retmax": 200,  # fetch 200 IDs per page
                "retstart": retstart,  # offset for pagination
                "retmode": "json",
                **_key(),
            },
        )
        r.raise_for_status()
        batch = r.json()["esearchresult"]["idlist"]
        if not batch:
            break  # no more results
        ids += batch
        retstart += len(batch)
        time.sleep(DELAY)
    return ids[:retmax]


def classify_licence(blob: str) -> str:
    """Classify a PMC <permissions>/<license> blob into a licence name.

    Extracted from fetch_article() on 2026-09-22 so it can be unit-tested --
    see tests/test_corpus_licence.py.  This is the single gate that keeps
    non-commercial papers out of a corpus shown in a public commercial talk,
    so it is worth testing rather than trusting.

    ORDER IS LOAD-BEARING: the restrictive licences are tested FIRST.
    This code used to test "cc by" before the non-commercial branch -- and the
    substring "cc by" is contained in "cc by-nc".  An article whose licence text
    spelled itself "CC BY-NC 4.0" (rather than the long
    "Attribution-NonCommercial" form) was therefore classified CC BY and KEPT:
    exactly the failure this filter exists to prevent.  The href test alone was
    safe ("/licenses/by/" has a trailing slash, so it cannot match
    "/licenses/by-nc/"); the free-text fallback was not.  Deny-first ordering
    makes the substring overlap harmless -- the same principle as Cedar's
    forbid-overrides-permit.

    Note CC BY-SA and CC BY-ND are in PMC's commercial-use group but are
    deliberately NOT in COMMERCIAL_LICENCES; naming them here means they are
    skipped explicitly rather than falling through to "CC BY".

    Args:
        blob: the licence href plus its element text, already lowercased.

    Returns:
        One of "CC0", "CC BY", "CC BY-NC", "CC BY-ND", "CC BY-SA", "UNKNOWN".
    """
    if "by-nc" in blob or "noncommercial" in blob:
        return "CC BY-NC"
    if "by-nd" in blob or "noderivs" in blob:
        return "CC BY-ND"
    if "by-sa" in blob or "sharealike" in blob:
        return "CC BY-SA"
    if "publicdomain" in blob or "cc0" in blob:
        return "CC0"
    if "creativecommons.org/licenses/by/" in blob or "cc by" in blob:
        return "CC BY"
    return "UNKNOWN"


def fetch_article(pmcid):
    """Fetch one PMC article, check its licence, and return its plain text.

    Downloads the article XML via the NCBI EFetch API, parses the licence
    from the <permissions> element, and extracts the body text.

    Articles are skipped (returning None for text) if:
      - The HTTP response is not 200 or the body is empty.
      - The XML cannot be parsed.
      - The licence is not CC0 or CC BY.
      - There is no <body> element (some PMC records have abstract-only entries).
      - The extracted text is shorter than 1,500 characters (too thin to be useful).

    Args:
        pmcid: the numeric PMC ID string (without the "PMC" prefix).

    Returns:
        (licence, text) where text is the plain string body, or (licence, None)
        if the article should be skipped.
    """
    r = requests.get(
        f"{EUTILS}/efetch.fcgi",
        params={"db": "pmc", "id": pmcid, "retmode": "xml", **_key()},
    )
    if r.status_code != 200 or not r.content:
        return None, None

    try:
        root = ET.fromstring(r.content)
    except ET.ParseError:
        # PMC XML is occasionally malformed.  Skip and move on.
        return None, None

    # Extract and classify the licence from the <permissions> section.
    # PMC uses href attributes and plain text in varying combinations,
    # so we concatenate both and search for known patterns.
    licence = "UNKNOWN"
    lic_el = root.find(".//permissions/license")
    if lic_el is not None:
        blob = (
            lic_el.get("{http://www.w3.org/1999/xlink}href", "") + " " + "".join(lic_el.itertext())
        ).lower()

        licence = classify_licence(blob)

    # Filter out anything that is not explicitly commercial-use.
    if licence not in COMMERCIAL_LICENCES:
        return licence, None

    # Extract the article body text.
    body = root.find(".//body")
    if body is None:
        # Abstract-only article -- not enough content for a knowledge base.
        return licence, None

    # Flatten all text nodes, strip whitespace, join with spaces.
    text = " ".join(t.strip() for t in body.itertext() if t.strip())

    # Reject articles under 1,500 characters -- they are too short to be useful
    # as knowledge base chunks and would waste ingestion budget.
    return (licence, text) if len(text) > 1500 else (licence, None)


def existing_ids(outdir=OUTDIR) -> set[str]:
    """Return the PMC IDs already downloaded into `outdir` (no "PMC" prefix).

    The basis of both resumption and the "already done, skip it" shortcut.
    Filenames are the source of truth rather than a state file: a state file can
    disagree with the directory, a directory cannot disagree with itself.
    """
    if not os.path.isdir(outdir):
        return set()
    return {
        f[3:-4]
        for f in os.listdir(outdir)
        if f.startswith("PMC") and f.endswith(".txt") and os.path.getsize(f"{outdir}/{f}") > 0
    }


def is_complete(outdir=OUTDIR, target=TARGET) -> bool:
    """True if `outdir` already holds a full corpus, so fetching can be skipped.

    Called by start.py so the single `make start` entry point can re-run end to
    end without spending 10-15 minutes re-downloading a corpus that is already
    on disk.
    """
    return len(existing_ids(outdir)) >= target


def main():
    """Main loop: search, filter, and save the corpus.  Resumable."""
    os.makedirs(OUTDIR, exist_ok=True)

    # Resume: whatever is already on disk counts toward TARGET.  A finished
    # corpus makes this a no-op, which is what lets `make start` be re-run.
    # OUTDIR is passed explicitly rather than relying on the default argument:
    # a default is bound once, at def time, so `existing_ids()` would silently
    # ignore a caller (or a test) that had changed OUTDIR.
    have = existing_ids(OUTDIR)
    if len(have) >= TARGET:
        print(f"Corpus already complete: {len(have)} articles in ./{OUTDIR}/ -- nothing to do.")
        print(f"(Delete ./{OUTDIR}/ if you want a genuinely fresh corpus.)")
        return
    if have:
        print(f"Resuming: {len(have)} of {TARGET} articles already in ./{OUTDIR}/")

    print(f"Searching PMC for open-access {GENE} articles...")

    # Over-fetch by 1.8× because many articles will be filtered out.
    # The typical yield after licence and length filtering is about 55%.
    ids = search_pmc(int(TARGET * 1.8))
    print(f"  {len(ids)} candidate PMCIDs")

    kept, skipped = len(have), 0
    for pmcid in ids:
        if kept >= TARGET:
            break
        if pmcid in have:
            continue  # already downloaded on an earlier run -- do not refetch

        licence, text = fetch_article(pmcid)
        time.sleep(DELAY)  # respect NCBI rate limits

        if text is None:
            skipped += 1
            continue

        # Write the article as plain text with a header comment.
        with open(f"{OUTDIR}/PMC{pmcid}.txt", "w", encoding="utf-8") as f:
            f.write(f"# PMC{pmcid}  licence: {licence}\n\n{text}")
        kept += 1

        if kept % 50 == 0:
            print(f"  kept {kept}  (skipped {skipped} non-commercial / empty)")

    print(f"\nDone. {kept} commercial-use articles in ./{OUTDIR}/")
    # No manual upload step any more.  `make build-kb` creates the bucket and
    # uploads this directory itself.
    print("Next:  make build-kb   (creates the S3 bucket, uploads these, provisions the KB)")


if __name__ == "__main__":
    main()
