"""
Licence-classification tests for corpus_fetch.classify_licence().

Why this file exists: the licence gate is the single thing keeping
non-commercial papers out of a corpus that gets projected during a public,
commercial conference talk.  Everything else in this repo is recoverable; a
CC BY-NC paper on the screen is not.

The bug that prompted these tests (fixed 2026-09-22): the classifier tested
"cc by" BEFORE the non-commercial branch, and "cc by" is a substring of
"cc by-nc".  Any article whose <permissions> text used the short spelling
"CC BY-NC 4.0" was classified CC BY and kept.  The URL form was unaffected
("/licenses/by/" has a trailing slash), which is why it survived review --
the free-text fallback was the hole.

No network, no AWS: these run against the pure function.
"""

import pytest

import corpus_fetch
from corpus_fetch import COMMERCIAL_LICENCES, classify_licence

# (licence blob as it appears in PMC XML, expected classification).
# Blobs are lowercased by the caller before classify_licence() sees them.
CASES = [
    # --- genuinely commercial-usable -------------------------------------
    ("http://creativecommons.org/licenses/by/4.0/ attribution 4.0", "CC BY"),
    ("cc by 4.0", "CC BY"),
    ("http://creativecommons.org/publicdomain/zero/1.0/", "CC0"),
    ("cc0 1.0 universal", "CC0"),
    # --- the regression: short-spelled non-commercial --------------------
    ("cc by-nc 4.0", "CC BY-NC"),
    ("cc by-nc-nd 4.0", "CC BY-NC"),
    ("cc by-nc-sa 4.0", "CC BY-NC"),
    # --- non-commercial in its other spellings ---------------------------
    ("http://creativecommons.org/licenses/by-nc/4.0/", "CC BY-NC"),
    ("attribution-noncommercial 4.0 international", "CC BY-NC"),
    # --- PMC calls these commercial-usable; we skip them anyway ----------
    ("cc by-sa 4.0", "CC BY-SA"),
    ("cc by-nd 4.0", "CC BY-ND"),
    ("attribution-sharealike 4.0", "CC BY-SA"),
    # --- no machine-readable CC licence ----------------------------------
    ("all rights reserved", "UNKNOWN"),
    ("", "UNKNOWN"),
]


@pytest.mark.parametrize("blob,expected", CASES)
def test_classify_licence(blob, expected):
    assert classify_licence(blob) == expected, (
        f"{blob!r} classified as {classify_licence(blob)!r}, expected {expected!r}"
    )


@pytest.mark.parametrize(
    "blob",
    [
        "cc by-nc 4.0",
        "cc by-nc-nd 4.0",
        "cc by-nc-sa 4.0",
        "http://creativecommons.org/licenses/by-nc/4.0/",
        "attribution-noncommercial 4.0 international",
    ],
)
def test_non_commercial_never_passes_the_gate(blob):
    """The property that actually matters, stated directly.

    Asserting the classification is useful; asserting that the classification
    keeps the paper OUT is the thing we care about.  This would have failed
    before 2026-09-22 for the short-spelled variants.
    """
    assert classify_licence(blob) not in COMMERCIAL_LICENCES


def test_commercial_licences_is_narrower_than_pmc_allows():
    """We deliberately keep less than PMC permits.

    PMC's commercial-use group is CC0, CC BY, CC BY-SA and CC BY-ND.  This repo
    accepts only the first two -- a deliberate narrowing, so a future reader
    does not "fix" the set to match PMC and widen the corpus by surprise.
    """
    assert sorted(COMMERCIAL_LICENCES) == ["CC BY", "CC0"]


# ----------------------------------------------------------------------
# Resumption -- the other reason this step can waste a stranger's afternoon
# ----------------------------------------------------------------------
# Fetching the corpus is the slowest thing in the whole setup (10-15 minutes,
# ~1,800 rate-limited NCBI calls).  `make start` re-runs the entire chain, so it
# MUST be able to tell "already done" from "not started" without going to the
# network.  These tests pin that, because a regression here means every re-run
# spends a quarter of an hour redownloading files that are already on disk.
def test_existing_ids_reads_the_directory(tmp_path):
    (tmp_path / "PMC111.txt").write_text("body")
    (tmp_path / "PMC222.txt").write_text("body")
    assert corpus_fetch.existing_ids(tmp_path) == {"111", "222"}


def test_existing_ids_ignores_junk_and_empty_files(tmp_path):
    (tmp_path / "PMC111.txt").write_text("body")
    (tmp_path / "PMC999.txt").write_text("")  # a truncated/interrupted download
    (tmp_path / "notes.md").write_text("body")
    (tmp_path / "PMC333.json").write_text("body")
    assert corpus_fetch.existing_ids(tmp_path) == {"111"}


def test_existing_ids_on_a_missing_directory_is_empty(tmp_path):
    assert corpus_fetch.existing_ids(tmp_path / "nope") == set()


def test_is_complete_is_false_when_partial(tmp_path):
    (tmp_path / "PMC1.txt").write_text("body")
    assert corpus_fetch.is_complete(tmp_path, target=3) is False


def test_is_complete_is_true_at_target(tmp_path):
    for i in range(3):
        (tmp_path / f"PMC{i}.txt").write_text("body")
    assert corpus_fetch.is_complete(tmp_path, target=3) is True


def test_main_does_nothing_when_the_corpus_is_complete(tmp_path, monkeypatch, capsys):
    """A complete corpus must short-circuit BEFORE any network call.

    search_pmc is replaced with a function that fails the test if it is reached,
    which is the only way to prove "no network" rather than assume it.
    """
    monkeypatch.setattr(corpus_fetch, "TARGET", 3)
    for i in range(3):
        (tmp_path / f"PMC{i}.txt").write_text("body")
    monkeypatch.setattr(corpus_fetch, "OUTDIR", str(tmp_path))
    monkeypatch.setattr(
        corpus_fetch, "search_pmc", lambda *_a, **_k: pytest.fail("hit the network")
    )
    corpus_fetch.main()
    assert "already complete" in capsys.readouterr().out
