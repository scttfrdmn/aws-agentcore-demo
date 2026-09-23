"""
test_bootstrap.py  --  the zero-config path, tested without AWS.

Two things in bootstrap.py are pure logic and are worth real tests, because both
fail in ways that are painful rather than obvious:

  * THE CONFIG REWRITER.  It edits a git-ignored file a human may have
    hand-tuned, so a bug there destroys work that cannot be recovered from git.
    These tests pin: round-trip correctness, idempotency (byte-identical on a
    second write), comment/formatting preservation, refusal to touch anything
    that is not a plain string assignment, creation from the example file when
    config.py is absent, and the backup.
  * BUCKET NAME DERIVATION.  An illegal S3 bucket name fails as
    `InvalidBucketName` mid-provision, and a name that does not start with
    "inside-the-lines" silently breaks teardown.py's name backstop -- which is
    how this repo previously orphaned resources in people's accounts.

No credentials, no network, no AWS calls: every test writes into tmp_path.
Nothing here imports config.py, and nothing here touches the real one.
"""

import shutil

import pytest

import bootstrap

# A miniature config.py with every shape the rewriter must cope with: a value to
# replace, an empty value, a value with a trailing comment, a bare comment, a
# multi-line dict, a number, an indented line, and a non-literal expression.
SAMPLE = '''"""Docstring with a = sign and a # hash."""

# a standalone comment that must survive
REGION = "us-west-2"  # trailing comment that must survive
ACCOUNT_ID = "000000000000"
BUCKET = "your-corpus-bucket"  # S3 bucket holding the paper corpus
KB_ID = ""
GUARDRAIL_VERSION = "DRAFT"

MODELS = {
    "haiku": "us.anthropic.claude-haiku-4-5",
    "opus": "us.anthropic.claude-opus-5",
}
EMBED_DIM = 1024
PORT = 8000
HAND_ROLLED = "a" + "b"


def helper():
    KB_ID = "not top level"  # noqa: F841
    return KB_ID
'''


@pytest.fixture
def cfg(tmp_path):
    """A throwaway config.py in tmp_path.  The real one is never touched."""
    p = tmp_path / "config.py"
    p.write_text(SAMPLE, encoding="utf-8")
    return p


# ----------------------------------------------------------------------
# Bucket name derivation
# ----------------------------------------------------------------------
def test_derived_bucket_name_shape():
    assert bootstrap.derive_bucket_name("123456789012") == "inside-the-lines-pcsk9-123456789012"


def test_derived_bucket_name_is_a_legal_s3_name():
    assert bootstrap.bucket_name_problem(bootstrap.derive_bucket_name("123456789012")) is None


def test_derived_bucket_name_is_unique_per_account():
    a = bootstrap.derive_bucket_name("111111111111")
    b = bootstrap.derive_bucket_name("222222222222")
    assert a != b


def test_derived_bucket_name_is_deterministic():
    # Determinism is load-bearing: a random suffix would make a second run
    # create a SECOND bucket and pay to ingest the corpus twice.
    assert bootstrap.derive_bucket_name("123456789012") == bootstrap.derive_bucket_name(
        "123456789012"
    )


def test_derived_bucket_name_is_found_by_teardowns_name_backstop():
    """teardown.py claims buckets whose normalised name starts "insidethelines".

    Duplicated here rather than imported, because importing teardown.py builds
    boto3 clients at module scope.  If this ever fails, the derived bucket has
    become invisible to teardown whenever the tag is missing -- the exact failure
    that previously left resources behind in people's accounts.
    """
    name = bootstrap.derive_bucket_name("123456789012")
    normalised = "".join(ch for ch in name.lower() if ch.isalnum())
    assert normalised.startswith("insidethelines")


@pytest.mark.parametrize("bad", ["", "12345678901", "1234567890123", "abcdefghijkl", "1234-5678"])
def test_derive_bucket_name_rejects_bad_account_ids(bad):
    with pytest.raises(ValueError):
        bootstrap.derive_bucket_name(bad)


@pytest.mark.parametrize(
    "name",
    [
        "ab",  # too short
        "a" * 64,  # too long
        "Has-Uppercase",
        "under_scores",
        "-leading-hyphen",
        "trailing-hyphen-",
        "two..dots",
        "192.168.0.1",  # IPv4-shaped
        "xn--puny",
        "something-s3alias",
    ],
)
def test_bucket_name_problem_catches_illegal_names(name):
    assert bootstrap.bucket_name_problem(name) is not None


@pytest.mark.parametrize("name", ["abc", "inside-the-lines-pcsk9-123456789012", "a.b-c1"])
def test_bucket_name_problem_accepts_legal_names(name):
    assert bootstrap.bucket_name_problem(name) is None


# ----------------------------------------------------------------------
# The config rewriter
# ----------------------------------------------------------------------
def test_write_config_round_trips_the_values(cfg):
    values = {"KB_ID": "ABCD1234", "GUARDRAIL_VERSION": "1"}
    bootstrap.write_config(values, path=cfg, echo=lambda *_: None)
    got = bootstrap.config_literals(cfg.read_text())
    assert got["KB_ID"] == "ABCD1234"
    assert got["GUARDRAIL_VERSION"] == "1"


def test_write_config_is_idempotent_byte_for_byte(cfg):
    values = {"KB_ID": "ABCD1234", "BUCKET": "inside-the-lines-pcsk9-123456789012"}
    bootstrap.write_config(values, path=cfg, echo=lambda *_: None)
    once = cfg.read_text()
    changed = bootstrap.write_config(values, path=cfg, echo=lambda *_: None)
    assert changed == []  # second run reports nothing changed
    assert cfg.read_text() == once  # and the file is untouched


def test_write_config_preserves_comments_and_layout(cfg):
    before = cfg.read_text()
    bootstrap.write_config({"ACCOUNT_ID": "123456789012"}, path=cfg, echo=lambda *_: None)
    after = cfg.read_text()

    assert "# a standalone comment that must survive" in after
    assert 'REGION = "us-west-2"  # trailing comment that must survive' in after
    assert "# S3 bucket holding the paper corpus" in after  # BUCKET's comment
    assert '"""Docstring with a = sign and a # hash."""' in after
    # exactly one line differs, and it is the one we asked for
    diff = [(a, b) for a, b in zip(before.split("\n"), after.split("\n"), strict=True) if a != b]
    assert diff == [('ACCOUNT_ID = "000000000000"', 'ACCOUNT_ID = "123456789012"')]


def test_write_config_keeps_the_trailing_comment_on_a_line_it_changes(cfg):
    bootstrap.write_config({"BUCKET": "my-bucket"}, path=cfg, echo=lambda *_: None)
    assert 'BUCKET = "my-bucket"  # S3 bucket holding the paper corpus' in cfg.read_text()


def test_write_config_backs_up_first(cfg):
    original = cfg.read_text()
    bootstrap.write_config({"KB_ID": "X"}, path=cfg, echo=lambda *_: None)
    backup = cfg.with_suffix(cfg.suffix + ".bak")
    assert backup.exists()
    assert backup.read_text() == original


def test_write_config_says_it_backed_up(cfg):
    lines = []
    bootstrap.write_config({"KB_ID": "X"}, path=cfg, echo=lines.append)
    assert any("backed up" in line for line in lines)


def test_write_config_prints_what_it_wrote(cfg):
    lines = []
    bootstrap.write_config({"KB_ID": "VISIBLE"}, path=cfg, echo=lines.append)
    assert any("VISIBLE" in line for line in lines)


def test_write_config_never_touches_a_dict_or_a_number(cfg):
    bootstrap.write_config(
        {"MODELS": "clobbered", "EMBED_DIM": "clobbered", "PORT": "clobbered"},
        path=cfg,
        echo=lambda *_: None,
    )
    got = bootstrap.config_literals(cfg.read_text())
    assert got["MODELS"] == {
        "haiku": "us.anthropic.claude-haiku-4-5",
        "opus": "us.anthropic.claude-opus-5",
    }
    assert got["EMBED_DIM"] == 1024
    assert got["PORT"] == 8000


def test_write_config_leaves_a_hand_written_expression_alone(cfg):
    """A non-literal assignment is a human's code.  Skip it and SAY so.

    Crucially it must also not be appended at the end of the file, which would
    silently shadow the expression.
    """
    lines = []
    bootstrap.write_config({"HAND_ROLLED": "nope"}, path=cfg, echo=lines.append)
    text = cfg.read_text()
    assert 'HAND_ROLLED = "a" + "b"' in text
    assert 'HAND_ROLLED = "nope"' not in text
    assert any("left HAND_ROLLED alone" in line for line in lines)


def test_write_config_ignores_assignments_that_are_not_top_level(cfg):
    """KB_ID is also assigned inside helper(); only the module-level one changes."""
    bootstrap.write_config({"KB_ID": "TOP"}, path=cfg, echo=lambda *_: None)
    text = cfg.read_text()
    assert 'KB_ID = "TOP"' in text
    assert 'KB_ID = "not top level"' in text


def test_write_config_appends_a_missing_key_once(cfg):
    values = {"GATEWAY_URL": "https://example.invalid/mcp"}
    bootstrap.write_config(values, path=cfg, echo=lambda *_: None)
    first = cfg.read_text()
    assert bootstrap.config_literals(first)["GATEWAY_URL"] == "https://example.invalid/mcp"
    assert first.count("GATEWAY_URL") == 1
    # ...and appending is itself idempotent: no second block, no second line
    bootstrap.write_config(values, path=cfg, echo=lambda *_: None)
    assert cfg.read_text() == first


def test_write_config_result_always_parses(cfg):
    bootstrap.write_config(
        {"KB_ID": "A", "DATA_SOURCE_ID": "B", "GATEWAY_ENGINE_ID": "C"},
        path=cfg,
        echo=lambda *_: None,
    )
    compile(cfg.read_text(), str(cfg), "exec")  # raises SyntaxError if broken


def test_write_config_creates_config_from_the_example_when_absent(tmp_path, monkeypatch):
    """The brief's explicit requirement: create it, say so, do not error."""
    example = tmp_path / "config.example.py"
    shutil.copy(bootstrap.EXAMPLE_PATH, example)
    monkeypatch.setattr(bootstrap, "EXAMPLE_PATH", example)
    # No credentials wanted in a test: stub the two AWS lookups ensure_config does.
    monkeypatch.setattr(bootstrap, "resolve_account_id", lambda *_a, **_k: "123456789012")
    monkeypatch.setattr(bootstrap, "choose_region", lambda *_a, **_k: "us-west-2")

    target = tmp_path / "config.py"
    lines = []
    bootstrap.write_config({"KB_ID": "MADE"}, path=target, echo=lines.append)

    assert target.exists()
    assert any("not found" in line for line in lines)
    got = bootstrap.config_literals(target.read_text())
    assert got["KB_ID"] == "MADE"


# ----------------------------------------------------------------------
# ensure_config: generation, and "a real value always wins"
# ----------------------------------------------------------------------
@pytest.fixture
def no_aws(tmp_path, monkeypatch):
    """Point bootstrap at tmp_path and stub out its two AWS lookups."""
    example = tmp_path / "config.example.py"
    shutil.copy(bootstrap.EXAMPLE_PATH, example)
    monkeypatch.setattr(bootstrap, "EXAMPLE_PATH", example)
    monkeypatch.setattr(bootstrap, "resolve_account_id", lambda *_a, **_k: "123456789012")
    monkeypatch.setattr(bootstrap, "choose_region", lambda *_a, **_k: "us-west-2")
    monkeypatch.setattr(bootstrap.boto3, "Session", lambda *_a, **_k: object())
    return tmp_path / "config.py"


def test_ensure_config_generates_a_complete_file(no_aws):
    filled = bootstrap.ensure_config(path=no_aws, echo=lambda *_: None)
    assert filled["ACCOUNT_ID"] == "123456789012"
    assert filled["BUCKET"] == "inside-the-lines-pcsk9-123456789012"
    got = bootstrap.config_literals(no_aws.read_text())
    assert got["ACCOUNT_ID"] == "123456789012"
    assert got["BUCKET"] == "inside-the-lines-pcsk9-123456789012"
    assert got["REGION"] == "us-west-2"
    # the template's comments came along, so the generated file is still the
    # documented one -- nobody has to run `cp config.example.py config.py`
    text = no_aws.read_text()
    assert "# --- pricing " in text
    assert "auto-derived AND auto-created" in text  # BUCKET's own comment


def test_ensure_config_is_idempotent(no_aws):
    bootstrap.ensure_config(path=no_aws, echo=lambda *_: None)
    once = no_aws.read_text()
    assert bootstrap.ensure_config(path=no_aws, echo=lambda *_: None) == {}
    assert no_aws.read_text() == once


def test_ensure_config_never_overwrites_a_real_value(no_aws):
    no_aws.write_text(
        'REGION = "eu-west-1"\nACCOUNT_ID = "999999999999"\nBUCKET = "my-own-bucket"\n',
        encoding="utf-8",
    )
    assert bootstrap.ensure_config(path=no_aws, echo=lambda *_: None) == {}
    got = bootstrap.config_literals(no_aws.read_text())
    assert (got["REGION"], got["ACCOUNT_ID"], got["BUCKET"]) == (
        "eu-west-1",
        "999999999999",
        "my-own-bucket",
    )


def test_ensure_config_fills_only_the_placeholders(no_aws):
    no_aws.write_text(
        'REGION = "eu-west-1"\nACCOUNT_ID = "000000000000"\nBUCKET = "my-own-bucket"\n',
        encoding="utf-8",
    )
    filled = bootstrap.ensure_config(path=no_aws, echo=lambda *_: None)
    assert filled == {"ACCOUNT_ID": "123456789012"}
    got = bootstrap.config_literals(no_aws.read_text())
    assert got["REGION"] == "eu-west-1"  # respected, even though it is unusual
    assert got["BUCKET"] == "my-own-bucket"


def test_ensure_config_survives_having_no_credentials(no_aws, monkeypatch):
    """No creds must leave a readable config.py, not a traceback.

    preflight.py is what then tells the user to run `aws configure`.
    """
    monkeypatch.setattr(bootstrap, "resolve_account_id", lambda *_a, **_k: None)
    bootstrap.ensure_config(path=no_aws, echo=lambda *_: None)
    got = bootstrap.config_literals(no_aws.read_text())
    assert got["ACCOUNT_ID"] == "000000000000"  # still the placeholder
    compile(no_aws.read_text(), str(no_aws), "exec")


# ----------------------------------------------------------------------
# safe_session: a bad AWS_PROFILE must be a sentence, not a traceback
# ----------------------------------------------------------------------
def test_safe_session_reports_a_missing_profile(monkeypatch):
    """`boto3.Session()` raises ProfileNotFound in its CONSTRUCTOR.

    Found while testing this work: with AWS_PROFILE set to a name that does not
    exist, the traceback landed before preflight could say anything. This test
    is the regression guard.
    """
    monkeypatch.setenv("AWS_PROFILE", "definitely-not-a-real-profile-xq7")
    session, problem = bootstrap.safe_session()
    assert session is None
    assert problem is not None
    assert "aws configure list-profiles" in problem  # names the fix


def test_safe_session_treats_an_empty_profile_as_unset(monkeypatch):
    """`AWS_PROFILE=` (nothing after the =) is a profile named "" to botocore."""
    monkeypatch.setenv("AWS_PROFILE", "")
    session, problem = bootstrap.safe_session()
    assert problem is None
    assert session is not None


def test_ensure_config_still_writes_a_file_with_a_broken_profile(no_aws, monkeypatch):
    """A bad profile must still leave a complete, parseable config.py behind."""
    monkeypatch.setattr(
        bootstrap, "safe_session", lambda: (None, "AWS_PROFILE is set to 'nope', but ...")
    )
    monkeypatch.delattr(bootstrap.boto3, "Session", raising=False)
    lines = []
    bootstrap.ensure_config(path=no_aws, echo=lines.append)
    assert no_aws.exists()
    compile(no_aws.read_text(), str(no_aws), "exec")
    got = bootstrap.config_literals(no_aws.read_text())
    assert got["REGION"] == bootstrap.DEFAULT_REGION
    assert got["ACCOUNT_ID"] == "000000000000"
    assert any("AWS_PROFILE" in line for line in lines)


# ----------------------------------------------------------------------
# The cost notice -- it must state real, non-zero, dated numbers
# ----------------------------------------------------------------------
def test_cost_notice_states_the_measured_numbers():
    text = bootstrap.cost_notice()
    assert "0.283" in text  # one-time ingestion
    assert "0.32" in text  # one full run
    assert "2026-09-22" in text  # dated, because rates move
    assert "teardown" in text
