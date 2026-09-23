#!/usr/bin/env python3
"""
bootstrap.py  --  everything a stranger used to have to do by hand.

This module exists because of one sentence from the owner of this repo:

    "They should NOT have to get anything right -- it should just work."

This is a PUBLIC repo for a conference talk.  Someone will clone it onto a
laptop five minutes before a session and run it.  Before this file existed they
had to get five things right by hand, any one of which failed the run:

  1. look up their 12-digit AWS account ID and paste it into config.py
  2. invent a globally-unique S3 bucket name (the placeholder
     "your-corpus-bucket" is already taken by somebody, so it always failed)
  3. create that bucket themselves -- nothing in the repo created it
  4. run `aws s3 sync corpus/ s3://THEIR-BUCKET/corpus/` from the README
  5. copy SEVEN generated resource IDs out of build_kb.py's stdout and hand-edit
     them into config.py without a typo

Every one of those is now automatic, and this file is where that happens:

    resolve_account_id()   STS get_caller_identity  -> the 12 digits (1)
    derive_bucket_name()   deterministic, unique by construction (2)
    ensure_bucket()        idempotent create_bucket, both region shapes (3)
    upload_corpus()        idempotent, parallel, pure boto3 (4)
    write_config()         surgical rewrite of config.py, backed up first (5)
    ensure_config()        generates config.py from scratch when absent

READ-ONLY-SAFE DESIGN NOTE.  Nothing in here deletes or overwrites a value a
human chose.  Every filler only replaces a value that is empty or is one of the
documented placeholders (see PLACEHOLDERS).  A real value already in config.py
always wins.  That rule is what makes it safe to run the whole chain again at
any time -- which is the point, because `make start` is resumable.

WHY THIS IS A ROOT-LEVEL MODULE, not part of the agentcore_demo package:
build_kb.py, corpus_fetch.py and teardown.py all live at the repo root and do
`import config as cfg`, where `config` is the root config.py.  Bootstrap has to
run BEFORE that import can succeed (it is what creates config.py), so it has to
be importable from the same place, with no dependency on the installed package.
tests/conftest.py already puts the repo root on sys.path for exactly this
reason, so tests/test_bootstrap.py can `import bootstrap`.
"""

from __future__ import annotations

import ast
import contextlib
import os
import re
import shutil
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import boto3
from botocore.exceptions import ClientError, ProfileNotFound

# ----------------------------------------------------------------------
# Paths and constants
# ----------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent
CONFIG_PATH = REPO_ROOT / "config.py"
EXAMPLE_PATH = REPO_ROOT / "config.example.py"
CORPUS_DIR = REPO_ROOT / "corpus"

# The region this demo is developed, rehearsed and priced against.  Used when
# the caller has no region configured at all, and as the fallback when their
# configured region cannot serve the models (see choose_region).
DEFAULT_REGION = "us-west-2"

# The stem of the derived corpus bucket name.  The account ID is appended, which
# is what makes the result globally unique BY CONSTRUCTION rather than by luck:
# S3's namespace is global, but an account ID is not shared, so
# "inside-the-lines-pcsk9-123456789012" cannot collide with another account's
# bucket.  This stem matches the bucket the owner is already running against, so
# an existing install keeps working unchanged.
#
# It also has to stay matched with teardown.py's NAME_PREFIX backstop
# ("inside-the-lines", normalised to "insidethelines"):
#   _norm("inside-the-lines-pcsk9-123456789012") -> "insidethelinespcsk9123456789012"
# which startswith "insidethelines", so teardown finds this bucket by NAME even
# if the tag is missing and config.py has been changed.  Renaming this stem to
# something that does not begin "inside the lines" would silently break that
# backstop -- tests/test_bootstrap.py pins it.
BUCKET_STEM = "inside-the-lines-pcsk9"

# The exact placeholder strings shipped in config.example.py.  A value equal to
# one of these (or empty) is "unset" and may be filled in; anything else is a
# human's choice and is left alone.
PLACEHOLDERS = {
    "ACCOUNT_ID": {"", "000000000000"},
    "BUCKET": {"", "your-corpus-bucket"},
    "REGION": {""},
}

# ----------------------------------------------------------------------
# What this costs.  Printed by start.py before anything is provisioned, so the
# number is visible without reading the README.
# ----------------------------------------------------------------------
# MEASURED against a real account on 2026-09-22 in us-west-2 -- not estimated.
# "Cost numbers must be real" is a standing guardrail in CLAUDE.md, so these are
# the observed figures, and they are dated because Bedrock rates move.
# A stranger's numbers will differ slightly: the corpus is whatever PMC returns
# on the day, so character count and vector count vary by a few percent.
COST_BUILD_USD = 0.283  # one-time ingestion: 56,599,850 chars ~= 14.1M tokens
COST_RUN_USD = 0.32  # one full five-beat run, measured $0.322552
COST_IDLE_USD_PER_MONTH = 0.013  # S3 Vectors $0.0113 + S3 corpus $0.0012


def cost_notice() -> str:
    """Return the plain-language cost summary printed before provisioning.

    Deliberately NOT a confirmation prompt.  The owner's call: document the cost
    clearly and proceed.  "Informed, not interrogated" -- stopping a scripted
    five-minute demo to ask "are you sure?" is its own sharp edge.
    """
    rows = (
        ("fetch the papers from PubMed Central", "$0.00", "free; 10-15 min"),
        ("build the knowledge base (one-time)", f"${COST_BUILD_USD:.3f}", "computed, not metered"),
        ("each full five-beat demo run", f"${COST_RUN_USD:.2f}", "measured"),
        ("leave it provisioned, never run it", f"${COST_IDLE_USD_PER_MONTH:.3f}", "per month"),
    )
    table = "\n".join(f"    {label:<38}{amount:>7}   {note}" for label, amount, note in rows)
    first_day = round((COST_BUILD_USD + COST_RUN_USD) * 100)
    return (
        "\n  What this costs\n"
        f"{table}\n"
        "\n"
        f"    First day, soup to nuts: about {first_day} cents.\n"
        f"    Every rehearsal after that: about {round(COST_RUN_USD * 100)} cents.\n"
        "    Q4 -- two frontier models reviewing in parallel -- is ~76% of a run.\n"
        "    `make teardown` removes everything, and verifies nothing is left.\n"
        "\n"
        "    Measured in us-west-2 on 2026-09-22. Bedrock rates move, and your\n"
        "    corpus is whatever PMC returns on the day, so treat these as\n"
        "    approximate rather than exact.\n"
    )


# ----------------------------------------------------------------------
# 0. Building a boto3 Session without a traceback
# ----------------------------------------------------------------------
def safe_session() -> tuple[boto3.Session | None, str | None]:
    """Build a boto3 Session, returning (session, problem) instead of raising.

    Returns (session, None) on success, (None, one-sentence problem) on failure.

    This exists because of a genuinely nasty edge found while testing this file:
    `boto3.Session()` ITSELF raises `ProfileNotFound` -- before any AWS call,
    before any of preflight.py's friendly messages can run -- if AWS_PROFILE
    names a profile that does not exist in ~/.aws/config.  A stranger who typos
    their profile name, or who copies an `AWS_PROFILE=` line out of the README
    with nothing after the `=`, got a 30-line botocore traceback ending in
    `ProfileNotFound: The config profile () could not be found`.

    An EMPTY AWS_PROFILE is treated as unset, because that is unambiguously what
    the user meant and botocore's "a profile literally named empty string" is
    never what anyone wants.
    """
    if os.environ.get("AWS_PROFILE", "").strip() == "" and "AWS_PROFILE" in os.environ:
        # Empty or whitespace: honour the intent, not the literal value.
        os.environ.pop("AWS_PROFILE")
    try:
        return boto3.Session(), None
    except ProfileNotFound:
        profile = os.environ.get("AWS_PROFILE", "(unset)")
        return None, (
            f"AWS_PROFILE is set to {profile!r}, but no such profile exists. "
            f"Run `aws configure list-profiles` to see the ones you have, then "
            f"`export AWS_PROFILE=<one of them>` -- or `unset AWS_PROFILE` to use "
            f"your default credentials."
        )
    except Exception as e:  # noqa: BLE001 -- a broken ~/.aws must not be a traceback
        return None, (
            f"Could not read your AWS configuration ({type(e).__name__}). "
            f"Check ~/.aws/config and ~/.aws/credentials, or run `aws configure`."
        )


# ----------------------------------------------------------------------
# 1. Account ID  --  was: "look up your 12-digit account ID and paste it"
# ----------------------------------------------------------------------
def resolve_account_id(session: boto3.Session | None = None) -> str | None:
    """Return the caller's 12-digit AWS account ID, or None if it can't be read.

    `sts get-caller-identity` is the one AWS call every caller can always make:
    it needs no IAM permission at all (it is implicitly allowed for any valid
    credential), so if this fails the credentials themselves are the problem --
    which is what preflight.py reports, in one sentence.

    Returns None rather than raising so that config generation can still produce
    a complete, readable config.py on a machine with no credentials yet.  The
    account ID is left as the placeholder and preflight says what to run.
    """
    try:
        return (session or boto3.Session()).client("sts").get_caller_identity()["Account"]
    except Exception:  # noqa: BLE001 -- any failure here means "no usable creds"
        return None


# ----------------------------------------------------------------------
# 2. Bucket name  --  was: "invent a globally unique S3 bucket name"
# ----------------------------------------------------------------------
# S3 bucket naming rules, transcribed from the Bucket naming rules doc.  Getting
# any of these wrong produces `InvalidBucketName`, which is exactly the class of
# failure this whole file exists to remove -- so the derived name is validated
# before it is ever sent to AWS.
_BUCKET_RE = re.compile(r"^[a-z0-9][a-z0-9.-]{1,61}[a-z0-9]$")
_IPV4_RE = re.compile(r"^\d{1,3}(\.\d{1,3}){3}$")
_BAD_PREFIXES = ("xn--", "sthree-", "amzn-s3-demo-")
_BAD_SUFFIXES = ("-s3alias", "--ol-s3", ".mrap", "--x-s3", "--table-s3")


def bucket_name_problem(name: str) -> str | None:
    """Return a human-readable reason `name` is not a legal S3 bucket name.

    None means the name is legal.  Kept as a separate pure function so
    tests/test_bootstrap.py can prove the derived name is legal without any
    credentials or network -- the rules are fiddly and a silent violation would
    surface as a cryptic InvalidBucketName mid-provision.
    """
    if not 3 <= len(name) <= 63:
        return f"must be 3-63 characters, got {len(name)}"
    if not _BUCKET_RE.match(name):
        return (
            "must be lowercase letters, digits, dots and hyphens only, "
            "starting and ending with a letter or digit"
        )
    if ".." in name:
        return "must not contain two adjacent dots"
    if _IPV4_RE.match(name):
        return "must not look like an IPv4 address"
    for p in _BAD_PREFIXES:
        if name.startswith(p):
            return f"must not start with the reserved prefix {p!r}"
    for s in _BAD_SUFFIXES:
        if name.endswith(s):
            return f"must not end with the reserved suffix {s!r}"
    return None


def derive_bucket_name(account_id: str, stem: str = BUCKET_STEM) -> str:
    """Derive the corpus bucket name for an account: "<stem>-<account-id>".

    Deterministic on purpose.  A random suffix would also be unique, but it
    would mean a second `make start` (or a run from a different clone of this
    repo) invented a SECOND bucket and paid to ingest the corpus twice -- and
    left the first one behind in the account.  Determinism is what makes the
    whole chain re-runnable.

    Raises:
        ValueError: if the account ID is not 12 digits, or the result would not
            be a legal bucket name.  Both are programmer errors, not user
            errors, so they are loud.
    """
    account_id = str(account_id).strip()
    if not (len(account_id) == 12 and account_id.isdigit()):
        raise ValueError(f"account ID must be 12 digits, got {account_id!r}")
    name = f"{stem}-{account_id}"
    problem = bucket_name_problem(name)
    if problem:
        raise ValueError(f"derived bucket name {name!r} is invalid: {problem}")
    return name


# ----------------------------------------------------------------------
# 3. Bucket creation  --  was: nothing in the repo created the bucket at all
# ----------------------------------------------------------------------
def ensure_bucket(
    bucket: str,
    region: str,
    s3=None,
    *,
    echo=print,
) -> str:
    """Create the corpus bucket if it does not exist.  Idempotent.

    Returns one of "created", "exists".

    THE us-east-1 SPECIAL CASE, which is the whole reason this is a function and
    not two lines inline.  `create_bucket` takes a LocationConstraint:

        us-east-1       CreateBucketConfiguration must be OMITTED.  Passing
                        {"LocationConstraint": "us-east-1"} is an error: S3
                        rejects it with InvalidLocationConstraint.  us-east-1 is
                        the legacy default and is expressed by saying nothing.
                        Note the failure is SERVER-side, not client-side:
                        "us-east-1" is absent from botocore's LocationConstraint
                        enum (checked 2026-09-22), but botocore does not enforce
                        enums, so `validate_parameters` accepts the bad call
                        happily.  There is no local check that catches this --
                        which is exactly why it is handled here in code.
        everywhere else CreateBucketConfiguration is REQUIRED.  Omitting it
                        creates the bucket in us-east-1 instead of the region
                        you asked for, which then makes every KB ingestion pay
                        cross-Region transfer -- a silent wrong answer, the
                        worst kind.

    Already-exists handling, in the order the API reports it:
      - head_bucket 200                -> exists and we can see it
      - head_bucket 403                -> exists, someone else owns it.  Only
                                          reachable if BUCKET was hand-edited to
                                          a taken name; the derived name cannot
                                          collide.  Reported, not retried.
      - head_bucket 404                -> does not exist, create it
      - BucketAlreadyOwnedByYou        -> a concurrent/earlier run beat us; fine
      - BucketAlreadyExists            -> taken by another account
    """
    s3 = s3 or boto3.client("s3", region_name=region)

    problem = bucket_name_problem(bucket)
    if problem:
        raise ValueError(
            f"BUCKET = {bucket!r} in config.py is not a legal S3 bucket name: {problem}"
        )

    try:
        s3.head_bucket(Bucket=bucket)
        echo(f"  corpus bucket exists: s3://{bucket}")
        return "exists"
    except ClientError as e:
        code = e.response.get("Error", {}).get("Code", "")
        status = e.response.get("ResponseMetadata", {}).get("HTTPStatusCode")
        if code == "403" or status == 403:
            raise RuntimeError(
                f"The S3 bucket {bucket!r} exists but belongs to another AWS account. "
                f"Change BUCKET in config.py to a name you own, or delete the BUCKET "
                f"line and re-run to let this script derive one."
            ) from e
        if code not in ("404", "NoSuchBucket") and status != 404:
            raise

    kwargs: dict = {"Bucket": bucket}
    if region != "us-east-1":
        kwargs["CreateBucketConfiguration"] = {"LocationConstraint": region}
    try:
        s3.create_bucket(**kwargs)
    except ClientError as e:
        code = e.response.get("Error", {}).get("Code", "")
        if code == "BucketAlreadyOwnedByYou":
            echo(f"  corpus bucket exists: s3://{bucket}")
            return "exists"
        if code == "BucketAlreadyExists":
            raise RuntimeError(
                f"The S3 bucket name {bucket!r} is already taken by another AWS "
                f"account. Change BUCKET in config.py."
            ) from e
        raise
    echo(f"  created corpus bucket: s3://{bucket}  ({region})")
    return "created"


# ----------------------------------------------------------------------
# 4. Corpus upload  --  was: README step 4, a manual `aws s3 sync`
# ----------------------------------------------------------------------
# WHY THE UPLOAD LIVES HERE AND IS CALLED FROM build_kb.py, NOT corpus_fetch.py
# ---------------------------------------------------------------------------
# It was a real choice; the reasoning, so nobody "tidies" it the other way:
#
#   * build_kb.py is the step that NEEDS the objects.  Its ingestion job reads
#     s3://BUCKET/CORPUS_PREFIX; if that prefix is empty the job "succeeds" with
#     zero documents and the demo then fails on stage with an empty KB.  Putting
#     the upload immediately before ingestion makes the precondition a
#     guarantee instead of a hope about command ordering.
#   * The brief's own test case -- "must work if the user runs only
#     `make build-kb` after a fresh `make corpus`" -- is satisfied by
#     construction this way.  In corpus_fetch.py it would NOT be: a user who ran
#     `make corpus`, then hit Ctrl-C, then ran `make build-kb` would provision
#     against an empty bucket.
#   * corpus_fetch.py has no AWS dependency today (just `requests`), so
#     `make corpus` works with no credentials at all.  Keeping it that way means
#     the slow, free, network-bound step is never blocked by an AWS problem.
#   * build_kb.py already owns the bucket (it creates and tags it).  Bucket
#     lifecycle and bucket contents belong together.
def upload_corpus(
    bucket: str,
    prefix: str,
    local_dir: str | os.PathLike = CORPUS_DIR,
    s3=None,
    *,
    region: str = DEFAULT_REGION,
    workers: int = 16,
    echo=print,
) -> tuple[int, int]:
    """Upload corpus/*.txt to s3://bucket/prefix.  Idempotent and parallel.

    Returns (uploaded, skipped).

    Pure boto3, deliberately: `aws s3 sync` would be one line, but the AWS CLI
    is a separate install that many people do not have, while boto3 is already a
    hard dependency of this package.  Shelling out to a tool that may not exist
    is exactly the sharp edge being removed.

    Idempotent: one `list_objects_v2` pass builds {key: size} for what is
    already under the prefix, and a local file is uploaded only if its key is
    missing or its size differs.  So a re-run after a completed upload costs a
    handful of LIST calls and finishes in about a second -- which is what makes
    `make start` cheap to re-run.

    Parallel: 1,000 files / ~48 MB is far too slow one PUT at a time (each is a
    fresh HTTPS round trip; serial takes minutes).  A ThreadPoolExecutor over a
    SINGLE boto3 client is the documented pattern -- boto3's low-level clients
    are thread-safe and the docs recommend creating one client and sharing it
    across threads (resources are the ones that are not thread-safe).

    Progress is printed because this takes a while and a silent multi-minute
    pause reads as a hang.
    """
    local_dir = Path(local_dir)
    if not local_dir.is_dir():
        raise RuntimeError(
            f"No corpus found at {local_dir}. Run `make corpus` first "
            f"(it downloads ~1,000 papers from PubMed Central; 10-15 minutes)."
        )
    files = sorted(p for p in local_dir.iterdir() if p.is_file() and p.suffix == ".txt")
    if not files:
        raise RuntimeError(
            f"{local_dir} exists but holds no .txt files. Run `make corpus` to "
            f"(re-)download the papers."
        )

    prefix = prefix if prefix.endswith("/") else prefix + "/"
    s3 = s3 or boto3.client("s3", region_name=region)

    # What is already up there?  One paginated LIST instead of 1,000 HEADs.
    remote: dict[str, int] = {}
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            remote[obj["Key"]] = obj["Size"]

    todo = [p for p in files if remote.get(f"{prefix}{p.name}") != p.stat().st_size]
    skipped = len(files) - len(todo)
    if not todo:
        echo(f"  corpus already uploaded: {skipped} files under s3://{bucket}/{prefix}")
        return 0, skipped

    total_bytes = sum(p.stat().st_size for p in todo)
    echo(
        f"  uploading {len(todo)} files ({total_bytes / 1e6:.1f} MB) to "
        f"s3://{bucket}/{prefix}  ({skipped} already there)"
    )

    done = 0
    lock = threading.Lock()

    def put(path: Path) -> None:
        nonlocal done
        with path.open("rb") as fh:
            s3.put_object(
                Bucket=bucket,
                Key=f"{prefix}{path.name}",
                Body=fh,
                ContentType="text/plain; charset=utf-8",
            )
        with lock:
            done += 1
            # Every 100 files, plus the last one, so the line always ends at 100%.
            if done % 100 == 0 or done == len(todo):
                echo(f"    {done}/{len(todo)} uploaded")

    with ThreadPoolExecutor(max_workers=workers) as pool:
        # list() forces every future to be consumed, so the FIRST exception is
        # re-raised here rather than swallowed.  A half-uploaded corpus that
        # reported success would be the worst possible outcome: ingestion would
        # succeed with a partial KB and the gap would only show up on stage.
        list(pool.map(put, todo))

    echo(f"  corpus in S3: {len(todo)} uploaded, {skipped} unchanged")
    return len(todo), skipped


# ----------------------------------------------------------------------
# 5. config.py  --  was: "paste these seven values into config.py"
# ----------------------------------------------------------------------
# This is the highest-risk thing in the repo: it edits a file a human may have
# hand-tuned, and config.py is git-ignored, so a mistake here is UNRECOVERABLE
# from git.  Four independent safeguards, in the order they fire:
#
#   1. A line is only touched if the WHOLE line matches _SIMPLE_ASSIGN below:
#      NAME = "a simple single-quoted or double-quoted string"  # optional comment
#      Nothing else can match -- not a dict literal, not a number, not a
#      continuation line, not an indented line, not an expression.  So MODELS,
#      PRICING and every comment in the file are unreachable by construction.
#   2. The new text is parsed with `ast` and the resulting values are read back
#      and compared to what we meant to write, BEFORE anything is written.
#   3. Only then is config.py copied to config.py.bak (and said so on stdout).
#   4. If the new text is byte-identical to the old, nothing is written at all --
#      so running build_kb.py twice cannot duplicate a line or churn the file.
_SIMPLE_ASSIGN = re.compile(
    r"""^
    (?P<name>[A-Za-z_][A-Za-z0-9_]*)   # a top-level name, at column 0
    (?P<eq>[ \t]*=[ \t]*)
    (?P<val>"[^"\\\n]*" | '[^'\\\n]*') # a plain string literal: no escapes
    (?P<rest>[ \t]*(?:\#.*)?)          # optional trailing comment, preserved
    $""",
    re.VERBOSE,
)

_APPEND_HEADER = "# --- added automatically by build_kb.py (keys were missing) ---"


def config_literals(text: str) -> dict[str, object]:
    """Read every top-level `NAME = <literal>` out of a config file's source.

    Uses `ast`, not import, so it works on a file that is not on sys.path and
    has no side effects.  Non-literal assignments (e.g. `X = os.environ[...]`)
    are skipped: they are values this module must never claim to understand.
    """
    out: dict[str, object] = {}
    for node in ast.parse(text).body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name):
                # A non-literal (e.g. `X = os.environ["Y"]`) is deliberately NOT
                # represented in the result: it is a value this module must never
                # claim to understand, and the absence is what makes
                # rewrite_assignments() leave it alone.
                with contextlib.suppress(ValueError, SyntaxError, TypeError):
                    out[target.id] = ast.literal_eval(node.value)
    return out


def config_assigned_names(text: str) -> set[str]:
    """Every name assigned at the top level, literal or not.

    Distinct from config_literals(): a name can be PRESENT but not a literal.
    Such a name must be left alone AND must not be appended to the end of the
    file, because an appended assignment would silently shadow the human's
    expression.  Keeping the two questions separate is what prevents that.
    """
    names: set[str] = set()
    for node in ast.parse(text).body:
        if isinstance(node, ast.Assign):
            names |= {t.id for t in node.targets if isinstance(t, ast.Name)}
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
    return names


def rewrite_assignments(text: str, values: dict[str, str]) -> tuple[str, list[str], list[str]]:
    """Rewrite the named top-level string assignments in `text`.

    Returns (new_text, changed, skipped):
        changed  names whose value was actually different and was replaced
        skipped  names present in the file but NOT in a rewritable shape

    Every top-level occurrence of a name is rewritten, not just the first.  If a
    file somehow assigns KB_ID twice, Python would use the LAST one, so updating
    only the first would leave a stale value winning -- the exact silent failure
    this function exists to prevent.

    The trailing comment on a line is preserved verbatim, because config.py's
    comments are how a reader knows what each value is for.
    """
    changed: list[str] = []
    seen: set[str] = set()
    out_lines: list[str] = []

    for line in text.split("\n"):
        m = _SIMPLE_ASSIGN.match(line)
        if m and m.group("name") in values:
            name = m.group("name")
            seen.add(name)
            new_line = f'{name}{m.group("eq")}"{values[name]}"{m.group("rest")}'
            if new_line != line:
                changed.append(name)
            out_lines.append(new_line)
        else:
            out_lines.append(line)

    present = config_assigned_names(text)
    skipped = sorted(n for n in values if n in present and n not in seen)

    new_text = "\n".join(out_lines)

    # Names that appear nowhere at top level get appended in one marked block.
    missing = [n for n in values if n not in present]
    if missing:
        if not new_text.endswith("\n"):
            new_text += "\n"
        block = [_APPEND_HEADER] + [f'{n} = "{values[n]}"' for n in missing]
        new_text += "\n" + "\n".join(block) + "\n"
        changed.extend(missing)

    return new_text, changed, skipped


def write_config(values: dict[str, str], path: Path = CONFIG_PATH, *, echo=print) -> list[str]:
    """Write `values` into config.py, safely and idempotently.

    Returns the list of keys actually changed (empty means "already correct").

    Replaces build_kb.py's old "Done. Paste these into config.py:" block.  It
    still PRINTS every value, so the user can see exactly what changed -- the
    write-back removes the transcription step, not the visibility.

    If config.py does not exist it is created from config.example.py first and
    that is stated on stdout, rather than erroring.
    """
    if not path.exists():
        echo(f"  {path.name} not found -- creating it from {EXAMPLE_PATH.name}")
        ensure_config(path=path, echo=echo)

    original = path.read_text(encoding="utf-8")
    new_text, changed, skipped = rewrite_assignments(original, values)

    for name in skipped:
        echo(
            f"  left {name} alone in {path.name}: it is not a plain string "
            f"literal, so it looks hand-written. Expected value: {values[name]!r}"
        )

    if new_text == original:
        echo(f"  {path.name} already up to date -- not rewritten")
        return []

    # Safeguard 2: prove the new text parses AND reads back as intended, BEFORE
    # touching the file.  If this raises, config.py is untouched.
    try:
        readback = config_literals(new_text)
    except SyntaxError as e:  # pragma: no cover -- only reachable on a bug here
        raise RuntimeError(
            f"refusing to write {path.name}: the rewritten file would not parse ({e})"
        ) from e
    for name in changed:
        if readback.get(name) != values[name]:
            raise RuntimeError(  # pragma: no cover -- only reachable on a bug here
                f"refusing to write {path.name}: {name} would read back as "
                f"{readback.get(name)!r}, not {values[name]!r}"
            )

    # Safeguard 3: back up first, and say so.  config.py is git-ignored, so
    # this .bak is the only undo that exists.
    backup = path.with_suffix(path.suffix + ".bak")
    shutil.copy2(path, backup)
    echo(f"  backed up {path.name} -> {backup.name}")

    path.write_text(new_text, encoding="utf-8")
    echo(f"  wrote {len(changed)} value(s) into {path.name}:")
    for name in changed:
        echo(f'    {name} = "{values[name]}"')
    return changed


def choose_region(account_models: dict[str, str], session: boto3.Session | None = None) -> str:
    """Pick the region to write into a brand-new config.py.

    Prefers the region the caller's environment already resolves to, but only if
    Bedrock in that region actually offers every model in `account_models`.
    Otherwise falls back to DEFAULT_REGION (us-west-2), which is where this demo
    is developed, rehearsed and priced.

    Why probe instead of just taking the caller's region: a stranger whose
    profile says eu-west-2 would otherwise get a config.py that cannot possibly
    work, and would have to diagnose it.  Why probe instead of just hard-coding
    us-west-2: someone who has deliberately set up Bedrock elsewhere should get
    their own region.  The probe is a single read-only `list-inference-profiles`.
    """
    session = session or boto3.Session()
    candidate = session.region_name
    if not candidate or candidate == DEFAULT_REGION:
        return DEFAULT_REGION
    try:
        available = active_inference_profiles(session, candidate)
    except Exception:  # noqa: BLE001 -- no creds / no Bedrock here; preflight reports it
        return DEFAULT_REGION
    missing = sorted(set(account_models.values()) - available)
    if missing:
        print(
            f"  your AWS region is {candidate}, but Bedrock there is missing "
            f"{len(missing)} of the demo's models -- using {DEFAULT_REGION} instead"
        )
        return DEFAULT_REGION
    return candidate


def active_inference_profiles(session: boto3.Session, region: str) -> set[str]:
    """Return the IDs of every ACTIVE system-defined inference profile in region.

    Shared by choose_region() and preflight.py.  `typeEquals="SYSTEM_DEFINED"`
    excludes application profiles a user may have made; the demo only ever names
    AWS's own `us.` Geo profiles.
    """
    br = session.client("bedrock", region_name=region)
    out: set[str] = set()
    token = None
    while True:
        kwargs: dict = {"maxResults": 100, "typeEquals": "SYSTEM_DEFINED"}
        if token:
            kwargs["nextToken"] = token
        page = br.list_inference_profiles(**kwargs)
        for p in page.get("inferenceProfileSummaries", []):
            if p.get("status") == "ACTIVE":
                out.add(p["inferenceProfileId"])
        token = page.get("nextToken")
        if not token:
            return out


def ensure_config(
    path: Path = CONFIG_PATH,
    example: Path = EXAMPLE_PATH,
    *,
    session: boto3.Session | None = None,
    echo=print,
) -> dict[str, str]:
    """Make sure config.py exists and its account/bucket/region are filled in.

    Returns the values it filled in (empty dict means nothing needed doing).

    Two cases:
      * config.py ABSENT -- generate it in full from config.example.py, which is
        tracked, so a fresh clone always has the template.  Every comment in the
        example comes along, so the generated file is still the documented one.
        Nobody ever runs `cp config.example.py config.py` by hand again.
      * config.py PRESENT -- respect it.  Only values that are empty or still
        one of the documented PLACEHOLDERS are filled.  A real value a human
        typed always wins; this function will not "correct" it.

    Safe to call on every run.  It is the first thing build_kb.py and start.py
    do, before `import config`, because it is what makes that import work.
    """
    creating = not path.exists()
    source = example if creating else path
    if not source.exists():
        raise RuntimeError(
            f"neither {path.name} nor {example.name} exists -- this clone of the "
            f"repo is incomplete; re-clone it"
        )
    text = source.read_text(encoding="utf-8")
    current = config_literals(text)
    if session is None:
        # safe_session(), not boto3.Session(): a bad AWS_PROFILE makes the
        # constructor itself raise, and a traceback here would land before
        # preflight.py ever gets to explain the problem.  A None session just
        # means the AWS-derived values stay as placeholders and preflight says
        # what to do -- config.py is still generated, complete and readable.
        session, session_problem = safe_session()
        if session_problem:
            echo(f"  {session_problem}")

    fill: dict[str, str] = {}

    # REGION.  Only chosen for a brand-new file; an existing REGION is a
    # deliberate choice (it decides where every resource lives) and is left be
    # unless it is blank.
    if creating or not str(current.get("REGION", "")).strip():
        models = current.get("MODELS") or {}
        fill["REGION"] = (
            choose_region(models if isinstance(models, dict) else {}, session)
            if session is not None
            else DEFAULT_REGION
        )
    region = fill.get("REGION") or str(current.get("REGION") or DEFAULT_REGION)

    # ACCOUNT_ID from STS.  A real value already present always wins.
    if str(current.get("ACCOUNT_ID", "")) in PLACEHOLDERS["ACCOUNT_ID"]:
        account = resolve_account_id(session) if session is not None else None
        if account:
            fill["ACCOUNT_ID"] = account
        else:
            echo(
                "  could not read your AWS account ID yet -- leaving the "
                "placeholder; the preflight message below says what to do"
            )
    account_id = fill.get("ACCOUNT_ID") or str(current.get("ACCOUNT_ID") or "")

    # BUCKET derived from the account ID -- unique by construction.  If there is
    # no account ID yet there can be no bucket name either; preflight stops the
    # run with the credentials message before anything needs one.
    have_account = len(account_id) == 12 and account_id.isdigit()
    if have_account and str(current.get("BUCKET", "")) in PLACEHOLDERS["BUCKET"]:
        fill["BUCKET"] = derive_bucket_name(account_id)

    if creating:
        new_text, _changed, _skipped = rewrite_assignments(text, fill)
        # Same validate-before-write discipline as write_config(), even though
        # there is no user data at risk here: a config.py that does not parse
        # would fail at `import config` with a traceback, which is precisely the
        # experience this file exists to prevent.
        config_literals(new_text)
        path.write_text(new_text, encoding="utf-8")
        echo(f"  generated {path.name} (from {example.name}) -- nothing to edit by hand:")
        for name, value in fill.items():
            echo(f'    {name} = "{value}"')
        echo(f"    region {region}")
        return fill

    if fill:
        write_config(fill, path=path, echo=echo)
    return fill
