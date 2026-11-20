"""
test_corpus_sync.py  --  several computers, one account: the bucket is the master copy.

Pins bootstrap section 4b.  Found 2026-11-20: two computers on one account had
built different corpora from PubMed Central on different days, 268 of 1,000
papers apart.  Once the bucket holds a complete corpus, every computer must use
it -- downloading what it lacks, setting aside (never deleting, never uploading)
what the bucket does not have.

No AWS: FakeS3 implements just the four calls bootstrap makes.
"""

from __future__ import annotations

import pytest
from botocore.exceptions import ClientError

import bootstrap

BUCKET, PREFIX = "inside-the-lines-pcsk9-123456789012", "corpus/"


class FakeS3:
    """In-memory S3: list_objects_v2 (paginated), download_file, put_object."""

    def __init__(self, files: dict[str, bytes] | None = None, *, error: str | None = None):
        self.objects = {f"{PREFIX}{k}": v for k, v in (files or {}).items()}
        self.error = error  # e.g. "NoSuchBucket", "AccessDenied"
        self.puts: list[str] = []
        self.downloads: list[str] = []

    def get_paginator(self, name: str):
        assert name == "list_objects_v2"
        return self

    def paginate(self, Bucket: str, Prefix: str):  # noqa: N803 -- boto3's spelling
        if self.error:
            raise ClientError({"Error": {"Code": self.error, "Message": "x"}}, "ListObjectsV2")
        contents = [
            {"Key": k, "Size": len(v)} for k, v in self.objects.items() if k.startswith(Prefix)
        ]
        yield {"Contents": contents}

    def download_file(self, bucket: str, key: str, path: str) -> None:
        self.downloads.append(key)
        with open(path, "wb") as f:
            f.write(self.objects[key])

    def put_object(self, Bucket: str, Key: str, Body, ContentType: str) -> None:  # noqa: N803
        self.puts.append(Key)
        self.objects[Key] = Body.read()


def _papers(n: int, start: int = 0) -> dict[str, bytes]:
    return {f"PMC{10_000_000 + i}.txt": f"paper {i}\n".encode() for i in range(start, start + n)}


def _write(dirpath, files: dict[str, bytes]) -> None:
    dirpath.mkdir(parents=True, exist_ok=True)
    for name, body in files.items():
        (dirpath / name).write_bytes(body)


@pytest.fixture
def dirs(tmp_path):
    return tmp_path / "corpus", tmp_path / "corpus-set-aside"


def _quiet(*_a, **_k) -> None:
    pass


# -- list_remote_corpus ----------------------------------------------------


def test_a_missing_bucket_is_an_empty_corpus_not_an_error():
    # The first computer on an account: nothing has been created yet.
    assert bootstrap.list_remote_corpus(BUCKET, PREFIX, FakeS3(error="NoSuchBucket")) == {}


def test_an_unreadable_bucket_raises_rather_than_reading_as_empty():
    # Treating AccessDenied as "empty" would fetch and upload a SECOND corpus.
    with pytest.raises(ClientError):
        bootstrap.list_remote_corpus(BUCKET, PREFIX, FakeS3(error="AccessDenied"))


def test_only_top_level_txt_files_count_as_the_corpus():
    s3 = FakeS3({"PMC1.txt": b"a", "notes.md": b"b", "sub/PMC2.txt": b"c"})
    assert bootstrap.list_remote_corpus(BUCKET, PREFIX, s3) == {"PMC1.txt": 1}


# -- mirror_corpus_from_bucket --------------------------------------------


def test_mirror_downloads_what_is_missing_and_what_differs(dirs):
    local, aside = dirs
    bucket = _papers(3)
    _write(local, {"PMC10000000.txt": bucket["PMC10000000.txt"], "PMC10000001.txt": b"old"})
    s3 = FakeS3(bucket)
    remote = bootstrap.list_remote_corpus(BUCKET, PREFIX, s3)

    downloaded, set_aside = bootstrap.mirror_corpus_from_bucket(
        BUCKET, PREFIX, remote, s3, local, aside, echo=_quiet
    )

    assert (downloaded, set_aside) == (2, 0)  # the size-differing one + the missing one
    assert sorted(p.name for p in local.iterdir()) == sorted(bucket)
    assert all((local / n).read_bytes() == b for n, b in bucket.items())


def test_mirror_sets_aside_local_extras_and_never_deletes_them(dirs):
    local, aside = dirs
    bucket, extras = _papers(2), _papers(2, start=50)
    _write(local, {**bucket, **extras})
    s3 = FakeS3(bucket)
    remote = bootstrap.list_remote_corpus(BUCKET, PREFIX, s3)

    _, set_aside = bootstrap.mirror_corpus_from_bucket(
        BUCKET, PREFIX, remote, s3, local, aside, echo=_quiet
    )

    assert set_aside == 2
    assert sorted(p.name for p in local.iterdir()) == sorted(bucket)
    assert all((aside / n).read_bytes() == b for n, b in extras.items())  # moved, intact
    assert s3.puts == []  # and never uploaded


def test_mirror_is_idempotent(dirs):
    local, aside = dirs
    s3 = FakeS3(_papers(3))
    remote = bootstrap.list_remote_corpus(BUCKET, PREFIX, s3)
    bootstrap.mirror_corpus_from_bucket(BUCKET, PREFIX, remote, s3, local, aside, echo=_quiet)
    s3.downloads.clear()

    again = bootstrap.mirror_corpus_from_bucket(
        BUCKET, PREFIX, remote, s3, local, aside, echo=_quiet
    )

    assert again == (0, 0)
    assert s3.downloads == []


# -- the decision ----------------------------------------------------------


def test_an_incomplete_bucket_is_left_for_this_computer_to_seed(dirs):
    local, aside = dirs
    _write(local, _papers(5, start=100))
    s3 = FakeS3(_papers(3))

    used = bootstrap.use_bucket_corpus_if_complete(
        BUCKET, PREFIX, 5, s3, echo=_quiet, local_dir=local, set_aside_dir=aside
    )

    assert used is False
    assert len(list(local.iterdir())) == 5  # local corpus untouched
    assert not aside.exists()


def test_a_complete_bucket_wins_over_a_different_complete_local_corpus(dirs):
    # Exactly the 2026-11-20 case: both sides "complete", different papers.
    local, aside = dirs
    shared, ours = _papers(5), _papers(5, start=3)  # 2 in common, 3 different each way
    _write(local, ours)
    s3 = FakeS3(shared)

    used = bootstrap.use_bucket_corpus_if_complete(
        BUCKET, PREFIX, 5, s3, echo=_quiet, local_dir=local, set_aside_dir=aside
    )

    assert used is True
    assert sorted(p.name for p in local.iterdir()) == sorted(shared)
    assert sorted(p.name for p in aside.iterdir()) == sorted(set(ours) - set(shared))


def test_sync_corpus_uploads_nothing_when_the_bucket_is_complete(dirs):
    local, aside = dirs
    _write(local, {**_papers(4), **_papers(3, start=70)})  # 3 papers the bucket lacks
    s3 = FakeS3(_papers(4))

    result = bootstrap.sync_corpus(
        BUCKET, PREFIX, 4, s3, local_dir=local, set_aside_dir=aside, echo=_quiet
    )

    assert result == "mirrored"
    assert s3.puts == []  # the extras did NOT grow the shared corpus
    assert len(list(aside.iterdir())) == 3


def test_sync_corpus_uploads_when_the_bucket_is_empty(dirs):
    # The first computer: ensure_bucket() has just created an empty bucket.
    local, aside = dirs
    _write(local, _papers(3))
    s3 = FakeS3()

    result = bootstrap.sync_corpus(
        BUCKET, PREFIX, 3, s3, local_dir=local, set_aside_dir=aside, echo=_quiet
    )

    assert result == "uploaded"
    assert sorted(s3.puts) == sorted(f"{PREFIX}{n}" for n in _papers(3))
