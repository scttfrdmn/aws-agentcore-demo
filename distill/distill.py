#!/usr/bin/env python3
"""
distill.py  --  distil an open-weight teacher into a small open-weight student,
                on this Mac (Apple silicon, MLX).  Offline; not part of `make start`.

    uv run python distill.py chunks              # corpus/*.txt -> data/chunks.jsonl
    uv run python distill.py questions --n 1200  # teacher writes researcher questions
    uv run python distill.py answers             # teacher answers them, demo-style
    uv run python distill.py split               # -> data/train|valid|test.jsonl
    (train with mlx_lm lora -- see README.md)
    uv run python distill.py compare ...         # base vs distilled vs teacher

Teacher and student are BOTH open-weight, deliberately (owner's decision,
2026-10-13).  Using Claude or GPT outputs to train another model runs into those
vendors' terms; an Apache-2.0 teacher run on our own hardware does not, and it
also keeps every byte of the corpus on this machine.

What the student learns is exactly beat 1's job: given retrieved passages in the
format agent.py builds (`_context()`), answer in <=150 words citing bare PMC IDs.
The system prompt is imported from questions.py, not copied, so the student is
trained on the instructions it will receive on stage.

The demo's own locked questions are NEVER trained on -- generated questions too
close to them are dropped (see _too_close_to_locked) -- so `compare` on them is
an honest out-of-sample test.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "src"))
from agentcore_demo import questions as Q  # noqa: E402

CORPUS = ROOT / "corpus"
DATA = HERE / "data"
# The teacher was already on this Mac, in Ollama: Qwen3.6 35B (MoE), Q4_K_M,
# Apache 2.0 -- no clause restricting use of its outputs to train other models.
TEACHER = "qwen3.6:35b"
OLLAMA = "http://127.0.0.1:11434/api/chat"
# Mistral architecture on purpose: Bedrock Custom Model Import keeps Converse
# working for Mistral/Llama imports, but not for Qwen3 or gpt-oss.  4-bit because
# this Mac's link runs at ~1 MB/s (measured 2026-10-13); bf16 would be 14.5 GB.
STUDENT = HERE / "models" / "student-mistral-7b-v0.3-4bit"

# Mirror build_kb.py's KB chunking: FIXED_SIZE, 512 tokens, 15% overlap.
# Approximated in words (~0.75 words/token) -- close enough for retrieval shape.
CHUNK_WORDS = 384
OVERLAP_WORDS = 58

PMC_RE = re.compile(r"PMC\d{5,9}")

QUESTION_SYSTEM = (
    "You write questions that a cardiovascular or lipid researcher would ask a "
    "search assistant over a library of PCSK9 papers. Given one passage, write ONE "
    "standalone question (at most 25 words) that the passage helps answer. Do not "
    "mention 'the passage', 'this study' or 'the authors'. Vary the kind of question: "
    "mechanism, clinical effect, genetics, safety, comparison, methods. "
    "Output only the question."
)


# -- helpers ----------------------------------------------------------------


def read_jsonl(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def context(chunks: list[dict]) -> str:
    """Byte-for-byte the format of agent.Agent._context()."""
    return "\n\n".join(f"[source: {c['source']}]\n{c['text']}" for c in chunks)


def user_prompt(question: str, chunks: list[dict]) -> str:
    """Byte-for-byte the user turn agent.question_1() sends."""
    return f"Passages:\n{context(chunks)}\n\n{question}"


def _too_close_to_locked(q: str) -> bool:
    """True if `q` overlaps a locked demo question enough to contaminate the test."""
    qs = set(_tokens(q))
    for locked in Q.QUESTIONS:
        ls = set(_tokens(locked))
        if qs and len(qs & ls) / len(qs | ls) > 0.5:
            return True
    return False


def _looks_like_references(text: str) -> bool:
    """Reference lists and tables make poor seeds: mostly digits and 'et al'."""
    digits = sum(ch.isdigit() for ch in text) / max(len(text), 1)
    return digits > 0.12 or text.count("et al") > 4 or text.count("doi") > 3


# The PMC search returns papers that mention PCSK9 only in passing (oncology,
# CRISPR...).  Seeding questions only from on-topic passages keeps the student a
# PCSK9 specialist; retrieval still searches every chunk, as the KB does.
ON_TOPIC = re.compile(r"pcsk9|ldl|cholesterol|lipoprotein|statin|lipid|inclisiran|evolocumab")


def _on_topic(text: str) -> bool:
    return len(ON_TOPIC.findall(text.lower())) >= 2


def student_messages(messages: list[dict]) -> list[dict]:
    """Fold the system prompt into the user turn, Mistral-style.

    Mistral-7B-Instruct-v0.3's chat template has no system role -- it raises
    "Only user and assistant roles are supported!".  Mistral's own convention
    (mistral-common) prepends the system prompt to the user message with a
    blank line, so that is what the student is trained and tested on.
    """
    system = next(m["content"] for m in messages if m["role"] == "system")
    rest = [m for m in messages if m["role"] != "system"]
    return [{"role": "user", "content": f"{system}\n\n{rest[0]['content']}"}, *rest[1:]]


# -- stages -----------------------------------------------------------------


def cmd_chunks(_args) -> None:
    rows = []
    files = sorted(CORPUS.glob("PMC*.txt"))
    if not files:
        sys.exit(f"no corpus at {CORPUS} -- run `uv run python corpus_fetch.py` in the repo root")
    for path in files:
        lines = path.read_text(encoding="utf-8").splitlines()
        body = " ".join(lines[2:] if lines and lines[0].startswith("#") else lines)
        words = body.split()
        step = CHUNK_WORDS - OVERLAP_WORDS
        for i in range(0, max(len(words) - OVERLAP_WORDS, 1), step):
            text = " ".join(words[i : i + CHUNK_WORDS])
            if len(text) > 200:
                rows.append({"id": f"{path.stem}-{i // step}", "source": path.stem, "text": text})
    write_jsonl(DATA / "chunks.jsonl", rows)
    print(f"{len(rows):,} chunks from {len(files):,} papers -> {DATA / 'chunks.jsonl'}")


def _teacher(messages: list[dict], max_tokens: int, temp: float) -> str:
    """One teacher call through Ollama's chat API, thinking OFF (as the demo does)."""
    import urllib.request

    body = {
        "model": TEACHER,
        "think": False,
        "stream": False,
        "options": {"temperature": temp, "num_ctx": 16384, "num_predict": max_tokens},
        "messages": messages,
    }
    req = urllib.request.Request(
        OLLAMA, json.dumps(body).encode(), {"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.load(r)["message"]["content"].strip()


def _batched(convs: list[list[dict]], max_tokens: int, temp: float, workers: int):
    """Run many teacher calls concurrently; yields (index, text) as they finish."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    t0 = time.time()
    with ThreadPoolExecutor(workers) as ex:
        futs = {ex.submit(_teacher, c, max_tokens, temp): i for i, c in enumerate(convs)}
        for done, fut in enumerate(as_completed(futs), 1):
            try:
                text = fut.result()
            except Exception as e:  # noqa: BLE001 -- one failed call must not stop a 1,000-call run
                print(f"  teacher call failed: {type(e).__name__}: {e}", flush=True)
                continue
            if done % 25 == 0 or done == len(convs):
                rate = done / (time.time() - t0)
                print(f"  {done:>5}/{len(convs)}  {rate:.2f}/s", flush=True)
            yield futs[fut], text


def cmd_questions(args) -> None:
    chunks = [
        c
        for c in read_jsonl(DATA / "chunks.jsonl")
        if _on_topic(c["text"]) and not _looks_like_references(c["text"])
    ]
    rng = random.Random(args.seed)
    by_paper: dict[str, list[dict]] = {}
    for c in chunks:
        by_paper.setdefault(c["source"], []).append(c)
    # Spread seeds across papers first, so no single review dominates.
    seeds = []
    papers = list(by_paper)
    while len(seeds) < args.n * 1.3 and papers:
        rng.shuffle(papers)
        for p in list(papers):
            if by_paper[p]:
                seeds.append(by_paper[p].pop(rng.randrange(len(by_paper[p]))))
            else:
                papers.remove(p)
    seeds = seeds[: int(args.n * 1.3)]
    convs = [
        [{"role": "system", "content": QUESTION_SYSTEM}, {"role": "user", "content": s["text"]}]
        for s in seeds
    ]
    out, seen = [], set()
    for i, q in _batched(convs, max_tokens=60, temp=0.8, workers=args.workers):
        q = q.splitlines()[0].strip().strip('"')
        key = " ".join(_tokens(q))
        if not q.endswith("?") or key in seen or _too_close_to_locked(q) or len(q.split()) > 30:
            continue
        seen.add(key)
        out.append({"question": q, "seed": seeds[i]["id"]})
        if len(out) >= args.n:
            break
    write_jsonl(DATA / "questions.jsonl", out)
    print(f"{len(out)} questions -> {DATA / 'questions.jsonl'}")


def _retriever(chunks: list[dict]):
    from rank_bm25 import BM25Okapi

    bm25 = BM25Okapi([_tokens(c["text"]) for c in chunks])

    def top_k(query: str, k: int) -> list[dict]:
        scores = bm25.get_scores(_tokens(query))
        best = sorted(range(len(chunks)), key=scores.__getitem__, reverse=True)[:k]
        return [chunks[i] for i in best]

    return top_k


def answer_problems(answer: str, chunks: list[dict]) -> list[str]:
    """Objective checks against beat 1's instructions.  Empty list == compliant."""
    return problems_for(answer, {c["source"] for c in chunks})


def problems_for(answer: str, allowed: set[str]) -> list[str]:
    """The checks themselves, given the set of PMC IDs that were in the passages.

    Deliberately mechanical -- no LLM judge.  Every number `compare` reports is
    one a researcher can re-check by eye: did it cite, did every citation point
    at a passage it was actually given, no URLs, within the length limit.
    """
    cited = set(PMC_RE.findall(answer))
    problems = []
    if not cited:
        problems.append("no citation")
    if cited - allowed:
        problems.append(f"cites {len(cited - allowed)} ID(s) not in the passages")
    if re.search(r"https?://|www\.", answer):
        problems.append("contains a URL")
    if len(answer.split()) > 170:
        problems.append(f"{len(answer.split())} words (limit 150)")
    return problems


def cmd_answers(args) -> None:
    chunks = read_jsonl(DATA / "chunks.jsonl")
    top_k = _retriever(chunks)
    qs = read_jsonl(DATA / "questions.jsonl")[: args.limit or None]
    t = time.time()
    retrieved = [top_k(q["question"], args.k) for q in qs]
    print(f"retrieved {args.k} passages for {len(qs)} questions in {time.time() - t:.0f}s")
    convs = [
        [
            {"role": "system", "content": Q.PLAIN_SYSTEM},
            {"role": "user", "content": user_prompt(q["question"], r)},
        ]
        for q, r in zip(qs, retrieved, strict=True)
    ]
    rows, rejected = [], 0
    for i, ans in _batched(convs, max_tokens=320, temp=0.3, workers=args.workers):
        problems = answer_problems(ans, retrieved[i])
        if problems:
            rejected += 1
            continue
        messages = [*convs[i], {"role": "assistant", "content": ans}]
        rows.append({"question": qs[i]["question"], "messages": messages})
    write_jsonl(DATA / "distilled.jsonl", rows)
    print(f"{len(rows)} kept, {rejected} rejected by answer_problems -> {DATA / 'distilled.jsonl'}")


def cmd_split(args) -> None:
    rows = read_jsonl(DATA / "distilled.jsonl")
    random.Random(args.seed).shuffle(rows)
    n_test, n_valid = args.test, args.valid
    test, valid, train = rows[:n_test], rows[n_test : n_test + n_valid], rows[n_test + n_valid :]
    for name, part in [("train", train), ("valid", valid), ("test", test)]:
        write_jsonl(
            DATA / f"{name}.jsonl", [{"messages": student_messages(r["messages"])} for r in part]
        )
    print(f"train {len(train)} / valid {len(valid)} / test {len(test)} -> {DATA}")


SOURCE_RE = re.compile(r"\[source: (PMC\d+)\]")


def _compare_cases(args) -> list[dict]:
    """Held-out test rows plus the demo's own locked Q1 and Q2 (never trained on)."""
    cases = []
    for r in read_jsonl(DATA / "test.jsonl")[: args.n]:
        folded = r["messages"][0]["content"]
        user = folded[len(Q.PLAIN_SYSTEM) + 2 :]
        cases.append({"label": "held-out", "question": user.rsplit("\n\n", 1)[-1], "user": user})
    top_k = _retriever(read_jsonl(DATA / "chunks.jsonl"))
    for i in (0, 1):
        q = Q.QUESTIONS[i]
        cases.append(
            {"label": f"locked Q{i + 1}", "question": q, "user": user_prompt(q, top_k(q, 12))}
        )
    for c in cases:
        c["allowed"] = set(SOURCE_RE.findall(c["user"]))
    return cases


def _run_student(cases: list[dict], adapter: str | None) -> list[dict]:
    from mlx_lm import generate, load
    from mlx_lm.sample_utils import make_sampler

    model, tok = load(str(STUDENT), adapter_path=adapter)
    out = []
    for c in cases:
        conv = student_messages(
            [{"role": "system", "content": Q.PLAIN_SYSTEM}, {"role": "user", "content": c["user"]}]
        )
        prompt = tok.apply_chat_template(conv, add_generation_prompt=True)
        t = time.time()
        text = generate(model, tok, prompt, max_tokens=320, sampler=make_sampler(temp=0.0))
        out.append({"text": text.strip(), "seconds": time.time() - t})
    return out


def _run_teacher(cases: list[dict]) -> list[dict]:
    out = []
    for c in cases:
        conv = [
            {"role": "system", "content": Q.PLAIN_SYSTEM},
            {"role": "user", "content": c["user"]},
        ]
        t = time.time()
        text = _teacher(conv, max_tokens=320, temp=0.0)
        out.append({"text": text, "seconds": time.time() - t})
    return out


def cmd_compare(args) -> None:
    cases = _compare_cases(args)
    runs = {
        "base student": _run_student(cases, None),
        "distilled student": _run_student(cases, args.adapter),
        f"teacher ({TEACHER})": _run_teacher(cases),
    }
    lines = [f"# Distillation comparison -- {len(cases)} cases\n"]
    lines.append("| model | compliant | mean words | mean seconds |")
    lines.append("|---|---|---|---|")
    for name, res in runs.items():
        ok = sum(not problems_for(r["text"], c["allowed"]) for r, c in zip(res, cases, strict=True))
        words = sum(len(r["text"].split()) for r in res) / len(res)
        secs = sum(r["seconds"] for r in res) / len(res)
        lines.append(f"| {name} | {ok}/{len(cases)} | {words:.0f} | {secs:.1f} |")
    for i, c in enumerate(cases):
        if not c["label"].startswith("locked"):
            continue
        lines.append(f"\n## {c['label']}: {c['question']}\n")
        for name, res in runs.items():
            probs = problems_for(res[i]["text"], c["allowed"]) or ["compliant"]
            lines.append(f"**{name}** ({res[i]['seconds']:.1f}s; {', '.join(probs)})\n")
            lines.append(res[i]["text"] + "\n")
    out = HERE / "runs" / "compare.md"
    out.parent.mkdir(exist_ok=True)
    out.write_text("\n".join(lines))
    dump = [
        {"case": {k: v for k, v in c.items() if k != "allowed"}}
        | {name: res[i] for name, res in runs.items()}
        for i, c in enumerate(cases)
    ]
    (HERE / "runs" / "compare.json").write_text(json.dumps(dump, indent=1))
    print("\n".join(lines[:6]))
    print(f"\nfull report -> {out}")


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("chunks")
    s = sub.add_parser("questions")
    s.add_argument("--n", type=int, default=1200)
    s.add_argument("--workers", type=int, default=4)
    s.add_argument("--seed", type=int, default=13)
    s = sub.add_parser("answers")
    s.add_argument("--k", type=int, default=12, help="passages per question (the demo uses 12)")
    s.add_argument("--limit", type=int, default=0)
    s.add_argument("--workers", type=int, default=4)
    s = sub.add_parser("split")
    s.add_argument("--test", type=int, default=40)
    s.add_argument("--valid", type=int, default=40)
    s.add_argument("--seed", type=int, default=13)
    s = sub.add_parser("compare")
    s.add_argument("--adapter", default=str(HERE / "adapters"))
    s.add_argument("--n", type=int, default=40, help="held-out cases (plus locked Q1, Q2)")
    args = p.parse_args()
    stages = {
        "chunks": cmd_chunks,
        "questions": cmd_questions,
        "answers": cmd_answers,
        "split": cmd_split,
        "compare": cmd_compare,
    }
    stages[args.cmd](args)


if __name__ == "__main__":
    main()
