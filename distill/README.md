# distill/ — a PCSK9 specialist, distilled on a laptop

Offline, optional, and **not** part of `make start`. This turns an open-weight
**teacher** into a small open-weight **student** that does beat 1's job: given
the passages retrieved from the knowledge base, answer in at most 150 words and
cite bare PMC IDs. Everything runs on one Mac. The corpus, the prompts and the
training data never leave it.

| role | model | licence | where it came from |
|---|---|---|---|
| teacher | Qwen3.6 35B (MoE), Q4_K_M | Apache 2.0 | already in Ollama on this Mac |
| student | Mistral-7B-Instruct-v0.3, 4-bit MLX | Apache 2.0 | `mlx-community/Mistral-7B-Instruct-v0.3-4bit` |

**Why both are open-weight.** It was the owner's decision. Training on Claude or GPT
outputs runs into those vendors' terms. AWS Service Terms §50.5 also makes
generating training data *on Bedrock* ambiguous. A teacher with an Apache 2.0 licence,
run on our own hardware, avoids both problems.

**Why a Mistral student.** Bedrock Custom Model Import keeps the Converse API
(and so the demo's code path) for Mistral and Llama architectures. It does not
for Qwen3 or gpt-oss. It is 4-bit because this Mac's link measured about 1 MB/s,
and bf16 weights are 14.5 GB.

## Run it

Use Python 3.11+ with `uv` and Ollama with `qwen3.6:35b` pulled. The corpus must
be in `../corpus/`; `uv run python corpus_fetch.py` from the repo root fetches it
(free, no AWS).

```bash
cd distill
uv sync
uv run python distill.py chunks              # corpus -> 512-token-ish chunks, as the KB does
uv run python distill.py questions --n 750   # teacher writes researcher questions
uv run python distill.py answers --k 6       # teacher answers them from 6 retrieved passages
uv run python distill.py split               # train / valid / test (+ Mistral formatting)
ollama stop qwen3.6:35b                      # free ~24 GB before training
uv run python -m mlx_lm lora -c lora.yaml    # LoRA; see lora.yaml for why each value
uv run python distill.py compare             # base vs distilled vs teacher
```

## What `compare` measures, and what it does not

The checks are mechanical and use no LLM judge. They test beat 1's own instructions:

- it cites at least one source;
- every PMC ID it cites is in the passages it was given;
- it contains no URLs;
- it stays within the length limit.

The test set is held-out generated questions, **plus the demo's locked Q1 and
Q2**. Generated questions that overlap a locked question too closely are
dropped, so Q1 and Q2 are never trained on. It shows whether the student learned
the job. It does not show whether every sentence is medically right; read the
side-by-side output in `runs/compare.md` for that.

## First run: 2026-10-13/14, M4 Max, 64 GB

**Data.** The teacher wrote 750 questions from on-topic passages. It answered
them from 6 retrieved passages each, and 748 answers passed `answer_problems`.
That split into 668 train, 40 valid and 40 test.

**Training.** 500 LoRA steps took 3.2 hours. Validation loss went
0.79 → 0.52 (100) → 0.54 → 0.50 → **0.41 (400)** → 0.49 (500). The
validation set has only 40 examples, so the curve is noisy. The step-400
checkpoint was kept.

**Results.** `compare` ran on 20 held-out questions plus locked Q1 and Q2.
Held-out questions had 6 passages; the locked ones had 12, as the demo sends.

| model | compliant | mean words | mean s (this Mac) |
|---|---|---|---|
| base student | 17/22 | 102 | 11.6 |
| **distilled student** | **22/22** | **69** | 12.8 |
| teacher (qwen3.6:35b) | 22/22 | 86 | 7.6 |

What these results do and do not show:

- **The distilled student learned the job.** It now follows beat 1's rules every
  time, as the teacher does. The base student broke them on 5 of 22: no
  citation, or over the length limit.
- **Its answers are shorter and cite fewer sources than the teacher's.** The
  teacher's best answers draw on several papers, while the student's tend to
  lean on one.
- **On locked Q1, both the student and the teacher answered only about Lp(a).**
  The local BM25 stand-in for the KB retrieved Lp(a) passages for that
  question. This is a retrieval artefact, not a model one, but Q1 is not a fair
  showcase from this run.
- **Smaller is not faster here.** The MoE teacher activates about 3B parameters
  per token and beats the dense 7B student on wall-clock.
- **Answers can be cut off mid-sentence.** One spot check ran into the 320-token
  `max_tokens` cap and was truncated, yet stayed under the 170-word check.
  `problems_for` does not detect truncation; that is a known gap.

**Output.** `fused/itl-pcsk9-mistral-7b/` holds the merged model:
`MistralForCausalLM`, bf16, 14.5 GB, Hugging Face safetensors plus the tokenizer.
It was produced by `mlx_lm fuse --dequantize` from the 4-bit base. It is the
input Custom Model Import expects, but it has **not been imported**: the
upload alone is hours on this link, and the import creates billable,
per-minute resources. Note that the base it was trained on was 4-bit, so
dequantising does not recover full bf16 quality.

## Dead ends, kept so nobody repeats them

These were measured on 2026-10-13 on an M4 Max with 64 GB.

- **12 passages per training example (what the demo sends).** This is about
  10k Mistral tokens. Truncation cut the answer off the end, and with
  `mask_prompt` the loss went NaN. Training therefore uses 6 passages; testing
  uses the demo's 12.
- **Learning rate 1e-4.** Diverged: validation loss went from 0.88 to 9.0 in 16
  steps. 2e-5 is stable.
- **No gradient checkpointing, with the teacher still loaded.** The run ran out of
  GPU memory.
- **Hugging Face's Xet downloader hung at 0 bytes.** Plain `curl -C -` on the
  `resolve/` URLs works.

## Next version of the talk

Distil a teacher too large for this Mac on SageMaker, and compare it against
this run.
