"""
cost.py  --  the cost meter for the demo.

Pure Python logic: real token counts and durations in, dollars out.
No AWS, no network I/O.  This means the entire cost accounting layer
is trivially testable without any cloud credentials (see tests/test_cost.py).

What gets measured vs. what gets computed:

  MEASURED (included in the run receipt / live total):
    LLM calls:        actual inputTokens + outputTokens from converse()
                      × per-million-token rates from config.PRICING.
    Code Interpreter: wall-clock seconds from code_interpreter_run()
                      × config.CODE_INTERPRETER_PER_SECOND.
    KB retrieval:     number of API calls × live rate from the Price List
                      (fetched once at startup by pricing.fetch_rates()).

  COMPUTED (shown in the sidebar KB panel, NOT in the receipt total):
    Ingestion:        corpus char count / 4 × embed rate.
                      (Bedrock ingestion API reports document counts only,
                       not token counts -- so we approximate from corpus size.)
    Vector storage:   vector count × per-vector bytes × storage rate.
                      (S3 Vectors GetVectorBucket has no sizeBytes field.)
    Corpus storage:   corpus dir size × S3 standard rate.
    These three are returned by backend.kb_setup_costs() and emitted as a
    separate setup_cost event.  They are NOT added to the run total.

Thread safety:
  Q3 runs Claude Opus and OpenAI GPT-6 Astra in parallel threads.  Both call
  meter.add_llm() concurrently.  The CostMeter uses a threading.Lock around
  the rows list to prevent data races.  The lock is intentionally not a
  dataclass field that shows up in repr/eq (init=False, repr=False,
  compare=False) so the meter can be compared in tests.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field


@dataclass
class CostRow:
    """One line on the receipt.

    Each method call (model invocation, Code Interpreter run, KB retrieval)
    produces one row.  The receipt event is a list of these rows plus the total.
    """

    step: str  # e.g. "Q1  synthesis", "Q2  analysis run"
    label: str  # human-readable model/service name for the UI
    in_tokens: int  # input tokens (0 for compute and retrieval rows)
    out_tokens: int  # output tokens (0 for compute and retrieval rows)
    usd: float  # cost in US dollars


@dataclass
class CostMeter:
    """Accumulates per-step cost across one demo run.

    Constructed fresh for each /ws WebSocket connection so every run
    starts with a clean receipt.  The pricing dict comes from config.PRICING;
    ci_per_second from config.CODE_INTERPRETER_PER_SECOND.
    """

    # tier -> (usd_per_1M_in, usd_per_1M_out)
    # Example: {"haiku": (1.00, 5.00), "sonnet": (3.00, 15.00), ...}
    pricing: dict[str, tuple[float, float]]

    # AgentCore Code Interpreter rate in USD per wall-clock second.
    # See config.py for the derivation from vCPU-hr + GB-hr pricing.
    ci_per_second: float = 0.0

    # KB retrieval cost in USD per 1,000 API calls.
    # Fetched from the AWS Price List at startup; 0.0 if unavailable.
    kb_query_usd_per_1k: float = 0.0

    # Accumulated rows -- one per model call, compute run, or retrieval.
    rows: list[CostRow] = field(default_factory=list)

    # Internal lock for thread-safe appends.
    # init=False: not a constructor parameter.
    # repr=False, compare=False: excluded from __repr__ and __eq__
    #   so test assertions on CostMeter values are not broken by the lock object.
    _lock: threading.Lock = field(
        default_factory=threading.Lock, init=False, repr=False, compare=False
    )

    def add_llm(self, step: str, tier: str, label: str, usage: dict) -> float:
        """Record one Bedrock model invocation and return its cost in USD.

        Computes cost as:
            (inputTokens / 1,000,000) × per_in_rate
          + (outputTokens / 1,000,000) × per_out_rate

        Thread-safe: safe to call from multiple threads concurrently (Q3
        calls this from both the Opus thread and the Astra thread).

        Args:
            step: receipt label for this step (e.g. "Q1  synthesis").
            tier: model tier key matching a key in self.pricing.
            label: human-readable model name for the UI (e.g. "Claude Haiku").
            usage: the usage dict from converse() with "inputTokens" / "outputTokens".

        Returns:
            The cost of this call in USD.
        """
        per_in, per_out = self.pricing[tier]
        i = int(usage.get("inputTokens", 0))
        o = int(usage.get("outputTokens", 0))
        usd = (i / 1_000_000) * per_in + (o / 1_000_000) * per_out
        with self._lock:
            self.rows.append(CostRow(step, label, i, o, usd))
        return usd

    def add_compute(self, step: str, seconds: float) -> float:
        """Record one Code Interpreter session by wall-clock duration.

        Cost = seconds × ci_per_second (see config.py for the derivation).
        in_tokens and out_tokens are 0 for compute rows.

        Args:
            step: receipt label (e.g. "Q2  analysis run").
            seconds: wall-clock duration returned by code_interpreter_run().

        Returns:
            The cost of this session in USD.
        """
        usd = seconds * self.ci_per_second
        with self._lock:
            self.rows.append(CostRow(step, "Code Interpreter", 0, 0, usd))
        return usd

    def add_retrieval(self, step: str, n_queries: int = 1) -> float:
        """Record KB retrieval API calls.

        Cost = (n_queries / 1,000) × kb_query_usd_per_1k.
        For a single retrieve() call, n_queries=1 and the cost is a
        fraction of a cent (at $0.40/1K, one call costs $0.0004).

        Args:
            step: receipt label (e.g. "Q1  retrieval").
            n_queries: number of retrieve() calls to record (usually 1).

        Returns:
            The cost of these retrieval calls in USD.
        """
        usd = (n_queries / 1_000) * self.kb_query_usd_per_1k
        with self._lock:
            self.rows.append(CostRow(step, "KB retrieval", 0, 0, usd))
        return usd

    @property
    def total(self) -> float:
        """Running total of all recorded costs, in USD."""
        return sum(r.usd for r in self.rows)

    def receipt(self) -> dict:
        """Return a JSON-serialisable receipt for the UI and the terminal runner.

        Used by the "receipt" event at the end of every run.  The UI renders
        this as the final itemised cost table.

        Returns:
            A dict with "rows" (list of row dicts) and "total" (float, USD).
        """
        return {
            "rows": [
                {
                    "step": r.step,
                    "label": r.label,
                    "in_tokens": r.in_tokens,
                    "out_tokens": r.out_tokens,
                    "usd": round(r.usd, 6),
                }
                for r in self.rows
            ],
            "total": round(self.total, 6),
        }
