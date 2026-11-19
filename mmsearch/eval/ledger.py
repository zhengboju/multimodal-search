"""Test-set ledger: every evaluation on a test split is appended here, so "test once" (plan §3.3) is auditable.

The ledger lives at storage/runs/test_ledger.jsonl. It never blocks — it warns when the same dataset has been
evaluated many times, which is the signal that hyper-parameters are being tuned on the test set.
"""

from __future__ import annotations

import time

from ..utils.misc import get_logger, read_jsonl, write_jsonl
from ..utils.paths import runs_dir
from ..utils.tracker import git_commit

log = get_logger(__name__)
WARN_AFTER = 10


def record_test_eval(dataset: str, method: str, metrics: dict) -> None:
    path = runs_dir() / "test_ledger.jsonl"
    prev = [r for r in read_jsonl(path) if r["dataset"] == dataset] if path.exists() else []
    if len(prev) >= WARN_AFTER:
        log.warning("dataset %s has been evaluated %d times — make sure you are not tuning on the test set", dataset, len(prev))
    write_jsonl([{"time": time.strftime("%Y-%m-%d %H:%M:%S"), "dataset": dataset, "method": method,
                  "commit": git_commit(), "metrics": metrics}], path, append=True)
