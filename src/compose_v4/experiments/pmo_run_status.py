"""Admissibility of recorded PMO task readings.

A completed run is not automatically a valid measurement.  The 250-call PMO gsk3b
task under run ``79fec498...`` charged its whole budget, wrote a well-formed ledger
and reported ``best_score 0.0`` -- and every one of those 250 zeros came from a
``FileNotFoundError`` that PyTDC swallowed into ``default_property``, not from the
chemistry (``diagnostics/pmo_gsk3b_oracle_diagnosis_v1.json``).  Nothing in the
artifact says so, so a summary that reads ``result.json`` and believes it would
publish a runtime defect as a performance number.

This module is the single place that answers "may this reading be reported?".  It is
deliberately per-TASK, not per-run: in that same run the two RDKit-only tasks
(perindopril_mpo, celecoxib_rediscovery) load no asset, were measured unaffected, and
remain valid.  Voiding the whole run would discard two good measurements; voiding
nothing would keep a bad one.

The ledger is fail-closed in the direction that matters: an unknown run/task pair is
ADMISSIBLE (the ledger records exceptions, not permissions), but a recorded exclusion
raises rather than warns.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

LEDGER = "diagnostics/pmo_run_status_ledger.json"

VALID = "VALID"
INVALID_ORACLE_RUNTIME_RELATIVE_PATH = "INVALID_ORACLE_RUNTIME_RELATIVE_PATH"


class PmoReadingExcluded(RuntimeError):
    """A PMO reading that the status ledger excludes from performance summaries."""


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


@lru_cache(maxsize=8)
def _load(root: str) -> dict:
    path = Path(root) / LEDGER
    if not path.exists():
        return {"entries": []}
    return json.loads(path.read_text())


def load_ledger(root: Path | None = None) -> dict:
    return _load(str(root or _repo_root()))


def entry_for(run_id: str, task: str, *, root: Path | None = None) -> dict | None:
    """The ledger entry for one (run, task), or None when none is recorded."""
    for entry in load_ledger(root).get("entries", []):
        if entry["run_id"] == run_id and entry["task"] == task:
            return entry
    return None


def status_of(run_id: str, task: str, *, root: Path | None = None) -> str:
    entry = entry_for(run_id, task, root=root)
    return VALID if entry is None else entry["status"]


def is_excluded(run_id: str, task: str, *, root: Path | None = None) -> bool:
    """Whether this reading must be left out of every PMO performance summary."""
    entry = entry_for(run_id, task, root=root)
    return bool(entry and entry.get("exclude_from_performance_summaries"))


def assert_admissible(run_id: str, task: str, *, root: Path | None = None) -> None:
    """Raise if this reading may not be reported as performance.

    Call this wherever a `result.json` becomes a published number.  Raising is the
    point: a warning next to a plausible score is read as a caveat, and the score is
    quoted anyway.
    """
    entry = entry_for(run_id, task, root=root)
    if entry and entry.get("exclude_from_performance_summaries"):
        raise PmoReadingExcluded(
            f"{task} @ {run_id[:12]}... is {entry['status']}: {entry['reason']}"
        )


def admissible_results(results: dict[str, dict], run_id: str, *,
                       root: Path | None = None) -> dict[str, dict]:
    """Filter a {task: result} mapping down to the readings that may be reported."""
    return {
        task: result for task, result in results.items()
        if not is_excluded(run_id, task, root=root)
    }
