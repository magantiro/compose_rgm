"""Runtime counters proving the RING_CORE_V1 zero-mixture (scaled-manifest) training path never touches any
de-novo resource (§6 of the owner mandate).

The gate + the de-novo builders increment these counters. Under ``--scaled-manifest`` the required invariant
is that every ``denovo_*`` counter is exactly 0 and every ``edit_*`` counter is positive -- proving the
zero-mixture decision happened BEFORE any de-novo compile / cache resolution / carbon-tree dataset, not merely
by zeroing a sampling weight after the fact. The CPU dry-launch serializes these; the fixture regression tests
assert them (including a mutation test that removes the early branch and observes a de-novo counter fire).
"""
from __future__ import annotations

from collections import Counter

# Ordered required-zero and required-positive keys (see mandate §6).
DENOVO_COUNTERS = (
    "denovo_path_compile_calls",
    "carbon_tree_prior_constructions",
    "denovo_cache_resolution_calls",
    "denovo_cache_open_calls",
    "denovo_dataset_constructions",
    "denovo_dataloader_constructions",
    "denovo_training_records_sampled",
)
EDIT_COUNTERS = (
    "edit_dataset_constructions",
    "edit_batches_emitted",
    "validation_edit_batches_emitted",
)

_COUNTERS: Counter = Counter()


def reset() -> None:
    _COUNTERS.clear()


def bump(name: str, n: int = 1) -> None:
    _COUNTERS[name] += int(n)


def get(name: str) -> int:
    return int(_COUNTERS.get(name, 0))


def snapshot() -> dict[str, int]:
    return {k: int(_COUNTERS.get(k, 0)) for k in (*DENOVO_COUNTERS, *EDIT_COUNTERS)}


def zero_mixture_ok() -> tuple[bool, dict]:
    """(ok, report). ok iff every de-novo counter is 0 and edit dataset+batches are positive."""
    snap = snapshot()
    denovo_clean = all(snap[k] == 0 for k in DENOVO_COUNTERS)
    edit_present = (
        snap["edit_dataset_constructions"] > 0
        and snap["edit_batches_emitted"] >= 1
        and snap["validation_edit_batches_emitted"] >= 1
    )
    return denovo_clean and edit_present, {
        "counters": snap,
        "denovo_all_zero": denovo_clean,
        "edit_path_reached": edit_present,
    }
