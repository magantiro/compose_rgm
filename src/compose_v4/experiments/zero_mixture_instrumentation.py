"""Runtime counters proving the RING_CORE_V1 zero-mixture (scaled-manifest) training path never touches any
de-novo resource, and the CPU dry-launch reaches the first editing + validation batch (owner mandate §3/§6).

The gate + builders increment these. Under ``--scaled-manifest`` every ``denovo_*`` counter must be exactly 0
and the production/edit counters positive -- proving the zero-mixture decision happened BEFORE any de-novo
compile / cache resolution / carbon-tree dataset, not merely by zeroing a sampling weight after the fact.
"""
from __future__ import annotations

from collections import Counter

DENOVO_COUNTERS = (
    "denovo_path_compile_calls",
    "carbon_tree_prior_constructions",
    "denovo_cache_resolution_calls",
    "denovo_cache_open_calls",
    "denovo_support_cache_resolution_calls",
    "denovo_support_cache_open_calls",
    "denovo_dataset_constructions",
    "denovo_dataloader_constructions",
    "denovo_training_records_sampled",
    "denovo_validation_records_sampled",
)
# (counter name, minimum required value) for the positive side of the invariant.
REQUIRED_POSITIVE = (
    ("production_catalog_constructions", 1),
    ("production_catalog_fingerprint_checks", 1),
    ("edit_manifest_loads", 1),
    ("edit_pool_open_calls", 1),
    ("edit_dataset_constructions", 1),
    ("edit_dataloader_constructions", 1),
    ("edit_training_batches_emitted", 1),
    ("edit_validation_batches_emitted", 1),
    ("model_forward_calls", 2),
    ("gm_loss_calls", 2),
)
REQUIRED_ZERO = (*DENOVO_COUNTERS, "optimizer_steps", "backward_calls")
_ALL_KEYS = (
    *DENOVO_COUNTERS,
    *(k for k, _ in REQUIRED_POSITIVE),
    "optimizer_steps",
    "backward_calls",
    "optimizer_step_calls",
    "scheduler_step_calls",
)

_COUNTERS: Counter = Counter()


def reset() -> None:
    _COUNTERS.clear()


def bump(name: str, n: int = 1) -> None:
    _COUNTERS[name] += int(n)


def get(name: str) -> int:
    return int(_COUNTERS.get(name, 0))


def snapshot() -> dict[str, int]:
    return {k: int(_COUNTERS.get(k, 0)) for k in _ALL_KEYS}


def zero_mixture_ok() -> tuple[bool, dict]:
    """(ok, report). ok iff every de-novo/optimizer counter is 0 and every required counter meets its floor."""
    snap = snapshot()
    zero_ok = all(snap[k] == 0 for k in REQUIRED_ZERO)
    positive_ok = all(snap[k] >= floor for k, floor in REQUIRED_POSITIVE)
    failing_zero = {k: snap[k] for k in REQUIRED_ZERO if snap[k] != 0}
    failing_positive = {k: snap[k] for k, floor in REQUIRED_POSITIVE if snap[k] < floor}
    return zero_ok and positive_ok, {
        "counters": snap,
        "denovo_and_optimizer_all_zero": zero_ok,
        "required_positive_met": positive_ok,
        "failing_zero_counters": failing_zero,
        "failing_positive_counters": failing_positive,
    }
