"""Acceptance measurement for the de-novo ring dependency-block schedule.

The de-novo base overproduces 3- and 4-membered rings by roughly an order of
magnitude against its own corpus.  The defect is NOT the reward and NOT the
learned policy: at the states where ring events fire, uniform mass over the
exact executable ring-template support is far above the catalog's unconditional
small-ring mass, and the policy already sits below that uniform baseline.  The
support the compiler presents is wrong before the policy decides.

This script measures that support under two compiled schedules on real training
molecules, trains nothing and calls no oracle:

``sequential``
    What Lineage B trained on.  Every ring transaction is deferred behind the
    whole graft phase AND the whole non-ring decoration phase.

``ring_dependency_block``
    Each ring system is committed at the earliest graft prefix the executor
    accepts it at, so the ring decision is supervised on the large, undecorated
    carbon skeleton.

Reporting rules this script enforces rather than leaves to the caller:

* The sample is drawn UNIFORMLY AT RANDOM from the recipe's own train
  partition.  Slice iteration order in this repository is lane- and
  family-sorted, so a leading-N slice is never a sample; the realized strata
  are reported so the draw can be audited rather than trusted.
* Endpoint exactness is checked per trace, by canonical key AND by array
  identity, and reported as N of N.  A schedule that changes an endpoint is a
  different corpus, not a repair.
* The FULL per-minimum-ring-size support histogram is reported beside the
  3/4-ring fraction.  Two supports with the same fraction can differ entirely
  in what else they offer, and only the histogram separates "deleted small
  templates" from "restored large ones".
* Molecules carrying no ring system are counted and excluded from the support
  statistics, because the repair has nothing to do on them.
* Cost is reported as a load-independent counter (support evaluations,
  molecules compiled) beside wall clock, because other jobs share this machine.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# The recipe Lineage B trained under (recipes/tree_fcd_transfer_stage3_flexible_graft.json).
MAX_ATOMS = 40
TRAIN_SIZE = 50_000
VALIDATION_SIZE = 2_000
TEST_SIZE = 2_000
CORPUS_SEED = 20260717
SMALL_RING_MAX = 4

CHECKPOINT_SHA256 = "c9d927510360ec6eb84ff8dae1a222b0b693a9bef0ca23bb5d9cca063025876c"
TRAIN_SMILES_SHA256 = "70526d92f1f08d8e292cb31218f81b6924a2182f772c43348015110669d47791"

ARMS = ("sequential", "exact_early_ring", "ring_dependency_block")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _assert_pinned(checkpoint: Path, corpus: Path) -> dict[str, str]:
    """Refuse to measure against artifacts the model was not trained with."""

    observed = {"checkpoint": _sha256(checkpoint), "train_smiles": _sha256(corpus)}
    expected = {"checkpoint": CHECKPOINT_SHA256, "train_smiles": TRAIN_SMILES_SHA256}
    if observed != expected:
        raise RuntimeError(f"pinned input mismatch: {observed} != {expected}")
    return observed


def load_train_partition(corpus: Path, cache_dir: Path) -> tuple[str, ...]:
    """Return the recipe's train split, cached under a key naming its RULE.

    The cache key carries the corpus digest AND every split parameter, so a
    cache built under different sizes or a different seed can never be served
    for this one -- the derived-artifact-cache failure this repository has
    already paid for once.
    """

    from compose_v4.data.cnof import load_cnof_corpus_split

    key = (
        f"{TRAIN_SMILES_SHA256}_{TRAIN_SIZE}_{VALIDATION_SIZE}_{TEST_SIZE}"
        f"_{MAX_ATOMS}_{CORPUS_SEED}"
    )
    cache_path = cache_dir / f"train_split_{key}.json"
    if cache_path.exists():
        return tuple(json.loads(cache_path.read_text())["train"])
    split = load_cnof_corpus_split(
        corpus,
        train_size=TRAIN_SIZE,
        validation_size=VALIDATION_SIZE,
        test_size=TEST_SIZE,
        max_atoms=MAX_ATOMS,
        seed=CORPUS_SEED,
        scan_all=True,
        workers=6,
    )
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps({"cache_key": key, "train": list(split.train)}))
    return tuple(split.train)


# ---- Worker ----

_STATE: dict[str, object] = {}


def _init_worker(checkpoint: str) -> None:
    import torch

    torch.set_num_threads(1)
    from evaluate_tracelet_rollouts import load_factorized_rollout_checkpoint

    model, payload = load_factorized_rollout_checkpoint(checkpoint)
    _STATE["model"] = model
    _STATE["prior"] = payload["tree_source_prior"]
    _STATE["templates"] = model.ring_system_templates


def _measure(job: tuple[int, str]) -> dict:
    """Compile one molecule under every arm and measure its ring support."""

    import numpy as np

    from compose_v4.chem.molecular_graph import smiles_to_molecular_graph
    from compose_v4.chem.state import pad_molecular_graph
    from compose_v4.eval.denovo_schedule_probe import (
        ring_size_support_histogram,
        small_ring_category_mask,
        small_ring_support_mass,
    )
    from compose_v4.rewrite.commuting_schedule import states_are_array_exact
    from compose_v4.rewrite.kernel import canonical_state_key, de_novo_rewrite_system
    from compose_v4.rewrite.trace import execute_trace
    from compose_v4.rewrite.tree_transport import compile_carbon_tree_to_target

    index, smiles = job
    model = _STATE["model"]
    prior = _STATE["prior"]
    templates = _STATE["templates"]
    if "category" not in _STATE:
        _STATE["category"] = small_ring_category_mask(
            templates, maximum_size=SMALL_RING_MAX
        )
    category = _STATE["category"]
    runtime = de_novo_rewrite_system()

    row: dict = {"index": index, "smiles": smiles, "arms": {}}
    try:
        target = pad_molecular_graph(smiles_to_molecular_graph(smiles), MAX_ATOMS)
    except (ValueError, RuntimeError, KeyError) as error:
        row["target_parse_error"] = f"{type(error).__name__}: {error}"
        return row
    row["heavy_atoms"] = int(target.n_real_atoms)
    target_key = canonical_state_key(target)

    # One source draw per molecule, shared by every arm, so an arm difference
    # can never come from a different draw.
    source = prior.sample(np.random.default_rng(CORPUS_SEED + index), n_slots=MAX_ATOMS)

    for arm in ARMS:
        entry: dict = {}
        try:
            trace = compile_carbon_tree_to_target(
                source,
                target,
                system=runtime,
                use_bond_reroute=True,
                align_source=False,
                flexible_size=True,
                typed_ring_payloads=True,
                ring_catalog=None,
                event_schedule=arm,
            )
        except Exception as error:  # noqa: BLE001 - recorded, never silently dropped
            entry["compile_error"] = f"{type(error).__name__}: {error}"
            row["arms"][arm] = entry
            continue
        endpoint, states = execute_trace(
            trace.source, trace.steps, system=runtime, return_states=True
        )
        entry["trace_length"] = len(trace.steps)
        entry["endpoint_canonical_key_matches"] = bool(
            canonical_state_key(endpoint) == target_key
        )
        entry["endpoint_array_exact"] = bool(states_are_array_exact(endpoint, target))
        entry["ring_dependency_block_commits"] = trace.metadata.get(
            "ring_dependency_block_commits"
        )
        entry["ring_dependency_block_deferred"] = trace.metadata.get(
            "ring_dependency_block_deferred"
        )
        events = []
        for position, step in enumerate(trace.steps):
            if step.rule_name != "ring_system_grow":
                continue
            state = states[position]
            support = model._ring_grow_support(state)
            mass, legal, legal_small = small_ring_support_mass(support, category)
            events.append(
                {
                    "index": position,
                    "fraction_of_trace": position / len(trace.steps),
                    "small_mass_uniform_support": mass,
                    "legal_template_count": legal,
                    "legal_small_template_count": legal_small,
                    "size_histogram": ring_size_support_histogram(support, templates),
                    "heavy_atoms_at_event": int(state.n_real_atoms),
                }
            )
        entry["ring_events"] = events
        row["arms"][arm] = entry
    return row


# ---- Driver ----


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-size", type=int, required=True)
    parser.add_argument("--seed", type=int, default=20260921)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(os.path.expanduser(
            "~/compose_denovo_artifacts/lineageB/checkpoint.best_so_far.pt"
        )),
    )
    parser.add_argument(
        "--corpus",
        type=Path,
        default=Path(os.path.expanduser(
            "~/compose_denovo_artifacts/guacamol/guacamol_subset_500000_seed0.smiles"
        )),
    )
    args = parser.parse_args()

    import numpy as np

    started = time.time()
    pinned = _assert_pinned(args.checkpoint, args.corpus)
    train = load_train_partition(args.corpus, args.checkpoint.parent.parent / "splits")
    if args.sample_size > len(train):
        raise ValueError(f"requested {args.sample_size} of {len(train)} train molecules")
    rng = np.random.default_rng(args.seed)
    chosen = sorted(int(i) for i in rng.choice(len(train), args.sample_size, replace=False))
    jobs = [(index, train[index]) for index in chosen]
    print(f"[sample] {len(jobs)} of {len(train)} train molecules, seed {args.seed}",
          flush=True)

    # This machine is shared, so a long run is never all-or-nothing: rows are
    # appended to a JSONL sidecar as they land and the summary is rewritten
    # periodically.  A run killed at any point is still readable, and a
    # relaunch can be scored against whatever completed.
    sidecar = args.output.with_suffix(".rows.jsonl")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    pool = None
    if args.workers > 1:
        pool = ProcessPoolExecutor(
            max_workers=args.workers,
            initializer=_init_worker,
            initargs=(str(args.checkpoint),),
        )
        stream = pool.map(_measure, jobs, chunksize=1)
    else:
        # Serial is the robust mode on this machine: a shared, heavily loaded
        # laptop has already killed a worker mid-run, and every number here is
        # a deterministic structural count, so serialising costs wall clock and
        # nothing else.
        _init_worker(str(args.checkpoint))
        stream = map(_measure, jobs)

    interrupted = None
    try:
        with sidecar.open("w") as handle:
            for done, row in enumerate(stream, start=1):
                rows.append(row)
                handle.write(json.dumps(row, sort_keys=True) + "\n")
                handle.flush()
                if done % 10 == 0:
                    print(
                        json.dumps(
                            {
                                "phase": "progress",
                                "done": done,
                                "total": len(jobs),
                                "elapsed_seconds": round(time.time() - started, 1),
                                **{
                                    arm: round(value, 4)
                                    for arm, value in _interim(rows).items()
                                },
                            }
                        ),
                        flush=True,
                    )
                if done % 25 == 0:
                    _write_interim(args, rows, pinned, len(train), started)
    except (BrokenProcessPool, KeyboardInterrupt, OSError) as error:
        # A machine failure must not discard completed measurements.  It is
        # recorded in the artifact so a partial run can never be mistaken for
        # a complete one.
        interrupted = f"{type(error).__name__}: {error}"
        print(json.dumps({"phase": "interrupted", "error": interrupted,
                          "completed": len(rows)}), flush=True)
    finally:
        if pool is not None:
            pool.shutdown(wait=False, cancel_futures=True)

    artifact = {
        "measurement": "denovo_ring_schedule_acceptance_v1",
        "trains_nothing": True,
        "oracle_calls": 0,
        "arms": list(ARMS),
        "pinned_inputs_sha256": pinned,
        "corpus_split": {
            "train_size": len(train),
            "train_partition_size": TRAIN_SIZE,
            "validation_size": VALIDATION_SIZE,
            "test_size": TEST_SIZE,
            "max_atoms": MAX_ATOMS,
            "corpus_seed": CORPUS_SEED,
        },
        "sampling": {
            "rule": "uniform at random without replacement from the train partition",
            "seed": args.seed,
            "sample_size": args.sample_size,
        },
        "status": "COMPLETE" if interrupted is None else "PARTIAL",
        "interrupted": interrupted,
        "molecules_completed": len(rows),
        "rows_sidecar": str(sidecar.name),
        "rows": rows,
        "wall_seconds": time.time() - started,
        "workers": args.workers,
    }
    artifact["summary"] = summarize(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=1, sort_keys=True) + "\n")
    print(json.dumps(artifact["summary"], indent=1, sort_keys=True), flush=True)
    print(f"[written] {args.output}", flush=True)


def _interim(rows: list[dict]) -> dict[str, float]:
    """Running mean small-ring support per arm, for the progress line."""

    out: dict[str, float] = {}
    for arm in ARMS:
        masses = [
            event["small_mass_uniform_support"]
            for row in rows
            for event in row.get("arms", {}).get(arm, {}).get("ring_events", [])
        ]
        if masses:
            out[arm] = sum(masses) / len(masses)
    return out


def _write_interim(args, rows, pinned, train_size, started) -> None:
    """Rewrite the summary artifact from whatever has completed so far."""

    payload = {
        "measurement": "denovo_ring_schedule_acceptance_v1",
        "status": "PARTIAL",
        "trains_nothing": True,
        "oracle_calls": 0,
        "arms": list(ARMS),
        "pinned_inputs_sha256": pinned,
        "corpus_split": {"train_size": train_size, "max_atoms": MAX_ATOMS,
                         "corpus_seed": CORPUS_SEED},
        "sampling": {
            "rule": "uniform at random without replacement from the train partition",
            "seed": args.seed,
            "sample_size_requested": args.sample_size,
            "sample_size_completed": len(rows),
        },
        "wall_seconds": time.time() - started,
        "workers": args.workers,
        "summary": summarize(rows),
    }
    args.output.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")


def summarize(rows: list[dict]) -> dict:
    """Reduce measured rows to the acceptance report.

    Every distribution carries its own ``n``.  An arm with no events reports
    ``None`` rather than a measured zero, and the ring-free molecules are
    counted separately because the repair has no work to do on them.
    """

    from statistics import median

    parsed = [row for row in rows if "target_parse_error" not in row]
    summary: dict = {
        "molecules_requested": len(rows),
        "molecules_parsed": len(parsed),
        "molecules_target_parse_failed": len(rows) - len(parsed),
        "arms": {},
    }
    ring_free = 0
    compiled_both = 0
    for row in parsed:
        entries = row.get("arms", {})
        if all("compile_error" not in entries.get(arm, {}) for arm in ARMS):
            compiled_both += 1
            if not entries[ARMS[0]].get("ring_events"):
                ring_free += 1
    summary["molecules_compiled_under_every_arm"] = compiled_both
    summary["molecules_without_a_ring_system"] = ring_free
    summary["molecules_in_repair_scope"] = compiled_both - ring_free

    heavy = [row["heavy_atoms"] for row in parsed if "heavy_atoms" in row]
    summary["realized_strata"] = {
        "heavy_atoms_n": len(heavy),
        "heavy_atoms_mean": (sum(heavy) / len(heavy)) if heavy else None,
        "heavy_atoms_median": median(heavy) if heavy else None,
        "heavy_atoms_min": min(heavy) if heavy else None,
        "heavy_atoms_max": max(heavy) if heavy else None,
        "ring_systems_per_molecule": dict(
            Counter(
                len(row["arms"][ARMS[0]].get("ring_events", []))
                for row in parsed
                if "compile_error" not in row["arms"].get(ARMS[0], {})
            )
        ),
    }

    for arm in ARMS:
        entries = [row["arms"][arm] for row in parsed if arm in row.get("arms", {})]
        compiled = [e for e in entries if "compile_error" not in e]
        events = [event for entry in compiled for event in entry.get("ring_events", [])]
        masses = [event["small_mass_uniform_support"] for event in events]
        legal = [event["legal_template_count"] for event in events]
        fractions = [event["fraction_of_trace"] for event in events]
        histogram: Counter = Counter()
        for event in events:
            histogram.update(event["size_histogram"])
        total_legal = sum(histogram.values())
        summary["arms"][arm] = {
            "molecules_compiled": len(compiled),
            "compile_failures": len(entries) - len(compiled),
            "endpoint_canonical_key_matches": sum(
                1 for e in compiled if e["endpoint_canonical_key_matches"]
            ),
            "endpoint_array_exact": sum(
                1 for e in compiled if e["endpoint_array_exact"]
            ),
            "ring_events_n": len(events),
            "small_mass_uniform_support_mean": (sum(masses) / len(masses)) if masses else None,
            "small_mass_uniform_support_median": median(masses) if masses else None,
            "legal_template_count_mean": (sum(legal) / len(legal)) if legal else None,
            "legal_template_count_median": median(legal) if legal else None,
            "ring_event_fraction_of_trace_mean": (
                (sum(fractions) / len(fractions)) if fractions else None
            ),
            "ring_event_fraction_of_trace_median": median(fractions) if fractions else None,
            "support_size_histogram": dict(sorted(histogram.items(), key=_size_key)),
            "support_size_histogram_share": {
                key: value / total_legal
                for key, value in sorted(histogram.items(), key=_size_key)
            }
            if total_legal
            else {},
        }
    return summary


def _size_key(item: tuple[str, int]) -> tuple[int, str]:
    key = item[0]
    return (int(key), key) if key.isdigit() else (10_000, key)


if __name__ == "__main__":
    main()
