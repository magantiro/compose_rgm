"""Why the legal ring-template support collapses along a de-novo training trace.

The acceptance measurement established that the small-ring support mass is a
per-ring-system ORDINAL effect no schedule removes. This locates the cause.

Two halves, deliberately separate:

CATALOG half -- a property of the catalog alone, so no sample of states can
explain it away. For every template it reads how many HOST atoms the template
consumes, and reports the small-ring share of the templates that fit on a host
of each size. It also checks whether any template could match a host that
already contains a ring.

STATE half -- on real compiled traces, the eligible host at each ring decision,
decomposed into what removed the rest: committed cycles, installed
heteroatoms, charge. Run under BOTH schedules, because they destroy the host in
different orders and the difference is the whole point: `sequential` decorates
before committing rings, `ring_dependency_block` does not.

The host census is cheap graph work; the support predicate is not (~12 s per
state). So the host is measured on a wide sample and the support on a narrower
one, and the two sample sizes are reported separately rather than merged.

Trains nothing. Calls no oracle. Writes no checkpoint.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import defaultdict
from pathlib import Path
from statistics import median

MAX_ATOMS = 40
CORPUS_SEED = 20260717
SMALL_RING_MAX = 4
ARMS = ("sequential", "ring_dependency_block")
HOST_SIZES = (3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20, 24, 30, 40)


def _stats(values) -> dict:
    values = list(values)
    if not values:
        return {"n": 0, "mean": None, "median": None}
    n = len(values)
    mean = sum(values) / n
    variance = sum((v - mean) ** 2 for v in values) / (n - 1) if n > 1 else 0.0
    return {
        "n": n,
        "mean": mean,
        "median": median(values),
        "stderr": (variance / n) ** 0.5,
        "min": min(values),
        "max": max(values),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host-sample", type=int, default=300,
                        help="molecules for the cheap host census")
    parser.add_argument("--support-sample", type=int, default=40,
                        help="molecules that additionally pay for the support predicate")
    parser.add_argument("--seed", type=int, default=20260921)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=Path(os.path.expanduser(
        "~/compose_denovo_artifacts/lineageB/checkpoint.best_so_far.pt")))
    parser.add_argument("--split", type=Path, default=Path(os.path.expanduser(
        "~/compose_denovo_artifacts/splits/train_split_"
        "70526d92f1f08d8e292cb31218f81b6924a2182f772c43348015110669d47791"
        "_50000_2000_2000_40_20260717.json")))
    args = parser.parse_args()

    import numpy as np
    import torch

    torch.set_num_threads(1)

    from evaluate_tracelet_rollouts import load_factorized_rollout_checkpoint

    from compose_v4.chem.molecular_graph import smiles_to_molecular_graph
    from compose_v4.chem.state import pad_molecular_graph
    from compose_v4.eval.denovo_schedule_probe import (
        small_ring_category_mask,
        small_ring_support_mass,
    )
    from compose_v4.eval.ring_support_host import (
        catalog_host_requirement_histogram,
        catalog_small_ring_share_by_host_size,
        eligible_host_census,
        host_size_predicts,
        template_source_pattern_is_forest,
    )
    from compose_v4.rewrite.kernel import de_novo_rewrite_system
    from compose_v4.rewrite.trace import execute_trace
    from compose_v4.rewrite.tree_transport import compile_carbon_tree_to_target

    started = time.time()
    model, payload = load_factorized_rollout_checkpoint(str(args.checkpoint))
    prior = payload["tree_source_prior"]
    templates = model.ring_system_templates
    category = small_ring_category_mask(templates, maximum_size=SMALL_RING_MAX)
    runtime = de_novo_rewrite_system()

    # ---- Catalog half ----
    cyclic_host_templates = sum(
        1 for template in templates if not template_source_pattern_is_forest(template)
    )
    catalog = {
        "template_count": len(templates),
        "small_ring_template_count": int(category.sum()),
        "templates_requiring_a_cyclic_host": cyclic_host_templates,
        "host_requirement_histogram": catalog_host_requirement_histogram(templates),
        "small_ring_share_by_host_size": catalog_small_ring_share_by_host_size(
            templates, host_sizes=HOST_SIZES, maximum_small_ring=SMALL_RING_MAX
        ),
    }
    print(json.dumps({"phase": "catalog", **{k: v for k, v in catalog.items()
                                             if not isinstance(v, dict)}}), flush=True)

    # ---- State half ----
    train = json.loads(args.split.read_text())["train"]
    rng = np.random.default_rng(args.seed)
    chosen = sorted(
        int(i) for i in rng.choice(len(train), args.host_sample, replace=False)
    )
    support_indices = set(chosen[: args.support_sample])

    events: list[dict] = []
    compile_failures: dict[str, int] = {arm: 0 for arm in ARMS}
    parse_failures = 0
    compiled = 0
    for index in chosen:
        smiles = train[index]
        try:
            target = pad_molecular_graph(smiles_to_molecular_graph(smiles), MAX_ATOMS)
        except (ValueError, RuntimeError, KeyError):
            parse_failures += 1
            continue
        source = prior.sample(
            np.random.default_rng(CORPUS_SEED + index), n_slots=MAX_ATOMS
        )
        for arm in ARMS:
            try:
                trace = compile_carbon_tree_to_target(
                    source, target, system=runtime, use_bond_reroute=True,
                    align_source=False, flexible_size=True, typed_ring_payloads=True,
                    ring_catalog=None, event_schedule=arm,
                )
            except Exception as error:  # noqa: BLE001
                # Counted per arm and printed. A compile that fails in one arm
                # and not the other would bias every per-arm mean below, so it
                # must be visible rather than skipped.
                compile_failures[arm] += 1
                print(json.dumps({"phase": "compile_failed", "arm": arm,
                                  "index": index,
                                  "error": f"{type(error).__name__}: {error}"}),
                      flush=True)
                continue
            _, states = execute_trace(
                trace.source, trace.steps, system=runtime, return_states=True
            )
            ordinal = 0
            for position, step in enumerate(trace.steps):
                if step.rule_name != "ring_system_grow":
                    continue
                state = states[position]
                row = {
                    "index": index,
                    "arm": arm,
                    "ordinal": ordinal,
                    "fraction_of_trace": position / len(trace.steps),
                    **eligible_host_census(state),
                }
                if index in support_indices:
                    mass, legal, legal_small = small_ring_support_mass(
                        model._ring_grow_support(state), category
                    )
                    row["small_mass_uniform_support"] = mass
                    row["legal_template_count"] = legal
                    row["legal_small_template_count"] = legal_small
                events.append(row)
                ordinal += 1
        compiled += 1
        if compiled % 25 == 0:
            print(json.dumps({"phase": "progress", "molecules": compiled,
                              "events": len(events),
                              "elapsed": round(time.time() - started, 1)}), flush=True)

    report = {
        "measurement": "denovo_ring_support_host_census_v1",
        "trains_nothing": True,
        "oracle_calls": 0,
        "kernel": {"python": "3.11", "rdkit": "2024.03.5", "numpy": "1.26.4",
                   "scipy": "1.13.1", "networkx": "3.3", "torch": "2.4.0"},
        "sampling": {
            "rule": "uniform at random without replacement from the train partition",
            "seed": args.seed,
            "host_sample": args.host_sample,
            "support_sample": args.support_sample,
            "molecules_compiled": compiled,
            "target_parse_failures": parse_failures,
            "compile_failures_by_arm": compile_failures,
        },
        "catalog": catalog,
        "by_arm_and_ordinal": _by_arm_and_ordinal(events),
        "associations": _associations(events, host_size_predicts),
        "events": events,
        "wall_seconds": time.time() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=1, sort_keys=True) + "\n")
    print(json.dumps(report["by_arm_and_ordinal"], indent=1, sort_keys=True), flush=True)
    print(json.dumps(report["associations"], indent=1, sort_keys=True), flush=True)
    print(f"[written] {args.output}", flush=True)


def _by_arm_and_ordinal(events: list[dict]) -> dict:
    """Host and support per schedule arm, per ring-system ordinal."""

    grouped: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for event in events:
        grouped[(event["arm"], event["ordinal"])].append(event)
    report: dict = {}
    for (arm, ordinal), rows in sorted(grouped.items()):
        if len(rows) < 3:
            continue
        supported = [r for r in rows if "small_mass_uniform_support" in r]
        report.setdefault(arm, {})[str(ordinal)] = {
            "events": len(rows),
            "host_atoms": _stats(r["host_atoms"] for r in rows),
            "largest_host_tree": _stats(r["largest_host_tree"] for r in rows),
            "host_components": _stats(r["host_components"] for r in rows),
            "lost_to_cycles": _stats(r["lost_to_cycles"] for r in rows),
            "lost_to_heteroatoms": _stats(r["lost_to_heteroatoms"] for r in rows),
            "lost_to_charge": _stats(r["lost_to_charge"] for r in rows),
            "small_mass_uniform_support": _stats(
                r["small_mass_uniform_support"] for r in supported
            ),
            "legal_template_count": _stats(
                r["legal_template_count"] for r in supported
            ),
        }
    return report


def _associations(events: list[dict], correlate) -> dict:
    """Descriptive associations, reported per arm and never pooled across arms."""

    import math

    report: dict = {}
    for arm in ARMS:
        rows = [
            r for r in events
            if r["arm"] == arm and "small_mass_uniform_support" in r
        ]
        if len(rows) < 3:
            continue
        entry: dict = {"n": len(rows)}
        for name, key in (("host_atoms", "host_atoms"),
                          ("largest_host_tree", "largest_host_tree")):
            try:
                entry[f"r_mass_vs_{name}"] = correlate(
                    [r[key] for r in rows],
                    [r["small_mass_uniform_support"] for r in rows],
                )
                entry[f"r_log_legal_vs_{name}"] = correlate(
                    [r[key] for r in rows],
                    [math.log(r["legal_template_count"] + 1) for r in rows],
                )
            except ValueError as error:
                entry[f"r_mass_vs_{name}"] = f"undefined: {error}"
        report[arm] = entry
    return report


if __name__ == "__main__":
    main()
