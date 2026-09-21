#!/usr/bin/env python3
"""Aggregate the sharded fragment-constrained sweep into the reportable table.

A task row is the UNWEIGHTED MEAN over that task's ten drugs, computed per
seed, then reported as mean +/- population standard deviation over the seeds --
the aggregation ``in_virtuo_gen/evaluation/downstream.py`` performs.
``scaffold_morphing`` is not a separate run: upstream copies the linker result,
so this copies it too and says so.

This refuses to aggregate unless every shard was produced by ONE sampler
configuration and ONE checkpoint.  Tuning per instance and reporting the
aggregate as a single method is the failure this check exists to make
impossible, and a hash comparison is the only form of that check that cannot be
satisfied by intent.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

METRIC_KEYS = (
    "validity",
    "uniqueness",
    "quality",
    "diversity",
    "distance",
    "distance_to_original_drug",
)
TASK_ORDER = (
    "motif_extension",
    "linker_design",
    "scaffold_morphing",
    "superstructure_generation",
    "scaffold_decoration",
)
MORPHING_SOURCE = "linker_design"


def load_shards(shard_dir: Path) -> list[dict]:
    shards = []
    for path in sorted(shard_dir.glob("*.json")):
        try:
            shards.append(json.loads(path.read_text()))
        except json.JSONDecodeError as exc:
            raise SystemExit(f"unreadable shard {path}: {exc}") from exc
    if not shards:
        raise SystemExit(f"no shards under {shard_dir}")
    return shards


def assert_one_frozen_configuration(shards: list[dict]) -> dict:
    sampler_hashes = {s["sampler"]["config_sha256"] for s in shards}
    checkpoints = {s["checkpoint"]["path"] for s in shards}
    steps = {s["checkpoint"]["completed_steps"] for s in shards}
    kernels = {s["kernel"]["rdkit"] for s in shards}
    if len(sampler_hashes) != 1:
        raise SystemExit(
            "shards disagree on the sampler configuration; the sweep is not one "
            f"method: {sorted(sampler_hashes)}"
        )
    if len(checkpoints) != 1 or len(steps) != 1:
        raise SystemExit(f"shards disagree on the checkpoint: {checkpoints} {steps}")
    if len(kernels) != 1:
        raise SystemExit(f"shards disagree on the rdkit kernel: {sorted(kernels)}")
    return {
        "sampler_config_sha256": sampler_hashes.pop(),
        "sampler_config": shards[0]["sampler"]["config"],
        "checkpoint": shards[0]["checkpoint"],
        "kernel": shards[0]["kernel"],
    }


def collect(shards: list[dict]) -> dict:
    # task -> seed -> drug -> row
    table: dict[str, dict[int, dict[str, dict]]] = defaultdict(lambda: defaultdict(dict))
    for shard in shards:
        for task, result in shard["results"].items():
            for drug, rows in result["per_drug"].items():
                for row in rows:
                    seed = int(row["seed"])
                    if drug in table[task][seed]:
                        raise SystemExit(
                            f"duplicate shard for {task}/{drug}/seed{seed}; refusing to "
                            "merge by picking one"
                        )
                    table[task][seed][drug] = row
    return table


def summarise(table: dict, expected_drugs: int, expected_seeds: int) -> dict:
    out: dict[str, dict] = {}
    for task, by_seed in table.items():
        per_seed_rows = {}
        for seed, by_drug in sorted(by_seed.items()):
            per_seed_rows[seed] = {
                key: statistics.fmean(
                    [by_drug[d]["official"][key] for d in sorted(by_drug)]
                )
                for key in METRIC_KEYS
            }
        complete = all(len(by_seed[s]) == expected_drugs for s in by_seed) and (
            len(by_seed) == expected_seeds
        )
        summary = {}
        for key in METRIC_KEYS:
            values = [per_seed_rows[s][key] for s in sorted(per_seed_rows)]
            summary[key] = {
                "mean": statistics.fmean(values),
                "std": statistics.pstdev(values) if len(values) > 1 else 0.0,
            }
        totals = defaultdict(int)
        for by_drug in by_seed.values():
            for row in by_drug.values():
                for field in (
                    "attempts",
                    "committed_endpoints",
                    "committed_chemically_valid",
                    "committed_fragment_preserving",
                    "emitted_nonempty",
                    "emitted_fragment_preserving",
                    "constraint_failures",
                    "budget_exhausted",
                ):
                    totals[field] += row.get(field, 0)
        out[task] = {
            "complete": complete,
            "seeds_present": sorted(by_seed),
            "drugs_per_seed": {s: len(by_seed[s]) for s in sorted(by_seed)},
            "per_seed": per_seed_rows,
            "summary": summary,
            "totals": dict(totals),
        }
    if MORPHING_SOURCE in out and "scaffold_morphing" not in out:
        copied = json.loads(json.dumps(out[MORPHING_SOURCE]))
        copied["copied_from"] = MORPHING_SOURCE
        out["scaffold_morphing"] = copied
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--shards",
        type=Path,
        default=Path("diagnostics/fragment_official_suite_v2/shards"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("diagnostics/fragment_official_suite_v2/suite_table.json"),
    )
    parser.add_argument("--expected-drugs", type=int, default=10)
    parser.add_argument("--expected-seeds", type=int, default=3)
    args = parser.parse_args()

    shards = load_shards(args.shards)
    identity = assert_one_frozen_configuration(shards)
    table = collect(shards)
    summary = summarise(table, args.expected_drugs, args.expected_seeds)

    payload = {
        "schema": "compose_fragment_official_suite_table_v1",
        "identity": identity,
        "shards": len(shards),
        "aggregation": (
            "per seed: unweighted mean over the task's drugs; reported as mean +/- "
            "population std over seeds; scaffold_morphing copies linker_design"
        ),
        "tasks": summary,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2))

    print(f"sampler {identity['sampler_config_sha256'][:16]}  rdkit {identity['kernel']['rdkit']}")
    header = f"{'task':26s} {'n':>5s} {'valid':>7s} {'uniq':>7s} {'qual':>7s} {'div':>7s} {'dist':>7s}"
    print(header)
    for task in TASK_ORDER:
        if task not in summary:
            continue
        block = summary[task]
        s = block["summary"]
        drugs = block["drugs_per_seed"]
        n = f"{sum(drugs.values())}"
        flag = "" if block["complete"] else "  [PARTIAL]"
        print(
            f"{task:26s} {n:>5s} "
            f"{s['validity']['mean']:6.2f} {s['uniqueness']['mean']:6.2f} "
            f"{s['quality']['mean']:6.2f} {s['diversity']['mean']:6.3f} "
            f"{s['distance']['mean']:6.3f}{flag}"
        )
    print(f"wrote {args.output}")


if __name__ == "__main__":
    raise SystemExit(main())
