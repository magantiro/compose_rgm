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
import math
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

# Tasks whose numbers describe a different task from the published one. See
# diagnostics/fragment_official_suite_v2/linker_design_invalidation.json: the two
# retained cores are joined by a direct bond that the region lock then pins, so
# the generated linker is always zero atoms long. Their shards are kept as
# evidence of the defect; their rows are not results.
WITHHELD_TASKS = {
    "linker_design": "zero-length linker pinned by the region lock",
    "scaffold_morphing": "copies linker_design, which is withheld",
}


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


def assert_denominators(shards: list[dict], expected_samples: int) -> dict:
    """Prove, over every row, that validity was scored out of the ATTEMPTS.

    Two identities are checked on every (task, drug, seed) row:

    ``attempts == validity_denominator == expected_samples``
        A row that requested 100 and scored 40 survivors would break this.

    ``official validity == 100 * emitted_nonempty / attempts``
        Every non-empty emission is asserted parseable, connected and
        canonical before scoring, so the official valid COUNT must equal the
        number of attempts that produced something.  If the two ever diverge,
        either an attempt was dropped from the denominator or an unparseable
        string was scored as valid -- both are the cherry-pick this check
        exists to exclude.
    """
    checked = 0
    for shard in shards:
        for task, result in shard["results"].items():
            for drug, rows in result["per_drug"].items():
                for row in rows:
                    where = f"{task}/{drug}/seed{row['seed']}"
                    if not (
                        row["attempts"]
                        == row["validity_denominator"]
                        == expected_samples
                    ):
                        raise SystemExit(
                            f"{where}: denominator is not the attempt count "
                            f"({row['attempts']}, {row['validity_denominator']}, "
                            f"expected {expected_samples})"
                        )
                    implied = 100.0 * row["emitted_nonempty"] / row["attempts"]
                    if abs(implied - row["official"]["validity"]) > 1e-9:
                        raise SystemExit(
                            f"{where}: official validity {row['official']['validity']} "
                            f"does not equal 100*emitted/attempts {implied}"
                        )
                    checked += 1
    return {"rows_checked": checked, "samples_per_prompt": expected_samples}


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
        # A drug/seed that emitted NOTHING has no molecules to measure distance
        # over, so its distance is NaN. That is a real absence, not a zero:
        # averaging it in as 0.0 would drag the task distance toward the prompt
        # exactly where the sampler failed hardest. Drop it from the mean and
        # COUNT the drops, so a row resting on fewer drugs is visible.
        per_seed_rows = {}
        undefined: dict[str, int] = defaultdict(int)
        for seed, by_drug in sorted(by_seed.items()):
            row = {}
            for key in METRIC_KEYS:
                values = [by_drug[d]["official"][key] for d in sorted(by_drug)]
                finite = [v for v in values if not math.isnan(v)]
                undefined[key] += len(values) - len(finite)
                row[key] = statistics.fmean(finite) if finite else float("nan")
            per_seed_rows[seed] = row
        complete = all(len(by_seed[s]) == expected_drugs for s in by_seed) and (
            len(by_seed) == expected_seeds
        )
        summary = {}
        for key in METRIC_KEYS:
            values = [
                per_seed_rows[s][key]
                for s in sorted(per_seed_rows)
                if not math.isnan(per_seed_rows[s][key])
            ]
            summary[key] = {
                "mean": statistics.fmean(values) if values else float("nan"),
                "std": statistics.pstdev(values) if len(values) > 1 else 0.0,
                "seeds_contributing": len(values),
                "drug_seed_cells_undefined": undefined[key],
            }
        totals: dict[str, int] = defaultdict(int)
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
        attempts = totals["attempts"] or 1
        committed = totals["committed_endpoints"] or 1
        decomposition = {
            # What COMPOSE emits, and therefore what the official metric scores.
            # An attempt that produced nothing admissible is an INVALID attempt.
            "strict_benchmark_validity_pct": summary["validity"]["mean"],
            # The official definition applied to every committed endpoint:
            # parseable, connected, salt-free. This is the number a generator
            # that did NOT self-censor on the prompt would report, and it is the
            # like-for-like comparison against a baseline whose evaluator never
            # checks the fragment at all.
            # THE BY-CONSTRUCTION CLAIM, stated as its own ratio: of the states
            # COMPOSE actually committed, how many are valid connected
            # molecules. Committed states are complete, supported and connected
            # by construction and the legal-event fiber is validity-closed, so
            # this is expected to be exactly 100% and is worth reporting
            # separately from any benchmark validity, which also folds in
            # "the trajectory produced nothing" and "the prompt was not
            # satisfied".
            "chemically_valid_share_of_committed_pct": 100.0
            * totals["committed_chemically_valid"]
            / committed,
            "chemical_validity_of_committed_pct": 100.0
            * totals["committed_chemically_valid"]
            / attempts,
            "committed_per_attempt_pct": 100.0 * totals["committed_endpoints"] / attempts,
            # Of the states COMPOSE committed, how many still contain the prompt
            # fragment(s) as a non-overlapping atom-level substructure.
            "fragment_containment_of_committed_pct": 100.0
            * totals["committed_fragment_preserving"]
            / committed,
            # Committed, fragment-containing, but the DECLARED attachment site
            # was not extended. Only tasks whose prompt carries a dummy can fail
            # this way; superstructure prompts carry none.
            "endpoint_constraint_failures": totals["constraint_failures"],
            "trajectories_hitting_rejection_budget": totals["budget_exhausted"],
        }
        out[task] = {
            "complete": complete,
            "withheld_reason": WITHHELD_TASKS.get(task),
            "constraint_decomposition": decomposition,
            "seeds_present": sorted(by_seed),
            "drugs_per_seed": {s: len(by_seed[s]) for s in sorted(by_seed)},
            "undefined_cells": dict(undefined),
            # A (drug, seed) cell that emitted nothing valid still contributes
            # to uniqueness (0, since valid_count is 0) and to diversity (0),
            # so once a task has many empty cells its uniqueness and diversity
            # describe the empty cells rather than the chemistry. The distance
            # undefined count IS the empty-cell count, because distance is the
            # one metric that refuses to be defined over nothing.
            "drug_seed_cells_with_no_valid_emission": undefined["distance"],
            "secondary_metrics_interpretable": undefined["distance"] == 0,
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
    parser.add_argument("--samples", type=int, default=100)
    args = parser.parse_args()

    shards = load_shards(args.shards)
    identity = assert_one_frozen_configuration(shards)
    denominators = assert_denominators(shards, args.samples)
    table = collect(shards)
    summary = summarise(table, args.expected_drugs, args.expected_seeds)

    payload = {
        "schema": "compose_fragment_official_suite_table_v1",
        "identity": identity,
        "shards": len(shards),
        "denominator_audit": denominators,
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
        if block.get("withheld_reason"):
            flag += f"  [WITHHELD: {block['withheld_reason']}]"
        print(
            f"{task:26s} {n:>5s} "
            f"{s['validity']['mean']:6.2f} {s['uniqueness']['mean']:6.2f} "
            f"{s['quality']['mean']:6.2f} {s['diversity']['mean']:6.3f} "
            f"{s['distance']['mean']:6.3f}{flag}"
        )
    print(f"wrote {args.output}")


if __name__ == "__main__":
    raise SystemExit(main())
