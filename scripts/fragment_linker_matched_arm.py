#!/usr/bin/env python3
"""Read the matched linker arm: same everything, one difference.

This is a DIAGNOSTIC, not a benchmark row.  ``linker_design`` is withheld by
``tools/aggregate_fragment_suite.py`` and stays withheld: the withholding is a
correctness guard, and the question of whether the composite transaction is a
capability EXTENSION rather than an acceleration is open at the time of writing.
Nothing here lifts it, and nothing here produces a number formatted as a row.

What it does produce is the comparison the v3 predeclaration says a PASS
licenses, with the realized-length distribution beside it, because a linker
number without that distribution cannot be told apart from the seed the harness
constructed.

THE MATCH IS CHECKED, NOT ASSERTED.  The two arms must agree on the checkpoint,
the sampler hash, the seeded bridge and the samples per prompt, and their
controller configurations must differ in ``path_program`` AND NOTHING ELSE.
Two arms that differ in more than the mechanism are not a measurement of it.

Denominators are stated everywhere they are used.  ``task_success`` is over
ATTEMPTS, which is what the corrected fragment table reports; ``commit_rate`` is
over attempts too; the realized-length share is over COMMITTED two-core
endpoints, which is the denominator the v3 falsifier used -- and which the
per-event gate is known to filter, so it is reported beside the commit rate and
never alone.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path

TASK = "linker_design"


def _load(arm_dir: Path) -> list[dict]:
    shards = []
    for path in sorted(arm_dir.glob("*.json")):
        shards.append(json.loads(path.read_text()))
    if not shards:
        raise SystemExit(f"no shards under {arm_dir}")
    return shards


def _rows(shards: list[dict]) -> list[dict]:
    rows = []
    for shard in shards:
        result = shard["results"].get(TASK)
        if not result:
            continue
        for drug, entries in result["per_drug"].items():
            for row in entries:
                rows.append({"drug": drug, **row})
    return rows


def _identity(shards: list[dict]) -> dict:
    def one(name, fn):
        values = {json.dumps(fn(s), sort_keys=True) for s in shards}
        if len(values) != 1:
            raise SystemExit(f"shards disagree on {name}: {sorted(values)}")
        return json.loads(values.pop())

    return {
        "sampler_sha256": one("sampler", lambda s: s["sampler"]["config_sha256"]),
        "controller": one("controller", lambda s: s["attachment_control"]["config"]),
        "controller_sha256": one(
            "controller hash", lambda s: s["attachment_control"]["config_sha256"]
        ),
        "checkpoint": one("checkpoint", lambda s: s["checkpoint"]["path"]),
        "linker_bridge_atoms": one(
            "seeded bridge", lambda s: s["protocol"]["linker_bridge_atoms"]
        ),
        "samples_per_prompt": one(
            "samples", lambda s: s["protocol"]["samples_per_prompt"]
        ),
        "shards": len(shards),
    }


def _summarise(rows: list[dict]) -> dict:
    by_seed: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        by_seed[int(row["seed"])].append(row)

    def per_seed(fn):
        values = [
            statistics.fmean([fn(r) for r in by_seed[s]]) for s in sorted(by_seed)
        ]
        return {
            "mean": statistics.fmean(values),
            "std": statistics.pstdev(values),
            "per_seed": values,
        }

    lengths: Counter[int] = Counter()
    seeded = {r.get("seeded_linker_length") for r in rows}
    for row in rows:
        lengths.update(row.get("realized_linker_lengths") or [])
    committed_two_core = sum(lengths.values())
    seed_length = next(iter(seeded)) if len(seeded) == 1 else None
    above = (
        sum(n for length, n in lengths.items() if seed_length is not None
            and length > seed_length)
    )
    return {
        "drugs": len({r["drug"] for r in rows}),
        "seeds": sorted(by_seed),
        "attempts": sum(r["attempts"] for r in rows),
        # Over ATTEMPTS, as the corrected fragment table reports it.
        "task_success_pct_of_attempts": per_seed(
            lambda r: 100.0 * r["emitted_nonempty"] / r["attempts"]
        ),
        "commit_rate_pct_of_attempts": per_seed(
            lambda r: 100.0 * r["committed_endpoints"] / r["attempts"]
        ),
        "chemical_validity_pct_of_attempts": per_seed(
            lambda r: 100.0 * r["committed_chemically_valid"] / r["attempts"]
        ),
        "fragment_containment_pct_of_attempts": per_seed(
            lambda r: 100.0 * r["committed_fragment_preserving"] / r["attempts"]
        ),
        "official_uniqueness": per_seed(lambda r: r["official"]["uniqueness"]),
        "official_quality": per_seed(lambda r: r["official"]["quality"]),
        "official_diversity": per_seed(lambda r: r["official"]["diversity"]),
        "mean_events": per_seed(lambda r: r["mean_events"]),
        "seeded_linker_length": seed_length,
        "realized_length_histogram": dict(sorted(lengths.items())),
        "committed_two_core_endpoints": committed_two_core,
        "above_seed": above,
        "above_seed_pct_of_committed": (
            100.0 * above / committed_two_core if committed_two_core else None
        ),
        "path_transactions": sum(r.get("path_transactions", 0) for r in rows),
        "path_transaction_refusals": sum(
            r.get("path_transaction_refusals", 0) for r in rows
        ),
        "path_rejections": sum(r.get("path_rejections", 0) for r in rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--baseline-arm", default="attachment")
    parser.add_argument("--treatment-arm", default="path_program")
    args = parser.parse_args()

    arms = {}
    identities = {}
    for name in (args.baseline_arm, args.treatment_arm):
        shards = _load(args.shard_root / name)
        identities[name] = _identity(shards)
        arms[name] = _summarise(_rows(shards))

    base, treat = identities[args.baseline_arm], identities[args.treatment_arm]
    mismatched = [
        key for key in ("sampler_sha256", "checkpoint", "linker_bridge_atoms",
                        "samples_per_prompt")
        if base[key] != treat[key]
    ]
    if mismatched:
        raise SystemExit(
            f"the arms are not matched; they differ in {mismatched}. Two arms "
            "differing in more than the mechanism are not a measurement of it."
        )
    differing_controller_keys = sorted(
        k for k in set(base["controller"]) | set(treat["controller"])
        if base["controller"].get(k) != treat["controller"].get(k)
    )
    if differing_controller_keys != ["path_program"]:
        raise SystemExit(
            "the controller configurations must differ in path_program and "
            f"nothing else; they differ in {differing_controller_keys}"
        )

    b, t = arms[args.baseline_arm], arms[args.treatment_arm]
    payload = {
        "schema": "compose_fragment_linker_matched_arm_v1",
        "status": (
            "DIAGNOSTIC, NOT A BENCHMARK ROW. linker_design remains withheld by "
            "the aggregator. This reports the matched comparison the v3 "
            "predeclaration licenses, with the realized-length distribution "
            "beside it; whether a row may be published depends on the open "
            "question of whether the composite is a capability EXTENSION, which "
            "the constituent scoring probe answers and this file does not."
        ),
        "predeclaration_discipline": (
            "NO THRESHOLD IS PREDECLARED HERE, because no pass/fail decision is "
            "taken from this arm. It is DESCRIPTIVE: it reports a comparison. "
            "Its size was fixed by the benchmark protocol before launch (10 "
            "drugs x 3 seeds x 100 samples = 3,000 attempts per arm), not "
            "chosen after seeing data, and the arms were launched together. If "
            "a DECISION is later taken from a linker comparison -- promoting a "
            "row, choosing between routes -- that decision needs its own "
            "predeclared threshold, n from a power calculation, and denominator, "
            "committed before its first sample. Reading a threshold off this "
            "artifact afterwards would be exactly the move the v3 "
            "predeclaration exists to prevent."
        ),
        "support_statement_that_must_accompany_any_linker_row": (
            "The composite is a capability EXTENSION on 8 of 10 released prompts "
            "and an ACCELERATION on 2 (see "
            "diagnostics/fragment_path_v3_classification_v1.json). On 8 prompts "
            "it performs a transition the proposal law does not propose, so a "
            "linker number from this arm is not a number the prior could have "
            "reached by sampling."
        ),
        "matched_on": {
            k: base[k] for k in
            ("sampler_sha256", "checkpoint", "linker_bridge_atoms", "samples_per_prompt")
        },
        "differs_only_in": differing_controller_keys,
        "identities": identities,
        "arms": arms,
        "deltas": {
            key: t[key]["mean"] - b[key]["mean"]
            for key in (
                "task_success_pct_of_attempts",
                "commit_rate_pct_of_attempts",
                "official_uniqueness",
                "official_quality",
                "official_diversity",
            )
        },
        "reading": (
            "The share above the seeded length is over COMMITTED two-core "
            "endpoints. The per-event gate is measured to convert seed-length "
            "COMMITS into non-commits, so that share is not comparable across "
            "arms on its own and is reported beside the commit rate, which is "
            "over attempts and is not filtered by the gate."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2))

    print(f"matched on: {payload['matched_on']}")
    print(f"differs only in: {differing_controller_keys}\n")
    header = f"{'metric':38s} {args.baseline_arm:>14s} {args.treatment_arm:>14s} {'delta':>9s}"
    print(header)
    for key in (
        "task_success_pct_of_attempts", "commit_rate_pct_of_attempts",
        "chemical_validity_pct_of_attempts", "fragment_containment_pct_of_attempts",
        "official_uniqueness", "official_quality", "official_diversity",
        "mean_events",
    ):
        print(
            f"{key:38s} {b[key]['mean']:14.2f} {t[key]['mean']:14.2f} "
            f"{t[key]['mean'] - b[key]['mean']:9.2f}"
        )
    print(f"\nseeded length: {b['seeded_linker_length']} (both arms)")
    for name, arm in ((args.baseline_arm, b), (args.treatment_arm, t)):
        print(
            f"{name:14s} committed_two_core={arm['committed_two_core_endpoints']:5d} "
            f"above_seed={arm['above_seed']:5d} "
            f"({arm['above_seed_pct_of_committed']}) "
            f"hist={arm['realized_length_histogram']}"
        )
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
