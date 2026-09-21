#!/usr/bin/env python3
"""Before/after table for attachment-aware fragment control, on the SAME instances.

Two arms, one frozen sampler configuration, one frozen controller
configuration, the identical (task, drug, seed) instance set.  The only thing
that differs between the arms is whether the controller reads the prompt's
DECLARED attachment interfaces.

Three columns, never conflated:

    chemical_validity       committed and chemically valid.  Reported over BOTH
                            denominators, because they answer different
                            questions:
                              * over COMMITTED endpoints -- the architectural
                                claim.  Committed states are complete, valid,
                                connected molecules by construction, so this is
                                expected to be exactly 100%, and anything else
                                is a defect in the executor or the adapter.
                              * over ATTEMPTS -- sampler efficiency, i.e. how
                                often an attempt produces a molecule at all.
                                This is the like-for-like column against a
                                baseline whose evaluator never inspects the
                                prompt.
    fragment_containment    the prompt fragment is present, over ATTEMPTS,
                            scored by the INDEPENDENT audit query rather than
                            by the function that decided what to emit.
    task_success            every declared attachment site is extended, over
                            ATTEMPTS.  This is the column the placement
                            mechanism moves.

Refusals, all of them hard:

* a shard whose sampler hash differs from its arm's;
* a shard whose attachment-control hash differs from its arm's, which is how a
  per-drug or per-task adjustment would show up;
* the two arms disagreeing on the sampler hash, which would mean the comparison
  is not controlled;
* the two arms covering different instance sets;
* any shard violating ``task_success <= containment <= produced`` or
  ``committed_chemically_valid == committed_endpoints``.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

TASK_ORDER = (
    "motif_extension",
    "superstructure_generation",
    "scaffold_decoration",
    "linker_design",
    "scaffold_morphing",
)
COLUMNS = (
    "chemical_validity_over_attempts",
    "fragment_containment",
    "task_success",
    "produced",
)


class ArmError(SystemExit):
    pass


def load_arm(shards: Path, *, label: str) -> dict:
    """Read one arm's shards and pin its identity."""
    rows: dict[str, list[dict]] = defaultdict(list)
    sampler_hashes: set[str] = set()
    control_hashes: set[str] = set()
    control_payloads: list[dict] = []
    kernels: set[str] = set()
    checkpoints: set[str] = set()
    instances: set[tuple[str, str, int]] = set()

    paths = sorted(shards.glob("*.json"))
    if not paths:
        raise ArmError(f"{label}: no shards under {shards}")
    for path in paths:
        shard = json.loads(path.read_text())
        sampler_hashes.add(shard["sampler"]["config_sha256"])
        control = shard.get("attachment_control")
        if control is None:
            # A shard produced before the controller existed is, by definition,
            # the baseline configuration; name it explicitly rather than
            # letting a missing key read as agreement.
            control_hashes.add("absent:pre_attachment_control")
        else:
            control_hashes.add(control["config_sha256"])
            control_payloads.append(control)
        kernels.add(json.dumps(shard["kernel"], sort_keys=True))
        checkpoints.add(str(shard["checkpoint"]["path"]))
        for task, result in shard["results"].items():
            for drug, entries in result["per_drug"].items():
                for row in entries:
                    rows[task].append({"drug": drug, **row})
                    instances.add((task, drug, int(row["seed"])))

    if len(sampler_hashes) != 1:
        raise ArmError(f"{label}: shards disagree on the sampler: {sorted(sampler_hashes)}")
    if len(control_hashes) != 1:
        raise ArmError(
            f"{label}: shards disagree on the attachment controller: "
            f"{sorted(control_hashes)} -- a per-instance configuration is not admissible"
        )
    if len(kernels) != 1:
        raise ArmError(f"{label}: shards disagree on the chemistry kernel")
    if len(checkpoints) != 1:
        raise ArmError(f"{label}: shards disagree on the checkpoint: {sorted(checkpoints)}")

    return {
        "label": label,
        "shard_count": len(paths),
        "sampler_sha256": sampler_hashes.pop(),
        "attachment_sha256": control_hashes.pop(),
        "attachment_config": (control_payloads[0]["config"] if control_payloads else None),
        "arm_name": (control_payloads[0]["arm"] if control_payloads else "frozen_sampler_baseline"),
        "kernel": json.loads(next(iter(kernels))),
        "checkpoint": checkpoints.pop(),
        "rows": rows,
        "instances": instances,
    }


def check_row_guards(task: str, row: dict) -> None:
    produced = int(row["committed_endpoints"])
    valid = int(row["committed_chemically_valid"])
    contained = int(row["committed_fragment_preserving"])
    success = int(row["emitted_nonempty"])
    where = f"{task}/{row['drug']}/seed{row['seed']}"
    if valid != produced:
        raise ArmError(
            f"{where}: committed_chemically_valid ({valid}) != committed endpoints "
            f"({produced}); committed states are valid BY CONSTRUCTION, so this is "
            "an executor or adapter defect, not a score"
        )
    if not success <= contained <= produced:
        raise ArmError(
            f"{where}: task_success ({success}) <= containment ({contained}) <= "
            f"produced ({produced}) is violated"
        )


def summarise(rows: list[dict], task: str) -> dict:
    by_seed: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        check_row_guards(task, row)
        by_seed[int(row["seed"])].append(row)

    per_seed: dict[int, dict[str, float]] = {}
    for seed, seed_rows in sorted(by_seed.items()):
        def mean(fn, _rows=seed_rows):
            return statistics.fmean([fn(r) for r in _rows])

        per_seed[seed] = {
            "chemical_validity_over_attempts": mean(
                lambda r: 100.0 * r["committed_chemically_valid"] / r["attempts"]
            ),
            "fragment_containment": mean(
                lambda r: 100.0 * r["committed_fragment_preserving"] / r["attempts"]
            ),
            "task_success": mean(lambda r: 100.0 * r["emitted_nonempty"] / r["attempts"]),
            "produced": mean(lambda r: 100.0 * r["committed_endpoints"] / r["attempts"]),
        }

    summary = {
        key: {
            "mean": statistics.fmean([per_seed[s][key] for s in sorted(per_seed)]),
            "std": statistics.pstdev([per_seed[s][key] for s in sorted(per_seed)]),
        }
        for key in COLUMNS
    }

    committed = sum(int(r["committed_endpoints"]) for r in rows)
    valid = sum(int(r["committed_chemically_valid"]) for r in rows)
    emitted = sum(int(r["emitted_nonempty"]) for r in rows)
    persisted = sum(len(r.get("committed_endpoint_smiles", ())) for r in rows)
    interfaces = {len(r.get("declared_interfaces", ())) for r in rows}
    return {
        "rows": len(rows),
        "summary": summary,
        "per_seed": per_seed,
        "chemical_validity_over_committed_pct": (
            100.0 * valid / committed if committed else float("nan")
        ),
        "committed_endpoints": committed,
        "emitted_after_task_filter": emitted,
        "censored_by_task_filter": committed - emitted,
        "censoring_pct_of_committed": (
            100.0 * (committed - emitted) / committed if committed else 0.0
        ),
        "committed_endpoints_persisted": persisted,
        "molecules_recoverable": persisted == committed,
        "declared_interfaces_per_prompt": sorted(interfaces),
        "redirections": sum(int(r.get("redirections", 0)) for r in rows),
        "interface_rejections": sum(int(r.get("interface_rejections", 0)) for r in rows),
        "staging_rejections": sum(int(r.get("staging_rejections", 0)) for r in rows),
        "separation_failures": sum(int(r.get("separation_failures", 0)) for r in rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-shards", required=True, type=Path)
    parser.add_argument("--attachment-shards", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="compare only the instances both arms cover (for an in-flight sweep)",
    )
    args = parser.parse_args()

    before = load_arm(args.baseline_shards, label="baseline")
    after = load_arm(args.attachment_shards, label="attachment_control")

    if before["sampler_sha256"] != after["sampler_sha256"]:
        raise ArmError(
            "the two arms do not share a sampler configuration "
            f"({before['sampler_sha256']} vs {after['sampler_sha256']}); the "
            "comparison would not be controlled"
        )
    if before["attachment_sha256"] == after["attachment_sha256"]:
        raise ArmError("both arms carry the same controller hash; there is nothing to compare")
    if before["checkpoint"] != after["checkpoint"]:
        raise ArmError("the two arms used different checkpoints")

    shared = before["instances"] & after["instances"]
    if before["instances"] != after["instances"]:
        missing = sorted(before["instances"] ^ after["instances"])[:8]
        if not args.allow_partial:
            raise ArmError(
                f"the arms cover different instance sets ({len(missing)}+ differ, "
                f"e.g. {missing}); pass --allow-partial to compare the overlap"
            )

    tasks: dict[str, dict] = {}
    for task in TASK_ORDER:
        before_rows = [
            r for r in before["rows"].get(task, ())
            if (task, r["drug"], int(r["seed"])) in shared
        ]
        after_rows = [
            r for r in after["rows"].get(task, ())
            if (task, r["drug"], int(r["seed"])) in shared
        ]
        if not before_rows or not after_rows:
            continue
        b = summarise(before_rows, task)
        a = summarise(after_rows, task)
        tasks[task] = {
            "instances": len({(r["drug"], r["seed"]) for r in before_rows}),
            "before": b,
            "after": a,
            "delta": {
                key: a["summary"][key]["mean"] - b["summary"][key]["mean"]
                for key in COLUMNS
            },
        }

    payload = {
        "schema": "compose_fragment_attachment_before_after_v1",
        "comparison": (
            "identical instances, identical sampler configuration, identical "
            "checkpoint; the arms differ only in whether the controller reads "
            "the prompt's declared attachment interfaces"
        ),
        "arms": {
            "before": {
                k: before[k]
                for k in (
                    "label", "arm_name", "shard_count", "sampler_sha256",
                    "attachment_sha256", "attachment_config", "kernel", "checkpoint",
                )
            },
            "after": {
                k: after[k]
                for k in (
                    "label", "arm_name", "shard_count", "sampler_sha256",
                    "attachment_sha256", "attachment_config", "kernel", "checkpoint",
                )
            },
        },
        "instances_compared": len(shared),
        "partial": before["instances"] != after["instances"],
        "column_semantics": {
            "chemical_validity_over_committed_pct": (
                "valid AND connected, over COMMITTED endpoints. The "
                "architectural claim; expected exactly 100."
            ),
            "chemical_validity_over_attempts": (
                "valid AND connected, over ATTEMPTS. Sampler efficiency, and "
                "the like-for-like column against a published validity number."
            ),
            "fragment_containment": "prompt fragment present, over ATTEMPTS.",
            "task_success": (
                "fragment present AND every declared attachment site extended, "
                "over ATTEMPTS."
            ),
        },
        "tasks": tasks,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2))

    width = 26
    print(
        f"{'task':{width}s} {'arm':10s} "
        f"{'chem_valid/committed':>20s} {'chem_valid/attempts':>19s} "
        f"{'containment':>11s} {'task_success':>12s}"
    )
    for task, block in tasks.items():
        for arm in ("before", "after"):
            row = block[arm]
            print(
                f"{task:{width}s} {arm:10s} "
                f"{row['chemical_validity_over_committed_pct']:>20.4f} "
                f"{row['summary']['chemical_validity_over_attempts']['mean']:>19.2f} "
                f"{row['summary']['fragment_containment']['mean']:>11.2f} "
                f"{row['summary']['task_success']['mean']:>12.2f}"
            )
        print(
            f"{'':{width}s} {'delta':10s} {'':>20s} "
            f"{block['delta']['chemical_validity_over_attempts']:>+19.2f} "
            f"{block['delta']['fragment_containment']:>+11.2f} "
            f"{block['delta']['task_success']:>+12.2f}"
        )
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
