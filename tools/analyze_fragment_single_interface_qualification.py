"""Audit held-prompt quality denominators without changing its frozen sampler.

The published IVG and GenMol quality metrics count distinct molecules passing
QED >= 0.6 and SA <= 4, then divide by *attempted* samples.  A committed-only
percentage and a unique-valid percentage are useful diagnostics, but neither
is the comparator metric.  Compute all three explicitly, per prompt first.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from rdkit import Chem, RDLogger, rdBase
from run_fragment_single_interface_qualification import (
    ARMS,
    TASK_DRUGS,
    _summarize,
    _unit_path,
)

from compose_v4.benchmark.fragment_official_metrics import _official_module_path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "diagnostics/fragment_single_interface_qualification_v1"
THRESHOLDS = {"qed_min": 0.6, "sa_max": 4.0}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _score(smiles: str, cache: dict[str, dict]) -> dict:
    if smiles not in cache:
        from in_virtuo_gen.utils.mol import compute_single_property

        molecule = Chem.MolFromSmiles(smiles)
        if molecule is None or len(Chem.GetMolFrags(molecule)) != 1:
            raise AssertionError(f"invalid or disconnected committed endpoint: {smiles}")
        canonical = Chem.MolToSmiles(molecule)
        if canonical != smiles:
            raise AssertionError(f"noncanonical committed endpoint: {smiles} != {canonical}")
        sa, qed = compute_single_property(smiles)
        cache[smiles] = {
            "sa": float(sa),
            "qed": float(qed),
            "quality_pass": bool(qed >= THRESHOLDS["qed_min"] and sa <= THRESHOLDS["sa_max"]),
            "heavy_atoms": molecule.GetNumHeavyAtoms(),
        }
    return cache[smiles]


def _prompt(row: dict, cache: dict[str, dict]) -> dict:
    attempts = row["attempt_records"]
    if len(attempts) != row["attempts"]:
        raise AssertionError("attempt records are censored")
    committed = [x["committed_smiles"] for x in attempts if x["committed_smiles"]]
    emitted_attempt_aligned = [x["emitted_smiles"] or "" for x in attempts]
    emitted = [smiles for smiles in emitted_attempt_aligned if smiles]
    if committed != row["committed_endpoint_smiles"]:
        raise AssertionError("attempt-aligned committed list differs from legacy list")
    if emitted_attempt_aligned != row["emitted_samples"]:
        raise AssertionError("attempt-aligned emitted list differs from legacy list")
    if len(committed) != row["committed_endpoints"]:
        raise AssertionError("committed count mismatch")
    if len(emitted) != row["emitted_nonempty"]:
        raise AssertionError("emitted count mismatch")
    for record in attempts:
        if record["emitted_smiles"] and record["emitted_smiles"] != record["committed_smiles"]:
            raise AssertionError("emission is not its committed molecule")
    unique_committed = sorted(set(committed))
    unique_emitted = sorted(set(emitted))
    for smiles in unique_committed:
        _score(smiles, cache)
    qualified_committed = sum(cache[s]["quality_pass"] for s in unique_committed)
    qualified_emitted = sum(cache[s]["quality_pass"] for s in unique_emitted)
    attempted = len(attempts)
    result = {
        "attempts": attempted,
        "committed_valid": len(committed),
        "committed_fragment_preserving": row["committed_fragment_preserving"],
        "committed_interfaces_covered": row["committed_interfaces_covered"],
        "unique_valid_committed": len(unique_committed),
        "emitted_task_success": len(emitted),
        "unique_valid_emitted": len(unique_emitted),
        "official_metrics": row["official"],
        "refusal_counts": {
            key: row[key]
            for key in (
                "budget_exhausted",
                "constraint_failures",
                "executor_refusals",
                "interface_rejections",
                "lock_rejections",
                "redirections",
                "staging_rejections",
            )
        },
        "distinct_qualified_committed": qualified_committed,
        "distinct_qualified_emitted": qualified_emitted,
        "quality_committed_per_attempt": qualified_committed / attempted,
        "quality_committed_per_commit": qualified_committed / len(committed) if committed else None,
        "quality_committed_per_unique_valid": qualified_committed / len(unique_committed)
        if unique_committed
        else None,
        "quality_official_emitted_per_attempt": qualified_emitted / attempted,
        "mean_qed_all_commits": _mean([cache[s]["qed"] for s in committed]),
        "mean_sa_all_commits": _mean([cache[s]["sa"] for s in committed]),
        "mean_heavy_atoms_all_commits": _mean([cache[s]["heavy_atoms"] for s in committed]),
        "qed_pass_all_commits": sum(cache[s]["qed"] >= THRESHOLDS["qed_min"] for s in committed),
        "sa_pass_all_commits": sum(cache[s]["sa"] <= THRESHOLDS["sa_max"] for s in committed),
        "joint_pass_all_commits": sum(cache[s]["quality_pass"] for s in committed),
    }
    if (
        abs(100 * result["quality_official_emitted_per_attempt"] - row["official"]["quality"])
        > 1e-9
    ):
        raise AssertionError("official quality disagrees with distinct emitted count")
    if len(committed) != row["committed_chemically_valid"]:
        raise AssertionError("committed chemical validity lost")
    return result


def _aggregate(prompts: dict[str, dict]) -> dict:
    summed = {
        key: sum(values[key] for values in prompts.values())
        for key in (
            "attempts",
            "committed_valid",
            "committed_fragment_preserving",
            "committed_interfaces_covered",
            "unique_valid_committed",
            "emitted_task_success",
            "unique_valid_emitted",
            "distinct_qualified_committed",
            "distinct_qualified_emitted",
            "qed_pass_all_commits",
            "sa_pass_all_commits",
            "joint_pass_all_commits",
        )
    }
    q = summed["distinct_qualified_committed"]
    summed.update(
        refusal_counts={
            key: sum(x["refusal_counts"][key] for x in prompts.values())
            for key in next(iter(prompts.values()))["refusal_counts"]
        },
        quality_committed_per_attempt=q / summed["attempts"],
        quality_committed_per_commit=q / summed["committed_valid"]
        if summed["committed_valid"]
        else None,
        quality_committed_per_unique_valid=q / summed["unique_valid_committed"]
        if summed["unique_valid_committed"]
        else None,
        quality_official_emitted_per_attempt=(
            summed["distinct_qualified_emitted"] / summed["attempts"]
        ),
        per_prompt_mean_official_quality=(
            sum(x["quality_official_emitted_per_attempt"] for x in prompts.values()) / len(prompts)
        ),
        per_prompt_mean_official_diversity=(
            sum(x["official_metrics"]["diversity"] for x in prompts.values()) / len(prompts)
        ),
        per_prompt_mean_official_uniqueness=(
            sum(x["official_metrics"]["uniqueness"] for x in prompts.values()) / len(prompts)
        ),
        per_prompt_mean_official_validity=(
            sum(x["official_metrics"]["validity"] for x in prompts.values()) / len(prompts)
        ),
        mean_qed_all_commits=(
            sum(
                x["mean_qed_all_commits"] * x["committed_valid"]
                for x in prompts.values()
                if x["committed_valid"]
            )
            / summed["committed_valid"]
        ),
        mean_sa_all_commits=(
            sum(
                x["mean_sa_all_commits"] * x["committed_valid"]
                for x in prompts.values()
                if x["committed_valid"]
            )
            / summed["committed_valid"]
        ),
        mean_heavy_atoms_all_commits=(
            sum(
                x["mean_heavy_atoms_all_commits"] * x["committed_valid"]
                for x in prompts.values()
                if x["committed_valid"]
            )
            / summed["committed_valid"]
        ),
    )
    return summed


def main() -> None:
    RDLogger.DisableLog("rdApp.warning")
    _official_module_path()  # Verify the previously frozen upstream files.
    summary_path = OUTPUT / "summary.json"
    # A predeclared gate can correctly fail before publishing summary.json.  The
    # 54 complete, hash-bound units remain evidence; never manufacture a passing
    # summary or silently skip failed prompts to analyze them.
    unit_paths = [
        _unit_path(OUTPUT, arm, task, drug)
        for arm, _ in ARMS
        for task, drugs in TASK_DRUGS
        for drug in drugs
    ]
    cache: dict[str, dict] = {}
    unit_hashes: dict[str, str] = {}
    raw_units: dict[tuple[str, str, str], dict] = {}
    by_arm: dict[str, dict[str, dict[str, dict]]] = {}
    identity = None
    for path in unit_paths:
        unit_name = str(path.relative_to(ROOT))
        unit_hashes[unit_name] = _sha256(path)
        unit = json.loads(path.read_text())
        if identity is None:
            identity = unit["identity"]
        if unit["identity"] != identity:
            raise AssertionError(f"identity mismatch: {unit_name}")
        arm, task, drug = unit["arm"], unit["task"], unit["drug"]
        raw_units[(arm, task, drug)] = unit
        row = unit["result"]["per_drug"][drug][0]
        by_arm.setdefault(arm, {}).setdefault(task, {})[drug] = _prompt(row, cache)
    if len(unit_hashes) != 54:
        raise AssertionError(f"expected 54 units, got {len(unit_hashes)}")
    aggregates = {
        arm: {task: _aggregate(prompts) for task, prompts in by_task.items()}
        for arm, by_task in by_arm.items()
    }
    if sum(x["attempts"] for tasks in aggregates.values() for x in tasks.values()) != 1080:
        raise AssertionError("held attempt count differs from frozen contract")
    routing_identity = {}
    for task, drugs in TASK_DRUGS:
        routing_identity[task.value] = {}
        for drug in drugs:
            rows = {
                arm: raw_units[(arm, task.value, drug)]["result"]["per_drug"][drug][0]
                for arm, _ in ARMS
            }
            count = len(rows["single_interface_policy"]["attachment_spec"]["interfaces"])
            candidate = rows["single_interface_policy"]["attempt_records"]
            strict_equal = candidate == rows["permanent_restriction"]["attempt_records"]
            release_equal = candidate == rows["global_release"]["attempt_records"]
            routing_identity[task.value][drug] = {
                "declared_interface_count": count,
                "strict_attempt_equal": strict_equal,
                "global_release_attempt_equal": release_equal,
            }
            if count != 1 and not strict_equal:
                raise AssertionError(f"multi/zero-interface routing changed: {task.value}/{drug}")
            if count == 0 and not release_equal:
                raise AssertionError(f"zero-interface no-op changed: {task.value}/{drug}")
    try:
        _summarize(raw_units)
    except AssertionError as error:
        frozen_gate = {"passed": False, "failure": str(error)}
    else:
        frozen_gate = {"passed": True, "failure": None}
    if summary_path.exists() != frozen_gate["passed"]:
        raise AssertionError("published summary status disagrees with frozen gate")
    artifact = {
        "schema": "fragment_single_interface_quality_denominators_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "held_identity": identity,
        "held_summary_sha256": _sha256(summary_path) if summary_path.exists() else None,
        "frozen_gate": frozen_gate,
        "routing_identity": routing_identity,
        "unit_sha256": unit_hashes,
        "analysis_source_sha256": _sha256(Path(__file__)),
        "rdkit_version": rdBase.rdkitVersion,
        "thresholds": THRESHOLDS,
        "denominator_definition": {
            "published_official": "distinct qualifying task-success emissions / attempted samples, per prompt",
            "committed_per_attempt": "distinct qualifying committed endpoints / attempted samples, per prompt",
            "committed_per_commit": "distinct qualifying committed endpoints / all valid commits, per prompt",
            "committed_per_unique_valid": "distinct qualifying committed endpoints / unique valid commits, per prompt",
        },
        "per_prompt": by_arm,
        "aggregate": aggregates,
        "unique_molecule_properties": cache,
    }
    path = OUTPUT / "quality_denominators.json"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(artifact, indent=2, sort_keys=True))
    temporary.replace(path)
    for arm, by_task in aggregates.items():
        for task, x in by_task.items():
            print(
                f"{arm}/{task}: committed-qualified {x['distinct_qualified_committed']}"
                f"/{x['attempts']} attempts, /{x['committed_valid']} commits, "
                f"/{x['unique_valid_committed']} unique; official emitted-qualified "
                f"{x['distinct_qualified_emitted']}/{x['attempts']}; "
                f"QED {x['mean_qed_all_commits']:.3f}, SA {x['mean_sa_all_commits']:.3f}"
            )
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
