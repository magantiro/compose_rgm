"""Build the 11-task answer-known PMO teacher-route atlas.

TASK 1 of the atlas programme.  Recovers, for every recorded teacher route:
source, destination, the exact program, structural checkpoints, replay status
under the CURRENT production executor, and the evaluator provenance that is (or
is not) attached to it.

The atlas is a DEVELOPMENT_INFORMED_DIAGNOSTIC challenge set.  It makes zero
objective calls: no score is computed, copied or invented here.  Recorded
scores found in the source artifacts are reported with their evaluator
provenance, or marked unknown.

Usage
-----
    KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 PYTHONPATH=src \
      python scripts/pmo_atlas_build.py --repo-root . \
        --output diagnostics/pmo_atlas_v1/atlas.json
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

from compose_v4.experiments.pmo_atlas_routes import (
    ATLAS_ARTIFACTS,
    ATLAS_DISTILLATION_RESULT,
    DEFAULT_CHECKPOINT_POSITIONS,
    DEVELOPMENT_INFORMED_LABEL,
    REGIME_STATEMENT,
    AtlasRoute,
    file_sha256,
    lineage_census,
    load_atlas,
    payload_sha256,
    replay_route,
    route_checkpoints,
)


def _revision(repo_root: Path) -> dict[str, Any]:
    def _git(*args: str) -> str | None:
        try:
            out = subprocess.run(
                ["git", *args],
                cwd=repo_root,
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
        except (FileNotFoundError, OSError, subprocess.SubprocessError):
            return None
        return out.stdout.strip() if out.returncode == 0 else None

    return {
        "commit": _git("rev-parse", "HEAD"),
        "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "worktree_dirty": bool(_git("status", "--porcelain")),
    }


def _software() -> dict[str, Any]:
    import numpy
    import rdkit

    software = {
        "python": platform.python_version(),
        "rdkit": rdkit.__version__,
        "numpy": numpy.__version__,
        "platform": platform.platform(),
    }
    try:
        import torch

        software["torch"] = torch.__version__
    except ImportError:
        software["torch"] = None
    return software


def _executor_identity() -> dict[str, Any]:
    from compose_v4.rewrite.action_codec_v4 import (
        codec_implementation_hash,
        supported_executor_rules,
    )

    return {
        "codec_implementation_sha256": codec_implementation_hash(),
        "supported_executor_rules": list(supported_executor_rules()),
        "rewrite_system": "editing_v2_semantic_rewrite_system",
    }


def _distillation_census(repo_root: Path) -> dict[str, Any]:
    """The frozen export's own census, reported for contrast, not as authority."""

    path = repo_root / ATLAS_DISTILLATION_RESULT
    if not path.is_file():
        return {"present": False}
    payload = json.loads(path.read_text())["payload"]
    summary = payload["summary"]
    return {
        "present": True,
        "file_sha256": file_sha256(path),
        "decision": payload["decision"],
        "limitations": payload["limitations"],
        "new_oracle_calls": payload["costs"]["new_oracle_calls"],
        "route_instances": summary["route_instances"],
        "unique_exact_traces": summary["unique_exact_traces"],
        "lineages": summary["lineages"],
        "task_route_census": summary["task_route_census"],
        "runtime_supported_route_instances": summary["runtime_supported_route_instances"],
        "runtime_supported_definition": (
            "len(route.actions) <= 32 in tools/pmo_route_distillation.py:362 -- a route "
            "LENGTH threshold, not an executor-support predicate"
        ),
        "executor_rule_counts": summary["executor_rule_counts"],
        "produced_under": payload["implementation"],
    }


#: The sibling result artifacts that carry the recorded objective values.  None
#: of the six route artifacts contains a score for a non-perindopril task; the
#: scores live here, beside the evaluator contract that produced them.
SCORE_ARTIFACTS: tuple[str, ...] = (
    "diagnostics/pmo_target_program_wave/result.json",
    "diagnostics/pmo_property_program_wave/result.json",
    "diagnostics/pmo_property_panel_refinement/result.json",
    "diagnostics/pmo_formula_median_panel_wave/result.json",
    "diagnostics/pmo_winner_program_curriculum/result.json",
)


def _recorded_scores(repo_root: Path) -> dict[str, Any]:
    """Recorded objective values with the evaluator identity that produced them.

    Nothing is recomputed, rescaled or carried across an evaluator change.  A
    recorded score is reported together with the evaluator contract beside it,
    so a reader can see that these numbers were produced under PyTDC 0.3.6 /
    RDKit 2024.03.5 and are therefore NOT comparable to a value computed under
    the current production PMO image.
    """

    per_task: dict[str, Any] = {}
    artifacts: dict[str, Any] = {}
    for relative in SCORE_ARTIFACTS:
        path = repo_root / relative
        if not path.is_file():
            artifacts[relative] = {"present": False}
            continue
        body = json.loads(path.read_text())
        body = body.get("payload", body)
        contract = body.get("contract", {})
        oracle_contract = contract.get("oracle", {})
        provenance = {
            "artifact": relative,
            "file_sha256": file_sha256(path),
            "software": body.get("software"),
            "oracle_contract": oracle_contract,
            "oracle_calls": body.get("oracle_calls"),
        }
        artifacts[relative] = provenance
        results = body.get("results")
        if results is None:
            # The perindopril winner result records its values at top level.
            results = {
                "perindopril_mpo": {
                    key: body[key]
                    for key in (
                        "auc_top10_official_10k",
                        "auc_top_ten",
                        "final_top10",
                        "final_top_ten_mean",
                        "best_score",
                        "oracle_calls",
                    )
                    if key in body
                }
            }
        for task, values in results.items():
            task_oracle = (contract.get("tasks", {}) or {}).get(task, {})
            entry = {
                "artifact": relative,
                "auc_top10_official_10k": values.get("auc_top10_official_10k")
                or values.get("auc_top_ten"),
                "final_top10": values.get("final_top10") or values.get("final_top_ten_mean"),
                "best_score": values.get("best_score"),
                "oracle_calls": values.get("oracle_calls"),
                "oracle_kind": task_oracle.get("oracle_kind"),
                "evaluator": {
                    "pytdc_version": oracle_contract.get("pytdc_version"),
                    "rdkit_version": oracle_contract.get("rdkit_version"),
                    "oracle_source_sha256": oracle_contract.get("oracle_source_sha256"),
                    "metric_budget": oracle_contract.get("metric_budget"),
                    "metric_frequency": oracle_contract.get("metric_frequency"),
                },
                "evaluator_identity_recorded": bool(
                    oracle_contract.get("pytdc_version")
                    and oracle_contract.get("oracle_source_sha256")
                ),
            }
            per_task.setdefault(task, []).append(entry)
    return {
        "per_task": per_task,
        "artifacts": artifacts,
        "comparability_warning": (
            "Recorded scores were produced under PyTDC 0.3.6 / RDKit 2024.03.5. The "
            "current production PMO container pins PyTDC 1.1.15 / RDKit 2023.9.6. "
            "These are different evaluator builds: recorded values must not be "
            "carried across, and gsk3b/jnk3 additionally used a local frozen forest "
            "(oracle_kind == 'frozen_tdc_forest'), not a PyTDC Oracle object."
        ),
    }


def _route_record(route: AtlasRoute, replay, checkpoints) -> dict[str, Any]:
    return {
        "task": route.task,
        "artifact": route.artifact,
        "lineage_id": route.lineage_id,
        "program_id": route.program_id,
        "role": route.role,
        "is_spine": route.is_spine,
        "source_smiles": route.source_smiles,
        "destination_smiles": route.destination_smiles,
        "destination_role": route.destination_role,
        "recorded_endpoint_smiles": route.recorded_endpoint_smiles,
        "primitive_steps": route.primitive_steps,
        "family_counts": dict(
            sorted(Counter(a["executor_rule"] for a in route.actions).items())
        ),
        "replay": {
            "exact": replay.exact,
            "steps_executed": replay.steps_executed,
            "steps_total": replay.steps_total,
            "endpoint_smiles": replay.endpoint_smiles,
            "endpoint_matches_record": replay.endpoint_matches_record,
            "endpoint_matches_destination": replay.endpoint_matches_destination,
            "intermediate_key_mismatches": list(replay.intermediate_key_mismatches),
            "out_of_support_indices": list(replay.out_of_support_indices),
            "failure": replay.failure,
        },
        "checkpoints": [
            {
                "label": c.label,
                "step_index": c.step_index,
                "fraction": round(c.fraction, 6),
                "remaining_steps": c.remaining_steps,
                "smiles": c.smiles,
                "heavy_atoms": c.heavy_atoms,
            }
            for c in checkpoints
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--output", default="diagnostics/pmo_atlas_v1/atlas.json")
    parser.add_argument(
        "--with-checkpoint-states",
        action="store_true",
        help="also persist the full 48-slot checkpoint states (large)",
    )
    args = parser.parse_args(argv)

    repo_root = Path(args.repo_root).resolve()
    started = time.time()
    dossier = load_atlas(repo_root)

    from compose_v4.rewrite.kernel import editing_v2_semantic_rewrite_system

    system = editing_v2_semantic_rewrite_system()

    records: list[dict[str, Any]] = []
    checkpoint_states: dict[str, Any] = {}
    for route in dossier.routes:
        replay = replay_route(route, system=system)
        checkpoints: tuple = ()
        if replay.exact:
            checkpoints = route_checkpoints(route, system=system)
            if args.with_checkpoint_states:
                for c in checkpoints:
                    checkpoint_states[f"{c.program_id}::{c.label}"] = c.state
        records.append(_route_record(route, replay, checkpoints))

    exact = sum(1 for r in records if r["replay"]["exact"])
    per_task: dict[str, Any] = {}
    for task in dossier.tasks:
        task_records = [r for r in records if r["task"] == task]
        spine = next((r for r in task_records if r["is_spine"]), None)
        per_task[task] = {
            "programs": len(task_records),
            "replayed_exact": sum(1 for r in task_records if r["replay"]["exact"]),
            "spine_program_id": spine["program_id"] if spine else None,
            "spine_primitive_steps": spine["primitive_steps"] if spine else None,
            "source_smiles": task_records[0]["source_smiles"],
            "destination_smiles": task_records[0]["destination_smiles"],
            "destination_role": task_records[0]["destination_role"],
        }

    source_counts = Counter(r["source_smiles"] for r in records)
    payload: dict[str, Any] = {
        "schema_version": "pmo_atlas_v1",
        "information_regime": DEVELOPMENT_INFORMED_LABEL,
        "information_regime_statement": REGIME_STATEMENT,
        "prohibited_use": (
            "Not a training distribution, prior, library or initialization for any "
            "scored no-prescreen PMO run."
        ),
        "new_oracle_calls": 0,
        "new_docking_calls": 0,
        "inputs": {
            a.relative_path: {
                "file_sha256": dossier.artifact_hashes.get(a.relative_path),
                "kind": a.kind,
            }
            for a in ATLAS_ARTIFACTS
        },
        "code_revision": _revision(repo_root),
        "software": _software(),
        "executor": _executor_identity(),
        "checkpoint_positions": [
            {"label": label, "fraction": fraction}
            for label, fraction in DEFAULT_CHECKPOINT_POSITIONS
        ],
        "summary": {
            "tasks": len(dossier.tasks),
            "routes": len(records),
            "spines": sum(1 for r in records if r["is_spine"]),
            "replayed_exact": exact,
            "replayed_inexact": len(records) - exact,
            "distinct_sources": len(source_counts),
            "shared_source_smiles": source_counts.most_common(1)[0][0],
            "shared_source_route_share": round(
                source_counts.most_common(1)[0][1] / len(records), 6
            ),
            "task_sections_on_shared_source": len(
                {
                    r["task"]
                    for r in records
                    if r["source_smiles"] == source_counts.most_common(1)[0][0]
                }
            ),
            "distinct_destinations": len(
                {r["destination_smiles"] for r in records if r["destination_smiles"]}
            ),
            "elapsed_seconds": round(time.time() - started, 3),
        },
        "lineage_census": lineage_census(dossier),
        "per_task": per_task,
        "recorded_scores": _recorded_scores(repo_root),
        "frozen_distillation_census": _distillation_census(repo_root),
        "routes": records,
    }
    if args.with_checkpoint_states:
        payload["checkpoint_states"] = checkpoint_states

    document = {"payload": payload, "payload_sha256": payload_sha256(payload)}
    output = repo_root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(document, indent=1, sort_keys=True))

    print(f"tasks={len(dossier.tasks)} routes={len(records)} exact={exact}")
    print(f"distinct sources={len(source_counts)}")
    for task, info in sorted(per_task.items()):
        print(
            f"  {task:26s} programs={info['programs']:3d} "
            f"exact={info['replayed_exact']:3d} spine_steps={info['spine_primitive_steps']}"
        )
    print(f"wrote {output} sha256={document['payload_sha256'][:16]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
