"""Run the grouped zero-oracle PMO local route-policy comparison."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import platform
import resource
import shutil
import subprocess
import tempfile
from pathlib import Path
from time import perf_counter

import numpy as np

from compose_v4.control.docking_value import identity
from compose_v4.experiments.pmo_route_policy_quality import (
    CHECKPOINT_SCHEMA,
    POLICIES,
    SCHEMA,
    PolicyConfig,
    action_metrics,
    component_metrics,
    decisions_from_dataset,
    evaluate_fold,
    factor_examples,
    fit_fold,
    predeclared_family_folds,
    runtime_checkpoint_leakage,
)

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = Path("configs/pmo_route_policy_quality_v1.json")
DEFAULT_OUTPUT = Path("diagnostics/pmo_route_policy_quality/attempt_1")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_contract(root: Path) -> dict:
    path = root / CONTRACT
    envelope = json.loads(path.read_text())
    payload = envelope.get("payload")
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != "pmo_route_policy_quality_contract_v1"
        or envelope.get("contract_sha256") != identity(payload)
    ):
        raise ValueError(f"invalid PMO route-policy quality contract: {path}")
    return payload


def _load_envelope(path: Path, expected_sha256: str, expected_payload: str) -> dict:
    observed = sha256_file(path)
    if observed != expected_sha256:
        raise ValueError(
            f"sealed PMO policy input hash mismatch for {path}: "
            f"expected {expected_sha256}, got {observed}"
        )
    raw = (
        gzip.decompress(path.read_bytes()).decode()
        if path.suffix == ".gz"
        else path.read_text()
    )
    envelope = json.loads(raw)
    payload = envelope.get("payload")
    if (
        not isinstance(payload, dict)
        or envelope.get("payload_sha256") != identity(payload)
        or envelope["payload_sha256"] != expected_payload
    ):
        raise ValueError(f"sealed PMO policy input envelope mismatch: {path}")
    return payload


def _json_ready(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_ready(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(child) for child in value]
    return value


def _publish(path: Path, payload: dict, *, compressed: bool = False) -> None:
    ready = _json_ready(payload)
    envelope = {"payload": ready, "payload_sha256": identity(ready)}
    raw = json.dumps(envelope, sort_keys=True, separators=(",", ":")) + "\n"
    if compressed:
        path.write_bytes(gzip.compress(raw.encode(), mtime=0))
    else:
        path.write_text(raw)


def _teacher_values(dataset: dict) -> set[str]:
    result = set()
    for row in dataset["training_provenance"]:
        result.update((str(row["trace_identity"]), str(row["lineage_identity"])))
        for member in row["members"]:
            result.update(
                str(value) for value in member.values() if isinstance(value, str)
            )
    for route in dataset["route_manifest"]:
        result.add(str(route["endpoint_identity"]))
    return result


def _validate_census(contract: dict, dataset: dict, audit: dict, decisions) -> None:
    expected = contract["expected"]
    observed = {
        "decisions": len(decisions),
        "runtime_supported_decisions": sum(row.runtime_supported for row in decisions),
        "local_only_decisions": sum(not row.runtime_supported for row in decisions),
        "unique_traces": len({row.trace_identity for row in decisions}),
        "runtime_supported_unique_traces": sum(
            row["runtime_complete_route_supported"] for row in dataset["route_manifest"]
        ),
        "lineages": len({row.lineage_identity for row in decisions}),
        "task_families": len({row.task_family for row in decisions}),
        "compound_stages": int(audit["summary"]["recognized_compound_stages"]),
        "created_handle_dependent_decisions": sum(
            bool(row.dependencies) for row in decisions
        ),
        "created_output_decisions": sum(row.creates_output for row in decisions),
    }
    if observed != expected:
        raise ValueError(
            f"sealed PMO route-policy census changed: expected {expected}, got {observed}"
        )
    if (
        audit["outputs"].get("training_dataset.json.gz")
        != contract["input"]["dataset_sha256"]
    ):
        raise ValueError(
            "route-distillation audit does not bind the policy input dataset"
        )


def _complete_abstention(dataset: dict, folds: tuple[dict, ...]) -> dict:
    by_family = {
        row["trace_identity"]: row["members"][0]["task_family"]
        for row in dataset["route_manifest"]
    }
    supported = {
        row["trace_identity"]
        for row in dataset["route_manifest"]
        if row["runtime_complete_route_supported"]
    }
    fold_rows = []
    for fold in folds:
        held_out = set(fold["held_out_task_families"])
        eligible = sum(by_family[trace] in held_out for trace in supported)
        fold_rows.append(
            {
                "fold": fold["fold"],
                "runtime_length_reference_traces": eligible,
                "policy_decodable_reference_traces": 0,
                "proposal_attempts": 0,
                "valid_complete_proposals": 0,
                "unique_complete_yield": 0,
                "exact_recall": 0.0,
                "transformation_equivalent_recall": 0.0,
                "coverage": 0.0,
                "precision": None,
                "shortfall": eligible,
            }
        )
    return {
        "status": "abstained_unsupported_by_sanitized_primitive_only_supervision",
        "reason": (
            "zero compound stages and no exact graph/action pairs remain for legal role "
            "rebinding, complete decoding or candidate-panel construction"
        ),
        "runtime_length_reference_traces": len(supported),
        "policy_decodable_reference_traces": 0,
        "proposal_attempts": 0,
        "valid_complete_proposals": 0,
        "unique_complete_yield": 0,
        "exact_recall": 0.0,
        "transformation_equivalent_recall": 0.0,
        "coverage": 0.0,
        "precision": None,
        "shortfall": len(supported),
        "folds": fold_rows,
        "long_route_local_decisions_counted_as_complete_recall": 0,
        "teacher_injection": False,
    }


def run(output: Path = ROOT / DEFAULT_OUTPUT, root: Path = ROOT) -> dict:
    if output.exists():
        raise ValueError(f"refusing to overwrite PMO route-policy output: {output}")
    began = perf_counter()
    contract = _load_contract(root)
    source = contract["input"]
    dataset = _load_envelope(
        root / source["dataset_path"],
        source["dataset_sha256"],
        source["dataset_payload_sha256"],
    )
    audit = _load_envelope(
        root / source["audit_path"],
        source["audit_sha256"],
        source["audit_payload_sha256"],
    )
    decisions = decisions_from_dataset(dataset)
    _validate_census(contract, dataset, audit, decisions)
    examples = factor_examples(decisions)
    folds = predeclared_family_folds(decisions, contract["folds"])
    model_config = PolicyConfig(
        maximum_selected_features_per_head=int(
            contract["models"]["maximum_selected_features_per_head"]
        ),
        uniform_probability_floor=float(
            contract["models"]["uniform_probability_floor"]
        ),
        centroid_distance_strength=float(
            contract["models"]["centroid_distance_strength"]
        ),
        minimum_scale=float(contract["models"]["minimum_scale"]),
    )
    runtime_folds, fold_results = [], []
    combined = {policy: [] for policy in POLICIES}
    fit_seconds = 0.0
    for fold in folds:
        fit_began = perf_counter()
        checkpoint = fit_fold(examples, set(fold["train_decisions"]), model_config)
        fit_seconds += perf_counter() - fit_began
        runtime_folds.append({"fold": fold["fold"], "checkpoint": checkpoint})
        policies = {}
        for policy, conditioned in zip(POLICIES, (False, True), strict=True):
            outcomes = evaluate_fold(
                checkpoint,
                examples,
                set(fold["test_decisions"]),
                conditioned=conditioned,
            )
            combined[policy].extend(outcomes)
            policies[policy] = {
                "components": component_metrics(outcomes),
                "factorized_actions": {
                    scope: action_metrics(outcomes, decisions, scope)
                    for scope in ("all", "runtime_supported", "local_only")
                },
            }
        fold_results.append(
            {
                "fold": fold["fold"],
                "held_out_task_families": fold["held_out_task_families"],
                "train_decisions": len(fold["train_decisions"]),
                "test_decisions": len(fold["test_decisions"]),
                "train_lineages": fold["train_lineages"],
                "test_lineages": fold["test_lineages"],
                "policies": policies,
            }
        )
    runtime = {
        "schema_version": CHECKPOINT_SCHEMA,
        "policy_role": "goal_free_hierarchical_local_action_ranker_not_complete_decoder",
        "folds": runtime_folds,
        "new_oracle_calls": 0,
    }
    leakage = runtime_checkpoint_leakage(runtime, _teacher_values(dataset))
    if leakage["forbidden_key_hits"] or leakage["forbidden_teacher_value_hits"]:
        raise RuntimeError(f"runtime checkpoint teacher-content leak: {leakage}")
    aggregate = {
        policy: {
            "components": component_metrics(tuple(rows)),
            "factorized_actions": {
                scope: action_metrics(tuple(rows), decisions, scope)
                for scope in ("all", "runtime_supported", "local_only")
            },
        }
        for policy, rows in combined.items()
    }
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()
    dirty = bool(
        subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=root, text=True
        ).strip()
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        runtime_path = temporary / contract["outputs"]["runtime_fold_checkpoints"]
        _publish(runtime_path, runtime, compressed=True)
        elapsed = perf_counter() - began
        raw_peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        peak_rss = int(raw_peak if platform.system() == "Darwin" else raw_peak * 1024)
        report = {
            "schema_version": SCHEMA,
            "decision": "local_policy_comparison_complete_complete_program_policy_unsupported",
            "contract_path": str(CONTRACT),
            "contract_sha256": sha256_file(root / CONTRACT),
            "inputs": {
                source["dataset_path"]: source["dataset_sha256"],
                source["audit_path"]: source["audit_sha256"],
            },
            "configuration": contract["models"],
            "census": contract["expected"],
            "folds": fold_results,
            "aggregate": aggregate,
            "complete_program_evaluation": _complete_abstention(dataset, folds),
            "hybrid_complete_candidate_ranker": {
                "status": "not_fit_no_legitimate_complete_candidate_panel",
                "teacher_injection": False,
            },
            "runtime_checkpoint_audit": leakage,
            "outputs": {
                contract["outputs"]["runtime_fold_checkpoints"]: sha256_file(
                    runtime_path
                )
            },
            "compute": {
                "device": "CPU",
                "workers": 1,
                "precision": "float64",
                "randomness": "none",
                "fit_seconds": fit_seconds,
                "total_seconds": elapsed,
                "peak_rss_bytes": peak_rss,
                "runtime_checkpoint_bytes": runtime_path.stat().st_size,
            },
            "implementation": {
                "revision": revision,
                "working_tree_dirty": dirty,
                "runner_sha256": sha256_file(Path(__file__)),
                "core_sha256": sha256_file(
                    root / "src/compose_v4/experiments/pmo_route_policy_quality.py"
                ),
                "python": platform.python_version(),
                "numpy": np.__version__,
            },
            "costs": {
                "new_oracle_calls": 0,
                "new_docking_calls": 0,
                "modal_launches": 0,
            },
            "limitations": [
                "All supervision is answer-known PMO development evidence.",
                "Task-family folds measure transfer to unseen families, not within-family interpolation.",
                "Local action likelihood does not establish complete executable program recovery.",
                "Long-route decisions are local-only and never count as complete-route support.",
                "No compound stage, complete decoder or complete candidate ranker was fit.",
            ],
        }
        _publish(temporary / contract["outputs"]["result"], report)
        temporary.replace(output)
    except Exception:
        shutil.rmtree(temporary)
        raise
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / DEFAULT_OUTPUT)
    result = run(parser.parse_args().output)
    print(
        json.dumps(
            {
                "decision": result["decision"],
                "aggregate": result["aggregate"],
                "complete_program_evaluation": result["complete_program_evaluation"],
                "compute": result["compute"],
            },
            indent=2,
        )
    )
