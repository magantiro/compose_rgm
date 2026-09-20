"""Seal the zero-oracle hybrid top-three plus Dynamic-v0 controller lock."""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import platform
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import rdkit

from compose_v4.control.docking_value import identity
from compose_v4.control.hybrid_top3_v0_controller import freeze_cell_schedule
from compose_v4.control.macro_archive_integration import seed_sequences
from compose_v4.control.target_conditioned_utility_selector import UtilityRanker

CONTRACT_SCHEMA = "t4_hybrid_top3_v0_controller_contract_v1"
CANDIDATE_LOCK_SCHEMA = "t4_hybrid_top3_v0_candidate_lock_v1"
ADMISSION_LOCK_SCHEMA = "t4_hybrid_top3_v0_admission_lock_v1"
RESULT_SCHEMA = "t4_hybrid_top3_v0_result_v1"


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_bytes(root: Path, revision: str, path: str) -> bytes:
    try:
        return subprocess.check_output(
            ["git", "show", f"{revision}:{path}"], cwd=root, stderr=subprocess.PIPE
        )
    except subprocess.CalledProcessError as error:
        detail = error.stderr.decode(errors="replace").strip()
        raise ValueError(f"bound Git input unavailable: {revision}:{path}: {detail}") from error


def _decode_json(raw: bytes, path: str) -> dict[str, Any]:
    if path.endswith(".gz"):
        raw = gzip.decompress(raw)
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def _bound_json(root: Path, spec: dict[str, Any]) -> dict[str, Any]:
    raw = _git_bytes(root, spec["revision"], spec["path"])
    if sha256_bytes(raw) != spec["sha256"]:
        raise ValueError(f"bound physical input changed: {spec['path']}")
    value = _decode_json(raw, spec["path"])
    expected = spec.get("payload_sha256")
    if expected is not None:
        payload = value.get("payload")
        claimed = value.get("payload_sha256", value.get("contract_sha256"))
        if not isinstance(payload, dict) or claimed != expected or identity(payload) != expected:
            raise ValueError(f"bound payload seal changed: {spec['path']}")
    return value


def load_contract(path: Path) -> dict[str, Any]:
    envelope = json.loads(path.read_text())
    payload = envelope.get("payload")
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != CONTRACT_SCHEMA
        or envelope.get("contract_sha256") != identity(payload)
    ):
        raise ValueError(f"invalid hybrid-controller contract: {path}")
    authority = payload["authority"]
    if any(
        authority[key] != expected
        for key, expected in {
            "oracle_calls_authorized": 0,
            "docking_calls_authorized": 0,
            "modal_launches_authorized": 0,
            "live_run_access_authorized": False,
            "model_fitting_authorized": False,
            "candidate_generation_authorized": False,
        }.items()
    ):
        raise ValueError("hybrid-controller lock must remain zero-oracle")
    if (
        payload["selection"]["target_conditioned_arm"]
        != "rejected_by_authoritative_v4_result_and_not_used"
    ):
        raise ValueError("target-conditioned arm must remain rejected")
    return payload


def _verify_revision_bindings(root: Path, contract: dict[str, Any]) -> None:
    inputs = contract["immutable_inputs"]
    for name in (
        "v4_utility_implementation",
        "v4_utility_result_revision",
        "macro_archive_integration",
        "qualification_v2",
    ):
        spec = inputs[name]
        tree = subprocess.check_output(
            ["git", "show", "-s", "--format=%T", spec["revision"]],
            cwd=root,
            text=True,
        ).strip()
        if tree != spec["tree"]:
            raise ValueError(f"bound Git tree changed: {name}")
    integration = inputs["macro_archive_integration"]
    for path_key, hash_key in (
        ("implementation_path", "implementation_sha256"),
        ("dynamic_v0_path", "dynamic_v0_sha256"),
    ):
        raw = _git_bytes(root, integration["revision"], integration[path_key])
        if sha256_bytes(raw) != integration[hash_key]:
            raise ValueError(f"bound integration implementation changed: {path_key}")


def _candidate_index(generator: dict[str, Any]) -> dict[tuple[str, str], tuple[dict, dict]]:
    payload = generator["payload"]
    if (
        payload.get("schema_version") != "t4_compositional_structural_subgoal_candidate_lock_v1"
        or payload.get("teacher_fields_present") is not False
        or payload.get("task_identity_present") is not False
        or payload.get("new_oracle_calls") != 0
    ):
        raise ValueError("baseline generator lock violates the frozen boundary")
    index: dict[tuple[str, str], tuple[dict, dict]] = {}
    for fold in payload["folds"]:
        for case in fold["cases"]:
            policies = {policy["policy_id"]: policy["candidates"] for policy in case["policies"]}
            learned = policies.get("balanced_joint_autoregressive")
            if learned is None or len(learned) != 128:
                raise ValueError("baseline learned candidate pool is not 128-wide")
            for rank, raw in enumerate(learned, start=1):
                candidate_id = identity(raw)
                key = (case["source_case_id"], candidate_id)
                if key in index:
                    raise ValueError("generator candidate identity repeats")
                index[key] = (case["source_state"], raw)
    return index


def _eligibility_rows(
    eligibility: dict[str, Any],
    *,
    cells: list[str],
    generator: dict[str, Any],
    generator_physical_sha256: str,
) -> dict[str, list[dict[str, Any]]]:
    payload = eligibility["payload"]
    if payload.get("schema_version") != "t4_compositional_generator_utility_eligibility_ledger_v1":
        raise ValueError("unexpected eligibility-ledger schema")
    expected = {
        "active_atoms_less_than_or_equal_to": 40,
        "qed_strictly_greater_than": 0.6,
        "sa_strictly_less_than": 4.0,
        "source_similarity_strictly_greater_than": 0.4,
    }
    if any(payload["eligibility"].get(key) != value for key, value in expected.items()):
        raise ValueError("strict eligibility contract changed")
    baseline = payload["candidate_locks"]["baseline"]
    if baseline["sha256"] != generator_physical_sha256:
        raise ValueError("eligibility ledger binds a different baseline lock")
    if baseline["payload_sha256"] != generator["payload_sha256"]:
        raise ValueError("eligibility and generator payload identities differ")
    generator_index = _candidate_index(generator)
    by_cell: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in payload["candidates"]:
        if (
            row.get("generator_revision") == "baseline"
            and row.get("arm_id") == "baseline_learned"
            and row.get("policy_id") == "balanced_joint_autoregressive"
            and row.get("cell") in cells
        ):
            by_cell[row["cell"]].append(row)
    if set(by_cell) != set(cells):
        raise ValueError("eligibility ledger lacks a declared cell")
    for cell, rows in sorted(by_cell.items()):
        rows.sort(key=lambda row: (int(row["rank"]), row["candidate_id"]))
        if len(rows) != 128 or [row["rank"] for row in rows] != list(range(1, 129)):
            raise ValueError(f"eligibility pool is not a complete 128-pool: {cell}")
        for row in rows:
            key = (row["source_case_id"], row["candidate_id"])
            source_state, raw = generator_index.get(key, (None, None))
            if (
                source_state is None
                or identity(raw) != row["candidate_id"]
                or source_state != row["source_state"]
                or raw["endpoint_state"] != row["endpoint_state"]
                or raw["actions"] != row["actions"]
                or raw["realization"] != row["realization"]
                or row["eligible"] != (not row["eligibility_exclusion_reasons"])
            ):
                raise ValueError(f"eligibility/generator exact join failed: {cell}/{row['rank']}")
    return by_cell


def _cell_seed(root_seed: int, cell: str) -> int:
    digest = hashlib.sha256(f"{root_seed}:{cell}".encode()).digest()
    return int.from_bytes(digest[:4], "big")


def _publish(path: Path, payload: dict[str, Any]) -> str:
    envelope = {"payload": payload, "payload_sha256": identity(payload)}
    content = json.dumps(envelope, sort_keys=True, indent=2, allow_nan=False).encode() + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        handle.write(content)
        temporary = Path(handle.name)
    os.replace(temporary, path)
    return sha256_bytes(content)


def run(root: Path, contract_path: Path, output: Path) -> dict[str, str]:
    dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True).strip()
    if dirty:
        raise RuntimeError("hybrid-controller lock requires clean committed source")
    code_revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()
    code_tree = subprocess.check_output(
        ["git", "show", "-s", "--format=%T", "HEAD"], cwd=root, text=True
    ).strip()
    contract = load_contract(contract_path)
    _verify_revision_bindings(root, contract)
    inputs = contract["immutable_inputs"]
    utility_contract = _bound_json(root, inputs["v4_utility_contract"])
    if utility_contract.get("contract_sha256") != inputs["v4_utility_contract"]["payload_sha256"]:
        raise ValueError("V4 utility contract identity changed")
    utility_result = _bound_json(root, inputs["v4_utility_result"])
    if (
        utility_result.get("fit_performed") is not True
        or utility_result.get("target_conditioning_improved_primary_metrics") is not False
        or utility_result.get("new_oracle_calls") != 0
        or utility_result.get("new_docking_calls") != 0
    ):
        raise ValueError("authoritative V4 negative result changed")
    generator = _bound_json(root, inputs["baseline_generator_lock"])
    eligibility = _bound_json(root, inputs["strict_eligibility_ledger"])
    qualification = _bound_json(root, inputs["qualification_v2"])
    if qualification["payload"].get("cells") != contract["cells"]:
        raise ValueError("qualification-v2 cell order changed")
    if (
        qualification["payload"]["prospective_design"]["matched_summary_calls"]
        != contract["continuation"]["report_calls"]
    ):
        raise ValueError("qualification-v2 reporting checkpoints changed")
    checkpoints: dict[int, dict[str, Any]] = {}
    checkpoint_hashes: dict[str, str] = {}
    for spec in inputs["v4_fold_checkpoints"]:
        value = _bound_json(root, spec)
        fold = int(spec["fold"])
        if (
            value.get("fold") != fold
            or value.get("contract_sha256") != inputs["v4_utility_contract"]["payload_sha256"]
        ):
            raise ValueError("utility checkpoint fold or contract changed")
        ranker = UtilityRanker.from_checkpoint(value["target_blind"])
        if ranker.conditioned:
            raise ValueError("target-blind checkpoint is unexpectedly conditioned")
        checkpoints[fold] = value
        checkpoint_hashes[str(fold)] = spec["sha256"]
    rows_by_cell = _eligibility_rows(
        eligibility,
        cells=contract["cells"],
        generator=generator,
        generator_physical_sha256=inputs["baseline_generator_lock"]["sha256"],
    )

    schedules = []
    for cell in contract["cells"]:
        fold = int(cell.rsplit("_", 1)[1])
        eligible = [row for row in rows_by_cell[cell] if row["eligible"]]
        schedule = freeze_cell_schedule(
            cell=cell,
            candidate_rows=eligible,
            ranker=UtilityRanker.from_checkpoint(checkpoints[fold]["target_blind"]),
            checkpoint_fold=fold,
            report_calls=contract["continuation"]["report_calls"],
        )
        schedule["candidate_pool_rows"] = len(rows_by_cell[cell])
        schedule["ineligible_candidate_rows"] = len(rows_by_cell[cell]) - len(eligible)
        schedule["checkpoint_sha256"] = checkpoint_hashes[str(fold)]
        schedule["rng_root_seed"] = _cell_seed(contract["continuation"]["rng_root_seed"], cell)
        schedule["rng_namespaces"] = seed_sequences(schedule["rng_root_seed"])
        schedules.append(schedule)

    supported = [row for row in schedules if row["status"] == "three_macro_calls_then_dynamic_v0"]
    abstained = [row for row in schedules if row["status"] != "three_macro_calls_then_dynamic_v0"]
    macro_calls = sum(len(row["initial_macro_calls"]) for row in schedules)
    v0_calls = sum(int(row["dynamic_v0_calls_through_100"]) for row in schedules)
    common = {
        "contract": {
            "path": str(contract_path.relative_to(root)),
            "physical_sha256": sha256_file(contract_path),
            "payload_sha256": identity(contract),
        },
        "input_provenance": contract["immutable_inputs"],
        "new_oracle_calls": 0,
        "new_docking_calls": 0,
        "modal_launches": 0,
        "live_run_accesses": 0,
        "producer": {
            "code_revision": code_revision,
            "code_tree": code_tree,
            "python": platform.python_version(),
            "rdkit": rdkit.__version__,
            "device": "cpu",
            "workers": 1,
        },
    }
    candidate_lock = {
        "schema_version": CANDIDATE_LOCK_SCHEMA,
        **common,
        "cells": schedules,
        "selection": contract["selection"],
        "target_conditioned_arm_used": False,
        "teacher_route_or_endpoint_used": False,
        "scoring_status": "UNAUTHORIZED_NOT_LAUNCHED",
    }
    candidate_physical = _publish(output / contract["outputs"]["candidate_lock"], candidate_lock)

    admission_cells = []
    for row in schedules:
        admission_cells.append(
            {
                "cell": row["cell"],
                "status": row["status"],
                "initial_macro_calls": [
                    {
                        "call": call["call"],
                        "candidate_id": call["candidate_id"],
                        "canonical_smiles": call["canonical_smiles"],
                        "selection_role": call["selection_role"],
                    }
                    for call in row["initial_macro_calls"]
                ],
                "conditional_admission": "admit_each_finite_scored_initial_candidate_to_one_run_local_archive_in_charged_call_order; retain_but_do_not_impute_or_replace_a_charged_failure",
                "archive_count_per_cell": 1,
                "dynamic_v0_start_call": row["dynamic_v0_start_call"],
                "dynamic_v0_calls_through_100": row["dynamic_v0_calls_through_100"],
                "rng_root_seed": row["rng_root_seed"],
                "rng_namespaces": row["rng_namespaces"],
            }
        )
    admission_lock = {
        "schema_version": ADMISSION_LOCK_SCHEMA,
        **common,
        "candidate_lock": {
            "physical_sha256": candidate_physical,
            "payload_sha256": identity(candidate_lock),
        },
        "cells": admission_cells,
        "future_budget_if_separately_authorized": {
            "cells": 5,
            "charged_calls_per_cell": 100,
            "maximum_total_charged_calls": 500,
            "frozen_macro_calls": macro_calls,
            "unchanged_dynamic_v0_calls": v0_calls,
            "maximum_concurrent_single_cpu_workers": 5,
            "gpu": False,
            "automatic_retry": False,
            "confirmation_calls": 0,
            "plateau_stopping": False,
        },
        "scoring_status": "UNAUTHORIZED_NOT_LAUNCHED",
    }
    admission_physical = _publish(output / contract["outputs"]["admission_lock"], admission_lock)

    result = {
        "schema_version": RESULT_SCHEMA,
        **common,
        "status": "complete_zero_oracle_support_conditioned_lock",
        "candidate_lock": {
            "physical_sha256": candidate_physical,
            "payload_sha256": identity(candidate_lock),
        },
        "admission_lock": {
            "physical_sha256": admission_physical,
            "payload_sha256": identity(admission_lock),
        },
        "summary": {
            "declared_cells": len(schedules),
            "three_candidate_macro_supported_cells": len(supported),
            "macro_support_abstention_cells": len(abstained),
            "all_five_cells_have_three_eligible_independent_candidates": len(supported) == 5,
            "supported_cells": [row["cell"] for row in supported],
            "abstention_cells": [row["cell"] for row in abstained],
            "frozen_initial_macro_calls": macro_calls,
            "unchanged_dynamic_v0_calls_if_full_five_cell_run_authorized": v0_calls,
            "total_calls_if_full_five_cell_run_authorized": macro_calls + v0_calls,
        },
        "cell_census": [
            {
                "cell": row["cell"],
                "pool": row["candidate_pool_rows"],
                "eligible": row["eligible_candidate_rows"],
                "distinct_eligible": row["distinct_eligible_endpoints"],
                "status": row["status"],
                "dynamic_v0_start_call": row["dynamic_v0_start_call"],
            }
            for row in schedules
        ],
        "future_scored_call_authorization_requirements": {
            "current_authority": "none",
            "required_exact_bindings": [
                "candidate_lock physical and payload SHA-256",
                "admission_lock physical and payload SHA-256",
                "clean launch Git revision and tree",
                "unchanged Dynamic-v0 implementation identity",
                "five target receptor, box, binary, preparation and docking-seed identities",
                "durable per-call query ledger and restart contract",
            ],
            "maximum_total_charged_calls": 500,
            "calls_by_cell": 100,
            "maximum_single_cpu_workers": 5,
            "initial_locked_macro_calls": macro_calls,
            "dynamic_v0_calls": v0_calls,
            "no_retry_no_confirmation_no_plateau": True,
            "interpretation": "five-cell answer-known development with generic support-conditioned fallback; not a full benchmark or IVG superiority result",
        },
        "limitations": [
            "three of five cells have no exact-eligible endpoint in the bound baseline learned 128-pool and therefore use Dynamic-v0 from call one",
            "predicted structural and target-blind utility scores are zero-oracle ordering signals, not evidence of docking improvement",
            "the target-conditioned arm is rejected and absent",
        ],
    }
    result_physical = _publish(output / contract["outputs"]["result"], result)
    readme = (
        "# Hybrid top-three plus Dynamic-v0 zero-oracle lock\n\n"
        f"Supported macro bootstraps: {len(supported)}/5 ({', '.join(row['cell'] for row in supported)}).\n\n"
        f"Support abstentions with unchanged Dynamic-v0 from call 1: {len(abstained)}/5 "
        f"({', '.join(row['cell'] for row in abstained)}).\n\n"
        "No oracle, docking, Modal or live-run action occurred. A separate exact-hash launch authorization is required.\n"
    )
    (output / contract["outputs"]["readme"]).write_text(readme)
    return {
        "candidate_lock": candidate_physical,
        "admission_lock": admission_physical,
        "result": result_physical,
        "readme": sha256_file(output / contract["outputs"]["readme"]),
    }


__all__ = ["load_contract", "run"]
