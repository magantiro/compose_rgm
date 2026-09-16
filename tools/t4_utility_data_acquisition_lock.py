#!/usr/bin/env python3
"""Build the score-blind, zero-oracle T4 utility data-acquisition lock."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from compose_v4.control.docking_value import identity
from compose_v4.experiments.t4_frozen_program_benchmark import strict_endpoint_scorer
from compose_v4.rewrite.kernel import canonical_state_key
from compose_v4.rewrite.trace_shard import decode_state
from tools import t4_target_conditioned_utility_selector as utility
from tools.t4_compositional_structural_subgoal_generator import _replay

CONTRACT = Path("configs/t4_utility_data_acquisition_lock_v1.json")
OUTPUT = Path("diagnostics/t4_utility_data_acquisition_lock/attempt_1")
PARP_CASE = "f060be55f96694dbcbe9af4d955a54b4a69b5ff40a249e2ecc915367e09d1740"
PARP_IDS = {
    "6ab68c2df5b637214cdb695ad1fa0b8302f5f17ad9fd38641f29f1355a4950fb",
    "fd4d754151f7312e8c6d4df7f79e83f77eaae6262f48d3943856138ffd40036c",
}
FA7_IDS = {
    "705e30bec99bf7ae6a51a7f862d6c4ff06bcb89f9f3964c78edbb5337b22af12",
    "d7c7b5625111f0c41fa24873e5dc4f0a3048f3d6a5670dc93f44268f645e3345",
}
RECEPTORS = {
    "fa7": "bfd705fb8220c52e1b447afce23a3810bc6deddb06983e98a2b0934b24ba092d",
    "parp1": "8d0891ddf915f51cf3108f39dd9dc01dfcf86be342c566021956afdd79eb4ad9",
}


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_bytes(root: Path, spec: Mapping[str, Any]) -> bytes:
    raw = subprocess.run(
        ["git", "show", f"{spec['git_revision']}:{spec['path']}"],
        cwd=root,
        check=True,
        capture_output=True,
    ).stdout
    observed = hashlib.sha256(raw).hexdigest()
    if observed != spec["sha256"]:
        raise ValueError(f"physical hash mismatch for {spec['path']}: {observed}")
    return raw


def _load(root: Path, spec: Mapping[str, Any], *, compressed: bool = False) -> Any:
    raw = _git_bytes(root, spec)
    value = json.loads(gzip.decompress(raw) if compressed else raw)
    if "payload_sha256" in spec:
        if value.get("payload_sha256") != spec["payload_sha256"]:
            raise ValueError(f"payload identity mismatch for {spec['path']}")
        if canonical_hash(value["payload"]) != spec["payload_sha256"]:
            raise ValueError(f"payload seal mismatch for {spec['path']}")
        return value["payload"]
    return value


def _evaluator(target: str) -> tuple[str, dict]:
    payload = {
        "schema_version": "t4_quickvina_endpoint_evaluator_v1",
        "target": target,
        "receptor_sha256": RECEPTORS[target],
        "qvina02_sha256": "f8ac045235025e98b15fd90aae6617edfdcc125081f72a5a5315db22be1f46e0",
        "box_definition": "BOXES in hash-bound modal_apps/genmol_t4_opt_app.py",
        "docking": {
            "cpu": 1,
            "exhaustiveness": 1,
            "ligand_preparation": "Open Babel --gen3d, unchanged production wrapper",
            "num_modes": 10,
        },
        "score_direction": "minimize",
        "score_unit": "kcal/mol as emitted by the bound QuickVina wrapper",
        "scoring_implementation_sha256": "9b635a9936493a5900b613cdce52f76c24efcf98f3f43b3b155448e0b1796bf0",
    }
    return identity(payload), payload


def _request(candidate: dict) -> dict:
    evaluator, evaluator_payload = _evaluator(candidate["target"])
    body = {
        "schema_version": "t4_docking_request_identity_v1",
        "target": candidate["target"],
        "canonical_smiles": candidate["canonical_smiles"],
        "docking_seed": 1701,
        "evaluator": evaluator,
        "evaluator_payload": evaluator_payload,
        "candidate_id": candidate["candidate_id"],
        "cell": candidate["cell"],
    }
    return {**body, "request_id": identity(body)}


def _check_score(expected: dict, observed: dict) -> None:
    for key in ("qed", "sa", "sim"):
        if not math.isclose(float(expected[key]), float(observed[key]), abs_tol=1e-12):
            raise ValueError(f"eligibility recomputation changed {key}")
    if (
        observed["oracle_eligible"] is not True
        or observed["endpoint_exclusion_reasons"]
    ):
        raise ValueError("locked endpoint is not strictly eligible")


def _candidate_record(
    *,
    source: str,
    cell: str,
    candidate_id: str,
    source_state: dict,
    endpoint_state: dict,
    action_count: int,
    action_sha256: str,
    properties: dict,
    lineage_ids: list[str],
    provenance: dict,
) -> dict:
    target, suffix = cell.rsplit("_", 1)
    endpoint = canonical_state_key(decode_state(endpoint_state))
    source_smiles = canonical_state_key(decode_state(source_state))
    scored = strict_endpoint_scorer(source_smiles, delta=0.4)({"smiles": endpoint})
    _check_score(properties, scored)
    return {
        "candidate_id": candidate_id,
        "source_artifact": source,
        "target": target,
        "cell": cell,
        "source_idx": int(suffix),
        "source_state": source_state,
        "endpoint_state": endpoint_state,
        "source_state_sha256": utility._state_hash(source_state),
        "endpoint_state_sha256": utility._state_hash(endpoint_state),
        "source_smiles": source_smiles,
        "canonical_smiles": endpoint,
        "scaffold": utility.murcko_scaffold(endpoint),
        "strict_eligibility": {
            key: scored[key]
            for key in (
                "qed",
                "sa",
                "sim",
                "oracle_eligible",
                "endpoint_exclusion_reasons",
            )
        },
        "action_count": action_count,
        "action_list_sha256": action_sha256,
        "lineage_ids": sorted(lineage_ids),
        "provenance": provenance,
        "docking_score": None,
        "scoring_status": "UNAUTHORIZED_NOT_LAUNCHED",
    }


def _load_candidates(root: Path, contract: dict) -> list[dict]:
    inputs = contract["inputs"]
    baseline = _load(root, inputs["compositional_candidate_lock"], compressed=True)
    if (
        baseline["new_oracle_calls"] != 0
        or baseline["teacher_fields_present"] is not False
        or baseline["task_identity_present"] is not False
    ):
        raise ValueError("compositional source is not the sealed zero-oracle pool")
    evaluation = _load(root, inputs["compositional_fold_2_evaluation"], compressed=True)
    reference = next(
        row for row in evaluation["cases"] if row["source_case_id"] == PARP_CASE
    )
    fold = next(row for row in baseline["folds"] if int(row["fold"]) == 2)
    case = next(row for row in fold["cases"] if row["source_case_id"] == PARP_CASE)
    if reference["source_state"] != case["source_state"]:
        raise ValueError("PARP1 source-state join changed")
    candidates = []
    found = set()
    for policy in case["policies"]:
        for row in policy["candidates"]:
            candidate_id = identity(row)
            if candidate_id not in PARP_IDS:
                continue
            found.add(candidate_id)
            states = _replay(case["source_state"], tuple(row["actions"]))
            if states[-1] != row["endpoint_state"]:
                raise ValueError(f"PARP1 replay changed for {candidate_id}")
            endpoint = canonical_state_key(decode_state(row["endpoint_state"]))
            properties = strict_endpoint_scorer(
                canonical_state_key(decode_state(case["source_state"])), delta=0.4
            )({"smiles": endpoint})
            patch_ids = row["patch_ids"]
            candidates.append(
                _candidate_record(
                    source="compositional_candidate_lock",
                    cell="parp1_2",
                    candidate_id=candidate_id,
                    source_state=case["source_state"],
                    endpoint_state=row["endpoint_state"],
                    action_count=len(row["actions"]),
                    action_sha256=canonical_hash(row["actions"]),
                    properties=properties,
                    lineage_ids=[
                        *(f"macro:{value}" for value in patch_ids),
                        "macro_program:" + ":".join(patch_ids),
                    ],
                    provenance={
                        "fold": 2,
                        "source_case_id": PARP_CASE,
                        "source_group": reference["source_group"],
                        "policy_id": policy["policy_id"],
                        "patch_ids": patch_ids,
                        "primitive_teacher_actions_used": row["realization"][
                            "primitive_teacher_actions_used"
                        ],
                    },
                )
            )
    if found != PARP_IDS:
        raise ValueError(f"missing PARP1 candidates: {sorted(PARP_IDS - found)}")

    fa7 = _load(root, inputs["program_retrieval_fa7_1"])
    library = _load(root, inputs["program_library"])
    if fa7["new_oracle_calls"] != 0 or any(
        row.get("score") is not None for row in fa7["candidates"]
    ):
        raise ValueError("FA7 source is not an unscored zero-oracle control pool")
    found = set()
    for row in fa7["candidates"]:
        candidate_id = row["candidate_id"]
        if candidate_id not in FA7_IDS:
            continue
        found.add(candidate_id)
        trace = row["trace"]
        replayed = _replay(row["source_state"], tuple(trace["actions"]))
        if list(replayed) != trace["states"] or trace["complete"] is not True:
            raise ValueError(f"FA7 exact replay changed for {candidate_id}")
        if canonical_state_key(decode_state(trace["states"][-1])) != row["endpoint"]:
            raise ValueError(f"FA7 endpoint changed for {candidate_id}")
        draw = row["provenance"]["metadata"]["draws"][0]
        parent = library[int(draw["program_index"])]
        if parent["source_groups"] != [
            "a918acb6b6313fbd8187ef5b2283a4fc87490942f7341c614dc60d11372a6667"
        ]:
            raise ValueError("FA7 parent program source group changed")
        parent_id = identity(parent["program"])
        candidates.append(
            _candidate_record(
                source="program_retrieval_fa7_1",
                cell="fa7_1",
                candidate_id=candidate_id,
                source_state=row["source_state"],
                endpoint_state=trace["states"][-1],
                action_count=len(trace["actions"]),
                action_sha256=canonical_hash(trace["actions"]),
                properties=row["provenance"]["properties"],
                lineage_ids=[f"program:{trace['program_id']}", f"program:{parent_id}"],
                provenance={
                    "source_group": row["source_group"],
                    "program_id": trace["program_id"],
                    "parent_program_id": parent_id,
                    "parent_program_index": int(draw["program_index"]),
                    "pool_index": int(draw["pool_index"]),
                    "parent_source_groups": parent["source_groups"],
                },
            )
        )
    if found != FA7_IDS:
        raise ValueError(f"missing FA7 candidates: {sorted(FA7_IDS - found)}")
    return sorted(candidates, key=lambda row: (row["cell"], row["candidate_id"]))


def _baseline_rows(root: Path, v3: dict):
    rows, exclusions = [], []
    for loader in (
        utility._load_second_generation,
        utility._load_program_pool,
        utility._load_delta06,
        utility._load_delta04,
    ):
        admitted, rejected = loader(root, v3)
        rows.extend(admitted)
        exclusions.extend(rejected)
    return rows, exclusions


def _simulate(root: Path, candidates: list[dict], v3: dict) -> dict:
    baseline, exclusions = _baseline_rows(root, v3)
    projected = []
    for index, candidate in enumerate(candidates):
        request = _request(candidate)
        projected.append(
            utility._make_row(
                row_id=f"acquisition:{request['request_id']}",
                artifact="t4_utility_data_acquisition_lock",
                target=candidate["target"],
                cell=candidate["cell"],
                delta=0.4,
                oracle_protocol=request["evaluator"],
                docking_seed=1701,
                docking_score=float(index % 2),
                source_state=candidate["source_state"],
                endpoint_state=candidate["endpoint_state"],
                lineage_ids=candidate["lineage_ids"],
                provenance={"simulation_only_candidate_id": candidate["candidate_id"]},
            )
        )
    kept, _all_exclusions, grouping = utility._leakage_filter(
        baseline + projected, list(exclusions)
    )
    projected_ids = {row.row_id for row in projected}
    kept_ids = {row.row_id for row in kept}
    if not projected_ids <= kept_ids:
        raise ValueError(
            f"locked candidates fail v3 grouping: {sorted(projected_ids - kept_ids)}"
        )
    support = utility._support_plan(kept)
    if support["supported_target_count"] < 3 or support["supported_strata"] < 5:
        raise ValueError("four-request lock does not satisfy the unchanged v3 gate")
    return {
        "schema_version": "t4_utility_data_acquisition_gate_simulation_v1",
        "evidence_status": "CONDITIONAL_STRUCTURAL_SIMULATION_NOT_MEASURED_UTILITY",
        "simulation_condition": "All four future calls return finite values and each two-row cell has a strict non-tied pair.",
        "hypothetical_scores_persisted": False,
        "baseline_admitted_rows_before_grouping": len(baseline),
        "baseline_final_rows": 47,
        "locked_request_rows": len(projected),
        "projected_final_rows": len(kept),
        "original_cross_fold_group_removals": grouping["excluded_rows"],
        "locked_rows_surviving_grouping": len(projected_ids & kept_ids),
        "post_exclusion_conflicting_groups": grouping[
            "post_exclusion_conflicting_group_count"
        ],
        "supported_targets": support["supported_targets"],
        "supported_target_count": support["supported_target_count"],
        "supported_strata": support["supported_strata"],
        "unchanged_gate": {
            "minimum_supported_targets": 3,
            "minimum_supported_strata": 5,
        },
        "passes_if_condition_holds": True,
    }


def _publish(path: Path, payload: dict) -> None:
    envelope = {"payload": payload, "payload_sha256": identity(payload)}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(envelope, indent=2, sort_keys=True) + "\n")


def run(root: Path, output: Path, contract_path: Path = CONTRACT) -> dict:
    document = json.loads((root / contract_path).read_text())
    contract = document["payload"]
    if canonical_hash(contract) != document["contract_sha256"]:
        raise ValueError("acquisition contract self-hash mismatch")
    for name in (
        "v3_result",
        "v3_data_audit",
        "v3_exclusion_ledger",
        "v3_split_manifest",
    ):
        _git_bytes(root, contract["inputs"][name])
    v3_document = _load(root, contract["inputs"]["v3_contract"])
    v3 = v3_document["payload"]
    candidates = _load_candidates(root, contract)
    if {row["candidate_id"] for row in candidates} != {
        row["candidate_id"] for row in contract["selection"]["candidates"]
    }:
        raise ValueError("candidate inventory differs from predeclared selection")
    if len(candidates) != 4:
        raise ValueError("request lock must contain exactly four candidates")
    requests = sorted(
        (_request(row) for row in candidates), key=lambda row: row["request_id"]
    )
    simulation = _simulate(root, candidates, v3)
    candidate_payload = {
        "schema_version": "t4_utility_data_acquisition_candidate_lock_v1",
        "contract_sha256": document["contract_sha256"],
        "source_bindings": contract["inputs"],
        "selection_rule": contract["selection"]["rule"],
        "candidates": candidates,
        "counts_per_cell": contract["selection"]["counts_per_cell"],
        "new_docking_calls": 0,
        "new_oracle_calls": 0,
        "modal_launches": 0,
        "live_run_accesses": 0,
    }
    candidate_hash = identity(candidate_payload)
    request_payload = {
        "schema_version": "t4_utility_data_acquisition_request_lock_v1",
        "contract_sha256": document["contract_sha256"],
        "candidate_lock_payload_sha256": candidate_hash,
        "requests": requests,
        "scoring_status": "UNAUTHORIZED_NOT_LAUNCHED",
        "call_ceiling_if_later_exactly_authorized": 4,
        "automatic_retries": 0,
        "replacement_or_backfill": False,
        "required_later_authorization": contract["authorization"][
            "required_later_authorization"
        ],
        "training_data_acquisition_only": True,
        "prospective_evaluation_separate": True,
        "new_docking_calls": 0,
        "new_oracle_calls": 0,
        "modal_launches": 0,
        "live_run_accesses": 0,
    }
    gate_payload = {**simulation, "candidate_lock_payload_sha256": candidate_hash}
    result = {
        "schema_version": "t4_utility_data_acquisition_lock_result_v1",
        "status": "LOCKED_UNAUTHORIZED",
        "request_count": 4,
        "counts_per_cell": contract["selection"]["counts_per_cell"],
        "candidate_lock_payload_sha256": candidate_hash,
        "request_lock_payload_sha256": identity(request_payload),
        "gate_simulation_payload_sha256": identity(gate_payload),
        "conditional_gate_result": {
            "supported_targets": simulation["supported_targets"],
            "supported_target_count": simulation["supported_target_count"],
            "supported_strata": simulation["supported_strata"],
            "passes_if_all_four_are_finite_and_non_tied_within_cell": True,
        },
        "scientific_interpretation": "This is a score-blind training-data acquisition lock, not measured utility evidence and not prospective evaluation.",
        "new_docking_calls": 0,
        "new_oracle_calls": 0,
        "modal_launches": 0,
        "live_run_accesses": 0,
    }
    for name, payload in (
        ("candidate_lock.json", candidate_payload),
        ("request_lock.json", request_payload),
        ("gate_simulation.json", gate_payload),
        ("result.json", result),
    ):
        _publish(output / name, payload)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    output = args.output if args.output.is_absolute() else args.root / args.output
    run(args.root, output)


if __name__ == "__main__":
    main()
