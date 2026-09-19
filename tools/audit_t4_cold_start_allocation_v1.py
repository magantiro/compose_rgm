"""Replay the T4 v1 cold-start allocation on immutable completed pools."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

from compose_v4.control.fiber_control import ProgramValue, SearchState
from compose_v4.control.t4_cold_start_allocation import (
    SCHEMA_VERSION as POLICY_SCHEMA_VERSION,
)
from compose_v4.control.t4_cold_start_allocation import cold_start_allocation_v1
from compose_v4.experiments.t4_integrated_route_fiber import select_batch

SCHEMA_VERSION = "t4_cold_start_allocation_replay_v1"


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_envelope(path: Path, expected_sha256: str) -> tuple[dict, dict]:
    actual_sha256 = _sha256(path)
    if actual_sha256 != expected_sha256:
        raise ValueError(f"{path}: outer SHA-256 {actual_sha256} != expected {expected_sha256}")
    envelope = json.loads(path.read_text())
    if set(envelope) != {"payload", "payload_sha256"}:
        raise ValueError(f"{path}: expected payload/payload_sha256 envelope")
    actual_payload_sha256 = hashlib.sha256(_canonical_bytes(envelope["payload"])).hexdigest()
    if actual_payload_sha256 != envelope["payload_sha256"]:
        raise ValueError(
            f"{path}: payload SHA-256 {actual_payload_sha256} != "
            f"declared {envelope['payload_sha256']}"
        )
    return envelope, {
        "path": str(path),
        "sha256": actual_sha256,
        "payload_sha256": actual_payload_sha256,
        "schema_version": envelope["payload"].get("schema_version"),
    }


def _git_revision(repo: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _selected_summary(row: dict) -> dict:
    return {
        "smiles": row["smiles"],
        "proposal_lane": row["proposal_lane"],
        "proposal_experts": list(row.get("proposal_experts") or []),
        "selection_kind": row["selection_kind"],
        "realized_primitive_band": row.get("realized_primitive_band"),
        "route_proposal_rank": row.get("route_proposal_rank"),
    }


def _available_experts(candidates: list[dict]) -> set[str]:
    result = set()
    for row in candidates:
        result.update(row.get("proposal_experts") or [row.get("proposal_lane")])
    result.discard(None)
    return result


def _initial_state(lock: dict) -> SearchState:
    archive = {}
    for parent in lock["parents"]:
        matching = [row for row in lock["candidate_pool"] if row.get("parent") == parent]
        if not matching:
            raise ValueError(f"round-one parent {parent!r} has no candidate provenance")
        scores = {float(row["parent_score"]) for row in matching}
        if len(scores) != 1:
            raise ValueError(f"round-one parent {parent!r} has conflicting scores {scores}")
        archive[parent] = scores.pop()
    return SearchState(archive=archive)


def _first_round(payload: dict, path: Path) -> dict:
    rounds = payload.get("rounds")
    if not rounds:
        raise ValueError(f"{path}: result/checkpoint has no completed round")
    first = rounds[0]
    if not isinstance(first, dict) or not first.get("docked"):
        raise ValueError(f"{path}: first completed round has no docked records")
    return first


def _locked_selection(lock: dict) -> list[dict]:
    if "selected_rows" in lock:
        return list(lock["selected_rows"])
    by_smiles = {row["smiles"]: row for row in lock["candidate_pool"]}
    selected = []
    for query in lock["queries"]:
        smiles = query["smiles"]
        if smiles not in by_smiles:
            raise ValueError(f"locked query {smiles!r} is absent from candidate pool")
        selected.append({**by_smiles[smiles], "selection_kind": query["selection_kind"]})
    return selected


def _replay_case(repo: Path, case: dict) -> dict:
    contract_path = repo / case["contract"]
    lock_path = Path(case["lock"])
    result_path = Path(case["result"])
    contract_envelope, contract_input = _load_envelope(contract_path, case["contract_sha256"])
    lock_envelope, lock_input = _load_envelope(lock_path, case["lock_sha256"])
    result_envelope, result_input = _load_envelope(result_path, case["result_sha256"])
    contract = contract_envelope["payload"]
    lock = lock_envelope["payload"]
    result = result_envelope["payload"]

    cell = case["cell"]
    if lock.get("cell") != cell or result.get("cell") != cell:
        raise ValueError(f"{case['label']}: cell identity mismatch")
    contract_hash = contract_envelope["payload_sha256"]
    if lock.get("contract_payload_sha256") != contract_hash:
        raise ValueError(f"{case['label']}: lock/contract lineage mismatch")
    if result.get("contract_payload_sha256") != contract_hash:
        raise ValueError(f"{case['label']}: result/contract lineage mismatch")
    if lock.get("round") != 1:
        raise ValueError(f"{case['label']}: expected round-one lock")
    first_round = _first_round(result, result_path)
    if first_round.get("candidate_lock_payload_sha256") != lock_envelope["payload_sha256"]:
        raise ValueError(f"{case['label']}: result/round-lock lineage mismatch")

    cell_contracts = {row["cell"]: row for row in contract["cells"]}
    if cell not in cell_contracts:
        raise ValueError(f"{case['label']}: cell missing from contract")
    seed = int(cell_contracts[cell]["controller_seed"])
    candidates = []
    for source in lock["candidate_pool"]:
        row = dict(source)
        row["fingerprint"] = set(row["fingerprint"])
        candidates.append(row)
    state = _initial_state(lock)

    original = select_batch(
        candidates,
        ProgramValue(),
        state,
        np.random.default_rng(seed),
        round_index=1,
        batch=int(contract["batch"]),
        exploration=int(contract["exploration"]),
        expert_floor_rounds=int(contract["expert_floor_rounds"]),
        route_scale_floor_rounds=int(contract["route_scale_floor_rounds"]),
    )
    locked_selection = _locked_selection(lock)
    original_smiles = [row["smiles"] for row in original]
    locked_smiles = [row["smiles"] for row in locked_selection]
    if original_smiles != locked_smiles:
        raise ValueError(f"{case['label']}: frozen selector replay did not reproduce lock")

    available_experts = _available_experts(candidates)
    plan = cold_start_allocation_v1(
        round_index=1,
        batch_size=int(contract["batch"]),
        available_experts=available_experts,
    )
    repaired = select_batch(
        candidates,
        ProgramValue(),
        state,
        np.random.default_rng(seed),
        round_index=1,
        batch=int(contract["batch"]),
        exploration=plan.exploration_slots,
        expert_floor_rounds=1,
        route_scale_floor_rounds=1,
        route_scale_floor_counts=plan.route_scale_floor_counts,
    )
    repaired_smiles = {row["smiles"] for row in repaired}

    docked = first_round["docked"]
    best_score = min(float(row["score"]) for row in docked if row.get("score") is not None)
    original_best = sorted(
        row["smiles"]
        for row in docked
        if row.get("score") is not None and float(row["score"]) == best_score
    )
    original_floor = {
        row["proposal_lane"]: row["smiles"]
        for row in locked_selection
        if row.get("selection_kind") == "expert_floor"
        and row.get("proposal_lane") in {"shallow", "anchored_replacement"}
    }
    preserved_floor = {
        expert: smiles in repaired_smiles for expert, smiles in sorted(original_floor.items())
    }
    probes = []
    for probe in case.get("diagnostic_probes", []):
        pool_rows = [row for row in candidates if row["smiles"] == probe["smiles"]]
        if len(pool_rows) != 1:
            raise ValueError(
                f"{case['label']}: diagnostic probe occurs {len(pool_rows)} times in pool"
            )
        row = pool_rows[0]
        probes.append(
            {
                **probe,
                "proposal_lane": row["proposal_lane"],
                "realized_primitive_band": row.get("realized_primitive_band"),
                "route_proposal_rank": row.get("route_proposal_rank"),
                "selected_by_original": row["smiles"] in set(original_smiles),
                "selected_by_repair": row["smiles"] in repaired_smiles,
            }
        )

    represented_experts = sorted(
        {
            expert
            for row in repaired
            for expert in (row.get("proposal_experts") or [row.get("proposal_lane")])
            if expert
        }
    )
    return {
        "label": case["label"],
        "cell": cell,
        "controller_seed": seed,
        "inputs": {
            "contract": contract_input,
            "lock": lock_input,
            "result_or_checkpoint": result_input,
        },
        "candidate_count": len(candidates),
        "original_replay_exact": True,
        "policy_plan": {
            "route_scale_floor_counts": plan.route_scale_floor_counts,
            "expert_floor_counts": plan.expert_floor_counts,
            "exploration_slots": plan.exploration_slots,
        },
        "original_selection": [_selected_summary(row) for row in original],
        "repaired_selection": [_selected_summary(row) for row in repaired],
        "repaired_large_route_ranks": sorted(
            int(row["route_proposal_rank"])
            for row in repaired
            if row.get("realized_primitive_band") == "large"
            and row.get("route_proposal_rank") is not None
        ),
        "represented_experts": represented_experts,
        "original_round_best": {
            "score": best_score,
            "smiles": original_best,
            "preserved_count": sum(smiles in repaired_smiles for smiles in original_best),
            "tied_count": len(original_best),
            "score_preserved": any(smiles in repaired_smiles for smiles in original_best),
        },
        "original_base_expert_floors": original_floor,
        "original_base_expert_floors_preserved": preserved_floor,
        "diagnostic_probes": probes,
    }


def _validate_abstention(repo: Path, case: dict) -> dict:
    contract_envelope, contract_input = _load_envelope(
        repo / case["contract"], case["contract_sha256"]
    )
    result_envelope, result_input = _load_envelope(Path(case["result"]), case["result_sha256"])
    result = result_envelope["payload"]
    if result.get("cell") != case["cell"]:
        raise ValueError(f"{case['label']}: result cell mismatch")
    if result.get("contract_payload_sha256") != contract_envelope["payload_sha256"]:
        raise ValueError(f"{case['label']}: result/contract lineage mismatch")
    first_round = _first_round(result, Path(case["result"]))
    return {
        "label": case["label"],
        "cell": case["cell"],
        "status": "abstained_missing_round_one_candidate_pool",
        "reason": case["reason"],
        "inputs": {"contract": contract_input, "result": result_input},
        "missing_lock_payload_sha256": first_round.get("candidate_lock_payload_sha256"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("configs/t4_cold_start_allocation_replay_v1.json"),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[1]
    manifest_path = args.manifest if args.manifest.is_absolute() else repo / args.manifest
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("policy_schema_version") != POLICY_SCHEMA_VERSION:
        raise ValueError("manifest/policy schema mismatch")
    cases = [_replay_case(repo, case) for case in manifest["cases"]]
    abstentions = [_validate_abstention(repo, case) for case in manifest["abstentions"]]

    preserved_best = sum(case["original_round_best"]["score_preserved"] for case in cases)
    preserved_floors = sum(
        all(case["original_base_expert_floors_preserved"].values()) for case in cases
    )
    probes = [probe for case in cases for probe in case["diagnostic_probes"]]
    payload = {
        "schema_version": SCHEMA_VERSION,
        "policy_schema_version": POLICY_SCHEMA_VERSION,
        "status": "selected_zero_oracle_policy",
        "scientific_problem": (
            "repair value-free round-one undercoverage of high-ranked large coordinated "
            "route transformations without target-specific information"
        ),
        "costs": {"oracle_calls": 0, "docking_calls": 0, "modal_launches": 0},
        "code_revision": _git_revision(repo),
        "audit_script_sha256": _sha256(Path(__file__)),
        "manifest": {"path": str(args.manifest), "sha256": _sha256(manifest_path)},
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
        "sample_counts": {
            "exact_replay_cases": len(cases),
            "abstentions": len(abstentions),
            "diagnostic_probes": len(probes),
        },
        "summary": {
            "original_locks_exactly_reproduced": sum(
                case["original_replay_exact"] for case in cases
            ),
            "original_round_best_score_preserved": preserved_best,
            "original_base_expert_floors_preserved": preserved_floors,
            "diagnostic_probes_selected_by_original": sum(
                probe["selected_by_original"] for probe in probes
            ),
            "diagnostic_probes_selected_by_repair": sum(
                probe["selected_by_repair"] for probe in probes
            ),
        },
        "decision": (
            "Use round-one route floors small=1, medium=1, large=3; retain one "
            "nonempty shallow and anchored floor; leave the eighth slot to ordinary "
            "exploration unless a nonempty optional protonation expert claims it."
        ),
        "claim_boundary": (
            "Retrospective selection replay on immutable completed candidate pools. "
            "It establishes coverage and preservation under an eight-call counterfactual, "
            "not prospective docking performance."
        ),
        "cases": cases,
        "abstentions": abstentions,
    }
    envelope = {
        "payload": payload,
        "payload_sha256": hashlib.sha256(_canonical_bytes(payload)).hexdigest(),
    }
    output = args.output if args.output.is_absolute() else repo / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    temporary.write_text(json.dumps(envelope, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, output)
    print(output)
    print(envelope["payload_sha256"])


if __name__ == "__main__":
    sys.exit(main())
