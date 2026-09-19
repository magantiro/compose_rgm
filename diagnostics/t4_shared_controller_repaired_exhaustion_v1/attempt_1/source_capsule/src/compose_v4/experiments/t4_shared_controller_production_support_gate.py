"""Deterministic zero-oracle gate for shared-controller proposal admission."""

from __future__ import annotations

import copy
import gzip
import hashlib
import json
import platform
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import rdkit

from compose_v4.control.fiber_control import SearchState
from compose_v4.experiments.t4_integrated_route_fiber import (
    EXPERTS,
    PROTONATION_AWARE_EXPERT,
)
from compose_v4.experiments.t4_shared_controller_checkpoint import (
    admit_candidate_union,
)
from compose_v4.experiments.t4_shared_controller_completion_contract import (
    payload_identity,
    sha256_file,
)
from compose_v4.experiments.t4_shared_controller_scored_runtime import (
    admit_runtime_proposal_pools,
    attach_generic_scale_band,
)

SCHEMA_VERSION = "t4_shared_controller_production_support_gate_result_v1"
CONTRACT_SCHEMA_VERSION = "t4_shared_controller_production_support_gate_contract_v1"
BRAF_CANDIDATE_SHA256 = (
    "96e3828141b0db1c6d681771bf0c87a2e5ceee400344e11af084a834b3feaf37"
)


def _load_envelope(path: Path) -> tuple[dict[str, Any], str]:
    envelope = json.loads(path.read_text())
    payload = envelope.get("payload")
    identity = envelope.get("payload_sha256")
    if not isinstance(payload, dict) or payload_identity(payload) != identity:
        raise ValueError(f"invalid deterministic envelope: {path}")
    return payload, str(identity)


def _read_jsonl_gzip(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rt") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _validate_input(
    root: Path, binding: Mapping[str, Any]
) -> tuple[Path, dict[str, Any] | None, str | None]:
    path = root / str(binding["path"])
    observed = sha256_file(path)
    if observed != binding.get("sha256"):
        raise ValueError(f"input physical identity drift: {path}")
    if "payload_sha256" not in binding:
        return path, None, None
    payload, identity = _load_envelope(path)
    if identity != binding["payload_sha256"]:
        raise ValueError(f"input payload identity drift: {path}")
    return path, payload, identity


def _controller_contract(
    scored_contract: Mapping[str, Any], gate_contract: Mapping[str, Any]
) -> dict[str, Any]:
    controller = copy.deepcopy(scored_contract["controller"])
    controller["proposal_experts"] = [*EXPERTS, PROTONATION_AWARE_EXPERT]
    controller["proposal"][PROTONATION_AWARE_EXPERT] = copy.deepcopy(
        gate_contract["protonation_aware_proposal"]
    )
    return {
        "support": scored_contract["support"],
        "controller": controller,
        "trajectory_distillation": {"enabled": True},
        "stale_braf_v4_reconciliation": copy.deepcopy(
            scored_contract["stale_braf_v4_reconciliation"]
        ),
    }


def _runtime_union(
    pools: Mapping[str, list[dict[str, Any]]],
    *,
    cell: Mapping[str, Any],
    contract: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    admitted, runtime_ledger = admit_runtime_proposal_pools(
        pools,
        cell=cell,
        contract=contract,
    )
    state = SearchState(archive={str(cell["source_smiles"]): 0.0})
    experts = tuple(contract["controller"]["proposal_experts"])
    union = admit_candidate_union(
        admitted,
        state=state,
        parents=[str(cell["source_smiles"])],
        experts=experts,
    )
    return union, runtime_ledger


def _braf_gate(
    *,
    route_rows: list[dict[str, Any]],
    retained_payload: Mapping[str, Any],
    scored_contract: Mapping[str, Any],
    runtime_contract: Mapping[str, Any],
) -> dict[str, Any]:
    cell = next(
        row for row in scored_contract["cells"] if row["cell_key"] == "braf_0_d06"
    )
    parent = str(cell["source_smiles"])
    route = []
    for source in route_rows:
        row = {
            **copy.deepcopy(source),
            "proposal_lane": "route_complete_region",
            "proposal_experts": ["route_complete_region"],
            "families": ["route_complete_region"],
            "program_families": ["route_complete_region"],
            "parent": parent,
            "parent_score": 0.0,
            "delta": float(cell["delta"]),
        }
        route.append(attach_generic_scale_band(row))
    retained_cell = next(
        row for row in retained_payload["cells"] if row["cell_key"] == cell["cell_key"]
    )
    for index, source in enumerate(retained_cell["eligible"], 1):
        primitives = int(source["primitive_edits"])
        route.append(
            attach_generic_scale_band(
                {
                    **copy.deepcopy(source),
                    "proposal_lane": "route_complete_region",
                    "proposal_experts": ["route_complete_region"],
                    "families": ["retained_core_prune"],
                    "program_families": ["retained_core_prune"],
                    "parent": parent,
                    "parent_score": 0.0,
                    "delta": float(cell["delta"]),
                    "created": 0,
                    "realized_primitives": primitives,
                    "realized_primitive_band": (
                        "small"
                        if primitives <= 3
                        else "medium" if primitives <= 11 else "large"
                    ),
                    "route_proposal_rank": len(route_rows) + index,
                }
            )
        )
    pools = {
        expert: [] for expert in runtime_contract["controller"]["proposal_experts"]
    }
    pools["route_complete_region"] = route
    union, ledger = _runtime_union(pools, cell=cell, contract=runtime_contract)
    identities = {
        hashlib.sha256(row["smiles"].encode()).hexdigest(): row for row in union
    }
    candidate = identities.get(BRAF_CANDIDATE_SHA256)
    stale = ledger["stale_query_exclusions"]
    return {
        "cell_key": cell["cell_key"],
        "source_global_index": cell["source_global_index"],
        "delta": cell["delta"],
        "candidate_input_count": len(route),
        "route_preflight_input_count": len(route_rows),
        "retained_core_input_count": len(retained_cell["eligible"]),
        "runtime_admission": ledger,
        "candidate_union_unique": len(union),
        "candidate_union_sha256": payload_identity(sorted(identities)),
        "required_candidate": {
            "canonical_smiles_sha256": BRAF_CANDIDATE_SHA256,
            "survived": candidate is not None,
            "proposal_experts": (
                candidate["proposal_experts"] if candidate is not None else []
            ),
            "route_proposal_rank": (
                candidate.get("route_proposal_rank") if candidate is not None else None
            ),
            "realized_primitive_band": (
                candidate.get("realized_primitive_band")
                if candidate is not None
                else None
            ),
        },
        "stale_candidates_excluded": stale["excluded_total"],
        "gate_pass": bool(candidate is not None and stale["excluded_total"] == 3),
    }


def _protonation_gate(
    *,
    protonation_payload: Mapping[str, Any],
    registry: list[dict[str, Any]],
    runtime_contract: Mapping[str, Any],
) -> dict[str, Any]:
    root = next(
        row
        for row in protonation_payload["roots"]
        if row["label"] == "primary_strict_support_failure"
    )
    source = str(registry[int(root["registry_index"])]["smiles"])
    if source != root["source"]:
        raise ValueError("protonation gate source-registry binding drift")
    cell = {
        "cell_key": "5ht1b_2_d06",
        "source_cell": "5ht1b_2",
        "source_global_index": int(root["registry_index"]),
        "source_smiles": source,
        "target": "5ht1b",
        "delta": float(root["delta"]),
    }
    rows = []
    expected_eligible = set()
    for source_row in root["candidates"]:
        endpoint = source_row["endpoint_metrics"]
        if endpoint["fully_eligible"]:
            expected_eligible.add(str(source_row["smiles"]))
        primitives = int(source_row["primitive_edits"])
        rows.append(
            attach_generic_scale_band(
                {
                    **copy.deepcopy(source_row),
                    "proposal_lane": PROTONATION_AWARE_EXPERT,
                    "proposal_experts": [PROTONATION_AWARE_EXPERT],
                    "families": sorted(set(source_row.get("program_families") or [])),
                    "proposal_program_sha256": payload_identity(source_row["actions"]),
                    "parent": source,
                    "parent_score": 0.0,
                    "delta": float(root["delta"]),
                    "realized_primitives": primitives,
                    "realized_primitive_band": (
                        "small"
                        if primitives <= 3
                        else "medium" if primitives <= 11 else "large"
                    ),
                }
            )
        )
    pools = {
        expert: [] for expert in runtime_contract["controller"]["proposal_experts"]
    }
    pools[PROTONATION_AWARE_EXPERT] = rows
    union, ledger = _runtime_union(pools, cell=cell, contract=runtime_contract)
    observed = {row["smiles"] for row in union}
    observed_hashes = sorted(
        hashlib.sha256(value.encode()).hexdigest() for value in observed
    )
    return {
        "cell_key": cell["cell_key"],
        "source_global_index": cell["source_global_index"],
        "delta": cell["delta"],
        "proposer_receipt_candidates": len(rows),
        "proposer_receipt_fully_eligible": len(expected_eligible),
        "runtime_admission": ledger,
        "candidate_union_unique": len(union),
        "candidate_union_sha256": payload_identity(observed_hashes),
        "all_expected_eligible_survived": observed == expected_eligible,
        "all_survivors_unseen": all(row["smiles"] != source for row in union),
        "surviving_program_kinds": dict(
            sorted(
                {
                    kind: sum(row.get("program_kind") == kind for row in union)
                    for kind in {row.get("program_kind") for row in union}
                }.items()
            )
        ),
        "surviving_canonical_smiles_sha256": observed_hashes,
        "gate_pass": bool(
            observed == expected_eligible
            and len(observed) >= 14
            and all(row["smiles"] != source for row in union)
        ),
    }


def scientific_projection(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Return the deterministic scientific content independent of checkout revision."""

    projected = copy.deepcopy(dict(payload))
    projected.get("implementation", {}).pop("code_revision", None)
    return projected


def run_gate(root: Path, contract_path: Path) -> dict[str, Any]:
    """Replay sealed proposal receipts through the exact production admission helpers."""

    root = root.resolve()
    contract, contract_identity = _load_envelope(contract_path.resolve())
    if contract.get("schema_version") != CONTRACT_SCHEMA_VERSION:
        raise ValueError("unexpected production support gate contract schema")
    if contract.get("status") != "AUTHORIZED_ZERO_ORACLE_RUNTIME_ADMISSION_GATE":
        raise ValueError("production support gate is not zero-oracle authorized")
    inputs = contract.get("inputs")
    if not isinstance(inputs, dict) or set(inputs) != {
        "braf_route_preflight_ledger",
        "protonation_proposal_gate",
        "retained_core_support_gate",
        "scored_contract_v4",
        "source_registry",
        "stale_reconciliation",
    }:
        raise ValueError("production support gate input census drift")
    loaded = {name: _validate_input(root, binding) for name, binding in inputs.items()}
    route_rows = _read_jsonl_gzip(loaded["braf_route_preflight_ledger"][0])
    protonation_payload = loaded["protonation_proposal_gate"][1]
    retained_payload = loaded["retained_core_support_gate"][1]
    scored_contract = loaded["scored_contract_v4"][1]
    if any(
        item is None
        for item in (protonation_payload, retained_payload, scored_contract)
    ):
        raise RuntimeError("gate envelope inputs were not loaded")
    registry = json.loads(loaded["source_registry"][0].read_text())
    stale_payload = loaded["stale_reconciliation"][1]
    if (
        stale_payload is None
        or scored_contract["stale_braf_v4_reconciliation"]["payload_sha256"]
        != loaded["stale_reconciliation"][2]
    ):
        raise ValueError("scored contract stale-reconciliation binding drift")
    runtime_contract = _controller_contract(scored_contract, contract)
    braf = _braf_gate(
        route_rows=route_rows,
        retained_payload=retained_payload,
        scored_contract=scored_contract,
        runtime_contract=runtime_contract,
    )
    protonation = _protonation_gate(
        protonation_payload=protonation_payload,
        registry=registry,
        runtime_contract=runtime_contract,
    )
    source_files = contract["implementation_files"]
    implementation_hashes = {
        relative: sha256_file(root / relative) for relative in source_files
    }
    result = {
        "schema_version": SCHEMA_VERSION,
        "evidence_status": "computed_deterministic_zero_oracle_runtime_admission",
        "scientific_problem": contract["scientific_problem"],
        "primary_model_output": contract["primary_model_output"],
        "central_claim_under_test": contract["central_claim_under_test"],
        "experimental_setting": contract["experimental_setting"],
        "claim_boundary": contract["claim_boundary"],
        "inputs": {
            name: {
                "path": binding["path"],
                "sha256": binding["sha256"],
                **(
                    {"payload_sha256": binding["payload_sha256"]}
                    if "payload_sha256" in binding
                    else {}
                ),
            }
            for name, binding in sorted(inputs.items())
        },
        "contract": {
            "path": str(contract_path.resolve().relative_to(root)),
            "sha256": sha256_file(contract_path),
            "payload_sha256": contract_identity,
        },
        "runtime_semantics": {
            "producer_vocabulary": runtime_contract["controller"]["proposal_experts"],
            "admission_helper": (
                "compose_v4.experiments.t4_shared_controller_scored_runtime."
                "admit_runtime_proposal_pools"
            ),
            "union_helper": (
                "compose_v4.experiments.t4_shared_controller_checkpoint."
                "admit_candidate_union"
            ),
            "preflight_and_runtime_share_exact_helpers": True,
            "sentinel_parent_score": 0.0,
            "sentinel_parent_score_used_for_selection_or_gate": False,
        },
        "cells": {"braf_0_d06": braf, "5ht1b_2_d06": protonation},
        "gate": {
            "braf_unseen_route_survives": braf["gate_pass"],
            "protonation_unseen_candidates_survive": protonation["gate_pass"],
            "passed": braf["gate_pass"] and protonation["gate_pass"],
        },
        "implementation": {
            "code_revision": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=root, text=True
            ).strip(),
            "source_files_sha256": implementation_hashes,
        },
        "software": {
            "python": platform.python_version(),
            "rdkit": rdkit.__version__,
            "platform": platform.platform(),
        },
        "costs": {
            "docking_calls": 0,
            "oracle_calls": 0,
            "modal_launches": 0,
            "live_run_reads": 0,
        },
    }
    if not result["gate"]["passed"]:
        raise RuntimeError("shared-controller production support gate failed")
    return result


__all__ = [
    "BRAF_CANDIDATE_SHA256",
    "CONTRACT_SCHEMA_VERSION",
    "SCHEMA_VERSION",
    "run_gate",
    "scientific_projection",
]
