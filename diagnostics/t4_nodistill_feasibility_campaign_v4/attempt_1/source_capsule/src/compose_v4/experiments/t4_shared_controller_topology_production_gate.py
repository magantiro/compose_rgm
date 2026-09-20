"""Zero-oracle production-path gate for the optional topology-macro expert."""

from __future__ import annotations

import copy
import hashlib
import inspect
import json
import platform
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any

from rdkit import rdBase

from compose_v4.control.fiber_control import SearchState
from compose_v4.control.generic_complete_program_composer_v2 import (
    propose_generic_topology_macro_programs,
)
from compose_v4.experiments.t4_integrated_route_fiber import (
    GENERIC_TOPOLOGY_MACRO_EXPERT,
    validate_expert_vocabulary,
)
from compose_v4.experiments.t4_nodistill_generic_composer_gate import (
    _publish_once,
    _read_envelope,
    sha256_file,
)
from compose_v4.experiments.t4_shared_controller_checkpoint import (
    admit_candidate_union,
)
from compose_v4.experiments.t4_shared_controller_completion_contract import (
    payload_identity,
)
from compose_v4.experiments.t4_shared_controller_scored_runtime import (
    admit_runtime_proposal_pools,
    generic_topology_macro_records,
    proposal_expert_vocabulary,
)

CONTRACT_RELATIVE_PATH = "configs/t4_shared_controller_topology_production_gate_v1.json"
ARTIFACT_ROOT_RELATIVE_PATH = (
    "diagnostics/t4_shared_controller_topology_production_gate_v1/attempt_1"
)
RESULT_SCHEMA = "t4_shared_controller_topology_production_gate_result_v1"
CELL_SHARD_SCHEMA = "t4_shared_controller_topology_production_cell_shard_v1"


def load_contract(repository_root: Path) -> tuple[dict[str, Any], str]:
    """Validate the self-hashed, zero-oracle gate contract and its inputs."""

    contract, contract_identity = _read_envelope(
        repository_root / CONTRACT_RELATIVE_PATH
    )
    if contract.get("schema_version") != (
        "t4_shared_controller_topology_production_gate_contract_v1"
    ):
        raise ValueError("unexpected topology production-gate schema")
    if contract.get("status") != "FROZEN_ZERO_ORACLE_PRODUCTION_PATH_GATE":
        raise ValueError("topology production gate is not frozen")
    if contract.get("costs") != {
        "oracle_calls": 0,
        "docking_calls": 0,
        "modal_launches": 0,
        "gpu_seconds": 0,
    }:
        raise ValueError("topology production gate is not zero-oracle")
    if not all(contract.get("runtime_prohibitions", {}).values()):
        raise ValueError("topology production runtime prohibition was weakened")
    for row in contract["inputs"].values():
        path = repository_root / row["path"]
        if sha256_file(path) != row["sha256"]:
            raise ValueError(f"topology production-gate input changed: {path}")
    if len(contract.get("cells", ())) != 8:
        raise ValueError("topology production gate requires eight distinct cells")
    keys = [row["cell_key"] for row in contract["cells"]]
    if len(set(keys)) != len(keys):
        raise ValueError("topology production gate repeats a cell")
    roles = Counter(row["evidence_role"] for row in contract["cells"])
    if roles != {"terminal_loss": 6, "contrasting_win": 2}:
        raise ValueError("topology production panel role census drift")
    return contract, contract_identity


def _bound_cells(
    repository_root: Path, contract: dict[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    source_registry = json.loads(
        (repository_root / contract["inputs"]["source_registry"]["path"]).read_text()
    )
    sources = {int(row["idx"]): row for row in source_registry}
    matrix = json.loads(
        (repository_root / contract["inputs"]["terminal_matrix"]["path"]).read_text()
    )
    matrix_rows = {
        f"{row['cell']}_d{round(10 * float(row['delta'])):02d}": row
        for row in matrix["records"]
    }
    cells = []
    evidence = {}
    for declared in contract["cells"]:
        source = sources.get(int(declared["source_index"]))
        row = matrix_rows.get(str(declared["cell_key"]))
        if source is None or row is None:
            raise ValueError(
                f"missing frozen source or matrix row: {declared['cell_key']}"
            )
        source_cell = str(declared["cell_key"]).rsplit("_d", 1)[0]
        if (
            not source_cell.startswith(f"{source['target']}_")
            or float(row["delta"]) != float(declared["delta"])
            or row["status_class"] not in {"complete", "exhausted"}
            or (
                declared["evidence_role"] == "terminal_loss"
                and row["outcome"] != "worse"
            )
            or (
                declared["evidence_role"] == "contrasting_win"
                and row["outcome"] != "better"
            )
        ):
            raise ValueError(f"frozen cell evidence drift: {declared['cell_key']}")
        cells.append(
            {
                "cell_key": declared["cell_key"],
                "source_cell": source_cell,
                "source_index": int(source["idx"]),
                "source_smiles": source["smiles"],
                "target": source["target"],
                "delta": float(declared["delta"]),
                "controller_seed": int(declared["controller_seed"]),
                "evidence_role": declared["evidence_role"],
            }
        )
        evidence[str(declared["cell_key"])] = copy.deepcopy(row)
    return cells, evidence


def _production_contract(
    repository_root: Path, contract: dict[str, Any]
) -> dict[str, Any]:
    source, _ = _read_envelope(
        repository_root / contract["inputs"]["production_stale_reconciliation"]["path"]
    )
    return {
        "support": contract["support"],
        "controller": copy.deepcopy(contract["controller"]),
        "trajectory_distillation": copy.deepcopy(contract["trajectory_distillation"]),
        "stale_braf_v4_reconciliation": copy.deepcopy(
            source["stale_braf_v4_reconciliation"]
        ),
    }


def _distribution(values: list[Any]) -> dict[str, int]:
    return dict(sorted(Counter(str(value) for value in values).items()))


def _summarize_cell(
    *,
    cell: dict[str, Any],
    matrix_row: dict[str, Any],
    production_contract: dict[str, Any],
) -> dict[str, Any]:
    controller = production_contract["controller"]
    experts = proposal_expert_vocabulary(production_contract)
    settings = controller["proposal"][GENERIC_TOPOLOGY_MACRO_EXPERT]
    raw, telemetry = generic_topology_macro_records(
        parent=cell["source_smiles"],
        proposal_seed_value=cell["controller_seed"],
        settings=settings,
    )
    for row in raw:
        row["parent_score"] = 0.0
        row["delta"] = cell["delta"]
    pools = {expert: [] for expert in experts}
    pools[GENERIC_TOPOLOGY_MACRO_EXPERT] = raw
    admitted, runtime_ledger = admit_runtime_proposal_pools(
        pools,
        cell=cell,
        contract=production_contract,
    )
    candidates = admit_candidate_union(
        admitted,
        state=SearchState(archive={cell["source_smiles"]: 0.0}),
        parents=[cell["source_smiles"]],
        experts=experts,
    )
    endpoint_hashes = [
        hashlib.sha256(row["smiles"].encode()).hexdigest() for row in candidates
    ]
    plans = [row["macro_plan"]["name"] for row in candidates]
    cycles = [row["observed_macro_fields"]["delta_cycle_rank"] for row in candidates]
    heavy = [row["observed_macro_fields"]["delta_heavy_atoms"] for row in candidates]
    interfaces = [
        row["observed_macro_fields"]["created_attachment_interface_count"]
        for row in candidates
    ]
    retained = [row["observed_macro_fields"]["retained_fraction"] for row in candidates]
    exact_count = sum(bool(row["exact_execution_verified"]) for row in raw)
    return {
        "cell_key": cell["cell_key"],
        "source_index": cell["source_index"],
        "source_smiles_sha256": hashlib.sha256(
            cell["source_smiles"].encode()
        ).hexdigest(),
        "target": cell["target"],
        "delta": cell["delta"],
        "controller_seed": cell["controller_seed"],
        "evidence_role": cell["evidence_role"],
        "frozen_matrix_evidence": {
            key: copy.deepcopy(matrix_row[key])
            for key in (
                "evidence_batch",
                "settled_calls",
                "status",
                "compose",
                "ivg",
                "gap",
                "outcome",
            )
        },
        "funnel": {
            "raw_exact_unique": len(raw),
            "exact_replay_verified": exact_count,
            "fiber_admitted_before_stale": runtime_ledger[
                "fiber_admitted_before_stale_by_expert"
            ][GENERIC_TOPOLOGY_MACRO_EXPERT],
            "stale_query_exclusions": runtime_ledger["stale_query_exclusions"][
                "excluded_by_expert"
            ][GENERIC_TOPOLOGY_MACRO_EXPERT],
            "selection_ready_after_runtime_admission": runtime_ledger[
                "selection_ready_by_expert"
            ][GENERIC_TOPOLOGY_MACRO_EXPERT],
            "archive_union_unique": len(candidates),
        },
        "selection_ready_support": {
            "endpoint_sha256": endpoint_hashes,
            "primitive_scale_counts": _distribution(
                [row["proposal_scale_band"] for row in candidates]
            ),
            "macro_plan_counts": _distribution(plans),
            "macro_plan_diversity": len(set(plans)),
            "cycle_gain_counts": _distribution(cycles),
            "cycle_gain_diversity": len(set(cycles)),
            "heavy_atom_delta_range": [min(heavy), max(heavy)] if heavy else None,
            "heavy_atom_delta_span": max(heavy) - min(heavy) if heavy else 0,
            "interface_count_range": (
                [min(interfaces), max(interfaces)] if interfaces else None
            ),
            "retained_fraction_range": (
                [min(retained), max(retained)] if retained else None
            ),
        },
        "generation_telemetry": telemetry,
        "runtime_admission_order": runtime_ledger["admission_order"],
    }


def _recommend_next_cell(cells: list[dict[str, Any]]) -> dict[str, Any] | None:
    eligible = [
        row
        for row in cells
        if row["evidence_role"] == "terminal_loss"
        and row["funnel"]["archive_union_unique"] > 0
    ]
    if not eligible:
        return None

    def rank(row: dict[str, Any]) -> tuple[Any, ...]:
        support = row["selection_ready_support"]
        return (
            -int(row["funnel"]["archive_union_unique"]),
            -int(support["macro_plan_diversity"]),
            -int(support["cycle_gain_diversity"]),
            -int(support["heavy_atom_delta_span"]),
            str(row["cell_key"]),
        )

    selected = min(eligible, key=rank)
    return {
        "cell_key": selected["cell_key"],
        "policy_rank_fields": {
            "selection_ready_unique": selected["funnel"]["archive_union_unique"],
            "macro_plan_diversity": selected["selection_ready_support"][
                "macro_plan_diversity"
            ],
            "cycle_gain_diversity": selected["selection_ready_support"][
                "cycle_gain_diversity"
            ],
            "heavy_atom_delta_span": selected["selection_ready_support"][
                "heavy_atom_delta_span"
            ],
        },
        "authority": "recommendation_only_no_scored_calls_authorized",
        "claim_boundary": (
            "chosen only from zero-oracle admitted yield and structural diversity; "
            "objective utility is unknown"
        ),
    }


def run_cell_shard(
    *, repository_root: Path, cell_key: str, output_path: Path
) -> dict[str, Any]:
    """Run and publish one independently resumable production-path cell."""

    contract, contract_identity = load_contract(repository_root)
    cells, matrix = _bound_cells(repository_root, contract)
    selected = [cell for cell in cells if cell["cell_key"] == cell_key]
    if len(selected) != 1:
        raise ValueError(f"cell is outside the frozen production panel: {cell_key}")
    production_contract = _production_contract(repository_root, contract)
    experts = proposal_expert_vocabulary(production_contract)
    if experts != validate_expert_vocabulary(
        contract["controller"]["proposal_experts"]
    ):
        raise ValueError("production expert vocabulary drift")
    settings_identity = payload_identity(
        contract["controller"]["proposal"][GENERIC_TOPOLOGY_MACRO_EXPERT]
    )
    controller_identity = payload_identity(contract["controller"])
    if output_path.exists():
        payload, payload_sha256 = _read_envelope(output_path)
        if (
            payload.get("schema_version") != CELL_SHARD_SCHEMA
            or payload.get("contract_payload_sha256") != contract_identity
            or payload.get("cell", {}).get("cell_key") != cell_key
            or payload.get("controller_config_sha256") != controller_identity
            or payload.get("topology_settings_sha256") != settings_identity
        ):
            raise ValueError(f"existing topology cell shard drift: {output_path}")
        return {
            "cell_key": cell_key,
            "path": str(output_path),
            "payload_sha256": payload_sha256,
            "reused": True,
        }
    cell = selected[0]
    summary = _summarize_cell(
        cell=cell,
        matrix_row=matrix[cell_key],
        production_contract=production_contract,
    )
    payload = {
        "schema_version": CELL_SHARD_SCHEMA,
        "evidence": "independent zero-oracle shared-controller production-path cell",
        "contract_payload_sha256": contract_identity,
        "controller_config_sha256": controller_identity,
        "topology_settings_sha256": settings_identity,
        "cell": summary,
        "costs": {
            "oracle_calls": 0,
            "docking_calls": 0,
            "modal_launches": 0,
            "gpu_seconds": 0,
        },
    }
    payload_sha256 = _publish_once(output_path, payload)
    return {
        "cell_key": cell_key,
        "path": str(output_path),
        "payload_sha256": payload_sha256,
        "reused": False,
    }


def reduce_cell_shards(
    *, repository_root: Path, shard_root: Path, output_path: Path
) -> dict[str, Any]:
    """Validate all eight independent shards and seal their deterministic reduction."""

    contract, contract_identity = load_contract(repository_root)
    cells, _matrix = _bound_cells(repository_root, contract)
    production_contract = _production_contract(repository_root, contract)
    experts = proposal_expert_vocabulary(production_contract)
    settings_identity = payload_identity(
        contract["controller"]["proposal"][GENERIC_TOPOLOGY_MACRO_EXPERT]
    )
    controller_identity = payload_identity(contract["controller"])
    rows = []
    shard_references = []
    for cell in cells:
        path = shard_root / f"{cell['cell_key']}.json"
        payload, payload_sha256 = _read_envelope(path)
        if (
            payload.get("schema_version") != CELL_SHARD_SCHEMA
            or payload.get("contract_payload_sha256") != contract_identity
            or payload.get("controller_config_sha256") != controller_identity
            or payload.get("topology_settings_sha256") != settings_identity
            or payload.get("cell", {}).get("cell_key") != cell["cell_key"]
            or payload.get("costs")
            != {
                "oracle_calls": 0,
                "docking_calls": 0,
                "modal_launches": 0,
                "gpu_seconds": 0,
            }
        ):
            raise ValueError(f"topology production cell shard changed: {path}")
        rows.append(payload["cell"])
        shard_references.append(
            {
                "cell_key": cell["cell_key"],
                "path": str(path.relative_to(repository_root)),
                "sha256": sha256_file(path),
                "payload_sha256": payload_sha256,
            }
        )

    raw_total = sum(row["funnel"]["raw_exact_unique"] for row in rows)
    exact_total = sum(row["funnel"]["exact_replay_verified"] for row in rows)
    selection_total = sum(row["funnel"]["archive_union_unique"] for row in rows)
    aggregate_cycles = {
        int(value)
        for row in rows
        for value in row["selection_ready_support"]["cycle_gain_counts"]
    }
    aggregate_plans = {
        value
        for row in rows
        for value in row["selection_ready_support"]["macro_plan_counts"]
    }
    recommendation = _recommend_next_cell(rows)
    producer_parameters = tuple(
        inspect.signature(propose_generic_topology_macro_programs).parameters
    )
    adapter_parameters = tuple(
        inspect.signature(generic_topology_macro_records).parameters
    )
    forbidden_generation_parameters = {
        "cell",
        "cell_key",
        "target",
        "delta",
        "support",
        "route_expert",
        "teacher_endpoint",
        "objective",
    }
    gates = {
        "complete_eight_cell_census": len(rows) == 8,
        "complete_six_terminal_loss_census": sum(
            row["evidence_role"] == "terminal_loss" for row in rows
        )
        == 6,
        "complete_two_contrasting_win_census": sum(
            row["evidence_role"] == "contrasting_win" for row in rows
        )
        == 2,
        "exact_execution_precision_one": raw_total > 0 and exact_total == raw_total,
        "same_expert_vocabulary_and_settings_all_cells": all(
            row["generation_telemetry"]["proposal_expert"]
            == GENERIC_TOPOLOGY_MACRO_EXPERT
            and row["generation_telemetry"]["attempts_per_plan"]
            == contract["controller"]["proposal"][GENERIC_TOPOLOGY_MACRO_EXPERT][
                "attempts_per_plan"
            ]
            for row in rows
        ),
        "endpoint_only_runtime_constraints": all(
            row["generation_telemetry"][
                "endpoint_constraints_applied_during_generation"
            ]
            is False
            and row["generation_telemetry"]["intermediate_endpoints_queried"] == 0
            and row["runtime_admission_order"][0] == "original_root_fiber"
            for row in rows
        ),
        "no_cell_specific_generation_branch": not (
            set(producer_parameters) | set(adapter_parameters)
        ).intersection(forbidden_generation_parameters),
        "aggregate_nonzero_selection_ready_unique": selection_total > 0,
        "aggregate_cycle_gain_diversity_at_least_two": len(aggregate_cycles) >= 2,
        "concrete_next_scored_cell_recommendation": recommendation is not None,
    }
    passed = all(gates.values())
    code_paths = (
        "src/compose_v4/control/generic_complete_program_composer_v2.py",
        "src/compose_v4/experiments/t4_integrated_route_fiber.py",
        "src/compose_v4/experiments/t4_shared_controller_scored_runtime.py",
        "src/compose_v4/experiments/t4_shared_controller_checkpoint.py",
        "src/compose_v4/experiments/t4_shared_controller_topology_production_gate.py",
        "modal_apps/t4_shared_controller_completion_v1_app.py",
    )
    payload = {
        "schema_version": RESULT_SCHEMA,
        "evidence": "computed zero-oracle shared-controller production-path gate",
        "decision": "PASS_PRODUCTION_PATH_GATE" if passed else "NO_PROMOTION",
        "contract": {
            "path": CONTRACT_RELATIVE_PATH,
            "sha256": sha256_file(repository_root / CONTRACT_RELATIVE_PATH),
            "payload_sha256": contract_identity,
        },
        "production_binding": {
            "proposal_experts": list(experts),
            "controller_config_sha256": controller_identity,
            "topology_settings_sha256": settings_identity,
            "topology_settings": copy.deepcopy(
                contract["controller"]["proposal"][GENERIC_TOPOLOGY_MACRO_EXPERT]
            ),
            "same_binding_all_cells": True,
            "producer_parameters": list(producer_parameters),
            "adapter_parameters": list(adapter_parameters),
        },
        "cell_shards": shard_references,
        "cells": rows,
        "aggregate": {
            "raw_exact_unique": raw_total,
            "exact_replay_verified": exact_total,
            "selection_ready_unique_sum": selection_total,
            "cells_with_selection_ready_candidates": sum(
                row["funnel"]["archive_union_unique"] > 0 for row in rows
            ),
            "macro_plan_diversity": len(aggregate_plans),
            "cycle_gain_diversity": len(aggregate_cycles),
            "cycle_gain_values": sorted(aggregate_cycles),
        },
        "next_scored_cell_recommendation": recommendation,
        "gates": {"values": gates, "passed": passed},
        "implementation": {
            "code_revision_before_result_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=repository_root, text=True
            ).strip(),
            "code_sha256": {
                path: sha256_file(repository_root / path) for path in code_paths
            },
            "python": platform.python_version(),
            "rdkit": rdBase.rdkitVersion,
        },
        "costs": {
            "oracle_calls": 0,
            "docking_calls": 0,
            "modal_launches": 0,
            "gpu_seconds": 0,
        },
        "claim_boundary": contract["claim_boundary"],
    }
    payload_sha256 = _publish_once(output_path, payload)
    return {
        "decision": payload["decision"],
        "payload_sha256": payload_sha256,
        "recommendation": recommendation,
        "aggregate": payload["aggregate"],
        "gates": payload["gates"],
    }


def run_gate(
    *, repository_root: Path, output_path: Path, shard_root: Path | None = None
) -> dict[str, Any]:
    """Run resumable cell shards sequentially, then reduce them.

    The command-line wrapper uses process-level parallelism for the same shard
    function.  Keeping this sequential entry point makes the scientific operation
    easy to test without changing any seed or reduction semantics.
    """

    contract, _contract_identity = load_contract(repository_root)
    root = shard_root or output_path.parent / "cell_shards"
    for cell in contract["cells"]:
        run_cell_shard(
            repository_root=repository_root,
            cell_key=cell["cell_key"],
            output_path=root / f"{cell['cell_key']}.json",
        )
    return reduce_cell_shards(
        repository_root=repository_root,
        shard_root=root,
        output_path=output_path,
    )


__all__ = [
    "ARTIFACT_ROOT_RELATIVE_PATH",
    "CELL_SHARD_SCHEMA",
    "CONTRACT_RELATIVE_PATH",
    "load_contract",
    "reduce_cell_shards",
    "run_cell_shard",
    "run_gate",
]
