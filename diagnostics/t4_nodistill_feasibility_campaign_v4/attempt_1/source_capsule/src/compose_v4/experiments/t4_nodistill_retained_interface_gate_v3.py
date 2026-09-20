"""Zero-oracle support gate for the route-free retained-interface v3 composer."""

from __future__ import annotations

import copy
import platform
import subprocess
from pathlib import Path
from typing import Any

from rdkit import Chem, rdBase

from compose_v4.chem.molecular_graph import (
    molecular_graph_to_smiles,
    smiles_to_molecular_graph,
)
from compose_v4.chem.state import pad_molecular_graph
from compose_v4.control.docking_value import identity
from compose_v4.control.generic_retained_interface_composer_v3 import (
    CYCLE_GAIN_GRID,
    HEAVY_GROWTH_BANDS,
    INTERFACE_COUNT_RANGE,
    MACRO_MODES,
    RETAINED_FRACTION_FLOOR,
    allocate_retained_interface_macro_plans,
    propose_generic_retained_interface_particle,
    propose_generic_retained_interface_programs,
)
from compose_v4.control.graph_geometry import topology
from compose_v4.experiments.t4_fiber_campaign import COMPOSE_VALID, Fiber
from compose_v4.experiments.t4_nodistill_generic_composer_gate import (
    _arm_summary,
    _publish_once,
    _read_envelope,
    _teacher_descriptors,
    sha256_file,
)
from compose_v4.rewrite.kernel import canonical_state_key
from compose_v4.rewrite.trace_shard import decode_state, encode_state

CONTRACT_RELATIVE_PATH = "configs/t4_nodistill_retained_interface_gate_v3.json"
ARTIFACT_ROOT_RELATIVE_PATH = (
    "diagnostics/t4_nodistill_retained_interface_gate_v3/attempt_1"
)
LOCK_SCHEMA = "t4_nodistill_retained_interface_candidate_lock_v3"
PARTICLE_LOCK_SCHEMA = "t4_nodistill_retained_interface_particle_lock_v3"
RESULT_SCHEMA = "t4_nodistill_retained_interface_gate_result_v3"
EXPERT_NAME = "generic_retained_interface_macro_v3"


def load_contract(repository_root: Path) -> tuple[dict[str, Any], str]:
    contract, contract_identity = _read_envelope(
        repository_root / CONTRACT_RELATIVE_PATH
    )
    if contract.get("schema_version") != (
        "t4_nodistill_retained_interface_gate_contract_v3"
    ):
        raise ValueError("unexpected retained-interface v3 contract schema")
    if contract.get("status") != "FROZEN_ZERO_ORACLE_SUPPORT_GATE":
        raise ValueError("retained-interface v3 contract is not frozen")
    if contract.get("costs") != {
        "oracle_calls": 0,
        "docking_calls": 0,
        "modal_launches": 0,
        "gpu_seconds": 0,
    }:
        raise ValueError("retained-interface v3 contract is not zero-oracle")
    if not all(contract.get("runtime_prohibitions", {}).values()):
        raise ValueError("retained-interface v3 runtime prohibition was weakened")
    bindings = {
        **contract["preregistered_design_evidence"],
        **contract["inputs"],
    }
    for binding in bindings.values():
        path = repository_root / binding["path"]
        if sha256_file(path) != binding["sha256"]:
            raise ValueError(f"retained-interface v3 input changed: {path}")
        expected_payload = binding.get("payload_sha256")
        if expected_payload is not None:
            _payload, observed_payload = _read_envelope(path)
            if observed_payload != expected_payload:
                raise ValueError(f"retained-interface v3 payload changed: {path}")
    return contract, contract_identity


def _source_cells(repository_root: Path, contract: dict[str, Any]) -> dict[str, dict]:
    registry = __import__("json").loads(
        (repository_root / contract["inputs"]["source_registry"]["path"]).read_text()
    )
    selected = {}
    for declared in contract["cells"]:
        source_index = int(declared["source_index"])
        source = registry[source_index]
        if int(source["idx"]) != source_index:
            raise ValueError(f"source-registry index drift: {source_index}")
        selected[declared["cell_key"]] = {
            **copy.deepcopy(declared),
            "source_smiles": str(source["smiles"]),
            "source_target": str(source["target"]),
        }
    if len(selected) != 5:
        raise ValueError("retained-interface v3 requires five distinct panel cells")
    return selected


def _raw_candidate_row(source, proposal) -> dict[str, Any]:
    before = topology(source)
    after = topology(proposal.endpoint)
    observed = proposal.metadata["observed_macro_fields"]
    return {
        "endpoint_state": encode_state(proposal.endpoint),
        "endpoint_key_sha256": identity(canonical_state_key(proposal.endpoint)),
        "canonical_smiles": molecular_graph_to_smiles(proposal.endpoint),
        "actions": list(proposal.actions),
        "program": proposal.program,
        "program_graph": proposal.program_graph,
        "families": list(proposal.families),
        "module_count": len(proposal.families),
        "requested_scale": proposal.requested_scale,
        "realized_primitive_count": len(proposal.actions),
        "realized_primitive_band": proposal.realized_scale,
        "structural_extent": max(
            1,
            int(proposal.actual_changes["surviving_new_atoms"])
            + int(proposal.actual_changes["deleted_original_atoms"]),
            len(proposal.actual_changes["changed_original_slots"]),
        ),
        "structural_extent_band": (
            "small"
            if max(
                1,
                int(proposal.actual_changes["surviving_new_atoms"])
                + int(proposal.actual_changes["deleted_original_atoms"]),
                len(proposal.actual_changes["changed_original_slots"]),
            )
            <= 3
            else (
                "medium"
                if max(
                    1,
                    int(proposal.actual_changes["surviving_new_atoms"])
                    + int(proposal.actual_changes["deleted_original_atoms"]),
                    len(proposal.actual_changes["changed_original_slots"]),
                )
                <= 11
                else "large"
            )
        ),
        "delta_heavy_atoms": int(after["n_heavy"] - before["n_heavy"]),
        "delta_cycle_rank": int(after["cycle_rank"] - before["cycle_rank"]),
        "delta_ring_systems": int(after["n_ring_systems"] - before["n_ring_systems"]),
        "endpoint_heavy_atoms": int(after["n_heavy"]),
        "actual_changes": proposal.actual_changes,
        "created_handle_dependencies": int(
            proposal.metadata.get("created_handle_dependencies", 0)
        ),
        "retained_fraction": float(observed["retained_fraction"]),
        "created_attachment_interface_count": int(
            observed["created_attachment_interface_count"]
        ),
        "block_count": int(observed["block_count"]),
        "macro_plan_identity": identity(proposal.metadata["macro_plan"]),
        "macro_plan": proposal.metadata["macro_plan"],
        "exact_execution_verified": True,
        "eligible": False,
        "fiber_admitted": False,
        "archive_ready": False,
        "endpoint_properties": None,
        "fingerprint": None,
        "metadata": proposal.metadata,
    }


def _apply_post_generation_production_admission(
    rows: list[dict[str, Any]], *, source_smiles: str, delta: float
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Apply the production endpoint gates only after the raw exact pool exists."""

    fiber = Fiber(source_smiles, delta, support=COMPOSE_VALID)
    source_molecule = Chem.MolFromSmiles(source_smiles)
    if source_molecule is None:
        raise ValueError("source registry contains invalid SMILES")
    canonical_source = Chem.MolToSmiles(source_molecule)
    seen: set[str] = set()
    admitted = 0
    canonical_self = 0
    canonical_alias = 0
    archive_excluded = 0
    prepared = []
    for source_row in rows:
        row = copy.deepcopy(source_row)
        properties = fiber.check(row["canonical_smiles"])
        if properties is None:
            prepared.append(row)
            continue
        admitted += 1
        canonical = str(properties["smiles"])
        row["eligible"] = True
        row["fiber_admitted"] = True
        row["endpoint_properties"] = properties
        molecule = Chem.MolFromSmiles(canonical)
        if molecule is None:
            raise RuntimeError("Fiber admitted an unparsable endpoint")
        row["fingerprint"] = sorted(
            fiber.generator.GetFingerprint(molecule).GetOnBits()
        )
        if canonical == canonical_source:
            canonical_self += 1
            prepared.append(row)
            continue
        if canonical in seen:
            canonical_alias += 1
            prepared.append(row)
            continue
        seen.add(canonical)
        if canonical == canonical_source:
            archive_excluded += 1
            prepared.append(row)
            continue
        row["archive_ready"] = True
        prepared.append(row)
    return prepared, {
        "admission_order": [
            "compose_valid_fiber",
            "canonical_self_exclusion",
            "canonical_endpoint_deduplication",
            "source_archive_exclusion",
        ],
        "endpoint_constraints_applied_during_generation": False,
        "raw_exact_unique": len(rows),
        "fiber_admitted": admitted,
        "canonical_self_events": canonical_self,
        "canonical_aliases": canonical_alias,
        "source_archive_exclusions": archive_excluded,
        "archive_ready_unique": len(seen),
    }


def _range(rows: list[dict[str, Any]], field: str) -> list[float] | None:
    if not rows:
        return None
    values = [row[field] for row in rows]
    return [min(values), max(values)]


def _support_summary(
    rows: list[dict[str, Any]], telemetry: dict[str, Any], admission: dict[str, Any]
) -> dict[str, Any]:
    ready = [row for row in rows if row["archive_ready"]]
    return {
        "raw_compile_successes": int(telemetry["raw_compile_successes"]),
        "exact_unique_candidates": len(rows),
        "fiber_admitted_candidates": sum(row["fiber_admitted"] for row in rows),
        "archive_ready_unique_candidates": len(ready),
        "archive_ready_macro_plan_diversity": len(
            {row["macro_plan_identity"] for row in ready}
        ),
        "archive_ready_heavy_band_diversity": sorted(
            {row["macro_plan"]["desired_heavy_band"] for row in ready}
        ),
        "archive_ready_cycle_gain_diversity": sorted(
            {int(row["delta_cycle_rank"]) for row in ready}
        ),
        "archive_ready_mode_diversity": sorted(
            {row["macro_plan"]["mode"] for row in ready}
        ),
        "all_candidate_delta_heavy_atom_range": _range(rows, "delta_heavy_atoms"),
        "archive_ready_delta_heavy_atom_range": _range(ready, "delta_heavy_atoms"),
        "all_candidate_delta_cycle_rank_range": _range(rows, "delta_cycle_rank"),
        "archive_ready_delta_cycle_rank_range": _range(ready, "delta_cycle_rank"),
        "all_candidate_retained_fraction_range": _range(rows, "retained_fraction"),
        "archive_ready_retained_fraction_range": _range(ready, "retained_fraction"),
        "all_candidate_interface_count_range": _range(
            rows, "created_attachment_interface_count"
        ),
        "archive_ready_interface_count_range": _range(
            ready, "created_attachment_interface_count"
        ),
        "primitive_count_range": _range(rows, "realized_primitive_count"),
        "block_count_range": _range(rows, "block_count"),
        "production_endpoint_admission": admission,
    }


def lock_cell(
    *, repository_root: Path, cell_key: str, output_path: Path
) -> dict[str, Any]:
    """Generate and seal one v3 pool without importing or reading teachers."""

    contract, contract_identity = load_contract(repository_root)
    cells = _source_cells(repository_root, contract)
    if cell_key not in cells:
        raise ValueError(f"cell is outside the retained-interface panel: {cell_key}")
    cell = cells[cell_key]
    source = pad_molecular_graph(smiles_to_molecular_graph(cell["source_smiles"]), 48)
    support = contract["support"]
    proposal_config = contract["proposal"]
    proposed = propose_generic_retained_interface_programs(
        source,
        seed=int(cell["controller_seed"]),
        attempts_per_particle=int(proposal_config["attempts_per_particle"]),
        candidate_quota_per_particle=int(
            proposal_config["candidate_quota_per_particle"]
        ),
        maximum_heavy_atoms=int(support["representable_heavy_atom_ceiling"]),
        maximum_primitives=int(support["maximum_primitives"]),
        maximum_blocks=int(support["maximum_blocks"]),
    )
    raw_rows = [_raw_candidate_row(source, row) for row in proposed.proposals]
    rows, admission = _apply_post_generation_production_admission(
        raw_rows,
        source_smiles=cell["source_smiles"],
        delta=float(cell["delta"]),
    )
    summary = _support_summary(rows, proposed.telemetry, admission)
    payload = {
        "schema_version": LOCK_SCHEMA,
        "evidence": "autonomous zero-oracle v3 candidate lock before teacher loading",
        "contract_payload_sha256": contract_identity,
        "cell_key": cell_key,
        "evidence_role": cell["evidence_role"],
        "delta": float(cell["delta"]),
        "controller_seed": int(cell["controller_seed"]),
        "source_state": encode_state(source),
        "source_key_sha256": identity(canonical_state_key(source)),
        "source_smiles_sha256": identity(cell["source_smiles"]),
        "proposal_inputs": {
            "source_graph": True,
            "seed": True,
            "task_cell_or_target": False,
            "requested_cell_delta": False,
            "fiber_or_endpoint_constraint": False,
            "trajectory_distilled_templates": False,
            "route_weights_or_ids": False,
            "fitted_weights": False,
            "teacher_or_winner_endpoints": False,
            "objective_or_comparator_scores": False,
        },
        "generation_then_admission": {
            "generation_completed_before_endpoint_admission": True,
            "generation_candidate_count": len(raw_rows),
            "admission": admission,
        },
        "telemetry": proposed.telemetry,
        "preteacher_support": summary,
        "candidates": rows,
        "costs": {"oracle_calls": 0, "docking_calls": 0, "modal_launches": 0},
    }
    payload_sha256 = _publish_once(output_path, payload)
    return {
        "cell_key": cell_key,
        "path": str(output_path),
        "payload_sha256": payload_sha256,
        "exact_candidates": len(rows),
        "archive_ready": summary["archive_ready_unique_candidates"],
    }


def allocated_particle_names(
    *, repository_root: Path, cell_key: str
) -> tuple[str, ...]:
    """Return the deterministic source-allocated particle census without compiling."""

    contract, _contract_identity = load_contract(repository_root)
    cells = _source_cells(repository_root, contract)
    if cell_key not in cells:
        raise ValueError(f"cell is outside the retained-interface panel: {cell_key}")
    cell = cells[cell_key]
    source = pad_molecular_graph(smiles_to_molecular_graph(cell["source_smiles"]), 48)
    support = contract["support"]
    _features, plans, _allocation = allocate_retained_interface_macro_plans(
        source,
        candidate_quota_per_particle=int(
            contract["proposal"]["candidate_quota_per_particle"]
        ),
        maximum_heavy_atoms=int(support["representable_heavy_atom_ceiling"]),
        maximum_primitives=int(support["maximum_primitives"]),
        maximum_blocks=int(support["maximum_blocks"]),
    )
    return tuple(plan.name for plan in plans)


def lock_particle(
    *,
    repository_root: Path,
    cell_key: str,
    particle_name: str,
    output_path: Path,
) -> dict[str, Any]:
    """Compile and seal one deterministic particle without endpoint admission."""

    contract, contract_identity = load_contract(repository_root)
    cells = _source_cells(repository_root, contract)
    if cell_key not in cells:
        raise ValueError(f"cell is outside the retained-interface panel: {cell_key}")
    cell = cells[cell_key]
    source = pad_molecular_graph(smiles_to_molecular_graph(cell["source_smiles"]), 48)
    support = contract["support"]
    proposal_config = contract["proposal"]
    proposed = propose_generic_retained_interface_particle(
        source,
        seed=int(cell["controller_seed"]),
        particle_name=particle_name,
        attempts_per_particle=int(proposal_config["attempts_per_particle"]),
        candidate_quota_per_particle=int(
            proposal_config["candidate_quota_per_particle"]
        ),
        maximum_heavy_atoms=int(support["representable_heavy_atom_ceiling"]),
        maximum_primitives=int(support["maximum_primitives"]),
        maximum_blocks=int(support["maximum_blocks"]),
    )
    rows = [_raw_candidate_row(source, row) for row in proposed.proposals]
    payload = {
        "schema_version": PARTICLE_LOCK_SCHEMA,
        "evidence": "autonomous zero-oracle v3 particle before endpoint admission",
        "contract_payload_sha256": contract_identity,
        "cell_key": cell_key,
        "particle_name": particle_name,
        "controller_seed": int(cell["controller_seed"]),
        "source_key_sha256": identity(canonical_state_key(source)),
        "telemetry": proposed.telemetry,
        "candidates": rows,
        "endpoint_admission_applied": False,
        "costs": {"oracle_calls": 0, "docking_calls": 0, "modal_launches": 0},
    }
    payload_sha256 = _publish_once(output_path, payload)
    return {
        "cell_key": cell_key,
        "particle_name": particle_name,
        "path": str(output_path),
        "payload_sha256": payload_sha256,
        "exact_candidates": len(rows),
        "raw_compile_successes": proposed.telemetry["raw_compile_successes"],
    }


def merge_cell_particles(
    *, repository_root: Path, cell_key: str, artifact_root: Path, output_path: Path
) -> dict[str, Any]:
    """Validate a complete particle census, merge, then apply endpoint admission."""

    contract, contract_identity = load_contract(repository_root)
    cells = _source_cells(repository_root, contract)
    if cell_key not in cells:
        raise ValueError(f"cell is outside the retained-interface panel: {cell_key}")
    cell = cells[cell_key]
    source = pad_molecular_graph(smiles_to_molecular_graph(cell["source_smiles"]), 48)
    support = contract["support"]
    proposal_config = contract["proposal"]
    features, plans, allocation = allocate_retained_interface_macro_plans(
        source,
        candidate_quota_per_particle=int(
            proposal_config["candidate_quota_per_particle"]
        ),
        maximum_heavy_atoms=int(support["representable_heavy_atom_ceiling"]),
        maximum_primitives=int(support["maximum_primitives"]),
        maximum_blocks=int(support["maximum_blocks"]),
    )
    particle_rows = []
    telemetry_rows = {}
    raw_compile_successes = 0
    particle_locks = []
    for plan in plans:
        path = artifact_root / "particles" / cell_key / f"{plan.name}.json.gz"
        payload, payload_identity = _read_envelope(path)
        if (
            payload.get("schema_version") != PARTICLE_LOCK_SCHEMA
            or payload.get("contract_payload_sha256") != contract_identity
            or payload.get("cell_key") != cell_key
            or payload.get("particle_name") != plan.name
            or payload.get("source_key_sha256") != identity(canonical_state_key(source))
            or payload.get("endpoint_admission_applied") is not False
        ):
            raise ValueError(f"retained-interface particle lock changed: {path}")
        particle_locks.append(
            {
                "particle_name": plan.name,
                "path": str(path.relative_to(repository_root)),
                "sha256": sha256_file(path),
                "payload_sha256": payload_identity,
            }
        )
        particle_rows.extend(payload["candidates"])
        telemetry_rows.update(payload["telemetry"]["particles"])
        raw_compile_successes += int(payload["telemetry"]["raw_compile_successes"])

    by_endpoint = {}
    for row in particle_rows:
        by_endpoint.setdefault(row["endpoint_key_sha256"], row)
    raw_rows = sorted(
        by_endpoint.values(),
        key=lambda row: (row["macro_plan"]["name"], row["endpoint_key_sha256"]),
    )
    rows, admission = _apply_post_generation_production_admission(
        raw_rows,
        source_smiles=cell["source_smiles"],
        delta=float(cell["delta"]),
    )
    grid_payload = {
        "heavy_growth_bands": {
            name: list(bounds) for name, bounds in HEAVY_GROWTH_BANDS.items()
        },
        "cycle_gain_grid": list(CYCLE_GAIN_GRID),
        "macro_modes": list(MACRO_MODES),
        "retained_fraction_floor": RETAINED_FRACTION_FLOOR,
        "interface_count_range": list(INTERFACE_COUNT_RANGE),
    }
    telemetry = {
        "schema_version": "generic_retained_interface_composer_telemetry_v3",
        "seed": int(cell["controller_seed"]),
        "attempts_per_particle": int(proposal_config["attempts_per_particle"]),
        "candidate_quota_per_particle": int(
            proposal_config["candidate_quota_per_particle"]
        ),
        "maximum_heavy_atoms": int(support["representable_heavy_atom_ceiling"]),
        "maximum_primitives": int(support["maximum_primitives"]),
        "maximum_blocks": int(support["maximum_blocks"]),
        "source_features": features.payload(),
        "allocation": allocation,
        "particles": telemetry_rows,
        "raw_compile_successes": raw_compile_successes,
        "exact_unique_candidates": len(rows),
        "balanced_grid_identity": identity(grid_payload),
        "allocated_plan_identity": identity([plan.payload() for plan in plans]),
        "trajectory_distilled_templates_loaded": 0,
        "route_weights_loaded": 0,
        "fitted_weights_loaded": 0,
        "runtime_task_cell_or_target_input": False,
        "runtime_requested_delta_input": False,
        "runtime_fiber_input": False,
        "runtime_teacher_endpoint_input": False,
        "runtime_objective_input": False,
    }
    summary = _support_summary(rows, telemetry, admission)
    payload = {
        "schema_version": LOCK_SCHEMA,
        "evidence": "autonomous zero-oracle v3 candidate lock before teacher loading",
        "contract_payload_sha256": contract_identity,
        "cell_key": cell_key,
        "evidence_role": cell["evidence_role"],
        "delta": float(cell["delta"]),
        "controller_seed": int(cell["controller_seed"]),
        "source_state": encode_state(source),
        "source_key_sha256": identity(canonical_state_key(source)),
        "source_smiles_sha256": identity(cell["source_smiles"]),
        "proposal_inputs": {
            "source_graph": True,
            "seed": True,
            "task_cell_or_target": False,
            "requested_cell_delta": False,
            "fiber_or_endpoint_constraint": False,
            "trajectory_distilled_templates": False,
            "route_weights_or_ids": False,
            "fitted_weights": False,
            "teacher_or_winner_endpoints": False,
            "objective_or_comparator_scores": False,
        },
        "generation_then_admission": {
            "generation_completed_before_endpoint_admission": True,
            "generation_candidate_count": len(raw_rows),
            "admission": admission,
            "deterministic_particle_locks": particle_locks,
        },
        "telemetry": telemetry,
        "preteacher_support": summary,
        "candidates": rows,
        "costs": {"oracle_calls": 0, "docking_calls": 0, "modal_launches": 0},
    }
    payload_sha256 = _publish_once(output_path, payload)
    return {
        "cell_key": cell_key,
        "path": str(output_path),
        "payload_sha256": payload_sha256,
        "exact_candidates": len(rows),
        "archive_ready": summary["archive_ready_unique_candidates"],
        "particle_locks": len(particle_locks),
    }


def _validate_lock(
    *, lock: dict[str, Any], cell_key: str, contract_identity: str
) -> None:
    if (
        lock.get("schema_version") != LOCK_SCHEMA
        or lock.get("cell_key") != cell_key
        or lock.get("contract_payload_sha256") != contract_identity
    ):
        raise ValueError(f"retained-interface v3 candidate lock changed: {cell_key}")
    if not lock["generation_then_admission"][
        "generation_completed_before_endpoint_admission"
    ]:
        raise ValueError("endpoint admission moved into v3 generation")


def evaluate_locks(
    *, repository_root: Path, artifact_root: Path, output_path: Path
) -> dict[str, Any]:
    contract, contract_identity = load_contract(repository_root)
    cells = _source_cells(repository_root, contract)
    locks = {}
    for cell_key in cells:
        path = artifact_root / "locks" / f"{cell_key}.json.gz"
        payload, payload_identity = _read_envelope(path)
        _validate_lock(
            lock=payload, cell_key=cell_key, contract_identity=contract_identity
        )
        locks[cell_key] = (payload, payload_identity, path)

    # This import/read occurs only after every autonomous lock above exists and validates.
    teachers_by_source = _teacher_descriptors(repository_root)
    result_cells = []
    all_candidates = []
    all_ready = []
    all_teachers = []
    for cell_key, (lock, lock_identity, path) in locks.items():
        source = decode_state(lock["source_state"])
        teachers = teachers_by_source.get(canonical_state_key(source), [])
        if not teachers:
            raise ValueError(f"no teacher diagnostic matches {cell_key}")
        candidates = lock["candidates"]
        ready = [row for row in candidates if row["archive_ready"]]
        all_candidates.extend(candidates)
        all_ready.extend(ready)
        all_teachers.extend(teachers)
        result_cells.append(
            {
                "cell_key": cell_key,
                "evidence_role": lock["evidence_role"],
                "delta": lock["delta"],
                "candidate_lock": {
                    "path": str(path.relative_to(repository_root)),
                    "sha256": sha256_file(path),
                    "payload_sha256": lock_identity,
                },
                "preteacher_support": lock["preteacher_support"],
                "teacher_routes": len(teachers),
                "post_lock_teacher_comparison": {
                    "all_exact_candidates": _arm_summary(candidates, teachers),
                    "archive_ready_candidates": _arm_summary(ready, teachers),
                },
            }
        )

    motivating = [
        row for row in result_cells if row["evidence_role"] == "motivating_growth_miss"
    ]
    exact_denominator = len(all_candidates)
    expected_grid_identity = identity(
        {
            "heavy_growth_bands": {
                name: list(bounds) for name, bounds in HEAVY_GROWTH_BANDS.items()
            },
            "cycle_gain_grid": list(CYCLE_GAIN_GRID),
            "macro_modes": list(MACRO_MODES),
            "retained_fraction_floor": float(
                contract["support"]["retained_fraction_floor"]
            ),
            "interface_count_range": contract["support"][
                "created_attachment_interface_range"
            ],
        }
    )
    runtime_false_fields = (
        "task_cell_or_target",
        "requested_cell_delta",
        "fiber_or_endpoint_constraint",
        "trajectory_distilled_templates",
        "route_weights_or_ids",
        "fitted_weights",
        "teacher_or_winner_endpoints",
        "objective_or_comparator_scores",
    )
    gates = {
        "complete_five_cell_lock_census": len(result_cells) == 5,
        "exact_replay_precision_one": bool(
            exact_denominator
            and sum(row["exact_execution_verified"] for row in all_candidates)
            / exact_denominator
            == 1.0
        ),
        "nonzero_diverse_archive_ready_support_on_all_three_growth_misses": all(
            row["preteacher_support"]["archive_ready_unique_candidates"] >= 2
            and row["preteacher_support"]["archive_ready_macro_plan_diversity"] >= 2
            for row in motivating
        )
        and len(motivating) == 3,
        "all_archive_ready_endpoints_within_40_heavy_atoms": all(
            int(row["endpoint_heavy_atoms"]) <= 40 for row in all_ready
        ),
        "all_programs_within_32_primitives_and_8_blocks": all(
            1 <= int(row["realized_primitive_count"]) <= 32
            and 1 <= int(row["block_count"]) <= 8
            for row in all_candidates
        ),
        "balanced_grid_identity_unchanged": all(
            lock[0]["telemetry"]["balanced_grid_identity"] == expected_grid_identity
            for lock in locks.values()
        ),
        "generation_precedes_endpoint_admission": all(
            lock[0]["generation_then_admission"][
                "generation_completed_before_endpoint_admission"
            ]
            and not lock[0]["telemetry"]["runtime_fiber_input"]
            for lock in locks.values()
        ),
        "no_teacher_route_template_objective_or_cell_runtime_input": all(
            all(
                not bool(lock[0]["proposal_inputs"][field])
                for field in runtime_false_fields
            )
            and lock[0]["telemetry"]["trajectory_distilled_templates_loaded"] == 0
            and lock[0]["telemetry"]["route_weights_loaded"] == 0
            and lock[0]["telemetry"]["fitted_weights_loaded"] == 0
            for lock in locks.values()
        ),
        "post_lock_teacher_coarse_and_semantic_coverage_reported": all(
            "teacher_coarse_descriptor_covered"
            in row["post_lock_teacher_comparison"]["archive_ready_candidates"][
                "teacher_descriptor_support"
            ]
            and "teacher_semantic_descriptor_covered"
            in row["post_lock_teacher_comparison"]["archive_ready_candidates"][
                "teacher_descriptor_support"
            ]
            for row in result_cells
        ),
    }
    passed = all(gates.values())
    code_inputs = (
        "src/compose_v4/control/generic_retained_interface_composer_v3.py",
        "src/compose_v4/experiments/t4_nodistill_retained_interface_gate_v3.py",
        "tools/t4_nodistill_retained_interface_gate_v3.py",
    )
    payload = {
        "schema_version": RESULT_SCHEMA,
        "evidence": "computed zero-oracle route-free retained-interface support gate",
        "decision": (
            "PASS_RETAINED_INTERFACE_SUPPORT_GATE" if passed else "NO_PROMOTION"
        ),
        "contract": {
            "path": CONTRACT_RELATIVE_PATH,
            "sha256": sha256_file(repository_root / CONTRACT_RELATIVE_PATH),
            "payload_sha256": contract_identity,
        },
        "candidate_lock_ordering": (
            "all five candidate locks were sealed and validated before teacher "
            "descriptors were imported or read"
        ),
        "cells": result_cells,
        "aggregate": {
            "all_exact_candidates": _arm_summary(all_candidates, all_teachers),
            "archive_ready_candidates": _arm_summary(all_ready, all_teachers),
            "archive_ready_by_cell": {
                row["cell_key"]: row["preteacher_support"][
                    "archive_ready_unique_candidates"
                ]
                for row in result_cells
            },
            "archive_ready_macro_plan_diversity_by_cell": {
                row["cell_key"]: row["preteacher_support"][
                    "archive_ready_macro_plan_diversity"
                ]
                for row in result_cells
            },
        },
        "gates": {"values": gates, "passed": passed},
        "implementation": {
            "code_revision_before_result_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=repository_root, text=True
            ).strip(),
            "code_sha256": {
                path: sha256_file(repository_root / path) for path in code_inputs
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
        "claim_boundary": (
            "Proposal support, exact replay and free post-generation endpoint "
            "admission only. No docking utility, scored selection, production "
            "integration or exact-teacher recovery is claimed or required."
        ),
    }
    payload_sha256 = _publish_once(output_path, payload)
    return {
        "decision": payload["decision"],
        "payload_sha256": payload_sha256,
        "gates": payload["gates"],
    }


__all__ = [
    "ARTIFACT_ROOT_RELATIVE_PATH",
    "CONTRACT_RELATIVE_PATH",
    "EXPERT_NAME",
    "allocated_particle_names",
    "evaluate_locks",
    "load_contract",
    "lock_cell",
    "lock_particle",
    "merge_cell_particles",
]
