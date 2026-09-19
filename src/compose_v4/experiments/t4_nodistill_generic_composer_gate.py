"""Zero-oracle support gate for the route-free generic program composer."""

from __future__ import annotations

import gzip
import hashlib
import json
import platform
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from rdkit import rdBase

from compose_v4.chem.molecular_graph import (
    molecular_graph_to_smiles,
    smiles_to_molecular_graph,
)
from compose_v4.chem.state import pad_molecular_graph
from compose_v4.control.docking_value import identity
from compose_v4.control.edit_program_graph import changed_input_sites
from compose_v4.control.generic_complete_program_composer import (
    SCALE_BANDS,
    primitive_scale,
    propose_generic_complete_programs,
)
from compose_v4.control.graph_geometry import topology
from compose_v4.control.retained_core_pruning import enumerate_retained_core_prunes
from compose_v4.experiments.t4_fiber_campaign import COMPOSE_VALID, Fiber
from compose_v4.rewrite.kernel import canonical_state_key
from compose_v4.rewrite.trace_shard import decode_state, encode_state

CONTRACT_RELATIVE_PATH = "configs/t4_nodistill_generic_composer_gate_v1.json"
ARTIFACT_ROOT_RELATIVE_PATH = "diagnostics/t4_nodistill_generic_composer_gate_v1/attempt_1"
LOCK_SCHEMA = "t4_nodistill_generic_composer_candidate_lock_v1"
RESULT_SCHEMA = "t4_nodistill_generic_composer_gate_result_v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    raw = gzip.decompress(path.read_bytes()) if path.suffix == ".gz" else path.read_bytes()
    return json.loads(raw)


def _read_envelope(path: Path) -> tuple[dict[str, Any], str]:
    envelope = _read_json(path)
    payload = envelope.get("payload")
    payload_sha256 = envelope.get("payload_sha256")
    if not isinstance(payload, dict) or identity(payload) != payload_sha256:
        raise ValueError(f"invalid self-hashed envelope: {path}")
    return payload, str(payload_sha256)


def _publish_once(path: Path, payload: dict[str, Any]) -> str:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite sealed artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload_sha256 = identity(payload)
    encoded = (
        json.dumps(
            {"payload": payload, "payload_sha256": payload_sha256},
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode()
    temporary = path.with_suffix(path.suffix + ".tmp")
    if path.suffix == ".gz":
        temporary.write_bytes(gzip.compress(encoded, mtime=0))
    else:
        temporary.write_bytes(encoded)
    temporary.replace(path)
    return payload_sha256


def load_contract(repository_root: Path) -> tuple[dict[str, Any], str]:
    path = repository_root / CONTRACT_RELATIVE_PATH
    contract, contract_identity = _read_envelope(path)
    if contract.get("schema_version") != ("t4_nodistill_generic_composer_gate_contract_v1"):
        raise ValueError("unexpected generic-composer contract schema")
    if contract.get("status") != "FROZEN_ZERO_ORACLE_SUPPORT_GATE":
        raise ValueError("generic-composer contract is not frozen")
    if contract.get("costs") != {
        "oracle_calls": 0,
        "docking_calls": 0,
        "modal_launches": 0,
        "gpu_seconds": 0,
    }:
        raise ValueError("generic-composer contract is not zero-oracle")
    if not all(contract.get("runtime_prohibitions", {}).values()):
        raise ValueError("generic-composer runtime prohibition was weakened")
    for row in contract["inputs"].values():
        input_path = repository_root / row["path"]
        if sha256_file(input_path) != row["sha256"]:
            raise ValueError(f"generic-composer input changed: {input_path}")
    audit, audit_identity = _read_envelope(
        repository_root / contract["inputs"]["motivating_audit"]["path"]
    )
    if audit_identity != contract["inputs"]["motivating_audit"]["payload_sha256"]:
        raise ValueError("motivating PARP audit payload changed")
    if audit.get("schema_version") != ("t4_compose_nodistill_parp1_full_vs_nodistill_audit_v1"):
        raise ValueError("unexpected motivating PARP audit schema")
    return contract, contract_identity


def _source_cells(repository_root: Path, contract: dict[str, Any]) -> dict[str, dict]:
    sources: dict[str, dict] = {}
    for source_name, input_name in (
        ("parp1", "parp1_source_contract"),
        ("transfer", "transfer_source_contract"),
    ):
        document = json.loads(
            (repository_root / contract["inputs"][input_name]["path"]).read_text()
        )
        for row in document["cells"]:
            sources[row["cell_key"]] = {**row, "source_contract": source_name}
    selected: dict[str, dict] = {}
    for declared in contract["cells"]:
        key = declared["cell_key"]
        source = sources.get(key)
        if source is None or source["source_contract"] != declared["source_contract"]:
            raise ValueError(f"generic-composer source cell changed: {key}")
        selected[key] = {**source, "controller_seed": declared["controller_seed"]}
    if len(selected) != 6:
        raise ValueError("generic-composer gate requires exactly six distinct cells")
    return selected


def _extent_scale(actual_changes: dict[str, Any]) -> tuple[int, str]:
    extent = max(
        1,
        int(actual_changes["surviving_new_atoms"]) + int(actual_changes["deleted_original_atoms"]),
        len(actual_changes["changed_original_slots"]),
    )
    return extent, "small" if extent <= 3 else "medium" if extent <= 11 else "large"


def _candidate_row(
    *,
    arm: str,
    source,
    endpoint,
    actions: tuple[dict[str, Any], ...],
    families: tuple[str, ...],
    requested_scale: str,
    metadata: dict[str, Any],
    program: dict[str, Any] | None,
    program_graph: dict[str, Any] | None,
    fiber: Fiber,
) -> dict[str, Any]:
    endpoint_smiles = molecular_graph_to_smiles(endpoint)
    exact_changes = changed_input_sites(source, endpoint, actions)
    primitive_count = len(actions)
    realized_scale = primitive_scale(primitive_count)
    extent, extent_band = _extent_scale(exact_changes)
    before = topology(source)
    after = topology(endpoint)
    eligibility = fiber.check(endpoint_smiles)
    return {
        "arm": arm,
        "endpoint_state": encode_state(endpoint),
        "endpoint_key_sha256": identity(canonical_state_key(endpoint)),
        "canonical_smiles": endpoint_smiles,
        "actions": list(actions),
        "program": program,
        "program_graph": program_graph,
        "families": list(families),
        "module_count": len(families),
        "requested_scale": requested_scale,
        "realized_primitive_count": primitive_count,
        "realized_primitive_band": realized_scale,
        "structural_extent": extent,
        "structural_extent_band": extent_band,
        "delta_heavy_atoms": after["n_heavy"] - before["n_heavy"],
        "delta_cycle_rank": after["cycle_rank"] - before["cycle_rank"],
        "delta_ring_systems": after["n_ring_systems"] - before["n_ring_systems"],
        "actual_changes": exact_changes,
        "created_handle_dependencies": int(metadata.get("created_handle_dependencies", 0)),
        "exact_execution_verified": True,
        "eligible": eligibility is not None,
        "eligibility": eligibility,
        "metadata": metadata,
    }


def lock_cell(*, repository_root: Path, cell_key: str, output_path: Path) -> dict[str, Any]:
    """Generate and seal candidates without importing or reading teacher data."""

    contract, contract_identity = load_contract(repository_root)
    cells = _source_cells(repository_root, contract)
    if cell_key not in cells:
        raise ValueError(f"cell is outside the frozen generic-composer panel: {cell_key}")
    cell = cells[cell_key]
    source = pad_molecular_graph(smiles_to_molecular_graph(cell["source_smiles"]), 48)
    fiber = Fiber(cell["source_smiles"], float(cell["delta"]), support=COMPOSE_VALID)
    proposal = contract["proposal"]

    composer = propose_generic_complete_programs(
        source,
        seed=int(cell["controller_seed"]),
        per_scale=int(proposal["composer_candidates_per_scale"]),
        attempts_per_scale=int(proposal["composer_attempts_per_scale"]),
        maximum_primitives=int(contract["support"]["maximum_primitives"]),
        maximum_blocks=int(contract["support"]["maximum_blocks"]),
    )
    composer_rows = [
        _candidate_row(
            arm="generic_complete_program_composer",
            source=source,
            endpoint=row.endpoint,
            actions=row.actions,
            families=row.families,
            requested_scale=row.requested_scale,
            metadata=row.metadata,
            program=row.program,
            program_graph=row.program_graph,
            fiber=fiber,
        )
        for row in composer.proposals
    ]

    retained_config = proposal["retained_core"]
    retained = enumerate_retained_core_prunes(
        source,
        maximum_fragment_atoms=int(retained_config["maximum_fragment_atoms"]),
        maximum_stages=int(retained_config["maximum_stages"]),
        maximum_primitives=int(retained_config["maximum_primitives"]),
        maximum_prefixes=int(retained_config["maximum_prefixes"]),
    )
    retained_by_scale: dict[str, list] = {band: [] for band in SCALE_BANDS}
    for row in retained:
        retained_by_scale[primitive_scale(len(row.actions))].append(row)
    baseline_rows = []
    baseline_limit = int(proposal["baseline_candidates_per_scale"])
    for scale in SCALE_BANDS:
        selected = sorted(
            retained_by_scale[scale], key=lambda row: canonical_state_key(row.product)
        )[:baseline_limit]
        for row in selected:
            baseline_rows.append(
                _candidate_row(
                    arm="generic_retained_core_only",
                    source=source,
                    endpoint=row.product,
                    actions=row.actions,
                    families=tuple(stage["family"] for stage in row.stages),
                    requested_scale=scale,
                    metadata={
                        "schema_version": "generic_retained_core_baseline_v1",
                        "created_handle_dependencies": 0,
                        "initial_stored_complete_routes": 0,
                        "source_library_rows_loaded": 0,
                        "trajectory_distilled_templates_loaded": 0,
                        "intermediate_task_evaluations": 0,
                        "task_oracle_information": False,
                    },
                    program=None,
                    program_graph=None,
                    fiber=fiber,
                )
            )

    payload = {
        "schema_version": LOCK_SCHEMA,
        "evidence": "autonomous zero-oracle candidate lock before teacher loading",
        "contract_payload_sha256": contract_identity,
        "cell_key": cell_key,
        "delta": float(cell["delta"]),
        "controller_seed": int(cell["controller_seed"]),
        "source_state": encode_state(source),
        "source_key_sha256": identity(canonical_state_key(source)),
        "source_smiles_sha256": identity(cell["source_smiles"]),
        "proposal_inputs": {
            "source_graph": True,
            "seed": True,
            "task_cell_or_target": False,
            "trajectory_distilled_templates": False,
            "route_weights_or_ids": False,
            "teacher_or_winner_endpoints": False,
            "objective_or_comparator_scores": False,
        },
        "arms": {
            "generic_complete_program_composer": {
                "maximum_candidate_budget": 3 * int(proposal["composer_candidates_per_scale"]),
                "telemetry": composer.telemetry,
                "candidates": composer_rows,
            },
            "generic_retained_core_only": {
                "maximum_candidate_budget": 3 * baseline_limit,
                "enumerated_exact_programs": len(retained),
                "available_by_scale": {band: len(retained_by_scale[band]) for band in SCALE_BANDS},
                "candidates": baseline_rows,
            },
        },
        "costs": {"oracle_calls": 0, "docking_calls": 0, "modal_launches": 0},
    }
    payload_sha256 = _publish_once(output_path, payload)
    return {
        "cell_key": cell_key,
        "path": str(output_path),
        "payload_sha256": payload_sha256,
        "composer_candidates": len(composer_rows),
        "baseline_candidates": len(baseline_rows),
    }


def _coarse_delta(value: int) -> str:
    if value <= -7:
        return "large_shrink"
    if value <= -3:
        return "medium_shrink"
    if value <= -1:
        return "local_shrink"
    if value == 0:
        return "same"
    if value <= 2:
        return "local_growth"
    if value <= 6:
        return "medium_growth"
    return "large_growth"


def _coarse_cycle(value: int) -> str:
    if value <= -2:
        return "multi_cycle_loss"
    if value == -1:
        return "one_cycle_loss"
    if value == 0:
        return "same"
    if value == 1:
        return "one_cycle_gain"
    return "multi_cycle_gain"


def _site_band(value: int) -> str:
    return "one" if value == 1 else "two" if value == 2 else "multi"


def _rewrite_mode(actual_changes: dict[str, Any]) -> str:
    deleted = int(actual_changes["deleted_original_atoms"])
    created = int(actual_changes["surviving_new_atoms"])
    if deleted and created:
        return "release_and_reconstruct"
    if deleted:
        return "release_only"
    if created:
        return "construct_only"
    return "retained_topology_or_attribute_rewrite"


def _semantic_descriptor(row: dict[str, Any]) -> tuple:
    changes = row["actual_changes"]
    regions = len(changes["source_induced_components"])
    return (
        row["realized_primitive_band"],
        _rewrite_mode(changes),
        _site_band(regions),
        _coarse_cycle(int(row["delta_cycle_rank"])),
    )


def _descriptors(row: dict[str, Any]) -> tuple[tuple, tuple]:
    exact = (
        row["realized_primitive_band"],
        int(row["delta_heavy_atoms"]),
        int(row["delta_cycle_rank"]),
        int(row["actual_changes"]["changed_site_count"]),
    )
    coarse = (
        row["realized_primitive_band"],
        _coarse_delta(int(row["delta_heavy_atoms"])),
        _coarse_cycle(int(row["delta_cycle_rank"])),
        _site_band(int(row["actual_changes"]["changed_site_count"])),
    )
    return exact, coarse


def _teacher_descriptors(repository_root: Path) -> dict[str, list[dict[str, Any]]]:
    """Load answer-known teachers only after every autonomous lock is sealed."""

    import sys

    sys.path.insert(0, str(repository_root))
    try:
        from tools.t4_structural_subgoal_audit import teacher_traces
    finally:
        sys.path.pop(0)

    grouped: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for teacher in teacher_traces():
        trace = teacher["trace"]
        source = decode_state(trace["states"][0])
        endpoint = decode_state(trace["states"][-1])
        actions = tuple(trace["actions"])
        changes = changed_input_sites(source, endpoint, actions)
        before = topology(source)
        after = topology(endpoint)
        primitive_count = len(actions)
        row = {
            "program_id_sha256": identity(teacher["program_id"]),
            "endpoint_key_sha256": identity(canonical_state_key(endpoint)),
            "realized_primitive_count": primitive_count,
            "realized_primitive_band": primitive_scale(primitive_count),
            "delta_heavy_atoms": after["n_heavy"] - before["n_heavy"],
            "delta_cycle_rank": after["cycle_rank"] - before["cycle_rank"],
            "actual_changes": changes,
        }
        exact, coarse = _descriptors(row)
        row["exact_descriptor"] = list(exact)
        row["coarse_descriptor"] = list(coarse)
        row["semantic_descriptor"] = list(_semantic_descriptor(row))
        grouped[canonical_state_key(source)].append(row)
    return grouped


def _arm_summary(
    candidates: list[dict[str, Any]], teachers: list[dict[str, Any]]
) -> dict[str, Any]:
    exact_teacher = {tuple(row["exact_descriptor"]) for row in teachers}
    coarse_teacher = {tuple(row["coarse_descriptor"]) for row in teachers}
    semantic_teacher = {tuple(row["semantic_descriptor"]) for row in teachers}
    teacher_endpoints = {row["endpoint_key_sha256"] for row in teachers}
    exact_candidates = [_descriptors(row)[0] for row in candidates]
    coarse_candidates = [_descriptors(row)[1] for row in candidates]
    semantic_candidates = [_semantic_descriptor(row) for row in candidates]
    exact_hits = [row in exact_teacher for row in exact_candidates]
    coarse_hits = [row in coarse_teacher for row in coarse_candidates]
    semantic_hits = [row in semantic_teacher for row in semantic_candidates]
    families = Counter(family for row in candidates for family in row.get("families", []))
    sequences = Counter("+".join(row.get("families", [])) for row in candidates)
    primitive_bands = Counter(row["realized_primitive_band"] for row in candidates)
    extent_bands = Counter(row["structural_extent_band"] for row in candidates)
    rewrite_modes = Counter(_rewrite_mode(row["actual_changes"]) for row in candidates)
    ring_families = {
        "append_ring",
        "construct_substituted_ring",
        "cycle_close",
        "cycle_open",
        "fuse_ring",
        "ring_path_remodel",
        "ring_system_restate",
    }
    nonring_families = {
        "bond_reroute",
        "carbonyl_insert",
        "functionalize",
        "heteroatom_substitute",
        "segment_grow",
        "segment_replace",
        "substituent_delete",
    }

    def family_set(row: dict[str, Any]) -> set[str]:
        return set(row.get("families", []))

    def change_regions(row: dict[str, Any]) -> int:
        return len(row["actual_changes"]["source_induced_components"])

    return {
        "unique_exact_endpoints": len({row["endpoint_key_sha256"] for row in candidates}),
        "eligible_endpoints": sum(bool(row["eligible"]) for row in candidates),
        "eligibility_precision": (
            sum(bool(row["eligible"]) for row in candidates) / len(candidates)
            if candidates
            else None
        ),
        "exact_execution": {
            "numerator": sum(bool(row["exact_execution_verified"]) for row in candidates),
            "denominator": len(candidates),
            "precision": (
                sum(bool(row["exact_execution_verified"]) for row in candidates) / len(candidates)
                if candidates
                else None
            ),
        },
        "primitive_band_counts": dict(sorted(primitive_bands.items())),
        "structural_extent_band_counts": dict(sorted(extent_bands.items())),
        "primitive_count_range": (
            [
                min(row["realized_primitive_count"] for row in candidates),
                max(row["realized_primitive_count"] for row in candidates),
            ]
            if candidates
            else None
        ),
        "delta_heavy_atom_range": (
            [
                min(row["delta_heavy_atoms"] for row in candidates),
                max(row["delta_heavy_atoms"] for row in candidates),
            ]
            if candidates
            else None
        ),
        "delta_cycle_rank_range": (
            [
                min(row["delta_cycle_rank"] for row in candidates),
                max(row["delta_cycle_rank"] for row in candidates),
            ]
            if candidates
            else None
        ),
        "multi_module_candidates": sum(row["module_count"] >= 2 for row in candidates),
        "dependency_bearing_candidates": sum(
            row["created_handle_dependencies"] > 0 for row in candidates
        ),
        "semantic_composition": {
            "rewrite_mode_counts": dict(sorted(rewrite_modes.items())),
            "release_and_reconstruct_candidates": sum(
                _rewrite_mode(row["actual_changes"]) == "release_and_reconstruct"
                for row in candidates
            ),
            "joint_constructed_attachment_candidates": sum(
                int(row["actual_changes"]["surviving_new_atoms"]) > 0
                and int(row["actual_changes"]["changed_site_count"]) >= 2
                for row in candidates
            ),
            "coordinated_multi_region_candidates": sum(
                change_regions(row) >= 2 for row in candidates
            ),
            "ring_plus_nonring_compositions": sum(
                bool(family_set(row).intersection(ring_families))
                and bool(family_set(row).intersection(nonring_families))
                for row in candidates
            ),
            "linker_or_reroute_compositions": sum(
                bool(
                    family_set(row).intersection(
                        {"bond_reroute", "ring_path_remodel", "segment_replace"}
                    )
                )
                for row in candidates
            ),
            "topology_changing_candidates": sum(
                int(row["delta_cycle_rank"]) != 0 for row in candidates
            ),
            "jump_candidates_for_recursive_refinement": sum(
                row["realized_primitive_band"] in {"medium", "large"}
                and row["module_count"] >= 2
                and row["structural_extent_band"] in {"medium", "large"}
                for row in candidates
            ),
            "small_refinement_candidates": sum(
                row["realized_primitive_band"] == "small" for row in candidates
            ),
        },
        "unique_family_count": len(families),
        "family_counts": dict(sorted(families.items())),
        "unique_composition_count": len(sequences),
        "composition_counts": dict(sorted(sequences.items())),
        "teacher_descriptor_support": {
            "teacher_exact_descriptor_covered": len(exact_teacher.intersection(exact_candidates)),
            "teacher_exact_descriptor_denominator": len(exact_teacher),
            "candidate_exact_descriptor_hits": sum(exact_hits),
            "candidate_exact_descriptor_denominator": len(candidates),
            "candidate_exact_descriptor_precision": (
                sum(exact_hits) / len(candidates) if candidates else None
            ),
            "teacher_coarse_descriptor_covered": len(
                coarse_teacher.intersection(coarse_candidates)
            ),
            "teacher_coarse_descriptor_denominator": len(coarse_teacher),
            "candidate_coarse_descriptor_hits": sum(coarse_hits),
            "candidate_coarse_descriptor_denominator": len(candidates),
            "candidate_coarse_descriptor_precision": (
                sum(coarse_hits) / len(candidates) if candidates else None
            ),
            "teacher_semantic_descriptor_covered": len(
                semantic_teacher.intersection(semantic_candidates)
            ),
            "teacher_semantic_descriptor_denominator": len(semantic_teacher),
            "candidate_semantic_descriptor_hits": sum(semantic_hits),
            "candidate_semantic_descriptor_denominator": len(candidates),
            "candidate_semantic_descriptor_precision": (
                sum(semantic_hits) / len(candidates) if candidates else None
            ),
            "exact_teacher_endpoint_recovery": sum(
                row["endpoint_key_sha256"] in teacher_endpoints for row in candidates
            ),
        },
    }


def evaluate_locks(
    *, repository_root: Path, artifact_root: Path, output_path: Path
) -> dict[str, Any]:
    contract, contract_identity = load_contract(repository_root)
    cells = _source_cells(repository_root, contract)
    locks: dict[str, tuple[dict[str, Any], str, Path]] = {}
    for cell_key in cells:
        path = artifact_root / "locks" / f"{cell_key}.json.gz"
        lock, lock_identity = _read_envelope(path)
        if (
            lock.get("schema_version") != LOCK_SCHEMA
            or lock.get("cell_key") != cell_key
            or lock.get("contract_payload_sha256") != contract_identity
        ):
            raise ValueError(f"candidate lock identity changed: {path}")
        locks[cell_key] = (lock, lock_identity, path)

    # The import and teacher read occur only after the complete six-lock census.
    teachers_by_source = _teacher_descriptors(repository_root)
    result_cells = []
    aggregate_candidates: dict[str, list[dict[str, Any]]] = defaultdict(list)
    aggregate_teachers: list[dict[str, Any]] = []
    for cell_key, (lock, lock_identity, path) in locks.items():
        source = decode_state(lock["source_state"])
        teachers = teachers_by_source.get(canonical_state_key(source), [])
        if not teachers:
            raise ValueError(f"no diagnostic teacher routes match source {cell_key}")
        arm_summaries = {}
        for arm, arm_payload in lock["arms"].items():
            candidates = arm_payload["candidates"]
            arm_summaries[arm] = _arm_summary(candidates, teachers)
            aggregate_candidates[arm].extend(candidates)
        aggregate_teachers.extend(teachers)
        result_cells.append(
            {
                "cell_key": cell_key,
                "delta": lock["delta"],
                "candidate_lock": {
                    "path": str(path.relative_to(repository_root)),
                    "sha256": sha256_file(path),
                    "payload_sha256": lock_identity,
                },
                "teacher_routes": len(teachers),
                "arms": arm_summaries,
            }
        )

    aggregate = {
        arm: _arm_summary(candidates, aggregate_teachers)
        for arm, candidates in sorted(aggregate_candidates.items())
    }
    composer = aggregate["generic_complete_program_composer"]
    baseline = aggregate["generic_retained_core_only"]
    exact_gain = (
        composer["teacher_descriptor_support"]["teacher_exact_descriptor_covered"]
        - baseline["teacher_descriptor_support"]["teacher_exact_descriptor_covered"]
    )
    coarse_gain = (
        composer["teacher_descriptor_support"]["teacher_coarse_descriptor_covered"]
        - baseline["teacher_descriptor_support"]["teacher_coarse_descriptor_covered"]
    )
    semantic_gain = (
        composer["teacher_descriptor_support"]["teacher_semantic_descriptor_covered"]
        - baseline["teacher_descriptor_support"]["teacher_semantic_descriptor_covered"]
    )
    engineering = {
        "complete_six_cell_lock_census": len(result_cells) == 6,
        "exact_execution_precision_one": all(
            row["arms"][arm]["exact_execution"]["precision"] == 1.0
            for row in result_cells
            for arm in row["arms"]
            if row["arms"][arm]["exact_execution"]["denominator"] > 0
        ),
        "joint_multimodule_support_every_cell": all(
            row["arms"]["generic_complete_program_composer"]["multi_module_candidates"] > 0
            for row in result_cells
        ),
        "small_medium_large_support_across_panel": all(
            composer["primitive_band_counts"].get(band, 0) > 0 for band in SCALE_BANDS
        ),
        "nonzero_created_handle_dependency_support_across_panel": (
            composer["dependency_bearing_candidates"] > 0
        ),
        "coherent_semantic_composition_across_panel": (
            composer["semantic_composition"]["coordinated_multi_region_candidates"] > 0
            and (
                composer["semantic_composition"]["release_and_reconstruct_candidates"] > 0
                or composer["semantic_composition"]["ring_plus_nonring_compositions"] > 0
            )
        ),
        "jump_and_refinement_support_across_panel": (
            composer["semantic_composition"]["jump_candidates_for_recursive_refinement"] > 0
            and composer["semantic_composition"]["small_refinement_candidates"] > 0
        ),
        "no_route_or_objective_runtime_inputs": all(
            all(
                not bool(lock[0]["proposal_inputs"][field])
                for field in (
                    "task_cell_or_target",
                    "trajectory_distilled_templates",
                    "route_weights_or_ids",
                    "teacher_or_winner_endpoints",
                    "objective_or_comparator_scores",
                )
            )
            for lock in locks.values()
        ),
    }
    scientific = {
        "exact_teacher_descriptor_coverage_gain_over_retained_core": exact_gain,
        "coarse_teacher_descriptor_coverage_gain_over_retained_core": coarse_gain,
        "semantic_teacher_descriptor_coverage_gain_over_retained_core": semantic_gain,
        "strict_descriptor_coverage_improvement": (
            exact_gain > 0 or coarse_gain > 0 or semantic_gain > 0
        ),
    }
    passed = all(engineering.values()) and scientific["strict_descriptor_coverage_improvement"]
    code_inputs = (
        "src/compose_v4/control/generic_complete_program_composer.py",
        "src/compose_v4/control/progressive_structured_sampler.py",
        "src/compose_v4/control/dynamic_program_synthesis_v1.py",
        "src/compose_v4/control/edit_program_graph.py",
        "src/compose_v4/control/retained_core_pruning.py",
        "src/compose_v4/experiments/t4_nodistill_generic_composer_gate.py",
    )
    payload = {
        "schema_version": RESULT_SCHEMA,
        "evidence": "computed zero-oracle route-free complete-program support gate",
        "decision": "PASS_SUPPORT_GATE" if passed else "NO_PROMOTION",
        "contract": {
            "path": CONTRACT_RELATIVE_PATH,
            "sha256": sha256_file(repository_root / CONTRACT_RELATIVE_PATH),
            "payload_sha256": contract_identity,
        },
        "candidate_lock_ordering": (
            "all six autonomous candidate locks were sealed and validated before "
            "teacher_traces was imported or read"
        ),
        "cells": result_cells,
        "aggregate": aggregate,
        "gates": {"engineering": engineering, "scientific": scientific, "passed": passed},
        "implementation": {
            "code_revision_before_result_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=repository_root, text=True
            ).strip(),
            "code_sha256": {path: sha256_file(repository_root / path) for path in code_inputs},
            "python": platform.python_version(),
            "rdkit": rdBase.rdkitVersion,
        },
        "costs": {"oracle_calls": 0, "docking_calls": 0, "modal_launches": 0},
        "claim_boundary": (
            "Proposal support and structural-descriptor evidence only. Teacher "
            "descriptors are retrospective diagnostics, not generation inputs or "
            "docking-utility evidence."
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
    "LOCK_SCHEMA",
    "RESULT_SCHEMA",
    "evaluate_locks",
    "load_contract",
    "lock_cell",
    "sha256_file",
]
