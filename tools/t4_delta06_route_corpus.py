#!/usr/bin/env python3
"""Export the frozen zero-oracle T4 delta-0.6 route-reference corpus."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import platform
import subprocess
from collections import Counter
from pathlib import Path
from statistics import median

import networkx as nx
import numpy as np
from rdkit import rdBase

from compose_v4.chem.molecular_graph import is_element
from compose_v4.control.dependency_region_program import trace_structure
from compose_v4.control.docking_value import identity
from compose_v4.control.structural_subgoal import (
    extract_structural_goal,
    instantiate_goal,
)
from compose_v4.control.structural_subgoal_components import (
    ExactComponentVocabulary,
    component_families,
    dependency_motif,
)
from compose_v4.control.structural_subgoal_policy import minimize_subgoal
from compose_v4.control.structural_subgoal_realizer import realize_structural_goal
from compose_v4.experiments.whole_ring_plan import execute_program
from compose_v4.rewrite.kernel import canonical_state_key
from compose_v4.rewrite.trace_shard import decode_state

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/t4_delta06_route_corpus_v1.json"
OUTPUT_SCHEMA = "t4_delta06_route_corpus_result_v1"
CORPUS_SCHEMA = "t4_delta06_route_training_corpus_v1"
COMPONENT_FAMILIES = ("target_topology", "attachment_pattern", "dependency_motif")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_contract() -> dict:
    envelope = json.loads(CONTRACT.read_text())
    payload = envelope.get("payload")
    if not isinstance(payload, dict) or envelope.get("contract_sha256") != identity(payload):
        raise ValueError(f"contract is not self-hashed: {CONTRACT}")
    oracle = payload["oracle"]
    if (
        any(
            oracle[field] != 0
            for field in (
                "calls_authorized",
                "docking_calls_authorized",
                "modal_launches_authorized",
            )
        )
        or oracle["model_training_authorized"]
    ):
        raise ValueError("delta-0.6 corpus contract authorizes external evaluation or training")
    return payload


def _atomic_publish(path: Path, payload: dict, *, compressed: bool = False) -> None:
    if path.exists():
        raise ValueError(f"refusing to overwrite route-corpus artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    envelope = {"payload": payload, "payload_sha256": identity(payload)}
    encoded = (json.dumps(envelope, sort_keys=True, separators=(",", ":")) + "\n").encode()
    if compressed:
        encoded = gzip.compress(encoded, mtime=0)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(encoded)
    temporary.replace(path)


def _selected_rows(audit: dict) -> list[dict]:
    selected = []
    for row in audit["pairs"]:
        references = [
            reference for reference in row["references"] if float(reference["delta"]) == 0.6
        ]
        if not references:
            continue
        selected.append({**row, "delta06_references": references})
    return sorted(selected, key=lambda row: row["pair_id"])


def _receipt_manifest(rows: list[dict]) -> list[dict]:
    return [
        {
            "pair_id": row["pair_id"],
            "path": f"diagnostics/ivg_winner_paths/{row['receipt']}",
            "sha256": row["receipt_sha256"],
            "status": row["status"],
        }
        for row in rows
    ]


def _load_inputs(contract: dict) -> tuple[dict, list[dict], list[dict]]:
    verified = []
    for key in (
        "ivg_t4_census",
        "seed_registry",
        "winner_witness_audit",
        "winner_witness_review",
        "winner_witness_preparation",
        "sealed_structural_realizer_summary",
    ):
        spec = contract["inputs"][key]
        path = ROOT / spec["path"]
        actual = sha256(path)
        if actual != spec["sha256"]:
            raise ValueError(f"input hash changed for {key}: {path}: {actual}")
        verified.append({"name": key, **spec})
    for key, spec in contract["inputs"]["sealed_implementation"].items():
        path = ROOT / spec["path"]
        actual = sha256(path)
        if actual != spec["sha256"]:
            raise ValueError(f"sealed implementation changed for {key}: {path}: {actual}")
        verified.append({"name": key, **spec})

    audit = json.loads((ROOT / contract["inputs"]["winner_witness_audit"]["path"]).read_text())
    rows = _selected_rows(audit)
    manifest = _receipt_manifest(rows)
    expected = contract["inputs"]["delta06_receipt_manifest"]
    if len(manifest) != expected["entries"] or identity(manifest) != expected["manifest_identity"]:
        raise ValueError("delta-0.6 receipt manifest identity or census changed")
    for spec in manifest:
        path = ROOT / spec["path"]
        actual = sha256(path)
        if actual != spec["sha256"]:
            raise ValueError(f"pair receipt hash changed: {path}: {actual}")
        verified.append({"name": f"pair_receipt:{spec['pair_id']}", **spec})
    return audit, rows, verified


def _graph_summary(graph) -> dict:
    active = [int(slot) for slot in np.flatnonzero(is_element(graph.atom_types))]
    edges = sum(
        bool(graph.bonds[left, right]) for i, left in enumerate(active) for right in active[i + 1 :]
    )
    nx_graph = nx.Graph()
    nx_graph.add_nodes_from(active)
    nx_graph.add_edges_from(
        (left, right)
        for i, left in enumerate(active)
        for right in active[i + 1 :]
        if graph.bonds[left, right]
    )
    components = nx.number_connected_components(nx_graph) if active else 0
    return {
        "active_atoms": len(active),
        "bonds": int(edges),
        "connected_components": int(components),
        "cycle_rank": int(edges - len(active) + components),
    }


def _route_geometry(states: tuple[dict, ...], actions: tuple[dict, ...]) -> dict:
    structure = trace_structure(states, actions)
    source = decode_state(states[0])
    final_source_tokens = {
        token for token in structure["final_active_slots"].values() if token[0] == "source"
    }
    source_count = len(structure["initial_token_slots"])
    touched = sorted(
        {
            token[1]
            for footprint in structure["footprints"]
            for token in footprint
            if token[0] == "source"
        }
    )
    if not touched:
        radius, radius_reason = None, "no_source_token_touched"
    else:
        graph = nx.Graph()
        active = [int(slot) for slot in np.flatnonzero(is_element(source.atom_types))]
        graph.add_nodes_from(active)
        graph.add_edges_from(
            (left, right)
            for i, left in enumerate(active)
            for right in active[i + 1 :]
            if source.bonds[left, right]
        )
        candidates = []
        for center in active:
            lengths = nx.single_source_shortest_path_length(graph, center)
            if all(slot in lengths for slot in touched):
                candidates.append(max(lengths[slot] for slot in touched))
        radius = min(candidates) if candidates else None
        radius_reason = None if candidates else "touched_source_tokens_disconnected"
    return {
        "touched_source_atoms": len(touched),
        "source_edit_radius": radius,
        "source_edit_radius_abstention": radius_reason,
        "retained_source_atoms": len(final_source_tokens),
        "source_atoms": source_count,
        "retained_source_atom_fraction": len(final_source_tokens) / source_count,
    }


def _component_payload(component) -> dict:
    return {
        "family": component.family,
        "node_labels": list(component.node_labels),
        "edges": [list(row) for row in component.edges],
        "directed": component.directed,
        "bucket_key": list(component.bucket_key),
    }


def _extract_components(goal, actions: tuple[dict, ...], regions: dict) -> tuple[list[dict], dict]:
    rows = []
    by_family = {family: [] for family in COMPONENT_FAMILIES}
    for index, (subgoal, region) in enumerate(
        zip(goal.subgoals, regions["components"], strict=True)
    ):
        template, retained = minimize_subgoal(subgoal)
        dependency = dependency_motif(
            actions,
            region,
            created_dependency_edges=tuple(regions["created_dependency_edges"]),
            cycle_dependency_edges=tuple(regions["cycle_dependency_edges"]),
        )
        families = component_families(template, dependency)
        selected = {
            family: [_component_payload(component) for component in families[family]]
            for family in COMPONENT_FAMILIES
        }
        for family in COMPONENT_FAMILIES:
            by_family[family].extend(families[family])
        rows.append(
            {
                "subgoal_index": index,
                "subgoal_id": subgoal.subgoal_id,
                "minimal_template_id": template.template_id,
                "retained_input_role_indices": list(retained),
                "components": selected,
            }
        )
    return rows, by_family


def _distribution(values: list[int | float]) -> dict:
    if not values:
        return {"count": 0, "minimum": None, "median": None, "maximum": None}
    return {
        "count": len(values),
        "minimum": min(values),
        "median": median(values),
        "maximum": max(values),
    }


def _gate(covered: int, denominator: int, *, exact: int | None = None) -> dict:
    exact = covered if exact is None else exact
    return {
        "covered": covered,
        "exact": exact,
        "denominator": denominator,
        "coverage": covered / denominator if denominator else None,
        "precision": exact / covered if covered else None,
    }


def _receipt_payload(row: dict, audit: dict) -> tuple[dict, dict]:
    path = ROOT / "diagnostics/ivg_winner_paths" / row["receipt"]
    envelope = json.loads(gzip.decompress(path.read_bytes()))
    if identity(envelope["payload"]) != envelope["payload_sha256"]:
        raise ValueError(f"corrupt pair payload: {path}")
    if envelope["scientific_identity"] != audit["scientific_identity"]:
        raise ValueError(f"pair receipt scientific identity mismatch: {path}")
    expected_pair = {key: row[key] for key in ("pair_id", "source", "target", "references")}
    if envelope["pair"] != expected_pair:
        raise ValueError(f"pair receipt identity mismatch: {path}")
    if envelope["payload"]["path"]["status"] != row["status"]:
        raise ValueError(f"pair receipt status mismatch: {path}")
    return envelope, envelope["payload"]["path"]


def build_corpus(
    contract: dict, audit: dict, rows: list[dict], *, code_revision: str
) -> tuple[dict, dict]:
    endpoints, routes, abstentions = [], [], []
    primitive_replays = structural_extractions = target_reconstructions = realizations = 0
    realizer_exact = 0
    route_lengths: list[int] = []
    region_counts: list[int] = []
    edit_radii: list[int] = []
    retained_fractions: list[float] = []
    cycle_deltas: list[int] = []
    all_components = {family: [] for family in COMPONENT_FAMILIES}

    for row in rows:
        receipt, path = _receipt_payload(row, audit)
        references = sorted(
            row["delta06_references"],
            key=lambda reference: (
                reference["cell"],
                reference["run_seed"],
                reference["reported_docking_score"],
            ),
        )
        reference_cells = sorted({reference["cell"] for reference in references})
        reference_proteins = sorted({reference["target"] for reference in references})
        if len(reference_cells) != 1 or len(reference_proteins) != 1:
            raise ValueError(f"delta-0.6 pair crosses cells or proteins: {row['pair_id']}")
        reference = {
            "pair_id": row["pair_id"],
            "cell": reference_cells[0],
            "protein": reference_proteins[0],
            "source_smiles": row["source"],
            "endpoint_smiles": row["target"],
            "evidence": contract["evidence_labels"]["endpoint"],
            "status": row["status"],
            "delta06_run_references": references,
            "delta06_run_reference_count": len(references),
            "training_weight": 1.0 if row["status"] == "witness_found" else 0.0,
            "receipt": {
                "path": f"diagnostics/ivg_winner_paths/{row['receipt']}",
                "sha256": row["receipt_sha256"],
                "payload_sha256": receipt["payload_sha256"],
            },
        }
        endpoints.append(reference)
        if row["status"] != "witness_found":
            abstentions.append(
                {
                    "pair_id": row["pair_id"],
                    "reason_code": row["status"],
                    "endpoint_reference_retained": True,
                    "training_weight": 0.0,
                }
            )
            continue

        states, actions = tuple(path["states"]), tuple(path["actions"])
        source, target = decode_state(states[0]), decode_state(states[-1])
        replayed, replay_receipt = execute_program(source, list(actions))
        replay_states_exact = replay_receipt["states"] == list(states)
        replay_endpoint_exact = canonical_state_key(replayed) == canonical_state_key(target)
        if not (replay_states_exact and replay_endpoint_exact):
            raise ValueError(f"exact primitive replay failed for {row['pair_id']}")
        primitive_replays += 1

        goal, bindings, regions = extract_structural_goal(states, actions)
        structural_extractions += 1
        reconstructed, bound_receipt = instantiate_goal(source, goal, bindings)
        target_exact = canonical_state_key(reconstructed) == canonical_state_key(target)
        if not target_exact:
            raise ValueError(f"address-free target reconstruction failed for {row['pair_id']}")
        target_reconstructions += 1

        realized = realize_structural_goal(source, goal, bindings)
        realized_ok = realized["status"] == "realized"
        realized_exact = bool(realized_ok and realized["endpoint_matches_bound_target"])
        realizations += int(realized_ok)
        realizer_exact += int(realized_exact)

        geometry = _route_geometry(states, actions)
        source_stats, target_stats = _graph_summary(source), _graph_summary(target)
        component_rows, component_objects = _extract_components(goal, actions, regions)
        for family in COMPONENT_FAMILIES:
            all_components[family].extend(component_objects[family])

        route_lengths.append(len(actions))
        region_counts.append(len(goal.subgoals))
        if geometry["source_edit_radius"] is not None:
            edit_radii.append(geometry["source_edit_radius"])
        retained_fractions.append(geometry["retained_source_atom_fraction"])
        cycle_deltas.append(target_stats["cycle_rank"] - source_stats["cycle_rank"])
        routes.append(
            {
                "route_id": identity(
                    {
                        "schema_version": "t4_delta06_route_record_v1",
                        "pair_id": row["pair_id"],
                        "goal_id": goal.goal_id,
                    }
                ),
                "pair_id": row["pair_id"],
                "cell": reference["cell"],
                "protein": reference["protein"],
                "evidence": contract["evidence_labels"]["route"],
                "observed_ivg_trajectory": False,
                "training_weight": 1.0,
                "source_state": states[0],
                "endpoint_state": states[-1],
                "primitive_actions": list(actions),
                "primitive_states": list(states),
                "primitive_count": len(actions),
                "primitive_replay": {
                    "states_exact": replay_states_exact,
                    "endpoint_exact": replay_endpoint_exact,
                },
                "dependency_regions": regions,
                "region_count": len(goal.subgoals),
                "structural_goal": goal.payload(),
                "address_free_bindings": [list(binding) for binding in bindings],
                "bound_target_receipt": bound_receipt,
                "exact_target_reconstruction": target_exact,
                "sealed_realizer": {
                    "status": realized["status"],
                    "actions": realized["actions"],
                    "states": realized["states"],
                    "expanded": realized["expanded"],
                    "attempted": realized["attempted"],
                    "best_mismatch": realized["best_mismatch"],
                    "compiler_strategy": realized.get("compiler_strategy"),
                    "endpoint_matches_bound_target": realized["endpoint_matches_bound_target"],
                    "subgoal_targets_match_within_complete_goal": realized[
                        "subgoal_targets_match_within_complete_goal"
                    ],
                    "primitive_teacher_actions_used": realized["primitive_teacher_actions_used"],
                },
                "geometry": {
                    **geometry,
                    "source": source_stats,
                    "target": target_stats,
                    "cycle_rank_delta": target_stats["cycle_rank"] - source_stats["cycle_rank"],
                },
                "component_records": component_rows,
            }
        )

    status_counts = Counter(row["status"] for row in rows)
    expected = contract["expected"]
    if (
        len(endpoints) != expected["endpoint_references"]
        or len(routes) != expected["exact_witnesses"]
    ):
        raise ValueError("endpoint or witness census changed")
    if (
        dict(sorted(Counter(row["reason_code"] for row in abstentions).items()))
        != expected["abstentions"]
    ):
        raise ValueError("abstention census changed")
    cells = sorted(
        {
            reference["cell"]
            for endpoint in endpoints
            for reference in endpoint["delta06_run_references"]
        }
    )
    proteins = sorted(
        {
            reference["target"]
            for endpoint in endpoints
            for reference in endpoint["delta06_run_references"]
        }
    )
    if len(cells) != expected["source_cells"] or len(proteins) != expected["proteins"]:
        raise ValueError("cell or protein census changed")

    diversity = {}
    for family in COMPONENT_FAMILIES:
        vocabulary = ExactComponentVocabulary(all_components[family])
        instances = len(all_components[family])
        diversity[family] = {
            "instances": instances,
            "exact_unique_classes": vocabulary.size,
            "unique_per_instance": vocabulary.size / instances if instances else None,
        }

    per_cell = {}
    for cell in cells:
        cell_endpoints = [row for row in endpoints if row["cell"] == cell]
        cell_routes = [row for row in routes if row["cell"] == cell]
        cell_abstentions = [
            row
            for row in abstentions
            if row["pair_id"] in {item["pair_id"] for item in cell_endpoints}
        ]
        per_cell[cell] = {
            "protein": cell_endpoints[0]["protein"],
            "endpoint_references": len(cell_endpoints),
            "delta06_run_references": sum(
                row["delta06_run_reference_count"] for row in cell_endpoints
            ),
            "exact_witnesses": len(cell_routes),
            "abstentions": dict(
                sorted(Counter(row["reason_code"] for row in cell_abstentions).items())
            ),
            "primitive_count": _distribution([row["primitive_count"] for row in cell_routes]),
            "dependency_region_count": _distribution([row["region_count"] for row in cell_routes]),
        }

    corpus = {
        "schema_version": CORPUS_SCHEMA,
        "scientific_identity": identity(
            {
                "contract": identity(contract),
                "receipt_manifest": contract["inputs"]["delta06_receipt_manifest"][
                    "manifest_identity"
                ],
                "code_revision": code_revision,
            }
        ),
        "evidence_boundary": contract["evidence_labels"],
        "support": contract["support"],
        "deduplication": contract["deduplication"],
        "endpoint_references": endpoints,
        "routes": routes,
        "abstentions": abstentions,
    }
    result = {
        "schema_version": OUTPUT_SCHEMA,
        "scientific_identity": corpus["scientific_identity"],
        "evidence": "computed zero-oracle training-data audit",
        "implementation": {
            "code_revision": code_revision,
            "python": platform.python_version(),
            "numpy": np.__version__,
            "rdkit": rdBase.rdkitVersion,
            "hardware_affects_scientific_result": False,
        },
        "census": {
            "endpoint_references": len(endpoints),
            "delta06_run_references": sum(row["delta06_run_reference_count"] for row in endpoints),
            "exact_witnesses": len(routes),
            "abstentions": len(abstentions),
            "status_counts": dict(sorted(status_counts.items())),
            "cells": len(cells),
            "proteins": len(proteins),
        },
        "gates": {
            "exact_primitive_replay": _gate(primitive_replays, len(routes)),
            "structural_goal_extraction": _gate(structural_extractions, len(routes)),
            "address_free_target_reconstruction": _gate(target_reconstructions, len(routes)),
            "sealed_structural_realization": _gate(realizations, len(routes), exact=realizer_exact),
        },
        "route_metrics": {
            "primitive_count": _distribution(route_lengths),
            "dependency_region_count": _distribution(region_counts),
            "source_edit_radius": _distribution(edit_radii),
            "source_edit_radius_abstentions": len(routes) - len(edit_radii),
            "retained_source_atom_fraction": _distribution(retained_fractions),
            "cycle_rank_delta_counts": dict(sorted(Counter(map(str, cycle_deltas)).items())),
            "component_diversity": diversity,
            "per_cell": per_cell,
        },
        "abstentions": dict(sorted(Counter(row["reason_code"] for row in abstentions).items())),
        "costs": {"oracle_calls": 0, "docking_calls": 0, "modal_launches": 0, "gpu_seconds": 0},
        "interpretation_limit": contract["interpretation_limit"],
        "limitations": [
            "Reported endpoint references are not observed IVG trajectories.",
            "Compiled witnesses are retrospective answer-known training evidence.",
            "Exact realization does not establish autonomous subgoal proposal quality or docking utility.",
        ],
    }
    return corpus, result


def run(output_dir: Path, *, code_revision: str) -> dict:
    if output_dir.exists():
        raise ValueError(f"refusing to overwrite route-corpus output directory: {output_dir}")
    actual_revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    if code_revision != actual_revision:
        raise ValueError(
            f"code revision mismatch: requested {code_revision}, current {actual_revision}"
        )
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip():
        raise ValueError("authoritative route-corpus export requires a clean committed worktree")
    contract = _load_contract()
    audit, rows, verified_inputs = _load_inputs(contract)
    corpus, result = build_corpus(contract, audit, rows, code_revision=code_revision)
    corpus_path = output_dir / "training_corpus.json.gz"
    result_path = output_dir / "result.json"
    _atomic_publish(corpus_path, corpus, compressed=True)
    result = {
        **result,
        "inputs": verified_inputs,
        "contract": {"path": str(CONTRACT.relative_to(ROOT)), "sha256": sha256(CONTRACT)},
        "corpus": {
            "path": str(corpus_path.relative_to(ROOT)),
            "sha256": sha256(corpus_path),
            "payload_sha256": identity(corpus),
        },
    }
    _atomic_publish(result_path, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--code-revision", required=True)
    args = parser.parse_args()
    result = run(args.output_dir, code_revision=args.code_revision)
    print(
        json.dumps(
            {
                "census": result["census"],
                "gates": result["gates"],
                "route_metrics": result["route_metrics"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
