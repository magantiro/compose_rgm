"""Semantic-coverage census of the effective packed RingCore MMP layer.

Operator counts do not answer whether a corpus teaches local substitutions,
ordinary lead optimization, linker moves, ring changes, or longer scaffold
restructuring.  This audit reconstructs the one-cut MMP semantics from the
exact states and executable actions that training consumed.

The original 363,456-row source pool is not present on the connected artifact
volume.  The packed derivative remains sufficient to reconstruct core size,
variable-fragment size, attachment count, ring participation, cardinality,
path scale, molecular similarity, scaffold relation, charge, and element
coverage.  The inability to compare those reconstructions with the original
compiler metadata is reported separately as a provenance defect.

The audit is read-only with respect to training artifacts.  Its only write is a
new immutable JSON under ``/_frozen_corpus_audits``.

Example:

    MODAL_PROFILE=nitya modal run modal_apps/audit_mmp_semantic_coverage.py \
      --output-name ringcore-v1-mmp-semantic-coverage-2026-07-29.json
"""

from __future__ import annotations

from collections import Counter
import gzip
import hashlib
import json
import math
from pathlib import Path
import platform
import re
from typing import Iterator

import modal


ROOT = Path(__file__).resolve().parents[1]
REMOTE_ROOT = Path("/root/compose_v4")
ARTIFACT_ROOT = Path("/artifacts")
UNIFIED_MANIFEST = ARTIFACT_ROOT / "UNIFIED_PACKED_MANIFEST.json"
MMP_PACK_MANIFEST = ARTIFACT_ROOT / "mmp_packed_v1" / "partition_manifest.json"
MMP_PACK_COMPLETE = ARTIFACT_ROOT / "mmp_packed_v1" / "MMP_PACK_COMPLETE.json"
MISSING_SOURCE_POOL = (
    ARTIFACT_ROOT / "edit_mining_full_broad_40" / "edit_pool_full.jsonl"
)
OUTPUT_DIRECTORY = ARTIFACT_ROOT / "_frozen_corpus_audits"
OUTPUT_NAME_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,159}\.json\Z")

EXPECTED_EFFECTIVE_COUNTS = {
    "train": 330_991,
    "validation": 15_775,
    "test": 14_253,
}
EXPECTED_POOL_RECORDS = 363_456
EXPECTED_POOL_SHA256 = (
    "3b29d7b84de92a7d2fdf931f55c1bf99c5783d6b2115aa5bf2429471007e4a64"
)
PARTITIONS = ("train", "validation", "test")
PATH_LENGTH_BIN_EDGES = (5, 9, 13)
MMP_LAYER_WEIGHT = 0.35
LATE_TIME_FRACTION = 0.5
OPERATIONAL_HORIZON = 16.0

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy==1.26.4", "rdkit==2024.3.5")
    .env(
        {
            "PYTHONPATH": str(REMOTE_ROOT / "src"),
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
        }
    )
    .add_local_dir(
        ROOT / "src",
        str(REMOTE_ROOT / "src"),
        copy=True,
        ignore=("**/__pycache__/**", "**/*.pyc"),
    )
)
app = modal.App("compose-v4-audit-mmp-semantic-coverage")
artifact_volume = modal.Volume.from_name(
    "compose-v4-artifacts",
    create_if_missing=False,
)


def _stable_hash(payload: object) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _counter_dict(counter: Counter) -> dict[str, int]:
    return {
        str(key): int(value)
        for key, value in sorted(counter.items(), key=lambda item: str(item[0]))
    }


def _float_counter_dict(counter: Counter) -> dict[str, float]:
    return {
        str(key): float(value)
        for key, value in sorted(counter.items(), key=lambda item: str(item[0]))
    }


def _path_bin(path_length: int) -> str:
    for index, edge in enumerate(PATH_LENGTH_BIN_EDGES):
        if path_length <= edge:
            return f"bin_{index}:length<={edge}"
    return f"bin_{len(PATH_LENGTH_BIN_EDGES)}:length>{PATH_LENGTH_BIN_EDGES[-1]}"


def _editing_scale(path_length: int) -> str:
    """Legacy capability descriptions, distinct from empirical sampler bins."""

    if path_length <= 2:
        return "precise_local_1_2"
    if path_length <= 6:
        return "ordinary_lead_optimization_3_6"
    return "longer_compositional_gt_6"


def _similarity_bin(similarity: float) -> str:
    if similarity < 0.2:
        return "[0.0,0.2)"
    if similarity < 0.4:
        return "[0.2,0.4)"
    if similarity < 0.6:
        return "[0.4,0.6)"
    if similarity < 0.8:
        return "[0.6,0.8)"
    if similarity < 1.0:
        return "[0.8,1.0)"
    return "1.0"


def _fraction_bin(value: float) -> str:
    if value < 0.5:
        return "[0.0,0.5)"
    if value < 0.7:
        return "[0.5,0.7)"
    if value < 0.85:
        return "[0.7,0.85)"
    if value < 0.95:
        return "[0.85,0.95)"
    return "[0.95,1.0]"


def _delta_class(delta: int) -> str:
    if delta < 0:
        return "shrink"
    if delta > 0:
        return "grow"
    return "same_cardinality"


def _pair_type(source_variable: int, target_variable: int) -> str:
    if source_variable == target_variable == 1:
        return "atom_swap"
    if source_variable == target_variable:
        return "isosteric"
    return "grow" if target_variable > source_variable else "shrink"


def _mean_production_time() -> float:
    normalized_mean = 0.5
    operational_mean = 1.0 - (
        1.0 - math.exp(-OPERATIONAL_HORIZON)
    ) / OPERATIONAL_HORIZON
    return (
        (1.0 - LATE_TIME_FRACTION) * normalized_mean
        + LATE_TIME_FRACTION * operational_mean
    )


def selected_mark_coefficient(path_length: int) -> float:
    """Expected sum of teacher rates over the sampled progress state."""

    if path_length <= 0:
        raise ValueError("path_length must be positive")
    # N_t | t ~ Binomial(K,t), so E[K-N_t] = K * (1-E[t]).
    return float(path_length) * (1.0 - _mean_production_time())


def _active_slots(state: dict) -> set[int]:
    import numpy as np

    from compose_v4.chem.molecular_graph import is_element

    atom_types = np.asarray(state["atom_types"], dtype=np.int32)
    return {int(index) for index in np.flatnonzero(is_element(atom_types))}


def _state_edges(state: dict, active: set[int]) -> tuple[tuple[int, int], ...]:
    return tuple(
        (int(left), int(right))
        for left, right, order in state["bonds"]
        if int(order) > 0 and int(left) in active and int(right) in active
    )


def _graph_cycle_rank(state: dict) -> int:
    """Return beta_1 = |E| - |V| + c on the exact active graph."""

    active = _active_slots(state)
    if not active:
        return 0
    edges = _state_edges(state, active)
    adjacency: dict[int, list[int]] = {vertex: [] for vertex in active}
    for left, right in edges:
        adjacency[left].append(right)
        adjacency[right].append(left)
    components = 0
    visited: set[int] = set()
    for root in sorted(active):
        if root in visited:
            continue
        components += 1
        stack = [root]
        visited.add(root)
        while stack:
            vertex = stack.pop()
            for neighbor in adjacency[vertex]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    stack.append(neighbor)
    return len(edges) - len(active) + components


def _ring_slots(state: dict) -> set[int]:
    """Vertices incident to a non-bridge edge in the exact active graph."""

    active = _active_slots(state)
    edges = _state_edges(state, active)
    adjacency: dict[int, list[int]] = {vertex: [] for vertex in active}
    for left, right in edges:
        adjacency[left].append(right)
        adjacency[right].append(left)

    discovery: dict[int, int] = {}
    low: dict[int, int] = {}
    bridges: set[tuple[int, int]] = set()
    clock = 0

    def visit(vertex: int, parent: int | None) -> None:
        nonlocal clock
        discovery[vertex] = clock
        low[vertex] = clock
        clock += 1
        for neighbor in adjacency[vertex]:
            if neighbor == parent:
                continue
            if neighbor not in discovery:
                visit(neighbor, vertex)
                low[vertex] = min(low[vertex], low[neighbor])
                if low[neighbor] > discovery[vertex]:
                    bridges.add(tuple(sorted((vertex, neighbor))))
            else:
                low[vertex] = min(low[vertex], discovery[neighbor])

    for vertex in sorted(active):
        if vertex not in discovery:
            visit(vertex, None)

    ring_vertices: set[int] = set()
    for left, right in edges:
        if tuple(sorted((left, right))) not in bridges:
            ring_vertices.update((left, right))
    return ring_vertices


def _boundary_edge_count(state: dict, variable: set[int]) -> int:
    active = _active_slots(state)
    return sum(
        int((left in variable) != (right in variable))
        for left, right in _state_edges(state, active)
    )


def reconstruct_one_cut_semantics(entry: dict) -> dict:
    """Reconstruct compiler descriptors from packed states/actions."""

    trace = entry["trace"]
    steps = trace["steps"]
    executor_rules = tuple(
        str(step["action"]["executor_rule"]) for step in steps
    )
    unexpected = sorted(
        set(executor_rules) - {"atom_delete", "atom_insert"}
    )
    if unexpected:
        raise ValueError(f"MMP trace contains non-one-cut operators: {unexpected}")

    first_insert = next(
        (index for index, rule in enumerate(executor_rules) if rule == "atom_insert"),
        len(executor_rules),
    )
    if any(rule != "atom_delete" for rule in executor_rules[:first_insert]):
        raise ValueError("MMP prefix before insertion is not all deletion")
    if any(rule != "atom_insert" for rule in executor_rules[first_insert:]):
        raise ValueError("MMP suffix after first insertion is not all insertion")

    deleted = {
        int(step["action"]["payload"]["v"])
        for step in steps
        if step["action"]["payload_type"] == "AtomDelete"
    }
    inserted = {
        int(step["action"]["payload"]["slot"])
        for step in steps
        if step["action"]["payload_type"] == "AtomInsert"
    }
    delete_count = sum(rule == "atom_delete" for rule in executor_rules)
    insert_count = sum(rule == "atom_insert" for rule in executor_rules)
    if len(deleted) != delete_count:
        raise ValueError("MMP path deletes a persistent slot more than once")
    if len(inserted) != insert_count:
        raise ValueError("MMP path inserts into a persistent slot more than once")

    source_state = entry["states"][0]
    target_state = entry["states"][-1]
    source_active = _active_slots(source_state)
    target_active = _active_slots(target_state)
    if not deleted <= source_active:
        raise ValueError("MMP deleted slot is not active in the exact source")
    if not inserted <= target_active:
        raise ValueError("MMP inserted slot is not active in the exact target")

    source_core = source_active - deleted
    target_core = target_active - inserted
    if len(source_core) != len(target_core):
        raise ValueError("source and target reconstructed core sizes disagree")
    if len(target_active) - len(source_active) != insert_count - delete_count:
        raise ValueError("MMP action counts do not reconstruct atom-count delta")

    source_boundary = _boundary_edge_count(source_state, deleted)
    target_boundary = _boundary_edge_count(target_state, inserted)
    source_ring = _ring_slots(source_state)
    target_ring = _ring_slots(target_state)
    source_cycle_rank = _graph_cycle_rank(source_state)
    target_cycle_rank = _graph_cycle_rank(target_state)
    variable_ring_atoms = len(deleted & source_ring) + len(inserted & target_ring)
    return {
        "path_length": len(steps),
        "delete_count": delete_count,
        "insert_count": insert_count,
        "constant_core_heavy": len(source_core),
        "source_variable_heavy": delete_count,
        "target_variable_heavy": insert_count,
        "heavy_delta": insert_count - delete_count,
        "pair_type": _pair_type(delete_count, insert_count),
        "source_attachment_count": source_boundary,
        "target_attachment_count": target_boundary,
        "variable_ring_atoms": variable_ring_atoms,
        "source_graph_cycle_rank": source_cycle_rank,
        "target_graph_cycle_rank": target_cycle_rank,
        "graph_cycle_rank_delta": target_cycle_rank - source_cycle_rank,
        "operator_signature": " -> ".join(executor_rules),
    }


def _packed_paths(
    manifest: dict,
    partition: str,
    *,
    root_override: Path | None = None,
) -> tuple[Path, ...]:
    root = (
        Path(manifest["roots"]["mmp_layer"])
        if root_override is None
        else Path(root_override)
    )
    paths = tuple(sorted((root / partition).glob("*.jsonl.gz")))
    declared = tuple(
        Path(relative).name
        for relative in manifest["layers"]["mmp_analogue"][partition]
    )
    if tuple(path.name for path in paths) != declared:
        raise ValueError(
            f"packed MMP shard mismatch for {partition}: "
            f"found {[path.name for path in paths]}, declared {list(declared)}"
        )
    return paths


def _iter_packed_entries(
    manifest: dict,
    partition: str,
    *,
    root_override: Path | None = None,
) -> Iterator[dict]:
    for path in _packed_paths(
        manifest,
        partition,
        root_override=root_override,
    ):
        with gzip.open(path, "rt") as handle:
            for line in handle:
                if line.strip():
                    yield json.loads(line)


def _molecule_features(smiles: str, generator):
    from rdkit import Chem
    from rdkit.Chem import rdMolDescriptors
    from rdkit.Chem.Scaffolds import MurckoScaffold

    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError(f"RDKit cannot parse effective MMP molecule {smiles!r}")
    elements = tuple(sorted({atom.GetSymbol() for atom in molecule.GetAtoms()}))
    charges = [int(atom.GetFormalCharge()) for atom in molecule.GetAtoms()]
    net_charge = sum(charges)
    if any(charges):
        charge_class = (
            "zwitterion_net_zero" if net_charge == 0 else "nonzero_net_charge"
        )
    else:
        charge_class = "neutral"
    scaffold = Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(molecule))
    return {
        "fingerprint": generator.GetFingerprint(molecule),
        "heavy_atoms": int(molecule.GetNumHeavyAtoms()),
        "elements": elements,
        "charge_class": charge_class,
        "scaffold": scaffold,
        # This is an RDKit ring-basis descriptor, not graph cycle rank.  beta_1
        # is computed independently from the exact packed state.
        "rdkit_ring_count": int(rdMolDescriptors.CalcNumRings(molecule)),
        "aromatic_ring_count": int(rdMolDescriptors.CalcNumAromaticRings(molecule)),
    }


def _empty_partition_stats() -> dict:
    return {
        "records": 0,
        "unique_sources": set(),
        "unique_targets": set(),
        "unique_scaffolds": set(),
        "pair_type": Counter(),
        "direction": Counter(),
        "path_length": Counter(),
        "production_path_bin": Counter(),
        "editing_scale": Counter(),
        "source_attachment_count": Counter(),
        "target_attachment_count": Counter(),
        "constant_core_heavy": Counter(),
        "source_variable_heavy": Counter(),
        "target_variable_heavy": Counter(),
        "variable_ring_atoms": Counter(),
        "source_graph_cycle_rank": Counter(),
        "target_graph_cycle_rank": Counter(),
        "graph_cycle_rank_delta": Counter(),
        "heavy_delta": Counter(),
        "delta_class": Counter(),
        "source_core_fraction_bin": Counter(),
        "target_core_fraction_bin": Counter(),
        "similarity_bin": Counter(),
        "scaffold_relation": Counter(),
        "source_rdkit_ring_count": Counter(),
        "target_rdkit_ring_count": Counter(),
        "rdkit_ring_count_delta": Counter(),
        "source_aromatic_ring_count": Counter(),
        "target_aromatic_ring_count": Counter(),
        "source_charge_class": Counter(),
        "target_charge_class": Counter(),
        "source_element_set": Counter(),
        "target_element_set": Counter(),
        "elements_added": Counter(),
        "elements_removed": Counter(),
        "operator_signature": Counter(),
        "joint_capability_cell": Counter(),
        "teacher_coefficient_by_pair_type": Counter(),
        "teacher_coefficient_by_editing_scale": Counter(),
        "teacher_coefficient_total": 0.0,
        "similarity_sum": 0.0,
        "similarity_min": 1.0,
        "similarity_max": 0.0,
    }


def _finalize_partition(stats: dict) -> dict:
    records = int(stats["records"])
    coefficient_total = float(stats["teacher_coefficient_total"])
    return {
        "records": records,
        "unique_sources": len(stats["unique_sources"]),
        "unique_targets": len(stats["unique_targets"]),
        "unique_scaffolds": len(stats["unique_scaffolds"]),
        "pair_type": _counter_dict(stats["pair_type"]),
        "direction": _counter_dict(stats["direction"]),
        "path_length": _counter_dict(stats["path_length"]),
        "production_path_bin": _counter_dict(stats["production_path_bin"]),
        "editing_scale": _counter_dict(stats["editing_scale"]),
        "source_attachment_count": _counter_dict(
            stats["source_attachment_count"]
        ),
        "target_attachment_count": _counter_dict(
            stats["target_attachment_count"]
        ),
        "constant_core_heavy": _counter_dict(stats["constant_core_heavy"]),
        "source_variable_heavy": _counter_dict(stats["source_variable_heavy"]),
        "target_variable_heavy": _counter_dict(stats["target_variable_heavy"]),
        "variable_ring_atoms": _counter_dict(stats["variable_ring_atoms"]),
        "source_graph_cycle_rank": _counter_dict(
            stats["source_graph_cycle_rank"]
        ),
        "target_graph_cycle_rank": _counter_dict(
            stats["target_graph_cycle_rank"]
        ),
        "graph_cycle_rank_delta": _counter_dict(
            stats["graph_cycle_rank_delta"]
        ),
        "heavy_delta": _counter_dict(stats["heavy_delta"]),
        "delta_class": _counter_dict(stats["delta_class"]),
        "source_core_fraction_bin": _counter_dict(
            stats["source_core_fraction_bin"]
        ),
        "target_core_fraction_bin": _counter_dict(
            stats["target_core_fraction_bin"]
        ),
        "similarity": {
            "mean": stats["similarity_sum"] / records if records else None,
            "min": stats["similarity_min"] if records else None,
            "max": stats["similarity_max"] if records else None,
            "histogram": _counter_dict(stats["similarity_bin"]),
        },
        "scaffold_relation": _counter_dict(stats["scaffold_relation"]),
        "source_rdkit_ring_count": _counter_dict(
            stats["source_rdkit_ring_count"]
        ),
        "target_rdkit_ring_count": _counter_dict(
            stats["target_rdkit_ring_count"]
        ),
        "rdkit_ring_count_delta": _counter_dict(
            stats["rdkit_ring_count_delta"]
        ),
        "source_aromatic_ring_count": _counter_dict(
            stats["source_aromatic_ring_count"]
        ),
        "target_aromatic_ring_count": _counter_dict(
            stats["target_aromatic_ring_count"]
        ),
        "source_charge_class": _counter_dict(stats["source_charge_class"]),
        "target_charge_class": _counter_dict(stats["target_charge_class"]),
        "source_element_set": _counter_dict(stats["source_element_set"]),
        "target_element_set": _counter_dict(stats["target_element_set"]),
        "elements_added": _counter_dict(stats["elements_added"]),
        "elements_removed": _counter_dict(stats["elements_removed"]),
        "operator_signature": _counter_dict(stats["operator_signature"]),
        "joint_capability_cell": _counter_dict(stats["joint_capability_cell"]),
        "expected_selected_mark_coefficient": {
            "total_conditional_within_mmp_layer": coefficient_total,
            "by_pair_type": _float_counter_dict(
                stats["teacher_coefficient_by_pair_type"]
            ),
            "by_editing_scale": _float_counter_dict(
                stats["teacher_coefficient_by_editing_scale"]
            ),
            "pair_type_share": {
                key: float(value) / coefficient_total
                for key, value in sorted(
                    stats["teacher_coefficient_by_pair_type"].items()
                )
            }
            if coefficient_total
            else {},
            "editing_scale_share": {
                key: float(value) / coefficient_total
                for key, value in sorted(
                    stats["teacher_coefficient_by_editing_scale"].items()
                )
            }
            if coefficient_total
            else {},
        },
    }


def build_audit(
    *,
    unified_manifest_path: Path = UNIFIED_MANIFEST,
    mmp_pack_manifest_path: Path = MMP_PACK_MANIFEST,
    mmp_pack_complete_path: Path = MMP_PACK_COMPLETE,
    mmp_layer_root: Path | None = None,
    source_pool_path: Path = MISSING_SOURCE_POOL,
) -> dict:
    """Build the read-only census from remote-mounted or local copied artifacts."""

    from rdkit import DataStructs
    from rdkit.Chem import rdFingerprintGenerator

    unified = json.loads(Path(unified_manifest_path).read_text())
    pack_manifest = json.loads(Path(mmp_pack_manifest_path).read_text())
    pack_complete = json.loads(Path(mmp_pack_complete_path).read_text())
    if unified.get("manifest_checksum") != "5c5c254e1054c081":
        raise ValueError("unexpected unified packed manifest")
    if pack_manifest.get("pool_sha256") != EXPECTED_POOL_SHA256:
        raise ValueError("MMP partition manifest has an unexpected source-pool hash")
    if not pack_complete.get("MMP_PACK_COMPLETE"):
        raise ValueError("MMP packed derivative is not complete")
    if int(pack_complete.get("pool_records", -1)) != EXPECTED_POOL_RECORDS:
        raise ValueError("MMP packed derivative has an unexpected source-pool size")

    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=2,
        fpSize=2048,
    )
    feature_cache: dict[str, dict] = {}
    stats_by_partition = {
        partition: _empty_partition_stats() for partition in PARTITIONS
    }
    path_bin_counts = {partition: Counter() for partition in PARTITIONS}
    coefficient_rows: list[tuple[str, str, str, int]] = []
    row_ids: set[int] = set()

    for partition in PARTITIONS:
        stats = stats_by_partition[partition]
        for entry in _iter_packed_entries(
            unified,
            partition,
            root_override=mmp_layer_root,
        ):
            trace = entry["trace"]
            metadata = trace.get("metadata") or {}
            if "row_id" not in metadata:
                raise ValueError("packed MMP row lacks source row_id")
            row_id = int(metadata["row_id"])
            if row_id in row_ids:
                raise ValueError(f"packed source row_id {row_id} is duplicated")
            row_ids.add(row_id)

            semantic = reconstruct_one_cut_semantics(entry)
            source = str(trace["source_smiles"])
            target = str(trace["target_smiles"])
            source_features = feature_cache.get(source)
            if source_features is None:
                source_features = _molecule_features(source, generator)
                feature_cache[source] = source_features
            target_features = feature_cache.get(target)
            if target_features is None:
                target_features = _molecule_features(target, generator)
                feature_cache[target] = target_features

            if source_features["heavy_atoms"] != (
                semantic["constant_core_heavy"]
                + semantic["source_variable_heavy"]
            ):
                raise ValueError("reconstructed MMP source atom accounting failed")
            if target_features["heavy_atoms"] != (
                semantic["constant_core_heavy"]
                + semantic["target_variable_heavy"]
            ):
                raise ValueError("reconstructed MMP target atom accounting failed")

            similarity = float(
                DataStructs.TanimotoSimilarity(
                    source_features["fingerprint"],
                    target_features["fingerprint"],
                )
            )
            pair_type = semantic["pair_type"]
            path_length = semantic["path_length"]
            editing_scale = _editing_scale(path_length)
            path_bin = _path_bin(path_length)
            core = semantic["constant_core_heavy"]
            source_variable = semantic["source_variable_heavy"]
            target_variable = semantic["target_variable_heavy"]
            heavy_delta = semantic["heavy_delta"]
            source_elements = set(source_features["elements"])
            target_elements = set(target_features["elements"])
            added = "+".join(sorted(target_elements - source_elements)) or "<none>"
            removed = "+".join(sorted(source_elements - target_elements)) or "<none>"
            if not source_features["scaffold"] and not target_features["scaffold"]:
                scaffold_relation = "both_acyclic"
            elif source_features["scaffold"] == target_features["scaffold"]:
                scaffold_relation = "same_murcko"
            else:
                scaffold_relation = "changed_murcko"
            rdkit_ring_delta = (
                target_features["rdkit_ring_count"]
                - source_features["rdkit_ring_count"]
            )
            source_fraction = core / float(core + source_variable)
            target_fraction = core / float(core + target_variable)
            joint_cell = "|".join(
                (
                    f"pair:{pair_type}",
                    f"scale:{editing_scale}",
                    f"delta:{_delta_class(heavy_delta)}",
                    f"cycle_delta:{semantic['graph_cycle_rank_delta']}",
                    f"variable_ring:{int(semantic['variable_ring_atoms'] > 0)}",
                    f"scaffold:{scaffold_relation}",
                    f"similarity:{_similarity_bin(similarity)}",
                )
            )

            stats["records"] += 1
            stats["unique_sources"].add(source)
            stats["unique_targets"].add(target)
            stats["unique_scaffolds"].update(
                (source_features["scaffold"], target_features["scaffold"])
            )
            stats["pair_type"][pair_type] += 1
            stats["direction"][str(trace.get("direction") or "<missing>")] += 1
            stats["path_length"][path_length] += 1
            stats["production_path_bin"][path_bin] += 1
            stats["editing_scale"][editing_scale] += 1
            stats["source_attachment_count"][
                semantic["source_attachment_count"]
            ] += 1
            stats["target_attachment_count"][
                semantic["target_attachment_count"]
            ] += 1
            stats["constant_core_heavy"][core] += 1
            stats["source_variable_heavy"][source_variable] += 1
            stats["target_variable_heavy"][target_variable] += 1
            stats["variable_ring_atoms"][semantic["variable_ring_atoms"]] += 1
            stats["source_graph_cycle_rank"][
                semantic["source_graph_cycle_rank"]
            ] += 1
            stats["target_graph_cycle_rank"][
                semantic["target_graph_cycle_rank"]
            ] += 1
            stats["graph_cycle_rank_delta"][
                semantic["graph_cycle_rank_delta"]
            ] += 1
            stats["heavy_delta"][heavy_delta] += 1
            stats["delta_class"][_delta_class(heavy_delta)] += 1
            stats["source_core_fraction_bin"][_fraction_bin(source_fraction)] += 1
            stats["target_core_fraction_bin"][_fraction_bin(target_fraction)] += 1
            stats["similarity_bin"][_similarity_bin(similarity)] += 1
            stats["similarity_sum"] += similarity
            stats["similarity_min"] = min(stats["similarity_min"], similarity)
            stats["similarity_max"] = max(stats["similarity_max"], similarity)
            stats["scaffold_relation"][scaffold_relation] += 1
            stats["source_rdkit_ring_count"][
                source_features["rdkit_ring_count"]
            ] += 1
            stats["target_rdkit_ring_count"][
                target_features["rdkit_ring_count"]
            ] += 1
            stats["rdkit_ring_count_delta"][rdkit_ring_delta] += 1
            stats["source_aromatic_ring_count"][
                source_features["aromatic_ring_count"]
            ] += 1
            stats["target_aromatic_ring_count"][
                target_features["aromatic_ring_count"]
            ] += 1
            stats["source_charge_class"][source_features["charge_class"]] += 1
            stats["target_charge_class"][target_features["charge_class"]] += 1
            stats["source_element_set"][
                "+".join(source_features["elements"])
            ] += 1
            stats["target_element_set"][
                "+".join(target_features["elements"])
            ] += 1
            stats["elements_added"][added] += 1
            stats["elements_removed"][removed] += 1
            stats["operator_signature"][semantic["operator_signature"]] += 1
            stats["joint_capability_cell"][joint_cell] += 1
            path_bin_counts[partition][path_bin] += 1
            coefficient_rows.append(
                (partition, pair_type, editing_scale, path_length)
            )

        expected = EXPECTED_EFFECTIVE_COUNTS[partition]
        if stats["records"] != expected:
            raise ValueError(
                f"effective {partition} MMP rows {stats['records']} != {expected}"
            )

    for partition, pair_type, editing_scale, path_length in coefficient_rows:
        path_bin = _path_bin(path_length)
        nonempty_bins = len(path_bin_counts[partition])
        record_probability = (
            1.0
            / float(nonempty_bins)
            / float(path_bin_counts[partition][path_bin])
        )
        coefficient = record_probability * selected_mark_coefficient(path_length)
        stats = stats_by_partition[partition]
        stats["teacher_coefficient_total"] += coefficient
        stats["teacher_coefficient_by_pair_type"][pair_type] += coefficient
        stats["teacher_coefficient_by_editing_scale"][editing_scale] += coefficient

    finalized = {
        partition: _finalize_partition(stats_by_partition[partition])
        for partition in PARTITIONS
    }
    result = {
        "schema": "compose.diagnostics.mmp_semantic_coverage",
        "schema_version": 1,
        "runtime": {
            "python": platform.python_version(),
            "numpy": __import__("numpy").__version__,
            "rdkit": __import__("rdkit").__version__,
        },
        "source": {
            "packed_manifest_checksum": unified["manifest_checksum"],
            "packed_effective_records": len(row_ids),
            "source_pool_sha256_from_partition_manifest": EXPECTED_POOL_SHA256,
            "partition_rule_version": pack_manifest[
                "mmp_partition_rule_version"
            ],
            "partition_rule_hash": pack_manifest[
                "pair_rule_implementation_hash"
            ],
            "semantic_source": (
                "exact packed source/target states plus executable actions"
            ),
        },
        "taxonomy": {
            "legacy_pair_types": ["atom_swap", "grow", "isosteric", "shrink"],
            "editing_scale": {
                "precise_local_1_2": "one or two compiled rewrites",
                "ordinary_lead_optimization_3_6": (
                    "three through six compiled rewrites"
                ),
                "longer_compositional_gt_6": (
                    "more than six compiled rewrites"
                ),
            },
            "warning": (
                "MMP pair_type is a directed variable-fragment-size label, "
                "not a complete editing-capability taxonomy."
            ),
        },
        "partitions": finalized,
        "training_law": {
            "mmp_layer_weight_in_full_training": MMP_LAYER_WEIGHT,
            "conditional_record_sampler": (
                "uniform nonempty production path-length bin, then uniform "
                "record within bin"
            ),
            "selected_mark_coefficient": (
                "record probability times E[K-N_t]; reported shares are "
                "conditional within MMP and the full law multiplies by 0.35"
            ),
        },
        "auditability_finding": {
            "packed_state_correctness_affected": False,
            "semantic_reconstruction_possible": True,
            "finding": (
                "The packed derivative retained row_id and exact states but "
                "did not copy compiler semantic descriptors. The original "
                "363,456-row source pool is absent from the connected artifact "
                "volume, so reconstructed descriptors cannot be compared back "
                "to the original metadata."
            ),
            "required_repair": (
                "Preserve the authoritative source pool and carry semantic "
                "descriptors plus their schema/hash into the next packed "
                "corpus contract."
            ),
            "missing_expected_path": str(source_pool_path),
            "missing_expected_path_observed": not Path(source_pool_path).exists(),
        },
    }
    result["content_sha256"] = _stable_hash(result)
    return result


@app.function(
    image=image,
    volumes={"/artifacts": artifact_volume},
    timeout=60 * 30,
    cpu=4.0,
    memory=8192,
)
def audit_remote(output_name: str) -> dict:
    if OUTPUT_NAME_PATTERN.fullmatch(output_name) is None:
        raise ValueError("output_name must be a safe JSON basename")
    result = build_audit()
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIRECTORY / output_name
    if output_path.exists():
        existing = json.loads(output_path.read_text())
        if existing != result:
            raise FileExistsError(
                f"{output_path} already exists with different content"
            )
    else:
        output_path.write_text(
            json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
        )
        artifact_volume.commit()
    train = result["partitions"]["train"]
    return {
        "output_path": str(output_path),
        "content_sha256": result["content_sha256"],
        "train_records": train["records"],
        "train_pair_types": train["pair_type"],
        "train_editing_scale": train["editing_scale"],
        "train_variable_ring_atoms": train["variable_ring_atoms"],
        "train_scaffold_relation": train["scaffold_relation"],
        "train_teacher_coefficient_pair_type_share": train[
            "expected_selected_mark_coefficient"
        ]["pair_type_share"],
    }


@app.local_entrypoint()
def main(
    output_name: str = "ringcore-v1-mmp-semantic-coverage-2026-07-29.json",
) -> None:
    print(json.dumps(audit_remote.remote(output_name), indent=2, sort_keys=True))
