#!/usr/bin/env python3
"""Audit AGILE/Ugi dominance and freeze a layer-aware lipid sampler.

The script compares the observed/measured R0 corpus with AGILE virtual-only
structures using exact canonical identity, ECFP4 nearest neighbors, scaffold
and lipid-core groupings, molecule/topology profiles, and R0-anchored structural
cells. It then selects the AGILE layer probability from an explicit
realism-versus-expected-coverage tradeoff rather than a hand-set percentage.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from rdkit import Chem, DataStructs, rdBase
from rdkit.Chem import rdFingerprintGenerator
from rdkit.SimDivFilters import rdSimDivPickers

from compose_v4.lipids.corpus_bias import (
    choose_equal_priority_minimax,
    expected_weighted_unique_coverage,
    hill_effective_numbers,
    jensen_shannon,
    lipid_topology_features,
    probability_vector,
    weighted_probability_vector,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = REPO_ROOT / "artifacts/datasets/compose_lipid_pretraining_v1"
FINGERPRINT_BITS = 2048
FINGERPRINT_RADIUS = 2
CELL_COUNTS = (64, 128, 256)
SEED = 20260720


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_group(prefix: str, value: str) -> str:
    digest = hashlib.sha256(f"{prefix}\0{value}".encode()).hexdigest()
    return f"{prefix}-{digest}"


def flatten_group_tokens(prefix: str, value: object) -> list[str]:
    if isinstance(value, dict):
        tokens: list[str] = []
        for key, nested in sorted(value.items()):
            tokens.extend(flatten_group_tokens(f"{prefix}:{key}", nested))
        return tokens
    if isinstance(value, list):
        tokens = []
        for nested in value:
            tokens.extend(flatten_group_tokens(prefix, nested))
        return tokens
    return [f"{prefix}={value}"]


def percentile(values: Sequence[float], probability: float) -> float:
    return float(np.quantile(np.asarray(values, dtype=float), probability))


def numerical_summary(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    return {
        "min": float(array.min()),
        "median": float(np.median(array)),
        "mean": float(array.mean()),
        "p05": float(np.quantile(array, 0.05)),
        "p95": float(np.quantile(array, 0.95)),
        "max": float(array.max()),
    }


def categorical_summary(values: Iterable[str]) -> dict[str, Any]:
    values = list(values)
    counts = Counter(values)
    return {
        "count": len(values),
        "unique": len(counts),
        "top": [
            {"value": value, "count": count, "fraction": count / len(values)}
            for value, count in counts.most_common(20)
        ],
        "effective_numbers": hill_effective_numbers(values),
    }


def load_corpora(
    input_dir: Path,
) -> tuple[list[dict[str, str]], list[dict[str, str]], dict[str, dict[str, str]]]:
    r0_path = input_dir / "r0_observed_real_structures.csv"
    union_path = input_dir / "canonical_structure_union.csv"
    with r0_path.open(newline="", encoding="utf-8") as handle:
        r0 = list(csv.DictReader(handle))
    r0_by_smiles = {row["canonical_isomeric_smiles"]: row for row in r0}
    aux: list[dict[str, str]] = []
    union_by_smiles: dict[str, dict[str, str]] = {}
    with union_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            smiles = row["canonical_isomeric_smiles"]
            union_by_smiles[smiles] = row
            sources = set(row["source_ids"].split("|"))
            if "agile_virtual12k" in sources and smiles not in r0_by_smiles:
                aux.append(row)
    if len(r0) != 15433 or len(aux) != 11076:
        raise RuntimeError(
            f"frozen corpus identity changed: expected R0=15433/AUX=11076, "
            f"found {len(r0)}/{len(aux)}"
        )
    return r0, aux, union_by_smiles


def parse_and_profile(
    rows: Sequence[Mapping[str, str]],
) -> tuple[list[Chem.Mol], list[dict[str, object]]]:
    molecules: list[Chem.Mol] = []
    profiles: list[dict[str, object]] = []
    for row in rows:
        smiles = row["canonical_isomeric_smiles"]
        molecule = Chem.MolFromSmiles(smiles)
        if molecule is None:
            raise ValueError(f"frozen canonical SMILES failed to parse: {smiles}")
        molecules.append(molecule)
        profiles.append(lipid_topology_features(molecule))
    return molecules, profiles


def fingerprint_molecules(molecules: Sequence[Chem.Mol]) -> list[DataStructs.ExplicitBitVect]:
    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=FINGERPRINT_RADIUS,
        fpSize=FINGERPRINT_BITS,
        includeChirality=True,
    )
    return [generator.GetFingerprint(molecule) for molecule in molecules]


def nearest_neighbors(
    query_fingerprints: Sequence[DataStructs.ExplicitBitVect],
    reference_fingerprints: Sequence[DataStructs.ExplicitBitVect],
    *,
    aligned_self: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    if aligned_self and len(query_fingerprints) != len(reference_fingerprints):
        raise ValueError("aligned self-neighbor inputs differ in length")
    similarities = np.empty(len(query_fingerprints), dtype=np.float32)
    indices = np.empty(len(query_fingerprints), dtype=np.int32)
    for query_index, fingerprint in enumerate(query_fingerprints):
        values = DataStructs.BulkTanimotoSimilarity(fingerprint, reference_fingerprints)
        if aligned_self:
            values[query_index] = -1.0
        neighbor_index = int(np.argmax(values))
        indices[query_index] = neighbor_index
        similarities[query_index] = values[neighbor_index]
    return similarities, indices


def nearest_neighbor_summary(values: np.ndarray) -> dict[str, Any]:
    return {
        **numerical_summary(values.tolist()),
        "threshold_counts": {
            f"ge_{threshold:.2f}": int(np.sum(values >= threshold))
            for threshold in (0.70, 0.80, 0.90, 0.95, 0.99, 1.00)
        },
        "threshold_fractions": {
            f"ge_{threshold:.2f}": float(np.mean(values >= threshold))
            for threshold in (0.70, 0.80, 0.90, 0.95, 0.99, 1.00)
        },
    }


def choose_anchors(
    r0_fingerprints: Sequence[DataStructs.ExplicitBitVect],
) -> list[int]:
    picker = rdSimDivPickers.MaxMinPicker()
    return list(
        picker.LazyBitVectorPick(
            r0_fingerprints,
            len(r0_fingerprints),
            max(CELL_COUNTS),
            seed=SEED,
        )
    )


def assign_cells(
    fingerprints: Sequence[DataStructs.ExplicitBitVect],
    anchor_fingerprints: Sequence[DataStructs.ExplicitBitVect],
) -> tuple[np.ndarray, np.ndarray]:
    assignments = np.empty(len(fingerprints), dtype=np.int32)
    similarities = np.empty(len(fingerprints), dtype=np.float32)
    for index, fingerprint in enumerate(fingerprints):
        values = DataStructs.BulkTanimotoSimilarity(fingerprint, anchor_fingerprints)
        assignment = int(np.argmax(values))
        assignments[index] = assignment
        similarities[index] = values[assignment]
    return assignments, similarities


def cell_calibrated_auxiliary_probabilities(
    r0_assignments: np.ndarray,
    auxiliary_assignments: np.ndarray,
    cell_count: int,
) -> np.ndarray:
    r0_counts = np.bincount(r0_assignments, minlength=cell_count).astype(float)
    auxiliary_counts = np.bincount(auxiliary_assignments, minlength=cell_count).astype(float)
    probabilities = np.zeros(len(auxiliary_assignments), dtype=float)
    occupied = auxiliary_counts > 0
    retained_r0_mass = float(r0_counts[occupied].sum())
    for cell in np.flatnonzero(occupied):
        cell_mass = r0_counts[cell] / retained_r0_mass
        probabilities[auxiliary_assignments == cell] = cell_mass / auxiliary_counts[cell]
    probabilities /= probabilities.sum()
    return probabilities


def cell_audit(
    r0_assignments: np.ndarray,
    auxiliary_assignments: np.ndarray,
    auxiliary_weights: np.ndarray,
    cell_count: int,
) -> dict[str, Any]:
    categories = [str(index) for index in range(cell_count)]
    r0_values = [str(value) for value in r0_assignments]
    auxiliary_values = [str(value) for value in auxiliary_assignments]
    r0_distribution = probability_vector(r0_values, categories)
    auxiliary_raw = probability_vector(auxiliary_values, categories)
    auxiliary_calibrated = weighted_probability_vector(
        auxiliary_values, categories, auxiliary_weights
    )
    return {
        "cell_count": cell_count,
        "r0_occupied_cells": int(np.count_nonzero(r0_distribution)),
        "auxiliary_occupied_cells": int(np.count_nonzero(auxiliary_raw)),
        "r0_cells_reached_by_auxiliary_fraction": float(
            np.count_nonzero((r0_distribution > 0) & (auxiliary_raw > 0))
            / np.count_nonzero(r0_distribution)
        ),
        "raw_jensen_shannon_bits": jensen_shannon(r0_distribution, auxiliary_raw),
        "cell_calibrated_jensen_shannon_bits": jensen_shannon(
            r0_distribution, auxiliary_calibrated
        ),
        "r0_effective_cells": {
            "shannon_q1": math.exp(
                -float(np.sum(r0_distribution[r0_distribution > 0] * np.log(
                    r0_distribution[r0_distribution > 0]
                )))
            ),
            "simpson_q2": 1.0 / float(np.sum(r0_distribution**2)),
        },
        "auxiliary_raw_effective_cells": {
            "shannon_q1": math.exp(
                -float(np.sum(auxiliary_raw[auxiliary_raw > 0] * np.log(
                    auxiliary_raw[auxiliary_raw > 0]
                )))
            ),
            "simpson_q2": 1.0 / float(np.sum(auxiliary_raw**2)),
        },
    }


def feature_axes(
    r0_profiles: Sequence[Mapping[str, object]],
    auxiliary_profiles: Sequence[Mapping[str, object]],
) -> dict[str, tuple[list[str], list[str]]]:
    direct_axes = (
        "size_bin",
        "charge_bin",
        "stereo_bin",
        "ring_bin",
        "aromatic_bin",
        "branching_bin",
        "unsaturation_bin",
        "long_tail_bin",
        "tail_architecture_proxy",
        "cleavable_motif_proxy",
        "murcko_scaffold",
        "heteroatom_connector_core",
    )
    axes: dict[str, tuple[list[str], list[str]]] = {
        axis: (
            [str(profile[axis]) for profile in r0_profiles],
            [str(profile[axis]) for profile in auxiliary_profiles],
        )
        for axis in direct_axes
    }
    all_elements = sorted(
        set().union(
            *(set(str(profile["elements"]).split("|")) for profile in r0_profiles),
            *(set(str(profile["elements"]).split("|")) for profile in auxiliary_profiles),
        )
    )
    for element in all_elements:
        axes[f"element_{element}"] = (
            [
                "present" if element in str(profile["elements"]).split("|") else "absent"
                for profile in r0_profiles
            ],
            [
                "present" if element in str(profile["elements"]).split("|") else "absent"
                for profile in auxiliary_profiles
            ],
        )
    return axes


def realism_penalty_curve(
    alphas: np.ndarray,
    axes: Mapping[str, tuple[list[str], list[str]]],
    auxiliary_weights: np.ndarray,
    r0_cells: np.ndarray,
    auxiliary_cells: np.ndarray,
    cell_count: int,
) -> tuple[np.ndarray, dict[str, float]]:
    aligned: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name, (r0_values, auxiliary_values) in axes.items():
        categories = sorted(set(r0_values) | set(auxiliary_values))
        aligned[name] = (
            probability_vector(r0_values, categories),
            weighted_probability_vector(auxiliary_values, categories, auxiliary_weights),
        )
    cell_categories = [str(index) for index in range(cell_count)]
    aligned[f"ecfp4_r0_cell_k{cell_count}"] = (
        probability_vector([str(value) for value in r0_cells], cell_categories),
        weighted_probability_vector(
            [str(value) for value in auxiliary_cells], cell_categories, auxiliary_weights
        ),
    )
    penalties = np.zeros(len(alphas), dtype=float)
    full_axis_divergence: dict[str, float] = {}
    for name, (r0_distribution, auxiliary_distribution) in aligned.items():
        per_alpha = []
        for alpha in alphas:
            mixture = (1.0 - alpha) * r0_distribution + alpha * auxiliary_distribution
            per_alpha.append(jensen_shannon(r0_distribution, mixture))
        penalties += np.asarray(per_alpha)
        full_axis_divergence[name] = float(per_alpha[-1])
    penalties /= len(aligned)
    if penalties[-1] > 0:
        penalties /= penalties[-1]
    return penalties, full_axis_divergence


def tradeoff_scenario(
    *,
    alphas: np.ndarray,
    axes: Mapping[str, tuple[list[str], list[str]]],
    auxiliary_weights: np.ndarray,
    novelty_weights: np.ndarray,
    r0_cells: np.ndarray,
    auxiliary_cells: np.ndarray,
    cell_count: int,
    draws: int,
    objective: str,
) -> dict[str, Any]:
    realism, full_axis_divergence = realism_penalty_curve(
        alphas,
        axes,
        auxiliary_weights,
        r0_cells,
        auxiliary_cells,
        cell_count,
    )
    coverage = np.asarray(
        [
            expected_weighted_unique_coverage(
                auxiliary_weights, novelty_weights, float(alpha), draws
            )
            for alpha in alphas
        ],
        dtype=float,
    )
    if coverage[-1] > 0:
        coverage /= coverage[-1]
    if objective == "equal_priority_minimax":
        selected = choose_equal_priority_minimax(alphas, realism, coverage)
    elif objective == "equal_priority_l2":
        regrets = np.sqrt(realism**2 + (1.0 - coverage) ** 2)
        index = int(np.argmin(regrets))
        selected = {
            "layer_probability": float(alphas[index]),
            "normalized_realism_penalty": float(realism[index]),
            "normalized_coverage_utility": float(coverage[index]),
            "l2_regret": float(regrets[index]),
        }
    else:
        raise ValueError(objective)
    return {
        "cell_count": cell_count,
        "draws": draws,
        "objective": objective,
        "selected": selected,
        "full_cap_axis_jensen_shannon_bits": full_axis_divergence,
        "curve": [
            {
                "layer_probability": float(alpha),
                "normalized_realism_penalty": float(realism_value),
                "normalized_coverage_utility": float(coverage_value),
            }
            for alpha, realism_value, coverage_value in zip(
                alphas, realism, coverage, strict=True
            )
        ],
    }


def corpus_profile(profiles: Sequence[Mapping[str, object]]) -> dict[str, Any]:
    numeric_fields = (
        "heavy_atoms",
        "formal_charge",
        "stereogenic_atom_count",
        "ring_count",
        "aromatic_ring_count",
        "carbon_branch_point_count",
        "cc_unsaturation_count",
        "long_aliphatic_terminus_count",
        "max_aliphatic_tail_depth",
        "motif_ester_count",
        "motif_amide_count",
        "motif_carbonate_count",
        "motif_disulfide_count",
        "motif_acetal_count",
    )
    categorical_fields = (
        "size_bin",
        "charge_bin",
        "elements",
        "stereo_bin",
        "ring_bin",
        "aromatic_bin",
        "branching_bin",
        "unsaturation_bin",
        "long_tail_bin",
        "tail_architecture_proxy",
        "cleavable_motif_proxy",
        "murcko_scaffold",
        "heteroatom_connector_core",
    )
    return {
        "count": len(profiles),
        "numeric": {
            field: numerical_summary([float(profile[field]) for profile in profiles])
            for field in numeric_fields
        },
        "categorical": {
            field: categorical_summary(str(profile[field]) for profile in profiles)
            for field in categorical_fields
        },
    }


def source_family_audit(r0: Sequence[Mapping[str, str]]) -> dict[str, Any]:
    source_memberships: Counter[str] = Counter()
    exclusive_memberships: Counter[str] = Counter()
    reaction_families: Counter[str] = Counter()
    for row in r0:
        sources = row["observed_source_ids"].split("|")
        source_memberships.update(sources)
        exclusive_memberships["|".join(sorted(sources))] += 1
        reaction_families.update(row["reaction_family_holdout_groups"].split("|"))
    return {
        "source_membership_counts_nonexclusive": dict(sorted(source_memberships.items())),
        "source_membership_combinations": dict(exclusive_memberships.most_common()),
        "reaction_family_membership_counts_nonexclusive": dict(
            sorted(reaction_families.items())
        ),
        "auxiliary_source": "agile_virtual12k",
        "auxiliary_reaction_family": "ugi_3cr_fixed_core",
    }


def write_nn_rows(
    path: Path,
    auxiliary: Sequence[Mapping[str, str]],
    r0: Sequence[Mapping[str, str]],
    similarities: np.ndarray,
    indices: np.ndarray,
    profiles: Sequence[Mapping[str, object]],
) -> None:
    fields = [
        "structure_sha256",
        "canonical_isomeric_smiles",
        "nearest_r0_structure_sha256",
        "nearest_r0_smiles",
        "ecfp4_tanimoto_to_nearest_r0",
        "novelty_one_minus_tanimoto",
        "murcko_scaffold",
        "heteroatom_connector_core",
        "tail_architecture_proxy",
        "cleavable_motif_proxy",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row, similarity, neighbor, profile in zip(
            auxiliary, similarities, indices, profiles, strict=True
        ):
            nearest = r0[int(neighbor)]
            writer.writerow(
                {
                    "structure_sha256": row["structure_sha256"],
                    "canonical_isomeric_smiles": row["canonical_isomeric_smiles"],
                    "nearest_r0_structure_sha256": nearest["structure_sha256"],
                    "nearest_r0_smiles": nearest["canonical_isomeric_smiles"],
                    "ecfp4_tanimoto_to_nearest_r0": f"{float(similarity):.8f}",
                    "novelty_one_minus_tanimoto": f"{1.0 - float(similarity):.8f}",
                    "murcko_scaffold": profile["murcko_scaffold"],
                    "heteroatom_connector_core": profile["heteroatom_connector_core"],
                    "tail_architecture_proxy": profile["tail_architecture_proxy"],
                    "cleavable_motif_proxy": profile["cleavable_motif_proxy"],
                }
            )


def write_sampler_rows(
    path: Path,
    r0: Sequence[Mapping[str, str]],
    auxiliary: Sequence[Mapping[str, str]],
    r0_profiles: Sequence[Mapping[str, object]],
    auxiliary_profiles: Sequence[Mapping[str, object]],
    r0_cells: np.ndarray,
    auxiliary_cells: np.ndarray,
    auxiliary_weights: np.ndarray,
    auxiliary_layer_probability: float,
    auxiliary_nn: np.ndarray,
) -> str:
    fields = [
        "layer_id",
        "structure_sha256",
        "canonical_isomeric_smiles",
        "source_ids",
        "reaction_family_holdout_groups",
        "component_holdout_groups_json",
        "study_split_groups_json",
        "linker_or_core_holdout_group",
        "split_group_tokens_json",
        "structure_cell_k128",
        "leakage_group_id",
        "prospective_lock_status",
        "split_eligibility",
        "frozen_group_assignment_sha256",
        "within_layer_probability",
        "layer_probability",
        "total_sampling_probability",
        "ecfp4_tanimoto_to_nearest_r0",
    ]
    r0_within = 1.0 / len(r0)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row, profile, cell in zip(r0, r0_profiles, r0_cells, strict=True):
            linker_group = stable_group(
                "connector", str(profile["heteroatom_connector_core"])
            )
            component_groups = row["component_holdout_groups_json"]
            frozen_groups = {
                "component": json.loads(component_groups),
                "linker_or_core": linker_group,
                "reaction_family": row["reaction_family_holdout_groups"],
                "study": json.loads(row["study_split_groups_json"]),
            }
            split_tokens = sorted(
                flatten_group_tokens("component", frozen_groups["component"])
                + flatten_group_tokens("linker_or_core", linker_group)
                + flatten_group_tokens(
                    "reaction_family",
                    row["reaction_family_holdout_groups"].split("|"),
                )
                + flatten_group_tokens("study", frozen_groups["study"])
            )
            writer.writerow(
                {
                    "layer_id": "r0_observed_real",
                    "structure_sha256": row["structure_sha256"],
                    "canonical_isomeric_smiles": row["canonical_isomeric_smiles"],
                    "source_ids": row["observed_source_ids"],
                    "reaction_family_holdout_groups": row[
                        "reaction_family_holdout_groups"
                    ],
                    "component_holdout_groups_json": component_groups,
                    "study_split_groups_json": row["study_split_groups_json"],
                    "linker_or_core_holdout_group": linker_group,
                    "split_group_tokens_json": json.dumps(split_tokens),
                    "structure_cell_k128": int(cell),
                    "leakage_group_id": row["leakage_group_id"],
                    "prospective_lock_status": row["prospective_lock_status"],
                    "split_eligibility": "r0_frozen_groups",
                    "frozen_group_assignment_sha256": hashlib.sha256(
                        json.dumps(frozen_groups, sort_keys=True).encode()
                    ).hexdigest(),
                    "within_layer_probability": f"{r0_within:.16g}",
                    "layer_probability": f"{1.0 - auxiliary_layer_probability:.16g}",
                    "total_sampling_probability": (
                        f"{(1.0 - auxiliary_layer_probability) * r0_within:.16g}"
                    ),
                    "ecfp4_tanimoto_to_nearest_r0": "1",
                }
            )
        for row, profile, cell, within, nn_similarity in zip(
            auxiliary,
            auxiliary_profiles,
            auxiliary_cells,
            auxiliary_weights,
            auxiliary_nn,
            strict=True,
        ):
            connector = str(profile["heteroatom_connector_core"])
            tail_proxy = str(profile["tail_architecture_proxy"])
            component_proxies = {
                "proxy_heteroatom_connector_core": stable_group("connector", connector),
                "proxy_tail_architecture": stable_group("tailproxy", tail_proxy),
            }
            linker_group = stable_group("connector", connector)
            frozen_groups = {
                "component_proxy": component_proxies,
                "linker_or_core": linker_group,
                "reaction_family": "ugi_3cr_fixed_core",
                "study": {"agile_virtual12k": ["AGILE_virtual12k"]},
            }
            split_tokens = sorted(
                flatten_group_tokens("component_proxy", component_proxies)
                + flatten_group_tokens("linker_or_core", linker_group)
                + flatten_group_tokens("reaction_family", "ugi_3cr_fixed_core")
                + flatten_group_tokens("study", frozen_groups["study"])
            )
            writer.writerow(
                {
                    "layer_id": "r1_auxiliary_agile_ugi",
                    "structure_sha256": row["structure_sha256"],
                    "canonical_isomeric_smiles": row["canonical_isomeric_smiles"],
                    "source_ids": "agile_virtual12k",
                    "reaction_family_holdout_groups": "ugi_3cr_fixed_core",
                    "component_holdout_groups_json": json.dumps(
                        component_proxies, sort_keys=True
                    ),
                    "study_split_groups_json": json.dumps(
                        frozen_groups["study"], sort_keys=True
                    ),
                    "linker_or_core_holdout_group": linker_group,
                    "split_group_tokens_json": json.dumps(split_tokens),
                    "structure_cell_k128": int(cell),
                    "leakage_group_id": f"mol-{row['structure_sha256']}",
                    "prospective_lock_status": "retrospective_virtual_auxiliary",
                    "split_eligibility": (
                        "structural_pretraining_only_exact_components_unresolved"
                    ),
                    "frozen_group_assignment_sha256": hashlib.sha256(
                        json.dumps(frozen_groups, sort_keys=True).encode()
                    ).hexdigest(),
                    "within_layer_probability": f"{float(within):.16g}",
                    "layer_probability": f"{auxiliary_layer_probability:.16g}",
                    "total_sampling_probability": (
                        f"{auxiliary_layer_probability * float(within):.16g}"
                    ),
                    "ecfp4_tanimoto_to_nearest_r0": f"{float(nn_similarity):.8f}",
                }
            )
    return sha256_file(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_INPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    r0, auxiliary, _ = load_corpora(input_dir)
    r0_molecules, r0_profiles = parse_and_profile(r0)
    auxiliary_molecules, auxiliary_profiles = parse_and_profile(auxiliary)
    r0_fingerprints = fingerprint_molecules(r0_molecules)
    auxiliary_fingerprints = fingerprint_molecules(auxiliary_molecules)

    auxiliary_to_r0, auxiliary_to_r0_indices = nearest_neighbors(
        auxiliary_fingerprints, r0_fingerprints
    )
    r0_self, _ = nearest_neighbors(r0_fingerprints, r0_fingerprints, aligned_self=True)
    auxiliary_self, _ = nearest_neighbors(
        auxiliary_fingerprints, auxiliary_fingerprints, aligned_self=True
    )
    anchor_indices = choose_anchors(r0_fingerprints)
    cell_results: dict[int, dict[str, Any]] = {}
    assignments: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for cell_count in CELL_COUNTS:
        anchor_fingerprints = [r0_fingerprints[index] for index in anchor_indices[:cell_count]]
        r0_cells, r0_anchor_similarity = assign_cells(r0_fingerprints, anchor_fingerprints)
        auxiliary_cells, auxiliary_anchor_similarity = assign_cells(
            auxiliary_fingerprints, anchor_fingerprints
        )
        auxiliary_weights = cell_calibrated_auxiliary_probabilities(
            r0_cells, auxiliary_cells, cell_count
        )
        assignments[cell_count] = (r0_cells, auxiliary_cells, auxiliary_weights)
        cell_results[cell_count] = {
            **cell_audit(r0_cells, auxiliary_cells, auxiliary_weights, cell_count),
            "r0_anchor_similarity": numerical_summary(r0_anchor_similarity.tolist()),
            "auxiliary_anchor_similarity": numerical_summary(
                auxiliary_anchor_similarity.tolist()
            ),
        }

    raw_auxiliary_fraction = len(auxiliary) / (len(r0) + len(auxiliary))
    alphas = np.linspace(0.0, raw_auxiliary_fraction, 101)
    axes = feature_axes(r0_profiles, auxiliary_profiles)
    novelty = 1.0 - auxiliary_to_r0.astype(float)
    primary_cell_count = min(CELL_COUNTS, key=lambda value: abs(value - math.sqrt(len(r0))))
    sensitivity: list[dict[str, Any]] = []
    for cell_count in CELL_COUNTS:
        r0_cells, auxiliary_cells, auxiliary_weights = assignments[cell_count]
        for horizon in (0.5, 1.0, 2.0):
            draws = int(round(len(r0) * horizon))
            for objective in ("equal_priority_minimax", "equal_priority_l2"):
                sensitivity.append(
                    tradeoff_scenario(
                        alphas=alphas,
                        axes=axes,
                        auxiliary_weights=auxiliary_weights,
                        novelty_weights=novelty,
                        r0_cells=r0_cells,
                        auxiliary_cells=auxiliary_cells,
                        cell_count=cell_count,
                        draws=draws,
                        objective=objective,
                    )
                )
    primary = next(
        result
        for result in sensitivity
        if result["cell_count"] == primary_cell_count
        and result["draws"] == len(r0)
        and result["objective"] == "equal_priority_minimax"
    )
    selected_auxiliary_probability = float(primary["selected"]["layer_probability"])
    sensitivity_selected = [
        float(result["selected"]["layer_probability"]) for result in sensitivity
    ]

    r0_cells, auxiliary_cells, auxiliary_weights = assignments[primary_cell_count]
    nn_path = output_dir / "agile_virtual_only_nn_to_r0.csv"
    write_nn_rows(
        nn_path,
        auxiliary,
        r0,
        auxiliary_to_r0,
        auxiliary_to_r0_indices,
        auxiliary_profiles,
    )
    sampler_path = output_dir / "layer_aware_sampler_rows.csv"
    sampler_hash = write_sampler_rows(
        sampler_path,
        r0,
        auxiliary,
        r0_profiles,
        auxiliary_profiles,
        r0_cells,
        auxiliary_cells,
        auxiliary_weights,
        selected_auxiliary_probability,
        auxiliary_to_r0,
    )

    audit = {
        "audit_version": "compose_lipid_agile_ugi_bias_v1",
        "software": {
            "rdkit": rdBase.rdkitVersion,
            "numpy": np.__version__,
            "fingerprint": {
                "type": "Morgan bit vector",
                "radius": FINGERPRINT_RADIUS,
                "bits": FINGERPRINT_BITS,
                "include_chirality": True,
            },
        },
        "corpus_counts": {
            "r0_observed_unique": len(r0),
            "agile_virtual_only_unique": len(auxiliary),
            "raw_union_total": len(r0) + len(auxiliary),
            "raw_agile_fraction": raw_auxiliary_fraction,
            "exact_cross_layer_canonical_duplicate_count": 0,
            "ecfp4_tanimoto_one_cross_layer_count": int(
                np.sum(auxiliary_to_r0 == 1.0)
            ),
            "identity_note": (
                "Tanimoto 1.0 is fingerprint equivalence, not exact molecular identity; "
                "the auxiliary layer was defined after exact canonical subtraction from R0"
            ),
        },
        "source_and_family": source_family_audit(r0),
        "profile": {
            "r0_observed": corpus_profile(r0_profiles),
            "agile_virtual_only": corpus_profile(auxiliary_profiles),
        },
        "nearest_neighbor": {
            "agile_virtual_only_to_r0": nearest_neighbor_summary(auxiliary_to_r0),
            "r0_within_layer": nearest_neighbor_summary(r0_self),
            "agile_virtual_only_within_layer": nearest_neighbor_summary(auxiliary_self),
        },
        "structural_cells": {
            str(cell_count): result for cell_count, result in cell_results.items()
        },
        "weight_selection": {
            "maximum_allowed_probability": raw_auxiliary_fraction,
            "maximum_definition": (
                "raw unique-union fraction; optimization may only down-weight AGILE"
            ),
            "primary_scenario": {
                "cell_count": primary_cell_count,
                "cell_count_rationale": "candidate in {64,128,256} nearest sqrt(R0 count)",
                "draws": len(r0),
                "draw_horizon_rationale": "one R0-equivalent training epoch",
                "objective": "equal-priority minimax normalized realism-versus-coverage regret",
                "objective_rationale": (
                    "normalizing both endpoints and minimizing worst regret avoids a fitted "
                    "or hand-set tradeoff coefficient"
                ),
                "selected": primary["selected"],
            },
            "selected_auxiliary_probability": selected_auxiliary_probability,
            "selected_r0_probability": 1.0 - selected_auxiliary_probability,
            "relative_reduction_from_raw_agile_fraction": (
                1.0 - selected_auxiliary_probability / raw_auxiliary_fraction
            ),
            "sensitivity": {
                "cell_counts": list(CELL_COUNTS),
                "r0_epoch_horizons": [0.5, 1.0, 2.0],
                "objectives": ["equal_priority_minimax", "equal_priority_l2"],
                "selected_probability_min": min(sensitivity_selected),
                "selected_probability_median": statistics.median(sensitivity_selected),
                "selected_probability_max": max(sensitivity_selected),
                "scenarios": sensitivity,
            },
        },
        "within_auxiliary_weighting": {
            "method": "R0-anchored ECFP4 structural-cell calibration",
            "cell_count": primary_cell_count,
            "definition": (
                "each occupied auxiliary cell receives the corresponding R0 cell mass; "
                "mass is uniform among auxiliary structures inside that cell"
            ),
            "purpose": (
                "prevent dense Ugi neighborhoods from receiving probability solely because "
                "they contain many enumerated variants"
            ),
        },
        "split_freeze": {
            "timing": "all group fields computed before sampling probabilities",
            "r0": (
                "preserve frozen source study/component/reaction groups from R0; add connector-core group"
            ),
            "agile_auxiliary": (
                "freeze reaction family and structural component proxies; exact A/B/C component "
                "identities are unresolved, so these rows are structural-pretraining-only"
            ),
            "no_claim": (
                "heteroatom connector and tail architecture proxies are not synthetic building-block identities"
            ),
        },
        "artifacts": {
            "nearest_neighbor_rows": {
                "path": str(nn_path.relative_to(REPO_ROOT)),
                "sha256": sha256_file(nn_path),
            },
            "sampler_rows": {
                "path": str(sampler_path.relative_to(REPO_ROOT)),
                "sha256": sampler_hash,
            },
        },
    }
    audit_path = output_dir / "agile_ugi_bias_audit.json"
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    manifest = {
        "manifest_version": "compose_lipid_layer_aware_sampler_v1",
        "layers": {
            "r0_observed_real": {
                "row_count": len(r0),
                "layer_probability": 1.0 - selected_auxiliary_probability,
                "within_layer": "uniform over unique canonical observed structures",
            },
            "r1_auxiliary_agile_ugi": {
                "row_count": len(auxiliary),
                "layer_probability": selected_auxiliary_probability,
                "within_layer": "R0-anchored ECFP4 structural-cell calibrated",
                "reaction_family": "ugi_3cr_fixed_core",
                "eligibility": (
                    "structural pretraining only until exact component identities and route "
                    "certification are recovered"
                ),
            },
        },
        "raw_agile_union_fraction": raw_auxiliary_fraction,
        "weight_selection_audit": {
            "path": str(audit_path.relative_to(REPO_ROOT)),
            "sha256": sha256_file(audit_path),
        },
        "sampler_rows": {
            "path": str(sampler_path.relative_to(REPO_ROOT)),
            "sha256": sampler_hash,
            "row_count": len(r0) + len(auxiliary),
            "total_probability_tolerance": 1e-10,
        },
        "selection_contract": {
            "layer_then_structure": True,
            "replacement": True,
            "split_groups_precede_selection": True,
            "agile_cannot_enter_observed_layer": True,
            "prospective_rows_allowed": False,
        },
    }
    manifest_path = output_dir / "layer_aware_training_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
