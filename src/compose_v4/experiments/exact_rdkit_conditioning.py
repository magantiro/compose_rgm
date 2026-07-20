"""Evaluation contracts for deterministic RDKit scalar conditioning targets."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from math import isfinite
from typing import Literal

import numpy as np
from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, QED

from compose_v4.chem.molecular_graph import MolecularGraph, molecular_graph_to_smiles
from compose_v4.chem.state import is_connected_or_null, is_valid_state
from compose_v4.experiments.griddd_conditional import molecular_ring_stratum
from compose_v4.rewrite.kernel import canonical_state_key


ExactRDKitPropertyName = Literal["qed", "molecular_weight", "crippen_logp"]
ExactRDKitScorer = Callable[[MolecularGraph], float]


GRIDDD_ZINC_REFERENCE_THRESHOLDS: Mapping[str, Mapping[str, float]] = {
    "qed": {"mae": 0.04, "validity_fraction": 0.872},
    "molecular_weight": {"mae": 4.89, "validity_fraction": 0.842},
    "crippen_logp": {"mae": 0.19, "validity_fraction": 0.879},
}


def _molecule(state: MolecularGraph) -> Chem.Mol:
    smiles = molecular_graph_to_smiles(state)
    molecule = None if smiles is None else Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError("exact-RDKit property requires a non-null valid molecule")
    return molecule


def exact_rdkit_qed(state: MolecularGraph) -> float:
    return float(QED.qed(_molecule(state)))


def exact_rdkit_molecular_weight(state: MolecularGraph) -> float:
    return float(Descriptors.MolWt(_molecule(state)))


def exact_rdkit_crippen_logp(state: MolecularGraph) -> float:
    return float(Crippen.MolLogP(_molecule(state)))


EXACT_RDKIT_SCORERS: Mapping[ExactRDKitPropertyName, ExactRDKitScorer] = {
    "qed": exact_rdkit_qed,
    "molecular_weight": exact_rdkit_molecular_weight,
    "crippen_logp": exact_rdkit_crippen_logp,
}


@dataclass(frozen=True)
class ExactRDKitTargetSpec:
    property_name: ExactRDKitPropertyName
    priority: int
    purpose: str

    def __post_init__(self) -> None:
        if self.property_name not in EXACT_RDKIT_SCORERS:
            raise ValueError("unsupported exact-RDKit target")
        if int(self.priority) <= 0:
            raise ValueError("target priority must be positive")
        if not self.purpose:
            raise ValueError("target purpose must be non-empty")

    @property
    def scorer(self) -> ExactRDKitScorer:
        return EXACT_RDKIT_SCORERS[self.property_name]

    @property
    def griddd_zinc_reference(self) -> Mapping[str, float]:
        return GRIDDD_ZINC_REFERENCE_THRESHOLDS[self.property_name]


def standard_exact_rdkit_target_specs() -> tuple[ExactRDKitTargetSpec, ...]:
    return (
        ExactRDKitTargetSpec("qed", 1, "lead-constrained QED optimization first"),
        ExactRDKitTargetSpec(
            "molecular_weight",
            2,
            "de novo target response and flexible-size proof second",
        ),
        ExactRDKitTargetSpec(
            "crippen_logp",
            3,
            "de novo lipophilicity target response third",
        ),
    )


@dataclass(frozen=True)
class TargetedCandidateRecord:
    target_value: float
    state: MolecularGraph | None
    trajectory_valid: bool
    trajectory_connected: bool
    metadata: Mapping[str, object] | None = None

    def __post_init__(self) -> None:
        if not isfinite(float(self.target_value)):
            raise ValueError("target value must be finite")


@dataclass(frozen=True)
class ExactRDKitTargetResponseEvaluator:
    spec: ExactRDKitTargetSpec

    def evaluate(
        self,
        records: Sequence[TargetedCandidateRecord],
    ) -> dict[str, object]:
        if not records:
            raise ValueError("target-response evaluation requires at least one attempt")
        rows: list[dict[str, object]] = []
        for index, record in enumerate(records):
            state = record.state
            valid = bool(state is not None and is_valid_state(state))
            connected = bool(state is not None and is_connected_or_null(state))
            prediction: float | None = None
            absolute_error: float | None = None
            atom_count: int | None = None
            if valid and connected and state is not None:
                prediction = float(self.spec.scorer(state))
                absolute_error = abs(prediction - float(record.target_value))
                molecule = _molecule(state)
                atom_count = int(molecule.GetNumHeavyAtoms())
            rows.append(
                {
                    "attempt_index": index,
                    "target_value": float(record.target_value),
                    "valid": valid,
                    "connected": connected,
                    "trajectory_valid": bool(record.trajectory_valid),
                    "trajectory_connected": bool(record.trajectory_connected),
                    "predicted_value": prediction,
                    "absolute_error": absolute_error,
                    "heavy_atom_count": atom_count,
                    "canonical_state_key": (
                        canonical_state_key(state) if valid and state is not None else None
                    ),
                    "ring_stratum": (
                        molecular_ring_stratum(state)
                        if valid and connected and state is not None
                        else None
                    ),
                    "metadata": dict(record.metadata or {}),
                }
            )
        valid_rows = [row for row in rows if row["predicted_value"] is not None]
        target_values = sorted({float(row["target_value"]) for row in rows})
        by_target = []
        for target in target_values:
            target_rows = [row for row in rows if float(row["target_value"]) == target]
            target_valid = [
                row for row in target_rows if row["predicted_value"] is not None
            ]
            by_target.append(
                {
                    "target_value": target,
                    "attempts": len(target_rows),
                    "valid_candidates": len(target_valid),
                    "validity_fraction": len(target_valid) / len(target_rows),
                    "mean_predicted_value": (
                        None
                        if not target_valid
                        else float(
                            np.mean(
                                [float(row["predicted_value"]) for row in target_valid]
                            )
                        )
                    ),
                    "mean_absolute_error": (
                        None
                        if not target_valid
                        else float(
                            np.mean(
                                [float(row["absolute_error"]) for row in target_valid]
                            )
                        )
                    ),
                    "mean_heavy_atom_count": (
                        None
                        if not target_valid
                        else float(
                            np.mean(
                                [float(row["heavy_atom_count"]) for row in target_valid]
                            )
                        )
                    ),
                    "ring_stratum_counts": {
                        name: sum(row["ring_stratum"] == name for row in target_valid)
                        for name in (
                            "acyclic",
                            "isolated_ring",
                            "fused_or_bridged",
                        )
                    },
                }
            )
        valid_fraction = len(valid_rows) / len(rows)
        mae = (
            None
            if not valid_rows
            else float(np.mean([float(row["absolute_error"]) for row in valid_rows]))
        )
        response_rows = [
            row
            for row in by_target
            if row["mean_predicted_value"] is not None
        ]
        predicted_monotonic = all(
            float(left["mean_predicted_value"])
            <= float(right["mean_predicted_value"])
            for left, right in zip(response_rows, response_rows[1:])
        )
        atom_count_monotonic = all(
            float(left["mean_heavy_atom_count"])
            <= float(right["mean_heavy_atom_count"])
            for left, right in zip(response_rows, response_rows[1:])
        )
        reference = self.spec.griddd_zinc_reference
        return {
            "format": "compose_v4_exact_rdkit_target_response_v1",
            "property_name": self.spec.property_name,
            "priority": self.spec.priority,
            "purpose": self.spec.purpose,
            "attempts": len(rows),
            "valid_candidates": len(valid_rows),
            "validity_fraction": valid_fraction,
            "mean_absolute_error_valid_candidates": mae,
            "all_trajectory_states_valid": all(
                bool(row["trajectory_valid"]) for row in rows
            ),
            "all_trajectory_states_connected": all(
                bool(row["trajectory_connected"]) for row in rows
            ),
            "target_response": {
                "mean_prediction_monotonic_non_decreasing": predicted_monotonic,
                "mean_heavy_atom_count_monotonic_non_decreasing": atom_count_monotonic,
                "flexible_size_proof_required": (
                    self.spec.property_name == "molecular_weight"
                ),
                "by_target": by_target,
            },
            "griddd_zinc_reported_reference": {
                "mae": float(reference["mae"]),
                "validity_fraction": float(reference["validity_fraction"]),
                "direct_comparison_claim_authorized": False,
            },
            "meets_numeric_reference_thresholds": (
                mae is not None
                and mae <= float(reference["mae"])
                and valid_fraction >= float(reference["validity_fraction"])
            ),
            "records": rows,
        }


__all__ = [
    "EXACT_RDKIT_SCORERS",
    "GRIDDD_ZINC_REFERENCE_THRESHOLDS",
    "ExactRDKitTargetResponseEvaluator",
    "ExactRDKitTargetSpec",
    "TargetedCandidateRecord",
    "exact_rdkit_crippen_logp",
    "exact_rdkit_molecular_weight",
    "exact_rdkit_qed",
    "standard_exact_rdkit_target_specs",
]
