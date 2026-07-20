"""Endpoint-property targets for directly conditioned molecular RGM training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, QED

from compose_v4.chem.molecular_graph import (
    MolecularGraph,
    molecular_graph_to_smiles,
)
from compose_v4.experiments.cnof_conditional import PathRecord


PropertyFunction = Callable[[Chem.Mol], float]

PROPERTY_FUNCTIONS: dict[str, PropertyFunction] = {
    "qed": lambda molecule: float(QED.qed(molecule)),
    "logp": lambda molecule: float(Crippen.MolLogP(molecule)),
    "molecular_weight": lambda molecule: float(Descriptors.MolWt(molecule)),
}


@dataclass(frozen=True)
class PropertyConditionNormalizer:
    names: tuple[str, ...]
    means: tuple[float, ...]
    standard_deviations: tuple[float, ...]

    def __post_init__(self) -> None:
        if not self.names or len(set(self.names)) != len(self.names):
            raise ValueError("property condition names must be non-empty and unique")
        if not (
            len(self.names) == len(self.means) == len(self.standard_deviations)
        ):
            raise ValueError("property condition normalization fields do not align")
        if any(name not in PROPERTY_FUNCTIONS for name in self.names):
            raise ValueError("unknown molecular property condition")
        if not np.isfinite(self.means).all() or not np.isfinite(
            self.standard_deviations
        ).all():
            raise ValueError("property normalization statistics must be finite")
        if any(value <= 0.0 for value in self.standard_deviations):
            raise ValueError("property standard deviations must be positive")

    def transform(self, raw_values: tuple[float, ...]) -> tuple[float, ...]:
        if len(raw_values) != len(self.names):
            raise ValueError("raw property vector has the wrong dimension")
        return tuple(
            (float(value) - mean) / standard_deviation
            for value, mean, standard_deviation in zip(
                raw_values,
                self.means,
                self.standard_deviations,
            )
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "names": list(self.names),
            "means": list(self.means),
            "standard_deviations": list(self.standard_deviations),
        }


def molecular_property_values(
    state: MolecularGraph,
    names: tuple[str, ...],
) -> tuple[float, ...]:
    smiles = molecular_graph_to_smiles(state)
    if smiles is None:
        raise ValueError("property targets require a non-null molecule")
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError("valid molecular state failed RDKit property conversion")
    try:
        return tuple(float(PROPERTY_FUNCTIONS[name](molecule)) for name in names)
    except KeyError as error:
        raise ValueError(f"unknown molecular property condition: {error.args[0]}") from error


def _raw_record_conditions(
    records: tuple[PathRecord, ...],
    names: tuple[str, ...],
) -> dict[str, tuple[float, ...]]:
    values: dict[str, tuple[float, ...]] = {}
    for record in records:
        candidate = molecular_property_values(record.path.trace.target, names)
        previous = values.setdefault(record.target_key, candidate)
        if not np.allclose(previous, candidate, rtol=0.0, atol=1e-8):
            raise ValueError("one target key maps to inconsistent property values")
    if not values:
        raise ValueError("cannot build property conditions from no records")
    return values


def fit_property_condition_normalizer(
    records: tuple[PathRecord, ...],
    names: tuple[str, ...],
) -> PropertyConditionNormalizer:
    if not names:
        raise ValueError("at least one property condition is required")
    raw = _raw_record_conditions(records, names)
    matrix = np.asarray(tuple(raw.values()), dtype=np.float64)
    means = matrix.mean(axis=0)
    standard_deviations = matrix.std(axis=0, ddof=1)
    if np.any(standard_deviations <= 1e-12):
        raise ValueError("a requested property is constant in the training targets")
    return PropertyConditionNormalizer(
        names=names,
        means=tuple(float(value) for value in means),
        standard_deviations=tuple(float(value) for value in standard_deviations),
    )


def standardized_record_conditions(
    records: tuple[PathRecord, ...],
    normalizer: PropertyConditionNormalizer,
) -> dict[str, tuple[float, ...]]:
    raw = _raw_record_conditions(records, normalizer.names)
    return {key: normalizer.transform(values) for key, values in raw.items()}


__all__ = [
    "PROPERTY_FUNCTIONS",
    "PropertyConditionNormalizer",
    "fit_property_condition_normalizer",
    "molecular_property_values",
    "standardized_record_conditions",
]
