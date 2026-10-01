"""Check whether a QED benchmark source survives the molecular representation."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from rdkit import Chem
from rdkit.Chem import QED, rdFingerprintGenerator

from compose_v4.chem.molecular_graph import (
    MolecularGraphError,
    molecular_graph_to_smiles,
    smiles_to_molecular_graph,
)


@dataclass(frozen=True)
class QEDSourceSupport:
    source: str
    represented: str | None
    supported: bool
    benchmark_equivalent: bool
    isomeric_identity_retained: bool
    reason: str | None

    def to_dict(self) -> dict[str, str | bool | None]:
        return asdict(self)


def audit_qed_source(source: str, *, max_active_atoms: int = 40) -> QEDSourceSupport:
    """Check graph support and invariance of the benchmark's QED and fingerprint."""
    original = Chem.MolFromSmiles(source)
    if original is None:
        return QEDSourceSupport(source, None, False, False, False, "rdkit_parse_failed")
    if original.GetNumHeavyAtoms() > max_active_atoms:
        return QEDSourceSupport(source, None, False, False, False, "active_atom_limit")

    try:
        graph = smiles_to_molecular_graph(source)
        represented = molecular_graph_to_smiles(graph)
    except MolecularGraphError:
        return QEDSourceSupport(source, None, False, False, False, "graph_not_representable")
    if represented is None:
        return QEDSourceSupport(source, None, False, False, False, "graph_decode_failed")

    decoded = Chem.MolFromSmiles(represented)
    if decoded is None:
        return QEDSourceSupport(source, represented, False, False, False, "rdkit_decode_failed")
    original_nonisomeric = Chem.MolToSmiles(original, isomericSmiles=False)
    decoded_nonisomeric = Chem.MolToSmiles(decoded, isomericSmiles=False)
    if original_nonisomeric != decoded_nonisomeric:
        return QEDSourceSupport(source, represented, False, False, False, "graph_identity_changed")

    fingerprint = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    equivalent = fingerprint.GetFingerprint(original) == fingerprint.GetFingerprint(decoded)
    equivalent = equivalent and abs(QED.qed(original) - QED.qed(decoded)) <= 1e-12
    isomeric_identity_retained = Chem.MolToSmiles(
        original, isomericSmiles=True
    ) == Chem.MolToSmiles(decoded, isomericSmiles=True)
    return QEDSourceSupport(
        source,
        represented,
        True,
        equivalent,
        isomeric_identity_retained,
        None if equivalent else "benchmark_metric_changed",
    )
