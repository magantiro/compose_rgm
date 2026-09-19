"""JAK2 proof-of-control: can the critic recognise the productive transformation?

One target, one question. Autonomous JAK2 currently sits near -10.7 while known strong
trajectories reach -11.6/-11.7. Everything here asks whether a controller can tell, from
a JAK2 state, which of the available transformations moves toward that basin.

Transitions are described GLOBALLY -- the source molecule, the program that ran, and the
molecule it produced -- because the local `(site, mode)` description was measured to be
value-blind on exactly this cell. The whole point of including the destination is that
the critic gets to judge where a candidate actually lands.

Splits are by CELL, so a JAK2 run never grades itself, and supervision is sibling-matched
so the critic cannot pass by learning which parents were already good.
"""

from __future__ import annotations

import gzip
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import QED, Crippen, Descriptors, rdMolDescriptors

RDLogger.DisableLog("rdApp.*")

SCHEMA_VERSION = "t4_jak2_control_v1"

MOLECULE_DESCRIPTORS = (
    "molecular_weight",
    "logp",
    "tpsa",
    "hbd",
    "hba",
    "rotatable_bonds",
    "rings",
    "aromatic_rings",
    "heteroatoms",
    "fraction_csp3",
    "qed",
    "heavy_atoms",
)

PROGRAM_DESCRIPTORS = (
    "primitive_count",
    "net_created",
    "net_deleted",
    "ring_count_change",
    "changed_slot_count",
    "input_endpoint_similarity",
    "ancestral_primitives",
)

EDIT_FAMILIES = (
    "atom_insert",
    "atom_delete",
    "atom_restate",
    "bond_reorder",
    "bond_reroute",
    "cycle_close",
    "cycle_open",
    "ring_system_restate",
    "ring_system_delete",
)


def describe_molecule(smiles: str) -> np.ndarray | None:
    mol = Chem.MolFromSmiles(smiles) if smiles else None
    if mol is None:
        return None
    return np.asarray(
        [
            Descriptors.MolWt(mol) / 500.0,
            Crippen.MolLogP(mol) / 5.0,
            rdMolDescriptors.CalcTPSA(mol) / 140.0,
            rdMolDescriptors.CalcNumHBD(mol) / 5.0,
            rdMolDescriptors.CalcNumHBA(mol) / 10.0,
            rdMolDescriptors.CalcNumRotatableBonds(mol) / 10.0,
            rdMolDescriptors.CalcNumRings(mol) / 6.0,
            rdMolDescriptors.CalcNumAromaticRings(mol) / 4.0,
            sum(1 for a in mol.GetAtoms() if a.GetSymbol() not in ("C", "H")) / 10.0,
            rdMolDescriptors.CalcFractionCSP3(mol),
            QED.qed(mol),
            mol.GetNumHeavyAtoms() / 40.0,
        ],
        dtype=float,
    )


def feature_names() -> tuple[str, ...]:
    return (
        *(f"source_{n}" for n in MOLECULE_DESCRIPTORS),
        *(f"result_{n}" for n in MOLECULE_DESCRIPTORS),
        *(f"delta_{n}" for n in MOLECULE_DESCRIPTORS),
        *PROGRAM_DESCRIPTORS,
        *(f"uses_{f}" for f in EDIT_FAMILIES),
    )


def describe_transition(row: dict) -> np.ndarray | None:
    """Source molecule, resulting molecule, their delta, and the program that connected them."""
    source = describe_molecule(row.get("input_smiles"))
    result = describe_molecule(row.get("endpoint"))
    if source is None or result is None:
        return None
    counts = row.get("rule_counts") or {}
    total = max(sum(counts.values()), 1)
    program = np.asarray(
        [
            (row.get("primitive_count") or 0) / 16.0,
            (row.get("net_created") or 0) / 8.0,
            (row.get("net_deleted") or 0) / 8.0,
            row.get("ring_count_change") or 0,
            (row.get("changed_slot_count") or 0) / 8.0,
            row.get("input_endpoint_similarity") or 0.0,
            (row.get("ancestral_primitives") or 0) / 20.0,
        ],
        dtype=float,
    )
    families = np.asarray([counts.get(f, 0) / total for f in EDIT_FAMILIES], dtype=float)
    return np.concatenate([source, result, result - source, program, families])


def load_target(path: Path, target: str) -> list[dict]:
    with gzip.open(Path(path), "rt") as handle:
        return [row for row in map(json.loads, handle) if row.get("target") == target]


def build(rows) -> dict:
    """Featurise every transition and group the ones that shared a parent."""
    features, kept = [], []
    for row in rows:
        vector = describe_transition(row)
        if vector is None:
            continue
        features.append(vector)
        kept.append(row)
    features = np.asarray(features)

    position = {id(row): i for i, row in enumerate(kept)}
    families = defaultdict(list)
    for row in kept:
        parent = row.get("genealogical_parent")
        if parent:
            families[(row["run"], parent)].append(position[id(row)])

    groups = [
        (members, [-kept[i]["score"] for i in members])  # higher is better
        for members in families.values()
        if len(members) >= 2
    ]
    return {"features": features, "rows": kept, "groups": groups}
