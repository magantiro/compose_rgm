"""Future program accessibility as a state variable.

A molecule-only critic asks whether a state looks good. The measured answer on this
branch is that it cannot tell: structure-only features scored AUC 0.495 against frontier
advance, exactly chance. The hypothesis this module tests is that the missing quantity is
not a property of the molecule at all, but of the PROGRAM SPACE the molecule opens --
two states with identical docking scores can have very different sets of legal
continuations, and only one of them can still be worked on.

    state quality  =  current molecular value  +  future program accessibility

Every feature here is computable at decision time with no oracle call, and each is a
capacity for some family of COMPOSE option rather than a generic descriptor:

    attachment capacity   open valences -- where any patch can bind at all
    scale capacity        slack to the active-atom cap -- how much can still be built
    ring-edit capacity    non-bridge ring bonds -- what `cycle_open` can act on
    deletion capacity     pendant atoms -- what can be removed without disconnection
    restatement capacity  heteroatom sites
    region diversity      distinct local environments, and their entropy

The entropy term is the honest form of "effective branching factor": a state offering
forty copies of one environment is not more manoeuvrable than one offering five distinct
ones, and a raw count would say otherwise.

CONFOUND WARNING, paid for once already on this branch. Any quantity correlated with how
much the historical search expanded a state will look predictive, because the labels
count descendants the search chose to generate -- descendant count alone scored AUC 0.851
on such a label. Expansion must be carried as a covariate in every arm, not removed from
one.
"""

from __future__ import annotations

import math
from collections import Counter

import numpy as np
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

SCHEMA_VERSION = "maneuverability_v1"

ACTIVE_ATOM_CAP = 40

FEATURE_NAMES = (
    "attachment_capacity",
    "scale_capacity",
    "ring_edit_capacity",
    "deletion_capacity",
    "restatement_capacity",
    "region_diversity",
    "region_entropy",
    "aromatic_fraction",
)


def _environment(atom) -> tuple:
    """A coarse local environment: element, degree, ring membership, aromaticity."""
    return (
        atom.GetSymbol(),
        atom.GetDegree(),
        atom.IsInRing(),
        atom.GetIsAromatic(),
    )


def maneuverability(smiles: str) -> np.ndarray | None:
    """Capacities of the option space reachable from this molecule."""
    mol = Chem.MolFromSmiles(smiles) if smiles else None
    if mol is None:
        return None
    atoms = list(mol.GetAtoms())
    heavy = mol.GetNumHeavyAtoms()
    if not heavy:
        return None

    open_valence = sum(atom.GetTotalNumHs() for atom in atoms)
    slack = max(ACTIVE_ATOM_CAP - heavy, 0)
    ring_info = mol.GetRingInfo()
    ring_bonds = [b for b in mol.GetBonds() if ring_info.NumBondRings(b.GetIdx())]
    # a ring bond is openable when removing it does not disconnect the graph, which for a
    # bond on at least one cycle is always true; bridges carry no ring membership
    ring_edit = len(ring_bonds)
    pendant = sum(1 for atom in atoms if atom.GetDegree() <= 1)
    hetero = sum(1 for atom in atoms if atom.GetSymbol() not in ("C", "H"))

    environments = Counter(_environment(atom) for atom in atoms)
    diversity = len(environments)
    total = sum(environments.values())
    entropy = -sum((n / total) * math.log(n / total) for n in environments.values())
    aromatic = sum(1 for atom in atoms if atom.GetIsAromatic()) / heavy

    return np.asarray(
        [
            open_valence / 10.0,
            slack / ACTIVE_ATOM_CAP,
            ring_edit / 10.0,
            pendant / 10.0,
            hetero / 10.0,
            diversity / 20.0,
            entropy / math.log(20.0),
            aromatic,
        ],
        dtype=float,
    )
