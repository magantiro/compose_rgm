"""Three PREDECLARED constraint families for the reversibility census.

WHY THIS FILE EXISTS
--------------------
The ring-system constraint failed its premise gate for a specific reason: a
broken fused ring system was never rebuilt, so violation was ABSORBING and
endpoint validity implied path validity. The pathwise/endpoint distinction only
has teeth when violation is REVERSIBLE -- when a trajectory can leave the
feasible set and come back.

This module declares three candidate families. All three are written down and
committed BEFORE any of them is measured, and they are ranked A > B > C in
advance. If more than one passes, the FIRST in this order is taken -- not the
one that looks best. If none passes, pathwise constraints leave the paper.

THRESHOLDS ARE NOT CHOSEN, THEY ARE READ
----------------------------------------
Every numeric bound below is loaded at runtime from an already-frozen held-in
artifact:

  * family B reads the cLogP quartiles from
    `diagnostics/retarget_goal_language_normalizers.json`, which was frozen for
    the retargeting lane and is marked
    `HELD_IN_NORMALIZERS_NO_THRESHOLD_SELECTED`;
  * family C reads the heavy-atom band already frozen in
    `pathwise_constraints.HEAVY_ATOM_BAND`, used unchanged by the retargeting
    cohort and by this lane's own panel.

Nothing here was tuned against trajectory data. Selecting a corridor because it
produces violations is the exact failure this census is guarding against, so
the corridors are the canonical summaries that already existed in the frozen
files -- the central half of the held-in distribution, and the panel's own size
band.

FAMILY A's SMARTS ARE LITERATURE ALERTS, NOT FITTED PATTERNS
------------------------------------------------------------
The undesired-group list is a standard medicinal-chemistry reactive/structural
alert set. It was written from chemical motivation and is fixed; it is not
derived from, filtered by, or checked against any COMPOSE trajectory.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import Crippen

from compose_v4.experiments.pathwise_constraints import HEAVY_ATOM_BAND

FAMILY_ORDER = ("A_undesired_motif", "B_physchem_corridor", "C_size_corridor")

#: Ranked BEFORE measurement. If several families pass, take the first.
FAMILY_PRECEDENCE_RULE = (
    "take the first PASSING family in FAMILY_ORDER, never the most favourable"
)

# ---------------------------------------------------------------------------
# FAMILY A -- undesired intermediate motifs
# ---------------------------------------------------------------------------
#: Standard reactive / undesired functional groups. A trajectory violates when
#: ANY of these is present at a committed state. Unlike a protected ring
#: system, an undesired group can appear at one step and be edited away at the
#: next, so return is chemically possible -- which is the whole point.
UNDESIRED_GROUPS: dict[str, str] = {
    "acyl_halide": "[CX3](=O)[F,Cl,Br,I]",
    "sulfonyl_halide": "[SX4](=O)(=O)[F,Cl,Br,I]",
    "aldehyde": "[CX3H1](=O)[#6]",
    "anhydride": "[CX3](=O)[OX2][CX3](=O)",
    "michael_acceptor": "[CX3]=[CX3][CX3]=[OX1]",
    "epoxide": "[OX2r3]1[#6r3][#6r3]1",
    "aziridine": "[NX3r3]1[#6r3][#6r3]1",
    "nitro": "[$([NX3](=O)=O),$([NX3+](=O)[O-])]",
    "azide": "[NX2]=[NX2+]=[NX1-]",
    "isocyanate": "[NX2]=[CX2]=[OX1]",
    "thiol": "[SX2H]",
    "peroxide": "[OX2][OX2]",
    "hydrazine": "[NX3][NX3]",
    "n_nitroso": "[NX3][NX2]=[OX1]",
}


@lru_cache(maxsize=64)
def _alert_queries() -> tuple[tuple[str, Chem.Mol], ...]:
    compiled = []
    for name, smarts in UNDESIRED_GROUPS.items():
        query = Chem.MolFromSmarts(smarts)
        if query is None:
            raise ValueError(f"undesired-group SMARTS failed to parse: {name} {smarts}")
        compiled.append((name, query))
    return tuple(compiled)


def undesired_hits(smiles: str) -> list[str]:
    """Names of undesired groups present. Empty list means the state is clean."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ["unparseable"]
    return [name for name, query in _alert_queries() if mol.HasSubstructMatch(query)]


# ---------------------------------------------------------------------------
# FAMILY B -- physicochemical corridor
# ---------------------------------------------------------------------------
DEFAULT_NORMALIZERS = "diagnostics/retarget_goal_language_normalizers.json"


@lru_cache(maxsize=8)
def clogp_corridor(normalizers_path: str = DEFAULT_NORMALIZERS) -> tuple[float, float]:
    """The central half of the FROZEN held-in cLogP distribution: [p25, p75].

    Read, never chosen. The interquartile range is the summary that already
    existed in the frozen normalizers file; it was not selected by looking at
    how many trajectories it would flag.
    """
    path = Path(normalizers_path)
    if not path.is_absolute():
        path = Path(__file__).resolve().parents[3] / normalizers_path
    norms = json.loads(path.read_text())["normalizers"]["clogp"]
    return float(norms["p25"]), float(norms["p75"])


def clogp_of(smiles: str) -> float:
    mol = Chem.MolFromSmiles(smiles)
    return float("nan") if mol is None else float(Crippen.MolLogP(mol))


def formal_charge_of(smiles: str) -> int:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return 99
    return Chem.GetFormalCharge(mol)


# ---------------------------------------------------------------------------
# FAMILY C -- structural size corridor
# ---------------------------------------------------------------------------
#: Reused verbatim from the already-frozen eligibility band. No new number.
HEAVY_ATOM_CORRIDOR = HEAVY_ATOM_BAND


def heavy_atoms_of(smiles: str) -> int:
    mol = Chem.MolFromSmiles(smiles)
    return -1 if mol is None else mol.GetNumHeavyAtoms()


def ring_count_of(smiles: str) -> int:
    mol = Chem.MolFromSmiles(smiles)
    return -1 if mol is None else mol.GetRingInfo().NumRings()


# ---------------------------------------------------------------------------
# the per-state predicates
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FamilySpec:
    key: str
    description: str
    threshold_provenance: str


FAMILIES: dict[str, FamilySpec] = {
    "A_undesired_motif": FamilySpec(
        key="A_undesired_motif",
        description="no undesired reactive group present at any committed state",
        threshold_provenance=(
            "fixed literature structural-alert list; not derived from any "
            "COMPOSE trajectory"
        ),
    ),
    "B_physchem_corridor": FamilySpec(
        key="B_physchem_corridor",
        description="cLogP(x_t) within the frozen held-in interquartile range",
        threshold_provenance=(
            "read at runtime from diagnostics/retarget_goal_language_"
            "normalizers.json (status HELD_IN_NORMALIZERS_NO_THRESHOLD_SELECTED)"
        ),
    ),
    "C_size_corridor": FamilySpec(
        key="C_size_corridor",
        description="heavy-atom count within the frozen panel eligibility band",
        threshold_provenance=(
            "reused verbatim from pathwise_constraints.HEAVY_ATOM_BAND, frozen "
            "for the retargeting cohort and this lane's panel"
        ),
    ),
}


def state_is_feasible(family: str, smiles: str) -> bool:
    """True when the state satisfies the family's constraint.

    An unparseable state is infeasible: a state COMPOSE cannot read is not a
    state that demonstrably satisfies anything.
    """
    if family == "A_undesired_motif":
        return not undesired_hits(smiles)
    if family == "B_physchem_corridor":
        value = clogp_of(smiles)
        if value != value:  # NaN
            return False
        low, high = clogp_corridor()
        return low <= value <= high
    if family == "C_size_corridor":
        count = heavy_atoms_of(smiles)
        if count < 0:
            return False
        return HEAVY_ATOM_CORRIDOR[0] <= count <= HEAVY_ATOM_CORRIDOR[1]
    raise KeyError(f"unknown family {family}")


def audit_trajectory(family: str, trajectory: list[str]) -> dict:
    """Pathwise audit for one realised trajectory under one family.

    `returned` is THE number this census exists to produce: a trajectory that
    left the feasible set and was back inside it at the endpoint. It is 0 for
    an absorbing constraint -- which is exactly what sank the ring-system
    family -- and positive only when violation is genuinely reversible.
    """
    flags = [state_is_feasible(family, key) for key in trajectory]
    if not flags:
        return {}
    source_feasible = flags[0]
    endpoint_feasible = flags[-1]
    violated = [i for i, ok in enumerate(flags[1:], start=1) if not ok]
    return {
        "states": len(trajectory),
        "source_feasible": source_feasible,
        "endpoint_feasible": endpoint_feasible,
        "any_violation": bool(violated),
        "violation_count": len(violated),
        "first_violation_index": violated[0] if violated else None,
        "returned": bool(violated and endpoint_feasible),
        "endpoint_valid_path_invalid": bool(violated and endpoint_feasible),
    }
