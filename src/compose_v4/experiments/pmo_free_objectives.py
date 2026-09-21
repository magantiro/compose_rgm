"""Exact offline re-derivations of PMO objectives that cost nothing to evaluate.

Two of the three tasks in the COMPOSE PMO pilot are pure similarity/descriptor
computations, so they can be reproduced locally in RDKit with no PyTDC, no
network, no asset file and -- critically -- no charged oracle call.  That makes
them the right development scorer for allocation work: unlike a synthetic
pseudo-objective, a result measured against these transfers to the real task,
because it IS the real task.

PARITY IS MEASURED, NOT ASSUMED.  ``tests/test_pmo_free_objectives.py`` checks
``celecoxib_rediscovery`` against all 250 charged rows of the completed 3x250
run recorded in ``diagnostics/pmo_3x250_autopsy_v1.json`` and requires
max_abs_delta == 0.0.  If RDKit drifts, that test fails rather than letting a
silently different objective stand in for the benchmark.

The oracle protocol string for anything scored here is ``local:...-free-rdkit-v1``,
matching the existing convention, so an artifact produced with these can never be
mistaken for a sealed charged receipt.
"""

from __future__ import annotations

import math

from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem, Descriptors

RDLogger.DisableLog("rdApp.*")

# ---- Reference molecules (benchmark constants, not tunable) ----

CELECOXIB = "CC1=CC=C(C=C1)C1=CC(=NN1C1=CC=C(C=C1)S(N)(=O)=O)C(F)(F)F"
PERINDOPRIL = "O=C(OCC)C(NC(C(=O)N1C(C(=O)O)CC2CCCCC12)C)CCC"

FREE_ORACLE_PROTOCOL = "local:pmo-free-rdkit-v1"


def _count_fingerprint(mol):
    """ECFP4 as an UNFOLDED COUNT fingerprint.

    The PMO similarity oracles use ``AllChem.GetMorganFingerprint`` (counts, not
    folded bits).  A folded binary fingerprint gives visibly different Tanimoto
    values and would silently stop being the benchmark objective.
    """
    return AllChem.GetMorganFingerprint(mol, 2)


def _similarity(smiles: str, reference_fp) -> float:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return 0.0
    return float(DataStructs.TanimotoSimilarity(reference_fp, _count_fingerprint(mol)))


_CELECOXIB_FP = _count_fingerprint(Chem.MolFromSmiles(CELECOXIB))
_PERINDOPRIL_FP = _count_fingerprint(Chem.MolFromSmiles(PERINDOPRIL))


def celecoxib_rediscovery(smiles: str) -> float:
    """ECFP4 count-fingerprint Tanimoto to celecoxib.  Verified exact on 250 rows."""
    return _similarity(smiles, _CELECOXIB_FP)


def _gaussian(value: float, mu: float, sigma: float) -> float:
    return math.exp(-0.5 * ((value - mu) / sigma) ** 2)


def perindopril_mpo(smiles: str) -> float:
    """Geometric mean of ECFP4 similarity to perindopril and a Gaussian on the
    aromatic-ring count (mu=2, sigma=0.5)."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return 0.0
    similarity = _similarity(smiles, _PERINDOPRIL_FP)
    rings = float(Descriptors.NumAromaticRings(mol))
    return float(math.sqrt(max(similarity, 0.0) * _gaussian(rings, 2.0, 0.5)))


FREE_OBJECTIVES = {
    "celecoxib_rediscovery": celecoxib_rediscovery,
    "perindopril_mpo": perindopril_mpo,
}


def build_free_scorer(task: str):
    """Return a zero-cost scorer for ``task``.

    Raises for any task that is NOT exactly reproducible offline, so an
    asset-backed oracle can never be silently replaced by an approximation.
    """
    if task not in FREE_OBJECTIVES:
        raise ValueError(
            f"{task!r} has no verified free re-derivation; available: "
            f"{sorted(FREE_OBJECTIVES)}. Asset-backed oracles (drd2, gsk3b, jnk3) "
            f"must not be approximated."
        )
    return FREE_OBJECTIVES[task]
