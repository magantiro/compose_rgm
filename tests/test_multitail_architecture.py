"""Iterative polysubstitution builds multi-tail lipids from polyamine heads.

Real ionizable lipidoids (C12-200 style) come from a polyamine substituted at
multiple N-H sites. This verifies the enumerator supports that via sequential
application at successive N-H sites.
"""

from __future__ import annotations

from pathlib import Path

from rdkit import Chem

from compose_v4.lipids.reaction_enumeration import BuildingBlock, ReactionEnumerator
from compose_v4.lipids.reaction_registry import ReactionRegistry

REPO_ROOT = Path(__file__).resolve().parents[1]
REGISTRY = REPO_ROOT / "configs/lipid_reactions/qualified_reaction_families_v1.json"

TAIL = "C=CC(=O)OCCCCCCCCCCCC"  # dodecyl acrylate


def _substituted_once(enum: ReactionEnumerator, amine_smiles: str) -> set[str]:
    amine = BuildingBlock("amine", "amine_head", amine_smiles, {})
    tail = BuildingBlock("tail", "alkyl_acrylate_or_acrylamide_tail", TAIL, {})
    # aza-Michael role order is [amine_head, acrylate]
    return {p.canonical_smiles for p in enum.react([amine, tail])}


def test_polyamine_accrues_multiple_tails() -> None:
    spec = ReactionRegistry.load(REGISTRY).by_id("aza_michael_amine_acrylate")
    enum = ReactionEnumerator(spec)
    # ethylenediamine has 4 N-H sites -> can accrue several tails
    depth1 = _substituted_once(enum, "NCCN")
    assert depth1, "mono-substitution should produce a product"
    # substitute again on a mono product -> di-tail
    depth2 = set()
    for inter in depth1:
        depth2 |= _substituted_once(enum, inter)
    assert depth2, "di-substitution should produce a product"

    def n_ester_tails(smiles: str) -> int:
        return len(Chem.MolFromSmiles(smiles).GetSubstructMatches(
            Chem.MolFromSmarts("[CX3](=O)[OX2][CH2]")))

    assert max(n_ester_tails(s) for s in depth1) >= 1
    assert max(n_ester_tails(s) for s in depth2) >= 2  # genuinely multi-tail


def test_substitution_terminates_when_no_nh_remains() -> None:
    spec = ReactionRegistry.load(REGISTRY).by_id("aza_michael_amine_acrylate")
    enum = ReactionEnumerator(spec)
    # a tertiary amine with no N-H cannot react
    assert _substituted_once(enum, "CCN(CC)CC") == set()
