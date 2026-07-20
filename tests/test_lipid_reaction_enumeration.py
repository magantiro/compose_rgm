from __future__ import annotations

from pathlib import Path

import pytest
from rdkit import Chem

from compose_v4.lipids.reaction_registry import ReactionRegistry, RegistryError
from compose_v4.lipids.streaming_enumeration import (
    CandidateProduct,
    StratifiedReservoir,
    product_bins,
    stable_priority,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def candidate(smiles: str, reaction_id: str = "test_route") -> CandidateProduct:
    molecule = Chem.MolFromSmiles(smiles)
    assert molecule is not None
    canonical = Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)
    return CandidateProduct(
        canonical_smiles=canonical,
        reaction_id=reaction_id,
        reactant_ids=(smiles,),
        reactant_roles=("test",),
        architecture_tags={
            "head_class": "amine",
            "tail_class": "alkyl",
            "linker_class": "ester",
            "degradability_class": "hydrolysable",
        },
        product_bins=product_bins(molecule),
    )


def test_committed_literature_specs_fail_closed() -> None:
    registry = ReactionRegistry.load(
        REPO_ROOT / "configs/lipid_reactions/pilot_literature_specs_v1.json"
    )
    preflight = registry.preflight()
    assert preflight["reaction_count"] == 3
    assert preflight["qualified_count"] == 0
    assert registry.qualified() == ()
    for reaction in registry.reactions:
        with pytest.raises(RegistryError, match="is not enumerable"):
            reaction.require_qualified()
        blockers = reaction.qualification_blockers()
        assert any("atom-mapped reaction SMARTS" in blocker for blocker in blockers)
        assert any("positive example" in blocker for blocker in blockers)


def test_stratified_reservoir_is_order_invariant_and_bounded() -> None:
    candidates = [candidate(smiles) for smiles in ("CCN", "CCCN", "CCCCN", "CCCCCN")]
    axes = ("reaction_family", "head_class", "tail_class")
    first = StratifiedReservoir(axes=axes, quota_per_stratum=2, seed="fixed")
    second = StratifiedReservoir(axes=axes, quota_per_stratum=2, seed="fixed")
    for item in candidates:
        first.consider(item)
    for item in reversed(candidates):
        second.consider(item)
    expected = sorted(candidates, key=lambda item: stable_priority("fixed", item.identity))[:2]
    assert [item.identity for item in first.selected()] == [item.identity for item in expected]
    assert [item.identity for item in second.selected()] == [item.identity for item in expected]
    assert list(first.stratum_counts.values()) == [2]


def test_product_bins_distinguish_core_pilot_axes() -> None:
    neutral_chain = Chem.MolFromSmiles("CCCCCC")
    charged_branched_ring = Chem.MolFromSmiles("C1CC[NH2+]CC1")
    unsaturated = Chem.MolFromSmiles("CCCC=CCCC")
    assert neutral_chain is not None
    assert charged_branched_ring is not None
    assert unsaturated is not None
    assert product_bins(neutral_chain)["charge_bin"] == "neutral"
    assert product_bins(charged_branched_ring)["charge_bin"] == "positive"
    assert product_bins(charged_branched_ring)["ring_bin"] == "ring"
    assert product_bins(unsaturated)["unsaturation_bin"] == "present"
