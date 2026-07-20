"""Qualification tests for the complementary reaction-family batch."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from rdkit import Chem

from compose_v4.lipids.reaction_enumeration import BuildingBlock, ReactionEnumerator
from compose_v4.lipids.reaction_registry import ReactionRegistry

REPO_ROOT = Path(__file__).resolve().parents[1]
REGISTRY = REPO_ROOT / "configs/lipid_reactions/qualified_reaction_families_v1.json"
AUDIT = REPO_ROOT / "artifacts/datasets/compose_lipid_pretraining_v1/reaction_families_qualification.json"

# reaction_id -> (blocks in role order, expected canonical product)
CASES = {
    "aza_michael_amine_acrylate": (
        [("amine_head", "CCCCCCCCCCCCN"), ("alkyl_acrylate_or_acrylamide_tail", "C=CC(=O)OC")],
        "CCCCCCCCCCCCNCCC(=O)OC"),
    "epoxide_opening_amine": (
        [("amine_head", "CCCCCCCCCCCCN"), ("alkyl_epoxide_tail", "CCCCCCCCC1CO1")],
        "CCCCCCCCCCCCNCC(O)CCCCCCCC"),
    "passerini_3cr": (
        [("carboxylic_acid_tail", "CC(=O)O"), ("aldehyde_body", "CCCCCCCC=O"),
         ("isocyanide_tail", "CCCCCCCCCCCC[N+]#[C-]")],
        "CCCCCCCC(OC(C)=O)C(=O)NCCCCCCCCCCCC"),
    "reductive_amination_amine_aldehyde": (
        [("amine_head", "CCCCCCCCCCCCN"), ("aldehyde_tail", "CCCCCCCCCCCC=O")],
        "CCCCCCCCCCCCNCCCCCCCCCCCC"),
    "amide_coupling_acid_amine": (
        [("carboxylic_acid_tail", "CCCCCCCCCCCC(=O)O"), ("amine_head", "CCCCCCCCCCCCN")],
        "CCCCCCCCCCCCNC(=O)CCCCCCCCCCC"),
    "thiol_michael_thioether": (
        [("thiol_tail", "CCCCCCCCCCCCS"), ("alkyl_acrylate_or_acrylamide_tail", "C=CC(=O)OCCCCCC")],
        "CCCCCCCCCCCCSCCC(=O)OCCCCCC"),
    "disulfide_coupling": (
        [("thiol_tail", "CCCCCCCCCCCCS"), ("thiol_head_or_tail", "CCCCCCS")],
        "CCCCCCSSCCCCCCCCCCCC"),
    "carbamate_amine_chloroformate": (
        [("amine_head", "CCCCCCCCCCCCN"), ("chloroformate_tail", "O=C(Cl)OCCCCCC")],
        "CCCCCCCCCCCCNC(=O)OCCCCCC"),
}


def _canon(s: str) -> str:
    return Chem.MolToSmiles(Chem.MolFromSmiles(s))


def test_all_families_qualified() -> None:
    families = ReactionRegistry.load(REGISTRY)
    ugi = ReactionRegistry.load(REPO_ROOT / "configs/lipid_reactions/qualified_reactions_v1.json")
    assert families.preflight()["qualified_count"] == 8
    assert ugi.preflight()["qualified_count"] == 1  # Ugi-3CR -> 9 complementary families total


@pytest.mark.parametrize("reaction_id", list(CASES))
def test_family_reconstructs_mechanism_product(reaction_id) -> None:
    blocks_spec, expected = CASES[reaction_id]
    spec = ReactionRegistry.load(REGISTRY).by_id(reaction_id)
    enumerator = ReactionEnumerator(spec)
    blocks = [BuildingBlock(f"{role}:0", role, smiles, {}) for role, smiles in blocks_spec]
    products = {p.canonical_smiles for p in enumerator.react(blocks)}
    assert _canon(expected) in products


def test_negatives_rejected_and_audit_hash_bound() -> None:
    registry = ReactionRegistry.load(REGISTRY)
    audit_sha = hashlib.sha256(AUDIT.read_bytes()).hexdigest()
    for spec in registry.reactions:
        assert spec.implementation["artifact_hash"] == audit_sha
    audit = json.loads(AUDIT.read_text())
    assert audit["all_qualified"] is True
    for family in audit["families"]:
        assert family["positive_reconstruction"]["reproduced_exactly"] is True
        assert family["negative_rejection"]["all_rejected"] is True


def test_architectures_are_distinct() -> None:
    registry = ReactionRegistry.load(REGISTRY)
    architectures = {spec.architecture for spec in registry.reactions}
    assert len(architectures) == 8  # genuinely complementary, not one scaffold
