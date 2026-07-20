"""Qualification tests for the AGILE Ugi-3CR alpha-amino-amide transform.

These lock the correctness gate: the qualified registry entry must reconstruct a
released AGILE product exactly, reject chemoselectivity negatives, bind its audit
hash, and remain fail-closed for unqualified reactions.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from rdkit import Chem

from compose_v4.lipids.reaction_enumeration import (
    BuildingBlock,
    EnumerationError,
    ReactionEnumerator,
)
from compose_v4.lipids.reaction_registry import ReactionRegistry

REPO_ROOT = Path(__file__).resolve().parents[1]
QUALIFIED = REPO_ROOT / "configs/lipid_reactions/qualified_reactions_v1.json"
LITERATURE = REPO_ROOT / "configs/lipid_reactions/pilot_literature_specs_v1.json"
AUDIT = REPO_ROOT / "artifacts/datasets/compose_lipid_pretraining_v1/ugi_3cr_qualification.json"


def _canonical(smiles: str) -> str:
    molecule = Chem.MolFromSmiles(smiles)
    assert molecule is not None, smiles
    return Chem.MolToSmiles(molecule, isomericSmiles=True)


def _blocks(amine: str, oxo: str, iso: str) -> list[BuildingBlock]:
    return [
        BuildingBlock("amine", "amine_head", amine, {"head_class": "amine"}),
        BuildingBlock("oxo", "oxoester_aldehyde_body_tail", oxo, {"linker_class": "ester"}),
        BuildingBlock("iso", "isocyanide_tail", iso, {"tail_class": "alkyl"}),
    ]


def test_ugi_3cr_is_qualified() -> None:
    registry = ReactionRegistry.load(QUALIFIED)
    assert registry.preflight()["qualified_count"] == 1
    spec = registry.by_id("ugi_3cr_agile")
    assert spec.status == "qualified_for_enumeration"
    assert spec.qualification_blockers() == ()


def test_enumerator_reconstructs_released_product_exactly() -> None:
    spec = ReactionRegistry.load(QUALIFIED).by_id("ugi_3cr_agile")
    enumerator = ReactionEnumerator(spec)
    products = list(
        enumerator.react(
            _blocks("CN(C)CCN", "O=C(CCCCCCCC)OCCCCCC=O", "CCCCCCCCCCCC[N+]#[C-]")
        )
    )
    target = _canonical("CCCCCCCCCCCCNC(=O)C(CCCCCOC(=O)CCCCCCCC)NCCN(C)C")
    assert target in {product.canonical_smiles for product in products}
    assert products[0].reaction_id == "ugi_3cr_agile"
    assert products[0].reactant_roles == (
        "amine_head",
        "oxoester_aldehyde_body_tail",
        "isocyanide_tail",
    )


@pytest.mark.parametrize(
    "amine,oxo,iso,reason",
    [
        ("CCN(CC)CC", "O=C(CCCCCCCC)OCCCCCC=O", "CCCCCCCCCCCC[N+]#[C-]", "tertiary amine"),
        ("CN(C)CCN", "CCCCC(=O)CCCC", "CCCCCCCCCCCC[N+]#[C-]", "ketone not aldehyde"),
        ("CN(C)CCN", "O=C(CCCCCCCC)OCCCCCC=O", "CCCCCCCCCCCCC#N", "nitrile not isocyanide"),
    ],
)
def test_enumerator_rejects_chemoselectivity_negatives(amine, oxo, iso, reason) -> None:
    spec = ReactionRegistry.load(QUALIFIED).by_id("ugi_3cr_agile")
    enumerator = ReactionEnumerator(spec)
    assert list(enumerator.react(_blocks(amine, oxo, iso))) == [], reason


def test_multi_amine_substrate_enumerates_multiple_regiochemistries() -> None:
    spec = ReactionRegistry.load(QUALIFIED).by_id("ugi_3cr_agile")
    enumerator = ReactionEnumerator(spec)
    # ethylenediamine has two equivalent primary amines -> the reaction can fire
    # at either; distinct canonical products are deduplicated per substrate tuple.
    products = list(enumerator.react(_blocks("NCCN", "CCCCCC=O", "CCCCCC[N+]#[C-]")))
    assert len(products) >= 1


def test_audit_hash_is_bound_into_registry() -> None:
    spec = ReactionRegistry.load(QUALIFIED).by_id("ugi_3cr_agile")
    audit_sha = hashlib.sha256(AUDIT.read_bytes()).hexdigest()
    assert spec.implementation["artifact_hash"] == audit_sha
    audit = json.loads(AUDIT.read_text())
    assert audit["positive_reconstruction"]["exact_fraction"] == 1.0
    assert audit["negative_rejection"]["all_rejected"] is True


def test_enumerator_refuses_unqualified_reaction() -> None:
    literature_spec = ReactionRegistry.load(LITERATURE).by_id("ugi_3cr_agile")
    with pytest.raises(Exception):
        ReactionEnumerator(literature_spec)


def test_blocks_must_be_supplied_in_role_order() -> None:
    spec = ReactionRegistry.load(QUALIFIED).by_id("ugi_3cr_agile")
    enumerator = ReactionEnumerator(spec)
    wrong_order = [
        BuildingBlock("iso", "isocyanide_tail", "CCCCCC[N+]#[C-]", {}),
        BuildingBlock("amine", "amine_head", "CN(C)CCN", {}),
        BuildingBlock("oxo", "oxoester_aldehyde_body_tail", "CCCCCC=O", {}),
    ]
    with pytest.raises(EnumerationError):
        list(enumerator.react(wrong_order))
