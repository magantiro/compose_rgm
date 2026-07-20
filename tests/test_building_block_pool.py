"""Invariants for the role-annotated building-block pool."""

from __future__ import annotations

import json
from pathlib import Path

from rdkit import Chem

from compose_v4.lipids.building_block_registry import load_pool

REPO_ROOT = Path(__file__).resolve().parents[1]
POOL = REPO_ROOT / "configs/lipid_reactions/building_block_pool_v1.json"

HANDLE_SMARTS = {
    "amine": "[NX3;H1,H2;!$(NC=O);!$(N=*)]",
    "carboxylic_acid": "[CX3](=[OX1])[OX2H1]",
    "aldehyde": "[CX3H1]=[OX1]",
    "epoxide": "[CH2]1[CH1][OX2]1",
    "acrylate": "[CH2]=[CH1][CX3](=[OX1])[#7,#8]",
    "isocyanide": "[C;-1,+0;X1]#[N;+1,+0;X2]",
    "thiol": "[SX2H1]",
    "chloroformate": "[Cl][CX3](=[OX1])[OX2]",
    "isocyanate": "[NX2]=[CX2]=[OX1]",
    "diol": "[OX2H1][CX4][CX4][OX2H1]",
    "dioxaphospholane": "[CH2]1[CH2][OX2][PX4](=[OX1])[OX2]1",
}


def _blocks() -> list[dict]:
    return json.loads(POOL.read_text())["blocks"]


def test_pool_has_multi_source_diversity() -> None:
    data = json.loads(POOL.read_text())
    assert data["total_blocks"] >= 200
    # at least one handle type per reaction role is present
    assert set(data["blocks_by_handle"]) == set(HANDLE_SMARTS)
    sources = data["blocks_by_source"]
    assert {"agile_measured", "curated_literature", "programmatic_rational"} <= set(sources)


def test_every_block_carries_its_intended_handle() -> None:
    for block in _blocks():
        mol = Chem.MolFromSmiles(block["canonical_smiles"])
        assert mol is not None, block
        patt = Chem.MolFromSmarts(HANDLE_SMARTS[block["handle"]])
        assert mol.HasSubstructMatch(patt), block


def test_registry_serves_blocks_for_reaction_roles() -> None:
    pool = load_pool(POOL)
    by_role = pool.blocks_for_roles(("amine_head", "isocyanide_tail", "carboxylic_acid_tail"))
    assert len(by_role["amine_head"]) >= 40
    assert len(by_role["isocyanide_tail"]) >= 10
    assert len(by_role["carboxylic_acid_tail"]) >= 10
    for role, blocks in by_role.items():
        for b in blocks:
            assert b.role == role
