"""Lipid region labeler + the region-specific heteroatom-abundance pattern.

Validates that lipid_region_labels assigns head/linker/tail sensibly on a known
di-ester amino-lipid, and that the corpus-wide pattern holds: tails are near-pure
carbon, linkers are heteroatom-rich, heads are nitrogen-rich -- the justification
for a region-conditioned element-insertion prior.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from rdkit import Chem

from compose_v4.lipids.region_labels import HEAD, LINKER, TAIL, lipid_region_labels

REPO_ROOT = Path(__file__).resolve().parents[1]
DIAG = REPO_ROOT / "diagnostics/region_heteroatom_abundance.json"

# a di-ester tertiary-amine lipid: two C12 tails, two esters, an N,N'-diamine core
LIPID = "CCCCCCCCCCCCOC(=O)CCN(C)CCN(C)CCC(=O)OCCCCCCCCCCCC"


def test_labeler_assigns_head_linker_tail() -> None:
    mol = Chem.MolFromSmiles(LIPID)
    labels = lipid_region_labels(mol)
    by_region = {}
    for atom, code in zip(mol.GetAtoms(), labels):
        by_region.setdefault(code, []).append(atom.GetSymbol())
    # every basic N sits in the head region
    n_idx = [a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() == "N"]
    assert all(labels[i] == HEAD for i in n_idx), "amine N not labeled head"
    # ester carbonyl oxygens land in a linker
    assert LINKER in by_region and "O" in by_region[LINKER]
    # the long alkyl chains are tail, and tail is all carbon
    assert TAIL in by_region and set(by_region[TAIL]) == {"C"}


def test_tail_is_near_pure_carbon() -> None:
    mol = Chem.MolFromSmiles(LIPID)
    labels = lipid_region_labels(mol)
    tail_syms = [a.GetSymbol() for a, c in zip(mol.GetAtoms(), labels) if c == TAIL]
    assert tail_syms and all(s == "C" for s in tail_syms)


@pytest.mark.skipif(not DIAG.exists(), reason="region heteroatom diagnostic not generated")
def test_corpus_heteroatom_pattern_is_region_specific() -> None:
    d = json.loads(DIAG.read_text())
    reg = d["per_region"]
    # tails near-pure carbon; linkers heteroatom-rich; strong separation
    assert reg["tail"]["heteroatom_fraction"] < 0.05
    assert reg["linker"]["heteroatom_fraction"] > 0.4
    assert reg["linker"]["heteroatom_fraction"] > reg["head"]["heteroatom_fraction"]
    # head is nitrogen-rich vs the (carbon) tail
    assert reg["head"]["element_fraction"].get("N", 0) > reg["tail"]["element_fraction"].get("N", 0)
