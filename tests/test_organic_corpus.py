"""Broad-organic corpus scope: membership, charge categories, census, and leakage-free split.

The scope decides which molecules B-edit trains/mines on. It must (a) admit broad organic elements and
representable charged states, (b) reject out-of-scope molecules with the correct COUNTED reason, (c)
distinguish the three charge categories, and (d) split TRAIN-only with no val/test leakage. The scope_hash
must bind the exact active vocabulary so a cross-scope load can be caught.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from compose_v4.data.organic_corpus import (
    BROAD_ORGANIC_NEUTRAL_V1,
    BROAD_ORGANIC_V1,
    REJECT_DISCONNECTED,
    REJECT_TOO_BIG,
    REJECT_UNSUPPORTED_ELEMENT,
    classify_smiles,
    load_organic_corpus_split,
    scan_corpus,
)

_LOCAL_CORPUS = Path("results/tree_fcd_transfer_stage1_factorized_v1/guacamol_heldout_val_5000_seed0.smiles")


def test_broad_scope_admits_organic_elements_and_charges() -> None:
    for smi in ["Cn1cnc2c1c(=O)n(C)c(=O)n2C", "CSc1ccccc1", "CS(=O)(=O)N", "Clc1ccccc1",
                "Brc1ccccc1", "Ic1ccccc1", "CCOP(=O)(OCC)C", "C[NH3+]", "CC(=O)[O-]",
                "C(C(=O)[O-])[NH3+]"]:
        ok, reason = classify_smiles(smi, BROAD_ORGANIC_V1)
        assert ok, f"{smi} should be in broad scope, got {reason}"


def test_neutral_scope_rejects_charges_but_keeps_broad_elements() -> None:
    # broad elements stay in; charged states are the only difference vs the production scope.
    assert classify_smiles("CSc1ccccc1", BROAD_ORGANIC_NEUTRAL_V1)[0]
    assert not classify_smiles("C[NH3+]", BROAD_ORGANIC_NEUTRAL_V1)[0]
    assert classify_smiles("C[NH3+]", BROAD_ORGANIC_V1)[0]


def test_out_of_scope_rejections_are_correctly_reasoned() -> None:
    assert classify_smiles("[Na+].CC(=O)[O-]", BROAD_ORGANIC_V1) == (False, REJECT_DISCONNECTED)
    assert classify_smiles("C[Si](C)(C)C", BROAD_ORGANIC_V1)[1] in (REJECT_UNSUPPORTED_ELEMENT, "unparseable")
    big = "C" * 60  # exceeds max_atoms=48
    assert classify_smiles(big, BROAD_ORGANIC_V1) == (False, REJECT_TOO_BIG)


def test_census_distinguishes_the_three_charge_categories() -> None:
    accepted, census = scan_corpus(
        ["C[NH3+]", "C(C(=O)[O-])[NH3+]", "CCO", "CSc1ccccc1", "[Na+].[Cl-]"], BROAD_ORGANIC_V1)
    cat = census["charge_category"]
    assert cat.get("nonzero_net_charge", 0) == 1      # C[NH3+]
    assert cat.get("zwitterion_net_zero", 0) == 1     # glycine zwitterion, net 0 but charged
    assert cat.get("neutral", 0) == 2                 # CCO, thioanisole
    assert census["contains_charged_atom"] == 2
    assert census["element_molecule_count"]["S"] == 1


def test_scope_hash_binds_vocabulary_and_charge_policy() -> None:
    assert BROAD_ORGANIC_V1.scope_hash() != BROAD_ORGANIC_NEUTRAL_V1.scope_hash()
    # stable across calls (deterministic)
    assert BROAD_ORGANIC_V1.scope_hash() == BROAD_ORGANIC_V1.scope_hash()
    assert "S" in BROAD_ORGANIC_V1.element_symbols() and "P" in BROAD_ORGANIC_V1.element_symbols()


@pytest.mark.skipif(not _LOCAL_CORPUS.exists(), reason="local corpus fixture absent")
def test_split_is_leakage_free_and_broad() -> None:
    sp = load_organic_corpus_split(_LOCAL_CORPUS, scope=BROAD_ORGANIC_V1,
                                   train_size=800, validation_size=100, test_size=100)
    assert len(sp.train) == 800 and len(sp.validation) == 100 and len(sp.test) == 100
    train = set(sp.train)
    assert not (train & set(sp.validation)) and not (train & set(sp.test))
    assert not (set(sp.validation) & set(sp.test))
    assert sp.scope["scope_hash"] == BROAD_ORGANIC_V1.scope_hash()
    assert sp.census["retained_fraction"] > 0.9  # broad scope keeps ~98%


def test_load_fails_loudly_when_under_requested() -> None:
    with pytest.raises(ValueError, match="eligible"):
        load_organic_corpus_split(_LOCAL_CORPUS, scope=BROAD_ORGANIC_V1,
                                  train_size=10_000_000, validation_size=1, test_size=1)
