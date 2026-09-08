"""Identity-safe contracts on synthetic, exact-slot executor fixtures."""

from dataclasses import replace

import numpy as np
import pytest
from rdkit import Chem

from compose_v4.chem.molecular_graph import SCAR_IDX, smiles_to_molecular_graph
from compose_v4.chem.state import pad_molecular_graph
from compose_v4.control.macro_engine import (
    MACRO_FAMILIES,
    append_system_closure,
    contract_for,
    ring_systems,
    slot_ring_systems,
    state_contract_for,
)
from compose_v4.control.option_continuation import OptionContinuationKernel, OptionState
from compose_v4.control.region_rewrite import Lineage, RewriteContext
from compose_v4.rewrite.kernel import canonical_state_key, editing_v2_rewrite_system
from compose_v4.rewrite.operators import BondInsert


def graph(smiles):
    return pad_molecular_graph(smiles_to_molecular_graph(smiles), 48)


def permute(state, order):
    return replace(
        state,
        atom_types=state.atom_types[order],
        bonds=state.bonds[np.ix_(order, order)],
        formal_charges=state.formal_charges[order],
        implicit_h_counts=state.implicit_h_counts[order],
    )


@pytest.mark.parametrize("size", [3, 4, 5, 6, 7, 8])
def test_pendant_sizes_keep_frozen_minimum(size):
    source = graph("c1ccccc1" + "C" * size)
    product = editing_v2_rewrite_system().apply(source, "bond_insert", BondInsert(6, 5 + size, 1))
    assert state_contract_for("append_system", source)(product) == (size >= 6)


def test_canonical_smiles_renumbering_does_not_reject_pendant():
    source = graph("c1ccccc1CCCCCC")
    product = editing_v2_rewrite_system().apply(source, "bond_insert", BondInsert(6, 11, 1))
    assert not append_system_closure(canonical_state_key(source))(canonical_state_key(product))
    assert state_contract_for("append_system", source)(product)
    assert not state_contract_for("append_system", source)(source)
    # Both coordinate changes move real atoms into sparse slots. Never permute
    # only one member of a source/product pair or reconstruct either by SMILES.
    for order in (np.arange(47, -1, -1), np.random.default_rng(41).permutation(48)):
        assert state_contract_for("append_system", permute(source, order))(permute(product, order))


@pytest.mark.parametrize("anchor", [5, 4, 2], ids=["spiro", "fused", "bridged"])
def test_closure_into_old_ring_is_not_pendant(anchor):
    source = graph("C1CCCCC1CCCCC")
    product = editing_v2_rewrite_system().apply(source, "bond_insert", BondInsert(anchor, 10, 1))
    assert len(slot_ring_systems(product)) == 1
    assert not state_contract_for("append_system", source)(product)


@pytest.mark.parametrize(
    "smiles", ["CCC", "c1ccccc1C1CCCCC1", "C1CCC2(CC1)CCCCC2", "C1CC2CCC1C2", "c1ccc2ccccc2c1"]
)
def test_slot_systems_agree_with_rdkit_shared_atom_definition(smiles):
    source = graph(smiles)
    mol = Chem.MolFromSmiles(smiles)
    expected = {frozenset().union(*group) for group in ring_systems(mol)}
    assert set(slot_ring_systems(source)) == expected


def test_scar_and_null_are_not_ring_atoms():
    source = graph("C1CCCCC1")
    source.atom_types[0] = SCAR_IDX
    # Malformed chemistry intentionally tests only the element predicate, not validity.
    assert slot_ring_systems(source) == ()


def test_mismatched_slot_layout_fails_loudly():
    source = graph("CCCCCC")
    with pytest.raises(ValueError, match="persistent-slot"):
        state_contract_for("append_system", source)(smiles_to_molecular_graph("C1CCCCC1"))


@pytest.mark.parametrize("macro", sorted(set(MACRO_FAMILIES) - {"append_system"}))
def test_other_contracts_preserve_legacy_predicate(macro):
    source = graph("c1ccccc1")
    old = contract_for(macro, canonical_state_key(source))
    new = state_contract_for(macro, source)
    assert (old is None) == (new is None)
    if old is not None:
        for smiles in ("c1ccccc1", "Cc1ccccc1", "Fc1ccccc1", "c1ccc2ccccc2c1"):
            product = graph(smiles)
            assert new(product) == old(canonical_state_key(product))


@pytest.mark.parametrize(
    "option,step,horizon", [("append_system", 0, 1), ("build_ring_system", 8, 11)]
)
def test_kernel_uses_exact_contract_and_normalizes_only_survivors(option, step, horizon):
    source = graph("c1ccccc1CCCCCC")
    node = OptionState(
        source,
        source,
        RewriteContext(frozenset(), frozenset(range(12)), (), "pendant", 0),
        Lineage.initial(range(12)),
        option,
        step,
        horizon,
        "synthetic-append-bundle",
    )

    def law(_graph):
        return (
            ("bond_insert", "bond_insert"),
            (BondInsert(6, 8, 1), BondInsert(6, 11, 1)),
            (0.9, 0.1),
        )

    kernel = OptionContinuationKernel(law, editing_v2_rewrite_system(), max_executor_applications=2)
    row = kernel.row(node)
    assert len(row.successors) == 1 and row.probabilities == (1.0,)
    assert row.successors[0].step == step + 1
    assert row.successors[0].bundle_id == node.bundle_id
    assert len(slot_ring_systems(row.successors[0].graph)) == 2
    assert kernel.work.rejected_products == 1 and kernel.work.legal_products == 1
