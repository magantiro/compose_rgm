"""AlkylGraft / AlkylPrune: verify the chain-continuation rewrite mechanics."""
from rdkit import Chem

from compose_v4.chem.molecular_graph import (
    BOND_DOUBLE, BOND_SINGLE, molecular_graph_to_smiles, smiles_to_molecular_graph,
)
from compose_v4.chem.state import pad_molecular_graph
from compose_v4.lipids.alkyl_graft import (
    AlkylPrune, apply_alkyl_graft, apply_alkyl_prune, build_graft,
    is_valid_alkyl_graft, is_valid_alkyl_prune,
)


def _pad(smi, extra):
    g = smiles_to_molecular_graph(smi)
    return g.n_atoms, pad_molecular_graph(g, g.n_atoms + extra)


def test_saturated_graft_extends_chain():
    n, padded = _pad("CCC", 12)  # propane; anchor = terminal C0
    graft = build_graft(anchor=0, bond_orders=[BOND_SINGLE] * 7, null_slots=range(n, n + 8))
    assert is_valid_alkyl_graft(padded, graft)
    smi = molecular_graph_to_smiles(apply_alkyl_graft(padded, graft))
    assert Chem.CanonSmiles(smi) == Chem.CanonSmiles("CCCCCCCCCCC")  # C11 linear


def test_unsaturated_graft_makes_one_double_bond():
    n, padded = _pad("CCC", 12)
    graft = build_graft(anchor=0, bond_orders=[1, 1, 1, BOND_DOUBLE, 1, 1, 1], null_slots=range(n, n + 8))
    assert is_valid_alkyl_graft(padded, graft)
    m = Chem.MolFromSmiles(molecular_graph_to_smiles(apply_alkyl_graft(padded, graft)))
    assert m is not None
    assert sum(1 for b in m.GetBonds() if b.GetBondTypeAsDouble() == 2.0) == 1


def test_branch_graft_midchain():
    n, padded = _pad("CCCCCC", 12)  # hexane; anchor = interior C2
    graft = build_graft(anchor=2, bond_orders=[1, 1, 1], null_slots=range(n, n + 4))
    assert is_valid_alkyl_graft(padded, graft)
    m = Chem.MolFromSmiles(molecular_graph_to_smiles(apply_alkyl_graft(padded, graft)))
    assert m is not None and m.GetNumHeavyAtoms() == 10  # 6 + 4 branch


def test_graft_prune_roundtrip():
    n, padded = _pad("CCC", 12)
    graft = build_graft(anchor=0, bond_orders=[1] * 5, null_slots=range(n, n + 6))
    grafted = apply_alkyl_graft(padded, graft)
    prune = AlkylPrune(anchor=0, slots=tuple(range(n, n + 6)))
    assert is_valid_alkyl_prune(grafted, prune)
    back = apply_alkyl_prune(grafted, prune)
    assert Chem.CanonSmiles(molecular_graph_to_smiles(back)) == Chem.CanonSmiles("CCC")


def test_graft_rejects_non_carbon():
    from compose_v4.lipids.alkyl_graft import AlkylGraft
    from compose_v4.rewrite.tracelets import AtomPayload
    from compose_v4.chem.molecular_graph import ELEMENT_TO_IDX
    n, padded = _pad("CCC", 4)
    bad = AlkylGraft(anchor=0,
                     atoms=(AtomPayload(slot=n, atom_type=ELEMENT_TO_IDX["N"], formal_charge=0, implicit_h_count=2),),
                     bond_orders=(), attachment_order=BOND_SINGLE)
    assert not is_valid_alkyl_graft(padded, bad)  # typing constraint: carbon only


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); print(f"PASS {name}")
