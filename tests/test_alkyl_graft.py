"""AlkylGraft / AlkylPrune: verify the chain-continuation rewrite mechanics."""
from rdkit import Chem

from compose_v4.chem.molecular_graph import (
    BOND_DOUBLE, BOND_SINGLE, molecular_graph_to_smiles, smiles_to_molecular_graph,
)
from compose_v4.chem.state import pad_molecular_graph
from compose_v4.rewrite.alkyl_graft import (
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
    from compose_v4.rewrite.alkyl_graft import AlkylGraft
    from compose_v4.rewrite.tracelets import AtomPayload
    from compose_v4.chem.molecular_graph import ELEMENT_TO_IDX
    n, padded = _pad("CCC", 4)
    bad = AlkylGraft(anchor=0,
                     atoms=(AtomPayload(slot=n, atom_type=ELEMENT_TO_IDX["N"], formal_charge=0, implicit_h_count=2),),
                     bond_orders=(), attachment_order=BOND_SINGLE)
    assert not is_valid_alkyl_graft(padded, bad)  # typing constraint: carbon only


def test_kernel_registered_apply_and_inverse():
    from compose_v4.rewrite.kernel import de_novo_rewrite_system
    from compose_v4.rewrite.trace import RewriteStep, inverse_step
    n, padded = _pad("CCC", 12)
    graft = build_graft(anchor=0, bond_orders=[1] * 5, null_slots=range(n, n + 6))
    rt = de_novo_rewrite_system()
    grafted = rt.apply(padded, "alkyl_graft", graft)  # official runtime path
    assert Chem.CanonSmiles(molecular_graph_to_smiles(grafted)) == Chem.CanonSmiles("CCCCCCCCC")  # C3+C6=C9
    # graft -> prune inverse, applied via runtime, returns to source
    prune_step = inverse_step(padded, RewriteStep("alkyl_graft", graft))
    assert prune_step.rule_name == "alkyl_prune"
    back = rt.apply(grafted, prune_step.rule_name, prune_step.action)
    assert Chem.CanonSmiles(molecular_graph_to_smiles(back)) == Chem.CanonSmiles("CCC")
    # prune -> graft inverse reconstructs the chain from the grafted source
    regraft = inverse_step(grafted, prune_step)
    assert regraft.rule_name == "alkyl_graft"
    assert Chem.CanonSmiles(molecular_graph_to_smiles(rt.apply(padded, "alkyl_graft", regraft.action))) == Chem.CanonSmiles("CCCCCCCCC")


def test_model_samples_and_applies_alkyl_graft():
    import numpy as np
    import torch
    from compose_v4.model.factorized_tracelet_rate_model import (
        FactorizedTraceletRateModel, MARK_RULE_NAMES,
    )
    from compose_v4.rewrite.typed_ring_catalog import TypedRingCatalog
    from compose_v4.rewrite.kernel import de_novo_rewrite_system
    assert MARK_RULE_NAMES[-1] == "alkyl_graft" and len(MARK_RULE_NAMES) == 11
    torch.manual_seed(0)
    model = FactorizedTraceletRateModel(TypedRingCatalog((), (), ()), hidden_dim=16, message_passing_steps=1).eval()
    state = pad_molecular_graph(smiles_to_molecular_graph("CCCC"), 24)
    rt = de_novo_rewrite_system()
    rng = np.random.default_rng(0)
    n_alkyl = n_ok = 0
    for _ in range(300):
        mark = model.sample_rewrite_mark(state, 0.5, rng)
        if mark.rule_name == "alkyl_graft":
            n_alkyl += 1
            if molecular_graph_to_smiles(rt.apply(state, "alkyl_graft", mark.action)):
                n_ok += 1
    assert n_alkyl > 0, "alkyl_graft never sampled"
    assert n_ok == n_alkyl, "sampled alkyl_graft failed to apply"


def test_teacher_collapse_reconstructs_and_grafts():
    from compose_v4.rewrite.tracelet_compiler import compile_null_to_target_tracelets
    from compose_v4.rewrite.kernel import default_rewrite_system
    from compose_v4.rewrite.alkyl_teacher import collapse_alkyl_runs
    rt = default_rewrite_system()
    smi = "CCCCCCCCCCC(CCCCCCCC)OC(=O)CCN(CCCN(C)C)CCC(=O)OCC(CCCCCC)CCCCCCCC"  # RM-60 skeleton
    trace = compile_null_to_target_tracelets(smiles_to_molecular_graph(smi), system=rt)
    col = collapse_alkyl_runs(trace, rt)
    assert col is not trace, "collapse did not apply"
    assert sum(1 for s in col.steps if s.rule_name == "alkyl_graft") >= 3
    assert len(col.steps) < len(trace.steps)

    def recon(t):
        st = t.source
        for s in t.steps:
            st = rt.apply(st, s.rule_name, s.action)
        return Chem.CanonSmiles(molecular_graph_to_smiles(st))

    assert recon(col) == recon(trace)  # graph-identical to the original teacher program


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); print(f"PASS {name}")
