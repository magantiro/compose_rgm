"""Stage 6: local plumbing verification of the checkpoint path -- the B->B-edit compatible/head-widening
init, the save/reload round-trip through the REAL loader, and loud-fail on metadata mismatch.

The production B checkpoint is not available locally, so these exercise the actual code paths with a tiny
B-compatible fixture (NOT an unrelated random model): _compatible_checkpoint_initialization (the
shape-compatible body transfer + fresh wider heads) and load_factorized_rollout_checkpoint (reconstruct
the organic/editing model from metadata + load_state_dict). A successful fixture test supports
GO_FOR_PRODUCTION_PREFLIGHT, not GO_FOR_FULL_A100 -- the real B checkpoint is still unloaded locally.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from compose_v4.chem.molecular_graph import ORGANIC_VOCABULARY, smiles_to_molecular_graph
from compose_v4.chem.source_prior import DegreeBoundedCarbonTreePrior
from compose_v4.chem.state import pad_molecular_graph
from compose_v4.model.factorized_tracelet_rate_model import FactorizedTraceletRateModel
from compose_v4.rewrite.tree_transport import compile_carbon_tree_to_target
from compose_v4.rewrite.typed_ring_catalog import build_typed_ring_catalog
from evaluate_tracelet_rollouts import load_factorized_rollout_checkpoint  # noqa: E402
from train_tracelet_cnof_gate import _compatible_checkpoint_initialization  # noqa: E402

_SLOTS = 16


def _catalog():
    def trace(smi):
        t = pad_molecular_graph(smiles_to_molecular_graph(smi), _SLOTS)
        src = DegreeBoundedCarbonTreePrior(sizes=(t.n_real_atoms,)).sample(
            np.random.default_rng(1), n_slots=_SLOTS)
        return compile_carbon_tree_to_target(src, t, use_bond_reroute=True, align_source=True)
    return build_typed_ring_catalog(tuple(trace(s) for s in ("c1ccccc1", "C1CCNCC1")))


def _b_model(catalog):
    torch.manual_seed(0)
    return FactorizedTraceletRateModel(catalog, hidden_dim=16, message_passing_steps=1).eval()


def _b_edit_model(catalog):
    torch.manual_seed(1)
    return FactorizedTraceletRateModel(
        catalog, hidden_dim=16, message_passing_steps=1, enable_ring_restates=True,
        enable_cyclic_graft=True, enable_heteroatom_scan=True, enable_ring_opening=True,
        atom_vocabulary=ORGANIC_VOCABULARY).eval()


def _payload(model, catalog, *, organic, corrupted):
    return {
        "state_dict": model.state_dict(),
        "ring_catalog": catalog,
        "tree_source_prior": DegreeBoundedCarbonTreePrior(sizes=(4,)),
        "hidden_dim": 16,
        "message_passing_steps": 1,
        "training_backend": "factorized_marks",
        "source_prior": "carbon_tree",
        "organic_vocabulary": organic,
        "corrupted_prior_mix": corrupted,
    }


def test_compatible_init_transfers_body_and_retains_wider_heads() -> None:
    # B (CNOF, 4-wide heads) -> B-edit (organic, 15-wide heads): shape-compatible body/embeddings transfer;
    # the wider (element,valence) heads keep the target's fresh init (no crash on the 4-vs-15 mismatch).
    catalog = _catalog()
    b, b_edit = _b_model(catalog), _b_edit_model(catalog)
    initialized, transferred, retained = _compatible_checkpoint_initialization(
        b_edit.state_dict(), {"state_dict": b.state_dict()})
    assert transferred, "no shape-compatible tensors transferred from B"
    assert "atom_embedding.weight" in transferred  # raw 12-element embedding is shared
    # the widened (element,valence) head is retained (fresh), not transferred.
    assert any(n.startswith("grow_root_head.") for n in retained)
    b_edit.load_state_dict(initialized)
    assert torch.equal(b_edit.atom_embedding.weight, b.atom_embedding.weight)  # body copied from B
    assert b_edit.grow_root_head.weight.shape[0] == len(ORGANIC_VOCABULARY)  # stayed wide


def test_save_reload_roundtrip_is_exact(tmp_path) -> None:
    catalog = _catalog()
    model = _b_edit_model(catalog)
    path = tmp_path / "b_edit.pt"
    torch.save(_payload(model, catalog, organic=True, corrupted=True), path)
    reloaded, meta = load_factorized_rollout_checkpoint(path)

    # reconstruction restored the editing capabilities from metadata.
    assert len(reloaded.atom_vocabulary) == len(ORGANIC_VOCABULARY)
    assert reloaded.enable_ring_restates and reloaded.enable_cyclic_graft and reloaded.enable_ring_opening

    state = pad_molecular_graph(smiles_to_molecular_graph("Cc1ccncc1CC"), _SLOTS)
    # fixed-batch output equivalence: reload matches the original to machine precision.
    for m, r in ((model, reloaded),):
        a = m.sample_rewrite_mark(state, 0.5, np.random.default_rng(7))
        b = r.sample_rewrite_mark(state, 0.5, np.random.default_rng(7))
        assert a.rule_name == b.rule_name and np.isclose(a.total_hazard, b.total_hazard, rtol=1e-5)
    # a short fixed-seed editing trajectory is identical.
    from compose_v4.rewrite.kernel import canonical_state_key, de_novo_rewrite_system
    sysr = de_novo_rewrite_system()

    def rollout(mdl):
        rng = np.random.default_rng(3)
        s, keys = state, []
        for _ in range(4):
            mk = mdl.sample_rewrite_mark(s, 0.5, rng)
            if mk.action is None:
                break
            s = sysr.apply(s, mk.rule_name, mk.action)
            keys.append(canonical_state_key(s))
        return keys

    assert rollout(model) == rollout(reloaded)


def test_load_reconstructs_de_novo_b_byte_identically(tmp_path) -> None:
    # a de-novo B checkpoint (no organic flag, corrupted=False) reconstructs as CNOF + editing off.
    catalog = _catalog()
    model = _b_model(catalog)
    path = tmp_path / "b.pt"
    torch.save(_payload(model, catalog, organic=False, corrupted=False), path)
    reloaded, _ = load_factorized_rollout_checkpoint(path)
    assert len(reloaded.atom_vocabulary) == 4  # CNOF
    assert not reloaded.enable_ring_restates and not reloaded.enable_cyclic_graft


def test_load_fails_loudly_on_missing_metadata(tmp_path) -> None:
    catalog = _catalog()
    payload = _payload(_b_edit_model(catalog), catalog, organic=True, corrupted=True)
    del payload["ring_catalog"]  # a required rollout-metadata key
    path = tmp_path / "bad.pt"
    torch.save(payload, path)
    with pytest.raises(ValueError, match="rollout metadata"):
        load_factorized_rollout_checkpoint(path)
