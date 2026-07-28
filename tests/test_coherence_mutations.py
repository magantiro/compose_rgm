"""Program-coherence mutation tests (§17): prove the coherence machinery is SENSITIVE -- each injected
divergence must be caught, so a GO verdict is meaningful. Mutations are applied locally + reverted (no
production state touched)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts" / "coherence"))

import pytest  # noqa: E402


def test_contract_fingerprint_changes_when_a_field_mutates():
    """§2: changing ONE contract field must change the production_contract_fingerprint -- so an incompatible
    consumer emitting the old fingerprint fails loudly."""
    from program_contract import build_contract, contract_fingerprint

    base = build_contract()
    base_fp = contract_fingerprint(base)
    for field, mutated in [
        ("max_atoms", 48),
        ("enable_ring_grow_macro", True),      # re-enable the legacy macro
        ("enable_cycle_ops", False),           # disable compositional ring support
        ("denovo_keep", 1),                    # break the zero-mixture contract
        ("atom_vocabulary", "CNOF_VOCABULARY"),
        ("operator_registry_hash", "deadbeefdeadbeef"),
        ("eval_operator_capability_fingerprint", "0000000000000000"),
    ]:
        m = dict(base)
        m[field] = mutated
        assert contract_fingerprint(m) != base_fp, f"fingerprint insensitive to {field}"


def test_disabling_editing_capabilities_trips_the_teacher_invariant():
    """§5/§7: a validation batch built WITHOUT the editing capabilities (the eval-collator bug class) must
    trip the teacher-in-candidate invariant -- proving a silently-different molecular process is caught."""
    from compose_v4.chem.molecular_graph import ORGANIC_VOCABULARY
    from compose_v4.experiments.corrupted_source_prior import build_corrupted_prior_records
    from compose_v4.experiments.factorized_mark_conditional import (
        TeacherOutsideCandidatesError,
        sample_factorized_mark_batch,
    )
    from warmstart_dry_run import build_production_ring_catalog

    catalog = build_production_ring_catalog(40)
    records, _ = build_corrupted_prior_records(
        ["O=C1NC(=O)c2ccccc21", "c1ccc2c(c1)ccc1ccccc12", "C1CCNCC1C(=O)O"],
        n_slots=40, depth_max=5, seed=17, catalog=catalog, vocabulary=ORGANIC_VOCABULARY,
        couplings_per_target=2,
    )
    tripped = False
    for r in records:
        try:
            # MUTATION: omit capabilities -> de-novo defaults (editing families off)
            sample_factorized_mark_batch(
                (r,), batch_size=4, seed=0, late_time_fraction=0.5, operational_horizon=16.0,
                progress_stratification_fraction=0.5, use_aromatic_bond_view=True, workers=0,
                ring_catalog=catalog, ring_electronic_mode="factorized_local",
            )
        except TeacherOutsideCandidatesError:
            tripped = True
            break
    assert tripped, "omitting editing capabilities must trip the teacher-in-candidate invariant"


def test_ring_grow_macro_reenable_is_caught_or_observable():
    """§17: the 'silently re-enable the legacy ring_system_grow macro' divergence must be caught. Under the
    RingCore contract (cycle ops on) re-enabling grow is now rejected at CONSTRUCTION by guard G1 -- a stronger
    guarantee than sampling-observability (the divergent model cannot even be built). In the de-novo regime
    (cycle ops off) grow is legitimately samplable, and toggling it is observable in sampling -- so a sampler
    that silently re-enables grow cannot pass unnoticed either way."""
    from compose_v4.chem.molecular_graph import ORGANIC_VOCABULARY, smiles_to_molecular_graph
    from compose_v4.chem.state import pad_molecular_graph
    from compose_v4.model.factorized_tracelet_rate_model import FactorizedTraceletRateModel
    from warmstart_dry_run import build_production_ring_catalog

    catalog = build_production_ring_catalog(40)
    chain = pad_molecular_graph(smiles_to_molecular_graph("CCCCCCCC"), 40)  # saturated chain -> grow fires

    # RingCore regime: re-enabling grow alongside cycle ops is caught at construction (guard G1).
    with pytest.raises(ValueError, match="mutually exclusive"):
        FactorizedTraceletRateModel(
            catalog, hidden_dim=16, message_passing_steps=1, atom_vocabulary=ORGANIC_VOCABULARY,
            enable_cycle_ops=True, enable_ring_grow_macro=True,
        )

    # De-novo regime (cycle ops off): toggling grow is observable in sampling.
    draws = {}
    for enable in (False, True):
        torch.manual_seed(0)
        model = FactorizedTraceletRateModel(
            catalog, hidden_dim=16, message_passing_steps=1, atom_vocabulary=ORGANIC_VOCABULARY,
            enable_cycle_ops=False, enable_ring_grow_macro=enable,
        ).eval()
        rng = np.random.default_rng(0)
        draws[enable] = sum(
            model.sample_rewrite_mark(chain, 0.3, rng).rule_name == "ring_system_grow" for _ in range(200)
        )
    assert draws[False] == 0, "grow disabled must never sample ring_system_grow"
    assert draws[True] > 0, "the mutation (grow enabled) must be observable as grow draws"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
