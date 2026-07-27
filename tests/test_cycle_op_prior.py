"""P3 supervision: build_cycle_op_records produces real, round-trip-verified cycle_open/cycle_close teacher
records that are in-mask (finite GM loss) for the enable_cycle_ops model."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from compose_v4.chem.molecular_graph import smiles_to_molecular_graph  # noqa: E402
from compose_v4.chem.source_prior import DegreeBoundedCarbonTreePrior  # noqa: E402
from compose_v4.chem.state import pad_molecular_graph  # noqa: E402
from compose_v4.experiments.cycle_op_prior import build_cycle_op_records  # noqa: E402
from compose_v4.model.factorized_tracelet_rate_model import (  # noqa: E402
    FactorizedTraceletRateModel,
    prepare_factorized_mark_batch,
)
from compose_v4.model.time_convention import frozen_time  # noqa: E402
from compose_v4.rewrite.kernel import canonical_state_key, de_novo_rewrite_system  # noqa: E402
from compose_v4.rewrite.typed_ring_catalog import build_typed_ring_catalog  # noqa: E402

_MOLS = ["c1ccccc1CO", "c1ccc2ccccc2c1", "C1CCNCC1", "c1ccncc1C(=O)O", "O=C1CCCCC1", "C1CC2CCC1C2"]


def _catalog():
    def trace(smi):
        t = pad_molecular_graph(smiles_to_molecular_graph(smi), 40)
        src = DegreeBoundedCarbonTreePrior(sizes=(t.n_real_atoms,)).sample(
            np.random.default_rng(1), n_slots=40
        )
        from compose_v4.rewrite.tree_transport import compile_carbon_tree_to_target

        return compile_carbon_tree_to_target(src, t, use_bond_reroute=True, align_source=True)

    return build_typed_ring_catalog(tuple(trace(s) for s in ("c1ccccc1", "c1ccncc1")))


def test_build_cycle_op_records_both_directions():
    records, attempted = build_cycle_op_records(_MOLS, n_slots=40, seed=0)
    assert attempted > 0 and len(records) == 2 * attempted  # both directions per ring bond
    fams = [r.path.trace.steps[0].rule_name for r in records]
    assert set(fams) == {"bond_delete", "bond_insert"}
    assert fams.count("bond_delete") == fams.count("bond_insert")


def test_cycle_op_records_round_trip_through_executor():
    system = de_novo_rewrite_system()
    records, _ = build_cycle_op_records(_MOLS, n_slots=40, seed=1)
    for r in records:
        trace = r.path.trace
        state = trace.source
        for step in trace.steps:
            state = system.apply(state, step.rule_name, step.action)
        assert canonical_state_key(state) == canonical_state_key(trace.target)


def test_cycle_op_records_are_in_mask_finite():
    cat = _catalog()
    torch.manual_seed(0)
    model = FactorizedTraceletRateModel(
        cat, hidden_dim=16, message_passing_steps=1, enable_cycle_ops=True
    ).eval()
    records, _ = build_cycle_op_records(_MOLS, n_slots=40, seed=2)
    assert records
    for r in records:
        trace = r.path.trace
        step = trace.steps[0]
        batch = prepare_factorized_mark_batch(
            (trace.source,), (frozen_time(0.2),), (step.action,), (step.rule_name,), (1.0,),
            ring_catalog=cat,
        )
        with torch.no_grad():
            pred = model.forward_mark_batch(batch)
        assert bool(torch.isfinite(pred.selected_mark_log_probability).all())
