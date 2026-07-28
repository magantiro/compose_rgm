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
from compose_v4.experiments.factorized_mark_conditional import (  # noqa: E402
    FactorizedMarkCollator,
    FactorizedMarkDataset,
    factorized_mark_bregman_loss,
)
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


def test_cycle_op_records_train_through_the_dataset_collator_path():
    """Regression: the ACTUAL training path (FactorizedMarkDataset.__getitem__ -> collator -> bregman) must
    accept cycle-op teacher records, whose rule_names are EXECUTOR names (bond_insert/bond_delete). The
    prepare_factorized_mark_batch tests bypass the dataset, so this path was previously unexercised and
    raised 'unsupported dense family: bond_delete' -- a Modal-training crash. Grow disabled (RingCore)."""
    cat = _catalog()
    records, _ = build_cycle_op_records(_MOLS, n_slots=40, seed=3)
    assert records
    dataset = FactorizedMarkDataset(
        records, start_index=0, length=24, seed=0, ring_catalog=cat,
        late_time_fraction=0.0, operational_horizon=16.0, progress_stratification_fraction=0.0,
    )
    examples = [dataset[i] for i in range(24)]  # must not raise on bond_insert/bond_delete
    teacher_names = {e.teacher_rule_name for e in examples if e.teacher_rule_name}
    assert teacher_names & {"bond_insert", "bond_delete"}
    collator = FactorizedMarkCollator(True, cat, compute_ring_grow_support=False)
    batch = collator(examples)
    assert batch.ring_grow_support_mask is None  # grow enumeration skipped
    torch.manual_seed(0)
    model = FactorizedTraceletRateModel(
        cat, hidden_dim=16, message_passing_steps=1,
        enable_cycle_ops=True, enable_ring_grow_macro=False,
    )
    loss = factorized_mark_bregman_loss(model.forward_mark_batch(batch), batch)
    loss.backward()
    assert bool(torch.isfinite(loss))
    grad_norm = sum(
        float(p.grad.norm()) ** 2 for p in model.parameters() if p.grad is not None
    ) ** 0.5
    assert grad_norm > 0.0


def test_cycle_op_records_are_in_mask_finite():
    cat = _catalog()
    torch.manual_seed(0)
    model = FactorizedTraceletRateModel(
        cat, hidden_dim=16, message_passing_steps=1,
        enable_cycle_ops=True, enable_ring_grow_macro=False,  # RingCore: cycle ops replace grow (guard G1)
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
