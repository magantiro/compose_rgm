"""A connected-nonleaf deletion must be scorable and learnable, not merely masked.

Exposing a candidate in the dense mask is necessary but not sufficient. The
repository has twice shipped an editing family whose histogram and replay checks
passed while the teacher scored `-inf` and the loss was `+inf`, because the mask
the teacher was scored against did not support the teacher's coordinate
(learnings 2026-07-24 and 2026-07-28: always training-smoke a new teacher source
before trusting it). `assert_teachers_in_exact_candidates` cannot catch this for
`atom_delete`: it only checks families carrying an explicit dynamic candidate
list, and per-coordinate families are validated by their masks at scoring time,
where an unsupported coordinate yields a silent `-inf` rather than an exception.

So this scores real connected-nonleaf `atom_delete` teachers end to end and
requires a finite log-probability, a finite Bregman loss, and finite gradients.
The Process-V1 model is scored on the same teachers as the negative control: it
must be `-inf`/`+inf`, which is precisely the support gap Process V2 closes.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from compose_v4.chem.molecular_graph import (
    ORGANIC_VOCABULARY,
    MolecularGraph,
    smiles_to_molecular_graph,
)
from compose_v4.chem.source_prior import DegreeBoundedCarbonTreePrior
from compose_v4.chem.state import pad_molecular_graph
from compose_v4.model.factorized_tracelet_rate_model import (
    LEGACY_ATOM_DELETE_ACTION_SEMANTICS,
    PROCESS_V2_ATOM_DELETE_ACTION_SEMANTICS,
    PROCESS_V2_EDITING_PROCESS_SEMANTICS,
    SEMANTIC_ATOM_RESTATE_ACTION_SEMANTICS,
    SEMANTIC_CYCLE_CLOSE_ACTION_SEMANTICS,
    SEMANTIC_CYCLE_OPEN_ACTION_SEMANTICS,
    SEMANTIC_EDITING_V2_PROCESS_SEMANTICS,
    SEMANTIC_RING_RESTATE_SCORER_MODE,
    FactorizedTraceletRateModel,
    factorized_mark_bregman_loss,
    prepare_factorized_mark_batch,
)
from compose_v4.rewrite.operators import AtomDelete, is_valid_atom_delete
from compose_v4.rewrite.process_v2_atom_delete import (
    process_v2_connected_nonleaf_atom_delete_mask,
)
from compose_v4.rewrite.tree_transport import compile_carbon_tree_to_target
from compose_v4.rewrite.typed_ring_catalog import build_typed_ring_catalog

_SLOTS = 24
_PANEL = (
    "C1CCCCC1",
    "C1CCOCC1",
    "CC1CCCCC1",
    "C1CC2CCC1CC2",
    "O=C1NC(O)C2CCCCC12",
    "C1CCSCC1",
    "c1ccc2c(c1)CCCC2",
    "ClC1CCCCC1",
)


def _state(smiles: str) -> MolecularGraph:
    return pad_molecular_graph(smiles_to_molecular_graph(smiles), _SLOTS)


@pytest.fixture(scope="module")
def ring_catalog():
    target = _state("c1ccccc1")
    source = DegreeBoundedCarbonTreePrior(sizes=(target.n_real_atoms,)).sample(
        np.random.default_rng(1), n_slots=_SLOTS
    )
    trace = compile_carbon_tree_to_target(
        source, target, use_bond_reroute=True, align_source=True
    )
    return build_typed_ring_catalog((trace,))


def _build_model(ring_catalog, *, process: str, delete_mode: str):
    torch.manual_seed(11)
    return FactorizedTraceletRateModel(
        ring_catalog,
        hidden_dim=12,
        message_passing_steps=1,
        enable_ring_restates=True,
        enable_heteroatom_scan=True,
        enable_cycle_ops=True,
        editing_process_semantics=process,
        atom_restate_action_semantics=SEMANTIC_ATOM_RESTATE_ACTION_SEMANTICS,
        ring_restate_scorer_mode=SEMANTIC_RING_RESTATE_SCORER_MODE,
        cycle_close_action_semantics=SEMANTIC_CYCLE_CLOSE_ACTION_SEMANTICS,
        cycle_open_action_semantics=SEMANTIC_CYCLE_OPEN_ACTION_SEMANTICS,
        atom_delete_action_semantics=delete_mode,
        enable_ring_grow_macro=False,
        enable_ring_system_delete=False,
        atom_vocabulary=ORGANIC_VOCABULARY,
    )


@pytest.fixture(scope="module")
def connected_nonleaf_teachers():
    """One executor-valid connected-nonleaf deletion per panel molecule."""

    rows = []
    for smiles in _PANEL:
        state = _state(smiles)
        admitted = np.flatnonzero(process_v2_connected_nonleaf_atom_delete_mask(state))
        if not admitted.size:
            continue
        action = AtomDelete(int(admitted[0]))
        assert is_valid_atom_delete(state, action)
        rows.append((state, action))
    assert len(rows) >= 6, "panel must supply several connected-nonleaf teachers"
    return tuple(rows)


def _teacher_batch(model, rows):
    capabilities = model.operator_capabilities
    states = tuple(state for state, _ in rows)
    return prepare_factorized_mark_batch(
        states,
        tuple(0.3 for _ in rows),
        tuple(action for _, action in rows),
        tuple("atom_delete" for _ in rows),
        tuple(1.0 for _ in rows),
        ring_catalog=model.ring_catalog,
        compute_ring_grow_support=capabilities.compute_ring_grow_support,
        compute_ring_restates=capabilities.compute_ring_restates,
        compute_cyclic_graft=capabilities.compute_cyclic_graft,
        compute_ring_opening=capabilities.compute_ring_opening,
        compute_ring_system_delete=capabilities.compute_ring_system_delete,
        editing_process_semantics=capabilities.editing_process_semantics,
        atom_restate_action_semantics=capabilities.atom_restate_action_semantics,
        ring_restate_scorer_mode=capabilities.ring_restate_scorer_mode,
        cycle_close_action_semantics=capabilities.cycle_close_action_semantics,
        cycle_open_action_semantics=capabilities.cycle_open_action_semantics,
        atom_delete_action_semantics=capabilities.atom_delete_action_semantics,
    )


def test_connected_nonleaf_teacher_scores_finite_and_backpropagates(
    ring_catalog, connected_nonleaf_teachers
) -> None:
    model = _build_model(
        ring_catalog,
        process=PROCESS_V2_EDITING_PROCESS_SEMANTICS,
        delete_mode=PROCESS_V2_ATOM_DELETE_ACTION_SEMANTICS,
    )
    batch = _teacher_batch(model, connected_nonleaf_teachers)
    outputs = model.forward_mark_batch(batch)

    log_probabilities = outputs.selected_mark_log_probability
    assert torch.isfinite(log_probabilities).all(), log_probabilities

    loss = factorized_mark_bregman_loss(outputs, batch)
    assert torch.isfinite(loss), loss

    model.zero_grad(set_to_none=True)
    loss.backward()
    gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.grad is not None
    ]
    assert gradients, "the teacher produced no gradient at all"
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
    assert any(bool(gradient.abs().sum() > 0) for gradient in gradients)


def test_the_same_teachers_are_unsupported_under_process_v1(
    ring_catalog, connected_nonleaf_teachers
) -> None:
    """The negative control: this is the support gap Process V2 exists to close.

    Without it, a finite loss above would not be evidence of anything new.
    """

    model = _build_model(
        ring_catalog,
        process=SEMANTIC_EDITING_V2_PROCESS_SEMANTICS,
        delete_mode=LEGACY_ATOM_DELETE_ACTION_SEMANTICS,
    )
    batch = _teacher_batch(model, connected_nonleaf_teachers)
    outputs = model.forward_mark_batch(batch)

    assert not torch.isfinite(outputs.selected_mark_log_probability).any()
    assert not torch.isfinite(factorized_mark_bregman_loss(outputs, batch))
