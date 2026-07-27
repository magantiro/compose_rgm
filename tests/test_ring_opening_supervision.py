"""Stage 5.6: clean ring-opening (`ring_system_delete`) must have real POSITIVE training supervision when
`enable_ring_opening` is on, using the SAME versioned ring catalog the rewrite system / enumerator /
executor / sampler use (never an ad hoc ring-perception).

The corruption generator emits `ring_system_delete` only when a ring catalog is threaded in
(`source_corruption.corrupt_to_source`: `if catalog is not None: enumerate_clean_ring_system_deletes`),
and the trainer passes `catalog=ring_catalog` at its corruption call site. These fixtures pin, for the
ring systems the operator actually supports (aromatic carbocycle, aromatic heterocycle, saturated ring,
heterocycle, fused/bicyclic) and a not-openable acyclic negative:

  (1) the ring-open action is in the legal (enumerable) set; (2) the corruption can select it and the
  saved trace contains it; (3) trace replay through the real executor succeeds; (4) every intermediate is
  valid + connected; (5) the clean delete preserves ring decoration (heteroatoms); (6) the inverse re-grows
  to the recorded target; and (7) a `ring_system_delete` POSITIVE teacher mark yields a finite loss and a
  finite, NONZERO gradient on the ring-delete head (positive supervision, not a suppressive denominator
  gradient).
"""
from __future__ import annotations

import numpy as np
import torch

from compose_v4.chem.molecular_graph import (
    ORGANIC_VOCABULARY,
    molecular_graph_to_smiles,
    smiles_to_molecular_graph,
)
from compose_v4.chem.source_prior import DegreeBoundedCarbonTreePrior
from compose_v4.chem.state import is_connected_or_null, is_valid_state, pad_molecular_graph
from compose_v4.model.factorized_tracelet_rate_model import (
    FactorizedTraceletRateModel,
    factorized_mark_bregman_loss,
    prepare_factorized_mark_batch,
)
from compose_v4.rewrite.kernel import canonical_state_key, de_novo_rewrite_system
from compose_v4.rewrite.ring_system_fiber import enumerate_clean_ring_system_deletes
from compose_v4.rewrite.source_corruption import make_edit_pair
from compose_v4.rewrite.tree_transport import compile_carbon_tree_to_target
from compose_v4.rewrite.trace import RewriteStep, execute_trace, inverse_step
from compose_v4.rewrite.typed_ring_catalog import build_typed_ring_catalog

_SLOTS = 24
_SYSTEM = de_novo_rewrite_system()
_RING_SMILES = ("c1ccccc1", "C1CCCCC1", "C1CCNCC1", "c1ccncc1", "c1ccc2ccccc2c1")


def _catalog():
    def trace(smi):
        target = pad_molecular_graph(smiles_to_molecular_graph(smi), _SLOTS)
        source = DegreeBoundedCarbonTreePrior(sizes=(target.n_real_atoms,)).sample(
            np.random.default_rng(1), n_slots=_SLOTS
        )
        return compile_carbon_tree_to_target(source, target, use_bond_reroute=True, align_source=True)

    return build_typed_ring_catalog(tuple(trace(s) for s in _RING_SMILES))


def _state(smi):
    return pad_molecular_graph(smiles_to_molecular_graph(smi), _SLOTS)


def test_clean_ring_open_in_legal_set_for_supported_ring_systems() -> None:
    # (1) legal-set membership: every supported ring system has >= 1 clean ring-open; acyclic has none.
    catalog = _catalog()
    for smi in _RING_SMILES:
        assert len(enumerate_clean_ring_system_deletes(_state(smi), catalog)) >= 1, smi
    assert len(enumerate_clean_ring_system_deletes(_state("CC(C)CC"), catalog)) == 0  # not openable


def _find_ring_delete_trace(catalog, *, tries=400):
    """A corruption TRIM trace (real->corrupt) whose steps include a ring_system_delete, with a catalog."""
    rng = np.random.default_rng(0)
    for _ in range(tries):
        smi = str(rng.choice(_RING_SMILES))
        state = _state(smi)
        trim, _ = make_edit_pair(
            state, int(rng.integers(1, 6)), system=_SYSTEM, rng=rng,
            catalog=catalog, vocabulary=ORGANIC_VOCABULARY,
        )
        if trim and any(s.rule_name == "ring_system_delete" for s in trim.steps):
            return smi, trim
    return None, None


def test_corruption_emits_ring_open_and_replays_valid() -> None:
    # (2) corruption selects it + (3) saved trace contains it + (4) replay valid+connected + replay==target.
    catalog = _catalog()
    _, trim = _find_ring_delete_trace(catalog)
    assert trim is not None, "corruption never produced a ring_system_delete with a catalog"
    assert any(s.rule_name == "ring_system_delete" for s in trim.steps)
    state = trim.source
    for step in trim.steps:
        state = _SYSTEM.apply(state, step.rule_name, step.action)
        assert is_valid_state(state) and is_connected_or_null(state)
    assert canonical_state_key(state) == canonical_state_key(trim.target)


def _element_multiset(state):
    real = state.atom_types[state.atom_types > 0]
    return sorted(int(a) for a in real)


def test_clean_ring_open_preserves_ring_heteroatoms() -> None:
    # (5) decoration-preserving: opening a heterocycle keeps its N/O (clean delete, not carbon-izing).
    catalog = _catalog()
    for smi, keep in (("C1CCNCC1", "N"), ("c1ccncc1", "N"), ("C1CCOCC1", "O")):
        state = _state(smi)
        deletes = enumerate_clean_ring_system_deletes(state, catalog)
        assert deletes, f"no clean ring-open enumerated for {smi}"
        opened = _SYSTEM.apply(state, "ring_system_delete", deletes[0])
        assert is_valid_state(opened) and is_connected_or_null(opened)
        # element multiset is preserved (only a ring bond opens; H re-derived) -> heteroatom kept.
        assert _element_multiset(opened) == _element_multiset(state)
        assert keep in (molecular_graph_to_smiles(opened) or "").upper()


def test_ring_open_inverse_round_trips() -> None:
    # (6) the inverse of each corruption step replays back to the recorded target (both directions exist).
    catalog = _catalog()
    _, trim = _find_ring_delete_trace(catalog)
    assert trim is not None
    states = [trim.source]
    for step in trim.steps:
        states.append(_SYSTEM.apply(states[-1], step.rule_name, step.action))
    inv = tuple(
        inverse_step(states[i], RewriteStep(step.rule_name, step.action))
        for i, step in reversed(list(enumerate(trim.steps)))
    )
    rebuilt = execute_trace(states[-1], inv, system=_SYSTEM)
    assert canonical_state_key(rebuilt) == canonical_state_key(trim.source)


def test_ring_open_positive_teacher_is_learned_not_suppressed() -> None:
    # (7) POSITIVE supervision (not a suppressive denominator gradient): a ring_system_delete teacher mark
    # is finite, the family head receives a finite nonzero gradient, and a short overfit DECREASES the
    # loss while INCREASING the selected ring-open mark's probability -- i.e. the family is a real target.
    catalog = _catalog()
    _, trim = _find_ring_delete_trace(catalog)
    assert trim is not None
    state = trim.source
    teacher_state = teacher_action = None
    for step in trim.steps:
        if step.rule_name == "ring_system_delete":
            teacher_state, teacher_action = state, step.action
            break
        state = _SYSTEM.apply(state, step.rule_name, step.action)
    assert teacher_action is not None

    torch.manual_seed(0)
    model = FactorizedTraceletRateModel(
        catalog, hidden_dim=16, message_passing_steps=1,
        enable_ring_opening=True, atom_vocabulary=ORGANIC_VOCABULARY,
    ).train()
    batch = prepare_factorized_mark_batch(
        (teacher_state,), (0.5,), (teacher_action,), ("ring_system_delete",), (1.0,),
        ring_catalog=catalog, compute_ring_opening=True,
    )

    def selected_logp():
        with torch.no_grad():
            return float(model.forward_mark_batch(batch).selected_mark_log_probability[0])

    assert np.isfinite(selected_logp())  # legal teacher (finite), not -inf
    logp_start = selected_logp()

    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    first_loss = last_loss = None
    family_grad = 0.0
    for step_i in range(40):
        opt.zero_grad()
        loss = factorized_mark_bregman_loss(model.forward_mark_batch(batch), batch)
        assert torch.isfinite(loss)
        loss.backward()
        if step_i == 0:
            family_grad = sum(
                float(p.grad.abs().sum()) for p in model.family_head.parameters() if p.grad is not None
            )
            first_loss = float(loss)
        opt.step()
        last_loss = float(loss)

    assert np.isfinite(family_grad) and family_grad > 0.0, "family head got no positive-target gradient"
    assert last_loss < first_loss, f"overfit did not reduce loss ({first_loss} -> {last_loss})"
    assert selected_logp() > logp_start, "ring-open teacher probability did not increase (suppressed?)"
