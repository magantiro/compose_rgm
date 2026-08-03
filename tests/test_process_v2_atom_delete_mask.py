"""The learned ``atom_delete`` candidate mask under Process V2.

Under Process V2 the batch's ``atom_delete_mask`` **equals**
``process_v2_atom_delete_mask``.  It is not a union with the legacy dense rule,
and it is neither a superset nor a subset of that rule: it adds the
connected-nonleaf fiber and removes every inherited candidate that fails a
common gate.

These tests pin that equality end to end, that the legacy branch is
byte-identical, that the mask agrees with an independently derived legality
oracle in both directions, that the other seven Active8 tables do not move, and
that a mixed V1/V2 batch or capability fails loudly.  The forward guard is
checked for EQUALITY rather than containment, because a containment check
accepts exactly the union this round removes and therefore could not detect the
defect it exists to guard.
"""

from __future__ import annotations

from dataclasses import fields as dataclass_fields, replace

import networkx as nx
import numpy as np
import pytest
import torch

from compose_v4.chem.aromaticity import resonance_invariant_bond_classes
from compose_v4.chem.graph_primitives import compute_topology_features
from compose_v4.chem.molecular_graph import (
    BOND_AROMATIC,
    ORGANIC_VOCABULARY,
    SCAR_IDX,
    MolecularGraph,
    is_element,
    smiles_to_molecular_graph,
)
from compose_v4.chem.source_prior import DegreeBoundedCarbonTreePrior
from compose_v4.chem.state import is_connected_or_null, pad_molecular_graph
from compose_v4.data.charge_policy import charge_policy_preserved
from compose_v4.experiments.cnof_conditional import PathRecord
from compose_v4.experiments.factorized_mark_conditional import (
    FactorizedMarkCollator,
    FactorizedMarkExample,
    _concatenate_factorized_mark_batches,
    sample_factorized_mark_batch,
)
from compose_v4.model.factorized_tracelet_rate_model import (
    ATOM_DELETE_ACTION_SEMANTICS,
    LEGACY_ATOM_DELETE_ACTION_SEMANTICS,
    LEGACY_ATOM_RESTATE_ACTION_SEMANTICS,
    LEGACY_CYCLE_CLOSE_ACTION_SEMANTICS,
    LEGACY_CYCLE_OPEN_ACTION_SEMANTICS,
    LEGACY_EDITING_PROCESS_SEMANTICS,
    LEGACY_RING_RESTATE_SCORER_MODE,
    PROCESS_V2_ATOM_DELETE_ACTION_SEMANTICS,
    PROCESS_V2_EDITING_PROCESS_SEMANTICS,
    SEMANTIC_ATOM_RESTATE_ACTION_SEMANTICS,
    SEMANTIC_CYCLE_CLOSE_ACTION_SEMANTICS,
    SEMANTIC_CYCLE_OPEN_ACTION_SEMANTICS,
    SEMANTIC_EDITING_V2_PROCESS_SEMANTICS,
    SEMANTIC_RING_RESTATE_SCORER_MODE,
    FactorizedTraceletRateModel,
    OperatorCapabilities,
    _graph_application_masks,
    prepare_factorized_mark_batch,
)
from compose_v4.rewrite.kernel import (
    canonical_state_key,
    editing_v2_semantic_rewrite_system,
)
from compose_v4.rewrite.operators import (
    AtomDelete,
    apply_atom_delete,
    enumerate_semantic_atom_restates,
    is_valid_atom_delete,
)
from compose_v4.rewrite.process_v2_atom_delete import process_v2_atom_delete_mask
from compose_v4.rewrite.progress import TraceProgressCTMC
from compose_v4.rewrite.trace import RewriteStep, RewriteTrace
from compose_v4.rewrite.tree_transport import compile_carbon_tree_to_target
from compose_v4.rewrite.typed_ring_catalog import build_typed_ring_catalog

_SLOTS = 24
_CONNECTED_NONLEAF_MINIMUM_DEGREE = 2

# The frozen production capability fingerprints that must not move.
_FROZEN_LEGACY_FINGERPRINT = "e787c852410c6b63"
_FROZEN_SEMANTIC_V1_FINGERPRINT = "d246bc88d8440d31"
_FROZEN_DE_NOVO_FINGERPRINT = "40510175845988f1"

# Bounded representative panel: charged (cation, anion, zwitterion), fused,
# bridged, spiro, S/Cl bearing, aromatic-plus-saturated mixtures, leaves and a
# singleton root.
_PANEL = (
    "C1CCCCC1",
    "C1CCOCC1",
    "C1CCSCC1",
    "c1ccccc1",
    "Cc1ccccc1",
    "c1ccc2ccccc2c1",
    "CC1CCCCC1",
    "C1CC2CCC1CC2",
    "C1CCC2(CC1)CCCC2",
    "O=C1NC(O)C2CCCCC12",
    "ClC1CCCCC1",
    "CC(=O)NC1CCCCC1",
    "O=S1(=O)CCCC1",
    "C1CC[NH2+]CC1",
    "C[N+](C)(C)CC(=O)[O-]",
    "[O-]C(=O)C1CCCCC1",
    "c1ccc(C2CCCCC2)cc1",
    "CSC1CCCCC1",
    "CCO",
    "C",
)

# Charged panel members, where the legacy rule and Process V2 disagree.
_CHARGED = ("C1CC[NH2+]CC1", "C[N+](C)(C)CC(=O)[O-]", "[O-]C(=O)C1CCCCC1")


def _state(smiles: str) -> MolecularGraph:
    return pad_molecular_graph(smiles_to_molecular_graph(smiles), _SLOTS)


@pytest.fixture(scope="module")
def ring_catalog():
    target = _state("c1ccccc1")
    source = DegreeBoundedCarbonTreePrior(sizes=(target.n_real_atoms,)).sample(
        np.random.default_rng(1), n_slots=_SLOTS
    )
    trace = compile_carbon_tree_to_target(
        source,
        target,
        use_bond_reroute=True,
        align_source=True,
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
    ).eval()


@pytest.fixture(scope="module")
def semantic_v1_model(ring_catalog):
    return _build_model(
        ring_catalog,
        process=SEMANTIC_EDITING_V2_PROCESS_SEMANTICS,
        delete_mode=LEGACY_ATOM_DELETE_ACTION_SEMANTICS,
    )


@pytest.fixture(scope="module")
def process_v2_model(ring_catalog, semantic_v1_model):
    model = _build_model(
        ring_catalog,
        process=PROCESS_V2_EDITING_PROCESS_SEMANTICS,
        delete_mode=PROCESS_V2_ATOM_DELETE_ACTION_SEMANTICS,
    )
    # Process V2 is a mask change over the existing per-slot delete head, so a
    # strict warm start from the V1 model must still load.
    model.load_state_dict(semantic_v1_model.state_dict(), strict=True)
    return model.eval()


def _batch(model, states, *, times=None):
    capabilities = model.operator_capabilities
    resolved_times = times if times is not None else tuple(
        0.2 + 0.05 * index for index in range(len(states))
    )
    return prepare_factorized_mark_batch(
        tuple(states),
        resolved_times,
        (None,) * len(states),
        (None,) * len(states),
        (0.0,) * len(states),
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


def _tables(model, batch):
    device_batch = batch.to(model.device)
    node, global_state, pair = model._encode_batch(device_batch)
    return model._action_tables(
        device_batch,
        node,
        global_state,
        pair,
        require_exact_ring_support=False,
    )


def _collator(model, **overrides):
    capabilities = model.operator_capabilities
    keywords = dict(
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
    keywords.update(overrides)
    return FactorizedMarkCollator(True, model.ring_catalog, **keywords)


# ---- Independent legality oracle (test-only) ----


def _real_atom_graph(state: MolecularGraph) -> nx.Graph:
    real = tuple(int(slot) for slot in np.flatnonzero(is_element(state.atom_types)))
    graph = nx.Graph()
    graph.add_nodes_from(real)
    graph.add_edges_from(
        (left, right)
        for offset, left in enumerate(real)
        for right in real[offset + 1 :]
        if int(state.bonds[left, right]) != 0
    )
    return graph


def _oracle_admission(state: MolecularGraph) -> np.ndarray:
    """Independent bounded oracle for the complete Process-V2 deletion fiber.

    Built directly from the production executor predicates, the frozen
    resonance-invariant aromatic view, graph connectivity and the charge policy.
    It never consults ``compose_v4.rewrite.process_v2_atom_delete``.
    """

    admitted = np.zeros(state.n_atoms, dtype=np.bool_)
    graph = _real_atom_graph(state)
    articulation = set(nx.articulation_points(graph))
    perceived = resonance_invariant_bond_classes(state)
    scar = state.atom_types == SCAR_IDX
    for slot in np.flatnonzero(is_element(state.atom_types)):
        slot = int(slot)
        action = AtomDelete(slot)
        if not is_valid_atom_delete(state, action):
            continue
        successor = apply_atom_delete(state, action)
        if not is_connected_or_null(successor):
            continue
        if not charge_policy_preserved(state, successor):
            continue
        if int(graph.degree[slot]) >= _CONNECTED_NONLEAF_MINIMUM_DEGREE:
            if bool((perceived[slot] == BOND_AROMATIC).any()):
                continue
            if slot in articulation:
                continue
            if bool(((state.bonds[slot] != 0) & scar).any()):
                continue
        admitted[slot] = True
    return admitted


# ---- Capability identity ----


def test_frozen_capability_fingerprints_are_unchanged_and_v2_differs() -> None:
    legacy = OperatorCapabilities(
        compute_ring_grow_support=False,
        compute_ring_restates=True,
        compute_cyclic_graft=True,
        compute_ring_opening=True,
        compute_ring_system_delete=False,
    )
    assert legacy.atom_delete_action_semantics == LEGACY_ATOM_DELETE_ACTION_SEMANTICS
    assert legacy.fingerprint() == _FROZEN_LEGACY_FINGERPRINT

    semantic_v1 = replace(
        legacy,
        editing_process_semantics=SEMANTIC_EDITING_V2_PROCESS_SEMANTICS,
        atom_restate_action_semantics=SEMANTIC_ATOM_RESTATE_ACTION_SEMANTICS,
        ring_restate_scorer_mode=SEMANTIC_RING_RESTATE_SCORER_MODE,
        cycle_close_action_semantics=SEMANTIC_CYCLE_CLOSE_ACTION_SEMANTICS,
        cycle_open_action_semantics=SEMANTIC_CYCLE_OPEN_ACTION_SEMANTICS,
    )
    assert semantic_v1.fingerprint() == _FROZEN_SEMANTIC_V1_FINGERPRINT

    process_v2 = replace(
        semantic_v1,
        editing_process_semantics=PROCESS_V2_EDITING_PROCESS_SEMANTICS,
        atom_delete_action_semantics=PROCESS_V2_ATOM_DELETE_ACTION_SEMANTICS,
    )
    assert process_v2.fingerprint() != semantic_v1.fingerprint()
    assert process_v2.fingerprint() != _FROZEN_LEGACY_FINGERPRINT
    assert OperatorCapabilities.de_novo().fingerprint() == _FROZEN_DE_NOVO_FINGERPRINT


def test_capability_rejects_a_mixed_process_and_atom_delete_mode() -> None:
    semantic_v1 = OperatorCapabilities(
        compute_ring_grow_support=False,
        compute_ring_restates=True,
        compute_cyclic_graft=True,
        compute_ring_opening=True,
        compute_ring_system_delete=False,
        editing_process_semantics=SEMANTIC_EDITING_V2_PROCESS_SEMANTICS,
        atom_restate_action_semantics=SEMANTIC_ATOM_RESTATE_ACTION_SEMANTICS,
        ring_restate_scorer_mode=SEMANTIC_RING_RESTATE_SCORER_MODE,
        cycle_close_action_semantics=SEMANTIC_CYCLE_CLOSE_ACTION_SEMANTICS,
        cycle_open_action_semantics=SEMANTIC_CYCLE_OPEN_ACTION_SEMANTICS,
    )
    with pytest.raises(ValueError, match="require the"):
        replace(
            semantic_v1,
            atom_delete_action_semantics=PROCESS_V2_ATOM_DELETE_ACTION_SEMANTICS,
        )
    with pytest.raises(ValueError, match="requires"):
        replace(
            semantic_v1,
            editing_process_semantics=PROCESS_V2_EDITING_PROCESS_SEMANTICS,
        )
    with pytest.raises(ValueError, match="unknown atom-delete action semantics"):
        replace(semantic_v1, atom_delete_action_semantics="not_a_mode")
    with pytest.raises(ValueError, match="require the"):
        OperatorCapabilities(
            atom_delete_action_semantics=PROCESS_V2_ATOM_DELETE_ACTION_SEMANTICS,
        )
    assert ATOM_DELETE_ACTION_SEMANTICS == (
        LEGACY_ATOM_DELETE_ACTION_SEMANTICS,
        PROCESS_V2_ATOM_DELETE_ACTION_SEMANTICS,
    )


def test_model_capability_carries_the_atom_delete_mode(
    semantic_v1_model,
    process_v2_model,
) -> None:
    assert semantic_v1_model.operator_capabilities.atom_delete_action_semantics == (
        LEGACY_ATOM_DELETE_ACTION_SEMANTICS
    )
    assert process_v2_model.operator_capabilities.atom_delete_action_semantics == (
        PROCESS_V2_ATOM_DELETE_ACTION_SEMANTICS
    )
    assert (
        process_v2_model.operator_capabilities.fingerprint()
        != semantic_v1_model.operator_capabilities.fingerprint()
    )


def test_model_rejects_a_mixed_process_and_atom_delete_mode(ring_catalog) -> None:
    with pytest.raises(ValueError, match="requires"):
        _build_model(
            ring_catalog,
            process=PROCESS_V2_EDITING_PROCESS_SEMANTICS,
            delete_mode=LEGACY_ATOM_DELETE_ACTION_SEMANTICS,
        )
    with pytest.raises(ValueError, match="require the"):
        _build_model(
            ring_catalog,
            process=SEMANTIC_EDITING_V2_PROCESS_SEMANTICS,
            delete_mode=PROCESS_V2_ATOM_DELETE_ACTION_SEMANTICS,
        )


# ---- Mask construction ----


def test_legacy_batch_mask_is_the_unchanged_v1_dense_mask(semantic_v1_model) -> None:
    for smiles in _PANEL:
        state = _state(smiles)
        batch = _batch(semantic_v1_model, (state,))
        atom_topology, _, _ = compute_topology_features(state)
        expected = _graph_application_masks(state, atom_topology)[0]
        assert np.array_equal(batch.atom_delete_mask[0].numpy(), expected), smiles
        assert batch.atom_delete_admission_mask is None
        assert batch.atom_delete_action_semantics == (
            LEGACY_ATOM_DELETE_ACTION_SEMANTICS
        )


def test_process_v2_batch_mask_equals_the_admission_authority(process_v2_model) -> None:
    """The core contract of this round: equality, not a union."""

    for smiles in _PANEL:
        state = _state(smiles)
        batch = _batch(process_v2_model, (state,))
        authority = process_v2_atom_delete_mask(state)

        assert batch.atom_delete_admission_mask is not None
        assert batch.atom_delete_admission_mask.dtype == torch.bool
        assert np.array_equal(batch.atom_delete_mask[0].numpy(), authority), smiles
        assert torch.equal(
            batch.atom_delete_mask,
            batch.atom_delete_admission_mask,
        ), smiles


def test_process_v2_mask_is_neither_a_superset_nor_a_subset_of_the_legacy_rule(
    semantic_v1_model,
    process_v2_model,
) -> None:
    """The defect, at the batch level.

    A union with the legacy rule would leave ``removed`` empty on every state.
    The charged fixtures make it non-empty, and every removed slot is one the
    charge policy refuses.
    """

    added_total = 0
    removed_total = 0
    removed_sources: list[str] = []
    for smiles in _PANEL:
        state = _state(smiles)
        legacy = _batch(semantic_v1_model, (state,)).atom_delete_mask[0].numpy()
        effective = _batch(process_v2_model, (state,)).atom_delete_mask[0].numpy()

        added = effective & ~legacy
        removed = legacy & ~effective
        added_total += int(added.sum())
        removed_total += int(removed.sum())
        if bool(removed.any()):
            removed_sources.append(smiles)

        graph = _real_atom_graph(state)
        for slot in np.flatnonzero(added):
            # Everything newly reached is connected-nonleaf.
            assert int(graph.degree[int(slot)]) >= _CONNECTED_NONLEAF_MINIMUM_DEGREE
        for slot in np.flatnonzero(removed):
            # Everything withdrawn is withdrawn by the charge policy alone.
            action = AtomDelete(int(slot))
            assert is_valid_atom_delete(state, action) is True, (smiles, int(slot))
            successor = apply_atom_delete(state, action)
            assert is_connected_or_null(successor) is True
            assert charge_policy_preserved(state, successor) is False, (smiles, slot)
    assert added_total > 0
    # Four methyl/oxide slots of the zwitterion and the carboxylate oxide.  The
    # piperidinium contributes none because the legacy rule never reached its
    # ring atoms in the first place.
    assert removed_total == 5, removed_total
    assert removed_sources == ["C[N+](C)(C)CC(=O)[O-]", "[O-]C(=O)C1CCCCC1"]
    assert set(removed_sources) <= set(_CHARGED)


def test_model_mask_never_admits_a_scar_or_null_slot(process_v2_model) -> None:
    base = _state("C1CCCCC1")
    atom_types = base.atom_types.copy()
    bonds = base.bonds.copy()
    hydrogens = base.implicit_h_counts.copy()
    atom_types[7] = SCAR_IDX
    bonds[7, 0] = bonds[0, 7] = 1
    hydrogens[0] -= 1
    scarred = MolecularGraph(atom_types, base.formal_charges.copy(), hydrogens, bonds)

    batch = _batch(process_v2_model, (scarred,))
    effective = batch.atom_delete_mask[0].numpy()
    non_element = ~is_element(scarred.atom_types)
    assert not bool(effective[non_element].any())
    # Slot 0 is SCAR-incident and connected-nonleaf, so it is deferred.
    assert tuple(int(slot) for slot in np.flatnonzero(effective)) == (1, 2, 3, 4, 5)


def test_legacy_rule_admits_only_degree_at_most_one_slots(semantic_v1_model) -> None:
    """The structural fact that makes the newly reached fiber disjoint from it."""

    for smiles in _PANEL:
        state = _state(smiles)
        legacy = _batch(semantic_v1_model, (state,)).atom_delete_mask[0].numpy()
        graph = _real_atom_graph(state)
        for slot in np.flatnonzero(legacy):
            assert int(graph.degree[int(slot)]) < _CONNECTED_NONLEAF_MINIMUM_DEGREE, (
                smiles,
                int(slot),
            )


def test_model_mask_agrees_with_an_independent_executor_oracle(
    process_v2_model,
) -> None:
    assert len(_PANEL) >= 12
    disagreements: list[tuple[str, int, bool, bool]] = []
    admitted_total = 0
    for smiles in _PANEL:
        state = _state(smiles)
        effective = _batch(process_v2_model, (state,)).atom_delete_mask[0].numpy()
        oracle = _oracle_admission(state)
        admitted_total += int(effective.sum())
        for slot in range(state.n_atoms):
            if bool(effective[slot]) != bool(oracle[slot]):
                disagreements.append(
                    (smiles, slot, bool(effective[slot]), bool(oracle[slot]))
                )
    assert disagreements == []
    assert admitted_total > 0


def test_every_admitted_delete_executes_to_the_exact_successor(
    process_v2_model,
) -> None:
    checked = 0
    for smiles in _PANEL:
        state = _state(smiles)
        batch = _batch(process_v2_model, (state,))
        for slot in np.flatnonzero(batch.atom_delete_mask[0].numpy()):
            slot = int(slot)
            action = AtomDelete(slot)
            assert is_valid_atom_delete(state, action) is True
            successor = apply_atom_delete(state, action)

            expected_types = state.atom_types.copy()
            expected_charges = state.formal_charges.copy()
            expected_hydrogens = state.implicit_h_counts.copy()
            expected_bonds = state.bonds.copy()
            for neighbor in np.flatnonzero(state.bonds[slot] != 0):
                order = int(state.bonds[slot, neighbor])
                expected_hydrogens[neighbor] += order if order < BOND_AROMATIC else 0
            expected_bonds[slot, :] = 0
            expected_bonds[:, slot] = 0
            expected_types[slot] = 0
            expected_charges[slot] = 0
            expected_hydrogens[slot] = 0

            assert np.array_equal(successor.atom_types, expected_types), (smiles, slot)
            assert np.array_equal(successor.formal_charges, expected_charges)
            assert np.array_equal(successor.implicit_h_counts, expected_hydrogens)
            assert np.array_equal(successor.bonds, expected_bonds)
            checked += 1
    assert checked > 0


def test_the_vectorized_charge_policy_removes_nothing_under_process_v2(
    semantic_v1_model,
    process_v2_model,
) -> None:
    """The candidate fiber and the scored fiber must be the same set.

    ``_action_tables`` intersects primitive support with a vectorized form of the
    charge policy.  Under Process V2 the exact CPU authority has already applied
    that policy, so the intersection must be a no-op and the scored mask must
    equal the batch mask.  Under the legacy rule it still bites, which is what
    made the round-one batch mask disagree with the mask the model scored.
    """

    legacy_dropped = 0
    for smiles in _PANEL:
        state = _state(smiles)
        v2_batch = _batch(process_v2_model, (state,))
        v2_scored = _tables(process_v2_model, v2_batch)[0]["atom_delete"]
        assert torch.equal(v2_scored, v2_batch.atom_delete_mask), smiles

        legacy_batch = _batch(semantic_v1_model, (state,))
        legacy_scored = _tables(semantic_v1_model, legacy_batch)[0]["atom_delete"]
        legacy_dropped += int(
            (legacy_batch.atom_delete_mask & ~legacy_scored).sum()
        )
    assert legacy_dropped == 5, legacy_dropped


# ---- Other Active8 tables ----


def test_other_active8_masks_and_logits_are_unchanged(
    semantic_v1_model,
    process_v2_model,
) -> None:
    for smiles in ("C1CCCCC1", "c1ccccc1", "CC1CCCCC1", "CCO", "C1CC[NH2+]CC1"):
        state = _state(smiles)
        legacy_masks, legacy_logits, _ = _tables(
            semantic_v1_model,
            _batch(semantic_v1_model, (state,)),
        )
        v2_masks, v2_logits, _ = _tables(
            process_v2_model,
            _batch(process_v2_model, (state,)),
        )
        assert set(legacy_masks) == set(v2_masks)
        assert set(legacy_logits) == set(v2_logits)
        for name in legacy_masks:
            if name == "atom_delete":
                continue
            assert torch.equal(legacy_masks[name], v2_masks[name]), (smiles, name)
        # Every head, including the per-slot delete head, is untouched; only
        # the delete MASK moves.  The family normalizer legitimately shifts
        # because the atom_delete support changed, so it is not compared here.
        for name in legacy_logits:
            assert torch.equal(legacy_logits[name], v2_logits[name]), (smiles, name)


# ---- Batch/model process binding ----


def test_process_v2_model_rejects_a_legacy_delete_batch(
    semantic_v1_model,
    process_v2_model,
) -> None:
    legacy_batch = _batch(semantic_v1_model, (_state("C1CCCCC1"),))
    with pytest.raises(ValueError, match="editing process semantics disagree"):
        _tables(process_v2_model, legacy_batch)

    forged = replace(
        legacy_batch,
        editing_process_semantics=PROCESS_V2_EDITING_PROCESS_SEMANTICS,
    )
    with pytest.raises(ValueError, match="atom-delete semantics disagree"):
        _tables(process_v2_model, forged)


def test_legacy_model_rejects_a_process_v2_batch(
    semantic_v1_model,
    process_v2_model,
) -> None:
    v2_batch = _batch(process_v2_model, (_state("C1CCCCC1"),))
    with pytest.raises(ValueError, match="editing process semantics disagree"):
        _tables(semantic_v1_model, v2_batch)

    forged = replace(
        _batch(semantic_v1_model, (_state("C1CCCCC1"),)),
        atom_delete_admission_mask=v2_batch.atom_delete_admission_mask,
    )
    with pytest.raises(ValueError, match="unexpectedly carries a Process-V2"):
        _tables(semantic_v1_model, forged)


def test_forward_requires_equality_not_containment(
    semantic_v1_model,
    process_v2_model,
) -> None:
    """The guard that would have caught the round-one union.

    Round one asserted that the admission mask was CONTAINED in the batch mask.
    A batch whose delete mask is ``legacy | admission`` satisfies containment
    while offering candidates the authority refuses, so that check could not
    detect the defect it existed to guard.  Equality rejects it.
    """

    state = _state("C[N+](C)(C)CC(=O)[O-]")
    v2_batch = _batch(process_v2_model, (state,))
    legacy_mask = _batch(semantic_v1_model, (state,)).atom_delete_mask
    union = legacy_mask | v2_batch.atom_delete_admission_mask
    assert not torch.equal(union, v2_batch.atom_delete_mask)
    assert bool((v2_batch.atom_delete_admission_mask & ~union).sum() == 0)

    with pytest.raises(ValueError, match="does not equal the batch atom-delete mask"):
        _tables(process_v2_model, replace(v2_batch, atom_delete_mask=union))

    # A strictly narrower batch mask is rejected in the same way.
    narrowed = v2_batch.atom_delete_mask.clone()
    narrowed[0, 6] = False
    with pytest.raises(ValueError, match="does not equal the batch atom-delete mask"):
        _tables(process_v2_model, replace(v2_batch, atom_delete_mask=narrowed))

    with pytest.raises(ValueError, match="lacks the exact uniform-gated"):
        _tables(process_v2_model, replace(v2_batch, atom_delete_admission_mask=None))

    wrong_shape = v2_batch.atom_delete_admission_mask[:, :-1]
    with pytest.raises(ValueError, match="wrong shape"):
        _tables(
            process_v2_model,
            replace(v2_batch, atom_delete_admission_mask=wrong_shape),
        )


def test_prepare_batch_fails_loudly_without_the_threaded_delete_mode(
    process_v2_model,
) -> None:
    """A builder that forgets the flag must raise, not collate a legacy mask."""

    capabilities = process_v2_model.operator_capabilities
    with pytest.raises(ValueError, match="Process-V2 Editing-V2 batch requires"):
        prepare_factorized_mark_batch(
            (_state("C1CCCCC1"),),
            (0.3,),
            (None,),
            (None,),
            (0.0,),
            ring_catalog=process_v2_model.ring_catalog,
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
        )


def test_prepare_batch_rejects_the_v2_delete_mode_outside_the_v2_process(
    semantic_v1_model,
) -> None:
    capabilities = semantic_v1_model.operator_capabilities
    with pytest.raises(ValueError, match="atom-delete semantics require the"):
        prepare_factorized_mark_batch(
            (_state("C1CCCCC1"),),
            (0.3,),
            (None,),
            (None,),
            (0.0,),
            ring_catalog=semantic_v1_model.ring_catalog,
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
            atom_delete_action_semantics=PROCESS_V2_ATOM_DELETE_ACTION_SEMANTICS,
        )
    with pytest.raises(ValueError, match="atom-delete semantics require the"):
        prepare_factorized_mark_batch(
            (_state("C1CCCCC1"),),
            (0.3,),
            (None,),
            (None,),
            (0.0,),
            ring_catalog=semantic_v1_model.ring_catalog,
            editing_process_semantics=LEGACY_EDITING_PROCESS_SEMANTICS,
            atom_restate_action_semantics=LEGACY_ATOM_RESTATE_ACTION_SEMANTICS,
            ring_restate_scorer_mode=LEGACY_RING_RESTATE_SCORER_MODE,
            cycle_close_action_semantics=LEGACY_CYCLE_CLOSE_ACTION_SEMANTICS,
            cycle_open_action_semantics=LEGACY_CYCLE_OPEN_ACTION_SEMANTICS,
            atom_delete_action_semantics=PROCESS_V2_ATOM_DELETE_ACTION_SEMANTICS,
        )


# ---- Batch reconstruction and collation ----


def test_batch_reconstruction_preserves_the_admission_mask(process_v2_model) -> None:
    states = tuple(_state(smiles) for smiles in ("C1CCCCC1", "CC1CCCCC1", "CCO"))
    batch = _batch(process_v2_model, states)
    assert batch.atom_delete_admission_mask.shape == batch.atom_delete_mask.shape

    for reconstructed in (
        batch.subbatch(1, 3),
        batch.to(torch.device("cpu")).subbatch(1, 3),
    ):
        assert reconstructed.atom_delete_action_semantics == (
            PROCESS_V2_ATOM_DELETE_ACTION_SEMANTICS
        )
        assert torch.equal(
            reconstructed.atom_delete_admission_mask,
            batch.atom_delete_admission_mask[1:3],
        )
        assert torch.equal(
            reconstructed.atom_delete_mask,
            batch.atom_delete_mask[1:3],
        )
        # The equality contract must survive every reconstruction.
        assert torch.equal(
            reconstructed.atom_delete_mask,
            reconstructed.atom_delete_admission_mask,
        )

    moved = batch.to(torch.device("cpu"))
    assert torch.equal(
        moved.atom_delete_admission_mask,
        batch.atom_delete_admission_mask,
    )
    assert moved.atom_delete_action_semantics == batch.atom_delete_action_semantics


def test_pin_memory_propagates_every_process_v2_field(
    process_v2_model, monkeypatch
) -> None:
    """The CPU-safe half: `pin_memory` must rebuild every field, on any machine.

    `pin_memory` is a hand-written field-by-field reconstruction, which is the
    exact shape that dropped `atom_delete_admission_mask` from
    `_index_factorized_batch`. What is worth testing here is that reconstruction,
    not the host allocator.

    `Tensor.pin_memory()` takes no device argument and dispatches to the current
    accelerator. On this repository's Macs that is MPS, which has no pinned host
    memory, so the real call raises (torch 2.11 and 2.13 alike) and the previous
    `except RuntimeError: pytest.skip(...)` meant the Process-V2 assertion below
    never executed on any developer machine. Substituting identity for the pin
    exercises the whole reconstruction without invoking an unsupported backend,
    which is why this test is the one that runs everywhere.
    """

    batch = _batch(process_v2_model, (_state("C1CCCCC1"), _state("CC1CCCCC1")))
    assert batch.atom_delete_admission_mask is not None

    monkeypatch.setattr(torch.Tensor, "pin_memory", lambda self: self, raising=True)
    rebuilt = batch.pin_memory()

    names = [item.name for item in dataclass_fields(type(batch))]
    assert "atom_delete_admission_mask" in names
    assert "atom_delete_action_semantics" in names
    for name in names:
        original = getattr(batch, name)
        produced = getattr(rebuilt, name)
        if original is None or produced is None:
            assert original is produced, name
        elif isinstance(original, torch.Tensor):
            assert torch.equal(original, produced), name
        else:
            assert original == produced, name


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="pinned host memory requires CUDA; MPS has no pinned-host-memory path",
)
def test_pin_memory_preserves_the_admission_mask(process_v2_model) -> None:
    """The accelerator-backed half, on hardware that genuinely supports pinning.

    Gated on the capability rather than on catching an exception from an
    unsupported backend: invoking `pin_memory` on MPS to discover it is
    unavailable is what made this test unsafe in the first place.
    """

    batch = _batch(process_v2_model, (_state("C1CCCCC1"), _state("CC1CCCCC1")))
    pinned = batch.pin_memory()
    assert pinned.atom_delete_action_semantics == (
        PROCESS_V2_ATOM_DELETE_ACTION_SEMANTICS
    )
    assert pinned.atom_delete_admission_mask.is_pinned()
    assert torch.equal(
        pinned.atom_delete_admission_mask.cpu(),
        batch.atom_delete_admission_mask,
    )
    assert torch.equal(
        pinned.atom_delete_mask.cpu(),
        pinned.atom_delete_admission_mask.cpu(),
    )


def test_direct_and_multiworker_collation_produce_identical_masks(
    process_v2_model,
) -> None:
    examples = [
        FactorizedMarkExample(
            state=_state(smiles),
            time=0.2 + 0.1 * index,
            teacher_action=None,
            teacher_rule_name=None,
            teacher_rate=0.0,
            importance_weight=1.0,
        )
        for index, smiles in enumerate(
            ("C1CCCCC1", "CC1CCCCC1", "c1ccccc1", "C[N+](C)(C)CC(=O)[O-]")
        )
    ]
    direct = _collator(process_v2_model)(examples)
    # Each DataLoader worker owns its own collator and chemistry cache; the
    # loader then concatenates the shard batches.
    merged = _concatenate_factorized_mark_batches(
        (
            _collator(process_v2_model)(examples[:2]),
            _collator(process_v2_model)(examples[2:]),
        )
    )
    assert merged.atom_delete_action_semantics == direct.atom_delete_action_semantics
    assert torch.equal(merged.atom_delete_mask, direct.atom_delete_mask)
    assert torch.equal(
        merged.atom_delete_admission_mask,
        direct.atom_delete_admission_mask,
    )
    assert torch.equal(merged.atom_delete_mask, merged.atom_delete_admission_mask)
    assert torch.equal(
        direct.atom_delete_mask,
        _batch(
            process_v2_model, tuple(example.state for example in examples)
        ).atom_delete_mask,
    )


def test_sample_factorized_mark_batch_threads_the_capability(
    semantic_v1_model,
    process_v2_model,
) -> None:
    """The eval/validation builder must consume the SAME V2 mask as training."""

    runtime = editing_v2_semantic_rewrite_system()
    source = _state("C1CCCCC1")
    action = enumerate_semantic_atom_restates(source)[0]
    target = runtime.apply(source, "atom_restate_semantic", action)
    record = PathRecord(
        target_key=canonical_state_key(target),
        path=TraceProgressCTMC(
            RewriteTrace(
                source=source,
                target=target,
                steps=(RewriteStep("atom_restate_semantic", action),),
                metadata={},
            ),
            system=runtime,
        ),
    )

    def sampled(model):
        return sample_factorized_mark_batch(
            (record,),
            batch_size=2,
            seed=3,
            late_time_fraction=0.0,
            operational_horizon=1.0,
            ring_catalog=model.ring_catalog,
            capabilities=model.operator_capabilities,
        )

    legacy = sampled(semantic_v1_model)
    assert legacy.atom_delete_action_semantics == LEGACY_ATOM_DELETE_ACTION_SEMANTICS
    assert legacy.atom_delete_admission_mask is None

    evaluated = sampled(process_v2_model)
    assert evaluated.atom_delete_action_semantics == (
        PROCESS_V2_ATOM_DELETE_ACTION_SEMANTICS
    )
    assert evaluated.atom_delete_admission_mask is not None
    assert torch.equal(
        evaluated.atom_delete_admission_mask,
        evaluated.atom_delete_mask,
    )
    for row, state in enumerate(evaluated.states):
        assert np.array_equal(
            evaluated.atom_delete_mask[row].numpy(),
            _oracle_admission(state),
        )
    # The model must be able to score the batch its own capability produced.
    _tables(process_v2_model, evaluated)


def test_concatenation_rejects_mixed_atom_delete_modes(
    semantic_v1_model,
    process_v2_model,
) -> None:
    state = _state("C1CCCCC1")
    v2_batch = _batch(process_v2_model, (state,))
    forged = replace(
        _batch(semantic_v1_model, (state,)),
        editing_process_semantics=PROCESS_V2_EDITING_PROCESS_SEMANTICS,
        atom_restate_action_semantics=v2_batch.atom_restate_action_semantics,
        ring_restate_scorer_mode=v2_batch.ring_restate_scorer_mode,
        cycle_close_action_semantics=v2_batch.cycle_close_action_semantics,
        cycle_open_action_semantics=v2_batch.cycle_open_action_semantics,
    )
    with pytest.raises(ValueError, match="mix semantic action process identities"):
        _concatenate_factorized_mark_batches((v2_batch, forged))


def test_chemistry_feature_cache_never_shares_a_v1_and_v2_mask(
    semantic_v1_model,
    process_v2_model,
) -> None:
    shared_cache: dict = {}
    state = _state("C1CCCCC1")

    def build(model):
        capabilities = model.operator_capabilities
        return prepare_factorized_mark_batch(
            (state,),
            (0.3,),
            (None,),
            (None,),
            (0.0,),
            ring_catalog=model.ring_catalog,
            chemistry_feature_cache=shared_cache,
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

    legacy = build(semantic_v1_model)
    v2 = build(process_v2_model)
    assert len(shared_cache) == 2
    assert not bool(legacy.atom_delete_mask.any())
    assert bool(v2.atom_delete_mask.any())
    assert not torch.equal(legacy.atom_delete_mask, v2.atom_delete_mask)
