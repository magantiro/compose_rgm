"""The learned ``atom_delete`` candidate mask under Process V2.

The effective Process-V2 mask is the DISJOINT UNION of the unchanged V1 dense
rule and the Process-V2 connected-nonleaf expansion.  These tests pin that the
legacy branch is byte-identical, that the expansion agrees with an independently
invoked production executor in both directions, that the other seven Active8
tables do not move, and that a mixed V1/V2 batch or capability fails loudly.
"""

from __future__ import annotations

from dataclasses import replace

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


def _oracle_connected_nonleaf_admission(state: MolecularGraph) -> np.ndarray:
    """Independent bounded oracle for the additional deletion fiber.

    Built directly from the production executor predicates, the frozen
    resonance-invariant aromatic view, graph connectivity and the charge policy.
    It never consults ``compose_v4.rewrite.process_v2_atom_delete``.
    """

    admitted = np.zeros(state.n_atoms, dtype=np.bool_)
    graph = _real_atom_graph(state)
    articulation = set(nx.articulation_points(graph))
    perceived = resonance_invariant_bond_classes(state)
    for slot in np.flatnonzero(is_element(state.atom_types)):
        slot = int(slot)
        if int(graph.degree[slot]) < _CONNECTED_NONLEAF_MINIMUM_DEGREE:
            continue
        if bool((perceived[slot] == BOND_AROMATIC).any()):
            continue
        if slot in articulation:
            continue
        action = AtomDelete(slot)
        if not is_valid_atom_delete(state, action):
            continue
        successor = apply_atom_delete(state, action)
        if not is_connected_or_null(successor):
            continue
        if not charge_policy_preserved(state, successor):
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


def test_process_v2_mask_is_the_disjoint_union_with_the_expansion(
    semantic_v1_model,
    process_v2_model,
) -> None:
    additions = 0
    for smiles in _PANEL:
        state = _state(smiles)
        legacy = _batch(semantic_v1_model, (state,)).atom_delete_mask[0].numpy()
        batch = _batch(process_v2_model, (state,))
        admission = batch.atom_delete_admission_mask[0].numpy()
        effective = batch.atom_delete_mask[0].numpy()

        assert batch.atom_delete_admission_mask.dtype == torch.bool
        assert not bool((legacy & admission).any()), smiles
        assert np.array_equal(effective, legacy | admission), smiles
        assert np.array_equal(effective & admission, admission), smiles
        additions += int(admission.sum())

        graph = _real_atom_graph(state)
        low_degree = np.zeros(state.n_atoms, dtype=np.bool_)
        for slot in graph.nodes:
            if int(graph.degree[slot]) < _CONNECTED_NONLEAF_MINIMUM_DEGREE:
                low_degree[int(slot)] = True
        null_or_scar = ~is_element(state.atom_types)
        preserved = low_degree | null_or_scar
        assert np.array_equal(effective & preserved, legacy & preserved), smiles
        assert not bool(admission[preserved].any()), smiles
    assert additions > 0


def test_model_mask_exposes_the_reference_connected_nonleaf_slots(
    semantic_v1_model,
    process_v2_model,
) -> None:
    """The exposed additional slots, read through the batch the model scores."""

    reference = (
        ("C1CCCCC1", (0, 1, 2, 3, 4, 5)),
        ("C1CCOCC1", (0, 1, 2, 3, 4, 5)),
        ("C1CCSCC1", (0, 1, 2, 3, 4, 5)),
        ("c1ccccc1", ()),
        ("Cc1ccccc1", ()),
        ("c1ccc2ccccc2c1", ()),
        ("CC1CCCCC1", (2, 3, 4, 5, 6)),
        ("C1CC2CCC1CC2", (0, 1, 2, 3, 4, 5, 6, 7)),
        ("O=C1NC(O)C2CCCCC12", (2, 5, 6, 7, 8, 9, 10)),
        ("CCO", ()),
        ("C", ()),
        ("C[N+](C)(C)CC(=O)[O-]", ()),
    )
    for smiles, expected in reference:
        state = _state(smiles)
        batch = _batch(process_v2_model, (state,))
        admission = batch.atom_delete_admission_mask[0].numpy()
        assert tuple(int(slot) for slot in np.flatnonzero(admission)) == expected, smiles
        legacy = _batch(semantic_v1_model, (state,)).atom_delete_mask[0].numpy()
        effective = batch.atom_delete_mask[0].numpy()
        assert tuple(int(slot) for slot in np.flatnonzero(effective & ~legacy)) == (
            expected
        ), smiles


def test_model_mask_never_admits_a_scar_or_null_slot(process_v2_model) -> None:
    base = _state("C1CCCCC1")
    atom_types = base.atom_types.copy()
    atom_types[7] = SCAR_IDX
    scarred = MolecularGraph(
        atom_types,
        base.formal_charges.copy(),
        base.implicit_h_counts.copy(),
        base.bonds.copy(),
    )
    batch = _batch(process_v2_model, (scarred,))
    effective = batch.atom_delete_mask[0].numpy()
    admission = batch.atom_delete_admission_mask[0].numpy()
    non_element = ~is_element(scarred.atom_types)
    assert not bool(effective[non_element].any())
    assert not bool(admission[non_element].any())
    assert tuple(int(slot) for slot in np.flatnonzero(admission)) == (0, 1, 2, 3, 4, 5)


def test_v1_admits_only_degree_at_most_one_slots(semantic_v1_model) -> None:
    """The structural fact that makes the union disjoint by construction."""

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
    semantic_v1_model,
    process_v2_model,
) -> None:
    assert len(_PANEL) >= 12
    disagreements: list[tuple[str, int, bool, bool]] = []
    admitted_total = 0
    for smiles in _PANEL:
        state = _state(smiles)
        legacy = _batch(semantic_v1_model, (state,)).atom_delete_mask[0].numpy()
        effective = _batch(process_v2_model, (state,)).atom_delete_mask[0].numpy()
        oracle = _oracle_connected_nonleaf_admission(state) | legacy
        admitted_total += int(effective.sum())
        for slot in range(state.n_atoms):
            if bool(effective[slot]) != bool(oracle[slot]):
                disagreements.append(
                    (smiles, slot, bool(effective[slot]), bool(oracle[slot]))
                )
    assert disagreements == []
    assert admitted_total > 0


def test_every_admitted_delete_executes_to_the_exact_successor(
    semantic_v1_model,
    process_v2_model,
) -> None:
    checked = 0
    for smiles in _PANEL:
        state = _state(smiles)
        legacy = _batch(semantic_v1_model, (state,)).atom_delete_mask[0].numpy()
        batch = _batch(process_v2_model, (state,))
        admission = batch.atom_delete_admission_mask[0].numpy()
        for slot in np.flatnonzero(admission):
            slot = int(slot)
            assert not bool(legacy[slot])
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


def test_scored_mask_keeps_every_admission_after_the_charge_policy(
    semantic_v1_model,
    process_v2_model,
) -> None:
    """``_action_tables`` intersects with the vectorized charge policy.

    The expansion is already resolved against the exact ``charge_policy_preserved``
    predicate, so the vectorized fast path must not drop any admitted slot; the
    preserved V1 leaves keep their historical charge-policy behaviour.
    """

    charged = 0
    for smiles in _PANEL:
        state = _state(smiles)
        legacy_scored = _tables(
            semantic_v1_model,
            _batch(semantic_v1_model, (state,)),
        )[0]["atom_delete"]
        batch = _batch(process_v2_model, (state,))
        v2_scored = _tables(process_v2_model, batch)[0]["atom_delete"]
        assert torch.equal(
            v2_scored,
            legacy_scored | batch.atom_delete_admission_mask,
        ), smiles
        if bool((state.formal_charges != 0).any()):
            charged += 1
    assert charged >= 3


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
        # because the atom_delete support grew, so it is not compared here.
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


def test_process_v2_model_requires_a_contained_admission_mask(
    semantic_v1_model,
    process_v2_model,
) -> None:
    state = _state("C1CCCCC1")
    v2_batch = _batch(process_v2_model, (state,))

    with pytest.raises(ValueError, match="lacks the exact connected-nonleaf"):
        _tables(process_v2_model, replace(v2_batch, atom_delete_admission_mask=None))

    legacy_mask = _batch(semantic_v1_model, (state,)).atom_delete_mask
    with pytest.raises(ValueError, match="not contained in the batch atom-delete mask"):
        _tables(process_v2_model, replace(v2_batch, atom_delete_mask=legacy_mask))

    wide = v2_batch.atom_delete_admission_mask[:, :-1]
    with pytest.raises(ValueError, match="wrong shape"):
        _tables(process_v2_model, replace(v2_batch, atom_delete_admission_mask=wide))


def test_prepare_batch_fails_loudly_without_the_threaded_delete_mode(
    process_v2_model,
) -> None:
    """A builder that forgets the flag must raise, not collate a V1 mask."""

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

    moved = batch.to(torch.device("cpu"))
    assert torch.equal(
        moved.atom_delete_admission_mask,
        batch.atom_delete_admission_mask,
    )
    assert moved.atom_delete_action_semantics == batch.atom_delete_action_semantics


def test_pin_memory_preserves_the_admission_mask(process_v2_model) -> None:
    batch = _batch(process_v2_model, (_state("C1CCCCC1"), _state("CC1CCCCC1")))
    try:
        pinned = batch.pin_memory()
    except RuntimeError as error:  # pragma: no cover - accelerator dependent
        pytest.skip(f"pinned host memory is unavailable on this machine: {error}")
    assert pinned.atom_delete_action_semantics == (
        PROCESS_V2_ATOM_DELETE_ACTION_SEMANTICS
    )
    assert torch.equal(
        pinned.atom_delete_admission_mask.cpu(),
        batch.atom_delete_admission_mask,
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
        for index, smiles in enumerate(("C1CCCCC1", "CC1CCCCC1", "c1ccccc1", "CCO"))
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
    assert torch.equal(
        direct.atom_delete_mask,
        _batch(process_v2_model, tuple(example.state for example in examples)).atom_delete_mask,
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
    assert evaluated.atom_delete_admission_mask.shape == evaluated.atom_delete_mask.shape
    assert not bool(
        (evaluated.atom_delete_admission_mask & ~evaluated.atom_delete_mask).any()
    )
    for row, state in enumerate(evaluated.states):
        expected = _oracle_connected_nonleaf_admission(state)
        assert np.array_equal(
            evaluated.atom_delete_admission_mask[row].numpy(),
            expected,
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
