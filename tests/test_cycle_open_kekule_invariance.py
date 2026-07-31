"""Fail-closed prospective invariance checks for contextual cycle opening."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from compose_v4.chem.aromaticity import resonance_invariant_bond_classes
from compose_v4.chem.molecular_graph import (
    ORGANIC_VOCABULARY,
    smiles_to_molecular_graph,
)
from compose_v4.chem.persistent_state_identity import persistent_slot_state_sha256
from compose_v4.chem.state import pad_molecular_graph
from compose_v4.experiments import cycle_open_kekule_invariance as diagnostic
from compose_v4.model.factorized_tracelet_rate_model import FactorizedTraceletRateModel
from compose_v4.rewrite.kernel import canonical_state_key, de_novo_rewrite_system
from compose_v4.rewrite.operators import BondDelete
from compose_v4.rewrite.typed_ring_catalog import TypedRingCatalog


def _state(smiles: str, slots: int = 16):
    return pad_molecular_graph(smiles_to_molecular_graph(smiles), slots)


def test_alternate_constructor_proves_semantic_and_exact_state_relationship() -> None:
    pair = diagnostic.build_alternate_kekule_pair(_state("Cc1cccc(Cl)c1"))

    assert pair.component_slots == tuple(sorted(pair.component_slots))
    assert len(pair.component_slots) == 6
    assert len(pair.component_edges) == 6
    assert canonical_state_key(pair.original) == canonical_state_key(pair.alternate)
    assert np.array_equal(
        resonance_invariant_bond_classes(pair.original),
        resonance_invariant_bond_classes(pair.alternate),
    )
    assert persistent_slot_state_sha256(pair.original) != persistent_slot_state_sha256(
        pair.alternate
    )
    assert np.array_equal(pair.original.atom_types, pair.alternate.atom_types)
    assert np.array_equal(pair.original.formal_charges, pair.alternate.formal_charges)
    assert np.array_equal(
        pair.original.implicit_h_counts,
        pair.alternate.implicit_h_counts,
    )


@pytest.mark.parametrize("smiles", ("C1CCCCC1", "c1ccc2ccccc2c1"))
def test_alternate_constructor_rejects_out_of_scope_ring_systems(smiles: str) -> None:
    with pytest.raises(diagnostic.CycleOpenKekuleInvarianceError):
        diagnostic.build_alternate_kekule_pair(_state(smiles))


def test_slot_transport_uses_new_to_old_permutation_convention() -> None:
    source = _state("c1ccccc1", 10)
    pair = diagnostic.build_alternate_kekule_pair(source)
    permutation = (7, 2, 9, 0, 4, 1, 6, 3, 5, 8)
    action = BondDelete(*pair.component_edges[0])
    transported = diagnostic.transport_bond_delete(
        action,
        permutation,
        n_slots=source.n_atoms,
    )
    relabeled = diagnostic.permute_persistent_slots(source, permutation)
    runtime = de_novo_rewrite_system()

    assert canonical_state_key(runtime.apply(source, "bond_delete", action)) == (
        canonical_state_key(runtime.apply(relabeled, "bond_delete", transported))
    )


def test_symmetric_support_preflight_passes_but_authorizes_nothing() -> None:
    report = diagnostic.run_cycle_open_kekule_invariance_diagnostic(_state("c1ccccc1"))

    assert report["status"] == "SUPPORT_PREFLIGHT_PASS_SCORING_NOT_REQUESTED"
    assert report["paper_claim_authorized"] is False
    assert report["training_authorized"] is False
    assert report["scorer_promotion_authorized"] is False
    assert report["frozen_e5_contract_modified"] is False
    assert report["source_evidence"]["canonical_source_identity_equal"] is True
    assert report["source_evidence"]["neural_bond_view_equal"] is True
    assert report["source_evidence"]["exact_state_identity_differs"] is True
    assert report["unit_mass_preflight"]["kekule_support_passed"] is True
    assert report["unit_mass_preflight"]["slot_gate_passed"] is True
    assert report["unit_mass_preflight"]["unit_law_passed"] is True
    assert report["scored_family_conditional"]["status"] == "NOT_REQUESTED"


@pytest.mark.parametrize(
    "smiles",
    (
        "Cc1cccc(Cl)c1",
        "COc1cc(Cl)ccc1N",
    ),
)
def test_asymmetric_support_mismatch_is_explicit_no_go_and_blocks_scoring(
    smiles: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden_scoring(*_args, **_kwargs):
        raise AssertionError("support preflight must block every neural scorer call")

    monkeypatch.setattr(
        diagnostic,
        "enumerate_factorized_marked_law",
        forbidden_scoring,
    )
    report = diagnostic.run_cycle_open_kekule_invariance_diagnostic(
        _state(smiles),
        models={"probe": SimpleNamespace(cycle_open_scorer_mode="exact_bond_contextual_probe")},
    )
    comparison = report["unit_mass_preflight"]["comparisons"]["original_vs_alternate"]

    assert report["status"] == "NO_GO_ALTERNATE_KEKULE_SUPPORT_MISMATCH"
    assert report["scorer_promotion_authorized"] is False
    assert report["unit_mass_preflight"]["kekule_support_passed"] is False
    assert comparison["support_identical"] is False
    assert comparison["l1_probability_residual"] == pytest.approx(2.0)
    assert comparison["left_only_successors"]
    assert comparison["right_only_successors"]
    assert report["scored_family_conditional"]["was_run"] is False
    assert report["scored_family_conditional"]["status"] == (
        "BLOCKED_BY_MODEL_INDEPENDENT_SUPPORT_OR_SLOT_MISMATCH"
    )


@pytest.fixture(scope="module")
def cycle_open_models():
    catalog = TypedRingCatalog((), (), ())
    models = {}
    for offset, mode in enumerate(("pair_linear", "exact_bond_contextual_probe")):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(20260731 + offset)
            models[mode] = FactorizedTraceletRateModel(
                catalog,
                hidden_dim=12,
                message_passing_steps=2,
                mark_dim=8,
                enable_cycle_ops=True,
                cycle_open_scorer_mode=mode,
                enable_ring_grow_macro=False,
                enable_ring_system_delete=False,
                atom_vocabulary=ORGANIC_VOCABULARY,
            ).eval()
    return models


def test_legacy_and_probe_are_scored_only_after_symmetric_support_passes(
    cycle_open_models,
) -> None:
    report = diagnostic.run_cycle_open_kekule_invariance_diagnostic(
        _state("c1ccccc1"),
        models=cycle_open_models,
    )

    assert report["status"] == "BOUNDED_PASS_NON_AUTHORIZING"
    assert report["scored_family_conditional"]["status"] == "PASS"
    assert report["scored_family_conditional"]["was_run"] is True
    assert set(report["scored_family_conditional"]["models"]) == {
        "pair_linear",
        "exact_bond_contextual_probe",
    }
    assert all(row["passes"] for row in report["scored_family_conditional"]["models"].values())
    assert report["scorer_promotion_authorized"] is False


def test_unsupported_scorer_mode_fails_loudly() -> None:
    with pytest.raises(
        diagnostic.CycleOpenKekuleInvarianceError,
        match="unsupported scorer modes",
    ):
        diagnostic.run_cycle_open_kekule_invariance_diagnostic(
            _state("c1ccccc1"),
            models={"bad": SimpleNamespace(cycle_open_scorer_mode="future_unfrozen_mode")},
        )
