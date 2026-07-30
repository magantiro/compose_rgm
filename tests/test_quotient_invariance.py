"""E5 canonical-quotient invariance over fixtures and real production rows."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from compose_v4.chem.molecular_graph import smiles_to_molecular_graph
from compose_v4.chem.state import pad_molecular_graph
from compose_v4.experiments.production_successor_kernel import (
    canonical_successor_result,
)
from compose_v4.experiments.quotient_invariance import (
    E5_RESULT_STATUS,
    EncodedMark,
    QuotientInvarianceError,
    artifact_self_hash,
    build_fixture_model,
    contract_self_hash,
    freeze_e5_artifact,
    load_e5_contract,
    mark_power_then_quotient,
    mark_top_k_then_quotient,
    max_probability_residual,
    quotient_mark_encoding,
    refine_within_fibers,
    run_e5_foundation,
    successor_doob_control,
    validate_e5_artifact,
)
from compose_v4.experiments.reference_successor_kernel import (
    compare_against_reference,
    reference_successor_batch,
)
from compose_v4.rewrite.kernel import de_novo_rewrite_system

_ROOT = Path(__file__).resolve().parent.parent
_CONTRACT_PATH = _ROOT / "configs/tasks/quotient.development.json"
_REGISTRY_PATH = _ROOT / "configs/experiment_registry.yaml"


@pytest.fixture(scope="module")
def contract():
    return load_e5_contract(_CONTRACT_PATH, registry_path=_REGISTRY_PATH)


@pytest.fixture(scope="module")
def model(contract):
    return build_fixture_model(contract)


def test_contract_is_self_hashed_and_bound_to_registry(contract):
    assert contract["contract_sha256"] == contract_self_hash(contract)
    assert contract["paper_claim_authorized"] is False
    assert contract["checkpoint_required"] is False


def test_contract_tampering_fails_closed(tmp_path):
    payload = json.loads(_CONTRACT_PATH.read_text())
    payload["sampling"]["draws_per_state"] += 1
    path = tmp_path / "tampered.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(QuotientInvarianceError, match="self-hash"):
        load_e5_contract(path, registry_path=_REGISTRY_PATH)


def test_within_fiber_refinement_preserves_successor_law_and_control():
    coarse = (
        EncodedMark("a", "y", 0.3),
        EncodedMark("b", "z", 0.7),
    )
    refined = refine_within_fibers(coarse, (0.2, 0.3, 0.5))
    coarse_law = quotient_mark_encoding(coarse)
    refined_law = quotient_mark_encoding(refined)
    values = {"y": 1.7, "z": 0.4}

    assert len(refined) == 3 * len(coarse)
    assert max_probability_residual(coarse_law, refined_law) <= 1e-12
    assert max_probability_residual(
        successor_doob_control(coarse_law, values),
        successor_doob_control(refined_law, values),
    ) <= 1e-12


def test_mark_level_power_and_top_k_are_explicit_negative_counterexamples():
    coarse = (
        EncodedMark("y.0", "y", 0.5),
        EncodedMark("z.0", "z", 0.5),
    )
    refined = (
        EncodedMark("y.0", "y", 0.125),
        EncodedMark("y.1", "y", 0.125),
        EncodedMark("y.2", "y", 0.125),
        EncodedMark("y.3", "y", 0.125),
        EncodedMark("z.0", "z", 0.5),
    )

    assert quotient_mark_encoding(coarse) == pytest.approx(
        quotient_mark_encoding(refined),
        abs=1e-12,
    )
    assert mark_power_then_quotient(coarse, beta=2.0)["y"] == pytest.approx(0.5)
    assert mark_power_then_quotient(refined, beta=2.0)["y"] == pytest.approx(0.2)
    assert mark_top_k_then_quotient(coarse, k=1) == {"y": pytest.approx(1.0)}
    assert mark_top_k_then_quotient(refined, k=1) == {"z": pytest.approx(1.0)}


@pytest.mark.parametrize(
    "smiles",
    ("CCO", "CC(C)C", "c1ccccc1", "C1CCCCC1", "C1CCNCC1"),
)
def test_authoritative_production_rows_match_fresh_dictionary_oracle(
    contract,
    model,
    smiles,
):
    state = pad_molecular_graph(
        smiles_to_molecular_graph(smiles),
        int(contract["model_fixture"]["max_atoms"]),
    )
    result = canonical_successor_result(
        model,
        state,
        float(contract["model_fixture"]["time"]),
    )
    reference = reference_successor_batch(
        state,
        [
            (
                mark.executor_rule_name,
                mark.action,
                mark.probability,
            )
            for mark in result.marked_law.marks
        ],
        system=de_novo_rewrite_system(),
        identity=result.batch.identity,
    )
    produced_probabilities = {
        successor.key: successor.probability
        for successor in result.batch.successors
    }
    reference_probabilities = {
        successor.key: successor.probability
        for successor in reference.successors
    }
    assert compare_against_reference(
        produced_probabilities,
        reference_probabilities,
        tolerance=float(
            contract["numeric_tolerances"]["production_vs_dictionary_oracle"]
        ),
    ) == []
    assert {
        successor.key: successor.alias_count
        for successor in result.batch.successors
    } == {
        successor.key: successor.alias_count
        for successor in reference.successors
    }
    assert result.batch.virtual_mass == pytest.approx(reference.virtual_mass, abs=2e-7)


def test_bounded_real_panel_passes_and_artifact_is_non_claiming(contract):
    artifact = run_e5_foundation(contract, registry_path=_REGISTRY_PATH)
    validate_e5_artifact(
        artifact,
        contract=contract,
        registry_path=_REGISTRY_PATH,
    )

    assert artifact["status"] == E5_RESULT_STATUS
    assert artifact["paper_claim_authorized"] is False
    assert artifact["checkpoint_used"] is None
    assert artifact["optimizer_steps"] == 0
    assert all(artifact["checks"].values())
    assert artifact["summary"]["maximum_alias_multiplicity"] >= 2
    assert artifact["summary"]["mark_level_power_divergence"] > 0.0
    assert artifact["summary"]["mark_level_top_k_divergence"] > 0.0
    assert artifact["artifact_sha256"] == artifact_self_hash(artifact)


def test_artifact_self_hash_detects_tampering(contract):
    artifact = run_e5_foundation(contract, registry_path=_REGISTRY_PATH)
    tampered = copy.deepcopy(artifact)
    tampered["summary"]["state_count"] += 1
    assert artifact_self_hash(tampered) != tampered["artifact_sha256"]
    with pytest.raises(QuotientInvarianceError, match="self-hash"):
        validate_e5_artifact(
            tampered,
            contract=contract,
            registry_path=_REGISTRY_PATH,
        )


def test_artifact_provenance_tampering_fails_closed(contract):
    artifact = run_e5_foundation(contract, registry_path=_REGISTRY_PATH)
    tampered = copy.deepcopy(artifact)
    tampered["provenance"]["registry_file_sha256"] = "0" * 64
    tampered["artifact_sha256"] = artifact_self_hash(tampered)
    with pytest.raises(QuotientInvarianceError, match="provenance"):
        validate_e5_artifact(
            tampered,
            contract=contract,
            registry_path=_REGISTRY_PATH,
        )


def test_artifact_freeze_is_immutable(contract, tmp_path):
    artifact = run_e5_foundation(contract, registry_path=_REGISTRY_PATH)
    path = tmp_path / "e5.json"
    freeze_e5_artifact(
        artifact,
        path,
        contract=contract,
        registry_path=_REGISTRY_PATH,
    )
    first = path.read_bytes()
    freeze_e5_artifact(
        artifact,
        path,
        contract=contract,
        registry_path=_REGISTRY_PATH,
    )
    assert path.read_bytes() == first

    collision = copy.deepcopy(artifact)
    collision["summary"]["state_count"] += 1
    collision["artifact_sha256"] = artifact_self_hash(collision)
    with pytest.raises(QuotientInvarianceError, match="provenance|check|status|collision"):
        freeze_e5_artifact(
            collision,
            path,
            contract=contract,
            registry_path=_REGISTRY_PATH,
        )


def test_result_producer_never_imports_dictionary_oracle():
    for relative in (
        "src/compose_v4/experiments/quotient_invariance.py",
        "scripts/run_e5_quotient_invariance.py",
    ):
        text = (_ROOT / relative).read_text()
        assert "reference_successor_kernel" not in text
        assert "reference_successor_batch" not in text


def test_fixture_sampling_is_reproducible(contract):
    first = run_e5_foundation(contract, registry_path=_REGISTRY_PATH)
    second = run_e5_foundation(contract, registry_path=_REGISTRY_PATH)
    first_sampling = [row["sampling"] for row in first["real_state_rows"]]
    second_sampling = [row["sampling"] for row in second["real_state_rows"]]
    assert first_sampling == second_sampling


def test_invalid_slot_refinement_is_rejected():
    mark = EncodedMark("a", "y", 1.0)
    with pytest.raises(QuotientInvarianceError, match="positive and sum to one"):
        refine_within_fibers((mark,), (0.4, 0.4))


def test_doob_zero_value_row_is_reported_undefined():
    with pytest.raises(QuotientInvarianceError, match="backward value is zero"):
        successor_doob_control({"y": 1.0}, {"y": 0.0})


def test_sampling_artifact_does_not_store_draw_vector(contract):
    artifact = run_e5_foundation(contract, registry_path=_REGISTRY_PATH)
    for row in artifact["real_state_rows"]:
        assert "counts" not in row["sampling"]
        assert "frequencies" not in row["sampling"]
        assert row["sampling"]["draws"] == contract["sampling"]["draws_per_state"]
