from __future__ import annotations

import copy
import hashlib
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from compose_v4.chem.molecular_graph import ELEMENT_TO_IDX, smiles_to_molecular_graph
from compose_v4.chem.source_prior import DegreeBoundedCarbonTreePrior
from compose_v4.chem.state import pad_molecular_graph
from compose_v4.control.reference_guidance import GuidanceConfig, guide_panel
from compose_v4.control.reference_programs import (
    FrozenProgramReference,
    ProgramInput,
    pmo_program_input,
    t4_program_input,
)
from compose_v4.experiments.production_successor_kernel import canonical_successor_result
from compose_v4.model.factorized_tracelet_rate_model import FactorizedTraceletRateModel
from compose_v4.model.reference_checkpoint import load_frozen_reference
from compose_v4.rewrite.action_codec_v4 import encode_action
from compose_v4.rewrite.kernel import canonical_state_key, editing_v2_semantic_rewrite_system
from compose_v4.rewrite.operators import AtomInsert
from compose_v4.rewrite.trace_shard import encode_state
from compose_v4.rewrite.typed_ring_catalog import TypedRingCatalog, ring_catalog_fingerprint


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def checkpoint(tmp_path):
    catalog = TypedRingCatalog((), (), ())
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(23)
        model = FactorizedTraceletRateModel(
            catalog,
            hidden_dim=16,
            message_passing_steps=1,
            enable_cycle_ops=True,
            enable_ring_grow_macro=False,
        )
    payload = {
        "checkpoint_kind": "selected_evaluation_model",
        "max_atoms": 40,
        "training_backend": "factorized_marks",
        "source_prior": "carbon_tree",
        "ring_catalog": catalog,
        "tree_source_prior": DegreeBoundedCarbonTreePrior(sizes=(2, 3)),
        "hidden_dim": 16,
        "message_passing_steps": 1,
        "enable_cycle_ops": True,
        "enable_ring_grow_macro": False,
        "state_dict": model.state_dict(),
    }
    path = tmp_path / "reference.pt"
    torch.save(payload, path)
    return path, payload, ring_catalog_fingerprint(catalog)


@pytest.fixture
def loaded(checkpoint):
    path, _, catalog = checkpoint
    return load_frozen_reference(
        path, expected_sha256=file_hash(path), expected_catalog_fingerprint=catalog
    )


def example_program(element="O"):
    source = pad_molecular_graph(smiles_to_molecular_graph("CC"), 12)
    hydrogens = {"O": 1, "N": 2}[element]
    action = AtomInsert(2, ELEMENT_TO_IDX[element], 0, hydrogens, ((1, 1),))
    successor = editing_v2_semantic_rewrite_system().apply(source, "atom_insert", action)
    trace = {
        "actions": [encode_action("atom_insert", action)],
        "states": [encode_state(source), encode_state(successor)],
    }
    endpoint = canonical_state_key(successor)
    return source, trace, endpoint


def programs():
    return tuple(
        ProgramInput.from_trace(element, endpoint, trace)
        for element in ("O", "N")
        for _, trace, endpoint in [example_program(element)]
    )


def test_loader_matches_existing_reconstruction_and_preserves_rng(checkpoint):
    from scripts.evaluate_tracelet_rollouts import load_factorized_rollout_checkpoint

    path, _, catalog = checkpoint
    state = torch.random.get_rng_state().clone()
    reference = load_frozen_reference(
        path, expected_sha256=file_hash(path), expected_catalog_fingerprint=catalog
    )
    assert torch.equal(state, torch.random.get_rng_state())
    old, _ = load_factorized_rollout_checkpoint(path)
    for key, value in old.state_dict().items():
        assert torch.equal(value, reference.model.state_dict()[key])
    for key in (
        "editing_process_semantics",
        "enable_cycle_ops",
        "enable_ring_grow_macro",
        "enable_ring_restates",
        "enable_ring_system_delete",
        "atom_delete_action_semantics",
    ):
        assert getattr(old, key) == getattr(reference.model, key)
    assert not reference.model.training
    assert all(not p.requires_grad for p in reference.model.parameters())


def test_wrong_hash_fails_before_deserialization(checkpoint, monkeypatch):
    path, _, catalog = checkpoint

    def forbidden(*args, **kwargs):
        raise AssertionError("must not unpickle unverified data")

    monkeypatch.setattr(torch, "load", forbidden)
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        load_frozen_reference(path, expected_sha256="0" * 64, expected_catalog_fingerprint=catalog)


def test_catalog_drift_fails(checkpoint):
    path, _, _ = checkpoint
    with pytest.raises(ValueError, match="catalog drift"):
        load_frozen_reference(
            path, expected_sha256=file_hash(path), expected_catalog_fingerprint="0" * 16
        )


@pytest.mark.parametrize(
    "key,value,match",
    [
        ("editing_process_semantics", "semantic_editing_v2", "non-legacy"),
        ("atom_delete_action_semantics", "different", "unsupported checkpoint"),
        ("corrupted_prior_mix", "false", "literal Boolean"),
        ("enable_cycle_ops", 1, "literal Boolean"),
        ("max_atoms", None, "max_atoms"),
    ],
)
def test_loader_does_not_guess_contradictory_metadata(checkpoint, key, value, match):
    path, payload, catalog = checkpoint
    payload[key] = value
    torch.save(payload, path)
    with pytest.raises(ValueError, match=match):
        load_frozen_reference(
            path, expected_sha256=file_hash(path), expected_catalog_fingerprint=catalog
        )


def test_real_model_scores_exact_programs_without_state_or_rng_changes(loaded):
    reference = FrozenProgramReference(loaded)
    before = {key: value.clone() for key, value in loaded.model.state_dict().items()}
    random_state = torch.random.get_rng_state().clone()
    first, second = reference.score(programs()), reference.score(programs())
    assert first == second
    assert all(item.status == "scored" and item.value < 0 for item in first.scores)
    assert torch.equal(random_state, torch.random.get_rng_state())
    assert all(torch.equal(before[key], value) for key, value in loaded.model.state_dict().items())
    assert all(p.grad is None for p in loaded.model.parameters())


def test_reference_score_aggregates_marks_by_canonical_successor(loaded):
    source, trace, endpoint = example_program()
    program = ProgramInput.from_trace("ethanol", endpoint, trace)
    score = FrozenProgramReference(loaded).score((program,)).scores[0]
    successors = canonical_successor_result(loaded.model, source, 0.5).batch.successors
    target = next(successor for successor in successors if successor.key == endpoint)
    assert target.alias_count > 1
    assert score.status == "scored"
    assert score.value == pytest.approx(math.log(target.probability), abs=2e-5)


def test_batch_size_preserves_score(loaded):
    assert FrozenProgramReference(loaded, batch_size=1).score(programs()) == FrozenProgramReference(
        loaded, batch_size=16
    ).score(programs())


def test_input_snapshot_does_not_alias_callers_trace():
    _, trace, endpoint = example_program()
    program = ProgramInput.from_trace("x", endpoint, trace)
    digest = program.trace_sha256
    trace["states"][0]["atom_types"][0] = 0
    assert program.trace_sha256 == digest
    assert json.loads(program.trace_json)["states"][0]["atom_types"][0] != 0


def test_pmo_and_t4_adapters_use_the_same_actual_program(loaded):
    source, trace, endpoint = example_program()
    pmo = pmo_program_input(
        {
            "candidate_id": "x",
            "source_state": trace["states"][0],
            "trace": trace,
            "endpoint": endpoint,
        }
    )
    t4 = t4_program_input(
        {
            "smiles": endpoint,
            "realized_actions": trace["actions"],
            "realized_endpoint_key": endpoint,
        },
        candidate_id="x",
        source=source,
    )
    assert pmo == t4
    assert FrozenProgramReference(loaded).score((pmo,)) == FrozenProgramReference(loaded).score(
        (t4,)
    )


def test_t4_missing_actions_is_not_a_fabricated_score(loaded):
    source, _, endpoint = example_program()
    program = t4_program_input({"smiles": endpoint}, candidate_id="x", source=source)
    score = FrozenProgramReference(loaded).score((program,)).scores[0]
    assert score.status == "missing_trace" and score.value is None


def test_wrong_pmo_source_is_rejected():
    _, trace, endpoint = example_program()
    with pytest.raises(ValueError, match="source_state differs"):
        pmo_program_input(
            {"candidate_id": "x", "source_state": {}, "trace": trace, "endpoint": endpoint}
        )


def test_t4_cannot_score_original_program_for_changed_endpoint():
    source, trace, _ = example_program()
    with pytest.raises(ValueError, match="do not produce the proposed endpoint"):
        t4_program_input(
            {"smiles": "CCN", "realized_endpoint_key": "CCN", "realized_actions": trace["actions"]},
            candidate_id="x",
            source=source,
        )


def test_false_intermediate_state_fails_exact_replay(loaded):
    _, trace, endpoint = example_program()
    trace["states"][0] = copy.deepcopy(trace["states"][1])
    with pytest.raises(ValueError):
        FrozenProgramReference(loaded).score((ProgramInput.from_trace("x", endpoint, trace),))


def test_wrong_endpoint_and_empty_program_are_rejected(loaded):
    _, trace, _ = example_program()
    scorer = FrozenProgramReference(loaded)
    with pytest.raises(ValueError, match="trace endpoint differs"):
        scorer.score((ProgramInput.from_trace("x", "CCN", trace),))
    with pytest.raises(ValueError, match="nonempty program"):
        scorer.score(
            (ProgramInput.from_trace("x", "CC", {"actions": [], "states": trace["states"][:1]}),)
        )


@pytest.mark.parametrize("mutation", ["train", "grad"])
def test_unfrozen_reference_is_rejected(loaded, mutation):
    if mutation == "train":
        loaded.model.train()
    else:
        next(loaded.model.parameters()).requires_grad_(True)
    with pytest.raises(ValueError, match="must remain frozen"):
        FrozenProgramReference(loaded).score(programs())


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_invalid_numerics_are_not_support_fallbacks(loaded, monkeypatch, value):
    monkeypatch.setattr(
        "compose_v4.control.reference_programs.forward_teacher_successor_batch",
        lambda model, batch, fibers: SimpleNamespace(
            selected_productive_successor_log_probability=torch.tensor([value])
        ),
    )
    with pytest.raises(FloatingPointError, match="invalid numerical"):
        FrozenProgramReference(loaded).score(programs())


def test_zero_canonical_successor_mass_is_reported(loaded, monkeypatch):
    monkeypatch.setattr(
        "compose_v4.control.reference_programs.forward_teacher_successor_batch",
        lambda model, batch, fibers: SimpleNamespace(
            selected_productive_successor_log_probability=torch.tensor([-float("inf")])
        ),
    )
    scored = FrozenProgramReference(loaded).score(programs())
    assert all(
        item.status == "unsupported_native_mark" and item.value is None for item in scored.scores
    )
    result = guide_panel(
        ("O", "N"), (0.5, 0.5), score=lambda: scored, config=GuidanceConfig("active", strength=0.25)
    )
    assert result.outcome == "baseline_fallback"


@pytest.mark.external_artifact
def test_shared_nll_checkpoint_real_inference_and_active_influence():
    root = Path(__file__).resolve().parents[1]
    reference = json.loads((root / "experiments/reference/model.json").read_text())
    location = Path(
        os.environ.get(
            "COMPOSE_REFERENCE_CHECKPOINT", root / "local_assets/fragments/r_theta_nll.pt"
        )
    )
    if not location.is_file():
        pytest.skip("fetch the NLL reference checkpoint through Git LFS")
    loaded = load_frozen_reference(
        location,
        expected_sha256=reference["checkpoint"]["sha256"],
        expected_catalog_fingerprint=reference["catalog"]["fingerprint"],
        catalog_path=root / "local_assets/fragments/catalog.json",
        expected_catalog_sha256=reference["catalog"]["sha256"],
    )
    scored = FrozenProgramReference(loaded).score(programs())
    assert all(item.status == "scored" for item in scored.scores)
    assert scored.scores[0].value != scored.scores[1].value
    result = guide_panel(
        ("O", "N"), (0.5, 0.5), score=lambda: scored, config=GuidanceConfig("active", strength=0.25)
    )
    assert result.probabilities_changed
