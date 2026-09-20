import json
from pathlib import Path

import numpy as np

from compose_v4.experiments.pmo_complete_route_dynamic_gate import (
    RUNTIME_FORBIDDEN_KEYS,
    _recursive_keys,
    fit_complete_route_checkpoints,
    historical_recursive_fiber_replay,
    load_envelope,
    sample_complete_route_program,
    teacher_forced_support,
)
from compose_v4.experiments.pmo_route_fiber_production_yield import (
    CORPUS,
    LEGAL_RUNTIME,
    _fold_checkpoints,
)
from compose_v4.rewrite.trace_shard import decode_state

ROOT = Path(__file__).resolve().parents[1]


def _corpus():
    return load_envelope(ROOT / CORPUS)


def test_complete_route_checkpoint_is_split_clean_and_teacher_free():
    result = fit_complete_route_checkpoints(_corpus())
    assert result["payload_sha256"]
    assert not (RUNTIME_FORBIDDEN_KEYS & _recursive_keys(result["payload"]))
    assert all(row["lineage_overlap"] == 0 for row in result["audit"].values())
    assert all(checkpoint["maximum_primitives"] == 32 for checkpoint in result["payload"]["folds"])


def test_runtime_support_abstains_for_long_perindopril_routes():
    support = teacher_forced_support(_corpus())
    assert support["celecoxib_rediscovery"]["complete_runtime_routes"] == 15
    assert support["gsk3b"]["complete_runtime_routes"] == 23
    assert support["gsk3b"]["local_decision_only_routes"] == 1
    assert support["perindopril_mpo"]["complete_runtime_routes"] == 0
    assert support["perindopril_mpo"]["local_decision_only_routes"] == 16
    assert support["perindopril_mpo"]["complete_route_abstention"] is True


def test_complete_sampler_executes_same_runtime_with_one_step_fixture():
    corpus = _corpus()
    checkpoints = fit_complete_route_checkpoints(corpus)["payload"]
    fold = 1
    checkpoint = next(row for row in checkpoints["folds"] if row["fold_index"] == fold)
    checkpoint = json.loads(json.dumps(checkpoint))
    checkpoint["lengths"] = list(range(1, 33))
    checkpoint["length_probabilities"] = [1.0] + [0.0] * 31
    route = next(row for row in corpus["routes"] if row["test_fold"] == fold and row["states"])
    legal = _fold_checkpoints(load_envelope(ROOT / LEGAL_RUNTIME))[fold]
    result = sample_complete_route_program(
        decode_state(route["states"][0]), checkpoint, legal, np.random.default_rng(3)
    )
    assert result["status"] == "complete"
    assert result["primitive_count"] == 1
    assert result["exact_replay"] is True


def test_historical_replay_is_recursive_and_hides_scores_until_selection():
    payload = load_envelope(ROOT / "diagnostics/pmo_online_policy/result_sealed.json")
    replay = historical_recursive_fiber_replay(payload, batch=8, maximum_calls=32)
    assert replay["historical_candidate_edges"] > 0
    assert 0 < replay["selected_edges"] <= 32
    assert replay["maximum_selected_generation"] >= 2
    assert replay["multi_generation_lineage_observed"] is True
    assert replay["scores_hidden_until_selection"] is True
    assert replay["new_oracle_calls"] == 0


def test_envelope_rejects_payload_tampering(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text('{"payload":{"x":1},"payload_sha256":"wrong"}')
    try:
        load_envelope(path)
    except ValueError as error:
        assert "payload hash mismatch" in str(error)
    else:
        raise AssertionError("tampered envelope was accepted")
