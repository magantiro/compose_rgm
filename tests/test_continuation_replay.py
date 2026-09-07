"""Recorded-row replay preserves exact fixture rows and abstains on missing data."""

import json

import numpy as np
import pytest

from compose_v4.experiments.continuation_gate import engineering_fixture
from compose_v4.experiments.continuation_profile import (
    canonical_bytes,
    run_profile,
    sha256_file,
    state_payload,
)
from compose_v4.experiments.continuation_replay import RecordedProfile, RecordedRowMissing, replay
from compose_v4.rewrite.kernel import RewriteSystem

CONFIG = {
    "seed": 1000,
    "samples_per_successor": 32,
    "alpha": 0.05,
    "kappa": 1.0,
    "exploration": 0.1,
    "max_expansions": 1000,
    "max_terminal_evaluations": 1000,
    "max_rollouts": 1000,
}


def recorded(tmp_path):
    node, kernel = engineering_fixture()
    report = run_profile(
        node,
        kernel.enumerate_law,
        kernel.system,
        {"max_executor_applications": 64, "max_expansions": 100, "max_terminal_evaluations": 100},
        tmp_path,
        snapshot_id="recorded-fixture-v1",
        commit_volume=lambda: None,
        progress={},
    )
    inventory = {
        "metadata": {"launch": {"run_id": "recorded-fixture-v1"}},
        "input": {"sha256": {r["path"]: r["sha256"] for r in report["rows"]}},
    }
    return node, kernel, inventory


def test_loaded_rows_match_direct_executor_fixture_without_new_execution(tmp_path, monkeypatch):
    node, kernel, inventory = recorded(tmp_path)
    direct = kernel.row(node)
    profile = RecordedProfile(tmp_path, inventory)
    root = profile.row(state_payload(node))
    assert root.probabilities == direct.probabilities
    assert tuple(canonical_bytes(s) for s in root.successors) == tuple(
        canonical_bytes(state_payload(s)) for s in direct.successors
    )
    # The wire schema uses JSON arrays; Python tuple/list implementation types
    # differ after decoding, but every serialized coordinate must be identical.
    assert canonical_bytes(profile.root) == canonical_bytes(state_payload(node))

    def forbidden(*args, **kwargs):
        pytest.fail("recorded replay must never invoke the molecular executor")

    monkeypatch.setattr(RewriteSystem, "apply", forbidden)
    result = replay(profile, CONFIG)
    assert result["status"] == "guided_sampled"
    assert result["decision"]["decision"]["probabilities"] == pytest.approx((0.95, 0.05))
    assert result["new_executor_calls"] == result["new_oracle_calls"] == 0
    assert not result["missing_rows"]


def test_missing_rows_are_unknown_not_empty_chemical_support(tmp_path):
    node, kernel, inventory = recorded(tmp_path)
    inventory["input"]["sha256"] = {
        p: h
        for p, h in inventory["input"]["sha256"].items()
        if json.loads((tmp_path / p).read_text())["source"]["step"] == 0
    }
    profile = RecordedProfile(tmp_path, inventory)
    with pytest.raises(RecordedRowMissing):
        profile.row(state_payload(kernel.row(node).successors[0]))
    profile = RecordedProfile(tmp_path, inventory)
    result = replay(profile, CONFIG)
    assert result["status"] == "incomplete_recorded_graph"
    assert result["decision"]["estimate"] is None
    assert result["decision"]["decision"]["successor_values"] is None
    assert result["decision"]["decision"]["probabilities"] == pytest.approx((0.5, 0.5))
    assert result["known_empty_row_hits"] == 0
    assert result["work"]["rollouts_completed"] == 0


def test_corruption_and_snapshot_mismatch_are_rejected(tmp_path):
    _, _, inventory = recorded(tmp_path)
    path = next(iter(inventory["input"]["sha256"]))
    inventory["input"]["sha256"][path] = "0" * 64
    with pytest.raises(ValueError, match="physical hash"):
        RecordedProfile(tmp_path, inventory)
    inventory["input"]["sha256"][path] = sha256_file(tmp_path / path)
    inventory["metadata"]["launch"]["run_id"] = "wrong-snapshot"
    with pytest.raises(ValueError, match="schema/snapshot"):
        RecordedProfile(tmp_path, inventory)


def test_exact_payload_key_preserves_context_and_lineage(tmp_path):
    node, _, inventory = recorded(tmp_path)
    original_key = node.key()
    profile = RecordedProfile(tmp_path, inventory)
    for field in ("formal_charges", "implicit_h_counts"):
        changed = state_payload(node)
        changed["graph"][field][0] += 1
        with pytest.raises(RecordedRowMissing):
            profile.row(changed)
    changed = state_payload(node)
    changed["lineage"]["next_id"] += 1
    with pytest.raises(RecordedRowMissing):
        profile.row(changed)
    assert np.isclose(sum(profile.row(state_payload(node)).probabilities), 1)
    assert node.key() == original_key
