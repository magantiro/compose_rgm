"""Bounded reconstruction mechanics, not a generalization evaluation."""

import numpy as np
import torch

from compose_v4.control import winner_imitation as imitation
from compose_v4.experiments.winner_paths import PathConfig, find_path
from compose_v4.rewrite.action_codec_v4 import decode_action
from compose_v4.rewrite.kernel import canonical_state_key, editing_v2_semantic_rewrite_system
from compose_v4.rewrite.trace_shard import decode_state


def fixture():
    route = find_path("CC", "CCC", PathConfig())
    return decode_state(route["source_state"]), decode_state(route["states"][-1])


def test_candidates_execute_and_ranker_receives_gradients():
    source, goal = fixture()
    system = editing_v2_semantic_rewrite_system()
    candidates = imitation.proposals(source, goal, frozenset(), system)
    assert candidates
    for candidate in candidates:
        family, action = decode_action(candidate.mark)
        assert canonical_state_key(system.apply(source, family, action)) == canonical_state_key(
            candidate.graph
        )
    x = imitation.features(source, goal, frozenset(), candidates)
    assert x.dtype == np.float32 and np.isfinite(x).all()
    model = imitation.ImitationRanker(x.shape[1], hidden=8)
    scores = model(torch.from_numpy(x))
    assert scores.shape == (len(candidates),)
    scores.sum().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_missing_proposals_are_not_replaced_by_teacher(monkeypatch):
    source, goal = fixture()
    monkeypatch.setattr(imitation, "_actions", lambda *args: iter(()))
    system = editing_v2_semantic_rewrite_system()
    assert imitation.proposals(source, goal, frozenset(), system) == []
    result = imitation.rollout(source, goal, frozenset(), system, None)
    assert result["status"] == "no_valid_proposal" and result["actions"] == []


def test_rollout_respects_budget_and_exact_goal():
    source, goal = fixture()
    system = editing_v2_semantic_rewrite_system()
    assert imitation.rollout(source, goal, frozenset(), system, None, 0)["status"] == "budget"
    assert imitation.rollout(goal, goal, frozenset(), system, None, 0)["status"] == "reconstructed"
