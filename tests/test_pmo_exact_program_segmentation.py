from __future__ import annotations

import gzip
import json
from pathlib import Path

from compose_v4.control.docking_value import identity
from compose_v4.experiments.pmo_exact_program_segmentation import (
    SegmentationConfig,
    segment_exact_trace,
)
from tools.pmo_exact_program_segmentation import _generate_panel, _load_contract
from tools.pmo_route_distillation import collect_routes, load_contract


def test_one_locked_trace_segments_with_exact_replay_and_complete_partition():
    route = collect_routes(load_contract())[0][0]
    result = segment_exact_trace(route.states, route.actions, SegmentationConfig())

    assert result["exact_replay_segments"] == len(result["segments"])
    assert result["segments"][0]["start"] == 0
    assert result["segments"][-1]["stop"] == len(route.actions)
    assert all(
        left["stop"] == right["start"]
        for left, right in zip(result["segments"], result["segments"][1:])
    )
    assert all(segment["primitive_count"] <= 8 for segment in result["segments"])


def test_generic_terminal_self_event_is_a_rejected_panel_attempt(monkeypatch):
    route = collect_routes(load_contract())[0][0]

    def self_event(source, rng, **kwargs):
        del rng, kwargs
        state = route.states[0]
        return source, object(), (), {"states": [state], "actions": []}, {}

    monkeypatch.setattr(
        "tools.pmo_exact_program_segmentation.synthesize_dynamic_program_v1",
        self_event,
    )
    panel = _generate_panel(
        route.lineage_identity,
        [
            {
                "source_state": route.states[0],
                "states": list(route.states),
                "trace_identity": route.trace_identity,
                "task_family": route.task_family,
                "test_fold": 0,
            }
        ],
        {
            "campaign_seed": 20260914,
            "attempts_per_lineage": 1,
            "max_modules": 3,
            "max_primitives": 32,
            "max_blocks": 8,
            "transformation_equivalence_radius": 2,
        },
    )

    assert panel["attempts"] == [
        {
            "attempt_id": "generic-0000",
            "rank": 1,
            "status": "rejected",
            "failure": "canonical_self_event",
            "endpoint_state": None,
        }
    ]


def test_sealed_result_preserves_split_replay_negative_yield_and_no_runtime_checkpoint():
    output = "diagnostics/pmo_exact_program_segmentation/attempt_2"
    with open(f"{output}/result.json") as handle:
        result_envelope = json.load(handle)
    with gzip.open(f"{output}/training_exact_segmented_corpus.json.gz", "rt") as handle:
        corpus_envelope = json.load(handle)

    result = result_envelope["payload"]
    corpus = corpus_envelope["payload"]
    assert result_envelope["payload_sha256"] == identity(result)
    assert corpus_envelope["payload_sha256"] == identity(corpus)
    assert result["gates"] == {
        "split_frozen_before_derivation": True,
        "zero_lineage_leakage": True,
        "exact_segment_replay_precision_one": True,
        "nonzero_multi_action_coverage": True,
        "nonzero_generic_negative_yield": True,
        "task_score_labels_absent": True,
        "runtime_checkpoint_absent": True,
    }
    assert result["segmentation"]["overall"]["complete_representation_routes"] == 8
    assert (
        result["next_policy_gate"][
            "every_held_out_fold_has_complete_representation_support"
        ]
        is False
    )
    assert result["negative_panels"]["attempts"] == 576
    assert result["negative_panels"]["unique_relation_counts"] == {
        "generic_negative": 559
    }
    assert corpus["runtime_checkpoint"] is False
    assert corpus["task_scores_loaded"] is False
    assert corpus["new_oracle_calls"] == 0


def test_contract_is_self_hashed_and_prohibits_runtime_checkpoint():
    contract = _load_contract(Path.cwd())
    assert contract["outputs"]["runtime_checkpoint"] is None
