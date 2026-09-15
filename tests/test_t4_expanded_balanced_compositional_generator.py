from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from compose_v4.experiments.t4_matched_pilot import unseal
from compose_v4.experiments.t4_route_policy_comparison import predeclared_source_folds
from tools.t4_compositional_structural_subgoal_generator import (
    CONTRACT,
    EXPANDED_CONTRACT,
    SEEDS,
    T4_CONTRACT,
    _balanced_weight_audit,
    _contract,
    _delta06_data,
    _training_events,
)
from tools.t4_program_vocabulary_audit import source_group_map

ROOT = Path(__file__).resolve().parents[1]
DELTA06_CORPUS = ROOT / "diagnostics/t4_delta06_route_corpus/attempt_1/training_corpus.json.gz"
DELTA06_RESULT = ROOT / "diagnostics/t4_delta06_route_corpus/attempt_1/result.json"


def _metadata() -> dict[str, dict]:
    return source_group_map(unseal(T4_CONTRACT), json.loads(SEEDS.read_text()))


def test_expanded_contract_preserves_model_decoder_evaluation_and_gate() -> None:
    baseline = _contract(CONTRACT)
    expanded = _contract(EXPANDED_CONTRACT)

    for key in ("generated_object", "model", "decoder", "evaluation"):
        assert expanded[key] == baseline[key]
    expanded_acceptance = dict(expanded["acceptance"])
    assert expanded_acceptance.pop("unchanged_from_baseline") is True
    assert expanded_acceptance == baseline["acceptance"]
    assert expanded["decoder"]["cutoffs"] == [8, 32, 128]


def test_delta06_adapter_admits_exact_deduplicated_witnesses() -> None:
    traces, records, summary = _delta06_data(
        DELTA06_CORPUS,
        DELTA06_RESULT,
        _metadata(),
    )

    assert summary["endpoint_references"] == 39
    assert summary["converted_routes"] == 32
    assert summary["subgoals"] == 45
    assert summary["abstentions"] == 7
    assert summary["observed_ivg_trajectories"] == 0
    assert not summary["failures"]
    assert len({row["route_id"] for row in traces}) == 32
    assert all(row["domain"] == "t4_delta06_reference" for row in traces)
    assert {row["source_group"] for row in traces} <= set(_metadata())
    assert all(row["route_id"].startswith("delta06:") for row in records)


def test_delta06_training_events_are_hierarchically_balanced() -> None:
    traces, _, _ = _delta06_data(DELTA06_CORPUS, DELTA06_RESULT, _metadata())
    events = []
    for trace in traces:
        rows, _ = _training_events(trace)
        events.extend(rows)

    audit = _balanced_weight_audit(events)

    assert np.isclose(audit["total_mass"], 1.0)
    assert audit["all_levels_equalized"] is True
    assert audit["domain"]["groups"] == 1
    assert audit["route"]["groups"] == 32
    assert audit["component"]["groups"] == 45


def test_delta06_routes_follow_the_unchanged_whole_source_folds() -> None:
    metadata = _metadata()
    traces, _, _ = _delta06_data(DELTA06_CORPUS, DELTA06_RESULT, metadata)

    for split in predeclared_source_folds(metadata):
        train_sources = set(split["train_sources"])
        held_sources = set(split["test_sources"])
        fitted = [row for row in traces if row["source_group"] in train_sources]

        assert train_sources.isdisjoint(held_sources)
        assert fitted
        assert all(row["source_group"] not in held_sources for row in fitted)
