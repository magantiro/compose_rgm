from __future__ import annotations

import json

import numpy as np
import pytest

from compose_v4.chem.molecular_graph import MolecularGraph
from compose_v4.control.complete_macro_selector import (
    CompleteMacroSelector,
    compiler_log_work,
    complete_macro_features,
    fit_complete_macro_selector,
    rank_complete_macros,
)
from compose_v4.control.compositional_structural_subgoal_generator import (
    generate_compositional_patches,
)
from compose_v4.control.docking_value import identity
from compose_v4.control.structural_subgoal_policy import ContextSubgoalRanker
from tools import t4_complete_macro_selector as selector_tool
from tools.t4_complete_macro_selector import _contract, _load_catalog


def _ethane() -> MolecularGraph:
    atom_types = np.zeros(48, dtype=np.int32)
    charges = np.zeros(48, dtype=np.int32)
    hydrogens = np.zeros(48, dtype=np.int32)
    bonds = np.zeros((48, 48), dtype=np.int32)
    atom_types[:2] = 2
    hydrogens[:2] = 3
    bonds[0, 1] = bonds[1, 0] = 1
    return MolecularGraph(atom_types, charges, hydrogens, bonds)


def _candidate_pool() -> tuple[MolecularGraph, list[dict]]:
    source = _ethane()
    candidates, _ = generate_compositional_patches(
        source,
        None,
        pool_size=4,
        first_event_beam=8,
        second_event_beam=1,
        second_event_expansion_per_prefix=2,
        per_rule_first_event_cap=2,
    )
    assert len(candidates) >= 3
    return source, [row.payload() for row in candidates]


def test_compiler_work_refuses_nonexact_or_teacher_assisted_receipts() -> None:
    _, candidates = _candidate_pool()
    assert compiler_log_work(candidates[0]) > 0

    broken = json.loads(json.dumps(candidates[0]))
    broken["realization"]["primitive_teacher_actions_used"] = 1
    with pytest.raises(ValueError, match="teacher-free exact realization"):
        compiler_log_work(broken)


def test_pairwise_selector_fit_is_source_balanced_and_round_trips() -> None:
    positives = [
        ("a", np.asarray([1.0, 0.0], dtype=np.float32)),
        ("b", np.asarray([0.8, 0.2], dtype=np.float32)),
    ]
    negatives = {
        "a": [(np.asarray([0.0, 1.0], dtype=np.float32), 2.0)],
        "b": [(np.asarray([0.1, 0.9], dtype=np.float32), 3.0)],
    }

    model, report = fit_complete_macro_selector(
        positives,
        negatives,
        updates=20,
        learning_rate=0.03,
        l2=0.01,
        seed=7,
        compiler_cost_weight=0.05,
    )

    assert report["ranker"]["pairs"] == 2
    assert CompleteMacroSelector.from_checkpoint(model.checkpoint()) == model
    serialized = json.dumps(model.checkpoint(), sort_keys=True).lower()
    for forbidden in ("source_group", "endpoint_state", "goal_payload", "patch_ids", "actions"):
        assert forbidden not in serialized


def test_ranking_preserves_pool_membership_and_puts_unique_macros_first() -> None:
    source, candidates = _candidate_pool()
    dimension = len(
        complete_macro_features(source, candidates[0]["endpoint_state"], candidates[0]["goal"])
    )
    ranker = ContextSubgoalRanker(
        tuple(np.zeros(dimension)),
        tuple(np.ones(dimension)),
        tuple(np.zeros(dimension)),
        "ranker-test",
    )
    selector = CompleteMacroSelector(ranker, 0.0, 1.0, 0.0, "selector-test")
    candidates[1]["patch_ids"] = list(candidates[0]["patch_ids"])

    ranked = rank_complete_macros(source, candidates, selector)

    assert {row["candidate_identity"] for row in ranked} == {identity(row) for row in candidates}
    signatures = [row["complete_macro_signature"] for row in ranked]
    first_repeat = next(
        (index for index, signature in enumerate(signatures) if signature in signatures[:index]),
        len(signatures),
    )
    assert len(set(signatures[:first_repeat])) == len(set(signatures))


def test_frozen_contract_and_catalog_bind_15_sources_and_77_routes() -> None:
    contract = _contract()
    lock, evaluations, catalog = _load_catalog(contract)

    assert contract["costs"]["candidate_generation_calls_authorized"] == 0
    assert lock["teacher_fields_present"] is False
    assert set(evaluations) == {0, 1, 2}
    assert len(catalog) == 15
    assert sum(len(row["teachers"]) for row in catalog.values()) == 77


def test_offline_ordering_quality_suppresses_degenerate_zero_time_rates(monkeypatch) -> None:
    report = {
        "schema_version": "route_proposal_quality_report_v1",
        "policies": {
            "p": {
                "aggregate": {
                    "attempts_per_second": 10**15,
                    "unique_endpoints_per_second": 10**15,
                }
            }
        },
        "limitations": [],
        "report_sha256": "old",
    }
    monkeypatch.setattr(selector_tool, "evaluate_quality", lambda payload, cutoffs: report)

    observed = selector_tool._offline_ordering_quality({}, cutoffs=(8, 16, 32))

    assert observed["policies"]["p"]["aggregate"]["attempts_per_second"] is None
    assert observed["policies"]["p"]["aggregate"]["unique_endpoints_per_second"] is None
    body = {key: value for key, value in observed.items() if key != "report_sha256"}
    assert observed["report_sha256"] == identity(body)
