from __future__ import annotations

import gzip
import json

from compose_v4.control.docking_value import identity
from compose_v4.experiments.pmo_route_policy_quality import (
    _dependency_target,
    decisions_from_dataset,
    predeclared_family_folds,
    runtime_checkpoint_leakage,
)
from tools.pmo_route_policy_quality import run


def test_created_handle_dependency_is_relative_and_exact():
    assert (
        _dependency_target(
            {
                "origin": "route_created",
                "created_ordinal": 7,
                "creation_lag": 3,
            }
        )
        == "route_created_lag_3"
    )
    assert _dependency_target({"origin": "preexisting"}) == "preexisting"


def test_frozen_task_family_folds_have_zero_lineage_leakage():
    with gzip.open(
        "diagnostics/pmo_route_distillation/attempt_1/training_dataset.json.gz",
        "rt",
    ) as handle:
        dataset = json.load(handle)["payload"]
    decisions = decisions_from_dataset(dataset)
    folds = predeclared_family_folds(
        decisions,
        [
            {
                "fold": 0,
                "held_out_task_families": ["bioactivity", "formula"],
                "expected_test_decisions": 2052,
            },
            {
                "fold": 1,
                "held_out_task_families": ["multi_property", "rediscovery"],
                "expected_test_decisions": 1797,
            },
            {
                "fold": 2,
                "held_out_task_families": ["druglikeness", "mpo", "similarity"],
                "expected_test_decisions": 2294,
            },
        ],
    )

    assert sum(len(row["test_decisions"]) for row in folds) == 6143
    assert all(
        not (set(row["train_lineages"]) & set(row["test_lineages"])) for row in folds
    )


def test_runtime_checkpoint_leakage_scan_fails_closed():
    checkpoint = {"heads": {"rule": {"all": {"classes": ["atom_insert"]}}}}
    assert runtime_checkpoint_leakage(checkpoint, {"celecoxib_rediscovery"}) == {
        "forbidden_key_hits": [],
        "forbidden_teacher_value_hits": [],
    }
    checkpoint["task"] = "celecoxib_rediscovery"
    audit = runtime_checkpoint_leakage(checkpoint, {"celecoxib_rediscovery"})
    assert audit["forbidden_key_hits"] == ["task"]
    assert audit["forbidden_teacher_value_hits"] == ["celecoxib_rediscovery"]


def test_sealed_pmo_policy_quality_comparison_is_local_only_and_zero_oracle(tmp_path):
    output = tmp_path / "attempt_1"
    report = run(output)
    with gzip.open(output / "runtime_fold_checkpoints.json.gz", "rt") as handle:
        runtime_envelope = json.load(handle)

    assert runtime_envelope["payload_sha256"] == identity(runtime_envelope["payload"])
    assert report["census"]["decisions"] == 6143
    assert report["census"]["created_handle_dependent_decisions"] == 1231
    assert report["census"]["created_output_decisions"] == 1531
    assert report["runtime_checkpoint_audit"] == {
        "forbidden_key_hits": [],
        "forbidden_teacher_value_hits": [],
    }
    assert report["complete_program_evaluation"]["coverage"] == 0.0
    assert report["complete_program_evaluation"]["precision"] is None
    assert report["complete_program_evaluation"]["unique_complete_yield"] == 0
    assert report["complete_program_evaluation"]["shortfall"] == 106
    assert report["hybrid_complete_candidate_ranker"]["status"].startswith("not_fit")
    assert report["costs"] == {
        "new_oracle_calls": 0,
        "new_docking_calls": 0,
        "modal_launches": 0,
    }
