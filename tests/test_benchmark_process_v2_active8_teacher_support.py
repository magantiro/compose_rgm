"""Focused tests for the Process-V2 teacher-support benchmark."""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import benchmark_process_v2_active8_teacher_support as benchmark  # noqa: E402

from compose_v4.chem.molecular_graph import is_element  # noqa: E402
from compose_v4.data.editing_corpus_contract import ACTIVE8_FAMILIES  # noqa: E402
from compose_v4.data.editing_v2_process_v2_active8_admission import (  # noqa: E402
    ProductionProcessV2BatchedTeacherSupportChecker,
    ProductionProcessV2SemanticExactCandidateChecker,
)


@pytest.fixture(scope="module")
def frozen_runtime():
    return benchmark.build_frozen_runtime()


@pytest.fixture(scope="module")
def jin_panel(frozen_runtime):
    return benchmark.build_jin_panel(
        frozen_runtime,
        csv_path=ROOT / benchmark.JIN_CSV,
        panel_size=16,
    )


def test_frozen_runtime_is_the_declared_production_shape(frozen_runtime) -> None:
    descriptor = benchmark.model_runtime_descriptor(frozen_runtime)
    assert descriptor["initialization_seed"] == 20260730
    assert descriptor["max_atoms"] == 40
    assert descriptor["hidden_dim"] == 256
    assert descriptor["message_passing_steps"] == 6
    assert descriptor["mark_dim"] == 32
    assert descriptor["dtype"] == "torch.float32"
    assert benchmark.CANDIDATE_TIME == 0.5
    assert benchmark.BATCH_SIZES == (8, 16, 32, 64)


def test_jin_panel_has_two_real_teachers_per_family_at_40_slots(jin_panel) -> None:
    entries, selection = jin_panel
    assert len(entries) == 16
    assert selection["per_family"] == 2
    assert selection["active_atom_limitation"] is None
    assert Counter(entry.model_family for entry in entries) == {
        family: 2 for family in ACTIVE8_FAMILIES
    }
    assert tuple(entry.panel_index for entry in entries) == tuple(range(16))
    for entry in entries:
        addressed, step_index = benchmark.query_from_entry(entry)
        assert step_index == 0
        assert addressed.path.state_at(0).n_atoms == 40
        assert int(is_element(addressed.path.state_at(0).atom_types).sum()) <= 40
        assert addressed.address.partition == "external_benchmark_not_selection"


def test_panel_round_trip_refuses_a_resealed_authority_grant(
    jin_panel, tmp_path: Path
) -> None:
    entries, selection = jin_panel
    payload = benchmark.panel_payload(
        entries,
        csv_sha256=benchmark._sha256(ROOT / benchmark.JIN_CSV),
        selection=selection,
    )
    path = tmp_path / "panel.json"
    benchmark._write_json_atomically(path, payload)
    assert benchmark.load_panel(path) == entries

    body = dict(payload)
    body["training_authorized"] = True
    body.pop("panel_sha256")
    tampered = {**body, "panel_sha256": benchmark.canonical_sha256(body)}
    benchmark._write_json_atomically(path, tampered)
    with pytest.raises(benchmark.TeacherSupportBenchmarkError, match="authorizing"):
        benchmark.load_panel(path)


def test_fast_and_slow_exact_common_evidence_match_on_frozen_jin_teacher(
    frozen_runtime, jin_panel
) -> None:
    entries, _selection = jin_panel
    query = benchmark.query_from_entry(entries[0])
    slow = ProductionProcessV2SemanticExactCandidateChecker(
        frozen_runtime.model,
        time=benchmark.CANDIDATE_TIME,
    ).evaluate(*query)
    fast = ProductionProcessV2BatchedTeacherSupportChecker(
        frozen_runtime.model,
        batch_size=8,
        time=benchmark.CANDIDATE_TIME,
    ).evaluate_many((query,))[0]
    assert benchmark.common_evidence(fast) == benchmark.common_evidence(slow)
    assert benchmark.common_evidence(fast)["supported"] is True


def test_report_refuses_one_fast_evidence_mismatch(
    frozen_runtime, jin_panel, monkeypatch: pytest.MonkeyPatch
) -> None:
    entries, selection = jin_panel
    panel = benchmark.panel_payload(
        entries,
        csv_sha256=benchmark._sha256(ROOT / benchmark.JIN_CSV),
        selection=selection,
    )
    measurement = {
        "method": "slow_full_quotient",
        "batch_size": None,
        "query_count": 16,
        "wall_seconds": 2.0,
        "cpu_seconds": 2.0,
        "queries_per_wall_second": 8.0,
        "queries_per_cpu_second": 8.0,
        "process_peak_rss_mb": 512.0,
        "evidence_sha256": "a" * 64,
        "evidence": [],
    }
    fast = {
        **measurement,
        "method": "fast_batched_teacher_support",
        "batch_size": 8,
        "evidence_sha256": "b" * 64,
    }
    with pytest.raises(benchmark.TeacherSupportBenchmarkError, match="differs"):
        benchmark.build_report(
            panel=panel,
            panel_selection=selection,
            runtime=frozen_runtime,
            slow=measurement,
            fast=(fast,),
            report_path=ROOT / benchmark.DEFAULT_REPORT,
        )


def test_report_carries_complete_false_authority_and_provenance(
    frozen_runtime, jin_panel, monkeypatch: pytest.MonkeyPatch
) -> None:
    entries, selection = jin_panel
    panel = benchmark.panel_payload(
        entries,
        csv_sha256=benchmark._sha256(ROOT / benchmark.JIN_CSV),
        selection=selection,
    )
    evidence_hash = "a" * 64
    slow = {
        "method": "slow_full_quotient",
        "batch_size": None,
        "query_count": 16,
        "wall_seconds": 4.0,
        "cpu_seconds": 3.0,
        "queries_per_wall_second": 4.0,
        "queries_per_cpu_second": 16 / 3,
        "process_peak_rss_mb": 700.0,
        "evidence_sha256": evidence_hash,
        "evidence": [],
    }
    fast = {
        **slow,
        "method": "fast_batched_teacher_support",
        "batch_size": 8,
        "wall_seconds": 1.0,
        "cpu_seconds": 0.8,
    }
    monkeypatch.setattr(benchmark, "_git_state", lambda _root: ("1" * 40, []))
    report = benchmark.build_report(
        panel=panel,
        panel_selection=selection,
        runtime=frozen_runtime,
        slow=slow,
        fast=(fast,),
        report_path=ROOT / benchmark.DEFAULT_REPORT,
    )
    assert report["status"] == benchmark.STATUS
    assert report["result"]["exact_common_evidence_parity"] is True
    assert report["result"]["fast_batched_teacher_support"][0][
        "wall_speedup_vs_slow"
    ] == 4.0
    assert all(
        report[field] is False for field in benchmark.authority_false_block()
    )
    assert report["provenance"]["panel_source"]["sha256"] == benchmark._sha256(
        ROOT / benchmark.JIN_CSV
    )
    assert report["configuration"]["model_runtime"]["max_atoms"] == 40
