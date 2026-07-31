from __future__ import annotations

import hashlib

from scripts.benchmark_lead_scope_coverage import REPO, build_artifact, coverage


BENCHMARK = REPO / "configs/benchmarks/jin_iclr19_qed_test_exact_v1.csv"


def test_benchmark_lead_scope_coverage_matches_frozen_census() -> None:
    result = coverage(BENCHMARK)

    assert result["n_leads"] == 800
    assert result["accepted"] == {
        "broad_organic_v1": 800,
        "broad_organic_neutral_v1": 557,
        "cnof_neutral": 264,
    }
    assert result["excluded_features_broad_scope"] == {}


def test_benchmark_lead_scope_artifact_binds_input_and_implementation() -> None:
    implementation = [
        {
            "path": "scripts/benchmark_lead_scope_coverage.py",
            "sha256": "a" * 64,
        }
    ]
    artifact = build_artifact(
        BENCHMARK,
        code_revision="b" * 40,
        implementation_files=implementation,
    )

    assert artifact["schema_version"] == 2
    assert artifact["evidence_class"] == "computed"
    assert artifact["provenance"]["code_revision"] == "b" * 40
    assert artifact["provenance"]["implementation_files"] == implementation
    assert artifact["provenance"]["inputs"] == [
        {
            "path": "configs/benchmarks/jin_iclr19_qed_test_exact_v1.csv",
            "sha256": hashlib.sha256(BENCHMARK.read_bytes()).hexdigest(),
            "role": "fixed_benchmark_support_census",
        }
    ]
    assert artifact["result"]["accepted"]["broad_organic_v1"] == 800
    assert (
        max(int(row.split(",")[6]) for row in BENCHMARK.read_text().splitlines()[1:])
        <= artifact["provenance"]["configuration"]["broad_organic_scope"]["max_atoms"]
    )
