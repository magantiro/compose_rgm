import json
from pathlib import Path

from compose_v4.control.docking_value import identity
from compose_v4.experiments.continuation_profile import sha256_file

ROOT = Path(__file__).resolve().parents[1]
RESULT = (
    ROOT
    / "diagnostics/t4_shared_retained_fiber_parp1_p0_rescue_v1/support_gate/"
    "result.json"
)
REPORT = RESULT.with_name("REPORT.md")


def test_parp_route_timeout_support_gate_is_sealed_and_complete() -> None:
    envelope = json.loads(RESULT.read_text())
    payload = envelope["payload"]

    assert envelope["payload_sha256"] == identity(payload)
    assert payload["schema_version"] == "t4_parp_route_timeout_support_gate_v1"
    assert payload["evidence_status"] == (
        "computed_zero_oracle_production_runtime_gate"
    )
    assert len(payload["code_revision"]) == 40
    assert payload["configuration"] == {
        "pool_size": 192,
        "realization_limit": 96,
        "beam_width": 48,
        "expansion_width": 48,
        "max_bindings_per_template": 4,
        "maximum_expansions": 4_000,
        "scale_balanced": True,
        "per_candidate_timeout_seconds": 10.0,
    }
    for relative, expected in {
        **payload["material_inputs_sha256"],
        **payload["implementation_inputs_sha256"],
    }.items():
        assert sha256_file(ROOT / relative) == expected

    result = payload["result"]
    statuses = result["realization_status_counts"]
    bands = result["realized_primitive_band_counts"]
    assert result["elapsed_seconds"] > 0.0
    assert result["attempted_realizations"] == sum(statuses.values()) == 96
    assert result["candidate_timeout_count"] == statuses[
        "realizer_candidate_timeout"
    ]
    assert result["candidate_timeout_count"] > 0
    assert result["returned_record_count"] == statuses["committed"]
    assert result["unique_returned_endpoint_count"] == result[
        "returned_record_count"
    ]
    assert sum(bands.values()) == result["returned_record_count"]
    assert all(bands[band] > 0 for band in ("small", "medium", "large"))
    assert result["exact_realization_precision"] == 1.0
    assert result["exact_realization_precision_numerator"] == result[
        "exact_realization_precision_denominator"
    ] == result["returned_record_count"]
    assert len(set(result["expected_endpoint_keys"])) == 3
    assert result["recovered_expected_endpoint_keys"] == result[
        "expected_endpoint_keys"
    ]
    assert result["expected_endpoint_recovery_count"] == 3
    assert result["missing_expected_endpoint_keys"] == []
    assert result["other_noncommitted_status_counts"] == {}
    assert payload["costs"] == {
        "oracle_calls": 0,
        "docking_calls": 0,
        "modal_launches": 0,
        "gpu_seconds": 0,
    }
    assert payload["gate"]["passed"] is True
    assert all(payload["gate"].values())

    report = REPORT.read_text()
    assert "**Decision: PASS**" in report
    assert "Exact realization precision: 87/87" in report
    assert "Expected endpoint recovery: 3/3" in report
