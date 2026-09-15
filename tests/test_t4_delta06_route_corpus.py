import gzip
import json
from collections import Counter
from pathlib import Path

import pytest

from compose_v4.control.docking_value import identity
from tools import t4_delta06_route_corpus as corpus


def test_contract_and_frozen_receipt_manifest_are_exact():
    contract = corpus._load_contract()
    audit, rows, verified = corpus._load_inputs(contract)

    assert audit["scientific_identity"]
    assert len(rows) == 39
    assert sum(len(row["delta06_references"]) for row in rows) == 49
    assert Counter(row["status"] for row in rows) == {
        "witness_found": 32,
        "unreachable_charge_change": 3,
        "search_unresolved": 2,
        "unsupported_representation": 2,
    }
    assert identity(corpus._receipt_manifest(rows)) == (
        "8e6578f1be0bff1929e76d89226f2bc3c7f7d710ae04b2bb012fc7ff633b5055"
    )
    assert len(verified) == 50


def test_selected_rows_drop_non_delta06_references_without_duplicate_examples():
    audit = {
        "pairs": [
            {
                "pair_id": "b",
                "references": [
                    {"delta": 0.4, "run_seed": 1},
                    {"delta": 0.6, "run_seed": 2},
                    {"delta": 0.6, "run_seed": 3},
                ],
            },
            {"pair_id": "a", "references": [{"delta": 0.4, "run_seed": 0}]},
        ]
    }
    rows = corpus._selected_rows(audit)
    assert [row["pair_id"] for row in rows] == ["b"]
    assert [row["run_seed"] for row in rows[0]["delta06_references"]] == [2, 3]


def test_distribution_and_gate_report_denominators_and_precision():
    assert corpus._distribution([1, 4, 2, 3]) == {
        "count": 4,
        "minimum": 1,
        "median": 2.5,
        "maximum": 4,
    }
    assert corpus._gate(3, 4, exact=2) == {
        "covered": 3,
        "exact": 2,
        "denominator": 4,
        "coverage": 0.75,
        "precision": 2 / 3,
    }
    assert corpus._gate(0, 0) == {
        "covered": 0,
        "exact": 0,
        "denominator": 0,
        "coverage": None,
        "precision": None,
    }


def test_atomic_publish_is_self_hashed_deterministic_and_refuses_overwrite(tmp_path: Path):
    first = tmp_path / "a.json.gz"
    second = tmp_path / "b.json.gz"
    payload = {"schema_version": "fixture_v1", "rows": [2, 1]}
    corpus._atomic_publish(first, payload, compressed=True)
    corpus._atomic_publish(second, payload, compressed=True)

    assert first.read_bytes() == second.read_bytes()
    envelope = json.loads(gzip.decompress(first.read_bytes()))
    assert envelope == {"payload": payload, "payload_sha256": identity(payload)}
    with pytest.raises(ValueError, match="refusing to overwrite"):
        corpus._atomic_publish(first, payload, compressed=True)


def test_full_conversion_replays_and_realizes_every_admitted_witness():
    contract = corpus._load_contract()
    audit, rows, _ = corpus._load_inputs(contract)
    training, result = corpus.build_corpus(contract, audit, rows, code_revision="focused_test")

    assert len(training["endpoint_references"]) == 39
    assert len(training["routes"]) == 32
    assert len(training["abstentions"]) == 7
    assert sum(route["region_count"] for route in training["routes"]) == 45
    assert all(not route["observed_ivg_trajectory"] for route in training["routes"])
    assert all(gate["coverage"] == gate["precision"] == 1.0 for gate in result["gates"].values())
    assert len(result["route_metrics"]["per_cell"]) == 15
