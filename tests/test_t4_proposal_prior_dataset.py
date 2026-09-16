"""Leakage, admission and determinism guards for the split-clean proposal-prior corpus.

The corpus exists to be split before anything is derived from it, so the tests that
matter here are the ones that fail when a group boundary is crossed, when a rejection
silently disappears, or when the published artifact stops matching its own invariants.
"""

from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

import pytest

from compose_v4.experiments.continuation_profile import canonical_bytes
from compose_v4.experiments.t4_matched_pilot import seal, unseal
from compose_v4.experiments.t4_proposal_prior_dataset import (
    SCHEMA_VERSION,
    admit,
    assert_group_disjointness,
    block_families,
    build,
    collapse_observations,
    coverage_decision,
    effective_sample_size,
    fold_assignment,
    group_disjointness,
    hierarchical_weights,
    lineage_components,
    load_contract,
    verify_contract_inputs,
    write_records,
)

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/t4_proposal_prior_dataset_v1.json"
ARTIFACT = ROOT / "diagnostics/t4_proposal_prior/dataset_v1"


def _contract():
    return load_contract(CONTRACT)


def _manifest(names=("run_checkpoint.json.gz",)):
    return {"checkpoints": {name: "0" * 64 for name in names}}


def _row(**overrides):
    row = {
        "cell": "braf_1",
        "target": "braf",
        "arm": "v0",
        "replicate": 0,
        "protocol": "p" * 64,
        "receipt_id": "r" * 64,
        "entry_id": "e" * 64,
        "entry_alternatives": 1,
        "endpoint": "CCO",
        "input_state_sha256": "s" * 64,
        "checkpoint": "/mirror/run_checkpoint.json.gz",
        "score": -9.5,
        "genealogical_parent": None,
        "parent_score": None,
        "query": 3,
        "primitive_count": 2,
        "changed_slot_count": 1,
        "net_created": 1,
        "net_deleted": 0,
        "ring_count_change": 0,
        "channel": "mutation",
        "planner_channel": None,
        "block_labels": ["0:carbonyl_insert"],
        "rule_counts": {"atom_insert": 2},
    }
    row.update(overrides)
    return row


# ---- Contract ----


def test_contract_payload_hash_is_enforced(tmp_path):
    payload = unseal(CONTRACT)
    assert load_contract(CONTRACT)["schema_version"] == SCHEMA_VERSION
    tampered = tmp_path / "tampered.json"
    envelope = json.loads(CONTRACT.read_text())
    envelope["payload"]["coverage_gate"]["hard"]["minimum_training_records_per_fold"] = 1
    tampered.write_text(json.dumps(envelope))
    with pytest.raises(ValueError, match="corrupt"):
        load_contract(tampered)
    assert payload["coverage_gate"]["hard"]["minimum_training_records_per_fold"] == 2000


def test_a_changed_declared_input_is_refused(tmp_path):
    contract = _contract()
    (tmp_path / Path(contract["inputs"]["scored_pack"]["path"]).parent).mkdir(
        parents=True, exist_ok=True
    )
    for entry in contract["inputs"].values():
        (tmp_path / entry["path"]).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / entry["path"]).write_bytes(b"not the frozen input")
    with pytest.raises(ValueError, match="contract input changed"):
        verify_contract_inputs(contract, tmp_path)


def test_declared_inputs_still_hash_to_their_frozen_identity():
    assert set(verify_contract_inputs(_contract(), ROOT)) == {
        "scored_pack",
        "scored_pack_manifest",
        "strategy_report",
        "checkpoint_manifest",
    }


# ---- Folds ----


def test_every_declared_cell_is_claimed_by_exactly_one_fold():
    assignment = fold_assignment(_contract())
    assert len(assignment) == 15
    assert sorted(set(assignment.values())) == ["5ht1b", "braf", "fa7", "jak2", "parp1"]
    assert all(cell.startswith(fold) for cell, fold in assignment.items())


def test_a_cell_claimed_by_two_folds_is_rejected():
    contract = _contract()
    contract["folds"]["members"]["braf"] = [*contract["folds"]["members"]["braf"], "jak2_0"]
    with pytest.raises(ValueError, match="claimed by two folds"):
        fold_assignment(contract)


def test_fold_membership_must_cover_the_declared_cells():
    contract = _contract()
    contract["folds"]["members"]["fa7"] = ["fa7_0"]
    with pytest.raises(ValueError, match="does not cover"):
        fold_assignment(contract)


# ---- Disjointness ----


def test_an_endpoint_shared_by_two_cells_is_caught():
    rows = [_row(), _row(cell="jak2_1", target="jak2", entry_id="x" * 64)]
    evidence = group_disjointness(rows)
    assert evidence["endpoint"]["values_crossing_groups"] == 1
    with pytest.raises(ValueError, match="not disjoint in: endpoint"):
        assert_group_disjointness(evidence)


def test_clean_rows_pass_disjointness():
    rows = [
        _row(),
        _row(
            cell="jak2_1",
            target="jak2",
            endpoint="CCN",
            entry_id="x" * 64,
            input_state_sha256="t" * 64,
            protocol="q" * 64,
        ),
    ]
    assert_group_disjointness(group_disjointness(rows))


# ---- Admission ----


def test_every_reason_code_fires_and_counts_reconcile():
    contract = _contract()
    manifest = _manifest()
    rows = [
        _row(),
        _row(score=None, receipt_id="a" * 64),
        _row(entry_alternatives=2, receipt_id="b" * 64),
        _row(cell="unknown_9", receipt_id="c" * 64),
        _row(checkpoint="/mirror/other.json.gz", receipt_id="d" * 64),
        _row(endpoint=None, receipt_id="f" * 64),
    ]
    admitted, excluded = admit(rows, contract, manifest)
    assert len(admitted) == 1
    assert len(admitted) + len(excluded) == len(rows)
    assert sorted(e["reason"] for e in excluded) == [
        "ambiguous_program_join",
        "missing_identity_field",
        "missing_or_nonfinite_score",
        "unbound_checkpoint",
        "undeclared_cell",
    ]


def test_a_nonfinite_score_is_not_admitted_as_a_number():
    admitted, excluded = admit([_row(score=float("nan"))], _contract(), _manifest())
    assert not admitted and excluded[0]["reason"] == "missing_or_nonfinite_score"


# ---- Collapse ----


def test_repeat_receipts_collapse_into_one_labelled_construction():
    rows = [
        _row(receipt_id="1" * 64, score=-11.3, replicate=0, query=5),
        _row(receipt_id="2" * 64, score=-7.6, replicate=1, query=None),
    ]
    (record,) = collapse_observations(rows)
    assert record["observation_count"] == 2
    assert record["score_mean"] == pytest.approx(-9.45)
    assert (record["score_min"], record["score_max"]) == (-11.3, -7.6)
    assert record["score_range"] == pytest.approx(3.7)
    assert record["replicates"] == [0, 1]
    assert record["call_indices"] == [5] and record["call_index_available"]


def test_several_recorded_parents_are_retained_not_collapsed():
    rows = [
        _row(receipt_id="1" * 64, genealogical_parent="p1", parent_score=-10.9),
        _row(receipt_id="2" * 64, genealogical_parent="p2", parent_score=-7.4),
    ]
    (record,) = collapse_observations(rows)
    assert record["parent_entry_ids"] == ["p1", "p2"]
    assert record["parent_scores"] == [-10.9, -7.4]


def test_one_construction_cannot_carry_two_endpoints():
    rows = [_row(receipt_id="1" * 64), _row(receipt_id="2" * 64, endpoint="CCN")]
    with pytest.raises(ValueError, match="conflicting endpoint"):
        collapse_observations(rows)


def test_block_labels_reduce_to_their_generic_family():
    assert block_families(
        [
            "composition_2:0:1:dependency_branch:1",
            "current:atom_restate_semantic",
            "compiled_complete_transformation",
            "0:pendant_benzene",
        ]
    ) == [
        "dependency_branch",
        "atom_restate_semantic",
        "compiled_complete_transformation",
        "pendant_benzene",
    ]


# ---- Lineage ----


def test_two_descendants_of_one_unscored_ancestor_share_a_component():
    records = [
        {"record_id": "a", "cell": "braf_1", "entry_id": "A", "parent_entry_ids": ["ghost"]},
        {"record_id": "b", "cell": "braf_1", "entry_id": "B", "parent_entry_ids": ["ghost"]},
        {"record_id": "c", "cell": "braf_1", "entry_id": "C", "parent_entry_ids": []},
    ]
    components = lineage_components(records)
    assert components["a"] == components["b"] != components["c"]


def test_identical_entry_ids_in_two_cells_do_not_merge():
    records = [
        {"record_id": "a", "cell": "braf_1", "entry_id": "A", "parent_entry_ids": ["Z"]},
        {"record_id": "b", "cell": "jak2_1", "entry_id": "A", "parent_entry_ids": ["Z"]},
    ]
    components = lineage_components(records)
    assert components["a"] != components["b"]


# ---- Weights ----


def test_equal_mass_per_target_then_cell_then_lineage():
    records = []
    for index, (target, cell) in enumerate(
        [("braf", "braf_1"), ("braf", "braf_2"), ("jak2", "jak2_1")]
    ):
        for step in range(index + 1):
            records.append(
                {
                    "record_id": f"{cell}-{step}",
                    "cell": cell,
                    "target": target,
                    "entry_id": f"{cell}-{step}",
                    "parent_entry_ids": [],
                }
            )
    weights = hierarchical_weights(records, _contract())
    assert sum(weights.values()) == pytest.approx(1.0)
    per_target = {"braf": 0.0, "jak2": 0.0}
    for record in records:
        per_target[record["target"]] += weights[record["record_id"]]
    assert per_target["braf"] == pytest.approx(0.5)
    assert per_target["jak2"] == pytest.approx(0.5)
    assert weights["braf_1-0"] == pytest.approx(0.25)


def test_an_undeclared_weight_hierarchy_is_refused():
    contract = _contract()
    contract["weights"]["hierarchy"] = ["cell", "record"]
    with pytest.raises(ValueError, match="unsupported declared weight hierarchy"):
        hierarchical_weights([], contract)


def test_effective_sample_size_matches_its_definition():
    assert effective_sample_size([0.25] * 4) == pytest.approx(4.0)
    assert effective_sample_size([0.97, 0.01, 0.01, 0.01]) == pytest.approx(1.0632, abs=1e-3)


# ---- Gate ----


def _census(**overrides):
    base = {
        "training": {
            "records": 5000,
            "cells": [f"c{i}" for i in range(12)],
            "targets": ["a", "b", "c", "d"],
            "source_contexts": 900,
            "lineage_components": 40,
            "observations": 5200,
            "bank_macro_records": 10,
            "family_counts": {
                family: 500
                for family in [
                    "substituent_delete",
                    "bond_reroute",
                    "carbonyl_insert",
                    "append_ring",
                    "ring_system_restate",
                    "construct_substituted_ring",
                ]
            },
        },
        "heldout": {
            "records": 800,
            "cells": ["h0"],
            "targets": ["e"],
            "source_contexts": 120,
            "lineage_components": 9,
            "observations": 830,
            "bank_macro_records": 3,
            "family_counts": {},
        },
    }
    for side, fields in overrides.items():
        base[side].update(fields)
    return {"only": base}


def test_a_satisfying_census_passes_the_frozen_gate():
    decision = coverage_decision(_census(), _contract())
    assert decision["decision"] == "GO" and decision["failures"] == []


def test_a_required_family_that_is_not_dense_forces_abstention():
    census = _census()
    census["only"]["training"]["family_counts"]["construct_substituted_ring"] = 12
    decision = coverage_decision(census, _contract())
    assert decision["decision"] == "ABSTAIN"
    assert any("construct_substituted_ring" in failure for failure in decision["failures"])
    assert decision["family_tiers"]["construct_substituted_ring"]["tier"] == "sparse"


def test_a_thin_heldout_side_forces_abstention():
    decision = coverage_decision(_census(heldout={"source_contexts": 3}), _contract())
    assert decision["decision"] == "ABSTAIN"
    assert any("source contexts" in failure for failure in decision["failures"])


def test_a_thin_training_side_forces_abstention():
    decision = coverage_decision(_census(training={"records": 10}), _contract())
    assert decision["decision"] == "ABSTAIN"
    assert any("training records" in failure for failure in decision["failures"])


# ---- Build ----


def _buildable_rows():
    rows = []
    for target, cells in _contract()["grouping"]["declared_cells"].items():
        for cell in cells:
            for index in range(3):
                rows.append(
                    _row(
                        cell=cell,
                        target=target,
                        entry_id=f"{cell}-{index}".ljust(64, "0"),
                        receipt_id=f"{cell}-r{index}".ljust(64, "0"),
                        endpoint=f"C{'C' * index}O-{cell}",
                        input_state_sha256=f"{cell}-s".ljust(64, "0"),
                        protocol=f"{cell}-p".ljust(64, "0"),
                        genealogical_parent=f"{cell}-0".ljust(64, "0") if index else None,
                    )
                )
    return rows


def test_build_produces_folds_that_no_lineage_component_crosses():
    audit, records = build(_contract(), _buildable_rows(), _manifest())
    assert audit["input_rows"] == audit["admitted_rows"] == 45
    assert audit["new_oracle_calls"] == 0 and audit["new_labels"] == 0
    folds = {}
    for record in records:
        folds.setdefault(record["lineage_component"], set()).add(record["fold"])
    assert all(len(seen) == 1 for seen in folds.values())
    assert all(record["fold"] == record["target"] for record in records)


def test_build_refuses_a_leaking_pack():
    rows = _buildable_rows()
    rows[-1]["endpoint"] = rows[0]["endpoint"]
    with pytest.raises(ValueError, match="not disjoint"):
        build(_contract(), rows, _manifest())


def test_the_record_table_is_byte_stable(tmp_path):
    _, records = build(_contract(), _buildable_rows(), _manifest())
    first = write_records(tmp_path / "a.jsonl.gz", records)
    second = write_records(tmp_path / "b.jsonl.gz", list(reversed(records)))
    assert first == second
    assert (tmp_path / "a.jsonl.gz").read_bytes() == (tmp_path / "b.jsonl.gz").read_bytes()


# ---- Published artifact ----


def _published():
    audit = unseal(ARTIFACT / "audit.json")
    with gzip.open(ARTIFACT / "records.jsonl.gz", "rt") as handle:
        records = [json.loads(line) for line in handle]
    return audit, records


def test_the_published_artifact_matches_its_own_record_table():
    audit, records = _published()
    body = b"".join(canonical_bytes(record) + b"\n" for record in records)
    assert hashlib.sha256(body).hexdigest() == audit["records_artifact"]["decompressed_sha256"]
    assert len(records) == audit["records"]
    assert audit["contract_sha256"] == hashlib.sha256(canonical_bytes(_contract())).hexdigest()


def test_the_published_artifact_has_no_identity_crossing_a_fold():
    _, records = _published()
    for field in (
        "endpoint_sha256",
        "input_state_sha256",
        "entry_id",
        "protocol",
        "lineage_component",
    ):
        folds = {}
        for record in records:
            folds.setdefault(record[field], set()).add(record["fold"])
        crossing = [value for value, seen in folds.items() if len(seen) > 1]
        assert crossing == [], f"{field} crosses a fold: {crossing[:3]}"


def test_the_published_artifact_carries_no_forbidden_runtime_field():
    _, records = _published()
    forbidden = {
        "endpoint_smiles",
        "input_smiles",
        "docking_seed",
        "controller_seed",
        "original_seed",
        "source_state",
        "program",
        "trace",
    }
    assert forbidden.isdisjoint(set().union(*(set(record) for record in records)))


def test_the_published_gate_decision_is_reproducible_from_its_own_census():
    audit, _ = _published()
    replayed = coverage_decision(audit["census"], _contract())
    assert replayed["decision"] == audit["gate"]["decision"]
    assert replayed["failures"] == audit["gate"]["failures"]


def test_seal_round_trip_is_required_by_the_published_audit(tmp_path):
    audit, _ = _published()
    seal(tmp_path / "copy.json", audit)
    assert unseal(tmp_path / "copy.json") == audit
    envelope = json.loads((tmp_path / "copy.json").read_text())
    envelope["payload"]["records"] = 1
    (tmp_path / "copy.json").write_text(json.dumps(envelope))
    with pytest.raises(ValueError, match="corrupt"):
        unseal(tmp_path / "copy.json")
