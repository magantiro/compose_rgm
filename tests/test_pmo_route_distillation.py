from __future__ import annotations

import gzip
import hashlib
import json

import pytest

from compose_v4.control.docking_value import identity
from compose_v4.experiments.editing_v2_evaluation_semantics import (
    production_state_from_smiles,
)
from tools.pmo_route_distillation import (
    FORBIDDEN_GENERIC_KEYS,
    _keys,
    _publish,
    _verified_envelope,
    action_role_supervision,
    run,
)


def test_action_role_supervision_uses_relative_created_handles_only():
    source = production_state_from_smiles("CC", max_atoms=48)
    created = {0: (3, 2)}
    action = {
        "executor_rule": "bond_reorder",
        "model_family": "bond_reorder",
        "payload": {"a": 0, "b": 1, "new_order": 2},
    }

    row, next_ordinal = action_role_supervision(
        source, action, created, step=5, next_ordinal=4
    )

    assert next_ordinal == 4
    assert row["operands"][0]["descriptor"]["origin"] == "route_created"
    assert row["operands"][0]["descriptor"]["created_ordinal"] == 3
    assert row["operands"][0]["descriptor"]["creation_lag"] == 3
    assert row["created_handle_dependencies"] == [
        {"operand": "endpoint_a", "created_ordinal": 3, "creation_lag": 3}
    ]
    assert not (FORBIDDEN_GENERIC_KEYS & set(_keys(row)))


def test_source_envelope_self_hash_mismatch_fails(tmp_path):
    path = tmp_path / "source.json"
    payload = {"schema_version": "fixture", "rows": 1}
    path.write_text(json.dumps({"payload": payload, "payload_sha256": "0" * 64}))

    with pytest.raises(ValueError, match="self-hash mismatch"):
        _verified_envelope(
            tmp_path, path.name, hashlib.sha256(path.read_bytes()).hexdigest()
        )


def test_locked_pmo_route_export_is_exact_balanced_and_has_no_actor(tmp_path):
    output = tmp_path / "attempt_1"
    report = run(output)
    dataset_path = output / "training_dataset.json.gz"
    envelope = json.loads(gzip.decompress(dataset_path.read_bytes()))
    dataset = envelope["payload"]

    assert envelope["payload_sha256"] == identity(dataset)
    assert report["summary"]["route_instances"] == 186
    assert report["summary"]["curriculum_programs"] == 181
    assert report["summary"]["witness_routes"] == 5
    assert report["summary"]["unique_exact_traces"] == 184
    assert report["summary"]["emitted_deduplicated_decisions"] == 6143
    assert report["summary"]["runtime_supported_route_instances"] == 107
    assert report["summary"]["long_route_local_only_instances"] == 79
    assert report["summary"]["recognized_compound_stages"] == 0
    assert report["summary"]["total_weight"] == pytest.approx(1.0)
    assert all(
        value == pytest.approx(1 / 7)
        for value in report["summary"]["family_weights"].values()
    )
    assert report["summary"]["generic_forbidden_key_hits"] == []
    assert report["summary"]["generic_forbidden_value_hits"] == []
    assert dataset["actor_training_performed"] is False
    assert dataset["module_count_targets_emitted"] is False
    assert dataset["binding_prototype_targets_emitted"] is False
    assert dataset["new_oracle_calls"] == 0

    duplicate = tmp_path / "duplicate.json.gz"
    _publish(duplicate, dataset, compressed=True)
    assert duplicate.read_bytes() == dataset_path.read_bytes()
