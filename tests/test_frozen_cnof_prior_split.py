"""Data-identity and leakage gates for a nested de novo prior comparison."""

from __future__ import annotations

import copy
import json

import pytest

from compose_v4.data.frozen_cnof_prior_split import (
    _digest,
    file_sha256,
    frozen_cnof_arm_identity,
    frozen_cnof_source_alias_identity,
    load_frozen_cnof_arm,
    prepare_frozen_cnof_split,
    validate_frozen_cnof_split,
)

MOLECULES = (
    "CCO",
    "CCN",
    "CCF",
    "c1ccccc1",
    "Cc1ccccc1",
    "C1CCCCC1",
    "C1CCNCC1",
    "c1ccncc1",
    "C1CCC1",
    "C1COC1",
    "C1CCCC1",
    "OCC",  # canonical duplicate of CCO
    "C.C",  # disconnected
    "C1CC",  # unparseable
    "ClC",  # unsupported element
    "[NH4+]",  # charged
    "CCCCCCCCCCCCCCCCCCCCC",  # too big
)


def _prepare(tmp_path):
    source = tmp_path / "source.smiles"
    source.write_text("\n".join(MOLECULES) + "\n")
    payload = prepare_frozen_cnof_split(
        source,
        expected_source_sha256=file_sha256(source),
        source_access_basis="fixture only",
        code_revision="fixture-revision",
        seed=37,
        max_atoms=20,
        small_train_size=2,
        validation_size=1,
        iid_test_size=1,
        scaffold_test_min=1,
        scaffold_test_max=2,
    )
    return source, payload


def test_nested_arms_share_validation_and_tests(tmp_path):
    _, payload = _prepare(tmp_path)
    manifest = tmp_path / "split.json"
    manifest.write_text(json.dumps(payload, sort_keys=True))
    small = load_frozen_cnof_arm(manifest, train_size=2)
    large = load_frozen_cnof_arm(manifest, train_size=len(payload["partitions"]["train"]))
    small_identity = frozen_cnof_arm_identity(manifest, train_size=2)
    large_identity = frozen_cnof_arm_identity(
        manifest, train_size=len(payload["partitions"]["train"])
    )
    assert small_identity["shared_source_prior_prefix_size"] == 2
    assert (
        small_identity["shared_source_prior_sha256"] == large_identity["shared_source_prior_sha256"]
    )
    assert small.train == large.train[:2]
    assert small.validation == large.validation
    assert small.test == large.test
    assert (
        small.scaffold_overlap_count
        == payload["census"]["training_arm_diagnostics"]["small"]["iid_scaffold_overlap_count"]
    )
    assert (
        large.scaffold_overlap_count
        == payload["census"]["training_arm_diagnostics"]["large"]["iid_scaffold_overlap_count"]
    )
    assert payload["census"]["source_rows"] == len(MOLECULES)
    assert payload["census"]["duplicates"] == 1
    assert payload["census"]["excluded"] == 5
    assert payload["census"]["exclusion_reasons"] == {
        "charged": 1,
        "disconnected": 1,
        "too_big": 1,
        "unparseable_or_unsupported_state": 1,
        "unsupported_element": 1,
    }
    validate_frozen_cnof_split(payload, deep=True)


def test_source_identity_and_manifest_integrity_fail_closed(tmp_path):
    source, payload = _prepare(tmp_path)
    with pytest.raises(ValueError, match="source SHA-256 mismatch"):
        prepare_frozen_cnof_split(
            source,
            expected_source_sha256="0" * 64,
            source_access_basis="fixture only",
            code_revision="fixture-revision",
            seed=37,
            max_atoms=20,
            small_train_size=2,
            validation_size=1,
            iid_test_size=1,
            scaffold_test_min=1,
            scaffold_test_max=2,
        )
    changed = copy.deepcopy(payload)
    changed["partitions"]["validation"][0][1] = "CC"
    with pytest.raises(ValueError, match="payload hash mismatch"):
        validate_frozen_cnof_split(changed)
    with pytest.raises(ValueError, match="eligible CNOF molecule census mismatch"):
        prepare_frozen_cnof_split(
            source,
            expected_source_sha256=file_sha256(source),
            source_access_basis="fixture only",
            code_revision="fixture-revision",
            seed=37,
            max_atoms=20,
            small_train_size=2,
            validation_size=1,
            iid_test_size=1,
            scaffold_test_min=1,
            scaffold_test_max=2,
            expected_eligible_unique=100,
        )


def test_split_is_byte_stable_and_bounds_are_enforced(tmp_path):
    source, first = _prepare(tmp_path)
    second = prepare_frozen_cnof_split(
        source,
        expected_source_sha256=file_sha256(source),
        source_access_basis="fixture only",
        code_revision="fixture-revision",
        seed=37,
        max_atoms=20,
        small_train_size=2,
        validation_size=1,
        iid_test_size=1,
        scaffold_test_min=1,
        scaffold_test_max=2,
    )
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)
    manifest = tmp_path / "split.json"
    manifest.write_text(json.dumps(first, sort_keys=True))
    with pytest.raises(ValueError, match="not either frozen arm"):
        load_frozen_cnof_arm(manifest, train_size=100)
    if len(first["partitions"]["train"]) > 3:
        with pytest.raises(ValueError, match="not either frozen arm"):
            load_frozen_cnof_arm(manifest, train_size=3)


def test_deep_validation_rejects_resealed_scaffold_leakage(tmp_path):
    _, payload = _prepare(tmp_path)
    changed = copy.deepcopy(payload)
    moved = changed["partitions"]["scaffold_test"].pop()
    displaced = changed["partitions"]["train"].pop()
    changed["partitions"]["train"].append(moved)
    changed["partitions"]["scaffold_test"].append(displaced)
    changed["partition_sha256"] = {
        name: _digest(rows) for name, rows in changed["partitions"].items()
    }
    changed["payload_sha256"] = _digest(
        {key: value for key, value in changed.items() if key != "payload_sha256"}
    )
    with pytest.raises(ValueError, match="scaffold group leaked"):
        validate_frozen_cnof_split(changed, deep=True)


def test_deep_validation_rejects_resealed_arm_diagnostic_tamper(tmp_path):
    _, payload = _prepare(tmp_path)
    changed = copy.deepcopy(payload)
    changed["census"]["training_arm_diagnostics"]["small"]["iid_scaffold_overlap_count"] += 1
    changed["payload_sha256"] = _digest(
        {key: value for key, value in changed.items() if key != "payload_sha256"}
    )
    with pytest.raises(ValueError, match="training-arm scaffold diagnostics mismatch"):
        validate_frozen_cnof_split(changed, deep=True)


def test_deep_validation_names_malformed_partition_row(tmp_path):
    _, payload = _prepare(tmp_path)
    changed = copy.deepcopy(payload)
    changed["partitions"]["validation"][0] = [1]
    changed["partition_sha256"]["validation"] = _digest(changed["partitions"]["validation"])
    changed["payload_sha256"] = _digest(
        {key: value for key, value in changed.items() if key != "payload_sha256"}
    )
    with pytest.raises(ValueError, match="malformed source row in validation"):
        validate_frozen_cnof_split(changed, deep=True)


def test_runtime_mount_alias_requires_exact_source_bytes(tmp_path):
    preparation = tmp_path / "prepare" / "source.smiles"
    preparation.parent.mkdir()
    preparation.write_text("CCO\nCCN\n")
    runtime = tmp_path / "mounted" / "source.smiles"
    runtime.parent.mkdir()
    runtime.write_bytes(preparation.read_bytes())
    expected = file_sha256(preparation)
    identity = frozen_cnof_source_alias_identity(runtime, expected_sha256=expected)
    assert identity == {
        "runtime_source_path": str(runtime),
        "runtime_source_sha256": expected,
    }
    runtime.write_text("CCO\nCCC\n")
    with pytest.raises(ValueError, match="source SHA-256 mismatch"):
        frozen_cnof_source_alias_identity(runtime, expected_sha256=expected)
    runtime.unlink()
    with pytest.raises(ValueError, match="missing frozen CNOF runtime source"):
        frozen_cnof_source_alias_identity(runtime, expected_sha256=expected)
