import pytest

from compose_v4.control.docking_value import identity
from compose_v4.control.route_complete_region_particle_receipts import (
    MANIFEST_FILENAME,
    CompleteCombinationParticleReceiptStore,
    make_complete_combination_parent_manifest,
    validate_complete_combination_parent_manifest,
)
from compose_v4.control.route_complete_region_particles import (
    RECEIPT_SCHEMA_VERSION,
    complete_combination_job_specs,
    reduce_route_complete_region_pool,
)

SOURCE_HASH = "a" * 64
CHECKPOINT_HASH = "b" * 64
TRAINING_HASH = "c" * 64
CODE_HASH = "d" * 64
CONFIG_HASH = "e" * 64


def _manifest(**overrides):
    arguments = {
        "source_state_sha256": SOURCE_HASH,
        "shared_checkpoint_sha256": CHECKPOINT_HASH,
        "expert_training_identity_sha256": TRAINING_HASH,
        "code_identities": {"particle_module": CODE_HASH},
        "config_identities": {"production_budgets": CONFIG_HASH},
    }
    arguments.update(overrides)
    return make_complete_combination_parent_manifest(**arguments)


def _receipt(spec, *, status="complete", source_hash=SOURCE_HASH):
    records = []
    telemetry = {
        "candidate_count": 0,
        "candidate_set_sha256": identity(records),
        "source_state_sha256": source_hash,
        "expert_training_identity_sha256": TRAINING_HASH,
    }
    status_flag = {
        "complete_with_realization_abstentions": ("compiler_abstentions", 1),
        "empty_supported_depth": (
            "combination_particle_empty_depth_abstention",
            True,
        ),
        "planning_capacity_abstention": (
            "combination_particle_planning_capacity_abstention",
            True,
        ),
        "target_capacity_abstention": (
            "combination_particle_target_capacity_abstention",
            True,
        ),
    }.get(status)
    if status_flag is not None:
        telemetry[status_flag[0]] = status_flag[1]
    payload = {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "job": spec.payload(),
        "status": status,
        "records": records,
        "telemetry": telemetry,
    }
    return {"payload": payload, "payload_sha256": identity(payload)}


def _publish_census(store, specs, *, scientific_status_job=None):
    for spec in specs:
        status = (
            "complete_with_realization_abstentions"
            if spec.job_id == scientific_status_job
            else "complete"
        )
        store.publish_receipt(spec, _receipt(spec, status=status))


def test_parent_manifest_is_invariant_and_freezes_exact_job_census():
    first = _manifest(
        code_identities={"zeta": CODE_HASH, "alpha": "f" * 64},
        config_identities={"route_policy": CONFIG_HASH},
    )
    second = _manifest(
        code_identities={"alpha": "f" * 64, "zeta": CODE_HASH},
        config_identities={"route_policy": CONFIG_HASH},
    )
    assert first == second
    payload = validate_complete_combination_parent_manifest(first)
    specs = sorted(complete_combination_job_specs(), key=lambda row: row.job_id)
    assert payload["job_count"] == 28
    assert payload["jobs"] == [spec.payload() for spec in specs]
    assert payload["supported_complete_depths"] == [1, 2, 3]
    assert payload["depth_4_complete_coverage"] is False
    assert payload["particle_budgets"]["max_planning_expansions"] == 4096
    assert not {
        "target",
        "target_name",
        "cell",
        "cell_key",
        "delta",
        "teacher",
        "smiles",
    }.intersection(payload)
    assert all(
        not {
            "target",
            "target_name",
            "cell",
            "cell_key",
            "delta",
            "teacher",
            "smiles",
        }.intersection(job)
        for job in payload["jobs"]
    )

    for forbidden_argument in ("target", "cell", "delta"):
        with pytest.raises(TypeError):
            _manifest(**{forbidden_argument: "must-not-affect-the-manifest"})


def test_manifest_and_receipts_are_publish_once_and_manifest_drift_is_refused(
    tmp_path,
):
    manifest = _manifest()
    store = CompleteCombinationParticleReceiptStore(tmp_path, manifest)
    assert (tmp_path / MANIFEST_FILENAME).is_file()
    manifest["payload"]["jobs"].clear()
    assert store.manifest_payload["job_count"] == 28
    assert len(store.manifest_payload["jobs"]) == 28
    spec = complete_combination_job_specs()[0]
    receipt = _receipt(spec)
    store.publish_receipt(spec, receipt)
    with pytest.raises(FileExistsError, match="already exists"):
        store.publish_receipt(spec, receipt)

    restarted = CompleteCombinationParticleReceiptStore(tmp_path, _manifest())
    assert restarted.load_receipt(spec) == receipt
    changed = _manifest(shared_checkpoint_sha256="f" * 64)
    with pytest.raises(ValueError, match="manifest identity mismatch"):
        CompleteCombinationParticleReceiptStore(tmp_path, changed)


def test_restart_reuses_valid_receipt_without_invoking_runner(tmp_path):
    manifest = _manifest()
    store = CompleteCombinationParticleReceiptStore(tmp_path, manifest)
    spec = complete_combination_job_specs()[0]
    receipt = _receipt(spec)
    store.publish_receipt(spec, receipt)
    calls = []

    def forbidden_runner(source, expert, job):
        calls.append((source, expert, job))
        raise AssertionError("restart recomputed a durable proposal receipt")

    loaded, reused = store.run_job_once(
        object(), object(), spec, runner=forbidden_runner
    )
    assert loaded == receipt
    assert reused is True
    assert calls == []


def test_settlement_is_independent_of_receipt_completion_order(tmp_path):
    manifest = _manifest()
    specs = list(complete_combination_job_specs())
    first = CompleteCombinationParticleReceiptStore(tmp_path / "forward", manifest)
    second = CompleteCombinationParticleReceiptStore(tmp_path / "reverse", manifest)
    scientific_job = specs[8].job_id
    _publish_census(first, specs, scientific_status_job=scientific_job)
    _publish_census(second, reversed(specs), scientific_status_job=scientific_job)

    legacy = [{"smiles": "CC", "route_proposal_rank": 1}]
    first_pool, first_telemetry = first.settle(legacy)
    second_pool, second_telemetry = second.settle(legacy)
    direct_pool, direct_telemetry = reduce_route_complete_region_pool(
        legacy,
        [
            _receipt(
                spec,
                status=(
                    "complete_with_realization_abstentions"
                    if spec.job_id == scientific_job
                    else "complete"
                ),
            )
            for spec in specs
        ],
    )
    assert first_pool == second_pool
    assert first_pool == direct_pool
    assert first_telemetry == second_telemetry
    assert {
        key: value
        for key, value in first_telemetry.items()
        if key != "particle_parent_manifest_sha256"
    } == direct_telemetry
    assert first_telemetry["particle_augmentation_status"] == "complete"
    assert first_telemetry["receipt_status_counts"] == {
        "complete": 27,
        "complete_with_realization_abstentions": 1,
    }


def test_missing_corrupt_and_failed_receipts_abstain_to_legacy_only(tmp_path):
    manifest = _manifest()
    specs = list(complete_combination_job_specs())
    legacy = [{"smiles": "CC", "route_proposal_rank": 1}]

    missing = CompleteCombinationParticleReceiptStore(tmp_path / "missing", manifest)
    pool, telemetry = missing.settle(legacy)
    assert pool == legacy
    assert telemetry["particle_augmentation_abstention_reason"] == (
        "missing_required_receipt"
    )

    corrupt = CompleteCombinationParticleReceiptStore(tmp_path / "corrupt", manifest)
    _publish_census(corrupt, specs)
    corrupt_path = tmp_path / "corrupt" / corrupt.receipt_path(specs[4])
    corrupt_path.write_text("{not-json\n")
    pool, telemetry = corrupt.settle(legacy)
    assert pool == legacy
    assert telemetry["particle_augmentation_abstention_reason"] == (
        "corrupt_required_receipt"
    )
    with pytest.raises(FileExistsError, match="already exists"):
        corrupt.publish_receipt(specs[4], _receipt(specs[4]))

    failed = CompleteCombinationParticleReceiptStore(tmp_path / "failed", manifest)
    for spec in specs:
        failed.publish_receipt(
            spec,
            _receipt(spec, status="failed" if spec == specs[3] else "complete"),
        )
    pool, telemetry = failed.settle(legacy)
    assert pool == legacy
    assert telemetry["particle_augmentation_abstention_reason"] == (
        "operationally_failed_required_receipt"
    )


def test_deadline_receipt_is_empty_durable_and_never_recomputed(tmp_path):
    manifest = _manifest()
    specs = list(complete_combination_job_specs())
    store = CompleteCombinationParticleReceiptStore(tmp_path, manifest)
    deadline, reused = store.mark_missing_at_deadline(specs[0])
    assert reused is False
    assert deadline["payload"]["status"] == "missing_at_deadline"
    assert deadline["payload"]["records"] == []
    assert deadline["payload"]["telemetry"]["candidate_count"] == 0
    assert store.mark_missing_at_deadline(specs[0]) == (deadline, True)
    for spec in specs[1:]:
        store.publish_receipt(spec, _receipt(spec))
    pool, telemetry = store.settle([{"smiles": "CC", "route_proposal_rank": 1}])
    assert pool == [{"smiles": "CC", "route_proposal_rank": 1}]
    assert telemetry["failed_jobs"] == [specs[0].job_id]


def test_receipt_telemetry_rejects_forbidden_nested_content(tmp_path):
    store = CompleteCombinationParticleReceiptStore(tmp_path, _manifest())
    spec = complete_combination_job_specs()[0]
    receipt = _receipt(spec)
    receipt["payload"]["telemetry"]["debug"] = {
        "teacher": {"smiles": "CC", "actions": [1, 2]}
    }
    receipt["payload_sha256"] = identity(receipt["payload"])
    with pytest.raises(ValueError, match="invalid complete-combination"):
        store.publish_receipt(spec, receipt)
    assert not (tmp_path / store.receipt_path(spec)).exists()


def test_receipt_with_wrong_parent_identity_is_never_published(tmp_path):
    store = CompleteCombinationParticleReceiptStore(tmp_path, _manifest())
    spec = complete_combination_job_specs()[0]
    with pytest.raises(ValueError, match="source-state identity mismatch"):
        store.publish_receipt(spec, _receipt(spec, source_hash="f" * 64))
    assert not (tmp_path / store.receipt_path(spec)).exists()
