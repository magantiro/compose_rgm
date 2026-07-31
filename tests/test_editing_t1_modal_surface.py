from __future__ import annotations

import copy
import hashlib
import json
import subprocess
from pathlib import Path

import pytest

pytest.importorskip("modal")

import modal_apps.run_editing_t1_successor_gate as t1_modal
import scripts.run_editing_t1_successor_gate as t1_worker
from compose_v4.experiments.editing_t1_panel import (
    EDITING_T1_CAPACITY_CENSUS_SCHEMA,
    EDITING_T1_CAPACITY_CENSUS_SCHEMA_VERSION,
    EDITING_T1_CAPACITY_CENSUS_STATUS,
    EDITING_T1_PANEL_SCHEMA,
    EDITING_T1_PANEL_SCHEMA_VERSION,
    EDITING_T1_PANEL_STATUS,
)
from compose_v4.experiments.editing_t1_successor_runtime import (
    EDITING_T1_V4_CAPACITY_CENSUS_RELATIVE_PATH,
    EDITING_T1_V4_PANEL_RELATIVE_PATH,
    EDITING_T1_V8_CONTRACT_RELATIVE_PATH,
    build_editing_t1_runtime_contract,
    validate_editing_t1_launch_authority,
)
from modal_apps.run_editing_t1_successor_gate import (
    FAMILIES,
    LOCAL_ADAPTER_FAMILIES,
    SCOPES,
    _require_clean_serialized_tree,
    build_t1_modal_launch_receipt,
    load_t1_modal_launch_receipt,
    t1_modal_launch_receipt_path,
    validate_t1_modal_launch_receipt,
    write_t1_modal_launch_receipt,
)
from modal_apps.run_editing_t1_successor_gate import (
    build_task_matrix as _build_task_matrix,
)

ACTIVE8_INVENTORY = "active8_trace_inventory_v1/ACTIVE8_TRACE_INVENTORY.json"
ACTIVE8_FILE_SHA256 = "a" * 64
RESOLVED_ACTIVE8_INVENTORY = "/artifacts/active8_trace_inventory_v1/ACTIVE8_TRACE_INVENTORY.json"
SOURCE_COMMIT = "c" * 40
RESULT_SHA256 = "d" * 64
CACHE_RECEIPTS = [
    {
        "packed_shard_content_sha256": "1" * 64,
        "cache_content_sha256": "2" * 64,
        "encoded_sha256": "3" * 64,
        "record_count": 1,
    }
]


def test_t1_worker_output_publication_survives_modal_hard_link_rejection(
    tmp_path: Path,
    monkeypatch,
) -> None:
    destination = tmp_path / "result.json"

    def reject_hard_link(*args, **kwargs):
        raise PermissionError(1, "operation not permitted")

    monkeypatch.setattr("os.link", reject_hard_link)
    t1_worker._write_if_absent(destination, {"status": "complete"})
    t1_worker._write_if_absent(destination, {"status": "complete"})

    assert json.loads(destination.read_text()) == {"status": "complete"}
    with pytest.raises(FileExistsError, match="already differs"):
        t1_worker._write_if_absent(destination, {"status": "different"})


CACHE_MANIFEST_RECEIPT = {
    "manifest_relative_path": "manifests/" + "4" * 64 + ".json",
    "manifest_sha256": "4" * 64,
    "manifest_file_sha256": "5" * 64,
    "manifest_file_bytes": 123,
    "selected_trace_set_sha256": "6" * 64,
    "initial_model_state_sha256": "7" * 64,
}


def _synthetic_launch_authority(
    *,
    repeated_families: set[str] | None = None,
):
    repeated_families = (
        {
            "atom_insert",
            "atom_restate",
            "bond_reroute",
            "cycle_attach",
            "ring_system_restate",
        }
        if repeated_families is None
        else set(repeated_families)
    )
    active8_identity = {
        "active8_inventory_manifest_file_sha256": ACTIVE8_FILE_SHA256,
        "active8_inventory_sha256": "b" * 64,
        "active8_effective_source_corpus_cache_sha256": "c" * 64,
        "active8_unified_packed_manifest_sha256": "d" * 64,
        "active8_support_contract_sha256": "e" * 64,
    }
    census_body = {
        "schema": EDITING_T1_CAPACITY_CENSUS_SCHEMA,
        "schema_version": EDITING_T1_CAPACITY_CENSUS_SCHEMA_VERSION,
        "status": EDITING_T1_CAPACITY_CENSUS_STATUS,
        "training_authorized": False,
        "forensics_file_sha256": "f" * 64,
        "gate_zero_runtime_contract_sha256": "e" * 64,
        "unified_packed_manifest_sha256": "d" * 64,
        "representability_overlay_sha256": "1" * 64,
        **active8_identity,
        "rows": [{"source_row_sha256": "2" * 64}],
    }
    census = {
        **census_body,
        "census_sha256": t1_modal._stable_sha256(census_body),
    }
    census_content = (
        json.dumps(census, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode()
    census_file_sha256 = hashlib.sha256(census_content).hexdigest()
    required_aliases = ["atom_insert", "bond_reroute", "cycle_attach"]
    selection = {
        "required_high_candidate_families": list(FAMILIES),
        "required_aliased_teacher_families": required_aliases,
    }
    panel_census = {
        "unique_high_candidate_stratum_available_by_family": {
            family: 1 for family in FAMILIES
        },
        "unique_high_candidate_stratum_selected_by_family": {
            family: 1 for family in FAMILIES
        },
        "unique_aliased_teacher_stratum_available_by_family": {
            family: int(family in required_aliases) for family in FAMILIES
        },
        "unique_aliased_teacher_stratum_selected_by_family": {
            family: int(family in required_aliases) for family in FAMILIES
        },
        "registered_high_candidate_families_all_covered": True,
        "registered_aliased_teacher_families_all_covered": True,
        "families_without_observed_within_family_multitarget_repeats": [
            family for family in FAMILIES if family not in repeated_families
        ],
    }
    panel_body = {
        "schema": EDITING_T1_PANEL_SCHEMA,
        "schema_version": EDITING_T1_PANEL_SCHEMA_VERSION,
        "status": EDITING_T1_PANEL_STATUS,
        "training_authorized": False,
        "source": {
            "forensics_file_sha256": "f" * 64,
            "semantic_sidecar_file_sha256": "7" * 64,
            "semantic_sidecar_manifest_file_sha256": "8" * 64,
            "charge_policy_audit_file_sha256": "3" * 64,
            "charge_policy_exclusions_file_sha256": "4" * 64,
            "charge_policy_exclusion_payload_sha256": "5" * 64,
            "charge_policy_source_input_inventory_sha256": "6" * 64,
            "capacity_census_file_sha256": census_file_sha256,
            "capacity_census_sha256": census["census_sha256"],
            **active8_identity,
        },
        "selection": selection,
        "census": panel_census,
        "thresholds": {
            "minimum_unique_state_teacher_successor_top1": None,
            "minimum_unique_state_teacher_successor_probability": None,
            "maximum_unique_state_teacher_successor_nll": None,
            "maximum_global_repeated_state_excess_nll_over_empirical_entropy": None,
        },
        "within_family_repeated_state_panels": {
            family: (
                [{"rows": [{"source_row_index": 0}]}]
                if family in repeated_families
                else []
            )
            for family in FAMILIES
        },
    }
    panel = {
        **panel_body,
        "artifact_sha256": t1_modal._stable_sha256(panel_body),
    }
    contract = build_editing_t1_runtime_contract(
        gate_zero_runtime_contract_sha256="e" * 64,
        panel=panel,
        capacity_census=census,
        capacity_census_file_sha256=census_file_sha256,
        active8_identity=active8_identity,
        optimization={
            "steps": 2,
            "learning_rate": 0.001,
            "weight_decay": 0.0,
            "scopes": ["heads_only", "heads_plus_local_adapter", "all"],
            "report_points": [1, 2],
        },
        thresholds={
            "minimum_unique_state_teacher_successor_top1": 0.8,
            "minimum_unique_state_teacher_successor_probability": 0.6,
            "maximum_unique_state_teacher_successor_nll": 0.7,
            "maximum_global_repeated_state_excess_nll_over_empirical_entropy": 0.2,
        },
    )
    return validate_editing_t1_launch_authority(
        contract=contract,
        panel=panel,
        capacity_census=census,
        capacity_census_file_sha256=census_file_sha256,
        expected_active8_inventory_manifest_file_sha256=ACTIVE8_FILE_SHA256,
    )


LAUNCH_AUTHORITY = _synthetic_launch_authority()


def test_t1_worker_uses_frozen_authority_without_full_census_replay():
    panel = t1_worker._execution_panel_from_frozen_authority(
        LAUNCH_AUTHORITY,
        semantic_sidecar_file_sha256="7" * 64,
        semantic_sidecar_manifest_file_sha256="8" * 64,
    )

    assert panel is LAUNCH_AUTHORITY.panel

    with pytest.raises(SystemExit, match="semantic_sidecar_file_sha256"):
        t1_worker._execution_panel_from_frozen_authority(
            LAUNCH_AUTHORITY,
            semantic_sidecar_file_sha256="9" * 64,
            semantic_sidecar_manifest_file_sha256="8" * 64,
        )


def _worker_result(
    *,
    run_label: str = "editing-t1-test-v1",
    family: str = "cycle_attach",
    panel_kind: str = "unique_state",
    scope: str = "heads_only",
    encoded: bytes = b"durable-result\n",
    observed_gpu_name: str = "NVIDIA L4",
):
    return {
        "run_label": run_label,
        "family": family,
        "panel_kind": panel_kind,
        "scope": scope,
        "requested_gpu_class": "L4",
        "observed_gpu_name": observed_gpu_name,
        "output": (
            f"/artifacts/_editing_t1_successor/{run_label}/{panel_kind}.{family}.{scope}.json"
        ),
        "output_sha256": hashlib.sha256(encoded).hexdigest(),
        "output_bytes": len(encoded),
        "result_sha256": RESULT_SHA256,
        "contract_sha256": "4" * 64,
        "numeric_thresholds_sha256": "5" * 64,
        "panel_artifact_sha256": "6" * 64,
        "panel_selection_sha256": "7" * 64,
        "panel_census_sha256": "8" * 64,
        "panel_capacity_strata_sha256": "9" * 64,
        "active8_inventory_manifest_file_sha256": ACTIVE8_FILE_SHA256,
        "active8_inventory_sha256": "b" * 64,
        "active8_effective_source_corpus_cache_sha256": "c" * 64,
        "active8_unified_packed_manifest_sha256": "d" * 64,
        "active8_support_contract_sha256": "e" * 64,
        "cache_receipts": copy.deepcopy(CACHE_RECEIPTS),
        "successor_cache_manifest_receipt": copy.deepcopy(
            CACHE_MANIFEST_RECEIPT
        ),
    }


def _one_arm_launch_receipt(
    *,
    run_label: str = "editing-t1-test-v1",
    result: dict | None = None,
):
    task = (
        run_label,
        "cycle_attach",
        "unique_state",
        "heads_only",
        RESOLVED_ACTIVE8_INVENTORY,
        ACTIVE8_FILE_SHA256,
    )
    return build_t1_modal_launch_receipt(
        run_label=run_label,
        source_commit=SOURCE_COMMIT,
        requested_gpu_class="L4",
        tasks=(task,),
        results=(_worker_result(run_label=run_label) if result is None else result,),
    )


def build_task_matrix(*args, **kwargs):
    kwargs.setdefault("active8_inventory", ACTIVE8_INVENTORY)
    kwargs.setdefault(
        "active8_inventory_file_sha256",
        ACTIVE8_FILE_SHA256,
    )
    kwargs.setdefault("launch_authority", LAUNCH_AUTHORITY)
    return _build_task_matrix(*args, **kwargs)


def test_t1_modal_surface_rejects_dirty_serialized_tree(
    monkeypatch,
):
    outputs = iter(("a" * 40, " M src/compose_v4/model.py\n"))

    def fake_run(*_args, **_kwargs):
        return type("Completed", (), {"stdout": next(outputs)})()

    monkeypatch.setattr(t1_modal.subprocess, "run", fake_run)
    with pytest.raises(RuntimeError, match="dirty serialized-code tree"):
        _require_clean_serialized_tree()


def test_t1_modal_surface_atomically_retains_independent_launch_receipt(
    tmp_path,
    monkeypatch,
):
    def reject_hard_link(*_args, **_kwargs):
        raise PermissionError(1, "operation not permitted")

    monkeypatch.setattr("os.link", reject_hard_link)
    receipt = _one_arm_launch_receipt()
    output = tmp_path / "receipts" / "editing-t1-test-v1.json"

    write_t1_modal_launch_receipt(receipt, output)
    observed = load_t1_modal_launch_receipt(output)

    assert observed == receipt
    assert observed["arms"][0]["output_sha256"] == hashlib.sha256(b"durable-result\n").hexdigest()
    assert observed["arms"][0]["result_sha256"] == RESULT_SHA256
    assert observed["arms"][0]["cache_receipts"] == CACHE_RECEIPTS
    assert observed["arms"][0]["result_relative_path"] == (
        "editing-t1-test-v1/unique_state.cycle_attach.heads_only.json"
    )
    assert t1_modal_launch_receipt_path("editing-t1-test-v1").name == ("editing-t1-test-v1.json")

    conflicting = _one_arm_launch_receipt(
        result=_worker_result(observed_gpu_name="NVIDIA A10"),
    )
    with pytest.raises(FileExistsError, match="already differs"):
        write_t1_modal_launch_receipt(conflicting, output)


def test_t1_modal_surface_rejects_incomplete_or_escaping_launch_receipts():
    incomplete = _worker_result()
    incomplete.pop("cache_receipts")
    with pytest.raises(ValueError, match="cache_receipts"):
        _one_arm_launch_receipt(result=incomplete)

    receipt = _one_arm_launch_receipt()
    escaped = copy.deepcopy(receipt)
    escaped["arms"][0]["result_relative_path"] = "../stolen.json"
    body = {key: value for key, value in escaped.items() if key != "receipt_sha256"}
    escaped["receipt_sha256"] = t1_modal._stable_sha256(body)
    with pytest.raises(ValueError, match="escapes"):
        validate_t1_modal_launch_receipt(escaped)


def test_t1_modal_surface_requires_explicit_family_and_defaults_to_one_arm():
    with pytest.raises(ValueError, match="families must be selected explicitly"):
        build_task_matrix("editing-t1-test-v1", families="")

    tasks = build_task_matrix(
        "editing-t1-test-v1",
        families="cycle_attach",
    )

    assert tasks == (
        (
            "editing-t1-test-v1",
            "cycle_attach",
            "unique_state",
            "heads_only",
            RESOLVED_ACTIVE8_INVENTORY,
            ACTIVE8_FILE_SHA256,
        ),
    )


def test_t1_modal_surface_requires_explicit_all_scopes_opt_in():
    tasks = build_task_matrix(
        "editing-t1-test-v1",
        families="cycle_attach",
        all_scopes=True,
    )

    assert tuple(task[3] for task in tasks) == SCOPES


def test_t1_modal_surface_skips_nonexistent_local_adapter_arms():
    tasks = build_task_matrix(
        "editing-t1-test-v1",
        families="",
        all_families=True,
        all_scopes=True,
    )
    assert len(tasks) == 2 * len(FAMILIES) + len(LOCAL_ADAPTER_FAMILIES)
    for task in tasks:
        if task[1] not in LOCAL_ADAPTER_FAMILIES:
            assert task[3] != "heads_plus_local_adapter"

    with pytest.raises(ValueError, match="changes no parameters"):
        build_task_matrix(
            "editing-t1-test-v1",
            families="atom_insert",
            scopes="heads_plus_local_adapter",
        )


def test_t1_modal_surface_does_not_hide_required_scope_failures(
    monkeypatch,
):
    original = t1_modal.require_editing_t1_family_scope_applicable

    def fail_required_scope(family: str, scope: str):
        if family == "atom_insert" and scope == "all":
            raise t1_modal.EditingT1RuntimeError("required all-scope failure")
        return original(family, scope)

    monkeypatch.setattr(
        t1_modal,
        "require_editing_t1_family_scope_applicable",
        fail_required_scope,
    )
    with pytest.raises(ValueError, match="required all-scope failure"):
        build_task_matrix(
            "editing-t1-test-v1",
            families="atom_insert",
            all_scopes=True,
        )


def test_t1_modal_surface_all_families_does_not_imply_all_scopes():
    tasks = build_task_matrix(
        "editing-t1-test-v1",
        families="",
        all_families=True,
    )

    assert len(tasks) == len(FAMILIES)
    assert {task[3] for task in tasks} == {"heads_only"}


def test_t1_modal_surface_rejects_unavailable_or_ambiguous_panels():
    with pytest.raises(
        ValueError,
        match="no empirical within-family repeated multi-successor panel",
    ):
        build_task_matrix(
            "editing-t1-test-v1",
            families="atom_delete",
            panel_kinds="within_family_repeated_state_distribution",
        )
    with pytest.raises(ValueError, match="requires exactly"):
        build_task_matrix(
            "editing-t1-test-v1",
            families="cycle_attach",
            panel_kinds="global_repeated_state_distribution",
        )


def test_t1_modal_surface_repeated_panel_is_explicit_and_bounded():
    tasks = build_task_matrix(
        "editing-t1-test-v1",
        families="cycle_attach,bond_reroute",
        panel_kinds="within_family_repeated_state_distribution",
        scopes="heads_only,all",
    )

    assert len(tasks) == 4
    assert {task[1] for task in tasks} == {"cycle_attach", "bond_reroute"}
    assert {task[2] for task in tasks} == {"within_family_repeated_state_distribution"}
    assert {task[3] for task in tasks} == {"heads_only", "all"}


def test_t1_modal_surface_derives_repeated_availability_from_selected_panel():
    authority = _synthetic_launch_authority(
        repeated_families={
            "atom_insert",
            "atom_delete",
            "atom_restate",
            "bond_reroute",
            "ring_system_restate",
        }
    )

    tasks = build_task_matrix(
        "editing-t1-test-v1",
        families="atom_delete",
        panel_kinds="within_family_repeated_state_distribution",
        launch_authority=authority,
    )
    assert tuple(task[1] for task in tasks) == ("atom_delete",)
    with pytest.raises(
        ValueError,
        match="no empirical within-family repeated multi-successor panel",
    ):
        build_task_matrix(
            "editing-t1-test-v1",
            families="cycle_attach",
            panel_kinds="within_family_repeated_state_distribution",
            launch_authority=authority,
        )


def test_t1_modal_surface_global_repeated_law_is_one_explicit_mixed_family_arm():
    tasks = build_task_matrix(
        "editing-t1-test-v1",
        families="all_families",
        panel_kinds="global_repeated_state_distribution",
    )

    assert tasks == (
        (
            "editing-t1-test-v1",
            "all_families",
            "global_repeated_state_distribution",
            "heads_only",
            RESOLVED_ACTIVE8_INVENTORY,
            ACTIVE8_FILE_SHA256,
        ),
    )

    with pytest.raises(ValueError, match="requires exactly"):
        build_task_matrix(
            "editing-t1-test-v1",
            families="all_families,cycle_attach",
            panel_kinds="global_repeated_state_distribution",
        )


def test_t1_modal_surface_requires_physical_active8_path_and_hash():
    with pytest.raises(ValueError, match="active8_inventory must be selected"):
        _build_task_matrix(
            "editing-t1-test-v1",
            families="cycle_attach",
            active8_inventory_file_sha256=ACTIVE8_FILE_SHA256,
        )
    with pytest.raises(ValueError, match="lowercase SHA-256"):
        _build_task_matrix(
            "editing-t1-test-v1",
            families="cycle_attach",
            active8_inventory=ACTIVE8_INVENTORY,
            active8_inventory_file_sha256="not-a-sha",
        )


def test_t1_modal_surface_accepts_inventory_under_symlinked_artifact_root(
    monkeypatch,
    tmp_path: Path,
):
    physical_root = tmp_path / "physical-artifacts"
    physical_root.mkdir()
    mounted_root = tmp_path / "artifacts"
    mounted_root.symlink_to(physical_root, target_is_directory=True)
    monkeypatch.setattr(t1_modal, "ARTIFACT_ROOT", mounted_root)

    resolved, digest = t1_modal._resolve_active8_launch_binding(
        str(mounted_root / "inventory.json"),
        ACTIVE8_FILE_SHA256,
    )

    assert resolved == str(physical_root / "inventory.json")
    assert digest == ACTIVE8_FILE_SHA256


def test_t1_modal_surface_rejects_symlink_escape_from_artifact_root(
    monkeypatch,
    tmp_path: Path,
):
    physical_root = tmp_path / "physical-artifacts"
    physical_root.mkdir()
    mounted_root = tmp_path / "artifacts"
    mounted_root.symlink_to(physical_root, target_is_directory=True)
    outside = tmp_path / "outside"
    outside.mkdir()
    (physical_root / "escape").symlink_to(outside, target_is_directory=True)
    monkeypatch.setattr(t1_modal, "ARTIFACT_ROOT", mounted_root)

    with pytest.raises(ValueError, match="mounted artifact volume"):
        t1_modal._resolve_active8_launch_binding(
            str(mounted_root / "escape" / "inventory.json"),
            ACTIVE8_FILE_SHA256,
        )


def test_t1_modal_surface_uses_only_current_authority_and_fails_closed(
    monkeypatch,
):
    assert t1_modal.T1_CONTRACT_RELATIVE_PATH == EDITING_T1_V8_CONTRACT_RELATIVE_PATH
    assert t1_modal.T1_PANEL_RELATIVE_PATH == EDITING_T1_V4_PANEL_RELATIVE_PATH
    assert (
        t1_modal.T1_CAPACITY_CENSUS_RELATIVE_PATH
        == EDITING_T1_V4_CAPACITY_CENSUS_RELATIVE_PATH
    )

    def absent_authority(**_kwargs):
        raise t1_modal.EditingT1RuntimeError("V4 launch authority is absent")

    monkeypatch.setattr(
        t1_modal,
        "load_editing_t1_launch_authority",
        absent_authority,
    )
    with pytest.raises(ValueError, match="V4 launch authority is absent"):
        _build_task_matrix(
            "editing-t1-test-v1",
            families="cycle_attach",
            active8_inventory=ACTIVE8_INVENTORY,
            active8_inventory_file_sha256=ACTIVE8_FILE_SHA256,
        )

    with pytest.raises(ValueError, match="another physical Active8"):
        _build_task_matrix(
            "editing-t1-test-v1",
            families="cycle_attach",
            active8_inventory=ACTIVE8_INVENTORY,
            active8_inventory_file_sha256="f" * 64,
            launch_authority=LAUNCH_AUTHORITY,
        )


def test_t1_modal_worker_names_every_current_authority_path_explicitly():
    command = t1_modal._editing_t1_worker_command(
        active8_inventory=RESOLVED_ACTIVE8_INVENTORY,
        active8_inventory_file_sha256=ACTIVE8_FILE_SHA256,
        family="cycle_attach",
        panel_kind="unique_state",
        scope="heads_only",
        successor_cache_root=t1_modal.ARTIFACT_ROOT / "cache",
        successor_cache_receipt=t1_modal.ARTIFACT_ROOT / "cache" / "receipt.json",
        output=t1_modal.ARTIFACT_ROOT / "result.json",
    )
    values_by_flag = {
        flag: command[command.index(flag) + 1]
        for flag in ("--t1-contract", "--panel", "--capacity-census")
    }
    assert values_by_flag == {
        "--t1-contract": str(
            t1_modal.REMOTE_ROOT / EDITING_T1_V8_CONTRACT_RELATIVE_PATH
        ),
        "--panel": str(
            t1_modal.REMOTE_ROOT / EDITING_T1_V4_PANEL_RELATIVE_PATH
        ),
        "--capacity-census": str(
            t1_modal.REMOTE_ROOT
            / EDITING_T1_V4_CAPACITY_CENSUS_RELATIVE_PATH
        ),
    }
    assert command[command.index("--successor-cache-root") + 1] == str(
        t1_modal.ARTIFACT_ROOT / "cache"
    )
    assert command[command.index("--successor-cache-receipt") + 1] == str(
        t1_modal.ARTIFACT_ROOT / "cache" / "receipt.json"
    )

    validation_command = t1_modal._editing_t1_cache_validation_command(
        active8_inventory=RESOLVED_ACTIVE8_INVENTORY,
        active8_inventory_file_sha256=ACTIVE8_FILE_SHA256,
        family="cycle_attach",
        panel_kind="unique_state",
        successor_cache_root=t1_modal.ARTIFACT_ROOT / "cache",
        receipt_path=t1_modal.ARTIFACT_ROOT / "cache" / "receipt.json",
    )
    assert "--validate-successor-cache" in validation_command
    assert validation_command[validation_command.index("--device") + 1] == "cpu"
    assert validation_command[
        validation_command.index("--successor-cache-receipt") + 1
    ] == str(t1_modal.ARTIFACT_ROOT / "cache" / "receipt.json")


def test_t1_modal_cpu_cache_build_precedes_failure_isolated_gpu_dispatch():
    source = Path(t1_modal.__file__).read_text()
    main_source = source[source.index("def main(") :]

    assert main_source.index("build_panel_successor_cache.starmap") < main_source.index(
        "runner.starmap"
    )
    assert "T1_CPU_CACHE_BUILD_FAILED_NO_GPU_DISPATCH" in main_source
    assert "return_exceptions=True" in main_source
    assert "_editing_t1_cache_validation_command" in source


def test_t1_modal_worker_failure_preserves_bounded_inner_diagnostics():
    completed = subprocess.CompletedProcess(
        args=["python", "worker.py"],
        returncode=7,
        stdout="prefix-" + "o" * 9000,
        stderr="prefix-" + "e" * 9000,
    )

    with pytest.raises(RuntimeError, match="T1 inner worker failed") as captured:
        t1_modal._require_worker_command_success(completed)

    message = str(captured.value)
    assert '"returncode": 7' in message
    assert "o" * 8000 in message
    assert "e" * 8000 in message
    assert "prefix-" not in message


def test_t1_modal_surface_isolates_failed_arm_from_successful_sibling():
    tasks = (
        (
            "editing-t1-test-v1",
            "cycle_attach",
            "unique_state",
            "heads_only",
            RESOLVED_ACTIVE8_INVENTORY,
            ACTIVE8_FILE_SHA256,
        ),
        (
            "editing-t1-test-v1",
            "bond_reorder",
            "unique_state",
            "heads_only",
            RESOLVED_ACTIVE8_INVENTORY,
            ACTIVE8_FILE_SHA256,
        ),
    )
    successful = _worker_result()
    error = RuntimeError("inner bond diagnostic")

    successful_tasks, successful_results, failures = (
        t1_modal.partition_t1_modal_results(tasks, (successful, error))
    )

    assert successful_tasks == (tasks[0],)
    assert successful_results == (successful,)
    assert failures == (
        {
            "family": "bond_reorder",
            "panel_kind": "unique_state",
            "scope": "heads_only",
            "error_type": "builtins.RuntimeError",
            "error_message": "inner bond diagnostic",
        },
    )


def test_t1_modal_surface_atomically_retains_failure_receipt(
    tmp_path: Path,
    monkeypatch,
):
    def reject_hard_link(*_args, **_kwargs):
        raise PermissionError(1, "operation not permitted")

    monkeypatch.setattr("os.link", reject_hard_link)
    failure = {
        "family": "bond_reorder",
        "panel_kind": "unique_state",
        "scope": "heads_only",
        "error_type": "builtins.RuntimeError",
        "error_message": "visible diagnostic",
    }
    receipt = t1_modal.build_t1_modal_failure_receipt(
        run_label="editing-t1-test-v1",
        source_commit=SOURCE_COMMIT,
        requested_gpu_class="L4",
        active8_inventory=ACTIVE8_INVENTORY,
        active8_inventory_file_sha256=ACTIVE8_FILE_SHA256,
        failures=(failure,),
    )
    output = tmp_path / "failures" / "editing-t1-test-v1.json"

    t1_modal.write_t1_modal_failure_receipt(receipt, output)
    observed = json.loads(output.read_text())

    assert observed == receipt
    assert observed["status"] == "INCOMPLETE_T1_MODAL_INVOCATION"
    assert observed["training_authorized"] is False
    assert observed["gate_decision"] is None
    assert observed["failures"] == [failure]
