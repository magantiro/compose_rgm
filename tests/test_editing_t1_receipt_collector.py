from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("modal")

import modal_apps.run_editing_t1_successor_gate as t1_modal
import scripts.collect_editing_t1_arm_results_manifest as collector

ACTIVE8_FILE_SHA256 = "a" * 64
ACTIVE8_INVENTORY = "/artifacts/active8_trace_inventory_v1/ACTIVE8_TRACE_INVENTORY.json"
SOURCE_COMMIT = "c" * 40
CACHE_RECEIPTS = [
    {
        "packed_shard_content_sha256": "1" * 64,
        "cache_content_sha256": "2" * 64,
        "encoded_sha256": "3" * 64,
        "record_count": 1,
    }
]
CACHE_MANIFEST_RECEIPT = {
    "manifest_relative_path": "manifests/" + "4" * 64 + ".json",
    "manifest_sha256": "4" * 64,
    "manifest_file_sha256": "5" * 64,
    "manifest_file_bytes": 123,
    "selected_trace_set_sha256": "6" * 64,
    "initial_model_state_sha256": "7" * 64,
}


def _contract():
    return SimpleNamespace(
        sha256="4" * 64,
        numeric_thresholds_sha256="5" * 64,
        payload={
            "panel_artifact_sha256": "6" * 64,
            "panel_selection_sha256": "7" * 64,
            "panel_census_sha256": "8" * 64,
            "panel_capacity_strata_sha256": "9" * 64,
            "active8_inventory_manifest_file_sha256": ACTIVE8_FILE_SHA256,
            "active8_inventory_sha256": "b" * 64,
            "active8_effective_source_corpus_cache_sha256": "c" * 64,
            "active8_unified_packed_manifest_sha256": "d" * 64,
            "active8_support_contract_sha256": "e" * 64,
        },
    )


def _arm(
    family: str,
    *,
    panel_kind: str = "unique_state",
    scope: str = "heads_only",
):
    return {
        "family": family,
        "panel_kind": panel_kind,
        "scope": scope,
    }


def _install_contract_and_manifest_fakes(
    monkeypatch,
    *,
    contract,
    planned,
    captured,
):
    monkeypatch.setattr(
        collector,
        "load_editing_t1_runtime_contract",
        lambda _path: contract,
    )
    monkeypatch.setattr(
        collector,
        "planned_t1_arms",
        lambda _contract: tuple(planned),
    )

    def fake_build(**kwargs):
        captured["build"] = kwargs
        ordered = [
            {
                **arm,
                "relative_path": kwargs["result_paths"][
                    (
                        arm["family"],
                        arm["panel_kind"],
                        arm["scope"],
                    )
                ]
                .relative_to(kwargs["manifest_directory"])
                .as_posix(),
            }
            for arm in planned
        ]
        return {
            "status": "COMPLETE_PHYSICAL_T1_ARM_RESULTS",
            "manifest_sha256": "f" * 64,
            "results": ordered,
        }

    def fake_validate(payload, **kwargs):
        captured["validate"] = {
            "payload": payload,
            **kwargs,
        }

    monkeypatch.setattr(
        collector,
        "build_t1_arm_results_manifest",
        fake_build,
    )
    monkeypatch.setattr(
        collector,
        "validate_t1_arm_results_manifest",
        fake_validate,
    )


def _write_invocation(
    *,
    result_root: Path,
    receipt_root: Path,
    run_label: str,
    arm,
    contract,
    source_commit: str = SOURCE_COMMIT,
    contract_sha256: str | None = None,
    materialize: bool = True,
):
    family = arm["family"]
    panel_kind = arm["panel_kind"]
    scope = arm["scope"]
    encoded = f"{run_label}:{family}:{panel_kind}:{scope}\n".encode()
    result_relative = Path(run_label) / f"{panel_kind}.{family}.{scope}.json"
    result_path = result_root / result_relative
    if materialize:
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_bytes(encoded)
    task = (
        run_label,
        family,
        panel_kind,
        scope,
        ACTIVE8_INVENTORY,
        ACTIVE8_FILE_SHA256,
    )
    worker_result = {
        "run_label": run_label,
        "family": family,
        "panel_kind": panel_kind,
        "scope": scope,
        "requested_gpu_class": "L4",
        "observed_gpu_name": "NVIDIA L4",
        "output": f"/artifacts/_editing_t1_successor/{result_relative.as_posix()}",
        "output_sha256": hashlib.sha256(encoded).hexdigest(),
        "output_bytes": len(encoded),
        "result_sha256": hashlib.sha256(b"semantic-result" + encoded).hexdigest(),
        "contract_sha256": (contract.sha256 if contract_sha256 is None else contract_sha256),
        "numeric_thresholds_sha256": contract.numeric_thresholds_sha256,
        "panel_artifact_sha256": contract.payload["panel_artifact_sha256"],
        "panel_selection_sha256": contract.payload["panel_selection_sha256"],
        "panel_census_sha256": contract.payload["panel_census_sha256"],
        "panel_capacity_strata_sha256": contract.payload["panel_capacity_strata_sha256"],
        **{field: contract.payload[field] for field in t1_modal.ACTIVE8_T1_IDENTITY_FIELDS},
        "cache_receipts": copy.deepcopy(CACHE_RECEIPTS),
        "successor_cache_manifest_receipt": copy.deepcopy(
            CACHE_MANIFEST_RECEIPT
        ),
    }
    receipt = t1_modal.build_t1_modal_launch_receipt(
        run_label=run_label,
        source_commit=source_commit,
        requested_gpu_class="L4",
        tasks=(task,),
        results=(worker_result,),
    )
    receipt_path = receipt_root / f"{run_label}.json"
    t1_modal.write_t1_modal_launch_receipt(receipt, receipt_path)
    return receipt_path, result_path


def test_t1_receipt_collector_builds_deterministic_physical_manifest(
    tmp_path,
    monkeypatch,
):
    contract = _contract()
    planned = [_arm("atom_insert"), _arm("cycle_attach")]
    captured = {}
    _install_contract_and_manifest_fakes(
        monkeypatch,
        contract=contract,
        planned=planned,
        captured=captured,
    )
    result_root = tmp_path / "materialized-results"
    result_root.mkdir()
    receipt_root = tmp_path / "receipts"
    contract_path = tmp_path / "runtime-contract.json"
    contract_path.write_text("{}\n")
    first_receipt, first_result = _write_invocation(
        result_root=result_root,
        receipt_root=receipt_root,
        run_label="editing-t1-atom-insert-v1",
        arm=planned[0],
        contract=contract,
    )
    second_receipt, second_result = _write_invocation(
        result_root=result_root,
        receipt_root=receipt_root,
        run_label="editing-t1-cycle-attach-v1",
        arm=planned[1],
        contract=contract,
    )
    output = result_root / "T1_ARM_RESULTS_MANIFEST.json"

    manifest = collector.collect_editing_t1_arm_results_manifest(
        runtime_contract_path=contract_path,
        result_directory=result_root,
        launch_receipt_paths=(second_receipt, first_receipt),
        output_path=output,
    )

    assert json.loads(output.read_bytes()) == manifest
    result_paths = captured["build"]["result_paths"]
    assert result_paths[("atom_insert", "unique_state", "heads_only")] == first_result.resolve()
    assert result_paths[("cycle_attach", "unique_state", "heads_only")] == second_result.resolve()
    retained = captured["build"]["retained_result_receipts"]
    assert (
        retained[("atom_insert", "unique_state", "heads_only")]["file_sha256"]
        == hashlib.sha256(first_result.read_bytes()).hexdigest()
    )
    assert (
        retained[("cycle_attach", "unique_state", "heads_only")]["cache_receipts"] == CACHE_RECEIPTS
    )

    repeated = collector.collect_editing_t1_arm_results_manifest(
        runtime_contract_path=contract_path,
        result_directory=result_root,
        launch_receipt_paths=(first_receipt, second_receipt),
        output_path=output,
    )
    assert repeated == manifest


def test_t1_receipt_collector_rejects_missing_duplicate_and_stale_arms(
    tmp_path,
    monkeypatch,
):
    contract = _contract()
    planned = [_arm("atom_insert"), _arm("cycle_attach")]
    _install_contract_and_manifest_fakes(
        monkeypatch,
        contract=contract,
        planned=planned,
        captured={},
    )
    result_root = tmp_path / "materialized-results"
    result_root.mkdir()
    receipt_root = tmp_path / "receipts"
    contract_path = tmp_path / "runtime-contract.json"
    contract_path.write_text("{}\n")
    first_receipt, _ = _write_invocation(
        result_root=result_root,
        receipt_root=receipt_root,
        run_label="editing-t1-atom-insert-v1",
        arm=planned[0],
        contract=contract,
    )
    with pytest.raises(
        collector.EditingT1ReceiptCollectionError,
        match="complete planned arm matrix",
    ):
        collector.collect_editing_t1_arm_results_manifest(
            runtime_contract_path=contract_path,
            result_directory=result_root,
            launch_receipt_paths=(first_receipt,),
            output_path=result_root / "manifest.json",
        )

    duplicate_receipt, _ = _write_invocation(
        result_root=result_root,
        receipt_root=receipt_root,
        run_label="editing-t1-atom-insert-v2",
        arm=planned[0],
        contract=contract,
    )
    with pytest.raises(
        collector.EditingT1ReceiptCollectionError,
        match="duplicate or conflict",
    ):
        collector.collect_editing_t1_arm_results_manifest(
            runtime_contract_path=contract_path,
            result_directory=result_root,
            launch_receipt_paths=(first_receipt, duplicate_receipt),
            output_path=result_root / "manifest.json",
        )

    stale_receipt, _ = _write_invocation(
        result_root=result_root,
        receipt_root=receipt_root,
        run_label="editing-t1-cycle-attach-stale-v1",
        arm=planned[1],
        contract=contract,
        contract_sha256="0" * 64,
    )
    with pytest.raises(
        collector.EditingT1ReceiptCollectionError,
        match="stale.*contract_sha256",
    ):
        collector.collect_editing_t1_arm_results_manifest(
            runtime_contract_path=contract_path,
            result_directory=result_root,
            launch_receipt_paths=(first_receipt, stale_receipt),
            output_path=result_root / "manifest.json",
        )


def test_t1_receipt_collector_rejects_dirty_receipts_and_result_path_escapes(
    tmp_path,
    monkeypatch,
):
    contract = _contract()
    planned = [_arm("cycle_attach")]
    _install_contract_and_manifest_fakes(
        monkeypatch,
        contract=contract,
        planned=planned,
        captured={},
    )
    result_root = tmp_path / "materialized-results"
    result_root.mkdir()
    receipt_root = tmp_path / "receipts"
    contract_path = tmp_path / "runtime-contract.json"
    contract_path.write_text("{}\n")
    receipt_path, _ = _write_invocation(
        result_root=result_root,
        receipt_root=receipt_root,
        run_label="editing-t1-cycle-attach-v1",
        arm=planned[0],
        contract=contract,
    )
    dirty = json.loads(receipt_path.read_bytes())
    dirty["arms"][0].pop("cache_receipts")
    receipt_path.write_text(json.dumps(dirty))
    with pytest.raises(
        collector.EditingT1ReceiptCollectionError,
        match="dirty or incomplete",
    ):
        collector.collect_editing_t1_arm_results_manifest(
            runtime_contract_path=contract_path,
            result_directory=result_root,
            launch_receipt_paths=(receipt_path,),
            output_path=result_root / "manifest.json",
        )

    escaped_receipt, expected_local_path = _write_invocation(
        result_root=result_root,
        receipt_root=receipt_root,
        run_label="editing-t1-cycle-attach-escape-v1",
        arm=planned[0],
        contract=contract,
        materialize=False,
    )
    outside = tmp_path / "outside"
    outside.mkdir()
    outside_result = outside / expected_local_path.name
    outside_result.write_bytes(b"external-result\n")
    os.symlink(
        outside,
        expected_local_path.parent,
        target_is_directory=True,
    )
    with pytest.raises(
        collector.EditingT1ReceiptCollectionError,
        match="escapes",
    ):
        collector.collect_editing_t1_arm_results_manifest(
            runtime_contract_path=contract_path,
            result_directory=result_root,
            launch_receipt_paths=(escaped_receipt,),
            output_path=result_root / "manifest.json",
        )


def test_t1_receipt_collector_rejects_mixed_source_commits(
    tmp_path,
    monkeypatch,
):
    contract = _contract()
    planned = [_arm("atom_insert"), _arm("cycle_attach")]
    _install_contract_and_manifest_fakes(
        monkeypatch,
        contract=contract,
        planned=planned,
        captured={},
    )
    result_root = tmp_path / "materialized-results"
    result_root.mkdir()
    receipt_root = tmp_path / "receipts"
    contract_path = tmp_path / "runtime-contract.json"
    contract_path.write_text("{}\n")
    first_receipt, _ = _write_invocation(
        result_root=result_root,
        receipt_root=receipt_root,
        run_label="editing-t1-atom-insert-v1",
        arm=planned[0],
        contract=contract,
    )
    second_receipt, _ = _write_invocation(
        result_root=result_root,
        receipt_root=receipt_root,
        run_label="editing-t1-cycle-attach-v1",
        arm=planned[1],
        contract=contract,
        source_commit="d" * 40,
    )
    with pytest.raises(
        collector.EditingT1ReceiptCollectionError,
        match="different source commits",
    ):
        collector.collect_editing_t1_arm_results_manifest(
            runtime_contract_path=contract_path,
            result_directory=result_root,
            launch_receipt_paths=(first_receipt, second_receipt),
            output_path=result_root / "manifest.json",
        )
