from __future__ import annotations

import json
from pathlib import Path

import pytest

from compose_v4.experiments import (
    t4_nodistill_topology_four_call_local_runtime as runtime,
)
from compose_v4.experiments.t4_nodistill_topology_four_call_local import (
    CONTAINER_RECIPE_RELATIVE_PATH,
    LOCAL_EXECUTION_CONTRACT_RELATIVE_PATH,
    MODAL_EXECUTION_PAYLOAD_SHA256,
    QVINA_SHA256,
    RECEPTOR_SHA256,
    build_local_execution_payload,
    load_envelope,
    payload_identity,
    publish_once_durable,
    validate_local_assets,
    validate_local_execution_binding,
)

ROOT = Path(__file__).resolve().parents[1]


def _placeholder_source_binding() -> dict[str, object]:
    return {
        "schema_version": "t4_exact_commit_local_source_binding_v1",
        "code_revision": "0" * 40,
        "git_tree": "1" * 40,
        "file_count": 0,
        "files": {},
    }


def test_local_binding_preserves_exact_queries_evaluator_and_budget() -> None:
    payload = build_local_execution_payload(ROOT, _placeholder_source_binding())
    scientific, _ = load_envelope(ROOT / payload["scientific_scored_contract"]["path"])
    assert payload["status"] == ("PENDING_EXACT_BINDING_SPECIFIC_LAUNCH_AUTHORIZATION")
    assert payload["query_lock"] == {
        "source": "scientific_scored_contract.queries",
        "ordered_queries_payload_sha256": payload_identity(scientific["queries"]),
        "ordered_query_ids": [row["query_id"] for row in scientific["queries"]],
        "query_count": 4,
        "attempt_ceiling_each": 1,
        "candidate_mutation": False,
    }
    assert payload["evaluator_protocol"]["qvina02_sha256"] == QVINA_SHA256
    assert payload["evaluator_protocol"]["receptor_sha256"] == RECEPTOR_SHA256
    assert payload["evaluator_protocol"]["docking_box"] == [
        [26.413, 11.282, 27.238],
        [18.521, 17.479, 19.995],
    ]
    assert payload["evaluator_protocol"]["docking_seed"] == 20260919
    assert payload["prior_modal_execution_binding"]["payload_sha256"] == (
        MODAL_EXECUTION_PAYLOAD_SHA256
    )
    assert payload["decision_equivalence"]["scientific_decision_change"] is False
    assert payload["execution"] == {
        "query_count": 4,
        "execution_order": "scientific_contract_query_order_sequential",
        "maximum_parallel_workers": 1,
        "cpu_per_worker": 1,
        "attempts_per_query": 1,
        "automatic_retries": 0,
        "replacement_queries": 0,
        "backfill_queries": 0,
        "replicate_calls": 0,
        "publish_complete_query_lock_before_first_container_score_process": True,
        "publish_all_reservations_before_first_container_score_process": True,
        "started_receipt_precedes_each_possible_score_call": True,
        "one_terminal_receipt_per_started_query": True,
        "started_without_terminal_is_terminally_incomplete_and_never_retried": True,
        "deterministic_reduction_order": "scientific_contract_query_order",
    }


def test_local_protocol_argv_is_exactly_modal_bound_adapter_protocol() -> None:
    payload = build_local_execution_payload(ROOT, _placeholder_source_binding())
    protocol = payload["evaluator_protocol"]
    assert protocol["ligand_preparation"] == [
        ["obabel", "-:{canonical_smiles}", "--gen3D", "-O", "{ligand_mol}"],
        ["obabel", "{ligand_mol}", "-O", "{ligand_pdbqt}"],
    ]
    command = protocol["docking_command"]
    assert command[0] == "/opt/dock/qvina02"
    assert command[command.index("--receptor") + 1] == ("/opt/dock/receptors/parp1.pdbqt")
    assert command[command.index("--cpu") + 1] == "1"
    assert command[command.index("--num_modes") + 1] == "10"
    assert command[command.index("--exhaustiveness") + 1] == "1"
    assert command[command.index("--seed") + 1] == "20260919"
    assert protocol["adapter"]["identical_to_modal_execution_v2_capsule"] is True


def test_container_recipe_and_runtime_forbid_source_egress_and_network() -> None:
    dockerfile = (ROOT / CONTAINER_RECIPE_RELATIVE_PATH).read_text()
    normalized = dockerfile.upper()
    assert "COPY " not in normalized
    assert "ADD " not in normalized
    assert "CURL " not in normalized
    assert "WGET " not in normalized
    assert "OPENBABEL" in normalized
    runtime_source = (
        ROOT / "src/compose_v4/experiments/t4_nodistill_topology_four_call_local_runtime.py"
    ).read_text()
    assert '"--network",\n        "none"' in runtime_source
    assert '"--platform",\n        "linux/amd64"' in runtime_source
    assert '"/workspace", read_only=True' in runtime_source
    assert runtime_source.index("validate_launch_authorization(root)") < (
        runtime_source.index("_validate_live_image(environment)")
    )
    assert Path(runtime.__file__).resolve().parents[3] == ROOT


def test_local_assets_match_locked_scientific_hashes() -> None:
    assert validate_local_assets() == {
        "qvina02_sha256": QVINA_SHA256,
        "receptor_sha256": RECEPTOR_SHA256,
    }


def test_durable_publisher_is_self_hashed_idempotent_and_immutable(
    tmp_path: Path,
) -> None:
    path = tmp_path / "receipt.json"
    payload = {"schema_version": "test_receipt_v1", "value": 1}
    identity = publish_once_durable(path, payload)
    observed, observed_identity = load_envelope(path)
    assert observed == payload
    assert observed_identity == identity
    assert publish_once_durable(path, payload) == identity
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        publish_once_durable(path, {**payload, "value": 2})


def test_launch_refuses_before_any_docker_call_without_specific_receipt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden() -> dict[str, object]:
        raise AssertionError("Docker must not be inspected before authorization")

    monkeypatch.setattr(runtime, "_inspect_image", forbidden)
    with pytest.raises(FileNotFoundError):
        runtime.launch(ROOT)


def test_sealed_local_binding_when_present() -> None:
    path = ROOT / LOCAL_EXECUTION_CONTRACT_RELATIVE_PATH
    if not path.exists():
        pytest.skip("local execution binding is sealed after its source commit")
    binding, identity, scientific = validate_local_execution_binding(
        ROOT, require_local_assets=True
    )
    envelope = json.loads(path.read_text())
    assert envelope["payload_sha256"] == identity
    assert binding["status"] == ("PENDING_EXACT_BINDING_SPECIFIC_LAUNCH_AUTHORIZATION")
    assert binding["query_lock"]["query_count"] == len(scientific["queries"]) == 4
    assert binding["costs_spent_during_preparation"] == {
        "docking_calls": 0,
        "oracle_calls": 0,
        "modal_calls": 0,
        "container_builds": 0,
        "container_runs": 0,
    }
