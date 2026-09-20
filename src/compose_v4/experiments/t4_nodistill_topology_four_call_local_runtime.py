"""Guarded local Docker runtime for the four-call PARP1 diagnostic.

The host-side launch path refuses before invoking Docker unless a zero-oracle
environment attestation and exact binding-specific authorization are present.
The container receives only a query identifier and derives the molecule from
the immutable scientific contract.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from compose_v4.experiments.t4_nodistill_topology_four_call_local import (
    LOCAL_ENVIRONMENT_ATTESTATION_RELATIVE_PATH,
    LOCAL_EXECUTION_CONTRACT_RELATIVE_PATH,
    LOCAL_QVINA_PATH,
    LOCAL_RECEIPT_ROOT_RELATIVE_PATH,
    LOCAL_RECEPTOR_PATH,
    QVINA_SHA256,
    RECEPTOR_SHA256,
    SCORED_CONTRACT_RELATIVE_PATH,
    file_sha256,
    load_envelope,
    payload_identity,
    publish_once_durable,
    validate_launch_authorization,
    validate_local_assets,
    validate_local_execution_binding,
)

IMAGE_TAG = "compose-t4-topology-four-call-local:v1"
QUERY_LOCK_SCHEMA = "t4_nodistill_topology_four_call_local_query_lock_v1"
RESERVATION_SCHEMA = "t4_nodistill_topology_four_call_local_reservation_v1"
START_SCHEMA = "t4_nodistill_topology_four_call_local_start_v1"
RESULT_SCHEMA = "t4_nodistill_topology_four_call_local_result_v1"
FAILURE_SCHEMA = "t4_nodistill_topology_four_call_local_failure_v1"
REDUCTION_SCHEMA = "t4_nodistill_topology_four_call_local_reduction_v1"


def _stamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _fsync_directory(path: Path) -> None:
    directory_fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _fsync_tree(root: Path) -> None:
    directories = [path for path in root.rglob("*") if path.is_dir()]
    for directory in sorted(directories, key=lambda value: len(value.parts), reverse=True):
        _fsync_directory(directory)
    _fsync_directory(root)


def _docker_mount(source: Path, destination: str, *, read_only: bool) -> str:
    option = f"type=bind,src={source.resolve()},dst={destination}"
    return f"{option},readonly" if read_only else option


def _docker_prefix(repository_root: Path, receipt_root: Path | None = None) -> list[str]:
    root = repository_root.resolve()
    command = [
        "docker",
        "run",
        "--rm",
        "--network",
        "none",
        "--platform",
        "linux/amd64",
        "--cpus",
        "1",
        "--memory",
        "4g",
        "--pids-limit",
        "256",
        "--read-only",
        "--tmpfs",
        "/tmp:rw,exec,nosuid,size=1g",
        "--tmpfs",
        "/opt/dock:rw,exec,nosuid,size=8m",
        "--mount",
        _docker_mount(root, "/workspace", read_only=True),
        "--mount",
        _docker_mount(LOCAL_QVINA_PATH, "/inputs/qvina02", read_only=True),
        "--mount",
        _docker_mount(LOCAL_RECEPTOR_PATH, "/inputs/parp1.pdbqt", read_only=True),
    ]
    if receipt_root is not None:
        command.extend(
            [
                "--mount",
                _docker_mount(receipt_root, "/receipts", read_only=False),
            ]
        )
    command.append(IMAGE_TAG)
    return command


def _inspect_image() -> dict[str, Any]:
    completed = subprocess.run(
        ["docker", "image", "inspect", IMAGE_TAG],
        capture_output=True,
        check=True,
        text=True,
    )
    rows = json.loads(completed.stdout)
    if not isinstance(rows, list) or len(rows) != 1:
        raise ValueError("local four-call image inspection is ambiguous")
    row = rows[0]
    if row.get("Os") != "linux" or row.get("Architecture") != "amd64":
        raise ValueError("local four-call image is not Linux/amd64")
    return row


def _container_preflight_payload(repository_root: Path) -> dict[str, Any]:
    root = repository_root.resolve()
    binding, binding_identity = load_envelope(root / LOCAL_EXECUTION_CONTRACT_RELATIVE_PATH)
    if binding.get("status") != "PENDING_EXACT_BINDING_SPECIFIC_LAUNCH_AUTHORIZATION":
        raise ValueError("local execution binding status changed")
    observed = {
        "qvina02_sha256": file_sha256(Path("/inputs/qvina02")),
        "receptor_sha256": file_sha256(Path("/inputs/parp1.pdbqt")),
    }
    if observed != {
        "qvina02_sha256": QVINA_SHA256,
        "receptor_sha256": RECEPTOR_SHA256,
    }:
        raise ValueError("container-mounted evaluator assets changed")
    openbabel = shutil.which("obabel")
    if openbabel is None:
        raise FileNotFoundError("container Open Babel executable is absent")
    return {
        "schema_version": "t4_nodistill_topology_four_call_container_preflight_v1",
        "status": "PASS_ZERO_ORACLE",
        "local_execution_contract_payload_sha256": binding_identity,
        "platform": {
            "os": platform.system().lower(),
            "architecture": platform.machine().lower(),
        },
        "python_version": platform.python_version(),
        "openbabel_version": subprocess.check_output(["obabel", "-V"], text=True).strip(),
        "openbabel_sha256": file_sha256(Path(openbabel)),
        **observed,
        "docking_calls": 0,
        "oracle_calls": 0,
    }


def attest_environment(repository_root: Path) -> dict[str, Any]:
    root = repository_root.resolve()
    binding, binding_identity, _scientific = validate_local_execution_binding(
        root, require_local_assets=True
    )
    image = _inspect_image()
    completed = subprocess.run(
        [*_docker_prefix(root), "container-preflight"],
        capture_output=True,
        check=True,
        text=True,
    )
    preflight = json.loads(completed.stdout)
    if (
        preflight.get("status") != "PASS_ZERO_ORACLE"
        or preflight.get("local_execution_contract_payload_sha256") != binding_identity
        or preflight.get("platform") != {"os": "linux", "architecture": "x86_64"}
        or preflight.get("qvina02_sha256") != QVINA_SHA256
        or preflight.get("receptor_sha256") != RECEPTOR_SHA256
        or not isinstance(preflight.get("openbabel_sha256"), str)
        or len(preflight["openbabel_sha256"]) != 64
        or preflight.get("docking_calls") != 0
    ):
        raise ValueError("local container zero-oracle preflight changed")
    payload = {
        "schema_version": ("t4_nodistill_topology_four_call_local_environment_attestation_v1"),
        "status": "PASS_ZERO_ORACLE",
        "local_execution_contract_payload_sha256": binding_identity,
        "recipe_sha256": binding["container"]["recipe"]["sha256"],
        "image_tag": IMAGE_TAG,
        "image_id": image["Id"],
        "repo_digests": sorted(image.get("RepoDigests") or []),
        "platform": {"os": "linux", "architecture": "amd64"},
        "python_version": preflight["python_version"],
        "openbabel_version": preflight["openbabel_version"],
        "openbabel_sha256": preflight["openbabel_sha256"],
        "qvina02_sha256": preflight["qvina02_sha256"],
        "receptor_sha256": preflight["receptor_sha256"],
        "runtime_network": "none",
        "zero_oracle_preflight": preflight,
        "docking_calls": 0,
        "oracle_calls": 0,
        "attested_at_utc": _stamp(),
    }
    path = root / LOCAL_ENVIRONMENT_ATTESTATION_RELATIVE_PATH
    identity = publish_once_durable(path, payload)
    return {"path": str(path), "payload_sha256": identity, "payload": payload}


def _validate_live_image(environment: dict[str, Any]) -> None:
    image = _inspect_image()
    if image.get("Id") != environment.get("image_id"):
        raise ValueError("local four-call image differs from authorized attestation")


def _run_id(
    binding_identity: str,
    environment_identity: str,
    authorization_identity: str,
    query_ids: list[str],
) -> str:
    return payload_identity(
        {
            "schema_version": "t4_nodistill_topology_four_call_local_run_id_v1",
            "local_execution_contract_payload_sha256": binding_identity,
            "environment_attestation_payload_sha256": environment_identity,
            "launch_authorization_payload_sha256": authorization_identity,
            "ordered_query_ids": query_ids,
        }
    )


def _publish_prepared_run(
    receipt_root: Path,
    run_id: str,
    binding_identity: str,
    environment_identity: str,
    authorization_identity: str,
    scientific: dict[str, Any],
) -> Path:
    run_root = receipt_root / run_id
    if run_root.exists():
        raise FileExistsError("local four-call run already exists; replay is forbidden")
    staging = receipt_root / f".{run_id}.{os.getpid()}.prepare"
    if staging.exists():
        raise FileExistsError("local four-call staging run already exists")
    staging.mkdir(parents=True)
    common = {
        "run_id": run_id,
        "scientific_contract_payload_sha256": scientific_contract_identity(scientific),
        "local_execution_contract_payload_sha256": binding_identity,
        "environment_attestation_payload_sha256": environment_identity,
        "launch_authorization_payload_sha256": authorization_identity,
    }
    queries = scientific["queries"]
    lock = {
        "schema_version": QUERY_LOCK_SCHEMA,
        **common,
        "query_ids": [row["query_id"] for row in queries],
        "queries": queries,
        "charged_call_ceiling": 4,
        "automatic_retries": 0,
        "replacement_queries": 0,
        "backfill_queries": 0,
        "replicate_calls": 0,
        "locked_at_utc": _stamp(),
    }
    publish_once_durable(staging / "query_lock.json", lock)
    for ordinal, query in enumerate(queries):
        reservation = {
            "schema_version": RESERVATION_SCHEMA,
            **common,
            "query_id": query["query_id"],
            "query_ordinal": ordinal,
            "endpoint_key_sha256": query["endpoint_key_sha256"],
            "attempt_ordinal": 1,
            "maximum_attempts": 1,
            "reservation_consumes_only_authorized_attempt": True,
            "retry_replacement_backfill_or_replication_authorized": False,
            "reserved_at_utc": _stamp(),
        }
        publish_once_durable(
            staging / "queries" / query["query_id"] / "reservation.json",
            reservation,
        )
    dispatch = {
        "schema_version": "t4_nodistill_topology_four_call_local_dispatch_v1",
        **common,
        "query_ids": lock["query_ids"],
        "all_reservations_published": True,
        "container_score_processes_planned": 4,
        "execution_order": "scientific_contract_query_order_sequential",
        "automatic_retries": 0,
        "published_at_utc": _stamp(),
    }
    publish_once_durable(staging / "dispatch.json", dispatch)
    _fsync_tree(staging)
    receipt_root.mkdir(parents=True, exist_ok=True)
    os.replace(staging, run_root)
    _fsync_directory(receipt_root)
    return run_root


def scientific_contract_identity(scientific: dict[str, Any]) -> str:
    return payload_identity(scientific)


def _terminal_paths(run_root: Path, query_id: str) -> tuple[Path, Path]:
    folder = run_root / "queries" / query_id
    return folder / "result.json", folder / "failure.json"


def _publish_host_failure(
    run_root: Path,
    query: dict[str, Any],
    common: dict[str, Any],
    error: Exception,
) -> dict[str, Any]:
    result_path, failure_path = _terminal_paths(run_root, query["query_id"])
    if result_path.exists() or failure_path.exists():
        terminals = [path for path in (result_path, failure_path) if path.exists()]
        if len(terminals) != 1:
            raise RuntimeError("local four-call query has ambiguous terminal receipts")
        return load_envelope(terminals[0])[0]
    payload = {
        "schema_version": FAILURE_SCHEMA,
        **common,
        "query_id": query["query_id"],
        "endpoint_key_sha256": query["endpoint_key_sha256"],
        "canonical_smiles": query["canonical_smiles"],
        "cell_key": query["cell_key"],
        "status": "failed_no_retry",
        "error_type": type(error).__name__,
        "error_message": str(error),
        "attempt_consumed": 1,
        "retry_authorized": False,
        "failed_at_utc": _stamp(),
    }
    publish_once_durable(failure_path, payload)
    return payload


def launch(repository_root: Path) -> dict[str, Any]:
    root = repository_root.resolve()
    (
        binding,
        environment,
        environment_identity,
        _authorization,
        authorization_identity,
    ) = validate_launch_authorization(root)
    binding_identity = load_envelope(root / LOCAL_EXECUTION_CONTRACT_RELATIVE_PATH)[1]
    validate_local_assets()
    _validate_live_image(environment)
    scientific, scientific_identity = load_envelope(root / SCORED_CONTRACT_RELATIVE_PATH)
    if scientific_identity != binding["scientific_scored_contract"]["payload_sha256"]:
        raise ValueError("local launch scientific contract changed")
    query_ids = [row["query_id"] for row in scientific["queries"]]
    run_id = _run_id(
        binding_identity,
        environment_identity,
        authorization_identity,
        query_ids,
    )
    receipt_root = root / LOCAL_RECEIPT_ROOT_RELATIVE_PATH
    run_root = _publish_prepared_run(
        receipt_root,
        run_id,
        binding_identity,
        environment_identity,
        authorization_identity,
        scientific,
    )
    common = {
        "run_id": run_id,
        "scientific_contract_payload_sha256": scientific_identity,
        "local_execution_contract_payload_sha256": binding_identity,
        "environment_attestation_payload_sha256": environment_identity,
        "launch_authorization_payload_sha256": authorization_identity,
    }
    terminals = []
    for query in scientific["queries"]:
        folder = run_root / "queries" / query["query_id"]
        started = {
            "schema_version": START_SCHEMA,
            **common,
            "query_id": query["query_id"],
            "endpoint_key_sha256": query["endpoint_key_sha256"],
            "attempt_ordinal": 1,
            "docking_seed": scientific["evaluator"]["docking_seed"],
            "qvina02_sha256": QVINA_SHA256,
            "receptor_sha256": RECEPTOR_SHA256,
            "started_at_utc": _stamp(),
        }
        publish_once_durable(folder / "started.json", started)
        try:
            subprocess.run(
                [
                    *_docker_prefix(root, run_root),
                    "score-one",
                    "--query-id",
                    query["query_id"],
                ],
                capture_output=True,
                check=True,
                text=True,
                timeout=420,
            )
            result_path, failure_path = _terminal_paths(run_root, query["query_id"])
            paths = [path for path in (result_path, failure_path) if path.exists()]
            if len(paths) != 1:
                raise RuntimeError("container did not publish exactly one terminal receipt")
            terminals.append(load_envelope(paths[0])[0])
        except Exception as error:  # noqa: BLE001 - started attempts cannot be retried
            terminals.append(_publish_host_failure(run_root, query, common, error))
            break
    return {
        "schema_version": "t4_nodistill_topology_four_call_local_launch_result_v1",
        "run_id": run_id,
        "terminal_count": len(terminals),
        "automatic_retries": 0,
        "replacement_queries": 0,
        "backfill_queries": 0,
        "replicate_calls": 0,
        "terminals": terminals,
    }


def _copy_verified_assets() -> None:
    destination = Path("/opt/dock")
    receptors = destination / "receptors"
    receptors.mkdir(parents=True, exist_ok=True)
    qvina = destination / "qvina02"
    receptor = receptors / "parp1.pdbqt"
    shutil.copyfile("/inputs/qvina02", qvina)
    shutil.copyfile("/inputs/parp1.pdbqt", receptor)
    qvina.chmod(0o555)
    if file_sha256(qvina) != QVINA_SHA256 or file_sha256(receptor) != RECEPTOR_SHA256:
        raise ValueError("copied container evaluator assets changed")


def _copy_file_durable(source: Path, destination: Path) -> str:
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite pose artifact: {destination}")
    with source.open("rb") as input_handle, destination.open("xb") as output_handle:
        shutil.copyfileobj(input_handle, output_handle)
        output_handle.flush()
        os.fsync(output_handle.fileno())
    _fsync_directory(destination.parent)
    return file_sha256(destination)


def score_one_in_container(repository_root: Path, query_id: str) -> dict[str, Any]:
    from compose_v4.experiments.t4_docking_adapter import dock_t4

    root = repository_root.resolve()
    binding, binding_identity = load_envelope(root / LOCAL_EXECUTION_CONTRACT_RELATIVE_PATH)
    scientific, scientific_identity = load_envelope(root / SCORED_CONTRACT_RELATIVE_PATH)
    if scientific_identity != binding["scientific_scored_contract"]["payload_sha256"]:
        raise ValueError("container scientific contract changed")
    matches = [row for row in scientific["queries"] if row["query_id"] == query_id]
    if len(matches) != 1:
        raise ValueError("container query is outside the exact four-query lock")
    query = matches[0]
    receipt_root = Path("/receipts")
    lock, _ = load_envelope(receipt_root / "query_lock.json")
    reservation, _ = load_envelope(receipt_root / "queries" / query_id / "reservation.json")
    started, _ = load_envelope(receipt_root / "queries" / query_id / "started.json")
    if (
        lock.get("schema_version") != QUERY_LOCK_SCHEMA
        or lock.get("queries") != scientific["queries"]
        or reservation.get("query_id") != query_id
        or reservation.get("maximum_attempts") != 1
        or started.get("query_id") != query_id
        or started.get("local_execution_contract_payload_sha256") != binding_identity
    ):
        raise ValueError("container durable query lock changed")
    folder = receipt_root / "queries" / query_id
    result_path, failure_path = _terminal_paths(receipt_root, query_id)
    if result_path.exists() or failure_path.exists():
        raise RuntimeError("container query already has a terminal receipt")
    _copy_verified_assets()
    common = {
        key: started[key]
        for key in (
            "run_id",
            "scientific_contract_payload_sha256",
            "local_execution_contract_payload_sha256",
            "environment_attestation_payload_sha256",
            "launch_authorization_payload_sha256",
        )
    }
    start_clock = time.perf_counter()
    tag = f"t4topo4local_{started['run_id'][:12]}_{query_id}"
    box = {
        "receptor": "/opt/dock/receptors/parp1.pdbqt",
        "coordinates": scientific["evaluator"]["docking_box"],
    }
    try:
        score = dock_t4(
            query["canonical_smiles"],
            tag,
            scientific["evaluator"]["docking_seed"],
            box=box,
            cpu=1,
        )
        if score is None or not math.isfinite(float(score)):
            raise RuntimeError("bound docking adapter returned no finite score")
        pose_hashes = {}
        for name in ("l.mol", "l.pdbqt", "o.pdbqt"):
            source = Path("/tmp") / tag / name
            if source.is_file():
                destination = folder / name
                pose_hashes[name] = _copy_file_durable(source, destination)
        terminal = {
            "schema_version": RESULT_SCHEMA,
            **common,
            "query_id": query_id,
            "endpoint_key_sha256": query["endpoint_key_sha256"],
            "canonical_smiles": query["canonical_smiles"],
            "cell_key": query["cell_key"],
            "status": "complete",
            "score": float(score),
            "score_direction": "minimize",
            "score_unit": "kcal/mol",
            "pose_artifact_sha256": pose_hashes,
            "docking_seconds": time.perf_counter() - start_clock,
            "attempt_consumed": 1,
            "completed_at_utc": _stamp(),
        }
        publish_once_durable(result_path, terminal)
    except Exception as error:  # noqa: BLE001 - terminal no-retry receipt is required
        terminal = {
            "schema_version": FAILURE_SCHEMA,
            **common,
            "query_id": query_id,
            "endpoint_key_sha256": query["endpoint_key_sha256"],
            "canonical_smiles": query["canonical_smiles"],
            "cell_key": query["cell_key"],
            "status": "failed_no_retry",
            "error_type": type(error).__name__,
            "error_message": str(error),
            "docking_seconds": time.perf_counter() - start_clock,
            "attempt_consumed": 1,
            "retry_authorized": False,
            "failed_at_utc": _stamp(),
        }
        publish_once_durable(failure_path, terminal)
    return terminal


def reduce_run(repository_root: Path, run_id: str) -> dict[str, Any]:
    root = repository_root.resolve()
    binding, binding_identity, scientific = validate_local_execution_binding(root)
    run_root = root / LOCAL_RECEIPT_ROOT_RELATIVE_PATH / run_id
    lock, _ = load_envelope(run_root / "query_lock.json")
    if (
        lock.get("queries") != scientific["queries"]
        or lock.get("local_execution_contract_payload_sha256") != binding_identity
    ):
        raise ValueError("local reduction query lock changed")
    rows = []
    incomplete = []
    for query in scientific["queries"]:
        folder = run_root / "queries" / query["query_id"]
        load_envelope(folder / "reservation.json")
        started = folder / "started.json"
        result_path, failure_path = _terminal_paths(run_root, query["query_id"])
        terminals = [path for path in (result_path, failure_path) if path.exists()]
        if not started.exists():
            incomplete.append({"query_id": query["query_id"], "state": "reserved_not_started"})
            continue
        load_envelope(started)
        if len(terminals) != 1:
            incomplete.append({"query_id": query["query_id"], "state": "started_without_terminal"})
            continue
        terminal, _ = load_envelope(terminals[0])
        if (
            terminal.get("query_id") != query["query_id"]
            or terminal.get("endpoint_key_sha256") != query["endpoint_key_sha256"]
            or terminal.get("canonical_smiles") != query["canonical_smiles"]
            or terminal.get("attempt_consumed") != 1
        ):
            raise ValueError("local terminal receipt changed")
        rows.append(terminal)
    successes = [row for row in rows if row["status"] == "complete"]
    failures = [row for row in rows if row["status"] != "complete"]
    payload = {
        "schema_version": REDUCTION_SCHEMA,
        "status": "complete" if not incomplete else "terminal_incomplete_no_retry",
        "run_id": run_id,
        "local_execution_contract_payload_sha256": binding_identity,
        "source_binding": binding["source_binding"],
        "query_count": 4,
        "reserved_attempts": 4,
        "started_attempts": len(rows)
        + len([row for row in incomplete if row["state"] == "started_without_terminal"]),
        "successful_scores": len(successes),
        "failed_scores": len(failures),
        "incomplete": incomplete,
        "results": rows,
        "best_score": min((row["score"] for row in successes), default=None),
        "automatic_retries": 0,
        "replacement_queries": 0,
        "backfill_queries": 0,
        "replicate_calls": 0,
        "reduced_at_utc": _stamp(),
        "claim_boundary": scientific["claim_boundary"],
    }
    publish_once_durable(run_root / "reduction.json", payload)
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("validate")
    subparsers.add_parser("attest-environment")
    subparsers.add_parser("launch")
    reduce_parser = subparsers.add_parser("reduce")
    reduce_parser.add_argument("--run-id", required=True)
    subparsers.add_parser("container-preflight")
    score_parser = subparsers.add_parser("score-one")
    score_parser.add_argument("--query-id", required=True)
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[3]
    if args.command == "validate":
        binding, identity, _ = validate_local_execution_binding(root, require_local_assets=True)
        result: Any = {"payload_sha256": identity, "status": binding["status"]}
    elif args.command == "attest-environment":
        result = attest_environment(root)
    elif args.command == "launch":
        result = launch(root)
    elif args.command == "reduce":
        result = reduce_run(root, args.run_id)
    elif args.command == "container-preflight":
        result = _container_preflight_payload(root)
    elif args.command == "score-one":
        result = score_one_in_container(root, args.query_id)
    else:  # pragma: no cover
        raise AssertionError(args.command)
    json.dump(result, sys.stdout, sort_keys=True, separators=(",", ":"))
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "attest_environment",
    "launch",
    "main",
    "reduce_run",
    "score_one_in_container",
]
