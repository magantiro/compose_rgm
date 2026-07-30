"""Pack one frozen V2 MMP analogue pool: partition once -> parallel pack -> reduce.

    caller-frozen pool identity
      -> ONE streaming partition pass (scaffold-keyed, exact reconciliation)
      -> N queued packers (one per raw shard, at most five concurrent immutable publishers)
      -> reducer (coverage, checksums, overlap, replay audit)
      -> MMP_PACK_COMPLETE

Why: corruption and cycle_ops are packed, but MMP still reconstructed every trace at load
(``rewrite_trace_from_record`` + ``TraceProgressCTMC``), measured at 13.74 ms per row scanned -- 83 min per
partition, and the loader scanned the whole pool once per partition. That dominated everything else and
would have burned ~2.8 h of A100 time before the first optimizer step.

Sharding arithmetic is MEASURED, not assumed (see _PACK_RATE): 86.4 records/s single core, 11.57 ms/record,
306 gz bytes/record -> 1.17 h serial, ~0.11 GB packed. Shards are duration-targeted so validation and test
do not become a long tail behind train.

The pool stays authoritative; this store is a deterministic derivative bound to the pool's content hash,
record count, frozen completion contract, compiler support contract and launch authorization. The V2
entrypoint has no default pool path, count or output directory. Its output namespace is content-addressed
and is structurally unable to target ``mmp_packed_v1``.

Before launching, freeze a JSON contract with this exact semantic payload (additional provenance fields are
allowed and become part of its SHA-256 identity):

    {
      "schema": "compose.mmp_analogue_pool.v2",
      "MMP_POOL_COMPLETE": true,
      "pool_path": "/artifacts/<v2-mining-run>/edit_pool_full.jsonl",
      "pool_sha256": "<64 lowercase hex>",
      "pool_records": 123,
      "analogue_support_contract": "<frozen compiler support contract>"
    }

Then launch with every required identity field explicitly supplied. A mismatch fails before partitioning or
packing begins; this app never infers a new scientific identity from whatever happens to occupy a path.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

import modal

ROOT = Path(__file__).resolve().parent.parent
REMOTE_ROOT = Path("/root/compose")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.4.0", "numpy==1.26.4", "scipy==1.13.1", "networkx==3.3", "rdkit==2024.3.5"
    )
    .env(
        {
            "PYTHONPATH": os.pathsep.join((str(REMOTE_ROOT / "src"), str(REMOTE_ROOT / "scripts"))),
            "PYTHONUNBUFFERED": "1",
            "OMP_NUM_THREADS": "1",
        }
    )
    .add_local_dir(
        ROOT / "src", str(REMOTE_ROOT / "src"), copy=True, ignore=("**/__pycache__/**", "**/*.pyc")
    )
    .add_local_dir(
        ROOT / "scripts",
        str(REMOTE_ROOT / "scripts"),
        copy=True,
        ignore=("**/__pycache__/**", "**/*.pyc"),
    )
)

app = modal.App("compose-v4-pack-mmp-pool")
artifact_volume = modal.Volume.from_name("compose-v4-artifacts", create_if_missing=True)

PARTITIONS = ("train", "validation", "test")
LAYER = "mmp_analogue"
POOL_CONTRACT_SCHEMA = "compose.mmp_analogue_pool.v2"
PACK_REQUEST_SCHEMA = "compose.mmp_packing_request.v2"
PACK_OUTPUT_SCHEMA = "compose.mmp_packed_store.v2"
_PACK_REQUEST_FILENAME = "V2_PACK_REQUEST.json"
_V2_OUTPUT_PREFIX = "mmp_packed_v2_"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_COMMIT_RE = re.compile(r"^[0-9a-f]{7,40}$")
_GATE_RE = re.compile(r"^[0-9a-f]{16,64}$")
_RAW_SHARD_RE = re.compile(r"^shard_[0-9]{4}\.jsonl$")

# ---- measured sharding arithmetic (scripts-derived, persisted into the manifest) --------------------
_PACK_RATE = {
    "records_per_second_single_core": 86.4,
    "pack_ms_per_record": 11.57,
    "partition_ms_per_record": 0.36,
    "gz_bytes_per_record": 306,
    "measured_on": "600-record stride sample of the real pool, mean K 6.08 vs pool 5.895",
}
_TARGET_SHARD_SECONDS = 240  # ~4 min: well above container startup, so overhead stays <25%
# Modal Volume v1 supports at most five concurrent small commit writers.  The
# logical task count may be larger; Modal queues it behind this function-level
# cap.  This budget assumes no other app is writing the same Volume concurrently.
MODAL_VOLUME_V1_MAX_CONCURRENT_WRITERS = 5
_VERIFY_FRACTION = 0.02
_SENTINEL_ENTRIES = 8
# v1: partition by target scaffold only (LEAKED: 1,325 sources spanned partitions)
# v2: require BOTH endpoints to map to the same partition; drop straddling pairs, counted
_MMP_PARTITION_RULE_VERSION = 2


class PackRequestError(ValueError):
    """A V2 pack request is incomplete, mutable, ambiguous or points outside its safe namespace."""


def _write_immutable_bytes(path: Path, content: bytes) -> bool:
    """Publish through the shared Modal-v1-compatible no-overwrite primitive."""

    import sys

    sys.path.insert(0, str(REMOTE_ROOT / "src"))
    from compose_v4.data.mmp_pool_freezer import write_bytes_if_absent_modal_volume_v1

    return write_bytes_if_absent_modal_volume_v1(path, content)


def _replace_mutable_bytes(path: Path, content: bytes) -> None:
    """Atomically replace one non-authoritative mutable status file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_name = handle.name
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
        temporary_name = None
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)


def packing_concurrency_plan(logical_tasks: int) -> dict[str, Any]:
    """Describe the exact writer and total-container peak for one driver."""

    if (
        isinstance(logical_tasks, bool)
        or not isinstance(logical_tasks, int)
        or logical_tasks < 0
    ):
        raise PackRequestError("logical packing task count must be a nonnegative integer")
    active_writers = min(logical_tasks, MODAL_VOLUME_V1_MAX_CONCURRENT_WRITERS)
    scheduled_waves = (
        logical_tasks + MODAL_VOLUME_V1_MAX_CONCURRENT_WRITERS - 1
    ) // MODAL_VOLUME_V1_MAX_CONCURRENT_WRITERS
    return {
        "logical_tasks": logical_tasks,
        "configured_writer_container_limit": MODAL_VOLUME_V1_MAX_CONCURRENT_WRITERS,
        "max_active_writer_containers": active_writers,
        "queued_tasks_after_first_wave": max(0, logical_tasks - active_writers),
        "scheduled_waves": scheduled_waves,
        "estimated_map_critical_path_seconds": scheduled_waves * _TARGET_SHARD_SECONDS,
        "peak_total_containers_including_driver": 1 + active_writers,
        "requires_exclusive_volume_writer_budget": True,
    }


def validate_pack_task_plan(
    plan: object,
    *,
    expected_shards: object,
) -> list[dict[str, Any]]:
    """Reject aliased or unsafe worker destinations before submitting the map."""

    if not isinstance(plan, list):
        raise PackRequestError("packing task plan must be a list")
    if (
        isinstance(expected_shards, bool)
        or not isinstance(expected_shards, int)
        or expected_shards != len(plan)
    ):
        raise PackRequestError(
            f"packing task plan has {len(plan)} shards; expected {expected_shards!r}"
        )
    seen: set[tuple[str, str]] = set()
    checked: list[dict[str, Any]] = []
    for index, raw_entry in enumerate(plan):
        if not isinstance(raw_entry, Mapping):
            raise PackRequestError(f"packing task {index} must be a mapping")
        entry = dict(raw_entry)
        partition = entry.get("partition")
        shard = entry.get("shard")
        records = entry.get("records")
        row_ids_sha256 = entry.get("row_ids_sha256")
        raw_content_sha256 = entry.get("raw_content_sha256")
        if partition not in PARTITIONS:
            raise PackRequestError(
                f"packing task {index} has invalid partition {partition!r}"
            )
        if not isinstance(shard, str) or not _RAW_SHARD_RE.fullmatch(shard):
            raise PackRequestError(f"packing task {index} has unsafe shard name {shard!r}")
        if isinstance(records, bool) or not isinstance(records, int) or records <= 0:
            raise PackRequestError(
                f"packing task {index} must contain a positive record count"
            )
        if not isinstance(row_ids_sha256, str) or not re.fullmatch(
            r"[0-9a-f]{16}", row_ids_sha256
        ):
            raise PackRequestError(
                f"packing task {index} has invalid row-id digest"
            )
        if not isinstance(raw_content_sha256, str) or not _SHA256_RE.fullmatch(
            raw_content_sha256
        ):
            raise PackRequestError(
                f"packing task {index} has invalid raw-content digest"
            )
        destination = (partition, shard)
        if destination in seen:
            raise PackRequestError(
                f"packing task plan aliases worker destination {partition}/{shard}"
            )
        seen.add(destination)
        checked.append(entry)
    return checked


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _require_full_sha256(value: str, *, field: str) -> str:
    normalized = str(value or "").strip()
    if not _SHA256_RE.fullmatch(normalized):
        raise PackRequestError(f"{field} must be a full 64-character lowercase SHA-256")
    return normalized


def _require_artifact_file(value: str, *, field: str) -> str:
    """Return one normalized file below /artifacts; reject aliases and traversal."""
    raw = str(value or "").strip()
    candidate = PurePosixPath(raw)
    if not candidate.is_absolute() or not candidate.parts or candidate.parts[1] != "artifacts":
        raise PackRequestError(f"{field} must be an absolute file below /artifacts")
    if ".." in candidate.parts or raw.endswith("/"):
        raise PackRequestError(f"{field} must not contain traversal or name a directory")
    normalized = str(candidate)
    if normalized != raw or normalized == "/artifacts":
        raise PackRequestError(f"{field} must be a normalized file path below /artifacts")
    return normalized


def build_pack_request(
    *,
    pool_path: str,
    expected_pool_sha256: str,
    expected_pool_records: int,
    pool_contract_path: str,
    expected_pool_contract_sha256: str,
    expected_analogue_support_contract: str,
    launch_commit: str,
    gate_sha256: str,
) -> dict[str, Any]:
    """Construct the only accepted V2 launch request and its content-addressed output identity."""
    normalized_pool_path = _require_artifact_file(pool_path, field="pool_path")
    normalized_contract_path = _require_artifact_file(
        pool_contract_path, field="pool_contract_path"
    )
    if normalized_contract_path == normalized_pool_path:
        raise PackRequestError("pool_contract_path must be distinct from pool_path")
    if (
        isinstance(expected_pool_records, bool)
        or not isinstance(expected_pool_records, int)
        or expected_pool_records <= 0
    ):
        raise PackRequestError("expected_pool_records must be a positive integer")
    support_contract = str(expected_analogue_support_contract or "").strip()
    if not support_contract:
        raise PackRequestError("expected_analogue_support_contract must be explicit and non-empty")
    commit = str(launch_commit or "").strip()
    if not _COMMIT_RE.fullmatch(commit):
        raise PackRequestError("launch_commit must be a 7-40 character lowercase Git SHA")
    gate = str(gate_sha256 or "").strip()
    if not _GATE_RE.fullmatch(gate):
        raise PackRequestError("gate_sha256 must be a 16-64 character lowercase hexadecimal digest")

    frozen_input = {
        "schema": PACK_REQUEST_SCHEMA,
        "pool_path": normalized_pool_path,
        "expected_pool_sha256": _require_full_sha256(
            expected_pool_sha256, field="expected_pool_sha256"
        ),
        "expected_pool_records": expected_pool_records,
        "pool_contract_path": normalized_contract_path,
        "expected_pool_contract_sha256": _require_full_sha256(
            expected_pool_contract_sha256, field="expected_pool_contract_sha256"
        ),
        "expected_analogue_support_contract": support_contract,
    }
    input_identity = _canonical_sha256(frozen_input)
    authorization = {"launch_commit": commit, "gate_sha256": gate}
    output_identity = _canonical_sha256(
        {
            "schema": PACK_OUTPUT_SCHEMA,
            "input_identity": input_identity,
            "authorization": authorization,
        }
    )
    return {
        **frozen_input,
        "input_identity": input_identity,
        "authorization": authorization,
        "output_identity": output_identity,
        "output_subdir": f"{_V2_OUTPUT_PREFIX}{output_identity[:20]}",
    }


def validate_pack_request(request: Mapping[str, Any]) -> dict[str, Any]:
    """Reconstruct and compare a serialized request so no identity field can be hand-edited."""
    if not isinstance(request, Mapping):
        raise PackRequestError("pack request must be a mapping")
    expected_keys = {
        "schema",
        "pool_path",
        "expected_pool_sha256",
        "expected_pool_records",
        "pool_contract_path",
        "expected_pool_contract_sha256",
        "expected_analogue_support_contract",
        "input_identity",
        "authorization",
        "output_identity",
        "output_subdir",
    }
    missing = sorted(expected_keys - set(request))
    extra = sorted(set(request) - expected_keys)
    if missing or extra:
        raise PackRequestError(f"pack request fields disagree: missing={missing}, extra={extra}")
    authorization = request.get("authorization")
    if not isinstance(authorization, Mapping):
        raise PackRequestError("authorization must be a mapping")
    rebuilt = build_pack_request(
        pool_path=str(request.get("pool_path", "")),
        expected_pool_sha256=str(request.get("expected_pool_sha256", "")),
        expected_pool_records=request.get("expected_pool_records", 0),
        pool_contract_path=str(request.get("pool_contract_path", "")),
        expected_pool_contract_sha256=str(request.get("expected_pool_contract_sha256", "")),
        expected_analogue_support_contract=str(
            request.get("expected_analogue_support_contract", "")
        ),
        launch_commit=str(authorization.get("launch_commit", "")),
        gate_sha256=str(authorization.get("gate_sha256", "")),
    )
    if dict(request) != rebuilt:
        raise PackRequestError("serialized pack request does not match its computed identities")
    if rebuilt["output_subdir"] == "mmp_packed_v1" or not rebuilt["output_subdir"].startswith(
        _V2_OUTPUT_PREFIX
    ):
        raise PackRequestError("V2 packing may only write its computed mmp_packed_v2_* namespace")
    return rebuilt


def _mounted_artifact_path(path: str, artifact_root: Path) -> Path:
    normalized = _require_artifact_file(path, field="artifact path")
    return artifact_root / PurePosixPath(normalized).relative_to("/artifacts")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def preflight_frozen_input(
    request: Mapping[str, Any], *, artifact_root: Path = Path("/artifacts")
) -> dict[str, Any]:
    """Verify the immutable upstream contract, pool bytes and row count before any expensive work."""
    checked = validate_pack_request(request)
    contract_path = _mounted_artifact_path(checked["pool_contract_path"], artifact_root)
    pool_path = _mounted_artifact_path(checked["pool_path"], artifact_root)
    if not contract_path.is_file():
        raise PackRequestError(f"frozen pool contract is absent: {checked['pool_contract_path']}")
    if not pool_path.is_file():
        raise PackRequestError(f"frozen pool is absent: {checked['pool_path']}")

    contract_bytes = contract_path.read_bytes()
    observed_contract_sha = hashlib.sha256(contract_bytes).hexdigest()
    if observed_contract_sha != checked["expected_pool_contract_sha256"]:
        raise PackRequestError(
            "frozen pool contract SHA-256 mismatch: "
            f"expected {checked['expected_pool_contract_sha256']}, got {observed_contract_sha}"
        )
    try:
        pool_contract = json.loads(contract_bytes)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise PackRequestError(f"frozen pool contract is not valid JSON: {exc}") from exc
    if not isinstance(pool_contract, Mapping):
        raise PackRequestError("frozen pool contract must be a JSON object")
    required_contract = {
        "schema": POOL_CONTRACT_SCHEMA,
        "MMP_POOL_COMPLETE": True,
        "pool_path": checked["pool_path"],
        "pool_sha256": checked["expected_pool_sha256"],
        "pool_records": checked["expected_pool_records"],
        "analogue_support_contract": checked["expected_analogue_support_contract"],
    }

    def _matches_contract_field(key: str, expected: Any) -> bool:
        observed = pool_contract.get(key)
        if key == "MMP_POOL_COMPLETE":
            return observed is True
        if key == "pool_records":
            return (
                isinstance(observed, int)
                and not isinstance(observed, bool)
                and observed == expected
            )
        return observed == expected

    disagreements = {
        key: {"expected": value, "observed": pool_contract.get(key)}
        for key, value in required_contract.items()
        if not _matches_contract_field(key, value)
    }
    if disagreements:
        raise PackRequestError(
            f"frozen pool contract disagrees with the launch request: {disagreements}"
        )

    digest = hashlib.sha256()
    records = 0
    first_row_error: str | None = None
    with pool_path.open("rb") as handle:
        for line_number, line in enumerate(handle, start=1):
            digest.update(line)
            if not line.strip():
                continue
            records += 1
            if first_row_error is not None:
                continue
            try:
                row = json.loads(line)
            except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                first_row_error = f"line {line_number} is not valid JSON: {exc}"
                continue
            if not isinstance(row, Mapping):
                first_row_error = f"line {line_number} is not a JSON object"
                continue
            metadata = row.get("metadata")
            row_support = (
                metadata.get("support_contract") if isinstance(metadata, Mapping) else None
            )
            if row_support != checked["expected_analogue_support_contract"]:
                first_row_error = (
                    f"line {line_number} support contract is {row_support!r}; expected "
                    f"{checked['expected_analogue_support_contract']!r}"
                )
    observed_pool_sha = digest.hexdigest()
    if observed_pool_sha != checked["expected_pool_sha256"]:
        raise PackRequestError(
            "frozen pool SHA-256 mismatch: "
            f"expected {checked['expected_pool_sha256']}, got {observed_pool_sha}"
        )
    if records != checked["expected_pool_records"]:
        raise PackRequestError(
            f"frozen pool has {records} records; expected {checked['expected_pool_records']}"
        )
    if first_row_error is not None:
        raise PackRequestError(f"frozen pool row contract failed: {first_row_error}")
    return {
        "input_identity": checked["input_identity"],
        "pool_sha256": observed_pool_sha,
        "pool_records": records,
        "pool_contract_sha256": observed_contract_sha,
        "row_support_contract_verified": True,
    }


def validate_existing_output_claim(existing: Mapping[str, Any], request: Mapping[str, Any]) -> None:
    """Permit an interrupted V2 build to resume only under its byte-identical request."""
    checked = validate_pack_request(request)
    if dict(existing) != checked:
        raise PackRequestError(
            "output namespace is already claimed by a different pack request; choose nothing manually"
        )


def require_unchanged_completed_shards(
    prior_complete: Mapping[str, Any],
    observed_shard_artifacts: list[dict[str, Any]],
) -> None:
    """Refuse a completed-store reuse when any packed or manifest byte changed."""

    expected = prior_complete.get("shard_artifacts")
    if not isinstance(expected, list):
        raise PackRequestError("existing completion artifact lacks packed-shard byte identities")
    if expected != observed_shard_artifacts:
        raise PackRequestError(
            "completed packed-shard bytes or manifests changed after freeze; "
            "refusing reuse or re-blessing"
        )


def verify_local_launch_commit(launch_commit: str, *, repo: Path = ROOT) -> None:
    """Bind the caller-supplied commit to the actual clean code that Modal will serialize."""
    result = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    actual = result.stdout.strip().lower()
    if not actual.startswith(launch_commit):
        raise PackRequestError(
            f"launch_commit {launch_commit!r} does not identify local HEAD {actual!r}"
        )
    dirty = subprocess.run(
        ["git", "-C", str(repo), "status", "--porcelain"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    code_dirty = [
        line
        for line in dirty.splitlines()
        if line[3:].startswith(("src/", "scripts/", "modal_apps/"))
    ]
    if code_dirty:
        raise PackRequestError(
            "refusing to launch from a dirty serialized-code tree; commit the V2 packer first: "
            f"{code_dirty[:12]}"
        )


def pair_rule_implementation_hash() -> str:
    """Hash of ``pair_partition``'s SOURCE.

    A hand-incremented version only invalidates a cache when someone remembers to bump it. Hashing the
    function body means an edit to the rule invalidates every derived artifact whether or not the version
    was touched.
    """
    import hashlib
    import inspect

    return hashlib.sha256(inspect.getsource(pair_partition).encode()).hexdigest()[:16]


def shards_for(record_count: int) -> int:
    """Duration-targeted shard count. Small partitions get 1 shard, not a proportional sliver."""
    if record_count <= 0:
        return 0
    seconds = record_count * _PACK_RATE["pack_ms_per_record"] / 1000.0
    return max(1, round(seconds / _TARGET_SHARD_SECONDS) or 1)


def pair_partition(source_smiles: str, target_smiles: str) -> tuple[str | None, str, str]:
    """Partition for one MMP pair, or None when its endpoints disagree.

    Returns ``(partition_or_None, source_scaffold, target_scaffold)``. An MMP pair has TWO endpoints;
    partitioning on the target alone let one source molecule reach several partitions through
    differently-scaffolded targets (measured: 1,325 sources spanned partitions). Both endpoints must map
    to the same partition, which makes molecule-level leakage impossible by construction.
    """
    from compose_v4.data.scaffold_partition import murcko_scaffold, partition_for_scaffold

    source = (source_smiles or "").strip()
    target = (target_smiles or "").strip()
    if not source or not target:
        return None, "", ""
    source_scaffold = murcko_scaffold(source)
    target_scaffold = murcko_scaffold(target)
    if not source_scaffold or not target_scaffold:
        return None, source_scaffold or "", target_scaffold or ""
    source_partition = partition_for_scaffold(source_scaffold, salt="ringcore-v1")
    target_partition = partition_for_scaffold(target_scaffold, salt="ringcore-v1")
    if source_partition != target_partition:
        return None, source_scaffold, target_scaffold
    return source_partition, source_scaffold, target_scaffold


def _status(subdir: str, status: str, **fields) -> dict:
    """RUNNING | PARTIAL | FAILED | COMPLETE. Only the reducer may declare COMPLETE."""
    payload = {"status": status, **fields}
    out = Path("/artifacts") / subdir
    out.mkdir(parents=True, exist_ok=True)
    _replace_mutable_bytes(
        out / "mmp_pack_status.json",
        (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode(),
    )
    artifact_volume.commit()
    print(
        json.dumps(
            {
                "phase": "status",
                **{k: v for k, v in payload.items() if k not in ("shards", "missing", "stale")},
            },
            sort_keys=True,
        )[:900],
        flush=True,
    )
    return payload


def _contract() -> dict:
    import sys

    sys.path.insert(0, str(REMOTE_ROOT / "src"))
    sys.path.insert(0, str(REMOTE_ROOT / "scripts"))
    from compose_v4.data.packed_trace_store import (
        PACKED_STORE_SCHEMA_VERSION,
        sampler_contract,
    )
    from compose_v4.data.scaffold_partition import partitioner_provenance
    from compose_v4.experiments.analogue_prior import (
        ANALOGUE_SUPPORT_CONTRACT,
    )
    from compose_v4.rewrite.action_codec import codec_implementation_hash
    from compose_v4.rewrite.trace_shard import TRACE_SCHEMA_VERSION
    from ring_core_identity import recompute_operator_registry_hash

    return {
        "packed_store_schema_version": PACKED_STORE_SCHEMA_VERSION,
        "trace_schema_version": TRACE_SCHEMA_VERSION,
        "codec_implementation_hash": codec_implementation_hash(),
        "operator_registry_hash": recompute_operator_registry_hash(),
        "analogue_support_contract": ANALOGUE_SUPPORT_CONTRACT,
        "partitioner": partitioner_provenance(),
        **sampler_contract(),
    }


def _assert_runtime_contract(request: Mapping[str, Any]) -> dict[str, Any]:
    """The caller freezes compiler semantics; current source code may not silently replace them."""
    checked = validate_pack_request(request)
    contract = _contract()
    observed = contract.get("analogue_support_contract")
    expected = checked["expected_analogue_support_contract"]
    if observed != expected:
        raise PackRequestError(
            "runtime analogue support contract disagrees with the frozen input contract: "
            f"expected {expected!r}, runtime exposes {observed!r}"
        )
    return contract


def _claim_output_namespace(
    request: Mapping[str, Any], runtime_contract: Mapping[str, Any]
) -> dict[str, Any]:
    """Claim or resume the computed V2 namespace; never rewrite an already complete derivative."""
    checked = validate_pack_request(request)
    subdir = checked["output_subdir"]
    root = Path("/artifacts") / subdir
    claim_path = root / _PACK_REQUEST_FILENAME
    complete_path = root / "MMP_PACK_COMPLETE.json"

    if root.exists():
        contents = sorted(path.name for path in root.iterdir())
        if contents and not claim_path.is_file():
            raise PackRequestError(
                f"{subdir} already contains data but has no {_PACK_REQUEST_FILENAME}; refusing adoption"
            )
    if claim_path.is_file():
        try:
            existing_claim = json.loads(claim_path.read_text())
        except json.JSONDecodeError as exc:
            raise PackRequestError(f"{subdir} has an invalid pack-request claim") from exc
        validate_existing_output_claim(existing_claim, checked)
    else:
        root.mkdir(parents=True, exist_ok=True)
        _write_immutable_bytes(
            claim_path,
            (json.dumps(checked, indent=2, sort_keys=True) + "\n").encode(),
        )
        artifact_volume.commit()

    if complete_path.is_file():
        try:
            complete = json.loads(complete_path.read_text())
        except json.JSONDecodeError as exc:
            raise PackRequestError(f"{subdir} has an invalid completion artifact") from exc
        if (
            complete.get("MMP_PACK_COMPLETE") is not True
            or complete.get("input_identity") != checked["input_identity"]
            or complete.get("output_identity") != checked["output_identity"]
            or complete.get("pack_request") != checked
            or complete.get("contract") != dict(runtime_contract)
            or not isinstance(complete.get("shard_artifacts"), list)
        ):
            raise PackRequestError(
                f"{subdir} has a completion artifact that disagrees with its frozen request"
            )
        return {"reused_complete": True, "payload": complete}
    return {"reused_complete": False}


@app.function(
    image=image,
    cpu=8.0,
    memory=32768,
    timeout=4 * 3600,
    max_containers=1,
    volumes={"/artifacts": artifact_volume},
)
def partition_pool(request: dict[str, Any]) -> dict:
    """Re-verify, then stream the pool once to assign partitions and write raw intermediate shards.

    Letting every packing worker scan the full pool would multiply the scan by the worker
    count. Partition assignment happens here, once, and every record is accounted for -- reconciliation to
    the expected total is required, and any rejection is counted with a reason rather than dropped.
    """
    import sys
    import time

    sys.path.insert(0, str(REMOTE_ROOT / "src"))
    sys.path.insert(0, str(REMOTE_ROOT / "scripts"))
    artifact_volume.reload()
    checked = validate_pack_request(request)
    preflight_frozen_input(checked)
    contract = _assert_runtime_contract(checked)
    subdir = checked["output_subdir"]
    pool_path = checked["pool_path"]
    expected_pool_sha256 = checked["expected_pool_sha256"]
    expected_pool_records = checked["expected_pool_records"]
    started = time.time()
    out_root = Path("/artifacts") / subdir / "_raw"

    # Reuse a prior partitioning when the pool and the partitioner are unchanged. The scan is cheap
    # relative to packing but it is pure overhead on a rebuild, and skipping it keeps
    # raw shard membership byte-identical -- which is what lets packed shards be reused too.
    existing_manifest = Path("/artifacts") / subdir / "partition_manifest.json"
    if existing_manifest.exists():
        try:
            previous = json.loads(existing_manifest.read_text())
        except json.JSONDecodeError as exc:
            raise PackRequestError(
                "existing partition manifest is invalid; refusing adoption"
            ) from exc
        frozen_fields_match = (
            previous.get("pool_sha256") == expected_pool_sha256
            and previous.get("input_identity") == checked["input_identity"]
            and previous.get("output_identity") == checked["output_identity"]
            and previous.get("pack_request") == checked
            and previous.get("contract") == contract
            # The app's own pairing rule is not covered by the partitioner provenance hash.
            and previous.get("mmp_partition_rule_version") == _MMP_PARTITION_RULE_VERSION
            and previous.get("pair_rule_implementation_hash") == pair_rule_implementation_hash()
            and previous.get("scanned") == expected_pool_records
        )
        if not frozen_fields_match:
            raise PackRequestError(
                "existing partition manifest disagrees with the frozen request/runtime contract; "
                "refusing overwrite or re-blessing"
            )
        previous_entries = validate_pack_task_plan(
            previous.get("shards"),
            expected_shards=previous.get("expected_shards"),
        )
        raw_mismatches = []
        for entry in previous_entries:
            raw_path = out_root / entry["partition"] / entry["shard"]
            if not raw_path.is_file() or _file_sha256(raw_path) != entry.get("raw_content_sha256"):
                raw_mismatches.append(f"{entry['partition']}/{entry['shard']}")
        if not raw_mismatches:
            print(
                json.dumps(
                    {
                        "phase": "partition_reuse",
                        "pool_sha256": expected_pool_sha256[:16],
                        "shards": previous["expected_shards"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            return previous
        raise PackRequestError(
            "existing raw partition shards are missing or changed; refusing overwrite: "
            f"{raw_mismatches[:8]}"
        )

    digest = hashlib.sha256()
    rows_by_partition: dict[str, list] = {p: [] for p in PARTITIONS}
    rejects: dict[str, int] = {}
    dropped_row_ids: list[int] = []
    molecules_by_partition: dict[str, set] = {p: set() for p in PARTITIONS}
    dropped_bias: dict[str, dict] = {
        k: {}
        for k in (
            "path_length",
            "direction",
            "atom_count_delta",
            "cycle_rank_delta",
            "operator_signature",
        )
    }

    def _bump(counter: dict, key: str) -> None:
        counter[key] = counter.get(key, 0) + 1

    sources_by_partition: dict[str, set] = {p: set() for p in PARTITIONS}
    scaffolds_by_partition: dict[str, set] = {p: set() for p in PARTITIONS}
    scanned = 0

    with open(pool_path, "rb") as raw:
        for line in raw:
            digest.update(line)
            text = line.strip()
            if not text:
                continue
            scanned += 1
            record = json.loads(text)
            target = (record.get("target_smiles") or "").strip()
            if not target:
                rejects["empty_target_smiles"] = rejects.get("empty_target_smiles", 0) + 1
                continue
            partition, source_scaffold, scaffold = pair_partition(
                record.get("source_smiles", ""), target
            )
            if partition is None:
                reason = (
                    "unassignable_scaffold"
                    if not (source_scaffold and scaffold)
                    else "endpoints_straddle_partitions"
                )
                rejects[reason] = rejects.get(reason, 0) + 1
                dropped_row_ids.append(scanned - 1)
                if reason == "endpoints_straddle_partitions":
                    # Descriptive bias audit ONLY -- it must never feed back into the split rule. Its job
                    # is to show the holdout correction did not quietly delete one transformation class.
                    _bump(dropped_bias["path_length"], str(record.get("path_length")))
                    _bump(dropped_bias["direction"], str(record.get("direction") or "unknown"))
                    _bump(dropped_bias["atom_count_delta"], str(record.get("atom_count_delta")))
                    _bump(dropped_bias["cycle_rank_delta"], str(record.get("cycle_rank_delta")))
                    _bump(
                        dropped_bias["operator_signature"],
                        ",".join(sorted((record.get("operator_histogram") or {}))),
                    )
                continue
            record["_row_id"] = scanned - 1  # exact original row index
            record["_scaffold"] = scaffold
            record["_source_scaffold"] = source_scaffold
            rows_by_partition[partition].append(record)
            sources_by_partition[partition].add(record.get("source_smiles", ""))
            scaffolds_by_partition[partition].add(scaffold)
            # BOTH molecules and BOTH scaffolds belong to the partition -- tracking only the source would
            # miss exactly the leak this rule exists to prevent.
            molecules_by_partition[partition].add(record.get("source_smiles", ""))
            molecules_by_partition[partition].add(target)
            scaffolds_by_partition[partition].add(source_scaffold)

    pool_sha256 = digest.hexdigest()
    if pool_sha256 != expected_pool_sha256:
        return _status(
            subdir,
            "FAILED",
            reason=(
                f"pool SHA-256 changed during partitioning: expected {expected_pool_sha256}, "
                f"got {pool_sha256}"
            ),
            input_identity=checked["input_identity"],
            output_identity=checked["output_identity"],
        )
    if scanned != expected_pool_records:
        return _status(
            subdir,
            "FAILED",
            reason=f"pool has {scanned} records, expected {expected_pool_records}",
            pool_sha256=pool_sha256,
            input_identity=checked["input_identity"],
            output_identity=checked["output_identity"],
        )

    # deterministic shard assignment inside each partition, duration-targeted
    out_root.mkdir(parents=True, exist_ok=True)
    plan: list[dict] = []
    for partition in PARTITIONS:
        records = rows_by_partition[partition]
        n_shards = shards_for(len(records))
        for shard_index in range(n_shards):
            slice_rows = records[shard_index::n_shards]  # stride keeps path-length mix even
            name = f"shard_{shard_index:04d}.jsonl"
            directory = out_root / partition
            directory.mkdir(parents=True, exist_ok=True)
            raw_content = "".join(
                json.dumps(r, sort_keys=True) + "\n" for r in slice_rows
            ).encode()
            _write_immutable_bytes(directory / name, raw_content)
            plan.append(
                {
                    "partition": partition,
                    "shard": name,
                    "records": len(slice_rows),
                    "raw_content_sha256": hashlib.sha256(raw_content).hexdigest(),
                    "row_ids_sha256": hashlib.sha256(
                        ",".join(str(r["_row_id"]) for r in slice_rows).encode()
                    ).hexdigest()[:16],
                }
            )
    plan = validate_pack_task_plan(plan, expected_shards=len(plan))
    expected_raw_paths = {
        out_root / entry["partition"] / entry["shard"] for entry in plan
    }
    observed_raw_paths = {
        path
        for partition in PARTITIONS
        for path in (out_root / partition).glob("*.jsonl")
        if (out_root / partition).is_dir()
    }
    if observed_raw_paths != expected_raw_paths:
        unexpected = sorted(str(path) for path in observed_raw_paths - expected_raw_paths)
        missing = sorted(str(path) for path in expected_raw_paths - observed_raw_paths)
        raise PackRequestError(
            "raw partition inventory disagrees with the deterministic task plan: "
            f"missing={missing[:8]}, unexpected={unexpected[:8]}"
        )

    # Zero overlap must hold at partition time. Discovering it in the reducer means the whole packing
    # fan-out was already paid for -- which is exactly what happened on the v1 rule.
    molecule_overlap, scaffold_overlap = {}, {}
    for i, a in enumerate(PARTITIONS):
        for b in PARTITIONS[i + 1 :]:
            shared_mols = molecules_by_partition[a] & molecules_by_partition[b]
            shared_scaffolds = scaffolds_by_partition[a] & scaffolds_by_partition[b]
            if shared_mols:
                molecule_overlap[f"{a}|{b}"] = len(shared_mols)
            if shared_scaffolds:
                scaffold_overlap[f"{a}|{b}"] = len(shared_scaffolds)
    if molecule_overlap or scaffold_overlap:
        return _status(
            subdir,
            "FAILED",
            reason="partitions overlap after the pair rule",
            molecule_overlap=molecule_overlap,
            scaffold_overlap=scaffold_overlap,
        )

    kept = sum(len(v) for v in rows_by_partition.values())
    if kept + sum(rejects.values()) != scanned:
        return _status(
            subdir,
            "FAILED",
            reason="record reconciliation failed",
            scanned=scanned,
            kept=kept,
            rejects=rejects,
        )

    manifest = {
        "schema": PACK_OUTPUT_SCHEMA,
        "pack_request": checked,
        "input_identity": checked["input_identity"],
        "output_identity": checked["output_identity"],
        "pool_path": pool_path,
        "pool_sha256": pool_sha256,
        "pool_contract_path": checked["pool_contract_path"],
        "pool_contract_sha256": checked["expected_pool_contract_sha256"],
        "scanned": scanned,
        "kept": kept,
        "rejects": rejects,
        "accepted_records": kept,
        "endpoints_straddle_partitions": rejects.get("endpoints_straddle_partitions", 0),
        "dropped_row_ids_sha256": hashlib.sha256(
            ",".join(str(r) for r in sorted(dropped_row_ids)).encode()
        ).hexdigest()[:16],
        "dropped_row_ids_count": len(dropped_row_ids),
        "unique_molecules_by_partition": {p: len(molecules_by_partition[p]) for p in PARTITIONS},
        "molecule_overlap": molecule_overlap,
        "scaffold_overlap": scaffold_overlap,
        "dropped_bias": dropped_bias,
        "mmp_partition_rule_version": _MMP_PARTITION_RULE_VERSION,
        "pair_rule_implementation_hash": pair_rule_implementation_hash(),
        "expected_shards": len(plan),
        "shards": plan,
        "counts_by_partition": {p: len(rows_by_partition[p]) for p in PARTITIONS},
        "unique_sources_by_partition": {p: len(sources_by_partition[p]) for p in PARTITIONS},
        "unique_scaffolds_by_partition": {p: len(scaffolds_by_partition[p]) for p in PARTITIONS},
        "sharding_arithmetic": {
            **_PACK_RATE,
            "target_shard_seconds": _TARGET_SHARD_SECONDS,
            "shards": len(plan),
            "max_containers": MODAL_VOLUME_V1_MAX_CONCURRENT_WRITERS,
            "expected_critical_path_seconds": packing_concurrency_plan(len(plan))[
                "estimated_map_critical_path_seconds"
            ],
            "concurrency": packing_concurrency_plan(len(plan)),
        },
        "contract": contract,
        "seconds": round(time.time() - started, 1),
    }
    _write_immutable_bytes(
        Path("/artifacts") / subdir / "partition_manifest.json",
        (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode(),
    )
    artifact_volume.commit()
    print(
        json.dumps(
            {
                "phase": "partitioned",
                "scanned": scanned,
                "kept": kept,
                "shards": len(plan),
                "rejects": rejects,
                "counts": manifest["counts_by_partition"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return manifest


@app.function(
    image=image,
    cpu=4.0,
    memory=16384,
    timeout=4 * 3600,
    max_containers=MODAL_VOLUME_V1_MAX_CONCURRENT_WRITERS,
    volumes={"/artifacts": artifact_volume},
)
def pack_mmp_shard(
    request: dict[str, Any],
    partition: str,
    name: str,
    pool_sha256: str,
    row_ids_sha256: str,
    raw_content_sha256: str,
) -> dict:
    """Pack one raw shard. Immutable finalize; reuse only on a full contract + identity match."""
    import sys
    import time

    sys.path.insert(0, str(REMOTE_ROOT / "src"))
    sys.path.insert(0, str(REMOTE_ROOT / "scripts"))
    from compose_v4.data.packed_trace_store import (
        assert_closed_form_applies,
        build_packed_entry,
        manifest_path_for,
        write_packed_shard,
    )
    from compose_v4.experiments.analogue_prior import rewrite_trace_from_record
    from compose_v4.rewrite.progress import TraceProgressCTMC
    from compose_v4.rewrite.trace_shard import encode_trace_record

    assert_closed_form_applies()
    artifact_volume.reload()
    checked = validate_pack_request(request)
    if pool_sha256 != checked["expected_pool_sha256"]:
        raise PackRequestError("shard task pool identity disagrees with its frozen pack request")
    contract = _assert_runtime_contract(checked)
    subdir = checked["output_subdir"]
    provenance = {
        "layer": LAYER,
        "input_identity": checked["input_identity"],
        "output_identity": checked["output_identity"],
        "pool_contract_sha256": checked["expected_pool_contract_sha256"],
        "source_pool_sha256": pool_sha256,
        "row_ids_sha256": row_ids_sha256,
        "raw_content_sha256": raw_content_sha256,
        "partition": partition,
        **contract,
    }
    packed_name = name + ".gz"
    dest = Path("/artifacts") / subdir / partition / packed_name

    manifest_path = manifest_path_for(dest)
    if dest.exists() and manifest_path.exists():
        try:
            existing = json.loads(manifest_path.read_text())
        except json.JSONDecodeError:
            existing = {}
        stored = existing.get("provenance", {})
        if (
            stored.get("input_identity") == checked["input_identity"]
            and stored.get("output_identity") == checked["output_identity"]
            and stored.get("pool_contract_sha256") == checked["expected_pool_contract_sha256"]
            and stored.get("source_pool_sha256") == pool_sha256
            and stored.get("row_ids_sha256") == row_ids_sha256
            and stored.get("raw_content_sha256") == raw_content_sha256
            and all(stored.get(k) == v for k, v in contract.items())
        ):
            print(json.dumps({"phase": "reuse", "shard": f"{partition}/{packed_name}"}), flush=True)
            return {
                "partition": partition,
                "shard": packed_name,
                "reused": True,
                "entries": existing.get("entries", 0),
            }
        raise PackRequestError(
            f"existing packed shard {partition}/{packed_name} disagrees with the frozen request; "
            "refusing overwrite or re-blessing"
        )
    if manifest_path.exists() and not dest.exists():
        raise PackRequestError(
            f"existing packed shard {partition}/{packed_name} has a manifest but no data; "
            "refusing adoption"
        )

    started = time.time()
    raw = Path("/artifacts") / subdir / "_raw" / partition / name
    observed_raw_sha256 = _file_sha256(raw)
    if observed_raw_sha256 != raw_content_sha256:
        raise PackRequestError(
            f"raw shard {partition}/{name} SHA-256 mismatch: expected {raw_content_sha256}, "
            f"got {observed_raw_sha256}"
        )
    entries, unbuildable = [], 0
    for line in raw.read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        try:
            trace = rewrite_trace_from_record(record)
        except Exception:  # noqa: BLE001 -- counted, never silently dropped
            unbuildable += 1
            continue
        # Normalize the compiler pool row to the V2 trace schema. The packed loader decodes actions with
        # the V2 codec; normalizing here also gives the MMP
        # layer the same schema and provenance as corruption/cycle_ops -- ONE loader path, not two.
        v2 = encode_trace_record(
            trace,
            n_slots=int(record.get("n_slots", 40)),
            seed=0,
            trace_id=f"mmp-{record['_row_id']}",
            partition=partition,
            layer=LAYER,
            direction=str(record.get("direction", "")),
            source_scaffold=str(record.get("_scaffold", "")),
            extra={
                **dict(trace.metadata or {}),
                "row_id": int(record["_row_id"]),
                "scaffold": str(record.get("_scaffold", "")),
                "source_smiles": str(record.get("source_smiles", "")),
                "target_smiles": str(record.get("target_smiles", "")),
            },
        )
        entries.append(build_packed_entry(v2, TraceProgressCTMC(trace)))

    # Serialize on container-local storage, then publish both artifacts through
    # the shared no-overwrite primitive.  Data is published first; if a worker
    # dies before its manifest is created, an exact deterministic retry may
    # finish the pair.  A different byte is always rejected.
    with tempfile.TemporaryDirectory(prefix="compose-mmp-pack-") as temporary_directory:
        temporary_shard = Path(temporary_directory) / packed_name
        manifest = write_packed_shard(
            temporary_shard,
            entries,
            provenance=provenance,
            deterministic_gzip=True,
        )
        temporary_manifest = manifest_path_for(temporary_shard)
        _write_immutable_bytes(dest, temporary_shard.read_bytes())
        _write_immutable_bytes(manifest_path, temporary_manifest.read_bytes())
    artifact_volume.commit()
    result = {
        "partition": partition,
        "shard": packed_name,
        "reused": False,
        "entries": manifest["entries"],
        "states": manifest["states"],
        "unbuildable": unbuildable,
        "seconds": round(time.time() - started, 1),
    }
    print(json.dumps({"phase": "packed", **result}, sort_keys=True), flush=True)
    return result


@app.function(
    image=image,
    cpu=16.0,
    memory=65536,
    timeout=4 * 3600,
    max_containers=1,
    volumes={"/artifacts": artifact_volume},
)
def reduce_mmp(request: dict[str, Any]) -> dict:
    """Refuse MMP_PACK_COMPLETE unless coverage, reconciliation, overlap and replay all pass."""
    import sys

    sys.path.insert(0, str(REMOTE_ROOT / "src"))
    sys.path.insert(0, str(REMOTE_ROOT / "scripts"))
    from compose_v4.data.packed_trace_store import (
        manifest_path_for,
        read_packed_shard,
        sentinel_replay_check,
    )

    artifact_volume.reload()
    checked = validate_pack_request(request)
    subdir = checked["output_subdir"]
    root = Path("/artifacts") / subdir
    plan = json.loads((root / "partition_manifest.json").read_text())
    plan_entries = validate_pack_task_plan(
        plan.get("shards"),
        expected_shards=plan.get("expected_shards"),
    )
    contract = _assert_runtime_contract(checked)
    complete_path = root / "MMP_PACK_COMPLETE.json"
    prior_complete = None
    if complete_path.is_file():
        try:
            prior_complete = json.loads(complete_path.read_text())
        except json.JSONDecodeError as exc:
            raise PackRequestError(
                "existing MMP completion artifact is invalid; refusing adoption"
            ) from exc
        if (
            prior_complete.get("MMP_PACK_COMPLETE") is not True
            or prior_complete.get("pack_request") != checked
            or prior_complete.get("input_identity") != checked["input_identity"]
            or prior_complete.get("output_identity") != checked["output_identity"]
            or prior_complete.get("contract") != contract
            or not isinstance(prior_complete.get("shard_artifacts"), list)
        ):
            raise PackRequestError(
                "existing MMP completion artifact disagrees with the frozen request"
            )
    if (
        plan.get("pack_request") != checked
        or plan.get("input_identity") != checked["input_identity"]
        or plan.get("output_identity") != checked["output_identity"]
        or plan.get("pool_sha256") != checked["expected_pool_sha256"]
        or plan.get("pool_contract_sha256") != checked["expected_pool_contract_sha256"]
        or plan.get("scanned") != checked["expected_pool_records"]
    ):
        return _status(
            subdir,
            "FAILED",
            reason="partition manifest disagrees with the frozen V2 pack request",
            input_identity=checked["input_identity"],
            output_identity=checked["output_identity"],
        )

    missing, stale, totals = [], [], {"entries": 0, "states": 0}
    shard_artifacts = []
    by_partition: dict[str, int] = {}
    for entry in plan_entries:
        partition, name = entry["partition"], entry["shard"] + ".gz"
        shard = root / partition / name
        manifest_path = manifest_path_for(shard)
        if not shard.exists() or not manifest_path.exists():
            missing.append(f"{partition}/{name}")
            continue
        manifest = json.loads(manifest_path.read_text())
        stored = manifest.get("provenance", {})
        if (
            stored.get("input_identity") != checked["input_identity"]
            or stored.get("output_identity") != checked["output_identity"]
            or stored.get("pool_contract_sha256") != checked["expected_pool_contract_sha256"]
            or stored.get("source_pool_sha256") != plan["pool_sha256"]
            or stored.get("row_ids_sha256") != entry["row_ids_sha256"]
            or stored.get("raw_content_sha256") != entry["raw_content_sha256"]
            or any(stored.get(k) != v for k, v in contract.items())
        ):
            stale.append(f"{partition}/{name}")
            continue
        totals["entries"] += manifest["entries"]
        totals["states"] += manifest["states"]
        by_partition[partition] = by_partition.get(partition, 0) + manifest["entries"]
        shard_artifacts.append(
            {
                "partition": partition,
                "shard": name,
                "packed_sha256": _file_sha256(shard),
                "manifest_sha256": _file_sha256(manifest_path),
                "entries": int(manifest["entries"]),
                "states": int(manifest["states"]),
            }
        )

    # an unexpected packed shard means the derivative drifted from its plan
    expected_names = {(e["partition"], e["shard"] + ".gz") for e in plan_entries}
    unexpected = [
        f"{p}/{f.name}"
        for p in PARTITIONS
        for f in sorted((root / p).glob("*.jsonl.gz"))
        if (root / p).is_dir()
        if (p, f.name) not in expected_names
    ]
    if missing or stale or unexpected:
        return _status(
            subdir, "PARTIAL", missing=missing, stale=stale, unexpected=unexpected, totals=totals
        )
    shard_artifacts = sorted(
        shard_artifacts,
        key=lambda item: (item["partition"], item["shard"]),
    )
    if prior_complete is not None:
        try:
            require_unchanged_completed_shards(
                prior_complete,
                shard_artifacts,
            )
        except PackRequestError as exc:
            return _status(
                subdir,
                "FAILED",
                reason=str(exc),
                input_identity=checked["input_identity"],
                output_identity=checked["output_identity"],
            )

    # exact reconciliation + zero duplicate row ids + zero cross-partition source/scaffold leakage
    seen_rows: set[int] = set()
    duplicate_rows = 0
    sources: dict[str, str] = {}
    scaffolds: dict[str, str] = {}
    leak_source, leak_scaffold = 0, 0
    path_lengths: dict[int, int] = {}
    for entry in plan_entries:
        partition, name = entry["partition"], entry["shard"] + ".gz"
        for trace, packed in read_packed_shard(root / partition / name):
            meta = trace.metadata or {}
            row_id = meta.get("row_id")
            if row_id is not None:
                if row_id in seen_rows:
                    duplicate_rows += 1
                seen_rows.add(row_id)
            source = meta.get("source_smiles")
            if source is not None:
                if sources.setdefault(source, partition) != partition:
                    leak_source += 1
            scaffold = meta.get("scaffold")
            if scaffold is not None:
                if scaffolds.setdefault(scaffold, partition) != partition:
                    leak_scaffold += 1
            path_lengths[packed.path_length] = path_lengths.get(packed.path_length, 0) + 1

    problems = []
    if totals["entries"] != plan["kept"]:
        problems.append(f"packed {totals['entries']} entries but partitioning kept {plan['kept']}")
    if plan["kept"] + sum(plan["rejects"].values()) != plan["scanned"]:
        problems.append("accepted + rejects does not reconcile to the pool record count")
    if duplicate_rows:
        problems.append(f"{duplicate_rows} duplicate row ids")
    if leak_source:
        problems.append(f"{leak_source} sources span partitions")
    if leak_scaffold:
        problems.append(f"{leak_scaffold} scaffolds span partitions")
    if problems:
        return _status(subdir, "FAILED", reason="; ".join(problems), totals=totals)

    audited = 0
    try:
        for entry in plan_entries[: min(len(plan_entries), 8)]:
            for _ in read_packed_shard(
                root / entry["partition"] / (entry["shard"] + ".gz"),
                verify_fraction=_VERIFY_FRACTION,
            ):
                audited += 1
    except Exception as exc:  # noqa: BLE001
        return _status(subdir, "FAILED", reason=f"replay audit failed: {exc}", totals=totals)

    sentinels = []
    for partition in ("train", "validation"):
        shard = next((e for e in plan_entries if e["partition"] == partition), None)
        if shard:
            sentinels.append(
                sentinel_replay_check(
                    root / partition / (shard["shard"] + ".gz"), entries=_SENTINEL_ENTRIES
                )
            )

    payload = {
        "schema": PACK_OUTPUT_SCHEMA,
        "MMP_PACK_COMPLETE": True,
        "pack_request": checked,
        "input_identity": checked["input_identity"],
        "output_identity": checked["output_identity"],
        "layer": LAYER,
        "pool_sha256": plan["pool_sha256"],
        "pool_contract_path": checked["pool_contract_path"],
        "pool_contract_sha256": checked["expected_pool_contract_sha256"],
        "pool_records": plan["scanned"],
        "packed_entries": totals["entries"],
        "packed_states": totals["states"],
        "rejects": plan["rejects"],
        "accepted_records": plan["accepted_records"],
        "endpoints_straddle_partitions": plan["endpoints_straddle_partitions"],
        "dropped_row_ids_sha256": plan["dropped_row_ids_sha256"],
        "dropped_row_ids_count": plan["dropped_row_ids_count"],
        "unique_molecules_by_partition": plan["unique_molecules_by_partition"],
        "partition_time_molecule_overlap": plan["molecule_overlap"],
        "partition_time_scaffold_overlap": plan["scaffold_overlap"],
        "dropped_bias": plan["dropped_bias"],
        "mmp_partition_rule_version": plan["mmp_partition_rule_version"],
        "pair_rule_implementation_hash": plan["pair_rule_implementation_hash"],
        "entries_by_partition": by_partition,
        "expected_shards": plan["expected_shards"],
        "unique_row_ids": len(seen_rows),
        "duplicate_row_ids": duplicate_rows,
        "cross_partition_source_leaks": leak_source,
        "cross_partition_scaffold_leaks": leak_scaffold,
        "path_length_histogram": {str(k): v for k, v in sorted(path_lengths.items())},
        "verify_fraction": _VERIFY_FRACTION,
        "audited_entries": audited,
        "sentinels": sentinels,
        "contract": contract,
        "shard_artifacts": shard_artifacts,
        "sharding_arithmetic": plan["sharding_arithmetic"],
        "authorization": checked["authorization"],
    }
    _write_immutable_bytes(
        root / "MMP_PACK_COMPLETE.json",
        (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode(),
    )
    artifact_volume.commit()
    _status(
        subdir,
        "COMPLETE",
        packed_entries=totals["entries"],
        expected_shards=plan["expected_shards"],
    )
    return payload


@app.function(
    image=image,
    cpu=2.0,
    timeout=12 * 3600,
    max_containers=1,
    volumes={"/artifacts": artifact_volume},
)
def driver(request: dict[str, Any]) -> dict:
    """Remote orchestration so --detach survives a client disconnect."""
    artifact_volume.reload()
    checked = validate_pack_request(request)
    from compose_v4.data.mmp_pool_freezer import run_write_once_storage_preflight

    storage = run_write_once_storage_preflight("/artifacts")
    artifact_volume.commit()
    if storage.get("status") != "PASS":
        raise PackRequestError(
            f"artifact Volume immutable-publication preflight failed: {storage}"
        )
    runtime_contract = _assert_runtime_contract(checked)
    # All bytes/counts/contracts are checked before claiming an output path or starting the fan-out.
    preflight = preflight_frozen_input(checked)
    claim = _claim_output_namespace(checked, runtime_contract)
    if claim["reused_complete"]:
        print(
            json.dumps(
                {
                    "phase": "complete_reuse_revalidation",
                    "output_subdir": checked["output_subdir"],
                    "output_identity": checked["output_identity"],
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return reduce_mmp.remote(checked)

    subdir = checked["output_subdir"]
    _status(
        subdir,
        "RUNNING",
        pack_request=checked,
        input_identity=checked["input_identity"],
        output_identity=checked["output_identity"],
        preflight=preflight,
        authorization=checked["authorization"],
    )

    plan = partition_pool.remote(checked)
    if plan.get("status") == "FAILED":
        return plan
    plan_entries = validate_pack_task_plan(
        plan.get("shards"),
        expected_shards=plan.get("expected_shards"),
    )
    tasks = [
        (
            checked,
            e["partition"],
            e["shard"],
            plan["pool_sha256"],
            e["row_ids_sha256"],
            e["raw_content_sha256"],
        )
        for e in plan_entries
    ]
    concurrency = packing_concurrency_plan(len(tasks))
    print(
        json.dumps(
            {
                "phase": "map_start",
                "tasks": len(tasks),
                "max_containers": MODAL_VOLUME_V1_MAX_CONCURRENT_WRITERS,
                "concurrency": concurrency,
            }
        ),
        flush=True,
    )
    results = list(pack_mmp_shard.starmap(tasks))
    print(
        json.dumps(
            {
                "phase": "map_done",
                "shards": len(results),
                "reused": sum(1 for r in results if r.get("reused")),
            }
        ),
        flush=True,
    )
    return reduce_mmp.remote(checked)


@app.local_entrypoint()
def main(
    pool_path: str,
    expected_pool_sha256: str,
    expected_pool_records: int,
    pool_contract_path: str,
    expected_pool_contract_sha256: str,
    expected_analogue_support_contract: str,
    commit: str,
    gate_sha: str,
):
    request = build_pack_request(
        pool_path=pool_path,
        expected_pool_sha256=expected_pool_sha256,
        expected_pool_records=expected_pool_records,
        pool_contract_path=pool_contract_path,
        expected_pool_contract_sha256=expected_pool_contract_sha256,
        expected_analogue_support_contract=expected_analogue_support_contract,
        launch_commit=commit,
        gate_sha256=gate_sha,
    )
    verify_local_launch_commit(request["authorization"]["launch_commit"])
    call = driver.spawn(request)
    print(
        json.dumps(
            {
                "phase": "launched",
                "driver_call_id": call.object_id,
                "input_identity": request["input_identity"],
                "output_identity": request["output_identity"],
                "output_subdir": request["output_subdir"],
                "authorization": request["authorization"],
                "poll": (
                    "modal volume get compose-v4-artifacts "
                    f"{request['output_subdir']}/mmp_pack_status.json -"
                ),
            },
            indent=2,
            sort_keys=True,
        )
    )
