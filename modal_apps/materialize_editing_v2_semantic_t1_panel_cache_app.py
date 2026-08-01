"""Materialize the exact train-only semantic T1 panel/cache inputs on Modal.

This CPU-only surface consumes the exact semantic migration, reusable chunk
cache, Active8 decision, and structural Gate 0 artifacts.  It freezes a
prospective bounded panel request before reading accepted train transitions,
delegates all scientific interpretation to the production decision index and
semantic T1 panel module, and publishes immutable content-addressed bytes.

The output prepares the unique-state capacity panel and its exact complete-trace
cache-input union.  It does not build the separate repeated-state empirical-law
panel and does not compile the successor-fiber cache.  Those are distinct,
provenance-bound stages.  This surface grants no T1, P50, training,
checkpoint-selection, or final-test authority and contains no optimizer,
threshold, hazard, path-metric, or P50 policy.
"""

from __future__ import annotations

import hashlib
import json
import platform
import re
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Any

import modal

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from compose_v4.experiments.editing_v2_semantic_t1_artifact_contracts import (  # noqa: E402
    NO_DOWNSTREAM_AUTHORITY,
    PANEL_ARTIFACT_FILENAME,
    PANEL_COMPLETION_FILENAME,
    PANEL_COMPLETION_SCHEMA,
    PANEL_COMPLETION_SCHEMA_VERSION,
    PANEL_COMPLETION_STATUS,
    PANEL_RUN_REQUEST_FILENAME,
    PANEL_RUN_REQUEST_SCHEMA,
    PANEL_RUN_REQUEST_SCHEMA_VERSION,
    PANEL_RUN_REQUEST_STATUS,
)

REMOTE_ROOT = Path("/root/compose")
ARTIFACT_ROOT = Path("/artifacts")
LAUNCHER_SOURCE = "modal_apps/materialize_editing_v2_semantic_t1_panel_cache_app.py"
PANEL_POLICY_SOURCE = "configs/editing_v2_semantic_t1_panel_policy_v1.json"
IMAGE_SOURCE_DIRECTORIES = ("src", "configs", "modal_apps")
OUTPUT_PREFIX = "/artifacts/editing_v2/semantic_t1_panel_cache_inputs"

RUN_REQUEST_FILENAME = PANEL_RUN_REQUEST_FILENAME
PANEL_FILENAME = PANEL_ARTIFACT_FILENAME
COMPLETION_FILENAME = PANEL_COMPLETION_FILENAME

SOURCE_REVISION_SCHEMA = "compose.editing.semantic_t1_modal_source_revision"
SOURCE_REVISION_SCHEMA_VERSION = 1
RUN_REQUEST_SCHEMA = PANEL_RUN_REQUEST_SCHEMA
RUN_REQUEST_SCHEMA_VERSION = PANEL_RUN_REQUEST_SCHEMA_VERSION
RUN_REQUEST_STATUS = PANEL_RUN_REQUEST_STATUS
COMPLETION_SCHEMA = PANEL_COMPLETION_SCHEMA
COMPLETION_SCHEMA_VERSION = PANEL_COMPLETION_SCHEMA_VERSION
COMPLETION_STATUS = PANEL_COMPLETION_STATUS
PANEL_POLICY_SCHEMA = "compose.editing_v2.semantic_t1_panel_policy"
PANEL_POLICY_SCHEMA_VERSION = 1
PANEL_POLICY_STATUS = "FROZEN_PROSPECTIVE_PANEL_NO_DOWNSTREAM_AUTHORITY"
PANEL_POLICY_PANEL_KIND = "unique_state_single_target_canonical_successor_capacity"
PANEL_POLICY_OBJECTIVE_UNIT = "exact_source_frozen_time_canonical_successor"
PANEL_POLICY_CACHE_HANDOFF = (
    "complete_train_trace_union_for_cpu_successor_fiber_cache_v1"
)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_NO_AUTHORITY = NO_DOWNSTREAM_AUTHORITY
_INPUT_NAMES = (
    "migration_completion",
    "chunk_cache_plan",
    "chunk_cache_global_completion",
    "decision_plan",
    "decision_completion",
    "gate_zero_evidence",
    "gate_zero_decision",
    "gate_zero_completion",
)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.4.0",
        "numpy==1.26.4",
        "scipy==1.13.1",
        "networkx==3.3",
        "rdkit==2024.3.5",
    )
    .env(
        {
            "PYTHONPATH": str(REMOTE_ROOT / "src"),
            "PYTHONUNBUFFERED": "1",
            "OMP_NUM_THREADS": "8",
        }
    )
)
for source_directory in IMAGE_SOURCE_DIRECTORIES:
    image = image.add_local_dir(
        ROOT / source_directory,
        str(REMOTE_ROOT / source_directory),
        copy=True,
        ignore=("**/__pycache__/**", "**/*.pyc"),
    )

app = modal.App("compose-v4-editing-v2-semantic-t1-panel-cache")
artifact_volume = modal.Volume.from_name(
    "compose-v4-artifacts", create_if_missing=False
)


def _canonical_bytes(value: object, *, pretty: bool = False) -> bytes:
    if pretty:
        return (
            json.dumps(
                value,
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
            ).encode("utf-8")
            + b"\n"
        )
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _sha256(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_sha256(value: object, *, field: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{field} must be a lowercase SHA-256")
    return value


def _require_commit(value: object, *, field: str) -> str:
    if not isinstance(value, str) or _COMMIT_RE.fullmatch(value) is None:
        raise ValueError(f"{field} must be a full lowercase Git commit")
    return value


def _artifact_path(value: str, *, artifact_root: Path, field: str) -> Path:
    pure = PurePosixPath(value)
    if (
        not value
        or not pure.is_absolute()
        or len(pure.parts) < 3
        or pure.parts[1] != "artifacts"
        or ".." in pure.parts
        or "\\" in value
        or str(pure) != value
        or value.endswith("/")
    ):
        raise ValueError(f"{field} must be a normalized path below /artifacts")
    root = Path(artifact_root).resolve()
    resolved = (root / Path(*pure.parts[2:])).resolve()
    if not resolved.is_relative_to(root):
        raise ValueError(f"{field} resolves outside artifact_root")
    return resolved


def _artifact_address(path: Path, *, artifact_root: Path) -> str:
    relative = Path(path).resolve().relative_to(Path(artifact_root).resolve())
    return str(PurePosixPath("/artifacts") / PurePosixPath(relative.as_posix()))


def _git(root: Path, *arguments: str) -> str:
    try:
        return subprocess.run(
            ("git", *arguments),
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError(
            f"cannot establish semantic T1 Git identity: git {' '.join(arguments)}"
        ) from error


def _serialized_source_paths(root: Path) -> tuple[str, ...]:
    paths = [LAUNCHER_SOURCE, PANEL_POLICY_SOURCE]
    paths.extend(
        path.relative_to(root).as_posix()
        for path in sorted((root / "src" / "compose_v4").rglob("*.py"))
        if path.is_file()
    )
    paths.extend(
        path.relative_to(root).as_posix()
        for path in sorted((root / "configs").rglob("*.json"))
        if path.is_file()
    )
    return tuple(paths)


def _serialized_source_hashes(root: Path) -> dict[str, str]:
    return {
        relative: _file_sha256(root / relative)
        for relative in _serialized_source_paths(root)
    }


def local_source_revision(
    *, expected_commit: str, repo_root: Path = ROOT
) -> dict[str, Any]:
    """Bind the exact clean tree and all code/config bytes in the image."""

    _require_commit(expected_commit, field="expected_commit")
    root = Path(repo_root)
    commit = _git(root, "rev-parse", "HEAD")
    tree = _git(root, "rev-parse", "HEAD^{tree}")
    dirty = _git(root, "status", "--porcelain=v1", "--untracked-files=all")
    if commit != expected_commit or dirty:
        raise RuntimeError("semantic T1 requires the exact clean committed worktree")
    hashes = _serialized_source_hashes(root)
    body: dict[str, Any] = {
        "schema": SOURCE_REVISION_SCHEMA,
        "schema_version": SOURCE_REVISION_SCHEMA_VERSION,
        "commit": commit,
        "tree": tree,
        "worktree_clean": True,
        "serialized_source_hashes": hashes,
        "serialized_source_hashes_sha256": _sha256(hashes),
    }
    return {**body, "source_revision_sha256": _sha256(body)}


def _validate_source_revision(value: object, *, remote_root: Path) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError("semantic T1 source revision must be an object")
    revision = dict(value)
    expected_fields = {
        "schema",
        "schema_version",
        "commit",
        "tree",
        "worktree_clean",
        "serialized_source_hashes",
        "serialized_source_hashes_sha256",
        "source_revision_sha256",
    }
    body = {
        key: item for key, item in revision.items() if key != "source_revision_sha256"
    }
    hashes = _serialized_source_hashes(remote_root)
    if (
        set(revision) != expected_fields
        or revision.get("schema") != SOURCE_REVISION_SCHEMA
        or revision.get("schema_version") != SOURCE_REVISION_SCHEMA_VERSION
        or revision.get("worktree_clean") is not True
        or revision.get("serialized_source_hashes") != hashes
        or revision.get("serialized_source_hashes_sha256") != _sha256(hashes)
        or revision.get("source_revision_sha256") != _sha256(body)
    ):
        raise RuntimeError("semantic T1 serialized source revision disagrees")
    _require_commit(revision.get("commit"), field="source_revision.commit")
    _require_commit(revision.get("tree"), field="source_revision.tree")
    return revision


def _validate_panel_request_payload(
    value: object,
    *,
    request_type: Any,
) -> Any:
    if not isinstance(value, Mapping):
        raise TypeError("semantic T1 panel request must be an object")
    payload = dict(value)
    limits = payload.get("maximum_entries_by_family")
    if not isinstance(limits, Mapping):
        raise TypeError("maximum_entries_by_family must be an object")
    try:
        request = request_type(
            request_id=payload.get("request_id"),
            source_revision_sha256=payload.get("source_revision_sha256"),
            support_time_hex=payload.get("support_time_hex"),
            maximum_entries_by_family=tuple(sorted(limits.items())),
            request_sha256=payload.get("request_sha256", ""),
        )
    except (TypeError, ValueError) as error:
        raise ValueError("semantic T1 panel request is invalid") from error
    if request.as_payload() != payload:
        raise ValueError("semantic T1 panel request fields or semantics disagree")
    return request


def _load_panel_policy(*, root: Path) -> dict[str, Any]:
    """Load the committed, self-hashed prospective panel policy."""

    path = Path(root) / PANEL_POLICY_SOURCE
    try:
        payload = json.loads(path.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"cannot load semantic T1 panel policy: {path}") from error
    if not isinstance(payload, dict):
        raise TypeError("semantic T1 panel policy must be an object")
    expected_fields = {
        "schema",
        "schema_version",
        "status",
        *_NO_AUTHORITY,
        "request_id",
        "support_time_hex",
        "maximum_entries_by_family",
        "panel_kind",
        "objective_unit",
        "cache_handoff",
        "repeated_state_panel_included",
        "empirical_multiplicity_receipts_included",
        "successor_fiber_cache_compiled",
        "hazard_included",
        "optimizer_policy_included",
        "gate_thresholds_included",
        "p50_policy_included",
        "policy_sha256",
    }
    body = dict(payload)
    supplied_sha256 = body.pop("policy_sha256", None)
    if (
        set(payload) != expected_fields
        or payload.get("schema") != PANEL_POLICY_SCHEMA
        or payload.get("schema_version") != PANEL_POLICY_SCHEMA_VERSION
        or payload.get("status") != PANEL_POLICY_STATUS
        or any(
            payload.get(field) is not expected
            for field, expected in _NO_AUTHORITY.items()
        )
        or payload.get("panel_kind") != PANEL_POLICY_PANEL_KIND
        or payload.get("objective_unit") != PANEL_POLICY_OBJECTIVE_UNIT
        or payload.get("cache_handoff") != PANEL_POLICY_CACHE_HANDOFF
        or payload.get("repeated_state_panel_included") is not False
        or payload.get("empirical_multiplicity_receipts_included") is not False
        or payload.get("successor_fiber_cache_compiled") is not False
        or payload.get("hazard_included") is not False
        or payload.get("optimizer_policy_included") is not False
        or payload.get("gate_thresholds_included") is not False
        or payload.get("p50_policy_included") is not False
        or supplied_sha256 != _sha256(body)
    ):
        raise RuntimeError("semantic T1 panel policy identity disagrees")
    return payload


def build_run_request(
    *,
    source_revision: Mapping[str, Any],
    input_records: Mapping[str, Mapping[str, str]],
    panel_request: Mapping[str, object],
    panel_policy_file_sha256: str,
    panel_policy_sha256: str,
    panel_implementation_sha256: str,
    output_prefix: str = OUTPUT_PREFIX,
) -> dict[str, Any]:
    """Content-address one exact, nonauthorizing T1 preparation request."""

    if set(input_records) != set(_INPUT_NAMES):
        raise ValueError("semantic T1 input inventory disagrees")
    normalized_inputs: dict[str, dict[str, str]] = {}
    for name in _INPUT_NAMES:
        record = input_records[name]
        if not isinstance(record, Mapping) or set(record) != {
            "artifact_path",
            "file_sha256",
        }:
            raise ValueError(f"semantic T1 input {name} fields disagree")
        artifact_path = str(record["artifact_path"])
        _artifact_path(artifact_path, artifact_root=ARTIFACT_ROOT, field=name)
        normalized_inputs[name] = {
            "artifact_path": artifact_path,
            "file_sha256": _require_sha256(
                record["file_sha256"], field=f"{name}.file_sha256"
            ),
        }
    request_payload = dict(panel_request)
    if any(
        request_payload.get(field) is not expected
        for field, expected in _NO_AUTHORITY.items()
    ):
        raise ValueError("semantic T1 panel request grants forbidden authority")
    _artifact_path(
        f"{output_prefix}/placeholder",
        artifact_root=ARTIFACT_ROOT,
        field="output_prefix",
    )
    body: dict[str, Any] = {
        "schema": RUN_REQUEST_SCHEMA,
        "schema_version": RUN_REQUEST_SCHEMA_VERSION,
        "status": RUN_REQUEST_STATUS,
        **_NO_AUTHORITY,
        "source_revision": dict(source_revision),
        "python_runtime": {
            "implementation": platform.python_implementation(),
            "version": platform.python_version(),
        },
        "inputs": normalized_inputs,
        "panel_request": request_payload,
        "panel_policy": {
            "source": PANEL_POLICY_SOURCE,
            "file_sha256": _require_sha256(
                panel_policy_file_sha256, field="panel_policy_file_sha256"
            ),
            "policy_sha256": _require_sha256(
                panel_policy_sha256, field="panel_policy_sha256"
            ),
        },
        "panel_implementation_sha256": _require_sha256(
            panel_implementation_sha256, field="panel_implementation_sha256"
        ),
        "output_prefix": output_prefix,
        "training_launched": False,
        "successor_cache_compiled": False,
    }
    return {**body, "run_identity_sha256": _sha256(body)}


def _imports(remote_root: Path = REMOTE_ROOT) -> dict[str, Any]:
    import sys

    source_root = str(Path(remote_root) / "src")
    if source_root not in sys.path:
        sys.path.insert(0, source_root)
    from compose_v4.data.editing_v2_semantic_active8_decision_source import (
        resolve_editing_v2_semantic_active8_decision_source,
    )
    from compose_v4.data.immutable_artifact import write_bytes_if_absent
    from compose_v4.experiments import editing_v2_semantic_gate_zero as gate_zero
    from compose_v4.experiments import editing_v2_semantic_t1_panel_cache as t1_panel

    return {
        "resolve_decision_source": (
            resolve_editing_v2_semantic_active8_decision_source
        ),
        "write_bytes_if_absent": write_bytes_if_absent,
        "gate_zero": gate_zero,
        "t1_panel": t1_panel,
    }


def _load_canonical_object(path: Path, *, field: str) -> dict[str, Any]:
    try:
        raw = Path(path).read_bytes()
        payload = json.loads(raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"cannot load {field}: {path}") from error
    if not isinstance(payload, dict) or raw != _canonical_bytes(payload) + b"\n":
        raise RuntimeError(f"{field} is not canonical JSON")
    return payload


def _require_gate_zero_siblings(
    *, evidence: Path, decision: Path, completion: Path, gate_zero: Any
) -> Path:
    expected = {
        evidence: gate_zero.EVIDENCE_FILENAME,
        decision: gate_zero.DECISION_FILENAME,
        completion: gate_zero.COMPLETION_FILENAME,
    }
    parents = {path.parent.resolve() for path in expected}
    if len(parents) != 1 or any(path.name != name for path, name in expected.items()):
        raise RuntimeError(
            "semantic T1 Gate-0 inputs must be the exact three sibling artifacts"
        )
    return next(iter(parents))


def _completion_body(
    *,
    run_request: Mapping[str, Any],
    request_path: Path,
    panel_path: Path,
    panel: Any,
    panel_file_sha256: str,
    panel_file_bytes: int,
    resolved_cache_trace_count: int,
    artifact_root: Path,
) -> dict[str, Any]:
    return {
        "schema": COMPLETION_SCHEMA,
        "schema_version": COMPLETION_SCHEMA_VERSION,
        "status": COMPLETION_STATUS,
        **_NO_AUTHORITY,
        "run_identity_sha256": run_request["run_identity_sha256"],
        "source_revision_sha256": run_request["source_revision"][
            "source_revision_sha256"
        ],
        "input_inventory_sha256": _sha256(run_request["inputs"]),
        "panel_policy": run_request["panel_policy"],
        "request_artifact_path": _artifact_address(
            request_path, artifact_root=artifact_root
        ),
        "request_file_sha256": _file_sha256(request_path),
        "panel_artifact_path": _artifact_address(
            panel_path, artifact_root=artifact_root
        ),
        "panel_artifact_sha256": panel.artifact_sha256,
        "panel_file_sha256": panel_file_sha256,
        "panel_file_bytes": panel_file_bytes,
        "panel_implementation_sha256": panel.panel_implementation_sha256,
        "decision_source_inventory_sha256": (
            panel.gate_zero_binding.decision_source_inventory_sha256
        ),
        "gate_zero_completion_sha256": panel.gate_zero_binding.completion_sha256,
        "counts": panel.identity_body()["counts"],
        "cache_trace_inputs_reopened": True,
        "resolved_cache_trace_count": resolved_cache_trace_count,
        "unique_state_panel_prepared": True,
        "repeated_state_panel_prepared": False,
        "empirical_multiplicity_receipts_consumed": False,
        "training_launched": False,
        "successor_cache_compiled": False,
        "next_stage_authorized": None,
    }


def _validate_completion(
    value: object,
    *,
    expected_body: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError("semantic T1 completion must be an object")
    completion = dict(value)
    body = dict(completion)
    supplied_sha256 = body.pop("completion_sha256", None)
    if body != dict(expected_body) or supplied_sha256 != _sha256(body):
        raise RuntimeError("semantic T1 completion identity disagrees")
    return completion


def _validated_panel(
    *,
    panel_path: Path,
    request: Any,
    index: Any,
    gate_zero_artifacts: Mapping[str, Any],
    decision_plan_path: Path,
    t1_panel: Any,
    gate_zero_contract: Any,
    remote_root: Path,
) -> Any:
    content = panel_path.read_bytes()
    panel = t1_panel.deserialize_semantic_t1_panel(
        content,
        expected_panel_implementation_sha256=(
            t1_panel.semantic_t1_panel_implementation_sha256(repo_root=remote_root)
        ),
    )
    binding = t1_panel.bind_verified_semantic_gate_zero_pass(
        index,
        gate_zero_artifacts,
        decision_plan_path=decision_plan_path,
        contract=gate_zero_contract,
        repo_root=remote_root,
    )
    if (
        panel.request != request
        or panel.gate_zero_binding != binding
        or dict(panel.decision_source_identity) != index.identity_payload()
    ):
        raise RuntimeError("persisted semantic T1 panel differs from exact request")
    return panel


def _driver_impl(
    *,
    source_revision: Mapping[str, Any],
    migration_completion: str,
    chunk_cache_plan: str,
    chunk_cache_global_completion: str,
    decision_plan: str,
    decision_completion: str,
    gate_zero_evidence: str,
    gate_zero_decision: str,
    gate_zero_completion: str,
    output_prefix: str,
    artifact_root: Path = ARTIFACT_ROOT,
    remote_root: Path = REMOTE_ROOT,
    stage_commit: Any | None = None,
) -> dict[str, Any]:
    revision = _validate_source_revision(source_revision, remote_root=remote_root)
    loaded = _imports(remote_root)
    gate_zero = loaded["gate_zero"]
    t1_panel = loaded["t1_panel"]
    supplied = {
        "migration_completion": migration_completion,
        "chunk_cache_plan": chunk_cache_plan,
        "chunk_cache_global_completion": chunk_cache_global_completion,
        "decision_plan": decision_plan,
        "decision_completion": decision_completion,
        "gate_zero_evidence": gate_zero_evidence,
        "gate_zero_decision": gate_zero_decision,
        "gate_zero_completion": gate_zero_completion,
    }
    mounted = {
        name: _artifact_path(value, artifact_root=artifact_root, field=name)
        for name, value in supplied.items()
    }
    index = loaded["resolve_decision_source"](
        migration_completion_path=mounted["migration_completion"],
        chunk_cache_plan_path=mounted["chunk_cache_plan"],
        chunk_cache_global_completion_path=mounted["chunk_cache_global_completion"],
        decision_plan_path=mounted["decision_plan"],
        decision_completion_path=mounted["decision_completion"],
        artifact_root=artifact_root,
        repo_root=remote_root,
    )
    gate_zero_directory = _require_gate_zero_siblings(
        evidence=mounted["gate_zero_evidence"],
        decision=mounted["gate_zero_decision"],
        completion=mounted["gate_zero_completion"],
        gate_zero=gate_zero,
    )
    gate_zero_contract = gate_zero.load_semantic_gate_zero_structural_contract(
        repo_root=remote_root
    )
    gate_zero_artifacts = gate_zero.load_semantic_gate_zero_structural_artifacts(
        output_directory=gate_zero_directory,
        index=index,
        decision_plan_path=mounted["decision_plan"],
        contract=gate_zero_contract,
        repo_root=remote_root,
    )
    evidence_runtime = gate_zero_artifacts["evidence"].get("model_runtime_identity")
    if not isinstance(evidence_runtime, Mapping):
        raise TypeError("semantic T1 Gate-0 model runtime identity is absent")
    expected_model_source = _require_sha256(
        evidence_runtime.get("source_revision_sha256"),
        field="Gate-0 model source revision",
    )
    policy = _load_panel_policy(root=remote_root)
    if (
        policy.get("panel_kind") != t1_panel.UNIQUE_PANEL_KIND
        or policy.get("objective_unit") != t1_panel.UNIQUE_OBJECTIVE_UNIT
        or policy.get("cache_handoff") != t1_panel.CACHE_HANDOFF
    ):
        raise RuntimeError(
            "semantic T1 panel policy differs from current panel semantics"
        )
    support_time_hex = policy.get("support_time_hex")
    try:
        support_time = float.fromhex(support_time_hex)
    except (TypeError, ValueError) as error:
        raise ValueError("support_time_hex must be a hexadecimal float") from error
    if support_time.hex() != support_time_hex:
        raise ValueError("support_time_hex must be canonical")
    request = t1_panel.SemanticT1PanelRequest.create(
        request_id=policy.get("request_id"),
        source_revision_sha256=expected_model_source,
        support_time=support_time,
        maximum_entries_by_family=dict(policy.get("maximum_entries_by_family", {})),
    )
    panel_implementation_sha256 = t1_panel.semantic_t1_panel_implementation_sha256(
        repo_root=remote_root
    )
    request_payload = _validate_panel_request_payload(
        request.as_payload(), request_type=t1_panel.SemanticT1PanelRequest
    ).as_payload()
    run_request = build_run_request(
        source_revision=revision,
        input_records={
            name: {
                "artifact_path": supplied[name],
                "file_sha256": _file_sha256(path),
            }
            for name, path in mounted.items()
        },
        panel_request=request_payload,
        panel_policy_file_sha256=_file_sha256(remote_root / PANEL_POLICY_SOURCE),
        panel_policy_sha256=policy["policy_sha256"],
        panel_implementation_sha256=panel_implementation_sha256,
        output_prefix=output_prefix,
    )
    output_parent = _artifact_path(
        f"{output_prefix}/placeholder",
        artifact_root=artifact_root,
        field="output_prefix",
    ).parent
    output_directory = output_parent / run_request["run_identity_sha256"]
    request_path = output_directory / RUN_REQUEST_FILENAME
    panel_path = output_directory / PANEL_FILENAME
    completion_path = output_directory / COMPLETION_FILENAME
    if (
        loaded["write_bytes_if_absent"](
            request_path, _canonical_bytes(run_request, pretty=True)
        )
        and stage_commit is not None
    ):
        stage_commit()

    if panel_path.exists():
        panel = _validated_panel(
            panel_path=panel_path,
            request=request,
            index=index,
            gate_zero_artifacts=gate_zero_artifacts,
            decision_plan_path=mounted["decision_plan"],
            t1_panel=t1_panel,
            gate_zero_contract=gate_zero_contract,
            remote_root=remote_root,
        )
        panel_reused = True
    else:
        panel = t1_panel.prepare_editing_v2_semantic_t1_panel(
            index,
            gate_zero_artifacts=gate_zero_artifacts,
            decision_plan_path=mounted["decision_plan"],
            request=request,
            gate_zero_contract=gate_zero_contract,
            repo_root=remote_root,
        )
        t1_panel.write_semantic_t1_panel(panel_path, panel)
        if stage_commit is not None:
            stage_commit()
        panel = _validated_panel(
            panel_path=panel_path,
            request=request,
            index=index,
            gate_zero_artifacts=gate_zero_artifacts,
            decision_plan_path=mounted["decision_plan"],
            t1_panel=t1_panel,
            gate_zero_contract=gate_zero_contract,
            remote_root=remote_root,
        )
        panel_reused = False

    panel_file_sha256 = _file_sha256(panel_path)
    panel_file_bytes = panel_path.stat().st_size
    resolved_cache_traces = t1_panel.resolve_semantic_t1_cache_trace_inputs(
        index,
        panel,
        repo_root=remote_root,
    )
    if len(resolved_cache_traces) != len(panel.cache_trace_inputs):
        raise RuntimeError("semantic T1 cache-input trace union is incomplete")
    completion_body = _completion_body(
        run_request=run_request,
        request_path=request_path,
        panel_path=panel_path,
        panel=panel,
        panel_file_sha256=panel_file_sha256,
        panel_file_bytes=panel_file_bytes,
        resolved_cache_trace_count=len(resolved_cache_traces),
        artifact_root=artifact_root,
    )
    completion = {
        **completion_body,
        "completion_sha256": _sha256(completion_body),
    }
    if (
        loaded["write_bytes_if_absent"](
            completion_path, _canonical_bytes(completion) + b"\n"
        )
        and stage_commit is not None
    ):
        stage_commit()
    persisted_completion = _load_canonical_object(
        completion_path, field="semantic T1 completion"
    )
    _validate_completion(persisted_completion, expected_body=completion_body)
    return {
        "run_root": _artifact_address(output_directory, artifact_root=artifact_root),
        "run_identity_sha256": run_request["run_identity_sha256"],
        "request_artifact_path": _artifact_address(
            request_path, artifact_root=artifact_root
        ),
        "panel_artifact_path": _artifact_address(
            panel_path, artifact_root=artifact_root
        ),
        "completion_artifact_path": _artifact_address(
            completion_path, artifact_root=artifact_root
        ),
        "panel_artifact_sha256": panel.artifact_sha256,
        "panel_file_sha256": panel_file_sha256,
        "completion_sha256": completion["completion_sha256"],
        "counts": completion["counts"],
        "panel_reused": panel_reused,
        "training_launched": False,
        "successor_cache_compiled": False,
        "t1_authorized": False,
        "bounded_p50_authorized": False,
    }


@app.function(
    image=image,
    cpu=8.0,
    memory=65536,
    timeout=24 * 3600,
    volumes={str(ARTIFACT_ROOT): artifact_volume},
)
def materialize_semantic_t1_panel_cache_inputs(
    *,
    source_revision: dict[str, object],
    migration_completion: str,
    chunk_cache_plan: str,
    chunk_cache_global_completion: str,
    decision_plan: str,
    decision_completion: str,
    gate_zero_evidence: str,
    gate_zero_decision: str,
    gate_zero_completion: str,
    output_prefix: str,
) -> dict[str, Any]:
    artifact_volume.reload()
    result = _driver_impl(
        source_revision=source_revision,
        migration_completion=migration_completion,
        chunk_cache_plan=chunk_cache_plan,
        chunk_cache_global_completion=chunk_cache_global_completion,
        decision_plan=decision_plan,
        decision_completion=decision_completion,
        gate_zero_evidence=gate_zero_evidence,
        gate_zero_decision=gate_zero_decision,
        gate_zero_completion=gate_zero_completion,
        output_prefix=output_prefix,
        stage_commit=artifact_volume.commit,
    )
    artifact_volume.commit()
    return result


@app.local_entrypoint()
def main(
    migration_completion: str,
    chunk_cache_plan: str,
    chunk_cache_global_completion: str,
    decision_plan: str,
    decision_completion: str,
    gate_zero_evidence: str,
    gate_zero_decision: str,
    gate_zero_completion: str,
    expected_commit: str,
    output_prefix: str = OUTPUT_PREFIX,
) -> None:
    source_revision = local_source_revision(expected_commit=expected_commit)
    result = materialize_semantic_t1_panel_cache_inputs.remote(
        source_revision=source_revision,
        migration_completion=migration_completion,
        chunk_cache_plan=chunk_cache_plan,
        chunk_cache_global_completion=chunk_cache_global_completion,
        decision_plan=decision_plan,
        decision_completion=decision_completion,
        gate_zero_evidence=gate_zero_evidence,
        gate_zero_decision=gate_zero_decision,
        gate_zero_completion=gate_zero_completion,
        output_prefix=output_prefix,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


__all__ = [
    "COMPLETION_FILENAME",
    "OUTPUT_PREFIX",
    "PANEL_FILENAME",
    "PANEL_POLICY_SOURCE",
    "RUN_REQUEST_FILENAME",
    "app",
    "build_run_request",
    "local_source_revision",
    "main",
    "materialize_semantic_t1_panel_cache_inputs",
]
