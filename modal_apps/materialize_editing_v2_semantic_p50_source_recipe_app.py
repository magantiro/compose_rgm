"""Materialize the semantic Editing-V2 P50 source and prepared recipe.

This CPU-only Modal surface reopens the complete semantic migration, chunk,
Active8, Gate 0, and T1 lineage.  Only an exact Gate 0 PASS and bounded-P50 T1
GO may reach candidate classification.  The producer then scans the admitted
train transitions once and freezes the deterministic 50 x 64 prepared stream.

The outputs are content addressed, immutable, and restart safe.  They retain
all five unresolved physical bindings and confer no training authority.  This
module deliberately contains no GPU, optimizer, successor-cache, baseline, or
training-launch path.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import re
import stat
import subprocess
import tempfile
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Any

import modal

ROOT = Path(__file__).resolve().parents[1]
REMOTE_ROOT = Path("/root/compose")
ARTIFACT_ROOT = Path("/artifacts")
LAUNCHER_SOURCE = "modal_apps/materialize_editing_v2_semantic_p50_source_recipe_app.py"
IMAGE_SOURCE_DIRECTORIES = ("src", "configs")
OUTPUT_PREFIX = "/artifacts/editing_v2/semantic_p50_source_recipe"
REQUEST_FILENAME = "SEMANTIC_P50_SOURCE_RECIPE_REQUEST.json"
RECIPE_FILENAME = "SEMANTIC_P50_PREPARED_RECIPE.json"
COMPLETION_FILENAME = "SEMANTIC_P50_SOURCE_RECIPE_COMPLETE.json"

SOURCE_REVISION_SCHEMA = "compose.editing_v2.semantic_p50_source_recipe_modal_source_revision"
REQUEST_SCHEMA = "compose.editing_v2.semantic_p50_source_recipe_request"
COMPLETION_SCHEMA = "compose.editing_v2.semantic_p50_source_recipe_completion"
SCHEMA_VERSION = 1
REQUEST_STATUS = "FROZEN_CPU_SOURCE_RECIPE_REQUEST_NO_TRAINING_AUTHORITY"
COMPLETION_STATUS = "COMPLETE_PREPARED_RECIPE_NO_TRAINING_AUTHORITY"

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_NO_AUTHORITY = {
    "training_authorized": False,
    "bounded_p50_authorized": False,
    "p500_authorized": False,
    "long_training_authorized": False,
    "checkpoint_selection_authorized": False,
    "final_test_selection_authorized": False,
}
_INPUT_NAMES = (
    "migration_completion",
    "chunk_cache_plan",
    "chunk_cache_global_completion",
    "decision_plan",
    "decision_completion",
    "gate_zero_evidence",
    "gate_zero_decision",
    "gate_zero_completion",
    "t1_decision",
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
image = image.add_local_file(
    ROOT / LAUNCHER_SOURCE,
    str(REMOTE_ROOT / LAUNCHER_SOURCE),
    copy=True,
)

app = modal.App("compose-v4-editing-v2-semantic-p50-source-recipe")
artifact_volume = modal.Volume.from_name("compose-v4-artifacts", create_if_missing=False)


def _canonical_bytes(value: object, *, newline: bool = False) -> bytes:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return encoded + (b"\n" if newline else b"")


def _sha(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def _load_canonical(path: Path, *, field: str) -> dict[str, Any]:
    source = Path(path)
    try:
        metadata = source.lstat()
    except FileNotFoundError:
        metadata = None
    if metadata is None or not stat.S_ISREG(metadata.st_mode):
        raise RuntimeError(f"{field} is absent: {source}")
    try:
        raw = source.read_bytes()
        payload = json.loads(raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"{field} is unreadable: {source}") from error
    if not isinstance(payload, dict) or raw != _canonical_bytes(payload, newline=True):
        raise RuntimeError(f"{field} must be canonical newline-terminated JSON")
    return payload


def _require_sha(value: object, *, field: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{field} must be a lowercase SHA-256")
    return value


def _artifact_path(value: str, *, artifact_root: Path, field: str) -> Path:
    if not isinstance(value, str):
        raise TypeError(f"{field} must be a /artifacts path")
    pure = PurePosixPath(value)
    if (
        not pure.is_absolute()
        or len(pure.parts) < 3
        or pure.parts[1] != "artifacts"
        or ".." in pure.parts
        or "\\" in value
        or str(pure) != value
        or value.endswith("/")
    ):
        raise ValueError(f"{field} must be a normalized path below /artifacts")
    root = Path(artifact_root).resolve()
    result = (root / Path(*pure.parts[2:])).resolve()
    if not result.is_relative_to(root):
        raise ValueError(f"{field} resolves outside artifact_root")
    return result


def _artifact_address(path: Path, *, artifact_root: Path) -> str:
    root = Path(artifact_root).resolve()
    resolved = Path(path).resolve()
    if not resolved.is_relative_to(root):
        raise ValueError("artifact resolves outside artifact_root")
    relative = resolved.relative_to(root)
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
            f"cannot establish semantic P50 source-recipe Git identity: git {' '.join(arguments)}"
        ) from error


def _serialized_source_paths(root: Path) -> tuple[str, ...]:
    paths = [LAUNCHER_SOURCE]
    for source_directory in IMAGE_SOURCE_DIRECTORIES:
        paths.extend(
            path.relative_to(root).as_posix()
            for path in sorted((Path(root) / source_directory).rglob("*"))
            if path.is_file() and path.suffix != ".pyc" and "__pycache__" not in path.parts
        )
    if len(paths) != len(set(paths)):
        raise RuntimeError("semantic P50 source-recipe inventory repeats a path")
    return tuple(paths)


def _serialized_source_hashes(root: Path) -> dict[str, str]:
    return {relative: _file_sha(root / relative) for relative in _serialized_source_paths(root)}


def local_source_revision(*, expected_commit: str, repo_root: Path = ROOT) -> dict[str, Any]:
    """Bind the exact clean commit and every source/config byte in the image."""

    if not isinstance(expected_commit, str) or _COMMIT_RE.fullmatch(expected_commit) is None:
        raise ValueError("expected_commit must be a full lowercase Git commit")
    root = Path(repo_root)
    commit = _git(root, "rev-parse", "HEAD")
    tree = _git(root, "rev-parse", "HEAD^{tree}")
    dirty = _git(root, "status", "--porcelain=v1", "--untracked-files=all")
    if commit != expected_commit or dirty:
        raise RuntimeError(
            "semantic P50 source-recipe materialization requires the exact clean committed worktree"
        )
    hashes = _serialized_source_hashes(root)
    tracked = set(_git(root, "ls-files").splitlines())
    if not hashes or not set(hashes).issubset(tracked):
        raise RuntimeError("every serialized semantic P50 source/config must be Git-tracked")
    body: dict[str, Any] = {
        "schema": SOURCE_REVISION_SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "commit": commit,
        "tree": tree,
        "worktree_clean": True,
        "serialized_source_hashes": hashes,
        "serialized_source_hashes_sha256": _sha(hashes),
    }
    return {**body, "source_revision_sha256": _sha(body)}


def _validate_source_revision(value: object, *, remote_root: Path) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError("semantic P50 source revision must be an object")
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
    body = dict(revision)
    supplied_sha = body.pop("source_revision_sha256", None)
    hashes = _serialized_source_hashes(remote_root)
    if (
        set(revision) != expected_fields
        or revision.get("schema") != SOURCE_REVISION_SCHEMA
        or revision.get("schema_version") != SCHEMA_VERSION
        or revision.get("worktree_clean") is not True
        or revision.get("serialized_source_hashes") != hashes
        or revision.get("serialized_source_hashes_sha256") != _sha(hashes)
        or supplied_sha != _sha(body)
        or not isinstance(revision.get("commit"), str)
        or _COMMIT_RE.fullmatch(revision["commit"]) is None
        or not isinstance(revision.get("tree"), str)
        or _COMMIT_RE.fullmatch(revision["tree"]) is None
    ):
        raise RuntimeError("semantic P50 serialized source revision disagrees")
    return revision


def _validated_inputs(supplied: Mapping[str, str], *, artifact_root: Path) -> dict[str, Path]:
    if set(supplied) != set(_INPUT_NAMES):
        raise ValueError("semantic P50 source-recipe input inventory disagrees")
    mounted = {
        name: _artifact_path(supplied[name], artifact_root=artifact_root, field=name)
        for name in _INPUT_NAMES
    }
    for name, path in mounted.items():
        if not path.is_file():
            raise RuntimeError(f"semantic P50 input {name} is absent: {path}")
    return mounted


def _input_records(
    supplied: Mapping[str, str], mounted: Mapping[str, Path]
) -> dict[str, dict[str, str]]:
    """Snapshot every physical prerequisite before any long corpus scan."""

    if set(supplied) != set(_INPUT_NAMES) or set(mounted) != set(_INPUT_NAMES):
        raise ValueError("semantic P50 physical input inventory disagrees")
    return {
        name: {
            "artifact_path": supplied[name],
            "file_sha256": _file_sha(mounted[name]),
        }
        for name in _INPUT_NAMES
    }


def _require_input_records_unchanged(
    mounted: Mapping[str, Path], expected: Mapping[str, Mapping[str, str]]
) -> None:
    """Fail if any prerequisite changed after its initial physical reopen."""

    if set(mounted) != set(_INPUT_NAMES) or set(expected) != set(_INPUT_NAMES):
        raise RuntimeError("semantic P50 input snapshot inventory disagrees")
    changed = [
        name
        for name in _INPUT_NAMES
        if not mounted[name].is_file() or _file_sha(mounted[name]) != expected[name]["file_sha256"]
    ]
    if changed:
        raise RuntimeError(
            f"semantic P50 physical prerequisites changed during compilation: {changed}"
        )


def build_run_request(
    *,
    source_revision: Mapping[str, Any],
    input_records: Mapping[str, Mapping[str, str]],
    source_identity: Mapping[str, Any],
    gate_zero_artifacts: Mapping[str, Mapping[str, Any]],
    t1_decision: Mapping[str, Any],
    recipe_policy: Mapping[str, Any],
    recipe_policy_file_sha256: str,
    capability_registry_sha256: str,
    output_prefix: str = OUTPUT_PREFIX,
) -> dict[str, Any]:
    """Content-address one exact CPU-only source and recipe compilation."""

    if set(input_records) != set(_INPUT_NAMES):
        raise ValueError("semantic P50 source-recipe input records disagree")
    normalized_inputs: dict[str, dict[str, str]] = {}
    for name in _INPUT_NAMES:
        record = input_records[name]
        if not isinstance(record, Mapping) or set(record) != {
            "artifact_path",
            "file_sha256",
        }:
            raise ValueError(f"semantic P50 input record {name} fields disagree")
        artifact_path = str(record["artifact_path"])
        _artifact_path(artifact_path, artifact_root=ARTIFACT_ROOT, field=name)
        normalized_inputs[name] = {
            "artifact_path": artifact_path,
            "file_sha256": _require_sha(record["file_sha256"], field=f"{name}.file_sha256"),
        }
    _artifact_path(
        f"{output_prefix}/placeholder",
        artifact_root=ARTIFACT_ROOT,
        field="output_prefix",
    )
    gate_evidence = gate_zero_artifacts.get("evidence")
    gate_decision = gate_zero_artifacts.get("decision")
    gate_completion = gate_zero_artifacts.get("completion")
    if not all(
        isinstance(item, Mapping) for item in (gate_evidence, gate_decision, gate_completion)
    ):
        raise ValueError("semantic Gate 0 artifact inventory is incomplete")
    body: dict[str, Any] = {
        "schema": REQUEST_SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "status": REQUEST_STATUS,
        **_NO_AUTHORITY,
        "source_revision": dict(source_revision),
        "runtime": {
            "device": "cpu",
            "python_implementation": platform.python_implementation(),
            "python_version": platform.python_version(),
            "torch_version": importlib.metadata.version("torch"),
            "numpy_version": importlib.metadata.version("numpy"),
            "scipy_version": importlib.metadata.version("scipy"),
            "networkx_version": importlib.metadata.version("networkx"),
            "rdkit_version": importlib.metadata.version("rdkit"),
        },
        "inputs": normalized_inputs,
        "source_identity": {
            "inventory_sha256": _require_sha(
                source_identity.get("inventory_sha256"),
                field="source_identity.inventory_sha256",
            ),
            "process_identity_sha256": _require_sha(
                source_identity.get("process_identity_sha256"),
                field="source_identity.process_identity_sha256",
            ),
            "model_runtime_identity_sha256": _require_sha(
                source_identity.get("model_runtime_identity_sha256"),
                field="source_identity.model_runtime_identity_sha256",
            ),
            "policy_sha256": _require_sha(
                source_identity.get("policy_sha256"),
                field="source_identity.policy_sha256",
            ),
            "decision_source_implementation_sha256": _require_sha(
                source_identity.get("decision_source_implementation_sha256"),
                field="source_identity.decision_source_implementation_sha256",
            ),
        },
        "gate_zero": {
            "evidence_sha256": _require_sha(
                gate_evidence.get("evidence_sha256"), field="gate_zero.evidence_sha256"
            ),
            "decision_sha256": _require_sha(
                gate_decision.get("decision_sha256"), field="gate_zero.decision_sha256"
            ),
            "completion_sha256": _require_sha(
                gate_completion.get("completion_sha256"),
                field="gate_zero.completion_sha256",
            ),
            "structural_result": gate_evidence.get("structural_result"),
        },
        "t1": {
            "decision_sha256": _require_sha(
                t1_decision.get("decision_sha256"), field="t1.decision_sha256"
            ),
            "status": t1_decision.get("status"),
            "initial_model_state_sha256": _require_sha(
                t1_decision.get("initial_model_state_sha256"),
                field="t1.initial_model_state_sha256",
            ),
            "selected_model_state_sha256": _require_sha(
                t1_decision.get("selected_model_state_sha256"),
                field="t1.selected_model_state_sha256",
            ),
        },
        "capability_registry_sha256": _require_sha(
            capability_registry_sha256, field="capability_registry_sha256"
        ),
        "recipe_policy_file_sha256": _require_sha(
            recipe_policy_file_sha256, field="recipe_policy_file_sha256"
        ),
        "recipe_policy_sha256": _require_sha(
            recipe_policy.get("policy_sha256"), field="recipe_policy.policy_sha256"
        ),
        "output_prefix": output_prefix,
    }
    return {**body, "run_identity_sha256": _sha(body)}


def _validate_prepared_recipe(
    prepared: Mapping[str, Any],
    *,
    recipe_stream: Any,
    expected_prerequisites: Mapping[str, Any],
) -> None:
    """Enforce the bounded prepared-only boundary before and after publication."""

    recipe = prepared.get("recipe")
    stream_contract = prepared.get("stream_contract")
    if not isinstance(recipe, Mapping) or not isinstance(stream_contract, Mapping):
        raise RuntimeError("semantic P50 prepared recipe lacks stream metadata")
    if (
        prepared.get("schema") != recipe_stream.PREPARED_SCHEMA
        or prepared.get("status") != recipe_stream.PREPARED_STATUS
        or prepared.get("prerequisites") != dict(expected_prerequisites)
        or prepared.get("required_physical_binding_purposes")
        != list(recipe_stream.REQUIRED_BINDING_PURPOSES)
        or prepared.get("unresolved_physical_bindings")
        != list(recipe_stream.REQUIRED_BINDING_PURPOSES)
        or len(prepared.get("ordered_stream_rows", ())) != recipe_stream.SCHEDULED_EXAMPLES
        or recipe.get("optimizer_steps") != recipe_stream.OPTIMIZER_STEPS
        or recipe.get("batch_size") != recipe_stream.BATCH_SIZE
        or stream_contract.get("scheduled_nonterminal_examples") != recipe_stream.SCHEDULED_EXAMPLES
        or any(prepared.get(name) is not False for name in _NO_AUTHORITY)
    ):
        raise RuntimeError("semantic P50 prepared recipe crosses its frozen boundary")


def _completion_payload(
    *,
    request: Mapping[str, Any],
    request_path: Path,
    published_source: Any,
    prepared: Mapping[str, Any],
    recipe_path: Path,
    gate_zero_artifacts: Mapping[str, Mapping[str, Any]],
    t1_decision: Mapping[str, Any],
    recipe_policy: Mapping[str, Any],
    source_revision: Mapping[str, Any],
    recipe_stream: Any,
) -> dict[str, Any]:
    """Build the exact completion from already verified physical artifacts."""

    body: dict[str, Any] = {
        "schema": COMPLETION_SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "status": COMPLETION_STATUS,
        **_NO_AUTHORITY,
        "training_launched": False,
        "request_relative_path": REQUEST_FILENAME,
        "request_file_sha256": _file_sha(request_path),
        "run_identity_sha256": request["run_identity_sha256"],
        "source_inventory_relative_path": (published_source.path.name),
        "source_inventory_file_sha256": _file_sha(published_source.path),
        "source_inventory_sha256": published_source.binding.source_inventory_sha256,
        "prepared_recipe_relative_path": RECIPE_FILENAME,
        "prepared_recipe_file_sha256": _file_sha(recipe_path),
        "prepared_recipe_sha256": prepared["prepared_recipe_sha256"],
        "candidate_inventory_sha256": prepared["candidate_inventory_sha256"],
        "candidate_count": prepared["candidate_count"],
        "scheduled_example_count": len(prepared["ordered_stream_rows"]),
        "optimizer_steps": recipe_stream.OPTIMIZER_STEPS,
        "batch_size": recipe_stream.BATCH_SIZE,
        "required_physical_binding_purposes": list(recipe_stream.REQUIRED_BINDING_PURPOSES),
        "unresolved_physical_bindings": list(recipe_stream.REQUIRED_BINDING_PURPOSES),
        "process_identity_sha256": published_source.binding.process_identity_sha256,
        "model_runtime_identity_sha256": (published_source.binding.model_runtime_identity_sha256),
        "gate_zero_evidence_sha256": gate_zero_artifacts["evidence"]["evidence_sha256"],
        "t1_decision_sha256": t1_decision["decision_sha256"],
        "recipe_policy_sha256": recipe_policy["policy_sha256"],
        "source_revision_sha256": source_revision["source_revision_sha256"],
    }
    return {**body, "completion_sha256": _sha(body)}


def _driver_result(
    *,
    output_directory: Path,
    request_path: Path,
    source_path: Path,
    recipe_path: Path,
    completion_path: Path,
    request: Mapping[str, Any],
    prepared: Mapping[str, Any],
    recipe_stream: Any,
    request_reused: bool,
    source_reused: bool,
    recipe_reused: bool,
    completion_reused: bool,
    artifact_root: Path,
) -> dict[str, Any]:
    return {
        "run_root": _artifact_address(output_directory, artifact_root=artifact_root),
        "request_artifact_path": _artifact_address(request_path, artifact_root=artifact_root),
        "source_inventory_artifact_path": _artifact_address(
            source_path, artifact_root=artifact_root
        ),
        "prepared_recipe_artifact_path": _artifact_address(
            recipe_path, artifact_root=artifact_root
        ),
        "completion_artifact_path": _artifact_address(completion_path, artifact_root=artifact_root),
        "run_identity_sha256": request["run_identity_sha256"],
        "candidate_count": prepared["candidate_count"],
        "scheduled_example_count": recipe_stream.SCHEDULED_EXAMPLES,
        "request_reused": request_reused,
        "source_inventory_reused": source_reused,
        "prepared_recipe_reused": recipe_reused,
        "completion_reused": completion_reused,
        "training_launched": False,
        "training_authorized": False,
        "bounded_p50_authorized": False,
        "unresolved_physical_bindings": list(recipe_stream.REQUIRED_BINDING_PURPOSES),
    }


def _imports(remote_root: Path = REMOTE_ROOT) -> dict[str, Any]:
    import sys

    source_root = str(remote_root / "src")
    if source_root not in sys.path:
        sys.path.insert(0, source_root)
    from compose_v4.data.editing_v2_semantic_active8_decision_source import (
        resolve_editing_v2_semantic_active8_decision_source,
    )
    from compose_v4.data.editing_v2_semantic_capability_cells import (
        load_semantic_capability_cell_registry,
    )
    from compose_v4.data.immutable_artifact import write_bytes_if_absent
    from compose_v4.experiments import editing_v2_semantic_gate_zero as gate_zero
    from compose_v4.experiments import editing_v2_semantic_p50_recipe_stream as recipe_stream
    from compose_v4.experiments import editing_v2_semantic_p50_source_inventory as source_inventory
    from compose_v4.experiments import editing_v2_semantic_t1_decision as t1_decision

    return {
        "gate_zero": gate_zero,
        "load_registry": load_semantic_capability_cell_registry,
        "recipe_stream": recipe_stream,
        "resolve_source": resolve_editing_v2_semantic_active8_decision_source,
        "source_inventory": source_inventory,
        "t1_decision": t1_decision,
        "write_bytes_if_absent": write_bytes_if_absent,
    }


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
    t1_decision: str,
    output_prefix: str,
    artifact_root: Path = ARTIFACT_ROOT,
    remote_root: Path = REMOTE_ROOT,
    stage_commit: Any | None = None,
) -> dict[str, Any]:
    revision = _validate_source_revision(source_revision, remote_root=remote_root)
    loaded = _imports(remote_root)
    supplied = {
        "migration_completion": migration_completion,
        "chunk_cache_plan": chunk_cache_plan,
        "chunk_cache_global_completion": chunk_cache_global_completion,
        "decision_plan": decision_plan,
        "decision_completion": decision_completion,
        "gate_zero_evidence": gate_zero_evidence,
        "gate_zero_decision": gate_zero_decision,
        "gate_zero_completion": gate_zero_completion,
        "t1_decision": t1_decision,
    }
    mounted = _validated_inputs(supplied, artifact_root=artifact_root)
    input_records = _input_records(supplied, mounted)

    index = loaded["resolve_source"](
        migration_completion_path=mounted["migration_completion"],
        chunk_cache_plan_path=mounted["chunk_cache_plan"],
        chunk_cache_global_completion_path=mounted["chunk_cache_global_completion"],
        decision_plan_path=mounted["decision_plan"],
        decision_completion_path=mounted["decision_completion"],
        artifact_root=artifact_root,
        repo_root=remote_root,
    )

    gate_zero = loaded["gate_zero"]
    expected_gate_paths = {
        "gate_zero_evidence": gate_zero.EVIDENCE_FILENAME,
        "gate_zero_decision": gate_zero.DECISION_FILENAME,
        "gate_zero_completion": gate_zero.COMPLETION_FILENAME,
    }
    gate_root = mounted["gate_zero_evidence"].parent
    if any(
        mounted[name].name != filename or mounted[name].parent != gate_root
        for name, filename in expected_gate_paths.items()
    ):
        raise RuntimeError("semantic Gate 0 artifacts must be exact named siblings")
    gate_contract = gate_zero.load_semantic_gate_zero_structural_contract(repo_root=remote_root)
    gate_artifacts = gate_zero.load_semantic_gate_zero_structural_artifacts(
        output_directory=gate_root,
        index=index,
        decision_plan_path=mounted["decision_plan"],
        contract=gate_contract,
        repo_root=remote_root,
    )
    if gate_artifacts["evidence"].get("structural_result") != "PASS":
        raise RuntimeError("semantic Gate 0 structural evidence is not PASS")

    t1_module = loaded["t1_decision"]
    if mounted["t1_decision"].name != t1_module.DECISION_FILENAME:
        raise RuntimeError("semantic T1 decision has the wrong filename")
    t1_payload = _load_canonical(mounted["t1_decision"], field="semantic T1 capacity decision")
    verified_t1 = t1_module.validate_semantic_t1_capacity_decision(
        t1_payload,
        decision_path=mounted["t1_decision"],
        repo_root=remote_root,
        expected_gate_zero_evidence_file_sha256=_file_sha(mounted["gate_zero_evidence"]),
        require_p50_go=True,
    )
    _require_input_records_unchanged(mounted, input_records)

    recipe_stream = loaded["recipe_stream"]
    policy_path = remote_root / recipe_stream.POLICY_RELATIVE_PATH
    recipe_policy, recipe_policy_file_sha256 = recipe_stream.load_semantic_p50_recipe_policy(
        policy_path
    )
    registry = loaded["load_registry"]()
    request = build_run_request(
        source_revision=revision,
        input_records=input_records,
        source_identity=index.identity_payload(),
        gate_zero_artifacts=gate_artifacts,
        t1_decision=verified_t1,
        recipe_policy=recipe_policy,
        recipe_policy_file_sha256=recipe_policy_file_sha256,
        capability_registry_sha256=registry.registry_sha256,
        output_prefix=output_prefix,
    )
    prefix_parent = _artifact_path(
        f"{output_prefix}/placeholder",
        artifact_root=artifact_root,
        field="output_prefix",
    ).parent
    output_directory = prefix_parent / request["run_identity_sha256"]
    request_path = output_directory / REQUEST_FILENAME
    request_created = loaded["write_bytes_if_absent"](
        request_path, _canonical_bytes(request, newline=True)
    )
    if request_created and stage_commit is not None:
        stage_commit()

    source_module = loaded["source_inventory"]
    source_path = output_directory / source_module.SEMANTIC_P50_SOURCE_INVENTORY_FILENAME
    source_reused = source_path.exists()
    published_source = source_module.publish_semantic_p50_source_inventory(
        index,
        decision_plan_path=mounted["decision_plan"],
        output_directory=output_directory,
    )
    if not source_reused and stage_commit is not None:
        stage_commit()
    verified_source = source_module.load_semantic_p50_source_inventory(
        published_source.path,
        expected_binding=published_source.binding,
        migration_completion_path=mounted["migration_completion"],
        chunk_cache_plan_path=mounted["chunk_cache_plan"],
        chunk_cache_global_completion_path=mounted["chunk_cache_global_completion"],
        decision_plan_path=mounted["decision_plan"],
        decision_completion_path=mounted["decision_completion"],
        artifact_root=artifact_root,
        repo_root=remote_root,
    )
    if verified_source.index.identity_payload() != index.identity_payload():
        raise RuntimeError("reopened semantic P50 source identity drifted")
    expected_prerequisites = recipe_stream.validate_semantic_p50_prerequisite_relationships(
        source=verified_source,
        gate_zero=gate_artifacts["evidence"],
        gate_zero_file_sha256=input_records["gate_zero_evidence"]["file_sha256"],
        t1_decision=verified_t1,
        t1_decision_file_sha256=input_records["t1_decision"]["file_sha256"],
        registry=registry,
    ).as_payload()
    _require_input_records_unchanged(mounted, input_records)

    recipe_path = output_directory / RECIPE_FILENAME
    completion_path = output_directory / COMPLETION_FILENAME
    if completion_path.exists():
        existing_request = _load_canonical(request_path, field="semantic P50 source request")
        prepared = _load_canonical(recipe_path, field="prepared semantic P50 recipe")
        _validate_prepared_recipe(
            prepared,
            recipe_stream=recipe_stream,
            expected_prerequisites=expected_prerequisites,
        )
        expected_completion = _completion_payload(
            request=request,
            request_path=request_path,
            published_source=published_source,
            prepared=prepared,
            recipe_path=recipe_path,
            gate_zero_artifacts=gate_artifacts,
            t1_decision=verified_t1,
            recipe_policy=recipe_policy,
            source_revision=revision,
            recipe_stream=recipe_stream,
        )
        observed_completion = _load_canonical(
            completion_path, field="semantic P50 source-recipe completion"
        )
        _require_input_records_unchanged(mounted, input_records)
        if existing_request != request or observed_completion != expected_completion:
            raise RuntimeError("completed semantic P50 source-recipe run differs on strict reopen")
        return _driver_result(
            output_directory=output_directory,
            request_path=request_path,
            source_path=published_source.path,
            recipe_path=recipe_path,
            completion_path=completion_path,
            request=request,
            prepared=prepared,
            recipe_stream=recipe_stream,
            request_reused=not request_created,
            source_reused=source_reused,
            recipe_reused=True,
            completion_reused=True,
            artifact_root=artifact_root,
        )

    candidates = recipe_stream.build_semantic_p50_candidate_inventory(
        source=verified_source,
        gate_zero_evidence_path=mounted["gate_zero_evidence"],
        t1_decision_path=mounted["t1_decision"],
        repo_root=remote_root,
        registry=registry,
    )
    _require_input_records_unchanged(mounted, input_records)
    if candidates.prerequisites.as_payload() != expected_prerequisites:
        raise RuntimeError("semantic P50 candidate scan reopened another prerequisite generation")
    prepared = recipe_stream.compile_semantic_p50_prepared_recipe(
        candidates, policy_path=policy_path
    )
    _validate_prepared_recipe(
        prepared,
        recipe_stream=recipe_stream,
        expected_prerequisites=expected_prerequisites,
    )

    with tempfile.TemporaryDirectory(prefix="compose-semantic-p50-recipe-") as scratch:
        staged_recipe = Path(scratch) / RECIPE_FILENAME
        recipe_stream.write_semantic_p50_prepared_recipe(staged_recipe, prepared)
        recipe_bytes = staged_recipe.read_bytes()
    if recipe_bytes != _canonical_bytes(prepared, newline=True):
        raise RuntimeError("semantic P50 recipe writer changed canonical bytes")
    recipe_created = loaded["write_bytes_if_absent"](recipe_path, recipe_bytes)
    if recipe_created and stage_commit is not None:
        stage_commit()
    reopened_recipe = _load_canonical(recipe_path, field="prepared semantic P50 recipe")
    if reopened_recipe != prepared:
        raise RuntimeError("published semantic P50 recipe differs from compilation")
    _validate_prepared_recipe(
        reopened_recipe,
        recipe_stream=recipe_stream,
        expected_prerequisites=expected_prerequisites,
    )
    _require_input_records_unchanged(mounted, input_records)

    completion = _completion_payload(
        request=request,
        request_path=request_path,
        published_source=published_source,
        prepared=prepared,
        recipe_path=recipe_path,
        gate_zero_artifacts=gate_artifacts,
        t1_decision=verified_t1,
        recipe_policy=recipe_policy,
        source_revision=revision,
        recipe_stream=recipe_stream,
    )
    completion_created = loaded["write_bytes_if_absent"](
        completion_path, _canonical_bytes(completion, newline=True)
    )
    if completion_created and stage_commit is not None:
        stage_commit()
    if (
        _load_canonical(completion_path, field="semantic P50 source-recipe completion")
        != completion
    ):
        raise RuntimeError("semantic P50 source-recipe completion differs on reopen")

    return _driver_result(
        output_directory=output_directory,
        request_path=request_path,
        source_path=published_source.path,
        recipe_path=recipe_path,
        completion_path=completion_path,
        request=request,
        prepared=prepared,
        recipe_stream=recipe_stream,
        request_reused=not request_created,
        source_reused=source_reused,
        recipe_reused=not recipe_created,
        completion_reused=not completion_created,
        artifact_root=artifact_root,
    )


@app.function(
    image=image,
    cpu=8.0,
    memory=65536,
    timeout=24 * 3600,
    max_containers=1,
    volumes={str(ARTIFACT_ROOT): artifact_volume},
)
def materialize_source_recipe(
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
    t1_decision: str,
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
        t1_decision=t1_decision,
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
    t1_decision: str,
    expected_commit: str,
    output_prefix: str = OUTPUT_PREFIX,
) -> None:
    source_revision = local_source_revision(expected_commit=expected_commit)
    result = materialize_source_recipe.remote(
        source_revision=source_revision,
        migration_completion=migration_completion,
        chunk_cache_plan=chunk_cache_plan,
        chunk_cache_global_completion=chunk_cache_global_completion,
        decision_plan=decision_plan,
        decision_completion=decision_completion,
        gate_zero_evidence=gate_zero_evidence,
        gate_zero_decision=gate_zero_decision,
        gate_zero_completion=gate_zero_completion,
        t1_decision=t1_decision,
        output_prefix=output_prefix,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


__all__ = [
    "COMPLETION_FILENAME",
    "OUTPUT_PREFIX",
    "RECIPE_FILENAME",
    "REQUEST_FILENAME",
    "app",
    "build_run_request",
    "local_source_revision",
    "main",
    "materialize_source_recipe",
]
