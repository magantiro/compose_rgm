"""Immutable, manifest-complete publication for a passing P50 sentinel."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
import shutil
from typing import Any, Mapping

import torch

P50_COMPLETION_SCHEMA = "compose.editing_p50_completion"
P50_COMPLETION_SCHEMA_VERSION = 1
P50_COMPLETION_STEPS = 50
P50_COMPLETION_MANIFEST = "completion.json"


class P50CompletionError(RuntimeError):
    """A P50 completion namespace is incomplete, mutable, or already occupied."""


@dataclass(frozen=True)
class PublishedP50Completion:
    namespace: Path
    manifest: Path
    selected_checkpoint: Path
    recovery_checkpoint: Path
    manifest_sha256: str
    selected_sha256: str
    recovery_sha256: str

    def report(self) -> dict[str, object]:
        return {
            "namespace": str(self.namespace),
            "manifest": str(self.manifest),
            "manifest_sha256": self.manifest_sha256,
            "selected_checkpoint": str(self.selected_checkpoint),
            "selected_checkpoint_sha256": self.selected_sha256,
            "recovery_checkpoint": str(self.recovery_checkpoint),
            "recovery_checkpoint_sha256": self.recovery_sha256,
        }


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def p50_completion_namespace(checkpoint_target: Path) -> Path:
    return checkpoint_target.with_name(
        f"{checkpoint_target.stem}.p50_complete"
    )


def _p50_staging_namespace(checkpoint_target: Path) -> Path:
    namespace = p50_completion_namespace(checkpoint_target)
    return namespace.with_name(f".{namespace.name}.staging")


def _legacy_p50_targets(checkpoint_target: Path) -> tuple[Path, ...]:
    return (
        checkpoint_target,
        checkpoint_target.with_name(
            f"{checkpoint_target.stem}.recovery{checkpoint_target.suffix}"
        ),
        checkpoint_target.with_name(
            f"{checkpoint_target.stem}.best_so_far{checkpoint_target.suffix}"
        ),
        checkpoint_target.with_name(
            f"{checkpoint_target.stem}.step50{checkpoint_target.suffix}"
        ),
        checkpoint_target.with_name(f".{checkpoint_target.name}.tmp"),
    )


def p50_completion_targets(checkpoint_target: Path) -> tuple[Path, ...]:
    """All names whose prior existence makes a P50 launch ambiguous."""

    target = Path(checkpoint_target)
    return (
        *_legacy_p50_targets(target),
        p50_completion_namespace(target),
        _p50_staging_namespace(target),
    )


def assert_p50_completion_targets_absent(checkpoint_target: Path) -> None:
    occupied = tuple(
        path for path in p50_completion_targets(checkpoint_target) if path.exists()
    )
    if occupied:
        raise P50CompletionError(
            "P50 completion targets already exist; refusing a destructive or "
            f"ambiguous launch: {[str(path) for path in occupied]}"
        )


def _checkpoint_filename(role: str, suffix: str) -> str:
    return f"{role}{suffix or '.pt'}"


def publish_p50_completion(
    checkpoint_target: Path,
    *,
    selected_payload: Mapping[str, object],
    recovery_payload: Mapping[str, object],
) -> PublishedP50Completion:
    """Publish both artifacts atomically by renaming one complete namespace."""

    target = Path(checkpoint_target)
    assert_p50_completion_targets_absent(target)
    namespace = p50_completion_namespace(target)
    staging = _p50_staging_namespace(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        staging.mkdir()
    except FileExistsError as error:
        raise P50CompletionError(
            f"P50 staging namespace already exists: {staging}"
        ) from error

    selected_name = _checkpoint_filename("selected", target.suffix)
    recovery_name = _checkpoint_filename("recovery", target.suffix)
    selected_path = staging / selected_name
    recovery_path = staging / recovery_name
    selected = {
        **selected_payload,
        "p50_completion_required": True,
        "p50_completion_role": "selected",
        "p50_completion_namespace": namespace.name,
        "p50_completion_manifest": P50_COMPLETION_MANIFEST,
    }
    recovery = {
        **recovery_payload,
        "p50_completion_required": True,
        "p50_completion_role": "recovery",
        "p50_completion_namespace": namespace.name,
        "p50_completion_manifest": P50_COMPLETION_MANIFEST,
    }
    published = False
    try:
        if selected.get("checkpoint_kind") != "selected_evaluation_model":
            raise P50CompletionError(
                "P50 selected payload has another checkpoint kind"
            )
        if recovery.get("checkpoint_kind") != "exact_training_recovery":
            raise P50CompletionError(
                "P50 recovery payload has another checkpoint kind"
            )
        if int(recovery.get("completed_steps", -1)) != P50_COMPLETION_STEPS:
            raise P50CompletionError(
                "P50 recovery payload is not the exact step-50 state"
            )
        torch.save(selected, selected_path)
        torch.save(recovery, recovery_path)
        selected_sha256 = _file_sha256(selected_path)
        recovery_sha256 = _file_sha256(recovery_path)
        manifest_payload = {
            "schema": P50_COMPLETION_SCHEMA,
            "schema_version": P50_COMPLETION_SCHEMA_VERSION,
            "status": "PASS",
            "completed_optimizer_steps": P50_COMPLETION_STEPS,
            "checkpoint_target": target.name,
            "artifacts": {
                "selected": {
                    "filename": selected_name,
                    "sha256": selected_sha256,
                    "checkpoint_kind": "selected_evaluation_model",
                },
                "recovery": {
                    "filename": recovery_name,
                    "sha256": recovery_sha256,
                    "checkpoint_kind": "exact_training_recovery",
                },
            },
        }
        manifest_path = staging / P50_COMPLETION_MANIFEST
        manifest_path.write_text(
            json.dumps(manifest_payload, indent=2, sort_keys=True) + "\n"
        )
        if namespace.exists():
            raise P50CompletionError(
                f"P50 completion namespace appeared concurrently: {namespace}"
            )
        os.rename(staging, namespace)
        published = True
    finally:
        if not published and staging.exists():
            shutil.rmtree(staging)

    final_manifest = namespace / P50_COMPLETION_MANIFEST
    return PublishedP50Completion(
        namespace=namespace,
        manifest=final_manifest,
        selected_checkpoint=namespace / selected_name,
        recovery_checkpoint=namespace / recovery_name,
        manifest_sha256=_file_sha256(final_manifest),
        selected_sha256=selected_sha256,
        recovery_sha256=recovery_sha256,
    )


def validate_p50_completion_member(
    path: Path,
    payload: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Validate manifest membership when a checkpoint declares it is required."""

    if "p50_completion_required" not in payload:
        return None
    if payload["p50_completion_required"] is not True:
        raise P50CompletionError(
            "p50_completion_required must be the literal Boolean true"
        )
    role = payload.get("p50_completion_role")
    if role not in {"selected", "recovery"}:
        raise P50CompletionError("P50 checkpoint has an invalid completion role")
    namespace_name = payload.get("p50_completion_namespace")
    if namespace_name != path.parent.name:
        raise P50CompletionError(
            "P50 checkpoint is outside its declared immutable namespace"
        )
    manifest_name = payload.get("p50_completion_manifest")
    if manifest_name != P50_COMPLETION_MANIFEST:
        raise P50CompletionError("P50 checkpoint names another completion manifest")
    manifest_path = path.parent / P50_COMPLETION_MANIFEST
    if not manifest_path.is_file():
        raise P50CompletionError("P50 completion manifest is absent")
    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise P50CompletionError("P50 completion manifest is unreadable") from error
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema") != P50_COMPLETION_SCHEMA
        or manifest.get("schema_version") != P50_COMPLETION_SCHEMA_VERSION
        or manifest.get("status") != "PASS"
        or manifest.get("completed_optimizer_steps") != P50_COMPLETION_STEPS
    ):
        raise P50CompletionError(
            "P50 completion manifest is not an exact step-50 PASS"
        )
    artifacts = manifest.get("artifacts")
    artifact = artifacts.get(role) if isinstance(artifacts, dict) else None
    if (
        not isinstance(artifact, dict)
        or artifact.get("filename") != path.name
        or artifact.get("checkpoint_kind") != payload.get("checkpoint_kind")
        or artifact.get("sha256") != _file_sha256(path)
    ):
        raise P50CompletionError(
            "P50 checkpoint bytes do not match the completion manifest"
        )
    return manifest


__all__ = [
    "P50CompletionError",
    "P50_COMPLETION_MANIFEST",
    "P50_COMPLETION_SCHEMA",
    "P50_COMPLETION_SCHEMA_VERSION",
    "PublishedP50Completion",
    "assert_p50_completion_targets_absent",
    "p50_completion_namespace",
    "p50_completion_targets",
    "publish_p50_completion",
    "validate_p50_completion_member",
]
