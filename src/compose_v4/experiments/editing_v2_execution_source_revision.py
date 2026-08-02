"""Canonical clean-source identity shared across Editing-V2 producers.

Producer-specific Modal attestations bind the exact bytes serialized into one
image.  Those attestations are intentionally different across producers.  The
nested execution-source revision defined here is the only identity that may be
compared across those producer namespaces.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from typing import Any

EXECUTION_SOURCE_REVISION_SCHEMA = "compose.editing_v2.semantic_p50_clean_source_revision"
EXECUTION_SOURCE_REVISION_SCHEMA_VERSION = 1

_GIT_RE = re.compile(r"^[0-9a-f]{40}$")
_SHA_RE = re.compile(r"^[0-9a-f]{64}$")
_FIELDS = {
    "schema",
    "schema_version",
    "commit",
    "tree",
    "worktree_clean",
    "source_revision_sha256",
}


class EditingV2ExecutionSourceRevisionError(ValueError):
    """A canonical execution-source revision is malformed or inconsistent."""


def _canonical_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise EditingV2ExecutionSourceRevisionError(
            "execution-source revision is not finite canonical JSON"
        ) from error


def _sha(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def build_execution_source_revision(*, commit: str, tree: str) -> dict[str, Any]:
    """Build the cross-producer identity for one clean committed source tree."""

    if _GIT_RE.fullmatch(commit) is None or _GIT_RE.fullmatch(tree) is None:
        raise EditingV2ExecutionSourceRevisionError(
            "execution-source commit and tree must be full lowercase Git objects"
        )
    body: dict[str, Any] = {
        "schema": EXECUTION_SOURCE_REVISION_SCHEMA,
        "schema_version": EXECUTION_SOURCE_REVISION_SCHEMA_VERSION,
        "commit": commit,
        "tree": tree,
        "worktree_clean": True,
    }
    return {**body, "source_revision_sha256": _sha(body)}


def validate_execution_source_revision(value: object) -> dict[str, Any]:
    """Strictly validate one canonical nested execution-source identity."""

    if not isinstance(value, Mapping) or set(value) != _FIELDS:
        raise EditingV2ExecutionSourceRevisionError("execution-source revision fields disagree")
    revision = dict(value)
    body = dict(revision)
    supplied = body.pop("source_revision_sha256")
    if (
        revision.get("schema") != EXECUTION_SOURCE_REVISION_SCHEMA
        or revision.get("schema_version") != EXECUTION_SOURCE_REVISION_SCHEMA_VERSION
        or _GIT_RE.fullmatch(str(revision.get("commit"))) is None
        or _GIT_RE.fullmatch(str(revision.get("tree"))) is None
        or revision.get("worktree_clean") is not True
        or not isinstance(supplied, str)
        or _SHA_RE.fullmatch(supplied) is None
        or supplied != _sha(body)
    ):
        raise EditingV2ExecutionSourceRevisionError("execution-source revision identity disagrees")
    return revision


__all__ = [
    "EXECUTION_SOURCE_REVISION_SCHEMA",
    "EXECUTION_SOURCE_REVISION_SCHEMA_VERSION",
    "EditingV2ExecutionSourceRevisionError",
    "build_execution_source_revision",
    "validate_execution_source_revision",
]
