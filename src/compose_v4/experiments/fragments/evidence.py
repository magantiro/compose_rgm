"""Validate the explicit fragment evidence index without loading ML dependencies.

Historical paths inside the source artifacts are provenance, never executable
input paths. Only repository-relative paths in the manifest are opened.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class EvidenceError(ValueError):
    """A source identity, population, or scientific invariant failed."""


@dataclass(frozen=True)
class Evidence:
    root: Path
    manifest_path: Path
    manifest: dict[str, Any]
    artifacts: dict[str, dict[str, Any]]


ARTIFACT_SCHEMAS = {
    "motif_selection": "fragment_motif_locked_prefix_v1",
    "decoration_selection": "fragment_locked_panel_selection_replay_v1",
    "linker_selection": "fragment_locked_panel_selection_replay_v1",
    "superstructure_learned": "fragment_superstructure_official_v2_result_v1",
    "superstructure_uniform": "fragment_superstructure_official_v2_result_v1",
    "superstructure_comparison": "fragment_superstructure_reference_final_v1",
    "motif_intervals": "fragment_common_panel_prompt_ci_v1",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise EvidenceError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _nonfinite(value: str) -> None:
    raise EvidenceError(f"non-finite JSON number: {value}")


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"), object_pairs_hook=_pairs, parse_constant=_nonfinite
        )
    except (OSError, UnicodeError, ValueError) as exc:
        raise EvidenceError(f"{path}: {exc}") from exc
    if not isinstance(value, dict):
        raise EvidenceError(f"{path}: expected a JSON object")
    return value


def contained_path(root: Path, relative: str) -> Path:
    if not isinstance(relative, str) or not relative:
        raise EvidenceError(f"invalid repository-relative path: {relative!r}")
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts:
        raise EvidenceError(f"path must stay inside the evidence root: {relative}")
    resolved = (root / path).resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise EvidenceError(f"symlink escapes the evidence root: {relative}")
    return resolved


def load_evidence(
    root: Path, manifest_relative: str = "experiments/fragments/manifest.json"
) -> Evidence:
    root = root.resolve()
    manifest_path = contained_path(root, manifest_relative)
    manifest = read_json(manifest_path)
    if manifest.get("schema") != "compose_fragment_evidence_manifest_v1":
        raise EvidenceError(f"{manifest_path}: unsupported manifest schema")
    if manifest.get("scope") != "saved_metric_reduction_only":
        raise EvidenceError(f"{manifest_path}: this reader does not support generation")
    specs = manifest.get("artifacts")
    if not isinstance(specs, dict) or set(specs) != set(ARTIFACT_SCHEMAS):
        raise EvidenceError(f"{manifest_path}: incomplete or unknown artifact roles")
    artifacts = {}
    seen_paths: set[Path] = set()
    for name, schema in ARTIFACT_SCHEMAS.items():
        spec = specs[name]
        if not isinstance(spec, dict) or not {"path", "sha256", "schema"} <= spec.keys():
            raise EvidenceError(f"{manifest_path}: invalid artifact specification for {name}")
        path = contained_path(root, spec["path"])
        expected = spec["sha256"]
        if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected):
            raise EvidenceError(f"{path}: invalid SHA-256 for {name}")
        if path in seen_paths:
            raise EvidenceError(f"{path}: reused for two different artifact roles")
        seen_paths.add(path)
        if not path.is_file():
            raise EvidenceError(f"{path}: missing {name}; expected SHA-256 {expected}")
        actual = sha256(path)
        if actual != expected:
            raise EvidenceError(f"{path}: SHA-256 mismatch; expected {expected}, found {actual}")
        value = read_json(path)
        if value.get("schema") != schema or spec["schema"] != schema:
            raise EvidenceError(f"{path}: expected schema {schema}, found {value.get('schema')!r}")
        artifacts[name] = value
    return Evidence(root, manifest_path, manifest, artifacts)
