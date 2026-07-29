"""Frozen exclusion overlay for teacher steps the production factorization cannot score.

The production factorization supports atom insertion with ZERO existing neighbours (``grow_root``) and
with EXACTLY ONE (``grow_connected``). An insertion connecting two or more existing atoms -- vertex
subdivision, the inverse of deleting a bridging atom -- has no head, and reaches
``_teacher_action_score`` as ``ValueError: factorized grow supports one existing neighbor``.

Why an overlay rather than a runtime rule
-----------------------------------------
Dropping such records at runtime "because they fail" is open-ended: a later model or enumerator change
would silently drop a growing fraction of the corpus while every count still looked healthy, and the
manifest identity would keep claiming a corpus that is no longer being trained on. So exclusions are
FROZEN into a versioned artifact:

  * the loader may omit ONLY records listed in the overlay, and counts them;
  * any unsupported teacher NOT listed fails loudly.

That converts "silently shrinking corpus" into "loud failure the first time support changes".

Excluding a trace removes BOTH directions of the transformation: a trace is drawn as a unit, and keeping
one direction while discarding its unsupported inverse would bias the family supervision asymmetrically.
The lost supervision is recorded explicitly.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

OVERLAY_SCHEMA = "compose.data.representability_overlay"
OVERLAY_SCHEMA_VERSION = 1

# Bump when the SUPPORT RULE changes, not merely when the excluded set changes.
REPRESENTABILITY_FILTER = "factorized_teacher_support_v2"
REASON_MULTI_NEIGHBOUR_INSERT = "MULTI_NEIGHBOUR_ATOM_INSERT_UNSUPPORTED"


class RepresentabilityViolation(RuntimeError):
    """An unsupported teacher was found that the frozen overlay does not list."""


def filter_implementation_hash() -> str:
    """Hash of this module, so a changed support rule invalidates every existing overlay."""
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:16]


def unsupported_steps(trace) -> list[dict]:
    """Every step of ``trace`` the production factorization cannot score, with its payload.

    Returns an empty list for a fully representable trace. The payload is recorded so an exclusion can be
    audited later without re-deriving it from the shard.
    """
    found = []
    for index, step in enumerate(trace.steps):
        neighbors = getattr(step.action, "neighbors", None)
        if neighbors is not None and len(neighbors) > 1:
            found.append({
                "step_index": index,
                "family": getattr(step, "rule_name", "atom_insert"),
                "reason": REASON_MULTI_NEIGHBOUR_INSERT,
                "payload": {"neighbors": [list(map(int, n)) for n in neighbors]},
            })
    return found


def trace_key(trace, record: dict | None = None) -> str:
    """Stable identity for an excluded trace: its recorded id when present, else a content digest."""
    if record:
        for field in ("trace_id", "row_id"):
            value = (record.get("metadata") or {}).get(field, record.get(field))
            if value is not None:
                return str(value)
    digest = hashlib.sha256()
    for step in trace.steps:
        digest.update(str(getattr(step, "rule_name", "")).encode())
        digest.update(str(getattr(step.action, "neighbors", "")).encode())
    return "sha:" + digest.hexdigest()[:24]


def build_overlay(exclusions: list[dict], *, counts: dict, enumerator_hash: str,
                  packed_manifest_hashes: dict) -> dict:
    """Freeze the census result into the artifact the loader is allowed to honour."""
    ordered = sorted(exclusions, key=lambda e: (e["layer"], e["partition"], e["trace_key"]))
    digest = hashlib.sha256()
    for entry in ordered:
        digest.update(f"{entry['layer']}/{entry['partition']}/{entry['trace_key']}".encode())
    for layer in sorted(counts):
        digest.update(f"{layer}:{counts[layer]['accepted']}".encode())
    return {
        "schema": OVERLAY_SCHEMA,
        "schema_version": OVERLAY_SCHEMA_VERSION,
        "representability_filter": REPRESENTABILITY_FILTER,
        "filter_implementation_hash": filter_implementation_hash(),
        "candidate_enumerator_hash": enumerator_hash,
        "packed_manifest_hashes": dict(packed_manifest_hashes),
        "exclusions": ordered,
        "counts": dict(counts),
        "effective_corpus_checksum": digest.hexdigest()[:16],
    }


def load_overlay(path: Path) -> dict:
    overlay = json.loads(Path(path).read_text())
    if overlay.get("schema") != OVERLAY_SCHEMA:
        raise RepresentabilityViolation(f"unexpected overlay schema: {overlay.get('schema')!r}")
    if overlay.get("schema_version") != OVERLAY_SCHEMA_VERSION:
        raise RepresentabilityViolation(
            f"overlay schema version {overlay.get('schema_version')!r} != {OVERLAY_SCHEMA_VERSION}"
        )
    if overlay.get("filter_implementation_hash") != filter_implementation_hash():
        raise RepresentabilityViolation(
            "representability filter changed since this overlay was frozen; re-run the census "
            f"(overlay {overlay.get('filter_implementation_hash')} != current "
            f"{filter_implementation_hash()})"
        )
    return overlay


def excluded_keys(overlay: dict, layer: str, partition: str) -> set[str]:
    return {
        entry["trace_key"]
        for entry in overlay.get("exclusions", ())
        if entry["layer"] == layer and entry["partition"] == partition
    }


def check_unlisted(trace, key: str, allowed: set[str], *, layer: str, partition: str) -> bool:
    """True if this trace is an ALLOWED exclusion; raise if it is unsupported but unlisted.

    The loud failure is the point: it means production support changed and the frozen overlay no longer
    describes the corpus, so the run must stop rather than quietly train on less data.
    """
    problems = unsupported_steps(trace)
    if not problems:
        return False
    if key in allowed:
        return True
    raise RepresentabilityViolation(
        f"unsupported teacher NOT listed in the frozen overlay: {layer}/{partition} trace {key}: "
        f"{problems}. Production factorization support changed, or the census is stale -- re-run the "
        "full representability census and refreeze the overlay."
    )
