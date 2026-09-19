"""V2 trace shards -- the persistent form of a compiled rewrite trace, built on RewriteActionCodecV2.

The historical pool format encoded a trace as ``source_smiles`` plus a list of two-family step dicts, so
corruption (seven-plus families, two with nested payloads) could not be persisted at all. This module adds
the V2 record: every step carries a fully-encoded action plus the executor rule, model family, and the
canonical key of the successor it produces, so a shard can be validated WITHOUT re-deriving anything.

Design notes:
  * A record is self-verifying. Each step stores its successor's canonical key, so a loader can replay the
    trace through the production executor and assert every intermediate matches -- catching silent executor
    or canonicalization drift at load time rather than at training time.
  * V1 (two-family MMP) records remain readable. ``load_trace_records`` dispatches on ``schema_version``,
    so the existing 363k-row MMP artifact is NOT rewritten to unblock corruption.
  * Nothing here re-implements chemistry: one executor (``RewriteSystem.apply``), one canonical successor
    (``canonical_state_key``), one action codec.
"""
from __future__ import annotations

import gzip
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterator

from compose_v4.chem.molecular_graph import (
    is_element,
    molecular_graph_to_smiles,
)
from compose_v4.rewrite.action_codec import (
    canonical_family,
    codec_implementation_hash,
    decode_action,
    encode_action,
)
from compose_v4.rewrite.kernel import canonical_state_key, de_novo_rewrite_system
from compose_v4.rewrite.trace import RewriteStep, RewriteTrace

TRACE_SCHEMA = "compose.rewrite.trace"
TRACE_SCHEMA_VERSION = 2
_SYSTEM = de_novo_rewrite_system()


class TraceShardError(ValueError):
    """A shard record failed validation. Always loud -- never repaired."""


def _cycle_rank(state) -> int:
    """Independent cycles = bonds - atoms + 1 for a connected molecule."""
    real = is_element(state.atom_types)
    idx = [i for i, ok in enumerate(real) if ok]
    bonds = sum(1 for i in idx for j in idx if i < j and int(state.bonds[i, j]))
    return bonds - len(idx) + 1


def _atom_count(state) -> int:
    return int(sum(1 for ok in is_element(state.atom_types) if ok))


def encode_state(state) -> dict:
    """Encode a padded state EXACTLY, by slot.

    A trace's actions address atoms by SLOT INDEX, so the source must be restored with its exact slot
    layout. Storing only SMILES is unsafe: measured on real corruption sources, 7 of 10 do NOT survive a
    SMILES round-trip index-exactly (RDKit re-canonicalization reorders atoms), which would silently point
    every action at a different atom. The historical MMP pool format gets away with source_smiles only
    because those traces are COMPILED FROM the re-parsed source, so their indices agree by construction;
    corruption sources come from the corpus loader's padding and do not.

    Bonds are stored as upper-triangular (i, j, order) triples -- exact and far smaller than n_slots^2.
    """
    n = len(state.atom_types)
    bonds = [
        [i, j, int(state.bonds[i, j])]
        for i in range(n) for j in range(i + 1, n) if int(state.bonds[i, j])
    ]
    return {
        "n_slots": n,
        "atom_types": [int(x) for x in state.atom_types],
        "formal_charges": [int(x) for x in state.formal_charges],
        "implicit_h_counts": [int(x) for x in state.implicit_h_counts],
        "bonds": bonds,
    }


def decode_state(payload: dict):
    """Rebuild the exact padded state written by ``encode_state``."""
    import numpy as np

    from compose_v4.chem.molecular_graph import MolecularGraph

    n = int(payload["n_slots"])
    atom_types = np.asarray(payload["atom_types"], dtype=np.int32)
    formal_charges = np.asarray(payload["formal_charges"], dtype=np.int32)
    implicit_h = np.asarray(payload["implicit_h_counts"], dtype=np.int32)
    if not (len(atom_types) == len(formal_charges) == len(implicit_h) == n):
        raise TraceShardError("state arrays disagree with n_slots")
    bonds = np.zeros((n, n), dtype=np.int32)
    for entry in payload["bonds"]:
        i, j, order = int(entry[0]), int(entry[1]), int(entry[2])
        bonds[i, j] = order
        bonds[j, i] = order
    return MolecularGraph(
        atom_types=atom_types, formal_charges=formal_charges,
        implicit_h_counts=implicit_h, bonds=bonds,
    )


def encode_trace_record(
    trace: RewriteTrace,
    *,
    n_slots: int,
    seed: int,
    trace_id: str,
    partition: str,
    layer: str = "corruption",
    direction: str = "",
    source_scaffold: str = "",
    extra: dict | None = None,
) -> dict:
    """Encode a trace into a self-verifying V2 record. Replays through the production executor so every
    stored successor key is the one the executor actually produces."""
    state = trace.source
    source_key = canonical_state_key(state)
    steps: list[dict] = []
    families: Counter = Counter()
    rules: Counter = Counter()
    for step in trace.steps:
        action_record = encode_action(step.rule_name, step.action)
        successor = _SYSTEM.apply(state, step.rule_name, step.action)
        successor_key = canonical_state_key(successor)
        steps.append({
            "action": action_record,
            "successor_key": successor_key,
            "atom_count_delta": _atom_count(successor) - _atom_count(state),
            "cycle_rank_delta": _cycle_rank(successor) - _cycle_rank(state),
        })
        families[action_record["model_family"]] += 1
        rules[step.rule_name] += 1
        state = successor
    return {
        "schema": TRACE_SCHEMA,
        "schema_version": TRACE_SCHEMA_VERSION,
        # EXACT source state (slot-addressed). source_smiles is human-readable metadata only and is
        # NOT used to rebuild the state -- see encode_state for why that would be unsafe.
        "source_state": encode_state(trace.source),
        "source_smiles": molecular_graph_to_smiles(trace.source),
        "source_key": source_key,
        "n_slots": int(n_slots),
        "steps": steps,
        "target_key": canonical_state_key(state),
        "target_smiles": molecular_graph_to_smiles(state),
        "path_length": len(trace.steps),
        "operator_histogram": dict(rules),
        "family_histogram": dict(families),
        "atom_count_delta": _atom_count(state) - _atom_count(trace.source),
        "cycle_rank_delta": _cycle_rank(state) - _cycle_rank(trace.source),
        "seed": int(seed),
        "trace_id": trace_id,
        "partition": partition,
        "layer": layer,
        "direction": direction,
        "source_scaffold": source_scaffold,
        "metadata": dict(extra or {}),
    }


def decode_trace_record(record: dict, *, validate: bool = True) -> RewriteTrace:
    """Rebuild a RewriteTrace from a V2 record. With ``validate`` (default), replays every step through the
    production executor and asserts the stored canonical successor keys match -- so executor or
    canonicalization drift fails at LOAD time, not silently during training."""
    if record.get("schema") != TRACE_SCHEMA:
        raise TraceShardError(f"unknown trace schema {record.get('schema')!r}")
    if record.get("schema_version") != TRACE_SCHEMA_VERSION:
        raise TraceShardError(f"unknown trace schema_version {record.get('schema_version')!r}")
    if "source_state" not in record:
        raise TraceShardError(
            f"trace {record.get('trace_id')!r} has no source_state. V2 records must carry the exact "
            f"slot-addressed source; rebuilding from source_smiles reorders atoms and would repoint "
            f"every action index."
        )
    source = decode_state(record["source_state"])
    if validate and canonical_state_key(source) != record["source_key"]:
        raise TraceShardError(
            f"source canonical key mismatch for trace {record.get('trace_id')!r}: the stored key does not "
            f"match the decoded source state (canonicalization drift?)"
        )
    state = source
    steps: list[RewriteStep] = []
    for i, entry in enumerate(record["steps"]):
        rule, action = decode_action(entry["action"])
        if canonical_family(rule) != entry["action"]["model_family"]:
            raise TraceShardError(f"step {i}: ontology disagreement")
        successor = _SYSTEM.apply(state, rule, action)
        if validate:
            got = canonical_state_key(successor)
            if got != entry["successor_key"]:
                raise TraceShardError(
                    f"step {i} ({rule}): replayed successor {got} != stored {entry['successor_key']}"
                )
        steps.append(RewriteStep(rule, action))
        state = successor
    if validate and canonical_state_key(state) != record["target_key"]:
        raise TraceShardError("final target canonical key mismatch")
    return RewriteTrace(
        source=source, target=state, steps=tuple(steps), metadata=dict(record.get("metadata", {}))
    )


# ---- shard IO -----------------------------------------------------------------------------------------


def write_shard(path: Path, records: list[dict]) -> dict:
    """Write gzipped JSONL. Returns shard stats incl. checksum."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(
        json.dumps(r, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n"
        for r in records
    ).encode()
    with gzip.open(path, "wb") as handle:
        handle.write(payload)
    return {
        "path": str(path),
        "records": len(records),
        # checksum over the UNCOMPRESSED payload so it is independent of gzip settings/timestamps
        "content_sha256": hashlib.sha256(payload).hexdigest(),
    }


def read_shard(path: Path) -> Iterator[dict]:
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def shard_manifest(
    *,
    shard_stats: dict,
    partition: str,
    counts: dict,
    rejections: dict,
    compiler_commit: str,
    program_contract_fingerprint: str,
    operator_registry_hash: str,
    capability_hash: str,
    extra: dict | None = None,
) -> dict:
    """Provenance a shard must carry so a cross-contract load fails loudly."""
    from compose_v4.chem import molecular_graph as _mg
    from compose_v4.rewrite import kernel as _kernel

    def _module_hash(module) -> str:
        return hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()[:16]

    return {
        "schema": TRACE_SCHEMA,
        "schema_version": TRACE_SCHEMA_VERSION,
        "codec_implementation_hash": codec_implementation_hash(),
        "program_contract_fingerprint": program_contract_fingerprint,
        "operator_registry_hash": operator_registry_hash,
        "capability_hash": capability_hash,
        # canonicalization and executor are the two functions a shard's stored keys depend on; pin both so
        # a shard compiled under different semantics fails loudly instead of training silently.
        "canonicalization_version": _module_hash(_mg),
        "executor_version": _module_hash(_kernel),
        "compiler_commit": compiler_commit,
        "source_partition": partition,
        "counts": dict(counts),
        "rejection_reasons": dict(rejections),
        **shard_stats,
        **(extra or {}),
    }


def load_trace_records(path: Path, *, validate: bool = True) -> list[Any]:
    """Version-dispatching loader: V2 shards and V1 (two-family MMP) pool rows both load through here, so
    the existing MMP artifact is never rewritten merely to unblock corruption."""
    from compose_v4.experiments.analogue_prior import rewrite_trace_from_record as _v1_trace

    out = []
    for record in read_shard(path):
        version = record.get("schema_version")
        if version == TRACE_SCHEMA_VERSION and record.get("schema") == TRACE_SCHEMA:
            out.append(decode_trace_record(record, validate=validate))
        elif version is None:
            out.append(_v1_trace(record))       # historical MMP pool row
        else:
            raise TraceShardError(f"unknown record schema/version: {record.get('schema')!r}/{version!r}")
    return out
