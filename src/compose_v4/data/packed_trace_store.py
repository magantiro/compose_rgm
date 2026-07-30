"""Pre-materialized trace states: the same CTMC, without the executor replay.

Measured on real shards, loading the corpus costs ~22 ms per corruption trace, and a profile attributes
almost all of it to ``molecular_graph_to_smiles`` / ``is_rdkit_valid`` -- roughly 32 RDKit round-trips per
record, run TWICE (``decode_trace_record`` replays, then ``TraceProgressCTMC.__init__`` replays again).
That work re-proves validity the build already established and certified in the shard manifest.

``validate=False`` does not help (measured 1.1x): the validity checks live inside the executor's
``apply``, not in the decoder's optional verification. The only way to avoid them without altering audited
executor semantics is to not replay at all -- so the states are materialized once, offline, and stored.

``PackedTraceProgress`` subclasses ``TraceProgressCTMC`` and overrides ONLY the replaying constructor.
Every other method -- ``state_at``, ``marginal``, ``sample_progress``, ``operational_jump_rate``,
``jump_rate``, ``sample`` -- is inherited unchanged, so it is literally the same code rather than a
parallel implementation that could drift. That is deliberate: the equivalence risk here is exactly the
"my reference is a copy" failure, and inheritance removes it by construction.

Measured against replay (per trace): corruption 22.45 -> 0.037 ms, cycle_ops 2.67 -> 0.018 ms,
mmp 0.091 ms; full train partition ~0.63 min instead of hours, at ~102 B/trace gzipped.
"""
from __future__ import annotations

import gzip
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from compose_v4.data.provenance_overlay import (
    load_overlay,
    overlay_path_for,
    shard_content_sha256,
)
from compose_v4.rewrite.action_codec import decode_action
from compose_v4.rewrite.kernel import canonical_state_key, de_novo_rewrite_system
from compose_v4.rewrite.progress import PowerSurvivalScheduler, TraceProgressCTMC
from compose_v4.rewrite.trace import RewriteStep, RewriteTrace
from compose_v4.rewrite.trace_shard import decode_state, encode_state

PACKED_STORE_SCHEMA = "compose.data.packed_trace"
PACKED_STORE_SCHEMA_VERSION = 1

# The packed progress sampler is a CLOSED FORM valid only under the unit-power scheduler: it substitutes
# alpha for t because alpha(t) = 1 - (1-t)^1 = t. A future scheduler change must invalidate this store (or
# force it through a generalized path) rather than silently reuse unit-power logic, so the scheduler
# identity travels with the artifact and is checked on load.
PROGRESS_SAMPLER_VERSION = 1
_SAMPLER_IMPLEMENTATION_SOURCES = (
    "src/compose_v4/data/packed_edit_cache.py",
    "src/compose_v4/rewrite/progress.py",
    "src/compose_v4/experiments/tracelet_conditional.py",
)


class PackedStoreError(RuntimeError):
    """The packed store is unusable or was built under a different contract."""


@dataclass(frozen=True)
class PackedTraceAddress:
    """Immutable identity of one trace inside one exact packed-shard artifact.

    ``entry_index`` is the zero-based nonblank JSONL row index in the packed
    shard. It is deliberately not the record's position in a concatenated
    training corpus: representability exclusions and layer concatenation may
    change that position, while this address must continue to identify the
    original immutable row.
    """

    packed_shard_content_sha256: str
    packed_shard_name: str
    entry_index: int
    trace_id: str
    layer: str
    partition: str
    source_key: str
    target_key: str
    path_length: int

    def __post_init__(self) -> None:
        digest = self.packed_shard_content_sha256
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError(
                "packed_shard_content_sha256 must be a lowercase SHA-256 digest"
            )
        if (
            not isinstance(self.packed_shard_name, str)
            or not self.packed_shard_name
            or Path(self.packed_shard_name).name != self.packed_shard_name
        ):
            raise ValueError("packed_shard_name must be one non-empty basename")
        for name in (
            "trace_id",
            "layer",
            "partition",
            "source_key",
            "target_key",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be a non-empty string")
        for name in ("entry_index", "path_length"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")


def sampler_contract() -> dict:
    """Identity of the progress/time sampling law this store's closed forms assume."""
    from compose_v4.rewrite.progress import PowerSurvivalScheduler

    scheduler = PowerSurvivalScheduler()
    repo = Path(__file__).resolve().parents[3]
    digest = hashlib.sha256()
    for rel in _SAMPLER_IMPLEMENTATION_SOURCES:
        source = repo / rel
        digest.update(rel.encode())
        digest.update(source.read_bytes() if source.exists() else b"<MISSING>")
    return {
        "scheduler_type": type(scheduler).__name__,
        "scheduler_power": float(scheduler.power),
        "time_sampling_implementation_hash": digest.hexdigest()[:16],
        "progress_sampler_version": PROGRESS_SAMPLER_VERSION,
    }


def assert_closed_form_applies() -> None:
    """The packed sampler treats alpha == t. Refuse to build or load a store when that is false."""
    from compose_v4.rewrite.progress import PowerSurvivalScheduler

    power = float(PowerSurvivalScheduler().power)
    if power != 1.0:
        raise PackedStoreError(
            f"packed progress sampling assumes alpha(t) == t (unit scheduler power), but power == {power}; "
            "the closed forms in packed_edit_cache must be generalized before this store can be used"
        )


class PackedTraceProgress(TraceProgressCTMC):
    """A ``TraceProgressCTMC`` whose states were materialized offline.

    The base constructor replays the trace through the executor to build ``_states``. Here they are
    supplied, so construction is a tuple assignment. The endpoint invariant the base class enforces is
    retained -- it is an array comparison, costs nothing, and is the property that would silently break
    if a store were ever paired with the wrong trace.
    """

    def __init__(self, trace, states, *, scheduler=None, system=None) -> None:
        states = tuple(states)
        if len(states) != len(trace.steps) + 1:
            raise PackedStoreError(
                f"packed trace has {len(states)} states for {len(trace.steps)} steps; "
                "expected one per progress position including the endpoint"
            )
        self.trace = trace
        self.scheduler = scheduler or PowerSurvivalScheduler()
        # The module-level guard only inspects the DEFAULT scheduler; an explicitly-supplied non-unit one
        # would otherwise slip past it and be used by the packed closed-form progress sampler.
        if float(self.scheduler.power) != 1.0:
            raise PackedStoreError(
                f"packed progress sampling assumes alpha(t) == t, but the supplied scheduler has "
                f"power == {self.scheduler.power}"
            )
        self.system = system  # only needed by the replaying paths, which this class never takes
        self.checkpoint_interval = None
        self._states = states
        self._checkpoints = ()
        endpoint, target = states[-1], trace.target
        if not (
            np.array_equal(endpoint.atom_types, target.atom_types)
            and np.array_equal(endpoint.bonds, target.bonds)
            and np.array_equal(endpoint.formal_charges, target.formal_charges)
        ):
            raise PackedStoreError("packed endpoint state does not equal the trace target")


@dataclass(frozen=True)
class AddressedPackedTrace:
    """One decoded packed row paired with its immutable corpus address."""

    address: PackedTraceAddress
    trace: RewriteTrace
    path: PackedTraceProgress


@dataclass(frozen=True)
class _DecodedPackedTrace:
    """Shared internal row representation for legacy and addressed readers."""

    trace: RewriteTrace
    path: PackedTraceProgress
    address: PackedTraceAddress | None


def pack_path(path: TraceProgressCTMC) -> list[dict]:
    """Materialize every progress state of a replayed path into the shard state encoding."""
    return [encode_state(path.state_at(progress)) for progress in range(path.path_length + 1)]


def unpack_path(trace, encoded_states) -> PackedTraceProgress:
    return PackedTraceProgress(trace, [decode_state(payload) for payload in encoded_states])


def decode_packed_trace(record: dict, states) -> RewriteTrace:
    """Rebuild a ``RewriteTrace`` with ZERO executor calls.

    ``trace_shard.decode_trace_record`` replays every step unconditionally -- ``validate`` gates only the
    key COMPARISONS, not the replay, because the final state is needed as ``target``. That replay is the
    dominant load cost. Here the states are already materialized, so source and target are reads and the
    steps are pure action decoding.

    The trade this makes explicit: executor drift is no longer caught by re-execution at load. It is
    caught instead by refusing a store whose ``operator_registry_hash`` / ``codec_implementation_hash``
    provenance disagrees, plus the optional ``verify_fraction`` replay audit in ``read_packed_shard``.
    Verify the contract, not the computation -- and keep a way to verify the computation on demand.
    """
    steps = []
    for entry in record["steps"]:
        rule, action = decode_action(entry["action"])
        steps.append(RewriteStep(rule, action))
    return RewriteTrace(
        source=states[0],
        target=states[-1],
        steps=tuple(steps),
        metadata=dict(record.get("metadata", {})),
    )


def replay_verify(record: dict, states) -> None:
    """Re-execute one packed entry and assert it reproduces the stored states. Audit path only."""
    system = de_novo_rewrite_system()
    state = states[0]
    for i, entry in enumerate(record["steps"]):
        rule, action = decode_action(entry["action"])
        state = system.apply(state, rule, action)
        got, want = canonical_state_key(state), canonical_state_key(states[i + 1])
        if got != want:
            raise PackedStoreError(
                f"packed store replay mismatch at step {i} ({rule}): {got} != stored {want}"
            )


def manifest_path_for(shard_path: Path) -> Path:
    """Sidecar manifest path for a shard, using the repo-wide convention.

    ``Path.with_suffix`` replaces the LAST suffix, so ``shard_0009.jsonl.gz`` -> ``shard_0009.jsonl.manifest.json``
    -- which is exactly what ``precompile_edit_data_app`` writes. Building it by string concatenation instead
    yields ``shard_0009.jsonl.gz.manifest.json`` and finds nothing. One helper, one convention.
    """
    return Path(shard_path).with_suffix(".manifest.json")


def source_shard_fingerprint(shard_path: Path) -> str:
    """The ``content_sha256`` of the audit shard a packed store derives from.

    The packed store is a DERIVATIVE: the audit shards remain the truth. Binding the derivative to its
    source's content hash is what makes that hierarchy enforceable -- a rebuilt source shard invalidates
    its packed store instead of silently pairing stale states with new traces.
    """
    manifest_path = manifest_path_for(shard_path)
    if not manifest_path.exists():
        raise PackedStoreError(f"source shard {shard_path} has no manifest to fingerprint")
    return json.loads(manifest_path.read_text())["content_sha256"]


def write_packed_shard(
    path: Path, entries: list[dict], *, provenance: dict
) -> dict:
    """Write a packed shard plus a sidecar manifest recording the contract it was built under.

    The manifest is not decoration: a store built under different capability flags enumerates different
    candidate families, and pairing it with a differently-configured trainer is the exact failure mode
    that produced "teacher outside exact dynamic candidates" on the eval path. ``read_packed_shard``
    refuses a mismatch rather than training on it. ``provenance`` should carry the source shard's
    ``content_sha256`` (see ``source_shard_fingerprint``) alongside the contract hashes.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt") as handle:
        for entry in entries:
            handle.write(json.dumps(entry, sort_keys=True) + "\n")
    assert_closed_form_applies()
    manifest = {
        "schema": PACKED_STORE_SCHEMA,
        "schema_version": PACKED_STORE_SCHEMA_VERSION,
        "sampler_contract": sampler_contract(),
        "entries": len(entries),
        "states": sum(len(entry["states"]) for entry in entries),
        "provenance": dict(provenance),
    }
    manifest_path_for(path).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def _validated_packed_manifest(
    path: Path,
    *,
    expected_provenance: dict | None = None,
) -> tuple[Path, dict]:
    """Validate one packed sidecar and return its path and decoded payload."""

    path = Path(path)
    manifest_path = manifest_path_for(path)
    if not manifest_path.exists():
        raise PackedStoreError(f"packed shard {path} has no manifest; refusing to load")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != PACKED_STORE_SCHEMA:
        raise PackedStoreError(f"unexpected packed-store schema: {manifest.get('schema')!r}")
    if manifest.get("schema_version") != PACKED_STORE_SCHEMA_VERSION:
        raise PackedStoreError(
            f"packed-store schema version {manifest.get('schema_version')!r} != "
            f"{PACKED_STORE_SCHEMA_VERSION}; rebuild the store"
        )
    assert_closed_form_applies()
    stored_contract = manifest.get("sampler_contract") or {}
    current_contract = sampler_contract()
    for key in ("scheduler_type", "scheduler_power", "progress_sampler_version"):
        if stored_contract.get(key) != current_contract[key]:
            raise PackedStoreError(
                f"sampler contract mismatch on {key}: store built under {stored_contract.get(key)!r}, "
                f"current is {current_contract[key]!r}; the packed closed-form progress law no longer "
                "applies to this store"
            )
    if expected_provenance:
        got = manifest.get("provenance") or {}
        for key, value in expected_provenance.items():
            if got.get(key) != value:
                raise PackedStoreError(
                    f"packed store provenance mismatch on {key}: {got.get(key)!r} != {value!r}"
                )
    return manifest_path, manifest


def _packed_shard_address_digest(path: Path, manifest_path: Path) -> str:
    """Return the exact packed-byte SHA, rejecting a present but invalid overlay."""

    sidecar = overlay_path_for(path)
    if not sidecar.exists():
        return shard_content_sha256(path)
    overlay = load_overlay(path, manifest_path)
    if overlay is None:
        raise PackedStoreError(
            f"packed shard {path} has an invalid provenance overlay; refusing addressed load"
        )
    digest = overlay.get("packed_shard_content_sha256")
    if not isinstance(digest, str):
        raise PackedStoreError(
            f"packed shard {path} provenance overlay lacks its content SHA-256"
        )
    return digest


def _required_address_string(record: dict, field: str) -> str:
    value = record.get(field)
    if not isinstance(value, str) or not value:
        raise PackedStoreError(
            f"packed trace address field {field!r} must be a non-empty string"
        )
    return value


def _address_for_packed_trace(
    *,
    shard_path: Path,
    shard_digest: str,
    entry_index: int,
    record: dict,
    trace: RewriteTrace,
    path: PackedTraceProgress,
    verify_endpoint_keys: bool,
) -> PackedTraceAddress:
    stored_path_length = record.get("path_length")
    if type(stored_path_length) is not int or stored_path_length < 0:
        raise PackedStoreError(
            "packed trace address field 'path_length' must be a nonnegative integer"
        )
    if stored_path_length != path.path_length:
        raise PackedStoreError(
            f"packed trace path_length {stored_path_length} != decoded {path.path_length}"
        )
    source_key = _required_address_string(record, "source_key")
    target_key = _required_address_string(record, "target_key")
    if verify_endpoint_keys:
        # Canonical molecular identity invokes RDKit. Keep it on the explicit
        # sampled audit path: doing this for every one of ~725k rows would
        # defeat the packed store's zero-RDKit production-load contract. The
        # exact shard-byte digest still binds these stored envelope values.
        decoded_source_key = canonical_state_key(trace.source)
        decoded_target_key = canonical_state_key(trace.target)
        if source_key != decoded_source_key:
            raise PackedStoreError(
                f"packed trace source_key {source_key!r} != decoded {decoded_source_key!r}"
            )
        if target_key != decoded_target_key:
            raise PackedStoreError(
                f"packed trace target_key {target_key!r} != decoded {decoded_target_key!r}"
            )
    return PackedTraceAddress(
        packed_shard_content_sha256=shard_digest,
        packed_shard_name=shard_path.name,
        entry_index=entry_index,
        trace_id=_required_address_string(record, "trace_id"),
        layer=_required_address_string(record, "layer"),
        partition=_required_address_string(record, "partition"),
        source_key=source_key,
        target_key=target_key,
        path_length=stored_path_length,
    )


def _iter_decoded_packed_shard(
    path: Path,
    *,
    expected_provenance: dict | None,
    verify_fraction: float,
    verify_seed: int,
    require_address: bool,
):
    """Shared decoder used by both public packed-shard readers."""

    if not 0.0 <= verify_fraction <= 1.0:
        raise ValueError("verify_fraction must lie in [0, 1]")
    path = Path(path)
    manifest_path, manifest = _validated_packed_manifest(
        path,
        expected_provenance=expected_provenance,
    )
    shard_digest = (
        _packed_shard_address_digest(path, manifest_path) if require_address else None
    )
    rng = np.random.default_rng(verify_seed)
    seen_trace_ids: set[str] = set()
    entry_index = 0
    with gzip.open(path, "rt") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            states = [decode_state(payload) for payload in entry["states"]]
            trace = decode_packed_trace(entry["trace"], states)
            verify_row = bool(verify_fraction and rng.random() < verify_fraction)
            if verify_row:
                replay_verify(entry["trace"], states)
            packed_path = PackedTraceProgress(trace, states)
            address = None
            if require_address:
                assert shard_digest is not None
                address = _address_for_packed_trace(
                    shard_path=path,
                    shard_digest=shard_digest,
                    entry_index=entry_index,
                    record=entry["trace"],
                    trace=trace,
                    path=packed_path,
                    verify_endpoint_keys=verify_row,
                )
                if address.trace_id in seen_trace_ids:
                    raise PackedStoreError(
                        f"duplicate trace_id {address.trace_id!r} in packed shard {path}"
                    )
                seen_trace_ids.add(address.trace_id)
            yield _DecodedPackedTrace(trace=trace, path=packed_path, address=address)
            entry_index += 1
    if require_address and manifest.get("entries") != entry_index:
        raise PackedStoreError(
            f"packed shard manifest declares {manifest.get('entries')!r} entries "
            f"but addressed reader decoded {entry_index}"
        )


def read_addressed_packed_shard(
    path: Path,
    *,
    expected_provenance: dict | None = None,
    verify_fraction: float = 0.0,
    verify_seed: int = 0,
):
    """Yield decoded rows with immutable packed-shard addresses.

    Unlike the legacy reader, this path requires complete trace-envelope
    identity fields and validates path length, row count and trace-ID
    uniqueness. The exact shard-byte digest binds the stored endpoint keys;
    rows selected by ``verify_fraction`` additionally recanonicalize endpoint
    states and compare those keys. A present provenance overlay must validate
    against the exact shard and manifest bytes.
    """

    for row in _iter_decoded_packed_shard(
        path,
        expected_provenance=expected_provenance,
        verify_fraction=verify_fraction,
        verify_seed=verify_seed,
        require_address=True,
    ):
        assert row.address is not None
        yield AddressedPackedTrace(
            address=row.address,
            trace=row.trace,
            path=row.path,
        )


def read_packed_shard(
    path: Path,
    *,
    expected_provenance: dict | None = None,
    verify_fraction: float = 0.0,
    verify_seed: int = 0,
):
    """Yield ``(trace, PackedTraceProgress)`` pairs, refusing an off-contract store.

    This legacy projection intentionally does not require the new address
    metadata, so previously readable stores and all callers retain their return
    shape and validation behavior. ``verify_fraction`` replays that fraction of
    entries through the real executor and asserts they reproduce the stored
    states.
    """

    for row in _iter_decoded_packed_shard(
        path,
        expected_provenance=expected_provenance,
        verify_fraction=verify_fraction,
        verify_seed=verify_seed,
        require_address=False,
    ):
        yield row.trace, row.path


def build_packed_entry(trace_record: dict, path: TraceProgressCTMC) -> dict:
    """One packed store row: the original trace record plus its materialized states."""
    return {"trace": trace_record, "states": pack_path(path)}


# ---- integrity policy ---------------------------------------------------------------------------------
# Production training loads with verify_fraction=0: the audit shards are authoritative, the Modal reducer
# fully verified them, and each packed shard is bound to its source's content hash. Re-executing every
# rewrite on every load would spend hours re-proving what the immutable source already certifies.
#
# What production DOES run is this fixed sentinel: a small, deterministic, first-N replay at startup. It
# is not random and not per-worker -- random replay inside dataloader workers would multiply the cost by
# the worker count, vary run to run, and still not be a proof.
SENTINEL_REPLAY_ENTRIES = 8


def sentinel_replay_check(shard_path: Path, *, entries: int = SENTINEL_REPLAY_ENTRIES) -> dict:
    """Replay the first ``entries`` packed traces and assert they reproduce their stored states.

    Deterministic by construction (first N, not sampled), so a failure is reproducible and a pass means
    the same thing on every run. Cheap enough to sit in the startup path of a multi-hour training job.
    """
    checked = 0
    with gzip.open(shard_path, "rt") as handle:
        for line in handle:
            line = line.strip()
            if not line or checked >= entries:
                break
            entry = json.loads(line)
            states = [decode_state(payload) for payload in entry["states"]]
            replay_verify(entry["trace"], states)
            checked += 1
    return {"shard": str(shard_path), "sentinel_entries_replayed": checked}
