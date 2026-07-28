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
import json
from pathlib import Path

import numpy as np

from compose_v4.rewrite.action_codec import decode_action
from compose_v4.rewrite.kernel import canonical_state_key, de_novo_rewrite_system
from compose_v4.rewrite.progress import PowerSurvivalScheduler, TraceProgressCTMC
from compose_v4.rewrite.trace import RewriteStep, RewriteTrace
from compose_v4.rewrite.trace_shard import decode_state, encode_state

PACKED_STORE_SCHEMA = "compose.data.packed_trace"
PACKED_STORE_SCHEMA_VERSION = 1


class PackedStoreError(RuntimeError):
    """The packed store is unusable or was built under a different contract."""


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


def write_packed_shard(
    path: Path, entries: list[dict], *, provenance: dict
) -> dict:
    """Write a packed shard plus a sidecar manifest recording the contract it was built under.

    The manifest is not decoration: a store built under different capability flags enumerates different
    candidate families, and pairing it with a differently-configured trainer is the exact failure mode
    that produced "teacher outside exact dynamic candidates" on the eval path. ``read_packed_shard``
    refuses a mismatch rather than training on it.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt") as handle:
        for entry in entries:
            handle.write(json.dumps(entry, sort_keys=True) + "\n")
    manifest = {
        "schema": PACKED_STORE_SCHEMA,
        "schema_version": PACKED_STORE_SCHEMA_VERSION,
        "entries": len(entries),
        "states": sum(len(entry["states"]) for entry in entries),
        "provenance": dict(provenance),
    }
    Path(str(path) + ".manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def read_packed_shard(
    path: Path,
    *,
    expected_provenance: dict | None = None,
    verify_fraction: float = 0.0,
    verify_seed: int = 0,
):
    """Yield ``(trace, PackedTraceProgress)`` pairs, refusing an off-contract store.

    ``verify_fraction`` replays that fraction of entries through the real executor and asserts they
    reproduce the stored states. 0.0 (default) is the fast production path; a small positive value gives
    a sampled integrity audit without paying full re-execution.
    """
    if not 0.0 <= verify_fraction <= 1.0:
        raise ValueError("verify_fraction must lie in [0, 1]")
    path = Path(path)
    manifest_path = Path(str(path) + ".manifest.json")
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
    if expected_provenance:
        got = manifest.get("provenance") or {}
        for key, value in expected_provenance.items():
            if got.get(key) != value:
                raise PackedStoreError(
                    f"packed store provenance mismatch on {key}: {got.get(key)!r} != {value!r}"
                )
    rng = np.random.default_rng(verify_seed)
    with gzip.open(path, "rt") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            states = [decode_state(payload) for payload in entry["states"]]
            trace = decode_packed_trace(entry["trace"], states)
            if verify_fraction and rng.random() < verify_fraction:
                replay_verify(entry["trace"], states)
            yield trace, PackedTraceProgress(trace, states)


def build_packed_entry(trace_record: dict, path: TraceProgressCTMC) -> dict:
    """One packed store row: the original trace record plus its materialized states."""
    return {"trace": trace_record, "states": pack_path(path)}
