"""Build the packed trace-state store from the validated audit shards (derivative, not truth).

    validated audit shards  ->  N parallel packers  ->  reduce + verify  ->  PACK_COMPLETE

Why this exists: loading the corpus by replaying traces costs ~21 ms/corruption-trace, because
``decode_trace_record`` replays every step through the executor unconditionally and
``TraceProgressCTMC.__init__`` replays a second time -- ~32 RDKit round-trips per record per replay,
re-proving validity the build already certified. Materializing the states once offline drops the load to
0.217 ms/trace (measured, 98x) for ~0.13 GB.

Two-level artifact discipline: the audit shards under ``edit_precompile_v1/`` remain the TRUTH; this store
is a DERIVATIVE. Every packed shard records its source shard's ``content_sha256``, so a rebuilt source
invalidates its packed store rather than silently pairing stale states with new traces. Nothing here may
declare success on its own -- only the reducer writes PACK_COMPLETE, and only after re-verifying a sample
by real executor replay.

    modal run --detach modal_apps/pack_trace_states_app.py --source-subdir edit_precompile_v1
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parent.parent
REMOTE_ROOT = Path("/root/compose")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.4.0", "numpy==1.26.4", "scipy==1.13.1", "networkx==3.3", "rdkit==2024.3.5")
    .env({
        "PYTHONPATH": os.pathsep.join((str(REMOTE_ROOT / "src"), str(REMOTE_ROOT / "scripts"))),
        "PYTHONUNBUFFERED": "1",
        "OMP_NUM_THREADS": "1",
    })
    .add_local_dir(ROOT / "src", str(REMOTE_ROOT / "src"), copy=True,
                   ignore=("**/__pycache__/**", "**/*.pyc"))
    .add_local_dir(ROOT / "scripts", str(REMOTE_ROOT / "scripts"), copy=True,
                   ignore=("**/__pycache__/**", "**/*.pyc"))
)

app = modal.App("compose-v4-pack-trace-states")
artifact_volume = modal.Volume.from_name("compose-v4-artifacts", create_if_missing=True)

LAYERS = ("corruption", "cycle_ops")
PARTITIONS = ("train", "validation", "test")
_MAX_CONTAINERS = 48
# A sampled replay audit inside the reducer: cheap (0.47 ms/trace at 2%) and it is the only thing that
# re-proves the stored states actually come from the recorded actions.
_VERIFY_FRACTION = 0.02


def _pack_status(subdir: str, status: str, **fields) -> dict:
    """Explicit terminal state: RUNNING | PARTIAL | FAILED | COMPLETE. Only the reducer may say COMPLETE."""
    payload = {"status": status, **fields}
    out = Path("/artifacts") / subdir
    out.mkdir(parents=True, exist_ok=True)
    (out / "pack_status.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    artifact_volume.commit()
    print(json.dumps({"phase": "status", **{k: v for k, v in payload.items() if k != "shards"}},
                     sort_keys=True)[:900], flush=True)
    return payload


def _shard_keys(source_root: Path) -> list[tuple[str, str, str]]:
    """(layer, partition, shard_filename) for every audit shard present."""
    keys = []
    for layer in LAYERS:
        for partition in PARTITIONS:
            directory = source_root / layer / partition
            if not directory.is_dir():
                continue
            for shard in sorted(directory.glob("*.jsonl.gz")):
                keys.append((layer, partition, shard.name))
    return keys


@app.function(image=image, cpu=8.0, timeout=4 * 3600, max_containers=_MAX_CONTAINERS,
              volumes={"/artifacts": artifact_volume})
def pack_shard(source_subdir: str, dest_subdir: str, layer: str, partition: str, name: str) -> dict:
    """Materialize every progress state of one audit shard. Idempotent: an up-to-date store is reused."""
    import sys
    sys.path.insert(0, str(REMOTE_ROOT / "src"))
    sys.path.insert(0, str(REMOTE_ROOT / "scripts"))
    import time

    from compose_v4.data.packed_trace_store import (
        assert_closed_form_applies,
        build_packed_entry,
        source_shard_fingerprint,
        write_packed_shard,
    )
    from compose_v4.rewrite.progress import TraceProgressCTMC
    from compose_v4.rewrite.trace_shard import decode_trace_record, read_shard
    from ring_core_identity import recompute_operator_registry_hash

    assert_closed_form_applies()
    artifact_volume.reload()
    source = Path("/artifacts") / source_subdir / layer / partition / name
    dest = Path("/artifacts") / dest_subdir / layer / partition / name
    fingerprint = source_shard_fingerprint(source)
    provenance = {
        "source_shard": f"{layer}/{partition}/{name}",
        "source_content_sha256": fingerprint,
        "operator_registry_hash": recompute_operator_registry_hash(),
    }

    manifest_path = Path(str(dest) + ".manifest.json")
    if dest.exists() and manifest_path.exists():
        existing = json.loads(manifest_path.read_text()).get("provenance", {})
        if existing.get("source_content_sha256") == fingerprint:
            print(json.dumps({"phase": "reuse", "shard": provenance["source_shard"]}), flush=True)
            return {"shard": provenance["source_shard"], "reused": True,
                    "entries": json.loads(manifest_path.read_text())["entries"]}

    started = time.time()
    entries = []
    for record in read_shard(source):
        # Replay ONCE here, offline, so the training path never has to.
        trace = decode_trace_record(record, validate=True)
        entries.append(build_packed_entry(record, TraceProgressCTMC(trace)))
    manifest = write_packed_shard(dest, entries, provenance=provenance)
    artifact_volume.commit()
    result = {"shard": provenance["source_shard"], "reused": False,
              "entries": manifest["entries"], "states": manifest["states"],
              "seconds": round(time.time() - started, 1)}
    print(json.dumps({"phase": "packed", **result}, sort_keys=True), flush=True)
    return result


@app.function(image=image, cpu=16.0, timeout=4 * 3600, volumes={"/artifacts": artifact_volume})
def reduce_and_verify(source_subdir: str, dest_subdir: str, expected: list, build_meta: dict) -> dict:
    """Refuse to declare COMPLETE unless every shard is present, on-contract, and passes a replay audit."""
    import sys
    sys.path.insert(0, str(REMOTE_ROOT / "src"))
    sys.path.insert(0, str(REMOTE_ROOT / "scripts"))

    from compose_v4.data.packed_trace_store import read_packed_shard, source_shard_fingerprint

    artifact_volume.reload()
    source_root = Path("/artifacts") / source_subdir
    dest_root = Path("/artifacts") / dest_subdir

    missing, stale, totals = [], [], {"entries": 0, "states": 0}
    per_layer: dict[str, int] = {}
    for layer, partition, name in (tuple(item) for item in expected):
        packed = dest_root / layer / partition / name
        manifest_path = Path(str(packed) + ".manifest.json")
        if not packed.exists() or not manifest_path.exists():
            missing.append(f"{layer}/{partition}/{name}")
            continue
        manifest = json.loads(manifest_path.read_text())
        want = source_shard_fingerprint(source_root / layer / partition / name)
        if manifest.get("provenance", {}).get("source_content_sha256") != want:
            stale.append(f"{layer}/{partition}/{name}")
            continue
        totals["entries"] += manifest["entries"]
        totals["states"] += manifest["states"]
        per_layer[layer] = per_layer.get(layer, 0) + manifest["entries"]

    if missing or stale:
        return _pack_status(dest_subdir, "PARTIAL", missing=missing, stale=stale, totals=totals)

    # Sampled replay audit -- the only check that re-proves the stored states follow from the actions.
    audited = 0
    try:
        for layer, partition, name in (tuple(item) for item in expected[: min(len(expected), 12)]):
            for _ in read_packed_shard(
                dest_root / layer / partition / name, verify_fraction=_VERIFY_FRACTION
            ):
                audited += 1
    except Exception as exc:  # noqa: BLE001 -- a failed audit must be a terminal FAILED, not a warning
        return _pack_status(dest_subdir, "FAILED", reason=f"replay audit failed: {exc}", totals=totals)

    payload = {
        "PACK_COMPLETE": True,
        "expected_shards": len(expected),
        "totals": totals,
        "entries_by_layer": per_layer,
        "verify_fraction": _VERIFY_FRACTION,
        "audited_entries": audited,
        "source_subdir": source_subdir,
        **build_meta,
    }
    (dest_root / "PACK_COMPLETE.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    artifact_volume.commit()
    _pack_status(dest_subdir, "COMPLETE", totals=totals, expected_shards=len(expected))
    return payload


@app.function(image=image, cpu=2.0, timeout=12 * 3600, volumes={"/artifacts": artifact_volume})
def driver(source_subdir: str, dest_subdir: str, commit: str, gate_sha: str) -> dict:
    """Runs REMOTELY so --detach survives: a local_entrypoint would exit and never issue the fan-out."""
    artifact_volume.reload()
    source_root = Path("/artifacts") / source_subdir
    if not (source_root / "BUILD_COMPLETE.json").exists():
        return _pack_status(
            dest_subdir, "FAILED",
            reason=f"{source_subdir} has no BUILD_COMPLETE.json; the audit corpus is not complete",
        )
    keys = _shard_keys(source_root)
    if not keys:
        return _pack_status(dest_subdir, "FAILED", reason="no audit shards found")

    build_meta = {"authorization": {"launch_commit": commit, "gate_sha256": gate_sha}}
    _pack_status(dest_subdir, "RUNNING", expected_shards=len(keys), **build_meta)
    results = list(pack_shard.starmap(
        [(source_subdir, dest_subdir, layer, partition, name) for layer, partition, name in keys]
    ))
    print(json.dumps({"phase": "map_done", "shards": len(results),
                      "reused": sum(1 for r in results if r.get("reused"))}), flush=True)
    return reduce_and_verify.remote(
        source_subdir, dest_subdir, [list(k) for k in keys], build_meta
    )


@app.local_entrypoint()
def main(
    source_subdir: str = "edit_precompile_v1",
    dest_subdir: str = "edit_packed_v1",
    commit: str = "",
    gate_sha: str = "",
):
    call = driver.spawn(source_subdir, dest_subdir, commit, gate_sha)
    print(json.dumps({
        "phase": "launched", "driver_call_id": call.object_id,
        "source_subdir": source_subdir, "dest_subdir": dest_subdir,
        "commit": commit, "gate_sha256": gate_sha,
        "poll": f"modal volume get compose-v4-artifacts {dest_subdir}/pack_status.json -",
    }, indent=2, sort_keys=True))
