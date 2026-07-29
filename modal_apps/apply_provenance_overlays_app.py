"""Apply provenance overlays to the immutable corruption/cycle packed shards.

The corruption/cycle packed shards were written by a packer that recorded the sampler contract but not
``codec_implementation_hash`` or the explicit codec/packed schema versions; the MMP shards record all of
it. That asymmetry is why the corpus satisfies BENCHMARK_CONTRACT but not SCIENTIFIC_TRAINING_CONTRACT.

This writes a SIDECAR per shard rather than repacking. The shards stay byte-identical -- they are already
certified by replay, and rewriting them purely to add metadata would discard that certification and cost
hours. Each overlay binds the shard's content hash and its original manifest hash, so it is void the
moment either changes.

    modal run modal_apps/apply_provenance_overlays_app.py --commit <sha>
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

app = modal.App("compose-v4-apply-provenance-overlays")
artifact_volume = modal.Volume.from_name("compose-v4-artifacts", create_if_missing=True)

LAYERS = ("corruption", "cycle_ops")
PARTITIONS = ("train", "validation", "test")
_SENTINEL_ENTRIES = 4


@app.function(image=image, cpu=16.0, memory=65536, timeout=4 * 3600,
              volumes={"/artifacts": artifact_volume})
def apply_overlays(packed_root: str, commit: str, mmp_root: str = "") -> dict:
    import sys

    sys.path.insert(0, str(REMOTE_ROOT / "src"))
    sys.path.insert(0, str(REMOTE_ROOT / "scripts"))
    from compose_v4.data.packed_trace_store import (
        PACKED_STORE_SCHEMA_VERSION,
        manifest_path_for,
        sentinel_replay_check,
    )
    from compose_v4.data.provenance_overlay import (
        build_overlay,
        load_overlay,
        overlay_path_for,
        tensorization_implementation_hash,
    )
    from compose_v4.rewrite.action_codec import codec_implementation_hash
    from compose_v4.rewrite.trace_shard import TRACE_SCHEMA_VERSION

    artifact_volume.reload()
    root = Path(packed_root)
    fields = {
        "codec_implementation_hash": codec_implementation_hash(),
        "trace_schema_version": TRACE_SCHEMA_VERSION,
        "packed_store_schema_version": PACKED_STORE_SCHEMA_VERSION,
        "tensorization_implementation_hash": tensorization_implementation_hash(),
    }

    # (layer_dir, root) pairs. The MMP store lives at its own root with partitions directly beneath it,
    # and although its packer recorded codec/schema provenance it predates tensorization_implementation_hash
    # -- so it needs an overlay for that field just as corruption/cycle need one for the codec fields.
    targets = [(layer, root) for layer in LAYERS]
    if mmp_root:
        targets.append(("", Path(mmp_root)))

    written, reused, shards = 0, 0, []
    for layer, layer_root in targets:
        for partition in PARTITIONS:
            directory = (layer_root / layer / partition) if layer else (layer_root / partition)
            if not directory.is_dir():
                continue
            for shard in sorted(directory.glob("*.jsonl.gz")):
                manifest_path = manifest_path_for(shard)
                if not manifest_path.exists():
                    return {"status": "FAILED", "reason": f"{shard} has no manifest"}
                if load_overlay(shard, manifest_path) is not None:
                    reused += 1
                    continue
                # Certify by REPLAY at overlay time: the overlay asserts provenance, so it must carry
                # evidence that this exact shard still reproduces its stored states.
                sentinel = sentinel_replay_check(shard, entries=_SENTINEL_ENTRIES)
                overlay = build_overlay(
                    shard, manifest_path, fields=fields, packer_commit=commit,
                    certification={
                        "sentinel_entries_replayed": sentinel["sentinel_entries_replayed"],
                        "replay_verified": True,
                        "certified_at_overlay_time": True,
                    },
                )
                overlay_path_for(shard).write_text(json.dumps(overlay, indent=2, sort_keys=True) + "\n")
                written += 1
                shards.append(f"{layer or 'mmp_analogue'}/{partition}/{shard.name}")

    artifact_volume.commit()
    result = {"status": "COMPLETE", "overlays_written": written, "overlays_reused": reused,
              "fields": fields, "packer_commit": commit, "shards": shards[:5],
              "total_shards": written + reused}
    print(json.dumps({k: v for k, v in result.items() if k != "shards"}, sort_keys=True), flush=True)
    return result


@app.local_entrypoint()
def main(packed_root: str = "/artifacts/edit_packed_v1", commit: str = "",
         mmp_root: str = "/artifacts/mmp_packed_v1"):
    print(json.dumps(apply_overlays.remote(packed_root, commit, mmp_root), indent=2, sort_keys=True))
