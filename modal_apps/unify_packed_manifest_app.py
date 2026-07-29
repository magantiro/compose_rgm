"""Emit the unified three-layer packed manifest on the artifact volume.

The manifest logic lives in ``scripts/build_unified_packed_manifest.py`` and is unit-tested there; this app
only runs it where the artifacts are. It refuses to write unless all three layers are COMPLETE and every
shared contract hash agrees, so the trainer can be handed one explicit file instead of discovering shards
by walking directories.

    modal run modal_apps/unify_packed_manifest_app.py
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

app = modal.App("compose-v4-unify-packed-manifest")
artifact_volume = modal.Volume.from_name("compose-v4-artifacts", create_if_missing=True)


@app.function(image=image, cpu=8.0, memory=32768, timeout=2 * 3600,
              volumes={"/artifacts": artifact_volume})
def unify(packed_root: str, mmp_root: str, out: str) -> dict:
    import sys

    sys.path.insert(0, str(REMOTE_ROOT / "src"))
    sys.path.insert(0, str(REMOTE_ROOT / "scripts"))
    from build_unified_packed_manifest import build

    artifact_volume.reload()
    manifest = build(Path(packed_root), Path(mmp_root))
    Path(out).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    artifact_volume.commit()
    print(json.dumps({"phase": "unified", "totals": manifest["totals"],
                      "checksum": manifest["manifest_checksum"]}, sort_keys=True), flush=True)
    return manifest


@app.local_entrypoint()
def main(
    packed_root: str = "/artifacts/edit_packed_v1",
    mmp_root: str = "/artifacts/mmp_packed_v1",
    out: str = "/artifacts/UNIFIED_PACKED_MANIFEST.json",
):
    manifest = unify.remote(packed_root, mmp_root, out)
    print(json.dumps({
        "totals": manifest["totals"],
        "counts": manifest["counts"],
        "layer_weights": manifest["layer_weights"],
        "shared_contract": manifest["shared_contract"],
        "manifest_checksum": manifest["manifest_checksum"],
    }, indent=2, sort_keys=True))
