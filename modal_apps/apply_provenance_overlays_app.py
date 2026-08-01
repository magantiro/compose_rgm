"""Apply provenance overlays to the immutable corruption/cycle packed shards.

The corruption/cycle packed shards were written by a packer that recorded the sampler contract but not
``codec_implementation_hash`` or the explicit codec/packed schema versions; the MMP shards record all of
it. That asymmetry is why the corpus satisfies BENCHMARK_CONTRACT but not SCIENTIFIC_TRAINING_CONTRACT.

This writes a SIDECAR per shard rather than repacking. The shards stay byte-identical -- they are already
certified by replay, and rewriting them purely to add metadata would discard that certification and cost
hours. Each overlay binds the shard's content hash and its original manifest hash, so it is void the
moment either changes.

    modal run modal_apps/apply_provenance_overlays_app.py \
      --commit <sha> \
      --packed-root <audit-packed-root> \
      --mmp-root /artifacts/mmp_packed_v2_f942fc2315cf9210f4c7 \
      --audit-pack-completion-sha256 <sha256> \
      --mmp-pack-completion-sha256 <sha256>
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parent.parent
REMOTE_ROOT = Path("/root/compose")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.4.0",
        "numpy==1.26.4",
        "scipy==1.13.1",
        "networkx==3.3",
        "rdkit==2024.3.5",
    )
    .env(
        {
            "PYTHONPATH": os.pathsep.join((str(REMOTE_ROOT / "src"), str(REMOTE_ROOT / "scripts"))),
            "PYTHONUNBUFFERED": "1",
            "OMP_NUM_THREADS": "1",
        }
    )
    .add_local_dir(
        ROOT / "src",
        str(REMOTE_ROOT / "src"),
        copy=True,
        ignore=("**/__pycache__/**", "**/*.pyc"),
    )
    .add_local_dir(
        ROOT / "scripts",
        str(REMOTE_ROOT / "scripts"),
        copy=True,
        ignore=("**/__pycache__/**", "**/*.pyc"),
    )
)

app = modal.App("compose-v4-apply-provenance-overlays")
artifact_volume = modal.Volume.from_name("compose-v4-artifacts", create_if_missing=False)

LAYERS = ("corruption", "cycle_ops")
PARTITIONS = ("train", "validation", "test")
AUDIT_PACK_COMPLETION_FILENAME = "PACK_COMPLETE.json"
MMP_PACK_COMPLETION_FILENAME = "MMP_PACK_COMPLETE.json"
FROZEN_MMP_V2_ROOT = "/artifacts/mmp_packed_v2_f942fc2315cf9210f4c7"
_SENTINEL_ENTRIES = 4
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
OVERLAY_COMPLETION_SCHEMA = "compose.editing_v2_upstream_overlay_completion"
OVERLAY_COMPLETION_SCHEMA_VERSION = 1
OVERLAY_COMPLETION_STATUS = "COMPLETE_IMMUTABLE_BYTE_BINDINGS_NO_TRAINING_AUTHORITY"
_UPSTREAM_COMPLETION_FIELDS = {
    "path",
    "file_sha256",
    "completion_flag",
    "inventory_origin",
    "expected_shards",
    "shard_inventory_sha256",
}


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _require_clean_launch(commit: str) -> None:
    """Refuse to serialize a dirty or differently committed source tree."""

    if _COMMIT_RE.fullmatch(commit) is None:
        raise RuntimeError("commit must be a full lowercase 40-character Git SHA")
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--porcelain=v1", "--untracked-files=normal"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if head != commit or dirty:
        raise RuntimeError(
            "provenance-overlay launch requires the exact clean committed source tree"
        )


def _verified_pack_completion(
    *,
    root: Path,
    filename: str,
    completion_flag: str,
    expected_file_sha256: str,
    mmp: bool,
) -> tuple[dict, list[dict]]:
    """Reconcile one exact upstream pack receipt against every live shard byte."""

    if _SHA256_RE.fullmatch(expected_file_sha256) is None:
        raise RuntimeError(f"expected SHA-256 for {completion_flag} must be supplied explicitly")
    completion_path = root / filename
    if not completion_path.is_file():
        raise RuntimeError(f"required upstream completion is absent: {completion_path}")
    observed_file_sha256 = _file_sha256(completion_path)
    if observed_file_sha256 != expected_file_sha256:
        raise RuntimeError(f"upstream completion bytes disagree for {completion_path}")
    try:
        completion = json.loads(completion_path.read_bytes())
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"upstream completion is invalid JSON: {completion_path}") from error
    if not isinstance(completion, dict) or completion.get(completion_flag) is not True:
        raise RuntimeError(f"{completion_flag} is not true in {completion_path}")
    expected_shards = completion.get("expected_shards")
    if type(expected_shards) is not int or expected_shards <= 0:
        raise RuntimeError(f"{completion_flag}.expected_shards must be positive")

    actual_paths: set[Path] = set()
    layers = ("mmp_analogue",) if mmp else LAYERS
    for layer in layers:
        for partition in PARTITIONS:
            directory = root / partition if mmp else root / layer / partition
            if not directory.is_dir():
                raise RuntimeError(f"required upstream packed directory is absent: {directory}")
            actual_paths.update(directory.glob("*.jsonl.gz"))
    if len(actual_paths) != expected_shards:
        raise RuntimeError(f"live shard count disagrees with {completion_flag}.expected_shards")

    rows = completion.get("shard_artifacts")
    inventory_origin = "completion_shard_artifacts"
    if rows is None and not mmp:
        # The immutable first-generation audit receipt predates per-shard
        # inventory rows. Reconstruct those identities from the exact live
        # bytes, then reconcile counts to its frozen replay-certified totals.
        inventory_origin = "reconciled_legacy_completion_totals"
        rows = []
        for shard in sorted(actual_paths):
            relative = shard.relative_to(root)
            layer, partition = relative.parts[:2]
            manifest_path = shard.with_suffix(".manifest.json")
            if not manifest_path.is_file():
                raise RuntimeError(f"legacy audit shard lacks its packed manifest: {shard}")
            try:
                manifest = json.loads(manifest_path.read_bytes())
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise RuntimeError(
                    f"legacy audit packed manifest is invalid JSON: {manifest_path}"
                ) from error
            rows.append(
                {
                    "layer": layer,
                    "partition": partition,
                    "shard": shard.name,
                    "packed_sha256": _file_sha256(shard),
                    "manifest_sha256": _file_sha256(manifest_path),
                    "entries": manifest.get("entries"),
                    "states": manifest.get("states"),
                }
            )
    if not isinstance(rows, list) or not rows or len(rows) != expected_shards:
        raise RuntimeError(f"{completion_flag} has no exact, reconciled shard inventory")

    normalized_inventory: list[dict] = []
    source_rows: list[dict] = []
    expected_paths: set[Path] = set()
    seen_cells_and_names: set[tuple[str, str, str]] = set()
    required_fields = {
        "partition",
        "shard",
        "packed_sha256",
        "manifest_sha256",
        "entries",
        "states",
    }
    if not mmp:
        required_fields.add("layer")
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or set(row) != required_fields:
            raise RuntimeError(f"{completion_flag}.shard_artifacts[{index}] fields disagree")
        layer = "mmp_analogue" if mmp else row["layer"]
        partition = row["partition"]
        shard_name = row["shard"]
        if (
            layer not in {*LAYERS, "mmp_analogue"}
            or partition not in PARTITIONS
            or not isinstance(shard_name, str)
            or not shard_name.endswith(".jsonl.gz")
            or Path(shard_name).name != shard_name
        ):
            raise RuntimeError(f"{completion_flag}.shard_artifacts[{index}] address is invalid")
        for field in ("packed_sha256", "manifest_sha256"):
            if _SHA256_RE.fullmatch(str(row[field])) is None:
                raise RuntimeError(f"{completion_flag}.shard_artifacts[{index}].{field} is invalid")
        for field in ("entries", "states"):
            if type(row[field]) is not int or row[field] <= 0:
                raise RuntimeError(
                    f"{completion_flag}.shard_artifacts[{index}].{field} must be positive"
                )
        cell_and_name = (layer, partition, shard_name)
        if cell_and_name in seen_cells_and_names:
            raise RuntimeError(f"{completion_flag} repeats a physical shard")
        seen_cells_and_names.add(cell_and_name)
        shard = root / partition / shard_name if mmp else root / layer / partition / shard_name
        manifest_path = shard.with_suffix(".manifest.json")
        if not shard.is_file() or not manifest_path.is_file():
            raise RuntimeError(f"upstream completion references absent shard bytes: {shard}")
        if (
            _file_sha256(shard) != row["packed_sha256"]
            or _file_sha256(manifest_path) != row["manifest_sha256"]
        ):
            raise RuntimeError(f"upstream completion shard bytes disagree: {shard}")
        expected_paths.add(shard)
        normalized_inventory.append(
            {
                "layer": layer,
                "partition": partition,
                "shard": shard_name,
                "packed_sha256": row["packed_sha256"],
                "manifest_sha256": row["manifest_sha256"],
                "entries": row["entries"],
                "states": row["states"],
            }
        )
        source_rows.append(
            {
                "layer": layer,
                "partition": partition,
                "shard_path": str(shard),
                "shard_file_sha256": row["packed_sha256"],
                "manifest_path": str(manifest_path),
                "manifest_file_sha256": row["manifest_sha256"],
                "entries": row["entries"],
                "states": row["states"],
            }
        )

    if actual_paths != expected_paths:
        raise RuntimeError(f"live shard inventory disagrees with {completion_flag}")
    normalized_inventory.sort(key=lambda item: (item["layer"], item["partition"], item["shard"]))
    source_rows.sort(key=lambda item: (item["layer"], item["partition"], item["shard_path"]))
    entries = sum(item["entries"] for item in normalized_inventory)
    states = sum(item["states"] for item in normalized_inventory)
    if mmp:
        if completion.get("packed_entries") != entries or completion.get("packed_states") != states:
            raise RuntimeError("MMP pack completion totals disagree with its exact shard inventory")
    else:
        entries_by_layer = {
            layer: sum(item["entries"] for item in normalized_inventory if item["layer"] == layer)
            for layer in LAYERS
        }
        if completion.get("totals") != {"entries": entries, "states": states} or (
            completion.get("entries_by_layer") != entries_by_layer
        ):
            raise RuntimeError(
                "audit pack completion totals disagree with its exact shard inventory"
            )
    identity = {
        "path": str(completion_path),
        "file_sha256": observed_file_sha256,
        "completion_flag": completion_flag,
        "inventory_origin": inventory_origin,
        "expected_shards": expected_shards,
        "shard_inventory_sha256": _canonical_sha256(normalized_inventory),
    }
    return identity, source_rows


def build_overlay_completion(
    *,
    commit: str,
    launcher_source_sha256: str,
    packed_root: str,
    mmp_root: str,
    upstream_completions: dict,
    new_overlay_fields: dict,
    shard_receipts: list[dict],
) -> dict:
    """Build the retry-invariant global receipt for exact overlay bytes."""

    if _COMMIT_RE.fullmatch(commit) is None:
        raise ValueError("commit must be a full lowercase 40-character Git SHA")
    if _SHA256_RE.fullmatch(launcher_source_sha256) is None:
        raise ValueError("launcher_source_sha256 must be a full lowercase SHA-256")
    if not packed_root.startswith("/artifacts/") or mmp_root != FROZEN_MMP_V2_ROOT:
        raise ValueError("packed roots must be absolute paths below /artifacts")
    if not isinstance(upstream_completions, dict) or set(upstream_completions) != {
        "audit_pack",
        "mmp_pack",
    }:
        raise ValueError("upstream_completions must bind audit_pack and mmp_pack")
    normalized_upstream: dict[str, dict] = {}
    for name in ("audit_pack", "mmp_pack"):
        identity = upstream_completions[name]
        if not isinstance(identity, dict) or set(identity) != _UPSTREAM_COMPLETION_FIELDS:
            raise ValueError(f"upstream_completions.{name} fields disagree")
        if _SHA256_RE.fullmatch(str(identity["file_sha256"])) is None or (
            _SHA256_RE.fullmatch(str(identity["shard_inventory_sha256"])) is None
        ):
            raise ValueError(f"upstream_completions.{name} SHA-256 is invalid")
        if type(identity["expected_shards"]) is not int or identity["expected_shards"] <= 0:
            raise ValueError(f"upstream_completions.{name}.expected_shards must be positive")
        normalized_upstream[name] = dict(identity)
    expected_completion_addresses = {
        "audit_pack": (
            f"{packed_root}/{AUDIT_PACK_COMPLETION_FILENAME}",
            "PACK_COMPLETE",
            "reconciled_legacy_completion_totals",
        ),
        "mmp_pack": (
            f"{mmp_root}/{MMP_PACK_COMPLETION_FILENAME}",
            "MMP_PACK_COMPLETE",
            "completion_shard_artifacts",
        ),
    }
    for name, (path, flag, origin) in expected_completion_addresses.items():
        if (
            normalized_upstream[name]["path"] != path
            or normalized_upstream[name]["completion_flag"] != flag
            or normalized_upstream[name]["inventory_origin"] != origin
        ):
            raise ValueError(f"upstream_completions.{name} address or completion flag disagrees")
    if not shard_receipts:
        raise ValueError("overlay completion requires at least one physical shard")
    required_receipt_fields = {
        "layer",
        "partition",
        "shard_path",
        "shard_file_sha256",
        "manifest_path",
        "manifest_file_sha256",
        "entries",
        "states",
        "overlay_path",
        "overlay_file_sha256",
        "overlay_fields_sha256",
        "effective_provenance_sha256",
    }
    normalized_receipts = []
    for index, receipt in enumerate(shard_receipts):
        if not isinstance(receipt, dict) or set(receipt) != required_receipt_fields:
            raise ValueError(f"shard_receipts[{index}] fields disagree")
        for field in (
            "shard_file_sha256",
            "manifest_file_sha256",
            "overlay_file_sha256",
            "overlay_fields_sha256",
            "effective_provenance_sha256",
        ):
            if _SHA256_RE.fullmatch(str(receipt[field])) is None:
                raise ValueError(f"shard_receipts[{index}].{field} is invalid")
        for field in ("entries", "states"):
            if type(receipt[field]) is not int or receipt[field] <= 0:
                raise ValueError(f"shard_receipts[{index}].{field} must be positive")
        layer = receipt["layer"]
        partition = receipt["partition"]
        shard_path = Path(str(receipt["shard_path"]))
        if (
            layer not in {*LAYERS, "mmp_analogue"}
            or partition not in PARTITIONS
            or shard_path.name == ""
            or not shard_path.name.endswith(".jsonl.gz")
        ):
            raise ValueError(f"shard_receipts[{index}] address is invalid")
        expected_shard = (
            Path(mmp_root) / partition / shard_path.name
            if layer == "mmp_analogue"
            else Path(packed_root) / layer / partition / shard_path.name
        )
        if (
            shard_path != expected_shard
            or Path(str(receipt["manifest_path"])) != expected_shard.with_suffix(".manifest.json")
            or receipt["overlay_path"] != f"{expected_shard}.provenance.json"
        ):
            raise ValueError(f"shard_receipts[{index}] path is outside its exact upstream cell")
        normalized_receipts.append(dict(receipt))
    normalized_receipts.sort(
        key=lambda item: (item["layer"], item["partition"], item["shard_path"])
    )
    shard_paths = [item["shard_path"] for item in normalized_receipts]
    if len(shard_paths) != len(set(shard_paths)):
        raise ValueError("overlay completion contains duplicate physical shards")
    expected_cells = {(layer, partition) for layer in LAYERS for partition in PARTITIONS}
    if mmp_root:
        expected_cells.update(("mmp_analogue", partition) for partition in PARTITIONS)
    observed_cells = {(item["layer"], item["partition"]) for item in normalized_receipts}
    if observed_cells != expected_cells:
        raise ValueError("overlay completion does not cover every required layer/partition cell")
    audit_shards = sum(item["layer"] in LAYERS for item in normalized_receipts)
    mmp_shards = sum(item["layer"] == "mmp_analogue" for item in normalized_receipts)
    if (
        audit_shards != normalized_upstream["audit_pack"]["expected_shards"]
        or mmp_shards != normalized_upstream["mmp_pack"]["expected_shards"]
    ):
        raise ValueError("overlay shard inventory count disagrees with upstream completions")
    for name, layers in (
        ("audit_pack", set(LAYERS)),
        ("mmp_pack", {"mmp_analogue"}),
    ):
        inventory = sorted(
            (
                {
                    "layer": item["layer"],
                    "partition": item["partition"],
                    "shard": Path(item["shard_path"]).name,
                    "packed_sha256": item["shard_file_sha256"],
                    "manifest_sha256": item["manifest_file_sha256"],
                    "entries": item["entries"],
                    "states": item["states"],
                }
                for item in normalized_receipts
                if item["layer"] in layers
            ),
            key=lambda item: (item["layer"], item["partition"], item["shard"]),
        )
        if _canonical_sha256(inventory) != normalized_upstream[name]["shard_inventory_sha256"]:
            raise ValueError(f"overlay shard inventory disagrees with upstream_completions.{name}")
    body = {
        "schema": OVERLAY_COMPLETION_SCHEMA,
        "schema_version": OVERLAY_COMPLETION_SCHEMA_VERSION,
        "status": OVERLAY_COMPLETION_STATUS,
        "training_authorized": False,
        "commit": commit,
        "launcher_source_sha256": launcher_source_sha256,
        "packed_root": packed_root,
        "mmp_root": mmp_root,
        "upstream_completions": normalized_upstream,
        "fields_for_new_overlays": dict(new_overlay_fields),
        "sentinel_entries_per_new_overlay": _SENTINEL_ENTRIES,
        "shards": normalized_receipts,
    }
    return {**body, "completion_sha256": _canonical_sha256(body)}


@app.function(
    image=image,
    cpu=16.0,
    memory=65536,
    timeout=4 * 3600,
    volumes={"/artifacts": artifact_volume},
)
def apply_overlays(
    packed_root: str,
    commit: str,
    mmp_root: str,
    *,
    launcher_source_sha256: str,
    expected_audit_pack_completion_sha256: str,
    expected_mmp_pack_completion_sha256: str,
) -> dict:
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
        effective_provenance,
        load_overlay,
        overlay_path_for,
        tensorization_implementation_hash,
    )
    from compose_v4.rewrite.action_codec import codec_implementation_hash
    from compose_v4.rewrite.trace_shard import TRACE_SCHEMA_VERSION

    artifact_volume.reload()
    if mmp_root != FROZEN_MMP_V2_ROOT:
        raise RuntimeError("editing-v2 provenance overlays require the frozen MMP V2 root")
    root = Path(packed_root)
    mmp_path = Path(mmp_root)
    audit_completion, audit_sources = _verified_pack_completion(
        root=root,
        filename=AUDIT_PACK_COMPLETION_FILENAME,
        completion_flag="PACK_COMPLETE",
        expected_file_sha256=expected_audit_pack_completion_sha256,
        mmp=False,
    )
    mmp_completion, mmp_sources = _verified_pack_completion(
        root=mmp_path,
        filename=MMP_PACK_COMPLETION_FILENAME,
        completion_flag="MMP_PACK_COMPLETE",
        expected_file_sha256=expected_mmp_pack_completion_sha256,
        mmp=True,
    )
    source_shards = sorted(
        [*audit_sources, *mmp_sources],
        key=lambda item: (item["layer"], item["partition"], item["shard_path"]),
    )
    fields = {
        "codec_implementation_hash": codec_implementation_hash(),
        "trace_schema_version": TRACE_SCHEMA_VERSION,
        "packed_store_schema_version": PACKED_STORE_SCHEMA_VERSION,
        "tensorization_implementation_hash": tensorization_implementation_hash(),
    }

    written, reused, shard_receipts = 0, 0, []
    for source in source_shards:
        shard = Path(source["shard_path"])
        manifest_path = Path(source["manifest_path"])
        if manifest_path_for(shard) != manifest_path:
            raise RuntimeError(f"upstream manifest path is nonstandard: {manifest_path}")
        overlay_path = overlay_path_for(shard)
        existing = load_overlay(shard, manifest_path)
        if existing is not None:
            reused += 1
            overlay = existing
        else:
            if overlay_path.exists():
                raise RuntimeError(
                    f"existing provenance overlay is invalid under the bound implementation: "
                    f"{overlay_path}"
                )
            # Certify by REPLAY at overlay time: the overlay asserts provenance, so it must carry
            # evidence that this exact shard still reproduces its stored states.
            sentinel = sentinel_replay_check(shard, entries=_SENTINEL_ENTRIES)
            overlay = build_overlay(
                shard,
                manifest_path,
                fields=fields,
                packer_commit=commit,
                certification={
                    "sentinel_entries_replayed": sentinel["sentinel_entries_replayed"],
                    "replay_verified": True,
                    "certified_at_overlay_time": True,
                },
            )
            encoded = json.dumps(overlay, indent=2, sort_keys=True).encode("utf-8") + b"\n"
            with overlay_path.open("xb") as handle:
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
            written += 1
        manifest = json.loads(manifest_path.read_bytes())
        # Historical overlays remain immutable. Record their actual fields and
        # effective provenance instead of relabelling them as newly generated.
        effective = effective_provenance(manifest, overlay)
        shard_receipts.append(
            {
                **source,
                "overlay_path": str(overlay_path),
                "overlay_file_sha256": _file_sha256(overlay_path),
                "overlay_fields_sha256": _canonical_sha256(overlay.get("fields") or {}),
                "effective_provenance_sha256": _canonical_sha256(effective),
            }
        )

    completion = build_overlay_completion(
        commit=commit,
        launcher_source_sha256=launcher_source_sha256,
        packed_root=packed_root,
        mmp_root=mmp_root,
        upstream_completions={
            "audit_pack": audit_completion,
            "mmp_pack": mmp_completion,
        },
        new_overlay_fields=fields,
        shard_receipts=shard_receipts,
    )
    completion_dir = (
        Path("/artifacts/editing_v2/upstream_overlays") / completion["completion_sha256"]
    )
    completion_dir.mkdir(parents=True, exist_ok=True)
    completion_path = completion_dir / "OVERLAY_COMPLETION.json"
    encoded_completion = json.dumps(completion, indent=2, sort_keys=True).encode("utf-8") + b"\n"
    if completion_path.exists():
        if completion_path.read_bytes() != encoded_completion:
            raise RuntimeError("existing overlay completion receipt has different bytes")
    else:
        with completion_path.open("xb") as handle:
            handle.write(encoded_completion)
            handle.flush()
            os.fsync(handle.fileno())
    artifact_volume.commit()
    result = {
        "status": "COMPLETE",
        "overlays_written": written,
        "overlays_reused": reused,
        "fields": fields,
        "packer_commit": commit,
        "total_shards": len(shard_receipts),
        "completion_path": str(completion_path),
        "completion_file_sha256": _file_sha256(completion_path),
        "completion_sha256": completion["completion_sha256"],
    }
    print(json.dumps(result, sort_keys=True), flush=True)
    return result


@app.local_entrypoint()
def main(
    commit: str,
    packed_root: str,
    mmp_root: str,
    audit_pack_completion_sha256: str,
    mmp_pack_completion_sha256: str,
):
    _require_clean_launch(commit)
    if mmp_root != FROZEN_MMP_V2_ROOT:
        raise RuntimeError(
            f"mmp_root must be the frozen Editing V2 MMP corpus root: {FROZEN_MMP_V2_ROOT}"
        )
    print(
        json.dumps(
            apply_overlays.remote(
                packed_root,
                commit,
                mmp_root,
                launcher_source_sha256=_file_sha256(Path(__file__)),
                expected_audit_pack_completion_sha256=(audit_pack_completion_sha256),
                expected_mmp_pack_completion_sha256=mmp_pack_completion_sha256,
            ),
            indent=2,
            sort_keys=True,
        )
    )
