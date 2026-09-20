"""Provenance overlays + the two enforcement contracts.

The corruption/cycle packed shards were written by an earlier packer that did not record
``codec_implementation_hash`` or the explicit codec/packed schema versions; the MMP shards do. Repacking
tens of gigabytes purely to add metadata would be wasteful AND would destroy the property that makes the
shards trustworthy -- that they are immutable artifacts already certified by replay.

So the missing provenance is supplied as a deterministic OVERLAY: a sidecar that binds the observed shard
to the fields it lacks, and is only valid while the shard's content hash is unchanged. The shard stays
byte-identical; the overlay carries the claim and the evidence for it.

Two enforcement levels:

  BENCHMARK_CONTRACT
      complete packed data, source hashes, operator/capability/scheduler/sampler agreement, explicit
      roots, zero fallback. Enough to trust a THROUGHPUT measurement whose checkpoint is discarded.

  SCIENTIFIC_TRAINING_CONTRACT
      additionally requires complete codec, schema and tensorization provenance for EVERY layer.
      A number that goes in a paper has to be reproducible from the artifact alone.

A throughput benchmark may run under the former. The training launcher must refuse anything short of the
latter -- an unreproducible corpus is a worse failure than a slow one.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

BENCHMARK_CONTRACT = "BENCHMARK_CONTRACT"
SCIENTIFIC_TRAINING_CONTRACT = "SCIENTIFIC_TRAINING_CONTRACT"

OVERLAY_SCHEMA = "compose.data.provenance_overlay"
OVERLAY_SCHEMA_VERSION = 1

# Required by BOTH contracts.
_BENCHMARK_FIELDS = (
    "operator_registry_hash",
    "scheduler_type",
    "scheduler_power",
    "progress_sampler_version",
    "time_sampling_implementation_hash",
)
# Additionally required for scientific training.
_SCIENTIFIC_FIELDS = (
    "codec_implementation_hash",
    "trace_schema_version",
    "packed_store_schema_version",
    "tensorization_implementation_hash",
)


class ContractViolation(RuntimeError):
    """The corpus does not satisfy the requested enforcement contract."""


def overlay_path_for(shard_path: Path) -> Path:
    return Path(str(shard_path) + ".provenance.json")


# The code that turns a packed record into model tensors. Two corpora built under different
# tensorization are not interchangeable even when every other hash agrees, so it is part of the
# scientific contract.
_TENSORIZATION_SOURCES = (
    "src/compose_v4/experiments/factorized_mark_conditional.py",
    "src/compose_v4/model/factorized_tracelet_rate_model.py",
    "src/compose_v4/chem/state.py",
    "src/compose_v4/data/charge_policy.py",
)


def tensorization_implementation_hash() -> str:
    """Hash of the collation/tensorization sources that turn packed records into model tensors."""
    repo = Path(__file__).resolve().parents[3]
    digest = hashlib.sha256()
    for rel in _TENSORIZATION_SOURCES:
        source = repo / rel
        digest.update(rel.encode())
        digest.update(source.read_bytes() if source.exists() else b"<MISSING>")
    return digest.hexdigest()[:16]


def upgrade_implementation_hash() -> str:
    """Hash of THIS module's source, so a changed overlay rule invalidates existing overlays."""
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:16]


def shard_content_sha256(shard_path: Path) -> str:
    digest = hashlib.sha256()
    with open(shard_path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_overlay(
    shard_path: Path,
    manifest_path: Path,
    *,
    fields: dict,
    packer_commit: str,
    certification: dict,
) -> dict:
    """Bind an immutable packed shard to the provenance fields its own manifest lacks.

    The overlay is only meaningful together with the exact bytes it was computed over, so it carries both
    the shard content hash and the original manifest hash. Either changing invalidates it.
    """
    missing = [key for key in _SCIENTIFIC_FIELDS if key not in fields]
    if missing:
        raise ContractViolation(f"overlay is missing required fields: {missing}")
    return {
        "schema": OVERLAY_SCHEMA,
        "schema_version": OVERLAY_SCHEMA_VERSION,
        "shard": shard_path.name,
        "packed_shard_content_sha256": shard_content_sha256(shard_path),
        "original_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "fields": dict(fields),
        "packer_commit": packer_commit,
        "upgrade_implementation_hash": upgrade_implementation_hash(),
        "certification": dict(certification),
    }


def load_overlay(shard_path: Path, manifest_path: Path) -> dict | None:
    """Return a VALID overlay, or None. An overlay whose bindings no longer hold is not returned."""
    path = overlay_path_for(shard_path)
    if not path.exists():
        return None
    try:
        overlay = json.loads(path.read_text())
    except json.JSONDecodeError:
        return None
    if overlay.get("schema") != OVERLAY_SCHEMA:
        return None
    if overlay.get("schema_version") != OVERLAY_SCHEMA_VERSION:
        return None
    if overlay.get("upgrade_implementation_hash") != upgrade_implementation_hash():
        return None
    if overlay.get("packed_shard_content_sha256") != shard_content_sha256(shard_path):
        return None
    if overlay.get("original_manifest_sha256") != hashlib.sha256(manifest_path.read_bytes()).hexdigest():
        return None
    return overlay


def effective_provenance(manifest: dict, overlay: dict | None) -> dict:
    """Manifest fields, plus overlay fields where the manifest is silent.

    An overlay may only ADD. If it disagrees with a field the manifest already records, that is a
    contradiction about the same artifact and must fail rather than be resolved by precedence.
    """
    merged = {**(manifest.get("provenance") or {}), **(manifest.get("sampler_contract") or {})}
    if overlay:
        for key, value in (overlay.get("fields") or {}).items():
            if key in merged and merged[key] != value:
                raise ContractViolation(
                    f"overlay contradicts the manifest on {key}: {merged[key]!r} != {value!r}"
                )
            merged.setdefault(key, value)
    return merged


def check_contract(layer_provenances: dict, *, level: str) -> dict:
    """Verify every layer satisfies ``level``; raise ``ContractViolation`` with what is missing.

    ``layer_provenances`` maps layer name -> effective provenance dict.
    """
    if level not in (BENCHMARK_CONTRACT, SCIENTIFIC_TRAINING_CONTRACT):
        raise ValueError(f"unknown contract level: {level!r}")
    required = _BENCHMARK_FIELDS + (
        _SCIENTIFIC_FIELDS if level == SCIENTIFIC_TRAINING_CONTRACT else ()
    )

    missing_by_layer = {}
    for layer, provenance in layer_provenances.items():
        missing = [key for key in required if provenance.get(key) is None]
        if missing:
            missing_by_layer[layer] = missing
    if missing_by_layer:
        raise ContractViolation(
            f"{level} not satisfied; missing provenance by layer: {missing_by_layer}"
        )

    # every layer must agree on the shared fields, not merely possess them
    disagreements = {}
    reference_layer = next(iter(layer_provenances))
    reference = layer_provenances[reference_layer]
    for layer, provenance in layer_provenances.items():
        for key in required:
            if provenance[key] != reference[key]:
                disagreements[key] = {reference_layer: reference[key], layer: provenance[key]}
    if disagreements:
        raise ContractViolation(f"{level} not satisfied; layers disagree: {disagreements}")

    return {"level": level, "layers": sorted(layer_provenances), "verified_fields": list(required)}
