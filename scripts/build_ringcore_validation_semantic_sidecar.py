#!/usr/bin/env python3
"""Freeze the exact RingCore-V1 validation semantic-cell sidecar."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

from compose_v4.data.packed_trace_store import read_addressed_packed_shard
from compose_v4.experiments.cnof_conditional import PathRecord
from compose_v4.experiments.ringcore_semantic_axes import (
    semantic_axis_labeler_contract_sha256,
)
from compose_v4.experiments.ringcore_semantic_sidecar import (
    SemanticSidecarConfig,
    SemanticSidecarError,
    SemanticSidecarProvenance,
    SemanticSidecarSourceShard,
    build_semantic_cell_sidecar,
    semantic_axis_labeler_source_sha256,
    write_semantic_cell_sidecar,
)

_CENSUS_SCHEMA = "compose.local_audit.current_labeler_full_validation_census"
_LAYER_TO_ADDRESS = {
    "general_corruption": "corruption",
    "cycle_operations": "cycle_ops",
    "mmp_analogue": "mmp_analogue",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def _stable_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _resolve_transfer_path(
    recorded: str,
    *,
    transfer_root: Path,
) -> Path:
    path = Path(recorded)
    if path.is_file():
        return path
    parts = path.parts
    for anchor in ("edit_packed_v1", "mmp_packed_v1"):
        if anchor in parts:
            candidate = transfer_root.joinpath(*parts[parts.index(anchor) :])
            if candidate.is_file():
                return candidate
    raise SemanticSidecarError(f"frozen transfer input is absent: {recorded}")


def _require_file_hash(path: Path, expected: str, *, name: str) -> None:
    observed = _sha256(path)
    if observed != expected:
        raise SemanticSidecarError(f"{name} hash mismatch: {observed} != {expected}")


def _load_frozen_inputs(
    *,
    census_path: Path,
    transfer_root: Path,
) -> tuple[
    dict[str, Any],
    SemanticSidecarConfig,
    SemanticSidecarProvenance,
    tuple[Path, ...],
]:
    census_bytes = census_path.read_bytes()
    census = json.loads(census_bytes)
    if census.get("schema") != _CENSUS_SCHEMA or census.get("schema_version") != 1:
        raise SemanticSidecarError("unexpected source semantic-census schema")
    labeler = census.get("labeler")
    source = census.get("source_provenance")
    aggregate = census.get("aggregate_census")
    if not all(isinstance(item, Mapping) for item in (labeler, source, aggregate)):
        raise SemanticSidecarError("source semantic census is incomplete")
    aggregate_body = {key: value for key, value in aggregate.items() if key != "census_sha256"}
    if _stable_sha256(aggregate_body) != aggregate.get("census_sha256"):
        raise SemanticSidecarError("source aggregate census self-hash mismatch")
    if (
        labeler.get("contract_sha256") != semantic_axis_labeler_contract_sha256()
        or labeler.get("source_sha256") != semantic_axis_labeler_source_sha256()
    ):
        raise SemanticSidecarError("source census does not bind the current semantic labeler")
    exclusions = source.get("representability_validation_exclusions")
    if exclusions != []:
        raise SemanticSidecarError(
            "this freezer requires explicit entry-index exclusions when the "
            "validation overlay excludes traces"
        )

    unified = transfer_root / "UNIFIED_PACKED_MANIFEST.json"
    representability = transfer_root / "REPRESENTABILITY_OVERLAY.json"
    _require_file_hash(
        unified,
        source["unified_packed_manifest_sha256"],
        name="unified packed manifest",
    )
    _require_file_hash(
        representability,
        source["representability_overlay_sha256"],
        name="representability overlay",
    )

    shard_specs: list[SemanticSidecarSourceShard] = []
    shard_paths: list[Path] = []
    for item in source.get("shards") or ():
        if not isinstance(item, Mapping):
            raise SemanticSidecarError("source shard inventory is malformed")
        packed_path = _resolve_transfer_path(
            str(item["packed_shard_path"]),
            transfer_root=transfer_root,
        )
        manifest_path = _resolve_transfer_path(
            str(item["manifest_path"]),
            transfer_root=transfer_root,
        )
        overlay_path = _resolve_transfer_path(
            str(item["provenance_overlay_path"]),
            transfer_root=transfer_root,
        )
        _require_file_hash(
            packed_path,
            str(item["packed_shard_sha256"]),
            name=f"packed shard {packed_path.name}",
        )
        _require_file_hash(
            manifest_path,
            str(item["manifest_sha256"]),
            name=f"packed manifest {packed_path.name}",
        )
        _require_file_hash(
            overlay_path,
            str(item["provenance_overlay_sha256"]),
            name=f"packed provenance overlay {packed_path.name}",
        )
        normalized_layer = str(item["layer"])
        try:
            address_layer = _LAYER_TO_ADDRESS[normalized_layer]
        except KeyError:
            raise SemanticSidecarError(
                f"unknown frozen validation layer {normalized_layer!r}"
            ) from None
        decoded_entries = int(item["decoded_entries"])
        decoded_states = int(item["decoded_states"])
        if (
            int(item["manifest_entries"]) != decoded_entries
            or int(item["manifest_states"]) != decoded_states
            or item["partition"] != "validation"
        ):
            raise SemanticSidecarError("source shard decoded census disagrees with its manifest")
        shard_specs.append(
            SemanticSidecarSourceShard(
                packed_shard_content_sha256=str(item["packed_shard_sha256"]),
                packed_shard_name=packed_path.name,
                layer=address_layer,
                partition="validation",
                packed_manifest_sha256=str(item["manifest_sha256"]),
                provenance_overlay_sha256=str(item["provenance_overlay_sha256"]),
                packed_entry_count=decoded_entries,
                effective_trace_count=decoded_entries,
                progress_row_count=decoded_states,
            )
        )
        shard_paths.append(packed_path)

    config = SemanticSidecarConfig(
        maximum_nonempty_cells=int(aggregate["maximum_nonempty_cells"]),
        expected_semantic_census_sha256=str(aggregate["census_sha256"]),
    )
    provenance = SemanticSidecarProvenance(
        source_census_artifact_sha256=hashlib.sha256(census_bytes).hexdigest(),
        unified_packed_manifest_sha256=str(source["unified_packed_manifest_sha256"]),
        representability_overlay_sha256=str(source["representability_overlay_sha256"]),
        labeler_source_sha256=str(labeler["source_sha256"]),
        source_shards=tuple(shard_specs),
    )
    return census, config, provenance, tuple(shard_paths)


def _records(shard_paths: tuple[Path, ...]) -> Iterator[PathRecord]:
    for shard_path in shard_paths:
        for addressed in read_addressed_packed_shard(
            shard_path,
            verify_fraction=0.0,
        ):
            yield PathRecord(
                target_key=addressed.address.target_key,
                path=addressed.path,
                corpus_address=addressed.address,
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--census", type=Path, required=True)
    parser.add_argument("--transfer-root", type=Path, required=True)
    parser.add_argument("--output-sidecar", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    args = parser.parse_args()

    census, config, provenance, shard_paths = _load_frozen_inputs(
        census_path=args.census,
        transfer_root=args.transfer_root,
    )
    artifact = build_semantic_cell_sidecar(
        _records(shard_paths),
        config=config,
        provenance=provenance,
    )
    if artifact.semantic_census != census["aggregate_census"]:
        raise SemanticSidecarError(
            "computed semantic census payload differs from the frozen census"
        )
    write_semantic_cell_sidecar(
        args.output_sidecar,
        args.output_manifest,
        artifact,
    )
    print(
        json.dumps(
            {
                "status": "FROZEN",
                "training_authorized": False,
                "sidecar_path": str(args.output_sidecar),
                "manifest_path": str(args.output_manifest),
                "manifest_sha256": artifact.manifest_sha256,
                "config_sha256": artifact.config.sha256,
                "provenance_sha256": artifact.provenance.sha256,
                "source_sha256": artifact.source_sha256,
                "sidecar_file_sha256": artifact.manifest()["sidecar"]["file_sha256"],
                "counts": artifact.manifest()["counts"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
