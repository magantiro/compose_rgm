#!/usr/bin/env python3
"""Freeze checkpoint-independent RingCore-V1 validation panel artifacts."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from compose_v4.data.production_edit_corpus import (  # noqa: E402
    load_production_edit_corpus,
)
from compose_v4.data.representability_overlay import (  # noqa: E402
    load_overlay,
)
from compose_v4.experiments.ringcore_successor_leaderboard import (  # noqa: E402
    load_json_object,
)
from compose_v4.experiments.ringcore_validation_panel import (  # noqa: E402
    FAMILY_FORENSICS_PANEL_ID,
    write_validation_panel,
)
from compose_v4.experiments.ringcore_validation_panel_builder import (  # noqa: E402
    build_family_forensics_validation_panel,
    build_production_validation_panel,
    panel_build_identity,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO / "configs" / "ringcore_v1_successor_leaderboard_v1.json",
    )
    parser.add_argument(
        "--inventory",
        type=Path,
        default=(
            REPO
            / "diagnostics"
            / "coherence"
            / "ringcore_v1_scientific_a7546e2_frozen_inventory.json"
        ),
    )
    parser.add_argument(
        "--panel-kind",
        choices=("production_law", FAMILY_FORENSICS_PANEL_ID),
        required=True,
    )
    parser.add_argument(
        "--production-stage",
        choices=("initial", "expanded"),
        default="initial",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--identity-output", type=Path, required=True)
    return parser


def _immutable_json_write(payload: object, path: Path) -> None:
    text = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(text)
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.read_text() != text:
                raise FileExistsError(f"immutable artifact already differs: {path}") from None
    finally:
        temporary.unlink(missing_ok=True)


def _validate_artifact_root(
    root: Path,
    *,
    config: dict,
) -> tuple[dict, dict]:
    unified_path = root / "UNIFIED_PACKED_MANIFEST.json"
    overlay_path = root / "REPRESENTABILITY_OVERLAY.json"
    unified = load_json_object(unified_path)
    overlay = load_overlay(overlay_path)
    validation = config["validation_data"]
    if unified.get("manifest_checksum") != validation["unified_manifest_checksum"]:
        raise SystemExit("unified packed-manifest checksum is off-protocol")
    if overlay.get("effective_corpus_checksum") != validation["representability_overlay_checksum"]:
        raise SystemExit("representability overlay checksum is off-protocol")
    if (unified.get("contract_levels") or {}).get("SCIENTIFIC_TRAINING_CONTRACT") != "PASS":
        raise SystemExit("unified corpus lacks SCIENTIFIC_TRAINING_CONTRACT")
    expected_weights = {
        layer: float(payload["trace_draw_weight"])
        for layer, payload in validation["layers"].items()
    }
    if unified.get("layer_weights") != expected_weights:
        raise SystemExit("unified corpus layer weights are off-protocol")
    expected_validation_counts = {
        layer: int(payload["records"]) for layer, payload in validation["layers"].items()
    }
    observed_validation_counts = {
        layer: int(payload["by_partition"]["validation"]["accepted"])
        for layer, payload in overlay["counts"].items()
    }
    if observed_validation_counts != expected_validation_counts:
        raise SystemExit("validation record census is off-protocol")
    return unified, overlay


def main() -> int:
    args = _parser().parse_args()
    config = load_json_object(args.config)
    inventory = load_json_object(args.inventory)
    unified, overlay = _validate_artifact_root(
        args.artifact_root,
        config=config,
    )
    precompiled = args.artifact_root / "edit_precompile_v1"
    packed = args.artifact_root / "edit_packed_v1"
    packed_mmp = args.artifact_root / "mmp_packed_v1"
    expected_contract = load_json_object(precompiled / "BUILD_COMPLETE.json").get("contract")
    layer_weights = {
        layer: float(payload["trace_draw_weight"])
        for layer, payload in config["validation_data"]["layers"].items()
    }
    sampling = config["validation_data"]["record_sampling"]
    corpus = load_production_edit_corpus(
        precompiled,
        mmp_pool_path=args.artifact_root / "<PACKED_MMP_ONLY>",
        partition="validation",
        layer_weights=layer_weights,
        expected_contract=expected_contract,
        path_length_bins=tuple(sampling["curriculum_bin_edges"]),
        cold_element_floor=float(sampling["cold_element_floor"]),
        packed_root=packed,
        packed_mmp_root=packed_mmp,
        representability_overlay=overlay,
        require_packed_mmp=True,
        verify_fraction=0.0,
        seed=int(config["panels"][args.panel_kind]["seed"]),
    )
    if args.panel_kind == "production_law":
        panel = build_production_validation_panel(
            corpus,
            stage=args.production_stage,
            config=config,
            inventory=inventory,
        )
    else:
        spec = config["panels"][FAMILY_FORENSICS_PANEL_ID]
        target = int(spec["initial_minimum_nonterminal_examples_per_family"])
        panel = build_family_forensics_validation_panel(
            corpus,
            requested_nonterminal_examples_by_family={
                family: target for family in spec["active_families"]
            },
            config=config,
            inventory=inventory,
        )
    write_validation_panel(
        panel,
        args.output,
        config=config,
        inventory=inventory,
    )
    identity = panel_build_identity(
        config=config,
        inventory=inventory,
        panel=panel,
    )
    identity.update(
        {
            "panel_path": str(args.output),
            "unified_manifest_checksum": unified["manifest_checksum"],
            "representability_overlay_checksum": overlay["effective_corpus_checksum"],
            "records_by_layer": corpus.provenance["records_by_layer"],
            "training_authorized": False,
            "checkpoint_scoring_performed": False,
            "checkpoint_ranking_performed": False,
            "checkpoint_selection_performed": False,
        }
    )
    _immutable_json_write(identity, args.identity_output)
    print(json.dumps(identity, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
