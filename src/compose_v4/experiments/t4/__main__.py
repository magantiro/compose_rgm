"""Validate assets or run one local T4 lead-optimization campaign."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from compose_v4.control.reference_guidance import GuidanceConfig
from compose_v4.control.reference_programs import FrozenProgramReference
from compose_v4.experiments.core_runtime import core_runtime_identity as runtime_identity
from compose_v4.experiments.t4_local_campaign import T4RunConfig, run_t4
from compose_v4.experiments.t4_quickvina import DockingConfig, QuickVinaEvaluator, verified_file
from compose_v4.experiments.t4_route_proposals import RouteProposalConfig, load_route_expert
from compose_v4.model.reference_checkpoint import load_frozen_reference


def read_config(path: Path) -> tuple[dict, T4RunConfig, DockingConfig]:
    try:
        raw = json.loads(path.read_text())
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid JSON configuration {path}: {error}") from error
    fields = {"schema_version", "search", "docking", "reference", "route_expert"}
    if (
        not isinstance(raw, dict)
        or set(raw) != fields
        or raw["schema_version"] != "compose.t4.config.v1"
    ):
        raise ValueError(f"{path}: expected compose.t4.config.v1 with fields {sorted(fields)}")
    try:
        search = dict(raw["search"])
        search["guidance"] = GuidanceConfig(**search.get("guidance", {}))
        search["route"] = RouteProposalConfig(**search.get("route", {}))
        if "lanes" in search:
            search["lanes"] = tuple(search["lanes"])
        config = T4RunConfig(**search)
        docking = dict(raw["docking"])
        for field in ("obabel", "qvina", "receptor"):
            docking[field] = (path.parent / docking[field]).resolve()
        for field in ("center", "size"):
            docking[field] = tuple(docking[field])
        for field in ("reference", "route_expert"):
            asset = raw[field]
            if asset is not None:
                expected = {"path", "sha256"} | (
                    {"catalog_fingerprint", "catalog_path", "catalog_sha256"}
                    if field == "reference"
                    else set()
                )
                if not isinstance(asset, dict) or set(asset) != expected:
                    raise ValueError(f"{field} expects exactly {sorted(expected)}")
                asset["path"] = str((path.parent / asset["path"]).resolve())
                if field == "reference":
                    asset["catalog_path"] = str((path.parent / asset["catalog_path"]).resolve())
        if (config.guidance.mode != "off") != (raw["reference"] is not None):
            raise ValueError(
                "reference asset is required in active/shadow modes and must be null in off mode"
            )
        if ("route_complete_region" in config.lanes) != (raw["route_expert"] is not None):
            raise ValueError("route_expert is required exactly when the route lane is enabled")
        return raw, config, DockingConfig(**docking)
    except (TypeError, KeyError, ValueError) as error:
        raise ValueError(f"invalid configuration {path}: {error}") from error


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("validate", "run", "resume"))
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.command != "validate" and args.output is None:
        parser.error("run/resume require --output")
    if args.command == "validate" and args.output is not None:
        parser.error("validate writes no output and does not take --output")
    raw, config, docking = read_config(args.config.resolve())
    implementation = runtime_identity()
    evaluator = QuickVinaEvaluator(docking, (args.output or Path(".")) / "docking")
    expert = None
    if raw["route_expert"] is not None:
        asset = raw["route_expert"]
        expert = load_route_expert(Path(asset["path"]), expected_sha256=asset["sha256"])
    if raw["reference"] is not None:
        verified_file(Path(raw["reference"]["path"]), raw["reference"]["sha256"])
        verified_file(Path(raw["reference"]["catalog_path"]), raw["reference"]["catalog_sha256"])
    if args.command == "validate":
        print(
            json.dumps(
                {
                    "status": "files_verified",
                    "evaluations": 0,
                    "search": asdict(config),
                    "evaluator": evaluator.identity(),
                    "reference": raw["reference"],
                    "note": "file hashes checked, neural checkpoint not loaded, executables not run",
                },
                sort_keys=True,
                indent=2,
            )
        )
        return
    import torch

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    reference = None
    if raw["reference"] is not None:
        asset = raw["reference"]
        reference = FrozenProgramReference(
            load_frozen_reference(
                Path(asset["path"]),
                expected_sha256=asset["sha256"],
                expected_catalog_fingerprint=asset["catalog_fingerprint"],
                catalog_path=Path(asset["catalog_path"]),
                expected_catalog_sha256=asset["catalog_sha256"],
            )
        )
    result = run_t4(
        output=args.output,
        config=config,
        evaluate=evaluator,
        evaluator_identity=evaluator.identity(),
        implementation_identity=implementation,
        reference=reference,
        route_expert=expert,
        asset_identity={"reference": raw["reference"], "route_expert": raw["route_expert"]},
        resume=args.command == "resume",
    )
    print(
        json.dumps(
            {
                key: result[key]
                for key in ("status", "charged_calls", "budget", "best_score", "best_smiles")
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
