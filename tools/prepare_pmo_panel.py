"""Prepare matched PMO campaign configs without an oracle call."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path

from compose_v4.experiments.continuation_profile import sha256_file, verify_file
from compose_v4.experiments.pmo.__main__ import read_config
from compose_v4.experiments.pmo_panel import (
    ARMS,
    GUIDANCE_MODES,
    cell_config,
    derived_seed,
    load_panel,
)
from compose_v4.model.reference_checkpoint import load_frozen_reference

ROOT = Path(__file__).resolve().parents[1]
TASK_DIR = ROOT / "experiments/pmo"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--oracle-python", required=True, type=Path)
    parser.add_argument("--oracle-assets", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--base-seed", required=True, type=int)
    parser.add_argument("--replicates", required=True, type=int)
    parser.add_argument("--strength", required=True, type=float)
    parser.add_argument("--tasks", nargs="+")
    parser.add_argument("--arms", nargs="+", choices=ARMS, default=["structured"])
    parser.add_argument("--guidance-modes", nargs="+", choices=GUIDANCE_MODES, default=["active"])
    args = parser.parse_args(argv)
    if not 1 <= args.replicates <= 100:
        parser.error("replicates must be in [1, 100]")
    if len(set(args.arms)) != len(args.arms) or len(set(args.guidance_modes)) != len(
        args.guidance_modes
    ):
        parser.error("proposal arms and guidance modes must be unique")
    if args.base_seed < 0:
        parser.error("base-seed must be nonnegative")
    output = args.output.resolve()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing to overwrite PMO panel: {output}")

    assets_path = TASK_DIR / "assets.json"
    assets = json.loads(assets_path.read_text())
    audit_path = TASK_DIR / assets["oracle_audit"]["path"]
    verify_file(audit_path, assets["oracle_audit"]["sha256"])
    audit = json.loads(audit_path.read_text())
    panel_path = TASK_DIR / "panel.json"
    panel = load_panel(panel_path, oracle_tasks=audit["tasks"])
    indices = panel["seed_indices"]
    excluded = set(panel["excluded_from_22_objective_mean"])
    available = tuple(task for task in indices if task not in excluded)
    requested = available if args.tasks is None else tuple(args.tasks)
    if not requested or len(set(requested)) != len(requested) or set(requested) - set(available):
        parser.error("tasks must be distinct members of the 22-objective protocol")
    tasks = tuple(sorted(requested, key=indices.__getitem__))

    oracle_python = args.oracle_python.resolve(strict=True)
    if not oracle_python.is_file() or not os.access(oracle_python, os.X_OK):
        raise ValueError(f"PMO oracle Python must be an executable file: {oracle_python}")
    oracle_assets = None if args.oracle_assets is None else args.oracle_assets.resolve(strict=True)
    if oracle_assets is None and any(
        audit["tasks"][task]["classification"].startswith("ASSET_BACKED") for task in tasks
    ):
        raise ValueError("asset-backed PMO tasks require --oracle-assets")
    if oracle_assets is not None:
        for asset in assets["oracle_assets"].values():
            verify_file(oracle_assets / asset["relative_path"], asset["sha256"])

    reference_path = ROOT / "experiments/reference/model.json"
    reference = json.loads(reference_path.read_text())
    model = reference["checkpoint"]
    ring_catalog = reference["catalog"]
    checkpoint = (
        (reference_path.parent / model["path"]).resolve()
        if args.checkpoint is None
        else args.checkpoint.resolve(strict=True)
    )
    catalog = (reference_path.parent / ring_catalog["path"]).resolve()
    load_frozen_reference(
        checkpoint,
        expected_sha256=model["sha256"],
        expected_catalog_fingerprint=ring_catalog["fingerprint"],
        catalog_path=catalog,
        expected_catalog_sha256=ring_catalog["sha256"],
    )
    template_path = TASK_DIR / "example.json"
    template = json.loads(template_path.read_text())
    if (
        template["reference"]["sha256"] != model["sha256"]
        or template["reference"]["catalog_sha256"] != ring_catalog["sha256"]
        or template["reference"]["catalog_fingerprint"] != ring_catalog["fingerprint"]
    ):
        raise ValueError("PMO template reference differs from the shared model manifest")
    for name in ("jump_plans", "initialization"):
        item = template[name]
        verify_file((template_path.parent / item["path"]).resolve(), item["sha256"])

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(dir=output.parent, prefix=f".{output.name}."))
    try:
        records = []
        for task in tasks:
            for replicate in range(args.replicates):
                seed = derived_seed(task, args.base_seed, replicate, indices)
                for arm in args.arms:
                    for mode in args.guidance_modes:
                        name = f"{task}_r{replicate:02d}_{arm}_{mode}.json"
                        config = cell_config(
                            template,
                            template_dir=template_path.parent,
                            task=task,
                            seed=seed,
                            arm=arm,
                            guidance_mode=mode,
                            strength=args.strength,
                            oracle_python=oracle_python,
                            oracle_assets=oracle_assets,
                            checkpoint=checkpoint,
                            catalog=catalog,
                        )
                        path = staging / name
                        path.write_text(json.dumps(config, sort_keys=True, indent=2) + "\n")
                        read_config(path)
                        records.append(
                            {
                                "path": name,
                                "sha256": sha256_file(path),
                                "task": task,
                                "replicate": replicate,
                                "seed": seed,
                                "arm": arm,
                                "guidance_mode": mode,
                            }
                        )
        manifest = {
            "schema_version": "compose.pmo.prepared_panel.v1",
            "panel_sha256": sha256_file(panel_path),
            "oracle_audit_sha256": assets["oracle_audit"]["sha256"],
            "template_sha256": sha256_file(template_path),
            "reference_manifest_sha256": sha256_file(reference_path),
            "checkpoint_sha256": model["sha256"],
            "catalog_sha256": ring_catalog["sha256"],
            "base_seed": args.base_seed,
            "seed_derivation": "base_seed + 1000 * fixed_task_index + replicate",
            "replicates": args.replicates,
            "strength": args.strength,
            "tasks": tasks,
            "arms": args.arms,
            "guidance_modes": args.guidance_modes,
            "excluded_from_22_objective_mean": sorted(excluded),
            "configurations": records,
            "oracle_calls": 0,
        }
        (staging / "manifest.json").write_text(
            json.dumps(manifest, sort_keys=True, indent=2) + "\n"
        )
        if output.exists() or output.is_symlink():
            raise FileExistsError(f"refusing to overwrite PMO panel: {output}")
        os.rename(staging, output)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    print(json.dumps({"output": str(output), "configs": len(records), "oracle_calls": 0}))


if __name__ == "__main__":
    main()
