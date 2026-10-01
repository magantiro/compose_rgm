"""Write all T4 cell configs after local asset verification. No docking calls."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path

from compose_v4.experiments.t4_panel import (
    cell_config,
    guidance_arm_config,
    load_panel,
    sha256,
    verify_cell_config,
)
from compose_v4.experiments.t4_quickvina import verified_file

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--obabel", required=True, type=Path)
    parser.add_argument("--qvina", required=True, type=Path)
    parser.add_argument("--receptors", required=True, type=Path)
    parser.add_argument(
        "--checkpoint", type=Path, default=ROOT / "local_assets/fragments/r_theta_nll.pt"
    )
    parser.add_argument("--base-seed", required=True, type=int)
    parser.add_argument("--replicates", required=True, type=int)
    parser.add_argument("--strength", required=True, type=float)
    parser.add_argument(
        "--guidance-modes",
        nargs="+",
        choices=("off", "shadow", "active"),
        default=("active",),
        help="reference-selection arms to prepare for every matched cell",
    )
    args = parser.parse_args()
    if not 1 <= args.replicates <= 100:
        parser.error("replicates must be in [1, 100]")
    if len(set(args.guidance_modes)) != len(args.guidance_modes):
        parser.error("guidance modes must be unique")
    output = args.output.resolve()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing to overwrite T4 panel: {output}")
    panel_path = ROOT / "experiments/t4/targets.json"
    template_path = ROOT / "experiments/t4/example.json"
    panel, leads = load_panel(panel_path)
    template = json.loads(template_path.read_text())
    checkpoint_sha256 = json.loads((ROOT / "experiments/fragments/assets.json").read_text())[
        "assets"
    ]["checkpoint"]["sha256"]
    verified_file(args.checkpoint, checkpoint_sha256)
    obabel_sha256 = sha256(args.obabel)
    verified_file(args.obabel, obabel_sha256, executable=True)
    verified_file(args.qvina, panel["qvina_sha256"], executable=True)
    for name, target in panel["targets"].items():
        verified_file(args.receptors / f"{name}.pdbqt", target["receptor_sha256"])
        verified_file(panel_path.parent / target["route_expert"], target["route_expert_sha256"])
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(dir=output.parent, prefix=f".{output.name}."))
    try:
        records = []
        for lead in leads:
            target = panel["targets"][lead["target"]]
            for delta in (0.4, 0.6):
                for replicate in range(args.replicates):
                    config = cell_config(
                        template,
                        lead,
                        target,
                        delta=delta,
                        replicate=replicate,
                        base_seed=args.base_seed,
                        strength=float(args.strength),
                        obabel=args.obabel,
                        obabel_sha256=obabel_sha256,
                        qvina=args.qvina,
                        receptor=args.receptors / f"{lead['target']}.pdbqt",
                        checkpoint=args.checkpoint,
                        checkpoint_sha256=checkpoint_sha256,
                        route_expert=panel_path.parent / target["route_expert"],
                        qvina_sha256=panel["qvina_sha256"],
                    )
                    for mode in args.guidance_modes:
                        arm_config = guidance_arm_config(config, mode)
                        suffix = (
                            f"_{mode}" if len(args.guidance_modes) > 1 or mode != "active" else ""
                        )
                        name = (
                            f"{lead['target']}_lead{lead['idx']:02d}_d{int(delta * 10):02d}"
                            f"_r{replicate:02d}{suffix}.json"
                        )
                        path = staging / name
                        path.write_text(json.dumps(arm_config, sort_keys=True, indent=2) + "\n")
                        verify_cell_config(path)
                        records.append(
                            {
                                "path": name,
                                "sha256": sha256(path),
                                "target": lead["target"],
                                "lead_index": lead["idx"],
                                "delta": delta,
                                "replicate": replicate,
                                "search_seed": arm_config["search"]["seed"],
                                "guidance_mode": mode,
                            }
                        )
        manifest = {
            "schema_version": "compose.t4.prepared_panel.v1",
            "source_panel_sha256": sha256(panel_path),
            "template_sha256": sha256(template_path),
            "checkpoint_sha256": checkpoint_sha256,
            "obabel_sha256": obabel_sha256,
            "qvina_sha256": panel["qvina_sha256"],
            "seed_derivation": "base_seed + 1000 * lead_index + replicate; shared across deltas",
            "base_seed": args.base_seed,
            "replicates": args.replicates,
            "strength": args.strength,
            "guidance_modes": args.guidance_modes,
            "configurations": records,
            "docking_calls": 0,
        }
        (staging / "manifest.json").write_text(
            json.dumps(manifest, sort_keys=True, indent=2) + "\n"
        )
        if output.exists() or output.is_symlink():
            raise FileExistsError(f"refusing to overwrite T4 panel: {output}")
        os.rename(staging, output)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    print(json.dumps({"output": str(output), "configs": len(records), "docking_calls": 0}))


if __name__ == "__main__":
    main()
