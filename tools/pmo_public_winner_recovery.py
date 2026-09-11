"""Recover a locked public target through saved-slot rewrites, not blind search."""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import math
import platform
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from rdkit import Chem, RDLogger, rdBase
from rdkit.Chem import Draw, rdMolDescriptors

from compose_v4.experiments.pmo_macro_probe import make_oracle
from compose_v4.experiments.t4_matched_pilot import unseal
from compose_v4.experiments.winner_paths import PathConfig, find_path_from_state, replay
from tools.ivg_winner_paths import digest, implementation_closure, publish, sha

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/pmo_public_winner_recovery.json"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    cfg = json.loads(CONFIG.read_text())
    if rdBase.rdkitVersion != cfg["evaluation"]["rdkit_version"]:
        raise ValueError("RDKit version differs from the target-recovery lock")
    if importlib.metadata.version("PyTDC") != cfg["evaluation"]["pytdc_version"]:
        raise ValueError("PyTDC version differs from the target-recovery lock")
    for suffix in ("pdf", "png"):
        asset = args.assets / f"perindopril_top_molecules.{suffix}"
        if sha(asset) != cfg["target"][f"{suffix}_sha256"]:
            raise ValueError(f"public source asset mismatch: {asset}")
    archive_path = ROOT / cfg["sources"]["best_archive"]
    if sha(archive_path) != cfg["sources"]["best_archive_sha256"]:
        raise ValueError("saved current-best archive differs from lock")
    archive = unseal(archive_path)["arms"][cfg["sources"]["best_arm"]]["archive"]
    best = max(archive, key=lambda row: (row["score"], row["id"]))
    roots_path = ROOT / cfg["sources"]["roots"]
    roots = json.loads(roots_path.read_text())["roots"]
    if len(roots) != 4 or cfg["evaluation"]["new_oracle_call_limit"] != 2:
        raise ValueError("target recovery requires four original roots and two score calls")
    sources = [
        {"id": "current_best", "smiles": best["smiles"], "state": best["node"]["graph"]},
        *[{"id": f"original_root_{i}", **row} for i, row in enumerate(roots)],
    ]
    molecule = Chem.MolFromSmiles(cfg["target"]["smiles"])
    if molecule is None:
        raise ValueError("locked public transcription is not a parseable molecule")
    target = {
        **cfg["target"],
        "canonical_smiles": Chem.MolToSmiles(molecule, isomericSmiles=False),
        "heavy_atoms": molecule.GetNumHeavyAtoms(),
        "aromatic_rings": rdMolDescriptors.CalcNumAromaticRings(molecule),
        "sssr_ring_sizes": sorted(map(len, molecule.GetRingInfo().AtomRings())),
    }
    oracle = make_oracle(cfg["task"], ROOT, {})  # Construct once; no score calls here.
    oracle_module = importlib.import_module("tdc.chem_utils.oracle.oracle")
    inputs = {
        str(path): sha(path)
        for path in (CONFIG, archive_path, roots_path, Path(oracle_module.__file__))
    }
    implementation = implementation_closure(Path(__file__).resolve())
    implementation["tools/ivg_winner_paths.py"] = sha(ROOT / "tools/ivg_winner_paths.py")
    record = {
        "schema_version": "pmo_public_winner_recovery_result_v1",
        "configuration": cfg,
        "input_sha256": inputs,
        "public_asset_directory": str(args.assets.resolve()),
        "implementation_sha256": implementation,
        "code_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "code_worktree_status": subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=ROOT, text=True
        ),
        "software": {
            "python": platform.python_version(),
            "rdkit": rdBase.rdkitVersion,
            "numpy": np.__version__,
            "pytdc": importlib.metadata.version("PyTDC"),
            "scipy": importlib.metadata.version("scipy"),
        },
        "hardware": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "device": "cpu",
        },
        "precision": "production PyTDC double-precision objective; integer executor states",
        "split": "answer-known development, no held-out evaluation",
        "seed": cfg["seed"],
        "target": target,
        "sources": sources,
        "source_count": len(sources),
        "exclusions": [],
    }
    # Source and implementation closure, not unrelated revision changes, bind reuse.
    run_id = digest(
        {"inputs": inputs, "implementation": implementation, "software": record["software"]}
    )
    record["run_id"] = run_id
    output = ROOT / "diagnostics/pmo_public_winner_recovery"
    output.mkdir(parents=True, exist_ok=True)
    Draw.MolToFile(molecule, str(output / "target.png"), size=(1200, 500))
    if args.prepare_only:
        print(json.dumps(target, sort_keys=True), flush=True)
        return
    RDLogger.DisableLog("rdApp.warning")
    record["started_utc"] = datetime.now(timezone.utc).isoformat()
    score_path = output / "scores.json"
    if score_path.exists():
        scored = json.loads(score_path.read_text())
        if scored["run_id"] != run_id:
            raise ValueError(
                "existing scores have different inputs or implementation; do not overwrite or rescore"
            )
    else:
        start = time.monotonic()
        baseline_value = oracle(best["smiles"])
        target_value = oracle(target["canonical_smiles"])
        scored = {
            **record,
            "new_oracle_calls": 2,
            "score_seconds": time.monotonic() - start,
            "oracle_ledger": [
                {
                    "role": "current_best_parity",
                    "smiles": best["smiles"],
                    "score": baseline_value,
                    "previous_score": best["score"],
                },
                {
                    "role": "public_target",
                    "smiles": target["canonical_smiles"],
                    "score": target_value,
                },
            ],
            "parity_passed": math.isfinite(baseline_value)
            and abs(baseline_value - best["score"])
            <= cfg["evaluation"]["parity_absolute_tolerance"],
            "printed_score_matches_rounding": math.isfinite(target_value)
            and round(target_value, 2) == target["reported_rounded_score"],
        }
        publish(score_path, scored)
    print(
        json.dumps(
            {
                k: scored[k]
                for k in ("oracle_ledger", "parity_passed", "printed_score_matches_rounding")
            }
        ),
        flush=True,
    )
    if not scored["parity_passed"] or not scored["printed_score_matches_rounding"]:
        raise ValueError(
            "scoring mismatch banked; do not reinterpret a mismatching public transcription"
        )
    paths = []
    for source in sources:
        destination = output / f"{source['id']}.json"
        if destination.exists():
            row = json.loads(destination.read_text())
            if row["run_id"] != run_id:
                raise ValueError(f"incompatible saved path: {destination}")
        else:
            print(f"searching {source['id']} (bounded, target-known)", flush=True)
            start = time.monotonic()
            result = find_path_from_state(
                source["state"],
                target["canonical_smiles"],
                PathConfig(**cfg["path_search"]),
                source["smiles"],
            )
            row = {
                "run_id": run_id,
                "source_id": source["id"],
                "seconds": time.monotonic() - start,
                "result": result,
            }
            if (
                result["status"] == "witness_found"
                and replay(source["state"], result["actions"], result["target_2d"])
                != result["states"]
            ):
                raise ValueError("independent replay disagrees with saved witness")
            publish(destination, row)
        paths.append(
            {
                "source_id": source["id"],
                "path": str(destination.relative_to(ROOT)),
                "sha256": sha(destination),
                "seconds": row["seconds"],
                "status": row["result"]["status"],
                "steps": row["result"].get("witness_steps"),
            }
        )
        print(json.dumps(paths[-1]), flush=True)
    publish(
        output / "result.json",
        {
            **scored,
            "paths": paths,
            "completed_utc": datetime.now(timezone.utc).isoformat(),
            "interpretation": "target-known compiler capability only; not a learned controller or blind winner discovery",
        },
    )


if __name__ == "__main__":
    main()
