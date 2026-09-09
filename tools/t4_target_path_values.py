"""Score saved development witnesses; never generate, replay, fit, or dock."""

from __future__ import annotations

import argparse
import gzip
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import numpy as np
from rdkit import Chem, rdBase
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Contrib.SA_Score import sascorer

from compose_v4.control.docking_value import graph_kernel, identity, molecular_features
from compose_v4.experiments.continuation_profile import sha256_file, verify_file
from compose_v4.experiments.t4_endpoint_selection import calculate_properties, feasible_endpoint
from compose_v4.rewrite.kernel import canonical_state_key
from compose_v4.rewrite.trace_shard import decode_state
from tools.ivg_winner_paths import implementation_closure, publish

ROOT = Path(__file__).resolve().parents[1]
CELL = "docking_parp1_idx0_thr4"


def curve_summary(values):
    scores = np.asarray(values, dtype=float)
    if scores.ndim != 1 or len(scores) < 2 or not np.isfinite(scores).all():
        raise ValueError("a path requires at least two finite scores")
    if np.any((scores < 0) | (scores > 1)):
        raise ValueError("path similarity must be in [0, 1]")
    decreases = np.flatnonzero(np.diff(scores) < -1e-12) + 1
    drawdown = np.maximum.accumulate(scores) - scores
    return {
        "source_similarity": float(scores[0]),
        "terminal_similarity": float(scores[-1]),
        "decreasing_steps": decreases.tolist(),
        "decreasing_edits": len(decreases),
        "max_drawdown": float(drawdown.max()),
        "max_drawdown_step": int(np.argmax(drawdown)),
    }


def score_saved_path(row, payload):
    path = payload["path"]
    if path["status"] != "witness_found" or len(path["states"]) != len(path["actions"]) + 1:
        raise ValueError("saved witness status or state/action count is invalid")
    target, target_features = molecular_features(row["target"])
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    seed_fp = generator.GetFingerprint(Chem.MolFromSmiles(row["source"]))
    states = []
    for step, state in enumerate(path["states"]):
        smiles = canonical_state_key(decode_state(state))
        _, features = molecular_features(smiles)
        properties = calculate_properties(
            Chem.MolFromSmiles(smiles),
            seed_fp=seed_fp,
            generator=generator,
            sa_scorer=sascorer.calculateScore,
            delta=0.4,
            qed_min=0.6,
            sa_max=4.0,
        )
        states.append(
            {
                "step": step,
                "smiles": smiles,
                "state_sha256": identity(state),
                "target_similarity": float(graph_kernel([features], [target_features])[0, 0]),
                "exact_target": smiles == target,
                "properties": properties,
                "preceding_mark": path["actions"][step - 1] if step else None,
            }
        )
    if not states[-1]["exact_target"]:
        raise ValueError("pinned-runtime saved endpoint does not equal the target")
    return {
        "pair_id": row["pair_id"],
        "target": target,
        "witness_steps": len(path["actions"]),
        "states": states,
        **curve_summary([s["target_similarity"] for s in states]),
        "t4_infeasible_steps": [
            s["step"] for s in states if not feasible_endpoint(s["properties"])
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--audit-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if rdBase.rdkitVersion != "2024.03.5":
        raise ValueError("saved path analysis requires RDKit 2024.03.5")
    dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True)
    if dirty.strip():
        raise ValueError("run from a clean committed tree; publish results outside it")
    verify_file(args.audit, args.audit_sha256)
    audit = json.loads(args.audit.read_text())
    selected = [r for r in audit["pairs"] if any(v["cell"] == CELL for v in r["references"])]
    if len(selected) != 3 or any(r["status"] != "witness_found" for r in selected):
        raise ValueError("expected all three saved PARP1 seed0 d=0.4 witnesses")
    started = perf_counter()
    hashes, results = {str(args.audit.resolve()): args.audit_sha256}, []
    for row in sorted(selected, key=lambda r: r["pair_id"]):
        path = args.audit.parent / row["receipt"]
        verify_file(path, row["receipt_sha256"])
        receipt = json.loads(gzip.decompress(path.read_bytes()))
        if identity(receipt["payload"]) != receipt["payload_sha256"]:
            raise ValueError(f"{path}: corrupt witness payload")
        hashes[str(path.resolve())] = row["receipt_sha256"]
        results.append(score_saved_path(row, receipt["payload"]))
    config = {
        "cell": CELL,
        "delta": 0.4,
        "qed_min": 0.6,
        "sa_max": 4.0,
        "score": "fixed_mean_morgan_atompair_tanimoto",
        "tolerance": 1e-12,
    }
    result = {
        "schema_version": "t4_target_path_values_v1",
        "configuration": config,
        "code_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "input_sha256": hashes,
        "implementation_sha256": {
            **implementation_closure(Path(__file__).resolve()),
            "tools/ivg_winner_paths.py": sha256_file(ROOT / "tools/ivg_winner_paths.py"),
        },
        "scoring_assets": {
            name: sha256_file(Path(sascorer.__file__).parent / name)
            for name in ("sascorer.py", "fpscores.pkl.gz")
        },
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "rdkit": rdBase.rdkitVersion,
        },
        "hardware": {
            "machine": platform.machine(),
            "system": platform.system(),
            "accelerator": None,
        },
        "split_identity": "already inspected IVG development witnesses; no held-out inputs",
        "seed": None,
        "randomness": "none",
        "precision": "float64 score arithmetic",
        "n_paths": len(results),
        "n_states": sum(len(r["states"]) for r in results),
        "exclusions": [],
        "results": results,
        "seconds": perf_counter() - started,
        "new_oracle_calls": 0,
        "learned_law_calls": 0,
        "executor_replays": 0,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "interpretation": "score valleys on saved paths only; not recovery, learned-law support, or proof of greedy failure",
    }
    publish(args.output, result)
    print(
        json.dumps(
            {
                "paths": [{k: v for k, v in r.items() if k != "states"} for r in results],
                "seconds": result["seconds"],
                "output": str(args.output),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
