"""Matched, small de novo C1 learned-legal-mark prior pilot.

The unchanged C1 endpoint records supply the control.  This is a zero-oracle
development screen, not a final de novo benchmark or checkpoint-selection run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
from pathlib import Path

import numpy as np
import rdkit
import torch
from evaluate_tracelet_rollouts import load_factorized_rollout_checkpoint

from compose_v4.eval.denovo_benchmark import denovo_benchmark_metrics
from compose_v4.eval.denovo_ring_marginal import RingSystemPlanPrior
from compose_v4.experiments.denovo_ring_plan import CatalogSignatureIndex, sample_denovo_arm
from compose_v4.rewrite.kernel import de_novo_rewrite_system

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "denovo_legal_mark_prior_pilot_v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run(contract_path: Path, output: Path) -> dict:
    contract = json.loads(contract_path.read_text())
    payload = contract["payload"]
    actual = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if actual != contract["payload_sha256"]:
        raise ValueError("de novo pilot contract payload hash mismatch")
    paths = {label: Path(info["path"]) for label, info in payload["inputs"].items()}
    for label, path in paths.items():
        if not path.is_file() or sha256(path) != payload["inputs"][label]["sha256"]:
            raise ValueError(f"de novo pilot {label} asset hash mismatch: {path}")
    saved = json.loads(paths["baseline_records"].read_text())
    baseline = {row["index"]: row for row in saved if row["arm"] == "C1"}
    indices = payload["selected_indices"]
    if len(indices) != len(set(indices)) or any(i not in baseline for i in indices):
        raise ValueError("selected pilot indices must be distinct saved C1 rows")
    torch.set_num_threads(1)
    model, checkpoint = load_factorized_rollout_checkpoint(str(paths["checkpoint"]))
    model.eval()
    runtime = de_novo_rewrite_system()
    model.sampling_logit_scale = float(payload["prior"]["sampling_logit_scale"])
    ring_prior = RingSystemPlanPrior.read(paths["ring_prior"])
    index = CatalogSignatureIndex.build(model.ring_system_templates)
    results = []
    for position, item in enumerate(indices):
        reference = baseline[item]
        sampled = sample_denovo_arm(
            model,
            arm="C1",
            rng=np.random.default_rng(reference["trajectory_seed"]),
            source_prior=checkpoint["tree_source_prior"],
            index=index,
            plan_prior=ring_prior,
            n_slots=40,
            operational_horizon=16.0,
            time_step=0.1,
            max_events=128,
            runtime=runtime,
        )
        if not sampled["valid_state"] or not sampled["connected"] or not sampled["smiles"]:
            raise RuntimeError(f"invalid or missing committed endpoint at index {item}")
        results.append(
            {
                "index": item,
                "trajectory_seed": reference["trajectory_seed"],
                "baseline_smiles": reference["canonical_smiles"],
                "candidate_smiles": sampled["smiles"],
                "event_rules": sampled["event_rules"],
                "ring_plan": sampled["ring_plan"],
                "valid_state": sampled["valid_state"],
                "connected": sampled["connected"],
                "proposal_work": "one learned action-table evaluation per draw, same as baseline",
            }
        )
        print(f"completed {position + 1}/{len(indices)}", flush=True)
    baseline_metrics = denovo_benchmark_metrics([baseline[i]["canonical_smiles"] for i in indices])
    candidate_metrics = denovo_benchmark_metrics([r["candidate_smiles"] for r in results])
    result = {
        "schema": SCHEMA,
        "evidence_class": "small_zero_oracle_development_ablation_not_official_benchmark",
        "contract_path": str(contract_path),
        "contract_sha256": sha256(contract_path),
        "contract_payload_sha256": actual,
        "code_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "implementation_sha256": {
            path: sha256(ROOT / path)
            for path in (
                "src/compose_v4/model/factorized_tracelet_rate_model.py",
                "scripts/denovo_legal_mark_prior_pilot.py",
                "src/compose_v4/experiments/denovo_ring_plan.py",
            )
        },
        "versions": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "rdkit": rdkit.__version__,
        },
        "attempted_per_arm": len(indices),
        "baseline_metrics": baseline_metrics,
        "candidate_metrics": candidate_metrics,
        "results": results,
        "input_sha256": {label: sha256(path) for label, path in paths.items()},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, output)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.contract, args.output)
    print(json.dumps({k: result[k] for k in ("baseline_metrics", "candidate_metrics")}, indent=2))


if __name__ == "__main__":
    main()
