"""One frozen, zero-oracle superstructure learned-law tilt pilot.

Uses the first ten attempts of official seed zero as the unchanged control;
generates ten fresh attempts per each of the same ten prompts under one opt-in
legal-mark ranker.  This is a development screen, not an official benchmark row.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
from pathlib import Path
from shutil import copytree

import numpy as np
import rdkit
import torch
from evaluate_tracelet_rollouts import load_factorized_rollout_checkpoint
from fetch_official_fragment_evaluator import default_cache_dir, verify_only
from run_fragment_constrained_suite import run_task
from run_fragment_superstructure_official_v2 import read_shard

from compose_v4.benchmark.fragment_attachment_control import AttachmentControlConfig
from compose_v4.benchmark.fragment_conditioned_sampler import SamplerConfig
from compose_v4.benchmark.fragment_constrained import FragmentTask, load_genmol_prompts
from compose_v4.benchmark.fragment_official_metrics import official_prompt_metrics
from compose_v4.model.legal_mark_prior import LearnedLegalMarkPrior
from compose_v4.rewrite.kernel import de_novo_rewrite_system

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "fragment_legal_mark_prior_pilot_v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _mean(rows: list[dict], key: str) -> float:
    return float(np.mean([row[key] for row in rows]))


def run(output: Path, *, attempts: int = 10) -> dict:
    pilot_contract_path = ROOT / "configs/fragment_legal_mark_prior_pilot_v1.json"
    pilot_contract = json.loads(pilot_contract_path.read_text())
    pilot_payload = pilot_contract["payload"]
    actual_payload_hash = hashlib.sha256(
        json.dumps(pilot_payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if actual_payload_hash != pilot_contract["payload_sha256"]:
        raise ValueError("pilot contract payload hash mismatch")
    planned = pilot_payload["candidate"]
    if attempts != planned["attempts_per_prompt"] or planned["seed"] != 0:
        raise ValueError("pilot attempt count or seed differs from frozen contract")
    official_contract_path = ROOT / "configs/fragment_superstructure_official_v2.json"
    if _sha256(official_contract_path) != pilot_payload["source_official_contract_sha256"]:
        raise ValueError("official baseline contract hash mismatch")
    official = json.loads(official_contract_path.read_text())["payload"]
    # An isolated worktree has no local evaluator cache. Reuse only the exact
    # hash-verified upstream blobs from the frozen run; never silently fetch a
    # different release or substitute a local metric implementation.
    baseline_cache = ROOT.parent / "fragment-interface-ablation-20260923" / ".official_eval_cache"
    verify_only(baseline_cache)
    if not default_cache_dir().exists():
        copytree(baseline_cache, default_cache_dir())
    verified_evaluator = verify_only()
    checkpoint = Path(official["checkpoint"]["path"])
    if _sha256(checkpoint) != official["checkpoint"]["sha256"]:
        raise ValueError("checkpoint hash mismatch")
    manifest = ROOT / official["prompt_manifest"]
    if _sha256(manifest) != official["material_sha256"][official["prompt_manifest"]]:
        raise ValueError("fragment prompt manifest hash mismatch")
    baseline_root = (
        ROOT.parent / "fragment-interface-ablation-20260923" / official["output_dir"] / "shards"
    )
    prompts = [
        p for p in load_genmol_prompts(manifest) if p.task == FragmentTask.SUPERSTRUCTURE_GENERATION
    ]
    if len(prompts) != 10 or len({p.drug_name for p in prompts}) != 10:
        raise ValueError("pilot requires ten distinct official superstructure prompts")
    baseline: dict[str, dict] = {}
    baseline_hashes: dict[str, str] = {}
    for prompt in prompts:
        path = baseline_root / f"superstructure_generation__{prompt.drug_name}__seed0.json"
        if not path.is_file():
            raise FileNotFoundError(f"official baseline seed-zero shard missing: {path}")
        row = read_shard(path, prompt.drug_name, 0, official)
        baseline_hashes[prompt.drug_name] = _sha256(path)
        samples = row["chemical_samples"][:attempts]
        emitted = row["emitted_samples"][:attempts]
        baseline[prompt.drug_name] = {
            "official": official_prompt_metrics(samples, expected_samples=attempts),
            "output_rate": sum(bool(s) for s in samples) / attempts,
            "fidelity_rate": sum(bool(s) for s in emitted) / attempts,
            "attempt_records": row["attempt_records"][:attempts],
        }

    torch.set_num_threads(1)
    model, _meta = load_factorized_rollout_checkpoint(str(checkpoint))
    model.eval()
    control = AttachmentControlConfig(
        enabled=True,
        condition_initial_locked_family=True,
        hard_lock_effective_chemistry=True,
    )
    config = SamplerConfig(**official["sampler_config"])
    prior = LearnedLegalMarkPrior(
        candidates=planned["legal_mark_candidates"],
        strength=planned["learned_log_probability_strength"],
    )
    candidate = run_task(
        model,
        de_novo_rewrite_system(),
        prompts,
        FragmentTask.SUPERSTRUCTURE_GENERATION,
        seeds=1,
        seed_list=[0],
        samples=attempts,
        config=config,
        control=control,
        learned_prior=prior,
    )
    if candidate["prompts_scored"] != 10 or candidate["build_failures"]:
        raise RuntimeError("pilot did not score all ten prompts")
    candidate_rows = {drug: rows[0] for drug, rows in candidate["per_drug"].items()}
    if any(
        row["committed_endpoints"] != row["committed_chemically_valid"]
        or row["committed_endpoints"] != row["committed_fragment_preserving"]
        for row in candidate_rows.values()
    ):
        raise RuntimeError("committed chemical validity or fragment preservation failed")
    baseline_rows = list(baseline.values())
    candidate_values = list(candidate_rows.values())
    summary = {
        "baseline_quality_percent": _mean([x["official"] for x in baseline_rows], "quality"),
        "candidate_quality_percent": _mean([x["official"] for x in candidate_values], "quality"),
        "baseline_diversity": _mean([x["official"] for x in baseline_rows], "diversity"),
        "candidate_diversity": _mean([x["official"] for x in candidate_values], "diversity"),
        "baseline_output_rate": _mean(baseline_rows, "output_rate"),
        "candidate_output_rate": _mean(
            [{"output_rate": x["committed_endpoints"] / attempts} for x in candidate_values],
            "output_rate",
        ),
        "baseline_fidelity_rate": _mean(baseline_rows, "fidelity_rate"),
        "candidate_fidelity_rate": _mean(
            [{"fidelity_rate": x["emitted_nonempty"] / attempts} for x in candidate_values],
            "fidelity_rate",
        ),
        "prior_admitted_offers": sum(x["prior_admitted_offers"] for x in candidate_values),
        "prior_rank_events": sum(x["prior_rank_events"] for x in candidate_values),
        "prior_nonfirst_selections": sum(x["prior_nonfirst_selections"] for x in candidate_values),
    }
    result = {
        "schema": SCHEMA,
        "evidence_class": "small_development_ablation_not_official_benchmark",
        "source_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "pilot_contract": str(pilot_contract_path),
        "pilot_contract_sha256": _sha256(pilot_contract_path),
        "pilot_contract_payload_sha256": actual_payload_hash,
        "inputs": {
            "official_contract": str(official_contract_path),
            "official_contract_sha256": _sha256(official_contract_path),
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": _sha256(checkpoint),
            "prompt_manifest": str(manifest),
            "prompt_manifest_sha256": _sha256(manifest),
            "baseline_shards_sha256": baseline_hashes,
            "official_evaluator_blobs_sha256": verified_evaluator,
        },
        "implementation_sha256": {
            path: _sha256(ROOT / path)
            for path in (
                "src/compose_v4/model/legal_mark_prior.py",
                "src/compose_v4/benchmark/fragment_conditioned_sampler.py",
                "tools/run_fragment_constrained_suite.py",
                "tools/run_fragment_legal_mark_prior_pilot.py",
            )
        },
        "versions": {
            "python": platform.python_version(),
            "rdkit": rdkit.__version__,
            "torch": torch.__version__,
            "numpy": np.__version__,
        },
        "configuration": {
            "attempts_per_prompt": attempts,
            "seed": 0,
            "prompts": sorted(baseline),
            "sampler": official["sampler_config"],
            "control": {
                "enabled": True,
                "condition_initial_locked_family": True,
                "hard_lock_effective_chemistry": True,
            },
            "prior": {"candidates": prior.candidates, "strength": prior.strength},
        },
        "summary": summary,
        "baseline": baseline,
        "candidate": candidate,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, output)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.output)
    print(json.dumps(result["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
