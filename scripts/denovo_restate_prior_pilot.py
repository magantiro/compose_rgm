"""Bounded, reversible C1 atom-restate fluorine-prior pilot.

This changes a proposal logit before sampling, never endpoint acceptance or
executor legality. Existing C1 rows at the same seeds are the paired control.
The output is exploratory development evidence, not an official benchmark row.
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
from denovo_postring_quality_audit import characterize, sha256
from evaluate_tracelet_rollouts import load_factorized_rollout_checkpoint

from compose_v4.chem.molecular_graph import ELEMENT_TO_IDX
from compose_v4.eval.denovo_benchmark import denovo_benchmark_metrics
from compose_v4.eval.denovo_ring_marginal import RingSystemPlanPrior
from compose_v4.experiments.denovo_ring_plan import (
    CatalogSignatureIndex,
    sample_denovo_arm,
)
from compose_v4.rewrite.kernel import de_novo_rewrite_system

SCHEMA = "denovo_restate_prior_pilot_v1"


def canonical_payload_hash(payload: dict) -> str:
    data = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(data).hexdigest()


def verify_contract(path: Path) -> dict:
    contract = json.loads(path.read_text())
    payload = contract["payload"]
    observed = canonical_payload_hash(payload)
    if observed != contract["payload_sha256"]:
        raise ValueError(f"contract payload SHA-256 mismatch: {path}")
    for label, source in payload["inputs"].items():
        source_path = Path(source["path"])
        actual = sha256(source_path)
        if actual != source["sha256"]:
            raise ValueError(f"{label} SHA-256 mismatch: {source_path}: {actual}")
    if sha256(Path(__file__)) != payload["implementation_sha256"]["pilot_script"]:
        raise ValueError("pilot script changed after protocol freeze")
    return contract


def _atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _summarize(rows: list[dict]) -> dict:
    metrics = denovo_benchmark_metrics([row["canonical_smiles"] for row in rows])
    return {
        "official_like_metrics": metrics,
        "mean_F_atoms": sum(row["elements"].get("F", 0) for row in rows) / len(rows),
        "fraction_with_F": sum(row["elements"].get("F", 0) > 0 for row in rows) / len(rows),
        "fraction_with_strained_ring": sum(3 in row["rings"] or 4 in row["rings"] for row in rows)
        / len(rows),
        "fraction_with_nonaromatic_unsaturated_5_6_ring": sum(
            row["nonaromatic_unsaturated_5_6_rings"] > 0 for row in rows
        )
        / len(rows),
        "mean_sa_fragment_score": sum(row["sa"]["fragment_score"] for row in rows) / len(rows),
    }


def run(config_path: Path, output: Path) -> dict:
    contract = verify_contract(config_path)
    payload = contract["payload"]
    paths = {label: Path(source["path"]) for label, source in payload["inputs"].items()}
    saved = json.loads(paths["baseline_records"].read_text())
    baseline_by_index = {row["index"]: row for row in saved if row["arm"] == "C1"}
    indices = [int(index) for index in payload["selected_indices"]]
    if len(indices) != len(set(indices)) or any(
        index not in baseline_by_index for index in indices
    ):
        raise ValueError("selected indices must be unique saved C1 identities")
    torch.set_num_threads(1)
    model, checkpoint = load_factorized_rollout_checkpoint(str(paths["checkpoint"]))
    model.eval()
    element = ELEMENT_TO_IDX["F"]
    f_classes = [
        index
        for index in range(len(model.atom_vocabulary))
        if model.atom_vocabulary.element_of(index) == element
    ]
    if len(f_classes) != 1:
        raise ValueError(f"expected one fluorine restate class; got {f_classes}")
    prior_before = model.atom_restate_log_prior.clone()
    with torch.no_grad():
        model.atom_restate_log_prior[f_classes[0]] += float(payload["F_restate_logit_delta"])
    index = CatalogSignatureIndex.build(model.ring_system_templates)
    plan_prior = RingSystemPlanPrior.read(paths["ring_prior"])
    runtime = de_novo_rewrite_system()
    output.mkdir(parents=True, exist_ok=True)
    results = []
    for position, selected_index in enumerate(indices):
        reference = baseline_by_index[selected_index]
        path = output / f"index_{selected_index:04d}.json"
        if path.exists():
            cached = json.loads(path.read_text())
            if cached["contract_payload_sha256"] != contract["payload_sha256"]:
                raise ValueError(f"stale pilot result under a different contract: {path}")
            if cached["trajectory_seed"] != reference["trajectory_seed"]:
                raise ValueError(f"pilot seed mismatch: {path}")
            results.append(cached)
            continue
        sampled = sample_denovo_arm(
            model,
            arm="C1",
            rng=np.random.default_rng(reference["trajectory_seed"]),
            source_prior=checkpoint["tree_source_prior"],
            index=index,
            plan_prior=plan_prior,
            n_slots=40,
            operational_horizon=16.0,
            time_step=0.1,
            max_events=128,
            runtime=runtime,
        )
        sampled["index"] = selected_index
        sampled["trajectory_seed"] = reference["trajectory_seed"]
        described = characterize(sampled, "C1_F_restate_prior", path)
        result = {
            "contract_payload_sha256": contract["payload_sha256"],
            "index": selected_index,
            "trajectory_seed": reference["trajectory_seed"],
            "baseline_smiles": reference["canonical_smiles"],
            "endpoint": described,
            "event_rules": sampled["event_rules"],
            "ring_plan": sampled["ring_plan"],
            "valid_state": sampled["valid_state"],
            "connected": sampled["connected"],
        }
        _atomic_json(path, result)
        if (
            not sampled["valid_state"]
            or not sampled["connected"]
            or described["group"] == "NO_ENDPOINT"
        ):
            raise RuntimeError(f"committed endpoint validity failed at index {selected_index}")
        results.append(result)
        if (position + 1) % 4 == 0:
            print(f"completed {position + 1}/{len(indices)}", flush=True)
    with torch.no_grad():
        model.atom_restate_log_prior.copy_(prior_before)
    baseline = [baseline_by_index[selected_index] for selected_index in indices]
    pilot = [result["endpoint"] for result in results]
    report = {
        "schema_version": SCHEMA,
        "evidence_class": "bounded_zero_oracle_development_pilot",
        "contract_payload_sha256": contract["payload_sha256"],
        "code_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "versions": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "rdkit": rdkit.__version__,
        },
        "attempted_per_arm": len(indices),
        "selected_indices": indices,
        "baseline": _summarize(baseline),
        "F_restate_prior": _summarize(pilot),
        "all_committed_valid_and_connected": all(
            result["valid_state"] and result["connected"] for result in results
        ),
    }
    _atomic_json(output / "report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run(args.contract, args.output)
    print(
        json.dumps(
            {
                "attempted_per_arm": report["attempted_per_arm"],
                "baseline": report["baseline"],
                "F_restate_prior": report["F_restate_prior"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
