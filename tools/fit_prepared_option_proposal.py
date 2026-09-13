"""Fit the unchanged proposal learner to prepared inverse-ring demonstrations.

Consumes immutable CPU preparation; performs no molecular replay or oracle call.
This is a ring-demonstration proposal ablation, not a future-value estimator.
"""

from __future__ import annotations

import argparse
import gzip
import json
import platform
import subprocess
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from rdkit import rdBase

from compose_v4.control.option_demonstrations import (
    DemonstrationFitConfig,
    append_ignored_context,
    batch_option_scores,
    descriptor_menu,
    fit_demonstration_actor,
    source_weights,
)
from compose_v4.control.option_features import (
    REGION_CONTEXT_NAMES,
    VALUE_CONTEXT_NAMES,
    structural_option_features,
)
from compose_v4.control.option_policy import (
    conservative_option_distribution,
    option_actor_parameter_id,
)
from compose_v4.control.option_selector import balanced_option_prior
from tools.ivg_winner_paths import implementation_closure, publish, sha
from tools.winner_option_proposal import evaluate

ROOT = Path(__file__).resolve().parents[1]


def run(args):
    if args.output.exists():
        raise ValueError(f"preserve existing fit: {args.output}")
    prepared = json.loads(args.prepared.read_text())
    data_path = args.prepared.parent / "demonstrations.json.gz"
    if (
        prepared["schema_version"] != "inverse_ring_preparation_v1"
        or sha(data_path) != prepared["demonstrations_sha256"]
    ):
        raise ValueError("prepared demonstration schema/hash mismatch")
    if prepared["software"]["rdkit"] != rdBase.rdkitVersion:
        raise ValueError("feature runtime must match preparation RDKit")
    rows = json.loads(gzip.decompress(data_path.read_bytes()))["rows"]
    train = [row for row in rows if row["role"] == "train"]
    held = [row for row in rows if row["role"] == "heldout_source_diagnostic"]
    if len(train) + len(held) != len(rows) or not train or not held:
        raise ValueError("prepared data has invalid or empty roles")
    if {row["source_id"] for row in train} & {row["source_id"] for row in held}:
        raise ValueError("source leakage in prepared data")
    if {row["smiles"] for row in train} & {row["smiles"] for row in held}:
        raise ValueError("intermediate-state leakage in prepared data")
    if {row["target_used_for_preparation_only"] for row in train} & {
        row["target_used_for_preparation_only"] for row in held
    }:
        raise ValueError("winner leakage in prepared data")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    start = perf_counter()
    config = DemonstrationFitConfig()
    names = descriptor_menu()
    reference = balanced_option_prior(names)
    actor, history = fit_demonstration_actor(
        [row["features"] for row in train],
        [names.index(row["option"]) for row in train],
        [row["source_id"] for row in train],
        names,
        reference,
        config=config,
    )
    fit_seconds = perf_counter() - start
    counts = np.zeros(len(names))
    for row, weight in zip(train, source_weights([row["source_id"] for row in train]), strict=True):
        counts[names.index(row["option"])] += weight
    marginal = (counts + reference) / 2
    marginal = np.asarray(
        conservative_option_distribution(names, reference, np.log(marginal / reference)).proposal
    )
    results, predictions = {}, {}
    for role, group in (("train", train), ("heldout_source_diagnostic", held)):
        results[role], predictions[role] = evaluate(actor, group, names, reference, marginal)
    # Profile the actual first-layer factorization against the existing serial
    # public forward call on a representative 32-molecule, full-menu batch.
    x = torch.tensor([row["features"] for row in train[:32]], dtype=torch.float32)
    options = torch.from_numpy(np.stack([structural_option_features(name) for name in names]))
    with torch.inference_mode():
        batch_option_scores(actor, x, options)  # warm both paths before timing
        torch.stack([actor(state, options) for state in x])
        before = perf_counter()
        for _ in range(10):
            serial = torch.stack([actor(state, options) for state in x])
        serial_seconds = (perf_counter() - before) / 10
        before = perf_counter()
        for _ in range(10):
            batched = batch_option_scores(actor, x, options)
        batch_seconds = (perf_counter() - before) / 10
        try:
            torch.testing.assert_close(serial, batched, rtol=1e-5, atol=2e-6)
            factored_gate = {"passed": True, "reason": None}
        except AssertionError as error:
            # Keep the original scorer when the optimization fails. Do not
            # loosen the registered tolerance to turn this into a pass.
            factored_gate = {"passed": False, "reason": str(error)}
    benchmark = {
        "molecules": len(x),
        "options": len(names),
        "repeats": 10,
        "serial_seconds": serial_seconds,
        "batched_seconds": batch_seconds,
        "max_absolute_difference": float((serial - batched).abs().max()),
        "argmax_equal": bool(torch.equal(serial.argmax(1), batched.argmax(1))),
        "equivalence_gate": factored_gate,
        "inference_implementation": "original_public_actor_forward",
    }
    runtime_actor = append_ignored_context(
        actor, len(REGION_CONTEXT_NAMES) + len(VALUE_CONTEXT_NAMES)
    )
    checkpoint = {
        "schema_version": "winner_demonstration_option_actor_v1",
        "evidence": "supervised proposal imitation only",
        "parameter_id": option_actor_parameter_id(runtime_actor),
        "state_dim": runtime_actor.state_dim,
        "option_dim": runtime_actor.option_dim,
        "hidden": config.hidden,
        "parameters": {
            name: value.detach().tolist()
            for name, value in sorted(runtime_actor.state_dict().items())
        },
        "ignored_runtime_context": [*VALUE_CONTEXT_NAMES, *REGION_CONTEXT_NAMES],
        "prepared_sha256": sha(args.prepared),
        "demonstrations_sha256": sha(data_path),
    }
    publish(args.output / "actor.json.gz", checkpoint, compressed=True)
    publish(args.output / "predictions.json.gz", predictions, compressed=True)
    nll = results["heldout_source_diagnostic"]["source_balanced_nll"]
    decision = (
        "conditioning_improves_retrospective_nll"
        if nll["actor"] < min(nll["reference"], nll["marginal"])
        else "do_not_promote_actor_on_this_evidence"
    )
    report = {
        "schema_version": "inverse_ring_proposal_fit_v1",
        "decision": decision,
        "evidence": "retrospective inverse-derived precursor classification, not autonomous search",
        "inputs_sha256": {
            str(args.prepared.resolve()): sha(args.prepared),
            str(data_path.resolve()): sha(data_path),
        },
        "implementation_sha256": implementation_closure(Path(__file__).resolve()),
        "code_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "working_tree_dirty": bool(
            subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip()
        ),
        "configuration": {
            **asdict(config),
            "max_kl": 1.0,
            "menu": names,
            "marginal_prior_weight": 0.5,
        },
        "split_sha256": prepared["split_sha256"],
        "exclusions": prepared["exclusions"],
        "software": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "rdkit": rdBase.rdkitVersion,
            "numpy": np.__version__,
        },
        "hardware": {
            "device": "CPU",
            "machine": platform.machine(),
            "threads": 1,
            "precision": "float32",
        },
        "results": results,
        "fit_history": history,
        "scoring_benchmark": benchmark,
        "costs": {
            "fit_seconds": fit_seconds,
            "total_seconds": perf_counter() - start,
            "new_docking_calls": 0,
            "molecular_replays": 0,
        },
        "outputs_sha256": {
            name: sha(args.output / name) for name in ("actor.json.gz", "predictions.json.gz")
        },
        "limitations": [
            "inverse-derived precursors, not benchmark initial seeds",
            "known winner data used explicitly",
            "no fresh final test",
            "diagnostic menu contains inapplicable distractors",
            "option choice only, not attachment or complete route recovery",
            "not a learned future value",
        ],
    }
    publish(args.output / "result.json", report)
    print(
        json.dumps(
            {
                "decision": decision,
                "heldout": results["heldout_source_diagnostic"],
                "benchmark": benchmark,
                "costs": report["costs"],
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())
