"""Prepare, fit and audit a target-free option proposal from saved IVG witnesses.

Local CPU development only. Previously inspected winners are retrospective data.
No docking, reference-law fitting, or future-value fitting occurs here.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import platform
import subprocess
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from rdkit import rdBase

from compose_v4.control.option_demonstrations import (
    DemonstrationFitConfig,
    append_ignored_context,
    descriptor_menu,
    fit_demonstration_actor,
    recognize_trace,
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
from compose_v4.control.trajectory_value import molecule_features
from compose_v4.rewrite.kernel import InvalidRewrite, canonical_state_key
from compose_v4.rewrite.trace_shard import decode_state
from tools.ivg_winner_paths import digest, implementation_closure, publish, sha

ROOT = Path(__file__).resolve().parents[1]
FOCUS_PAIR = "5705946f25f35dd88189b1204f01f294e96a43e78fd1c5030967c48cecceccc8"


def source_id(source):
    return hashlib.sha256(source.encode()).hexdigest()


@lru_cache(maxsize=8192)
def features(smiles):
    return molecule_features(smiles).tolist()


def evaluate(actor, rows, menu, reference, marginal):
    options = torch.from_numpy(np.stack([structural_option_features(name) for name in menu]))
    weights = source_weights([row["source_id"] for row in rows])
    probabilities, reference_probabilities, marginal_probabilities = [], [], []
    hits, marginal_hits, predictions = [], [], []
    started = perf_counter()
    with torch.inference_mode():
        for offset in range(0, len(rows), 32):
            batch = rows[offset : offset + 32]
            x = torch.tensor([row["features"] for row in batch], dtype=torch.float32)
            # The original public forward is the inference authority. The
            # factored optimization is not admitted until its numerical gate
            # passes on the fitted model, not only small random fixtures.
            scores = torch.stack([actor(state, options) for state in x]).numpy()
            for row, score in zip(batch, scores, strict=True):
                distribution = conservative_option_distribution(menu, reference, score)
                index = menu.index(row["option"])
                q = np.asarray(distribution.proposal)
                probabilities.append(float(q[index]))
                reference_probabilities.append(float(reference[index]))
                marginal_probabilities.append(float(marginal[index]))
                hits.append(int(np.argmax(q)) == index)
                marginal_hits.append(int(np.argmax(marginal)) == index)
                predictions.append(
                    {
                        "decision_id": row["decision_id"],
                        "source_id": row["source_id"],
                        "option": row["option"],
                        "compound": row["compound"],
                        "reference_probability": float(reference[index]),
                        "marginal_probability": float(marginal[index]),
                        "proposal_probability": float(q[index]),
                        "kl": distribution.kl,
                        "top_option": menu[int(np.argmax(q))],
                    }
                )
    p, r, m = map(np.asarray, (probabilities, reference_probabilities, marginal_probabilities))
    summary = {
        "decisions": len(rows),
        "sources": len({row["source_id"] for row in rows}),
        "source_balanced_nll": {
            "reference": float(-weights @ np.log(r)),
            "marginal": float(-weights @ np.log(m)),
            "actor": float(-weights @ np.log(p)),
        },
        "source_balanced_mean_demo_probability": {
            "reference": float(weights @ r),
            "marginal": float(weights @ m),
            "actor": float(weights @ p),
        },
        "source_balanced_top_label_agreement": {
            "reference": float(
                weights
                @ np.asarray(
                    [int(np.argmax(reference)) == menu.index(row["option"]) for row in rows]
                )
            ),
            "marginal": float(weights @ np.asarray(marginal_hits)),
            "actor": float(weights @ np.asarray(hits)),
        },
        "compound_decisions": sum(row["compound"] for row in rows),
        "seconds": perf_counter() - started,
    }
    return summary, predictions


def run(args):
    if args.output.exists():
        raise ValueError(f"preserve previous experiment: {args.output}")
    if rdBase.rdkitVersion != "2024.03.5":
        raise ValueError("use pinned RDKit 2024.03.5 for exact witness replay")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    started = perf_counter()
    config = DemonstrationFitConfig()
    audit = json.loads(args.audit.read_text())
    pairs = audit["pairs"]
    # Freeze source roles before reading trajectories or extracting labels.
    sources = sorted({source_id(row["source"]) for row in pairs})
    focus = source_id(next(row["source"] for row in pairs if row["pair_id"] == FOCUS_PAIR))
    ranked = sorted(
        (key for key in sources if key != focus), key=lambda key: digest([config.seed, key])
    )
    heldout = {focus, *ranked[: max(0, math.ceil(len(sources) / 4) - 1)]}
    roles = {key: "heldout_source_diagnostic" if key in heldout else "train" for key in sources}
    split = {
        "schema_version": "winner_option_source_split_v1",
        "source_roles": roles,
        "seed": config.seed,
        "focus_pair_excluded_from_fit": FOCUS_PAIR,
        "selection": "focus source plus SHA-ranked sources up to ceil(n_sources/4)",
        "audit_sha256": sha(args.audit),
        "evidence": "retrospective; all winners previously inspected, not a fresh final test",
    }
    publish(args.output / "split.json", split)
    menu = descriptor_menu()
    reference = balanced_option_prior(menu)
    rows, exclusions, inputs = [], [], {str(args.audit.resolve()): sha(args.audit)}
    recognized_counts = Counter()
    primitive_count = 0
    for pair in pairs:
        if pair["status"] != "witness_found":
            exclusions.append({"pair_id": pair["pair_id"], "reason": pair["status"]})
            continue
        receipt_path = args.audit.parent / pair["receipt"]
        actual = sha(receipt_path)
        if actual != pair["receipt_sha256"]:
            raise ValueError(f"{receipt_path}: receipt hash mismatch")
        inputs[str(receipt_path.resolve())] = actual
        receipt = json.loads(gzip.decompress(receipt_path.read_bytes()))
        if digest(receipt["payload"]) != receipt["payload_sha256"]:
            raise ValueError(f"{receipt_path}: payload hash mismatch")
        path = receipt["payload"]["path"]
        if (
            path["source_state"] != path["states"][0]
            or receipt["pair"]["pair_id"] != pair["pair_id"]
        ):
            raise ValueError(f"{receipt_path}: source or pair identity mismatch")
        # Older witnesses can fail pinned-runtime replay. Keep their exact
        # failure rather than repairing states, suppressing it, or fitting them.
        try:
            segments = recognize_trace(path["states"], path["actions"])
        except (InvalidRewrite, ValueError) as error:
            exclusions.append(
                {
                    "pair_id": pair["pair_id"],
                    "reason": "pinned_replay_rejected",
                    "detail": str(error),
                }
            )
            continue
        primitive_count += len(path["actions"])
        key = source_id(pair["source"])
        for segment in segments:
            if segment.option not in menu:
                raise ValueError(
                    f"{pair['pair_id']}: recognized descriptor absent from declared diagnostic menu"
                )
            smiles = canonical_state_key(decode_state(path["states"][segment.start]))
            recognized_counts[segment.option] += 1
            rows.append(
                {
                    "decision_id": f"{pair['pair_id']}:{segment.start}:{segment.stop}",
                    "pair_id": pair["pair_id"],
                    "source_id": key,
                    "role": roles[key],
                    "smiles": smiles,
                    "features": features(smiles),
                    **asdict(segment),
                    "receipt_sha256": actual,
                }
            )
    if not rows:
        raise ValueError("no exact-replayed demonstrations remain; do not fit")
    # Equalize source contribution after canonical decision deduplication.
    # Different valid option labels at one molecule remain distinct examples.
    unique = {}
    for row in rows:
        key = (row["source_id"], row["smiles"], row["option"])
        if key in unique:
            unique[key]["aliases"].append(row["decision_id"])
        else:
            unique[key] = {**row, "aliases": [row["decision_id"]]}
    rows = list(unique.values())
    train = [row for row in rows if row["role"] == "train"]
    held = [row for row in rows if row["role"] != "train"]
    if not train or not held:
        raise ValueError("source split has no retained train or heldout decisions")
    # Avoid identical intermediate-state leakage despite disjoint sources.
    train_states = {row["smiles"] for row in train}
    overlap = [row["decision_id"] for row in held if row["smiles"] in train_states]
    held = [row for row in held if row["smiles"] not in train_states]
    if not held:
        raise ValueError("no heldout decisions after intermediate-state overlap exclusion")
    prepare_seconds = perf_counter() - started
    publish(
        args.output / "demonstrations.json.gz",
        {"schema_version": "winner_option_demonstrations_v1", "rows": rows},
        compressed=True,
    )
    print(
        json.dumps(
            {
                "prepared_decisions": len(rows),
                "train": len(train),
                "heldout": len(held),
                "compound_segments": sum(row["compound"] for row in rows),
                "seconds": prepare_seconds,
            }
        ),
        flush=True,
    )
    fit_start = perf_counter()
    actor, history = fit_demonstration_actor(
        [row["features"] for row in train],
        [menu.index(row["option"]) for row in train],
        [row["source_id"] for row in train],
        menu,
        reference,
        config=config,
    )
    fit_seconds = perf_counter() - fit_start
    # One prior pseudo-observation per source; no heldout frequencies are used.
    counts = np.zeros(len(menu))
    weights = source_weights([row["source_id"] for row in train])
    for row, weight in zip(train, weights, strict=True):
        counts[menu.index(row["option"])] += weight
    marginal = (counts + reference) / 2
    marginal = np.asarray(
        conservative_option_distribution(menu, reference, np.log(marginal / reference)).proposal
    )
    results, predictions = {}, {}
    for role, selected in (("train", train), ("heldout_source_diagnostic", held)):
        results[role], predictions[role] = evaluate(actor, selected, menu, reference, marginal)
    runtime_actor = append_ignored_context(
        actor, len(VALUE_CONTEXT_NAMES) + len(REGION_CONTEXT_NAMES)
    )
    parameters = {
        name: tensor.detach().tolist()
        for name, tensor in sorted(runtime_actor.state_dict().items())
    }
    checkpoint = {
        "schema_version": "winner_demonstration_option_actor_v1",
        "evidence": "supervised proposal imitation only",
        "parameter_id": option_actor_parameter_id(runtime_actor),
        "state_dim": runtime_actor.state_dim,
        "option_dim": runtime_actor.option_dim,
        "hidden": config.hidden,
        "parameters": parameters,
        "ignored_runtime_context": [*VALUE_CONTEXT_NAMES, *REGION_CONTEXT_NAMES],
        "split_sha256": sha(args.output / "split.json"),
        "demonstrations_sha256": sha(args.output / "demonstrations.json.gz"),
    }
    publish(args.output / "actor.json.gz", checkpoint, compressed=True)
    publish(args.output / "predictions.json.gz", predictions, compressed=True)
    nll = results["heldout_source_diagnostic"]["source_balanced_nll"]
    report = {
        "schema_version": "winner_option_proposal_diagnostic_v1",
        "evidence": "retrospective source-heldout option imitation; not autonomous discovery or docking improvement",
        "configuration": {
            **asdict(config),
            "max_kl": 1.0,
            "menu": menu,
            "marginal_prior_weight": 0.5,
        },
        "inputs_sha256": dict(sorted(inputs.items())),
        "split": split,
        "implementation_sha256": implementation_closure(Path(__file__).resolve()),
        "code_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "working_tree_dirty": bool(
            subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip()
        ),
        "software": {
            "python": platform.python_version(),
            "rdkit": rdBase.rdkitVersion,
            "torch": torch.__version__,
            "numpy": np.__version__,
        },
        "hardware": {
            "device": "CPU",
            "machine": platform.machine(),
            "threads": 1,
            "precision": "float32",
        },
        "seed_derivation": "fixed fit seed; SHA-ranked source split; independent minibatch generator",
        "access_basis_and_upstream": audit["access_basis_and_upstream"],
        "counts": {
            "pairs": len(pairs),
            "retained_pairs": len({row["pair_id"] for row in rows}),
            "primitive_steps": primitive_count,
            "deduplicated_decisions": len(rows),
            "compound_decisions": sum(row["compound"] for row in rows),
            "recognized_options_before_dedup": dict(sorted(recognized_counts.items())),
        },
        "exclusions": exclusions,
        "heldout_intermediate_overlap_exclusions": overlap,
        "results": results,
        "fit_history": history,
        "costs": {
            "prepare_seconds": prepare_seconds,
            "fit_seconds": fit_seconds,
            "total_seconds": perf_counter() - started,
            "new_docking_calls": 0,
            "gpu_seconds": 0,
            "reference_law_evaluations": 0,
        },
        "outputs_sha256": {
            name: sha(args.output / name)
            for name in (
                "split.json",
                "demonstrations.json.gz",
                "actor.json.gz",
                "predictions.json.gz",
            )
        },
        "decision": "conditioning_improves_retrospective_nll"
        if nll["actor"] < min(nll["marginal"], nll["reference"])
        else "do_not_promote_actor_on_this_evidence",
        "limitations": [
            "all source winners were previously inspected",
            "offline descriptor row is not product-applicability filtered",
            "contiguous witness segmentation is not an optimal macro decomposition",
            "source-level generalization is not scaffold-disjoint generalization",
            "demonstration probability is not docking value or autonomous route recovery",
            "proposal training is winner-informed and must be separately declared in benchmark comparisons",
        ],
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    publish(args.output / "result.json", report)
    print(
        json.dumps(
            {
                "decision": report["decision"],
                "heldout": results["heldout_source_diagnostic"],
                "costs": report["costs"],
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--audit", type=Path, default=ROOT / "diagnostics/ivg_winner_paths/audit.json"
    )
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())
