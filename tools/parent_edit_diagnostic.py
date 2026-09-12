"""Chronological, already-scored parent/edit selection audit; zero new oracles."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from time import perf_counter

import numpy as np
from rdkit import rdBase

from compose_v4.control.docking_value import identity
from compose_v4.control.edit_learning_data import mutation_contrast_panel, observed_interaction
from compose_v4.control.edit_program import EditProgram
from compose_v4.control.edit_program_graph import compile_program_graph, execute_program_graph
from compose_v4.control.parent_edit_model import (
    MUTATION_RECIPE,
    RECIPE,
    ParentEditFeatures,
    ParentEditModel,
    select_parent_edits,
)
from compose_v4.control.program_decomposition import verified_branches
from compose_v4.control.program_mutation import branch_components, replace_branch, select_branch
from compose_v4.control.program_task import ProgramTask
from compose_v4.experiments.continuation_profile import publish_json
from compose_v4.experiments.t4_matched_pilot import unseal
from compose_v4.rewrite.kernel import canonical_state_key
from compose_v4.rewrite.trace_shard import decode_state
from tools.ivg_winner_paths import implementation_closure, sha

ROOT = Path(__file__).resolve().parents[1]
CELLS = ("jak2_1", "fa7_0", "braf_1", "5ht1b_0")


def fit_next(output):
    """Use all measured mutation outcomes/repeats for future decisions only."""
    started, inputs, summaries = perf_counter(), {}, []
    for cell in CELLS:
        initial_path = ROOT / f"diagnostics/t4_program_curriculum/attempt_1/{cell}_archive.json"
        initial = json.loads(initial_path.read_text())
        inputs[str(initial_path.relative_to(ROOT))] = sha(initial_path)
        domain = initial["optimizer"]["oracle_protocol"]
        entries, observations = {}, {}
        for arm in ("score_rank", "score_blind"):
            path = ROOT / f"diagnostics/t4_second_generation/scoring_1/{cell}_{arm}_archive.json"
            inputs[str(path.relative_to(ROOT))] = sha(path)
            archive = json.loads(path.read_text())["optimizer"]
            if (
                identity({k: v for k, v in archive.items() if k != "snapshot_id"})
                != archive["snapshot_id"]
            ):
                raise ValueError(f"corrupt measured archive: {path}")
            if archive["oracle_protocol"] != domain:
                raise ValueError("training archives have different target protocols")
            entries.update(archive["entries"])
            for key, row in archive["observations"].items():
                if key in observations and observations[key] != row:
                    raise ValueError("cross-arm observation receipt conflict")
                observations[key] = row
        feature, training = ParentEditFeatures(), []
        for entry in entries.values():
            parent_id = entry.get("provenance", {}).get("entry_id")
            parent = entries.get(parent_id) if parent_id is not None else None
            if parent_id is not None and parent is None:
                raise ValueError("measured mutation lost its exact selected parent")
            if parent is not None:
                state = parent["trace"]["states"][-1]
                parent_scores = [-entry["provenance"]["parent_measured_score"]]
            else:
                state = entry["source_state"]
                parent_scores = [
                    -r["ds"]
                    for r in initial["initialization_receipts"]
                    if r["role"] == "seed_control"
                ]
            values = feature.with_mutation_context(
                entry, parent_state=state, parent_scores=parent_scores, parent_record=parent
            )
            for receipt, observation in observations.items():
                if observation["endpoint"] == entry["endpoint"]:
                    training.append(
                        {
                            "endpoint": entry["endpoint"],
                            "features": values,
                            "receipt_id": receipt,
                            "utility": -observation["score"],
                            "oracle_protocol": domain,
                        }
                    )
        model = ParentEditModel.fit(
            training, oracle_protocol=domain, input_sha256=identity(inputs), recipe=MUTATION_RECIPE
        )
        publish_json(output / f"{cell}_model.json", model.payload)
        summaries.append(
            {
                "cell": cell,
                "unique_endpoints": model.payload["training_endpoint_count"],
                "program_representations": len(entries),
                "measured_observations": len(observations),
                "model_sha256": model.payload["model_sha256"],
                "evaluation": "none; do not evaluate on these training pools",
            }
        )
    result = {
        "schema_version": "parent_edit_next_fit_v1",
        "recipe": MUTATION_RECIPE,
        "summaries": summaries,
        "inputs": inputs,
        "implementation": implementation_closure(Path(__file__).resolve()),
        "git_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "rdkit": rdBase.rdkitVersion,
        },
        "hardware": {
            "device": "cpu",
            "precision": "float64",
            "workers": 1,
            "machine": platform.machine(),
        },
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "new_oracle_calls": 0,
        "seconds": perf_counter() - started,
        "qualification": "unqualified next-decision models, not performance evidence",
        "label_policy": "all actual repeats retained, mean per endpoint, no best-repeat target",
    }
    publish_json(output / "result.json", {**result, "result_sha256": identity(result)})
    print(json.dumps(summaries, indent=2))


def donation_witness(admitted):
    attempts = []
    for cell, records in admitted.items():
        known = {row[0]["endpoint"] for row in records}
        for entry, program, binding in records:
            source = decode_state(entry["source_state"])
            for donor, donor_program, donor_binding in records:
                if len(attempts) >= 64:
                    return None, attempts
                if (
                    donor["entry_id"] == entry["entry_id"]
                    or donor["source_state"] != entry["source_state"]
                ):
                    continue
                removed = branch_components(program)[0]
                part, roots = select_branch(donor_program, branch_components(donor_program)[-1])
                try:
                    changed, assigned = replace_branch(
                        source,
                        program,
                        binding,
                        removed,
                        part,
                        tuple(donor_binding[i] for i in roots),
                        max_primitives=32,
                        max_blocks=8,
                    )
                    _, trace = execute_program_graph(
                        source,
                        compile_program_graph(changed),
                        assigned,
                        max_primitives=32,
                        max_blocks=8,
                    )
                except ValueError as error:
                    attempts.append({"cell": cell, "status": "rejected", "reason": str(error)})
                    continue
                attempts.append({"cell": cell, "status": "executed", "endpoint": trace["endpoint"]})
                if trace["endpoint"] not in known:
                    return {
                        "cell": cell,
                        "recipient": entry["entry_id"],
                        "donor": donor["entry_id"],
                        "removed_branch": removed,
                        "retained_blocks": len(program.blocks) - len(removed),
                        "program": changed.payload(),
                        "assignment": assigned,
                        "trace": trace,
                        "scored": False,
                        "claim": "new executable endpoint, not a docking improvement",
                    }, attempts
    return None, attempts


def braf_contrast(output, inputs):
    """Replay the already reported BRAF edit pair, not an autonomous success test."""
    paths = (
        "diagnostics/t4_program_curriculum/attempt_1/braf_1_archive.json",
        "diagnostics/t4_second_generation/attempt_2/braf_1_score_blind_batch.json",
        "diagnostics/t4_second_generation/scoring_1/remote_result.json",
    )
    for relative in paths:
        inputs[relative] = sha(ROOT / relative)
    archive = json.loads((ROOT / paths[0]).read_text())["optimizer"]
    pool = json.loads((ROOT / paths[1]).read_text())
    scored = unseal(ROOT / paths[2])
    labels = {r["smiles"]: r["ds"] for r in scored["rows"] if r["cell"] == "braf_1"}
    winner = min(pool["candidates"], key=lambda c: (labels[c["endpoint"]], c["candidate_id"]))
    entry = archive["entries"][winner["provenance"]["entry_id"]]
    edits = winner["provenance"]["metadata"]["mutations"]
    if [r["kind"] for r in edits] != ["extend_segment", "attachment"]:
        raise ValueError("the declared BRAF paired-mutation example changed")
    task = ProgramTask(
        "braf_1",
        archive["oracle_protocol"],
        "t4",
        canonical_state_key(decode_state(entry["source_state"])),
        0.4,
    )
    panel = mutation_contrast_panel(
        entry,
        attachment=winner["assignment"],
        parameter_move=edits[0]["kind"],
        parameter_choice=edits[0]["choice"],
        eligibility=task.endpoint_evaluator(),
    )
    if panel["arms"]["joint"].get("endpoint") != winner["endpoint"]:
        raise ValueError("BRAF contrast does not reproduce the recorded joint mutation")
    observations = [
        {**row, "receipt_id": receipt} for receipt, row in archive["observations"].items()
    ]
    observations.extend(
        {
            "endpoint": row["smiles"],
            "score": row["ds"],
            "oracle_protocol": row["oracle_protocol"],
            "receipt_id": identity(row),
        }
        for row in scored["rows"]
        if row["cell"] == "braf_1" and row["ds"] is not None
    )
    interpretation = observed_interaction(panel, observations, task=task)
    publish_json(
        output / "braf_contrast.json",
        {
            "panel": panel,
            "existing_observation_result": interpretation,
            "inputs": {p: inputs[p] for p in paths},
            "selection_role": "retrospective known successful joint edit",
        },
    )
    return {
        "arms": {
            name: {"status": row["status"], "endpoint": row.get("endpoint")}
            for name, row in panel["arms"].items()
        },
        **interpretation,
    }


def branch_audit(output):
    """Re-evaluate only the changed decomposition, not completed model fits."""
    started, reports, inputs, admitted = perf_counter(), [], {}, {}
    for cell in CELLS:
        path = ROOT / f"diagnostics/t4_program_curriculum/attempt_1/{cell}_archive.json"
        inputs[str(path.relative_to(ROOT))] = sha(path)
        archive = json.loads(path.read_text())["optimizer"]
        if (
            identity({k: v for k, v in archive.items() if k != "snapshot_id"})
            != archive["snapshot_id"]
        ):
            raise ValueError(f"corrupt measured archive: {path}")
        for entry in archive["entries"].values():
            program, assignment, report = verified_branches(
                decode_state(entry["source_state"]),
                EditProgram.from_payload(entry["program"]),
                tuple(entry["assignment"]),
                max_primitives=32,
            )
            reports.append(
                {
                    "cell": cell,
                    "entry_id": entry["entry_id"],
                    **report,
                    "program": program.payload() if report["status"] == "verified" else None,
                    "assignment": assignment if report["status"] == "verified" else None,
                }
            )
            if report["status"] == "verified":
                admitted.setdefault(cell, []).append((entry, program, assignment))
    donation, attempts = donation_witness(admitted)
    contrast = braf_contrast(output, inputs)
    result = {
        "schema_version": "measured_program_branch_audit_v1",
        "reports": reports,
        "branch_donation": donation,
        "donation_attempts": attempts,
        "braf_contrast": contrast,
        "inputs": inputs,
        "implementation": implementation_closure(Path(__file__).resolve()),
        "git_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "rdkit": rdBase.rdkitVersion,
        },
        "hardware": {"device": "cpu", "workers": 1, "machine": platform.machine()},
        "randomization": "none; deterministic observed-write components and executor replay",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "new_oracle_calls": 0,
        "seconds": perf_counter() - started,
        "verified": sum(r["status"] == "verified" for r in reports),
        "attempted": len(reports),
        "evidence_role": "retrospective structural decomposition, not docking improvement",
    }
    publish_json(output / "result.json", {**result, "result_sha256": identity(result)})
    print(
        json.dumps({k: result[k] for k in ("attempted", "verified", "seconds", "new_oracle_calls")})
    )


def run(output):
    started = perf_counter()
    inputs = {}

    def load(relative):
        path = ROOT / relative
        inputs[relative] = sha(path)
        return json.loads(path.read_text())

    score_path = "diagnostics/t4_second_generation/scoring_1/remote_result.json"
    load(score_path)
    scored = unseal(ROOT / score_path)
    labels = {(r["cell"], r["smiles"]): r for r in scored["rows"]}
    summaries, branches = [], []
    for cell in CELLS:
        saved = load(f"diagnostics/t4_program_curriculum/attempt_1/{cell}_archive.json")
        archive = saved["optimizer"]
        if (
            identity({k: v for k, v in archive.items() if k != "snapshot_id"})
            != archive["snapshot_id"]
        ):
            raise ValueError(f"corrupt initial measured archive: {cell}")
        domain = archive["oracle_protocol"]
        feature = ParentEditFeatures()
        training = []
        seed_rows = [r for r in saved["initialization_receipts"] if r["role"] == "seed_control"]
        if len(seed_rows) != 1 or seed_rows[0]["ds"] is None:
            raise ValueError(f"missing unique first-generation seed observation: {cell}")
        for entry in archive["entries"].values():
            observations = [
                (k, v)
                for k, v in archive["observations"].items()
                if v["endpoint"] == entry["endpoint"]
            ]
            features = feature(
                entry, parent_state=entry["source_state"], parent_scores=[-seed_rows[0]["ds"]]
            )
            for receipt_id, observed in observations:
                training.append(
                    {
                        "features": features,
                        "endpoint": entry["endpoint"],
                        "receipt_id": receipt_id,
                        "utility": -observed["score"],
                        "oracle_protocol": domain,
                    }
                )
            _, _, report = verified_branches(
                decode_state(entry["source_state"]),
                EditProgram.from_payload(entry["program"]),
                tuple(entry["assignment"]),
                max_primitives=32,
            )
            branches.append({"cell": cell, "entry_id": entry["entry_id"], **report})
        model = ParentEditModel.fit(training, oracle_protocol=domain, input_sha256=identity(inputs))
        publish_json(output / f"{cell}_model.json", model.payload)
        archive_utility = [
            float(
                np.mean(
                    [-v["score"] for v in archive["observations"].values() if v["endpoint"] == s]
                )
            )
            for s in sorted({v["endpoint"] for v in archive["observations"].values()})
        ]
        for arm in ("score_rank", "score_blind"):
            pool = load(f"diagnostics/t4_second_generation/attempt_2/{cell}_{arm}_batch.json")
            body = {
                k: v
                for k, v in pool.items()
                if k not in ("batch_id", "proposal_seconds", "new_oracle_calls")
            }
            if identity(body) != pool["batch_id"]:
                raise ValueError(f"corrupt completed candidate lock: {cell}/{arm}")
            candidates = pool["candidates"]
            if {r["endpoint"] for r in candidates} & set(model.payload["training_endpoints"]):
                raise ValueError(f"chronological endpoint leakage in {cell}/{arm}")
            features, outcomes = [], []
            for candidate in candidates:
                parent = archive["entries"][candidate["provenance"]["entry_id"]]
                parent_scores = [
                    -v["score"]
                    for v in archive["observations"].values()
                    if v["endpoint"] == parent["endpoint"]
                ]
                features.append(
                    feature(
                        candidate,
                        parent_state=parent["trace"]["states"][-1],
                        parent_scores=parent_scores,
                    )
                )
                observed = labels[cell, candidate["endpoint"]]
                if observed["ds"] is None or observed["oracle_protocol"] != domain:
                    raise ValueError(
                        f"missing or mismatched completed query: {cell}/{candidate['candidate_id']}"
                    )
                outcomes.append(-observed["ds"])
            prediction, disagreement = model.predict(features, oracle_protocol=domain)
            count = min(4, len(candidates))
            selection = select_parent_edits(
                [c["candidate_id"] for c in candidates],
                prediction,
                archive_utility,
                k=1,
                count=count,
                seed=RECIPE["seed"],
                mode="learned",
                diagnostic=True,
            )
            observed = np.array(outcomes)
            subsets = list(combinations(range(len(candidates)), count))
            uniform_best = np.array([max(observed[list(indices)]) for indices in subsets])
            chosen_best = float(max(observed[selection["indices"]]))
            incumbent = max(archive_utility)
            summaries.append(
                {
                    "cell": cell,
                    "arm_pool": arm,
                    "pool_count": len(candidates),
                    "training_unique_endpoints": len(model.payload["training_endpoints"]),
                    "model_sha256": model.payload["model_sha256"],
                    "selection": selection,
                    "uniform_subset_count": len(subsets),
                    "best_selected_docking": -chosen_best,
                    "uniform_expected_best_docking": -float(np.mean(uniform_best)),
                    "selected_utility_minus_uniform": chosen_best - float(np.mean(uniform_best)),
                    "archive_gain_minus_uniform": max(chosen_best - incumbent, 0)
                    - float(np.mean(np.maximum(uniform_best - incumbent, 0))),
                    "candidate_predictions": [
                        {
                            "candidate_id": c["candidate_id"],
                            "predicted_utility": float(p),
                            "uncalibrated_disagreement": float(d),
                            "observed_docking": -u,
                        }
                        for c, p, d, u in zip(
                            candidates, prediction, disagreement, outcomes, strict=True
                        )
                    ],
                }
            )
    result = {
        "schema_version": "parent_edit_chronological_diagnostic_v1",
        "recipe": RECIPE,
        "evidence_role": "retrospective development; all evaluation pools had already been inspected",
        "split": "first-generation observations only for fit; second-generation first queries for audit; no confirmation labels fitted",
        "inputs": inputs,
        "implementation": implementation_closure(Path(__file__).resolve()),
        "git_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "working_tree_sources_bound_by_content_hashes": True,
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "rdkit": rdBase.rdkitVersion,
        },
        "hardware": {
            "machine": platform.machine(),
            "device": "cpu",
            "precision": "float64",
            "workers": 1,
        },
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "new_oracle_calls": 0,
        "summaries": summaries,
        "decomposition": branches,
        "positive_archive_gain_pools": sum(
            r["archive_gain_minus_uniform"] > 1e-9 for r in summaries
        ),
        "negative_archive_gain_pools": sum(
            r["archive_gain_minus_uniform"] < -1e-9 for r in summaries
        ),
        "verified_decompositions": sum(r["status"] == "verified" for r in branches),
        "seconds": perf_counter() - started,
        "decision": "not qualified for production guidance; prospective matched support/budget evidence still required",
    }
    publish_json(output / "result.json", {**result, "result_sha256": identity(result)})
    print(
        json.dumps(
            {
                k: result[k]
                for k in (
                    "new_oracle_calls",
                    "seconds",
                    "verified_decompositions",
                    "positive_archive_gain_pools",
                    "negative_archive_gain_pools",
                )
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--branch-audit-only", action="store_true")
    parser.add_argument("--fit-next", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("refusing to overwrite an existing diagnostic directory")
    if args.branch_audit_only and args.fit_next:
        raise SystemExit("choose one diagnostic mode")
    (fit_next if args.fit_next else branch_audit if args.branch_audit_only else run)(args.output)
