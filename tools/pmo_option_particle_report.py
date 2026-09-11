"""Audit saved particle trajectories and chemistry with no oracle/model calls."""

import argparse
import hashlib
import json
import tarfile
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator, rdMolDescriptors

from compose_v4.control.archive_allocation import sample_archive_parents
from compose_v4.control.docking_value import identity
from compose_v4.control.donor_memory import build_memory, proposal_identity
from compose_v4.control.local_endpoint_selector import active as local_active
from compose_v4.control.local_endpoint_selector import policy_identity as local_policy_identity
from compose_v4.control.option_particles import advance, log_potentials
from compose_v4.experiments.continuation_profile import publish_json, sha256_file
from compose_v4.experiments.inference_package import software
from compose_v4.experiments.pmo_branch_policy import worker_identity


def quantiles(values):
    return (
        dict(zip(("min", "median", "max"), map(float, np.quantile(values, [0, 0.5, 1]))))
        if values
        else None
    )


def proposal_depths(parents, proposed, depths):
    """Count new edits along sampled ancestry, not a shared historical prefix."""
    result = []
    for parent, child, depth in zip(parents, proposed, depths, strict=True):
        if child is None:
            result.append(None)
            continue
        if (
            parent is None
            or depth is None
            or child["chain"] != parent["chain"] + [child["id"]]
            or child["primitives"] - parent["primitives"] != child["primitive_count"]
            or child["primitive_count"] < 1
        ):
            raise ValueError("new primitive depth lacks a matching exact parent chain/count")
        result.append(depth + child["primitive_count"])
    return result


def slot_policy_identity(configuration, arm, step, slot, broad_policy, memory_id=None):
    """Identify the actually drawn channel without merging local and broad work."""
    if arm not in configuration.get("local_selector_arms", []) or not local_active(
        configuration["seed"], step, slot
    ):
        return broad_policy
    if memory_id is None:
        raise ValueError("local selection receipt requires its frozen donor memory")
    return local_policy_identity(memory_id, configuration["endpoint_model_sha256"])


def report(path):
    r = json.loads(path.read_text())
    c, n = r["configuration"], r["configuration"]["particles"]
    if identity({k: v for k, v in c.items() if k != "contract_sha256"}) != c["contract_sha256"]:
        raise ValueError("saved particle contract hash mismatch")
    archive = path.with_name("source_snapshot.tar.gz")
    with tarfile.open(archive, "r:gz") as tar:
        for p, h in r["image_revision"]["serialized_sources"].items():
            if hashlib.sha256(tar.extractfile(p).read()).hexdigest() != h:
                raise ValueError(f"executed source archive mismatch: {p}")
        prepared = tar.extractfile(c["prepared"]["path"]).read()
        if hashlib.sha256(prepared).hexdigest() != c["prepared"]["sha256"]:
            raise ValueError("prepared score history does not match the contract")
        for key in ("model", "training"):
            if key in c:
                payload = tar.extractfile(c[key]["path"]).read()
                if hashlib.sha256(payload).hexdigest() != c[key]["sha256"]:
                    raise ValueError(f"executed proposal {key} differs from the contract")
        prepared_data = json.loads(prepared)
        history = dict(prepared_data["observed"])
    for q in r["oracle_rows"]:
        if q["smiles"] in history or q["status"] != "complete":
            raise ValueError("repeated historical query or incomplete oracle receipt")
        history[q["smiles"]] = q["score"]
    if (
        len(r["oracle_rows"]) != r["new_oracle_calls"]
        or len(r["oracle_rows"]) > c["new_oracle_limit"]
    ):
        raise ValueError("oracle census differs or exceeds the locked budget")
    rows = []
    fresh_scores = {q["smiles"]: q["score"] for q in r["oracle_rows"]}
    max_replay_error = 0.0
    fpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    for arm in c["arms"]:
        selection_mode = c.get("selection_modes", {}).get(arm, arm)
        weights, previous = [-float(np.log(n))] * n, [0.0] * n
        proposals = []
        offspring_deltas = []
        selected = []
        live_before = [True] * n
        parents, depths, observed_depths = r["initial_parents"], [0] * n, []
        archive_mode = c.get("parent_selection_modes", {}).get(arm) == "archive"
        archive_nodes = {p["smiles"]: p for p in parents}
        archive_depths = {p["smiles"]: 0 for p in parents}
        used_workers = set()
        memory_audits = []
        local_audits = []
        attempts = failures = 0
        for round_row in r["rounds"]:
            a = round_row["arms"][arm]
            if any(p is not None and history[p["smiles"]] != p["score"] for p in a["proposals"]):
                raise ValueError("proposal score differs from its actual oracle receipt")
            step = round_row["boundary"]
            policy_id = c.get("proposal_ids", {}).get(arm)
            memory_id = None
            if c.get("donor_memory_modes"):
                memory = build_memory(
                    prepared_data["initial_donor_memory"],
                    archive_nodes,
                    mode=c["donor_memory_modes"][arm],
                )
                policy_id = proposal_identity(memory["memory_id"])
                memory_id = memory["memory_id"]
                initial_donors = {p["smiles"] for p in prepared_data["initial_donor_memory"]}
                donors_used = []
                donor_draws = []
                for slot, child in enumerate(a["proposals"]):
                    draw_rng = np.random.default_rng(
                        np.random.SeedSequence([c["seed"], step, slot, 776])
                    )
                    if draw_rng.random() >= c["memory_recipe"]["donor_probability"]:
                        continue
                    index = int(draw_rng.choice(len(memory["rows"]), p=memory["probabilities"]))
                    selected_donor = memory["rows"][index]["smiles"]
                    if child is not None and (
                        child["bundle"]["option"] != "donor_transplant"
                        or child["bundle"]["donor_index"] != index
                    ):
                        raise ValueError("completed donor selection differs from its RNG draw")
                    donor_draws.append(
                        {
                            "slot": slot,
                            "smiles": selected_donor,
                            "new_donor": selected_donor not in initial_donors,
                            "completed": child is not None,
                        }
                    )
                for child in a["proposals"]:
                    if child is None or child["bundle"]["option"] != "donor_transplant":
                        continue
                    bundle = child["bundle"]
                    donor = memory["rows"][bundle["donor_index"]]
                    if (
                        bundle["donor_memory_id"] != memory["memory_id"]
                        or bundle["donor_smiles"] != donor["smiles"]
                        or abs(
                            bundle["donor_probability"]
                            - memory["probabilities"][bundle["donor_index"]]
                        )
                        > 1e-12
                    ):
                        raise ValueError("donor memory differs from the arm's actual scored prefix")
                    donors_used.append(
                        {
                            "smiles": donor["smiles"],
                            "new_donor": donor["smiles"] not in initial_donors,
                            "child": child["smiles"],
                            "score": child["score"],
                            "parent_score": child["parent_score"],
                        }
                    )
                memory_audits.append(
                    {
                        "boundary": step,
                        "memory_id": memory["memory_id"],
                        "donors": len(memory["rows"]),
                        "attempted_uses": donor_draws,
                        "completed_uses": donors_used,
                    }
                )
            for slot, parent in enumerate(parents):
                if parent is None:
                    continue
                selected_policy = slot_policy_identity(c, arm, step, slot, policy_id, memory_id)
                used_workers.add(worker_identity(step, slot, parent, selected_policy))
                child = a["proposals"][slot]
                local_draw = selected_policy != policy_id
                if child is not None and (
                    (child["bundle"]["option"] == "local_endpoint_selector") != local_draw
                    or (local_draw and child["primitive_count"] != 1)
                ):
                    raise ValueError("local/broad channel or primitive count differs from its draw")
                if local_draw:
                    local_audits.append(
                        {
                            "boundary": step,
                            "slot": slot,
                            "parent_score": parent["score"],
                            "completed": child is not None,
                            "score": None if child is None else child["score"],
                            "policy_id": selected_policy,
                        }
                    )
            terminal = step == c["boundaries"]
            proposed_depths = proposal_depths(parents, a["proposals"], depths)
            rng = np.random.default_rng(np.random.SeedSequence([c["seed"], step, 999]))
            if archive_mode:
                for p, depth in zip(a["proposals"], proposed_depths, strict=True):
                    if p is not None:
                        archive_nodes.setdefault(p["smiles"], p)
                        archive_depths.setdefault(p["smiles"], depth)
                next_parents, selection = sample_archive_parents(
                    archive_nodes, n, rng, exploration=c["archive_exploration"]
                )
                if selection != a["archive_selection"]:
                    raise ValueError(f"archive parent selection differs: {arm}/{step}")
                next_depths = [archive_depths[p["smiles"]] for p in next_parents]
            else:
                psi = log_potentials(
                    selection_mode,
                    a["scores"],
                    a["future_values"],
                    terminal=terminal,
                    beta=c["beta"],
                )
                update = advance(
                    weights,
                    previous,
                    psi,
                    [p is not None for p in a["proposals"]],
                    rng,
                    resample=selection_mode != "reference" and not terminal,
                )
                for key in ("status", "resampled", "indices", "log_potential"):
                    if update[key] != a[key]:
                        raise ValueError(f"particle decision replay mismatch: {arm}/{step}/{key}")
                # Numerical tolerance only; selection and zero-weight masks stay exact.
                for key in ("weights", "ess", "log_weights"):
                    expected = a[key] if isinstance(a[key], list) else [a[key]]
                    actual = update[key] if isinstance(update[key], list) else [update[key]]
                    if [x is None for x in expected] != [x is None for x in actual]:
                        raise ValueError("particle zero-weight mask changed")
                    error = max(
                        (
                            abs(x - y)
                            for x, y in zip(actual, expected, strict=True)
                            if x is not None
                        ),
                        default=0,
                    )
                    max_replay_error = max(max_replay_error, error)
                    if error > 1e-12:
                        raise ValueError(f"particle numerical replay mismatch: {arm}/{step}/{key}")
                weights, previous = update["log_weights"], update["log_potential"]
                next_depths = [proposed_depths[i] for i in update["indices"]]
                next_parents = [a["proposals"][i] for i in update["indices"]]
            offspring_deltas.extend(
                child["score"] - parent["score"]
                for parent, child in zip(parents, a["proposals"], strict=True)
                if child is not None
            )
            observed_depths.extend(d for d in proposed_depths if d is not None)
            depths, parents = next_depths, next_parents
            attempts += sum(live_before)
            failures += sum(
                live and p is None for live, p in zip(live_before, a["proposals"], strict=True)
            )
            live_before = [p is not None for p in next_parents]
            proposals.extend(p for p in a["proposals"] if p is not None)
            selected.append(
                {
                    "boundary": step,
                    "ess": a.get("ess"),
                    "resampled": a.get("resampled"),
                    "parent_selection": "archive" if archive_mode else "smc",
                    "best": a["best"],
                    "top10_mean": a["top10_mean"],
                }
            )

        products = sorted(r["archives"][arm])
        fps = [fpgen.GetFingerprint(Chem.MolFromSmiles(s)) for s in products]
        distances = [
            1 - DataStructs.TanimotoSimilarity(fps[i], fps[j])
            for i in range(len(fps))
            for j in range(i)
        ]
        roots = {p["smiles"]: p["score"] for p in r["initial_parents"]}
        independently_queried = dict(roots)
        independently_queried.update({p["smiles"]: p["score"] for p in proposals})
        if independently_queried != r["archives"][arm]:
            raise ValueError("archive excludes an evaluated candidate or imports unqueried labels")
        arm_workers = [w for w in r["workers"] if w["worker_id"] in used_workers]
        if {w["worker_id"] for w in arm_workers} != used_workers:
            raise ValueError("parent ancestry lacks matching worker receipts")
        # Older arms share proposal workers; these are not additive arm costs.
        audited_attempts = [a for w in arm_workers for a in w["attempts"]]
        has_attempt_census = all("proposal_attempts" in a for a in audited_attempts)
        local_attempts = [a for a in audited_attempts if a.get("component") == "local_endpoint"]
        sampled_attempts = (
            sum(a["proposal_attempts"] for a in audited_attempts)
            if has_attempt_census and not local_attempts
            else None
        )
        steps = sum(a["primitive_steps"] for a in audited_attempts) if has_attempt_census else None
        generated_fresh = {
            p["smiles"]: p["score"] for p in proposals if p["smiles"] in fresh_scores
        }
        rows.append(
            {
                "arm": arm,
                **r["arms"][arm],
                "particle_replay_verified": not archive_mode,
                "archive_selection_replay_verified": archive_mode,
                **({"donor_memory_replay": memory_audits} if memory_audits else {}),
                **({"local_channel_replay": local_audits} if local_audits else {}),
                "complete_options": len(proposals),
                "attempted_options": attempts,
                "failed_options": failures,
                "offered_particle_slots": n * c["boundaries"],
                "unique_generated_candidates": len({p["smiles"] for p in proposals}),
                "newly_queried_candidates": len(generated_fresh),
                "newly_queried_best": max(generated_fresh.values(), default=None),
                "offspring_improvement": {
                    "distribution": quantiles(offspring_deltas),
                    "mean": float(np.mean(offspring_deltas)) if offspring_deltas else None,
                    "improved": sum(d > 0 for d in offspring_deltas),
                    "tied": sum(d == 0 for d in offspring_deltas),
                    "worse": sum(d < 0 for d in offspring_deltas),
                },
                "proposal_sampling": {
                    "attempts": sampled_attempts,
                    "primitive_steps": steps,
                    "attempts_per_step": sampled_attempts / steps
                    if steps and sampled_attempts is not None
                    else None,
                    "local_support_census": {
                        "draws": len(local_attempts),
                        "positive_mass_marks": sum(a["proposal_attempts"] for a in local_attempts),
                        "canonical_products": sum(a["available_products"] for a in local_attempts),
                        "predicted_products": sum(a["unqueried_products"] for a in local_attempts),
                        "selection_seconds": sum(a["selection_seconds"] for a in local_attempts),
                        "note": "Enumeration counts are not stochastic proposal attempts; mixed-channel attempt ratios are undefined.",
                    },
                    "what_kl": quantiles(
                        [
                            a["what_allocation"]["kl"]
                            for a in audited_attempts
                            if a.get("what_allocation")
                        ]
                    ),
                    "worker_seconds": sum(w["seconds"] for w in arm_workers),
                    "cost_note": "Workers can be shared between arms with the same proposal identity; do not add shared costs twice.",
                },
                "new_primitive_depth": quantiles(observed_depths),
                "primitive_steps_per_option": quantiles([p["primitive_count"] for p in proposals]),
                "option_counts": dict(Counter(p["bundle"]["option"] for p in proposals)),
                "intended_release": quantiles([p["bundle"]["r_release"] for p in proposals]),
                "realized_coherent": quantiles(
                    [p["structural_change"]["largest_changed_fraction"] for p in proposals]
                ),
                "cycle_rank_deltas": dict(
                    Counter(p["structural_change"]["d_cycle_rank"] for p in proposals)
                ),
                "ring_system_deltas": dict(
                    Counter(p["structural_change"]["d_ring_systems"] for p in proposals)
                ),
                "mean_pairwise_morgan_distance": float(np.mean(distances)),
                "top10": [
                    {
                        "smiles": s,
                        "score": v,
                        "heavy_atoms": Chem.MolFromSmiles(s).GetNumHeavyAtoms(),
                        "aromatic_rings": rdMolDescriptors.CalcNumAromaticRings(
                            Chem.MolFromSmiles(s)
                        ),
                    }
                    for s, v in sorted(r["archives"][arm].items(), key=lambda x: (-x[1], x[0]))[:10]
                ],
                "rounds": selected,
            }
        )
    recovery_path = path.with_name("resume_after_preemption.json")
    recovery = None
    if recovery_path.exists():
        recovered = json.loads(recovery_path.read_text())
        original = json.loads(path.with_name("spawn.json").read_text())
        if recovered["task"] != original["task"] or recovered[
            "original_receipt_sha256"
        ] != sha256_file(path.with_name("spawn.json")):
            raise ValueError("recovery did not preserve the original deployed task")
        recovery = {
            "receipt_sha256": sha256_file(recovery_path),
            "original_call_id": original["call_id"],
            "resumed_call_id": recovered["call_id"],
            "same_deployed_task": True,
            "submission_to_finish_seconds": (
                datetime.fromisoformat(r["finished_at"])
                - datetime.fromisoformat(original["task"]["started_at"])
            ).total_seconds(),
            "final_driver_session_seconds": r["seconds"],
            "cost_limit": "submission-to-finish includes downtime; worker totals exclude lost preemption work; final session time is not full experiment wall time",
        }
    return {
        "schema_version": "option_particles_report_v1",
        "run_id": r["run_id"],
        "input_paths_sha256": {
            str(p): sha256_file(p) for p in (path, path.with_name("spawn.json"), archive)
        },
        "analyzer_sha256": sha256_file(Path(__file__)),
        "particle_replay_tolerance": 1e-12,
        "maximum_numeric_replay_error": max_replay_error,
        "particle_decisions_match_exactly": True,
        "analysis_software": software(),
        "configuration": c,
        "executed_revision": r["image_revision"],
        "arms": rows,
        "new_oracle_calls": r["new_oracle_calls"],
        "oracle_seconds": sum(x["oracle_seconds"] for x in r["oracle_rows"]),
        "worker_tasks": len(r["workers"]),
        "wall_seconds": r["seconds"]
        if recovery is None
        else recovery["submission_to_finish_seconds"],
        "recovery": recovery,
        "proposal_wall_seconds": sum(x["proposal_seconds"] for x in r["rounds"]),
        "worker_seconds_sum": sum(x["seconds"] for x in r["workers"]),
        "law_seconds_sum": sum(x["law_work"]["law_seconds"] for x in r["workers"]),
        "driver_io": r["io_timings"],
        "historical_unique_labels_available": r["historical_unique_labels"],
        "initial_best": r["initial_metrics"]["best"],
        "any_best_improvement": any(
            a["best"] > r["initial_metrics"]["best"] for a in r["arms"].values()
        ),
        "interpretation": "warm-development particle comparison; inspect the measured arms and initial best, not official PMO AUC or blind generalization",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    value = report(args.result)
    publish_json(args.output, value)
    print(
        json.dumps(
            {
                k: value[k]
                for k in ("new_oracle_calls", "oracle_seconds", "worker_tasks", "wall_seconds")
            }
        )
    )
    for a in value["arms"]:
        print(
            json.dumps(
                {
                    k: a[k]
                    for k in (
                        "arm",
                        "best",
                        "top10_mean",
                        "new_primitive_depth",
                        "complete_options",
                        "cycle_rank_deltas",
                        "mean_pairwise_morgan_distance",
                    )
                }
            )
        )


if __name__ == "__main__":
    main()
