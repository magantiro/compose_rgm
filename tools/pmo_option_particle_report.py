"""Audit saved particle trajectories and chemistry with no oracle/model calls."""

import argparse
import hashlib
import json
import tarfile
from collections import Counter
from pathlib import Path

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator, rdMolDescriptors

from compose_v4.control.docking_value import identity
from compose_v4.control.option_particles import advance, log_potentials
from compose_v4.experiments.continuation_profile import publish_json, sha256_file
from compose_v4.experiments.inference_package import software


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
        history = dict(json.loads(prepared)["observed"])
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
    max_replay_error = 0.0
    fpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    for arm in c["arms"]:
        weights, previous = [-float(np.log(n))] * n, [0.0] * n
        proposals = []
        selected = []
        live_before = [True] * n
        parents, depths, observed_depths = r["initial_parents"], [0] * n, []
        attempts = failures = 0
        for round_row in r["rounds"]:
            a = round_row["arms"][arm]
            if any(p is not None and history[p["smiles"]] != p["score"] for p in a["proposals"]):
                raise ValueError("proposal score differs from its actual oracle receipt")
            step = round_row["boundary"]
            terminal = step == c["boundaries"]
            psi = log_potentials(
                arm, a["scores"], a["future_values"], terminal=terminal, beta=c["beta"]
            )
            update = advance(
                weights,
                previous,
                psi,
                [p is not None for p in a["proposals"]],
                np.random.default_rng(np.random.SeedSequence([c["seed"], step, 999])),
                resample=arm != "reference" and not terminal,
            )
            for key in ("status", "resampled", "indices", "log_potential"):
                if update[key] != a[key]:
                    raise ValueError(f"particle decision replay mismatch: {arm}/{step}/{key}")
            # The first cross-platform audit exposed only floating roundoff
            # (max ESS difference 4.45e-15), not changed ancestry. Do not demand
            # bit-identical exp/log on ARM macOS and x86 Linux. The 1e-12
            # diagnostic tolerance does not alter the running ESS trigger.
            for key in ("weights", "ess", "log_weights"):
                expected = a[key] if isinstance(a[key], list) else [a[key]]
                actual = update[key] if isinstance(update[key], list) else [update[key]]
                if [x is None for x in expected] != [x is None for x in actual]:
                    raise ValueError("particle zero-weight mask changed")
                error = max(
                    (abs(x - y) for x, y in zip(actual, expected, strict=True) if x is not None),
                    default=0,
                )
                max_replay_error = max(max_replay_error, error)
                if error > 1e-12:
                    raise ValueError(f"particle numerical replay mismatch: {arm}/{step}/{key}")
            weights, previous = update["log_weights"], update["log_potential"]
            proposed_depths = proposal_depths(parents, a["proposals"], depths)
            observed_depths.extend(d for d in proposed_depths if d is not None)
            depths = [proposed_depths[i] for i in update["indices"]]
            parents = [a["proposals"][i] for i in update["indices"]]
            attempts += sum(live_before)
            failures += sum(
                live and p is None for live, p in zip(live_before, a["proposals"], strict=True)
            )
            live_before = [a["proposals"][i] is not None for i in update["indices"]]
            proposals.extend(p for p in a["proposals"] if p is not None)
            selected.append(
                {
                    "boundary": step,
                    "ess": a["ess"],
                    "resampled": a["resampled"],
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
        rows.append(
            {
                "arm": arm,
                **r["arms"][arm],
                "particle_replay_verified": True,
                "complete_options": len(proposals),
                "attempted_options": attempts,
                "failed_options": failures,
                "offered_particle_slots": n * c["boundaries"],
                "unique_generated_candidates": len({p["smiles"] for p in proposals}),
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
        "wall_seconds": r["seconds"],
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
