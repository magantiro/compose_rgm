"""Summarize a completed online-policy development run without new scoring."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from rdkit import Chem, rdBase

from compose_v4.experiments.continuation_profile import publish_json, sha256_file
from compose_v4.experiments.pmo_online_policy import load_contract
from compose_v4.experiments.t4_matched_pilot import unseal

ROOT = Path(__file__).resolve().parents[1]


def bounds(values):
    return (
        dict(zip(("min", "median", "max"), np.quantile(values, [0, 0.5, 1]).tolist()))
        if values
        else None
    )


def summarize(result, prepared, phases):
    if result["status"] != "complete_bounded_development" or result["evaluation_policy_frozen"]:
        raise ValueError("requires a completed repeated-feedback development result")
    if (
        result["new_oracle_calls"] > 96
        or result["historical_calls"] != prepared["historical_calls"]
    ):
        raise ValueError("oracle accounting disagrees with the frozen budget/history")
    if len(phases) != 8 or len(result["online_updates"]) != 7:
        raise ValueError("missing locked round or policy update")
    known_labels = dict(prepared["observed"])
    for state in result["arms"].values():
        for event in state["selections"]:
            if event["status"] != "scored":
                continue
            s, value = event["candidate"]["smiles"], event["score"]
            if s in known_labels and abs(known_labels[s] - value) > 1e-12:
                raise ValueError("inconsistent recorded deterministic labels")
            known_labels[s] = value
    summaries = {}
    for arm, state in result["arms"].items():
        observed = dict(prepared["observed"])
        initial = sum(sorted(observed.values(), reverse=True)[:10]) / 10
        previous, area = initial, 0.0
        records = state["selections"]
        if len(records) != 32 or len(state["curve"]) != 32:
            raise ValueError(f"wrong request count for {arm}")
        for event, curve in zip(records, state["curve"]):
            choices = [
                r
                for r in phases[event["round"]]["choices"]["choices"]
                if r["arm"] == arm and r["slot"] == event["slot"]
            ]
            if len(choices) != 1:
                raise ValueError("missing or duplicated locked selection")
            choice = choices[0]
            if event["status"] == "scored":
                candidate = event["candidate"]
                if choice["selected"] is None or choice["pool"][choice["selected"]] != candidate:
                    raise ValueError("scored candidate differs from its lock")
                observed[candidate["smiles"]] = event["score"]
            elif choice["selected"] is not None:
                raise ValueError("abstention disagrees with lock")
            best, top = max(observed.values()), sum(sorted(observed.values(), reverse=True)[:10])
            if abs(best - curve["best"]) > 1e-12 or abs(top - curve["top10_sum"]) > 1e-12:
                raise ValueError("recorded best/top-ten curve does not replay")
            area += (previous + top / 10) / 2
            previous = top / 10
        if abs(state["best"] - best) > 1e-12 or abs(state["top10_sum"] - top) > 1e-12:
            raise ValueError("final archive summary does not replay")
        scored = [r for r in records if r["status"] == "scored"]
        candidates = [r["candidate"] for r in scored]
        choices = [
            {"round": n, **r}
            for n, p in phases.items()
            for r in p["choices"]["choices"]
            if r["arm"] == arm
        ]
        policies = [phases[i]["choices"]["policy_by_arm"][arm] for i in range(1, 9)]
        parent_sizes = [
            Chem.MolFromSmiles(c["parent"]["smiles"]).GetNumHeavyAtoms() for c in choices
        ]
        known_counts, missed_gaps, complete_contrasts, missed_examples = [], [], 0, []
        for choice in choices:
            known = [
                known_labels[c["smiles"]] for c in choice["pool"] if c["smiles"] in known_labels
            ]
            known_counts.append(len(known))
            if len(known) >= 2 and choice["selected"] is not None:
                selected_smiles = choice["pool"][choice["selected"]]["smiles"]
                missed_gaps.append(max(known) - known_labels[selected_smiles])
                complete_contrasts += len(known) == len(choice["pool"])
                if missed_gaps[-1] > 1e-12:
                    index = max(
                        (i for i, c in enumerate(choice["pool"]) if c["smiles"] in known_labels),
                        key=lambda i: known_labels[choice["pool"][i]["smiles"]],
                    )
                    candidate = choice["pool"][index]
                    missed_examples.append(
                        {
                            "round": choice["round"],
                            "slot": choice["slot"],
                            "selected_smiles": selected_smiles,
                            "selected_score": known_labels[selected_smiles],
                            "selected_probability": choice["allocation"]["probabilities"][
                                choice["selected"]
                            ],
                            "better_known_smiles": candidate["smiles"],
                            "better_known_score": known_labels[candidate["smiles"]],
                            "better_known_probability": choice["allocation"]["probabilities"][
                                index
                            ],
                        }
                    )
        if arm != "learned" and len(set(policies)) != 1:
            raise ValueError("a frozen control changed policy")
        if arm == "learned" and policies[1:] != [
            r["model_sha256"] for r in result["online_updates"]
        ]:
            raise ValueError("online update identities disagree with next-round locks")
        summaries[arm] = {
            "requests": len(records),
            "scored_requests": len(scored),
            "abstentions": len(records) - len(scored),
            "unique_selected_molecules": len({c["smiles"] for c in candidates}),
            "initial_best": state["initial"]["best"],
            "final_best": best,
            "initial_top10_mean": initial,
            "final_top10_mean": top / 10,
            "warm_window_top10_trapezoid_auc": area / len(records),
            "mean_selected_score": state["mean_selected_score"],
            "mean_parent_delta": state["mean_parent_delta"],
            "parent_improvements": sum(r["parent_delta"] > 1e-12 for r in scored),
            "parent_losses": sum(r["parent_delta"] < -1e-12 for r in scored),
            "options": dict(sorted(Counter(c["bundle"]["option"] for c in candidates).items())),
            "cycle_rank_deltas": dict(
                sorted(Counter(c["structural_change"]["d_cycle_rank"] for c in candidates).items())
            ),
            "ring_system_deltas": dict(
                sorted(
                    Counter(c["structural_change"]["d_ring_systems"] for c in candidates).items()
                )
            ),
            "intended_release": bounds([c["bundle"]["r_release"] for c in candidates]),
            "realized_largest_changed_fraction": bounds(
                [c["structural_change"]["largest_changed_fraction"] for c in candidates]
            ),
            "pool_sizes": bounds([len(c["pool"]) for c in choices]),
            "sampled_parent_heavy_atoms": bounds(parent_sizes),
            "parent_requests_at_40_atom_limit": sum(n == 40 for n in parent_sizes),
            "parent_requests_with_fewer_than_5_free_atoms": sum(n > 35 for n in parent_sizes),
            "excluded_known_candidates": sum(len(c["excluded"]) for c in choices),
            "distinct_policy_hashes": len(set(policies)),
            "hindsight_known_pool_diagnostic": {
                "labelled_pool_entries": sum(known_counts),
                "all_pool_entries": sum(len(c["pool"]) for c in choices),
                "decisions_with_at_least_two_known_labels": len(missed_gaps),
                "fully_labelled_contrasts": complete_contrasts,
                "missed_better_known_alternative": sum(g > 1e-12 for g in missed_gaps),
                "mean_gap_to_best_known_alternative": float(np.mean(missed_gaps))
                if missed_gaps
                else None,
                "examples": missed_examples,
                "scope": "retrospective partial-label diagnostic using already-charged outcomes across arms; no unobserved labels inferred, no training, not unbiased policy regret",
            },
            "best_smiles": sorted(s for s, v in observed.items() if abs(v - best) < 1e-12),
        }
    workers = [w for p in phases.values() for w in p["proposals"].values()]
    attempts = [a for w in workers for a in w["attempts"]]
    shared = []
    for number, phase in phases.items():
        choices = {(r["arm"], r["slot"]): r for r in phase["choices"]["choices"]}
        for slot in range(4):
            frozen, online = choices["frozen", slot], choices["learned", slot]
            if not frozen["pool"] or not online["pool"]:
                continue
            if frozen["parent"] != online["parent"] or [c["smiles"] for c in frozen["pool"]] != [
                c["smiles"] for c in online["pool"]
            ]:
                continue
            shared.append(
                {
                    "round": number,
                    "slot": slot,
                    "selection_changed": frozen["selected"] != online["selected"],
                    "allocation_half_l1": float(
                        np.abs(
                            np.asarray(frozen["allocation"]["probabilities"])
                            - np.asarray(online["allocation"]["probabilities"])
                        ).sum()
                        / 2
                    ),
                }
            )
    return {
        "arms": summaries,
        "workers": len(workers),
        "worker_seconds_sum": sum(w["seconds"] for w in workers),
        "worker_seconds": bounds([w["seconds"] for w in workers]),
        "worker_initialization_seconds": bounds([w["initialization_seconds"] for w in workers]),
        "completed_draws": sum(a["status"] == "complete" for a in attempts),
        "draw_status": dict(sorted(Counter(a["status"] for a in attempts).items())),
        "draw_options": dict(
            sorted(
                Counter(
                    a["bundle"]["option"] if a["bundle"] else "no_option" for a in attempts
                ).items()
            )
        ),
        "draw_seconds": bounds([a["proposal_seconds"] for a in attempts]),
        "online_fit_seconds": [r["seconds"] for r in result["online_updates"]],
        "run_seconds": result["seconds"],
        "historical_calls": result["historical_calls"],
        "new_physical_calls": result["new_oracle_calls"],
        "online_vs_frozen_shared_pools": shared,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError(f"refusing to overwrite {args.output}")
    paths = [args.snapshot / "result.json", ROOT / "diagnostics/pmo_online_policy/prepared.json"]
    result, prepared = unseal(paths[0]), json.loads(paths[1].read_text())
    contract = load_contract(ROOT)
    if result["contract_sha256"] != contract["contract_sha256"]:
        raise ValueError("result belongs to another experimental contract")
    paths.extend([ROOT / "configs/pmo_online_policy.json", ROOT / contract["protocol"]["path"]])
    phases = {}
    for number in range(1, 9):
        phases[number] = {}
        for name in ("choices", "proposals"):
            path = args.snapshot / "phase" / str(number) / f"{name}.json"
            phases[number][name] = unseal(path)
            paths.append(path)
    report = {
        "schema_version": "pmo_online_audit_v1",
        "at": datetime.now(timezone.utc).isoformat(),
        "run_id": result["run_id"],
        "contract_sha256": result["contract_sha256"],
        "producer_revision": result["code_revision"],
        "audit_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "input_sha256": {str(p): sha256_file(p) for p in paths},
        "implementation_sha256": sha256_file(Path(__file__)),
        "configuration": {
            "new_oracle_calls": 0,
            "new_fits": 0,
            "random_sampling": False,
            "seed": None,
            "split": "inspected_chronological_development",
            "device": "cpu",
            "precision": "float64",
            "auc": "trapezoid including warm-start at request 0, normalized over 32 requests; not official PMO AUC",
        },
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "rdkit": rdBase.rdkitVersion,
        },
        **summarize(result, prepared, phases),
    }
    publish_json(args.output, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
