#!/usr/bin/env python3
"""Hard-constrained valid-fiber controller: fixed-lead GrIDDD-style QED panel.

Diagnostic/ceiling method (best-first search over VALID molecular successors,
with a HARD Tanimoto>=0.40 constraint enforced in the legal fiber, and an
anytime best-feasible memory). This is NOT the deployable method (best-first
collapses diversity); it is (a) an upper-bound estimate of what constrained
optimization can reach on our frozen base, and (b) a generator of value-function
training labels. Every scored candidate is a complete, valid, oracle-scoreable
molecule -- the property the whole approach rests on.

Reports per-lead best-feasible QED and the GrIDDD success (>=0.90 at sim>=0.40)
so we can estimate a success rate vs GrIDDD's 45.1%. Runs on the neutral
C/N/O/F-compatible subset of the frozen Jin QED leads (our base is C/N/O/F only).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator

from compose_v4.chem.molecular_graph import (
    molecular_graph_to_smiles,
    smiles_to_molecular_graph,
)
from compose_v4.chem.state import pad_molecular_graph
from compose_v4.model.time_convention import frozen_time
from compose_v4.rewrite.kernel import de_novo_rewrite_system

RDLogger.DisableLog("rdApp.*")


def _load_base_sampler(checkpoint: str):
    if __package__:
        from scripts.evaluate_tracelet_rollouts import (
            load_factorized_rollout_checkpoint,
        )
        from scripts.run_griddd_analytic_zero_sidecar_smoke import (
            AnalyticPancakeQuotientSampler,
            qed_state_oracle,
        )
    else:
        from evaluate_tracelet_rollouts import load_factorized_rollout_checkpoint
        from run_griddd_analytic_zero_sidecar_smoke import (
            AnalyticPancakeQuotientSampler,
            qed_state_oracle,
        )

    base, _ = load_factorized_rollout_checkpoint(checkpoint)
    base.eval()
    return AnalyticPancakeQuotientSampler(base), qed_state_oracle


_FP = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


def _fp(state):
    mol = Chem.MolFromSmiles(molecular_graph_to_smiles(state))
    return _FP.GetFingerprint(mol) if mol is not None else None


def _canon(state):
    mol = Chem.MolFromSmiles(molecular_graph_to_smiles(state))
    return Chem.MolToSmiles(mol) if mol is not None else None


def best_first_constrained(
    lead_state,
    *,
    sampler,
    qed_oracle,
    seed: int,
    similarity_minimum: float,
    budget: int,
    proposals_per_expansion: int,
    frontier_maximum: int,
):
    """Best-first search over valid successors under a hard similarity floor.

    Returns (best_feasible_qed, best_similarity, oracle_calls, labels), where
    labels is a list of (canonical_smiles, qed, similarity) for every distinct
    feasible molecule scored -- the raw material for a value function.
    """

    rewrite = de_novo_rewrite_system()
    lead_fp = _fp(lead_state)
    lead_qed = qed_oracle(lead_state)
    rng = np.random.default_rng(seed)
    frontier = [(lead_qed, 0.0, lead_state)]
    best_qed, best_state = lead_qed, lead_state
    seen = {_canon(lead_state)}
    labels: list[tuple[str, float, float]] = []
    calls = 0
    while calls < budget and frontier:
        frontier.sort(key=lambda item: -item[0])
        _, node_time, node = frontier.pop(0)
        for _ in range(proposals_per_expansion):
            if calls >= budget:
                break
            mark = sampler.sample_rewrite_mark(node, frozen_time(node_time), rng)
            if mark.action is None:
                continue
            try:
                successor = rewrite.apply(node, mark.rule_name, mark.action)
            except Exception:
                continue
            key = _canon(successor)
            if key is None or key in seen:
                continue
            seen.add(key)
            fingerprint = _fp(successor)
            if fingerprint is None:
                continue
            similarity = DataStructs.TanimotoSimilarity(lead_fp, fingerprint)
            qed = qed_oracle(successor)
            calls += 1
            if similarity >= similarity_minimum:
                frontier.append((qed, node_time + 0.1, successor))
                labels.append((key, float(qed), float(similarity)))
                if qed > best_qed:
                    best_qed, best_state = qed, successor
        frontier = sorted(frontier, key=lambda item: -item[0])[:frontier_maximum]
    best_similarity = DataStructs.TanimotoSimilarity(lead_fp, _fp(best_state))
    return float(best_qed), float(best_similarity), calls, labels


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("/private/tmp/pancake_checkpoint/checkpoint.recovery.pt"),
    )
    parser.add_argument(
        "--leads",
        type=Path,
        required=True,
        help="JSON list of [lead_index, smiles, qed] (CNOF-compatible).",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-leads", type=int, default=12)
    parser.add_argument("--seed", type=int, default=20260726)
    parser.add_argument("--similarity-minimum", type=float, default=0.40)
    parser.add_argument("--target-qed", type=float, default=0.90)
    parser.add_argument("--budget", type=int, default=640)
    parser.add_argument("--proposals-per-expansion", type=int, default=16)
    parser.add_argument("--frontier-maximum", type=int, default=48)
    parser.add_argument("--n-slots", type=int, default=40)
    args = parser.parse_args()

    sampler, qed_oracle = _load_base_sampler(str(args.checkpoint))
    leads = json.loads(args.leads.read_text())[: args.max_leads]

    results: list[dict[str, object]] = []
    for index, smiles, listed_qed in leads:
        try:
            state = pad_molecular_graph(
                smiles_to_molecular_graph(smiles), args.n_slots
            )
        except Exception as error:  # noqa: BLE001
            results.append({"lead_index": index, "error": str(error)})
            _write(results, args)
            continue
        lead_qed = qed_oracle(state)
        best_qed, best_similarity, calls, labels = best_first_constrained(
            state,
            sampler=sampler,
            qed_oracle=qed_oracle,
            seed=args.seed,
            similarity_minimum=args.similarity_minimum,
            budget=args.budget,
            proposals_per_expansion=args.proposals_per_expansion,
            frontier_maximum=args.frontier_maximum,
        )
        success = bool(
            best_qed >= args.target_qed and best_similarity >= args.similarity_minimum
        )
        results.append(
            {
                "lead_index": index,
                "lead_smiles": smiles,
                "lead_qed": lead_qed,
                "best_feasible_qed": best_qed,
                "best_similarity": best_similarity,
                "success": success,
                "oracle_calls": calls,
                "distinct_feasible_scored": len(labels),
            }
        )
        _write(results, args)
        print(
            f"lead {index}: lead_qed {lead_qed:.3f} -> best {best_qed:.3f} "
            f"(sim {best_similarity:.3f}) success={success} calls={calls}",
            flush=True,
        )

    scored = [r for r in results if "success" in r]
    n = len(scored)
    successes = sum(1 for r in scored if r["success"])
    beat = sum(1 for r in scored if r["best_feasible_qed"] > r["lead_qed"])
    summary = {
        "format": "compose_v4_valid_fiber_controller_panel_v1",
        "method": "hard_constrained_best_first_valid_fiber_search_ceiling",
        "note": "upper-bound/ceiling + label generator; NOT the deployable diverse method",
        "leads_scored": n,
        "success_at_target": successes,
        "success_rate": (successes / n) if n else None,
        "beat_lead_and_feasible": beat,
        "mean_best_feasible_qed": (
            sum(r["best_feasible_qed"] for r in scored) / n if n else None
        ),
        "griddd_reference_success_rate": 0.451,
        "target_qed": args.target_qed,
        "similarity_minimum": args.similarity_minimum,
        "budget_per_lead": args.budget,
        "results": results,
    }
    args.output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: summary[k] for k in ("leads_scored", "success_at_target", "success_rate", "mean_best_feasible_qed", "beat_lead_and_feasible")}))


def _write(results: list[dict[str, object]], args) -> None:
    args.output.write_text(
        json.dumps({"partial": True, "results": results}, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
