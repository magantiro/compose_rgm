#!/usr/bin/env python3
"""Value-guided Sequential Monte Carlo controller over the rewrite CTMC.

This is the DEPLOYABLE conditional method (design: docs/CONDITIONAL_CONTROLLER_DESIGN.md),
as opposed to the best-first `griddd_valid_fiber_controller_panel.py` which only
estimates a ceiling and collapses diversity. Here a population of particles
(each a real, valid, oracle-scoreable molecule) is propagated under Lineage B's
base CTMC, weighted by a Feynman-Kac potential toward the reward-tilted target
`pi(x) ~ p_B(x) exp(r(x)/alpha) 1[x in F]`, with the similarity constraint F
enforced HARD in the legal fiber (infeasible proposals are never accepted).

Phase 1 (this file): V0 reward-difference potential `G = exp((r(y)-r(x))/alpha)`,
using the exact oracle at each real intermediate -- our unique advantage. A
learned twist / Doob value (V1, DEFT/RTB) replaces V0 for the reported result;
the SMC scaffold, resampling, hard fiber, and matched oracle accounting are shared.

Reports per-lead best-feasible QED, GrIDDD-style success (>=target at sim>=floor),
population diversity (mean pairwise 1-Tanimoto), and matched oracle-call budget.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold

from compose_v4.chem.molecular_graph import (
    molecular_graph_to_smiles,
    smiles_to_molecular_graph,
)
from compose_v4.chem.state import pad_molecular_graph
from compose_v4.experiments.canonical_successor_distillation import (
    PancakeQuotientCalibration,
)
from compose_v4.model.time_convention import frozen_time
from compose_v4.rewrite.kernel import de_novo_rewrite_system

RDLogger.DisableLog("rdApp.*")

_FP = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


def _load_base_sampler(checkpoint: str, atom_delete_log_rate_adjustment: float):
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
    calibration = PancakeQuotientCalibration(
        atom_delete_log_rate_adjustment=atom_delete_log_rate_adjustment
    )
    return AnalyticPancakeQuotientSampler(base, calibration=calibration), qed_state_oracle


def _fp(state):
    mol = Chem.MolFromSmiles(molecular_graph_to_smiles(state))
    return _FP.GetFingerprint(mol) if mol is not None else None


def _canon(state):
    mol = Chem.MolFromSmiles(molecular_graph_to_smiles(state))
    return Chem.MolToSmiles(mol) if mol is not None else None


def _systematic_resample(weights: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Return resampled indices (low-variance/systematic)."""
    n = len(weights)
    positions = (rng.random() + np.arange(n)) / n
    cumulative = np.cumsum(weights)
    cumulative[-1] = 1.0
    return np.searchsorted(cumulative, positions)


def _mean_pairwise_diversity(fingerprints: list) -> float:
    if len(fingerprints) < 2:
        return 0.0
    total, count = 0.0, 0
    for i in range(len(fingerprints)):
        for j in range(i + 1, len(fingerprints)):
            total += 1.0 - DataStructs.TanimotoSimilarity(fingerprints[i], fingerprints[j])
            count += 1
    return total / count if count else 0.0


def value_guided_smc(
    lead_state,
    *,
    sampler,
    qed_oracle,
    rewrite,
    seed: int,
    similarity_minimum: float,
    budget: int,
    n_particles: int,
    max_steps: int,
    time_step: float,
    horizon: float,
    alpha_start: float,
    alpha_end: float,
    feasible_attempts: int,
    twist=None,
    lookahead_rollouts: int = 0,
    lookahead_depth: int = 4,
    scaffold_mol=None,
    measure_scaffold_mol=None,
    required_smarts=None,
    forbidden_smarts=None,
    measure_required_smarts=None,
    measure_forbidden_smarts=None,
    dump_population=False,
):
    """Propagate a particle population under the base CTMC toward high reward
    inside the hard similarity fiber. Returns a result dict."""

    rng = np.random.default_rng(seed)
    lead_fp = _fp(lead_state)
    lead_qed = qed_oracle(lead_state)

    # QED cache doubles as the matched oracle-call meter: one call per distinct
    # molecule ever scored (same accounting as the valid-fiber ceiling).
    qed_cache: dict[str, float] = {_canon(lead_state): lead_qed}

    def scored_qed(state) -> float | None:
        key = _canon(state)
        if key is None:
            return None
        if key not in qed_cache:
            qed_cache[key] = qed_oracle(state)
        return qed_cache[key]

    twist_value_cache: dict[str, float] = {}
    lookahead_cache: dict[str, float] = {}

    def _lookahead_value(state) -> float:
        # V2: EXACT Monte-Carlo lookahead -- estimate the best feasible reward
        # reachable from `state` with a few short STOCHASTIC base rollouts (not a
        # greedy beam expansion). Every scored endpoint goes through scored_qed,
        # so its oracle calls count against the same budget (honest accounting).
        best = scored_qed(state)
        best = float(best) if best is not None else 0.0
        for _ in range(lookahead_rollouts):
            node, node_time = state, 0.0
            for _ in range(lookahead_depth):
                if len(qed_cache) - 1 >= budget:
                    break
                mark = sampler.sample_rewrite_mark(node, frozen_time(node_time), rng)
                if mark.action is None:
                    break
                try:
                    node = rewrite.apply(node, mark.rule_name, mark.action)
                except Exception:  # noqa: BLE001
                    break
                fingerprint = _fp(node)
                if fingerprint is None:
                    break
                if DataStructs.TanimotoSimilarity(lead_fp, fingerprint) >= similarity_minimum:
                    reached = scored_qed(node)
                    if reached is not None and reached > best:
                        best = float(reached)
                node_time += time_step
        return best

    def scored_value(state) -> float:
        # Steering value in the FK potential (reward/success always use scored_qed):
        # V2 exact lookahead > learned twist > immediate reward (V0).
        key = _canon(state)
        if key is None:
            return 0.0
        if lookahead_rollouts > 0:
            if key not in lookahead_cache:
                lookahead_cache[key] = _lookahead_value(state)
            return lookahead_cache[key]
        if twist is None:
            value = scored_qed(state)
            return float(value) if value is not None else 0.0
        if key not in twist_value_cache:
            twist_value_cache[key] = twist.value(key)
        return twist_value_cache[key]

    states = [lead_state] * n_particles
    times = [0.0] * n_particles
    terminal = [False] * n_particles
    log_weights = np.zeros(n_particles)

    best_qed, best_state = lead_qed, lead_state
    best_qed_trace: list[tuple[int, float]] = [(0, float(lead_qed))]  # (oracle_calls, best_so_far)
    feasible_population: dict[str, object] = {}  # canon -> fingerprint

    for step in range(max_steps):
        if len(qed_cache) - 1 >= budget:  # -1: lead itself is not a search call
            break
        alpha = alpha_start * (alpha_end / alpha_start) ** (step / max(max_steps - 1, 1))
        for i in range(n_particles):
            if terminal[i] or times[i] >= horizon:
                continue
            current = states[i]
            current_qed = scored_qed(current)
            current_val = scored_value(current)
            accepted = None
            for _ in range(feasible_attempts):
                if len(qed_cache) - 1 >= budget:
                    break
                mark = sampler.sample_rewrite_mark(current, frozen_time(times[i]), rng)
                if mark.action is None:  # TERMINAL: productive hazard ~ 0
                    terminal[i] = True
                    break
                try:
                    successor = rewrite.apply(current, mark.rule_name, mark.action)
                except Exception:  # noqa: BLE001
                    continue
                key = _canon(successor)
                if key is None:
                    continue
                fingerprint = _fp(successor)
                if fingerprint is None:
                    continue
                if DataStructs.TanimotoSimilarity(lead_fp, fingerprint) < similarity_minimum:
                    continue  # HARD fiber: similarity constraint
                if scaffold_mol is not None or required_smarts is not None or forbidden_smarts is not None:
                    successor_mol = Chem.MolFromSmiles(key)
                    if successor_mol is None:
                        continue
                    if scaffold_mol is not None and not successor_mol.HasSubstructMatch(scaffold_mol):
                        continue  # HARD fiber: exact fixed-scaffold constraint
                    if required_smarts is not None and not successor_mol.HasSubstructMatch(required_smarts):
                        continue  # HARD fiber: required substructure (must be present)
                    if forbidden_smarts is not None:
                        _forb = forbidden_smarts if isinstance(forbidden_smarts, (list, tuple)) else (forbidden_smarts,)
                        if any(successor_mol.HasSubstructMatch(f) for f in _forb):
                            continue  # HARD fiber: forbidden substructure(s) absent at EVERY committed state (pathwise)
                accepted = (successor, key, fingerprint)
                break
            if accepted is None:
                continue
            successor, key, fingerprint = accepted
            new_qed = scored_qed(successor)
            if new_qed is None:
                continue
            new_val = scored_value(successor)
            log_weights[i] += (new_val - current_val) / alpha  # FK potential (twist or reward)
            states[i] = successor
            times[i] += time_step
            feasible_population[key] = fingerprint
            if new_qed > best_qed:
                best_qed, best_state = new_qed, successor
            # anytime trace: best feasible reward vs distinct oracle calls so far
            best_qed_trace.append((len(qed_cache) - 1, float(best_qed)))

        # normalize, measure ESS, resample if degenerate
        shifted = log_weights - log_weights.max()
        weights = np.exp(shifted)
        weights /= weights.sum()
        ess = 1.0 / np.sum(weights**2)
        if ess < n_particles / 2:
            idx = _systematic_resample(weights, rng)
            states = [states[j] for j in idx]
            times = [times[j] for j in idx]
            terminal = [terminal[j] for j in idx]
            log_weights = np.zeros(n_particles)
        if all(terminal[i] or times[i] >= horizon for i in range(n_particles)):
            break

    best_similarity = DataStructs.TanimotoSimilarity(lead_fp, _fp(best_state))
    diversity = _mean_pairwise_diversity(list(feasible_population.values()))
    check_scaffold = scaffold_mol if scaffold_mol is not None else measure_scaffold_mol
    scaffold_preserved = True
    if scaffold_mol is not None:
        best_mol = Chem.MolFromSmiles(_canon(best_state))
        scaffold_preserved = best_mol is not None and best_mol.HasSubstructMatch(scaffold_mol)
    # fraction of the distinct feasible population that preserves the scaffold --
    # with the constraint ON this is 1.0 by construction; with it OFF (measure_only)
    # this is the generate-then-filter baseline's usable fraction.
    population_mols = [Chem.MolFromSmiles(k) for k in feasible_population]
    population_mols = [m for m in population_mols if m is not None]

    def _pop_frac(pattern, want_present):
        # fraction of the distinct feasible population that SATISFIES the rule.
        # With the rule enforced this is 1.0 by construction; measured OFF it is
        # the generate-then-filter baseline's usable fraction. `pattern` may be a
        # single mol or a tuple of forbidden mols (satisfied = matches none).
        if pattern is None or not population_mols:
            return None
        pats = pattern if isinstance(pattern, (list, tuple)) else (pattern,)
        ok = sum(int(any(m.HasSubstructMatch(p) for p in pats) == want_present) for m in population_mols)
        return ok / len(population_mols)

    req_pattern = required_smarts if required_smarts is not None else measure_required_smarts
    forb_pattern = forbidden_smarts if forbidden_smarts is not None else measure_forbidden_smarts
    return {
        "lead_qed": float(lead_qed),
        "best_feasible_qed": float(best_qed),
        "best_similarity": float(best_similarity),
        "oracle_calls": len(qed_cache) - 1,
        "distinct_feasible": len(feasible_population),
        "population_diversity": float(diversity),
        "scaffold_preserved": bool(scaffold_preserved),
        "scaffold_preservation_fraction": _pop_frac(check_scaffold, True),
        "required_satisfaction_fraction": _pop_frac(req_pattern, True),
        "forbidden_satisfaction_fraction": _pop_frac(forb_pattern, False),
        "best_qed_trace": best_qed_trace,  # anytime curve: (oracle_calls, best_so_far)
        "best_state_smiles": _canon(best_state),  # the argmax-reward molecule
        # the distinct feasible-population SMILES (canonical), for offline evaluation of a
        # multi-constraint design spec (the conjunction feasibility funnel).
        "population_smiles": sorted(feasible_population.keys()) if dump_population else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("/private/tmp/lineage_b_checkpoint/checkpoint.best_so_far.pt"),
    )
    parser.add_argument("--leads", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-leads", type=int, default=12)
    parser.add_argument("--seed", type=int, default=20260726)
    parser.add_argument("--similarity-minimum", type=float, default=0.40)
    parser.add_argument("--target-qed", type=float, default=0.90)
    parser.add_argument("--budget", type=int, default=640)
    parser.add_argument("--n-particles", type=int, default=32)
    parser.add_argument("--max-steps", type=int, default=64)
    parser.add_argument("--time-step", type=float, default=0.1)
    parser.add_argument("--horizon", type=float, default=16.0)
    parser.add_argument("--alpha-start", type=float, default=0.20)
    parser.add_argument("--alpha-end", type=float, default=0.03)
    parser.add_argument("--feasible-attempts", type=int, default=8)
    parser.add_argument("--n-slots", type=int, default=40)
    parser.add_argument(
        "--atom-delete-log-rate-adjustment",
        type=float,
        default=-0.5,
        help="Base-sampler calibration; -0.5 matches the ceiling run, 0.0 = zero-calibration.",
    )
    parser.add_argument(
        "--value-twist-checkpoint",
        type=Path,
        default=None,
        help="Optional trained ValueTwistNet (griddd_value_twist.py) for V1 steering.",
    )
    parser.add_argument(
        "--value-lookahead-rollouts",
        type=int,
        default=0,
        help="V2: exact Monte-Carlo lookahead rollouts per state (0 = off). Budget-counted.",
    )
    parser.add_argument("--value-lookahead-depth", type=int, default=4)
    parser.add_argument(
        "--scaffold-constraint",
        choices=("none", "murcko", "measure_only"),
        default="none",
        help="'murcko': require each lead's Bemis-Murcko scaffold in EVERY candidate "
        "(exact fiber constraint -- the decisive hard-structural-constraint experiment).",
    )
    parser.add_argument(
        "--required-substructure",
        type=str,
        default=None,
        help="SMARTS every candidate MUST contain (hard fiber constraint).",
    )
    parser.add_argument(
        "--forbidden-substructure",
        type=str,
        default=None,
        help="SMARTS every candidate must NOT contain (hard fiber constraint).",
    )
    parser.add_argument(
        "--dump-population",
        action="store_true",
        help="emit the distinct feasible-population SMILES for offline design-spec evaluation.",
    )
    args = parser.parse_args()

    sampler, qed_oracle = _load_base_sampler(
        str(args.checkpoint), args.atom_delete_log_rate_adjustment
    )
    rewrite = de_novo_rewrite_system()
    twist = None
    if args.value_twist_checkpoint is not None:
        if __package__:
            from scripts.griddd_value_twist import load_value_twist
        else:
            from griddd_value_twist import load_value_twist
        twist = load_value_twist(str(args.value_twist_checkpoint))
    _req = Chem.MolFromSmarts(args.required_substructure) if args.required_substructure else None
    _forb = Chem.MolFromSmarts(args.forbidden_substructure) if args.forbidden_substructure else None
    _measure = args.scaffold_constraint == "measure_only"  # measure baseline, don't enforce
    required_smarts = None if _measure else _req
    forbidden_smarts = None if _measure else _forb
    measure_required_smarts = _req if _measure else None
    measure_forbidden_smarts = _forb if _measure else None
    leads = json.loads(args.leads.read_text())[: args.max_leads]

    results: list[dict[str, object]] = []
    for index, smiles, _listed_qed in leads:
        try:
            state = pad_molecular_graph(smiles_to_molecular_graph(smiles), args.n_slots)
        except Exception as error:  # noqa: BLE001
            results.append({"lead_index": index, "error": str(error)})
            continue
        scaffold_mol = None
        measure_scaffold_mol = None
        if args.scaffold_constraint in ("murcko", "measure_only"):
            lead_mol = Chem.MolFromSmiles(smiles)
            scaf = MurckoScaffold.GetScaffoldForMol(lead_mol) if lead_mol is not None else None
            scaf = scaf if scaf is not None and scaf.GetNumAtoms() > 0 else None
            if args.scaffold_constraint == "murcko":
                scaffold_mol = scaf
            else:
                measure_scaffold_mol = scaf
        outcome = value_guided_smc(
            state,
            sampler=sampler,
            qed_oracle=qed_oracle,
            rewrite=rewrite,
            seed=args.seed + int(index),
            similarity_minimum=args.similarity_minimum,
            budget=args.budget,
            n_particles=args.n_particles,
            max_steps=args.max_steps,
            time_step=args.time_step,
            horizon=args.horizon,
            alpha_start=args.alpha_start,
            alpha_end=args.alpha_end,
            feasible_attempts=args.feasible_attempts,
            twist=twist,
            lookahead_rollouts=args.value_lookahead_rollouts,
            lookahead_depth=args.value_lookahead_depth,
            scaffold_mol=scaffold_mol,
            measure_scaffold_mol=measure_scaffold_mol,
            required_smarts=required_smarts,
            forbidden_smarts=forbidden_smarts,
            measure_required_smarts=measure_required_smarts,
            measure_forbidden_smarts=measure_forbidden_smarts,
            dump_population=args.dump_population,
        )
        success = bool(
            outcome["best_feasible_qed"] >= args.target_qed
            and outcome["best_similarity"] >= args.similarity_minimum
        )
        results.append({"lead_index": index, "lead_smiles": smiles, "success": success, **outcome})
        args.output.write_text(json.dumps({"partial": True, "results": results}, indent=2) + "\n")
        print(
            f"lead {index}: {outcome['lead_qed']:.3f} -> {outcome['best_feasible_qed']:.3f} "
            f"(sim {outcome['best_similarity']:.3f}) success={success} "
            f"calls={outcome['oracle_calls']} div={outcome['population_diversity']:.3f}",
            flush=True,
        )

    scored = [r for r in results if "success" in r]
    n = len(scored)
    successes = sum(1 for r in scored if r["success"])
    summary = {
        "format": "compose_v4_value_guided_smc_controller_v1",
        "method": "value_guided_sequential_monte_carlo_reward_potential_hard_fiber",
        "base_checkpoint": str(args.checkpoint),
        "atom_delete_log_rate_adjustment": args.atom_delete_log_rate_adjustment,
        "value_twist": (str(args.value_twist_checkpoint) if args.value_twist_checkpoint else None),
        "value_lookahead_rollouts": args.value_lookahead_rollouts,
        "leads_scored": n,
        "success_at_target": successes,
        "success_rate": (successes / n) if n else None,
        "mean_best_feasible_qed": (sum(r["best_feasible_qed"] for r in scored) / n) if n else None,
        "mean_population_diversity": (sum(r["population_diversity"] for r in scored) / n) if n else None,
        "target_qed": args.target_qed,
        "similarity_minimum": args.similarity_minimum,
        "budget_per_lead": args.budget,
        "n_particles": args.n_particles,
        "griddd_reference_success_rate": 0.451,
        "best_first_ceiling_success_rate": 0.3333,
        "results": results,
    }
    args.output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: summary[k] for k in (
        "leads_scored", "success_at_target", "success_rate",
        "mean_best_feasible_qed", "mean_population_diversity")}))


if __name__ == "__main__":
    main()
