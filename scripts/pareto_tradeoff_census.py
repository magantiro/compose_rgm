"""Stage 0 tradeoff census for target-free Pareto / preference control.

Decides ONE thing: which objective pair, if any, can support a Pareto-control
experiment. It writes no controller, runs no arm, and is deliberately incapable
of expressing a preference between controllers.

WHY THIS RUNS BEFORE ANY CONTROLLER EXISTS
------------------------------------------
Exact-target recovery asks "can control reach a known molecule?", and a known
molecule hands the controller Tanimoto-to-the-answer -- a privileged heuristic
that does not exist in a real design problem. Pareto control removes the answer
and asks whether the SAME frozen process can pursue a user-specified tradeoff.
That question is only well posed if the objective pair actually HAS a tradeoff,
and whether it does is a property of the chemistry and the executor, not of the
controller. So it is measured first, against thresholds written down first.

The specific failure this guards against is on the record: a calibration in this
project reported "29/30 vs 28/30" and read it as a result. It was a CEILING --
the task was easy enough that every arm reached the top, so no controller
difference could have appeared. That is invisible after the fact unless the
headroom thresholds were fixed in advance. They are, in
docs/workstreams/pareto-control/PROTOCOL.md section 6.4, committed before this
script was run.

THE PAIR ORDER IS PREDECLARED AND IS NOT REORDERED HERE
    1. potency vs developability
    2. potency vs source similarity
    3. developability vs source similarity
The first pair clearing all five gates is adopted. This script reports all three
so the reader can see what the rejected pairs looked like, but the ADOPTION rule
is positional, not "best".

TWO INSTRUMENTS, OPPOSITE BIASES
--------------------------------
I-A  the real model-gated canonical successor fiber (~334 successors/state,
     ~5.5 s/state, local). The legal mask is structural, so the support
     enumerates WITHOUT the frozen R_theta weights; this script reads support
     and properties only and never reads a transition probability.
I-B  one-cut matched molecular pairs over the held-in pool. Median degree 1, so
     it understates tradeoff availability -- but it is drawn from real molecules
     rather than from the executor's own action inventory, so it can tell us
     whether a tradeoff is a fact about chemistry or an artifact of the operator
     set.

HELD-IN ONLY. `reserve_source_keys` is never read; the script asserts it.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

# ---------------------------------------------------------------------------
# Frozen constants. Every one of these is quoted in PROTOCOL.md and none may be
# changed by this script.
# ---------------------------------------------------------------------------

#: Held-in goal-language normalizers, diagnostics/retarget_goal_language_normalizers.json
CENTRE_P, S_P = -5.521657630461303, 2.6160788815821414
S_QED, S_LOGP = 0.3139245718923388, 2.0833575000000017

#: Frozen developability region and soft-min, from retarget_goal_calibration_app.py
QED_FLOOR = 0.6
LOGP_BOX = (1.0, 4.0)
MARGIN_CLIP = 1.5
SOFTMIN_TAU = 0.25

#: Analytic ceiling of the CLIPPED soft-min: both margins at +1.5.
Z_D_CEILING = float(-SOFTMIN_TAU * np.log(2.0 * np.exp(-MARGIN_CLIP / SOFTMIN_TAU)))

#: Frozen cohort size band (retarget_select_calibration_sources.py), outcome-independent.
MIN_HEAVY, MAX_HEAVY = 18, 38
CANONICAL_SLOTS = 48
TIME_POINT = 0.5

#: Preference grid and budget, PROTOCOL.md section 7.1.
PREFERENCE_GRID = (0.1, 0.3, 0.5, 0.7, 0.9)
FINE_GRID = tuple(np.linspace(0.0, 1.0, 101))
CHEBYSHEV_RHO = 1e-3
BUDGET = 6

#: Predeclared pair order. NOT reordered after results. PROTOCOL.md section 5.
PAIR_ORDER = (
    ("potency_vs_developability", "P", "D"),
    ("potency_vs_source_similarity", "P", "S"),
    ("developability_vs_source_similarity", "D", "S"),
)

#: Headroom gate thresholds, PROTOCOL.md section 6.4. Fixed before any number.
G1_MAX_RHO = 0.70
G2_MIN_TRADEOFF_FRACTION = 0.20
G2_MIN_EACH_DIRECTION = 0.05
G2_MIN_STATE_AVAILABILITY = 0.25
G3_MAX_BINDING_SHARE = 0.90
G4_MAX_REACH_FRACTION = 0.85
G5_MIN_DISTINCT_SELECTIONS = 2.0
G5_MAX_UNANIMOUS_FRACTION = 0.50

SEED = 20260813


# ---------------------------------------------------------------------------
# Objectives. All MAXIMIZED, all in held-in IQR units.
# ---------------------------------------------------------------------------

def z_potency(logodds: np.ndarray) -> np.ndarray:
    return (logodds - CENTRE_P) / S_P


def z_developability(qed: np.ndarray, clogp: np.ndarray) -> np.ndarray:
    m_qed = (qed - QED_FLOOR) / S_QED
    m_logp = np.minimum(clogp - LOGP_BOX[0], LOGP_BOX[1] - clogp) / S_LOGP
    stack = np.clip(np.stack([m_qed, m_logp]), -MARGIN_CLIP, MARGIN_CLIP)
    return -SOFTMIN_TAU * np.log(np.sum(np.exp(-stack / SOFTMIN_TAU), axis=0))


def chebyshev(z: np.ndarray, weight: float, utopia: np.ndarray) -> np.ndarray:
    """Augmented weighted Chebyshev, MINIMIZED. z is (n, 2).

    Chebyshev rather than a weighted sum because every Pareto point is the
    Chebyshev optimum for some weight, whereas a weighted sum can only ever
    select points on the convex hull -- it would systematically miss nonconvex
    regions of the front and make the front look smaller than it is.
    """
    gap = utopia[None, :] - z
    w = np.array([weight, 1.0 - weight])
    return np.max(w[None, :] * gap, axis=1) + CHEBYSHEV_RHO * np.sum(gap, axis=1)


def chebyshev_binding(z: np.ndarray, weight: float, utopia: np.ndarray) -> np.ndarray:
    gap = utopia[None, :] - z
    w = np.array([weight, 1.0 - weight])
    return np.argmax(w[None, :] * gap, axis=1)


def pareto_front_indices(z: np.ndarray) -> np.ndarray:
    """Indices of the nondominated set for MAXIMIZED objectives."""
    n = len(z)
    keep = np.ones(n, dtype=bool)
    for i in range(n):
        if not keep[i]:
            continue
        dominated = np.all(z <= z[i], axis=1) & np.any(z < z[i], axis=1)
        keep &= ~dominated
    return np.flatnonzero(keep)


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------

def batch_properties(smiles: list[str], oracle) -> dict[str, np.ndarray]:
    from rdkit import Chem
    from rdkit.Chem import Crippen, QED

    logodds = np.asarray(oracle.margin_many(smiles), dtype=float)
    qed = np.full(len(smiles), np.nan)
    clogp = np.full(len(smiles), np.nan)
    heavy = np.zeros(len(smiles), dtype=int)
    for i, s in enumerate(smiles):
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            continue
        try:
            qed[i] = QED.qed(mol)
        except Exception:  # noqa: BLE001
            pass
        clogp[i] = Crippen.MolLogP(mol)
        heavy[i] = mol.GetNumHeavyAtoms()
    return {"logodds": logodds, "qed": qed, "clogp": clogp, "heavy": heavy}


def fingerprints(smiles: list[str]):
    from rdkit import Chem
    from rdkit.Chem import rdFingerprintGenerator

    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    out = []
    for s in smiles:
        mol = Chem.MolFromSmiles(s)
        out.append(None if mol is None else gen.GetFingerprint(mol))
    return out


def tanimoto_to_reference(reference, others) -> np.ndarray:
    from rdkit import DataStructs

    if reference is None:
        return np.full(len(others), np.nan)
    return np.array([np.nan if f is None else DataStructs.TanimotoSimilarity(reference, f)
                     for f in others], dtype=float)


# ---------------------------------------------------------------------------
# Instrument I-A: the real model-gated canonical successor fiber
# ---------------------------------------------------------------------------

def build_local_kernel(active8: Path, gate_zero: Path, matscorer: Path):
    """Enumerates the FROZEN legal support without the frozen R_theta weights.

    The legal mask comes from the model's structural action tables, not from its
    parameters. This census reads support and properties only and never reads a
    transition probability, so materialized (untrained) weights are sufficient
    and the checkpoint on the Modal volume is not needed.
    """
    import torch

    from compose_v4.experiments.editing_v2_process_v2_t1_panel import (
        open_process_v2_t1_source,
    )
    from compose_v4.experiments.editing_v2_process_v2_t1_runtime import (
        build_process_v2_score_revised_scratch_runtime,
        load_materialized_scorer_state,
    )

    source = open_process_v2_t1_source(
        active8,
        gate_zero_decision_path=gate_zero,
        artifact_root=gate_zero.parent.parent.parent,
        repo_root=REPO,
    )
    bundle = load_materialized_scorer_state(matscorer)
    runtime, _b, _c = build_process_v2_score_revised_scratch_runtime(
        source, materialized_state=bundle)
    model = runtime.model
    model.eval()
    torch.set_grad_enabled(False)
    return model


def enumerate_support(model, smiles: str) -> list[str]:
    import torch

    from compose_v4.chem.molecular_graph import smiles_to_molecular_graph
    from compose_v4.chem.state import pad_molecular_graph
    from compose_v4.experiments.production_successor_kernel import (
        canonical_successor_result,
    )

    try:
        state = pad_molecular_graph(smiles_to_molecular_graph(smiles), CANONICAL_SLOTS)
    except Exception:  # noqa: BLE001
        return []
    with torch.no_grad():
        result = canonical_successor_result(model, state, float(TIME_POINT))
    return [s.key for s in result.batch.successors]


# ---------------------------------------------------------------------------
# The census itself
# ---------------------------------------------------------------------------

def objective_matrix(props: dict[str, np.ndarray], tanimoto: np.ndarray,
                     centre_s: float, s_s: float) -> dict[str, np.ndarray]:
    return {
        "P": z_potency(props["logodds"]),
        "D": z_developability(props["qed"], props["clogp"]),
        "S": (tanimoto - centre_s) / s_s,
    }


def census_states(states: list[dict], key_a: str, key_b: str,
                  utopia: np.ndarray) -> dict[str, Any]:
    """C2, C3, C4 over a list of decision states for ONE objective pair."""
    quadrants = {"pp": 0, "pm": 0, "mp": 0, "mm": 0, "zero": 0}
    n_moves = 0
    has_pm = has_mp = 0
    front_sizes, fine_distinct, distinct_counts, unanimous = [], [], [], 0
    binding = [0, 0]
    n_states = 0
    front_ranges_a, front_ranges_b = [], []

    for st in states:
        za = st["z"][key_a]
        zb = st["z"][key_b]
        ok = np.isfinite(za) & np.isfinite(zb)
        if ok.sum() < 2:
            continue
        z = np.stack([za[ok], zb[ok]], axis=1)
        n_states += 1

        # --- C3: sign quadrant of each legal move, relative to the state -----
        da = za[ok] - st["state_z"][key_a]
        db = zb[ok] - st["state_z"][key_b]
        eps = 1e-9
        pm = (da > eps) & (db < -eps)
        mp = (da < -eps) & (db > eps)
        pp = (da > eps) & (db > eps)
        mm = (da < -eps) & (db < -eps)
        quadrants["pm"] += int(pm.sum())
        quadrants["mp"] += int(mp.sum())
        quadrants["pp"] += int(pp.sum())
        quadrants["mm"] += int(mm.sum())
        quadrants["zero"] += int(len(da) - pm.sum() - mp.sum() - pp.sum() - mm.sum())
        n_moves += len(da)
        has_pm += int(pm.any())
        has_mp += int(mp.any())

        # --- C2: front size and extent --------------------------------------
        front = pareto_front_indices(z)
        front_sizes.append(len(front))
        front_ranges_a.append(float(np.ptp(z[front, 0])))
        front_ranges_b.append(float(np.ptp(z[front, 1])))
        fine = {int(np.argmin(chebyshev(z, w, utopia))) for w in FINE_GRID}
        fine_distinct.append(len(fine))

        # --- C4: preference distinguishability -------------------------------
        picks = []
        for w in PREFERENCE_GRID:
            scores = chebyshev(z, w, utopia)
            idx = int(np.argmin(scores))
            picks.append(idx)
            binding[int(chebyshev_binding(z[idx: idx + 1], w, utopia)[0])] += 1
        distinct_counts.append(len(set(picks)))
        unanimous += int(len(set(picks)) == 1)

    total_binding = max(sum(binding), 1)
    return {
        "n_states": n_states,
        "n_moves": n_moves,
        "quadrant_fractions": {k: v / max(n_moves, 1) for k, v in quadrants.items()},
        "tradeoff_move_fraction": (quadrants["pm"] + quadrants["mp"]) / max(n_moves, 1),
        "state_offers_a_up_b_down": has_pm / max(n_states, 1),
        "state_offers_a_down_b_up": has_mp / max(n_states, 1),
        "mean_front_size": float(np.mean(front_sizes)) if front_sizes else 0.0,
        "mean_front_extent_a": float(np.mean(front_ranges_a)) if front_ranges_a else 0.0,
        "mean_front_extent_b": float(np.mean(front_ranges_b)) if front_ranges_b else 0.0,
        "mean_fine_grid_distinct": float(np.mean(fine_distinct)) if fine_distinct else 0.0,
        "mean_distinct_selections": float(np.mean(distinct_counts)) if distinct_counts else 0.0,
        "unanimous_fraction": unanimous / max(n_states, 1),
        "binding_share": [b / total_binding for b in binding],
        "max_binding_share": max(binding) / total_binding,
    }


def greedy_single_objective_rollout(model, oracle, source: str, objective: str,
                                    budget: int, centre_s: float, s_s: float,
                                    fps_cache: dict) -> dict[str, Any]:
    """Roll out `budget` edits maximizing ONE objective. This is the C5/G4
    instrument: "does the budget exhaust this axis?" is a question about what a
    single-objective policy can reach, and nothing weaker answers it.

    Deliberately NOT "the best candidate available at one state". With a
    ~589-wide fiber, `max over candidates >= pooled p99` fires with probability
    0.997 whatever the truth is; it has no falsifying range and is not a
    measurement. See DECISION_LOG D-007.
    """
    from rdkit import Chem
    from rdkit.Chem import Crippen, QED, rdFingerprintGenerator
    from rdkit import DataStructs

    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

    def fp(smi: str):
        if smi not in fps_cache:
            mol = Chem.MolFromSmiles(smi)
            fps_cache[smi] = None if mol is None else gen.GetFingerprint(mol)
        return fps_cache[smi]

    ref = fp(source)

    def score(keys: list[str]) -> np.ndarray:
        if objective == "P":
            return z_potency(np.asarray(oracle.margin_many(keys), dtype=float))
        if objective == "D":
            qed, clogp = [], []
            for k in keys:
                mol = Chem.MolFromSmiles(k)
                if mol is None:
                    qed.append(np.nan); clogp.append(np.nan); continue
                try:
                    qed.append(QED.qed(mol))
                except Exception:  # noqa: BLE001
                    qed.append(np.nan)
                clogp.append(Crippen.MolLogP(mol))
            return z_developability(np.asarray(qed), np.asarray(clogp))
        tan = np.array([np.nan if (f := fp(k)) is None
                        else DataStructs.TanimotoSimilarity(ref, f) for k in keys])
        return (tan - centre_s) / s_s

    current = source
    trace = [float(score([source])[0])]
    for _ in range(budget):
        candidates = enumerate_support(model, current)
        if not candidates:
            break
        values = score(candidates)
        if not np.isfinite(values).any():
            break
        best = int(np.nanargmax(values))
        current = candidates[best]
        trace.append(float(values[best]))
    return {"source": source, "objective": objective, "endpoint": current,
            "trace": trace, "start": trace[0], "end": trace[-1],
            "movement": trace[-1] - trace[0], "edits": len(trace) - 1}


def committed_reach_fractions(calibration: Path, utopia_p: float) -> dict[str, Any]:
    """C5/G4 from COMMITTED held-in real-fiber rollouts.

    This is the only part of the census that uses multi-edit trajectories, and
    it deliberately reuses an existing committed artifact rather than launching
    a run: the question "does a 6-edit budget exhaust this axis?" was already
    answered for potency and developability by a real fiber rollout.
    """
    data = json.loads(calibration.read_text())
    per_source = data["per_source"]

    # Potency: the A-prefix maximises DRD2 log-odds for 3 real edits.
    best_logodds = np.array([max(t["drd2_logodds"] for t in p["trajectory"])
                             for p in per_source])
    potency_reach = float(np.mean(z_potency(best_logodds) >= utopia_p))

    # Developability: 3 further real edits under the frozen region objective.
    develop_binary = float(np.mean([p["develop_greedy_success"] for p in per_source]))
    develop_ceiling = float(np.mean(
        [p["develop_greedy_score"] >= Z_D_CEILING - 1e-3 for p in per_source]))
    return {
        "source": str(calibration),
        "n_sources": len(per_source),
        "edits_potency": 3,
        "edits_developability": 3,
        "P_reach_fraction_vs_utopia_p99": potency_reach,
        "D_reach_fraction_binary_region": develop_binary,
        "D_reach_fraction_clip_ceiling": develop_ceiling,
        "z_D_clip_ceiling": Z_D_CEILING,
        "note": ("Both are measured at 3 edits, not 6. A 6-edit budget can only "
                 "increase a reach fraction, so a value already above the G4 "
                 "threshold at 3 edits is conclusive; a value below it at 3 "
                 "edits is not conclusive for 6 and is reported as such."),
    }


def evaluate_gates(pair_name: str, a: str, b: str, rho_pool: float,
                   ia: dict, ib: dict, reach: dict[str, float]) -> dict[str, Any]:
    """The five predeclared gates. Thresholds are module constants, fixed in
    PROTOCOL.md before any of these numbers existed."""

    def g2(stats: dict) -> tuple[bool, list[str]]:
        why = []
        if stats["tradeoff_move_fraction"] < G2_MIN_TRADEOFF_FRACTION:
            why.append(f"tradeoff moves {stats['tradeoff_move_fraction']:.3f} "
                       f"< {G2_MIN_TRADEOFF_FRACTION}")
        for k, label in (("pm", f"{a}+/{b}-"), ("mp", f"{a}-/{b}+")):
            if stats["quadrant_fractions"][k] < G2_MIN_EACH_DIRECTION:
                why.append(f"{label} {stats['quadrant_fractions'][k]:.3f} "
                           f"< {G2_MIN_EACH_DIRECTION}")
        for k, label in (("state_offers_a_up_b_down", f"states offering {a}+/{b}-"),
                         ("state_offers_a_down_b_up", f"states offering {a}-/{b}+")):
            if stats[k] < G2_MIN_STATE_AVAILABILITY:
                why.append(f"{label} {stats[k]:.3f} < {G2_MIN_STATE_AVAILABILITY}")
        return (not why), why

    def g5(stats: dict) -> tuple[bool, list[str]]:
        why = []
        if stats["mean_distinct_selections"] < G5_MIN_DISTINCT_SELECTIONS:
            why.append(f"mean distinct selections "
                       f"{stats['mean_distinct_selections']:.2f} "
                       f"< {G5_MIN_DISTINCT_SELECTIONS}")
        if stats["unanimous_fraction"] > G5_MAX_UNANIMOUS_FRACTION:
            why.append(f"unanimous states {stats['unanimous_fraction']:.3f} "
                       f"> {G5_MAX_UNANIMOUS_FRACTION}")
        return (not why), why

    g2_ia, g2_ia_why = g2(ia)
    g2_ib, g2_ib_why = g2(ib) if ib["n_states"] else (None, ["I-B unavailable"])
    g5_ia, g5_ia_why = g5(ia)
    g5_ib, g5_ib_why = g5(ib) if ib["n_states"] else (None, ["I-B unavailable"])

    def combine(pass_ia: bool, pass_ib: bool | None) -> tuple[bool, str]:
        """PROTOCOL.md section 6.2. I-A is primary because the controller can
        only act through the executable support; a tradeoff invisible to the
        executor is not actionable."""
        if pass_ib is None:
            return pass_ia, "I-A only"
        if pass_ia and pass_ib:
            return True, "both instruments agree"
        if pass_ia and not pass_ib:
            return True, "OPERATOR_SET_DEPENDENT: present in the fiber, absent in MMP pairs"
        if not pass_ia and pass_ib:
            return False, "fails on the executable support; MMP agreement is not actionable"
        return False, "both instruments agree"

    g2_pass, g2_note = combine(g2_ia, g2_ib)
    g5_pass, g5_note = combine(g5_ia, g5_ib)

    g1_pass = rho_pool < G1_MAX_RHO
    g3_pass = ia["max_binding_share"] <= G3_MAX_BINDING_SHARE
    g4_fail_axes = [k for k, v in reach.items() if v > G4_MAX_REACH_FRACTION]
    g4_pass = not g4_fail_axes

    gates = {
        "G1_alignment": {
            "pass": bool(g1_pass), "statistic": "spearman_rho_pool",
            "value": rho_pool, "reject_if": f">= {G1_MAX_RHO}",
            "falsifying_range": "[-1, +1]; rho >= 0.70 rejects",
        },
        "G2_local_tradeoff": {
            "pass": bool(g2_pass), "note": g2_note,
            "I_A": {"pass": bool(g2_ia), "why_failed": g2_ia_why},
            "I_B": {"pass": g2_ib, "why_failed": g2_ib_why},
            "falsifying_range": "[0, 1] per component",
        },
        "G3_no_domination": {
            "pass": bool(g3_pass), "statistic": "max_binding_share",
            "value": ia["max_binding_share"], "reject_if": f"> {G3_MAX_BINDING_SHARE}",
            "falsifying_range": "[0.5, 1.0]; 0.5 is perfectly balanced",
        },
        "G4_no_saturation": {
            "pass": bool(g4_pass), "reach_fractions": reach,
            "failing_axes": g4_fail_axes, "reject_if": f"any > {G4_MAX_REACH_FRACTION}",
            "falsifying_range": "[0, 1] per axis",
        },
        "G5_front_richness": {
            "pass": bool(g5_pass), "note": g5_note,
            "I_A": {"pass": bool(g5_ia), "why_failed": g5_ia_why,
                    "mean_distinct_selections": ia["mean_distinct_selections"],
                    "unanimous_fraction": ia["unanimous_fraction"]},
            "I_B": {"pass": g5_ib, "why_failed": g5_ib_why},
            "falsifying_range": "[1, 5] distinct selections; 1 means preferences collapse",
        },
    }
    all_pass = all(g["pass"] for g in gates.values())
    return {"pair": pair_name, "gates": gates, "verdict": "PASS" if all_pass else "FAIL",
            "failed": [k for k, g in gates.items() if not g["pass"]]}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reserve", type=Path,
                        default=REPO / "diagnostics/editing_v2_matched_validation_reserve_ids.json.gz")
    parser.add_argument("--manifest", type=Path,
                        default=REPO / "artifacts/oracles/drd2_svm_v1/drd2_oracle_manifest.json")
    parser.add_argument("--calibration", type=Path,
                        default=REPO / "diagnostics/retarget_calibration_result_3plus3_fixed.json")
    parser.add_argument("--active8", type=Path, default=Path(os.environ.get(
        "COMPOSE_ACTIVE8",
        "/Users/rmaganti/compose_trainset_backup/localprep/artifacts/editing_v2/"
        "process_v2_active8/8ecc0e5e825a15200560c58960d4f662c9ca23be785c825c9c24aa73308144bb")))
    parser.add_argument("--gate-zero", type=Path, default=Path(os.environ.get(
        "COMPOSE_GATE_ZERO",
        "/Users/rmaganti/compose_trainset_backup/localprep/artifacts/editing_v2/"
        "process_v2_gate_zero_v7/DECISION.json")))
    parser.add_argument("--matscorer", type=Path, default=Path(os.environ.get(
        "COMPOSE_MATSCORER",
        "/Users/rmaganti/compose_trainset_backup/materialized_scorer")))
    parser.add_argument("--sources", type=int, default=120)
    parser.add_argument("--reach-sources", type=int, default=20)
    parser.add_argument("--pool-sample", type=int, default=20000)
    parser.add_argument("--mmp-pool", type=int, default=96094)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    from rdkit import RDLogger

    from build_analogue_trace_pool import mine_one_cut_pairs
    from compose_v4.drd2_oracle import load_default_oracle

    RDLogger.DisableLog("rdApp.*")
    started = time.perf_counter()
    oracle = load_default_oracle(str(args.manifest))

    # --- Held-in only. The reserve key is never touched. --------------------
    split = json.load(gzip.open(args.reserve, "rt"))
    held_in = sorted(split["training_source_keys"])
    assert "reserve_source_keys" in split, "unexpected split schema"
    del split  # the held-out list is not carried past this point
    print(f"held-in pool {len(held_in):,}  (held-out reserve NOT read)", flush=True)

    rng = random.Random(SEED)

    # --- Phase 0: freeze the scales BEFORE any gate statistic ---------------
    pool = list(held_in)
    rng.shuffle(pool)
    pool_sample = pool[: args.pool_sample]
    pool_props = batch_properties(pool_sample, oracle)
    pool_zp = z_potency(pool_props["logodds"])
    pool_zd = z_developability(pool_props["qed"], pool_props["clogp"])
    finite = np.isfinite(pool_zp) & np.isfinite(pool_zd)
    print(f"[{time.perf_counter()-started:6.1f}s] scored pool sample "
          f"{int(finite.sum()):,}", flush=True)

    from scipy.stats import spearmanr
    rho_pool_pd = float(spearmanr(pool_zp[finite], pool_zd[finite]).statistic)

    # --- Instrument I-A: real fiber ----------------------------------------
    eligible = [s for s, h in zip(pool_sample, pool_props["heavy"])
                if MIN_HEAVY <= h <= MAX_HEAVY]
    rng.shuffle(eligible)
    chosen = eligible[: args.sources]
    print(f"eligible held-in sources in [{MIN_HEAVY},{MAX_HEAVY}] heavy atoms: "
          f"{len(eligible):,}; censusing {len(chosen)}", flush=True)

    model = build_local_kernel(args.active8, args.gate_zero, args.matscorer)
    print(f"[{time.perf_counter()-started:6.1f}s] local kernel built", flush=True)

    ia_states: list[dict] = []
    step_rng = random.Random(SEED + 1)
    for i, src in enumerate(chosen):
        # Depth 0: the source itself.
        d0 = enumerate_support(model, src)
        if not d0:
            continue
        ia_states.append({"source": src, "depth": 0, "state": src, "candidates": d0})
        # Depth 1: a UNIFORMLY RANDOM legal successor. Objective-independent by
        # construction, so the same depth-1 states serve all three candidate
        # pairs and no pair is handed a state selected in its favour.
        step_state = d0[step_rng.randrange(len(d0))]
        d1 = enumerate_support(model, step_state)
        if d1:
            ia_states.append({"source": src, "depth": 1,
                              "state": step_state, "candidates": d1})
        if (i + 1) % 20 == 0:
            print(f"  [{time.perf_counter()-started:6.1f}s] {i+1}/{len(chosen)} sources, "
                  f"{len(ia_states)} states", flush=True)

    print(f"[{time.perf_counter()-started:6.1f}s] I-A: {len(ia_states)} decision states, "
          f"mean fiber "
          f"{np.mean([len(s['candidates']) for s in ia_states]):.0f}", flush=True)

    # --- Instrument I-B: one-cut matched pairs ------------------------------
    mmp_pool = held_in[: args.mmp_pool]
    pairs = mine_one_cut_pairs(mmp_pool)
    neighbours: dict[str, list[str]] = {}
    for x, y in pairs:
        neighbours.setdefault(x, []).append(y)
        neighbours.setdefault(y, []).append(x)
    eligible_set = set(eligible)
    ib_states = [{"source": s, "depth": 0, "state": s, "candidates": n}
                 for s, n in neighbours.items() if len(n) >= 2 and s in eligible_set]
    print(f"[{time.perf_counter()-started:6.1f}s] I-B: {len(pairs):,} MMP pairs, "
          f"{len(ib_states)} eligible decision states", flush=True)

    # --- Score every state and candidate ------------------------------------
    def decorate(states: list[dict]) -> None:
        universe = sorted({s["state"] for s in states} |
                          {c for s in states for c in s["candidates"]} |
                          {s["source"] for s in states})
        props = batch_properties(universe, oracle)
        fps = fingerprints(universe)
        index = {s: i for i, s in enumerate(universe)}
        zp = z_potency(props["logodds"])
        zd = z_developability(props["qed"], props["clogp"])
        for st in states:
            idx = np.array([index[c] for c in st["candidates"]])
            ref = fps[index[st["source"]]]
            tan = tanimoto_to_reference(ref, [fps[j] for j in idx])
            tan_state = tanimoto_to_reference(
                ref, [fps[index[st["state"]]]])[0]
            st["_raw"] = {"zp": zp[idx], "zd": zd[idx], "tan": tan,
                          "state_zp": zp[index[st["state"]]],
                          "state_zd": zd[index[st["state"]]],
                          "state_tan": tan_state}

    decorate(ia_states)
    decorate(ib_states)
    print(f"[{time.perf_counter()-started:6.1f}s] properties scored", flush=True)

    # Similarity normalizer: SAME RECIPE as the frozen normalizers (held-in
    # median and held-in IQR), applied to the property they did not cover.
    # Computed here, in Phase 0, before any gate statistic is evaluated.
    all_tan = np.concatenate([s["_raw"]["tan"] for s in ia_states])
    all_tan = all_tan[np.isfinite(all_tan)]
    centre_s = float(np.median(all_tan))
    s_s = float(np.percentile(all_tan, 75) - np.percentile(all_tan, 25))
    if s_s <= 0:
        s_s = 1.0
    print(f"frozen similarity normalizer: centre {centre_s:.4f}  s {s_s:.4f}", flush=True)

    for states in (ia_states, ib_states):
        for st in states:
            raw = st["_raw"]
            st["z"] = {"P": raw["zp"], "D": raw["zd"],
                       "S": (raw["tan"] - centre_s) / s_s}
            st["state_z"] = {"P": raw["state_zp"], "D": raw["state_zd"],
                             "S": (raw["state_tan"] - centre_s) / s_s}

    # Utopia / reference points, frozen from held-in.
    utopia_all = {
        "P": float(np.nanpercentile(pool_zp, 99)),
        "D": float(np.nanpercentile(pool_zd, 99)),
        "S": float(np.nanpercentile((all_tan - centre_s) / s_s, 99)),
    }
    reference_all = {
        "P": float(np.nanpercentile(pool_zp, 5)),
        "D": float(np.nanpercentile(pool_zd, 5)),
        "S": float(np.nanpercentile((all_tan - centre_s) / s_s, 5)),
    }
    print("utopia z*", {k: round(v, 4) for k, v in utopia_all.items()},
          " reference r", {k: round(v, 4) for k, v in reference_all.items()}, flush=True)

    # --- C5 / G4 reach fractions -------------------------------------------
    # Corroborating evidence from committed real-fiber rollouts (3 edits).
    reach_pd = committed_reach_fractions(args.calibration, utopia_all["P"])

    # The gate statistic itself: single-objective greedy rollouts at the FULL
    # 6-edit budget, run here. Ceilings:
    #   D  analytic -- the clipped soft-min cannot exceed Z_D_CEILING
    #   P  no analytic ceiling; held-in pool p99
    #   S  no ceiling is reachable at all (T = 1 needs zero edits), so instead
    #      the axis is tested for INERTNESS: an objective that six real edits
    #      cannot move is not controllable and degenerates the pair.
    ceilings = {"D": Z_D_CEILING - 1e-3, "P": utopia_all["P"]}
    fps_cache: dict[str, Any] = {}
    reach_sources = chosen[: args.reach_sources]
    rollouts: dict[str, list[dict]] = {}
    for objective in ("P", "D", "S"):
        rollouts[objective] = [
            greedy_single_objective_rollout(model, oracle, src, objective, BUDGET,
                                            centre_s, s_s, fps_cache)
            for src in reach_sources]
        print(f"[{time.perf_counter()-started:6.1f}s] reach rollouts {objective}: "
              f"{len(rollouts[objective])} sources, median movement "
              f"{np.median([r['movement'] for r in rollouts[objective]]):+.3f}",
              flush=True)

    reach_by_objective = {}
    for objective in ("P", "D"):
        ends = np.array([r["end"] for r in rollouts[objective]])
        reach_by_objective[objective] = float(np.mean(ends >= ceilings[objective]))
    s_movement = np.array([abs(r["movement"]) for r in rollouts["S"]])
    s_inert = float(np.median(s_movement) < 0.05)
    #: An inert axis is treated as fully saturated: nothing a controller does
    #: moves it, so every arm ends in the same place on that axis.
    reach_by_objective["S"] = 1.0 if s_inert else 0.0

    reach_detail = {
        "instrument": "single-objective greedy rollouts at the full 6-edit budget",
        "n_sources": len(reach_sources),
        "ceilings": ceilings,
        "P": {"reach_fraction": reach_by_objective["P"],
              "median_end": float(np.median([r["end"] for r in rollouts["P"]])),
              "median_movement": float(np.median([r["movement"] for r in rollouts["P"]]))},
        "D": {"reach_fraction": reach_by_objective["D"],
              "median_end": float(np.median([r["end"] for r in rollouts["D"]])),
              "median_movement": float(np.median([r["movement"] for r in rollouts["D"]])),
              "analytic_clip_ceiling": Z_D_CEILING},
        "S": {"inert": bool(s_inert), "median_abs_movement": float(np.median(s_movement)),
              "rule": "no ceiling is reachable (T=1 needs zero edits); the axis is "
                      "tested for inertness instead, threshold 0.05 in z_S units",
              "reach_fraction_assigned": reach_by_objective["S"],
              # Auditable evidence for the inertness verdict: if a
              # similarity-maximizing policy can always find a successor whose
              # ECFP4 fingerprint is identical to the source, "stay similar" is
              # free and the axis exerts no tradeoff pressure at all.
              "endpoint_tanimoto_to_source": [
                  float(r["trace"][-1] * s_s + centre_s) for r in rollouts["S"]],
              "traces": [r["trace"] for r in rollouts["S"]]},
        "withdrawn_statistic": (
            "An earlier draft used 'the best candidate at a state reaches the pooled "
            "p99'. With a ~589-wide fiber that fires with probability 0.997 whatever "
            "the truth is -- no falsifying range, so not a measurement. Withdrawn "
            "before any pair verdict was read; see DECISION_LOG D-007."),
        "corroboration_3_edit_committed": reach_pd,
    }
    print("G4 reach fractions", {k: round(v, 3) for k, v in reach_by_objective.items()},
          flush=True)

    # --- Run the census on all three pairs, in the predeclared order --------
    results, adopted = [], None
    for name, a, b in PAIR_ORDER:
        utopia = np.array([utopia_all[a], utopia_all[b]])
        ia = census_states(ia_states, a, b, utopia)
        ib = census_states(ib_states, a, b, utopia)
        if a == "P" and b == "D":
            rho = rho_pool_pd
        else:
            za = np.concatenate([st["z"][a] for st in ia_states])
            zb = np.concatenate([st["z"][b] for st in ia_states])
            ok = np.isfinite(za) & np.isfinite(zb)
            rho = float(spearmanr(za[ok], zb[ok]).statistic)
        gate = evaluate_gates(name, a, b, rho,  ia, ib,
                              {a: reach_by_objective[a], b: reach_by_objective[b]})
        entry = {"pair": name, "objective_a": a, "objective_b": b,
                 "spearman_rho": rho, "I_A": ia, "I_B": ib, "gate": gate}
        results.append(entry)
        if adopted is None and gate["verdict"] == "PASS":
            adopted = name
        print(f"\n=== {name}  rho={rho:+.3f}  {gate['verdict']}", flush=True)
        if gate["failed"]:
            print(f"    failed: {', '.join(gate['failed'])}", flush=True)
        print(f"    I-A  tradeoff moves {ia['tradeoff_move_fraction']:.3f}  "
              f"states {a}+/{b}- {ia['state_offers_a_up_b_down']:.3f}  "
              f"{a}-/{b}+ {ia['state_offers_a_down_b_up']:.3f}", flush=True)
        print(f"    I-A  distinct selections {ia['mean_distinct_selections']:.2f}/5  "
              f"unanimous {ia['unanimous_fraction']:.3f}  "
              f"front {ia['mean_front_size']:.1f}  "
              f"fine-grid distinct {ia['mean_fine_grid_distinct']:.2f}", flush=True)
        print(f"    I-B  tradeoff moves {ib['tradeoff_move_fraction']:.3f}  "
              f"distinct selections {ib['mean_distinct_selections']:.2f}/5", flush=True)

    payload = {
        "schema": "compose.pareto.tradeoff_census",
        "status": "SMOKE_HELD_IN",
        "held_out_opened": False,
        "protocol": "docs/workstreams/pareto-control/PROTOCOL.md",
        "predeclared_pair_order": [p[0] for p in PAIR_ORDER],
        "predeclared_gate_thresholds": {
            "G1_max_rho": G1_MAX_RHO,
            "G2_min_tradeoff_fraction": G2_MIN_TRADEOFF_FRACTION,
            "G2_min_each_direction": G2_MIN_EACH_DIRECTION,
            "G2_min_state_availability": G2_MIN_STATE_AVAILABILITY,
            "G3_max_binding_share": G3_MAX_BINDING_SHARE,
            "G4_max_reach_fraction": G4_MAX_REACH_FRACTION,
            "G5_min_distinct_selections": G5_MIN_DISTINCT_SELECTIONS,
            "G5_max_unanimous_fraction": G5_MAX_UNANIMOUS_FRACTION,
        },
        "seed": SEED,
        "budget": BUDGET,
        "preference_grid": list(PREFERENCE_GRID),
        "instruments": {
            "I_A": {"kind": "model-gated canonical successor fiber",
                    "n_states": len(ia_states),
                    "mean_fiber_width": float(np.mean(
                        [len(s["candidates"]) for s in ia_states])) if ia_states else 0.0,
                    "depths": [0, 1],
                    "reads_transition_probabilities": False,
                    "gate_zero": str(args.gate_zero),
                    "support_identical_to_v6": True},
            "I_B": {"kind": "one-cut matched molecular pairs",
                    "n_states": len(ib_states), "n_pairs": len(pairs)},
        },
        "frozen_scales": {
            "utopia_p99": utopia_all, "reference_p5": reference_all,
            "similarity_centre": centre_s, "similarity_iqr": s_s,
            "recipe": "held-in median and held-in IQR, the same recipe as the frozen "
                      "goal-language normalizers, applied to the one property they "
                      "did not cover",
        },
        "C5_reachability": reach_detail,
        "G4_reach_by_objective": reach_by_objective,
        "pairs": results,
        "adopted_pair": adopted,
        "adoption_rule": ("FIRST pair in the predeclared order clearing all five "
                          "gates. Positional, not best-of."),
        "seconds": round(time.perf_counter() - started, 1),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, default=float) + "\n")
    digest = hashlib.sha256(args.out.read_bytes()).hexdigest()
    print(f"\nadopted pair: {adopted}")
    print(f"wrote {args.out}  sha256 {digest[:16]}  "
          f"[{time.perf_counter()-started:.0f}s]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
