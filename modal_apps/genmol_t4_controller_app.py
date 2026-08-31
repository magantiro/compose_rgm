"""The strong COMPOSE controller for T4. One architecture, switchable control.

Level 1 stays frozen as a baseline. This is a separate engine built around the
same frozen R_theta, addressing the three defects the diagnostics actually
found rather than the one we first assumed:

  parp1  129 feasible molecules enumerated, 4 docked. One scalar weight was
         deciding BOTH molecular navigation and oracle allocation.
  fa7    route steps rank at median 7.2%, never outside APPLY_CAP, yet the
         lineage carrying them is discarded. Collapse, not misranking.
  braf   no route found by the beam procedure at all.

So the backbone is the fix, and adaptive h-control sits on top of it:

    frozen R_theta
      + persistent multi-lineage graph search with a transposition table
      + SEPARATE search and oracle archives
      + online docking surrogate driving acquisition
      + adaptive finite-horizon h-control                   [switchable]

CONTROL is a one-flag ablation inside this same engine, so the comparison is
free of implementation differences:

    control=immediate   score a successor by its own predicted desirability
    control=future_h    score it by E_{R_theta}[g_t(X_b) | X_0 = y], the
                        desirability of the futures it leaves reachable

The h-transform is APPROXIMATE here and is not claimed otherwise: g_t is built
from an online surrogate and h is a bounded Monte Carlo estimate, so Theorem 3
does not apply. It is an adaptive Doob-style controller, not the exact one.

Nothing is tuned per receptor, seed or delta. R_theta is never retrained.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import modal

from modal_apps.genmol_t4_opt_app import (
    APPLY_CAP, CANONICAL_SLOTS, QED_MIN, SA_MAX, TIME_POINT,
    ARTIFACT_ROOT, REMOTE_ROOT, ROOT, artifact_volume, _dock, _dock_many, _runtime,
)
from modal_apps.genmol_t4_opt_app import image as _opt_image

image = _opt_image.add_local_file(
    ROOT / "modal_apps/genmol_t4_opt_app.py",
    str(REMOTE_ROOT / "modal_apps/genmol_t4_opt_app.py"), copy=True)

app = modal.App("genmol-t4-controller")

# ---- frozen policy constants. Global, never per-cell. ----
N_LINEAGE      = 8       # active parents kept, vs 3 collapsing ones in Level 1
ROLLOUTS       = 6       # R_theta continuations per h estimate
H_BUDGET       = 40      # max successors expanded inside one h estimate
TAU_U          = 1.0     # temperature on predicted docking desirability
TAU_V          = 0.10    # graded feasibility temperature, search only
ENSEMBLE       = 8       # bootstrap members in the ridge surrogate
RIDGE          = 1.0
WARMUP         = 24      # diverse feasible dockings before the surrogate leads
# Docking workers per cell. MEASURED, not chosen: w=4 gave 2.52x, w=8 only 2.07x,
# and w=16 collapsed to 0.48x while silently returning results for just 4 of 20
# molecules. More workers is not safer here -- it loses dockings without erroring.
DOCK_WORKERS   = 4
MACRO_L            = (2, 4)   # extra endpoint horizons; L=1 is the ordinary fiber
# FIXED GLOBAL BUDGET, not per-parent. The high-recall version generated
# 3 x 8 parents x 2 horizons = 48 trajectories and drove candidate pools to ~4,200,
# but macro generation became 93% of round time and rounds hit 250-320s. Cells whose
# round exceeds the mean time-to-preemption cannot complete a round AT ALL: they
# resume, work, and are killed before the next persist. Six of fourteen cells were
# stuck at round 1 indefinitely. The scientific question is whether nonlocal
# endpoints improve the candidate distribution, not whether we can afford to
# enumerate thousands of them.
MACRO_BUDGET       = 16       # total sampled trajectories per round, all parents
# Soft-reward weight. Docking desirability and constraint shortfall are each
# standardised within the round before combining, so LAMBDA is scale-free and one
# global value applies to every receptor (fa7 sits near -8, 5ht1b near -13).
LAMBDA_SOFT        = 1.0
# Online purpose adapter. R_theta stays FROZEN; a small task-specific term is
# learned at inference time from counted docking observations only.
ADAPT_LR           = 0.05     # advantage-weighted step size
ADAPT_REPLAY       = 200      # evaluated molecules retained for updates
ADAPT_CLIP         = 3.0      # bound on s_psi so the adapter cannot swamp R_theta
# ---- Arm D: faithful InVirtuoGen lead-optimisation settings, transplanted.
# Their published command: max_oracle_calls 1000, tot_offspring 100,
# offspring_size 50, num_reinforce_steps 50, experience_replay_size 100,
# clip_eps 0.5. Only representation-specific generation is replaced -- offspring
# come from executable COMPOSE transitions instead of fragment-flow sampling.
IVG_OFFSPRING      = 100      # LOGICAL cycle size. Docked in chunks of DOCK_CHUNK;
IVG_DOCK_CHUNK     = 20       # tot_offspring=100 is not 100-way Vina concurrency.
IVG_UPDATES        = 50       # policy updates after each completed cycle
IVG_REPLAY         = 100
IVG_CLIP           = 0.5
# ---- Arm E: trajectory-level adaptation.
# The structural audit is the reason these horizons are long. InVirtuoGen's
# winners sit 8-15 legal edits from the benchmark seed: 5ht1b s7 needs +15 heavy
# atoms and +3 rings, parp1 s0 +14 and +3, braf s9 -14, s11 -10, s10 -8. Every
# arm so far searched 1-4 edits out and re-ranked what it found there. An
# offspring must therefore be a MULTI-STEP executable path, and the horizons must
# reach the distances the targets actually sit at.
TRAJ_L             = (1, 2, 4, 8, 16)
TRAJ_PER_CYCLE     = 100      # offspring per cycle, matching the IVG cycle size
TRAJ_UPDATES       = 50
TRAJ_CLIP          = 0.5
PER_ROUND_TRAJ     = 20       # trajectory offspring per round == docking calls
TRAJ_FANOUT        = 8        # containers per cell; 10 cells x 8 = 80 (cap 100)
MACRO_PRIOR        = 1e-3     # R_theta mass placeholder for a sampled endpoint


# cpu=4 so DOCK_WORKERS=4 concurrent QuickVina processes each own a core. The
# previous cpu=1 with "--cpu 4" passed to QuickVina oversubscribed one core 4x and
# measured no benefit (61.1s vs 57.5s serial).
@app.function(image=image, cpu=(2.0, 2.0), memory=int(6 * 1024),
              timeout=60 * 60, max_containers=80,
              volumes={str(ARTIFACT_ROOT): artifact_volume})
def walk_paths(job: dict[str, Any]) -> list[dict[str, Any]]:
    """Sample executable trajectories. FAITHFUL to the in-process version.

    The 20 offspring of a round are independent paths, so they fan out to
    containers instead of running sequentially. Two properties keep this exactly
    equivalent to the serial sampler:

      1. Each offspring gets its OWN seed, derived deterministically from
         (cell, round, offspring index). Without this every container would share
         a seed and walk correlated paths -- fewer distinct offspring dressed up
         as 20.
      2. The sampler is the same top-APPLY_CAP mark pool and the same
         probability-weighted single-mark application as _step, so the transition
         distribution is unchanged. Only WHERE it executes differs.

    Returns the full transition list per path, which is what the endpoint reward
    has to be credited across.
    """

    import os, sys
    import numpy as np
    os.chdir("/tmp"); sys.path.insert(0, str(REMOTE_ROOT / "src"))
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")
    from compose_v4.chem.molecular_graph import smiles_to_molecular_graph
    from compose_v4.chem.state import pad_molecular_graph
    from compose_v4.experiments.production_successor_kernel import (
        canonical_state_key, enumerate_factorized_marked_law)
    rt = _runtime(); model, system = rt["model"], rt["system"]
    LAW: dict[str, Any] = {}

    def step(smi, rng):
        if smi not in LAW:
            try:
                st = pad_molecular_graph(smiles_to_molecular_graph(smi), CANONICAL_SLOTS)
                law = enumerate_factorized_marked_law(model, st, float(TIME_POINT))
                pr = np.array([m.probability for m in law.marks], float)
                ix = np.argsort(-pr)[:APPLY_CAP]
                LAW[smi] = (st, law, ix, pr[ix] / pr[ix].sum())
            except Exception:
                LAW[smi] = None
        e = LAW[smi]
        if e is None:
            return None
        st, law, ix, w = e
        mk = law.marks[int(ix[int(rng.choice(len(ix), p=w))])]
        try:
            y = canonical_state_key(system.apply(st, mk.executor_rule_name, mk.action))
        except Exception:
            return None
        return y if y and y != smi else None

    out = []
    for spec in job["specs"]:
        rng = np.random.default_rng(spec["seed"])
        cur = spec["start"]; steps = []
        for _ in range(int(spec["L"])):
            nxt = step(cur, rng)
            if nxt is None:
                break
            steps.append([cur, nxt]); cur = nxt
        # Return EVERY visited state, not just the terminal one. A path visits L
        # complete, valid molecules; T4 permits returning any molecule we dock,
        # and QED/SA/similarity are free cheminformatics. Docking the terminal
        # state alone threw away L-1 candidates for free: braf produced feas 0 on
        # every round because an R_theta-driven walk has NO feasibility awareness
        # and drifts out of the region without re-entering. The controller owns
        # props(), so it picks which visited state to spend the docking call on.
        out.append({"end": cur if steps else None, "steps": steps,
                    "visited": [s[1] for s in steps],
                    "L": int(spec["L"]), "idx": spec["idx"]})
    return out


@app.function(image=image, cpu=(4.0, 4.0), memory=int(8 * 1024),
              timeout=10 * 60 * 60, max_containers=32,
              volumes={str(ARTIFACT_ROOT): artifact_volume})
def run_cell(task: dict[str, Any]) -> dict[str, Any]:
    import sys, os
    import numpy as np
    sys.path.insert(0, str(REMOTE_ROOT / "src"))
    from rdkit import Chem, DataStructs, RDLogger
    from rdkit.Chem import QED, AllChem, RDConfig
    RDLogger.DisableLog("rdApp.*")
    sys.path.append(os.path.join(RDConfig.RDContribDir, "SA_Score"))
    import sascorer
    from compose_v4.chem.molecular_graph import smiles_to_molecular_graph
    from compose_v4.chem.state import pad_molecular_graph
    from compose_v4.experiments.production_successor_kernel import (
        canonical_state_key, enumerate_factorized_marked_law)

    rt = _runtime(); model, system = rt["model"], rt["system"]
    seed, delta, target = task["smiles"], task["delta"], task["target"]
    control, budget = task["control"], task["budget"]
    do_dock = task.get("dock", True)
    rng = np.random.default_rng(task["seed_rng"])
    gen = AllChem.GetMorganGenerator(radius=2, fpSize=2048)
    seed_fp = gen.GetFingerprint(Chem.MolFromSmiles(seed))
    t0 = time.time()

    # ---------------- transposition table: one entry per canonical molecule ----
    TT: dict[str, dict] = {}

    def props(smi):
        e = TT.get(smi)
        if e is not None and "v" in e:
            return e
        m = Chem.MolFromSmiles(smi)
        if m is None:
            TT[smi] = {"bad": True}; return TT[smi]
        q = float(QED.qed(m)); s = float(sascorer.calculateScore(m))
        sim = float(DataStructs.TanimotoSimilarity(seed_fp, gen.GetFingerprint(m)))
        # the full constraint-shortfall VECTOR, not one scalar
        c = (max(0.0, QED_MIN - q) / QED_MIN,
             max(0.0, s - SA_MAX) / SA_MAX,
             max(0.0, delta - sim) / delta)
        e = {"qed": q, "sa": s, "sim": sim, "c": c, "v": max(c),
             "fp": np.array(gen.GetFingerprint(m), dtype=np.float32)}
        TT[smi] = e
        return e

    def fiber(smi):
        e = TT.setdefault(smi, {})
        if "fiber" in e:
            return e["fiber"]
        try:
            st = pad_molecular_graph(smiles_to_molecular_graph(smi), CANONICAL_SLOTS)
            law = enumerate_factorized_marked_law(model, st, float(TIME_POINT))
        except Exception:
            e["fiber"] = {}; return e["fiber"]
        if not law.marks:
            e["fiber"] = {}; return e["fiber"]
        pr = np.array([m.probability for m in law.marks], float)
        out = {}
        for i in np.argsort(-pr)[:APPLY_CAP]:
            mk = law.marks[int(i)]
            try:
                y = canonical_state_key(system.apply(st, mk.executor_rule_name, mk.action))
            except Exception:
                continue
            if y and y != smi and y not in out:
                out[y] = float(pr[int(i)])
        e["fiber"] = out
        return out

    # ---------------- online docking surrogate: bootstrap ridge, numpy only ---
    class Surrogate:
        """Ranking-oriented. Thompson sampling = draw one bootstrap member."""
        def __init__(self):
            self.W = None
        def fit(self, X, y):
            n, d = X.shape
            if n < 8:
                self.W = None; return
            Ws = []
            for _ in range(ENSEMBLE):
                idx = rng.integers(0, n, n)
                Xb, yb = X[idx], y[idx]
                A = Xb.T @ Xb + RIDGE * np.eye(d, dtype=np.float32)
                Ws.append(np.linalg.solve(A, Xb.T @ yb))
            self.W = np.stack(Ws)
        def sample(self, X):
            if self.W is None:
                return np.zeros(len(X))
            return X @ self.W[rng.integers(0, len(self.W))]
        def mean(self, X):
            if self.W is None:
                return np.zeros(len(X))
            return X @ self.W.mean(0)

    sur = Surrogate()

    # ---------------- online purpose adapter ----------------
    # pi_psi(y|x) ∝ R_theta(y|x) * exp(s_psi(y)),  s_psi = w . fingerprint(y)
    #
    # WHY THIS IS NOT THE FAILED DELTA-D PREDICTOR. That test asked whether an
    # edit's docking EFFECT is supervised-predictable, and the proxy answer was no.
    # This does not predict anything: it is advantage-weighted policy improvement.
    # Molecules that actually received high reward get their features up-weighted,
    # so probability mass moves toward the chemistry that worked. Credit assignment,
    # not regression.
    #
    # R_theta IS NEVER MODIFIED. s_psi starts at zero, so the controller begins
    # exactly at the frozen reference process and adapts only its PURPOSE. That
    # keeps the reusable-process claim intact, unlike per-target generator
    # fine-tuning.
    adapt_w = np.zeros(2048, dtype=np.float64)
    replay: list = []

    def s_psi(fps):
        if not np.any(adapt_w):
            return np.zeros(len(fps))
        return np.clip(np.stack(fps) @ adapt_w, -ADAPT_CLIP, ADAPT_CLIP)

    def traj_sample(x0, L):
        """One offspring: an L-step executable path. Returns (endpoint, [(x,y)...]).

        Every intermediate is a complete valid molecule; QED/SA/similarity are
        checked at the ENDPOINT only, which is what T4 requires. Returning the
        transition list is the point -- endpoint reward is credited to every
        decision along the path, not just the last one.
        """

        cur = x0; steps = []
        for _ in range(L):
            nxt_ = _step(cur)
            if nxt_ is None:
                break
            steps.append((cur, nxt_)); cur = nxt_
        return (cur, steps) if steps else (None, [])

    def traj_update(paths):
        """Advantage-weighted update on SUMMED path log-probability.

        A high-reward endpoint makes the whole sequence of executable decisions
        that reached it more likely, rather than only its final edit. This is the
        COMPOSE analogue of whole-molecule refinement: credit assignment along a
        legal trajectory. R_theta is never modified -- only s_psi moves.
        """

        nonlocal adapt_w
        if not paths:
            return
        r = np.array([p[1] for p in paths], dtype=np.float64)
        A = (r - r.mean()) / (r.std() + 1e-9)
        # one feature vector per path = sum of its transition endpoint features,
        # which is the gradient of sum_l log pi(y_l|x_l) w.r.t. a linear s_psi
        # MEAN, not sum. Summing transition features makes the vector norm scale
        # with path length, so a 16-step offspring gets ~16x the gradient of a
        # 1-step one regardless of its reward -- a length bias masquerading as a
        # learning signal. Averaging keeps horizons comparable, which matters
        # because TRAJ_L spans 1 to 16.
        F = np.stack([np.mean([props(y)["fp"] for _, y in p[0]], axis=0)
                      if p[0] else np.zeros(2048) for p in paths]).astype(np.float64)
        base = F @ adapt_w
        for _ in range(TRAJ_UPDATES):
            ratio = np.exp(np.clip(F @ adapt_w - base, -10, 10))
            lo, hi = 1.0 - TRAJ_CLIP, 1.0 + TRAJ_CLIP
            live = ((ratio > lo) & (ratio < hi)) | ((ratio <= lo) & (A < 0)) \
                   | ((ratio >= hi) & (A > 0))
            g = F.T @ (ratio * A * live) / len(A)
            n = np.linalg.norm(g)
            if n == 0:
                break
            adapt_w += (ADAPT_LR / 5.0) * g / n
            adapt_w = np.clip(adapt_w, -ADAPT_CLIP, ADAPT_CLIP)

    def ivg_reward(y, ds):
        """InVirtuoGen's soft lead reward: S = (-DS/15) * (1 - penalty(QED,SA,SIM)).

        Their form exactly, with penalty built from our already-normalised
        shortfall vector so no new chemistry-specific quantity is introduced.
        Hard constraints still decide what may be RETURNED; this only shapes
        search, which is the point -- it gives graded signal outside the feasible
        region, where braf spends its entire budget.
        """

        c = props(y)["c"]
        penalty = float(min(1.0, max(0.0, max(c))))
        return (-float(ds) / 15.0) * (1.0 - penalty)

    def ivg_update(batch):
        """IVG_UPDATES clipped policy steps on the linear adapter after one cycle."""
        nonlocal adapt_w
        replay.extend(batch)
        del replay[:-IVG_REPLAY]
        if len(replay) < 8:
            return
        F = np.stack([b[0] for b in replay]).astype(np.float64)
        r = np.array([b[1] for b in replay], dtype=np.float64)
        A = (r - r.mean()) / (r.std() + 1e-9)
        w_old = adapt_w.copy()
        base = F @ w_old
        for _ in range(IVG_UPDATES):
            ratio = np.exp(np.clip(F @ adapt_w - base, -10, 10))
            lo, hi = 1.0 - IVG_CLIP, 1.0 + IVG_CLIP
            # PPO: a sample contributes gradient only where it is NOT clipped
            live = ((ratio > lo) & (ratio < hi)) | ((ratio <= lo) & (A < 0)) \
                   | ((ratio >= hi) & (A > 0))
            g = F.T @ (ratio * A * live) / len(A)
            n = np.linalg.norm(g)
            if n == 0:
                break
            adapt_w += (ADAPT_LR / 5.0) * g / n
            adapt_w = np.clip(adapt_w, -ADAPT_CLIP, ADAPT_CLIP)

    def adapt_update(batch):
        """batch: list of (fingerprint, soft_reward) from THIS docking round."""
        nonlocal adapt_w
        replay.extend(batch)
        del replay[:-ADAPT_REPLAY]
        if len(replay) < 8:
            return
        F = np.stack([b[0] for b in replay]).astype(np.float64)
        r = np.array([b[1] for b in replay], dtype=np.float64)
        A = (r - r.mean()) / (r.std() + 1e-9)      # normalised advantage
        g = F.T @ A / len(A)                        # advantage-weighted feature mean
        n = np.linalg.norm(g)
        if n > 0:
            adapt_w += ADAPT_LR * g / n             # unit-norm step: LR is scale-free

    # ---------------- terminal desirability g_t ----------------
    def g_terminal(smi, tilde):
        """Hard C times predicted docking desirability, once we can estimate it.

        Before there is feasible terminal mass, a pure indicator makes every
        rollout zero and the planner is blind, so search (and ONLY search) uses
        a graded feasibility surrogate. Reported candidates are always subject
        to the exact endpoint constraints.
        """
        e = props(smi)
        if e.get("bad"):
            return 0.0
        if e["v"] > 0.0:
            return 0.0 if tilde is not None else float(np.exp(-e["v"] / TAU_V))
        if tilde is None:
            return 1.0
        return float(np.exp(tilde(smi) / TAU_U))

    def h_hat(y, b, tilde):
        """E_{R_theta}[g_t(X_b) | X_0 = y] by bounded cached continuations."""
        if b <= 0:
            return g_terminal(y, tilde)
        key = ("h", y, b, tilde is not None)
        if key in TT:
            return TT[key]
        spent, acc = 0, []
        for _ in range(ROLLOUTS):
            cur = y
            for _d in range(b):
                f = fiber(cur)
                if not f or spent >= H_BUDGET:
                    break
                ks = list(f); w = np.array([f[k] for k in ks]); w /= w.sum()
                cur = ks[int(rng.choice(len(ks), p=w))]
                spent += 1
            acc.append(g_terminal(cur, tilde))
        val = float(np.mean(acc)) if acc else 0.0
        TT[key] = val
        return val

    _LAW: dict[str, Any] = {}
    def _step(smi):
        """Sample ONE successor. Does NOT enumerate the fiber.

        A trajectory step consumes one successor, so calling fiber() -- which
        executes all APPLY_CAP=300 rewrites -- to pick one is ~285x more work than
        the step needs. Measured: 292s of fiber for 96 macro steps, which made the
        compressed macro budget no faster than the uncompressed one. The budget was
        never the bottleneck; the per-step implementation was.

        The mark pool is the same top-APPLY_CAP by model probability that fiber()
        uses. The difference is that fiber() renormalises AFTER canonical dedup and
        this samples before it, which is recorded here rather than absorbed silently.
        """

        if smi not in _LAW:
            try:
                st = pad_molecular_graph(smiles_to_molecular_graph(smi), CANONICAL_SLOTS)
                law = enumerate_factorized_marked_law(model, st, float(TIME_POINT))
                pr_ = np.array([m.probability for m in law.marks], float)
                ix = np.argsort(-pr_)[:APPLY_CAP]
                _LAW[smi] = (st, law, ix, pr_[ix] / pr_[ix].sum())
            except Exception:
                _LAW[smi] = None
        e = _LAW[smi]
        if e is None:
            return None
        st, law, ix, w = e
        mk = law.marks[int(ix[int(rng.choice(len(ix), p=w))])]
        try:
            y = canonical_state_key(system.apply(st, mk.executor_rule_name, mk.action))
        except Exception:
            return None
        return y if y and y != smi else None

    def _macro(x, L):
        """L sampled R_theta edits. Intermediates need only be EXECUTABLE."""
        cur = x
        for _ in range(L):
            nxt_ = _step(cur)
            if nxt_ is None:
                return None
            cur = nxt_
        return cur if cur != x else None

    # ---------------- archives, deliberately separate ----------------
    A_search: list[dict] = [{"smi": seed, **{k: v for k, v in props(seed).items() if k != "fp"}}]
    A_oracle: list[dict] = []
    docked: dict[str, float] = {}
    Xs, ys = [], []
    log = []

    def refresh_surrogate():
        if len(ys) >= 8:
            sur.fit(np.stack(Xs), np.array(ys, dtype=np.float32))

    def tilde_fn():
        if sur.W is None or len(ys) < WARMUP:
            return None
        Wm = sur.W[rng.integers(0, len(sur.W))]
        return lambda s: float(props(s)["fp"] @ Wm)

    def _feasible_parents(pool, n_slots):
        """Parents chosen by OBSERVED DOCKING SCORE among feasible molecules.

        THE DEFECT THIS FIXES. keep_lineages selects on the constraint-shortfall
        vector c, and every feasible molecule has c = [0,0,0]. So once feasible, a
        -12.0 molecule and a -8.5 molecule are IDENTICAL to the parent selector,
        and continuation is decided by Tanimoto diversity alone. The measured
        consequence: parp1 seed0 docked 175 feasible molecules and finished at
        -8.5 against GenMol's -10.6, and 5ht1b seed7 at delta=0.6 went
        43,40,27,9,0,0,0,... feasible candidates per round because nothing
        preferentially retained feasible lineages at all.

        The benchmark is min D(x) SUBJECT TO C(x), not find-any-x-with-C(x). Once
        v = 0 the shortfall vector has done its job and the objective takes over.

        RANK-BASED, NOT SOFTMAX ON D. Docking has ~0.7 kcal/mol replicate spread
        and different receptors sit at different absolute scales (fa7 near -8,
        5ht1b near -13), so a temperature on raw magnitude would mean something
        different per target. Rank is scale-free.

        INVARIANT: if any feasible molecule has ever been docked, the best one is
        always a parent. Feasible lineages cannot silently vanish again.
        """

        scored = [(p, docked[p["smi"]]) for p in pool
                  if p["smi"] in docked and docked[p["smi"]] != 0.0]
        if not scored or n_slots <= 0:
            return []
        scored.sort(key=lambda z: z[1])            # more negative = better
        out = [scored[0][0]]                       # the elite, unconditionally
        shortlist = [p for p, _ in scored[1:max(n_slots * 4, 8)]]
        while len(out) < n_slots and shortlist:    # then spread structurally
            best, bd = None, -1.0
            for q in shortlist:
                if q in out:
                    continue
                d = min(1.0 - DataStructs.TanimotoSimilarity(
                    gen.GetFingerprint(Chem.MolFromSmiles(q["smi"])),
                    gen.GetFingerprint(Chem.MolFromSmiles(z["smi"]))) for z in out)
                if d > bd:
                    best, bd = q, d
            if best is None:
                break
            out.append(best); shortlist.remove(best)
        return out

    def keep_lineages(pool):
        """Multi-lineage frontier: Pareto on the shortfall vector, then diverse.

        Level 1 kept the top-3 by a single scalar and collapsed. Here a molecule
        that is worse overall but better on ANY component survives, which is
        what preserves the lineage the route replay showed being discarded.

        Under the dockelite arm this handles the INFEASIBLE regime only; feasible
        parents come from _feasible_parents, ranked by the actual objective.
        """
        if control == "dockelite":
            n_f = N_LINEAGE // 2
            F = _feasible_parents(pool, n_f)
            infeas = [p for p in pool if props(p["smi"])["v"] > 0.0]
            rest = _shortfall_parents(infeas or pool, N_LINEAGE - len(F))
            seen_, merged = set(), []
            for p in F + rest:
                if p["smi"] not in seen_:
                    seen_.add(p["smi"]); merged.append(p)
            return merged[:N_LINEAGE] if merged else _shortfall_parents(pool, N_LINEAGE)
        if control == "dual_macro":
            # ADDITIVE, not a split. dockelite took half of the eight slots for
            # docking elites and fa7 regressed 1.3 kcal/mol with its feasible
            # population collapsing 87 -> 17: exploration capacity was the thing
            # being spent. Here the baseline eight exploration parents are kept
            # EXACTLY as they were, and up to eight docking elites are added
            # alongside, so objective exploitation costs the constraint search
            # nothing. Parents are free; only docking calls are budgeted, and the
            # per-round docking budget is unchanged.
            base_p = _shortfall_parents(pool, N_LINEAGE)
            seen_ = {p["smi"] for p in base_p}
            for p in _feasible_parents(pool, N_LINEAGE):
                if p["smi"] not in seen_:
                    seen_.add(p["smi"]); base_p.append(p)
            return base_p
        return _shortfall_parents(pool, N_LINEAGE)

    def _shortfall_parents(pool, n_slots):
        if not pool or n_slots <= 0:
            return []
        pts = [(p, props(p["smi"])["c"]) for p in pool]
        front = []
        for p, c in pts:
            if not any(all(o <= x for o, x in zip(c2, c)) and o_better(c2, c)
                       for _, c2 in pts):
                front.append(p)
        if len(front) < n_slots:
            rest = sorted((p for p in pool if p not in front),
                          key=lambda p: props(p["smi"])["v"])
            front += rest[:n_slots - len(front)]
        if len(front) <= n_slots:
            return front
        chosen = [min(front, key=lambda p: props(p["smi"])["v"])]
        while len(chosen) < n_slots:
            best, bd = None, -1.0
            for p in front:
                if p in chosen:
                    continue
                d = min(1.0 - DataStructs.TanimotoSimilarity(
                    gen.GetFingerprint(Chem.MolFromSmiles(p["smi"])),
                    gen.GetFingerprint(Chem.MolFromSmiles(q["smi"]))) for q in chosen)
                if d > bd:
                    best, bd = p, d
            chosen.append(best)
        return chosen

    def o_better(a, b):
        return any(x < y for x, y in zip(a, b))

    # ---------------- resume from a preempted attempt ----------------
    _prev = Path("/artifacts/t4_controller") / \
        f"{task['tag']}_{task['idx']}_{task['target']}_d{task['delta']}_{control}.json"
    _resumed_rounds = 0
    try:
        artifact_volume.reload()
        if _prev.exists():
            _p = json.loads(_prev.read_text())
            # Resume a COMPLETE cell too when the budget has been raised: the
            # 200-call table is continued to 500 and 1000 on the same states, so
            # "complete at 200" must not block "continue to 500". The guard is
            # whether calls remain, not whether the previous stage finished.
            _spent = len((_p.get("resume") or {}).get("docked", {}))
            if _p.get("resume") and _spent < budget:
                r_ = _p["resume"]
                docked.update({k: float(v) for k, v in r_.get("docked", {}).items()})
                A_oracle.extend(r_.get("A_oracle", []))
                for smi in r_.get("A_search_smi", []):
                    pr_ = props(smi)
                    if not pr_.get("bad"):
                        A_search.append({"smi": smi,
                                         **{k: v for k, v in pr_.items() if k != "fp"}})
                for smi, ds in docked.items():
                    pr_ = props(smi)
                    if not pr_.get("bad") and ds != 0.0:
                        Xs.append(pr_["fp"]); ys.append(-float(ds))
                if r_.get("adapt_w"):
                    adapt_w[:] = np.array(r_["adapt_w"], dtype=np.float64)
                for fp_, rr_ in (r_.get("replay") or []):
                    replay.append((np.array(fp_, dtype=np.float64), float(rr_)))
                log.extend(_p.get("per_round", []))
                _resumed_rounds = len(log)
                refresh_surrogate()
                print(f"  RESUMED {task['target']} seed{task['idx']}: "
                      f"{len(docked)} docked, {len(A_oracle)} feasible, "
                      f"{_resumed_rounds} rounds already done", flush=True)
    except Exception as e:
        print(f"  !! resume failed (starting fresh): {type(e).__name__}: {e}", flush=True)

    # ---------------- main loop ----------------
    # PER-STAGE TIMERS AND PER-ROUND PERSIST. The previous run produced nothing
    # in 2h45m across three cells because a cell only wrote at the very end and
    # printed nothing while it worked, so there was no way to tell a slow round
    # from a hung one. Both of those are now observable from the outside.
    T = {"fiber": 0.0, "score": 0.0, "archive": 0.0, "dock": 0.0}

    def _build_out(rd, n_calls, log):
        best = min(A_oracle, key=lambda a: a["ds"]) if A_oracle else None
        return {**{k: v for k, v in task.items() if k != "smiles"},
                "rounds": rd, "n_calls": n_calls,
                "n_feasible_docked": len(A_oracle),
                "first_feasible_call": A_oracle[0]["call"] if A_oracle else None,
                "best_ds": best["ds"] if best else None,
                "best_smiles": best["smi"] if best else None,
                "reached_feasible": any(l["n_feasible"] > 0 for l in log),
                "first_feasible_round": next(
                    (l["round"] for l in log if l["n_feasible"] > 0), None),
                "per_round": log, "timers": {k: round(v, 2) for k, v in T.items()},
                "horizon_stats": {str(L): v for L, v in horizon_stats.items()},
                "best_ds_found_by_horizon": best_ds_seen[1],
                "complete": False, "seconds": round(time.time() - t0, 1),
                # RESUME STATE. Modal preempts these containers every ~8-10 min and
                # restarts them from scratch with the same input. At 50 rounds for a
                # 1,000-call official cell, a from-scratch restart means the cell can
                # NEVER finish. Everything needed to continue is therefore written
                # every round: the docking ledger is the expensive part and is exactly
                # what must not be recomputed.
                # adapt_w and replay MUST persist. Without them a preemption
                # silently resets the learned policy to zero and the arm reports a
                # null result that looks scientific rather than infrastructural --
                # exactly the false negative the whole D experiment must avoid.
                "resume": {"docked": docked,
                           "A_oracle": A_oracle,
                           "A_search_smi": [a["smi"] for a in A_search[-400:]],
                           "adapt_w": adapt_w.tolist() if np.any(adapt_w) else None,
                           "replay": [[b[0].tolist(), float(b[1])] for b in replay[-100:]]}}

    def _persist(o):
        try:
            d = Path("/artifacts/t4_controller"); d.mkdir(parents=True, exist_ok=True)
            (d / f"{task['tag']}_{task['idx']}_{task['target']}_"
                 f"d{task['delta']}_{control}.json").write_text(
                json.dumps(o, indent=1))
            artifact_volume.commit()
        except Exception as e:
            print(f"  !! persist failed: {e}", flush=True)

    parent_of: dict[str, str] = {}
    traj_by_end: dict[str, tuple] = {}
    horizon_stats = {int(L): {"n": 0, "r": 0.0} for L in TRAJ_L}
    best_ds_seen = [None, None]        # (best docking score, horizon that produced it)
    n_calls, rd = len(docked), _resumed_rounds
    while n_calls < budget or (not do_dock and rd < task.get("rounds", 10)):
        rd += 1
        parents = keep_lineages(A_search)
        _t = time.time()
        cand: dict[str, float] = {}
        # For the traj arm the offspring ARE the trajectories, so the one-step
        # fiber must NOT also run. It was filling cand with ~1,700 local
        # candidates against ~40 trajectory endpoints, so selection drew 20 from
        # 1,740 and trajectory endpoints won ~2% of the picks: measured
        # horizon_visits {1:0, 2:0, 4:1, 8:0, 16:0} over 60 calls. The arm was
        # running as the baseline. L=1 already covers the single-edit case.
        if control != "traj":
            for p in parents:
                for y, r in fiber(p["smi"]).items():
                    if y in docked or y in cand:
                        continue
                    if not props(y).get("bad"):
                        cand[y] = r
                        parent_of.setdefault(y, p["smi"])
        if control == "traj":
            # An offspring is a PATH, not an edit. Up to 16 executable decisions
            # cost ONE docking call, which is the oracle efficiency this benchmark
            # rewards. The transition list is kept per endpoint so the endpoint
            # reward can be credited to every decision that produced it -- the
            # unrewarded intermediate steps are exactly what the audit says must
            # be learned (growth stalls at 0.55 similarity, deletion reaches 0.87).
            traj_by_end.clear()
            hs_ = [int(l) for l in TRAJ_L]
            n_off = PER_ROUND_TRAJ * 2                 # oversample; dedup after
            specs = []
            for j in range(n_off):
                p_ = parents[int(rng.integers(0, len(parents)))]
                specs.append({"start": p_["smi"],
                              "L": hs_[j % len(hs_)],  # even coverage of horizons
                              "idx": j,
                              # deterministic per-offspring seed: distinct paths
                              "seed": int(task["seed_rng"]) * 1000003 + rd * 1009 + j})
            chunks = [{"specs": specs[i::TRAJ_FANOUT]} for i in range(TRAJ_FANOUT)]
            got = []
            n_err = 0
            for res in walk_paths.map(chunks, order_outputs=False,
                                      return_exceptions=True,
                                      wrap_returned_exceptions=False):
                if not isinstance(res, list):
                    n_err += 1
                    if n_err <= 2:
                        print(f"  !! walk_paths failed: {type(res).__name__}: "
                              f"{str(res)[:200]}", flush=True)
                    continue
                got.extend(res)
            if n_err:
                print(f"  !! {n_err}/{len(chunks)} walk_paths chunks failed this round",
                      flush=True)
            _eff = []
            for g in got:
                # Spend the one docking call on the BEST state the path reached,
                # not the last. Ties on v (all-feasible paths) break toward the
                # SHORTEST prefix, so a feasible state found at step 3 is not
                # charged the drift of steps 4..16.
                vis = g.get("visited") or ([g["end"]] if g["end"] else [])
                bi, bv, end = -1, None, None
                for i_, s_ in enumerate(vis):
                    if not s_ or s_ in docked or s_ in cand:
                        continue
                    pr_ = props(s_)
                    if pr_.get("bad"):
                        continue
                    v_ = float(pr_["v"])
                    if bv is None or v_ < bv - 1e-12:
                        bi, bv, end = i_, v_, s_
                if end is not None:
                    cand[end] = float(MACRO_PRIOR)
                    traj_by_end[end] = ([tuple(s) for s in g["steps"][:bi + 1]],
                                        bi + 1)
                    _eff.append((int(g["L"]), bi + 1, bv))
                    if g["steps"]:
                        parent_of.setdefault(end, g["steps"][0][0])
            if _eff:
                _fe = [e for e in _eff if e[2] == 0.0]
                print(f"    traj: {len(_eff)} paths  L_eff/L_sampled "
                      f"{np.mean([e[1] for e in _eff]):.1f}/"
                      f"{np.mean([e[0] for e in _eff]):.1f}  "
                      f"min_v {min(e[2] for e in _eff):.4f}  "
                      f"feasible_states {len(_fe)}  "
                      f"(endpoint-only would give "
                      f"{sum(1 for e in _eff if e[1] == e[0] and e[2] == 0.0)})",
                      flush=True)
                    # NOTE: no per-molecule depth map in this controller (that is
                    # the PMO one). Path length is carried in traj_by_end[end][1].
        if control == "dual_macro":
            # NONLOCAL EXECUTABLE ENDPOINTS. Every candidate above is ONE edit
            # away, and two failures are exactly that locality: 5ht1b d=0.6 ran
            # its feasible neighbourhood to zero (43,40,27,9,0,0,...), and braf
            # d=0.6 needs QED and similarity to trade against each other, which no
            # single edit can do. T4 applies QED/SA/similarity to the RETURNED
            # molecule only, so intermediates are free to violate them: a
            # trajectory may leave the feasible region and re-enter somewhere
            # better. Only x_L is ever tested. This adds NO oracle calls -- the
            # existing acquisition still picks the same 20 molecules to dock.
            # Budget split evenly across horizons; parents drawn round-robin across
            # BOTH archives so neither exploration nor docking exploitation
            # monopolises the macro allocation.
            per_h = max(1, MACRO_BUDGET // len(MACRO_L))
            for L in MACRO_L:
                for j in range(per_h):
                    p = parents[j % len(parents)]
                    y = _macro(p["smi"], int(L))
                    if y and y not in docked and y not in cand and not props(y).get("bad"):
                        cand[y] = float(MACRO_PRIOR)
        T["fiber"] += time.time() - _t
        if not cand:
            break
        keys = list(cand)
        tl = tilde_fn()

        # ---- CONTROL: the one-flag ablation ----
        _t = time.time()
        if control == "future_h":
            b = task.get("horizon", 4)
            score = np.array([cand[y] * h_hat(y, b, tl) for y in keys])
        elif control in ("adapt", "ivg"):
            # R_theta mass reweighted by the learned purpose term. g_terminal is
            # retained so feasibility guidance is not lost before the adapter has
            # seen any reward.
            sp = s_psi([props(y)["fp"] for y in keys])
            score = np.array([cand[y] * g_terminal(y, tl) for y in keys]) * np.exp(sp)
        else:                                    # immediate
            score = np.array([cand[y] * g_terminal(y, tl) for y in keys])
        score = np.clip(score, 1e-30, None)
        T["score"] += time.time() - _t

        # ---- search archive: keep promising states, feasible or not ----
        _t = time.time()
        top = list(np.argsort(-score)[:N_LINEAGE * 4])

        # ELITISM ON THE NAVIGATION SIGNAL. The score above is
        # R_theta(y|x) * g_terminal(y), so a candidate that is nearly FEASIBLE but
        # that R_theta considers an unlikely edit falls outside the top-32 and is
        # never stored. That is not hypothetical: fa7 reached min_v 0.0207 at round
        # 6 and was back at 0.0614 by round 7, and parp1 went 0.0432 -> 0.0517.
        # min_v RISING between rounds is the signature of that loss.
        #
        # ELITISM IS PER-COMPONENT, NOT ON THE SCALAR max. v(x) is a MAX over
        # (QED, SA, similarity) shortfalls, so a molecule that trades similarity
        # for QED does not look better -- the max simply switches to the other
        # term. Admitting only the best scalar-v candidates therefore reinforces
        # whichever constraint is currently binding and never builds on the
        # molecule that relieved a DIFFERENT one. That is a general defect of a
        # max-scalar over competing constraints, not a property of any target:
        # wherever two constraints pull opposite ways, the search gets no gradient
        # along the only direction that can satisfy both. Keeping the best
        # candidate on EACH component preserves that lineage.
        #
        # This is the same reasoning the archive's Pareto-on-shortfall-vector rule
        # already encodes; the scalar admission gate was quietly undoing it.
        v_rank = list(np.argsort([props(y)["v"] for y in keys])[:N_LINEAGE])
        _C = np.array([props(y)["c"] for y in keys], dtype=float)   # (n, 3)
        for j in range(_C.shape[1]):
            for i in np.argsort(_C[:, j])[:max(2, N_LINEAGE // 2)]:
                if int(i) not in v_rank:
                    v_rank.append(int(i))
        for i in v_rank:
            if int(i) not in top:
                top.append(int(i))
        for i in top:
            A_search.append({"smi": keys[int(i)],
                             **{k: v for k, v in props(keys[int(i)]).items() if k != "fp"}})
        T["archive"] += time.time() - _t

        # ---- oracle archive: FEASIBLE FIRST, then acquisition ----
        feas = [y for y in keys if props(y)["v"] == 0.0]
        _t = time.time()
        if do_dock:
            per = min(IVG_OFFSPRING if control == "ivg" else task.get("per_round", 20),
                      budget - n_calls)
            if feas:
                fX = np.stack([props(y)["fp"] for y in feas])
                acq = sur.sample(fX) if sur.W is not None else rng.random(len(feas))
                pick = [feas[int(i)] for i in np.argsort(-acq)[:per]]
            else:
                pick = []
            if len(pick) < per:                  # fill from the soft frontier
                rest = [y for y in keys if y not in pick]
                if control in ("softreward", "adapt", "ivg") and rest:
                    # SOFT REWARD FOR INFEASIBLE STATES.
                    #
                    # The default fill ranks by score = R_theta(y) * exp(-v/TAU_V):
                    # model probability times constraint shortfall. DOCKING NEVER
                    # ENTERS. braf is the extreme case -- it has zero feasible
                    # candidates every round, so all 20 calls per round already go
                    # here, 200 counted docking scores per cell, and every one of
                    # them is used only to fit a surrogate that has no influence on
                    # which infeasible molecule is chosen next. The cell has never
                    # crossed in any arm.
                    #
                    # Here observed docking and graded shortfall drive the choice
                    # jointly, which is what lets a lineage trade similarity for QED
                    # and still be recognised as promising. Both terms are
                    # standardised within the round, so no receptor-specific scale
                    # or temperature is implied.
                    rX = np.stack([props(y)["fp"] for y in rest])
                    u = sur.mean(rX) if sur.W is not None else np.zeros(len(rest))
                    vv = np.array([props(y)["v"] for y in rest], float)
                    z = lambda a: (a - a.mean()) / (a.std() + 1e-9)
                    soft = (z(u) if np.std(u) > 0 else np.zeros(len(rest))) \
                           - LAMBDA_SOFT * z(vv)
                    pick += [rest[int(i)] for i in np.argsort(-soft)[:per - len(pick)]]
                else:
                    w = score[[keys.index(y) for y in rest]]; w = w / w.sum()
                    extra = rng.choice(len(rest), size=min(per - len(pick), len(rest)),
                                       replace=False, p=w)
                    pick += [rest[int(i)] for i in extra]
            # BATCHED. The round's dockings are conditionally independent -- the
            # archive updates only after all of them -- so they run concurrently.
            # _dock_many preserves order and returns exactly one entry per input,
            # so the accounting below is byte-for-byte what the serial loop did.
            if control == "ivg":
                # A 100-offspring CYCLE, docked in safe chunks. Rewards accumulate
                # across the whole cycle and the policy updates once, afterwards.
                _scores = []
                for _c0 in range(0, len(pick), IVG_DOCK_CHUNK):
                    _scores += _dock_many(pick[_c0:_c0 + IVG_DOCK_CHUNK], target,
                                          f"h{task['idx']}_{rd}_{_c0}",
                                          workers=DOCK_WORKERS, cpu_per_dock=1)
            else:
                _scores = _dock_many(pick, target, f"h{task['idx']}_{rd}",
                                     workers=DOCK_WORKERS, cpu_per_dock=1)
            assert len(_scores) == len(pick), "dock_many must return one result per molecule"
            for y, ds in zip(pick, _scores):
                n_calls += 1
                docked[y] = ds if ds is not None else 0.0
                if ds is not None:
                    Xs.append(props(y)["fp"]); ys.append(-ds)   # larger is better
                    if props(y)["v"] == 0.0:
                        # PARENT RECORDED. Edit-level analysis -- "does this
                        # particular rewrite from this particular molecule improve
                        # docking?" -- needs (x -> y) pairs, and absolute docking
                        # has already been measured as barely rankable (Spearman
                        # ~0.2). Storing only the child made that question
                        # unanswerable from banked runs.
                        A_oracle.append({"smi": y, "ds": ds, "round": rd,
                                         "call": n_calls,
                                         "parent": parent_of.get(y)})
            refresh_surrogate()
            if control == "traj":
                paths = []
                for y in pick:
                    ds_ = docked.get(y)
                    if ds_ in (None, 0.0) or y not in traj_by_end:
                        continue
                    steps, L = traj_by_end[y]
                    r_ = ivg_reward(y, ds_)
                    paths.append((steps, r_))
                    horizon_stats[L]["n"] += 1
                    horizon_stats[L]["r"] += r_
                    if best_ds_seen[0] is None or ds_ < best_ds_seen[0]:
                        best_ds_seen[0] = ds_; best_ds_seen[1] = L
                if paths:
                    traj_update(paths)
            if control == "ivg":
                bb = [(props(y)["fp"].astype(np.float64), ivg_reward(y, docked[y]))
                      for y in pick if docked.get(y) not in (None, 0.0)]
                if bb:
                    ivg_update(bb)
            if control == "adapt":
                # soft reward on every COUNTED molecule, feasible or not: this is
                # the signal braf never had. u = -ds (larger better), minus the
                # graded constraint shortfall.
                bb = []
                for y in pick:
                    ds_ = docked.get(y)
                    if ds_ is None or ds_ == 0.0:
                        continue
                    bb.append((props(y)["fp"].astype(np.float64),
                               (-float(ds_)) - LAMBDA_SOFT * 10.0 * props(y)["v"]))
                if bb:
                    adapt_update(bb)
        T["dock"] += time.time() - _t

        log.append({"round": rd, "n_cand": len(keys), "n_feasible": len(feas),
                    "n_docked_total": n_calls, "n_lineages": len(parents),
                    "min_v": round(min(props(y)["v"] for y in keys), 4),
                    "surrogate_n": len(ys),
                    "horizon_visits": {str(L): v["n"] for L, v in horizon_stats.items()},
                    "horizon_mean_reward": {str(L): (round(v["r"]/v["n"], 4) if v["n"] else None)
                                            for L, v in horizon_stats.items()},
                    "best_ds_horizon": best_ds_seen[1],
                    "t_fiber": round(T["fiber"], 2), "t_score": round(T["score"], 2),
                    "t_archive": round(T["archive"], 2), "t_dock": round(T["dock"], 2),
                    "elapsed": round(time.time() - t0, 1)})
        print(f"  [{task['target']}/{control}] rd {rd:>2}  cand {len(keys):>5}  "
              f"feas {len(feas):>4}  calls {n_calls:>4}  |  fiber {T['fiber']:>7.1f}s  "
              f"score {T['score']:>7.1f}s  arch {T['archive']:>5.1f}s  dock {T['dock']:>7.1f}s  "
              f"|  {time.time() - t0:>7.1f}s total", flush=True)
        _persist(_build_out(rd, n_calls, log))
        if rd >= task.get("max_rounds", 10**9):
            break
        if not do_dock and rd >= task.get("rounds", 10):
            break

    out = _build_out(rd, n_calls, log)
    out["complete"] = True
    _persist(out)
    return out


@app.function(image=image, cpu=(1.0, 1.0), memory=2048, timeout=12 * 60 * 60,
              volumes={str(ARTIFACT_ROOT): artifact_volume})
def drive(cells: str, control: str, budget: int, dock: bool, tag: str,
          horizon: int, rounds: int, max_rounds: int = 10**9,
          seedset: str = "dev", deltas: str = "0.4") -> dict[str, Any]:
    """seedset selects DEV (policy development) or OFFICIAL (the 15 published seeds).

    Frozen decision 4 of the T4 amendment keeps these disjoint: the policy is
    developed on dev seeds and the published cells are run once under it. The
    official file is a bare list; the dev file wraps its list in {"seeds": ...}.
    """

    fn = ("docs/GENMOL_T4_SEEDS.json" if seedset.lower().startswith("off")
          else "docs/GENMOL_T4_DEV_SEEDS.json")
    raw = json.loads((REMOTE_ROOT / fn).read_text())
    seeds = raw["seeds"] if isinstance(raw, dict) else raw
    DL = tuple(float(x) for x in deltas.split(",") if x.strip())
    print(f"  seedset={seedset} ({fn}, {len(seeds)} seeds)  deltas={DL}", flush=True)
    idxs = [int(x) for x in cells.split(",") if x != ""]
    arms = [c.strip() for c in control.split(",")]
    tasks = []
    for i in idxs:
        for arm in arms:
            for dl in DL:
                s_ = seeds[i]
                # The two seed files carry different field names: dev uses
                # chembl/qed/sa, the published file uses published_ds/seed_qed/
                # seed_sa and has no chembl id. Read both rather than assuming one.
                tasks.append({"idx": i, "target": s_["target"],
                              "chembl": s_.get("chembl") or f"seed{i}",
                              "smiles": s_["smiles"],
                              "qed": s_.get("qed", s_.get("seed_qed")),
                              "sa": s_.get("sa", s_.get("seed_sa")),
                              "published_ds": s_.get("published_ds"),
                              "delta": dl,
                              "control": arm, "budget": budget, "dock": dock,
                              "tag": tag, "horizon": horizon, "rounds": rounds,
                              "max_rounds": max_rounds,
                              "per_round": 20, "seed_rng": 20260821 + i})
    print(f"{len(tasks)} runs  arms={arms}  budget={budget}  dock={dock}\n", flush=True)
    out = []
    for r in run_cell.map(tasks, order_outputs=False, return_exceptions=True,
                          wrap_returned_exceptions=False):
        if not isinstance(r, dict):
            print(f"  !! {type(r).__name__}: {str(r)[:140]}", flush=True); continue
        out.append(r)
        print(f"  {r['target']:6s} {r['control']:9s} feas@rd {str(r['first_feasible_round']):>4}  "
              f"feas_docked {r['n_feasible_docked']:>3}  calls {r['n_calls']:>4}  "
              f"bestDS {r['best_ds'] if r['best_ds'] is not None else 'NONE'}", flush=True)
    if not out:
        d = Path("/artifacts/t4_controller")
        if d.exists():
            out = [json.loads(f.read_text()) for f in sorted(d.glob(f"{tag}_*.json"))]
            print(f"  recovered {len(out)} from volume", flush=True)
    if not out:
        raise RuntimeError("no runs completed")
    return {"runs": out, "tag": tag}


@app.function(image=image, cpu=(1.0, 1.0), memory=1024, timeout=600,
              volumes={str(ARTIFACT_ROOT): artifact_volume})
def salvage(tag: str = "") -> dict[str, Any]:
    d = Path("/artifacts/t4_controller")
    g = f"{tag}_*.json" if tag else "*.json"
    runs = [json.loads(f.read_text()) for f in sorted(d.glob(g))] if d.exists() else []
    print(f"  salvaged {len(runs)}", flush=True)
    return {"runs": runs, "salvaged": True}


@app.local_entrypoint()
def main(cells: str = "6,18,2", control: str = "immediate,future_h",
         budget: int = 200, dock: bool = True, tag: str = "h2",
         horizon: int = 4, rounds: int = 10, max_rounds: int = 10**9,
         seedset: str = "dev", deltas: str = "0.4", out: str = "") -> None:
    try:
        o = drive.remote(cells, control, budget, dock, tag, horizon, rounds,
                         max_rounds, seedset, deltas)
    except Exception as e:
        print(f"  driver failed ({type(e).__name__}); salvaging")
        o = salvage.remote(tag)
    p = Path(__file__).resolve().parents[1] / (out or f"diagnostics/genmol_t4_controller_{tag}.json")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(o, indent=1))
    print(f"\nwrote {p}")
