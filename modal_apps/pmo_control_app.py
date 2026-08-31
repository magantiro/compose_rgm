"""Step C: oracle-prescreened COMPOSE with an ONLINE-ADAPTIVE task surrogate.

THE QUESTION. Our matched lane scored 249,455 molecules and used the labels to
pick 100 starting states, discarding 249,355 of them. GenMol turns the same
labels into a task-specific fragment vocabulary it consults throughout
generation. So the lane was never actually matched -- it was strictly weaker.
This asks whether COMPOSE can exploit the SAME labels as dense guidance
throughout a multi-lineage search.

WHY ONLINE ADAPTATION IS NOT OPTIONAL. Every label in the prescreen corpus is
<= 0.5261 on scaffold_hop, and GenMol reports 0.936. The moment search leaves the
corpus -- which the oracle-greedy diagnostic showed it does, reaching 0.6285 --
the surrogate is extrapolating with no supervision. The counted oracle calls are
the ONLY signal available in that region, so they must feed back into the model.
A frozen prior is kept as an arm precisely so this claim is tested, not assumed.

The few online labels must not be washed out by 249k offline samples, so the
prior is held fixed and a residual is fitted only to counted observations:

    f_t(x) = prior(x) + r_t(x)

with r_t a bootstrap ridge ensemble over Morgan features. That also supplies the
uncertainty the acquisition arms need -- extrapolation is the problem here, so
pure surrogate greed is the wrong default and is tested rather than assumed.

ARMS (one flag each, everything else identical):
    no_surrogate     R_theta only. The matched-init baseline, re-run at this budget.
    frozen_greedy    prior only, never updated. Isolates the value of adaptation.
    online_greedy    prior + online residual, rank by the mean.
    online_thompson  prior + online residual, rank by one sampled ensemble member.
    online_ucb       prior + online residual, rank by mean + kappa * sd.

REPORTING. Every run records the official counted-only top_auc(finish=True),
normalised by 10,000, because that is the only number comparable to GenMol's.
Primed molecules never enter that ledger, exactly as GenMol's initial population
never enters mol_buffer.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

import modal

from modal_apps.genmol_t4_opt_app import (
    APPLY_CAP, CANONICAL_SLOTS, TIME_POINT,
    ARTIFACT_ROOT, REMOTE_ROOT, ROOT, artifact_volume, _runtime,
)
from modal_apps.genmol_t4_opt_app import image as _opt_image

image = (
    _opt_image
    .pip_install("PyTDC==0.3.6", "requests", "fuzzywuzzy", "seaborn", "networkx")
    .add_local_file(ROOT / "modal_apps/genmol_t4_opt_app.py",
                    str(REMOTE_ROOT / "modal_apps/genmol_t4_opt_app.py"), copy=True)
    .add_local_file(ROOT / "artifacts/oracles/molleo_task3_v1/jnk3_forest.npz",
                    "/frozen/jnk3_forest.npz", copy=True)
    .add_local_file(ROOT / "artifacts/oracles/molleo_task3_v1/gsk3b_forest.npz",
                    "/frozen/gsk3b_forest.npz", copy=True)
)

app = modal.App("pmo-control")

ARMS = ["no_surrogate", "frozen_greedy", "online_greedy", "online_thompson", "online_ucb",
        "L1", "mixedL", "puct", "bo"]
# Macro-proposal horizons. One counted oracle call buys a coherent L-edit move;
# intermediates are NEVER evaluated. L1 is the same optimizer restricted to L=1,
# so the pair is a one-flag ablation of trajectory length alone.
HORIZONS = (1, 2, 4, 8)

# ---- PUCT. ONE generic setting for every task; never tuned per task. Tuning 23
# ---- algorithms and reporting the max is not a reusable controller.
C_PUCT   = 1.5     # exploration weight on the R_theta prior
PW_K     = 4.0     # progressive widening: children = ceil(PW_K * N**PW_ALPHA)
PW_ALPHA = 0.5
C_HORIZON = 1.0    # UCB constant for the online horizon bandit

# ---- COMPOSE-BO. R_theta generates executable possibilities; BO decides which
# ---- one deserves the scarce oracle call. One global setting for every task.
BO_LIB      = 40    # uncharged library per round. LIB/BATCH = 2 walks per counted
                    # call, the SAME generation cost as mixedL, so the arms differ in
                    # how candidates are CHOSEN rather than in how many are generated.
BO_BATCH    = 20    # oracle calls spent per round, chosen by expected improvement
BO_AMP      = 1.0   # Tanimoto kernel amplitude
BO_NOISE    = 1e-3  # observation noise
BO_XI       = 0.01  # EI exploration offset
CHECKPOINTS = (25, 50, 100, 250, 500, 1000, 2000)
N_LINEAGE, PER_ROUND = 12, 20
ENSEMBLE, RIDGE, KAPPA = 8, 1.0, 1.0
# Corpus maximum is READ FROM EACH TASK'S BANK, never hardcoded. A per-task
# constant table silently produces a wrong threshold the moment a task is added,
# and this metric only means anything relative to that task's own prescreen.
CORPUS_MAX: dict[str, float] = {}


def _shim():
    import sys, types
    six = types.ModuleType("rdkit.six")
    six.iteritems = lambda d, **k: iter(d.items())
    six.itervalues = lambda d, **k: iter(d.values())
    six.iterkeys = lambda d, **k: iter(d.keys())
    six.string_types = (str,)
    sys.modules["rdkit.six"] = six
    import rdkit
    rdkit.six = six


def _top_auc(buffer, top_n, finish, freq_log, max_oracle_calls):
    """PMO's top_auc, verbatim. Counted-only buffer; see pmo_matched_init_app."""
    import numpy as np
    sum_, prev, called = 0, 0, 0
    ordered = list(sorted(buffer.items(), key=lambda kv: kv[1][1], reverse=False))
    for idx in range(freq_log, min(len(buffer), max_oracle_calls), freq_log):
        temp = sorted(ordered[:idx], key=lambda kv: kv[1][0], reverse=True)[:top_n]
        now = float(np.mean([i[1][0] for i in temp]))
        sum_ += freq_log * (now + prev) / 2
        prev = now; called = idx
    temp = sorted(ordered, key=lambda kv: kv[1][0], reverse=True)[:top_n]
    now = float(np.mean([i[1][0] for i in temp]))
    sum_ += (len(buffer) - called) * (now + prev) / 2
    if finish and len(buffer) < max_oracle_calls:
        sum_ += (max_oracle_calls - len(buffer)) * now
    return sum_ / max_oracle_calls


class Residual:
    """Bootstrap ridge on Morgan features, fitted ONLY to counted observations.

    Deliberately not a retrain of the prior. Fitting 250 online points jointly
    with 249,455 offline ones would drown exactly the samples that carry the new
    region's signal; a residual keeps their influence undiluted.
    """

    def __init__(self, rng, dim=2048):
        self.rng, self.dim, self.W = rng, dim, None

    def fit(self, X, r):
        import numpy as np
        n = len(X)
        if n < 8:
            self.W = None; return
        W = []
        for _ in range(ENSEMBLE):
            ix = self.rng.integers(0, n, n)
            A = X[ix].T @ X[ix] + RIDGE * np.eye(self.dim, dtype=np.float32)
            W.append(np.linalg.solve(A, X[ix].T @ r[ix]))
        self.W = np.stack(W)

    def predict(self, X):
        import numpy as np
        if self.W is None:
            return np.zeros(len(X)), np.zeros(len(X))
        P = X @ self.W.T
        return P.mean(1), P.std(1)

    def sample(self, X):
        import numpy as np
        if self.W is None:
            return np.zeros(len(X))
        return X @ self.W[self.rng.integers(0, ENSEMBLE)]


@app.function(image=image, cpu=(2.0, 2.0), memory=int(8 * 1024),
              timeout=4 * 60 * 60, max_containers=24,
              volumes={str(ARTIFACT_ROOT): artifact_volume})
def run_arm(job: dict[str, Any]) -> dict[str, Any]:
    import os, sys
    import numpy as np
    _shim(); os.chdir("/tmp")
    sys.path.insert(0, str(REMOTE_ROOT / "src"))
    from rdkit import Chem, DataStructs, RDLogger
    from rdkit.Chem import AllChem
    RDLogger.DisableLog("rdApp.*")
    import torch
    from compose_v4.chem.molecular_graph import smiles_to_molecular_graph
    from compose_v4.chem.state import pad_molecular_graph
    from compose_v4.experiments.production_successor_kernel import (
        canonical_state_key, enumerate_factorized_marked_law)
    from compose_v4.benchmark.molleo_task3 import OracleMeter, CountingRule, BudgetExceeded
    from tdc import Oracle

    task, arm, budget = job["task"], job["arm"], job["budget"]
    rng = np.random.default_rng(job["seed"])
    t0 = time.time()
    artifact_volume.reload()
    rt = _runtime(); model, system = rt["model"], rt["system"]

    if task in ("jnk3", "gsk3b"):
        from compose_v4.benchmark.oracles.forest import FrozenForest
        ff = FrozenForest(f"/frozen/{task}_forest.npz")
        def raw(s):
            m = Chem.MolFromSmiles(s)
            if m is None: return (0.0,)
            a = np.zeros((1,), dtype=np.int8)
            DataStructs.ConvertToNumpyArray(
                AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048), a)
            return (float(ff.probabilities(a.astype(np.float64).reshape(1, -1))[0]),)
    else:
        o = Oracle(name=task)
        raw = lambda s: (float(o(s)),)

    # A NaN oracle value is a FAILED evaluation, not a score. Left alone it
    # propagates through np.mean into top10, into top_auc, and into the 22-task
    # sum -- one molecule silently voids the aggregate. Failure scores zero,
    # which is also what an unparseable molecule already scores.
    _raw = raw
    def raw(s):
        v = _raw(s)
        return tuple(0.0 if (x != x or x in (float("inf"), float("-inf"))) else x
                     for x in v)

    canon = lambda s: (lambda m: Chem.MolToSmiles(m) if m else None)(Chem.MolFromSmiles(s))
    meter = OracleMeter(evaluate=raw, budget=budget,
                        counting_rule=CountingRule.PER_MOLECULE,
                        canonicalize=canon, n_objectives=1)

    gen = AllChem.GetMorganGenerator(radius=2, fpSize=2048)
    FP: dict[str, Any] = {}
    def fp(s):
        if s not in FP:
            m = Chem.MolFromSmiles(s)
            a = np.zeros((2048,), dtype=np.float32)
            if m is not None:
                b = np.zeros((2048,), dtype=np.int8)
                DataStructs.ConvertToNumpyArray(gen.GetFingerprint(m), b)
                a = b.astype(np.float32)
            FP[s] = a
        return FP[s]

    # ---- frozen prior from the 249,455 prescreen labels ----
    # WHITELIST, not a blacklist. This was an exclusion list, so adding the puct
    # arm silently opted it INTO loading a surrogate it never uses, and every
    # task without a fitted model died on FileNotFoundError. Naming the arms that
    # DO use the prior means a new arm defaults to not using one.
    SURROGATE_ARMS = ("frozen_greedy", "online_greedy", "online_thompson", "online_ucb")
    use_prior = arm in SURROGATE_ARMS
    prior_fn = lambda S: np.zeros(len(S))
    if use_prior:
        meta = np.load(f"/artifacts/pmo_surrogate/model_{task}.npz", allow_pickle=True)
        if str(meta["kind"][0]) == "mlp":
            net = torch.nn.Sequential(
                torch.nn.Linear(2048, 256), torch.nn.ReLU(),
                torch.nn.Linear(256, 64), torch.nn.ReLU(), torch.nn.Linear(64, 1))
            net.load_state_dict(torch.load(f"/artifacts/pmo_surrogate/model_{task}.pt"))
            net.eval()
            mu, sd = float(meta["mu"][0]), float(meta["sd"][0])
            def prior_fn(S):
                with torch.no_grad():
                    X = torch.from_numpy(np.stack([fp(s) for s in S]))
                    return net(X).view(-1).numpy() * sd + mu
        else:
            w, b = meta["w"], float(meta["b"][0])
            prior_fn = lambda S: np.stack([fp(s) for s in S]) @ w + b

    LAW: dict[str, Any] = {}
    def step_sample(smi):
        """Sample ONE successor from R_theta. Does not enumerate the fiber.

        A rollout consumes one successor, so materialising all APPLY_CAP of them
        to pick one is ~285x more work than the move needs -- the same defect
        that made the T4 future_h arm cost 107 hours per round. We sample a mark
        from the top-APPLY_CAP marks by their model probabilities and apply only
        that one. The pool matches fiber()'s; the difference is that fiber()
        renormalises after canonical dedup while this samples before it, which
        is recorded here rather than silently absorbed.
        """

        if smi not in LAW:
            try:
                st = pad_molecular_graph(smiles_to_molecular_graph(smi), CANONICAL_SLOTS)
                law = enumerate_factorized_marked_law(model, st, float(TIME_POINT))
                pr = np.array([m.probability for m in law.marks], float)
                idx = np.argsort(-pr)[:APPLY_CAP]
                LAW[smi] = (st, law, idx, pr[idx] / pr[idx].sum())
            except Exception:
                LAW[smi] = None
        e = LAW[smi]
        if e is None:
            return None
        st, law, idx, w = e
        mk = law.marks[int(idx[int(rng.choice(len(idx), p=w))])]
        try:
            y = canonical_state_key(system.apply(st, mk.executor_rule_name, mk.action))
        except Exception:
            return None
        return y if y and y != smi else None

    def macro(x, L):
        """Walk L sampled edits. Intermediates are never oracle-evaluated."""
        cur = x
        for _ in range(L):
            nxt_ = step_sample(cur)
            if nxt_ is None:
                break
            cur = nxt_
        return cur if cur != x else None

    gen_fp = AllChem.GetMorganGenerator(radius=2, fpSize=2048)

    TT: dict[str, dict] = {}
    def fiber(smi):
        if smi in TT:
            return TT[smi]
        try:
            st = pad_molecular_graph(smiles_to_molecular_graph(smi), CANONICAL_SLOTS)
            law = enumerate_factorized_marked_law(model, st, float(TIME_POINT))
        except Exception:
            TT[smi] = {}; return TT[smi]
        if not law.marks:
            TT[smi] = {}; return TT[smi]
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
        TT[smi] = out
        return out

    bank_meta = json.loads(Path(f"/artifacts/pmo_matched_init/bank_{task}.json").read_text())
    bank = bank_meta["bank"]
    pop: dict[str, dict] = {}
    ledger: dict[str, list] = {}
    depth: dict[str, int] = {}
    for b in bank:
        c = canon(b["smiles"])
        if c and c not in pop:
            pop[c] = {"smi": c, "u": float(b["u"]), "d": 0}; depth[c] = 0
            meter.prime(c, (float(b["u"]),))

    # ------------------------------------------------------------ COMPOSE-BO ----
    def _bo():
        """Dynamic executable library + online Tanimoto GP + expected improvement.

        THE SEPARATION THIS EXPLOITS. R_theta proposes what is chemically
        reachable; BO decides which of those proposals is worth a counted oracle
        call. PMO's whole difficulty is query scarcity, so spending calls to
        accumulate tree statistics (as PUCT does, at one call per simulation) is
        a poor bargain -- it lost to plain archive search on every task at this
        budget. Here every call either exploits a predicted optimum or resolves
        genuine uncertainty, which is what EI buys.

        NO HORIZON BANDIT. Candidates are generated with equal quotas across
        L in {1,2,4,8} and EI ranks the ENDPOINTS. If L=8 endpoints are worth
        querying, EI selects them; the controller therefore learns horizon
        preference implicitly, through the same statistic it uses for everything
        else. This deliberately avoids the failure we just measured, where a
        separate horizon-UCB had an exploration bonus (0.43) an order of
        magnitude larger than the signal it was meant to detect (0.08), and so
        never left uniform allocation.

        THE GP IS FITTED ONLY TO COUNTED OBSERVATIONS. The frozen ZINC surrogate
        is not used: it scored Spearman 0.0688 on COMPOSE-generated molecules
        above the corpus maximum, so it is actively misleading exactly where
        search operates. Prescreen information enters through the initial
        archive, not through the value model.
        """

        import torch as _t
        fps: dict[str, Any] = {}
        def fpv(s):
            if s not in fps:
                m = Chem.MolFromSmiles(s)
                a = np.zeros((2048,), dtype=np.float32)
                if m is not None:
                    b = np.zeros((2048,), dtype=np.int8)
                    DataStructs.ConvertToNumpyArray(gen_fp.GetFingerprint(m), b)
                    a = b.astype(np.float32)
                fps[s] = a
            return fps[s]

        def tanimoto(A, B):
            num = A @ B.T
            na = A.sum(1)[:, None]; nb = B.sum(1)[None, :]
            den = na + nb - num
            return np.where(den > 0, num / np.maximum(den, 1e-9), 0.0)

        def gp_posterior(Xtr, ytr, Xs):
            """Exact GP. n <= a few hundred here, so a direct solve is fine."""
            n = len(Xtr)
            K = BO_AMP * tanimoto(Xtr, Xtr) + (BO_NOISE + 1e-6) * np.eye(n, dtype=np.float64)
            mu_y = float(ytr.mean())
            try:
                alpha = np.linalg.solve(K, ytr - mu_y)
            except np.linalg.LinAlgError:
                return np.full(len(Xs), mu_y), np.full(len(Xs), np.sqrt(BO_AMP))
            Ks = BO_AMP * tanimoto(Xs, Xtr)
            mu = mu_y + Ks @ alpha
            try:
                V = np.linalg.solve(K, Ks.T)
            except np.linalg.LinAlgError:
                return mu, np.full(len(Xs), np.sqrt(BO_AMP))
            var = BO_AMP - np.sum(Ks * V.T, axis=1)
            return mu, np.sqrt(np.maximum(var, 1e-12))

        def expected_improvement(mu, sd, fstar):
            z = (mu - fstar - BO_XI) / np.maximum(sd, 1e-9)
            zt = _t.from_numpy(np.asarray(z, dtype=np.float64))
            cdf = (0.5 * (1.0 + _t.erf(zt / np.sqrt(2.0)))).numpy()
            pdf = (np.exp(-0.5 * z ** 2) / np.sqrt(2.0 * np.pi))
            return (mu - fstar - BO_XI) * cdf + sd * pdf

        rd_ = 0
        while meter.remaining > 0:
            rd_ += 1
            ranked = sorted(pop.values(), key=lambda z: -z["u"])
            anchors = ranked[:N_LINEAGE]
            if not anchors:
                break
            # --- uncharged executable library, equal quotas across horizons ---
            lib: dict[str, int] = {}
            per_h = max(1, BO_LIB // len(HORIZONS))
            _t0lib = time.time()
            for L in HORIZONS:
                got, tries = 0, 0
                while got < per_h and tries < per_h * 3:
                    tries += 1
                    a_ = anchors[int(rng.integers(0, len(anchors)))]
                    y = macro(a_["smi"], int(L))
                    if y and y not in lib and y not in pop:
                        lib[y] = int(L); got += 1
            if not lib:
                break
            if rd_ == 1:
                print(f"  [{task}/bo] MICROBENCH round 1: library {len(lib)} in "
                      f"{time.time()-_t0lib:.1f}s -> ~{(time.time()-_t0lib)*budget/BO_BATCH/60:.1f} "
                      f"min projected", flush=True)

            # --- GP on counted observations only ---
            obs = [(v[0], k) for k, v in ledger.items()]
            keys = list(lib)
            Xs = np.stack([fpv(y) for y in keys]).astype(np.float64)
            if len(obs) >= 5:
                Xtr = np.stack([fpv(s) for _, s in obs]).astype(np.float64)
                ytr = np.array([u for u, _ in obs], dtype=np.float64)
                mu, sd = gp_posterior(Xtr, ytr, Xs)
                fstar = float(ytr.max())
                acq = expected_improvement(mu, sd, fstar)
            else:
                acq = rng.random(len(keys))       # cold start: uniform

            k_ = min(BO_BATCH, len(keys), meter.remaining)
            pick = [keys[int(i)] for i in np.argsort(-acq)[:k_]]
            for y in pick:
                try:
                    u = meter(y)[0]
                except BudgetExceeded:
                    break
                pop[y] = {"smi": y, "u": u, "d": depth.get(y, 0) + lib.get(y, 1)}
                if y not in ledger:
                    ledger[y] = [float(u), len(ledger) + 1]
            if len(pop) > 400:
                pop2 = {z["smi"]: z for z in sorted(pop.values(), key=lambda z: -z["u"])[:400]}
                pop.clear(); pop.update(pop2)
            # EVERY round persists and prints. Waiting for a call-checkpoint meant
            # the first observable output was two rounds in, which is how this arm
            # looked hung for fifty minutes rather than merely slow.
            s = snap()
            hsel = {L: sum(1 for y in pick if lib.get(y) == L) for L in HORIZONS}
            print(f"  [{task}/bo] rd {rd_:>2} @{s['calls']:>4}  top10c {s['top10_counted']:.4f}  "
                  f"best {s['best_counted']:.4f}  lib {len(lib)}  EI-picked by L {hsel}  "
                  f"{s['t']:>6.0f}s", flush=True)
            while nxt and meter.spent >= nxt[0]:
                curve.append(s); nxt.pop(0)
            _flush(task, arm, job, curve, ledger, meter, rd_, bank_meta, False)
        curve.append(snap())
        return _flush(task, arm, job, curve, ledger, meter, rd_, bank_meta, True)

    # ---------------------------------------------------------------- PUCT ----
    def _puct():
        """Transposition-aware PUCT over the executable molecular graph.

        WHY NOT DOOB-h HERE. PMO is a budgeted BLACK-BOX regime: the objective is
        reachable only through counted oracle calls, and we measured that an
        offline g fitted to the prescreen corpus is uninformative (Spearman
        0.0688) on the chemistry COMPOSE actually generates. Propagating
        h_b(x) = E[g(X_b)] through that g would be future-aware control built on
        noise. Here future value is instead LEARNED ON-POLICY from realised
        oracle returns and backed up through the graph. Same control principle --
        choose edits by what they lead to -- different estimator, chosen by the
        information regime rather than by which one happened to win.

        Nodes are canonical molecules, so two edit paths reaching the same
        molecule SHARE statistics: the transposition table is keyed by canonical
        SMILES, which is the right structure for a graph rather than a tree.

        A simulation costs exactly ONE counted oracle call, spent at the endpoint
        of an L-edit macro. Intermediates are executable molecules but are never
        charged and are allowed to be worse -- that is the whole point.

        L is chosen ONLINE by a UCB bandit over realised improvement per call, so
        the controller decides how nonlocal to move rather than being told.
        """

        N: dict[str, int] = {}
        W: dict[str, float] = {}
        kids: dict[str, list[str]] = {}
        prior: dict[tuple, float] = {}
        parent_of: dict[str, str] = {}

        roots = [q["smi"] for q in sorted(pop.values(), key=lambda z: -z["u"])]
        for r_ in roots:
            N.setdefault(r_, 1); W.setdefault(r_, pop[r_]["u"])

        hN = {L: 0 for L in HORIZONS}
        hW = {L: 0.0 for L in HORIZONS}

        def pick_horizon():
            tot = sum(hN.values())
            for L in HORIZONS:
                if hN[L] == 0:
                    return L
            return max(HORIZONS, key=lambda L: hW[L] / hN[L]
                       + C_HORIZON * math.sqrt(math.log(max(tot, 2)) / hN[L]))

        def widen(x):
            """Expand one more child, sampled from R_theta. Never enumerate 300."""
            cur = kids.setdefault(x, [])
            want = math.ceil(PW_K * (N.get(x, 1) ** PW_ALPHA))
            if len(cur) >= want:
                return
            f_ = fiber(x)
            if not f_:
                return
            fresh = [(y, pr) for y, pr in f_.items() if y not in cur]
            if not fresh:
                return
            ys = [y for y, _ in fresh]
            ws = np.array([pr for _, pr in fresh], float); ws = ws / ws.sum()
            y = ys[int(rng.choice(len(ys), p=ws))]
            cur.append(y)
            prior[(x, y)] = float(dict(fresh)[y])
            parent_of.setdefault(y, x)
            N.setdefault(y, 0); W.setdefault(y, 0.0)

        def puct_pick(x):
            ks = kids.get(x, [])
            if not ks:
                return None
            nx = max(N.get(x, 1), 1)
            best, bs = None, -1e18
            for y in ks:
                q = W.get(y, 0.0) / N[y] if N.get(y, 0) else 0.0
                u = C_PUCT * prior.get((x, y), 1e-3) * math.sqrt(nx) / (1 + N.get(y, 0))
                if q + u > bs:
                    best, bs = y, q + u
            return best

        sims = 0
        while meter.remaining > 0:
            sims += 1
            # --- selection: descend from the best root by PUCT ---
            x = max(roots, key=lambda r_: (W.get(r_, 0.0) / max(N.get(r_, 1), 1))
                    + C_PUCT * math.sqrt(math.log(max(sims, 2))) / (1 + N.get(r_, 0)))
            path = [x]
            for _ in range(8):
                widen(x)
                nxt_ = puct_pick(x)
                if nxt_ is None or N.get(nxt_, 0) == 0:
                    if nxt_ is not None:
                        path.append(nxt_)
                    break
                x = nxt_; path.append(x)

            # --- simulation: an L-edit macro from the leaf, endpoint only ---
            L = pick_horizon()
            leaf = path[-1]
            end = leaf if L == 0 else (macro(leaf, L) or leaf)
            try:
                r_obs = meter(end)[0]
            except BudgetExceeded:
                break
            if end not in pop:
                pop[end] = {"smi": end, "u": r_obs, "d": depth.get(leaf, 0) + L}
            if end not in ledger:
                ledger[end] = [float(r_obs), len(ledger) + 1]

            # --- backup: realised return along the visited path ---
            for nd in path:
                N[nd] = N.get(nd, 0) + 1
                W[nd] = W.get(nd, 0.0) + r_obs
            base = pop[leaf]["u"] if leaf in pop else 0.0
            hN[L] += 1
            hW[L] += max(r_obs - base, 0.0)

            if len(pop) > 400:
                keep = sorted(pop.values(), key=lambda z: -z["u"])[:400]
                keep_s = {z["smi"] for z in keep} | set(roots)
                pop2 = {z["smi"]: z for z in pop.values() if z["smi"] in keep_s}
                pop.clear(); pop.update(pop2)
            while nxt and meter.spent >= nxt[0]:
                s = snap(); curve.append(s); nxt.pop(0)
                hb = {L: (round(hW[L]/hN[L], 5) if hN[L] else None, hN[L]) for L in HORIZONS}
                print(f"  [{task}/puct] @{s['calls']:>4}  top10c {s['top10_counted']:.4f}  "
                      f"best {s['best_counted']:.4f}  nodes {len(N):>5}  sims {sims:>4}  "
                      f"horizons {hb}  {s['t']:>6.0f}s", flush=True)
                _flush(task, arm, job, curve, ledger, meter, sims, bank_meta, False)
        curve.append(snap())
        o = _flush(task, arm, job, curve, ledger, meter, sims, bank_meta, True)
        o["puct"] = {"nodes": len(N), "sims": sims,
                     "horizon_visits": {str(L): hN[L] for L in HORIZONS},
                     "horizon_mean_gain": {str(L): (hW[L]/hN[L] if hN[L] else None)
                                           for L in HORIZONS}}
        return o

    res = Residual(rng)
    Xo, ro = [], []
    cmax = float(bank_meta.get("top1", 0.0))
    CORPUS_MAX[task] = cmax

    def snap():
        v = sorted((p["u"] for p in pop.values()), reverse=True)
        cv = sorted((x[0] for x in ledger.values()), reverse=True)
        return {"calls": meter.spent,
                "top10_all": float(np.mean(v[:10])) if v else 0.0,
                "top10_counted": float(np.mean(cv[:10])) if cv else 0.0,
                "best_counted": cv[0] if cv else 0.0,
                "n_above_corpus_max": int(sum(1 for x in cv if x > cmax)),
                "lineages": len(pop), "t": round(time.time() - t0, 1)}

    curve, nxt = [snap()], [c for c in CHECKPOINTS if c <= budget]

    # Called here, not at definition: _puct closes over snap/curve/nxt/cmax, and
    # invoking it any earlier raises NameError on the first checkpoint.
    if arm == "puct":
        return _puct()
    if arm == "bo":
        return _bo()
    rd, stop = 0, False
    while meter.remaining > 0 and not stop:
        rd += 1
        ranked = sorted(pop.values(), key=lambda p: -p["u"])
        elite = ranked[:N_LINEAGE // 2]
        rest = ranked[N_LINEAGE // 2:]
        div = list(rng.choice(rest, size=min(N_LINEAGE - len(elite), len(rest)),
                              replace=False)) if rest else []
        parents = elite + list(div)
        cand: dict[str, float] = {}
        if arm in ("L1", "mixedL"):
            hs = (1,) if arm == "L1" else HORIZONS
            # SIZED FROM THE COST, NOT GUESSED. Every new state a trajectory
            # visits needs its own R_theta law, so attempts x mean(L) is the real
            # unit of work: the previous 240x{1,2,4,8} cap meant up to 900 law
            # computations per round, minutes of work to fill one round. We need
            # PER_ROUND candidates, so ask for PER_ROUND and stop at 2x attempts.
            tries, _t_mac = 0, time.time()
            while len(cand) < PER_ROUND and tries < PER_ROUND * 2:
                tries += 1
                par_ = parents[int(rng.integers(0, len(parents)))]
                L = int(hs[int(rng.integers(0, len(hs)))])
                y = macro(par_["smi"], L)
                if y and y not in cand and y not in pop:
                    cand[y] = 1.0
                    depth.setdefault(y, depth.get(par_["smi"], 0) + L)
            if rd == 1:
                print(f"  [{task}/{arm}] MICROBENCH round 1: {tries} attempts -> "
                      f"{len(cand)} candidates in {time.time()-_t_mac:.1f}s "
                      f"({len(LAW)} laws) -> ~{(time.time()-_t_mac)*budget/PER_ROUND/60:.1f} "
                      f"min projected", flush=True)
        else:
            for p in parents:
                for y, r in fiber(p["smi"]).items():
                    if y not in cand and y not in pop:
                        cand[y] = r
                        depth.setdefault(y, depth.get(p["smi"], 0) + 1)
        if not cand:
            break
        keys = list(cand)
        rth = np.array([cand[y] for y in keys])

        if arm in ("L1", "mixedL"):
            k = min(PER_ROUND, len(keys), meter.remaining)
            pick = [keys[int(i)] for i in rng.choice(len(keys), size=k, replace=False)]
        elif arm == "no_surrogate":
            acq = rth / rth.sum()
            k = min(PER_ROUND, len(keys), meter.remaining)
            pick = [keys[int(i)] for i in rng.choice(len(keys), size=k, replace=False, p=acq)]
        else:
            base = prior_fn(keys)
            if arm == "frozen_greedy":
                sc = base
            else:
                X = np.stack([fp(y) for y in keys])
                m_, s_ = res.predict(X)
                if arm == "online_thompson":
                    sc = base + res.sample(X)
                elif arm == "online_ucb":
                    sc = base + m_ + KAPPA * s_
                else:
                    sc = base + m_
            k = min(PER_ROUND, len(keys), meter.remaining)
            pick = [keys[int(i)] for i in np.argsort(-sc)[:k]]

        for y in pick:
            try:
                u = meter(y)[0]
            except BudgetExceeded:
                stop = True; break
            pop[y] = {"smi": y, "u": u, "d": depth.get(y, 0)}
            if y not in ledger:
                ledger[y] = [float(u), len(ledger) + 1]
            if use_prior:
                Xo.append(fp(y)); ro.append(float(u) - float(prior_fn([y])[0]))
        if arm.startswith("online") and len(Xo) >= 8:
            res.fit(np.stack(Xo), np.array(ro, dtype=np.float32))
        if len(pop) > 400:
            pop = {p["smi"]: p for p in sorted(pop.values(), key=lambda p: -p["u"])[:400]}
        while nxt and meter.spent >= nxt[0]:
            s = snap(); curve.append(s); nxt.pop(0)
            print(f"  [{task}/{arm}] @{s['calls']:>4}  top10c {s['top10_counted']:.4f}  "
                  f"best {s['best_counted']:.4f}  >corpusmax {s['n_above_corpus_max']:>4}  "
                  f"lin {s['lineages']:>3}  {s['t']:>6.0f}s", flush=True)
            _flush(task, arm, job, curve, ledger, meter, rd, bank_meta, False)
    curve.append(snap())
    out = _flush(task, arm, job, curve, ledger, meter, rd, bank_meta, True)
    print(f"  [{task}/{arm}] DONE top10c {out['top10_counted']:.4f}  "
          f"auc10k {out['auc_top10_official_10k']:.4f}  "
          f"above-corpus {out['n_above_corpus_max']}  {out['seconds']:.0f}s", flush=True)
    return out


def _flush(task, arm, job, curve, ledger, meter, rd, bank_meta, done):
    import numpy as np
    cv = sorted((x[0] for x in ledger.values()), reverse=True)
    o = {"task": task, "arm": arm, "seed": job["seed"], "budget": job["budget"],
         "complete": done, "rounds": rd,
         "calls_spent": meter.spent, "primed": meter.n_primed,
         "uncounted_prescreen_calls": bank_meta.get("uncounted_calls"),
         "top10_counted": float(np.mean(cv[:10])) if cv else 0.0,
         "best_counted": cv[0] if cv else 0.0,
         "corpus_max": CORPUS_MAX.get(task, 0.0),
         "n_above_corpus_max": int(sum(1 for x in cv if x > CORPUS_MAX.get(task, 0.0))),
         "auc_top10_official_10k": _top_auc(ledger, 10, True, 100, 10000),
         "ledger_n": len(ledger), "curve": curve,
         "seconds": curve[-1]["t"] if curve else 0.0}
    d = Path("/artifacts/pmo_control"); d.mkdir(parents=True, exist_ok=True)
    (d / f"{task}_{arm}_s{job['seed']}.json").write_text(json.dumps(o, indent=1))
    artifact_volume.commit()
    return o


@app.function(image=image, cpu=(1.0, 1.0), memory=2048, timeout=8 * 60 * 60,
              volumes={str(ARTIFACT_ROOT): artifact_volume})
def drive(task: str, arms: str, budget: int, seeds: str) -> dict[str, Any]:
    """task may be a comma-separated list, or ALL for every banked task.

    ALL resolves from the volume rather than a hardcoded list, so a task without
    a bank is skipped instead of failing mid-sweep, and drd2 -- whose oracle
    extraction was never verified -- is excluded by simply never having a bank.
    """

    artifact_volume.reload()
    if task.upper() == "ALL":
        d = Path("/artifacts/pmo_matched_init")
        T = sorted(p.name[5:-5] for p in d.glob("bank_*.json"))
    else:
        T = [x.strip() for x in task.split(",") if x.strip()]
    A = [a.strip() for a in arms.split(",") if a.strip()]
    S = [int(s) for s in seeds.split(",") if s.strip()]
    jobs = [{"task": tk, "arm": a, "budget": budget, "seed": s}
            for tk in T for a in A for s in S]
    print(f"  tasks: {', '.join(T)}", flush=True)
    print(f"  {len(jobs)} runs = {len(T)} tasks x {len(A)} arms x {len(S)} seeds, "
          f"{budget} calls each\n", flush=True)
    out = []
    for r in run_arm.map(jobs, order_outputs=False, return_exceptions=True,
                         wrap_returned_exceptions=False):
        if not isinstance(r, dict):
            print(f"  !! {type(r).__name__}: {str(r)[:200]}", flush=True); continue
        out.append(r)
    return {"runs": out}


@app.function(image=image, cpu=(1.0, 1.0), memory=1024, timeout=600,
              volumes={str(ARTIFACT_ROOT): artifact_volume})
def salvage(task: str = "", arms: str = "", seeds: str = "") -> dict[str, Any]:
    """Recover THIS run's persisted results, not every file in the directory.

    The unfiltered version returned all 57 runs ever written here when a driver
    died, and the caller wrote them into that run's output file. A salvage that
    silently substitutes unrelated results is worse than an empty one: the file
    looks like data and is not. Filters are therefore required to match.
    """

    d = Path("/artifacts/pmo_control")
    if not d.exists():
        return {"runs": [], "salvaged": True}
    T = {x.strip() for x in task.split(",") if x.strip()}
    A = {x.strip() for x in arms.split(",") if x.strip()}
    S = {int(x) for x in seeds.split(",") if x.strip()}
    runs = []
    for f in sorted(d.glob("*.json")):
        try:
            r = json.loads(f.read_text())
        except Exception:
            continue
        if T and r.get("task") not in T: continue
        if A and r.get("arm") not in A: continue
        if S and r.get("seed") not in S: continue
        runs.append(r)
    print(f"  salvaged {len(runs)} matching task={sorted(T) or 'ANY'} "
          f"arms={sorted(A) or 'ANY'} seeds={sorted(S) or 'ANY'}", flush=True)
    return {"runs": runs, "salvaged": True}


@app.local_entrypoint()
def main(task: str = "scaffold_hop", arms: str = "", budget: int = 250,
         seeds: str = "1,2", out: str = "") -> None:
    arms = arms or ",".join(ARMS)
    try:
        o = drive.remote(task, arms, budget, seeds)
    except Exception as e:
        print(f"  driver failed ({type(e).__name__}); salvaging THIS run only")
        o = salvage.remote(task, arms, seeds)
    p = Path(__file__).resolve().parents[1] / (out or f"diagnostics/pmo_control_{task}.json")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(o, indent=1))
    print(f"\nwrote {p}")
