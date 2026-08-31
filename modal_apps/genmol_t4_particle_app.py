"""Zero-oracle particle controller for T4. GATES 1-3 SPEND NO DOCKING CALLS.

A particle population over complete molecules, propagated by the frozen R_theta
and twisted by a cheap future-feasibility value:

    pi(a|x,z,b) ∝ R_theta(a|x) * exp[ beta * V_cheap(T(x,a), z, b-1) ]

This is the long-horizon Feynman-Kac deployment: sample from the LOCALLY TWISTED
proposal and carry the local normaliser as the particle weight, which is the
low-variance choice and keeps the algorithm a proper SMC rather than a heuristic
reweighting. S_dock is absent here BY DESIGN -- the point of the gates is to
prove the free constraints can be navigated before any Vina call is authorised.

Nothing is per-receptor. braf must discover deletion and parp1 growth from the
goal z and the budget b, not from their names.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any
import collections  # noqa: F401

import modal

from modal_apps.genmol_t4_opt_app import (
    APPLY_CAP, CANONICAL_SLOTS, QED_MIN, SA_MAX, TIME_POINT,
    ARTIFACT_ROOT, REMOTE_ROOT, ROOT, artifact_volume, _runtime,
)
from modal_apps.genmol_t4_opt_app import image as _opt_image

image = _opt_image.add_local_file(
    ROOT / "modal_apps/genmol_t4_opt_app.py",
    str(REMOTE_ROOT / "modal_apps/genmol_t4_opt_app.py"), copy=True)

app = modal.App("genmol-t4-particle")

# ---- frozen. global, never per-cell, never per-receptor. ----
N_PART      = 48     # particles
POOL_M      = 96     # top-M marks twisted per step. Declared approximation:
                     # pi is computed over the top-M of the APPLY_CAP pool, not
                     # all 300. 96*17.4ms = 1.7s against a 6.74s R_theta call.
PER_RULE_K  = 20     # minimum marks kept PER OPERATOR RULE, whatever their rank.
                     # RAISED 12 -> 20 on measurement: the parp1 witness route's
                     # ENTRY action is a cycle_close at within-family rank 18
                     # (global rank 414), so a floor of 12 excluded the only
                     # known way into the basin. Full-route within-family ranks
                     # are [0,0,0,1,2,2,3,5,7,18,20,29,109,137]; 20 covers the
                     # median and the entry move. It does NOT cover the 109/137
                     # tail -- that is the macro engine's job, not the floor's.
                     # A twist can only reweight the support it is given. Taking
                     # the global top-32 by R_theta probability alone let the
                     # highest-probability operator monopolise the pool, so
                     # deletion moves -- median rank ~7.2% of APPLY_CAP=300, i.e.
                     # routinely outside a top-32 cut -- were unreachable no
                     # matter how strongly V favoured them. braf moved only
                     # dheavy -0.8 in 4 edits for this reason. Operator-type
                     # coverage is generic: no rule is named or preferred.
STAGE_LEN   = 8      # edits per stage; state reuse banks between stages
N_STAGES    = 3      # 3 x 8 = 24 effective edits
BETA        = 4.0    # twist strength on V_cheap
ESS_FRAC    = 0.3    # resample below this fraction of N. Lowered from 0.5:
                     # with the ring term and the similarity barrier both active
                     # the feasible region is narrow, weights concentrate, and
                     # resampling at 0.5 collapsed 24 lineages to 5 in eight
                     # depths. Resample only on genuine degeneracy.
FANOUT      = 32     # containers per run; particles split across them.
                     # MEASURED: 164s/depth of which only ~20s was compute --
                     # one advance.map() per depth costs ~1.8s per chunk in
                     # dispatch, so 80 chunks against 50 particles paid 80
                     # round-trips to do 50 particles of work. Right-sized to
                     # ~3-4 specs per chunk, which minimises (compute/N + N*1.8).
_OLD_FANOUT = 80     # was 8,
                     # i.e. 10% of the 80-container cap, while segments multiply
                     # the per-depth cost by the mean segment length (~3.8). At
                     # 8 containers a depth cost ~400s and a 72-depth run ~8h.
FRONTIER    = 24     # banked states carried between stages
# Generic displacement ladder. NOT a single D fitted to what a competitor found:
# every stratum is carried simultaneously and the counted docking reward decides
# later which one pays. d=0 is the plain feasibility target, so the ladder
# strictly contains the previous controller as its lowest rung.
LADDER      = (0, 5, 10, 15, 20)   # displacement rungs. The controller's DEPTH
                     # is what had to grow: reaching +14 heavy on parp1 costs 23
                     # OPERATIONS (17 insert + 3 delete + 3 close), not 14, because
                     # |dHeavy| is a NET measure that hides deletions, ring
                     # closures and replaced atoms. Runs now use 5 stages x 8 = 40
                     # edits so the required ~23-30 operations fit with headroom.
GAMMA_RULE  = 0.75   # operator-marginal debiasing exponent. See _debias().
# SEGMENT LENGTHS. The route audit measured the bridge: the first ~5-6 atoms of
# an appended fragment create NO shared structure with any good endpoint,
# because a Morgan radius-2 environment does not match until the inserted atom's
# own neighbours exist. parp1 sat at 31 shared bits for 24 consecutive steps and
# 5ht1b oscillated 22<->21 for 17 rounds under a 1-step-greedy guide.
# A controller that scores every intermediate therefore CANNOT cross the bridge.
# Particles advance by a whole segment and are judged only at its end. The rungs
# straddle the measured bridge length so no rung is assumed correct: 1 and 3 are
# shorter than the bridge, 6/9/12 exceed it, and the counted docking reward
# decides which pays.
SEGMENTS    = (1, 2, 4, 8)
# The short rungs are RESTORED. Dropping them to save iterations was a mistake
# based on a wrong reading of their role: they are not there to cross bridges,
# they are the corrective moves that REPAIR feasibility after a long segment
# overshoots. With only k=4/8 available every move overshot, sim fell to 0.22,
# feasibility went to ~0, and the archive held 21 states -- so only 17 of 50
# docking calls were even spendable and the best score fell to -9.2 from -10.5.
# Speed comes from capping ROUNDS instead (see `rounds` below), which leaves
# every rung in place.
# TEMPORALLY EXTENDED OPTIONS. The segment gate showed that an UNGUIDED k-step
# interior is starved: eight random length-8 walks cover an astronomically
# smaller fraction of the space than eight one-step samples, so large k lost at
# equal sample budget. The fix is not exponentially more samples -- it is to give
# the segment interior a generic STRUCTURAL MODE while leaving the actual legal
# edit and site to R_theta. A grow-option segment then performs k insertions
# instead of winning k independent lotteries.
#
# The modes are receptor-agnostic and all are carried simultaneously; the counted
# docking reward decides which pays. Nothing encodes "parp1 needs insertions".
_OPEN  = ("cycle_open", "ring_system_restate", "bond_reroute")
_GROW  = ("atom_insert",)
_SHRINK = ("atom_delete",)
_CLOSE = ("cycle_close", "cycle_insert", "ring_system_restate")
_REARR = ("bond_reroute", "bond_reorder", "atom_restate_semantic")


def option_families(mode, hop, seg):
    """An option is a PROGRAM over the segment, not one repeated family.

    'construct_ring' is grow, grow, ..., then close: a ring closure is often not
    even legal until its precursor chain exists, so eight consecutive close
    attempts is not a ring-building program. The schedule is generic -- it names
    no receptor and encodes no target -- and R_theta still chooses which legal
    edit and which site inside whichever family the program calls for.
    """
    if mode == "grow":
        return _GROW
    if mode == "shrink":
        return _SHRINK
    if mode == "construct_ring":
        return _GROW if hop < max(1, seg - 1) else _CLOSE
    if mode == "rebuild":
        # OPEN -> then adaptive grow/close (see one_step). Read off the parp1 witness route, which goes
        # seed -> (tricycle OPENED at depth 3) -> grow -> (RECLOSED at depth 12)
        # -> -13.6. Binding is FLAT across the opened phase (-7.2 -> -7.6) and
        # only pays at the reclose, so no controller that scores individual
        # edits can hold the program together. The schedule is generic: open,
        # extend, close. No target, no receptor, no landmark structure.
        if hop == 0:
            return _OPEN
        return _GROW if hop < max(2, seg - 1) else _CLOSE
    if mode == "local":
        return _REARR
    return ()                            # 'mixed': no restriction


MODES = ("local", "grow", "shrink", "construct_ring", "rebuild", "mixed")
MIN_QUOTA = 2        # protected particles per (option, segment-length) cell
BETA_DOCK   = 1.0    # weight on the online docking term S_dock
DOCK_PER_STAGE = 40  # COUNTED oracle calls spent per stage on the frontier


def _debias(pr, rules, gamma):
    """Partially flatten the OPERATOR-TYPE marginal of R_theta.

    Measured on the parp1 seed: 57 ring-closing marks share 7.2e-5 of the mass
    (mean 1.3e-6 each) while 134 atom_insert marks share 0.457 (mean 3.4e-3) --
    ring closure is under-weighted ~2,700x per move. This is flat in process
    time (7.2e-5 to 7.7e-5 over t in [0.05, 0.9]), so it is not an artefact of
    where we sample the process, and the under-ranked moves are GOOD: their
    products have QED 0.82-0.88 at SA 3.9-4.4.

    The likely cause is the factorisation itself: atom_insert commits to one
    site, cycle_close to two, so a multi-site operator pays a product of two
    small per-site probabilities regardless of product quality.

    R_theta's WITHIN-operator ranking is informative (it separates QED 0.88
    closures from QED 0.54 ones) so that ordering is preserved exactly. Only the
    BETWEEN-operator marginal is rescaled:

        p'(y) ∝ p(y) * (m_r / P(rule=r))^gamma

    gamma = 0 leaves R_theta untouched; gamma = 1 makes every operator type
    equiprobable in aggregate. This is a declared, uniform correction applied to
    every rule identically -- no rule is named, preferred, or tuned per target.
    """
    import numpy as np
    pr = np.asarray(pr, dtype=np.float64)
    if gamma <= 0:
        return pr / pr.sum()
    tot = collections.defaultdict(float)
    for p_, r_ in zip(pr, rules):
        tot[r_] += float(p_)
    n_rules = max(1, len(tot))
    out = np.array([p_ * ((1.0 / n_rules) / max(tot[r_], 1e-300)) ** gamma
                    for p_, r_ in zip(pr, rules)], dtype=np.float64)
    s = out.sum()
    return out / s if s > 0 else pr / pr.sum()


def _dfeat(pr):
    """Low-dimensional PHYSICOCHEMICAL features for S_dock.

    S_dock previously regressed docking score on 2048-bit Morgan fingerprints
    from ~100 counted calls -- 20x more features than samples, which is why the
    ablation came out null and why earlier surrogates on this task sat at
    Spearman ~0.2. The signal is present in a handful of interpretable
    descriptors: measured over 1,207 of OUR OWN docked molecules, Spearman
    against ds is logP -0.43, rings -0.40, aromatic atoms -0.37, TPSA -0.35,
    heavy -0.34, while QED is -0.01. Thirteen features against a hundred samples
    is a well-determined fit.

    Nothing here is receptor-specific: the SAME descriptor basis is used for
    every target and the weights are learned per target from counted calls.
    """
    import numpy as np
    return np.array([pr["logP"] / 5.0, pr["rings"] / 6.0, pr["arom"] / 4.0,
                     pr["tpsa"] / 100.0, pr["heavy"] / 40.0, pr["mw"] / 500.0,
                     pr["hbd"] / 5.0, pr["hba"] / 8.0, pr["rotb"] / 8.0,
                     pr["fsp3"], pr["qed"], pr["sa"] / 10.0, 1.0],
                    dtype=np.float64)


def _feat(pr, b):
    """Features for V_cheap. Free properties + remaining budget only."""
    import numpy as np
    return np.array([pr["qed"], pr["sa"] / 10.0, pr["sim"],
                     pr["dheavy"] / 20.0, pr["drings"] / 5.0,
                     float(b) / 24.0, 1.0], dtype=np.float64)


@app.function(image=image, cpu=(2.0, 2.0), memory=int(6 * 1024),
              timeout=60 * 60 * 4, max_containers=80,
              # keep workers warm BETWEEN depths: the model load is ~170s and a
              # depth is ~40s of work, so cold-starting every depth dominated the
              # measured 150s/depth.
              scaledown_window=900,
              enable_memory_snapshot=True,
              volumes={str(ARTIFACT_ROOT): artifact_volume})
def advance(job: dict[str, Any]) -> list[dict[str, Any]]:
    """Advance a chunk of particles ONE step under the twisted proposal.

    Returns per particle: chosen successor, log local normaliser (the weight
    increment), and the free properties of the successor. One R_theta call per
    particle per step; the twist costs POOL_M applies on top of it.
    """
    import os, sys
    import numpy as np
    os.chdir("/tmp"); sys.path.insert(0, str(REMOTE_ROOT / "src"))
    from rdkit import Chem, DataStructs, RDLogger
    from rdkit.Chem import QED, AllChem, RDConfig, Descriptors
    from rdkit.Chem import rdMolDescriptors as rdMD
    RDLogger.DisableLog("rdApp.*")
    sys.path.append(os.path.join(RDConfig.RDContribDir, "SA_Score"))
    import sascorer
    from compose_v4.chem.molecular_graph import smiles_to_molecular_graph
    from compose_v4.chem.state import pad_molecular_graph
    from compose_v4.experiments.production_successor_kernel import (
        canonical_state_key, enumerate_factorized_marked_law)
    from compose_v4.inference.twisted_smc import twist
    from compose_v4.gates.med_chem_gate import is_executable as _MED_CHEM_OK

    rt = _runtime(); model, system = rt["model"], rt["system"]
    delta = float(job["delta"]); x0 = job["x0"]
    beta = float(job["beta"])
    W = {str(k): np.array(v, dtype=np.float64)
         for k, v in (job.get("w_by_d") or {}).items()}
    w_dock = np.array(job.get("w_dock") or [], dtype=np.float64)
    bdock = float(job.get("beta_dock") or 0.0)
    m0 = Chem.MolFromSmiles(x0)
    fp0 = AllChem.GetMorganFingerprintAsBitVect(m0, 2, nBits=2048)
    _A0 = set(fp0.GetOnBits())
    h0 = m0.GetNumHeavyAtoms()
    r0 = Chem.GetSSSR(m0) if isinstance(Chem.GetSSSR(m0), int) \
        else len(Chem.GetSymmSSSR(m0))

    def props(smi):
        m = Chem.MolFromSmiles(smi) if smi else None
        if m is None:
            return None
        if not _MED_CHEM_OK(smi):
            # PATHWISE gate only: is this an executable molecule? Applying the
            # full med-chem gate here strangled the search -- a particle could
            # not pass THROUGH a temporarily unattractive state to reach a good
            # one, and the first three GATED30 cells came back at -9.7/-8.2/-7.8
            # against an ungated archive's -9.9/-10.2/-8.8, barely displaced
            # from their seeds. QED/SA/similarity belong at the RETURNED
            # molecule, which is also what T4 itself requires.
            return None
        try:
            q = float(QED.qed(m)); sa = float(sascorer.calculateScore(m))
        except Exception:
            return None
        fp = AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048)
        sim = float(DataStructs.TanimotoSimilarity(fp0, fp))
        _bs = set(fp.GetOnBits())
        _new = len(_bs - _A0)                          # novel environments
        _lost = len(_A0 - _bs)                         # DESTROYED seed environments
        c = [max(0.0, (QED_MIN - q)) / QED_MIN,
             max(0.0, (sa - SA_MAX)) / SA_MAX,
             max(0.0, (delta - sim)) / delta]
        nr = len(Chem.GetSymmSSSR(m))
        try:
            _d = {"logP": float(Descriptors.MolLogP(m)),
                  "tpsa": float(Descriptors.TPSA(m)),
                  "mw": float(Descriptors.MolWt(m)),
                  "rings": float(rdMD.CalcNumRings(m)),
                  "arom": float(rdMD.CalcNumAromaticRings(m)),
                  "hbd": float(rdMD.CalcNumHBD(m)),
                  "hba": float(rdMD.CalcNumHBA(m)),
                  "rotb": float(rdMD.CalcNumRotatableBonds(m)),
                  "fsp3": float(rdMD.CalcFractionCSP3(m))}
        except Exception:
            return None
        return {"smi": smi, "qed": q, "sa": sa, "sim": sim, "fp": np.array(fp),
                "newbits": int(_new), "lostbits": int(_lost),
                **_d,
                "c": c, "v": max(c),
                "heavy": m.GetNumHeavyAtoms(), "rings": nr,
                "dheavy": m.GetNumHeavyAtoms() - h0, "drings": nr - r0}

    def v_cheap(pr, b, d):
        """V = log h_hat_b(.; z, d), so beta=1 IS the exact Doob transform.

        h_b(x;z) = Pr_R[g_z(X_b)=1 | X_0=x] is estimated by fitted value
        iteration on FREE rollout data (see fit_h). Budget dependence then lives
        inside h_b itself and no pressure schedule is imposed by hand.

        Stage 0 has no fitted h yet, so it falls back to the analytic bootstrap
        -- which is DIAGNOSTIC ONLY and is not the reportable controller.
        """
        w_v = W.get(str(d))
        if w_v is not None and np.any(w_v):
            z = float(_feat(pr, b) @ w_v)
            return -float(np.logaddexp(0.0, -z))        # log sigmoid(z)
        # Stage-0 bootstrap. The two terms carry DIFFERENT budget semantics and
        # must not share a schedule:
        #   feasibility -> divided by (1+b), so a bridge state is free to look
        #                  bad early and pressure rises as the horizon closes;
        #   displacement -> NOT divided, because reaching d atoms needs sustained
        #                  pressure over the whole path.
        # Dividing displacement by (1+b) too made one atom of progress worth
        # 0.125/21 ~ 0.006 of V at b=20, i.e. a 1.02x reweight under beta=4 --
        # numerically nothing, which is why braf drifted at the untwisted rate of
        # -0.08 atoms/edit (measured: shrink mass 0.241, grow mass 0.161).
        # Scale CALIBRATED from the measured one-step mass split, not swept:
        # probe(braf) gives shrink 0.241 vs grow 0.161, so selecting progress
        # ~60% of the time needs an odds ratio ~7, i.e. dV = log(7)/beta = 0.49
        # per atom at beta=4. gap/2 delivers 0.5.
        # Displacement is the VECTOR Delta(x), not heavy atoms alone. Measured
        # over 1,207 of our own docked molecules, ring count is the #2 binding
        # predictor (Spearman -0.40, second only to logP -0.43), and the audit
        # says the growth cells need +3 rings. Every run so far delivered
        # drings ~ 0.0 while hitting +14 heavy, which is why reaching the right
        # SIZE never reached the right SCORE. One ring is weighted as ~3 heavy
        # atoms, the ratio in which the audit's targets move (+14 heavy, +3 rings).
        gap_h = max(0.0, float(d) - abs(float(pr["dheavy"])))
        d_ring = float(d) / 5.0                       # +3 rings at the d=16 rung
        gap_r = max(0.0, d_ring - abs(float(pr["drings"]))) * 3.0
        gap = gap_h + gap_r
        # SIMILARITY EFFICIENCY. Tanimoto gives sim ~ |A| / (|A| + new_bits), so
        # the budget that limits displacement is NOVEL FINGERPRINT BITS, not atom
        # count. Measured on the published constraint-satisfying molecules vs
        # ours, at the SAME similarity (~0.42): they spend 1.3-1.8 novel bits per
        # added atom and buy +14/+15 atoms, we spend 10-15 and buy +1/+2. Growth
        # that reuses environments the molecule already contains is nearly free
        # in similarity; growth that invents new environments is not.
        # This is read off the Tanimoto algebra and measured on our own edits --
        # it names no target and copies no molecule.
        # Reward atoms-per-novel-bit rather than penalising bits-per-atom. The
        # penalty form divides by max(1,|dheavy|), so at dheavy ~ 0 it charged
        # every candidate a large constant and fought the FIRST growth step --
        # mean displacement stalled at +4.3 while single particles reached +13.
        # This form is 0 at zero displacement (neutral) and grows as efficient
        # growth is found. Published molecules run ~0.6-0.8 atoms per novel bit
        # at the same similarity; our edits ran ~0.07-0.1.
        _eff = abs(float(pr["dheavy"])) / max(1.0, float(pr.get("newbits", 0)))
        gap = gap - 2.0 * min(_eff, 1.5)
        # SCAFFOLD PRESERVATION. Tanimoto is (|A| - lost) / (|A| + new), so a
        # DESTROYED seed environment costs the numerator while a new one costs
        # only the denominator -- losing a bit is roughly twice as expensive as
        # adding one, and buys nothing. Structural inspection of the published
        # molecules shows exactly this: they retain the seed scaffold verbatim
        # and append a second ring system through a linker at one peripheral
        # position, while our best parp1 molecule chlorinated the core and
        # cyclised a side chain -- rearranging the scaffold rather than
        # extending it (lost 16-17 bits for +1 to +2 atoms). Charging lost bits
        # directly is read off the metric, not copied from their chemistry.
        gap = gap + 0.15 * float(pr.get("lostbits", 0))
        # Constraints are NOT interchangeable in how they relax with budget.
        # QED and SA are RECOVERABLE: a later edit can raise QED or lower SA, so
        # a bridge state may violate them early and repair them as b -> 0. That
        # is what the (1+b) relaxation is for.
        # Similarity to x_0 is QUASI-MONOTONE DECREASING: edits drive it down and
        # adding more atoms never brings it back. Relaxing it early is therefore
        # a one-way ticket out of the feasible set. Measured: growth reaches
        # |dheavy| = +8 but sim falls to 0.34-0.43 against delta = 0.4, so d8
        # reach stayed 0 -- the displaced states existed and were simply
        # infeasible. Similarity gets FULL weight at every b.
        c_ = pr["c"]                      # (qed, sa, sim) shortfalls
        recoverable = float(c_[0] + c_[1]) / (1.0 + float(b))
        # A LINEAR sim shortfall normalised by delta is far too flat at the wall:
        # at sim 0.388 vs delta 0.40 it contributes 0.03 against a displacement
        # term of 0.5, so displacement outweighed the binding constraint 16:1 and
        # particles walked straight out of the feasible set (measured: dheavy
        # +7.3 at sim 0.388). Similarity is a HARD endpoint constraint that is
        # quasi-monotone and unrecoverable, so it gets a quadratic barrier with a
        # safety margin instead -- the standard treatment, not a fitted weight.
        MARGIN = 0.05
        slack = float(pr["sim"]) - delta
        barrier = (max(0.0, MARGIN - slack) / MARGIN) ** 2   # 0 outside, ->1 at
        return -gap / 2.0 - recoverable - barrier            # delta, 4 at -margin

    out = []

    def one_step(smi, b, rng, d_rung, twisted=True, mode="mixed",
                 hop_i=0, seg_i=1):
        """One transition. Returns (successor, logZ, props) or None.

        twisted=False proposes from the (debiased) reference law alone, which is
        what a segment interior uses.
        """
        try:
            st = pad_molecular_graph(smiles_to_molecular_graph(smi),
                                     CANONICAL_SLOTS)
            law = enumerate_factorized_marked_law(model, st, float(TIME_POINT))
        except Exception:
            return None
        pr_ = np.array([m.probability for m in law.marks], float)
        order = np.argsort(-pr_)
        ix_l, seen_ = [], set()
        by_rule: dict[str, int] = {}
        for j in order:                           # per-rule coverage over the
                                                  # FULL law. Scoping this to
                                                  # order[:APPLY_CAP] made the
                                                  # guarantee vacuous: parp1's 57
                                                  # ring-forming marks all rank
                                                  # >=305 against a cap of 300, so
                                                  # every ring-forming move was
                                                  # excluded and no reweighting
                                                  # could ever recover it.
            rn = law.marks[int(j)].executor_rule_name
            if by_rule.get(rn, 0) < PER_RULE_K:
                by_rule[rn] = by_rule.get(rn, 0) + 1
                ix_l.append(int(j)); seen_.add(int(j))
        for j in order[:APPLY_CAP]:               # then global top-M
            if len(ix_l) >= POOL_M:
                break
            if int(j) not in seen_:
                ix_l.append(int(j)); seen_.add(int(j))
        fams = option_families(mode, hop_i, seg_i)
        if mode in ("construct_ring", "rebuild"):
            # ADAPTIVE, not one-shot. Firing a single closure after exactly k-1
            # growth steps requires that a legal closure exist at that precise
            # state and site, which is a very low-probability way to build a
            # ring -- the previous schedule produced net drings of -0.4 to +0.1,
            # i.e. no rings at all. Instead: ask at EVERY hop whether a legal
            # closure opportunity now exists, and take it when it does, growing
            # the precursor otherwise.
            # ADAPTIVE for BOTH ring programs. `rebuild` was left one-shot --
            # a single close fired at the last hop -- which is precisely the
            # schedule that produced zero rings for construct_ring before it was
            # made adaptive. Measured: rebuild SEGring 0.5-0.6 against
            # construct_ring's 2.9-6.1, similarity never recovered from the
            # bridge (0.48 -> 0.23 monotone) and the feasible frontier drained.
            avail = {law.marks[int(j)].executor_rule_name for j in ix_l}
            close_now = [f for f in _CLOSE if f in avail]
            if mode == "rebuild" and hop_i == 0:
                fams = _OPEN            # open first, then behave adaptively
            else:
                fams = tuple(close_now) if (close_now and hop_i >= 1) else _GROW
            # RETURNABILITY TRIGGER. A fixed grow^(k-1) -> close schedule closes
            # far too late: measured, the controller reached +24 heavy with +4.8
            # rings before closing, while every bridge state on the witness route
            # escapes to FEASIBLE in one closure at only +4 heavy. Similarity
            # cannot be bought back once the molecule has doubled. So instead of
            # a schedule: at every intermediate, if ANY legal closure returns the
            # molecule to the terminal-feasible set, take it NOW and end the
            # option. The margin min(QED-0.6, 4-SA, SIM-delta) >= 0 is exactly
            # v == 0, so no new constant is introduced. Uses only the task's own
            # constraints -- no target, no receptor, no fitted "close at +4".
            # never fire before `rebuild` has actually opened (hop 0 is OPEN),
            # and never on the very first hop of construct_ring either.
            if mode in ("rebuild", "construct_ring") and close_now and hop_i >= 1:
                _rets = []
                for _j in ix_l:
                    if law.marks[int(_j)].executor_rule_name not in _CLOSE:
                        continue
                    _mk = law.marks[int(_j)]
                    try:
                        _y = canonical_state_key(
                            system.apply(st, _mk.executor_rule_name, _mk.action))
                    except Exception:
                        continue
                    if not _y or _y == smi:
                        continue
                    _p = props(_y)
                    if _p is not None and float(_p["v"]) == 0.0:
                        _rets.append((float(_p["sim"]), _y, _p))
                # DO NOT close at the first returnable opportunity. Measured:
                # closing on first availability fired at +1 to +4 heavy atoms and
                # produced a docked pool with median dheavy +1 / drings +0
                # (best -8.6), while the witness route closes at +6 heavy / +2
                # rings (-10.8) and reaches +14/+3 (-13.6). Closing early locks
                # in undisplaced molecules.
                #
                # Instead: keep GROWING while returnability is ROBUST, and close
                # only when it becomes FRAGILE -- i.e. when just one returnable
                # closure remains. That uses the structure of the constraint set
                # itself (how many legal closures still land feasible) rather
                # than a fitted "close at +N heavy" threshold.
                if len(_rets) == 1:
                    return _rets[0][1], 0.0, _rets[0][2], True   # last chance
                if len(_rets) >= 2:
                    fams = _GROW      # returnability is robust: keep building
        if fams:
            # the OPTION selects the structural mode; R_theta still selects which
            # legal edit and which site within that mode. Fall back to the full
            # pool when the mode offers nothing here, so an option can never make
            # a state a dead end.
            sub = [j for j in ix_l
                   if law.marks[int(j)].executor_rule_name in fams]
            if sub:
                ix_l = sub
        ix = np.array(ix_l, dtype=int)
        w = _debias(pr_[ix], [law.marks[int(j)].executor_rule_name for j in ix],
                    float(job.get("gamma_rule", GAMMA_RULE)))
        ys, keep = [], []
        for j, k in enumerate(ix):
            mk = law.marks[int(k)]
            try:
                y = canonical_state_key(system.apply(st, mk.executor_rule_name,
                                                     mk.action))
            except Exception:
                continue
            if not y or y == smi:
                continue
            p = props(y)
            if p is None:
                continue
            # PATHWISE we enforce EXECUTABLE LEGALITY ONLY. T4 constrains the
            # RETURNED molecule, not every intermediate COMPOSE state, and an
            # intermediate may legitimately have worse QED/SA/similarity while
            # crossing a bridge. A hard per-edit similarity filter (which this
            # code previously applied) reintroduces exactly the one-step myopia
            # the route audit identified as the failure mode. Free-constraint
            # control now happens at SEGMENT BOUNDARIES, below.
            ys.append(p); keep.append(w[j])
        if not ys:
            return None
        keep = np.array(keep, float)
        if not twisted:
            # INSIDE a segment the proposal is the reference process itself.
            # Greedily re-scoring after every edit is what produced the measured
            # insert/delete oscillation, so purpose is applied at the boundary.
            tw0, lz0 = twist(keep, np.zeros(len(ys)))
            pick0 = int(rng.choice(len(ys), p=tw0))
            return ys[pick0]["smi"], float(lz0), ys[pick0], False
        vv = np.array([v_cheap(p, max(0, b - 1), d_rung) for p in ys], float)
        if w_dock.size and bdock:
            # S_dock: the ONLY term carrying protein information. Fitted on
            # counted docking calls, standardised so its scale does not depend on
            # how many calls have been spent. Separate from h by construction --
            # h keeps the particle feasible and displaced, S_dock says which
            # displaced direction binds.
            fpm = np.stack([_dfeat(p) for p in ys]).astype(np.float64)
            sd = fpm @ w_dock
            sd = (sd - sd.mean()) / (sd.std() + 1e-9)
            vv = vv + bdock * sd
        # twist() is the unit-tested primitive: its log-normaliser is a TRUE
        # logsumexp. The earlier inlined version subtracted the max without
        # adding it back, which cancels in the (renormalised) proposal but left
        # every particle weight short by a state-dependent constant and so
        # distorted resampling. Invisible at beta=0; only bites once twisted.
        tw, logZ = twist(keep, beta * vv)
        if not np.isfinite(logZ):
            return None
        pick = int(rng.choice(len(ys), p=tw))
        return ys[pick]["smi"], float(logZ), ys[pick], False

    # SEGMENT ADVANCE. The route audit measured a ~5-6 edit bridge over which no
    # intermediate carries reward, so a controller that judges every intermediate
    # cannot cross it. A particle advances a whole segment and is scored only at
    # the segment END; intermediates are never given the chance to be vetoed.
    # Segment length is a per-particle stratum, straddling the measured bridge,
    # so no length is assumed correct.
    for spec in job["specs"]:
        smi = spec["smi"]; b = int(spec["b"])
        rng = np.random.default_rng(spec["seed"])
        seg = max(1, int(spec.get("seg", 1)))
        d_rung = spec.get("d", 0)
        cur, last = smi, None
        realized = 0
        stop_at = -1
        for hop in range(seg):
            r_ = one_step(cur, max(0, b - hop), rng, d_rung, twisted=False,
                          mode=spec.get("mode", "mixed"), hop_i=hop, seg_i=seg)
            if r_ is None:
                stop_at = hop             # keep the longest valid PREFIX
                break
            cur, _lz, last, _closed = r_
            realized += 1
            if _closed:
                break                     # returnable closure taken: option done
        if last is None or cur == smi:
            out.append({"idx": spec["idx"], "smi": None}); continue
        # Feynman-Kac at SEGMENT granularity: the segment is proposed from the
        # reference process, so the importance weight is the boundary potential
        # exp(beta*V + bdock*S_dock) evaluated on the segment ENDPOINT. Credit
        # for the whole segment therefore attaches to every edit that produced
        # it, which is the temporal-credit fix the route audit called for.
        # THREE DISTINCT LEVELS, deliberately not collapsed:
        #   individual edits      -> executable legality only
        #   segment boundaries    -> FUTURE-returnability control (V), no hard cut
        #   reported/docked state -> hard QED/SA/SIM, enforced by the archive
        # Hard-rejecting a boundary below delta would recreate the one-step
        # myopia one level up: a bridge spanning more than one segment must be
        # allowed to sit below delta at an intermediate boundary. Returnability
        # is guaranteed instead by the ARCHIVE, which only ever admits feasible
        # states, so whatever we dock or report satisfies the constraints by
        # construction while the walk itself stays free.
        vend = v_cheap(last, max(0, b - seg), d_rung)
        if w_dock.size and bdock:
            sdv = float(_dfeat(last) @ w_dock)
            vend = vend + bdock * sdv
        acc = float(beta) * float(vend)
        out.append({"idx": spec["idx"], "smi": cur, "logw": acc,
                    "seg_used": realized, "seg_asked": seg,
                    "stopped_at": stop_at,
                    "mode": spec.get("mode", "mixed"),
                    # (C) PER-SEGMENT deltas: what THIS option actually changed,
                    # not the cumulative displacement an inherited particle
                    # already carried. The cumulative version made `local` read
                    # +8.5 when local operations barely move heavy-atom count.
                    "seg_dheavy": int(last["dheavy"]) - int(props(smi)["dheavy"]),
                    "seg_drings": int(last["drings"]) - int(props(smi)["drings"]),
                    "pr": {k: (v.tolist() if hasattr(v, "tolist") else v)
                           for k, v in last.items() if k not in ("smi", "fp")}})
    return out


def fit_h(rows, l2=1.0, iters=200, lr=0.5):
    """Fitted finite-horizon value: logistic regression of

        label(y, b) = 1[ the continuation from y reached the feasible set
                         within its remaining budget b ]

    on (QED, SA, sim, dheavy, drings, b, 1). The labels come from rollouts we
    already paid for with free properties, so h costs ZERO oracle calls. This is
    the object that lets a bridge state look bad now and still be favoured: its
    label depends on what happened AFTER it, not on its own feasibility.
    """
    import numpy as np
    if not rows:
        return [0.0] * 7
    X = np.array([r[0] for r in rows], dtype=np.float64)
    y = np.array([r[1] for r in rows], dtype=np.float64)
    if y.max() == y.min():                 # degenerate: no signal to fit
        return [0.0] * 7
    w = np.zeros(X.shape[1], dtype=np.float64)
    n = len(y)
    for _ in range(iters):
        z = X @ w
        p = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
        g = X.T @ (y - p) / n - l2 * w / n
        gn = float(np.linalg.norm(g))
        if gn < 1e-9:
            break
        w += lr * g / gn
    return [float(v) for v in w]


@app.function(image=image, cpu=(4.0, 4.0), memory=int(8 * 1024),
              timeout=60 * 60 * 10, max_containers=16,
              volumes={str(ARTIFACT_ROOT): artifact_volume})
def run_gate(task: dict[str, Any]) -> dict[str, Any]:
    """One cell, one beta. Staged particle SMC with state reuse. NO DOCKING."""
    import numpy as np
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")

    x0 = task["smiles"]; delta = float(task["delta"]); beta = float(task["beta"])
    tag = task["tag"]; nstage = int(task.get("stages", N_STAGES))
    slen = int(task.get("stage_len", STAGE_LEN))
    npart = int(task.get("n_part", N_PART))
    rng = np.random.default_rng(int(task["seed_rng"]))
    from modal_apps.genmol_t4_opt_app import _dock_many
    from rdkit.Chem import AllChem, QED as _QED, Descriptors as _Desc, RDConfig as _RC
    from rdkit.Chem import rdMolDescriptors as _rdMD
    import os as _os, sys as _sys
    _sys.path.append(_os.path.join(_RC.RDContribDir, "SA_Score"))
    import sascorer as _sas

    def _pcprops(smi):
        m = Chem.MolFromSmiles(smi)
        return {"logP": float(_Desc.MolLogP(m)), "tpsa": float(_Desc.TPSA(m)),
                "mw": float(_Desc.MolWt(m)), "rings": float(_rdMD.CalcNumRings(m)),
                "arom": float(_rdMD.CalcNumAromaticRings(m)),
                "hbd": float(_rdMD.CalcNumHBD(m)), "hba": float(_rdMD.CalcNumHBA(m)),
                "rotb": float(_rdMD.CalcNumRotatableBonds(m)),
                "fsp3": float(_rdMD.CalcFractionCSP3(m)),
                "heavy": float(m.GetNumHeavyAtoms()),
                "qed": float(_QED.qed(m)), "sa": float(_sas.calculateScore(m))}
    ladder = [int(x) for x in (task.get("ladder") or LADDER)]
    w_dock = []
    docked: dict[str, float] = {}
    dock_budget = int(task.get("dock_budget") or 0)
    w_by_d = {str(d): [0.0] * 7 for d in ladder}

    banked = {d: [x0] for d in ladder}
    n_resample = [0]
    per_depth, frontier, best_overall = [], {}, {"v": 9e9, "smi": None, "depth": 0}
    t_start = time.time()
    feas_any = 0
    for stage in range(nstage):
        # state reuse: stage s starts from the states stage s-1 banked
        # particles spread across the WHOLE ladder simultaneously
        cur = []
        for i in range(npart):
            d_ = ladder[i % len(ladder)]
            pool = banked.get(d_) or banked.get(0) or [x0]
            # Round-robin over the CROSS PRODUCT. The previous strides
            # (i // len(ladder), i // (len(ladder)*len(SEGMENTS))) meant that
            # with 40 particles only `local` and `grow` existed at all, and with
            # 64 `construct_ring` got 4 particles and `mixed` got none -- so
            # every "construct_ring produced no rings" reading was measuring a
            # mode that barely existed rather than a mode that failed.
            # EXPLICIT Cartesian cell: rung x segment-length x mode. No clever
            # indexing -- every intended combination gets particles by
            # construction, which the previous strides did not.
            _cell = i % (len(ladder) * len(SEGMENTS) * len(MODES))
            d_ = ladder[_cell % len(ladder)]
            seg_ = SEGMENTS[(_cell // len(ladder)) % len(SEGMENTS)]
            mode_ = MODES[(_cell // (len(ladder) * len(SEGMENTS))) % len(MODES)]
            # AN OPTION'S SEGMENT MUST BE AT LEAST ITS PROGRAM LENGTH.
            # `rebuild` is open -> grow... -> reclose, so it needs >= 3-4 edits;
            # at the measured realized length of 2.3 it executed the open and a
            # grow and NEVER reached the close. The population then sat in the
            # infeasible region it had opened into (sim 0.19, 1 of 52 feasible)
            # because the only move that returns to feasibility is the reclose.
            if mode_ in ("rebuild", "construct_ring") and seg_ < 4:
                seg_ = 4 if (_cell % 2 == 0) else 8
            cur.append({"smi": pool[i % len(pool)],
                        "w": 0.0, "d": d_, "seg": seg_, "mode": mode_,
                        "edits": 0,
                        "lin": f"s{stage}d{d_}g{seg_}{mode_}p{i}", "hist": []})
        # Cap the number of ADVANCE ROUNDS rather than the segment ladder. The
        # loop previously ran stage_len times because it waited for k=1
        # particles to spend their whole budget one edit at a time, which set
        # the iteration count -- and therefore the wall clock -- for everyone.
        _rounds = int(task.get("rounds") or slen)
        for d in range(min(slen, _rounds)):
            b = slen - d - 1
            # equal EDIT budget per stage across segment strata: a particle
            # stops advancing once it has spent stage_len edits, so a seg=12
            # stratum does not simply get 12x the depth of a seg=1 stratum.
            specs = [{"smi": c["smi"], "b": b, "idx": i, "d": c["d"],
                      "seg": c.get("seg", 1), "mode": c.get("mode", "mixed"),
                      "seed": int(task["seed_rng"]) * 7919 + stage * 131
                              + d * 17 + i}
                     for i, c in enumerate(cur)
                     if c["smi"] and c.get("edits", 0) < slen]
            if not specs:
                break
            chunks = [{"specs": specs[i::FANOUT], "delta": delta, "x0": x0,
                       "beta": beta, "w_by_d": w_by_d,
                       "w_dock": list(w_dock),
                       "gamma_rule": float(task.get("gamma_rule", GAMMA_RULE)),
                       "beta_dock": float(task.get("beta_dock") or 0.0)}
                      for i in range(FANOUT) if specs[i::FANOUT]]
            got, nerr = [], 0
            for res in advance.map(chunks, order_outputs=False,
                                   return_exceptions=True,
                                   wrap_returned_exceptions=False):
                if not isinstance(res, list):
                    nerr += 1
                    if nerr <= 2:
                        print(f"  !! advance failed: {type(res).__name__}: "
                              f"{str(res)[:200]}", flush=True)
                    continue
                got.extend(res)
            if nerr:
                print(f"  !! {nerr}/{len(chunks)} advance chunks failed", flush=True)
            by = {g["idx"]: g for g in got}
            seg_deltas = [(int(g.get("seg_dheavy", 0)), g.get("mode", "mixed"))
                          for g in got if g.get("smi")]
            seg_rings = [(int(g.get("seg_drings", 0)), g.get("mode", "mixed"))
                         for g in got if g.get("smi")]
            seg_real = [(int(g.get("seg_used", 0)), g.get("mode", "mixed"))
                        for g in got if g.get("smi")]
            n_advanced = len(seg_deltas)
            _asked = collections.Counter(int(g.get("seg_asked", 1))
                                         for g in got if g.get("smi"))
            _stop = collections.Counter(int(g.get("stopped_at", -1))
                                        for g in got if g.get("smi"))
            for i, c in enumerate(cur):
                if i not in by and c["smi"] and c.get("edits", 0) >= slen:
                    by[i] = {"idx": i, "smi": c["smi"], "logw": 0.0,
                             "seg_used": 0, "pr": None}
            nxt, logw, stats = [], [], []
            for i, c in enumerate(cur):
                g = by.get(i)
                if not g or not g.get("smi"):
                    # segment rejected at the boundary: the particle survives at
                    # its current state and may try a different option next round
                    if c["smi"]:
                        nxt.append(dict(c)); logw.append(c["w"])
                    continue
                if g.get("pr") is None:      # exhausted particle, carried as-is
                    nxt.append(dict(c)); logw.append(c["w"]); continue
                nxt.append({"smi": g["smi"], "w": c["w"] + g["logw"],
                            "lin": c["lin"], "d": c["d"],
                            "seg": c.get("seg", 1), "mode": c.get("mode", "mixed"),
                            "edits": c.get("edits", 0) + int(g.get("seg_used", 1)),
                            "hist": c["hist"] + [(list(_feat(g["pr"], b)),
                                                  float(g["pr"]["v"]),
                                                  int(g["pr"]["dheavy"]))]})
                logw.append(c["w"] + g["logw"]); stats.append(g["pr"])
                p = g["pr"]
                if p["v"] < best_overall["v"]:
                    best_overall = {"v": float(p["v"]), "smi": g["smi"],
                                    "depth": stage * slen + d + 1}
                if p["v"] == 0.0:
                    feas_any += 1
                if p["v"] == 0.0 or len(frontier) < 8000:
                    # Delta(x) = (edit depth, |dheavy|, |drings|, 1-sim).
                    # Kept as a VECTOR: a hand-weighted scalar would re-introduce
                    # a chosen notion of "far", and "explore" must not collapse
                    # into "keep adding carbon".
                    frontier[g["smi"]] = {
                        "v": float(p["v"]), "d": int(c["d"]),
                        "depth": stage * slen + d + 1,
                        "dh": int(p["dheavy"]), "dr": int(p["drings"]),
                        "dist": 1.0 - float(p["sim"]),
                        # OPTION PROVENANCE. Without this a docked score cannot
                        # be attributed to the option that produced it, which is
                        # the whole causal question: did grow / construct_ring
                        # generate the good binders, or did they arrive anyway?
                        "mode": c.get("mode", "mixed"),
                        "seg_asked": int(c.get("seg", 1)),
                        "seg_realized": int(g.get("seg_used", 0)),
                        "lin": c.get("lin", "")}
            if not nxt:
                break
            # PROTECTED OPTION STRATA. Resampling globally (or by displacement
            # rung) let the construct_ring option be wiped out inside a single
            # stage -- which is the delayed-credit pathology again, now at the
            # OPTION level: a ring-building program looks worthless until its
            # precursor chain exists. So resample WITHIN option, and hold a
            # minimum quota per option during exploration so no option can be
            # eliminated before it has completed a segment. The mixture is then
            # allowed to move only on evidence the option actually had a chance
            # to produce.
            from compose_v4.inference.twisted_smc import (
                ess as _ess, systematic_resample as _sysres)
            # census BEFORE resampling, by (mode, segment length)
            _pre = collections.Counter((c2.get("mode"), c2.get("seg"))
                                       for c2 in nxt)
            # Resample within (option, segment length). Protecting only the
            # option let long segments be eliminated: a longer segment drifts
            # further, so its endpoint scores worse under the boundary potential
            # and resampling wipes it out -- the same delayed-credit pathology
            # one level down. Measured: realized length collapsed to 1.0 across
            # every mode, i.e. only seg=1 particles survived.
            groups = {(m_, s_): [i for i, c2 in enumerate(nxt)
                                 if c2.get("mode") == m_ and c2.get("seg") == s_]
                      for m_ in MODES for s_ in SEGMENTS}
            groups = {m_: ii for m_, ii in groups.items() if ii}
            ess = float(np.mean([_ess(np.array([nxt[i]["w"] for i in ii]))
                                 for ii in groups.values()])) if groups else 0.0
            _lw = np.array([c2["w"] for c2 in nxt])
            if _lw.size:
                _md = float(np.median(_lw))
                for c2, v2 in zip(nxt, np.clip(_lw, _md - 4.0, _md + 4.0)):
                    c2["w"] = float(v2)
            if groups and ess < ESS_FRAC * (len(nxt) / max(1, len(groups))):
                idx = []
                for m_, ii in groups.items():
                    sub = _sysres(np.array([nxt[i]["w"] for i in ii]), rng.random())
                    idx.extend([ii[int(j)] for j in sub])
                idx = np.array(idx, dtype=int) if idx else np.arange(len(nxt))
                n_resample[0] += 1
                nxt = [{"smi": nxt[int(j)]["smi"], "w": 0.0,
                        "lin": nxt[int(j)]["lin"], "d": nxt[int(j)]["d"],
                        "seg": nxt[int(j)].get("seg", 1),
                        "mode": nxt[int(j)].get("mode", "mixed"),
                        "edits": nxt[int(j)].get("edits", 0),
                        "hist": list(nxt[int(j)]["hist"])} for j in idx]
            # quota top-up: revive any option that has fallen below MIN_QUOTA by
            # re-seeding it from the archive, so an option is never extinct.
            have = {(m_, s_): sum(1 for c2 in nxt if c2.get("mode") == m_
                                  and c2.get("seg") == s_)
                    for m_ in MODES for s_ in SEGMENTS}
            pool_ = [s_ for s_, m_ in sorted(frontier.items(),
                                             key=lambda kv: kv[1]["v"])[:64]] or [x0]
            for (m_, s_) in have:
                need_ = MIN_QUOTA - have[(m_, s_)]
                for q in range(max(0, need_)):
                    nxt.append({"smi": pool_[q % len(pool_)], "w": 0.0,
                                "lin": f"revive{stage}_{m_}_{s_}_{q}", "d": 0,
                                "seg": s_, "mode": m_, "edits": 0, "hist": []})
            _post = collections.Counter((c2.get("mode"), c2.get("seg"))
                                        for c2 in nxt)
            _long = {m_: (sum(v for (mm, ss), v in _pre.items()
                              if mm == m_ and ss >= 4),
                          sum(v for (mm, ss), v in _post.items()
                              if mm == m_ and ss >= 4)) for m_ in MODES}
            print(f"    (mode,k>=4) before->after resample: "
                  + "  ".join(f"{m_}:{a}->{b}" for m_, (a, b) in _long.items()),
                  flush=True)
            cur = nxt
            if not stats:
                # the returnability trigger can end every segment in a round
                # early. Emit a minimal row rather than dropping it: skipping
                # left per_depth empty and made the run look untelemetried.
                per_depth.append({"stage": stage, "depth": stage * slen + d + 1,
                                  "n": 0, "note": "no particle advanced"})
                cur = nxt
                continue
            rec = {"stage": stage, "depth": stage * slen + d + 1,
                   "n": len(stats), "ess": ess,
                   "min_v": float(min(s["v"] for s in stats)),
                   "n_feas": int(sum(1 for s in stats if s["v"] == 0.0)),
                   "mean_dheavy": float(np.mean([s["dheavy"] for s in stats])),
                   "max_dheavy": int(max(s["dheavy"] for s in stats)),
                   "mean_drings": float(np.mean([s["drings"] for s in stats])),
                   "max_drings": int(max(s["drings"] for s in stats)),
                   "mean_sim": float(np.mean([s["sim"] for s in stats])),
                   "mean_qed": float(np.mean([s["qed"] for s in stats])),
                   "mean_sa": float(np.mean([s["sa"] for s in stats])),
                   # collapse guard: terminal feasibility bought by driving 48
                   # particles into one lineage is a hidden failure, not a win.
                   "n_lineage": len({c["lin"] for c in cur}),
                   "n_unique_smi": len({c["smi"] for c in cur}),
                   "n_resample": int(n_resample[0]),
                   "seg_mix": {str(s_): sum(1 for c2 in cur
                                            if c2.get("seg") == s_)
                               for s_ in SEGMENTS},
                   # which OPTIONS survive resampling is the learned mixture
                   "mode_mix": {m_: sum(1 for c2 in cur
                                        if c2.get("mode") == m_) for m_ in MODES},
                   # ZERO-ORACLE MECHANISM GATE. Per option: net structural
                   # progress and the insert<->delete cancellation that the route
                   # audit measured as the 5ht1b failure signature. An option
                   # that grows without cancelling is crossing the plateau.
                   # PER-SEGMENT contribution of each option this round
                   "mode_dheavy": {m_: round(float(np.mean(
                       [sd for sd, mm in seg_deltas if mm == m_] or [0.0])), 2)
                       for m_ in MODES},
                   "mode_drings": {m_: round(float(np.mean(
                       [sr for sr, mm in seg_rings if mm == m_] or [0.0])), 2)
                       for m_ in MODES},
                   "mode_realized": {m_: round(float(np.mean(
                       [rl for rl, mm in seg_real if mm == m_] or [0.0])), 1)
                       for m_ in MODES},
                   "mode_feas": {m_: int(sum(
                       1 for s, c2 in zip(stats, cur)
                       if c2.get("mode") == m_ and s["v"] == 0.0)) for m_ in MODES}}
            per_depth.append(rec)
            print(f"  [{task['target']}/b{beta:g}] st{stage} d{rec['depth']:2d} "
                  f"n{rec['n']:3d} ess{ess:5.1f} min_v {rec['min_v']:.4f} "
                  f"feas {rec['n_feas']:3d} dheavy {rec['mean_dheavy']:+5.1f}"
                  f"/{rec['max_dheavy']:+3d} drings {rec['mean_drings']:+4.1f} "
                  f"sim {rec['mean_sim']:.3f} SEGdh {rec['mode_dheavy']} "
                  f"SEGring {rec['mode_drings']} len {rec['mode_realized']} "
                  f"asked {dict(_asked)} stop@ {dict(_stop)} "
                  f"lin {rec['n_lineage']:2d} "
                  f"uniq {rec['n_unique_smi']:2d} | {time.time()-t_start:.0f}s",
                  flush=True)
        # Bank per stratum, Pareto-nondominated over the cheap displacement
        # vector among FEASIBLE states, so each rung keeps a diverse frontier
        # rather than one "farthest" molecule. Selection here is an adaptive
        # restart policy, NOT part of the FK target -- see the amendment.
        banked = {}
        for d_ in ladder:
            # Bank BOTH: feasible states (returnable) AND displaced infeasible
            # ones (bridge continuation). Preferring feasible-only here is what
            # restarted every stage before the bridge.
            _feas = [(s_, m_) for s_, m_ in frontier.items()
                     if m_["d"] == d_ and m_["v"] == 0.0]
            _bridge = [(s_, m_) for s_, m_ in frontier.items()
                       if m_["d"] == d_ and m_["v"] > 0.0 and abs(m_["dh"]) >= 2]
            _bridge.sort(key=lambda kv: -abs(kv[1]["dh"]))
            pool = (_feas + _bridge[:max(8, len(_feas))]) or \
                   [(s_, m_) for s_, m_ in frontier.items() if m_["d"] == d_]
            keys = [(m_["depth"], abs(m_["dh"]), abs(m_["dr"]), m_["dist"])
                    for _, m_ in pool]
            front = []
            for i_, k_ in enumerate(keys):
                if not any(all(kj >= ki for kj, ki in zip(keys[j_], k_))
                           and keys[j_] != k_ for j_ in range(len(keys))):
                    front.append(i_)
            if not front:
                front = list(range(len(pool)))
            front.sort(key=lambda i_: -(keys[i_][1] + keys[i_][2] + keys[i_][3]))
            banked[d_] = [pool[i_][0] for i_ in front[:FRONTIER]] or [x0]
        _reach = {d_: sum(1 for m_ in frontier.values()
                          if m_["v"] == 0.0 and abs(m_["dh"]) >= d_)
                  for d_ in ladder}
        print(f"  [{task['target']}/b{beta:g}] STAGE {stage} ladder-reach "
              f"(feasible AND |dheavy|>=d): "
              + "  ".join(f"d{d_}:{_reach[d_]}" for d_ in ladder), flush=True)

        # Fitted value iteration on the stage's own rollouts. label(y,b)=1 iff
        # the continuation from y reached feasibility within its remaining
        # budget -- credit flows BACKWARD from outcomes, which is what makes a
        # bridge state defensible.
        # ---- COUNTED oracle calls: dock part of the frontier, fit S_dock ----
        if dock_budget and len(docked) < dock_budget:
            import numpy as _np
            pool_ = [(s_, m_) for s_, m_ in frontier.items()
                     if m_["v"] == 0.0 and s_ not in docked]
            per_stage = max(1, dock_budget // max(1, nstage))
            n_take = min(per_stage, dock_budget - len(docked))
            if w_dock and pool_:
                # ACQUISITION RANKED BY S_dock. Previously the surrogate was
                # fitted and then ignored: candidates were docked most-displaced
                # first, so the oracle budget was spent by a rule that knew
                # nothing about binding. That is why the S_dock ablation came out
                # null -- the model never touched the decision it was fitted for.
                # Two thirds exploit the surrogate, one third keeps the most
                # displaced states so the frontier is not narrowed to whatever
                # the early surrogate already likes.
                Xp = _np.array([_dfeat(_pcprops(s_)) for s_, _ in pool_],
                               dtype=_np.float64)
                pred = Xp @ _np.array(w_dock)
                order = _np.argsort(-pred)
                n_ex = int(n_take * 2 / 3)
                take = [pool_[int(i)][0] for i in order[:n_ex]]
                rest = sorted([pm for pm in pool_ if pm[0] not in set(take)],
                              key=lambda kv: -abs(kv[1]["dh"]))
                take += [s_ for s_, _ in rest[:n_take - len(take)]]
            else:
                cand = [s_ for s_, m_ in sorted(pool_,
                                                key=lambda kv: -abs(kv[1]["dh"]))]
                take = cand[:n_take]
            if take:
                ds_ = _dock_many(take, task["target"],
                                 f"sd_{tag}_{task['idx']}_{stage}",
                                 workers=4, cpu_per_dock=1)
                for s_, v_ in zip(take, ds_):
                    if v_ is not None and v_ != 0.0:
                        docked[s_] = float(v_)
                if len(docked) >= 8:
                    import numpy as _np
                    Xd = _np.array([_dfeat(_pcprops(s_)) for s_ in docked],
                                   dtype=_np.float64)
                    yd = _np.array([-docked[s_] for s_ in docked])   # higher=better
                    yd = (yd - yd.mean()) / (yd.std() + 1e-9)
                    # ridge, closed form; 2048 features on tens of rows so the
                    # penalty is doing real work and is not optional
                    A = Xd.T @ Xd + 1.0 * _np.eye(Xd.shape[1])
                    w_dock = list(_np.linalg.solve(A, Xd.T @ yd))
                    best_ = min(docked.values())
                    print(f"  [{task['target']}/b{beta:g}] S_dock fitted on "
                          f"{len(docked)} COUNTED calls, best {best_:.1f}",
                          flush=True)
        rows_by_d = {d_: [] for d_ in ladder}
        for c in cur:
            h_ = c.get("hist") or []
            d_ = c["d"]
            for ti in range(len(h_)):
                # label(y,b) = 1 iff the continuation from y reached
                # (feasible AND |dheavy| >= d) within its remaining budget.
                reached = 1.0 if any(v_ == 0.0 and abs(dh_) >= d_
                                     for _, v_, dh_ in h_[ti:]) else 0.0
                rows_by_d[d_].append((h_[ti][0], reached))
        _fit = []
        for d_ in ladder:
            rr = rows_by_d[d_]
            if not rr:
                continue
            w_new = fit_h(rr)
            if any(w_new):
                w_by_d[str(d_)] = w_new
            _fit.append(f"d{d_}:{sum(r[1] for r in rr):.0f}/{len(rr)}")
        if _fit:
            print(f"  [{task['target']}/b{beta:g}] fitted h per stratum "
                  f"(reached/rows, ZERO oracle calls): " + "  ".join(_fit),
                  flush=True)

    res = {"target": task["target"], "idx": task["idx"], "delta": delta,
           "beta": beta, "tag": tag, "seed": x0,
           "per_depth": per_depth, "best": best_overall,
           "n_feasible_states": feas_any,
           "ladder": ladder,
           # the headline gate number: feasible AND displaced by at least d
           "ladder_reach": {str(d_): sum(1 for m_ in frontier.values()
                                         if m_["v"] == 0.0 and abs(m_["dh"]) >= d_)
                            for d_ in ladder},
           "ladder_reach_signed": {str(d_): {
               "grow": sum(1 for m_ in frontier.values()
                           if m_["v"] == 0.0 and m_["dh"] >= d_),
               "shrink": sum(1 for m_ in frontier.values()
                             if m_["v"] == 0.0 and -m_["dh"] >= d_)}
               for d_ in ladder},
           "frontier_top": [[s, m] for s, m in
                            sorted(frontier.items(),
                                   key=lambda kv: -(abs(kv[1]["dh"])
                                                    + abs(kv[1]["dr"])))[:96]],
           "banked_last": {str(k): v[:16] for k, v in banked.items()},
           "w_by_d": w_by_d,
           "docked": docked,
           # per-docked-molecule provenance, joined from the frontier
           "docked_provenance": {s_: {k2: v2 for k2, v2 in
                                      (frontier.get(s_) or {}).items()}
                                 for s_ in docked},
           "n_oracle_calls": len(docked),
           "best_docked": (min(docked.values()) if docked else None),
           "best_docked_smi": (min(docked, key=docked.get) if docked else None),
           "wall_s": time.time() - t_start}
    try:
        d = ARTIFACT_ROOT / "t4_particle"
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{tag}_{task['idx']}_{task['target']}_d{delta}_b{beta:g}.json"
         ).write_text(json.dumps(res))
        artifact_volume.commit()
    except Exception as e:
        print(f"  !! save failed: {type(e).__name__}: {e}", flush=True)
    return res


@app.function(image=image, cpu=(1.0, 1.0), memory=2048,
              timeout=60 * 60 * 12,
              volumes={str(ARTIFACT_ROOT): artifact_volume})
def drive(tasks: list[dict]) -> list[dict]:
    """SERVER-SIDE driver.

    `modal run --detach` on a LOCAL entrypoint keeps only the last triggered
    function alive once the parent disconnects -- Modal warns about this at
    launch, and it is what killed the reachability audit when the connection
    dropped. Driving the fan-out from inside Modal makes the local process
    irrelevant, so the job survives the laptop closing.
    """
    got = []
    for r in run_gate.map(tasks, order_outputs=False, return_exceptions=True,
                          wrap_returned_exceptions=False):
        if isinstance(r, dict):
            got.append(r)
            print(f"  done {r['target']} s{r['idx']} d{r['delta']} "
                  f"best_docked={r.get('best_docked')} "
                  f"calls={r.get('n_oracle_calls')}", flush=True)
        else:
            print(f"  !! {type(r).__name__}: {str(r)[:200]}", flush=True)
    return got


@app.local_entrypoint()
def main(cells: str = "0,7,10,11", deltas: str = "0.4,0.6",
         betas: str = "0,4", tag: str = "gate", stages: int = 3,
         stage_len: int = 8, n_part: int = 48, out: str = "",
         dock_budget: int = 0, beta_dock: float = 0.0,
         gamma_rule: float = GAMMA_RULE, rounds: int = 0):
    seeds = json.loads((ROOT / "docs/GENMOL_T4_SEEDS.json").read_text())
    want = [int(c) for c in cells.split(",") if c.strip() != ""]
    dl = [float(x) for x in deltas.split(",")]
    bl = [float(x) for x in betas.split(",")]
    tasks = []
    for i in want:
        s = seeds[i] if isinstance(seeds, list) else seeds[str(i)]
        smi = s["smiles"] if isinstance(s, dict) else s
        tgt = s.get("target") if isinstance(s, dict) else None
        for dd in dl:
            for bb in bl:
                tasks.append({"rounds": rounds, "gamma_rule": gamma_rule,
                              "dock_budget": dock_budget,
                              "beta_dock": beta_dock,
                              "smiles": smi, "target": tgt or f"cell{i}",
                              "idx": i, "delta": dd, "beta": bb, "tag": tag,
                              "stages": stages, "stage_len": stage_len,
                              "n_part": n_part,
                              "seed_rng": 1000 + i * 31 + int(dd * 10)})
    print(f"  {len(tasks)} gate runs = {len(want)} cells x {len(dl)} deltas "
          f"x {len(bl)} betas, {n_part} particles, "
          f"{stages}x{stage_len}={stages*stage_len} edits, "
          f"dock_budget={dock_budget} beta_dock={beta_dock}",
          flush=True)
    # spawn() queues the driver server-side and returns immediately, so the
    # local process can exit (or the laptop can close) without touching the job.
    h = drive.spawn(tasks)
    print(f"  SPAWNED server-side driver, call id {h.object_id}", flush=True)
    print(f"  results land on the volume under t4_particle/{tag}_*.json", flush=True)
    print(f"  safe to close the laptop once the app shows a running task", flush=True)


@app.function(image=image, cpu=(2.0, 2.0), memory=int(6 * 1024),
              timeout=60 * 30, volumes={str(ARTIFACT_ROOT): artifact_volume})
def probe(smi: str) -> dict:
    """What does ONE step actually offer? Decides whether deletion is reachable
    at all, or merely improbable, before any control question is asked."""
    import os, sys, collections
    import numpy as np
    os.chdir("/tmp"); sys.path.insert(0, str(REMOTE_ROOT / "src"))
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")
    from compose_v4.chem.molecular_graph import smiles_to_molecular_graph
    from compose_v4.chem.state import pad_molecular_graph
    from compose_v4.experiments.production_successor_kernel import (
        canonical_state_key, enumerate_factorized_marked_law)
    rt = _runtime(); model, system = rt["model"], rt["system"]
    h0 = Chem.MolFromSmiles(smi).GetNumHeavyAtoms()
    st = pad_molecular_graph(smiles_to_molecular_graph(smi), CANONICAL_SLOTS)
    law = enumerate_factorized_marked_law(model, st, float(TIME_POINT))
    pr = np.array([m.probability for m in law.marks], float)
    order = np.argsort(-pr)
    rule_mass = collections.Counter(); rule_n = collections.Counter()
    ring_mass = collections.Counter(); ring_n = collections.Counter()
    ring_rules = collections.Counter(); ring_ranks = []
    dh_mass = collections.Counter(); dh_n = collections.Counter()
    first_del_rank = None
    for rank, j in enumerate(order):          # FULL law, not just top-APPLY_CAP
        mk = law.marks[int(j)]
        rule_mass[mk.executor_rule_name] += float(pr[j])
        rule_n[mk.executor_rule_name] += 1
        try:
            y = canonical_state_key(system.apply(st, mk.executor_rule_name,
                                                 mk.action))
            m2 = Chem.MolFromSmiles(y) if y else None
            if m2 is None:
                continue
            d = m2.GetNumHeavyAtoms() - h0
        except Exception:
            continue
        key = "shrink" if d < 0 else ("grow" if d > 0 else "same")
        dh_mass[key] += float(pr[j]); dh_n[key] += 1
        try:
            dr = len(Chem.GetSymmSSSR(m2)) - len(Chem.GetSymmSSSR(Chem.MolFromSmiles(smi)))
        except Exception:
            dr = 0
        rk = "ring+" if dr > 0 else ("ring-" if dr < 0 else "ring0")
        ring_mass[rk] += float(pr[j]); ring_n[rk] += 1
        if dr > 0:
            ring_rules[mk.executor_rule_name] += 1
            ring_ranks.append(rank)
        if d < 0 and first_del_rank is None:
            first_del_rank = rank
    tot = float(pr.sum())
    return {"smi": smi, "heavy": h0, "n_marks": len(law.marks),
            "total_mass_in_cap": tot,
            "by_rule_mass": {k: round(v / tot, 5) for k, v in rule_mass.most_common()},
            "by_rule_n": dict(rule_n.most_common()),
            "dheavy_mass": {k: round(v / tot, 5) for k, v in dh_mass.items()},
            "dheavy_n": dict(dh_n),
            "ring_mass": {k: round(v / tot, 5) for k, v in ring_mass.items()},
            "ring_n": dict(ring_n),
            "ring_forming_rules": dict(ring_rules),
            "ring_ranks": sorted(ring_ranks)[:12],
            "n_ring_forming": len(ring_ranks),
            "apply_cap": APPLY_CAP,
            "first_shrink_rank": first_del_rank}


@app.local_entrypoint()
def probe_seeds(cells: str = "10,0"):
    seeds = json.loads((ROOT / "docs/GENMOL_T4_SEEDS.json").read_text())
    for i in [int(c) for c in cells.split(",")]:
        s = seeds[i]
        r = probe.remote(s["smiles"])
        print(f"\n=== cell {i} {s.get('target')}  heavy={r['heavy']}  "
              f"marks={r['n_marks']} ===")
        print(f"  mass by dheavy : {r['dheavy_mass']}")
        print(f"  count by dheavy: {r['dheavy_n']}")
        print(f"  first shrink at rank: {r['first_shrink_rank']}")
        print(f"  RING mass      : {r['ring_mass']}")
        print(f"  RING count     : {r['ring_n']}")
        print(f"  ring-FORMING rules: {r['ring_forming_rules']}")
        print(f"  ring-forming marks: {r['n_ring_forming']} of {r['n_marks']}; "
              f"ranks {r['ring_ranks']} (APPLY_CAP={r['apply_cap']})")
        print(f"  mass by rule   : {r['by_rule_mass']}")
        print(f"  count by rule  : {r['by_rule_n']}")


@app.function(image=image, cpu=(2.0, 2.0), memory=int(6 * 1024),
              timeout=60 * 60, volumes={str(ARTIFACT_ROOT): artifact_volume})
def probe_time(smi: str, times: list[float]) -> list[dict]:
    """Is ring formation intrinsically low-mass, or low-mass AT THIS TIME POINT?

    R_theta(y|x) is evaluated at a single TIME_POINT. If ring closure carries
    real mass at some other point on the process, then the operator is not
    disfavoured by the model -- we are simply sampling the process where ring
    closure does not happen, which is a very different finding.
    """
    import os, sys, collections
    import numpy as np
    os.chdir("/tmp"); sys.path.insert(0, str(REMOTE_ROOT / "src"))
    from rdkit import Chem, RDLogger
    from rdkit.Chem import QED, RDConfig
    RDLogger.DisableLog("rdApp.*")
    sys.path.append(os.path.join(RDConfig.RDContribDir, "SA_Score"))
    import sascorer
    from compose_v4.chem.molecular_graph import smiles_to_molecular_graph
    from compose_v4.chem.state import pad_molecular_graph
    from compose_v4.experiments.production_successor_kernel import (
        canonical_state_key, enumerate_factorized_marked_law)
    rt = _runtime(); model, system = rt["model"], rt["system"]
    st = pad_molecular_graph(smiles_to_molecular_graph(smi), CANONICAL_SLOTS)
    m0 = Chem.MolFromSmiles(smi); r0 = len(Chem.GetSymmSSSR(m0))
    out = []
    for tp in times:
        law = enumerate_factorized_marked_law(model, st, float(tp))
        pr = np.array([m.probability for m in law.marks], float)
        order = np.argsort(-pr)
        rmass = 0.0; rn = 0; first = None; quality = []
        by_rule = collections.Counter()
        for rank, j in enumerate(order):
            mk = law.marks[int(j)]
            by_rule[mk.executor_rule_name] += float(pr[j])
            try:
                y = canonical_state_key(system.apply(st, mk.executor_rule_name,
                                                     mk.action))
                m2 = Chem.MolFromSmiles(y) if y else None
                if m2 is None:
                    continue
                if len(Chem.GetSymmSSSR(m2)) > r0:
                    rmass += float(pr[j]); rn += 1
                    if first is None:
                        first = rank
                    if len(quality) < 12:
                        quality.append((round(float(QED.qed(m2)), 3),
                                        round(float(sascorer.calculateScore(m2)), 2)))
            except Exception:
                continue
        tot = float(pr.sum())
        out.append({"t": tp, "ring_mass": rmass / max(tot, 1e-30), "ring_n": rn,
                    "first_rank": first, "n_marks": len(law.marks),
                    "ring_quality_qed_sa": quality,
                    "top_rules": {k: round(v / max(tot, 1e-30), 4)
                                  for k, v in by_rule.most_common(5)}})
    return out


@app.local_entrypoint()
def timescan(cell: int = 0, times: str = "0.1,0.25,0.5,0.75,0.9"):
    seeds = json.loads((ROOT / "docs/GENMOL_T4_SEEDS.json").read_text())
    s = seeds[cell]
    ts = [float(x) for x in times.split(",")]
    print(f"=== cell {cell} {s['target']} : ring-forming mass vs process time ===")
    for r in probe_time.remote(s["smiles"], ts):
        print(f"  t={r['t']:<5} ring_mass={r['ring_mass']:.2e}  n_ring={r['ring_n']:>3}  "
              f"first_rank={r['first_rank']}  marks={r['n_marks']}")
        print(f"        top rules: {r['top_rules']}")
        if r["ring_quality_qed_sa"]:
            print(f"        ring-product (QED,SA): {r['ring_quality_qed_sa'][:6]}")


@app.local_entrypoint()
def probe_pmo(n: int = 3):
    """Is the operator-marginal bias a property of R_theta, or of the T4 seeds?

    Same probe on PMO starting molecules. If ring closure is under-ranked by a
    comparable factor here, the bias belongs to the model and the debiasing
    correction is task-independent -- a materially stronger claim than a T4 fix.
    """
    bank = json.loads((ROOT / "docs/PMO_INIT_BANK.json").read_text())
    for smi in bank["smiles"][:n]:
        r = probe.remote(smi)
        rm = r["ring_mass"].get("ring+", 0.0)
        ins = r["by_rule_mass"].get("atom_insert", 0.0)
        nr = r.get("n_ring_forming", 0)
        print(f"\n  {smi[:52]}  heavy={r['heavy']} marks={r['n_marks']}")
        print(f"    ring-forming n={nr}  mass={rm:.3e}  first_rank={r.get('ring_ranks')[:1]}"
              f"  (APPLY_CAP={r.get('apply_cap')})")
        print(f"    atom_insert mass={ins:.4f} over n={r['by_rule_n'].get('atom_insert',0)}")
        if nr and rm > 0:
            per_ring = rm / nr
            per_ins = ins / max(r["by_rule_n"].get("atom_insert", 1), 1)
            print(f"    UNDER-WEIGHTING per move: {per_ins / per_ring:,.0f}x")
        print(f"    rules: { {k: round(v,4) for k,v in r['by_rule_mass'].items()} }")


@app.local_entrypoint()
def probe_smiles(smi: str = ""):
    """Run the operator-bias probe on an arbitrary molecule.

    Used to test whether the ring under-ranking is a property of R_theta itself
    or an artefact of the T4 seeds. If PMO starting molecules show a comparable
    under-weighting, the debiasing correction is task-independent.
    """
    if not smi:
        smi = json.loads((ROOT / "docs/PMO_INIT_BANK.json").read_text())["smiles"][0]
    r = probe.remote(smi)
    rm = r["ring_mass"].get("ring+", 0.0)
    nr = r.get("n_ring_forming", 0)
    ins = r["by_rule_mass"].get("atom_insert", 0.0)
    n_ins = r["by_rule_n"].get("atom_insert", 0)
    print(f"\n  {smi}")
    print(f"  heavy={r['heavy']} marks={r['n_marks']} APPLY_CAP={r.get('apply_cap')}")
    print(f"  ring-forming: n={nr} mass={rm:.3e} first_ranks={r.get('ring_ranks')[:4]}")
    print(f"  atom_insert : n={n_ins} mass={ins:.4f}")
    if nr and rm > 0 and n_ins:
        print(f"  UNDER-WEIGHTING per move: {(ins/n_ins)/(rm/nr):,.0f}x")
    print(f"  rules: { {k: round(v,4) for k,v in r['by_rule_mass'].items()} }")
