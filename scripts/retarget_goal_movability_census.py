"""How far can ONE local structural change move each candidate goal property?

Counts only; freezes nothing, selects no goal.

The feasibility census answered "how rare is each goal on the pool". It did not
answer the question that actually decides whether a 4-edit post-switch budget
can reach a goal: **is the property locally movable, or is it a cliff that a
handful of edits cannot climb?**

Real one-cut matched pairs are the right instrument. An MMP pair is, by
construction, one local structural change on a shared core -- the same kind of
move the executor makes -- and it needs no kernel, so this runs locally on the
held-out reserve. If a single matched-pair change almost never moves a property,
four edits will not either, and a goal defined on that property is unreachable
from an arbitrary source regardless of how good the controller is.

The comparison across properties is the point. A property whose MMP deltas are
broad and continuous (QED, cLogP) supports a goal reachable by search. A
property whose deltas are near-zero almost everywhere with a thin tail of large
jumps (an activity cliff) does not: greedy has nothing to follow and lookahead
has nothing to find, which is the shape a planning probe reports as "no signal".

DRD2 movement is reported in LOG-ODDS, not probability. Probability saturates at
both ends, so a probability delta understates movement exactly where the pool
lives (median P(active) is a few thousandths).
"""

from __future__ import annotations

import argparse
import gzip
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

LOGP_BOX = (1.0, 4.0)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reserve", type=Path,
                        default=Path("diagnostics/editing_v2_matched_validation_reserve_ids.json.gz"))
    parser.add_argument("--manifest", type=Path,
                        default=Path("artifacts/oracles/drd2_svm_v1/drd2_oracle_manifest.json"))
    parser.add_argument("--sources", type=int, default=10653)
    parser.add_argument("--seed", type=int, default=20260812)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    from rdkit import Chem, RDLogger
    from rdkit.Chem import Crippen, QED

    from build_analogue_trace_pool import mine_one_cut_pairs
    from compose_v4.drd2_oracle import load_default_oracle

    RDLogger.DisableLog("rdApp.*")
    started = time.perf_counter()

    reserve = json.load(gzip.open(args.reserve, "rt"))
    pool = sorted(reserve["reserve_source_keys"])
    random.Random(args.seed).shuffle(pool)
    pool = pool[: args.sources]
    print(f"mining one-cut pairs over {len(pool):,} held-out molecules "
          f"(seed {args.seed})", flush=True)

    pairs = mine_one_cut_pairs(pool)
    print(f"[{time.perf_counter()-started:5.1f}s] {len(pairs):,} real matched pairs",
          flush=True)
    if not pairs:
        raise SystemExit("no matched pairs mined")

    members = sorted({s for pair in pairs for s in pair[:2]})
    index = {s: i for i, s in enumerate(members)}

    oracle = load_default_oracle(str(args.manifest))
    # Log-odds, not probability: the pool sits where probability is saturated.
    margin = np.asarray(oracle.margin_many(members), dtype=float)
    qed, logp = [], []
    for s in members:
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            qed.append(np.nan); logp.append(np.nan); continue
        try:
            qed.append(QED.qed(mol))
        except Exception:
            qed.append(np.nan)
        logp.append(Crippen.MolLogP(mol))
    qed = np.asarray(qed); logp = np.asarray(logp)
    print(f"[{time.perf_counter()-started:5.1f}s] scored {len(members):,} molecules",
          flush=True)

    ia = np.array([index[p[0]] for p in pairs])
    ib = np.array([index[p[1]] for p in pairs])
    keep = np.isfinite(margin[ia]) & np.isfinite(margin[ib]) & \
        np.isfinite(qed[ia]) & np.isfinite(qed[ib]) & \
        np.isfinite(logp[ia]) & np.isfinite(logp[ib])
    ia, ib = ia[keep], ib[keep]
    print(f"{len(ia):,} pairs with all three properties\n")

    # Unordered pairs: take |delta| for spread, and the achievable gain in the
    # favourable direction, which is what a controller could actually exploit.
    properties = {
        "drd2_logodds": margin,
        "qed": qed,
        "clogp": logp,
    }
    deltas = {}
    print("per-EDIT movability across real matched pairs")
    print(f"{'property':>14} {'|d| p50':>9} {'|d| p90':>9} {'|d| p99':>9} "
          f"{'max':>9} {'frac |d|<0.01':>15}")
    for name, values in properties.items():
        d = np.abs(values[ib] - values[ia])
        deltas[name] = {
            "p50": float(np.percentile(d, 50)), "p90": float(np.percentile(d, 90)),
            "p99": float(np.percentile(d, 99)), "max": float(d.max()),
            "fraction_below_0.01": float((d < 0.01).mean()),
            "fraction_below_0.1": float((d < 0.1).mean()),
        }
        print(f"{name:>14} {deltas[name]['p50']:>9.4f} {deltas[name]['p90']:>9.4f} "
              f"{deltas[name]['p99']:>9.4f} {d.max():>9.4f} "
              f"{deltas[name]['fraction_below_0.01']:>14.1%}")

    # Neighbour degree is reported because it bounds what any per-molecule
    # statistic can mean here. Most reserve molecules have very few matched
    # neighbours, so "median best available gain" is a statement about MMP
    # density, not about the property. Every decisive number below is therefore
    # computed over EDITS, not over molecules.
    degree = np.bincount(np.concatenate([ia, ib]), minlength=len(members))
    degree = degree[degree > 0]
    print(f"\nmatched-neighbour degree: {len(degree):,} molecules with >=1 "
          f"neighbour, median {int(np.percentile(degree, 50))}, "
          f"p90 {int(np.percentile(degree, 90))}, max {int(degree.max())}")

    # Directional gains. Each unordered pair contributes both directions, so the
    # distribution is symmetric by construction and the upper tail is exactly
    # "what a single favourable edit can buy".
    print("\nsingle-edit gain in the favourable direction (per EDIT, degree-independent)")
    print(f"{'property':>14} {'p50>0':>9} {'p75':>9} {'p90':>9} {'p99':>9} {'max':>9}")
    gains = {}
    for name, values in properties.items():
        signed = np.concatenate([values[ib] - values[ia], values[ia] - values[ib]])
        positive = signed[signed > 0]
        gains[name] = {"positive_edits": int(positive.size),
                       "total_directed_edits": int(signed.size)}
        for c in (50, 75, 90, 99):
            gains[name][f"p{c}"] = float(np.percentile(positive, c))
        gains[name]["max"] = float(positive.max())
        print(f"{name:>14} " + " ".join(f"{gains[name][f'p{c}']:>9.4f}"
                                        for c in (50, 75, 90, 99))
              + f" {positive.max():>9.4f}")

    # Translate the DRD2 requirement into the units the goal is stated in, then
    # ask what fraction of real single edits could deliver it.
    need, med = {}, float(np.percentile(margin, 50))
    print("\nDRD2: log-odds a MEDIAN reserve molecule must climb to reach P(active)")
    signed_drd2 = np.concatenate([margin[ib] - margin[ia], margin[ia] - margin[ib]])
    for threshold in (0.3, 0.5, 0.7):
        target = float(np.log(threshold / (1 - threshold)))
        climb = target - med
        per_edit = climb / 4.0
        # Fraction of real single edits big enough to sustain a 4-edit ascent,
        # and fraction big enough to do it in ONE cliff jump.
        sustain = float((signed_drd2 >= per_edit).mean())
        one_jump = float((signed_drd2 >= climb).mean())
        need[str(threshold)] = {
            "target_logodds": target, "median_start_logodds": med,
            "climb_required": climb, "per_edit_required_over_4": per_edit,
            "fraction_of_edits_at_or_above_per_edit_requirement": sustain,
            "fraction_of_edits_achieving_the_whole_climb": one_jump,
        }
        print(f"  P>={threshold}: climb {climb:+.3f} = {per_edit:+.3f}/edit over 4. "
              f"{sustain:.2%} of real edits clear the per-edit bar, "
              f"{one_jump:.2%} clear it in one jump")

    # BEST-OF-N. The instrument sees a median of 1 matched neighbour, but the
    # real successor fiber is ~500 wide, so a controller picks the best of many
    # rather than the only one on offer. This bounds how much that rescues the
    # DRD2 goal: draw N gains from the measured favourable distribution, take
    # the max, repeat for 4 steps, and ask whether the climb is covered.
    # Deliberately optimistic in three ways -- fiber successors are small,
    # highly correlated perturbations rather than independent MMP-scale draws;
    # gains are assumed additive; and every step is assumed to find its best.
    # So "still out of reach" is trustworthy, "reachable" is not a promise.
    rng = np.random.default_rng(args.seed)
    positive_drd2 = signed_drd2[signed_drd2 > 0]
    print("\nbest-of-N per step, if the fiber offered N independent MMP-scale choices")
    print(f"{'N':>6} {'E[best step]':>13} {'4 steps':>9}   covers P>=0.3 / 0.5 / 0.7")
    best_of_n = {}
    for n in (1, 5, 20, 100, 500):
        draws = rng.choice(positive_drd2, size=(4000, n), replace=True).max(axis=1)
        step = float(draws.mean())
        four = 4 * step
        covers = [four >= need[str(t)]["climb_required"] for t in (0.3, 0.5, 0.7)]
        best_of_n[str(n)] = {"expected_best_step": step, "four_steps": four,
                             "covers_0.3": covers[0], "covers_0.5": covers[1],
                             "covers_0.7": covers[2]}
        print(f"{n:>6} {step:>13.3f} {four:>9.3f}   "
              + " / ".join("yes" if c else " no" for c in covers))

    payload = {
        "schema": "compose.retarget.goal_movability_census",
        "best_of_n_per_step": best_of_n,
        "status": "COUNTS_ONLY_FREEZES_NOTHING_SELECTS_NO_GOAL",
        "instrument": ("real one-cut matched pairs on the held-out reserve; one "
                       "local structural change per pair, no kernel required"),
        "caveat": ("MMP neighbours are a PROXY for the executor's 1-edit "
                   "neighbourhood, not the same set. The executor may reach "
                   "moves MMP mining does not, and vice versa. This bounds "
                   "local movability; it does not enumerate the real fiber."),
        "per_edit_bar_caveat": ("The per-edit requirement divides the climb "
                                "evenly over 4 edits and assumes gains add. "
                                "Real ascents are not additive, so clearing the "
                                "bar is necessary, not sufficient."),
        "sources_sampled": len(pool),
        "matched_pairs": int(len(ia)),
        "molecules_scored": len(members),
        "absolute_deltas": deltas,
        "single_edit_gain_favourable_direction": gains,
        "neighbour_degree_median": int(__import__("numpy").percentile(degree, 50)),
        "drd2_climb_required": need,
        "seed": args.seed,
    }
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
