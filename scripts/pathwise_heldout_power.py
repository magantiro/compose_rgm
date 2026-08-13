"""Power the pathwise held-out confirmation panel. DESIGN ONLY -- no run.

Sizes the panel for the INTERSECTION of the two preregistered criteria:

    P1  lower bound of the 95% source-clustered bootstrap CI on p_hidden > 1/3
    P2  lower bound of the 95% source-clustered bootstrap CI on mean dV  > -delta

WHY A SIMULATION AND NOT A CLOSED FORM. Two reasons, both about honesty rather
than taste.

  * `dV` is not remotely normal. Of the 24 Stage B source-level differences, 5
    are EXACTLY zero -- the pathwise and endpoint arms took the identical
    trajectory, because the unconstrained optimum never left the corridor -- and
    the remainder are heavy-tailed. A z-based n would misstate the tail the
    criterion actually depends on.
  * P1 and P2 are correlated across sources (-0.319 in Stage B: sources that
    hide an excursion tend to be the sources where the constraint bites). A
    closed form for the JOINT criterion would have to assume independence, which
    would overstate joint power. Resampling whole sources -- carrying the hidden
    indicator and the dV value together -- gets the dependence for free.

WHAT STAGE B IS AND IS NOT USED FOR. It supplies the plug-in population from
which sources are resampled, i.e. the VARIANCE and the planning values. It does
NOT supply delta. Delta is fixed from the frozen held-in property normalisation
in the preregistration; see the delta section there.

The planning prevalence is `endpoint_verified`'s 0.5833, the LOWER of the two
Stage B arms, so the primary is powered conservatively. If the true rate sits at
the bottom of Stage B's interval the confirmation will correctly fail to confirm
-- that is what a confirmation is for, and it is not a reason to plan on the
greedy arm's friendlier 0.6667.

CAVEAT, stated rather than buried: the plug-in population is 24 sources, so its
tails are coarse. These are planning numbers, not guarantees.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np

#: A2's V5a source-spread criterion, frozen before Stage B ran. Reused verbatim.
PREVALENCE_FLOOR = 1.0 / 3.0

#: Held-in DRD2 interquartile ranges. Fixed in the preregistration, NOT here.
DELTA = 0.25

STAGE_B_COMMIT = "bcc4a40"
STAGE_B_SOURCES = 24


def stage_b_plugin(commit: str = STAGE_B_COMMIT) -> tuple[np.ndarray, np.ndarray]:
    """(hidden-path indicator, paired terminal difference) per Stage B source."""
    hidden, delta = [], []
    for index in range(STAGE_B_SOURCES):
        blob = subprocess.run(
            ["git", "show", f"{commit}:diagnostics/pathwise_stage_b_shards/{index:03d}.json"],
            capture_output=True, text=True, check=True).stdout
        arms = json.loads(blob)["arms"]
        ev = arms["endpoint_verified"]
        hidden.append(bool(ev["endpoint_in_C"]) and ev["intermediate_violation_count"] > 0)
        delta.append(float(arms["pathwise_verified"]["U_P"]) - float(ev["U_P"]))
    return np.array(hidden, float), np.array(delta, float)


def power(hidden: np.ndarray, delta_obs: np.ndarray, n: int, *, margin: float,
          shift: float = 0.0, draws: int = 6000, boot: int = 2000,
          chunk: int = 250, seed: int = 20260816) -> dict:
    """Joint power at panel size `n`, by resampling whole Stage B sources."""
    rng = np.random.default_rng(seed)
    p1_hits = p2_hits = both = 0
    done = 0
    while done < draws:
        take = min(chunk, draws - done)
        # (take, n) synthetic panels; the SAME source index carries both series.
        src = rng.integers(0, len(hidden), (take, n))
        h, d = hidden[src], delta_obs[src] + shift
        # (take, boot, n) source-clustered bootstrap within each synthetic panel.
        idx = rng.integers(0, n, (take, boot, n))
        hb = np.take_along_axis(h[:, None, :], idx, axis=2).mean(axis=2)
        db = np.take_along_axis(d[:, None, :], idx, axis=2).mean(axis=2)
        p1 = np.percentile(hb, 2.5, axis=1) > PREVALENCE_FLOOR
        p2 = np.percentile(db, 2.5, axis=1) > -margin
        p1_hits += int(p1.sum())
        p2_hits += int(p2.sum())
        both += int((p1 & p2).sum())
        done += take
    return {"n": n, "p1": p1_hits / draws, "p2": p2_hits / draws,
            "joint": both / draws}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path,
                        default=Path("diagnostics/pathwise_heldout_power.json"))
    parser.add_argument("--draws", type=int, default=6000)
    args = parser.parse_args()

    hidden, delta_obs = stage_b_plugin()
    print(f"plug-in population, n={len(hidden)} Stage B sources")
    print(f"  p_hidden (endpoint_verified)  {hidden.mean():.4f}")
    print(f"  dV mean {delta_obs.mean():+.4f}  sd {delta_obs.std(ddof=1):.4f}  "
          f"exact ties {int((delta_obs == 0).sum())}")
    print(f"  corr(hidden, dV)              {np.corrcoef(hidden, delta_obs)[0, 1]:+.3f}")
    print(f"  criteria: p_hidden CI-low > {PREVALENCE_FLOOR:.4f}, "
          f"dV CI-low > {-DELTA:+.2f}\n")

    grid = [power(hidden, delta_obs, n, margin=DELTA, draws=args.draws)
            for n in (24, 36, 40, 44, 48, 52, 56)]
    print(f"{'n':>4} {'P1':>7} {'P2':>7} {'joint':>7}")
    for row in grid:
        mark = "   <- CHOSEN" if row["n"] == 48 else ""
        print(f"{row['n']:>4} {row['p1']:>7.3f} {row['p2']:>7.3f} "
              f"{row['joint']:>7.3f}{mark}")

    # What a stricter margin would have cost, so "why 0.25" is auditable.
    print(f"\nP2 at n=48 under alternative margins (delta is NOT chosen here):")
    margins = {}
    for m in (0.10, 0.15, 0.20, 0.25, 0.30):
        margins[m] = power(hidden, delta_obs, 48, margin=m, draws=args.draws)["p2"]
        print(f"  delta={m:.2f}   P2={margins[m]:.3f}")

    # If the true cost is worse than Stage B measured, P2 should and does fail.
    print(f"\nP2 at n=48, delta={DELTA}, sensitivity to the TRUE mean cost:")
    sens = {}
    for true_mean in (0.0, -0.023, -0.050, -0.105):
        shift = true_mean - float(delta_obs.mean())
        sens[true_mean] = power(hidden, delta_obs, 48, margin=DELTA,
                                shift=shift, draws=args.draws)["p2"]
        note = {0.0: "  (no cost at all)", -0.023: "  (Stage B verified)",
                -0.105: "  (Stage B GREEDY -- should fail)"}.get(true_mean, "")
        print(f"  true mean {true_mean:+.3f}   P2={sens[true_mean]:.3f}{note}")

    args.out.write_text(json.dumps({
        "schema": "compose.pathwise.heldout_power",
        "status": "DESIGN_ONLY_NO_HELDOUT_SOURCE_OPENED",
        "plugin_population": {
            "commit": STAGE_B_COMMIT, "sources": STAGE_B_SOURCES,
            "p_hidden_endpoint_verified": float(hidden.mean()),
            "dV_mean": float(delta_obs.mean()),
            "dV_sd": float(delta_obs.std(ddof=1)),
            "dV_exact_ties": int((delta_obs == 0).sum()),
            "corr_hidden_dV": float(np.corrcoef(hidden, delta_obs)[0, 1]),
            "role": ("supplies VARIANCE and planning values only; delta comes "
                     "from the frozen held-in normalisation, not from here"),
        },
        "criteria": {"prevalence_floor": PREVALENCE_FLOOR,
                     "prevalence_floor_provenance":
                         "Stage A2 V5a_source_spread, frozen before Stage B ran",
                     "noninferiority_margin": DELTA,
                     "test_form": "95% source-clustered bootstrap CI lower bound",
                     "multiplicity": ("intersection-union: the main-figure verdict "
                                      "requires BOTH, so each is tested at full "
                                      "alpha and the panel is sized on the joint")},
        "grid": grid,
        "chosen_n": 48,
        "binding_constraint": "P1",
        "p2_by_margin_at_n48": {str(k): v for k, v in margins.items()},
        "p2_sensitivity_to_true_cost_at_n48": {str(k): v for k, v in sens.items()},
        "caveat": ("the plug-in population is 24 sources, so its tails are "
                   "coarse; these are planning numbers, not guarantees"),
    }, indent=2) + "\n")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
