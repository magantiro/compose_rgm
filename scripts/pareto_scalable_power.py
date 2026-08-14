"""Power the fresh scalable-Pareto panel USING THE FINAL ANALYSIS PROCEDURE.

WHY THIS REPLACES THE FIRST CALCULATION. The initial n=24 planning used a normal
approximation, `lower = r - 1.96 * sd / sqrt(n)`, while the final analysis will
report a **source-level paired bootstrap** CI. Planning under one procedure and
reporting under another is exactly the kind of mismatch a reviewer finds, and the
two do not agree in general -- the bootstrap of a ratio is skewed, and the normal
approximation is symmetric by construction.

So this script simulates the ACTUAL estimand and the ACTUAL interval:

    r_i = HV_i(budgeted greedy) / HV_i(full greedy)          per source
    report: source-level bootstrap 95% CI on mean(r_i)
    criterion: lower bound > 0.90

VARIANCE SOURCE. The existing 12 sources supply the spread, and nothing else --
they are planning input, never the load-bearing efficiency estimate. The proxy is
the paired controller HV difference (verified - greedy) over mean full-greedy HV,
because a paired difference between two controllers on the SAME source is the
same shape as full-vs-budgeted on the same source. It is conservative: a
shortlist retaining 56% of the fiber should perturb less than swapping the
controller outright.

The ratio is simulated on the LOG scale so it cannot go negative, which a normal
approximation on a ratio permits and which would silently inflate the tail.
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np

RETENTION_FLOOR = 0.90
BOOT_DRAWS = 4000


def bootstrap_lower(r: np.ndarray, rng: np.random.Generator,
                    draws: int = BOOT_DRAWS) -> float:
    """The interval the final analysis will report: source-level, paired."""
    idx = rng.integers(0, len(r), (draws, len(r)))
    return float(np.percentile(r[idx].mean(axis=1), 2.5))


def power(n: int, true_ret: float, sd_ratio: float, *, trials: int = 4000,
          seed: int = 20260814) -> float:
    rng = np.random.default_rng(seed)
    # Log-normal so a simulated retention can never be negative.
    sigma = np.sqrt(np.log1p((sd_ratio / true_ret) ** 2))
    mu = np.log(true_ret) - sigma ** 2 / 2
    hits = 0
    for _ in range(trials):
        r = rng.lognormal(mu, sigma, n)
        hits += bootstrap_lower(r, rng) > RETENTION_FLOOR
    return hits / trials


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--smoke", type=Path,
                    default=Path("local_runtime/pareto_control_smoke"))
    ap.add_argument("--out", type=Path,
                    default=Path("diagnostics/pareto_scalable_power.json"))
    args = ap.parse_args()

    rows = [json.loads(Path(f).read_text())
            for f in sorted(glob.glob(str(args.smoke / "*.json")))
            if "partial" not in f]
    g = np.array([r["arms"]["greedy_pref"]["normalized_hypervolume"] for r in rows])
    v = np.array([r["arms"]["verified_pref"]["normalized_hypervolume"] for r in rows])
    sd_ratio = float((v - g).std(ddof=1) / g.mean())

    print("variance planning from the existing 12 (planning input only)")
    print(f"  paired controller HV diff sd {(v - g).std(ddof=1):.4f}")
    print(f"  mean full-greedy HV          {g.mean():.4f}")
    print(f"  implied retention-ratio sd   {sd_ratio:.4f}\n")
    print(f"criterion: source-level BOOTSTRAP 95% CI lower bound on mean r_i "
          f"> {RETENTION_FLOOR}")
    print("  (the same estimand and the same interval the final analysis uses)\n")

    grid = {}
    print(f"{'true retention':>16}" + "".join(f"{f'n={n}':>10}" for n in (16, 20, 24, 30)))
    for true_ret in (0.93, 0.95, 0.97, 1.00):
        row = {}
        line = f"{true_ret:>16.2f}"
        for n in (16, 20, 24, 30):
            p = power(n, true_ret, sd_ratio)
            row[n] = p
            line += f"{p:>10.3f}"
        grid[true_ret] = row
        print(line)

    chosen = grid[0.95][24]
    print(f"\n  n=24 at true retention 0.95 -> power {chosen:.3f}")

    args.out.write_text(json.dumps({
        "schema": "compose.pareto.scalable_power",
        "status": "DESIGN_ONLY",
        "estimand": "r_i = HV_i(budgeted greedy) / HV_i(full greedy), source-level",
        "interval": "source-level paired bootstrap, 2.5th percentile of the mean",
        "criterion": f"lower bound > {RETENTION_FLOOR}",
        "supersedes": ("the initial normal-approximation planning; planning and "
                       "reporting must use the same procedure"),
        "variance_proxy": {
            "paired_controller_hv_diff_sd": float((v - g).std(ddof=1)),
            "mean_full_greedy_hv": float(g.mean()),
            "implied_retention_ratio_sd": sd_ratio,
            "why_conservative": ("a shortlist retaining 56% of the fiber should "
                                 "perturb less than swapping the controller"),
        },
        "power_grid": {str(k): {str(n): p for n, p in v_.items()}
                       for k, v_ in grid.items()},
        "chosen_n": 24,
    }, indent=2) + "\n")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
