"""Paired per-parent comparison of two arms, from a finished comparison artifact.

The arms are matched: every arm sees the same parents, the same per-draw seeds
and the same declared module counts.  So the informative statistic is the PAIRED
per-parent difference, not the difference of two pooled means -- pooling throws
away the pairing that the harness went to trouble to create, and it lets a
single permissive parent carry the result.

Reported: the mean and median paired difference, how many parents moved each
way, an exact two-sided sign test, and a bootstrap interval resampling PARENTS
(the unit the claim is about), not draws.

Reads only; writes a summary beside the artifact. Zero oracle calls.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

SCHEMA_VERSION = "route_prior_paired_summary_v1"

#: Metrics whose per-parent values the artifact carries for every arm.
PAIRED_METRICS = (
    "complete_program_yield",
    "legal_execution_rate",
    "distinct_endpoints",
    "teacher_region_recall",
)


def sign_test(differences, *, tolerance: float = 1e-12) -> dict:
    """Exact two-sided sign test, ties discarded.

    Deliberately distribution-free: fifteen parents is far too few to lean on a
    normal approximation, and per-parent yields are bounded rates.
    """

    positive = sum(1 for d in differences if d > tolerance)
    negative = sum(1 for d in differences if d < -tolerance)
    n = positive + negative
    if n == 0:
        return {"positive": 0, "negative": 0, "ties": len(differences), "p_value": 1.0}
    extreme = min(positive, negative)
    tail = sum(math.comb(n, k) for k in range(extreme + 1)) / (2.0**n)
    return {
        "positive": positive,
        "negative": negative,
        "ties": len(differences) - n,
        "p_value": min(1.0, 2.0 * tail),
    }


def bootstrap_interval(differences, *, draws: int = 20000, seed: int = 20260921) -> dict:
    rng = np.random.default_rng(seed)
    values = np.asarray(differences, dtype=float)
    if values.size == 0:
        return {"low": 0.0, "high": 0.0}
    means = values[rng.integers(0, values.size, size=(draws, values.size))].mean(axis=1)
    return {
        "low": float(np.percentile(means, 2.5)),
        "high": float(np.percentile(means, 97.5)),
        "resampled": "parents",
        "draws": draws,
    }


def compare(rows, treatment: str, control: str) -> dict:
    out = {}
    for metric in PAIRED_METRICS:
        pairs = [
            (row["arms"][treatment][metric], row["arms"][control][metric])
            for row in rows
            if treatment in row.get("arms", {}) and control in row.get("arms", {})
        ]
        if not pairs:
            continue
        differences = [float(a) - float(b) for a, b in pairs]
        out[metric] = {
            "parents": len(pairs),
            "treatment_mean": float(np.mean([a for a, _ in pairs])),
            "control_mean": float(np.mean([b for _, b in pairs])),
            "mean_paired_difference": float(np.mean(differences)),
            "median_paired_difference": float(np.median(differences)),
            "sign_test": sign_test(differences),
            "bootstrap_95": bootstrap_interval(differences),
        }
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    payload = json.loads(Path(args.comparison).read_text())
    contrasts = {
        "declared_prior_vs_declared_uniform": ("declared_prior", "declared_uniform"),
        "declared_prior_families_vs_uniform": (
            "declared_prior_families",
            "declared_uniform",
        ),
        "declared_prior_region_vs_uniform": (
            "declared_prior_region",
            "declared_uniform",
        ),
        "production_route_law_vs_v1": ("production_route_law", "production_v1"),
    }
    summary = {}
    for population in ("t4_heldout", "generic"):
        rows = [
            row
            for row in payload["per_parent"].get(population, [])
            if "arms" in row
        ]
        if not rows:
            continue
        summary[population] = {
            name: compare(rows, treatment, control)
            for name, (treatment, control) in contrasts.items()
        }

    out = {
        "schema_version": SCHEMA_VERSION,
        "paired_contrasts": summary,
        "source_comparison": args.comparison,
        "note": "arms are matched per parent, per draw seed and per declared "
        "module count, so the paired difference is the informative statistic",
    }
    Path(args.out).write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
