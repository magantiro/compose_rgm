"""Does the learned law RANK the teacher's region above chance on unseen sources?

This is the coordinate-level question the region law is answerable for, and it
is separate from -- and much cheaper than -- complete-program yield.  It asks
only: out of the bridge-separated regions this unseen parent offers, where does
the law put the one the teacher actually released?

The control is analytic, not sampled.  Under a uniform law over ``N`` regions of
which ``T`` are teacher targets, ``P(a target lands in the top k)`` is
``1 - C(N-T, k) / C(N, k)`` exactly, so the comparison needs no Monte Carlo and
carries no sampling error on the control side.  The learned side is a
deterministic weight ordering, so it carries none either.

Leave-source-out throughout: each source is ranked under the fold prior fitted
without any route from that source molecule.

Zero oracle calls; zero chemistry beyond the region enumeration the proposal
path already performs.
"""

from __future__ import annotations

import argparse
import glob
import gzip
import hashlib
import json
import math
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from route_prior_heldout_comparison import _teacher_released

from compose_v4.control.bridge_region_law import bridge_separated_regions
from compose_v4.control.route_program_prior import RouteProgramPrior
from compose_v4.experiments.route_program_prior_eval import parent_state

SCHEMA_VERSION = "route_prior_region_ranking_v1"

RECALL_AT = (1, 3, 5, 10)


def _revision() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (FileNotFoundError, OSError, subprocess.CalledProcessError):
        return "unknown"


def uniform_recall_at(total: int, targets: int, k: int) -> float:
    """Exact P(at least one of `targets` lands in a uniformly drawn top `k`)."""

    if targets <= 0 or total <= 0:
        return 0.0
    k = min(k, total)
    if total - targets < k:
        return 1.0
    return 1.0 - math.comb(total - targets, k) / math.comb(total, k)


def uniform_reciprocal_rank(total: int, targets: int) -> float:
    """Expected 1/rank of the FIRST target under a uniform random order.

    The first of `targets` marked items among `total` has rank `r` with
    probability `C(total-r, targets-1) / C(total, targets)`, so this is an exact
    expectation rather than an approximation.
    """

    if targets <= 0 or total <= 0:
        return 0.0
    denominator = math.comb(total, targets)
    return sum(
        math.comb(total - rank, targets - 1) / denominator / rank
        for rank in range(1, total - targets + 2)
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    started = time.monotonic()
    fit_raw = Path(args.fit).read_bytes()
    fit = json.loads(fit_raw.decode())
    priors = {k: RouteProgramPrior.from_payload(v) for k, v in fit["priors"].items()}
    fold_of = {
        source: name for name, held in fit["folds"].items() for source in held
    }

    routes: dict[str, list] = {}
    smiles_of: dict[str, str] = {}
    for path in sorted(glob.glob("diagnostics/ivg_winner_paths/pairs/*.json.gz")):
        record = json.loads(gzip.decompress(Path(path).read_bytes()).decode())
        route = record["payload"]["path"]
        if route["status"] != "witness_found":
            continue
        source = str(record["pair"]["source"])
        source_id = hashlib.sha256(source.encode()).hexdigest()
        smiles_of[source_id] = source
        routes.setdefault(source_id, []).append(
            (tuple(route["states"]), tuple(route["actions"]))
        )

    per_source = []
    for source_id, traces in sorted(routes.items()):
        graph = parent_state(smiles_of[source_id])
        support = bridge_separated_regions(graph, maximum=None)
        released = []
        for states, actions in traces:
            released.extend(_teacher_released(states, actions, graph))
        fragments = [set(region.fragment) for region in support]
        target_positions = set()
        for release in released:
            best, best_size = None, None
            for position, fragment in enumerate(fragments):
                if release <= fragment and (best_size is None or len(fragment) < best_size):
                    best, best_size = position, len(fragment)
            if best is not None:
                target_positions.add(best)
        if not target_positions or not support:
            per_source.append(
                {"source_id": source_id, "skipped": "no bound teacher region"}
            )
            continue

        prior = priors[fold_of[source_id]]
        weights = prior.region_weights(graph, support)
        # Descending weight, ties broken by the enumeration order the support
        # already fixes, so the ordering is deterministic.
        order = list(np.argsort(-weights, kind="stable"))
        ranks = sorted(order.index(position) + 1 for position in target_positions)
        total, targets = len(support), len(target_positions)
        per_source.append(
            {
                "source_id": source_id,
                "fold": fold_of[source_id],
                "heavy_atoms": int(graph.n_real_atoms),
                "support_regions": total,
                "teacher_target_regions": targets,
                "best_teacher_rank_learned": ranks[0],
                "learned": {
                    f"recall_at_{k}": float(ranks[0] <= k) for k in RECALL_AT
                },
                "learned_reciprocal_rank": 1.0 / ranks[0],
                "uniform": {
                    f"recall_at_{k}": uniform_recall_at(total, targets, k)
                    for k in RECALL_AT
                },
                "uniform_reciprocal_rank": uniform_reciprocal_rank(total, targets),
            }
        )

    scored = [row for row in per_source if "skipped" not in row]
    headline = {
        "sources_scored": len(scored),
        "sources_skipped": len(per_source) - len(scored),
        "median_support_regions": float(
            np.median([r["support_regions"] for r in scored])
        ),
        "median_teacher_targets": float(
            np.median([r["teacher_target_regions"] for r in scored])
        ),
        "learned": {
            **{
                f"recall_at_{k}": float(
                    np.mean([r["learned"][f"recall_at_{k}"] for r in scored])
                )
                for k in RECALL_AT
            },
            "mean_reciprocal_rank": float(
                np.mean([r["learned_reciprocal_rank"] for r in scored])
            ),
        },
        "uniform": {
            **{
                f"recall_at_{k}": float(
                    np.mean([r["uniform"][f"recall_at_{k}"] for r in scored])
                )
                for k in RECALL_AT
            },
            "mean_reciprocal_rank": float(
                np.mean([r["uniform_reciprocal_rank"] for r in scored])
            ),
        },
    }

    payload = {
        "schema_version": SCHEMA_VERSION,
        "headline": headline,
        "per_source": per_source,
        "regime": "leave-source-out; every source ranked under a prior fitted "
        "without any route from that source molecule",
        "control": "analytic uniform over the same region support; no sampling "
        "error on either side",
        "costs": {"oracle_calls": 0, "docking_calls": 0, "modal_launches": 0},
        "inputs": {
            "fit": {"path": args.fit, "sha256": hashlib.sha256(fit_raw).hexdigest()}
        },
        "implementation": {
            "code_revision": _revision(),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "rdkit": __import__("rdkit").__version__,
            "wall_seconds": round(time.monotonic() - started, 2),
        },
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    body = json.dumps(payload, indent=1, sort_keys=True)
    out.write_text(body + "\n")
    print(json.dumps(headline, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
