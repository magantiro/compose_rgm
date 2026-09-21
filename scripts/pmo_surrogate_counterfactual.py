"""Matched-budget counterfactual: same pool, same information, different pick.

Consumes a capture from `pmo_surrogate_pool_capture.py`, which recorded -- for
every round of a REAL COMPOSE campaign -- the full candidate pool the
production allocator saw and the subset it actually charged.  Every pool member
is then scored with the exact free re-derivation of the benchmark objective, so
the counterfactual "what if a different 16 had been bought" is answerable at
zero oracle cost.

WHY THE PRIMARY METRIC IS PER-ROUND AND DOES NOT COMPOUND
---------------------------------------------------------
Changing which molecules are bought changes the archive, which changes what the
proposer emits next, so the captured pools for later rounds are only valid under
the production trajectory.  Re-simulating across rounds would quietly evaluate
arms on pools that could not have existed.  The primary comparison therefore
holds each round independent: identical pool, identical purchased archive (the
production one, which is real paid-for data), and only the choice differs.  That
under-states any compounding advantage, which is the conservative direction.

THE DIAGNOSTIC THAT SEPARATES THE TWO HYPOTHESES
------------------------------------------------
`proposal_headroom` reports, per round, whether the pool contained ANY molecule
better than the best already purchased.  If it usually did not, the run is
limited by what COMPOSE proposes and no ranker can fix it; if it usually did and
the production allocator failed to buy it, the run is limited by allocation.
That distinction is the finding, so it is computed first and reported whether or
not the surrogate wins.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np

from compose_v4.experiments.pmo_free_objectives import (
    FREE_ORACLE_PROTOCOL,
    build_free_scorer,
)
from compose_v4.experiments.pmo_surrogate_allocator import (
    MorganFeaturizer,
    OnlineArchive,
    SurrogateAllocator,
    build_surrogate,
    engagement_fraction,
)


def _initialization_endpoints() -> list[str]:
    from compose_v4.experiments import pmo_dynamic_v21 as v21

    root = Path(__file__).resolve().parents[1]
    lock = json.loads((root / v21.INITIALIZATION).read_text())
    if lock.get("count") != len(lock["candidates"]):
        raise ValueError("initialization lock count disagrees with its candidates")
    return [row["endpoint"] for row in lock["candidates"]]


def _mean(values):
    return float(statistics.fmean(values)) if values else None


def build_arms(seed: int) -> dict:
    """Policies compared at matched budget.  `production` is the real selection."""
    arms = {
        "random": {"kind": "random"},
        "production": {"kind": "production"},
        "pool_ceiling": {"kind": "ceiling"},
    }
    for name in ("gp_tanimoto", "knn_tanimoto", "gp_linear", "bagged_linear", "mean"):
        arms[f"{name}/greedy"] = {
            "kind": "surrogate", "surrogate": name,
            "acquisition": "greedy", "beta": 0.0, "explore": 0.0,
        }
    arms["gp_tanimoto/ucb_b1"] = {
        "kind": "surrogate", "surrogate": "gp_tanimoto",
        "acquisition": "ucb", "beta": 1.0, "explore": 0.0,
    }
    arms["gp_tanimoto/ucb_b1_x25"] = {
        "kind": "surrogate", "surrogate": "gp_tanimoto",
        "acquisition": "ucb", "beta": 1.0, "explore": 0.25,
    }
    arms["gp_tanimoto/ei"] = {
        "kind": "surrogate", "surrogate": "gp_tanimoto",
        "acquisition": "ei", "beta": 0.0, "explore": 0.0,
    }
    return arms


def evaluate_capture(capture: dict, *, seeds: list[int], min_fit_rows: int = 8) -> dict:
    task = capture["task"]
    scorer = build_free_scorer(task)
    pools = capture["pools"]
    featurizer = MorganFeaturizer()

    score_cache: dict[str, float] = {}

    def score(smiles: str) -> float:
        value = score_cache.get(smiles)
        if value is None:
            value = float(scorer(smiles))
            score_cache[smiles] = value
        return value

    # The production archive: the 16 initialization molecules are not in the
    # captured pools, so seed from the run's own frozen initialization lock --
    # the same task-blind lock the scored run used, verified by its own sha256
    # inside _load_initialization.
    init_smiles = capture.get("initialization_endpoints") or _initialization_endpoints()
    archive = OnlineArchive()
    for smiles in init_smiles:
        archive.observe(smiles, score(smiles), provenance="initialization", step=0)

    rounds_out = []
    arm_specs = build_arms(seeds[0])
    per_arm_selected: dict[str, list[float]] = {name: [] for name in arm_specs}
    per_arm_max: dict[str, list[float]] = {name: [] for name in arm_specs}
    rank_seconds: dict[str, list[float]] = {name: [] for name in arm_specs}

    for entry in pools:
        pool = [row for row in entry["pool"] if row.get("endpoint")]
        if not pool:
            continue
        smiles = [row["endpoint"] for row in pool]
        truth = np.asarray([score(s) for s in smiles], dtype=np.float64)
        selected_ids = set(entry.get("selected_candidate_ids") or [])
        k = max(1, len(selected_ids)) if selected_ids else 16
        k = min(k, len(pool))

        incumbent = archive.best
        ceiling_idx = list(np.argsort(-truth)[:k])
        production_idx = [i for i, row in enumerate(pool)
                          if row.get("candidate_id") in selected_ids]

        headroom = {
            "round": entry.get("batch_index"),
            "pool_size": len(pool),
            "k": k,
            "selection_ratio": round(len(pool) / k, 3),
            "archive_best_before": incumbent,
            "pool_best": float(truth.max()),
            "pool_contains_improvement": bool(truth.max() > incumbent),
            "n_pool_above_incumbent": int((truth > incumbent).sum()),
            "production_mean": _mean([truth[i] for i in production_idx]),
            "production_max": max((truth[i] for i in production_idx), default=None),
            "ceiling_mean": _mean([truth[i] for i in ceiling_idx]),
            "ceiling_max": float(truth[ceiling_idx[0]]),
            "archive_size_before": len(archive),
        }

        # Surrogate engagement on this round's pool, using only purchased rows.
        if len(archive) >= min_fit_rows:
            X_train, kept = featurizer.featurize(archive.smiles())
            X_pool, pool_kept = featurizer.featurize(smiles)
            if len(kept) >= min_fit_rows and pool_kept:
                model = build_surrogate("gp_tanimoto").fit(X_train, archive.scores()[kept])
                pred = model.predict(X_pool)
                headroom["engagement_greedy"] = engagement_fraction(
                    pred, incumbent, kind="greedy"
                )
                headroom["engagement_ucb_b1"] = engagement_fraction(
                    pred, incumbent, kind="ucb", beta=1.0
                )
                order = np.argsort(-pred.mean)
                headroom["surrogate_spearman"] = _spearman(pred.mean, truth)
                headroom["surrogate_topk_mean"] = _mean(
                    [float(truth[i]) for i in order[:k]]
                )

        rounds_out.append(headroom)

        for name, spec in arm_specs.items():
            if spec["kind"] == "production":
                picks, elapsed = production_idx, 0.0
            else:
                picks, elapsed = _select(
                    spec, pool, smiles, truth, k, archive, featurizer, seeds, min_fit_rows
                )
            if not picks:
                continue
            per_arm_selected[name].append(_mean([float(truth[i]) for i in picks]))
            per_arm_max[name].append(max(float(truth[i]) for i in picks))
            rank_seconds[name].append(elapsed)

        # Advance the archive along the PRODUCTION trajectory, which is what
        # actually happened and what every arm is conditioned on.
        for i in production_idx:
            archive.observe(smiles[i], float(truth[i]),
                            provenance=f"round_{entry.get('batch_index')}",
                            step=int(entry.get("batch_index") or 0))

    arms_out = {}
    for name in arm_specs:
        arms_out[name] = {
            "rounds": len(per_arm_selected[name]),
            "mean_selected_score": _mean(per_arm_selected[name]),
            "mean_best_in_batch": _mean(per_arm_max[name]),
            "mean_rank_seconds": _mean(rank_seconds[name]),
        }
    production = arms_out.get("production", {}).get("mean_selected_score")
    ceiling = arms_out.get("pool_ceiling", {}).get("mean_selected_score")
    for name, row in arms_out.items():
        value = row["mean_selected_score"]
        if value is not None and production is not None:
            row["delta_vs_production"] = value - production
            if ceiling is not None and ceiling > production:
                row["fraction_of_available_headroom"] = (
                    (value - production) / (ceiling - production)
                )

    return {
        "schema_version": "pmo_surrogate_counterfactual_v1",
        "task": task,
        "charged_oracle_calls": 0,
        "oracle_protocol": FREE_ORACLE_PROTOCOL,
        "surrogate_training_disclosure": {
            "training_rows": "only molecules purchased earlier in this same run",
            "provenance_counts": archive.training_provenance(),
            "external_scored_database_used": False,
            "pretrained_activity_model_used": False,
            "reference_molecules_used": False,
            "features": featurizer.identity,
        },
        "n_rounds": len(rounds_out),
        "rounds": rounds_out,
        "proposal_headroom": _headroom_summary(rounds_out),
        "arms": arms_out,
    }


def _headroom_summary(rounds: list[dict]) -> dict:
    if not rounds:
        return {}
    with_improvement = [r for r in rounds if r["pool_contains_improvement"]]
    return {
        "rounds": len(rounds),
        "rounds_whose_pool_contained_an_improvement": len(with_improvement),
        "fraction_of_rounds_with_any_improvement": len(with_improvement) / len(rounds),
        "mean_pool_size": _mean([r["pool_size"] for r in rounds]),
        "mean_selection_ratio": _mean([r["selection_ratio"] for r in rounds]),
        "mean_n_pool_above_incumbent": _mean([r["n_pool_above_incumbent"] for r in rounds]),
        "mean_production_max": _mean([r["production_max"] for r in rounds if r["production_max"] is not None]),
        "mean_ceiling_max": _mean([r["ceiling_max"] for r in rounds]),
        "mean_engagement_greedy": _mean(
            [r["engagement_greedy"] for r in rounds if "engagement_greedy" in r]
        ),
        "mean_engagement_ucb_b1": _mean(
            [r["engagement_ucb_b1"] for r in rounds if "engagement_ucb_b1" in r]
        ),
        "mean_surrogate_spearman": _mean(
            [r["surrogate_spearman"] for r in rounds if r.get("surrogate_spearman") is not None]
        ),
        "interpretation": (
            "fraction_of_rounds_with_any_improvement near 0 means the pool, not the "
            "allocator, is the binding constraint"
        ),
    }


def _spearman(a: np.ndarray, b: np.ndarray) -> float | None:
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return None
    ra, rb = _rank(a), _rank(b)
    ra, rb = ra - ra.mean(), rb - rb.mean()
    denom = float(np.sqrt((ra**2).sum() * (rb**2).sum()))
    return float((ra * rb).sum() / denom) if denom > 0 else None


def _rank(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x)
    ranks = np.empty(len(x), dtype=np.float64)
    ranks[order] = np.arange(len(x), dtype=np.float64)
    return ranks


def _select(spec, pool, smiles, truth, k, archive, featurizer, seeds, min_fit_rows):
    """Return (indices, seconds).  The production arm is handled by the caller,
    which already holds the real selection."""
    kind = spec["kind"]
    if kind == "ceiling":
        return list(np.argsort(-truth)[:k]), 0.0
    if kind == "random":
        rng = np.random.default_rng(seeds[0] + len(archive))
        return [int(i) for i in rng.choice(len(pool), size=k, replace=False)], 0.0
    allocator = SurrogateAllocator(
        surrogate_name=spec["surrogate"],
        acquisition=spec["acquisition"],
        beta=spec["beta"],
        explore_fraction=spec["explore"],
        min_fit_rows=min_fit_rows,
        featurizer=featurizer,
        seed=seeds[0],
    )
    started = time.perf_counter()
    decision = allocator.select_batch(archive, smiles, k)
    return decision.selected, time.perf_counter() - started


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", required=True, nargs="+")
    parser.add_argument("--out", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    args = parser.parse_args()

    reports = []
    for path in args.capture:
        capture = json.loads(Path(path).read_text())
        reports.append(evaluate_capture(capture, seeds=args.seeds))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"reports": reports}, indent=1, sort_keys=True))
    for report in reports:
        print(f"\n=== {report['task']} ===")
        print(json.dumps(report["proposal_headroom"], indent=1))
        print(json.dumps(report["arms"], indent=1))


if __name__ == "__main__":
    main()
