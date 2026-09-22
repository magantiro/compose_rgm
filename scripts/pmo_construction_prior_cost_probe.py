#!/usr/bin/env python3
"""Measure what the construction prior COSTS on the real shallow lane, zero oracle calls.

The scored A/B runs under a PER-LANE wall clock (``config.wall_seconds = 45.0``,
checked in ``dynamic_program_synthesis_v21._generate_channel_pool``).  A proposal law
that costs a model forward per construction step can therefore reduce the number of
candidates that lane contributes, which would change the CHANNEL MIX of the 16 queried
candidates and confound a matched comparison.  This probe measures that directly:
the same parents, the same seeds, the same production ``synthesize_dynamic_program``,
differing only in ``successor_prior``.

Reports the quantity the confound actually turns on -- how many constructions complete
inside one 45-second lane budget -- not a microbenchmark of the prior in isolation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from compose_v4.chem.molecular_graph import smiles_to_molecular_graph
from compose_v4.chem.state import pad_molecular_graph
from compose_v4.control.dynamic_program_synthesis import synthesize_dynamic_program

ROOT = Path(__file__).resolve().parents[1]
INIT = ROOT / "diagnostics/parent_edit_cycles/prepared/init_20260921.json"
PMO_SLOTS = 48
WALL_SECONDS = 45.0


def _parents(limit: int):
    rows = json.loads(INIT.read_text())["candidates"]
    out = []
    for row in rows[:limit]:
        graph = pad_molecular_graph(smiles_to_molecular_graph(row["endpoint"]), PMO_SLOTS)
        out.append((row["endpoint"], graph))
    return out


def _arm(parents, prior, *, attempts: int, seed: int):
    """One arm: `attempts` construction draws per parent, same seed schedule both arms."""

    began = perf_counter()
    completed = failed = 0
    prior_tagged = 0
    per_draw = []
    for index, (_smiles, graph) in enumerate(parents):
        for attempt in range(attempts):
            rng = np.random.default_rng(np.random.SeedSequence([seed, index, attempt]))
            step = perf_counter()
            try:
                _, _program, _binding, _trace, metadata = synthesize_dynamic_program(
                    graph, rng, max_modules=3, max_primitives=32, max_blocks=8,
                    successor_prior=prior,
                )
            except ValueError:
                failed += 1
                per_draw.append(perf_counter() - step)
                continue
            completed += 1
            if metadata.get("construction_law"):
                prior_tagged += 1
            per_draw.append(perf_counter() - step)
    elapsed = perf_counter() - began
    draws = completed + failed
    seconds_per_draw = elapsed / draws if draws else None
    return {
        "draws": draws,
        "completed": completed,
        "execution_rejected": failed,
        "construction_law_tagged": prior_tagged,
        "elapsed_seconds": round(elapsed, 3),
        "seconds_per_draw": None if seconds_per_draw is None else round(seconds_per_draw, 4),
        "median_seconds_per_draw": round(float(np.median(per_draw)), 4) if per_draw else None,
        # The quantity the confound turns on: the lane stops at 16 candidates OR 45 s.
        "draws_per_45s_lane": None if not seconds_per_draw else round(WALL_SECONDS / seconds_per_draw, 1),
        "completions_per_45s_lane": (
            None if not seconds_per_draw
            else round(WALL_SECONDS / seconds_per_draw * (completed / draws), 1)
        ),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parents", type=int, default=8)
    ap.add_argument("--attempts", type=int, default=6)
    ap.add_argument("--seed", type=int, default=20260922)
    ap.add_argument("--floor", type=float, default=0.05)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    from compose_v4.control.pmo_construction_prior import (
        ConstructionPriorSpec,
        build_construction_prior,
        resolve_checkpoint_path,
    )

    parents = _parents(args.parents)
    spec = ConstructionPriorSpec(
        checkpoint_path=str(resolve_checkpoint_path()),
        floor=args.floor,
        temperature=args.temperature,
    )
    load_began = perf_counter()
    prior = build_construction_prior(spec)
    load_seconds = perf_counter() - load_began

    off = _arm(parents, None, attempts=args.attempts, seed=args.seed)
    on = _arm(parents, prior, attempts=args.attempts, seed=args.seed)

    ratio = (
        None if not off["seconds_per_draw"]
        else round(on["seconds_per_draw"] / off["seconds_per_draw"], 2)
    )
    payload = {
        "schema_version": "pmo_construction_prior_cost_probe_v1",
        "oracle_calls_spent": 0,
        "parents": args.parents,
        "attempts_per_parent": args.attempts,
        "seed": args.seed,
        "lane_wall_seconds": WALL_SECONDS,
        "channel_candidate_limit": 16,
        "checkpoint": {
            "path": spec.checkpoint_path,
            "sha256": spec.checkpoint_sha256,
            "status": spec.status,
            "load_seconds": round(load_seconds, 2),
        },
        "off": off,
        "on": on,
        "on_over_off_seconds_ratio": ratio,
        "prior_statistics": prior.statistics,
        "verdict": (
            "UNKNOWN" if ratio is None
            else "NO_LANE_ATTRITION" if on["completions_per_45s_lane"] >= 16
            else "LANE_ATTRITION_RISK"
        ),
        "verdict_note": (
            "NO_LANE_ATTRITION means the ON arm still completes at least "
            "CHANNEL_CANDIDATE_LIMIT=16 constructions inside one 45 s lane budget, so "
            "the lane stops on the candidate cap in BOTH arms and the wall clock is "
            "not the binding constraint. It does NOT mean the prior is free."
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: v for k, v in payload.items() if k != "prior_statistics"},
                     indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
