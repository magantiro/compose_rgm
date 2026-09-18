"""Adaptive versus reward-blind selection on LITERALLY the same candidate pools.

The question this run exists to answer is narrow on purpose:

    given a rich pool of exactly feasible COMPOSE program endpoints, does docking
    feedback improve which of them we choose to dock next?

Nothing else may differ between the arms. So the two selectors run in lockstep inside one
process against one shared pool per round: the pool is generated once from the union of
both frontiers, both arms see the identical list, each picks its own batch of eight, the
union of their picks is docked once, and each arm is charged eight calls and is shown only
its own outcomes. Pools are therefore identical by construction rather than by matched
seeds, and the docking protocol, seed, budget and starting archive are shared.

TWO KNOWN BIASES, both against the adaptive arm, stated so neither is mistaken for rigour:

  * the pool is grown from the UNION of frontiers, so the blind arm gets to propose from
    parents the adaptive arm discovered. If adaptive still separates, it separates while
    handing its discoveries to the control.
  * a molecule both arms pick is docked once and its single score is given to both, so
    docking noise cannot manufacture a difference between them on shared picks.

WHY NO Q3 HERE. The receding-horizon branch value exists (`t4_fiber_expansion.branch_value`)
and is deliberately not used. Adding it before one-step selection is shown to beat blind
selection would leave two untested mechanisms sharing one result. Program depth is a
different knob and stays at three modules: the action is the whole variable-duration
program, and multi-region options are first class.

Selection ranks predicted ENDPOINT score, never improvement over a candidate's own parent.
The previous run ranked gain and bought -6.8 endpoints while holding a -9.10 incumbent.
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np

from compose_v4.control.fiber_control import (
    ProgramValue,
    SearchState,
    acquisition,
    endpoint_utility,
    should_stop,
)
from compose_v4.data.durable_path import require_durable_path
from compose_v4.experiments.t4_fiber_campaign import Fiber, prepare
from compose_v4.experiments.t4_fiber_expansion import expand_frontier

SCHEMA_VERSION = "t4_fiber_paired_campaign_v1"

JAK2_ROOT = "COC(=O)CC2Nc1ccccc1c3ccnc4[nH]cc2c34"
ROOT_CONTEXT = {"qed": 0.712, "similarity": 1.0, "sa": 3.09}


def _docker(target: str, seed: int):
    """The counted oracle. Every call against the development target is charged."""
    import modal

    remote = modal.Function.from_name("jak2-contrastive-dock", "dock_batch")

    def dock(records, tag: str) -> dict[str, float]:
        if not records:
            return {}
        request = {
            "target": target,
            "seed": seed,
            "candidates": [
                {
                    "index": i, "bundle": tag, "value": 0, "smiles": r["smiles"],
                    "qed": r["qed"], "similarity": r["similarity"], "sa": r["sa"],
                }
                for i, r in enumerate(records)
            ],
        }
        answer = json.loads(remote.remote(json.dumps(request)))
        return {r["smiles"]: r["score"] for r in answer["results"] if r["score"] is not None}

    return dock


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=JAK2_ROOT)
    parser.add_argument("--target", default="jak2")
    parser.add_argument("--delta", type=float, default=0.6)
    parser.add_argument("--budget", type=int, default=112, help="charged calls PER ARM")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--draws", type=int, default=720, help="raw programs per parent")
    parser.add_argument("--shards", type=int, default=8)
    parser.add_argument("--parents", type=int, default=4, help="frontier width PER ARM")
    parser.add_argument("--horizon", type=int, default=3, help="modules per program")
    parser.add_argument("--support", default="benchmark",
                        choices=("benchmark", "legacy_screened"),
                        help="benchmark = the task's own criterion, the only fair one "
                             "for an IVG comparison")
    parser.add_argument("--exploration", type=int, default=2,
                        help="random picks per batch; the rest are the model's ranking")
    parser.add_argument("--seed", type=int, default=20260918)
    parser.add_argument("--out", default="diagnostics/t4_fiber_control/paired.json")
    options = parser.parse_args()

    destination = require_durable_path(options.out, role="paired campaign result")
    destination.parent.mkdir(parents=True, exist_ok=True)

    fiber = Fiber(options.root, options.delta, support=options.support)
    rng = np.random.default_rng(options.seed)
    dock = _docker(options.target, options.seed)

    # One shared starting archive: the root is docked once and both arms are given that
    # same number, so neither arm starts from a luckier measurement of the same molecule.
    root_score = dock([{"smiles": options.root, **ROOT_CONTEXT}], "root")
    if options.root not in root_score:
        raise RuntimeError("the root failed to dock; nothing downstream is interpretable")
    arms = {
        "adaptive": {
            "state": SearchState(archive=dict(root_score), budget=options.budget - 1),
            "value": ProgramValue(), "x": [], "y": [], "previous": None, "stopped": False,
        },
        "blind": {
            "state": SearchState(archive=dict(root_score), budget=options.budget - 1),
            "value": None, "x": [], "y": [], "previous": None, "stopped": False,
        },
    }
    for arm in arms.values():
        arm["previous"] = arm["state"].incumbent

    print(f"[paired] {options.target} delta={options.delta}; {options.budget} charged calls per arm", flush=True)
    print(f"[paired] root {options.root} scores {root_score[options.root]:.2f}", flush=True)
    print(f"[paired] pool from {options.draws} raw programs per parent over {options.shards} shards", flush=True)
    print(f"[paired] selection: predicted ENDPOINT score, one step, program depth "
          f"{options.horizon}; fiber support = {options.support}\n", flush=True)

    rounds, round_index = [], 0
    while any(a["state"].budget >= options.batch and not a["stopped"] for a in arms.values()):
        round_index += 1
        started = time.time()

        # ---- one shared pool, grown from the union of both frontiers ----
        frontier, seen = [], set()
        for name, arm in arms.items():
            if arm["stopped"]:
                continue
            for parent in arm["state"].parents(limit=options.parents, rng=rng):
                if parent not in seen:
                    seen.add(parent)
                    frontier.append((parent, name))
        scored_frontier = [
            (
                parent,
                min(a["state"].archive[parent] for a in arms.values()
                    if parent in a["state"].archive),
            )
            for parent, _ in frontier
        ]
        pool = expand_frontier(
            scored_frontier, options.root, options.delta,
            draws=options.draws, workers=options.shards, horizon=options.horizon,
            seed=options.seed + 104729 * round_index, support=options.support,
        )
        generated = time.time() - started

        # ---- each arm selects from the identical list ----
        picks, requested, ranking = {}, {}, {}
        for name, arm in arms.items():
            if arm["stopped"] or arm["state"].budget < options.batch:
                continue
            fresh = [r for r in pool.values() if r["smiles"] not in arm["state"].archive]
            if not fresh:
                arm["stopped"] = True
                continue
            prepared = prepare(fresh, arm["state"], fiber)
            model = arm["value"] or ProgramValue()
            if arm["value"] is not None and should_stop(arm["state"], prepared, model):
                arm["stopped"] = True
                continue
            selected = acquisition(
                prepared, model, arm["state"], rng,
                batch=options.batch,
                exploration=options.batch if arm["value"] is None else options.exploration,
            )
            chosen = [prepared[i] for i in selected]
            predicted = model.predict(np.asarray([c["features"] for c in chosen]))
            picks[name] = list(zip(chosen, (float(p) for p in predicted)))
            # The ranking the model assigned to the WHOLE pool before anything was
            # docked. Used after the fact to place the round's best molecule in it: if a
            # -10.7 candidate was present and ranked 180th, reward learning is the
            # remaining problem, and no amount of acquisition machinery fixes it.
            if arm["value"] is not None:
                order = np.argsort(-endpoint_utility(prepared, model))
                ranking[name] = {
                    prepared[index]["smiles"]: position
                    for position, index in enumerate(order, start=1)
                }
                ranking[f"{name}_pool"] = len(prepared)
            for candidate, _ in picks[name]:
                requested[candidate["smiles"]] = candidate
        if not picks:
            print(f"[round {round_index}] no arm can select; stopping", flush=True)
            break

        # ---- dock the union once; a shared pick cannot differ between arms by noise ----
        scores = dock(list(requested.values()), f"r{round_index}")

        record = {
            "round": round_index, "pool": len(pool), "parents": len(frontier),
            "multi_region_in_pool": sum(1 for r in pool.values() if r["regions"] > 1),
            "generation_seconds": round(generated, 1),
            "docked_distinct": len(requested), "arms": {},
        }
        for name, arm in arms.items():
            if name not in picks:
                record["arms"][name] = {"stopped": True, "best": arm["state"].incumbent}
                continue
            state = arm["state"]
            docked = []
            for candidate, expectation in picks[name]:
                score = scores.get(candidate["smiles"])
                docked.append({
                    "smiles": candidate["smiles"], "parent": candidate["parent"],
                    "parent_score": candidate["parent_score"], "families": candidate["families"],
                    "regions": candidate["regions"], "created": candidate["created"],
                    "deleted": candidate["deleted"], "similarity": candidate["similarity"],
                    "qed": candidate["qed"], "sa": candidate["sa"],
                    "predicted_endpoint": candidate["parent_score"] - expectation,
                    "score": score,
                })
                if score is None:
                    continue
                state.archive[candidate["smiles"]] = score
                arm["x"].append(candidate["features"])
                arm["y"].append(candidate["parent_score"] - score)
            state.budget -= len(picks[name])
            if arm["value"] is not None:
                arm["value"].fit(arm["x"], arm["y"])
            improved = state.incumbent < arm["previous"] - 1e-9
            winner = min((d for d in docked if d["score"] is not None),
                         key=lambda d: d["score"], default=None)
            state.history.append({"round": round_index, "improved": improved})
            arm["previous"] = state.incumbent
            record["arms"][name] = {
                "best": state.incumbent, "calls": options.budget - state.budget,
                "improved": improved, "docked": docked,
                "improving_regions": winner["regions"] if improved and winner else None,
                "improving_families": winner["families"] if improved and winner else None,
                "round_best": winner["score"] if winner else None,
            }
        # Where the round's actual best molecule sat in the adaptive model's pre-docking
        # ranking of the entire pool, and how well that ranking predicted what came back.
        every = [
            d for arm in record["arms"].values() for d in arm.get("docked", [])
            if d["score"] is not None
        ]
        if every and "adaptive" in ranking:
            champion = min(every, key=lambda d: d["score"])
            record["model_rank_of_round_best"] = ranking["adaptive"].get(champion["smiles"])
            record["ranked_pool"] = ranking.get("adaptive_pool")
            record["round_best_score"] = champion["score"]
        history_x = arms["adaptive"]["x"]
        if len(history_x) >= 8:
            pairs = [
                (d["predicted_endpoint"], d["score"])
                for entry in rounds + [record]
                for d in entry["arms"].get("adaptive", {}).get("docked", [])
                if d["score"] is not None and d.get("predicted_endpoint") is not None
            ]
            if len(pairs) >= 8:
                predicted_all = np.asarray([a for a, _ in pairs])
                realized_all = np.asarray([b for _, b in pairs])
                if predicted_all.std() > 1e-9:
                    record["rho_predicted_realized"] = float(
                        np.corrcoef(predicted_all, realized_all)[0, 1]
                    )
        rounds.append(record)

        line = (f"[round {round_index}] pool {len(pool):>5} "
                f"({record['multi_region_in_pool']} multi) from {len(frontier)} parents "
                f"| gen {generated:5.0f}s | docked {len(requested):>2}")
        for name in ("adaptive", "blind"):
            arm = record["arms"][name]
            mark = "*" if arm.get("improved") else " "
            line += f" | {name} {arm['best']:6.2f}{mark}@{arm.get('calls', '-')}"
        rho = record.get("rho_predicted_realized")
        rank, ranked = record.get("model_rank_of_round_best"), record.get("ranked_pool")
        line += f" | rho {rho:+.3f}" if rho is not None else " | rho     -"
        if rank is not None and ranked:
            line += f" | best ranked {rank}/{ranked} ({record['round_best_score']:.2f})"
        print(line, flush=True)
        destination.write_text(json.dumps({
            "schema_version": SCHEMA_VERSION, "options": vars(options),
            "root_score": root_score[options.root], "rounds": rounds,
            "archives": {n: a["state"].archive for n, a in arms.items()},
        }, indent=1))

    print("\n[paired] FINAL", flush=True)
    for name, arm in arms.items():
        state = arm["state"]
        print(f"  {name:>8}  best {state.incumbent:6.2f}  after {options.budget - state.budget} calls"
              f"  archive {len(state.archive)}", flush=True)
        for smiles, score in sorted(state.archive.items(), key=lambda kv: kv[1])[:3]:
            print(f"            {score:7.2f}  {smiles}", flush=True)
    print(f"\n[paired] written to {destination}", flush=True)


if __name__ == "__main__":
    main()
