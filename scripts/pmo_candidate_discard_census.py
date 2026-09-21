"""WHERE does the PMO path discard candidates? Zero oracle calls.

WHY THIS IS A SEPARATE GATE FROM THE SCORING-IDENTITY GATE
-----------------------------------------------------------
`pmo_proposal_scoring_identity_gate.py` establishes that the object a proposal channel
conditions is the object that gets scored (measured `recovered_fraction` 1.0000). That is
necessary and NOT sufficient for a selection layer to bite.

On the T4 path, `expand` keeps EVERY eligible endpoint: it builds each variant, calls
`fiber.check`, and on a pass writes `found[gate["smiles"]]`. No budget, no top-k, no
selection among eligible endpoints. So a law RANKING those endpoints cannot change what
comes back even though the ranked and gated molecules are provably identical. Ranking
only matters where something is DISCARDED.

So a component must be sited where capacity actually binds:

  GENERATION mixtures change what EXISTS and are never exposed to this failure.
  SELECTION layers (a credit allocator, a contextual bandit over
  `(G, region, prototype, scale)`) bite ONLY if the stage they sit at discards.

This census measures every discard site on the real path, at PRODUCTION settings, by
execution:

  1. per-channel proposal   attempts -> eligible rows, split by refusal reason, and
                            whether the channel stopped on CHANNEL_CANDIDATE_LIMIT
                            (capacity) or on `wall_seconds` (time)
  2. cross-channel dedup    pools -> merged
  3. ALLOCATION             merged -> selected, i.e. `candidates_per_batch`
  4. query lock             proposed -> locked

The number that decides where a bandit goes is `discarded_by_allocation`. If the merged
pool exceeds `candidates_per_batch`, allocation is capacity-limited and a selection layer
sited there is load-bearing. If it does not, the oracle budget is the only real
bottleneck and the bandit belongs at query selection, not at proposal ranking.
"""

from __future__ import annotations

import argparse
import collections
import json
from dataclasses import replace
from pathlib import Path

from rdkit import Chem, RDLogger

from compose_v4.control.adaptive_program_optimizer import ProgramSearchConfig
from compose_v4.control.dynamic_program_synthesis_v21 import (
    CHANNEL_CANDIDATE_LIMIT,
    initial_dynamic_program_batch_v21,
)
from compose_v4.control.pmo_population_controller import PmoPopulationController
from compose_v4.experiments.editing_v2_evaluation_semantics import (
    production_state_from_smiles,
)

RDLogger.DisableLog("rdApp.*")
ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = json.loads(
    (ROOT / "diagnostics/pmo_joint_dependency_jump_gate_v1/attempt_2/checkpoints.json").read_text()
)["payload"]["checkpoints"]["shared_all_routes"]

#: Production PMO settings (`experiments/pmo_population_v1.py`). Measuring at smaller
#: settings would understate the pool and therefore understate the discard.
ATTEMPTS_PER_BATCH = 128
QUERIES_PER_ROUND = 16
WALL_SECONDS = 45.0


def _eligibility(row):
    return {"oracle_eligible": Chem.MolFromSmiles(row["smiles"]) is not None}


def run(*, smiles: str, seed: int, rounds: int) -> dict:
    source = production_state_from_smiles(smiles, max_atoms=48)
    config = replace(
        ProgramSearchConfig.program_only_recipe(seed=seed, score_direction="maximize"),
        attempts_per_batch=ATTEMPTS_PER_BATCH,
        candidates_per_batch=QUERIES_PER_ROUND,
        wall_seconds=WALL_SECONDS,
        parent_allocation="niche_score",
    )
    batch = initial_dynamic_program_batch_v21(
        source, (), config,
        source_group="discard-census", oracle_protocol="free-no-oracle",
        eligibility=_eligibility,
    )
    controller = PmoPopulationController(
        config, source_group="discard-census", oracle_protocol="free-no-oracle",
        hierarchy=None, jump_checkpoint=CHECKPOINT, enable_online_memory=True,
    )
    for index, candidate in enumerate(batch["candidates"][:8]):
        controller.add_measured_program(candidate, receipt_id=f"seed{index}", score=0.5)

    rows = []
    for round_index in range(rounds):
        proposed = controller.propose_batch(_eligibility)
        # NOTE the key: `propose_batch` publishes the full pool as `eligible_pool`.
        # `proposal_pool` is the LOCKED batch's key, set later by `lock_query_subset`;
        # reading that name here raises rather than silently measuring the wrong set.
        pool = proposed["eligible_pool"]["candidates"]
        selected = proposed["candidates"]
        allocation = proposed["allocation"]
        statuses = collections.Counter(a["status"] for a in proposed.get("attempts", []))
        by_channel = allocation.get("available_by_channel", {})
        # A channel that stopped AT the limit was capacity-bound; below it, time- or
        # rejection-bound. The distinction decides whether more search would help.
        at_capacity = {
            channel: count >= CHANNEL_CANDIDATE_LIMIT for channel, count in by_channel.items()
        }
        ids = [row["candidate_id"] for row in selected]
        locked = controller.lock_query_subset(
            proposed["batch_id"], ids,
            {"policy": "discard_census_all", "selected_ids": ids},
        )
        rows.append({
            "round": round_index,
            "attempt_statuses": dict(statuses),
            "pool_by_channel": by_channel,
            "channel_at_candidate_limit": at_capacity,
            "merged_pool": len(pool),
            "candidates_per_batch": QUERIES_PER_ROUND,
            "selected": len(selected),
            "discarded_by_allocation": len(pool) - len(selected),
            "allocation_is_capacity_limited": len(pool) > QUERIES_PER_ROUND,
            "proposed_to_lock": len(selected),
            "locked": len(locked["candidates"]),
            "discarded_by_lock": len(selected) - len(locked["candidates"]),
            "allocation_mode": allocation.get("mode"),
            "credit_cells_available": allocation.get("credit_allocation", {}).get(
                "cells_available"
            ),
        })
        # Score everything locked so the next round is a real continuation.
        receipts = {r["candidate_id"]: f"r{round_index}-{r['candidate_id'][:10]}"
                    for r in locked["candidates"]}
        controller.observe_batch(
            locked["batch_id"],
            [
                {"candidate_id": r["candidate_id"], "receipt_id": receipts[r["candidate_id"]],
                 "score": 0.5, "oracle_protocol": controller.oracle_protocol}
                for r in locked["candidates"]
            ],
        )
    return {"source": smiles, "seed": seed, "rounds": rows}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--seeds", type=int, nargs="+", default=[11])
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    sources = ("CC(=O)Nc1ccc(O)cc1", "O=C(NC1CCNCC1)c1ccccc1")
    results = []
    for smiles in sources:
        for seed in args.seeds:
            row = run(smiles=smiles, seed=seed, rounds=args.rounds)
            results.append(row)
            for entry in row["rounds"]:
                print(
                    f"{smiles[:22]:<24} seed={seed} r{entry['round']} "
                    f"pool={entry['merged_pool']:<3} selected={entry['selected']:<3} "
                    f"DISCARDED_BY_ALLOCATION={entry['discarded_by_allocation']:<3} "
                    f"lock_discard={entry['discarded_by_lock']} "
                    f"by_channel={entry['pool_by_channel']}",
                    flush=True,
                )
    every = [entry for row in results for entry in row["rounds"]]
    capacity_limited = sum(e["allocation_is_capacity_limited"] for e in every)
    total_discarded = sum(e["discarded_by_allocation"] for e in every)
    lock_discarded = sum(e["discarded_by_lock"] for e in every)
    verdict = (
        "ALLOCATION_IS_CAPACITY_LIMITED_SELECTION_BITES"
        if capacity_limited == len(every) and total_discarded > 0
        else "ALLOCATION_SOMETIMES_CAPACITY_LIMITED"
        if capacity_limited
        else "ALLOCATION_INERT_ORACLE_BUDGET_IS_THE_ONLY_BOTTLENECK"
    )
    report = {
        "schema_version": "pmo_candidate_discard_census_v1",
        "oracle_calls": 0,
        "settings": {
            "attempts_per_batch": ATTEMPTS_PER_BATCH,
            "candidates_per_batch": QUERIES_PER_ROUND,
            "wall_seconds": WALL_SECONDS,
            "channel_candidate_limit": CHANNEL_CANDIDATE_LIMIT,
        },
        "rounds_measured": len(every),
        "rounds_where_allocation_was_capacity_limited": capacity_limited,
        "total_discarded_by_allocation": total_discarded,
        "total_discarded_by_lock": lock_discarded,
        "verdict": verdict,
        "results": results,
    }
    print(
        f"\nrounds={len(every)}  capacity-limited={capacity_limited}"
        f"  discarded_by_allocation={total_discarded}"
        f"  discarded_by_lock={lock_discarded}"
    )
    print(f"VERDICT: {verdict}")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + "\n")
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
