"""Measure the eligible-endpoint yield curve for fa7_0 at delta=0.6 against draw budget.

WHY
---
`fa7_0` terminated live at `candidate_exhaustion` after one charged call -- the
seed's own docking score -- because the campaign app ends a cell the first time
selection returns nothing.  Whether escalating the proposal draw budget is the
right lever is an empirical question about THIS cell, and it is answerable with
zero oracle calls: the eligibility gate is `Fiber.check`, which is local.

WHAT IT RUNS
------------
The production proposal path, `t4_fiber_campaign.expand`, on the real fa7_0 seed
at the contract's horizon, with the contract's region law and completion law
resolved through their own contract surfaces -- not reconstructed here, so a
drift in either surface shows up as a failure rather than as a quietly different
arm.

It reports, per lane and per draw budget, how many DISTINCT eligible endpoints
appeared, and whether each retains the amidine (the FA7 S1-pocket pharmacophore
the unconditioned path was measured to delete).  Benchmark eligibility and
chemical plausibility are reported as two numbers, never one.

ZERO oracle calls, zero docking, zero Modal.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from rdkit import Chem, RDLogger

from compose_v4.control.completion_law_contract import completion_law_for_proposal_lane
from compose_v4.control.region_law_contract import region_law_for_proposal_lane
from compose_v4.experiments.t4_fiber_campaign import Fiber, expand

RDLogger.DisableLog("rdApp.*")

#: The amidine of the fa7 seed: the S1-pocket pharmacophore.  An endpoint that
#: deletes it can still pass every benchmark threshold, which is exactly why
#: this is reported beside the count rather than folded into it.
AMIDINE = Chem.MolFromSmarts("[CX3](=[NX2])[NX3]")


def _retains_amidine(smiles: str) -> bool:
    mol = Chem.MolFromSmiles(smiles)
    return bool(mol is not None and mol.HasSubstructMatch(AMIDINE))


def _laws(contract: dict, lane: str, seed_smiles: str, fiber: Fiber, horizon: int):
    """Resolve both laws through their production contract surfaces."""

    def _probe(law, attempt):
        return expand(
            seed_smiles,
            0.0,
            fiber,
            np.random.default_rng(90_000 + attempt),
            draws=1,
            multi_region=False,
            horizon=horizon,
            proposal_lane="shallow",
            region_law=law,
        )

    def _completion_probe(law, attempt):
        return expand(
            seed_smiles,
            0.0,
            fiber,
            np.random.default_rng(91_000 + attempt),
            draws=1,
            multi_region=False,
            horizon=horizon,
            proposal_lane="shallow",
            completion_law=law,
        )

    region_law, region_telemetry = region_law_for_proposal_lane(
        contract,
        lane=lane,
        delta=contract["delta"],
        reference_smiles=seed_smiles,
        draw=_probe,
    )
    completion_law, completion_telemetry = completion_law_for_proposal_lane(
        contract,
        lane=lane,
        delta=contract["delta"],
        reference_smiles=seed_smiles,
        draw=_completion_probe,
    )
    return region_law, completion_law, {**region_telemetry, **completion_telemetry}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", required=True)
    parser.add_argument("--cell", default="fa7_0")
    parser.add_argument("--lane", default="shallow", choices=["shallow", "anchored_replacement"])
    parser.add_argument("--draws", type=int, nargs="+", default=[480, 960, 1920])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    contract = json.loads(Path(args.contract).read_text())["payload"]
    cell = next(row for row in contract["cells"] if row["cell"] == args.cell)
    seed_smiles = cell["smiles"]
    horizon = contract["proposal"][args.lane]["horizon"]
    fiber = Fiber(seed_smiles, contract["delta"], support=contract["support"])
    region_law, completion_law, telemetry = _laws(
        contract, args.lane, seed_smiles, fiber, horizon
    )
    print(f"laws for lane={args.lane}: {json.dumps(telemetry, sort_keys=True)}", flush=True)

    rows = []
    for draws in args.draws:
        for seed_index in args.seeds:
            proposal_seed = int(cell["controller_seed"] + 1_000_003 * (seed_index + 1))
            started = time.time()
            records = expand(
                seed_smiles,
                -7.5,
                fiber,
                np.random.default_rng(proposal_seed),
                draws=draws,
                multi_region=True,
                horizon=horizon,
                proposal_lane=args.lane,
                region_law=region_law,
                completion_law=completion_law,
            )
            endpoints = sorted({row["smiles"] for row in records})
            retaining = [smiles for smiles in endpoints if _retains_amidine(smiles)]
            row = {
                "lane": args.lane,
                "draws": draws,
                "seed_index": seed_index,
                "proposal_seed": proposal_seed,
                "eligible_distinct": len(endpoints),
                "eligible_retaining_amidine": len(retaining),
                "elapsed_seconds": round(time.time() - started, 2),
                "endpoints": [
                    {
                        "smiles": record["smiles"],
                        # `Fiber.check` returns `qed` and `sa`. An earlier version of
                        # this script read `quality`/`access`, which would have raised
                        # only once an eligible endpoint was actually found -- i.e.
                        # exactly when the probe mattered. Assert the shape instead.
                        "similarity": round(float(record["similarity"]), 4),
                        "qed": round(float(record["qed"]), 4),
                        "sa": round(float(record["sa"]), 4),
                        "retains_amidine": _retains_amidine(record["smiles"]),
                    }
                    for record in sorted(records, key=lambda r: r["smiles"])
                ],
            }
            rows.append(row)
            print(
                f"[{args.lane}] draws={draws} seed={seed_index} "
                f"eligible={row['eligible_distinct']} "
                f"amidine={row['eligible_retaining_amidine']} "
                f"{row['elapsed_seconds']}s",
                flush=True,
            )
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            Path(args.out).write_text(
                json.dumps(
                    {
                        "schema_version": "t4_fa7_0_support_expansion_yield_v1",
                        "contract": args.contract,
                        "cell": args.cell,
                        "lane": args.lane,
                        "law_telemetry": telemetry,
                        "oracle_calls": 0,
                        "rows": rows,
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
