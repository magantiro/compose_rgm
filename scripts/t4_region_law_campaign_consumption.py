"""OFF vs ON on the CAMPAIGN's own proposal path, on real exhausted T4 cells.

WHAT THIS ANSWERS
-----------------
``scripts/t4_bridge_region_law_gate.py`` measured the repaired law by walking
``BridgeRegionLaw`` directly: it proved the law re-ranks regions.  It did NOT
run the campaign, so it could not show that the campaign consumes the law --
and on the launch branch it did not, because no production caller passed
``region_law=``.

This driver closes that gap.  It calls ``t4_fiber_campaign.expand`` -- the exact
function the Modal proposal worker calls, on the exact lane, with the draws and
horizon the cell's own signed contract declares -- once with the contract field
ABSENT and once with it PRESENT, from the same seed.  The only difference
between the two arms is the contract field.

ZERO ORACLE CALLS.  ``Fiber.check`` is free: similarity, QED, SA and the
structural gate.  Nothing here docks, and no oracle or receptor is loaded.

WHAT TO READ
------------
``eligible`` is the count of distinct endpoints passing the production
``Fiber.check`` -- the same gate whose empty result is reported live as
``candidate_exhaustion``.  ``max_excision`` is the largest single-module heavy
atom loss the lane reached, which is the axis v1's eight-atom cap closes and the
axis every known witness for these cells needs.

The control cells are reported beside the failed ones on purpose: a repair that
rescues a dead cell by breaking a working one is not a repair.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from compose_v4.chem.molecular_graph import smiles_to_molecular_graph
from compose_v4.chem.state import pad_molecular_graph
from compose_v4.control.region_law_contract import (
    CONTRACT_FIELD,
    CONTRACT_LANE,
    FREE_GATE_MARGIN_V1,
    assert_region_law_is_consumed,
    resolve_region_law,
)
from compose_v4.experiments.t4_fiber_campaign import Fiber, expand
from compose_v4.experiments.t4_matched_pilot import unseal

SCHEMA_VERSION = "t4_region_law_campaign_consumption_v1"

#: The T4 proposal executor requires 48 slots, exactly as `expand` builds them.
PROPOSAL_SLOTS = 48

#: Cells that terminated live at `candidate_exhaustion`, and the sibling
#: controls that searched normally. Read from the contract, never inferred.
FAILED = {"braf_0", "braf_1", "fa7_0", "fa7_2"}


def _cell_rows(payload: dict, wanted: tuple[str, ...] | None) -> list[dict]:
    rows = payload["cells"]
    if wanted:
        by_name = {row["cell"]: row for row in rows}
        missing = [name for name in wanted if name not in by_name]
        if missing:
            raise ValueError(f"contract has no cells {missing}")
        return [by_name[name] for name in wanted]
    return list(rows)


def _with_region_law(payload: dict, request) -> dict:
    """The same contract with the field switched on; nothing else moves."""

    mutated = json.loads(json.dumps(payload))
    mutated["proposal"][CONTRACT_LANE][CONTRACT_FIELD] = request
    return mutated


def _arm(cell: dict, payload: dict, *, region_law, draws: int, seed: int) -> dict:
    lane = payload["proposal"][CONTRACT_LANE]
    source = pad_molecular_graph(
        smiles_to_molecular_graph(cell["smiles"]), PROPOSAL_SLOTS
    )
    fiber = Fiber(cell["smiles"], payload["delta"], support=payload["support"])
    started = time.time()
    records = expand(
        cell["smiles"],
        0.0,
        fiber,
        np.random.default_rng(seed),
        draws=draws,
        multi_region=True,
        horizon=lane["horizon"],
        proposal_lane=CONTRACT_LANE,
        region_law=region_law,
    )
    heavy = int(source.n_real_atoms)
    excisions = [heavy - int(row["heavy"]) for row in records]
    return {
        "eligible": len(records),
        "max_excision": max(excisions, default=0),
        "excisions_over_v1_cap": sum(1 for value in excisions if value > 8),
        "best_similarity": max((row["similarity"] for row in records), default=None),
        "best_qed": max((row["qed"] for row in records), default=None),
        "example": min(
            (row for row in records),
            key=lambda row: -row["similarity"],
            default=None,
        ),
        "seconds": round(time.time() - started, 2),
    }


def run(contract_path: Path, *, cells, draws, seed_offset, out: Path) -> dict:
    payload = unseal(contract_path)
    declared_draws = payload["proposal"][CONTRACT_LANE]["draws"]
    used_draws = draws or declared_draws
    on_payload = _with_region_law(payload, FREE_GATE_MARGIN_V1)
    rows = []
    for cell in _cell_rows(payload, cells):
        seed = int(cell["controller_seed"]) + seed_offset
        off_law = resolve_region_law(
            payload, delta=payload["delta"], reference_smiles=cell["smiles"]
        )
        on_law = resolve_region_law(
            on_payload, delta=on_payload["delta"], reference_smiles=cell["smiles"]
        )
        assert off_law is None, "an absent field must resolve to no law"

        def _probe(law, attempt, _cell=cell, _payload=payload, _seed=seed):
            fiber = Fiber(
                _cell["smiles"], _payload["delta"], support=_payload["support"]
            )
            return expand(
                _cell["smiles"],
                0.0,
                fiber,
                np.random.default_rng(_seed + 1_000_003 * (attempt + 1)),
                draws=1,
                multi_region=False,
                horizon=_payload["proposal"][CONTRACT_LANE]["horizon"],
                proposal_lane=CONTRACT_LANE,
                region_law=law,
            )

        consumption_attempts = assert_region_law_is_consumed(_probe)
        off = _arm(cell, payload, region_law=None, draws=used_draws, seed=seed)
        on = _arm(cell, on_payload, region_law=on_law, draws=used_draws, seed=seed)
        rows.append(
            {
                "cell": cell["cell"],
                "status": "failed_live" if cell["cell"] in FAILED else "control",
                "delta": payload["delta"],
                "heavy_atoms": int(
                    pad_molecular_graph(
                        smiles_to_molecular_graph(cell["smiles"]), PROPOSAL_SLOTS
                    ).n_real_atoms
                ),
                "seed": seed,
                "region_law_consumption_attempts": consumption_attempts,
                "off": off,
                "on": on,
            }
        )
        row = rows[-1]
        print(
            f"{row['cell']:>8} {row['status']:>12}  "
            f"OFF eligible={off['eligible']:<5} maxcut={off['max_excision']:<3} "
            f"|  ON eligible={on['eligible']:<5} maxcut={on['max_excision']:<3} "
            f"over_cap={on['excisions_over_v1_cap']}",
            flush=True,
        )
    report = {
        "schema_version": SCHEMA_VERSION,
        "contract": str(contract_path),
        "contract_payload_sha256": json.loads(contract_path.read_text())[
            "payload_sha256"
        ],
        "oracle_calls": 0,
        "docking_calls": 0,
        "lane": CONTRACT_LANE,
        "declared_draws": declared_draws,
        "draws_used": used_draws,
        "region_law_request": FREE_GATE_MARGIN_V1,
        "cells": rows,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1, sort_keys=True))
    print(f"\nwrote {out}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--cell", action="append", default=None)
    parser.add_argument(
        "--draws",
        type=int,
        default=0,
        help="override the contract's declared shallow draws (0 = use the contract)",
    )
    parser.add_argument("--seed-offset", type=int, default=0)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    run(
        args.contract,
        cells=tuple(args.cell) if args.cell else None,
        draws=args.draws,
        seed_offset=args.seed_offset,
        out=args.out,
    )


if __name__ == "__main__":
    main()
