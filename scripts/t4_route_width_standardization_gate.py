"""Does standardizing the route lane to 48/48/64 preserve PARP1/BRAF route support?

THE QUESTION, AND WHY IT IS A MEASUREMENT
------------------------------------------
Five T4 contracts configure `proposal.route_complete_region` two ways, split by
protein: parp1/braf at `beam_width 32 / expansion_width 24 / realization_limit 32`
and jak2/fa7/5ht1b at `48 / 48 / 64`.  Those values are EXECUTABLE -- they go
straight into `propose_route_expert_candidates` -- so the panel currently runs two
route searches keyed on target identity, which is the "family of controllers"
failure the routing and the frozen ladder exist to remove.

The owner's decision is to standardize on 48/48/64.  The tempting argument is that
a wider search must contain a narrower one, so parp1/braf can only gain.  **That
argument is wrong for a beam.**  Widening `beam_width` changes which partial
programs survive each layer and widening `expansion_width` changes how many
successors each survivor contributes, so the set of complete programs reaching
realization can move rather than grow.  Nesting has to be measured.

WHAT IS HELD FIXED
------------------
The cell and its docked root, the per-protein route expert checkpoint, the
production `Fiber` (delta read from the CONTRACT), the production round-one
proposal seed, and every other route-lane field.  Only the three widths differ.

The route lane takes no RNG -- `propose_route_expert_candidates` has no `rng`
parameter -- so given a state and its settings the lane is DETERMINISTIC and the
comparison carries no sampling noise at all.  The seed is still derived exactly as
production derives it, so a future lane that did consume RNG would be compared
fairly rather than silently.

WHAT IS SPENT
-------------
CPU.  No oracle call, no docking.  The lane is gated by the unmodified production
`Fiber.check`, which is pure RDKit.

THE CRITERIA ARE SEALED, NOT CHOSEN AFTERWARDS
------------------------------------------------
`diagnostics/t4_route_width_standardization/predeclaration_v1.json` was committed
before this script produced a number, and this driver refuses to emit a verdict if
that file's content hash has moved.  Selecting the other width after seeing the
result, or quietly retaining per-target widths, is exactly what the seal prevents.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

from compose_v4.control.docking_value import identity
from compose_v4.experiments.t4_fiber_campaign import Fiber
from compose_v4.experiments.t4_integrated_route_fiber import EXPERTS
from compose_v4.experiments.t4_matched_pilot import unseal
from compose_v4.experiments.t4_unified_proposal import proposal_unit

SCHEMA_VERSION = "t4_route_width_standardization_gate_v1"

ROOT = Path(__file__).resolve().parents[1]
LANE = "route_complete_region"

PREDECLARATION = "diagnostics/t4_route_width_standardization/predeclaration_v1.json"
#: The hash of the sealed criteria. A driver that cannot verify its own
#: predeclaration must not publish a verdict.
PREDECLARATION_SHA256 = (
    "82e356a047331a440faef128b35da23bab059c433b7d6094b39f9c3a203c78c0"
)

#: The two arms. Everything not named here is read from the cell's own contract,
#: so a field that is not under test cannot drift between arms.
ARMS = {
    "narrow": {"beam_width": 32, "expansion_width": 24, "realization_limit": 32},
    "wide": {"beam_width": 48, "expansion_width": 48, "realization_limit": 64},
}

#: PARP1 and BRAF only: they are the targets the decision moves.
CELLS = ("parp1_0", "parp1_1", "parp1_2", "braf_0", "braf_1", "braf_2")

#: C4's bound. A third of the 1800 s production proposal-worker timeout.
ELAPSED_BUDGET_SECONDS = 600.0


def protein_of(cell: str) -> str:
    return cell.rsplit("_", 1)[0]


def load_cell(cell: str, *, root: Path = ROOT) -> tuple[dict, dict]:
    """The unified contract payload and the cell row. Delta from the CONTRACT."""

    payload = unseal(root / f"configs/t4_unified_controller_{protein_of(cell)}_d06_v1.json")
    row = next(item for item in payload["cells"] if item["cell"] == cell)
    return payload, row


def route_expert(cell: str, *, root: Path = ROOT):
    relative = (
        "diagnostics/t4_held_target_distillation_quality_v1/"
        f"{protein_of(cell)}_checkpoint.json"
    )
    envelope = json.loads((root / relative).read_text())
    if identity(envelope["payload"]) != envelope["payload_sha256"]:
        raise ValueError(f"{relative} payload hash does not verify")
    from compose_v4.control.route_distilled_goal_expert import RouteDistilledGoalExpert

    return RouteDistilledGoalExpert.from_checkpoint(envelope["payload"]["expert"])


def assert_predeclaration_intact(*, root: Path = ROOT) -> str:
    """Refuse to run against criteria that moved after they were sealed."""

    payload = json.loads((root / PREDECLARATION).read_text())
    actual = identity(payload)
    if actual != PREDECLARATION_SHA256:
        raise ValueError(
            f"{PREDECLARATION} hashes to {actual}, not the sealed "
            f"{PREDECLARATION_SHA256}; the criteria moved and no verdict may be "
            "published against them"
        )
    return actual


def run_arm(cell: str, arm: str, *, expert, root: Path = ROOT) -> dict:
    """One arm on one cell, through the PRODUCTION proposal unit."""

    payload, row = load_cell(cell, root=root)
    contract = {
        **payload,
        "proposal": {
            **payload["proposal"],
            LANE: {**payload["proposal"][LANE], **ARMS[arm]},
        },
    }
    fiber = Fiber(row["smiles"], payload["delta"], support=payload["support"])
    seed = int(
        row["controller_seed"]
        + 1_000_003 * 1
        + 10_007 * 0
        + 101 * EXPERTS.index(LANE)
    )
    started = time.time()
    answer = proposal_unit(
        contract,
        {
            "expert": LANE,
            "parent": row["smiles"],
            "parent_score": 0.0,
            "original_seed": row["smiles"],
            "proposal_seed": seed,
        },
        fiber=fiber,
        route_expert=expert,
    )
    elapsed = time.time() - started
    eligible = sorted({record["smiles"] for record in answer["records"]})
    return {
        "cell": cell,
        "arm": arm,
        "settings": {k: contract["proposal"][LANE][k] for k in sorted(ARMS[arm])},
        "delta_from_contract": payload["delta"],
        "proposal_seed": seed,
        "eligible_count": len(eligible),
        "eligible_smiles": eligible,
        "records_returned": answer["eligible_records_returned"],
        "telemetry": answer["telemetry"],
        "elapsed_seconds": round(elapsed, 2),
    }


def compare(cell: str, *, root: Path = ROOT) -> dict:
    """Both arms on one cell, and the containment relation between them."""

    expert = route_expert(cell, root=root)
    narrow = run_arm(cell, "narrow", expert=expert, root=root)
    wide = run_arm(cell, "wide", expert=expert, root=root)
    lost = sorted(set(narrow["eligible_smiles"]) - set(wide["eligible_smiles"]))
    gained = sorted(set(wide["eligible_smiles"]) - set(narrow["eligible_smiles"]))
    return {
        "cell": cell,
        "narrow": narrow,
        "wide": wide,
        "wide_contains_narrow": not lost,
        "lost_count": len(lost),
        "lost_smiles": lost,
        "gained_count": len(gained),
        "gained_smiles": gained,
        "yield_delta": wide["eligible_count"] - narrow["eligible_count"],
        "c1_recovery": not lost,
        "c2_yield": wide["eligible_count"] >= narrow["eligible_count"],
        "c4_cost": wide["elapsed_seconds"] <= ELAPSED_BUDGET_SECONDS,
    }


def reduce_cells(rows: list[dict], *, root: Path = ROOT) -> dict:
    """The verdict, against the SEALED criteria and nothing else."""

    if len(rows) != len(CELLS):
        raise ValueError(
            f"a verdict needs all {len(CELLS)} cells, got {len(rows)}; an incomplete "
            "comparison must not decide a frozen configuration"
        )
    c1 = all(row["c1_recovery"] for row in rows)
    c2 = all(row["c2_yield"] for row in rows)
    c4 = all(row["c4_cost"] for row in rows)
    green = c1 and c2 and c4
    return {
        "schema_version": SCHEMA_VERSION,
        "predeclaration": PREDECLARATION,
        "predeclaration_sha256": assert_predeclaration_intact(root=root),
        "question": (
            "does standardizing the route lane to 48/48/64 preserve the PARP1/BRAF "
            "route support that 32/24/32 produced?"
        ),
        "arms": ARMS,
        "lane_is_deterministic": (
            "propose_route_expert_candidates takes no rng, so given a state and its "
            "settings this lane carries no sampling noise"
        ),
        "criteria": {
            "C1_recovery_wide_contains_narrow": c1,
            "C2_yield_never_drops": c2,
            "C4_cost_within_600s_per_cell": c4,
        },
        "verdict": "GREEN" if green else "NOT_GREEN",
        "cells": {row["cell"]: row for row in rows},
        "totals": {
            "narrow_eligible": sum(row["narrow"]["eligible_count"] for row in rows),
            "wide_eligible": sum(row["wide"]["eligible_count"] for row in rows),
            "lost_total": sum(row["lost_count"] for row in rows),
            "gained_total": sum(row["gained_count"] for row in rows),
            "narrow_seconds": round(
                sum(row["narrow"]["elapsed_seconds"] for row in rows), 2
            ),
            "wide_seconds": round(sum(row["wide"]["elapsed_seconds"] for row in rows), 2),
        },
        "oracle_calls": 0,
        "docking_calls": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--cells", default="")
    args = parser.parse_args()

    assert_predeclaration_intact()
    cells = [name for name in args.cells.split(",") if name] or list(CELLS)
    rows = []
    for cell in cells:
        row = compare(cell)
        rows.append(row)
        print(
            f"{cell:9s} narrow={row['narrow']['eligible_count']:3d} "
            f"({row['narrow']['elapsed_seconds']:6.1f}s)  "
            f"wide={row['wide']['eligible_count']:3d} "
            f"({row['wide']['elapsed_seconds']:6.1f}s)  "
            f"lost={row['lost_count']} gained={row['gained_count']} "
            f"contains={row['wide_contains_narrow']}",
            flush=True,
        )
    payload = reduce_cells(rows)
    args.destination.parent.mkdir(parents=True, exist_ok=True)
    args.destination.write_text(
        json.dumps(
            {"payload": payload, "payload_sha256": identity(payload)},
            indent=1,
            sort_keys=True,
        )
    )
    print(json.dumps({**payload["criteria"], "verdict": payload["verdict"],
                      **payload["totals"]}, indent=1), flush=True)


if __name__ == "__main__":  # pragma: no cover - CLI
    main()
