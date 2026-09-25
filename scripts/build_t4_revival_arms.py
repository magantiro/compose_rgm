"""The nitya-workspace revival of the T4 replicate cells the rahul panel cannot finish.

WHAT IS BEING REVIVED, AND WHY IT IS SAFE
-----------------------------------------
Four cells of the ten-arm replicate panel died inside the round-0 preemption
window: they published `round_000_lock.json` and were preempted before the first
checkpoint, so `_resume_state` raises "an unfinished query lock exists with no
recoverable checkpoint".  With `retries: 0` and that guard, the rahul-94866
`run_cell` container for each is PERMANENTLY unable to write again -- a restarted
container raises on the same guard.  So these four cells, and ONLY these four,
have no live writer and can be re-run elsewhere without a divergent duplicate.

    fa7  d0.4   fa7_1_r3, fa7_2_r2, fa7_2_r3
    braf d0.4   braf_2_r3

MEASURED, not assumed (`/tmp/t4arms.json`, and the locks themselves): each dead
cell holds exactly one `round_000_lock.json` carrying ONE query and no
`charged_before`, and no `checkpoint.json` or `result.json`.  Every other cell of
the panel holds a checkpoint or a result and is therefore live, and fourteen
further cells are QUEUED on rahul -- their `run_cell` calls exist and will take a
container as slots free -- so those are explicitly NOT revived here.  A queued
cell is racy; a dead cell is not.

BUDGET
------
That one root query is of uncertain resolution, and this repository's rule is
that an uncertain call must not become a free call.  So each revived cell is
debited 1 against its authorized 250 and the arm ceiling is DERIVED as the sum,
never written down.  `charged_calls_per_cell` stays at 250 because it is a
panel-uniformity field; the CEILING carries the debit, which is exactly the
mechanism the contracts' own `replicate_policy.budget_note` already describes for
the seven historical rows at 249/248.  Lifetime spend per revived cell is
therefore 1 (rahul) + 249 (nitya) = 250, the authorized figure exactly.

SEPARATE ARTIFACTS ON PURPOSE
-----------------------------
The revival writes to its own volume, app and output namespace (`-rev`).  A Modal
volume name is scoped to a workspace, so reusing the rahul names would have
produced two distinct volumes with one name -- a confusion this repository has
already paid for twice.  The names are made distinct so reconciliation must state
which workspace it read.

This tool does NOT authorize a launch.  It writes
`status = FROZEN_PENDING_OWNER_LAUNCH_AUTHORIZATION` with both authorization
booleans false, exactly as the panel sealer does.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from compose_v4.control.docking_value import identity
from compose_v4.experiments.continuation_profile import sha256_file

#: (target, delta tag) -> {cell: charged calls already spent on rahul}.
#: Measured from each cell's own `round_000_lock.json`: one query, no
#: `charged_before`, no checkpoint.
REVIVE: dict[tuple[str, str], dict[str, int]] = {
    # WAVE 1, already running on nitya.
    ("fa7", "d04"): {"fa7_1_r3": 1, "fa7_2_r2": 1, "fa7_2_r3": 1},
    ("braf", "d04"): {"braf_2_r3": 1},
    # WAVE 2.  Eleven further cells died in the same round-0 window and sat idle for
    # 362-764 minutes: one round_000_lock.json, one query, no checkpoint, no result.
    # Measured per cell rather than assumed; every one debits exactly 1.
    ("5ht1b", "d04"): {"5ht1b_2_r2": 1, "5ht1b_2_r3": 1},
    ("jak2", "d04"): {"jak2_2_r2": 1, "jak2_2_r3": 1},
    ("jak2", "d06"): {"jak2_2_r2": 1, "jak2_2_r3": 1},
    ("parp1", "d04"): {"parp1_2_r3": 1},
    ("parp1", "d06"): {"parp1_1_r3": 1, "parp1_2_r2": 1, "parp1_2_r3": 1},
    ("braf", "d06"): {"braf_2_r3": 1},
}

CHARGED_CALLS_PER_CELL = 250

WRAPPER_TEMPLATE = '''"""Revival launcher for {target} at delta={delta}: the round-0-preempted cells.

Sets only the arm's identity. The CONTROLLER is byte-identical to every other arm
and to the live rahul-94866 panel -- `modal_apps/t4_unified_controller_app.py` is
unchanged from 4a600557 -- so the only thing that differs is WHICH cells run and
WHERE the artifacts land.

Pinned in its contract's `runtime_inputs_sha256` and baked into the image, because
`_validate_task` re-hashes every pinned entry inside the container and the wrapper
is what selects the arm.
"""

from __future__ import annotations

import os

os.environ.setdefault(
    "COMPOSE_HELD_CONTRACT",
    "configs/t4_unified_controller_{target}_{tag}_rev_v1.json",
)
os.environ.setdefault(
    "COMPOSE_HELD_CHECKPOINT",
    "{checkpoint}",
)
os.environ.setdefault("COMPOSE_HELD_VOLUME", "{volume}")
os.environ.setdefault("COMPOSE_HELD_OUTPUT", "{output}")
os.environ.setdefault("COMPOSE_HELD_APP", "{appname}")
os.environ.setdefault("COMPOSE_HELD_RECEPTOR_NAME", "{target}")
os.environ.setdefault(
    "COMPOSE_HELD_WRAPPER",
    "modal_apps/t4_unified_controller_{target}_{tag}_rev_app.py",
)

from modal_apps.t4_unified_controller_app import app, main  # noqa: F401
'''


def _payload(path: Path) -> dict:
    return json.loads(path.read_text())["payload"]


def main() -> None:
    total = 0
    for (target, tag), spend in sorted(REVIVE.items()):
        source_path = ROOT / f"configs/t4_unified_controller_{target}_{tag}_r23_v1.json"
        base = _payload(source_path)
        source_sha = json.loads(source_path.read_text())["payload_sha256"]
        delta = base["delta"]

        volume = f"compose-t4-unified-controller-{target}-rev-v1"
        output = f"/unified_{target}_rev"
        appname = f"compose-t4-unified-controller-{target}-{tag}-rev-v1"
        checkpoint = base["unified_controller"]["round_one_route_expert"]

        wrapper = ROOT / f"modal_apps/t4_unified_controller_{target}_{tag}_rev_app.py"
        wrapper.write_text(
            WRAPPER_TEMPLATE.format(
                target=target,
                tag=tag,
                delta=delta,
                checkpoint=checkpoint,
                volume=volume,
                output=output,
                appname=appname,
            )
        )

        cells = [row for row in base["cells"] if row["cell"] in spend]
        missing = set(spend) - {row["cell"] for row in cells}
        if missing:
            raise SystemExit(f"{target} {tag}: cells absent from the source arm: {sorted(missing)}")

        payload = dict(base)
        payload["cells"] = cells
        payload["charged_calls_per_cell"] = CHARGED_CALLS_PER_CELL
        payload["total_charged_call_ceiling"] = sum(
            CHARGED_CALLS_PER_CELL - spend[row["cell"]] for row in cells
        )
        payload["schema_version"] = "t4_unified_controller_revival_contract_v1"
        payload["scored_launch_authorized"] = False
        payload["modal_launch_authorized"] = False
        payload["status"] = "FROZEN_PENDING_OWNER_LAUNCH_AUTHORIZATION"
        payload.pop("authorization", None)
        payload["revival_policy"] = {
            "revives": sorted(spend),
            "workspace": "nitya",
            "source_arm": str(source_path.relative_to(ROOT)),
            "source_arm_payload_sha256": source_sha,
            "source_arm_workspace": "rahul-94866",
            "why_these_cells_are_free": (
                "Each holds exactly one round_000_lock.json with one query and no "
                "checkpoint.json, so t4_unified_controller_app._resume_state raises "
                "'an unfinished query lock exists with no recoverable checkpoint'. "
                "With retries: 0 and that guard, a restarted rahul-94866 container "
                "raises on the same guard and can never write the cell again. No "
                "other cell of the panel is revived: fourteen are QUEUED on rahul "
                "with live run_cell calls and would produce divergent duplicates."
            ),
            "prior_charged_calls_debited": dict(sorted(spend.items())),
            "prior_charge_evidence": (
                "the cell's own round_000_lock.json; the query's resolution is "
                "uncertain and an uncertain call is charged, never freed"
            ),
            "lifetime_charged_calls_per_cell": CHARGED_CALLS_PER_CELL,
            "ceiling_rule": (
                "DERIVED as sum over cells of (250 - prior spend), never written "
                "down; charged_calls_per_cell stays at the panel-uniform 250 and "
                "the ceiling carries the debit, the same mechanism "
                "replicate_policy.budget_note describes for the historical 249/248 rows"
            ),
            "controller_bytes_unchanged": (
                "modal_apps/t4_unified_controller_app.py is byte-identical to the "
                "live rahul-94866 arms at 4a600557; only the cell set, the output "
                "namespace and the workspace differ"
            ),
            "separate_namespace_reason": (
                "a Modal volume name is workspace-scoped, so reusing the rahul names "
                "would give two distinct volumes one name; -rev keeps them distinct"
            ),
        }
        payload["claim_boundary"] = (
            base["claim_boundary"]
            + " REVIVAL ARM: the subset of replicate cells lost to the round-0 "
            "preemption window, re-run on the nitya workspace under byte-identical "
            "controller bytes with each cell's one prior charged call debited. "
            "Report these rows as part of the replicate panel, naming the revival."
        )

        pins = dict(base["runtime_inputs_sha256"])
        pins.pop(f"modal_apps/t4_unified_controller_{target}_{tag}_r23_app.py", None)
        pins[str(wrapper.relative_to(ROOT))] = ""
        payload["runtime_inputs_sha256"] = {
            relative: sha256_file(ROOT / relative) for relative in sorted(pins)
        }

        destination = ROOT / f"configs/t4_unified_controller_{target}_{tag}_rev_v1.json"
        destination.write_text(
            json.dumps(
                {"payload": payload, "payload_sha256": identity(payload)},
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        total += payload["total_charged_call_ceiling"]
        print(
            f"{destination.name:52s} delta={delta} cells={len(cells)} "
            f"ceiling={payload['total_charged_call_ceiling']} "
            f"sha256={identity(payload)[:8]}"
        )
    print(f"TOTAL DERIVED CEILING FOR THE REVIVAL = {total}")


if __name__ == "__main__":
    main()
