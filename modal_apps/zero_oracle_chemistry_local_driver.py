"""Local baseline + equivalence driver for the zero-oracle chemistry fan-out.

Three stages, all on this machine, all zero-oracle:

    baseline   call the PRODUCTION function (``t4_v2_feasibility_gate.run_cell``) over a
               set of cells exactly as the serial local script does, and time it
    decompose  run the same work through the harness's unit decomposition
               (``execute_item`` + the workload reducer) IN THIS PROCESS
    compare    canonical, timing-stripped diff of two artifacts, with the first
               disagreeing JSON path named

``baseline`` is the reference the whole speedup claim rests on, so it drives the real
production function rather than a transcription of it: if ``run_cell`` changes, this arm
changes with it and the comparison can actually fail.

    python modal_apps/zero_oracle_chemistry_local_driver.py baseline \
        --cells fa7_0,braf_2 --budget 600 --out /tmp/local_baseline.json
    python modal_apps/zero_oracle_chemistry_local_driver.py compare \
        --left /tmp/local_baseline.json --right /tmp/modal_merged.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modal_apps import zero_oracle_chemistry_workloads as workloads


def _gate():
    return workloads._t4_gate()


def stage_baseline(arguments: argparse.Namespace) -> dict:
    """The serial local run: the production ``run_cell``, one cell after another."""

    gate = _gate()
    audit = gate.load_v1()
    sources = gate.v1_sources(audit)
    cells = [cell for cell in arguments.cells.split(",") if cell] or list(
        gate.FAILED + gate.CONTROLS + gate.SANITY
    )
    arms = tuple(a for a in arguments.arms.split(",") if a) or tuple(gate.ARMS)
    payload: dict = {
        "schema_version": gate.law.SCHEMA_VERSION,
        "budget": arguments.budget,
        "seed": arguments.seed,
        "delta": arguments.delta,
        "chemistry_filter": True,
        "prune_pathological_parents": True,
        "cells": {},
    }
    timings: dict[str, float] = {}
    started = time.time()
    for cell in cells:
        cell_started = time.time()
        payload["cells"][cell] = gate.run_cell(
            cell,
            sources[cell],
            audit,
            budget=arguments.budget,
            seed=arguments.seed,
            ablate=not arguments.no_ablate,
            delta=arguments.delta,
            arms=arms,
            attribution=not arguments.no_attribution,
        )
        timings[cell] = round(time.time() - cell_started, 2)
        print(f"[baseline] {cell} {timings[cell]}s", flush=True)
    wall = round(time.time() - started, 2)
    print(f"[baseline] total {wall}s over {len(cells)} cells", flush=True)
    return {
        "artifact": payload,
        "timing": {
            "stage": "baseline_serial_local",
            "wall_seconds": wall,
            "per_cell_seconds": timings,
            "cells": len(cells),
        },
    }


def stage_decompose(arguments: argparse.Namespace) -> dict:
    """The harness decomposition, executed serially in THIS process.

    Same units the Modal map would run, so a disagreement against ``baseline`` is a
    decomposition bug and a disagreement against the Modal artifact is transport or
    environment -- the two failure modes are separated by construction.
    """

    definition = workloads.workload(arguments.workload)
    # The SAME parameter dict the Modal launcher passes; every planner absorbs the keys
    # it does not use, so the two sides cannot drift in what they plan.
    parameters = {
        "cells": tuple(c for c in arguments.cells.split(",") if c),
        "arms": tuple(a for a in arguments.arms.split(",") if a),
        "budget": arguments.budget,
        "seed": arguments.seed,
        "delta": arguments.delta,
        "ablate": not arguments.no_ablate,
        "attribution": not arguments.no_attribution,
        "granularity": arguments.granularity,
    }
    items = definition["plan"](**parameters)
    rows = []
    started = time.time()
    unit_seconds = 0.0
    for index, item in enumerate(items, start=1):
        row = workloads.execute_item(item)
        unit_seconds += float(row["wall_seconds"])
        rows.append(row)
        print(
            f"[decompose] {index}/{len(items)} {item['kind']} {item.get('cell')} "
            f"{row['wall_seconds']}s",
            flush=True,
        )
    wall = round(time.time() - started, 2)
    merged = definition["reduce"](items, rows)
    print(f"[decompose] total {wall}s over {len(items)} units", flush=True)
    return {
        "artifact": merged,
        "timing": {
            "stage": f"decompose_serial_local_{arguments.granularity}",
            "wall_seconds": wall,
            "unit_seconds_total": round(unit_seconds, 2),
            "units": len(items),
            "unit_seconds_max": round(
                max(float(row["wall_seconds"]) for row in rows), 2
            ),
        },
    }


def _first_difference(left: object, right: object, path: str = "$") -> str | None:
    """Name the first disagreeing JSON path.  ``None`` means the two are identical."""

    if type(left) is not type(right):
        return f"{path}: type {type(left).__name__} vs {type(right).__name__}"
    if isinstance(left, dict):
        for key in sorted(set(left) | set(right)):
            if key not in left:
                return f"{path}.{key}: missing on the left"
            if key not in right:
                return f"{path}.{key}: missing on the right"
            found = _first_difference(left[key], right[key], f"{path}.{key}")
            if found:
                return found
        return None
    if isinstance(left, list):
        if len(left) != len(right):
            return f"{path}: length {len(left)} vs {len(right)}"
        for index, (one, other) in enumerate(zip(left, right)):
            found = _first_difference(one, other, f"{path}[{index}]")
            if found:
                return found
        return None
    if left != right:
        return f"{path}: {left!r} vs {right!r}"
    return None


def compare_artifacts(
    left: dict, right: dict, workload_name: str = "t4_v2_feasibility"
) -> dict:
    """Canonical, timing-stripped equivalence of two merged artifacts."""

    strip = workloads.workload(workload_name)["strip_timing"]

    def drop_bookkeeping(payload: dict) -> dict:
        # Harness provenance, not chemistry.  Safe to exclude from the hash BECAUSE an
        # incomplete artifact is already unequal by content: the units it is missing are
        # absent from ``cells``.  Reported separately so the reader still sees it.
        return {
            key: value
            for key, value in payload.items()
            if key not in ("complete", "missing_units")
        }

    left_clean = strip(drop_bookkeeping(left))
    right_clean = strip(drop_bookkeeping(right))
    left_identity = workloads.canonical_sha256(left_clean)
    right_identity = workloads.canonical_sha256(right_clean)
    per_cell = {}
    for cell in sorted(set(left_clean.get("cells", {})) | set(right_clean.get("cells", {}))):
        one = left_clean.get("cells", {}).get(cell)
        other = right_clean.get("cells", {}).get(cell)
        per_cell[cell] = {
            "left_sha256": workloads.canonical_sha256(one) if one is not None else None,
            "right_sha256": (
                workloads.canonical_sha256(other) if other is not None else None
            ),
            "identical": one is not None and other is not None and one == other,
            "first_difference": (
                _first_difference(one, other, f"$.cells.{cell}")
                if one is not None and other is not None
                else "cell missing on one side"
            ),
        }
    return {
        "left_complete": left.get("complete", True),
        "right_complete": right.get("complete", True),
        "left_missing_units": len(left.get("missing_units") or []),
        "right_missing_units": len(right.get("missing_units") or []),
        "timing_fields_excluded": (
            list(workloads.T4_V2_TIMING_FIELDS)
            if workload_name == "t4_v2_feasibility"
            else []
        ),
        "left_sha256": left_identity,
        "right_sha256": right_identity,
        "identical": left_identity == right_identity,
        "first_difference": _first_difference(left_clean, right_clean),
        "cells_compared": len(per_cell),
        "cells_identical": sum(1 for row in per_cell.values() if row["identical"]),
        "per_cell": per_cell,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("baseline", "decompose", "compare"))
    parser.add_argument("--workload", default="t4_v2_feasibility")
    parser.add_argument("--cells", default="")
    parser.add_argument("--arms", default="")
    parser.add_argument("--budget", type=int, default=6000)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--delta", type=float, default=0.6)
    parser.add_argument("--granularity", default="arm", choices=("arm", "cell"))
    parser.add_argument("--no-ablate", action="store_true")
    parser.add_argument("--no-attribution", action="store_true")
    parser.add_argument("--out", default="")
    parser.add_argument("--left", default="")
    parser.add_argument("--right", default="")
    arguments = parser.parse_args()

    if arguments.stage == "compare":
        left = json.loads(Path(arguments.left).read_text())
        right = json.loads(Path(arguments.right).read_text())
        # Accept either a bare artifact or a {"artifact": ...} envelope on either side.
        left = left.get("artifact", left) if isinstance(left, dict) else left
        right = right.get("artifact", right) if isinstance(right, dict) else right
        result = compare_artifacts(left, right, arguments.workload)
        if arguments.out:
            Path(arguments.out).write_text(
                json.dumps(result, indent=1, sort_keys=True) + "\n"
            )
        summary = dict(result)
        summary.pop("per_cell")
        print(json.dumps(summary, indent=1, sort_keys=True))
        return

    result = (
        stage_baseline(arguments)
        if arguments.stage == "baseline"
        else stage_decompose(arguments)
    )
    result["zero_oracle_evidence"] = workloads.assert_zero_oracle()
    result["workload"] = arguments.workload
    if arguments.out:
        Path(arguments.out).write_text(
            json.dumps(result, indent=1, sort_keys=True, default=str) + "\n"
        )
        print(f"[out] {arguments.out}", flush=True)
    print(json.dumps(result["timing"], indent=1, sort_keys=True))


if __name__ == "__main__":
    main()
