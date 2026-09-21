"""Matched offline-vs-production distribution comparison for the PMO jump lane.

Holds the 95 shared plan latents FIXED and varies only the parent population, so
any move is a property of the parents the production lane binds against -- not of
the plans, the specification, the search, or the interpreter.

Four parent strata:
  teacher_witness     the route source that GENERATED the plan (a realization is
                      known to exist, except where the runtime fiber cannot
                      re-offer a teacher step -- 14 of 95 plans, measured by
                      ``pmo_realization_witness_gate_v1``).
  teacher_root        the 7 distinct teacher route sources.
  production_init     the pinned 16-molecule production initialization -- the
                      parents the offline repair benchmark used.
  production_archive  the DISTINCT parents the jump lane actually bound against in
                      the completed 3x250 scored run.  These are archive entries
                      reached by shallow/structured edits, not initialization
                      molecules, which is the population the offline benchmark
                      never contained.

Two zero-search certificates per pair, both from the PINNED module:
  operand_feasible    ``static_first_step_feasible`` -- role 0 has an operand
                      descriptor some active atom carries.
  root_admissible     ``propagate`` at the root returns no refusal.  This is the
                      STRICTLY stronger depth-0 proof: it also forward-checks slot
                      capacity, the element/charge deletion budget, and created
                      handle reachability against the whole plan suffix.

Zero oracle calls.  Pinned modules are imported, never modified.
"""

from __future__ import annotations

import argparse
import collections
import gzip
import json
import statistics
from pathlib import Path
from time import perf_counter
from typing import Any

from compose_v4.chem.molecular_graph import is_element
from compose_v4.control import pmo_realization as PR
from compose_v4.control.docking_value import identity
from compose_v4.rewrite.trace_shard import decode_state

SCHEMA = "pmo_jump_distribution_shift_v1"
CHECKPOINT = Path("diagnostics/pmo_joint_dependency_jump_gate_v1/attempt_2/checkpoints.json")
PRODUCTION_INIT = Path("diagnostics/parent_edit_cycles/prepared/init_20260921.json")
TEACHER_CORPUS = Path(
    "diagnostics/pmo_dependency_region_program_v2/attempt_1/"
    "training_dependency_region_corpus.json.gz"
)
TASKS = ("gsk3b", "perindopril_mpo", "celecoxib_rediscovery")
JUMP = "joint_dependency_region_jump"


def load_plans() -> list[dict[str, Any]]:
    checkpoint = json.loads(CHECKPOINT.read_text())["payload"]["checkpoints"][
        "shared_all_routes"
    ]
    if checkpoint.get("fit_scope") != "shared_all_routes":
        raise ValueError("unexpected jump checkpoint fit scope")
    return list(checkpoint["plan_latents"])


def load_teacher_corpus() -> dict[str, Any]:
    with gzip.open(TEACHER_CORPUS, "rt") as handle:
        return json.load(handle)["payload"]


def strata(data_root: Path) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}

    body = json.loads(PRODUCTION_INIT.read_text())
    out["production_init"] = [
        {"parent_id": f"production_init:{i}", "state": row["state"], "depth": 0}
        for i, row in enumerate(body["candidates"])
    ]

    corpus = load_teacher_corpus()
    roots, seen = [], set()
    for route in corpus["routes"]:
        key = json.dumps(route["states"][0], sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        roots.append(
            {"parent_id": f"teacher_root:{len(roots)}", "state": route["states"][0],
             "depth": 0}
        )
    out["teacher_root"] = roots

    entries = json.loads((data_root.parent / "parent_entries.json").read_text())
    used: set[str] = set()
    for task in TASKS:
        payload = json.loads((data_root / task / "round_0014" / "complete.json").read_text())
        for batch in payload["snapshot"]["history"]:
            for attempt in batch["batch"]["attempts"]:
                if attempt.get("planner_channel") == JUMP:
                    used.add(attempt["entry_id"])
    out["production_archive"] = [
        {
            "parent_id": f"production_archive:{entry_id[:12]}",
            "state": entries[entry_id]["trace"]["states"][-1],
            "depth": len(entries[entry_id]["trace"]["actions"]),
        }
        for entry_id in sorted(used)
    ]
    return out


def witness_pairs() -> dict[str, list[dict[str, Any]]]:
    from compose_v4.control.pmo_joint_dependency_jump import _generic_role_sequence

    corpus = load_teacher_corpus()
    witnesses: dict[str, list[dict[str, Any]]] = {}
    for index, route in enumerate(corpus["routes"]):
        if not route["dependency_region_program"].get("complete_representation_supported"):
            continue
        roles = _generic_role_sequence(route)
        plan_id = identity({"schema_version": "pmo_joint_role_plan_v1", "roles": roles})
        witnesses.setdefault(plan_id, []).append(
            {"parent_id": f"teacher_witness:{index}", "state": route["states"][0],
             "depth": 0}
        )
    return witnesses


def shape(graph) -> dict[str, int]:
    active = int(is_element(graph.atom_types).sum())
    return {
        "heavy_atoms": active,
        "n_slots": len(graph.atom_types),
        "free_slots": len(graph.atom_types) - active,
    }


def certify(graph, plan: dict[str, Any], spec) -> dict[str, Any]:
    """Both zero-search depth-0 certificates, from the pinned implementation."""

    operand = PR.static_first_step_feasible(graph, plan, spec)
    prefix = PR._Prefix(graph=graph, actions=(), created=(), next_ordinal=0)
    refusal = PR.propagate(prefix, 0, plan, PR.plan_demand(plan), spec)
    return {
        "operand_feasible": bool(operand),
        "root_admissible": refusal is None,
        "root_refusal": refusal,
    }


def _by_heavy_atom_band(rows: list[dict[str, Any]], n_plans: int) -> dict[str, Any]:
    """Depth-0 admissibility as a function of parent size.

    The plans are held fixed, so any monotone trend here is a statement about how
    much molecule a plan needs in order to be admissible at all.
    """

    bands: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in rows:
        heavy = row["heavy_atoms"]
        band = ("01-09" if heavy < 10 else "10-14" if heavy < 15 else
                "15-19" if heavy < 20 else "20-24" if heavy < 25 else
                "25-29" if heavy < 30 else "30+")
        bands[band].append(row)
    return {
        band: {
            "parents": len(group),
            "pairs": len(group) * n_plans,
            "root_admissible": sum(r["root_admissible_plans"] for r in group),
            "root_admissible_fraction": (
                sum(r["root_admissible_plans"] for r in group) / (len(group) * n_plans)
            ),
            "median_depth": statistics.median([r["depth"] for r in group]),
        }
        for band, group in sorted(bands.items())
    }


def summarize(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    ordered = sorted(values)
    return {
        "n": len(ordered),
        "min": ordered[0],
        "median": statistics.median(ordered),
        "mean": statistics.fmean(ordered),
        "p90": ordered[min(len(ordered) - 1, int(0.9 * len(ordered)))],
        "max": ordered[-1],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default=str(Path.home() / "compose_pmo_data/rounds"))
    parser.add_argument("--out", default="diagnostics/pmo_jump_distribution_shift_v1.json")
    args = parser.parse_args()

    spec = PR.PRODUCTION_SPECIFICATION
    plans = load_plans()
    groups = strata(Path(args.data_root))
    began = perf_counter()

    report: dict[str, Any] = {
        "schema_version": SCHEMA,
        "new_oracle_calls": 0,
        "plans": len(plans),
        "specification": spec.name,
        "strata": {},
        "parent_axes": {},
        "by_plan_scale": {},
    }

    for name, parents in groups.items():
        shapes = [shape(decode_state(p["state"])) for p in parents]
        report["parent_axes"][name] = {
            "parents": len(parents),
            "heavy_atoms": summarize([s["heavy_atoms"] for s in shapes]),
            "free_slots": summarize([s["free_slots"] for s in shapes]),
            "n_slots": summarize([float(s["n_slots"]) for s in shapes]),
            "depth_from_root": summarize([float(p["depth"]) for p in parents]),
        }
        operand = root = 0
        refusals: collections.Counter = collections.Counter()
        scale_tab: dict[str, dict[str, int]] = {}
        per_parent: list[dict[str, Any]] = []
        for parent in parents:
            graph = decode_state(parent["state"])
            parent_shape = shape(graph)
            parent_root = 0
            for plan in plans:
                certificate = certify(graph, plan, spec)
                operand += certificate["operand_feasible"]
                root += certificate["root_admissible"]
                parent_root += certificate["root_admissible"]
                refusals[certificate["root_refusal"] or "ADMISSIBLE"] += 1
                count = int(plan["primitive_count"])
                band = "small<=7" if count <= 7 else (
                    "medium8-15" if count <= 15 else (
                        "large16-26" if count < 27 else "teacher_scale>=27"
                    )
                )
                bucket = scale_tab.setdefault(band, {"pairs": 0, "operand": 0, "root": 0})
                bucket["pairs"] += 1
                bucket["operand"] += certificate["operand_feasible"]
                bucket["root"] += certificate["root_admissible"]
            per_parent.append({
                "parent_id": parent["parent_id"],
                "depth": parent["depth"],
                **parent_shape,
                "root_admissible_plans": parent_root,
                "root_admissible_fraction": parent_root / len(plans),
            })
        pairs = len(parents) * len(plans)
        report["strata"][name] = {
            "pairs": pairs,
            "operand_feasible": operand,
            "operand_feasible_fraction": operand / pairs,
            "root_admissible": root,
            "root_admissible_fraction": root / pairs,
            "root_refusal_histogram": dict(refusals.most_common()),
        }
        report.setdefault("per_parent", {})[name] = per_parent
        report.setdefault("admissibility_by_parent_heavy_atoms", {})[name] = (
            _by_heavy_atom_band(per_parent, len(plans))
        )
        report["by_plan_scale"][name] = {
            band: {
                **bucket,
                "operand_fraction": bucket["operand"] / bucket["pairs"],
                "root_fraction": bucket["root"] / bucket["pairs"],
            }
            for band, bucket in sorted(scale_tab.items())
        }

    # matched witness pairs: plan bound back onto its own generating source
    witnesses = witness_pairs()
    by_id = {plan["plan_id"]: plan for plan in plans}
    wpairs = woperand = wroot = 0
    wrefusals: collections.Counter = collections.Counter()
    for plan_id, rows in witnesses.items():
        plan = by_id.get(plan_id)
        if plan is None:
            continue
        for row in rows:
            certificate = certify(decode_state(row["state"]), plan, spec)
            wpairs += 1
            woperand += certificate["operand_feasible"]
            wroot += certificate["root_admissible"]
            wrefusals[certificate["root_refusal"] or "ADMISSIBLE"] += 1
    report["strata"]["teacher_witness_matched"] = {
        "pairs": wpairs,
        "operand_feasible": woperand,
        "operand_feasible_fraction": woperand / wpairs if wpairs else None,
        "root_admissible": wroot,
        "root_admissible_fraction": wroot / wpairs if wpairs else None,
        "root_refusal_histogram": dict(wrefusals.most_common()),
        "note": "each plan bound onto the teacher route that generated it",
    }

    report["elapsed_seconds"] = perf_counter() - began
    Path(args.out).write_text(json.dumps(report, indent=1, sort_keys=True))
    for name, block in report["strata"].items():
        print(f"{name:26s} pairs={block['pairs']:6d} "
              f"operand={block['operand_feasible_fraction']:.4f} "
              f"root_admissible={block['root_admissible_fraction']:.4f}")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
