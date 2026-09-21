#!/usr/bin/env python3
"""Zero-oracle jump-lane realization audit, one row per teacher root.

The jump lane is the ONLY autonomous lane whose programs reach teacher scale, so it
is the only lane where a teacher-scale transformation can be lost at realization
rather than at scale.  This driver measures, for every distinct teacher root and
every plan in the sealed task-blind bank:

  * role-0 feasibility        -- does ANY legal successor of the root carry the
                                 operand descriptor the plan's first role demands
  * binding depth             -- the deepest role index at which at least one beam
                                 prefix survived, under the PRODUCTION beam width
  * descriptor projections    -- role-0 feasibility with one descriptor field removed
                                 from BOTH the stored plan and the observed role

METHOD.  Binding is never transcribed.  ``bind_joint_plan`` is called unmodified and
its own ``_role_identity`` decides; depth is recovered by WRAPPING the module global
``enumerate_role_successors`` that the binder resolves at call time, in this process
only, and recording the largest ``step`` it is asked for.  Cost is reported as
successors enumerated, never seconds.

A projection that matches under a coarser descriptor realizes a DIFFERENT
transformation.  It LOCATES the constraint; it does not license relaxing it.

No oracle is constructed and no call is charged.
"""

from __future__ import annotations

import argparse
import copy
import json
from collections import Counter, defaultdict
from pathlib import Path

from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

from pmo_teacher_route_gap_v1 import ROOT, load_routes, route_objective

from compose_v4.control import pmo_action_roles
from compose_v4.control import pmo_joint_dependency_jump as jump
from compose_v4.rewrite.trace_shard import decode_state

PRODUCTION_BEAM_WIDTH = 4
CHECKPOINT = ROOT / "diagnostics/pmo_joint_dependency_jump_gate_v1/attempt_2/checkpoints.json"

# Fields of ``pmo_action_roles.atom_role``; the two creation fields only exist on a
# route-created operand, so they are ablated together.
ABLATIONS = {
    "neighbor_element_histogram": ("neighbor_element_histogram",),
    "implicit_hydrogens": ("implicit_hydrogens",),
    "bond_class_histogram": ("bond_class_histogram",),
    "degree": ("degree",),
    "formal_charge": ("formal_charge",),
    "creation_order": ("created_ordinal", "creation_lag"),
    "origin": ("origin",),
}


def load_plans() -> list[dict]:
    checkpoint = json.loads(CHECKPOINT.read_text())["payload"]["checkpoints"][
        "shared_all_routes"
    ]
    if checkpoint.get("fit_scope") != "shared_all_routes":
        raise ValueError("the PMO jump bank must be shared and task blind")
    return checkpoint["plan_latents"]


def project_plan(plan: dict, drop: tuple[str, ...]) -> dict:
    """A copy of the plan with ``drop`` removed from every operand descriptor."""
    projected = copy.deepcopy(plan)
    for role in projected["roles"]:
        for operand in role.get("operands", ()):
            for field in drop:
                operand["descriptor"].pop(field, None)
    return projected


class _Instrumented:
    """Wrap the binder's successor enumerator: record depth and enumeration cost."""

    def __init__(self, drop: tuple[str, ...] = ()) -> None:
        self.drop = drop
        self.max_step = -1
        self.successors = 0
        self.calls = 0

    def __enter__(self):
        self._enumerate = jump.enumerate_role_successors
        self._atom_role = pmo_action_roles.atom_role
        self._jump_atom_role = jump.atom_role
        drop = self.drop

        def atom_role(*args, **kwargs):
            role = self._atom_role(*args, **kwargs)
            for field in drop:
                role.pop(field, None)
            return role

        def enumerate_role_successors(graph, role, *, created=None, step=0):
            self.calls += 1
            self.max_step = max(self.max_step, step)
            rows = self._enumerate(graph, role, created=created, step=step)
            self.successors += len(rows)
            return rows

        if drop:
            pmo_action_roles.atom_role = atom_role
            jump.atom_role = atom_role
        jump.enumerate_role_successors = enumerate_role_successors
        return self

    def __exit__(self, *exc) -> None:
        jump.enumerate_role_successors = self._enumerate
        pmo_action_roles.atom_role = self._atom_role
        jump.atom_role = self._jump_atom_role


def truncate(plan: dict, roles: int) -> dict:
    return {**plan, "roles": plan["roles"][:roles]}


def probe(source, plan: dict, *, roles: int | None, drop: tuple[str, ...]) -> dict:
    """Run the PRODUCTION binder on (a prefix of) a (possibly projected) plan."""
    candidate = project_plan(plan, drop) if drop else plan
    if roles is not None:
        candidate = truncate(candidate, roles)
    with _Instrumented(drop) as probe_state:
        try:
            bindings = jump.bind_joint_plan(
                source, candidate, beam_width=PRODUCTION_BEAM_WIDTH
            )
            error = None
        except Exception as failure:  # noqa: BLE001 - report any binder refusal verbatim
            bindings, error = [], f"{type(failure).__name__}: {failure}"
    return {
        "bindings": len(bindings),
        "deepest_role_reached": probe_state.max_step,
        "successors_enumerated": probe_state.successors,
        "enumerator_calls": probe_state.calls,
        "error": error,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--prefix-roles",
        type=int,
        default=0,
        help=(
            "also bind the first N roles of every role-0-feasible plan and record the "
            "deepest role index a beam prefix survived; 0 disables the depth ladder"
        ),
    )
    options = parser.parse_args()

    corpus = load_routes()
    roots: dict[str, int] = {}
    root_state: dict[int, dict] = {}
    root_objectives: dict[int, set[str]] = defaultdict(set)
    for route in corpus:
        key = json.dumps(route["source_state"], sort_keys=True)
        if key not in roots:
            roots[key] = len(roots)
            root_state[roots[key]] = route["source_state"]
        root_objectives[roots[key]].add(route_objective(route))

    plans = load_plans()
    per_root = {}
    for root_id in sorted(root_state):
        source = decode_state(root_state[root_id])
        rows = []
        for plan in plans:
            row = {
                "plan_id": plan["plan_id"],
                "plan_primitives": plan["primitive_count"],
                "plan_components": plan["component_count"],
                "role0_rule": plan["roles"][0]["executor_rule"],
            }
            baseline = probe(source, plan, roles=1, drop=())
            row["role0"] = baseline
            row["ablation"] = {
                name: probe(source, plan, roles=1, drop=fields)["bindings"] > 0
                for name, fields in ABLATIONS.items()
            }
            if options.prefix_roles and baseline["bindings"] > 0:
                row["prefix"] = probe(
                    source, plan, roles=min(options.prefix_roles, len(plan["roles"])), drop=()
                )
            rows.append(row)
        feasible = [row for row in rows if row["role0"]["bindings"] > 0]
        summary = {
            "plans": len(rows),
            "role0_feasible": len(feasible),
            "role0_feasible_rate": len(feasible) / len(rows),
            "role0_rule_feasibility": {
                rule: {
                    "plans": sum(row["role0_rule"] == rule for row in rows),
                    "feasible": sum(
                        row["role0_rule"] == rule and row["role0"]["bindings"] > 0
                        for row in rows
                    ),
                }
                for rule in sorted({row["role0_rule"] for row in rows})
            },
            "ablation_role0_feasible_rate": {
                name: sum(row["ablation"][name] for row in rows) / len(rows)
                for name in ABLATIONS
            },
            "successors_enumerated_role0": sum(
                row["role0"]["successors_enumerated"] for row in rows
            ),
        }
        if options.prefix_roles:
            prefix = [row["prefix"] for row in rows if "prefix" in row]
            summary["prefix_bind"] = {
                "roles_attempted": options.prefix_roles,
                "plans_attempted": len(prefix),
                "prefix_bindings": sum(row["bindings"] > 0 for row in prefix),
                "deepest_role_reached": dict(
                    sorted(Counter(row["deepest_role_reached"] for row in prefix).items())
                ),
                "successors_enumerated": sum(row["successors_enumerated"] for row in prefix),
            }
        per_root[str(root_id)] = {
            "root_id": root_id,
            "objectives": sorted(root_objectives[root_id]),
            "summary": summary,
            "plans": rows,
        }
        print(
            f"root {root_id}: role0 {summary['role0_feasible']}/{summary['plans']}"
            f" objectives={sorted(root_objectives[root_id])}",
            flush=True,
        )

    payload = {
        "schema_version": "pmo_teacher_capability_jump_lane_v1",
        "charged_oracle_calls": 0,
        "beam_width": PRODUCTION_BEAM_WIDTH,
        "prefix_roles": options.prefix_roles,
        "checkpoint": str(CHECKPOINT.relative_to(ROOT)),
        "plan_bank": len(plans),
        "projection_caveat": (
            "A projection that matches under a coarser descriptor realizes a DIFFERENT "
            "transformation. It locates the constraint; it does not license relaxing it."
        ),
        "roots": per_root,
    }
    Path(options.out).parent.mkdir(parents=True, exist_ok=True)
    Path(options.out).write_text(json.dumps(payload, sort_keys=True, indent=1) + "\n")
    print(f"wrote {options.out}")


if __name__ == "__main__":
    main()
