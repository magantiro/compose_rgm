"""Autopsy of the PMO jump lane's binding failures: WHICH constraint refuses, and why.

WHAT THIS IS
------------
``pmo_population_controller._generate_jump_pool`` calls
``pmo_realization.realize_plan_binding`` and, on an empty result, records ONE string:
``"joint plan has no legal binding on this parent (<outcome>)"``.  Over the arm-B
celecoxib 250-call run that string covers **357 of 357** jump attempts.  Four outcomes
are already preserved inside it (``proven_incompatible`` 346, ``search_budget_exhausted``
11), but an outcome is not a MECHANISM: ``proven_incompatible`` is returned both by a
plan whose role 0 no atom of the parent can carry and by a plan that reached depth 14
and ran out of created handles.

This module decomposes that outcome into the refusal the search ITSELF computed, by
WRAPPING ``pmo_realization.propagate`` rather than transcribing it.  The wrapper calls
the production function and records its return value, so the recorded reason IS the
decision the search acted on -- a transcription could drift from the code it claims to
describe and would have nothing to check itself against.

Two quantities, kept separate because they answer different questions:

  ROOT FEASIBILITY -- a zero-search necessary condition on role 0 alone
    (``pmo_realization.static_first_step_feasible``).  Cheap, exact, and the quantity a
    descriptor PROJECTION moves.  It is NECESSARY, NEVER SUFFICIENT: a pair that clears
    step 0 may still be refused at any later step, by the completion conditions, or by
    the budget.  A step-0 number quoted as a binding rate is an over-claim.

  REALIZED OUTCOME -- what the production search returns at the production budget.

DESCRIPTOR PROJECTION IS A DIAGNOSTIC, NOT A LICENCE
----------------------------------------------------
``field_projection_lift`` reports, per descriptor field, how many pairs become step-0
feasible when that field alone is dropped from the comparison.  A match under a coarser
descriptor realizes a DIFFERENT transformation from the one the plan declares.  The
projection LOCATES the binding constraint; it does not establish that relaxing it is
sound.  Every caller must label its output accordingly.

INVARIANTS TESTED
-----------------
  * the wrapper never changes a decision: recorded reason == the production function's
    return on every call, and the wrapped search's outcome equals the unwrapped one;
  * ``SPEC_V1_EXACT`` and a relaxed-nothing specification give the SAME step-0 verdict
    (the production predicate compares whole-role identity, but step-0 availability is a
    descriptor-key computation over ``spec.compared_fields``, which coincide);
  * dropping a field can only ADD step-0-feasible pairs, never remove one (monotonicity).
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from compose_v4.control import pmo_realization as realization
from compose_v4.control.pmo_realization import (
    RELAXABLE_COMPONENTS,
    SPEC_V1_EXACT,
    RoleSpecification,
    static_first_step_feasible,
)

SCHEMA = "pmo_jump_binding_autopsy_v1"

# The specification whose ``compared_fields`` are the full descriptor.  Provably the same
# step-0 predicate as the production ``SPEC_V1_EXACT`` (see module docstring); used where
# a non-identity comparison is needed so that a single field can be projected out.
SPEC_V1_FIELDS = RoleSpecification("v1_fields", frozenset())

# Every descriptor component a projection may remove.  The semantic core
# (``atom_type``/``formal_charge``) and the dataflow edge (``origin``/``created_ordinal``)
# are included ONLY so the diagnostic can report what they cost; projecting them is not a
# proposal, and ``pmo_realization`` refuses to relax them in any shipped specification.
PROJECTABLE_FIELDS = (
    *RELAXABLE_COMPONENTS,
    "atom_type",
    "formal_charge",
    "origin",
    "created_ordinal",
)

# ---- refusal taxonomy ----
#
# Keys are the reason strings ``pmo_realization.propagate`` itself returns, plus the
# outcomes ``realize`` reports.  The mechanism names are this module's reading of the
# code that produces each string and are recorded beside the raw string, never instead
# of it.
REFUSAL_MECHANISM = {
    "insufficient_slot_capacity": "capacity",
    "insufficient_preexisting_atoms_for_remaining_deletions": "region_size_mismatch",
    "insufficient_preexisting_atoms_of_a_required_identity": "operand_state_incompatible",
    "required_created_handle_is_unreachable": "dataflow_step_order",
    "no_atom_carries_a_required_operand_descriptor": "operand_descriptor_overspecified",
}


@contextmanager
def recording_propagate() -> Iterator[list[dict[str, Any]]]:
    """Record every ``propagate`` refusal the production search acts on.

    The production function decides; this only observes.  ``pmo_realization.realize``
    resolves ``propagate`` from its own module globals at call time, so rebinding the
    module attribute intercepts the real call site without touching the search.
    """

    original = realization.propagate
    log: list[dict[str, Any]] = []

    def wrapper(prefix, step, plan, demand, spec):
        reason = original(prefix, step, plan, demand, spec)
        log.append({"step": int(step), "reason": reason})
        return reason

    realization.propagate = wrapper
    try:
        yield log
    finally:
        realization.propagate = original


def first_refusal(log: list[dict[str, Any]]) -> dict[str, Any] | None:
    """The first refusal recorded, or ``None`` when nothing was refused."""

    for row in log:
        if row["reason"] is not None:
            return dict(row)
    return None


def root_refusal(log: list[dict[str, Any]]) -> str | None:
    """The refusal at step 0, i.e. before the search expanded anything.

    ``None`` means the root survived forward checking, whatever happened afterwards.
    """

    for row in log:
        if row["step"] == 0:
            return row["reason"]
    return None


def deepest_refused_step(log: list[dict[str, Any]]) -> int:
    return max((row["step"] for row in log), default=-1)


def diagnose_pair(
    source,
    plan: dict[str, Any],
    *,
    specification: RoleSpecification = SPEC_V1_EXACT,
    node_budget: int = realization.PRODUCTION_NODE_BUDGET,
    seconds_cap: float = realization.PRODUCTION_SECONDS_CAP,
    max_realizations: int = realization.PRODUCTION_MAX_REALIZATIONS,
) -> dict[str, Any]:
    """Run the production binder on one (parent, plan) pair and record WHY it refused.

    The returned ``outcome``/``bindings`` are the production values: this function adds
    evidence beside the search, it does not substitute for it.
    """

    with recording_propagate() as log:
        result = realization.realize_plan_binding(
            source,
            plan,
            specification=specification,
            node_budget=node_budget,
            seconds_cap=seconds_cap,
            max_realizations=max_realizations,
        )
    root = root_refusal(log)
    first = first_refusal(log)
    counts: dict[str, int] = {}
    for row in log:
        if row["reason"] is not None:
            counts[row["reason"]] = counts.get(row["reason"], 0) + 1
    return {
        "outcome": result["outcome"],
        "bindings": len(result["bindings"]),
        "nodes_expanded": int(result["nodes_expanded"]),
        "successors_enumerated": int(result["successors_enumerated"]),
        "max_depth_reached": int(result["max_depth_reached"]),
        "pruned_by_propagation": int(result.get("pruned_by_propagation", 0)),
        "seconds": float(result["seconds"]),
        "invalid_reasons": list(result.get("invalid_reasons", ())),
        "completion_rejections": sorted(set(result.get("completion_rejections", ()))),
        "root_refusal": root,
        "root_refused": root is not None,
        "first_refusal_step": None if first is None else first["step"],
        "first_refusal_reason": None if first is None else first["reason"],
        "refusal_counts": counts,
        "deepest_step_examined": deepest_refused_step(log),
        "propagate_calls": len(log),
        "plan_primitive_count": int(plan["primitive_count"]),
    }


def classify(row: dict[str, Any]) -> str:
    """One mechanism label per diagnosed pair, from the evidence the search produced.

    Order matters and encodes what the search proved: a plan that is statically invalid
    is not a fact about the parent; a budget hit proves nothing at all; and a pair whose
    ROOT survived forward checking but which still expanded no role-consistent child was
    refused by the role-match predicate rather than by any propagated constraint.
    """

    if row["outcome"] == realization.OUTCOME_INVALID_PLAN:
        return "invalid_plan"
    if row["bindings"]:
        return "bound"
    if row["outcome"] == realization.OUTCOME_EXHAUSTED:
        return "search_budget_exhausted"
    reason = row["first_refusal_reason"]
    if reason is not None:
        label = REFUSAL_MECHANISM[reason]
        return f"{label}@root" if row["first_refusal_step"] == 0 else f"{label}@depth"
    if row["completion_rejections"]:
        return "declared_completion_conditions"
    return "no_role_consistent_successor"


def field_projection_lift(
    source,
    plan: dict[str, Any],
    fields: tuple[str, ...] = PROJECTABLE_FIELDS,
) -> dict[str, Any]:
    """Step-0 feasibility under the full descriptor and under each single-field drop.

    DIAGNOSTIC ONLY.  A step-0 match under a coarser descriptor identifies a DIFFERENT
    atom, hence a different transformation; this locates the constraint and licenses
    nothing.  Reported as a NECESSARY condition, never as a binding rate.
    """

    baseline = bool(static_first_step_feasible(source, plan, SPEC_V1_FIELDS))
    per_field = {}
    for name in fields:
        spec = RoleSpecification(f"drop_{name}", frozenset({name}))
        per_field[name] = bool(static_first_step_feasible(source, plan, spec))
    return {"baseline_feasible": baseline, "single_drop_feasible": per_field}


def role_based_specification(name: str = "role_based_v1") -> RoleSpecification:
    """The descriptor a ROLE-based option would compare.

    Keeps the semantic core (``atom_type``, ``formal_charge``) and the dataflow edge
    (``origin``, ``created_ordinal``); drops every component that describes the SOURCE
    molecule's local neighbourhood or the teacher's schedule.  Identical in content to
    ``pmo_realization.SPEC_R2``; named separately so a result can be attributed to this
    design rather than to the pre-existing relaxation arm.
    """

    return RoleSpecification(name, frozenset(RELAXABLE_COMPONENTS))


def step0_feasible_under(source, plan: dict[str, Any], spec: RoleSpecification) -> bool:
    return bool(static_first_step_feasible(source, plan, spec))


# ---- structural shape of a realized transformation ----
#
# The measure the PMO structural census uses to compare a proposal against the macro a
# productive transition requires (``scripts/pmo_large_proposal_structure.py``): compare
# the endpoint to its own SOURCE by MCS, call every endpoint atom outside the common
# substructure CHANGED, and take the connected components of that set as the changed
# REGIONS.  Reproduced here so a realized binding can be scored on the SAME axes as the
# generic sampler it would replace; ``tests`` pin it on hand-checkable pairs.
#
# ``retained_fraction`` here is the MCS core over the source's heavy atoms, which is NOT
# the controller's own ``_retained_fraction`` rule.  Both are reported by callers and the
# two must never be quoted interchangeably.


def structural_delta(source_smiles: str, endpoint_smiles: str, *, timeout: int = 5):
    """Region-level delta of ``endpoint`` relative to ``source``; ``None`` if unparseable."""

    import networkx as nx
    from rdkit import Chem
    from rdkit.Chem import rdFMCS

    before = Chem.MolFromSmiles(source_smiles)
    after = Chem.MolFromSmiles(endpoint_smiles)
    if before is None or after is None:
        return None
    result = rdFMCS.FindMCS(
        [before, after], timeout=timeout, completeRingsOnly=False, ringMatchesRingOnly=True
    )
    if result.canceled or result.numAtoms == 0:
        core, matched = 0, set()
    else:
        query = Chem.MolFromSmarts(result.smartsString)
        match = after.GetSubstructMatch(query)
        core, matched = result.numAtoms, set(match) if match else set()
    changed = [atom.GetIdx() for atom in after.GetAtoms() if atom.GetIdx() not in matched]
    graph = nx.Graph()
    graph.add_nodes_from(changed)
    changed_set = set(changed)
    for bond in after.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if i in changed_set and j in changed_set:
            graph.add_edge(i, j)
    components = [len(part) for part in nx.connected_components(graph)] if changed else []
    return {
        "d_heavy": after.GetNumHeavyAtoms() - before.GetNumHeavyAtoms(),
        "d_rings": after.GetRingInfo().NumRings() - before.GetRingInfo().NumRings(),
        "retained_core": core,
        "retained_fraction_mcs": core / max(before.GetNumHeavyAtoms(), 1),
        "n_changed_regions": len(components),
        "largest_changed_region": max(components) if components else 0,
        "total_changed": sum(components),
    }


__all__ = [
    "PROJECTABLE_FIELDS",
    "REFUSAL_MECHANISM",
    "SCHEMA",
    "SPEC_V1_FIELDS",
    "classify",
    "deepest_refused_step",
    "diagnose_pair",
    "field_projection_lift",
    "first_refusal",
    "recording_propagate",
    "role_based_specification",
    "root_refusal",
    "step0_feasible_under",
    "structural_delta",
]
