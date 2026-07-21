"""Teacher-program transform: collapse linear saturated-carbon `atom_insert` runs
into single `alkyl_graft` steps, so training data teaches the model to build tails
in one high-level move. Verified by reconstruction: if the collapsed program does
not rebuild the identical target, the original trace is returned unchanged.

This is the step-5 piece of the AlkylGraft integration (docs/CHAIN_REWRITE_DESIGN.md).
"""
from __future__ import annotations

from compose_v4.chem.molecular_graph import (
    BOND_SINGLE,
    ELEMENT_TO_IDX,
    molecular_graph_to_smiles,
)
from compose_v4.rewrite.alkyl_graft import ALKYL_MAX_LENGTH, build_graft
from compose_v4.rewrite.kernel import RewriteSystem
from compose_v4.rewrite.trace import RewriteStep, RewriteTrace

_C = ELEMENT_TO_IDX["C"]


def _smiles(state) -> str | None:
    try:
        return molecular_graph_to_smiles(state)
    except Exception:
        return None


def collapse_alkyl_runs(
    trace: RewriteTrace, system: RewriteSystem, *, min_run: int = 2
) -> RewriteTrace:
    """Return a trace with maximal linear saturated-C atom_insert runs collapsed to
    `alkyl_graft` steps. Falls back to the original trace if verification fails."""
    steps = list(trace.steps)
    n = len(steps)
    out: list[RewriteStep] = []
    i = 0
    changed = False
    while i < n:
        s = steps[i]
        if s.rule_name != "atom_insert" or int(s.action.atom_type) != _C:
            out.append(s)
            i += 1
            continue
        first = s.action
        is_root = len(first.neighbors) == 0
        anchor = None
        if not is_root:
            if len(first.neighbors) == 1 and int(first.neighbors[0][1]) == BOND_SINGLE:
                anchor = int(first.neighbors[0][0])
            else:
                out.append(s)
                i += 1
                continue
        run_slots = [int(first.slot)]
        j = i + 1
        cap = ALKYL_MAX_LENGTH + (1 if is_root else 0)  # graft length <= ALKYL_MAX_LENGTH
        while (
            j < n
            and steps[j].rule_name == "atom_insert"
            and int(steps[j].action.atom_type) == _C
            and len(steps[j].action.neighbors) == 1
            and int(steps[j].action.neighbors[0][0]) == run_slots[-1]
            and int(steps[j].action.neighbors[0][1]) == BOND_SINGLE
            and len(run_slots) < cap
        ):
            run_slots.append(int(steps[j].action.slot))
            j += 1
        graft_slots = run_slots[1:] if is_root else run_slots
        graft_anchor = run_slots[0] if is_root else anchor
        if len(graft_slots) >= min_run:
            if is_root:
                out.append(s)  # keep the root atom, graft onto it
            out.append(
                RewriteStep(
                    "alkyl_graft",
                    build_graft(graft_anchor, (BOND_SINGLE,) * (len(graft_slots) - 1), graft_slots),
                )
            )
            changed = True
            i = j
        else:
            out.append(s)
            i += 1
    if not changed:
        return trace
    candidate = RewriteTrace(
        trace.source, trace.target, tuple(out), {**dict(trace.metadata), "alkyl_collapsed": True}
    )
    state = trace.source
    try:
        for st in out:
            state = system.apply(state, st.rule_name, st.action)
    except Exception:
        return trace
    if _smiles(state) is None or _smiles(state) != _smiles(trace.target):
        return trace  # verification failed -> keep the exact original teacher program
    return candidate
