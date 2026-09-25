"""Arm C: resample the PLANNED CONNECTIONS between successive edits, nothing else.

WHAT THIS CHANGES, AND WHAT IT DELIBERATELY DOES NOT
----------------------------------------------------
A structured program is built by the UNCHANGED machinery from the arm's own current
parent, with its region-replacement families, jump-plan access and archived-program
mutation/recombination rules intact, and it must pass that machinery's own success
filter first.  The resulting program is then kept as a RECIPE and exactly one thing is
varied: wherever an operation refers to an atom CREATED by an earlier operation, that
operand is resampled uniformly among the currently LIVE created atoms that leave the
operation admissible.

Preserved byte-for-byte: the primitive-operation sequence, its length, the block
boundaries, every payload field, and every binding to an ORIGINAL SOURCE atom
(`{"input": i}`).  Only `{"created": j}` operands move, and never the birth slot of an
`atom_insert`, which `execute_bound_program` assigns from `fresh_slot` and which
`EditProgram` requires to be the next sequential handle.

The originally prescribed binding stays ELIGIBLE wherever it remains admissible; the
intervention does not force a change.  A step whose live created set admits exactly one
binding is recorded as HAVING NO ALTERNATIVE rather than as intervened, so a panel on
which bindings are forced reports itself as uninformative instead of appearing to work.

This is NOT a consistent renaming of handles, which would leave the chemistry identical.
Substitution is per-operand against the live set, so whenever an alternative is chosen
the executed molecular graph genuinely differs.

ADMISSIBILITY IS THE EXECUTOR'S, NOT A SURROGATE
------------------------------------------------
Each candidate binding is offered to `execute_program` exactly as
`execute_bound_program` would, so every semantic-admission, liveness, connectivity,
valence and capacity check is the production one.  A candidate is eligible iff the real
executor accepts it.

FAILURE IS FROZEN AND VISIBLE
-----------------------------
If no admissible binding completes the next prescribed operation, the modified proposal
FAILS under the ordinary `ProgramExecutionError` path.  There is no fallback to the
original program, no replacement chain and no redraw.  The receipt distinguishes a
failure of the ORIGINAL structured construction (which happens before this module runs)
from one introduced by REBINDING, and records the step at which it occurred.

RANDOMNESS IS ISOLATED
----------------------
Binding choices come from a dedicated generator seeded from the run seed and a stable
proposal identifier via BLAKE2b -- never Python's process-salted `hash()`, and never by
consuming draws from the arbitration or proposal streams, so every surrounding stream
advances exactly as it would without the intervention.  With `identity=True` the module
must reproduce the original program and consume nothing, which is the development check
that the plumbing is inert when it should be.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict
from itertools import product
from time import perf_counter

import numpy as np

from compose_v4.control.edit_program import (
    EditProgram,
    ProgramExecutionError,
    _ordered,
    _slots,
    atom_signature,
)
from compose_v4.experiments.whole_ring_plan import execute_program, fresh_slot
from compose_v4.rewrite.kernel import InvalidRewrite, canonical_state_key
from compose_v4.rewrite.trace_shard import encode_state

BIRTH_RULE = "atom_insert"
#: Declared ceiling on the joint candidate set examined at one step.  When the live set
#: makes the product larger the step KEEPS ITS ORIGINAL BINDING and is recorded as
#: `cap_bound`; this is a stated restriction of the intervention, not a rescue on failure.
MAX_JOINT_CANDIDATES = 256


def binding_rng(run_seed: int, proposal_id: str) -> np.random.Generator:
    """Deterministic, process-independent generator for one proposal's bindings."""
    digest = hashlib.blake2b(
        f"{int(run_seed)}:{proposal_id}".encode(), digest_size=8
    ).digest()
    return np.random.default_rng(int.from_bytes(digest, "big"))


def created_operands(record: dict) -> tuple[int, ...]:
    """This mark's `created` operands, EXCLUDING an `atom_insert`'s own birth slot."""
    found: list[int] = []

    def collect(ref):
        kind, index = next(iter(ref.items()))
        if kind == "created":
            found.append(int(index))
        return ref

    _slots(record, collect)
    if record["executor_rule"] == BIRTH_RULE:
        birth = int(next(iter(record["payload"]["slot"].items()))[1])
        if birth in found:
            found.remove(birth)
    return tuple(dict.fromkeys(found))


def _substitute(record: dict, mapping: dict[int, int]) -> dict:
    """Rewrite created operands through `mapping`; the birth slot is never touched."""
    birth = None
    if record["executor_rule"] == BIRTH_RULE:
        birth = int(next(iter(record["payload"]["slot"].items()))[1])
    first_slot = {"seen": False}

    def swap(ref):
        kind, index = next(iter(ref.items()))
        if kind != "created":
            return ref
        index = int(index)
        if birth is not None and index == birth and not first_slot["seen"]:
            first_slot["seen"] = True
            return ref
        return {"created": int(mapping.get(index, index))}

    return _slots(record, swap)


def execute_rebound_program(
    source,
    program: EditProgram,
    assignment: tuple[int, ...],
    *,
    rng,
    max_primitives: int = 24,
    max_blocks: int = 5,
    mutable_slots: frozenset[int] | None = None,
    identity: bool = False,
) -> tuple[object, dict]:
    """Execute `program` with its created-atom operands resampled step by step.

    Mirrors `execute_bound_program` exactly except at steps carrying created operands,
    where the admissible joint bindings are found by OFFERING each to the executor and
    one is drawn uniformly.  `identity=True` keeps every original binding and consumes
    no randomness, which must reproduce `execute_bound_program` byte for byte.
    """
    if len(program.marks) > max_primitives or len(program.blocks) > max_blocks:
        raise ValueError("complete program exceeds its declared primitive/block budget")
    if len(assignment) != len(program.input_atoms) or len(set(assignment)) != len(
        assignment
    ):
        raise ValueError("program requires an injective complete input binding")
    for i, slot in enumerate(assignment):
        if atom_signature(source, slot)[:2] != program.input_atoms[i][:2]:
            raise ValueError("attachment element/charge differs from the program input")
        if mutable_slots is not None and slot not in mutable_slots:
            raise ValueError("program attachment touches preserved context")
        if any(
            int(source.bonds[slot, other]) != program.input_bonds[i][j]
            for j, other in enumerate(assignment)
        ):
            raise ValueError("attachment binding disagrees with required input connectivity")

    handles = {("input", i): slot for i, slot in enumerate(assignment)}
    # `created_count` must be independent of the LIVE list: handle indices are assigned
    # sequentially by `EditProgram` and never reused, so deriving the next index from
    # `len(live_created)` collides with a surviving handle as soon as one is deleted.
    live_created: list[int] = []
    created_count = 0
    current, states, actions = source, [encode_state(source)], []
    audit: list[dict] = []
    rewritten: list[str] = []

    for step, text in enumerate(program.marks):
        record = json.loads(text)
        rule = record["executor_rule"]
        if rule == BIRTH_RULE:
            ref = next(iter(record["payload"]["slot"].items()))
            handles[ref] = fresh_slot(current)

        operands = created_operands(record)
        available = tuple(h for h in live_created if ("created", h) in handles)
        joint = len(available) ** len(operands) if operands else 0
        cap_bound = bool(operands) and joint > MAX_JOINT_CANDIDATES

        if operands and not cap_bound and not identity and available:
            candidates = [
                dict(zip(operands, choice))
                for choice in product(available, repeat=len(operands))
            ]
        else:
            candidates = [{index: index for index in operands}] if operands else [{}]

        admissible: list[tuple[dict, object, dict]] = []
        last_error: Exception | None = None
        for mapping in candidates:
            candidate_record = _substitute(record, mapping) if mapping else record
            try:

                def resolve(ref, _record=candidate_record):
                    return handles[next(iter(ref.items()))]

                bound = _ordered(_slots(candidate_record, resolve))
                successor, result = execute_program(current, [bound])
            except (ValueError, InvalidRewrite, KeyError) as error:
                last_error = error
                continue
            admissible.append((mapping, successor, {"bound": bound, "result": result}))

        if not admissible:
            raise ProgramExecutionError(
                step,
                f"no admissible created-atom binding: {last_error}",
                {
                    "actions": actions,
                    "states": states,
                    "complete": False,
                    "failure_stage": "rebinding",
                    "candidates_examined": len(candidates),
                    "audit": audit,
                },
            )

        pick = 0 if (identity or len(admissible) == 1) else int(rng.integers(len(admissible)))
        mapping, current, carried = admissible[pick]
        identity_map = {index: index for index in operands}
        audit.append(
            {
                "step": step,
                "rule": rule,
                "created_operands": len(operands),
                "live_created": len(available),
                "candidates_examined": len(candidates),
                "admissible": len(admissible),
                "identity_admissible": any(m == identity_map for m, _, _ in admissible),
                "changed": bool(operands) and mapping != identity_map,
                "cap_bound": cap_bound,
            }
        )
        # Bookkeeping must follow the CHOSEN mark, never the prescribed one.  A rebound
        # `atom_delete` retires a DIFFERENT atom than the recipe named, so popping the
        # prescribed handle leaves the executed deletion's handle live and kills one that
        # still exists -- which `EditProgram.validate` then rejects on a later reference.
        chosen_record = _substitute(record, mapping) if mapping else record
        rewritten.append(
            json.dumps(chosen_record, sort_keys=True, separators=(",", ":"))
        )
        states.append(carried["result"]["states"][-1])
        actions.append(carried["bound"])
        if rule == BIRTH_RULE:
            live_created.append(created_count)
            created_count += 1
        if rule == "atom_delete":
            dead = next(iter(chosen_record["payload"]["v"].items()))
            handles.pop((dead[0], int(dead[1])))
            if dead[0] == "created" and int(dead[1]) in live_created:
                live_created.remove(int(dead[1]))

    steps_with_operands = [a for a in audit if a["created_operands"]]
    return current, {
        "schema_version": "rebound_edit_program_receipt_v1",
        "program_id": program.program_id,
        "assignment": list(assignment),
        "actions": actions,
        "states": states,
        "endpoint": canonical_state_key(current),
        "primitive_edits": len(actions),
        "blocks": [asdict(block) for block in program.blocks],
        "complete": True,
        "marks": tuple(rewritten),
        "rebinding": {
            "identity": bool(identity),
            "steps": len(program.marks),
            "steps_with_created_operands": len(steps_with_operands),
            "steps_with_alternatives": sum(
                1 for a in steps_with_operands if a["admissible"] > 1
            ),
            "steps_changed": sum(1 for a in audit if a["changed"]),
            "steps_cap_bound": sum(1 for a in audit if a["cap_bound"]),
            "audit": audit,
        },
        "support": "production_executor; reference_probability_not_evaluated",
    }


# ---- Production seam --------------------------------------------------------

#: Arm C is opt-in and ABSENT means byte-identical arm A.  There is deliberately no
#: "identity law" registered as an off switch: a value that looks like off but reaches
#: the executor by a different route is worse than no name at all.
ENV_FLAG = "PMO_BINDING_REBIND"
STAGE_NAME = "created_atom_rebinding"

#: Live footprint counters, streamed per round.  A mechanism that reaches a small
#: fraction of proposals must SAY SO while the run is happening, not be discovered
#: afterwards: the completion-law A/B spent a whole scored run measuring a treatment that
#: reached 20% of the proposals it governed.
STATS: dict = {
    "proposals": 0,
    "applicable": 0,
    "with_alternatives": 0,
    "changed_binding": 0,
    "changed_endpoint": 0,
    "refused_rebinding": 0,
    "refused_reschedule": 0,
    "refused_replay": 0,
    "seconds": 0.0,
}


def _observe(**counts) -> None:
    for key, value in counts.items():
        STATS[key] = STATS.get(key, 0) + value


def snapshot_stats(reset: bool = True) -> dict:
    """Per-round footprint, then zero the counters for the next round."""
    n = max(STATS["proposals"], 1)
    out = {
        **{k: v for k, v in STATS.items()},
        "applicable_rate": round(STATS["applicable"] / n, 4),
        "changed_endpoint_rate": round(STATS["changed_endpoint"] / n, 4),
        "seconds_per_proposal": round(STATS["seconds"] / n, 4),
    }
    if reset:
        for key, value in STATS.items():
            STATS[key] = 0 if isinstance(value, int) else 0.0
    return out


def binding_arm_enabled() -> bool:
    """True iff arm C is requested.  Arm B and arm C are MUTUALLY EXCLUSIVE.

    Arm B discards the structured program and replaces it with a uniform legal chain, so
    rebinding its created atoms would measure neither arm.  Enabling both is a launch
    error and fails closed here rather than producing a third, unnamed condition.
    """
    from compose_v4.control.pmo_uniform_chain import ENV_FLAG as CHAIN_FLAG

    enabled = os.environ.get(ENV_FLAG) == "1"
    if enabled and os.environ.get(CHAIN_FLAG) == "1":
        raise RuntimeError(
            f"{ENV_FLAG} and {CHAIN_FLAG} are mutually exclusive ablation arms"
        )
    return enabled


def execute_program_graph_rebound(
    source,
    graph,
    assignment,
    *,
    run_seed: int,
    occurrence: int,
    max_primitives: int = 24,
    max_blocks: int = 5,
    mutable_slots=None,
):
    """`execute_program_graph` with created-atom operands resampled.

    The scheduled program -- the recipe whose operation sequence, payloads and source
    bindings are being HELD FIXED -- is produced by the unchanged scheduler, so this
    differs from production in exactly one call.  Every derived field is recomputed from
    the MODIFIED execution, never carried over from the prescribed program, so selection
    features describe the molecule that was actually built.

    `occurrence` is a stable per-run proposal ORDINAL and is part of the generator key.
    Without it the key would be purely content-addressed (`program_id` is a content hash
    of the marks), so re-proposing the same recipe on the same parent would replay the
    identical bindings every time and make arm C far less stochastic than intended.

    SELF-CONSISTENCY IS ENFORCED, NOT ASSUMED.  Rebinding changes the dataflow graph, so
    the modified program's own canonical schedule need not reproduce the order that was
    just executed -- and the archive's admission check replays a stored program through
    `compile_program_graph` + `scheduled_program`.  A proposal whose modified program does
    not reschedule and replay to exactly the molecule and state sequence it produced is
    therefore REFUSED here, rather than being stored as an entry whose recorded program
    explains a different molecule.  That refusal is labelled and never rescued.
    """
    from compose_v4.control.docking_value import identity as _identity
    from compose_v4.control.edit_program import execute_bound_program
    from compose_v4.control.edit_program_graph import (
        changed_input_sites,
        compile_program_graph,
        scheduled_program,
    )

    program, order, timeline = scheduled_program(graph, source.n_real_atoms)
    proposal_id = (
        f"{program.program_id}:{canonical_state_key(source)}"
        f":{tuple(assignment)}:{int(occurrence)}"
    )
    began = perf_counter()
    try:
        product, receipt = execute_rebound_program(
            source,
            program,
            assignment,
            rng=binding_rng(run_seed, proposal_id),
            max_primitives=max_primitives,
            max_blocks=max_blocks,
            mutable_slots=mutable_slots,
        )
    except ProgramExecutionError:
        _observe(proposals=1, refused_rebinding=1, seconds=perf_counter() - began)
        raise
    audit = receipt["rebinding"]
    _observe(
        proposals=1,
        applicable=1 if audit["steps_with_created_operands"] else 0,
        with_alternatives=1 if audit["steps_with_alternatives"] else 0,
        changed_binding=1 if audit["steps_changed"] else 0,
        seconds=perf_counter() - began,
    )

    modified = EditProgram(
        program.input_atoms,
        program.input_bonds,
        program.environments,
        tuple(receipt["marks"]),
        program.blocks,
    )
    rescheduled, modified_order, modified_timeline = scheduled_program(
        compile_program_graph(modified), source.n_real_atoms
    )
    if tuple(rescheduled.marks) != tuple(modified.marks):
        _observe(refused_reschedule=1)
        raise ProgramExecutionError(
            len(receipt["actions"]),
            "modified program does not reschedule to the executed order",
            {
                "actions": receipt["actions"],
                "states": receipt["states"],
                "complete": False,
                "failure_stage": "rebinding_reschedule",
            },
        )
    _, replay = execute_bound_program(
        source,
        rescheduled,
        assignment,
        max_primitives=max_primitives,
        max_blocks=max_blocks,
        mutable_slots=mutable_slots,
    )
    if replay["endpoint"] != receipt["endpoint"] or replay["states"] != receipt["states"]:
        _observe(refused_replay=1)
        raise ProgramExecutionError(
            len(receipt["actions"]),
            "modified program fails exact replay of its own execution",
            {
                "actions": receipt["actions"],
                "states": receipt["states"],
                "complete": False,
                "failure_stage": "rebinding_replay",
            },
        )
    # "Did the intervention change the MOLECULE" is a counterfactual against the
    # PRESCRIBED program, not against the source, and the two differ: a substituted
    # binding can legally rebuild the same canonical molecule (3 of 40 on the development
    # panel).  The prescribed program is therefore executed too whenever a binding
    # actually moved.  It is deterministic and consumes no randomness, so it cannot
    # perturb the arm; it costs one extra execution on the minority of proposals that
    # changed, and it is what lets the scored run report its own endpoint-change rate
    # instead of inferring one.
    prescribed_endpoint = None
    if audit["steps_changed"]:
        try:
            _, prescribed = execute_bound_program(
                source,
                program,
                assignment,
                max_primitives=max_primitives,
                max_blocks=max_blocks,
                mutable_slots=mutable_slots,
            )
            prescribed_endpoint = prescribed["endpoint"]
            if prescribed_endpoint != receipt["endpoint"]:
                _observe(changed_endpoint=1)
        except (ValueError, InvalidRewrite):
            # The prescribed program is arm A's own executed program, so this should not
            # arise; if it does the comparison is recorded as unavailable rather than
            # silently counted either way.
            prescribed_endpoint = "unavailable"
    return product, {
        **receipt,
        # Dependency metadata describes the MODIFIED references.  The order and the
        # capacity timeline are the modified program's own, which the check above has
        # just proved identical to the order actually executed -- so nothing was
        # silently rescheduled and the held-fixed operation sequence is intact.
        "graph_id": _identity(compile_program_graph(modified).payload()),
        "selected_program_id": modified.program_id,
        "prescribed_program_id": graph.program.program_id,
        "block_order": modified_order,
        "capacity_timeline": modified_timeline,
        "planned_block_order": order,
        "planned_capacity_timeline": timeline,
        "modified_program": modified.payload(),
        "scheduling": "deterministic serial; executor revalidation after every primitive",
        "proposal_law": "new optimization proposal; no reference likelihood claimed",
        "actual_changes": changed_input_sites(source, product, receipt["actions"]),
        "ablation_stage": STAGE_NAME,
        "ablation_proposal_id": proposal_id,
        "prescribed_endpoint": prescribed_endpoint,
        "endpoint_changed_by_rebinding": (
            None
            if prescribed_endpoint in (None, "unavailable")
            else prescribed_endpoint != receipt["endpoint"]
        ),
    }
