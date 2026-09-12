"""Transferable, attachment-bound primitive programs with persistent atom handles.

These are optional executor-supported optimization proposals, not samples from a
certified R_theta law. A program contains no target endpoint or source molecule.
Its primitive semantics and validation belong exclusively to the existing codec
and executor. Input and newly-created atom identities never alias on slot reuse.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import asdict, dataclass
from itertools import pairwise

import numpy as np

from compose_v4.chem.molecular_graph import MolecularGraph, is_element
from compose_v4.control.docking_value import identity
from compose_v4.experiments.whole_ring_plan import execute_program, fresh_slot
from compose_v4.rewrite.action_codec_v4 import decode_action, encode_action
from compose_v4.rewrite.kernel import InvalidRewrite, canonical_state_key
from compose_v4.rewrite.trace_shard import encode_state

SCHEMA = "attachment_edit_program_v1"


def _json(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _slots(record: dict, transform: Callable) -> dict:
    """Map only slot operands of the authoritative V4 payload shapes."""
    value = json.loads(_json(record))
    rule, payload = value["executor_rule"], value["payload"]
    if rule == "atom_insert":
        payload["slot"] = transform(payload["slot"])
        payload["neighbors"] = [[transform(i), order] for i, order in payload["neighbors"]]
    elif rule in ("atom_delete", "atom_restate_semantic"):
        payload["v"] = transform(payload["v"])
    elif rule in ("cycle_close", "cycle_open", "bond_reorder", "bond_reroute"):
        for field in ("a", "b", "u", "v") if rule == "bond_reroute" else ("a", "b"):
            payload[field] = transform(payload[field])
    elif rule == "ring_system_restate":
        for change in payload["changes"]:
            change["a"], change["b"] = transform(change["a"]), transform(change["b"])
    else:
        raise ValueError(f"unsupported program executor rule: {rule!r}")
    return value


def _ordered(record: dict) -> dict:
    """Canonicalize undirected operands after binding, then use the real codec."""
    rule, payload = record["executor_rule"], record["payload"]
    if rule in ("cycle_close", "cycle_open", "bond_reorder", "bond_reroute"):
        payload["a"], payload["b"] = sorted((payload["a"], payload["b"]))
        if rule == "bond_reroute":
            payload["u"], payload["v"] = sorted((payload["u"], payload["v"]))
    if rule == "ring_system_restate":
        for change in payload["changes"]:
            change["a"], change["b"] = sorted((change["a"], change["b"]))
        payload["changes"].sort(key=lambda item: (item["a"], item["b"]))
    family, action = decode_action(record)
    return encode_action(family, action)


def atom_signature(graph: MolecularGraph, slot: int) -> tuple[int, int, int, int]:
    if (
        type(slot) is not int
        or not 0 <= slot < graph.n_atoms
        or not is_element(graph.atom_types[slot])
    ):
        raise ValueError(f"invalid real-atom attachment slot: {slot!r}")
    return (
        int(graph.atom_types[slot]),
        int(graph.formal_charges[slot]),
        int(graph.implicit_h_counts[slot]),
        int(np.count_nonzero(graph.bonds[slot])),
    )


def environment(graph: MolecularGraph, slot: int) -> str:
    """Slot-independent one-hop attachment context; no endpoint/task information."""
    neighbors = sorted(
        (int(graph.bonds[slot, other]), *atom_signature(graph, int(other)))
        for other in np.flatnonzero(graph.bonds[slot])
    )
    return _json((atom_signature(graph, slot), neighbors))


@dataclass(frozen=True)
class ProgramBlock:
    label: str
    stop: int


@dataclass(frozen=True)
class EditProgram:
    input_atoms: tuple[tuple[int, int, int, int], ...]
    input_bonds: tuple[tuple[int, ...], ...]
    environments: tuple[str, ...]
    marks: tuple[str, ...]
    blocks: tuple[ProgramBlock, ...]

    def __post_init__(self):
        n = len(self.input_atoms)
        if not n or len(self.environments) != n or len(self.input_bonds) != n:
            raise ValueError("program input atom/context dimensions disagree")
        if any(len(row) != n for row in self.input_bonds):
            raise ValueError("program input bond matrix has the wrong shape")
        if any(len(row) != 4 or any(type(v) is not int for v in row) for row in self.input_atoms):
            raise ValueError("program atom signatures must have four integer fields")
        if not self.marks or not self.blocks:
            raise ValueError("program needs primitive marks and complete blocks")
        if any(not b.label or type(b.stop) is not int for b in self.blocks):
            raise ValueError("program block label/stop is malformed")
        stops = (0, *(b.stop for b in self.blocks))
        if stops[-1] != len(self.marks) or any(b <= a for a, b in pairwise(stops)):
            raise ValueError("program blocks must partition the full primitive trace")
        active = {("input", i) for i in range(n)}
        next_created = 0
        for text in self.marks:
            record = json.loads(text)
            rule = record["executor_rule"]
            if rule == "atom_insert":
                ref = record["payload"]["slot"]
                if ref != {"created": next_created}:
                    raise ValueError("birth handles must be sequential and never reused")
                active.add(("created", next_created))
                next_created += 1

            def validate(ref):
                if not isinstance(ref, dict) or len(ref) != 1:
                    raise ValueError("program slot requires one typed atom reference")
                kind, index = next(iter(ref.items()))
                if type(index) is not int or (kind, index) not in active:
                    raise ValueError(f"unbound, deleted or malformed atom reference: {ref}")
                return index if kind == "input" else n + index

            _ordered(_slots(record, validate))
            if rule == "atom_delete":
                active.remove(next(iter(record["payload"]["v"].items())))

    @property
    def program_id(self) -> str:
        return identity(self.payload())

    def payload(self) -> dict:
        return {"schema_version": SCHEMA, **asdict(self)}

    @classmethod
    def from_payload(cls, payload: dict) -> EditProgram:
        if payload.get("schema_version") != SCHEMA or set(payload) != {
            "schema_version",
            "input_atoms",
            "input_bonds",
            "environments",
            "marks",
            "blocks",
        }:
            raise ValueError("unexpected edit-program schema or fields")
        return cls(
            tuple(tuple(row) for row in payload["input_atoms"]),
            tuple(tuple(row) for row in payload["input_bonds"]),
            tuple(payload["environments"]),
            tuple(payload["marks"]),
            tuple(ProgramBlock(**row) for row in payload["blocks"]),
        )


def extract_program(
    source: MolecularGraph, stages: list[dict]
) -> tuple[EditProgram, tuple[int, ...]]:
    """Verify a stored trace, erase source addresses, retain cross-block handles."""
    original = encode_state(source)
    roots, handles, marks, blocks = [], {}, [], []
    current = source
    created = 0
    for stage_index, stage in enumerate(stages):
        if encode_state(current) != stage["states"][0] or not stage["actions"]:
            raise ValueError("saved program boundary is empty or loses exact state")
        after, receipt = execute_program(current, stage["actions"])
        if receipt["states"] != stage["states"] or receipt["endpoint"] != stage["endpoint"]:
            raise ValueError("saved program trace fails exact executor replay")
        for record in stage["actions"]:
            rule, action = decode_action(record)
            if rule == "atom_insert":
                handles[action.slot] = {"created": created}
                created += 1

            def reference(slot):
                if slot not in handles:
                    atom_signature(source, slot)
                    if slot not in roots:
                        roots.append(slot)
                    handles[slot] = {"input": roots.index(slot)}
                return handles[slot]

            marks.append(_json(_slots(record, reference)))
            if rule == "atom_delete":
                handles.pop(action.v)
        blocks.append(ProgramBlock(stage.get("name", f"block_{stage_index}"), len(marks)))
        current = after
    if encode_state(source) != original:
        raise RuntimeError("program extraction mutated its source")
    program = EditProgram(
        tuple(atom_signature(source, i) for i in roots),
        tuple(tuple(int(source.bonds[i, j]) for j in roots) for i in roots),
        tuple(environment(source, i) for i in roots),
        tuple(marks),
        tuple(blocks),
    )
    return program, tuple(roots)


@dataclass(frozen=True)
class BindingCensus:
    assignments: tuple[tuple[int, ...], ...]
    context_distances: tuple[int, ...]
    visits: int
    truncated: bool


def attachment_bindings(
    program: EditProgram,
    graph: MolecularGraph,
    *,
    max_bindings: int = 32,
    max_visits: int = 4096,
    contextual: bool = True,
    mutable_slots: frozenset[int] | None = None,
) -> BindingCensus:
    """Bounded injective attachment search; report caps instead of claiming coverage."""
    if any(type(v) is not int or v < 1 for v in (max_bindings, max_visits)):
        raise ValueError("attachment search caps must be positive integers")
    real = tuple(int(i) for i in np.flatnonzero(is_element(graph.atom_types)))
    if mutable_slots is not None and not mutable_slots <= set(real):
        raise ValueError("mutable slots contain an absent or non-element atom")
    signatures = {i: atom_signature(graph, i) for i in real}
    contexts = {i: environment(graph, i) for i in real}
    candidates, distances = [], []
    for signature, context in zip(program.input_atoms, program.environments, strict=True):
        options = [i for i in real if signatures[i][:2] == signature[:2]]
        if mutable_slots is not None:
            options = [i for i in options if i in mutable_slots]
        distance = {
            i: int(contexts[i] != context)
            + sum(a != b for a, b in zip(signatures[i], signature, strict=True))
            for i in options
        }
        candidates.append(sorted(options, key=lambda i: (distance[i] if contextual else 0, i)))
        distances.append(distance)
    order = sorted(range(len(candidates)), key=lambda i: (len(candidates[i]), i))
    binding, assignments, scores = {}, [], []
    visits, truncated = 0, False

    def visit(depth):
        nonlocal visits, truncated
        if len(assignments) >= max_bindings or visits >= max_visits:
            truncated = True
            return
        visits += 1
        if depth == len(order):
            row = tuple(binding[i] for i in range(len(order)))
            assignments.append(row)
            scores.append(sum(distances[i][slot] for i, slot in enumerate(row)))
            return
        i = order[depth]
        for slot in candidates[i]:
            if slot in binding.values() or any(
                int(graph.bonds[slot, other]) != program.input_bonds[i][j]
                for j, other in binding.items()
            ):
                continue
            binding[i] = slot
            visit(depth + 1)
            del binding[i]
            if truncated:
                return

    visit(0)
    return BindingCensus(tuple(assignments), tuple(scores), visits, truncated)


class ProgramExecutionError(ValueError):
    """A proposal failed; its valid prefix is not a completed endpoint candidate."""

    def __init__(self, step: int, reason: str, receipt: dict):
        super().__init__(f"program step {step}: {reason}")
        self.step, self.receipt = step, receipt


def execute_bound_program(
    source: MolecularGraph,
    program: EditProgram,
    assignment: tuple[int, ...],
    *,
    max_primitives: int = 24,
    max_blocks: int = 5,
    mutable_slots: frozenset[int] | None = None,
) -> tuple[MolecularGraph, dict]:
    """Execute a fully chosen program without intermediate task-score pruning."""
    if len(program.marks) > max_primitives or len(program.blocks) > max_blocks:
        raise ValueError("complete program exceeds its declared primitive/block budget")
    if len(assignment) != len(program.input_atoms) or len(set(assignment)) != len(assignment):
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
    current, states, actions = source, [encode_state(source)], []
    for step, text in enumerate(program.marks):
        record = json.loads(text)
        rule = record["executor_rule"]
        try:
            if rule == "atom_insert":
                ref = next(iter(record["payload"]["slot"].items()))
                handles[ref] = fresh_slot(current)

            def resolve(ref):
                return handles[next(iter(ref.items()))]

            bound = _ordered(_slots(record, resolve))
            current, result = execute_program(current, [bound])
        except (ValueError, InvalidRewrite) as error:
            raise ProgramExecutionError(
                step, str(error), {"actions": actions, "states": states, "complete": False}
            ) from error
        states.append(result["states"][-1])
        actions.append(bound)
        if rule == "atom_delete":
            handles.pop(next(iter(record["payload"]["v"].items())))
    return current, {
        "schema_version": "bound_edit_program_receipt_v1",
        "program_id": program.program_id,
        "assignment": list(assignment),
        "actions": actions,
        "states": states,
        "endpoint": canonical_state_key(current),
        "primitive_edits": len(actions),
        "blocks": [asdict(block) for block in program.blocks],
        "complete": True,
        "support": "production_executor; reference_probability_not_evaluated",
    }
