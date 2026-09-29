"""Opt-in region-to-pendant replacement, composed solely of existing rewrites.

Remove the released region while the frozen complement stays connected, then
delegate construction to an existing parameterized ring program. No endpoint
templates, oracle filters, or changes to the molecular executor occur here.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from compose_v4.chem.molecular_graph import MolecularGraph, is_element
from compose_v4.control.region_rewrite import RewriteContext
from compose_v4.control.ring_program import RingProgress, RingSpec, ring_spec
from compose_v4.rewrite.trace_shard import decode_state, encode_state

PREFIX = "replace_region:"


@lru_cache(maxsize=128)
def replacement_spec(option: str) -> RingSpec | None:
    if not option.startswith(PREFIX):
        return None
    spec = ring_spec(option[len(PREFIX) :])
    if spec is None or spec.topology != "pendant":
        raise ValueError("region replacement currently requires an explicit pendant construction")
    return spec


def replacement_options(ring_options):
    return tuple(PREFIX + o for o in ring_options if ring_spec(o).topology == "pendant")


def released_slots(graph, context):
    return frozenset(
        int(i) for i in context.locus - context.frozen if is_element(graph.atom_types[i])
    )


def cycle_rank(graph):
    slots = np.flatnonzero(is_element(graph.atom_types))
    edges = np.count_nonzero(graph.bonds[np.ix_(slots, slots)]) // 2
    return int(edges - len(slots) + 1)  # complete connected states only


def maximum_horizon(graph, context, spec):
    return len(released_slots(graph, context)) + cycle_rank(graph) + spec.horizon


def applicable(graph, context, spec):
    slots = released_slots(graph, context)
    return bool(
        slots
        and context.frozen
        and context.k_components == 1
        and context.terminal_context_slots
        and graph.n_real_atoms - len(slots) + spec.growth <= 40
    )


@dataclass(frozen=True)
class ReplacementProgress:
    removed: int = 0
    opened: int = 0
    build_origin: MolecularGraph | None = None
    build_step: int = 0
    ring: RingProgress | None = None

    def validate(self, graph, origin, context: RewriteContext, step, spec):
        if any(type(v) is not int or v < 0 for v in (self.removed, self.opened, self.build_step)):
            raise ValueError("replacement counters must be nonnegative integers")
        # Before growth, the context's locus contains only original slots.
        original_slots = released_slots(origin, context) - context.frozen
        if self.opened > cycle_rank(origin) or self.removed > len(original_slots):
            raise ValueError("replacement pruning exceeds its structural bound")
        if step != self.removed + self.opened + self.build_step:
            raise ValueError("replacement primitive clock disagrees with phase counters")
        if self.build_origin is None:
            if self.ring is not None or self.build_step or self.removed == len(original_slots):
                raise ValueError("invalid replacement pruning phase")
            if graph.n_real_atoms != origin.n_real_atoms - self.removed:
                raise ValueError("replacement pruning changed unexpected atom count")
            if (
                sum(is_element(graph.atom_types[i]) for i in original_slots)
                != len(original_slots) - self.removed
            ):
                raise ValueError("replacement pruned outside the original released region")
        else:
            if self.removed != len(original_slots) or self.ring is None:
                raise ValueError("replacement growth began before removing the released region")
            if self.build_origin.n_real_atoms != origin.n_real_atoms - self.removed:
                raise ValueError("replacement construction origin has wrong cardinality")
            if any(is_element(self.build_origin.atom_types[i]) for i in original_slots):
                raise ValueError("replacement build origin retains an original released atom")
            self.ring.validate(graph, self.build_origin, context.locus, self.build_step, spec)

    def complete(self, spec):
        return self.build_origin is not None and self.build_step == spec.horizon

    def key(self):
        graph = self.build_origin
        encoded = (
            None
            if graph is None
            else tuple(
                (a.dtype.str, a.shape, a.tobytes())
                for a in (
                    graph.atom_types,
                    graph.bonds,
                    graph.formal_charges,
                    graph.implicit_h_counts,
                )
            )
        )
        return self.removed, self.opened, encoded, self.build_step, self.ring

    def payload(self):
        return {
            "schema_version": "region_replacement_progress_v1",
            "removed": self.removed,
            "opened": self.opened,
            "build_step": self.build_step,
            "build_origin": None if self.build_origin is None else encode_state(self.build_origin),
            "ring": None if self.ring is None else self.ring.payload(),
        }

    @classmethod
    def from_payload(cls, value):
        if (
            set(value)
            != {"schema_version", "removed", "opened", "build_origin", "build_step", "ring"}
            or value["schema_version"] != "region_replacement_progress_v1"
        ):
            raise ValueError("unknown replacement progress schema")
        return cls(
            value["removed"],
            value["opened"],
            None if value["build_origin"] is None else decode_state(value["build_origin"]),
            value["build_step"],
            None if value["ring"] is None else RingProgress.from_payload(value["ring"]),
        )
