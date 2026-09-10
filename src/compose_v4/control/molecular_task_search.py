"""WHERE/WHAT/HOW adapter over the existing executable option reference.

Selections are augmented graph states, not molecular edits. Only HOW calls the
executor, and option completion returns to WHERE with its true remaining budget.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

from compose_v4.chem.molecular_graph import NULL_IDX, MolecularGraph, is_element
from compose_v4.control.carbonyl_option import INSERT_RING_CARBONYL_OPTION, CarbonylProgress
from compose_v4.control.fused_option import BUILD_FUSED_RING_OPTION, FusedProgress
from compose_v4.control.option_continuation import (
    OptionContinuationKernel,
    OptionState,
    exact_graph_key,
)
from compose_v4.control.option_selector import (
    applicable_options,
    balanced_option_prior,
    option_horizon,
    retain_product_applicable_options,
)
from compose_v4.control.region import Region, enumerate_regions
from compose_v4.control.region_rewrite import Lineage, admissible_indices
from compose_v4.control.region_selector import region_distribution
from compose_v4.control.ring_program import RingProgress, default_ring_options, ring_spec
from compose_v4.control.task_search import SearchRow
from compose_v4.experiments.t4_warm_continuation import exact_context
from compose_v4.rewrite.kernel import canonical_state_key


@dataclass(frozen=True)
class MolecularSearchState:
    graph: MolecularGraph
    lineage: Lineage
    budget: int
    root_id: str
    stage: str = "where"
    region: Region | None = None
    active: OptionState | None = None

    def __post_init__(self):
        if type(self.budget) is not int or self.budget < 0 or not self.root_id:
            raise ValueError("hierarchy requires a nonnegative primitive budget and root identity")
        if self.stage not in ("where", "what", "how") or not 1 <= self.graph.n_real_atoms <= 40:
            raise ValueError(
                "hierarchy requires a declared stage and a complete 1..40-atom molecule"
            )
        if (self.stage == "where") != (self.region is None):
            raise ValueError("region must exist exactly during WHAT and HOW")
        if (self.stage == "how") != (self.active is not None):
            raise ValueError("an active option is required exactly during HOW")
        if self.active is not None and (
            self.active.remaining > self.budget
            or self.active.remaining < 1
            or exact_graph_key(self.graph) != exact_graph_key(self.active.graph)
        ):
            raise ValueError("active option exceeds budget or disagrees with exact graph")
        if self.budget == 0 and self.stage != "where":
            raise ValueError("zero remaining primitive budget must be a terminal WHERE state")

    def key(self) -> tuple:
        return (
            self.root_id,
            self.stage,
            self.budget,
            exact_graph_key(self.graph),
            tuple(sorted(self.lineage.id_of.items())),
            self.lineage.next_id,
            self.region.key() if self.region else None,
            self.active.key() if self.active else None,
        )

    @classmethod
    def start(cls, graph: MolecularGraph, *, budget: int, root_id: str):
        return cls(
            graph, Lineage.initial(np.flatnonzero(is_element(graph.atom_types))), budget, root_id
        )


class MolecularHierarchy:
    def __init__(
        self,
        kernel: OptionContinuationKernel,
        *,
        generic_horizon: int = 3,
        ring_options: tuple[str, ...] | None = None,
        lazy_applicability: bool = False,
        include_carbonyl_options: bool = False,
    ):
        if type(generic_horizon) is not int or generic_horizon < 1:
            raise ValueError("generic_horizon must be a positive primitive count")
        self.kernel, self.generic_horizon = kernel, generic_horizon
        self.ring_options = default_ring_options() if ring_options is None else ring_options
        self.lazy_applicability = lazy_applicability
        if type(include_carbonyl_options) is not bool:
            raise ValueError("include_carbonyl_options must be an explicit boolean")
        self.include_carbonyl_options = include_carbonyl_options
        self._selection_rows = {}

    def sample_reference(self, node: MolecularSearchState, rng) -> MolecularSearchState | None:
        """Reference-only planning draw; no task tilt or sampled support truncation."""
        if node.stage != "how":
            key = node.key()
            if key not in self._selection_rows:
                self._selection_rows[key] = self.row(node)
            row = self._selection_rows[key]
            if not row.successors:
                return None
            return row.successors[int(rng.choice(len(row.successors), p=row.reference))]
        active = self.kernel.lazy_row(node.active).sample(rng)
        return None if active is None else self._successor(node, active)

    @staticmethod
    def _successor(node, active):
        if active.remaining:
            return MolecularSearchState(
                active.graph,
                active.lineage,
                node.budget - 1,
                node.root_id,
                "how",
                node.region,
                active,
            )
        return MolecularSearchState(active.graph, active.lineage, node.budget - 1, node.root_id)

    def option_state(self, node: MolecularSearchState, option: str, *, context=None) -> OptionState:
        if context is None:
            context = exact_context(node.graph, canonical_state_key(node.graph), node.region)
        horizon = option_horizon(option, min(self.generic_horizon, node.budget))
        bundle = hashlib.sha256(repr((node.key(), option)).encode()).hexdigest()[:20]
        return OptionState(
            node.graph,
            node.graph,
            context,
            node.lineage,
            option,
            0,
            horizon,
            bundle,
            FusedProgress() if option == BUILD_FUSED_RING_OPTION else None,
            ring_progress=RingProgress() if ring_spec(option) is not None else None,
            carbonyl_progress=CarbonylProgress() if option == INSERT_RING_CARBONYL_OPTION else None,
        )

    def row(self, node: MolecularSearchState) -> SearchRow[MolecularSearchState]:
        if node.budget == 0:
            return SearchRow((), (), (), (), 0.1)
        if node.stage == "where":
            regions = [
                r for r in enumerate_regions(canonical_state_key(node.graph)) if 1 <= r.size <= 24
            ]
            regions, scores, _ = region_distribution(regions)
            return SearchRow(
                tuple(
                    MolecularSearchState(
                        node.graph, node.lineage, node.budget, node.root_id, "what", r
                    )
                    for r in regions
                ),
                tuple(repr(r.key()) for r in regions),
                tuple(s.base_probability for s in scores),
                tuple(s.floor_probability for s in scores),
                0.2,
            )
        if node.stage == "what":
            families, actions, _ = self.kernel.enumerate_law(node.graph)
            context = exact_context(node.graph, canonical_state_key(node.graph), node.region)
            indices, _ = admissible_indices(families, actions, context)
            free = min(int(np.sum(node.graph.atom_types == NULL_IDX)), 40 - node.graph.n_real_atoms)
            options = tuple(
                o
                for o in applicable_options(
                    families,
                    indices,
                    n_free_slots=free,
                    include_fused=True,
                    include_carbonyl=self.include_carbonyl_options,
                    ring_options=self.ring_options,
                )
                if option_horizon(o, min(self.generic_horizon, node.budget)) <= node.budget
            )
            states = {o: self.option_state(node, o, context=context) for o in options}
            options = retain_product_applicable_options(
                options,
                lambda o: (
                    self.kernel.lazy_row(states[o]).has_product()
                    if self.lazy_applicability
                    else bool(self.kernel.row(states[o]).successors)
                ),
            )
            return SearchRow(
                tuple(
                    MolecularSearchState(
                        node.graph,
                        node.lineage,
                        node.budget,
                        node.root_id,
                        "how",
                        node.region,
                        states[o],
                    )
                    for o in options
                ),
                options,
                tuple(balanced_option_prior(options, exploration=0)),
                tuple(np.full(len(options), 1 / len(options))),
                0.1,
            )
        row = self.kernel.row(node.active)
        successors = [self._successor(node, active) for active in row.successors]
        labels = tuple(f"{rule}:{action!r}" for rule, action in self.kernel.marks(node.active))
        return SearchRow(tuple(successors), labels, row.probabilities, row.probabilities, 0.1)
