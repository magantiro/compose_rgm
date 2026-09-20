"""End-to-end option-boundary runtime over the production molecular hierarchy.

The runtime composes the existing three controller levels without redefining
their scientific roles:

* WHERE is sampled from the frozen region row exposed by ``MolecularHierarchy``;
* WHAT is proposed by ``OptionBoundaryController`` against the balanced option
  reference law;
* HOW is executed by ``sample_option_trajectory`` through the production
  ``OptionContinuationKernel`` and executor.

Particle accounting uses the complete reference/proposal path ratio in log
space. The runtime accepts a frozen, oriented utility callback but owns no task
oracle. Candidate docking/scoring remains a separate locked-batch operation.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass

import numpy as np

from compose_v4.control.batch_acquisition import select_qpo_batch
from compose_v4.control.docking_value import identity
from compose_v4.control.graph_geometry import structural_displacement, topology
from compose_v4.control.molecular_search_codec import (
    decode_option,
    decode_search_state,
    encode_search_state,
)
from compose_v4.control.molecular_task_search import MolecularHierarchy, MolecularSearchState
from compose_v4.control.option_continuation import (
    OptionState,
    sample_option_trajectory,
)
from compose_v4.control.option_controller import OptionBoundaryController
from compose_v4.control.option_features import (
    boundary_value_features,
    option_state_features,
)
from compose_v4.control.option_selector import BUILD_RING_SYSTEM_OPTION, GENERIC_OPTION
from compose_v4.control.persistent_option_smc import (
    OptionTransition,
    PersistentOptionParticle,
    PersistentOptionPopulation,
    advance_population,
)
from compose_v4.control.trajectory_value import molecule_features
from compose_v4.rewrite.kernel import canonical_state_key


@dataclass(frozen=True)
class HowControl:
    """Frozen configuration for the existing within-option controller."""

    estimator: str
    snapshot_id: str
    max_expansions: int
    max_terminal_evaluations: int
    samples_per_successor: int = 32
    confidence_alpha: float = 0.05
    max_rollouts: int = 1024

    def __post_init__(self) -> None:
        if self.estimator not in ("reference", "exact", "sampled") or not self.snapshot_id:
            raise ValueError("HOW control requires a supported estimator and snapshot identity")
        for name in (
            "max_expansions",
            "max_terminal_evaluations",
            "samples_per_successor",
            "max_rollouts",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"HOW {name} must be a nonnegative integer")
        if self.samples_per_successor < 1 or self.max_rollouts < 1:
            raise ValueError("sampled HOW budgets must be positive")
        if not math.isfinite(self.confidence_alpha) or not 0 < self.confidence_alpha < 1:
            raise ValueError("HOW confidence alpha must lie in (0, 1)")


def _stream_seed(identity_value: str, boundary: int) -> int:
    digest = hashlib.sha256(f"{identity_value}|{boundary}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


class OptionControllerRuntime:
    """A frozen three-level controller advanced one complete option at a time."""

    def __init__(
        self,
        hierarchy: MolecularHierarchy,
        controller: OptionBoundaryController,
        *,
        total_options: int,
        reference_snapshot: str,
        feature_snapshot: str,
        utility_snapshot: str,
        how: HowControl,
        utility: Callable[[MolecularSearchState], float],
        how_terminal_weight: Callable[[OptionState], float],
        how_fallback_weight: Callable[[OptionState], float],
        molecular_feature_fn: Callable[[MolecularSearchState], Sequence[float]] | None = None,
        enable_build_ring_system: bool = False,
        allowed_options: frozenset[str] | None = None,
    ) -> None:
        if hierarchy.kernel is None:
            raise ValueError("runtime requires a production option continuation kernel")
        if (
            isinstance(total_options, bool)
            or not isinstance(total_options, int)
            or not 1 <= total_options <= 16
        ):
            raise ValueError("total option budget must be an integer in 1..16")
        if not reference_snapshot or not feature_snapshot or not utility_snapshot:
            raise ValueError("reference, feature and utility snapshots are required")
        if type(enable_build_ring_system) is not bool:
            raise ValueError("compound-option activation must be an explicit boolean")
        if allowed_options is not None and (
            not isinstance(allowed_options, frozenset) or GENERIC_OPTION not in allowed_options
        ):
            raise ValueError("an allowed option set must be a frozenset containing generic")
        missing = set(range(1, total_options + 1)) - set(controller.horizons)
        if missing:
            raise ValueError(f"value snapshot lacks required option horizons: {sorted(missing)}")
        self.hierarchy, self.controller, self.how = hierarchy, controller, how
        self.total_options = total_options
        self.utility = utility
        self.how_terminal_weight = how_terminal_weight
        self.how_fallback_weight = how_fallback_weight
        self.molecular_feature_fn = molecular_feature_fn or (
            lambda node: molecule_features(canonical_state_key(node.graph))
        )
        self.enable_build_ring_system = enable_build_ring_system
        self.allowed_options = allowed_options
        body = {
            "schema_version": "option_controller_runtime_snapshot_v1",
            "reference_snapshot": reference_snapshot,
            "feature_snapshot": feature_snapshot,
            "utility_snapshot": utility_snapshot,
            "controller_snapshot": controller.snapshot,
            "how": asdict(how),
            "total_options": total_options,
            "hierarchy": {
                "generic_horizon": hierarchy.generic_horizon,
                "ring_options": list(hierarchy.ring_options),
                "lazy_applicability": hierarchy.lazy_applicability,
                "include_carbonyl_options": hierarchy.include_carbonyl_options,
                "include_region_replacement": hierarchy.include_region_replacement,
            },
            "enable_build_ring_system": enable_build_ring_system,
            "allowed_options": None if allowed_options is None else sorted(allowed_options),
        }
        self.snapshot_payload = body
        self.snapshot = identity(body)
        self._utility_cache: dict[tuple, float] = {}
        self._molecular_feature_cache: dict[tuple, np.ndarray] = {}

    def _utility(self, node: MolecularSearchState) -> float:
        key = node.key()
        if key not in self._utility_cache:
            value = float(self.utility(node))
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError("oriented boundary utility must be finite in [0, 1]")
            self._utility_cache[key] = value
        return self._utility_cache[key]

    def _molecular_features(self, node: MolecularSearchState) -> np.ndarray:
        key = node.key()
        if key not in self._molecular_feature_cache:
            values = np.asarray(self.molecular_feature_fn(node), dtype=np.float32)
            if values.ndim != 1 or not values.size or not np.isfinite(values).all():
                raise ValueError("molecular feature callback must return a finite vector")
            self._molecular_feature_cache[key] = values
        return self._molecular_feature_cache[key]

    def value_features(
        self, node: MolecularSearchState, *, incumbent: float, remaining_options: int
    ) -> np.ndarray:
        return boundary_value_features(
            self._molecular_features(node),
            current_utility=self._utility(node),
            incumbent_utility=incumbent,
            remaining_options=remaining_options,
        )

    def actor_features(
        self,
        node: MolecularSearchState,
        *,
        incumbent: float,
        remaining_options: int,
    ) -> np.ndarray:
        if node.stage != "what" or node.region is None:
            raise ValueError("actor features require a realized WHERE decision")
        region = node.region
        return option_state_features(
            self._molecular_features(node),
            current_utility=self._utility(node),
            incumbent_utility=incumbent,
            remaining_options=remaining_options,
            released_fraction=region.released_fraction,
            region_size=region.size,
            total_atoms=region.n_atoms_total,
            boundary_bonds=region.arity,
            context_components=region.n_context_components,
            region_has_ring=region.region_has_ring,
            ring_boundary_bonds=region.n_ring_boundary_bonds,
        )

    def start_population(
        self,
        states: Sequence[MolecularSearchState],
        *,
        incumbent: float,
        seed: int,
    ) -> PersistentOptionPopulation:
        nodes = tuple(states)
        if not nodes or any(node.stage != "where" for node in nodes):
            raise ValueError("runtime populations start at complete WHERE-boundary states")
        if not math.isfinite(incumbent) or not 0 <= incumbent <= 1:
            raise ValueError("oriented incumbent must be finite in [0, 1]")
        potentials = []
        for node in nodes:
            path_incumbent = max(float(incumbent), self._utility(node))
            potentials.append(
                self.controller.log_potential(
                    self.value_features(
                        node,
                        incumbent=path_incumbent,
                        remaining_options=self.total_options,
                    ),
                    remaining_options=self.total_options,
                    incumbent=path_incumbent,
                )
            )
        return PersistentOptionPopulation.start(
            [encode_search_state(node) for node in nodes],
            seed=seed,
            controller_snapshot=self.snapshot,
            log_potentials=potentials,
            control_context={
                "schema_version": "option_controller_runtime_context_v1",
                "runtime_snapshot": self.snapshot,
                "incumbent": float(incumbent),
            },
        )

    @staticmethod
    def _path_incumbent(particle: PersistentOptionParticle, initial_incumbent: float) -> float:
        """Recover the best utility already realized on this exact particle path."""

        best = float(initial_incumbent)
        for boundary in particle.history:
            value = boundary.get("after_utility")
            if value is None:
                continue
            value = float(value)
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError("persistent history contains an invalid boundary utility")
            best = max(best, value)
        return best

    def _what_row(self, node: MolecularSearchState):
        row = self.hierarchy.row(node)
        indices = [
            index
            for index, option in enumerate(row.labels)
            if (self.enable_build_ring_system or option != BUILD_RING_SYSTEM_OPTION)
            and (self.allowed_options is None or option in self.allowed_options)
        ]
        labels = tuple(row.labels[index] for index in indices)
        if GENERIC_OPTION not in labels:
            raise RuntimeError("filtered applicable option row lost generic")
        reference = np.asarray(row.reference, dtype=float)[indices]
        if not len(reference) or reference.sum() <= 0:
            raise RuntimeError("filtered applicable option row has zero mass")
        reference /= reference.sum()
        return labels, tuple(row.successors[index] for index in indices), reference

    @staticmethod
    def _dead_transition(particle: PersistentOptionParticle, reason: str) -> OptionTransition:
        return OptionTransition(
            particle.particle_id,
            None,
            0.0,
            0.0,
            None,
            False,
            {"status": reason},
        )

    def _transition(
        self,
        particle: PersistentOptionParticle,
        *,
        incumbent: float,
        remaining_options: int,
        boundary: int,
    ) -> OptionTransition:
        if not particle.alive:
            return self._dead_transition(particle, "already_dead")
        node = decode_search_state(particle.exact_state)
        if node.stage != "where":
            raise ValueError("persistent runtime state is not at an option boundary")
        path_incumbent = self._path_incumbent(particle, incumbent)
        rng_seed = _stream_seed(particle.rng_identity, boundary)
        rng = np.random.default_rng(rng_seed)
        where = self.hierarchy.row(node)
        if not where.successors:
            return self._dead_transition(particle, "no_applicable_region")
        region_probabilities = np.asarray(where.reference, dtype=float)
        region_index = int(rng.choice(len(where.successors), p=region_probabilities))
        what_node = where.successors[region_index]
        options, how_nodes, option_reference = self._what_row(what_node)
        selected = self.controller.decide(
            options,
            option_reference,
            self.actor_features(
                what_node,
                incumbent=path_incumbent,
                remaining_options=remaining_options,
            ),
            rng,
        )
        how_node = how_nodes[selected.selected_index]
        result = sample_option_trajectory(
            how_node.active,
            self.hierarchy.kernel,
            self.how_terminal_weight,
            self.how_fallback_weight,
            snapshot_id=self.how.snapshot_id,
            seed=int(rng.integers(0, 2**63)),
            max_expansions=self.how.max_expansions,
            max_terminal_evaluations=self.how.max_terminal_evaluations,
            estimator=self.how.estimator,
            samples_per_successor=self.how.samples_per_successor,
            confidence_alpha=self.how.confidence_alpha,
            max_rollouts=self.how.max_rollouts,
        )
        if result["status"] == "executor_budget_exhausted":
            from compose_v4.control.continuation import ContinuationBudgetExceeded

            raise ContinuationBudgetExceeded(
                "HOW executor budget exhausted before option completion"
            )
        region_log_probability = math.log(float(region_probabilities[region_index]))
        option_log_reference = math.log(selected.reference_probability)
        option_log_proposal = math.log(selected.proposal_probability)
        log_reference = (
            region_log_probability
            + option_log_reference
            + result["conditional_reference_path_logp"]
        )
        log_proposal = (
            region_log_probability + option_log_proposal + result["conditional_path_logq"]
        )
        deterministic_how = {
            key: value for key, value in result.items() if key not in ("seconds", "kernel_work")
        }
        operational = {
            "how_seconds": result["seconds"],
            "kernel_work": result["kernel_work"],
        }
        audit = {
            "status": result["status"],
            "runtime_snapshot": self.snapshot,
            "rng_seed": rng_seed,
            "root_id": node.root_id,
            "region": {
                "key": repr(what_node.region.key()),
                "generator": what_node.region.generator,
                "interface": what_node.region.interface,
                "released_fraction": what_node.region.released_fraction,
                "size": what_node.region.size,
                "reference_probability": float(region_probabilities[region_index]),
            },
            "option": selected.selected,
            "applicable_options": list(options),
            "option_reference_probability": selected.reference_probability,
            "option_proposal_probability": selected.proposal_probability,
            "option_kl": selected.distribution.kl,
            "how": deterministic_how,
            "before_utility": self._utility(node),
            "path_incumbent_before": path_incumbent,
        }
        if result["status"] != "complete":
            return OptionTransition(
                particle.particle_id,
                None,
                log_reference,
                log_proposal,
                None,
                False,
                audit,
                operational,
            )
        final = decode_option(result["final_option_state"])
        primitive_count = len(result["trace"])
        if final.remaining or primitive_count < 1 or primitive_count > node.budget:
            raise RuntimeError("completed HOW result violates its primitive budget")
        following = MolecularSearchState(
            final.graph,
            final.lineage,
            node.budget - primitive_count,
            node.root_id,
        )
        next_remaining = remaining_options - 1
        utility = self._utility(following)
        next_incumbent = max(path_incumbent, utility)
        features = self.value_features(
            following, incumbent=next_incumbent, remaining_options=next_remaining
        )
        next_log_potential = self.controller.log_potential(
            features,
            remaining_options=next_remaining,
            incumbent=next_incumbent,
            terminal_best=next_incumbent if next_remaining == 0 else None,
        )
        audit.update(
            {
                "after_utility": utility,
                "path_incumbent_after": next_incumbent,
                "primitive_count": primitive_count,
                "structural_change": structural_displacement(
                    node.graph, final.graph, node.lineage, final.lineage
                ),
                "topology": topology(final.graph),
            }
        )
        return OptionTransition(
            particle.particle_id,
            encode_search_state(following),
            log_reference,
            log_proposal,
            next_log_potential,
            True,
            audit,
            operational,
        )

    def advance(
        self,
        population: PersistentOptionPopulation,
        *,
        incumbent: float,
        resample: bool = True,
    ) -> tuple[PersistentOptionPopulation, dict]:
        if population.controller_snapshot != self.snapshot:
            raise ValueError("persistent population belongs to another runtime snapshot")
        context = population.control_context
        if (
            not isinstance(context, dict)
            or context.get("schema_version") != "option_controller_runtime_context_v1"
            or context.get("runtime_snapshot") != self.snapshot
            or context.get("incumbent") != float(incumbent)
        ):
            raise ValueError("persistent population has a changed or unbound control context")
        if population.option_boundary >= self.total_options:
            raise ValueError("option population has exhausted its declared horizon")
        if not math.isfinite(incumbent) or not 0 <= incumbent <= 1:
            raise ValueError("oriented incumbent must be finite in [0, 1]")
        remaining = self.total_options - population.option_boundary
        transitions = tuple(
            self._transition(
                particle,
                incumbent=incumbent,
                remaining_options=remaining,
                boundary=population.option_boundary,
            )
            for particle in population.particles
        )
        following, receipt = advance_population(population, transitions, resample=resample)
        return following, {
            "schema_version": "option_controller_runtime_boundary_v1",
            "runtime_snapshot": self.snapshot,
            "boundary": following.option_boundary,
            "remaining_options": self.total_options - following.option_boundary,
            "particle_update": receipt,
            "transitions": [
                {
                    **transition.audit,
                    **(
                        {"operational": transition.operational}
                        if transition.operational is not None
                        else {}
                    ),
                }
                for transition in transitions
            ],
        }

    def locked_candidates(self, population: PersistentOptionPopulation) -> tuple[dict, ...]:
        """Canonical-deduplicate complete boundary molecules before acquisition."""

        if population.controller_snapshot != self.snapshot:
            raise ValueError("candidate population belongs to another runtime snapshot")
        by_smiles: dict[str, tuple[PersistentOptionParticle, MolecularSearchState]] = {}
        for particle in population.particles:
            if not particle.alive:
                continue
            node = decode_search_state(particle.exact_state)
            if node.stage != "where":
                raise ValueError("only complete option-boundary states can be locked")
            smiles = canonical_state_key(node.graph)
            previous = by_smiles.get(smiles)
            if previous is None or (
                particle.log_weight,
                particle.particle_id,
            ) > (previous[0].log_weight, previous[0].particle_id):
                by_smiles[smiles] = (particle, node)
        rows = []
        for smiles in sorted(by_smiles):
            particle, node = by_smiles[smiles]
            candidate_id = identity(
                {
                    "runtime_snapshot": self.snapshot,
                    "boundary": population.option_boundary,
                    "exact_state_id": particle.exact_state_id,
                    "canonical_smiles": smiles,
                }
            )
            rows.append(
                {
                    "candidate_id": candidate_id,
                    "canonical_smiles": smiles,
                    "locked": True,
                    "complete": True,
                    "exact_state": particle.exact_state,
                    "exact_state_id": particle.exact_state_id,
                    "particle_id": particle.particle_id,
                    "log_weight": particle.log_weight,
                    "history": list(particle.history),
                    "option_boundary": population.option_boundary,
                    "primitive_budget_remaining": node.budget,
                    "control_context": population.control_context,
                }
            )
        return tuple(rows)

    def acquire(
        self,
        population: PersistentOptionPopulation,
        posterior_samples: np.ndarray,
        *,
        batch_size: int,
    ) -> dict:
        candidates = self.locked_candidates(population)
        acquisition = select_qpo_batch(candidates, posterior_samples, batch_size=batch_size)
        return {**acquisition, "runtime_snapshot": self.snapshot, "candidates": candidates}
