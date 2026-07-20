"""Inference-only CTMC rate calibration by exact thinning.

The wrapper in this module never renormalizes the base model's marked event
distribution and never enables a new action.  A sampled legal mark is accepted
with probability ``exp(log_rate_adjustment)``.  A rejected mark remains a
virtual CTMC jump: operational time advances, but the molecular state does not
change.  This is the standard thinning construction for multiplying selected
transition intensities while preserving the base clock and every unmodified
rate.

Only non-positive adjustments are supported.  That restriction makes the
calibration exact without adding a dominating proposal process.  It is enough
for the current diagnosed failures: excessive atom deletion and excessive
three/four-member ring-system growth.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite, log
from typing import Protocol

import networkx as nx
import numpy as np

from compose_v4.chem.molecular_graph import MolecularGraph, is_element
from compose_v4.model.factorized_tracelet_rate_model import SampledRewriteMark
from compose_v4.rewrite.kernel import de_novo_rewrite_system
from compose_v4.rewrite.tracelets import RingSystemGrow


class RewriteMarkSampler(Protocol):
    def sample_rewrite_mark(
        self,
        state: MolecularGraph,
        time: float,
        rng: np.random.Generator,
    ) -> SampledRewriteMark:
        ...


def ring_system_grow_cycle_sizes(
    state: MolecularGraph,
    action: RingSystemGrow,
) -> tuple[int, ...]:
    """Return minimum-cycle-basis sizes installed by a ring-system action."""

    successor = de_novo_rewrite_system().apply(
        state,
        "ring_system_grow",
        action,
    )
    members = tuple(
        int(vertex)
        for vertex in action.system_atoms
        if bool(is_element(successor.atom_types[int(vertex)]))
    )
    graph = nx.Graph()
    graph.add_nodes_from(members)
    graph.add_edges_from(
        (left, right)
        for offset, left in enumerate(members)
        for right in members[offset + 1 :]
        if int(successor.bonds[left, right]) != 0
    )
    return tuple(sorted(len(cycle) for cycle in nx.minimum_cycle_basis(graph)))


def ring_system_grow_contains_small_ring(
    state: MolecularGraph,
    action: RingSystemGrow,
    *,
    maximum_size: int = 4,
) -> bool:
    """Return whether a proposed whole-ring rewrite contains a small cycle."""

    if maximum_size < 3:
        raise ValueError("maximum small-ring size must be at least three")
    cycle_sizes = ring_system_grow_cycle_sizes(state, action)
    return bool(cycle_sizes) and min(cycle_sizes) <= int(maximum_size)


TRIPLE_BOND_INSTALLING_RULES = frozenset({"atom_insert", "bond_reorder"})


def mark_installs_triple_bond(
    state: MolecularGraph,
    rule_name: str,
    action: object,
) -> bool:
    """Return whether a sampled mark increases the count of order-3 (triple) bonds.

    Only connected atom insertion and bond reorder can introduce a new triple
    bond, and every triple is created exactly once by such a mark, so a positive
    count delta after applying the action attributes the installation
    unambiguously.  Graft/reroute relocates existing bonds and preserves the
    total triple count, so it is intentionally excluded.  Triples never occur
    inside committed ring systems in the matched corpus, so whole-ring actions
    are excluded as well.
    """

    if rule_name not in TRIPLE_BOND_INSTALLING_RULES:
        return False
    before = int((np.triu(np.asarray(state.bonds)) == 3).sum())
    successor = de_novo_rewrite_system().apply(state, str(rule_name), action)
    after = int((np.triu(np.asarray(successor.bonds)) == 3).sum())
    return after > before


@dataclass(frozen=True)
class ThinnedRateCalibrationSampler:
    """Multiply selected base-model rates without renormalizing other marks.

    ``family_log_rate_adjustments`` contains ``(rule_name, log_multiplier)``
    pairs.  ``small_ring_log_rate_adjustment`` is added only to
    ``ring_system_grow`` actions containing a cycle no larger than
    ``small_ring_maximum_size``.  Every adjustment must be finite and no larger
    than zero.
    """

    base_sampler: RewriteMarkSampler
    family_log_rate_adjustments: tuple[tuple[str, float], ...] = ()
    small_ring_log_rate_adjustment: float = 0.0
    small_ring_maximum_size: int = 4
    triple_bond_log_rate_adjustment: float = 0.0
    _family_adjustments: dict[str, float] = field(
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        adjustments: dict[str, float] = {}
        for raw_name, raw_value in self.family_log_rate_adjustments:
            name = str(raw_name)
            value = float(raw_value)
            if not name:
                raise ValueError("calibrated rewrite-family name cannot be empty")
            if name in adjustments:
                raise ValueError(f"duplicate calibrated rewrite family: {name}")
            if not isfinite(value) or value > 0.0:
                raise ValueError("family log-rate adjustments must be finite and non-positive")
            adjustments[name] = value
        small_ring_adjustment = float(self.small_ring_log_rate_adjustment)
        if not isfinite(small_ring_adjustment) or small_ring_adjustment > 0.0:
            raise ValueError(
                "small-ring log-rate adjustment must be finite and non-positive"
            )
        if int(self.small_ring_maximum_size) < 3:
            raise ValueError("maximum small-ring size must be at least three")
        triple_bond_adjustment = float(self.triple_bond_log_rate_adjustment)
        if not isfinite(triple_bond_adjustment) or triple_bond_adjustment > 0.0:
            raise ValueError(
                "triple-bond log-rate adjustment must be finite and non-positive"
            )
        object.__setattr__(self, "_family_adjustments", adjustments)

    @property
    def calibration_signature(self) -> dict[str, object]:
        return {
            "method": "exact_ctmc_thinning_v1",
            "family_log_rate_adjustments": dict(sorted(self._family_adjustments.items())),
            "small_ring_log_rate_adjustment": float(
                self.small_ring_log_rate_adjustment
            ),
            "small_ring_maximum_size": int(self.small_ring_maximum_size),
            "triple_bond_log_rate_adjustment": float(
                self.triple_bond_log_rate_adjustment
            ),
        }

    def sample_rewrite_mark(
        self,
        state: MolecularGraph,
        time: float,
        rng: np.random.Generator,
    ) -> SampledRewriteMark:
        sampled = self.base_sampler.sample_rewrite_mark(state, time, rng)
        if sampled.action is None or float(sampled.total_hazard) <= 1e-12:
            return sampled

        adjustment = float(self._family_adjustments.get(sampled.rule_name, 0.0))
        labels = []
        if adjustment < 0.0:
            labels.append(sampled.rule_name)
        calibrates_ring = (
            self.small_ring_log_rate_adjustment < 0.0
            and sampled.rule_name == "ring_system_grow"
            and isinstance(sampled.action, RingSystemGrow)
        )
        if calibrates_ring:
            if ring_system_grow_contains_small_ring(
                state,
                sampled.action,
                maximum_size=int(self.small_ring_maximum_size),
            ):
                ring_adjustment = float(self.small_ring_log_rate_adjustment)
                adjustment += ring_adjustment
                if ring_adjustment < 0.0:
                    labels.append("small_ring")

        if self.triple_bond_log_rate_adjustment < 0.0 and mark_installs_triple_bond(
            state,
            sampled.rule_name,
            sampled.action,
        ):
            adjustment += float(self.triple_bond_log_rate_adjustment)
            labels.append("triple_bond")

        if adjustment >= 0.0:
            return sampled
        uniform = max(float(rng.random()), float(np.nextafter(0.0, 1.0)))
        if log(uniform) < adjustment:
            return sampled
        label = ":".join(labels) or sampled.rule_name
        return SampledRewriteMark(
            float(sampled.total_hazard),
            f"<VIRTUAL_RATE_CALIBRATION:{label}>",
            None,
        )


__all__ = [
    "RewriteMarkSampler",
    "ThinnedRateCalibrationSampler",
    "mark_installs_triple_bond",
    "ring_system_grow_contains_small_ring",
    "ring_system_grow_cycle_sizes",
]
