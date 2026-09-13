"""Context-conditioned complete-program retrieval, the non-neural proposal arm.

Only observed program structure enters retrieval. Scores, winner endpoints and
target fingerprints are deliberately absent. This component changes proposal
geometry, not R_theta, Q(M), or the existing option KL controller. The caller
retains its broad reference channel through an explicit mixture decision.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from compose_v4.control.edit_program import EditProgram, attachment_bindings


@dataclass(frozen=True)
class ProgramEntry:
    program: EditProgram
    source_groups: tuple[str, ...]

    def __post_init__(self):
        if not self.source_groups or tuple(sorted(set(self.source_groups))) != self.source_groups:
            raise ValueError("program source groups must be nonempty, unique and sorted")


def source_balanced_prior(entries: tuple[ProgramEntry, ...]) -> np.ndarray:
    if not entries or len({entry.program.program_id for entry in entries}) != len(entries):
        raise ValueError("program bank must be nonempty and program-unique")
    sources = sorted({source for entry in entries for source in entry.source_groups})
    counts = {source: sum(source in entry.source_groups for entry in entries) for source in sources}
    weights = np.asarray([sum(1 / counts[s] for s in entry.source_groups) for entry in entries])
    return weights / weights.sum()


def choose_channel(rng, *, program_probability: float = 0.9) -> str:
    """The mandatory reference branch is delegated to the unchanged caller."""
    if not math.isfinite(program_probability) or not 0 < program_probability < 1:
        raise ValueError("both program and broad-reference channels require positive probability")
    return "program" if rng.random() < program_probability else "reference"


def propose_programs(
    graph,
    entries: tuple[ProgramEntry, ...],
    *,
    seed: int,
    count: int = 32,
    contextual: bool = True,
    temperature: float = 1.0,
    max_bindings: int = 32,
    max_visits: int = 4096,
    mutable_slots: frozenset[int] | None = None,
    ranked_count: int = 0,
) -> dict:
    """Sample fixed complete programs and sites, before executing/ranking endpoints.

    Uses a common capped binding pool for paired context/unguided arms. Source
    balancing prevents many decompositions of one source dominating the prior.
    Probabilities refer to this finite proposal pool, not the full reference law.
    """
    if (
        type(count) is not int
        or count < 1
        or type(ranked_count) is not int
        or ranked_count < 0
        or not math.isfinite(temperature)
        or temperature <= 0
    ):
        raise ValueError("proposal count/temperature must be positive")
    prior = source_balanced_prior(entries)
    slots, base, distances, censuses = [], [], [], []
    for index, entry in enumerate(entries):
        census = attachment_bindings(
            entry.program,
            graph,
            max_bindings=max_bindings,
            max_visits=max_visits,
            # The shared support is fixed independently of the comparison arm.
            contextual=True,
            mutable_slots=mutable_slots,
        )
        censuses.append(
            {
                "program_id": entry.program.program_id,
                "visits": census.visits,
                "bindings": len(census.assignments),
                "truncated": census.truncated,
            }
        )
        for binding, distance in zip(census.assignments, census.context_distances, strict=True):
            slots.append((index, binding))
            base.append(prior[index] / len(census.assignments))
            distances.append(distance)
    if not slots:
        return {"status": "no_bindings", "draws": [], "censuses": censuses}
    weights = np.asarray(base, dtype=float)
    weights /= weights.sum()
    log_weight = np.log(weights) - (np.asarray(distances) / temperature if contextual else 0)
    proposal = np.exp(log_weight - log_weight.max())
    proposal /= proposal.sum()
    # Preserve finite-pool exploration independently of the caller's broad mixture.
    proposal = 0.1 * weights + 0.9 * proposal
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(slots), size=count, replace=True, p=proposal)
    draws = [
        {
            "program_index": slots[int(i)][0],
            "assignment": list(slots[int(i)][1]),
            "proposal_probability": float(proposal[i]),
            "base_probability": float(weights[i]),
            "context_distance": distances[i],
            "pool_index": int(i),
        }
        for i in indices
    ]
    result = {
        "status": "proposed",
        "draws": draws,
        "censuses": censuses,
        "pool_size": len(slots),
        "contextual": contextual,
        "probability_domain": "capped program-binding pool; not R_theta",
    }
    if ranked_count:
        # Prefer high-probability contextual matches, but first expose at most
        # one binding per program. This prevents a simple program with many
        # symmetric bindings from monopolizing a bounded direct-retrieval slice.
        ordered = sorted(
            range(len(slots)),
            key=lambda i: (
                -float(proposal[i]),
                distances[i],
                entries[slots[i][0]].program.program_id,
                slots[i][1],
            ),
        )
        priority, used = [], set()
        for unique_programs in (True, False):
            for raw in ordered:
                program_index, binding = slots[raw]
                program_id = entries[program_index].program.program_id
                if unique_programs and program_id in used:
                    continue
                record = {
                    "program_index": program_index,
                    "assignment": list(binding),
                    "proposal_probability": float(proposal[raw]),
                    "base_probability": float(weights[raw]),
                    "context_distance": distances[raw],
                    "pool_index": raw,
                    "priority": len(priority),
                    "priority_policy": "context_probability_then_unique_program_v1",
                }
                priority.append(record)
                used.add(program_id)
                if len(priority) == min(ranked_count, len(slots)):
                    result["ranked_draws"] = priority
                    return result
    return result
