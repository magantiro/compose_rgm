"""Preflight that every COMPOSE evaluation must pass before it does any work.

WHY THIS EXISTS
---------------
A molecular graph rebuilt from SMILES carries exactly ``n_real_atoms`` slots.
Atom insertion is a BIRTH operation that needs a free slot, so an unpadded
source silently drops the entire ``atom_insert`` family -- measured at 35-100%
of the legal support depending on molecule size (1267 -> 1930 marks at 36 heavy
atoms; 179 -> 358 at 15) -- and only recovers it once a deletion happens to free
a slot mid-trajectory.

Nothing about that failure is loud.  The rollout still runs, the NLL still
computes, the gate still "passes."  It invalidated three completed experiments
before it was noticed, and the tell was subtle: ``atom_insert`` appeared at 10%
of sampled moves in ROLLOUTS but 0% in single-state enumeration.  For a project
whose distinctive claim includes native atom birth and death, evaluating on
states that cannot express birth is the most dangerous silent failure available.

WHAT IS CHECKED
---------------
The probe's realized legal-family census is compared against the census the
production compiler recorded for the SAME source state.  That is strictly
stronger than asserting ``n_slots == max_atoms``: a state can carry the right
slot count and still be running the wrong rewrite system, catalog, or family
gating, and this catches all of those.

Corpus states carry ``n_slots``; prefer decoding the recorded payload over
re-parsing SMILES.  Where an experiment must start from a bare SMILES string,
build it with :func:`production_state_from_smiles`.
"""

from __future__ import annotations

import collections
from typing import Any, Iterable, Mapping

#: Slot capacity of the production editing model; corpus states are padded here.
PRODUCTION_MAX_ATOMS = 40


class EvaluationSemanticsError(RuntimeError):
    """The evaluation state representation disagrees with the production kernel."""


def production_state_from_smiles(smiles: str, max_atoms: int = PRODUCTION_MAX_ATOMS):
    """Build an evaluation state with production slot semantics.

    Never call ``smiles_to_molecular_graph`` directly in an experiment: it
    returns a TIGHT graph with no free slot, which removes the whole
    ``atom_insert`` family from the enumerated support.
    """

    from compose_v4.chem.molecular_graph import smiles_to_molecular_graph
    from compose_v4.chem.state import pad_molecular_graph

    return pad_molecular_graph(smiles_to_molecular_graph(smiles), int(max_atoms))


def realized_family_census(model, state, time: float = 0.5) -> dict[str, int]:
    """The legal-family census this evaluation path actually produces."""

    import torch

    from compose_v4.experiments.production_successor_kernel import (
        enumerate_factorized_marked_law,
    )

    with torch.no_grad():
        law = enumerate_factorized_marked_law(model, state, float(time))
    return dict(collections.Counter(mark.family_name for mark in law.marks))


def assert_production_state_semantics(
    model,
    samples: Iterable[tuple[Any, Mapping[str, int]]],
    *,
    time: float = 0.5,
) -> dict[str, Any]:
    """Raise unless every sampled state reproduces the compiler-recorded census.

    ``samples`` yields ``(state, recorded_marks_by_family)`` pairs, where the
    recorded census comes from the corpus entry for that same source state.
    Returns evidence on success; carries no authority.
    """

    checked = 0
    for state, recorded in samples:
        realized = realized_family_census(model, state, time)
        expected = {str(f): int(c) for f, c in dict(recorded).items() if int(c) > 0}
        observed = {str(f): int(c) for f, c in realized.items() if int(c) > 0}
        if observed != expected:
            disagreeing = {f: c for f, c in expected.items() if observed.get(f, 0) != c}
            unexpected = {f: c for f, c in observed.items() if f not in expected}
            raise EvaluationSemanticsError(
                "evaluation legal-family census disagrees with the production compiler "
                f"for this source.\n  disagreeing: {disagreeing}\n  unexpected: {unexpected}\n"
                "  the usual cause is a TIGHT graph rebuilt from SMILES (no free slot, so "
                "atom_insert is absent); build it with production_state_from_smiles, or "
                "decode the corpus exact_state payload instead of re-parsing SMILES"
            )
        checked += 1
    if not checked:
        raise EvaluationSemanticsError(
            "preflight ran on zero samples; an empty check proves nothing"
        )
    return {
        "schema": "compose.editing_v2.evaluation_semantics_preflight",
        "schema_version": 1,
        "status": "PREFLIGHT_EVIDENCE_ONLY_NO_AUTHORITY",
        "states_checked": checked,
    }


__all__ = [
    "PRODUCTION_MAX_ATOMS",
    "EvaluationSemanticsError",
    "assert_production_state_semantics",
    "production_state_from_smiles",
    "realized_family_census",
]
