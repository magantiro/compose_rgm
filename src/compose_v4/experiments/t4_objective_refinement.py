"""Terminal local repair of T4 endpoints that the fiber gate refused.

WHY THIS EXISTS
---------------
A T4 cell reports ``candidate_exhaustion`` when a round produces zero eligible
endpoints.  That label conflates two different failures, and the support-stage
audit separated them:

  * a SUPPORT failure -- the proposal distribution never reaches the region
    where an eligible molecule lives;
  * a LOCAL failure -- the proposal reaches the region and the endpoint misses a
    bound by an amount one rewrite covers.

More proposal volume is the right answer to the first and the wrong answer to
the second.  Measured on ``fa7_0`` at delta 0.6, the best of 1,535
similarity-passing endpoints carries a QED margin of ``-0.001663`` while holding
a similarity margin of ``+0.072`` and an SA margin of ``+1.720``.  Drawing more
programs from the same distribution does not close 0.0017 of QED; one rewrite
does -- and none of the four eligible successors it finds appear among the 9,527
distinct endpoints that cell produced.

WHAT THIS DOES
--------------
Given an endpoint the production :class:`~compose_v4.experiments.t4_fiber_campaign.Fiber`
refused, enumerate the EXACT legal fiber at production slot semantics and return
the successors the same gate accepts.  Every quantity used here -- QED,
similarity, SA -- is free to evaluate; this module never asks the oracle.

Which endpoints to refine is decided by their own violation scalar ``v``, never
by which bound they fail and never by a target, cell or seed identity --
:func:`select_refinement_panel`.  Two distinct repairs show up under that one
selector: objective POLISH (fa7_0, trim a peripheral aliphatic carbon) and
artifact REPAIR (braf_1, open a spurious ring the proposal lane fused across a
benzene).  A refinement that fires on one cell and no other is a finding about
that cell, not a policy, so the caller reports the per-cell yield rather than
assuming it.

INVARIANTS
----------
* the source graph is PADDED to ``PRODUCTION_MAX_ATOMS`` before enumeration.
  A graph rebuilt from SMILES is TIGHT and carries no free slot, which silently
  deletes the whole ``atom_insert`` family -- measured here at 305 of 733 marks
  (42%) on the ``fa7_0`` leader.  Refining on a tight graph is an artifact.
* eligibility is decided by the production ``Fiber``, imported rather than
  restated, so this module cannot drift from the campaign's own gate.
* ``oracle_calls`` is 0 by construction and is reported so a caller can assert
  it.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from rdkit import Chem, RDLogger
from rdkit.Chem import QED
from rdkit.Contrib.SA_Score import sascorer

from compose_v4.chem.molecular_graph import (
    MolecularGraphError,
    molecular_graph_to_smiles,
    smiles_to_molecular_graph,
)
from compose_v4.chem.state import pad_molecular_graph
from compose_v4.control.generic_legal_action_policy import enumerate_legal_successors
from compose_v4.experiments.editing_v2_evaluation_semantics import PRODUCTION_MAX_ATOMS
from compose_v4.experiments.t4_endpoint_selection import calculate_properties
from compose_v4.experiments.t4_fiber_campaign import (
    COMPOSE_VALID,
    QED_MIN,
    REPRESENTABLE_HEAVY_ATOMS,
    SA_MAX,
    Fiber,
)

RDLogger.DisableLog("rdApp.*")

SCHEMA_VERSION = "t4_objective_refinement_v1"

#: Panels are selected by the endpoint's own violation scalar ``v``, NOT by
#: which bound it fails.  This is a measurement, and it corrected an earlier
#: reading of mine taken from fa7_0 alone.
#:
#: On fa7_0 the two selectors look identical: the 5 panel endpoints failing ONLY
#: QED are exactly the 5 that lift, and the other 20 lift nothing.  Generalising
#: from that, a first version refined only QED near-misses.  braf_1 refutes it --
#: there the violation-ranked panel lifts 5 endpoints, while the QED-only panel
#: given the top 20 of a 1,404-endpoint trigger set lifts 1.  Four of braf_1's
#: five rescues were SIMILARITY failures repaired jointly with QED by a single
#: ``cycle_open`` (e.g. QED 0.6481 -> 0.7574 AND similarity 0.5658 -> 0.6857).
#:
#: Both mechanisms are general and they are different.  fa7_0 is objective
#: POLISH: trim a peripheral aliphatic carbon, moving MW, ALOGP and ROTB
#: together.  braf_1 is artifact REPAIR: the proposal lane fused a spurious ring
#: across a benzene (``c1c2cc(...)cc1-2``) and opening it restores similarity and
#: drug-likeness at once.  A bound-specific trigger sees only the first.
VIOLATION_RANKED_PANEL = "least_violation_v1"


class SourceNotRepresentable(ValueError):
    """The endpoint parses as a molecule but is outside the COMPOSE vocabulary.

    T4 proposal lanes emit species RDKit will parse and COMPOSE will not build a
    graph for -- radicals above all, which is why ``Fiber`` carries a structural
    validity gate at all.  Such an endpoint has no legal fiber, so it cannot be
    refined; callers batching over a panel must be able to skip it rather than
    die inside the chemistry layer.
    """


# ---- Property helpers -------------------------------------------------------


def properties_for_fiber(mol, fiber: Fiber) -> dict | None:
    """The production T4 property vector for `mol`, or None if it is not scorable.

    Takes an already-built ``Fiber`` so a caller scoring a whole shard pays the
    seed fingerprint once rather than per molecule.
    """

    if mol is None:
        return None
    return calculate_properties(
        mol,
        seed_fp=fiber.seed,
        generator=fiber.generator,
        sa_scorer=sascorer.calculateScore,
        delta=fiber.delta,
        qed_min=QED_MIN,
        sa_max=SA_MAX,
    )


def endpoint_properties(seed_smiles: str, delta: float, smiles: str) -> dict | None:
    """Free property vector (qed, sa, sim, v) of `smiles` against `seed_smiles`."""

    return properties_for_fiber(Chem.MolFromSmiles(smiles), Fiber(seed_smiles, delta))


def free_objective_near_miss(properties: dict, delta: float) -> bool:
    """True when the endpoint passes similarity and SA and fails only QED.

    A DIAGNOSTIC LABEL, not the selector.  It says which bound was missing, and
    on fa7_0 it happens to be perfectly predictive (5 fired, 5 lifted; 20 did
    not fire, none lifted).  It is not predictive in general -- see
    ``VIOLATION_RANKED_PANEL`` for the braf_1 measurement that refutes using it
    to choose what to refine.
    """

    if properties is None:
        return False
    if properties["sim"] < delta or properties["sa"] > SA_MAX:
        return False
    return properties["qed"] < QED_MIN


def select_refinement_panel(candidates, size: int) -> list[dict]:
    """The `size` least-violating candidates, whatever bound they fail.

    ``candidates`` are mappings carrying at least ``smiles`` and the violation
    scalar ``v`` from :func:`properties_for_fiber`.  Already-eligible candidates
    (``v <= 0``) are dropped: there is nothing to repair.  Ties break on SMILES
    so the panel is deterministic, and selection reads no target, cell or seed
    identity -- the same rule ``t4_repair_neighbors.select_panel`` applies.
    """

    if size < 0:
        raise ValueError(f"panel size must be non-negative, got {size}")
    violating = [row for row in candidates if row["v"] > 0]
    violating.sort(key=lambda row: (row["v"], row["smiles"]))
    return violating[:size]


# ---- Refinement -------------------------------------------------------------


@dataclass(frozen=True)
class Refinement:
    """One legal single-rewrite successor and what it did to the free objective."""

    smiles: str
    rule: str
    qed: float
    similarity: float
    sa: float
    heavy: int
    alerts: int
    eligible: bool
    qed_gain: float
    similarity_delta: float
    sa_delta: float


def _legal_successor_smiles(smiles: str, max_atoms: int) -> list[tuple[str, str]]:
    """(rule, SMILES) for every distinct legal successor of the PADDED graph."""

    try:
        graph = pad_molecular_graph(smiles_to_molecular_graph(smiles), int(max_atoms))
    except MolecularGraphError as error:
        raise SourceNotRepresentable(
            f"endpoint is outside the COMPOSE graph vocabulary: {smiles!r}"
        ) from error
    out: list[tuple[str, str]] = []
    seen: set[str] = set()
    for successor in enumerate_legal_successors(graph):
        try:
            product = molecular_graph_to_smiles(successor.successor)
        except Exception:  # noqa: BLE001,S112 - an unrenderable product is not a candidate
            continue
        if not product or product in seen:
            continue
        seen.add(product)
        out.append((successor.rule, product))
    return out


def enumerate_refinements(
    seed_smiles: str,
    delta: float,
    smiles: str,
    *,
    support: str = COMPOSE_VALID,
    max_atoms: int = PRODUCTION_MAX_ATOMS,
) -> tuple[Refinement, ...]:
    """Every legal single rewrite of `smiles`, scored on the free objective.

    The returned tuple is ordered by descending QED and is NOT filtered to the
    eligible ones -- the ``eligible`` flag carries the production gate's verdict
    so a caller can measure the whole local trade surface, not only its winners.
    """

    fiber = Fiber(seed_smiles, delta, support=support)
    base = properties_for_fiber(Chem.MolFromSmiles(smiles), fiber)
    if base is None:
        raise ValueError(f"refinement source is not a scorable molecule: {smiles!r}")

    rows: list[Refinement] = []
    for rule, product in _legal_successor_smiles(smiles, max_atoms):
        mol = Chem.MolFromSmiles(product)
        if mol is None or "." in product or mol.GetNumHeavyAtoms() > REPRESENTABLE_HEAVY_ATOMS:
            continue
        props = properties_for_fiber(mol, fiber)
        if props is None:
            continue
        rows.append(
            Refinement(
                smiles=Chem.MolToSmiles(mol),
                rule=rule,
                qed=props["qed"],
                similarity=props["sim"],
                sa=props["sa"],
                heavy=mol.GetNumHeavyAtoms(),
                alerts=int(QED.properties(mol).ALERTS),
                eligible=fiber.check(product) is not None,
                qed_gain=props["qed"] - base["qed"],
                similarity_delta=props["sim"] - base["sim"],
                sa_delta=props["sa"] - base["sa"],
            )
        )
    rows.sort(key=lambda row: (-row.qed, row.smiles))
    return tuple(rows)


def refine_endpoint(
    seed_smiles: str,
    delta: float,
    smiles: str,
    *,
    support: str = COMPOSE_VALID,
    max_atoms: int = PRODUCTION_MAX_ATOMS,
) -> dict:
    """Refine one near-miss endpoint and report the evidence.

    Returns the full local trade surface plus the eligible subset, so a caller
    can hand the eligible candidates back to the existing selection loop.
    """

    fiber = Fiber(seed_smiles, delta, support=support)
    base = properties_for_fiber(Chem.MolFromSmiles(smiles), fiber)
    report = {
        "schema_version": SCHEMA_VERSION,
        "seed_smiles": seed_smiles,
        "delta": delta,
        "source_smiles": smiles,
        "source_properties": base,
        "source_eligible": fiber.check(smiles) is not None,
        "triggered": free_objective_near_miss(base, delta),
        "source_representable": True,
        "successors_scored": 0,
        "successors_holding_sim_and_sa": 0,
        "eligible_count": 0,
        "eligible": [],
        "best_qed": None,
        "oracle_calls": 0,
    }
    try:
        rows = enumerate_refinements(
            seed_smiles, delta, smiles, support=support, max_atoms=max_atoms
        )
    except SourceNotRepresentable:
        report["source_representable"] = False
        return report

    eligible = [row for row in rows if row.eligible]
    holding = [row for row in rows if row.similarity >= delta and row.sa <= SA_MAX]
    report.update(
        {
            "successors_scored": len(rows),
            "successors_holding_sim_and_sa": len(holding),
            "eligible_count": len(eligible),
            "eligible": [asdict(row) for row in eligible],
            "best_qed": rows[0].qed if rows else None,
        }
    )
    return report


__all__ = [
    "PRODUCTION_MAX_ATOMS",
    "SCHEMA_VERSION",
    "VIOLATION_RANKED_PANEL",
    "Refinement",
    "SourceNotRepresentable",
    "endpoint_properties",
    "enumerate_refinements",
    "free_objective_near_miss",
    "properties_for_fiber",
    "refine_endpoint",
    "select_refinement_panel",
]
