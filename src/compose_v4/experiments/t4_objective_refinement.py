"""Terminal free-objective refinement of T4 endpoints that miss only a free bound.

WHY THIS EXISTS
---------------
A T4 cell reports ``candidate_exhaustion`` when a round produces zero eligible
endpoints.  That label conflates two different failures, and the support-stage
audit separated them:

  * a SUPPORT failure -- the proposal distribution never reaches the region
    where an eligible molecule lives;
  * an OBJECTIVE failure -- the proposal reaches the region, the similarity and
    SA constraints are satisfied with margin, and the endpoint misses the QED
    bound by a hair.

More proposal volume is the right answer to the first and the wrong answer to
the second.  Measured on ``fa7_0`` at delta 0.6, the best of 1,535
similarity-passing endpoints carries a QED margin of ``-0.001663`` while holding
a similarity margin of ``+0.072`` and an SA margin of ``+1.720``.  Drawing more
programs from the same distribution does not close 0.0017 of QED; one rewrite
does.

WHAT THIS DOES
--------------
Given an endpoint the production :class:`~compose_v4.experiments.t4_fiber_campaign.Fiber`
refused, enumerate the EXACT legal fiber at production slot semantics and return
the successors the same gate accepts.  Every quantity used here -- QED,
similarity, SA -- is free to evaluate; this module never asks the oracle.

The trigger is a property of the endpoint's own violation vector, never of the
target, cell or seed: :func:`free_objective_near_miss` fires when the endpoint
satisfies every constraint the refinement cannot cheaply repair and fails only
on bounds that a local edit moves.  A refinement that fires on one cell and no
other is a finding about that cell, not a policy, so the caller is expected to
report the per-cell yield rather than assume it.

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

#: Bounds a single local rewrite can realistically move, and bounds it cannot.
#: QED is a smooth function of eight molecular descriptors and responds to a
#: one-atom edit; similarity to a FIXED external seed and SA do not improve
#: reliably by editing away from that seed, so an endpoint failing those is a
#: support problem and is deliberately NOT refined here.
REFINABLE_BOUNDS = ("qed",)


class SourceNotRepresentable(ValueError):
    """The endpoint parses as a molecule but is outside the COMPOSE vocabulary.

    T4 proposal lanes emit species RDKit will parse and COMPOSE will not build a
    graph for -- radicals above all, which is why ``Fiber`` carries a structural
    validity gate at all.  Such an endpoint has no legal fiber, so it cannot be
    refined; callers batching over a panel must be able to skip it rather than
    die inside the chemistry layer.
    """


# ---- Property helpers -------------------------------------------------------


def _properties(mol, fiber: Fiber) -> dict | None:
    """The production T4 property vector for `mol`, or None if it is not scorable."""

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

    return _properties(Chem.MolFromSmiles(smiles), Fiber(seed_smiles, delta))


def free_objective_near_miss(properties: dict, delta: float) -> bool:
    """True when the endpoint fails ONLY on bounds a local rewrite can move.

    General over structure and target: it reads the endpoint's own violation
    vector, never a target, cell or seed identity.
    """

    if properties is None:
        return False
    if properties["sim"] < delta or properties["sa"] > SA_MAX:
        return False
    return properties["qed"] < QED_MIN


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
    base = _properties(Chem.MolFromSmiles(smiles), fiber)
    if base is None:
        raise ValueError(f"refinement source is not a scorable molecule: {smiles!r}")

    rows: list[Refinement] = []
    for rule, product in _legal_successor_smiles(smiles, max_atoms):
        mol = Chem.MolFromSmiles(product)
        if mol is None or "." in product or mol.GetNumHeavyAtoms() > REPRESENTABLE_HEAVY_ATOMS:
            continue
        props = _properties(mol, fiber)
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
    base = _properties(Chem.MolFromSmiles(smiles), fiber)
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
    "REFINABLE_BOUNDS",
    "SCHEMA_VERSION",
    "Refinement",
    "SourceNotRepresentable",
    "endpoint_properties",
    "enumerate_refinements",
    "free_objective_near_miss",
    "refine_endpoint",
]
