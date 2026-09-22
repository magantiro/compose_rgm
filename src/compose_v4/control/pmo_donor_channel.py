"""Donor recombination as a PMO GENERATION channel, drawing its cuts from the T4 law.

WHAT THIS CHANNEL PROPOSES
--------------------------
One pendant exchange between the current parent and a molecule THIS RUN has already
scored: cut a bridge-separated substituent off the parent, cut one off a scored donor,
and join the donor's at the parent's anchor.  The target is CONSTRUCTED from that
explicit retained/added split -- it is never searched for -- and
``donor_program.compile_transplant`` then finds an executable route to it and RAISES if
replay changed a retained slot's element, charge or connectivity.

That last property is why this is not the compiler that was refuted.
``compile_source_to_target`` accepts an ARBITRARY pair and is
``delete_to_null_then_construct_v1`` -- retained_fraction 0.000 on 4 of 4 drug-like
pairs.  ``compile_transplant`` does not accept an arbitrary pair, so it cannot route
through the null state: preservation here is by construction, not by search quality.

THE LAW IS REUSED, NOT REWRITTEN
--------------------------------
MEASURED over 40 real scored molecules: ``donor_program.pendant_cuts`` returns exactly
the ``bond_order == 1`` subset of ``bridge_region_law.bridge_separated_regions`` -- 860
of each, ZERO disagreements -- and a ``PendantCut``'s ``root`` is exactly recoverable
from a ``BridgeRegion`` as the unique fragment atom adjacent to the anchor (0 failures
over all 860).  ``bridge_separated_regions`` is a strict superset, additionally
enumerating double (212) and triple (4) bond bridges, which ``transplant_plan`` refuses.

So the cut draw needs no new law.  :class:`~compose_v4.control.bridge_region_law.
BridgeRegionLaw` already supplies an uncapped support, a Boltzmann tilt on a caller
supplied margin, a support floor that makes it a RE-RANKING rather than a filter, and a
weighted order without replacement.  It was built and measured on T4; using it here gives
it a second, independent validation rather than a second implementation.

WHY THE TEMPERATURE IS NOT T4'S
-------------------------------
``bridge_region_law.MARGIN_TEMPERATURE`` is 0.1 in NORMALIZED FREE-GATE UNITS -- slack
against a similarity delta, a QED floor and an SA ceiling.  This channel's margin is a
RELEASED FRACTION, a different quantity on a different scale, and borrowing T4's constant
would be a unit error rather than a shared default.  :data:`DONOR_MARGIN_TEMPERATURE` is
the value the feasibility measurement was taken at and is named separately for that
reason.

WHAT THE MARGIN MEANS ON EACH SIDE -- SAME FORMULA, TWO DIFFERENT REASONS
-------------------------------------------------------------------------
``excise_region`` removes the fragment, so ``-(released fraction)`` is
``-(len(fragment) / n_real)`` on whichever molecule it is applied to.

* On the PARENT it is a RETENTION preference: keep most of a molecule the run already
  liked, rather than amputating it.
* On the DONOR it is an INSTALL-SIZE preference: take a small graft, because program
  length is roughly removed + installed and the controller has a realization ceiling.

The two are the same function for different reasons and are documented as such rather
than described as one idea.

WHY IT IS SITED AT GENERATION
-----------------------------
This branch measured the PMO discard points: ``_credit_allocate`` discards 16-17 of ~32
candidates per round and is capacity-limited on 6 of 6 rounds, while ``lock_query_subset``
discards none.  A GENERATION channel changes what exists and so is never exposed to the
ranking-inert failure where a law reorders endpoints that are all returned anyway; any
SELECTION this channel later earns must sit at ``_credit_allocate``, which is where
discarding actually happens.

THIS IS NOT A SCORED RESULT.  The feasibility that motivates the channel is one task, one
ledger, 60 pairs, offline compilation, replayed scores and zero oracle calls.  Medium
confidence, and the channel ships default-OFF.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from compose_v4.chem.molecular_graph import MolecularGraph, is_element
from compose_v4.control.bridge_region_law import (
    SUPPORT_FLOOR,
    BridgeRegion,
    BridgeRegionLaw,
)
from compose_v4.control.donor_program import PendantCut, compile_transplant

#: The channel's name in provenance and in the controller's per-channel counters.
DONOR_CHANNEL = "donor_transplant"

#: Boltzmann scale for the released-fraction margin. NOT `bridge_region_law`'s 0.1,
#: which is denominated in free-gate slack; see the module docstring.
DONOR_MARGIN_TEMPERATURE = 0.25


class DonorLawProbe(BaseException):
    """Raised from inside a law to prove the production path reaches it.

    Deliberately a ``BaseException`` and not a ``ValueError``/``RuntimeError``: the donor
    generation path catches both of those per attempt, exactly as the jump lane does, so
    a probe raising either would be swallowed by the very code being observed.
    """


# ---- The BridgeRegion <-> PendantCut correspondence ----------------------


def pendant_cut_from_region(
    graph: MolecularGraph, region: BridgeRegion
) -> PendantCut | None:
    """The ``PendantCut`` a ``BridgeRegion`` denotes, or ``None`` if there is not one.

    ``None`` is returned for a region whose bridge is not a SINGLE bond, because
    ``transplant_plan`` admits only single-bond bridges and would raise.  Returning
    ``None`` rather than raising keeps the caller's draw a re-ranking: a non-convertible
    region is skipped and the next one in the weighted order is tried, so the law's
    support is not silently narrowed by the conversion.

    ``root`` is the unique fragment atom adjacent to the anchor. It is unique because
    cutting that single bridge is what separated the fragment from the anchor in the
    first place, so a second fragment-anchor bond would contradict the enumeration.
    """
    if region.bond_order != 1:
        return None
    fragment = {int(slot) for slot in region.fragment}
    touching = [
        int(other)
        for other in np.flatnonzero(graph.bonds[region.anchor])
        if int(other) in fragment
    ]
    if len(touching) != 1:
        # Not reachable for a region this enumerator produced. Refuse rather than pick
        # one, because guessing here would fabricate a different transformation.
        return None
    return PendantCut(int(region.anchor), touching[0], tuple(sorted(fragment)))


def released_fraction_margin(graph: MolecularGraph) -> Callable[[MolecularGraph], float]:
    """``child -> -(released fraction)``, closed over the molecule being cut.

    Task-independent by construction: it reads atom COUNTS and nothing else -- no oracle
    value, no task identity, no declared target, no similarity reference. That is what
    lets this law run on a PMO task, where `free_gate_margin_law`'s similarity delta and
    reference molecule do not exist.
    """
    total = int(np.count_nonzero(is_element(graph.atom_types)))
    if total < 1:
        raise ValueError("a molecule with no real atoms has no region support")

    def margin(child: MolecularGraph) -> float:
        kept = int(np.count_nonzero(is_element(child.atom_types)))
        return (kept / total) - 1.0

    return margin


def donor_region_law(
    graph: MolecularGraph,
    *,
    maximum: int | None = None,
    floor: float = SUPPORT_FLOOR,
    temperature: float = DONOR_MARGIN_TEMPERATURE,
) -> BridgeRegionLaw:
    """The retentive law for one molecule: uncapped support, tilted toward small cuts."""
    return BridgeRegionLaw(
        maximum=maximum,
        margin=released_fraction_margin(graph),
        floor=floor,
        temperature=temperature,
    )


#: The law the shipped `donor_memory.RECIPE` describes as
#: `uniform_oriented_single_bridge`. Kept so the counterfactual can be NAMED rather than
#: reconstructed, and so a comparison against it cannot drift.
UNIFORM_ORIENTED_SINGLE_BRIDGE = BridgeRegionLaw(maximum=None, margin=None)


# ---- The named cut-law arms ---------------------------------------------

#: The retentive draw: the uniform SUPPORT, tilted toward cuts that release little.
#: This is the PRODUCTION DEFAULT. Measured over 60 pairs of top-scoring molecules from a
#: completed 250-call charged celecoxib ledger, same pairs and same compiler, differing
#: ONLY in the cut draw: retained median 0.324 -> 0.722 and program length median 36 -> 17
#: primitives. The length column is the one that decides something, because this branch
#: measured the controller's realization ceiling at 23 primitives -- the shipped uniform
#: draw puts the MEDIAN transplant above it and the retentive draw puts it below.
RETENTIVE_RELEASED_FRACTION = "retentive_released_fraction_v1"

#: The shipped `donor_memory.RECIPE["cut_distribution"]`, kept as a NAMED arm so a matched
#: comparison selects it by name rather than reconstructing it. It resolves to ``None``,
#: which is `donor_transplant_draw`'s unlawed `rng.permutation` branch -- the literal
#: shipped draw, not a `BridgeRegionLaw` that merely reproduces its support.
UNIFORM_ORIENTED_SINGLE_BRIDGE_ARM = "uniform_oriented_single_bridge"

#: Arm name -> law factory, where ``None`` is the unlawed draw. Resolution goes through
#: this table rather than a boolean so an unknown arm is a hard error instead of a silent
#: fallback to whichever branch a stray truth value happened to select.
CUT_LAWS: dict[str, Callable[[MolecularGraph], BridgeRegionLaw] | None] = {
    RETENTIVE_RELEASED_FRACTION: donor_region_law,
    UNIFORM_ORIENTED_SINGLE_BRIDGE_ARM: None,
}

#: What an unflagged donor lane draws. Named separately from the arm it currently points
#: at, so moving the default is a one-line, greppable change rather than an edit buried in
#: a signature.
DEFAULT_CUT_LAW = RETENTIVE_RELEASED_FRACTION


def resolve_cut_law(name: str) -> Callable[[MolecularGraph], BridgeRegionLaw] | None:
    """The law factory a named arm denotes. Raises on an unknown name.

    Raising matters more than it looks. The two arms differ in their PROBABILITIES over a
    shared support, so a typo that fell back to the other arm would produce a run that
    looks exactly like the one that was asked for and measures the other one.
    """
    if name not in CUT_LAWS:
        raise ValueError(
            f"unknown donor cut law {name!r}; the named arms are {sorted(CUT_LAWS)}"
        )
    return CUT_LAWS[name]


# ---- Generation ----------------------------------------------------------


@dataclass(frozen=True)
class DonorProposal:
    """One compiled transplant, with the evidence needed to judge it offline."""

    donor: str
    endpoint: str
    actions: tuple
    states: tuple
    retained_fraction: float
    removed_atoms: int
    added_atoms: int
    primitive_steps: int

    def payload(self) -> dict:
        return {
            "donor": self.donor,
            "endpoint": self.endpoint,
            "retained_fraction": self.retained_fraction,
            "removed_atoms": self.removed_atoms,
            "added_atoms": self.added_atoms,
            "primitive_steps": self.primitive_steps,
        }


def donor_transplant_draw(
    source: MolecularGraph,
    donors: list[tuple[str, MolecularGraph]],
    rng,
    *,
    law: Callable[[MolecularGraph], BridgeRegionLaw] | None = None,
    max_attempts: int = 8,
) -> tuple[DonorProposal | None, dict]:
    """One transplant draw: a parent cut and a donor cut, both from ``law``.

    ``law`` is a FACTORY, because the retentive law is closed over the molecule it cuts
    and the parent and donor are different molecules.  ``None`` means the shipped uniform
    draw, and it is the only byte-identical OFF: a `BridgeRegionLaw` with `margin=None`
    reproduces the uniform SUPPORT but still consumes `rng.random` through
    `BridgeRegionLaw.order`, where an unlawed permutation would not -- so a "uniform law"
    passed as an off switch would move every existing draw.  Absent is off; uniform is a
    named arm.

    Returns the proposal (or ``None``) and a status census, so a refusal is readable
    instead of collapsing into one string.
    """
    census: dict[str, int] = {}

    def note(status: str) -> None:
        census[status] = census.get(status, 0) + 1

    if not donors:
        note("no_donor_available")
        return None, census

    source_law = law(source) if law is not None else None
    source_regions = (
        source_law.order(source, rng)
        if source_law is not None
        else list(rng.permutation(np.asarray(_uniform_regions(source), dtype=object)))
    )
    for attempt in range(max_attempts):
        if attempt >= len(source_regions):
            note("source_support_exhausted")
            break
        source_cut = pendant_cut_from_region(source, source_regions[attempt])
        if source_cut is None:
            note("source_region_not_single_bond")
            continue
        donor_smiles, donor = donors[int(rng.integers(len(donors)))]
        donor_law = law(donor) if law is not None else None
        donor_regions = (
            donor_law.order(donor, rng)
            if donor_law is not None
            else list(rng.permutation(np.asarray(_uniform_regions(donor), dtype=object)))
        )
        if not donor_regions:
            note("donor_has_no_region")
            continue
        donor_cut = pendant_cut_from_region(donor, donor_regions[0])
        if donor_cut is None:
            note("donor_region_not_single_bond")
            continue
        result = compile_transplant(source, donor, source_cut, donor_cut)
        status = str(result.get("status"))
        note(status)
        if status != "compiled":
            continue
        return (
            DonorProposal(
                donor=donor_smiles,
                endpoint=result["smiles"],
                actions=tuple(result["actions"]),
                states=tuple(result["states"]),
                retained_fraction=1.0 - float(result["released_fraction"]),
                removed_atoms=int(result["removed_atoms"]),
                added_atoms=int(result["added_atoms"]),
                primitive_steps=int(result["primitive_steps"]),
            ),
            census,
        )
    return None, census


def _uniform_regions(graph: MolecularGraph) -> list[BridgeRegion]:
    return list(UNIFORM_ORIENTED_SINGLE_BRIDGE.regions(graph))


# ---- The consumption gate ------------------------------------------------


def assert_donor_law_is_consumed(draw: Callable[[object], object], *, seeds: int = 16) -> dict:
    """Prove the caller's own draw closure REACHES the law. Raises if it does not.

    This exists because a validated mechanism behind an opt-in argument is INERT until a
    caller passes it, and this repository has shipped three such mechanisms.  Reading a
    signature cannot detect it: in both recorded cases the argument existed and was
    dropped one hop later.  So the check installs a law that RAISES and requires the
    production path to reach it.

    ``seeds`` is FIXED and plural on purpose.  A single draw can legitimately miss the
    law -- the draw may refuse every region for an unrelated reason -- so a random single
    seed would make a wiring defect and an unlucky draw indistinguishable.  With fixed
    seeds the answer is deterministic: for a given code state this always passes or
    always fails.
    """

    def probing_law(_graph: MolecularGraph) -> BridgeRegionLaw:
        raise DonorLawProbe("the donor region law was reached")

    for seed in range(seeds):
        try:
            draw(probing_law, np.random.default_rng([20260921, seed]))
        except DonorLawProbe:
            return {"consumed": True, "seed": seed, "seeds_tried": seed + 1}
    raise AssertionError(
        "the donor region law was NEVER consulted across "
        f"{seeds} fixed seeds -- the channel is wired but inert"
    )
