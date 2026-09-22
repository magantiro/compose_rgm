"""Online task-adaptive structural proposal memory for the PMO controller.

WHAT THIS CHANGES
-----------------
This module changes **which proposals are generated**, not which finished
candidates are kept.  It produces a :class:`MemoryRegionLaw` -- a region-draw
law in the sense of :mod:`compose_v4.control.bridge_region_law` -- that is
consulted inside ``compile_generic_module`` while a program is being built, by
``_delete_pendant_fragment``, for the ``substituent_delete`` and
``segment_replace`` module families.  The draw it biases happens *before any
candidate exists*, so no re-ranking of finished endpoints is involved and no
channel quota is touched.

    proposal path (shallow lane)
      PmoPopulationController.propose_batch
        -> _generate_channel_pool                       (v21)
          -> _channel_proposal                          (v21)
            -> DynamicProgramOptimizer._mutate
              -> synthesize_dynamic_program(region_law=<THIS>)
                -> compile_generic_module(region_law=<THIS>)
                  -> _delete_pendant_fragment(law=<THIS>)
                    -> law.order(graph, rng)   <== the distribution changes HERE

INTEGRATION -- THE ONE CALL SITE THE COORDINATOR ADDS
-----------------------------------------------------
``_channel_proposal`` is reached as ``self._channel_proposal(...)`` from
``_generate_channel_pool``, so it can be overridden in
``pmo_population_controller.py`` -- the file the controller owns -- without
touching ``dynamic_program_synthesis.py``, which is pinned by roughly twenty
live T4 contracts, or ``dynamic_program_synthesis_v21.py``.  Note that
overriding ``_mutate`` would NOT work: v21 calls
``DynamicProgramOptimizer._mutate(self, entry)`` unbound, so a subclass
override is never consulted.

Add to ``PmoPopulationController``::

    self.online_memory = OnlineProposalMemory()      # in __init__

    def _channel_proposal(self, channel, entry):     # the one call site
        return memory_channel_proposal(
            self, channel, entry, super()._channel_proposal, self.online_memory
        )

and feed it in ``observe_batch``, where counted scores already arrive::

    self.online_memory.observe_scored_molecule(
        endpoint=..., score=..., graph=decode_state(...)
    )
    self.online_memory.observe_transition(...)

Until something passes the memory, this module is INERT -- which is the
failure this repository has hit three times.  ``memory_channel_proposal``
therefore refuses to be a no-op silently: with a warm memory on the shallow
lane it never falls back, and
``tests/test_pmo_online_memory.py::test_the_adapter_installs_the_law_on_the_production_path``
drives the adapter through the shipped consumption probe, so dropping the
``region_law=`` keyword turns a named test red.

SCOPE, STATED UP FRONT
----------------------
Only the **shallow** lane is biased.  The structured lane calls
``synthesize_structured_program``, whose signature has no ``region_law``
parameter at all, so the keyword cannot reach its draw site; claiming that
lane would be claiming a hook that does not exist.  The jump lane is a
separate proposer.

Measured from the completed PMO-v2 run's own final-round channel counters
(three tasks summed, read from the round snapshots rather than from the
attribution report, whose embedded counters are the stale round-13 reading it
corrects in its own text): shallow 979 proposals / 725 executions / 319 oracle
selections, structured 939 / 688 / 302, jump 1,091 / 3 / 3, total 3,009 / 1,416
/ 624.  So the shallow lane is 32.5% of proposals and **51.1% of everything
that reached the oracle** -- a live lane and not a corner, but one lane of
three, and any result should say so.

TWO MEMORIES, AND WHY THEY ARE NOT SYMMETRIC
--------------------------------------------
:class:`EditOutcomeMemory` (**primary**) stores counted transitions
``(G, Z, G', f(G), f(G'))``: what region was touched, at what scale, by which
module family, and what the score did.  It localizes evidence to the edit.

:class:`DonorRegionMemory` (**secondary**) stores regions appearing in
high-scoring scored molecules, with full provenance.  These are
**associations, never causal labels**: a high-scoring molecule containing a
fragment is not evidence the fragment caused the score.

The two are deliberately **not** equal partners.  Measured on the completed
PMO-v2 run, the gsk3b oracle -- a fingerprint RandomForest -- correlates
``r(score, QED) = -0.433`` and ``r(score, SA) = +0.684``: its top scorers are
off-manifold molecules carrying bare phosphorus, hypervalent iodine and
peroxide-hydrazine chains.  On such an oracle, selecting regions by the score
rank of the molecule that contained them learns the *exploitation*, and it
looks like it is working because the score rises.  So:

* the donor memory can only **modulate** a context the edit memory already has
  counted evidence for, and can never by itself promote one (see
  :meth:`OnlineProposalMemory.context_value`);
* every donor row keeps the endpoint and the call ordinal it came from, so any
  later claim built on it can be checked against the molecule it came from.

THE SIZE CONFOUND, AND WHY THE EVIDENCE IS RESIDUALIZED
-------------------------------------------------------
``r(score, heavy_atoms)`` is ``+0.67`` to ``+0.81`` on all three measured
tasks, and the run's archive is full of one- and two-atom parents.  A memory
fitted on raw score change would therefore learn "add atoms" and dress it up as
structural knowledge.  :class:`SizeResidual` fits the mean outcome per
heavy-atom-change bucket and the memory learns only the **residual**: does this
kind of region do better than a size-matched edit?  ``report()`` carries the
size model so the confound stays visible.

WHAT THE MEMORY IS ALLOWED TO KNOW
----------------------------------
Only two things: the task-independent structure of states it is shown, and
**counted** oracle observations from the run in progress -- each one a
``(endpoint, score)`` pair the run charged to its own ledger.

It never sees the oracle's internals, a target SMILES, a hidden component
score, a task identifier, or any molecule or winner identity carried in from a
previous run.  There is no task-keyed branch anywhere in this module: two tasks
differ only by the counted observations they produced.  :meth:`certification`
states the exclusion in the artifact.

The memory also computes **no molecular property** -- no QED, no SA, no
similarity to anything.  That is deliberate: where a property is the objective,
evaluating it off-ledger to choose a proposal is uncounted objective
evaluation.  Chemical character is reported by the *gate harness*, which is a
diagnostic and never feeds back into a draw.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from compose_v4.chem.molecular_graph import MolecularGraph, is_element
from compose_v4.control.bridge_region_law import (
    SUPPORT_FLOOR,
    BridgeRegion,
    BridgeRegionLaw,
    RegionRealizationError,
    excise_region,
)

SCHEMA = "pmo_online_memory_v1"

#: Frontier width.  PMO grades a run on the mean of its top ten distinct
#: scored molecules, so that is the quantity an allocation objective must move.
TOP_K = 10

#: Observations required before a context's evidence is used at full strength.
#: Below it the estimate is shrunk toward zero, so a single lucky edit cannot
#: capture the draw.
SHRINKAGE = 8.0

#: Heavy-atom-change bucket width for the size-confound control.
SIZE_BUCKET = 4

#: Tilt strength, in ROBUST STANDARD DEVIATIONS of the value spread over the
#: source's own regions -- a dimensionless quantity.
#:
#: It is not a Boltzmann temperature in score units, and that distinction was
#: found by the offline gate rather than reasoned out in advance.  A fixed
#: temperature of 0.05 produced a max/min weight ratio of only 1.74x on gsk3b,
#: because the learned values there span 0.028 while on perindopril_mpo they
#: span 0.071.  Scores live on different scales per task, so a temperature
#: fixed in score units is a different tilt strength on every task -- which
#: would make the controller task-dependent by accident, exactly what the
#: one-controller requirement forbids.  Normalizing first makes the strength
#: mean the same thing everywhere.
TILT = 1.5

#: Retained so a caller may still request an absolute-units tilt; the memory
#: law normalizes instead and does not read it.
TEMPERATURE = 0.05

#: Largest share of a context's value the donor association may contribute.
#: The donor memory is an association, so it is capped well below the counted
#: edit evidence it modulates.
DONOR_CEILING = 0.25


# ---- The stratified scored bank ------------------------------------------
#
# The bank is the run's own counted population, offered as structural MATERIAL
# rather than as a ranking.  It exists because a top-N-by-score pool is a pool
# of whatever the search has drifted toward: on this repository's own measured
# PMO runs the population leaves the drug-like manifold on almost every task,
# so recombining the top of that population recombines the drift.  Stratifying
# for structural coverage and for demonstrated productivity, beside score,
# keeps material in the bank that score alone would have discarded.

#: Total selection size.  Matched to InVirtuoGen's no-prescreen replay buffer
#: (300) so the donor lane draws from a population of comparable size to the
#: comparator's, rather than from a number chosen here.
BANK_CAPACITY = 300

#: Per-stratum capacities.  They SUM to :data:`BANK_CAPACITY` and are equal: this
#: is a DECLARED split, not a calibrated one, and it is named so that changing it
#: is a one-line, greppable edit rather than a tuned constant hidden in a method.
ELITE_CAPACITY = 100
PROMISING_CAPACITY = 100
DIVERSE_CAPACITY = 100

ELITE = "elite"
PROMISING = "promising"
DIVERSE = "diverse"

#: Stratum order is also PRECEDENCE: a molecule that qualifies for more than one
#: stratum occupies exactly one slot, in this order.  Disjointness matters because
#: the draw picks a stratum first and then a member, so a molecule appearing in two
#: strata would receive two shares of the mass.
STRATA = (ELITE, PROMISING, DIVERSE)
STRATUM_CAPACITY = {
    ELITE: ELITE_CAPACITY,
    PROMISING: PROMISING_CAPACITY,
    DIVERSE: DIVERSE_CAPACITY,
}

#: Draw mass per stratum, BEFORE redistribution over the strata that are non-empty.
#:
#: Uniform, and the uniformity is the point: the shares are declared rather than
#: fitted, so no stratum's weight encodes a result.  It is also why the draw must be
#: STRATUM-FIRST.  Drawing uniformly over the union would give the elite stratum its
#: share by size instead of by declaration -- the same defect as the 2026-07-24
#: corruption-selection finding, where a weighted permutation over all instances
#: over-picked whichever family had the most of them.
STRATUM_WEIGHT = {ELITE: 1.0 / 3.0, PROMISING: 1.0 / 3.0, DIVERSE: 1.0 / 3.0}


# ---- Structural context --------------------------------------------------


@dataclass(frozen=True)
class RegionContext:
    """A task-independent structural descriptor of one region draw.

    Keyed on what the draw *is* -- how big, what it hangs off, whether it
    carries a ring -- and never on which molecule, parent or task it came from,
    so evidence generalizes across parents instead of memorizing one lineage.
    """

    size_bucket: int
    anchor_element: int
    bond_order: int
    anchor_in_ring: bool
    fragment_has_ring: bool

    @classmethod
    def of(cls, graph: MolecularGraph, region: BridgeRegion) -> RegionContext:
        return cls(
            size_bucket=min(region.size, 24) // SIZE_BUCKET,
            anchor_element=int(graph.atom_types[region.anchor]),
            bond_order=int(region.bond_order),
            anchor_in_ring=_in_ring(graph, region.anchor),
            fragment_has_ring=any(_in_ring(graph, a) for a in region.fragment),
        )

    def key(self) -> tuple:
        return (
            self.size_bucket,
            self.anchor_element,
            self.bond_order,
            self.anchor_in_ring,
            self.fragment_has_ring,
        )


def _in_ring(graph: MolecularGraph, slot: int) -> bool:
    """True when ``slot`` lies on a cycle of the real-atom graph.

    A vertex is on a cycle exactly when some incident edge is not a bridge, so
    this reuses the same two-sides test the region enumeration uses rather than
    perceiving aromaticity, which would require a chemistry round-trip.
    """

    real = {int(i) for i in np.flatnonzero(is_element(graph.atom_types))}
    adjacency = {
        s: {int(j) for j in np.flatnonzero(graph.bonds[s]) if int(j) in real}
        for s in real
    }
    for other in adjacency.get(slot, ()):  # an edge on a cycle keeps both ends joined
        seen, stack = {slot}, [slot]
        cut = frozenset((slot, other))
        while stack:
            at = stack.pop()
            for nxt in adjacency[at]:
                if frozenset((at, nxt)) == cut or nxt in seen:
                    continue
                seen.add(nxt)
                stack.append(nxt)
        if other in seen:
            return True
    return False


# ---- The frontier the run is actually graded on --------------------------


@dataclass
class FrontierLedger:
    """Counted scored molecules and the top-ten mean they induce.

    ``U10`` is the mean of the top :data:`TOP_K` **distinct** scored molecules,
    which is the quantity PMO's top-ten AUC integrates.  ``frontier_gain`` is
    what a new observation would have been worth to it -- the allocation
    objective the brief asks for, and the reason an edit lifting a strong parent
    0.80 -> 0.84 can outrank one lifting a weak parent 0.05 -> 0.25.
    """

    scores: dict[str, float] = field(default_factory=dict)

    def observe(self, endpoint: str, score: float) -> None:
        previous = self.scores.get(endpoint)
        if previous is not None and previous != score:
            raise ValueError(f"counted score for {endpoint!r} changed")
        self.scores[endpoint] = float(score)

    def top_k_mean(
        self, extra: float | None = None, *, exclude: str | None = None
    ) -> float:
        values = [
            value for key, value in self.scores.items() if key != exclude
        ]
        if extra is not None:
            values.append(float(extra))
        head = sorted(values, reverse=True)[:TOP_K]
        return float(sum(head) / TOP_K) if head else 0.0

    def frontier_gain(self, score: float, *, exclude: str | None = None) -> float:
        """``U10(A + {G'}) - U10(A)`` -- never negative, zero off the frontier.

        ``exclude`` removes one endpoint from the BASELINE, which is what makes
        the answer independent of whether the caller has already recorded the
        observation.  Without it the result depends on update order: recording
        the molecule first and then asking what it contributed counts it twice
        and understates its arrival.  Order-dependence in a learning target is
        the kind of defect that produces a plausible number rather than an
        error, so the ordering is removed rather than documented.
        """

        return max(
            0.0,
            self.top_k_mean(score, exclude=exclude)
            - self.top_k_mean(exclude=exclude),
        )


@dataclass
class ScoredMoleculeBank:
    """Every molecule this run has COUNTED, with the evidence each stratum needs.

    THE BANK RETAINS, THE SELECTION BOUNDS
    --------------------------------------
    Rows are kept for every counted observation; :data:`BANK_CAPACITY` bounds the
    SELECTION, not the buffer.  That is a deliberate divergence from a fixed-size
    replay buffer, and it is not a size saving: eviction would be irreversible and
    wrong here, because whether a molecule belongs in the *promising* stratum is
    decided by observations that arrive AFTER it -- a molecule evicted at call 200
    for a low score can become the best donor in the run at call 600 when its child
    improves.  The frontier ledger beside this one already retains unboundedly for
    the same reason, so this costs nothing new.

    WHAT EACH STRATUM IS EVIDENCE OF, AND WHAT IT IS NOT
    ----------------------------------------------------
    ``elite``      counted score.  The strongest evidence available and the most
                   drifted: on a predictor-backed oracle the top scorers are the
                   off-manifold molecules.
    ``promising``  counted PARENT-RELATIVE improvement delivered by this molecule's
                   own children.  This is the only stratum whose evidence is about
                   the molecule as a SOURCE of edits rather than as a product, which
                   is exactly the role a donor plays.
    ``diverse``    structural coverage: the best-scoring representative of a basin
                   the higher strata do not already occupy.  A coverage backstop, and
                   an ASSOCIATION at most -- occupying a distinct scaffold is not
                   evidence that the scaffold is good.

    Precedence is ``elite -> promising -> diverse`` and the strata are DISJOINT.

    INFORMATION REGIME
    ------------------
    Score comes from the run's own charged ledger; improvement is the difference of
    two charged scores; the basin is the Bemis-Murcko scaffold of a molecule the run
    has already produced.  The basin is supplied BY THE CALLER, using the same
    :func:`compose_v4.control.pmo_credit.basin_label` the allocator already groups by,
    so this module still computes no chemistry and the bank's notion of "structurally
    distinct" is the controller's existing one rather than a second definition.
    """

    rows: dict[str, dict[str, Any]] = field(default_factory=dict)

    # -- updates ----------------------------------------------------------

    def observe(self, *, endpoint: str, score: float, basin: str | None = None) -> None:
        """Bank one counted observation.

        Refuses a changed score for the same molecule, exactly as
        :class:`FrontierLedger` does: two different counted values for one endpoint
        means the ledger join is wrong, and silently keeping either one would put a
        number in the bank that no charged call produced.
        """

        row = self.rows.get(endpoint)
        if row is None:
            self.rows[endpoint] = {
                "score": float(score),
                "basin": basin,
                "children": 0,
                "improvement": 0.0,
            }
            return
        if row["score"] != float(score):
            raise ValueError(f"counted score for {endpoint!r} changed")
        if row["basin"] is None and basin is not None:
            row["basin"] = basin

    def observe_lineage(
        self, *, parent_endpoint: str, parent_score: float | None, child_score: float
    ) -> bool:
        """Credit a parent with one counted child and its positive improvement.

        Returns whether the parent was known.  A parent the bank has never seen is
        NOT created here: a bank row must be a molecule the ledger charged for, and
        inventing one from a provenance field would put an unscored molecule into a
        pool whose whole claim is that every member was counted.

        ``parent_score is None`` means the transition carries no parent baseline, so
        there is no improvement to measure; the child is still counted, because "this
        molecule has been used as a parent" is itself evidence the promising stratum
        needs in order to rank by improvement PER USE rather than by luck of reuse.
        """

        row = self.rows.get(parent_endpoint)
        if row is None:
            return False
        row["children"] += 1
        if parent_score is not None:
            row["improvement"] += max(0.0, float(child_score) - float(parent_score))
        return True

    # -- selection --------------------------------------------------------

    def strata(self, *, capacity: int = BANK_CAPACITY) -> dict[str, list[str]]:
        """The stratified selection: disjoint, deterministic, ordered within stratum.

        ``capacity`` scales the three declared per-stratum capacities together, so a
        caller asking for a smaller bank gets the same MIX rather than a truncated
        elite.  Every order is fully tie-broken on the endpoint string: a selection
        whose order depended on dict insertion would not survive a resume.
        """

        if capacity < 0:
            raise ValueError("bank capacity cannot be negative")
        scale = capacity / BANK_CAPACITY if BANK_CAPACITY else 0.0
        taken: set[str] = set()

        def room(name: str) -> int:
            return round(STRATUM_CAPACITY[name] * scale)

        elite = [
            endpoint
            for endpoint, _ in sorted(
                self.rows.items(), key=lambda kv: (-kv[1]["score"], kv[0])
            )
        ][: room(ELITE)]
        taken.update(elite)

        promising = [
            endpoint
            for endpoint, _ in sorted(
                (
                    (endpoint, row)
                    for endpoint, row in self.rows.items()
                    if endpoint not in taken and row["improvement"] > 0.0
                ),
                key=lambda kv: (-kv[1]["improvement"], -kv[1]["score"], kv[0]),
            )
        ][: room(PROMISING)]
        taken.update(promising)

        # One representative per basin the higher strata do not already occupy, and
        # the basin's own best scorer represents it: the stratum buys COVERAGE, so
        # spending its slots on several members of one scaffold would buy nothing.
        occupied = {
            self.rows[endpoint]["basin"]
            for endpoint in taken
            if self.rows[endpoint]["basin"] is not None
        }
        best_of_basin: dict[str, tuple[float, str]] = {}
        for endpoint, row in self.rows.items():
            basin = row["basin"]
            if endpoint in taken or basin is None or basin in occupied:
                continue
            current = best_of_basin.get(basin)
            if current is None or (-row["score"], endpoint) < (-current[0], current[1]):
                best_of_basin[basin] = (row["score"], endpoint)
        diverse = [
            endpoint
            for _, endpoint in sorted(
                best_of_basin.values(), key=lambda pair: (-pair[0], pair[1])
            )
        ][: room(DIVERSE)]

        return {ELITE: elite, PROMISING: promising, DIVERSE: diverse}

    def weights(self, strata: dict[str, list[str]]) -> dict[str, float]:
        """Declared stratum mass, renormalized over the strata that have members.

        An empty stratum yields its share rather than wasting it; it does not silently
        shrink the others' relative proportions.
        """

        live = {name: STRATUM_WEIGHT[name] for name in STRATA if strata.get(name)}
        total = sum(live.values())
        if not total:
            return {}
        return {name: weight / total for name, weight in live.items()}

    # -- reporting --------------------------------------------------------

    def report(self, *, capacity: int = BANK_CAPACITY) -> dict[str, Any]:
        selection = self.strata(capacity=capacity)
        return {
            "counted_molecules": len(self.rows),
            "basins": len(
                {r["basin"] for r in self.rows.values() if r["basin"] is not None}
            ),
            "rows_without_basin": sum(1 for r in self.rows.values() if r["basin"] is None),
            "parents_with_an_improving_child": sum(
                1 for r in self.rows.values() if r["improvement"] > 0.0
            ),
            "capacity": int(capacity),
            "selected": {name: len(members) for name, members in selection.items()},
            "weights": self.weights(selection),
        }


# ---- Size-confound control ----------------------------------------------


@dataclass
class SizeResidual:
    """Mean observed outcome per heavy-atom-change bucket.

    Exists so the memory cannot pass off the corpus-wide size gradient as
    structural knowledge.  The residual it returns is the part of an outcome a
    size-matched edit would NOT have produced.
    """

    totals: dict[int, float] = field(default_factory=lambda: defaultdict(float))
    counts: dict[int, int] = field(default_factory=lambda: defaultdict(int))

    @staticmethod
    def bucket(delta_heavy: int) -> int:
        return int(np.sign(delta_heavy)) * (min(abs(int(delta_heavy)), 24) // SIZE_BUCKET)

    def observe(self, delta_heavy: int, outcome: float) -> None:
        at = self.bucket(delta_heavy)
        self.totals[at] += float(outcome)
        self.counts[at] += 1

    def expected(self, delta_heavy: int) -> float:
        at = self.bucket(delta_heavy)
        n = self.counts.get(at, 0)
        if n == 0:
            grand = sum(self.totals.values())
            total = sum(self.counts.values())
            return float(grand / total) if total else 0.0
        return float(self.totals[at] / n)

    def residual(self, delta_heavy: int, outcome: float) -> float:
        return float(outcome) - self.expected(delta_heavy)

    def report(self) -> dict[str, Any]:
        return {
            "buckets": {
                str(k): {"n": self.counts[k], "mean_outcome": self.totals[k] / self.counts[k]}
                for k in sorted(self.counts)
            },
            "bucket_width_heavy_atoms": SIZE_BUCKET,
        }


# ---- Memory 1: edit outcomes (PRIMARY) -----------------------------------


@dataclass
class EditOutcomeMemory:
    """Counted transitions ``(G, Z, G', f(G), f(G'))``, keyed by region context.

    Each row records what changed and what happened to the score.  The stored
    statistic is the **size-residualized** outcome, so a context is credited
    only for doing better than a size-matched edit.
    """

    rows: list[dict[str, Any]] = field(default_factory=list)
    totals: dict[tuple, float] = field(default_factory=lambda: defaultdict(float))
    counts: dict[tuple, int] = field(default_factory=lambda: defaultdict(int))

    def observe(
        self,
        *,
        context: RegionContext,
        family: str,
        delta_heavy: int,
        residual: float,
        parent_endpoint: str,
        child_endpoint: str,
        parent_score: float | None,
        child_score: float,
    ) -> None:
        key = context.key()
        self.totals[key] += float(residual)
        self.counts[key] += 1
        self.rows.append(
            {
                "context": key,
                "family": family,
                "delta_heavy": int(delta_heavy),
                "residual": float(residual),
                "parent_endpoint": parent_endpoint,
                "child_endpoint": child_endpoint,
                "parent_score": parent_score,
                "child_score": float(child_score),
            }
        )

    def value(self, context: RegionContext) -> tuple[float, int]:
        """Shrunk residual effect of this context, with its observation count."""

        key = context.key()
        n = self.counts.get(key, 0)
        if n == 0:
            return 0.0, 0
        mean = self.totals[key] / n
        return float(mean * (n / (n + SHRINKAGE))), n


# ---- Memory 2: donor regions (SECONDARY, association only) ---------------


@dataclass
class DonorRegionMemory:
    """Regions observed inside scored molecules, with provenance retained.

    A row asserts only *co-occurrence*: this region was present in a molecule
    that scored this much, at this counted call.  It does not assert that the
    region caused the score, and on a predictor-backed oracle it is often
    evidence of the opposite.  Provenance is kept so a later claim can be
    audited against the molecule it came from.
    """

    rows: list[dict[str, Any]] = field(default_factory=list)
    totals: dict[tuple, float] = field(default_factory=lambda: defaultdict(float))
    counts: dict[tuple, int] = field(default_factory=lambda: defaultdict(int))

    def observe(
        self,
        *,
        context: RegionContext,
        endpoint: str,
        score: float,
        ordinal: int,
    ) -> None:
        key = context.key()
        self.totals[key] += float(score)
        self.counts[key] += 1
        self.rows.append(
            {
                "context": key,
                "source_endpoint": endpoint,
                "source_score": float(score),
                "counted_call_ordinal": int(ordinal),
                "claim": "association_only",
            }
        )

    def association(self, context: RegionContext) -> tuple[float, int]:
        """Mean score of molecules containing this context, centred on the mean.

        Centring is what makes this a contrast rather than a level: a context
        present in everything scores zero, not "good".
        """

        key = context.key()
        n = self.counts.get(key, 0)
        total_n = sum(self.counts.values())
        if n == 0 or total_n == 0:
            return 0.0, 0
        grand = sum(self.totals.values()) / total_n
        mean = self.totals[key] / n
        return float((mean - grand) * (n / (n + SHRINKAGE))), n


# ---- The law -------------------------------------------------------------


@dataclass(frozen=True)
class MemoryRegionLaw(BridgeRegionLaw):
    """A region law whose weights come from counted run evidence.

    Overrides :meth:`BridgeRegionLaw.weights` rather than supplying a ``margin``
    callable, because a margin sees only the realized child while the evidence
    is keyed on the *region* -- its size, its anchor and what it hangs off.
    ``order`` is inherited unchanged, so the draw remains the same
    Efraimidis-Spirakis weighted sampling without replacement the tested law
    uses, and the support floor is likewise inherited: every region that was
    drawable keeps strictly positive probability, so this is a RE-RANKING and
    can never delete part of the support.
    """

    memory: OnlineProposalMemory | None = None
    tilt: float = TILT

    def weights(
        self, graph: MolecularGraph, regions, /
    ) -> np.ndarray:
        """Weights from normalized learned value, with the inherited floor.

        Values are standardized against the spread over THIS source's own
        regions before the tilt is applied, so the tilt strength is in robust
        standard deviations and means the same thing whatever scale the task's
        scores happen to live on.  A source whose regions all look alike gets a
        degenerate spread and falls back to uniform, which is the correct
        answer: there is nothing to prefer.
        """

        if self.memory is None:
            return super().weights(graph, regions)
        values = np.empty(len(regions), dtype=float)
        realizable = np.ones(len(regions), dtype=bool)
        for position, region in enumerate(regions):
            try:
                excise_region(graph, region)
            except RegionRealizationError:
                realizable[position] = False
                values[position] = 0.0
                continue
            values[position] = self.memory.context_value(
                RegionContext.of(graph, region), size=region.size
            )
            self.memory.cost["region_weight_evaluations"] += 1
        if not realizable.any():
            return np.full(len(regions), self.floor)
        live = values[realizable]
        centre = float(np.median(live))
        spread = float(np.median(np.abs(live - centre))) * 1.4826
        if spread <= 1e-12:
            spread = float(live.std())
        out = np.empty(len(regions), dtype=float)
        for position in range(len(regions)):
            if not realizable[position]:
                out[position] = self.floor
                continue
            if spread <= 1e-12:
                out[position] = 1.0
                continue
            z = (values[position] - centre) / spread
            out[position] = max(
                self.floor, math.exp(float(np.clip(z, -6.0, 6.0)) * self.tilt)
            )
        return out


# ---- The memory ----------------------------------------------------------


@dataclass
class OnlineProposalMemory:
    """Both memories plus the frontier ledger they are scored against.

    Updated only from counted observations of the run in progress.  Cold (no
    counted evidence) it returns ``None`` from :meth:`region_law`, which is the
    only byte-identical "off": a uniform law object would reproduce v1's support
    but not v1's draws, because an unlawed ``_delete_pendant_fragment`` consumes
    ``rng.permutation`` while any law consumes ``rng.random``.
    """

    frontier: FrontierLedger = field(default_factory=FrontierLedger)
    size: SizeResidual = field(default_factory=SizeResidual)
    edits: EditOutcomeMemory = field(default_factory=EditOutcomeMemory)
    donors: DonorRegionMemory = field(default_factory=DonorRegionMemory)
    #: The run's stratified scored population, offered as structural MATERIAL to the
    #: donor recombination lane.  Distinct from `donors` above, which holds region
    #: CONTEXTS for the proposal law; this holds whole counted molecules.
    bank: ScoredMoleculeBank = field(default_factory=ScoredMoleculeBank)
    ordinal: int = 0
    donor_ceiling: float = DONOR_CEILING
    #: Proposal compute, counted SEPARATELY from oracle calls and in
    #: load-independent units.  Wall clock cannot be used here: other jobs run
    #: on the same machine, and an easy generator that completes reliably would
    #: otherwise look cheap while monopolising the run.  `region_weight_
    #: evaluations` is the work this memory itself adds; `syntheses` is the
    #: number of production proposal draws it served.
    cost: dict[str, int] = field(
        default_factory=lambda: {"syntheses": 0, "region_weight_evaluations": 0}
    )

    # -- durable state ----------------------------------------------------

    def payload(self) -> dict[str, Any]:
        """Serialise every counted observation this memory holds.

        Without this a resumed campaign silently restarts the memory COLD and
        re-learns from nothing -- arm B would drift back toward arm A for its
        remaining rounds and the divergence would be invisible in the artifact.
        That is the same failure the credit and pool-continuity payloads exist to
        prevent, recorded in `PmoPopulationController.snapshot`.

        ``totals``/``counts`` are keyed by tuples, which JSON cannot express as
        object keys, so they are written as explicit [key, value] pairs rather
        than stringified -- a stringified key cannot be read back to the tuple it
        came from without a parser that would drift from this writer.
        """

        def pairs(mapping: dict) -> list:
            # Keys are tuples for the context maps and plain ints for the size
            # residual, so encode a scalar key as itself rather than forcing it
            # through list() -- which raised on the int-keyed map.
            def encode(key):
                return list(key) if isinstance(key, tuple) else key

            return [
                [encode(key), value]
                for key, value in sorted(mapping.items(), key=lambda kv: str(kv[0]))
            ]

        return {
            "schema_version": "pmo_online_memory_payload_v1",
            "ordinal": int(self.ordinal),
            "donor_ceiling": float(self.donor_ceiling),
            "cost": dict(self.cost),
            "frontier": dict(self.frontier.scores),
            "size": {"totals": pairs(self.size.totals), "counts": pairs(self.size.counts)},
            "edits": {
                "rows": list(self.edits.rows),
                "totals": pairs(self.edits.totals),
                "counts": pairs(self.edits.counts),
            },
            "donors": {
                "rows": list(self.donors.rows),
                "totals": pairs(self.donors.totals),
                "counts": pairs(self.donors.counts),
            },
            # Serialised for the same reason as everything else here: a resumed arm
            # whose bank restarted empty would draw its donors from the last few
            # rounds only, and report itself as the arm that ran the whole way.
            "bank": {"rows": dict(self.bank.rows)},
        }

    def restore_payload(self, payload: dict[str, Any]) -> None:
        """Rebuild from :meth:`payload`. Refuses a payload it did not write."""

        if payload.get("schema_version") != "pmo_online_memory_payload_v1":
            raise ValueError(
                "online memory payload schema changed; refusing to resume a memory "
                "whose encoding this reader did not write"
            )

        def unpair(rows, cast) -> dict:
            return {
                (tuple(key) if isinstance(key, list) else key): cast(value)
                for key, value in rows
            }

        self.ordinal = int(payload["ordinal"])
        self.donor_ceiling = float(payload["donor_ceiling"])
        self.cost = dict(payload["cost"])
        self.frontier.scores = dict(payload["frontier"])
        self.size.totals = defaultdict(float, unpair(payload["size"]["totals"], float))
        self.size.counts = defaultdict(int, unpair(payload["size"]["counts"], int))
        self.edits.rows = list(payload["edits"]["rows"])
        self.edits.totals = defaultdict(float, unpair(payload["edits"]["totals"], float))
        self.edits.counts = defaultdict(int, unpair(payload["edits"]["counts"], int))
        self.donors.rows = list(payload["donors"]["rows"])
        self.donors.totals = defaultdict(float, unpair(payload["donors"]["totals"], float))
        self.donors.counts = defaultdict(int, unpair(payload["donors"]["counts"], int))
        # A payload written before the bank existed carries none. Rebuilding it from
        # `frontier` would be silently WRONG, not merely partial: the frontier has no
        # basin and no lineage, so every row would land in the elite stratum and the
        # resumed arm would run a top-N pool under the stratified arm's name. An empty
        # bank refills from the rounds that follow and is visible as such in `report`.
        self.bank.rows = {
            endpoint: dict(row)
            for endpoint, row in (payload.get("bank") or {}).get("rows", {}).items()
        }

    # -- updates ----------------------------------------------------------

    def observe_scored_molecule(
        self,
        *,
        endpoint: str,
        score: float,
        graph: MolecularGraph | None = None,
        basin: str | None = None,
    ) -> None:
        """Record one counted oracle observation.

        ``graph`` is optional; when supplied its regions are entered in the
        donor memory.  The frontier is updated either way, because the frontier
        is what the run is graded on and every counted call moves it.

        ``basin`` is the caller's structural label for this molecule -- the same
        Bemis-Murcko basin the credit allocator groups by.  It is passed IN rather
        than computed here so this module still performs no chemistry, and so the
        bank's notion of "structurally distinct" is the controller's existing one
        rather than a second definition that could drift from it.  ``None`` banks the
        molecule but leaves it ineligible to REPRESENT a basin in the diverse
        stratum, and :meth:`ScoredMoleculeBank.report` counts those rows.
        """

        self.frontier.observe(endpoint, score)
        self.bank.observe(endpoint=endpoint, score=score, basin=basin)
        self.ordinal += 1
        if graph is None:
            return
        from compose_v4.control.bridge_region_law import bridge_separated_regions

        for region in bridge_separated_regions(graph, maximum=None):
            self.donors.observe(
                context=RegionContext.of(graph, region),
                endpoint=endpoint,
                score=score,
                ordinal=self.ordinal,
            )

    def observe_transition(
        self,
        *,
        parent_graph: MolecularGraph,
        parent_endpoint: str,
        parent_score: float | None,
        child_endpoint: str,
        child_score: float,
        child_heavy: int,
        family: str,
        touched_slots: tuple[int, ...],
    ) -> bool:
        """Record one counted transition and attribute it to a region context.

        The outcome learned is the **frontier contribution** of the child, with
        the parent-relative change folded in as the learning evidence the brief
        asks for -- then residualized against the size model so the corpus-wide
        "bigger scores better" gradient is not relearned as chemistry.

        Returns whether the transition could be attributed to a region.
        """

        # Banked FIRST, and deliberately above the attribution below: `_attribute`
        # returns None whenever no region overlaps the slots the edit touched, and the
        # lineage evidence is valid regardless of whether the edit can be localized to
        # a region. Recording it after the early return would silently drop the
        # promising stratum's evidence for exactly the transitions that are hardest to
        # attribute.
        self.bank.observe_lineage(
            parent_endpoint=parent_endpoint,
            parent_score=parent_score,
            child_score=child_score,
        )
        delta_heavy = int(child_heavy) - int(parent_graph.n_real_atoms)
        gain = self.frontier.frontier_gain(child_score, exclude=child_endpoint)
        relative = 0.0 if parent_score is None else float(child_score) - float(parent_score)
        outcome = gain + 0.5 * relative
        self.size.observe(delta_heavy, outcome)
        context = self._attribute(parent_graph, touched_slots)
        if context is None:
            return False
        self.edits.observe(
            context=context,
            family=family,
            delta_heavy=delta_heavy,
            residual=self.size.residual(delta_heavy, outcome),
            parent_endpoint=parent_endpoint,
            child_endpoint=child_endpoint,
            parent_score=parent_score,
            child_score=child_score,
        )
        return True

    @staticmethod
    def _attribute(
        graph: MolecularGraph, touched_slots: tuple[int, ...]
    ) -> RegionContext | None:
        """The region whose fragment best matches the slots the edit touched.

        Attribution is by overlap against the parent's own region support, so a
        transition is credited to a region the law could actually have drawn --
        not to an ad-hoc set of slots the law has no way to select.
        """

        from compose_v4.control.bridge_region_law import bridge_separated_regions

        touched = {int(s) for s in touched_slots}
        if not touched:
            return None
        best, best_score = None, 0.0
        for region in bridge_separated_regions(graph, maximum=None):
            fragment = set(region.fragment)
            union = fragment | touched
            overlap = len(fragment & touched) / len(union) if union else 0.0
            if overlap > best_score:
                best, best_score = region, overlap
        if best is None or best_score <= 0.0:
            return None
        return RegionContext.of(graph, best)

    # -- reads ------------------------------------------------------------

    def context_value(self, context: RegionContext, *, size: int | None = None) -> float:
        """Learned value of drawing this region context.

        Two terms, kept separate on purpose:

        ``scale``     what the run's counted data says an edit of this SIZE
                      does, from :class:`SizeResidual`.  This is the corpus-wide
                      size gradient.  It is a real, counted property of the
                      task and it belongs in a *proposal* distribution -- but it
                      is not chemistry, so it is reported on its own and never
                      folded into the structural claim.
        ``residual``  what this region CONTEXT does beyond a size-matched edit.
                      This is the only term that can support a claim that the
                      memory learned something structural.

        Without the scale term the law is size-blind, and since a molecule has
        more large regions than small ones, mass drifts to enormous excisions
        purely because nothing contradicts them.  Without the residual term
        there is no structural learning at all.  Both are needed, and
        :meth:`report` prints them separately so "it learned to make molecules
        bigger" can never be reported as "it learned chemistry".

        The donor association may only **modulate** a context the edit memory
        already has counted evidence for, bounded by ``donor_ceiling``.  It can
        never by itself promote an unseen context.  That gate is the
        architectural answer to a predictor-backed oracle whose top scorers are
        off-manifold: donor membership there is evidence of exploitation.
        """

        scale = 0.0 if size is None else self.size.expected(-int(size))
        effect, n = self.edits.value(context)
        if n == 0:
            return float(scale)
        association, _ = self.donors.association(context)
        bounded = max(-self.donor_ceiling, min(self.donor_ceiling, association))
        return float(scale + effect + bounded * abs(effect))

    def value_terms(self, context: RegionContext, size: int) -> dict[str, Any]:
        """The same value, decomposed, for the gate to report."""

        effect, n = self.edits.value(context)
        association, donor_n = self.donors.association(context)
        return {
            "scale": float(self.size.expected(-int(size))),
            "residual": float(effect),
            "edit_observations": int(n),
            "donor_association": float(association),
            "donor_observations": int(donor_n),
            "total": self.context_value(context, size=size),
        }

    @property
    def warm(self) -> bool:
        return bool(self.edits.counts)

    def region_law(self, *, maximum: int | None = None) -> MemoryRegionLaw | None:
        """The law to pass as ``region_law=``, or ``None`` while cold."""

        if not self.warm:
            return None
        return MemoryRegionLaw(
            maximum=maximum,
            margin=None,
            floor=SUPPORT_FLOOR,
            temperature=TEMPERATURE,
            memory=self,
            tilt=TILT,
        )

    # -- artifact ---------------------------------------------------------

    def allocation_priority(
        self, predicted_endpoint_value: float, *, parent_score: float | None = None
    ) -> dict[str, float]:
        """The frontier-aligned priority of a candidate, for the allocator.

        PMO grades a run on the mean of its top ten distinct scored molecules,
        so what a candidate is worth is what it would add to THAT -- not how
        far it moves its own parent.  ``frontier_gain`` is
        ``U10(A + {G'}) - U10(A)`` at the predicted endpoint value, which is
        why an edit taking a strong parent 0.80 -> 0.84 can outrank one taking
        a weak parent 0.05 -> 0.25: the second changes its parent more and the
        top ten not at all.

        ``parent_relative`` is returned beside it because it remains the right
        LEARNING EVIDENCE for what an edit does -- it is simply not the
        allocation objective.  The caller decides how to combine them; nothing
        here collapses the two.

        This does not replace the controller's credit allocator or its
        exploration floor.  It supplies the frontier term that allocator
        currently has no way to see.
        """

        value = float(predicted_endpoint_value)
        return {
            "predicted_endpoint_value": value,
            "frontier_gain": self.frontier.frontier_gain(value),
            "parent_relative": (
                0.0 if parent_score is None else value - float(parent_score)
            ),
            "u10_now": self.frontier.top_k_mean(),
        }

    def certification(self) -> dict[str, Any]:
        """The information-boundary statement recorded beside any result."""

        return {
            "schema": SCHEMA,
            "consumes": [
                "structure of states the controller shows it",
                "counted oracle observations from THIS run only",
            ],
            "excludes": [
                "oracle internals or model weights",
                "target or reference SMILES",
                "hidden component scores of a composite objective",
                "task identity, task->program lookup, or any per-task branch",
                "same-task winner molecules or winner identities from prior runs",
                "uncounted same-task history",
                (
                    "any molecular property computed off-ledger for selection; "
                    "this module computes none"
                ),
            ],
            "counted_observations": len(self.frontier.scores),
            "donor_claim": "association_only; provenance retained per row",
            "primary_evidence": "edit_outcome_memory",
            "donor_can_promote_unseen_context": False,
        }

    def report(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA,
            "counted_observations": len(self.frontier.scores),
            "u10": self.frontier.top_k_mean(),
            "edit_contexts": len(self.edits.counts),
            "edit_rows": len(self.edits.rows),
            "donor_contexts": len(self.donors.counts),
            "donor_rows": len(self.donors.rows),
            "size_model": self.size.report(),
            "scored_bank": self.bank.report(),
            "proposal_cost": dict(self.cost),
            "certification": self.certification(),
        }


# ---- Adapter -------------------------------------------------------------


def memory_channel_proposal(
    optimizer,
    channel: str,
    entry: dict[str, Any],
    fallback,
    memory: OnlineProposalMemory | None,
    *,
    shallow_channel: str = "shallow_program_channel",
):
    """Drop-in replacement for ``DynamicV21ProgramOptimizer._channel_proposal``.

    Delegates to ``fallback`` -- the unmodified production implementation --
    whenever the memory is absent or cold, or the lane is not the shallow one.
    That delegation is byte-identical: no law object is constructed, so the
    unlawed ``rng.permutation`` draw is preserved exactly.

    On the shallow lane with a warm memory it runs the same
    ``synthesize_dynamic_program`` the production path runs, with the learned
    law supplied through the existing ``region_law=`` keyword.  Nothing about
    the executor, the eligibility gate or the candidate record changes.
    """

    if memory is None or not memory.warm or channel != shallow_channel:
        return fallback(channel, entry)
    from compose_v4.control.dynamic_program_synthesis import synthesize_dynamic_program
    from compose_v4.rewrite.trace_shard import decode_state

    law = memory.region_law()
    if law is None:
        return fallback(channel, entry)
    memory.cost["syntheses"] += 1
    source = decode_state(entry["trace"]["states"][-1])
    _, program, binding, _, metadata = synthesize_dynamic_program(
        source,
        optimizer.shallow_rng,
        max_modules=3,
        max_primitives=optimizer.config.max_primitives,
        max_blocks=optimizer.config.max_blocks,
        region_law=law,
    )
    return (
        source,
        program,
        binding,
        {
            "dynamic_generic_composition": metadata,
            "proposal_kind": "mutation",
            "pmo_online_memory": {
                "schema": SCHEMA,
                "edit_contexts": len(memory.edits.counts),
                "counted_observations": len(memory.frontier.scores),
            },
        },
    )
