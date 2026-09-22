"""The generic zero-support fallback: what the controller does when it has nothing.

WHAT THIS IS
------------
A second, structurally different proposal procedure that the T4 controller runs
when its primary procedure has produced no usable candidate at all.  It is the
same shape as a numerical solver switching strategy when one method stalls: the
fallback is part of the declared algorithm, it is available to every cell, and
its parameters are fixed before the algorithm is run rather than chosen after a
result is seen.

THE TRIGGER, AND WHY IT IS THE TERMINAL ONE
-------------------------------------------
The shipped round loop is

    proposals -> merge -> candidates (endpoints not already in the archive)
      -> selected = select_batch(candidates, ..., batch = room >= 1)
      -> if not selected:      <-- TERMINAL candidate_exhaustion
      -> dock, update, and if budget <= 0:   TERMINAL complete_budget

``select_batch`` returns empty exactly when ``candidates`` is empty (``room`` is
at least one whenever a round starts, because a round only starts with budget
remaining), so "this round produced zero eligible candidates" and "this cell
terminates at ``candidate_exhaustion``" are THE SAME EVENT.  A round that yields
nothing does not merely fail, it ends the cell.

The fallback therefore fires on entry to that branch and nowhere else.  Two
consequences, and both are load-bearing:

* Every round that wrote a lock had at least one eligible candidate, so it never
  entered the branch, so on a row that reached its budget the fallback is
  unreachable.  The primary trajectory of such a row is unchanged, and the proof
  is structural rather than statistical -- no RNG is drawn and no state is
  mutated before the check.
* A per-round trigger on some weaker condition (say, "fewer than n candidates")
  WOULD reach inside successful cells and would owe a full re-run.  That is why
  the terminal condition, not a threshold, is what this module implements.

THE MECHANISM
-------------
Every clean eligible witness measured on the exhausted T4 delta=0.6 cells is a
net EXCISION of roughly 7 to 15 heavy atoms, and the primary lane's region draw
is capped at 8 atoms.  The fallback is the general form of that move class:

    for each bridge-separated region of the parent, ordered by the task's own
    declared free-gate margin, realise the excision through the production
    executor and attach K independent replacement completions to the retained
    anchor; gate every endpoint with the unmodified production gate.

The completion of length zero -- the degenerate member of the family, i.e. the
pure excision -- is enumerated exactly once per region, so the fallback's
support provably contains the plain excision of every region it considers.  The
other K-1 completions are drawn from the SAME distribution the production
``segment_replace`` module uses: a length uniform on 1..min(8, capacity) and a
chain over ``("C", "N", "O")``, with the anchor pinned to the retained atom.
No new chemistry constant is introduced; both are imported from the production
module.

WHY IT CAN SUCCEED WHERE THE PRIMARY LANE CANNOT
------------------------------------------------
Two independent reasons, both measured elsewhere in this repository:

* the primary shallow lane cannot express a coherent 9-15 atom excision in one
  draw, and expressing it as two or three independent bounded cuts makes its
  proposal mass a product of per-module terms;
* ``t4_fiber_campaign.expand`` does not gate the molecule its program builds --
  it abstracts the program to a structural goal, re-binds it and gates whatever
  ``instantiate_goal`` returns, so the program's own endpoint reaches the gate
  with measured frequency 0.  The fallback gates the executed endpoint directly
  and never enters that layer.

WHAT THE FALLBACK IS ALLOWED TO KNOW
-------------------------------------
The parent state, the task's declared similarity reference and delta, and the
task's declared free thresholds.  It reads no cell identifier, no protein name,
no seed index, no docking score and no witness molecule.  Asked "why does this
fire here", the answer is "the primary procedure returned nothing"; asked "why
this region", the answer is "this parent has bridge-separated substituents and
their children differ in declared-gate margin".
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from compose_v4.chem.molecular_graph import (
    MolecularGraph,
    molecular_graph_to_smiles,
    smiles_to_molecular_graph,
)
from compose_v4.chem.state import pad_molecular_graph
from compose_v4.control.bridge_region_law import (
    BridgeRegion,
    FreeFeasibilityGate,
    free_gate_margin_law,
)
from compose_v4.control.dynamic_program_synthesis import (
    MAX_SEGMENT_LENGTH,
    _delete_pendant_fragment,
    _grow_actions,
)
from compose_v4.experiments.whole_ring_plan import execute_program

SCHEMA_VERSION = "t4_zero_support_fallback_v1"

#: Slot capacity of a T4 proposal state. This is the SLOT COUNT of the padded
#: array, not the heavy-atom ceiling: `t4_fiber_campaign` builds every parent as
#: `pad_molecular_graph(..., 48)` and separately refuses any ENDPOINT above 40
#: heavy atoms at the gate. The two numbers are different quantities.
PROPOSAL_SLOTS = 48

#: Attempt budget per parent, work-matched to the primary lane's own per-parent
#: draw count (`proposal.shallow.draws` is 480 in every T4 contract).
ATTEMPTS_PER_PARENT = 480
#: Regions considered per parent, in free-gate-margin order. Measured over all
#: fifteen T4 seeds the full bridge-region support is at most 42, so this cap
#: does not bind on the panel; it exists so a pathological parent cannot blow
#: the attempt budget, and the law decides which regions survive truncation.
MAX_REGIONS_PER_PARENT = 60
#: Completions attached to each region: one empty completion (the pure excision)
#: plus this many minus one drawn completions. 60 * 8 == 480.
COMPLETIONS_PER_REGION = 8


@dataclass(frozen=True)
class _FixedRegionLaw:
    """A one-region law, so the PRODUCTION excision path does the excision.

    ``_delete_pendant_fragment`` consults its ``law`` through ``order`` alone.
    Handing it a law that offers exactly one region makes it excise THAT region
    with its own deletion schedule, its own executor calls and its own
    acceptance rule -- nothing about the excision is transcribed here, which is
    what lets a drift in the executor fail this module rather than pass it.
    """

    region: BridgeRegion

    def order(self, graph: MolecularGraph, rng) -> list[BridgeRegion]:
        return [self.region]


@dataclass
class FallbackWork:
    """Load-independent work counters. Seconds are not a cost metric here."""

    regions_considered: int = 0
    regions_excised: int = 0
    attempts: int = 0
    completions_executed: int = 0
    endpoints_gated: int = 0
    distinct_valid_endpoints: int = 0
    distinct_eligible_endpoints: int = 0

    def as_dict(self) -> dict:
        return {
            "regions_considered": self.regions_considered,
            "regions_excised": self.regions_excised,
            "attempts": self.attempts,
            "completions_executed": self.completions_executed,
            "endpoints_gated": self.endpoints_gated,
            "distinct_valid_endpoints": self.distinct_valid_endpoints,
            "distinct_eligible_endpoints": self.distinct_eligible_endpoints,
        }


def fallback_seed(
    controller_seed: int, *, round_index: int, parent_index: int, replicate: int = 0
) -> int:
    """The fallback's proposal seed, derived exactly as the primary lane derives its own.

    The primary controller uses
    ``controller_seed + 1_000_003*round + 10_007*parent + 101*expert``; the
    fallback uses the same shape with a fixed offset so a fallback draw can
    never collide with a primary draw in the same round.
    """

    return int(
        controller_seed
        + 900_007
        + 1_000_003 * int(round_index)
        + 10_007 * int(parent_index)
        + 101 * int(replicate)
    )


def proposal_state(smiles: str) -> MolecularGraph:
    """The parent as the T4 proposal path builds it: 48 slots, never re-parsed later."""

    return pad_molecular_graph(smiles_to_molecular_graph(smiles), PROPOSAL_SLOTS)


def _completion_lengths(rng, *, capacity: int, count: int) -> list[int]:
    """One empty completion, then ``count - 1`` drawn as ``segment_replace`` draws.

    ``segment_replace`` uses ``rng.integers(1, capacity + 1)`` with
    ``capacity = min(MAX_SEGMENT_LENGTH, 40 - n_real_atoms)``; that expression is
    reproduced here against the CONTRACTED state, exactly as the production
    module computes it after its own deletion.
    """

    lengths = [0]
    if capacity >= 1:
        lengths.extend(int(rng.integers(1, capacity + 1)) for _ in range(count - 1))
    return lengths


def fallback_candidates(
    parent_smiles: str,
    rng,
    *,
    check: Callable[[str], dict | None],
    reference_smiles: str,
    delta: float,
    max_regions: int = MAX_REGIONS_PER_PARENT,
    completions_per_region: int = COMPLETIONS_PER_REGION,
    attempt_budget: int = ATTEMPTS_PER_PARENT,
) -> tuple[list[dict], FallbackWork]:
    """Eligible endpoints from one parent under the zero-support fallback.

    ``check`` is the caller's eligibility kernel and is expected to be the
    UNMODIFIED production ``t4_fiber_campaign.Fiber.check``; it is injected so
    this control-layer module does not import an experiment module, and the
    tests assert the injected gate is that one.
    """

    if max_regions < 1 or completions_per_region < 1 or attempt_budget < 1:
        raise ValueError("fallback budget parameters must be positive")
    work = FallbackWork()
    source = proposal_state(parent_smiles)
    law = free_gate_margin_law(
        FreeFeasibilityGate(delta=float(delta)), reference_smiles, maximum=None
    )
    ordered = law.order(source, rng)[:max_regions]
    work.regions_considered = len(ordered)

    found: dict[str, dict] = {}
    valid: set[str] = set()
    for region in ordered:
        if work.attempts >= attempt_budget:
            break
        try:
            delete_actions, contracted, anchor, path = _delete_pendant_fragment(
                source, rng, law=_FixedRegionLaw(region)
            )
        except ValueError:
            # The production excision refused this region. Counted, not hidden.
            continue
        work.regions_excised += 1
        capacity = min(MAX_SEGMENT_LENGTH, 40 - contracted.n_real_atoms)
        lengths = _completion_lengths(
            rng, capacity=capacity, count=completions_per_region
        )
        for length in lengths:
            if work.attempts >= attempt_budget:
                break
            work.attempts += 1
            if length == 0:
                product, elements = contracted, []
            else:
                try:
                    grow_actions, _, elements = _grow_actions(
                        contracted, rng, length=length, elements=("C", "N", "O"),
                        anchor=anchor,
                    )
                    product, _ = execute_program(
                        source, [*delete_actions, *grow_actions]
                    )
                except (ValueError, KeyError, IndexError):
                    continue
            work.completions_executed += 1
            endpoint = molecular_graph_to_smiles(product)
            if endpoint is None:
                continue
            valid.add(endpoint)
            work.endpoints_gated += 1
            gate = check(endpoint)
            if gate is None or gate["smiles"] in found:
                continue
            found[gate["smiles"]] = {
                **gate,
                "parent": parent_smiles,
                "proposal_lane": "zero_support_fallback",
                "region_size": int(region.size),
                "retained_anchor": int(anchor),
                "deleted_atoms": len(path),
                "inserted_atoms": int(length),
                "inserted_elements": list(elements),
                "heavy_delta": int(gate["heavy"]) - int(source.n_real_atoms),
                "delta": float(delta),
            }
    work.distinct_valid_endpoints = len(valid)
    work.distinct_eligible_endpoints = len(found)
    return list(found.values()), work
