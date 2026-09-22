"""One T4 controller: every proposal kernel wired from time zero, routed by STATE.

THE CLAIM THIS MODULE EXISTS TO MAKE HONEST
-------------------------------------------
The published T4 panel used a different support operator on different cells, chosen
by a human after watching a cell fail.  That is legitimate development history and it
is not one algorithm, so it forces a disclosure sentence.  This module removes the
need for that sentence by making the choice automatic:

    q(a | G) = w_local(G, S) q_local + w_region(G, S) q_region + w_state(G, S) q_state

The weights are a function of the MOLECULAR state ``G`` and the SEARCH state ``S``.
No target name, no protein, no cell id, no seed index, no docking score, no reward, no
archive value can reach them -- and that is enforced STRUCTURALLY rather than by a
naming convention: ``kernel_weights`` accepts two frozen dataclasses whose fields are
enumerated here, and neither can carry an identity or a score.  Adding such a field is
the only way to smuggle one in, and ``tests/test_t4_unified_routing.py`` fails if the
field sets move.

THE TWO MECHANISMS, AND WHY THEY ARE COMPLEMENTARY RATHER THAN ARBITRARY
------------------------------------------------------------------------
MEASURED on all fifteen T4 starting states, zero oracle calls, under the pinned kernel
(rdkit 2024.3.5), by running the PRODUCTION excision path over every bridge-separated
region of each source:

    cell      regions  executable  refused  refused-by-charge-policy
    parp1_*     6-10      all          0            0
    braf_*     32-42      all          0            0
    fa7_*      26-38      all          0            0
    jak2_*      8-10      all          0            0
    5ht1b_0       16        7          9            9
    5ht1b_1       10        5          5            5
    5ht1b_2       16        7          9            9

The executor's charge policy refuses a region excision on exactly the states that
carry a formal charge, and on no others; every refusal is attributable to that policy.
So the region kernel LOSES SUPPORT precisely where a charge/protonation-aware kernel
is needed.  The two kernels are not two labels for the same job -- one is degraded
exactly where the other becomes necessary, and that is a property of the molecule and
the executor, measurable without naming a protein.

WHAT MOLECULAR STATE ALONE CANNOT DO, STATED PLAINLY
-----------------------------------------------------
It cannot predict WHICH cells are support-limited.  Heavy-atom count does not separate
them: ``braf_2`` at 37 heavy atoms searches while ``fa7_2`` at 35 does not, and
``5ht1b_0`` at 39 searches while ``braf_1`` at 39 does not.  That is why the ramp term
is a SEARCH-state quantity -- the measured emptiness of the candidate pool -- and not a
molecular one.  A rule that claimed to predict support limitation from the source
molecule would be fitting the panel.

CONSEQUENCE FOR THE NON-TRIGGER ROWS
------------------------------------
``support_ramp`` is exactly zero while the primary pool is non-empty, so on any round
that produced an eligible candidate the weights are ``(1, 0, 0)`` and the controller is
byte-identical to the primary lane.  A cell that never empties its pool therefore runs
the historical primary trajectory unchanged, and the proof is structural: no alternate
kernel is consulted and no RNG is drawn for one.

A CONFOUND THIS PANEL CANNOT RESOLVE, AND THE CONTROL THAT DOES
----------------------------------------------------------------
On these fifteen leads, "carries a formal charge" is perfectly confounded with "is a
5HT1B lead" -- all three 5HT1B sources are cations and no other source is charged.  The
panel alone therefore cannot distinguish routing on charge from routing on identity.
Two things resolve it and both are required: the structural guarantee above (identity
cannot reach the function), and an OUT-OF-PANEL control over task-independent leads
showing the weights track charge rather than provenance.  See
``scripts/t4_blind_routing_gate.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, fields

from compose_v4.chem.molecular_graph import MolecularGraph, is_element
from compose_v4.control.bridge_region_law import bridge_separated_regions, excise_region
from compose_v4.control.dynamic_program_synthesis import MAX_SEGMENT_LENGTH
from compose_v4.data.charge_policy import charge_policy_preserved
from compose_v4.rewrite.operators import enumerate_atom_protonation_restates

SCHEMA_VERSION = "t4_unified_routing_v1"

#: The proposal kernels, all available from time zero. Names match the committed
#: blind-routing evidence (`diagnostics/t4_state_routing_v1/blind_routing_v1.json`).
KERNELS = ("local", "region", "state_aware")

#: The heavy-atom ceiling the fiber applies to ENDPOINTS. Distinct from the 48 SLOT
#: capacity a T4 proposal source is padded to; both hold at once.
REPRESENTABLE_HEAVY_ATOMS = 40
#: Slot capacity of a T4 proposal state.
PROPOSAL_SLOTS = 48
#: The largest region the LOCAL kernel's own draw can express in one module.
LOCAL_REGION_CAP = MAX_SEGMENT_LENGTH


@dataclass(frozen=True)
class MolecularApplicability:
    """What the parent molecule makes possible. Chemistry only.

    Every field is a count or a capacity derived from the molecular graph and the
    production executor. There is deliberately no name, no identifier, no score and
    no provenance here: the routing cannot read what it is not given.
    """

    heavy_atoms: int
    growth_headroom: int
    net_formal_charge: int
    charged_centres: int
    total_regions: int
    executable_regions: int
    charge_refused_regions: int
    large_regions: int
    protonation_sites: int

    def as_record(self) -> dict:
        return {field.name: getattr(self, field.name) for field in fields(self)}


@dataclass(frozen=True)
class SearchProgress:
    """What the search has observed so far. Counts only, never values.

    ``eligible_pool_size`` is a COUNT of admitted candidates, not their scores, and
    no docking value, incumbent or improvement appears anywhere in this record. That
    is what makes the ramp reward-free.
    """

    eligible_pool_size: int
    consecutive_empty_rounds: int
    rounds_completed: int

    def as_record(self) -> dict:
        return {field.name: getattr(self, field.name) for field in fields(self)}


def molecular_applicability(source: MolecularGraph) -> MolecularApplicability:
    """Measure a parent's kernel applicability with the PRODUCTION excision path.

    Regions come from ``bridge_separated_regions`` and each candidate child is judged
    by ``charge_policy_preserved`` -- the same predicate ``whole_ring_plan`` applies at
    every step of a real program -- so a drift in either fails this function rather
    than passing it.

    The source must carry the T4 proposal slot capacity. A tight graph silently
    deletes the whole insertion family from the legal support, which would change
    every number here, so it is refused rather than accepted quietly.
    """

    slots = int(source.atom_types.shape[0])
    if slots != PROPOSAL_SLOTS:
        raise ValueError(
            f"a T4 proposal source carries {PROPOSAL_SLOTS} slots, got {slots}; a tight "
            "graph deletes the insertion family from the legal support"
        )
    heavy = int(source.n_real_atoms)
    charges = source.formal_charges[is_element(source.atom_types)]
    try:
        sites = len(list(enumerate_atom_protonation_restates(source)))
    except TypeError:  # pragma: no cover - signature variance across revisions
        sites = len(list(enumerate_atom_protonation_restates(source, None)))
    regions = bridge_separated_regions(source, maximum=None)
    executable = 0
    charge_refused = 0
    for region in regions:
        try:
            child = excise_region(source, region)
        except (ValueError, KeyError, IndexError):
            # An unrealizable region is counted as neither executable nor
            # charge-refused; it is excluded rather than silently attributed.
            continue
        if charge_policy_preserved(source, child):
            executable += 1
        else:
            charge_refused += 1
    return MolecularApplicability(
        heavy_atoms=heavy,
        growth_headroom=max(0, REPRESENTABLE_HEAVY_ATOMS - heavy),
        net_formal_charge=int(charges.sum()),
        charged_centres=int((charges != 0).sum()),
        total_regions=len(regions),
        executable_regions=executable,
        charge_refused_regions=charge_refused,
        large_regions=sum(1 for region in regions if region.size > LOCAL_REGION_CAP),
        protonation_sites=sites,
    )


def support_ramp(search: SearchProgress) -> float:
    """How far the primary lane has run out of support, in [0, 1].

    Exactly 0 while the pool is non-empty, which is what makes a healthy round
    byte-identical to the primary lane. It reaches 1 on the first empty pool, because
    an empty pool IS the terminal condition in the shipped loop -- ``select_batch``
    returns empty exactly when ``candidates`` is empty -- so there is no later round
    in which to ramp further.
    """

    if int(search.eligible_pool_size) > 0:
        return 0.0
    return 1.0


def applicability_mask(applicability: MolecularApplicability) -> dict[str, int]:
    """Which kernels this molecule's chemistry PERMITS. Zero elsewhere.

    This is a mask, not a classifier. The controller never decides semantically that
    "this cell has a protonation bottleneck" -- it only asks which kernels the state
    chemistry admits, and lets support exhaustion decide whether any of them is
    needed. That distinction is what makes the two healthy charged cells
    (``5ht1b_0``, ``5ht1b_1``) consistent with the rule rather than counterexamples
    to it: the state-aware kernel is APPLICABLE on all three 5HT1B leads and
    ACTIVATES only where support runs out.

    ``local`` is always permitted. A molecule carrying formal charge admits
    ``state_aware`` edits; a neutral one is the generic structural case the region
    kernel covers.
    """

    charged = int(applicability.net_formal_charge) != 0
    return {"local": 1, "region": int(not charged), "state_aware": int(charged)}


def kernel_weights(
    applicability: MolecularApplicability, search: SearchProgress
) -> dict[str, float]:
    """Proposal-compute weights over ``KERNELS``, from molecular and search state alone.

    Neither argument can carry a target identity or a reward, so the routing provably
    reads neither. While the pool is non-empty the ramp is zero and the weights are
    ``(1, 0, 0)`` -- byte-identical to the primary lane, with no alternate kernel
    consulted and no RNG drawn for one.
    """

    ramp = support_ramp(search)
    mask = applicability_mask(applicability)
    return {
        "local": 1.0,
        "region": ramp * mask["region"],
        "state_aware": ramp * mask["state_aware"],
    }


def activated_kernel(
    applicability: MolecularApplicability, search: SearchProgress
) -> str | None:
    """The non-local kernel this state allocates proposal compute to, or ``None``.

    ``None`` while the primary pool is non-empty: that is the whole of the
    non-trigger guarantee, and it is structural rather than statistical.
    """

    weights = kernel_weights(applicability, search)
    fired = [
        kernel for kernel in ("region", "state_aware") if weights.get(kernel, 0.0) > 0.0
    ]
    if not fired:
        return None
    if len(fired) > 1:  # pragma: no cover - the mask is exclusive by construction
        raise ValueError(f"the applicability mask is not exclusive: {fired}")
    return fired[0]


def applicable_kernel_if_exhausted(applicability: MolecularApplicability) -> str:
    """Which kernel WOULD activate on an empty pool. The blind-routing table's column.

    Defined for every state, exhausted or not, so the routing can be audited on cells
    that never trigger -- which is where a rule that secretly keyed on target identity
    would be caught.
    """

    kernel = activated_kernel(
        applicability, SearchProgress(eligible_pool_size=0, consecutive_empty_rounds=1, rounds_completed=0)
    )
    if kernel is None:  # pragma: no cover - the mask always permits exactly one
        raise ValueError("no kernel is applicable on an empty pool")
    return kernel


def charge_refused_region_share(applicability: MolecularApplicability) -> float:
    """The GRADED signal, recorded and deliberately NOT consulted by the mask.

    MEASURED across the fifteen T4 sources: the executor's charge policy refuses
    9 of 16 pendant regions on ``5ht1b_2``, 9 of 16 on ``5ht1b_0``, 5 of 10 on
    ``5ht1b_1`` and ZERO on every neutral cell. It is a continuous mechanistic
    statement of the same fact the binary flag encodes -- the region kernel loses
    support exactly where a charge-aware kernel becomes necessary -- and it is the
    declared fallback if the binary proves brittle beyond this panel.

    It is computed and reported so a future decision has the number, and it is not
    wired into ``applicability_mask``, because swapping the decision rule after
    seeing a result is the thing this whole design exists to prevent.
    """

    total = int(applicability.total_regions)
    if total <= 0:
        return 0.0
    return applicability.charge_refused_regions / total


def routing_decision(
    applicability: MolecularApplicability, search: SearchProgress
) -> dict:
    """The full record a round lock should carry, so a routing choice is auditable."""

    weights = kernel_weights(applicability, search)
    return {
        "schema_version": SCHEMA_VERSION,
        "applicability": applicability.as_record(),
        "search": search.as_record(),
        "support_ramp": support_ramp(search),
        "applicability_mask": applicability_mask(applicability),
        "weights": weights,
        "activated_kernel": activated_kernel(applicability, search),
        "applicable_kernel_if_exhausted": applicable_kernel_if_exhausted(applicability),
        "charge_refused_region_share_diagnostic_only": charge_refused_region_share(
            applicability
        ),
    }


def graded_kernel_weights(
    applicability: MolecularApplicability, search: SearchProgress
) -> dict[str, float]:
    """The PORTFOLIO allocation: proposal compute split by MEASURED support share.

    This is the declared graded alternative to the binary mask, computed alongside it
    and NOT shipped as the decision. It allocates the two alternate kernels in
    proportion to the share of this molecule's bridge-separated regions the executor
    accepts and the share it refuses under the charge policy -- the same measured
    quantity, partitioned, rather than a flag.

    WHY IT IS WORTH CARRYING. The binary mask keys on net formal charge, and on this
    fifteen-lead panel that is perfectly confounded with "is a 5HT1B lead". The graded
    form keys on a mechanism instead: the region kernel measurably LOSES support where
    the charge policy refuses its excisions, and that is a continuous quantity with a
    causal story. ``kernel_allocation_agreement`` records whether the two agree.
    """

    ramp = support_ramp(search)
    weights = {"local": 1.0, "region": 0.0, "state_aware": 0.0}
    total = int(applicability.total_regions)
    if ramp <= 0.0 or total <= 0:
        return weights
    weights["region"] = ramp * applicability.executable_regions / total
    weights["state_aware"] = ramp * applicability.charge_refused_regions / total
    return weights


def graded_dominant_kernel(applicability: MolecularApplicability) -> str | None:
    """Which alternate kernel the graded portfolio would favour on an empty pool."""

    weights = graded_kernel_weights(
        applicability,
        SearchProgress(eligible_pool_size=0, consecutive_empty_rounds=1, rounds_completed=0),
    )
    alternates = {k: weights[k] for k in ("region", "state_aware")}
    if max(alternates.values()) <= 0.0:
        return None
    return max(alternates, key=alternates.get)


def kernel_allocation_agreement(applicability: MolecularApplicability) -> dict:
    """Does the shipped binary mask agree with the graded portfolio on this state?

    Recorded so a future decision to swap rules has the evidence, and so a state
    where they diverge is visible rather than discovered later.
    """

    binary = applicable_kernel_if_exhausted(applicability)
    graded = graded_dominant_kernel(applicability)
    return {
        "binary": binary,
        "graded": graded,
        "agree": binary == graded,
        "graded_weights": graded_kernel_weights(
            applicability,
            SearchProgress(eligible_pool_size=0, consecutive_empty_rounds=1, rounds_completed=0),
        ),
        "protonation_sites": int(applicability.protonation_sites),
        "protonation_precondition_matches_binary": (
            (applicability.protonation_sites > 0) == (binary == "state_aware")
        ),
    }
