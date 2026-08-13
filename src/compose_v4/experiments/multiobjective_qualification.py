"""Common resource vocabulary and panel-admission rules for the multiobjective package.

This module exists because three separate lanes each froze a three-counter
accounting scheme, using three different sets of names for overlapping concepts,
and a main-paper Pareto table has to put them in one row-space without silently
equating things that are not equal.

THE THREE EXISTING VOCABULARIES
--------------------------------
1. `CostLedger` in `pareto_control.py` (Lane 4, branch `codex/compose-pareto-control`)::

       native_oracle_calls   distinct molecules scored
       raw_oracle_calls      every scoring invocation, cache hits included
       kernel_calls          successor enumerations

2. `pareto_oracle_semantics.py` (Lane 4, post-hoc semantic correction)::

       raw_instrument_oracle_requests   every request the PROGRAM made
       algorithmic_oracle_requests      what the METHOD consumed to decide
       benchmark_eval_requests          post-generation scoring; ours, not the
                                        method's.  The committed JSON emits this
                                        under the key `harness_only_requests`.

3. `OracleCounts` in `oracle_accounting.py` (Workstream D, branch
   `codex/compose-baseline-qualification`)::

       oracle_requests
       unique_valid_canonical_evaluations
       evaluator_calls
       duplicate_requests / cache_hits / failed_proposals

They do not line up one-to-one, and two of the mismatches decide comparisons:

* Lane 4's `raw_oracle_calls` and Workstream D's `oracle_requests` are the same
  concept (algorithmic demand, duplicates included) under different names.
* Lane 4's `native_oracle_calls` and Workstream D's
  `unique_valid_canonical_evaluations` are *nearly* the same concept but not
  identical: the Workstream D counter excludes proposals RDKit cannot parse,
  and Lane 4's process cannot emit an unparseable successor, so the two agree
  only because COMPOSE's support is legal by construction.  That agreement is a
  property of COMPOSE, not of the counter, and it fails for any external method
  that can propose an invalid molecule.

THE AXIS THAT DID NOT EXIST AND HAS TO
---------------------------------------
None of the three vocabularies has a name for **surrogate calls** — objective
information a method obtains from a learned model trained on oracle labels,
rather than from the oracle.

That omission is not cosmetic.  HN-GFN's GFlowNet is trained against a proxy and
the true oracle is consulted only on the batch selected in each outer round
(`main_mobo.py`: `num_init_examples=200`, `num_outer_loop_iters=8`,
`num_samples=100`).  COMPOSE consults the frozen evaluator directly at every
decision.  Comparing the two on "oracle calls" alone reports the difference
between *paying the oracle* and *amortizing the oracle*, while appearing to
report a difference in efficiency.

So `surrogate_calls` is a first-class axis here, and a comparison whose methods
disagree on whether they have one is returned as INCOMPARABLE rather than as a
number.  This is the project rule "every efficiency claim must name its resource
axis" made mechanical: the axis cannot be omitted, because the comparison
function refuses to run without it.

WHAT THIS MODULE DELIBERATELY DOES NOT DO
------------------------------------------
It does not compute hypervolume, it does not rank methods, and it does not
choose a budget.  It maps counters onto a shared vocabulary, decides which
fairness panel a method may enter, and refuses the comparisons that the frozen
protocol says are inadmissible.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal


class ResourceAxisError(ValueError):
    """A counter name has no declared meaning in the common vocabulary."""


class PanelAdmissionError(ValueError):
    """A method was offered to a panel whose semantics it does not satisfy."""


class FrozenScaleError(ValueError):
    """A hypervolume reference was supplied from something other than the freeze."""


# ---------------------------------------------------------------------------
# The common resource vocabulary
# ---------------------------------------------------------------------------

#: Canonical axis name -> what it counts. Every axis is reported separately and
#: none is ever summed into another. The ordering is the reporting order.
RESOURCE_AXES: dict[str, str] = {
    "algorithmic_oracle_requests": (
        "objective information the METHOD consumed to make decisions, "
        "duplicates included. The oracle-demand axis the paper uses."
    ),
    "unique_oracle_evaluations": (
        "distinct valid canonical molecules the true objective was evaluated "
        "on. The benchmark-native convention and the only PMO-comparable count."
    ),
    "evaluator_calls": (
        "expensive evaluator executions actually performed after caching. "
        "Real work, not algorithmic demand."
    ),
    "surrogate_calls": (
        "objective information obtained from a LEARNED model standing in for "
        "the oracle. Zero for COMPOSE. Large for any Bayesian-optimization "
        "method. Never added to any oracle axis."
    ),
    "benchmark_eval_requests": (
        "evaluations performed only AFTER generation, to score and report the "
        "produced molecules. Ours, not the method's. Never added back into "
        "the algorithmic counter."
    ),
    "kernel_calls": (
        "generative / successor-enumeration calls. COMPOSE-internal; external "
        "methods have no shared notion of one."
    ),
    "completed_trajectories": (
        "complete molecular trajectories produced. COMPOSE-internal; HN-GFN "
        "and de novo generators do not share this object."
    ),
    "wall_core_seconds": (
        "wall time multiplied by cores. SECONDARY ONLY -- never the primary "
        "axis of an efficiency claim, and never compared across hardware."
    ),
}

#: Axes that are COMPOSE-internal: they exist only inside our process and may
#: never carry an external cross-method claim.
COMPOSE_INTERNAL_AXES: frozenset[str] = frozenset(
    {"kernel_calls", "completed_trajectories"}
)

#: Axes that may never be the primary axis of a claim.
SECONDARY_ONLY_AXES: frozenset[str] = frozenset({"wall_core_seconds"})

#: Local counter name -> canonical axis, per source vocabulary. A name absent
#: here raises rather than being dropped: a silently ignored counter is how an
#: efficiency claim loses its axis.
AXIS_ALIASES: dict[str, dict[str, str]] = {
    # Lane 4, src/compose_v4/experiments/pareto_control.py :: CostLedger
    "pareto_control.CostLedger": {
        "raw_oracle_calls": "algorithmic_oracle_requests",
        "native_oracle_calls": "unique_oracle_evaluations",
        "kernel_calls": "kernel_calls",
    },
    # Lane 4, src/compose_v4/experiments/pareto_oracle_semantics.py, and the
    # committed diagnostics/pareto_semantic_oracle_accounting.json. Note the
    # JSON key is `harness_only_requests` while the module prose calls the same
    # concept `benchmark_eval_requests`; both are accepted, deliberately.
    "pareto_oracle_semantics": {
        "algorithmic_oracle_requests": "algorithmic_oracle_requests",
        "raw_instrument_oracle_requests": "_raw_instrument",
        "harness_only_requests": "benchmark_eval_requests",
        "benchmark_eval_requests": "benchmark_eval_requests",
        "native_oracle_calls": "unique_oracle_evaluations",
        "kernel_calls": "kernel_calls",
    },
    # Workstream D, src/compose_v4/experiments/oracle_accounting.py
    "oracle_accounting.OracleCounts": {
        "oracle_requests": "algorithmic_oracle_requests",
        "unique_valid_canonical_evaluations": "unique_oracle_evaluations",
        "evaluator_calls": "evaluator_calls",
        "duplicate_requests": "_bookkeeping",
        "cache_hits": "_bookkeeping",
        "failed_proposals": "_bookkeeping",
    },
}

#: `raw_instrument_oracle_requests` is a faithful execution record, not a
#: resource axis: it is the object a serial-vs-parallel parity replay must match
#: exactly. It is carried through under its own name and never mapped onto an
#: axis, because doing so would attribute our harness overhead to the method.
PASSTHROUGH_AXES: frozenset[str] = frozenset({"_raw_instrument", "_bookkeeping"})


def canonical_axis(name: str, vocabulary: str) -> str:
    """Map a lane-local counter name onto the common axis vocabulary."""
    try:
        table = AXIS_ALIASES[vocabulary]
    except KeyError as exc:
        raise ResourceAxisError(
            f"unknown counter vocabulary {vocabulary!r}; "
            f"known: {sorted(AXIS_ALIASES)}"
        ) from exc
    try:
        return table[name]
    except KeyError as exc:
        raise ResourceAxisError(
            f"counter {name!r} has no declared meaning in vocabulary "
            f"{vocabulary!r}. Declare it before reporting it -- a counter "
            f"dropped in translation is an efficiency claim without an axis."
        ) from exc


def reconcile_ledger(ledger: dict[str, Any], vocabulary: str) -> dict[str, Any]:
    """Translate one lane's ledger into the common axis vocabulary.

    Returns a dict keyed by canonical axis name, plus a `passthrough` sub-dict
    holding the counters that are records rather than axes. Unknown counters
    raise; they are never silently discarded.
    """
    axes: dict[str, Any] = {}
    passthrough: dict[str, Any] = {}
    for name, value in ledger.items():
        target = canonical_axis(name, vocabulary)
        if target in PASSTHROUGH_AXES:
            passthrough[name] = value
            continue
        if target in axes:
            raise ResourceAxisError(
                f"two counters in {vocabulary!r} both map to axis {target!r}; "
                f"the mapping is ambiguous and must be resolved, not summed"
            )
        axes[target] = value
    axes["passthrough"] = passthrough
    return axes


# ---------------------------------------------------------------------------
# Source-conditioning semantics and panel admission
# ---------------------------------------------------------------------------

Support = Literal["YES", "NO", "UNVERIFIED"]

#: Panel A is the COMPOSE-native source-conditioned control experiment: every
#: arm starts from the SAME supplied molecule under the SAME edit budget.
PANEL_A = "PANEL_A_SOURCE_CONDITIONED"
#: Panel B is ordinary global multiobjective competence: all methods share the
#: same global task freedom and the same objective budget.
PANEL_B = "PANEL_B_GLOBAL_COMPETENCE"
#: Neither: the method's native object is not the one under study.
PANEL_NONE = "NOT_ADMISSIBLE"


@dataclass(frozen=True)
class SourceConditioningRecord:
    """What a method natively supports, with the citation that settles each cell.

    `evidence` must name a paper section, or a file path with a line range, or a
    commit. A cell without evidence is `UNVERIFIED` by construction: the project
    rule is that a cited-but-absent source is worse than a wrong number.
    """

    method: str
    #: Can the method begin from an exact molecule we supply?
    supplied_source: Support
    #: Can it be held to a fixed edit / trust-region budget from that molecule?
    edit_budget: Support
    #: Can it produce output conditioned on a requested preference vector?
    preference_conditioned: Support
    #: citation -> what it establishes
    evidence: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for axis in ("supplied_source", "edit_budget", "preference_conditioned"):
            value = getattr(self, axis)
            if value not in ("YES", "NO", "UNVERIFIED"):
                raise PanelAdmissionError(
                    f"{self.method}.{axis} = {value!r}; must be YES, NO or UNVERIFIED"
                )
            if value != "UNVERIFIED" and axis not in self.evidence:
                raise PanelAdmissionError(
                    f"{self.method}.{axis} asserts {value!r} with no evidence. "
                    f"Supply a citation or record it as UNVERIFIED."
                )


def panel_admission(record: SourceConditioningRecord) -> str:
    """Which fairness panel this method may enter.

    Panel A requires ALL THREE of supplied source, edit budget and preference
    conditioning to be natively supported. A method missing any of them has a
    different reachable set from COMPOSE's, and placing its front beside
    COMPOSE's per-source fronts would grant it a larger reachable set while
    calling the comparison source-matched.

    A method that is preference-conditioned but global goes to Panel B. A method
    that is neither is not admissible as a numerical baseline at all.
    """
    triple = (
        record.supplied_source,
        record.edit_budget,
        record.preference_conditioned,
    )
    if triple == ("YES", "YES", "YES"):
        return PANEL_A
    if "UNVERIFIED" in triple:
        # Unresolved is not the same as absent. It stays out of Panel A and out
        # of the main table until the cell is settled from a primary source.
        return PANEL_NONE
    if record.preference_conditioned == "YES":
        return PANEL_B
    return PANEL_NONE


def assert_not_cross_panel(
    method_panels: dict[str, str], comparison: tuple[str, str]
) -> None:
    """Raise if a comparison places a Panel A method against a Panel B method.

    This is the check that stops the single most damaging fairness error
    available here: reporting a global method's pooled front beside COMPOSE's
    per-source fronts as though the two were the same task.
    """
    left, right = comparison
    for name in comparison:
        if name not in method_panels:
            raise PanelAdmissionError(f"no panel recorded for {name!r}")
    if method_panels[left] != method_panels[right]:
        raise PanelAdmissionError(
            f"{left} is {method_panels[left]} and {right} is "
            f"{method_panels[right]}; these are different tasks. Report them in "
            f"separate panels, never as one comparison."
        )


# ---------------------------------------------------------------------------
# Efficiency comparability
# ---------------------------------------------------------------------------

ADMISSIBLE = "ADMISSIBLE"
INCOMPARABLE_SURROGATE = "INCOMPARABLE_SURROGATE_ASYMMETRY"
INCOMPARABLE_INTERNAL_AXIS = "INCOMPARABLE_COMPOSE_INTERNAL_AXIS"
INCOMPARABLE_SECONDARY_AXIS = "INCOMPARABLE_SECONDARY_AXIS_AS_PRIMARY"


def oracle_efficiency_verdict(
    axis: str,
    left: dict[str, Any],
    right: dict[str, Any],
) -> tuple[str, str]:
    """Whether an efficiency comparison on `axis` may be reported at all.

    Returns `(verdict, reason)`. The verdict is ADMISSIBLE only when the axis is
    shared, is not COMPOSE-internal, is not secondary-only, and the two methods
    agree on whether they consume a surrogate.

    The surrogate check is the substantive one. A method that spends its
    objective information on a learned proxy and touches the true oracle only to
    label a batch is not "more oracle-efficient" than one that pays the oracle
    directly -- it has moved the cost onto an axis the comparison does not show.
    """
    if axis not in RESOURCE_AXES:
        raise ResourceAxisError(f"{axis!r} is not a declared resource axis")
    if axis in COMPOSE_INTERNAL_AXES:
        return (
            INCOMPARABLE_INTERNAL_AXIS,
            f"{axis} is COMPOSE-internal; external methods have no shared "
            f"notion of it and are never plotted on it",
        )
    if axis in SECONDARY_ONLY_AXES:
        return (
            INCOMPARABLE_SECONDARY_AXIS,
            f"{axis} is secondary only and may not carry a primary claim",
        )
    for side, ledger in (("left", left), ("right", right)):
        if axis not in ledger:
            return (
                INCOMPARABLE_INTERNAL_AXIS,
                f"{side} does not report {axis}",
            )
    left_surrogate = int(left.get("surrogate_calls", 0) or 0)
    right_surrogate = int(right.get("surrogate_calls", 0) or 0)
    if (left_surrogate > 0) != (right_surrogate > 0):
        return (
            INCOMPARABLE_SURROGATE,
            f"one method consumes a learned surrogate ({left_surrogate} vs "
            f"{right_surrogate} surrogate_calls) and the other does not. On "
            f"{axis} this reports amortization, not efficiency. Report both "
            f"axes side by side and name which is which.",
        )
    return (ADMISSIBLE, f"both methods report {axis} under the same surrogate regime")


# ---------------------------------------------------------------------------
# The frozen hypervolume normalization
# ---------------------------------------------------------------------------

#: Where the scales were frozen, and the only admissible provenance for them.
FROZEN_SCALE_SOURCE = "diagnostics/pareto_tradeoff_census.json :: frozen_scales"

#: Provenance strings a caller may declare. Anything else raises.
ADMISSIBLE_SCALE_PROVENANCE: frozenset[str] = frozenset({FROZEN_SCALE_SOURCE})


def load_frozen_scales(
    census_path: str | Path, pair: str | None = None
) -> dict[str, Any]:
    """Read the frozen nadir and utopia for the adopted objective pair.

    The nadir `r` is the held-in p5 and the utopia `z*` is the held-in p99.
    `z*` is a NORMALIZER and not a cap: held-in-scaled hypervolume may exceed 1,
    and the correct response to that is to report it descriptively, never to
    clip it or to introduce a clipped variant after seeing a result.
    """
    census = json.loads(Path(census_path).read_text())
    adopted = census["adopted_pair"] if pair is None else pair
    entry = next(p for p in census["pairs"] if p["pair"] == adopted)
    key_a, key_b = entry["objective_a"], entry["objective_b"]
    scales = census["frozen_scales"]
    return {
        "pair": adopted,
        "objective_a": key_a,
        "objective_b": key_b,
        "nadir_p5": [scales["reference_p5"][key_a], scales["reference_p5"][key_b]],
        "utopia_p99": [scales["utopia_p99"][key_a], scales["utopia_p99"][key_b]],
        "provenance": FROZEN_SCALE_SOURCE,
        "is_a_cap": False,
        "note": (
            "z* is the held-in p99, not an attainable utopia. Values above 1.0 "
            "are real achievement beyond the reference box. Do not clip."
        ),
    }


def assert_reference_is_frozen(provenance: str) -> None:
    """Refuse a hypervolume reference that was derived from an outcome.

    The failure this prevents is specific: defining the reference point from the
    best front observed makes the winning method's own achievement the yardstick
    it is measured against.
    """
    if provenance not in ADMISSIBLE_SCALE_PROVENANCE:
        raise FrozenScaleError(
            f"hypervolume reference provenance {provenance!r} is not the freeze. "
            f"The only admissible provenance is {FROZEN_SCALE_SOURCE}. A "
            f"reference minted from a front -- any front, including COMPOSE's "
            f"-- measures a method against its own achievement."
        )
