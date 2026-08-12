"""Claim 2: three transport laws over ONE executable canonical-successor support.

The scientific object of Workstream B is a comparison of *laws*, not of
*supports*.  Every arm sees exactly the same set of distinct productive
canonical successors at every state; only the probability assigned to them
differs.  That is the ``law_only`` relation of
``compose_v4.experiments.successor_kernel.assert_arms_comparable`` and it is
what makes any trajectory difference attributable to what ``R_theta`` learned.

    r_theta            P_theta(y | x), the frozen learned reference process
    uniform_canonical  1 / |N+(x)|
    empirical_family   sum over families k reaching y of qhat(k) / |N_k(x)|

``qhat`` is the frozen realized training family law restricted to families with
a legal productive successor at ``x`` and renormalized there.  That definition
is not invented here: it is the frozen definition committed at
``configs/comparator_registry_v3.json`` under
``frozen_definitions.state_independent_empirical_family_prior`` and used by the
Experiment 1 evaluator (``modal_apps/experiment1_reference_law_app.py``).  This
module reuses it so the two experiments cannot silently disagree about what the
baseline is.

Why one row object instead of three kernels
-------------------------------------------
``UniformSuccessorKernel`` already exists and is a probability wrapper over a
base kernel.  The empirical-family law needs something no ``SuccessorBatch``
carries: **which operator families reach each canonical successor**.  Building
that by calling the kernel a second time would double the dominant cost (mark
execution and canonicalization), so instead one enumeration produces one
``SuccessorRow`` carrying keys, learned probabilities, alias counts *and*
family attribution, and all three laws are pure functions of that row.

The row is deliberately a plain, JSON-serializable value.  Everything below the
enumeration boundary therefore runs and is unit-tested locally, with no model,
no checkpoint and no Modal runtime.

Instrument note
---------------
``arm_divergence`` exists because "three arms" is a claim about the *laws*, not
about the code paths.  At a state with one legal successor all three arms are
identical by construction and the trajectory carries no information about which
law produced it.  A characterization run that never checks this can report
three curves that are three drawings of the same process.
"""

from __future__ import annotations

from bisect import bisect_right
from collections import Counter
from dataclasses import dataclass, field
from itertools import accumulate
from math import fsum, isfinite
from typing import Any, Mapping, Sequence

#: Absolute tolerance for the "a law is a probability distribution" invariant.
#: Matches ``successor_kernel.NORMALIZATION_TOLERANCE``; loose enough for
#: float accumulation over a few thousand successors, tight enough that a
#: genuinely unnormalized law fails instead of quietly biasing every rollout.
LAW_NORMALIZATION_TOLERANCE = 1e-6

ARM_REFERENCE = "r_theta"
ARM_UNIFORM = "uniform_canonical"
ARM_EMPIRICAL_FAMILY = "empirical_family"

#: Frozen arm order. Fixed here so shards, analysis and figures cannot disagree.
ARMS: tuple[str, ...] = (ARM_REFERENCE, ARM_UNIFORM, ARM_EMPIRICAL_FAMILY)


class TransportLawError(ValueError):
    """A transport law broke an invariant of the shared successor support."""


@dataclass(frozen=True)
class SuccessorRow:
    """One state's productive canonical-successor support with family attribution.

    ``successor_keys`` are DISTINCT canonical molecular keys in sorted order --
    aliases are already collapsed, exactly as ``SuccessorBatch`` requires.
    ``families[i]`` is the set of operator families whose marks execute to
    ``successor_keys[i]``; a successor reachable through several operators
    appears once and lists every family that reaches it, which is what makes
    the cross-family sum in the empirical-family law a sum rather than a double
    count.

    ``reference_probabilities`` is the frozen learned law conditioned on taking
    a productive jump.  ``virtual_mark_count`` records canonical self-events,
    which are not molecular jumps and are excluded from every arm.
    """

    source_key: str
    successor_keys: tuple[str, ...]
    reference_probabilities: tuple[float, ...]
    alias_counts: tuple[int, ...]
    families: tuple[tuple[str, ...], ...]
    cells: tuple[tuple[str, ...], ...] = ()
    virtual_mark_count: int = 0

    def __post_init__(self) -> None:
        width = len(self.successor_keys)
        if len(set(self.successor_keys)) != width:
            raise TransportLawError(
                "successor keys are not distinct -- aliases were not collapsed"
            )
        if tuple(sorted(self.successor_keys)) != self.successor_keys:
            raise TransportLawError("successor keys must be in sorted order")
        if self.source_key in self.successor_keys:
            raise TransportLawError(
                "a successor carries the source key; a self-transition is not a molecular jump"
            )
        for name, value in (
            ("reference_probabilities", self.reference_probabilities),
            ("alias_counts", self.alias_counts),
            ("families", self.families),
        ):
            if len(value) != width:
                raise TransportLawError(f"{name} has width {len(value)}, expected {width}")
        if self.cells and len(self.cells) != width:
            raise TransportLawError("cells must be empty or aligned with successor keys")
        for index, probability in enumerate(self.reference_probabilities):
            if not isfinite(probability) or probability < 0.0:
                raise TransportLawError(
                    f"reference probability {probability!r} at {index} is not a probability"
                )
        for index, count in enumerate(self.alias_counts):
            if count < 1:
                raise TransportLawError(
                    f"alias_count {count} at {index}; a merged successor came from >=1 mark"
                )
        for index, names in enumerate(self.families):
            if not names:
                raise TransportLawError(
                    f"successor {self.successor_keys[index]!r} lists no reaching family"
                )
            if tuple(sorted(set(names))) != tuple(names):
                raise TransportLawError("reaching families must be sorted and distinct")
        if width:
            _assert_normalized(self.reference_probabilities, ARM_REFERENCE)

    @property
    def support_size(self) -> int:
        """|N+(x)|: distinct productive canonical successors."""
        return len(self.successor_keys)

    @property
    def is_terminal(self) -> bool:
        return not self.successor_keys

    @property
    def family_successor_counts(self) -> dict[str, int]:
        """|N_k(x)|: distinct canonical successors reachable by each family.

        This is the denominator of the within-family uniform law.  It counts
        SUCCESSORS, not marks: two marks of one family that execute to the same
        molecule contribute one, which is why the empirical-family law is a
        law over molecules rather than over syntax.
        """
        counts: Counter[str] = Counter()
        for names in self.families:
            counts.update(set(names))
        return dict(counts)

    @property
    def legal_families(self) -> tuple[str, ...]:
        return tuple(sorted(self.family_successor_counts))

    def to_json(self) -> dict[str, Any]:
        return {
            "source_key": self.source_key,
            "successor_keys": list(self.successor_keys),
            "reference_probabilities": list(self.reference_probabilities),
            "alias_counts": list(self.alias_counts),
            "families": [list(names) for names in self.families],
            "cells": [list(names) for names in self.cells],
            "virtual_mark_count": self.virtual_mark_count,
        }

    @classmethod
    def from_json(cls, payload: Mapping[str, Any]) -> "SuccessorRow":
        return cls(
            source_key=str(payload["source_key"]),
            successor_keys=tuple(payload["successor_keys"]),
            reference_probabilities=tuple(float(v) for v in payload["reference_probabilities"]),
            alias_counts=tuple(int(v) for v in payload["alias_counts"]),
            families=tuple(tuple(names) for names in payload["families"]),
            cells=tuple(tuple(names) for names in payload.get("cells", ())),
            virtual_mark_count=int(payload.get("virtual_mark_count", 0)),
        )


def _assert_normalized(probabilities: Sequence[float], label: str) -> None:
    total = fsum(probabilities)
    if abs(total - 1.0) > LAW_NORMALIZATION_TOLERANCE:
        raise TransportLawError(
            f"{label} sums to {total!r}, not 1 (tolerance {LAW_NORMALIZATION_TOLERANCE}); "
            "a transport law must be normalized over the distinct successor support"
        )


# ---- the three laws -------------------------------------------------------


def reference_law(row: SuccessorRow) -> tuple[float, ...]:
    """The frozen learned reference process, as enumerated."""
    return tuple(row.reference_probabilities)


def uniform_canonical_law(row: SuccessorRow) -> tuple[float, ...]:
    """Uniform over DISTINCT productive canonical successors: 1 / |N+(x)|.

    Uniform over *marks* would instead reward whichever molecule happens to
    have the most syntactic encodings, which is a fact about the action codec
    rather than about chemistry.  Alias collapse happens before this law, so
    a successor reachable by five marks receives the same mass as one reachable
    by a single mark.
    """
    if row.is_terminal:
        return ()
    share = 1.0 / float(row.support_size)
    return tuple(share for _ in row.successor_keys)


def empirical_family_law(
    row: SuccessorRow,
    family_frequencies: Mapping[str, float],
) -> tuple[float, ...]:
    """State-independent empirical family prior, uniform within family.

        qhat(k) = q(k) restricted to families legal at x, renormalized
        P(y|x)  = sum over families k reaching y of qhat(k) / |N_k(x)|

    This is the STRONG form of the unlearned baseline: it receives the correct
    global operator frequencies, the exact legal support, state-dependent
    family availability, and alias aggregation.  What it does not receive is
    learned molecular context.

    The cross-family sum is what makes this normalized: summing over all y
    gives sum_k qhat(k) * |N_k(x)| / |N_k(x)| = 1.  Dropping the sum -- taking
    only one "primary" family per successor -- would silently produce a
    sub-probability and bias every downstream statistic.
    """
    if row.is_terminal:
        return ()
    per_family = row.family_successor_counts
    legal = {
        name: float(family_frequencies.get(name, 0.0))
        for name, count in per_family.items()
        if count > 0
    }
    mass = fsum(legal.values())
    if mass <= 0.0:
        raise TransportLawError(
            f"no legal family at {row.source_key!r} carries frozen empirical mass; "
            f"legal families {sorted(per_family)} are absent from the frozen law. "
            "Falling back to uniform here would silently make this arm a second "
            "copy of uniform_canonical."
        )
    qhat = {name: value / mass for name, value in legal.items()}
    probabilities = tuple(
        fsum(qhat.get(name, 0.0) / per_family[name] for name in set(names))
        for names in row.families
    )
    _assert_normalized(probabilities, ARM_EMPIRICAL_FAMILY)
    return probabilities


#: Frozen arm registry: name -> callable(row, family_frequencies) -> law.
def law_for_arm(
    arm: str,
    row: SuccessorRow,
    family_frequencies: Mapping[str, float],
) -> tuple[float, ...]:
    """Dispatch by frozen arm name so no caller can invent a fourth law."""
    if arm == ARM_REFERENCE:
        return reference_law(row)
    if arm == ARM_UNIFORM:
        return uniform_canonical_law(row)
    if arm == ARM_EMPIRICAL_FAMILY:
        return empirical_family_law(row, family_frequencies)
    raise TransportLawError(f"unknown arm {arm!r}; expected one of {list(ARMS)}")


# ---- sampling -------------------------------------------------------------


def sample_index(probabilities: Sequence[float], variate: float) -> int:
    """Inverse-CDF sample from ``probabilities`` using one uniform in [0, 1).

    Taking the variate as an ARGUMENT rather than drawing it internally is what
    lets every arm consume the identical uniform at the identical step -- common
    random numbers.  Under the shared sorted key order this couples the arms:
    where two laws agree they take the same action, so a divergence in the
    realized trajectories is attributable to the laws and not to the sampler's
    seed.  This is a variance reduction, and it is declared rather than
    discovered.
    """
    if not probabilities:
        raise TransportLawError("cannot sample from an empty support")
    if not 0.0 <= variate < 1.0:
        raise TransportLawError(f"variate {variate!r} is outside [0, 1)")
    cumulative = list(accumulate(probabilities))
    index = bisect_right(cumulative, variate * cumulative[-1])
    return min(index, len(probabilities) - 1)


# ---- instrument checks ----------------------------------------------------


def total_variation(left: Sequence[float], right: Sequence[float]) -> float:
    """TV distance between two laws over the SAME ordered support."""
    if len(left) != len(right):
        raise TransportLawError("total variation needs two laws over one support")
    return 0.5 * fsum(abs(a - b) for a, b in zip(left, right))


@dataclass(frozen=True)
class ArmDivergence:
    """How far apart the arms' laws actually are at one state.

    A characterization run whose arms coincide is not measuring three transport
    laws.  ``support_size == 1`` forces coincidence; beyond that, coincidence
    would mean the learned law is numerically indistinguishable from an
    unlearned one at this state, which is itself a reportable finding rather
    than something to average away.
    """

    source_key: str
    support_size: int
    pairwise: dict[str, float] = field(default_factory=dict)

    @property
    def minimum(self) -> float:
        return min(self.pairwise.values()) if self.pairwise else 0.0

    @property
    def degenerate(self) -> bool:
        """True when no pair of arms differs measurably at this state."""
        return self.support_size <= 1 or self.minimum <= LAW_NORMALIZATION_TOLERANCE


def arm_divergence(
    row: SuccessorRow,
    family_frequencies: Mapping[str, float],
) -> ArmDivergence:
    """Pairwise TV between all three arms at one state."""
    if row.is_terminal:
        return ArmDivergence(source_key=row.source_key, support_size=0)
    laws = {arm: law_for_arm(arm, row, family_frequencies) for arm in ARMS}
    pairwise: dict[str, float] = {}
    for index, left in enumerate(ARMS):
        for right in ARMS[index + 1 :]:
            pairwise[f"{left}|{right}"] = total_variation(laws[left], laws[right])
    return ArmDivergence(
        source_key=row.source_key,
        support_size=row.support_size,
        pairwise=pairwise,
    )


# ---- the enumeration boundary ---------------------------------------------


def enumerate_successor_row(model: Any, state: Any, time: float) -> SuccessorRow:
    """Build one ``SuccessorRow`` from the frozen model with ONE mark execution.

    Every import is local so the pure law layer above stays importable, and
    testable, without torch, a checkpoint or the Modal runtime.

    Why this composition and not ``canonical_successor_result``
    ----------------------------------------------------------
    ``canonical_successor_result`` returns probabilities but throws away the
    mark-to-successor map, so it cannot say which operator families reach a
    given molecule -- exactly what the empirical-family arm needs.  Calling it
    *and* compiling the partition would execute and canonicalize every legal
    mark twice, and mark execution is the dominant cost of the whole
    experiment.  So this reuses the two production functions that already
    factor the work the right way:

        compile_state_successor_map          executes each mark once, groups by
                                             canonical key, keeps family
                                             attribution, drops self-events;
        forward_compiled_successor_partitions scores those static groups.

    No aggregation logic is reimplemented here.  In particular the segmented
    log-sum-exp over each successor fiber, the alias collapse and the
    productive/virtual mass split all remain in production code, and
    ``forward_compiled_successor_partitions`` independently proves that
    productive and virtual coordinates exhaust unit mark mass.
    """
    import torch

    from compose_v4.experiments.factorized_successor_training import (
        compile_state_successor_map,
        forward_compiled_successor_partitions,
    )
    from compose_v4.experiments.factorized_mark_conditional import (
        operator_capability_batch_kwargs,
    )
    from compose_v4.model.factorized_tracelet_rate_model import (
        prepare_factorized_mark_batch,
    )

    partition = compile_state_successor_map(model, state, time=float(time))
    virtual_mark_count = len(partition.virtual_marks)
    if not partition.successor_groups:
        # A terminal state of the SHARED support. Every arm sees it identically;
        # which law arrived here is the only thing that differs.
        return SuccessorRow(
            source_key=partition.source_key,
            successor_keys=(),
            reference_probabilities=(),
            alias_counts=(),
            families=(),
            cells=(),
            virtual_mark_count=virtual_mark_count,
        )

    batch = prepare_factorized_mark_batch(
        (state,),
        (float(time),),
        (None,),
        (None,),
        (0.0,),
        use_aromatic_bond_view=True,
        ring_catalog=model.ring_catalog,
        **operator_capability_batch_kwargs(model.operator_capabilities),
    ).to(model.device)
    with torch.no_grad():
        prediction = forward_compiled_successor_partitions(model, batch, (partition,))

    keys = tuple(prediction.successor_keys[0])
    probabilities = tuple(
        float(value) for value in prediction.successor_log_probabilities[0].exp().tolist()
    )
    groups = partition.successor_groups
    if tuple(group.target_key for group in groups) != keys:
        raise TransportLawError(
            "scored successor keys do not follow the compiled partition order"
        )
    families = tuple(
        tuple(sorted({alias.family_name for alias in group.aliases})) for group in groups
    )
    cells = tuple(
        tuple(sorted({f"{alias.family_name}:{alias.table_name}" for alias in group.aliases}))
        for group in groups
    )
    # Renormalize the float32 row into an exact probability vector. The kernel
    # already proved normalization to 2e-5 in float32; this removes the residual
    # so the sampler's inverse CDF and the strict law invariant agree.
    total = fsum(probabilities)
    if not total > 0.0:
        raise TransportLawError("scored reference row carries no productive mass")
    probabilities = tuple(value / total for value in probabilities)

    return SuccessorRow(
        source_key=partition.source_key,
        successor_keys=keys,
        reference_probabilities=probabilities,
        alias_counts=tuple(len(group.marks) for group in groups),
        families=families,
        cells=cells,
        virtual_mark_count=virtual_mark_count,
    )


def cross_check_against_production_kernel(
    model: Any,
    state: Any,
    time: float,
    row: SuccessorRow,
    *,
    tolerance: float = 1e-4,
) -> dict[str, Any]:
    """Verify a row against ``canonical_successor_result`` on the SAME state.

    This costs a second mark execution, so it runs on a declared subsample
    rather than every state.  It exists because ``enumerate_successor_row``
    reaches the learned law by a different production route than the kernel
    every other COMPOSE experiment consumes; if the two ever disagreed, every
    number in this workstream would be measuring a private kernel.
    """
    import torch

    from compose_v4.experiments.production_successor_kernel import (
        canonical_successor_result,
    )

    with torch.no_grad():
        result = canonical_successor_result(model, state, float(time))
    batch = result.batch
    reference_keys = tuple(successor.key for successor in batch.successors)
    reference_probabilities = {
        successor.key: successor.probability for successor in batch.successors
    }
    keys_match = reference_keys == row.successor_keys
    worst = 0.0
    if keys_match:
        worst = max(
            (
                abs(probability - reference_probabilities[key])
                for key, probability in zip(row.successor_keys, row.reference_probabilities)
            ),
            default=0.0,
        )
    aliases_match = keys_match and tuple(
        successor.alias_count for successor in batch.successors
    ) == row.alias_counts
    return {
        "source_key": row.source_key,
        "keys_match": bool(keys_match),
        "alias_counts_match": bool(aliases_match),
        "max_absolute_probability_difference": float(worst),
        "agrees": bool(keys_match and aliases_match and worst <= tolerance),
        "tolerance": float(tolerance),
        "production_support_size": batch.support_size,
        "row_support_size": row.support_size,
    }


__all__ = [
    "ARMS",
    "ARM_EMPIRICAL_FAMILY",
    "ARM_REFERENCE",
    "ARM_UNIFORM",
    "ArmDivergence",
    "LAW_NORMALIZATION_TOLERANCE",
    "SuccessorRow",
    "TransportLawError",
    "arm_divergence",
    "cross_check_against_production_kernel",
    "empirical_family_law",
    "enumerate_successor_row",
    "law_for_arm",
    "reference_law",
    "sample_index",
    "total_variation",
    "uniform_canonical_law",
]
