"""Applicability-aware prior for the middle COMPOSE option layer.

The three controller levels have separate jobs::

    Q(M | x, z) -> Q(o | x, M, z) -> q(w | x, M, o)
       WHERE            WHAT               HOW

This module implements only the first, deliberately unlearned version of
``Q(o | x, M, z)``.  One option is sampled for each region draw, so adding an
option cannot clone a draw from ``Q(M)`` or silently increase that region's
outer-controller mass.  The prior is balanced over semantic purpose groups,
then mixed with a uniform exploration floor over the applicable options.

``generic`` is always present.  It recovers the qualified raw region law
exactly, so macros remain proposal channels rather than a replacement grammar.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import numpy as np

from compose_v4.control.fused_option import BUILD_FUSED_RING_OPTION, FUSED_HORIZON
from compose_v4.control.macro_engine import (
    BUILD_RING_SYSTEM,
    MACRO_FAMILIES,
    macro_action_distribution,
    proposal_support,
)

GENERIC_OPTION = "generic"
BUILD_RING_SYSTEM_OPTION = "build_ring_system"

# Stable, reviewable order.  Do not derive serialization order from a set.
MACRO_OPTIONS = (
    "local",
    "grow",
    "append",
    "scaffold_extend",
    "decorate",
    "cyclize",
    "append_system",
    "annulate",
    "small_ring",
    "aromatize",
    "restate",
    "open",
    "rebuild",
    "shrink",
)
OPTIONS = (GENERIC_OPTION,) + MACRO_OPTIONS + (BUILD_RING_SYSTEM_OPTION,)
OPT_IN_OPTIONS = (BUILD_FUSED_RING_OPTION,)

# Balance purposes first and variants second.  This prevents, for example,
# adding another ring topology from silently increasing total ring-option mass.
OPTION_GROUPS = {
    "generic": (GENERIC_OPTION,),
    "local_state": ("local", "aromatize", "restate"),
    "material": ("grow", "append", "scaffold_extend", "decorate"),
    "ring_topology": (
        "cyclize",
        "append_system",
        "annulate",
        "small_ring",
        BUILD_RING_SYSTEM_OPTION,
    ),
    "restructure": ("open", "rebuild", "shrink"),
}
OPTION_GROUP_BY_NAME = {
    option: group for group, options in OPTION_GROUPS.items() for option in options
}


@dataclass(frozen=True)
class OptionChoice:
    """One draw from the auditable, non-learned option prior."""

    selected: str
    applicable: tuple[str, ...]
    probabilities: tuple[float, ...]

    @property
    def selected_probability(self) -> float:
        return self.probabilities[self.applicable.index(self.selected)]


@dataclass(frozen=True)
class ConditionedActionDistribution:
    """Option-conditioned primitive support at one program step."""

    option: str
    active_macro: str | None
    indices: np.ndarray
    probabilities: np.ndarray


def bundle_identity(parent: str, parent_lineage_id: int, region_key, option: str) -> str:
    """Stable identity for one distinct outer-controller search bundle."""

    if option not in OPTIONS + OPT_IN_OPTIONS:
        raise KeyError(f"unknown option {option!r}; known: {OPTIONS}")
    payload = json.dumps(
        {
            "parent": str(parent),
            "parent_lineage_id": int(parent_lineage_id),
            "region": region_key,
            "option": option,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:20]


def primitive_option_at_step(option: str, step: int) -> str | None:
    """Return the primitive-support macro active at zero-based ``step``.

    ``None`` means unrestricted generic support.  A second ``None`` case is a
    completed compound program; callers distinguish it using
    :func:`option_horizon`.
    """

    if option == GENERIC_OPTION:
        return None
    if option == BUILD_FUSED_RING_OPTION:
        return "scaffold_extend" if step < 4 else "annulate" if step < FUSED_HORIZON else None
    if option == BUILD_RING_SYSTEM_OPTION:
        offset = int(step)
        for macro, length in BUILD_RING_SYSTEM:
            if offset < int(length):
                return macro
            offset -= int(length)
        return None
    if option not in MACRO_FAMILIES:
        raise KeyError(f"unknown option {option!r}; known: {OPTIONS}")
    return option


def option_horizon(option: str, generic_horizon: int) -> int:
    """Primitive horizon for an option.

    ``generic`` retains the qualified multi-step region rewrite.  Each ordinary
    macro option conditions one primitive transition.  The compound option is
    indivisible and stops after its declared program length.
    """

    if option == BUILD_RING_SYSTEM_OPTION:
        return sum(int(length) for _macro, length in BUILD_RING_SYSTEM)
    if option == BUILD_FUSED_RING_OPTION:
        return FUSED_HORIZON
    if option not in OPTIONS:
        raise KeyError(f"unknown option {option!r}; known: {OPTIONS}")
    return int(generic_horizon) if option == GENERIC_OPTION else 1


def applicable_options(
    families,
    admissible_indices,
    *,
    n_free_slots: int | None = None,
    include_fused: bool = False,
) -> tuple[str, ...]:
    """Options with initial primitive support inside the selected region.

    Later phases of a compound program are state dependent and cannot be
    required at the initial state.  ``build_ring_system`` is initially
    applicable when its first phase is supported and all eight declared growth
    slots are available.  Failure in a later phase is an explicit UNSAT
    outcome, never a hidden fallback to generic support.
    """

    fam = np.asarray(list(families), dtype=object)
    idx = np.asarray(list(admissible_indices), dtype=int)
    if idx.size and (int(idx.min()) < 0 or int(idx.max()) >= len(fam)):
        raise IndexError(f"admissible action index outside [0, {len(fam)}): {idx.tolist()}")
    present = set(fam[idx].tolist()) if idx.size else set()
    out = [GENERIC_OPTION]
    for option in MACRO_OPTIONS:
        if present.intersection(MACRO_FAMILIES[option]):
            out.append(option)
    first_macro, first_length = BUILD_RING_SYSTEM[0]
    enough_slots = n_free_slots is None or int(n_free_slots) >= int(first_length)
    if enough_slots and present.intersection(MACRO_FAMILIES[first_macro]):
        out.append(BUILD_RING_SYSTEM_OPTION)
    # Necessary family/slot check only. Opt-in callers MUST also test the
    # stateful kernel's first row through retain_product_applicable_options.
    if (
        include_fused
        and (n_free_slots is None or int(n_free_slots) >= 4)
        and present.intersection(MACRO_FAMILIES["scaffold_extend"])
    ):
        out.append(BUILD_FUSED_RING_OPTION)
    return tuple(out)


def retain_product_applicable_options(applicable, has_clean_product) -> tuple[str, ...]:
    """Keep macros with a clean executable product; never remove ``generic``.

    Family support is a necessary but insufficient precondition for macros with
    topology or composition contracts.  The caller owns execution and supplies
    the product-level predicate so this policy module stays independent of the
    molecular executor.
    """

    opts = tuple(applicable)
    if GENERIC_OPTION not in opts:
        raise ValueError("generic must be present before product applicability filtering")
    return (GENERIC_OPTION,) + tuple(
        option for option in opts if option != GENERIC_OPTION and bool(has_clean_product(option))
    )


def balanced_option_prior(
    applicable,
    *,
    exploration: float = 0.10,
) -> np.ndarray:
    """Purpose-balanced prior with a uniform per-option exploration floor.

    The base distribution is uniform over represented purpose groups and then
    uniform over applicable variants within each group.  Mixing a uniform
    distribution over applicable options gives every option at least
    ``exploration / n_applicable`` probability without letting variant count
    dominate the remaining mass.
    """

    opts = tuple(applicable)
    if not opts:
        raise ValueError("applicable option set is empty; generic must be present")
    if len(opts) != len(set(opts)):
        raise ValueError(f"duplicate applicable options: {opts}")
    groups = OPTION_GROUPS
    if BUILD_FUSED_RING_OPTION in opts:
        groups = {**groups, "ring_topology": groups["ring_topology"] + OPT_IN_OPTIONS}
    known = set(OPTIONS + OPT_IN_OPTIONS)
    unknown = [option for option in opts if option not in known]
    if unknown:
        raise KeyError(f"unknown applicable options: {unknown}; known: {OPTIONS}")
    eps = float(exploration)
    if not 0.0 <= eps <= 1.0:
        raise ValueError(f"exploration must lie in [0, 1], got {exploration}")

    represented = tuple(
        group for group in groups if any(option in opts for option in groups[group])
    )
    base = np.zeros(len(opts), dtype=float)
    for group in represented:
        members = [i for i, option in enumerate(opts) if option in groups[group]]
        for i in members:
            base[i] = 1.0 / (len(represented) * len(members))
    floor = np.full(len(opts), 1.0 / len(opts), dtype=float)
    q = (1.0 - eps) * base + eps * floor
    return q / q.sum()


def sample_option(applicable, rng, *, exploration: float = 0.10) -> OptionChoice:
    """Draw exactly one option for one draw from ``Q(M | x, z)``."""

    opts = tuple(applicable)
    q = balanced_option_prior(opts, exploration=exploration)
    selected = opts[int(rng.choice(len(opts), p=q))]
    return OptionChoice(
        selected=selected,
        applicable=opts,
        probabilities=tuple(float(x) for x in q),
    )


def conditioned_action_distribution(
    families,
    probabilities,
    admissible,
    option: str,
    *,
    step: int = 0,
    clean=None,
    temperature: float = 2.0,
    exploration: float = 0.15,
):
    """Restrict and normalize the primitive law for one option step.

    For ``generic`` this is exactly the normalized qualified region law.  For
    a macro it delegates to the existing ``proposal_support`` and
    ``macro_action_distribution`` machinery.  ``clean`` is a product-level
    mask, allowing the caller to enforce validity, frozen context, and the
    macro-local contract before normalization.
    """

    if option == BUILD_FUSED_RING_OPTION:
        raise ValueError("build_fused_ring requires the stateful OptionContinuationKernel")
    fams = list(families)
    probs = np.asarray(probabilities, dtype=float)
    if probs.ndim != 1 or len(fams) != len(probs):
        raise ValueError(
            f"families/probabilities must be aligned 1D arrays, got {len(fams)} and {probs.shape}"
        )
    admissible = np.asarray(list(admissible), dtype=int)
    if admissible.size and (int(admissible.min()) < 0 or int(admissible.max()) >= len(probs)):
        raise IndexError(
            f"admissible action index outside [0, {len(probs)}): {admissible.tolist()}"
        )
    if clean is None:
        clean_mask = np.ones(len(probs), dtype=bool)
    else:
        clean_mask = np.asarray(clean, dtype=bool)
        if clean_mask.shape != probs.shape:
            raise ValueError(
                f"clean mask shape {clean_mask.shape} != probability shape {probs.shape}"
            )
    in_region = np.zeros(len(probs), dtype=bool)
    in_region[admissible] = True
    clean_mask = clean_mask & in_region

    active = primitive_option_at_step(option, int(step))
    if option == GENERIC_OPTION:
        idx = np.flatnonzero(clean_mask)
        p = probs[idx]
        if idx.size == 0:
            q = np.zeros(0, dtype=float)
        elif np.isfinite(p).all() and p.sum() > 0:
            q = p / p.sum()
        else:
            q = np.full(idx.size, 1.0 / idx.size)
        return ConditionedActionDistribution(option, None, idx, q)

    if active is None:
        return ConditionedActionDistribution(
            option, None, np.zeros(0, dtype=int), np.zeros(0, dtype=float)
        )
    support = proposal_support(fams, probs)
    idx, q = macro_action_distribution(
        fams,
        probs,
        support,
        active,
        clean_mask,
        temperature=float(temperature),
        epsilon=float(exploration),
    )
    return ConditionedActionDistribution(option, active, idx, q)
