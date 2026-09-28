"""Persistent anchor context across the modules of one generic program.

v1 draws every module's growth anchor uniformly and independently over the whole
molecule and then walks the tip forward, so a K-module program is K mostly unrelated
local edits rather than one coordinated construction.  Measured consequence: a
transformation needing two or three primitives on the SAME atom carries essentially no
proposal mass.  A tert-butyl is executable from three ``atom_insert`` primitives on one
anchor -- verified by construction, ``CNCc1ccccc1 -> CC(C)(C)NCc1ccccc1`` -- and was
observed 0 times across 2,803 production draws on every one of five live molecules, at
every module cap from three to eight.  Raising the cap does not help, because each added
module still redraws its own anchor: the defect is coordination, not program length.

``FocusState`` remembers the anchor a module grew from and the tip it grew to.
``AnchorFocusPolicy`` decides whether the next module returns to that anchor (which is
what produces a branch), continues from the tip (v1's implicit chain behaviour), or
draws a fresh site.  Passing ``None`` keeps v1 byte-identical INCLUDING its RNG
consumption, which is why the policy is consulted only when one is supplied.

The policy carries no task information: it sees the state and the previous module's
anchor, never a target, a score or an objective.
"""
from __future__ import annotations

from dataclasses import dataclass

from compose_v4.chem.state import is_element

# ---- Focus state ----


@dataclass(frozen=True)
class FocusState:
    """Where the previous module acted.  Slots are stable across modules."""

    anchor: int | None = None
    tip: int | None = None


def viable_anchor(graph, slot) -> bool:
    """Can ``slot`` carry another substituent on ``graph``?

    States are slot-stable, so a slot recorded by an earlier module still addresses the
    same atom -- unless that atom was deleted, or its last hydrogen was consumed.  Both
    are ordinary outcomes of an intervening module, so an unusable focus falls back to
    the uniform draw rather than refusing the family.
    """
    if slot is None:
        return False
    slot = int(slot)
    types = graph.atom_types
    if not 0 <= slot < len(types):
        return False
    if not bool(is_element(types)[slot]):
        return False
    return int(graph.implicit_h_counts[slot]) >= 1


# ---- Policy ----


@dataclass(frozen=True)
class AnchorFocusPolicy:
    """Choose the next module's growth anchor from the previous module's focus.

    ``reuse_anchor`` is the branch-producing branch: it returns to the atom the last
    module grew FROM, so a second insertion lands beside the first rather than after it.
    ``continue_tip`` extends from the atom the last module grew TO, which is what v1
    does within a single module.  The remaining mass is a fresh uniform draw, which is
    what v1 does between modules.
    """

    reuse_anchor: float = 0.40
    continue_tip: float = 0.25

    def __post_init__(self) -> None:
        if not 0.0 <= self.reuse_anchor <= 1.0:
            raise ValueError("reuse_anchor must be a probability")
        if not 0.0 <= self.continue_tip <= 1.0:
            raise ValueError("continue_tip must be a probability")
        if self.reuse_anchor + self.continue_tip > 1.0:
            raise ValueError("focus branch probabilities must not exceed one")

    def resolve(self, graph, rng, focus: FocusState | None) -> int | None:
        """Return the anchor slot to grow from, or ``None`` for the uniform draw.

        Exactly one random number is consumed per call whatever the outcome, so the
        stream does not depend on which branch a given state happens to admit.
        """
        if focus is None:
            return None
        draw = float(rng.random())
        if draw < self.reuse_anchor and viable_anchor(graph, focus.anchor):
            return int(focus.anchor)
        if draw < self.reuse_anchor + self.continue_tip and viable_anchor(
            graph, focus.tip
        ):
            return int(focus.tip)
        return None


# ---- Coherent local construction ----


@dataclass(frozen=True)
class LocalConstructionPolicy:
    """Let ONE growth module emit a coordinated construction instead of a chain.

    ``_grow_actions`` walks its insertion point forward unconditionally (``at =
    action.slot``), so a length-3 module is always a 3-chain.  Building a branch
    therefore needed three SEPARATE modules to independently draw a growth family and
    then independently land on the same atom.  Measured on a live molecule: the family
    mix is near-uniform over thirteen families, so programs with three growth modules
    are 0.17% of draws, and the anchor-reuse policy could bite on only 1.0% -- which is
    why anchor reuse alone moved tert-butyl 0 -> 0.

    This removes the multiplicative family penalty rather than the anchor penalty: one
    growth draw (~1/13) can now return to the atom it started from between insertions,
    so a coordinated construction costs one family draw instead of three aligned ones.
    It adds no family, no template and no element: the primitives, the element sets and
    the executor are exactly as they were.  ``None`` keeps the linear walk byte-identical
    including RNG consumption.
    """

    branch: float = 0.45

    def __post_init__(self) -> None:
        if not 0.0 <= self.branch <= 1.0:
            raise ValueError("branch must be a probability")

    def returns_to_origin(self, graph, rng, origin) -> bool:
        """Should the next insertion go back to ``origin`` rather than the tip?

        Consumes exactly one random number per call so the stream does not depend on
        whether the origin happens to still have capacity.
        """
        draw = float(rng.random())
        return draw < self.branch and viable_anchor(graph, origin)
