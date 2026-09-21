"""Attachment-aware fragment control: q(Z | G, C_frag) with declared interfaces.

The fragment-constrained benchmark hands the generator a constraint
specification ``C_frag`` that names two different things:

``retained region``
    which atoms and bonds must survive unchanged.  This repository already
    consumes that part, as :class:`~compose_v4.benchmark
    .fragment_conditioned_sampler.RegionLock`, and it is why fragment
    CONTAINMENT is at or near the chemical-validity ceiling.

``allowed attachment interfaces``
    which atoms of that region are declared open, and how many external heavy
    neighbours each must acquire.  The released prompts carry this as dummy
    atoms (``[1*]``, ``[2*]``, ...), one per open valence.  Nothing downstream
    of the prompt parser consumed it, so the proposal distribution grew
    wherever the source-agnostic prior felt like growing and the declared site
    was satisfied only by chance.

This module supplies the second part.  It is a CONSTRAINT-DERIVED controller:
every decision it makes is a function of the declared specification and of
structural features of the current state.  It never reads the benchmark drug,
the task label, or any instance identity, and it holds one frozen parameter set
across every drug and every task.  A prompt that declares no interface --
``superstructure_generation`` declares none -- gets a vacuous controller that
changes nothing, which is the correct reading of "no declared interface", not a
special case for that task.

Three mechanisms, all pathwise
------------------------------

``interface admission``
    A bond between a locked slot and a non-locked slot may exist only where the
    locked slot is a declared interface.  Growth off an undeclared part of the
    retained core is deleted from the legal-event fiber rather than filtered at
    the endpoint.  Vacuous when no interface is declared.

``attachment-first staging``
    While some declared interface is still short of its required external
    heavy-neighbour count, only events that strictly reduce the unsatisfied set
    are admitted.  Coverage is therefore established before free elaboration
    begins, in at most one event per declared interface.  Since deletions and
    reroutes can un-cover an interface, the set is recomputed at every step and
    staging re-engages on its own.

``attachment redirection``
    The structural action ``Z = (R, H, alpha, D)`` carries attachment
    ``alpha`` as a controller degree of freedom.  During staging, an
    ``atom_insert`` the prior anchored somewhere inadmissible is re-anchored to
    the least-covered unsatisfied interface, keeping the prior's PAYLOAD -- the
    element, the bond order, the derived hydrogen count.  The prior chooses
    WHAT to attach; the declared constraint chooses WHERE.  The target is
    picked deterministically (fewest external neighbours, then lowest slot), so
    the controller consumes no random draws and an instance with no declared
    interface keeps a bit-identical RNG stream.

The controller never widens the fiber.  A redirected action is still executed
by the production executor and still has to pass the region lock, so no state
that violates the retained region can be committed by this path.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace

from compose_v4.chem.molecular_graph import MolecularGraph, is_element
from compose_v4.rewrite.operators import AtomInsert


@dataclass(frozen=True)
class AttachmentControlConfig:
    """One frozen parameter set, used for every drug and every task."""

    enabled: bool = False
    # Refuse a bond from an undeclared locked atom to a new atom.
    restrict_interfaces: bool = True
    # Admit only coverage-increasing events while an interface is unsatisfied.
    attachment_first: bool = True
    # Re-anchor an inadmissible atom_insert onto an unsatisfied interface.
    redirect_attachment: bool = True
    # Release the constructed linker join from the region lock so a genuine
    # linker can displace it.  Vacuous for single-core prompts.
    release_linker_join: bool = True
    # Refuse an endpoint whose two retained cores are still directly bonded.
    require_separated_cores: bool = True
    # Lower a redirected bond order to the free valence the declared site
    # actually has.  Without this the prior's order is carried onto a site that
    # cannot accept it and the executor refuses the whole event.
    realize_redirected_order: bool = True
    # ---- Two-interface path construction ----
    #
    # A declared-interface specification cannot express "build a linker between
    # these two sites".  Measured: with a seeded one-atom bridge the seed
    # already covers BOTH declared interfaces at event 0, so coverage staging is
    # vacuous from the first step and 121 of 122 committed endpoints hold the
    # seeded length.  Coverage is the wrong predicate; the length of the
    # core-to-core PATH is the right one, and it must be a decision variable
    # rather than whatever the prior happens to leave behind.
    #
    # Vacuous unless the prompt declares two retained regions, which is a
    # property of the SPECIFICATION and never of a drug or a task name.
    path_program: bool = False
    # Inclusive band the realized path length is steered into.  One band for
    # every instance; the target is drawn per TRAJECTORY from it, so a run
    # explores the declared band instead of pinning one length per prompt.
    path_length_min: int = 2
    path_length_max: int = 5


@dataclass(frozen=True)
class AttachmentSpec:
    """Declared interfaces for one prompt, in canonical slot coordinates.

    ``lock_groups`` is one frozenset of slots per declared fragment.  For a
    single-core prompt there is one group; for a linker prompt there are two,
    and ``released_pairs`` holds the slot pair carrying the constructed join
    bond between them.
    """

    interfaces: tuple[int, ...]
    requirements: tuple[tuple[int, int], ...]
    lock_groups: tuple[frozenset[int], ...]
    released_pairs: frozenset[frozenset[int]] = frozenset()

    @property
    def declares_interfaces(self) -> bool:
        return bool(self.interfaces)

    def requirement_of(self, slot: int) -> int:
        for candidate, count in self.requirements:
            if candidate == slot:
                return count
        return 0

    def identity_payload(self) -> dict:
        """Address-free description, for the run artifact."""
        return {
            "interfaces": list(self.interfaces),
            "requirements": [list(pair) for pair in self.requirements],
            "lock_group_sizes": [len(group) for group in self.lock_groups],
            "released_pairs": [sorted(pair) for pair in sorted(map(sorted, self.released_pairs))],
        }


# ---- Coverage ----


def external_neighbour_count(
    state: MolecularGraph, slot: int, locked: Sequence[int] | frozenset[int]
) -> int:
    """Heavy neighbours of ``slot`` that are NOT part of the retained region.

    This mirrors the benchmark's own attachment test, which counts a declared
    site's neighbours that fall outside the fragment's substructure embedding.
    The region lock guarantees the locked slots ARE such an embedding, so a
    count taken here is sufficient for that test even though the benchmark is
    free to satisfy it with a different embedding.
    """
    locked_set = frozenset(locked)
    real = is_element(state.atom_types)
    row = state.bonds[slot]
    return int(
        sum(
            1
            for other in range(state.n_atoms)
            if other != slot
            and other not in locked_set
            and bool(real[other])
            and int(row[other]) > 0
        )
    )


class AttachmentController:
    """Admission and redirection for one prompt's declared interfaces."""

    def __init__(
        self,
        spec: AttachmentSpec,
        locked_slots: Sequence[int],
        config: AttachmentControlConfig,
    ) -> None:
        self._spec = spec
        self._locked = frozenset(int(slot) for slot in locked_slots)
        self._config = config
        self._interfaces = frozenset(spec.interfaces)
        self._undeclared = tuple(sorted(self._locked - self._interfaces))

    @property
    def spec(self) -> AttachmentSpec:
        return self._spec

    @property
    def config(self) -> AttachmentControlConfig:
        """The frozen parameter set.  One object serves every drug and task."""
        return self._config

    @property
    def active(self) -> bool:
        """False when the prompt declares no interface: the controller is a no-op."""
        return self._config.enabled and self._spec.declares_interfaces

    # ---- Coverage over the declared interfaces ----

    def coverage(self, state: MolecularGraph) -> dict[int, int]:
        return {
            slot: external_neighbour_count(state, slot, self._locked)
            for slot in self._spec.interfaces
        }

    def unsatisfied(self, state: MolecularGraph) -> tuple[int, ...]:
        covered = self.coverage(state)
        return tuple(
            slot
            for slot in self._spec.interfaces
            if covered[slot] < self._spec.requirement_of(slot)
        )

    def all_interfaces_covered(self, state: MolecularGraph) -> bool:
        return not self.unsatisfied(state)

    def cores_are_separated(self, state: MolecularGraph) -> bool:
        """True when no two distinct retained cores are directly bonded.

        Vacuous for a single-core prompt.  For a linker prompt this is the
        difference between a genuine linker and the zero-atom join the prompt
        was CONSTRUCTED with, which the benchmark's own endpoint test cannot
        tell apart because each core satisfies the other's attachment
        requirement.
        """
        groups = self._spec.lock_groups
        if len(groups) < 2:
            return True
        for index, left in enumerate(groups):
            for right in groups[index + 1 :]:
                for i in left:
                    row = state.bonds[i]
                    if any(int(row[j]) > 0 for j in right):
                        return False
        return True

    def realized_linker_length(self, state: MolecularGraph) -> int | None:
        """Atoms on the shortest core-to-core path, excluding both cores.

        This is the number that says whether a linker was DESIGNED or merely
        inherited.  The corrected start state seeds an unlocked bridge, because
        the executor refuses a disconnected pair and the alternative -- joining
        the cores directly -- cannot express the task at all.  A seed is not a
        result, so the seed length has to be reported beside every linker row
        and the distribution above it is what shows the generator did work.

        Returns ``None`` for a single-core prompt, where the quantity is not
        defined, and ``None`` when no core-to-core path exists.  ``0`` means the
        cores are directly bonded, i.e. the zero-atom linker.
        """
        groups = self._spec.lock_groups
        if len(groups) < 2:
            return None
        source, target = groups[0], groups[1]
        # Breadth-first over NON-core atoms: the path length we want counts only
        # the atoms between the cores, so core slots are never intermediates.
        frontier = {int(i) for i in source}
        interior = frozenset().union(*groups)
        seen = set(frontier)
        distance = 0
        while frontier:
            nxt: set[int] = set()
            for i in frontier:
                row = state.bonds[i]
                for j in range(len(row)):
                    if int(row[j]) <= 0 or j in seen:
                        continue
                    if j in target:
                        return distance
                    if j in interior:
                        continue
                    if not is_element(int(state.atom_types[j])):
                        continue
                    nxt.add(j)
                    seen.add(j)
            frontier = nxt
            distance += 1
        return None

    # ---- Pathwise admission ----

    # ---- Two-interface path construction ----

    @property
    def path_active(self) -> bool:
        """True only when the SPECIFICATION declares two retained regions.

        Single-core prompts have one lock group and the program is vacuous, the
        same way the attachment controller is vacuous on a prompt that declares
        no interface.  Nothing here reads a drug or a task.
        """
        return (
            self._config.enabled
            and self._config.path_program
            and len(self._spec.lock_groups) >= 2
        )

    def path_target(self, rng) -> int:
        """Draw this trajectory's target path length from the declared band.

        Drawn per TRAJECTORY, never fixed per instance: one band covers every
        prompt, and a run explores it rather than pinning a single length to a
        drug.
        """
        low = int(self._config.path_length_min)
        high = max(low, int(self._config.path_length_max))
        return int(rng.integers(low, high + 1))

    def path_unsatisfied(self, state: MolecularGraph, target: int) -> bool:
        """True while the realized core-to-core path is shorter than ``target``.

        The predicate is the MEASUREMENT: ``realized_linker_length`` is the same
        function the artifact reports, so the program and the number that judges
        it cannot disagree.  A path longer than the target is left alone -- the
        program builds a linker, it does not trim one, and refusing longer states
        would reject chemistry the prior legitimately produced.
        """
        if not self.path_active:
            return False
        realized = self.realized_linker_length(state)
        return realized is not None and realized < target

    def path_permits(
        self, successor: MolecularGraph, predecessor: MolecularGraph, target: int
    ) -> tuple[bool, str]:
        """Admit only events that lengthen the path while it is short.

        This is the path analogue of attachment-first staging, and it is what
        coverage staging cannot express: with a seeded bridge both interfaces
        are already covered, so a coverage predicate is satisfied at event 0
        while the path is still one atom long.
        """
        if not self.path_active:
            return True, ""
        if not self.path_unsatisfied(predecessor, target):
            return True, ""
        before = self.realized_linker_length(predecessor)
        after = self.realized_linker_length(successor)
        if after is None:
            # The event severed the cores; a linker prompt has no such endpoint.
            return False, "path_disconnected"
        if before is not None and after <= before:
            return False, "no_path_progress"
        return True, ""

    def permits(self, successor: MolecularGraph, predecessor: MolecularGraph) -> tuple[bool, str]:
        """Admit or refuse a candidate successor; returns (ok, reason_code)."""
        if not self.active:
            return True, ""
        if self._config.restrict_interfaces:
            real = is_element(successor.atom_types)
            for slot in self._undeclared:
                row = successor.bonds[slot]
                for other in range(successor.n_atoms):
                    if other in self._locked or other == slot:
                        continue
                    if bool(real[other]) and int(row[other]) > 0:
                        return False, "undeclared_interface"
        if self._config.attachment_first:
            before = self.unsatisfied(predecessor)
            if before:
                after = self.unsatisfied(successor)
                if len(after) >= len(before):
                    return False, "no_coverage_progress"
        return True, ""

    # ---- Attachment redirection (the alpha component of Z) ----

    def redirect_target(self, state: MolecularGraph) -> int | None:
        """The least-covered unsatisfied interface; deterministic, draws no randomness."""
        pending = self.unsatisfied(state)
        if not pending:
            return None
        covered = self.coverage(state)
        return min(pending, key=lambda slot: (covered[slot], slot))

    def redirect(self, rule_name: str, action, state: MolecularGraph):
        """Re-anchor an inadmissible ``atom_insert`` onto an unsatisfied interface.

        Returns the action unchanged when redirection does not apply.  The
        payload -- element, formal charge, derived hydrogen count, bond order --
        is the prior's; only the anchor is the constraint's.
        """
        if not self.active or not self._config.redirect_attachment:
            return action
        if rule_name != "atom_insert" or not isinstance(action, AtomInsert):
            return action
        if len(action.neighbors) != 1:
            # A rootless insert has no anchor to redirect and a multi-anchor
            # insert is not produced by this sampler; leave both alone.
            return action
        anchor, order = action.neighbors[0]
        anchor = int(anchor)
        if anchor in self._interfaces:
            return action
        target = self.redirect_target(state)
        if target is None or target == anchor:
            return action
        order = int(order)
        if self._config.realize_redirected_order:
            free = int(state.implicit_h_counts[target])
            if free <= 0:
                # The declared site has no valence left; nothing legal can be
                # realized there, so leave the prior's action to be refused on
                # its own terms rather than manufacturing an illegal one.
                return action
            if order > free:
                # Constrained realization: keep the prior's element and
                # re-derive the hydrogen count for the reduced order, exactly
                # as the prior derives it from the element's valence.
                action = replace(
                    action,
                    implicit_h_count=int(action.implicit_h_count) + order - free,
                )
                order = free
        return replace(action, neighbors=((int(target), order),))


# ---- Diagnostics ----


def interface_coverage_report(
    state: MolecularGraph, spec: AttachmentSpec, locked_slots: Sequence[int]
) -> dict:
    """Per-interface coverage of one state, for the run artifact."""
    locked = frozenset(int(slot) for slot in locked_slots)
    covered = {
        slot: external_neighbour_count(state, slot, locked) for slot in spec.interfaces
    }
    return {
        "interfaces": list(spec.interfaces),
        "required": {str(slot): spec.requirement_of(slot) for slot in spec.interfaces},
        "covered": {str(slot): covered[slot] for slot in spec.interfaces},
        "satisfied": all(
            covered[slot] >= spec.requirement_of(slot) for slot in spec.interfaces
        ),
    }


__all__ = [
    "AttachmentControlConfig",
    "AttachmentController",
    "AttachmentSpec",
    "external_neighbour_count",
    "interface_coverage_report",
]
