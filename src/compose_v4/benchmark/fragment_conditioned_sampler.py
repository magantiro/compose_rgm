"""Fragment-conditioned sampling from the learned COMPOSE rewrite process.

The fragment-constrained benchmark supplies a retained core (or two, for linker
design) and asks for completions.  COMPOSE's base model is a source-AGNOSTIC
prior ``Q_theta(y | x, t)`` over legal rewrites; task conditioning is applied at
inference rather than by retraining the generator.  This module applies it as a
PATHWISE CONSTRAINT, which is the repository's native mechanism: the constraint
holds at every committed state, not only at the endpoint, and it is enforced by
deleting violating marks from the legal-event fiber rather than by scoring or
filtering endpoints afterwards.

The constraint is a REGION LOCK on the retained core.  ``MolecularGraph`` states
are slot-stable and ``smiles_to_molecular_graph`` assigns slot ``i`` to RDKit
atom ``i``, so the core's atoms occupy a known, fixed set of slots.  A successor
is admitted only when, for every locked slot, the element and formal charge are
unchanged, and for every pair of locked slots the bond order is unchanged.
Hydrogen counts are deliberately NOT locked: they are derived, and locking them
would forbid the very growth the benchmark asks for.  New atoms and new bonds
from a locked atom to a new atom are therefore free, while deleting, retyping,
recharging or re-bonding the core is refused.

This is a restriction of the learned law to its constraint-preserving
sub-fiber, sampled by bounded rejection.  Accepted events are draws from the
restricted-and-renormalised jump law; when the rejection budget is exhausted the
trajectory stops rather than committing a violating state, so no emitted
molecule ever breaks the prompt.

What this module does NOT do: it calls no oracle, no docking function and no
property objective.  QED and SA are the benchmark's own inexpensive quality
diagnostic and are computed by the official evaluator downstream, never here.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field, replace

import numpy as np
from rdkit import Chem

from compose_v4.benchmark.fragment_attachment_control import (
    AttachmentControlConfig,
    AttachmentController,
    AttachmentSpec,
)
from compose_v4.benchmark.fragment_constrained import (
    FragmentPrompt,
    FragmentTask,
    _fragment_spec,
    check_fragment_constraint,
)
from compose_v4.chem.molecular_graph import (
    MolecularGraph,
    molecular_graph_to_smiles,
    smiles_to_molecular_graph,
)
from compose_v4.chem.state import pad_molecular_graph
from compose_v4.model.time_convention import frozen_time
from compose_v4.rewrite.operators import BondDelete, BondInsert

# Slot budget for the proposal state.  This is a PADDING width, not the
# ``REPRESENTABLE_HEAVY_ATOMS`` endpoint ceiling: an unpadded (tight) graph has
# no free slot, which silently removes the whole ``atom_insert`` family from the
# legal support and would make growth impossible.
DEFAULT_PROPOSAL_SLOTS = 48


@dataclass(frozen=True)
class SamplerConfig:
    n_slots: int = DEFAULT_PROPOSAL_SLOTS
    operational_horizon: float = 16.0
    max_events: int = 32
    # Bounded rejection budget per event.  Exhausting it stops the trajectory;
    # it never commits a state that violates the region lock.
    mark_attempts_per_event: int = 24


@dataclass(frozen=True)
class PromptContext:
    """Start state plus the locked region for one benchmark prompt."""

    prompt: FragmentPrompt
    start_state: MolecularGraph
    locked_slots: tuple[int, ...]
    start_smiles: str
    # The benchmark's OWN declared attachment interfaces, in canonical slot
    # coordinates.  Always derived, so it is visible in the artifact even when
    # the controller that consumes it is switched off.
    attachment: AttachmentSpec | None = None


@dataclass
class SamplingReceipt:
    completed: int = 0
    lock_rejections: int = 0
    executor_refusals: int = 0
    budget_exhausted: int = 0
    constraint_failures: int = 0
    events: list[int] = field(default_factory=list)
    families: dict[str, int] = field(default_factory=dict)
    # Every endpoint COMPOSE actually committed, whether or not it satisfied the
    # prompt's attachment condition.  Committed states are valid and connected by
    # construction, so this list is the evidence for the chemical-validity claim
    # and must be kept separate from benchmark (strict) validity.
    committed_endpoints: list[str] = field(default_factory=list)
    # Attachment-control accounting.  All zero when the controller is off or
    # when the prompt declares no interface.
    interface_rejections: int = 0
    staging_rejections: int = 0
    redirections: int = 0
    separation_failures: int = 0
    interface_covered: int = 0
    # Realized linker length per COMMITTED endpoint of a two-core prompt, in the
    # same order as ``committed_endpoints``.  A seeded bridge means separation is
    # satisfied from the start, so this distribution -- not the separation count
    # -- is what distinguishes a designed linker from an inherited seed.  Empty
    # for single-core prompts, where the quantity is undefined.
    linker_lengths: list[int] = field(default_factory=list)
    # Events refused because they did not lengthen a path still short of its
    # target.  Zero unless the path program is active.
    path_rejections: int = 0
    # The per-trajectory target drawn from the declared band, one per attempt.
    path_targets: list[int] = field(default_factory=list)
    # Composite path transactions committed, and refused.
    path_transactions: int = 0
    path_transaction_refusals: int = 0


class FragmentConditioningError(RuntimeError):
    """The prompt could not be turned into a legal, representable start state."""


# ---- Retained core construction ----


def retained_core(fragment_smiles: str) -> tuple[Chem.Mol, tuple[int, ...]]:
    """Return the dummy-stripped core and the core-atom indices that carried a dummy.

    Dummy isotope labels are serialization markers, not addresses.  Removing the
    dummy leaves its neighbour with a free valence that RDKit fills with an
    implicit hydrogen, which is exactly the H-capped retained core.
    """
    mol = Chem.MolFromSmiles(fragment_smiles)
    if mol is None:
        raise FragmentConditioningError(f"unparseable fragment: {fragment_smiles!r}")

    # Mark the dummies' neighbours so they can be found again after the edit
    # renumbers atoms.  An attachment site is a ROLE, not a fixed index.
    editable = Chem.RWMol(mol)
    marker = 0
    for atom in editable.GetAtoms():
        if atom.GetAtomicNum() != 0:
            continue
        for nbr in atom.GetNeighbors():
            if nbr.GetAtomicNum() != 0 and nbr.GetAtomMapNum() == 0:
                marker += 1
                nbr.SetAtomMapNum(marker)

    # Turn each dummy into a HYDROGEN rather than deleting it.  Deleting the
    # atom leaves its neighbour a free valence that RDKit will not fill when the
    # neighbour's H count is pinned by bracket notation (``[C@H]``) or when the
    # neighbour is an aromatic nitrogen, which then fails to kekulize.  Capping
    # with H is also what the retained core physically is.
    for atom in editable.GetAtoms():
        if atom.GetAtomicNum() == 0:
            atom.SetAtomicNum(1)
            atom.SetIsotope(0)
            atom.SetNoImplicit(False)
            atom.SetFormalCharge(0)

    core = editable.GetMol()
    try:
        Chem.SanitizeMol(core)
        core = Chem.RemoveHs(core)
    except Exception as exc:
        raise FragmentConditioningError(f"core did not sanitize: {exc}") from exc

    sites = tuple(
        sorted(a.GetIdx() for a in core.GetAtoms() if a.GetAtomMapNum() != 0)
    )
    for atom in core.GetAtoms():
        atom.SetAtomMapNum(0)
    return core, sites


def _declared_sites(fragment: str) -> tuple[tuple[int, int], ...]:
    """(core atom index, required external heavy neighbours) for one fragment.

    Indices are positions in the DUMMY-STRIPPED core.  ``retained_core`` and
    ``_fragment_spec`` build that core by different routes -- capping the dummy
    with hydrogen and removing it, respectively -- so the agreement of their
    atom orders is asserted in ``tests/test_fragment_attachment_control.py``
    across every released fragment rather than assumed here.
    """
    return tuple(_fragment_spec(fragment).attachment_requirements)


def build_prompt_context(
    prompt: FragmentPrompt,
    *,
    config: SamplerConfig | None = None,
    control: AttachmentControlConfig | None = None,
    linker_bridge_atoms: int = 0,
) -> PromptContext:
    """Build the fixed start state and the locked slot set for one prompt.

    Single-core tasks start at the retained core itself.

    Linker design cannot start from the two cores alone: the executor only
    admits CONNECTED states, so a disconnected pair is not a legal state, and
    asking a source-agnostic prior to re-derive the second core by chance is not
    conditioning.  Something has to bridge them, and ``linker_bridge_atoms``
    chooses what.

    ``linker_bridge_atoms == 0`` joins the two declared sites with a DIRECT
    bond.  This is the original construction and it is MEASURED to be
    structurally unable to express the task: the join consumes the very hydrogen
    each declared site needed, so on four of the ten released drugs both sites
    start with zero free valence and no first event can increase coverage.
    Every emitted "linker" is then zero atoms long, which the benchmark's own
    endpoint test cannot detect because each core satisfies the other core's
    attachment requirement.  Retained only so the invalidated rows stay
    reproducible.

    ``linker_bridge_atoms >= 1`` inserts that many UNLOCKED carbons in a chain
    between the two declared sites.  The cores are then separated by
    construction, each declared site's external neighbour is a linker atom
    rather than the opposite core, and the chain is free to grow, shrink and
    change element because it belongs to neither retained region.  One atom is
    the minimum that makes the task expressible; it is a seed to design from,
    not a linker, so any row built this way must report the realized linker
    length beside it.
    """
    config = config or SamplerConfig()
    if linker_bridge_atoms < 0:
        raise FragmentConditioningError(
            f"linker_bridge_atoms must be non-negative, got {linker_bridge_atoms}"
        )

    if prompt.task in (FragmentTask.LINKER_DESIGN, FragmentTask.SCAFFOLD_MORPHING):
        if len(prompt.fragments) != 2:
            raise FragmentConditioningError(
                f"{prompt.task.value} expects two fragments, got {len(prompt.fragments)}"
            )
        left, left_sites = retained_core(prompt.fragments[0])
        right, right_sites = retained_core(prompt.fragments[1])
        if not left_sites or not right_sites:
            raise FragmentConditioningError("linker fragment declares no attachment site")
        combined = Chem.RWMol(Chem.CombineMols(left, right))
        offset = left.GetNumAtoms()
        join = (left_sites[0], right_sites[0] + offset)
        # Each core was capped with H where its dummy was.  The join bond
        # consumes that valence, so the cap has to come back off or the atom is
        # over-valent.  Pin the resulting H count explicitly: leaving it implicit
        # lets RDKit re-derive the cap and fail sanitization again.
        for index in join:
            atom = combined.GetAtomWithIdx(index)
            total_h = atom.GetTotalNumHs()
            if total_h < 1:
                raise FragmentConditioningError(
                    f"attachment atom {index} has no hydrogen to replace with the join bond"
                )
            atom.SetNoImplicit(True)
            atom.SetNumExplicitHs(total_h - 1)
        if linker_bridge_atoms == 0:
            combined.AddBond(join[0], join[1], Chem.BondType.SINGLE)
            bridge_slots: tuple[int, ...] = ()
            join_is_direct = True
        else:
            # Chain the seeded carbons between the two declared sites.  They are
            # appended after both cores, so core slot indices are unchanged and
            # ``group_bounds`` below still addresses the retained regions.
            previous = join[0]
            for _ in range(linker_bridge_atoms):
                index = combined.AddAtom(Chem.Atom(6))
                combined.AddBond(previous, index, Chem.BondType.SINGLE)
                previous = index
            combined.AddBond(previous, join[1], Chem.BondType.SINGLE)
            bridge_slots = tuple(
                range(
                    left.GetNumAtoms() + right.GetNumAtoms(),
                    left.GetNumAtoms() + right.GetNumAtoms() + linker_bridge_atoms,
                )
            )
            join_is_direct = False
        declared = [
            (site, count) for site, count in _declared_sites(prompt.fragments[0])
        ] + [
            (site + offset, count)
            for site, count in _declared_sites(prompt.fragments[1])
        ]
        group_bounds = (
            tuple(range(left.GetNumAtoms())),
            tuple(range(left.GetNumAtoms(), left.GetNumAtoms() + right.GetNumAtoms())),
        )
        # Only a DIRECT join has to be released from the lock; a seeded bridge is
        # already outside every locked region, so there is nothing to release.
        join_pair = join if join_is_direct else None
        start = combined.GetMol()
        try:
            Chem.SanitizeMol(start)
        except Exception as exc:
            raise FragmentConditioningError(
                f"joined linker start did not sanitize: {exc}"
            ) from exc
        # The bridge carbons are deliberately NOT locked: they are the linker the
        # generator is being asked to design.
        locked = tuple(range(left.GetNumAtoms() + right.GetNumAtoms()))
        assert all(slot not in locked for slot in bridge_slots)
    else:
        core, _sites = retained_core(prompt.fragments[0])
        start = core
        locked = tuple(range(core.GetNumAtoms()))
        declared = list(_declared_sites(prompt.fragments[0]))
        group_bounds = (tuple(range(core.GetNumAtoms())),)
        join_pair = None

    start_smiles = Chem.MolToSmiles(start)
    # Re-parse from canonical SMILES so slot i corresponds to RDKit atom i of the
    # SAME molecule object the lock indices are read from.
    canonical = Chem.MolFromSmiles(start_smiles)
    if canonical is None:
        raise FragmentConditioningError(f"start molecule lost on round trip: {start_smiles!r}")
    order = canonical.GetSubstructMatch(start)
    if len(order) != start.GetNumAtoms():
        raise FragmentConditioningError("could not map start molecule onto its canonical form")
    locked_canonical = tuple(sorted(order[i] for i in locked))

    try:
        graph = smiles_to_molecular_graph(start_smiles)
    except Exception as exc:
        raise FragmentConditioningError(
            f"start state is not representable by this model: {exc}"
        ) from exc
    if len(graph.atom_types) > config.n_slots:
        raise FragmentConditioningError(
            f"start state needs {len(graph.atom_types)} slots, budget is {config.n_slots}"
        )
    padded = pad_molecular_graph(graph, config.n_slots)

    # ``order[i]`` is the canonical slot of start-molecule atom ``i``; the
    # declared interfaces are addressed in the same coordinates as the lock.
    interfaces = tuple(sorted(order[site] for site, _count in declared))
    requirements = tuple(
        sorted((order[site], int(count)) for site, count in declared)
    )
    lock_groups = tuple(
        frozenset(order[i] for i in bounds) for bounds in group_bounds
    )
    release = control is not None and control.enabled and control.release_linker_join
    released_pairs = (
        frozenset({frozenset({order[join_pair[0]], order[join_pair[1]]})})
        if join_pair is not None and release
        else frozenset()
    )
    attachment = AttachmentSpec(
        interfaces=interfaces,
        requirements=requirements,
        lock_groups=lock_groups,
        released_pairs=released_pairs,
    )
    return PromptContext(
        prompt=prompt,
        start_state=padded,
        locked_slots=locked_canonical,
        start_smiles=start_smiles,
        attachment=attachment,
    )


# ---- Pathwise region lock ----


class RegionLock:
    """Admits exactly the successors that leave the retained core untouched."""

    def __init__(
        self,
        state: MolecularGraph,
        locked_slots: Sequence[int],
        *,
        released_pairs: frozenset[frozenset[int]] = frozenset(),
    ) -> None:
        self._slots = tuple(locked_slots)
        self._types = {i: int(state.atom_types[i]) for i in self._slots}
        self._charges = {i: int(state.formal_charges[i]) for i in self._slots}
        # A RELEASED pair is one this adapter constructed rather than one the
        # prompt declared: the linker join bond exists only because the
        # executor refuses a disconnected state, so pinning its order would
        # make a genuine linker unreachable.  Element and charge stay locked on
        # both of its endpoints; only the bond between them is free.
        self._released = frozenset(
            frozenset(int(x) for x in pair) for pair in released_pairs
        )
        self._bonds = {
            (i, j): int(state.bonds[i][j])
            for index, i in enumerate(self._slots)
            for j in self._slots[index + 1 :]
            if frozenset({i, j}) not in self._released
        }

    @property
    def locked_slots(self) -> tuple[int, ...]:
        return self._slots

    @property
    def released_pairs(self) -> frozenset[frozenset[int]]:
        return self._released

    def permits(self, successor: MolecularGraph) -> bool:
        for slot, expected in self._types.items():
            if int(successor.atom_types[slot]) != expected:
                return False
        for slot, expected in self._charges.items():
            if int(successor.formal_charges[slot]) != expected:
                return False
        for (i, j), expected in self._bonds.items():
            if int(successor.bonds[i][j]) != expected:
                return False
        return True


# ---- Conditioned sampling ----



def _attempt_path_transaction(
    model, system, state, controller, lock, rng, receipt, *, payload_draws: int = 32
):
    """Execute the three-event path-lengthening transaction, or return None.

    ATOMIC: all three events must execute and the final state must satisfy the
    region lock and the interface controller, or nothing is committed and the
    caller keeps the state it had.  Each constituent is a move the proposal law
    can rank -- measured IN_SUPPORT on every constituent of every drug where the
    transaction executes -- so this is an ACCELERATION: it performs in sequence
    what the prior can propose but essentially never proposes in order.  With
    the program off, 1,440 events across 120 rollouts never once lengthened a
    path.

    The prior chooses WHAT to insert and the constraint chooses WHERE, which is
    the same division the attachment redirection already uses: the inserted
    atom's payload is taken from an ``atom_insert`` the model itself proposes at
    the chosen site, and the transaction is abandoned rather than invented if
    the model offers none.
    """
    sites = controller.path_transaction_sites(state)
    if sites is None:
        return None
    path_atom, far_anchor, free_slot = sites

    # ---- payload from the prior, never invented here ----
    payload = None
    for _ in range(payload_draws):
        try:
            mark = model.sample_rewrite_mark(state, 0.0, rng)
        except Exception:  # noqa: BLE001, S112 -- a refused draw is simply not a payload
            continue
        if mark.rule_name != "atom_insert":
            continue
        action = mark.action
        if not any(n[0] == path_atom for n in getattr(action, "neighbors", ())):
            continue
        payload = action
        break
    if payload is None:
        return None

    step_insert = replace(payload, slot=free_slot, neighbors=((path_atom, 1),))
    try:
        after_insert = system.apply(state, "atom_insert", step_insert)
        after_close = system.apply(
            after_insert, "bond_insert", BondInsert(a=free_slot, b=far_anchor, order=1)
        )
        after_open = system.apply(
            after_close, "bond_delete", BondDelete(a=path_atom, b=far_anchor)
        )
    except Exception:  # noqa: BLE001
        receipt.path_transaction_refusals += 1
        return None

    if not lock.permits(after_open):
        receipt.path_transaction_refusals += 1
        return None
    admitted, _reason = controller.permits(after_open, state)
    if not admitted:
        receipt.path_transaction_refusals += 1
        return None
    before = controller.realized_linker_length(state)
    after = controller.realized_linker_length(after_open)
    if after is None or (before is not None and after <= before):
        receipt.path_transaction_refusals += 1
        return None
    receipt.path_transactions += 1
    return after_open


def sample_completion(
    model,
    system,
    context: PromptContext,
    rng: np.random.Generator,
    *,
    config: SamplerConfig | None = None,
    receipt: SamplingReceipt | None = None,
    control: AttachmentControlConfig | None = None,
) -> str | None:
    """Sample ONE completion of ``context``'s prompt from the learned process.

    Returns the canonical SMILES of a completion that satisfies the benchmark
    constraint, or ``None`` when the trajectory produced nothing admissible.
    ``None`` is a genuine failed attempt and must be scored as one: silently
    retrying until something lands would report a success rate the sampler does
    not have.
    """
    config = config or SamplerConfig()
    receipt = receipt if receipt is not None else SamplingReceipt()
    control = control or AttachmentControlConfig()
    spec = context.attachment or AttachmentSpec((), (), (frozenset(context.locked_slots),))
    controller = AttachmentController(spec, context.locked_slots, control)
    # The path target is drawn ONCE per trajectory, from the declared band, and
    # only when the specification declares two retained regions.  Drawing it
    # here rather than per event keeps one linker goal per trajectory; drawing
    # it at all is skipped when the program is vacuous, so a single-core prompt
    # keeps a bit-identical RNG stream.
    path_target = controller.path_target(rng) if controller.path_active else 0
    lock = RegionLock(
        context.start_state,
        context.locked_slots,
        released_pairs=spec.released_pairs,
    )

    state = context.start_state
    operational_time = 0.0
    events = 0

    while events < config.max_events and operational_time < config.operational_horizon:
        time_feature = frozen_time(operational_time)
        # While the path is short of this trajectory's target, try the composite
        # transaction FIRST.  A single event cannot lengthen the path -- measured,
        # 0 of 30 lock-passing events did -- so without this the program has
        # nothing to admit and the trajectory stalls at the seeded length.
        if controller.path_unsatisfied(state, path_target):
            lengthened = _attempt_path_transaction(
                model, system, state, controller, lock, rng, receipt
            )
            if lengthened is not None:
                state = lengthened
                events += 1
                continue
        accepted = None
        for _ in range(config.mark_attempts_per_event):
            try:
                mark = model.sample_rewrite_mark(state, time_feature, rng)
            except Exception:  # noqa: BLE001
                receipt.executor_refusals += 1
                continue
            # Attachment (``alpha``) is the controller's degree of freedom: the
            # prior keeps the payload, the declared constraint picks the site.
            # ``redirect`` is the identity whenever the controller is inactive,
            # so the off path executes exactly the mark the prior sampled.
            action = controller.redirect(mark.rule_name, mark.action, state)
            if action is not mark.action:
                receipt.redirections += 1
            try:
                successor = system.apply(state, mark.rule_name, action)
            except Exception:  # noqa: BLE001
                receipt.executor_refusals += 1
                continue
            if not lock.permits(successor):
                receipt.lock_rejections += 1
                continue
            admitted, reason = controller.permits(successor, state)
            if admitted:
                admitted, reason = controller.path_permits(
                    successor, state, path_target
                )
            if not admitted:
                if reason == "undeclared_interface":
                    receipt.interface_rejections += 1
                elif reason in ("no_path_progress", "path_disconnected"):
                    receipt.path_rejections += 1
                else:
                    receipt.staging_rejections += 1
                continue
            accepted = (mark, successor)
            break

        if accepted is None:
            receipt.budget_exhausted += 1
            break

        mark, successor = accepted
        receipt.families[mark.rule_name] = receipt.families.get(mark.rule_name, 0) + 1
        hazard = float(getattr(mark, "total_hazard", 0.0) or 0.0)
        # Reference holding time.  Only the endpoint enters the benchmark
        # metrics, so this sets trajectory LENGTH, not any reported number.
        operational_time += (
            float(rng.exponential(1.0 / hazard)) if hazard > 0.0 else config.operational_horizon
        )
        state = successor
        events += 1

    if controller.path_active:
        receipt.path_targets.append(path_target)
    receipt.events.append(events)
    if events == 0:
        return None

    smiles = molecular_graph_to_smiles(state)
    if not smiles:
        return None
    receipt.committed_endpoints.append(smiles)
    realized_length = controller.realized_linker_length(state)
    if realized_length is not None:
        receipt.linker_lengths.append(realized_length)
    if controller.active and controller.all_interfaces_covered(state):
        receipt.interface_covered += 1
    result = check_fragment_constraint(context.prompt, smiles)
    if not result.satisfied or result.canonical_smiles is None:
        receipt.constraint_failures += 1
        return None
    # A linker prompt is CONSTRUCTED with its two cores directly bonded, and
    # the benchmark's endpoint test cannot see that: each core satisfies the
    # other's attachment requirement, so a zero-atom "linker" passes it.  This
    # is the one place the adapter is STRICTER than the published check.
    if (
        control.enabled
        and control.require_separated_cores
        and not controller.cores_are_separated(state)
    ):
        receipt.separation_failures += 1
        return None
    receipt.completed += 1
    return result.canonical_smiles


__all__ = [
    "DEFAULT_PROPOSAL_SLOTS",
    "AttachmentControlConfig",
    "FragmentConditioningError",
    "PromptContext",
    "RegionLock",
    "SamplerConfig",
    "SamplingReceipt",
    "build_prompt_context",
    "retained_core",
    "sample_completion",
]
