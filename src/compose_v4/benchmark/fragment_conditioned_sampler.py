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
from dataclasses import dataclass, field

import numpy as np
from rdkit import Chem

from compose_v4.benchmark.fragment_constrained import (
    FragmentPrompt,
    FragmentTask,
    check_fragment_constraint,
)
from compose_v4.chem.molecular_graph import (
    MolecularGraph,
    molecular_graph_to_smiles,
    smiles_to_molecular_graph,
)
from compose_v4.chem.state import pad_molecular_graph
from compose_v4.model.time_convention import frozen_time

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


def build_prompt_context(
    prompt: FragmentPrompt, *, config: SamplerConfig | None = None
) -> PromptContext:
    """Build the fixed start state and the locked slot set for one prompt.

    Single-core tasks start at the retained core itself.  Linker design starts at
    the two cores joined by one bond between their declared attachment atoms:
    the executor only admits connected states, so a disconnected pair is not a
    legal state, and asking a source-agnostic prior to re-derive the second core
    by chance is not conditioning.  Both cores are locked, so every committed
    state retains both and only the join region is free to be edited.
    """
    config = config or SamplerConfig()

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
        combined.AddBond(join[0], join[1], Chem.BondType.SINGLE)
        start = combined.GetMol()
        try:
            Chem.SanitizeMol(start)
        except Exception as exc:
            raise FragmentConditioningError(
                f"joined linker start did not sanitize: {exc}"
            ) from exc
        locked = tuple(range(left.GetNumAtoms() + right.GetNumAtoms()))
    else:
        core, _sites = retained_core(prompt.fragments[0])
        start = core
        locked = tuple(range(core.GetNumAtoms()))

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
    return PromptContext(
        prompt=prompt,
        start_state=padded,
        locked_slots=locked_canonical,
        start_smiles=start_smiles,
    )


# ---- Pathwise region lock ----


class RegionLock:
    """Admits exactly the successors that leave the retained core untouched."""

    def __init__(self, state: MolecularGraph, locked_slots: Sequence[int]) -> None:
        self._slots = tuple(locked_slots)
        self._types = {i: int(state.atom_types[i]) for i in self._slots}
        self._charges = {i: int(state.formal_charges[i]) for i in self._slots}
        self._bonds = {
            (i, j): int(state.bonds[i][j])
            for index, i in enumerate(self._slots)
            for j in self._slots[index + 1 :]
        }

    @property
    def locked_slots(self) -> tuple[int, ...]:
        return self._slots

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


def sample_completion(
    model,
    system,
    context: PromptContext,
    rng: np.random.Generator,
    *,
    config: SamplerConfig | None = None,
    receipt: SamplingReceipt | None = None,
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
    lock = RegionLock(context.start_state, context.locked_slots)

    state = context.start_state
    operational_time = 0.0
    events = 0

    while events < config.max_events and operational_time < config.operational_horizon:
        time_feature = frozen_time(operational_time)
        accepted = None
        for _ in range(config.mark_attempts_per_event):
            try:
                mark = model.sample_rewrite_mark(state, time_feature, rng)
            except Exception:  # noqa: BLE001
                receipt.executor_refusals += 1
                continue
            try:
                successor = system.apply(state, mark.rule_name, mark.action)
            except Exception:  # noqa: BLE001
                receipt.executor_refusals += 1
                continue
            if not lock.permits(successor):
                receipt.lock_rejections += 1
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

    receipt.events.append(events)
    if events == 0:
        return None

    smiles = molecular_graph_to_smiles(state)
    if not smiles:
        return None
    receipt.committed_endpoints.append(smiles)
    result = check_fragment_constraint(context.prompt, smiles)
    if not result.satisfied or result.canonical_smiles is None:
        receipt.constraint_failures += 1
        return None
    receipt.completed += 1
    return result.canonical_smiles


__all__ = [
    "DEFAULT_PROPOSAL_SLOTS",
    "FragmentConditioningError",
    "PromptContext",
    "RegionLock",
    "SamplerConfig",
    "SamplingReceipt",
    "build_prompt_context",
    "retained_core",
    "sample_completion",
]
