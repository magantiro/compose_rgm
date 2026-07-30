"""Production canonical-successor kernel for the factorized COMPOSE editing model.

The trained model defines a normalized law over executable rewrite marks.  This
module performs the one derivation used by every molecular experiment:

    mark coordinates
        -> executable production actions
        -> production executor
        -> canonical molecular identity
        -> segmented sum over each successor fiber.

The implementation deliberately reuses the model's action tables and family
factorization.  It does not construct a second chemistry enumerator.  The slow
dictionary implementation remains a test-only oracle and is never imported
here.

RingCore-V1 scope
-----------------
The production editing process uses compositional cycle close/open and disables
``ring_system_grow``.  Supporting the legacy macro would require enumerating its
factorized electronic decoder, not merely its template head.  This module
therefore fails closed if that macro is enabled instead of silently scoring an
incomplete fiber.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from math import exp, isfinite
from typing import Any

import numpy as np
import torch

from compose_v4.chem.molecular_graph import NULL_IDX, MolecularGraph
from compose_v4.experiments.successor_kernel import (
    CanonicalSuccessor,
    KernelIdentity,
    SuccessorBatch,
    SupportSignature,
    validate_successor_batch,
)
from compose_v4.model.factorized_tracelet_rate_model import (
    MARK_RULE_NAMES,
    MARK_RULE_TO_INDEX,
    FactorizedMarkBatch,
    FactorizedTraceletRateModel,
    _masked_family_logits,
    prepare_factorized_mark_batch,
)
from compose_v4.model.segmented_successor import segmented_successor_logprobs
from compose_v4.rewrite.kernel import (
    RewriteSystem,
    canonical_state_key,
    de_novo_rewrite_system,
)
from compose_v4.rewrite.operators import (
    AtomDelete,
    AtomInsert,
    AtomRestate,
    BondDelete,
    BondInsert,
    BondReorder,
    BondReroute,
)


class ProductionSuccessorKernelError(RuntimeError):
    """The production marked law could not be derived exactly."""


@dataclass(frozen=True)
class ScoredRewriteMark:
    """One legal production mark and its normalized model log probability."""

    family_name: str
    table_name: str
    executor_rule_name: str
    action: Any
    coordinate: tuple[int, ...]
    log_probability: float

    @property
    def probability(self) -> float:
        return exp(self.log_probability)


@dataclass(frozen=True)
class FactorizedMarkedLaw:
    """The complete normalized marked law out of one state."""

    source_key: str
    marks: tuple[ScoredRewriteMark, ...]
    total_hazard: float
    family_log_probabilities: tuple[float, ...]
    enabled_families: tuple[str, ...]

    @property
    def total_probability(self) -> float:
        return float(sum(mark.probability for mark in self.marks))


@dataclass(frozen=True)
class SuccessorKernelDiagnostics:
    """Geometry and raw/productive mass for one canonical successor row."""

    raw_mark_count: int
    productive_mark_count: int
    virtual_self_mark_count: int
    canonical_successor_count: int
    raw_productive_mass: float
    virtual_self_mass: float
    alias_multiplicities: tuple[int, ...]
    marks_by_family: dict[str, int]
    productive_marks_by_family: dict[str, int]


@dataclass(frozen=True)
class SuccessorKernelResult:
    """A validated successor batch together with its forensic diagnostics."""

    batch: SuccessorBatch
    diagnostics: SuccessorKernelDiagnostics
    marked_law: FactorizedMarkedLaw


_TABLE_FAMILIES = (
    ("atom_insert", "grow_root"),
    ("atom_insert", "grow_connected"),
    ("atom_delete", "atom_delete"),
    ("atom_restate", "atom_restate"),
    ("bond_reorder", "bond_reorder"),
    ("bond_reroute", "bond_reroute"),
    ("cycle_insert", "cycle_insert"),
    ("cycle_attach", "cycle_attach"),
    ("ring_system_delete", "ring_system_delete"),
    ("ring_system_restate", "ring_system_restate"),
)


def _one_state_batch(
    model: FactorizedTraceletRateModel,
    state: MolecularGraph,
    time: float,
    *,
    prepared_batch: FactorizedMarkBatch | None,
) -> FactorizedMarkBatch:
    if prepared_batch is None:
        capabilities = model.operator_capabilities
        prepared_batch = prepare_factorized_mark_batch(
            (state,),
            (float(time),),
            (None,),
            (None,),
            (0.0,),
            use_aromatic_bond_view=True,
            ring_catalog=model.ring_catalog,
            compute_ring_grow_support=capabilities.compute_ring_grow_support,
            compute_ring_restates=capabilities.compute_ring_restates,
            compute_cyclic_graft=capabilities.compute_cyclic_graft,
            compute_ring_opening=capabilities.compute_ring_opening,
            compute_ring_system_delete=capabilities.compute_ring_system_delete,
        )
    if prepared_batch.batch_size != 1:
        raise ProductionSuccessorKernelError(
            f"a one-state kernel row received batch_size={prepared_batch.batch_size}"
        )
    if prepared_batch.states[0] is not state:
        expected = canonical_state_key(state)
        observed = canonical_state_key(prepared_batch.states[0])
        if observed != expected:
            raise ProductionSuccessorKernelError(
                f"prepared batch state {observed!r} does not match requested state {expected!r}"
            )
    return replace(
        prepared_batch,
        times=torch.tensor((float(time),), dtype=prepared_batch.times.dtype),
    ).to(model.device)


def _default_kernel_identity(
    model: FactorizedTraceletRateModel,
) -> KernelIdentity:
    """Capability-bearing fallback for direct callers outside the shared evaluator."""

    capabilities = model.operator_capabilities
    flags = (
        ("enable_ring_restates", capabilities.compute_ring_restates),
        ("enable_cyclic_graft", capabilities.compute_cyclic_graft),
        ("enable_heteroatom_scan", model.enable_heteroatom_scan),
        ("enable_ring_opening", capabilities.compute_ring_opening),
        ("enable_cycle_ops", model.enable_cycle_ops),
        ("enable_ring_system_delete", capabilities.compute_ring_system_delete),
    )
    if model.enable_cycle_ops:
        ringcore_configuration = "ringcore_v1_compositional_cycle_ops"
    elif model.enable_ring_grow_macro:
        ringcore_configuration = "legacy_ring_grow_macro"
    else:
        ringcore_configuration = "cycle_and_ring_grow_disabled"
    if not model.enable_ring_system_delete:
        ringcore_configuration += ":ring_system_delete_disabled"
    return KernelIdentity(
        implementation="factorized_ringcore_segmented_pushforward",
        support_signature=SupportSignature(
            capability_flags=flags,
            canonicalizer_version="canonical_state_key",
            embedded_jump_chain_policy="fixed_step_embedded_jump_chain",
            ringcore_configuration=ringcore_configuration,
        ),
    )


def _coordinate_action(
    model: FactorizedTraceletRateModel,
    state: MolecularGraph,
    batch: FactorizedMarkBatch,
    *,
    family_name: str,
    table_name: str,
    coordinate: tuple[int, ...],
) -> tuple[str, Any]:
    null_slots = tuple(int(v) for v in np.flatnonzero(state.atom_types == NULL_IDX))

    if table_name == "grow_root":
        if not null_slots:
            raise ProductionSuccessorKernelError("root insertion has no persistent null slot")
        (atom_index,) = coordinate
        atom_type = int(model.atom_vocabulary.element_of(atom_index))
        return "atom_insert", AtomInsert(
            null_slots[0],
            atom_type,
            0,
            int(model.cnof_valences[atom_index]),
            (),
        )
    if table_name == "grow_connected":
        if not null_slots:
            raise ProductionSuccessorKernelError("connected insertion has no persistent null slot")
        neighbor, order_index, atom_index = coordinate
        order = order_index + 1
        atom_type = int(model.atom_vocabulary.element_of(atom_index))
        return "atom_insert", AtomInsert(
            null_slots[0],
            atom_type,
            0,
            int(model.cnof_valences[atom_index]) - order,
            ((neighbor, order),),
        )
    if table_name == "atom_delete":
        (vertex,) = coordinate
        return "atom_delete", AtomDelete(vertex)
    if table_name == "atom_restate":
        vertex, atom_index = coordinate
        atom_type = int(model.atom_vocabulary.element_of(atom_index))
        bond_valence = sum(
            _bond_valence_deltas()[int(order)]
            for order in state.bonds[vertex]
        )
        # Production currently stores formal charge but does not design it; all
        # vocabulary rows therefore use charge zero in this head.
        return "atom_restate", AtomRestate(
            vertex,
            atom_type,
            0,
            int(model.cnof_valences[atom_index]) - bond_valence,
        )
    if table_name == "bond_reorder":
        a, b, order_index = coordinate
        return "bond_reorder", BondReorder(a, b, order_index + 1)
    if table_name == "bond_reroute":
        moved, target = coordinate
        removed = int(batch.graft_remove_neighbors[0, moved, target])
        if removed < 0:
            raise ProductionSuccessorKernelError(
                f"legal graft coordinate {(moved, target)} lacks a removed neighbor"
            )
        return "bond_reroute", BondReroute(
            a=moved,
            b=removed,
            u=moved,
            v=target,
        )
    if table_name == "cycle_insert":
        if not model.enable_cycle_ops:
            raise ProductionSuccessorKernelError(
                "RingCore production scoring requires compositional cycle operations"
            )
        a, b, order_index = coordinate
        return "bond_insert", BondInsert(a, b, order_index + 1)
    if table_name == "cycle_attach":
        if not model.enable_cycle_ops:
            raise ProductionSuccessorKernelError(
                "RingCore production scoring requires compositional cycle operations"
            )
        a, b = coordinate
        return "bond_delete", BondDelete(a, b)
    if table_name == "ring_system_delete":
        (action_index,) = coordinate
        candidates = batch.ring_delete_actions
        if candidates is None or action_index >= len(candidates[0]):
            raise ProductionSuccessorKernelError(
                "ring-system-delete coordinate is not aligned to its production candidates"
            )
        return "ring_system_delete", candidates[0][action_index]
    if table_name == "ring_system_restate":
        (action_index,) = coordinate
        if action_index >= len(batch.ring_restate_actions[0]):
            raise ProductionSuccessorKernelError(
                "ring-system-restate coordinate is not aligned to its production candidates"
            )
        return "ring_system_restate", batch.ring_restate_actions[0][action_index]
    raise ProductionSuccessorKernelError(
        f"no production action decoder for {family_name}/{table_name}"
    )


def _bond_valence_deltas() -> tuple[int, ...]:
    # Imported lazily to keep the coordinate decoder tied to the same table the
    # model uses for aromatic/Kekule bond classes.
    from compose_v4.chem.molecular_graph import BOND_CLASS_TO_H_CHANGE

    return tuple(int(value) for value in BOND_CLASS_TO_H_CHANGE)


def enumerate_factorized_marked_law(
    model: FactorizedTraceletRateModel,
    state: MolecularGraph,
    time: float,
    *,
    prepared_batch: FactorizedMarkBatch | None = None,
    normalization_tolerance: float = 2e-5,
) -> FactorizedMarkedLaw:
    """Return every legal RingCore mark with its exact normalized model mass."""

    if model.enable_ring_grow_macro:
        raise ProductionSuccessorKernelError(
            "the legacy ring_system_grow macro is enabled; RingCore-V1 requires it disabled, and "
            "scoring only its template head would omit the electronic decoder"
        )
    if not model.enable_cycle_ops:
        raise ProductionSuccessorKernelError(
            "compositional cycle close/open are disabled; this is not the RingCore-V1 editing process"
        )

    batch = _one_state_batch(
        model,
        state,
        time,
        prepared_batch=prepared_batch,
    )
    with torch.no_grad():
        node, global_state, pair = model._encode_batch(batch)
        masks, logits, action_log_z = model._action_tables(
            batch,
            node,
            global_state,
            pair,
            require_exact_ring_support=False,
        )
        enabled = torch.isfinite(action_log_z)
        family_logits = _masked_family_logits(
            model._family_base_logits(batch, global_state),
            action_log_z,
            enabled,
            rate_factorization=model.rate_factorization,
        )
        family_log_probabilities = torch.log_softmax(family_logits, dim=-1)[0]
        has_legal_mark = bool(enabled[0].any())
        total_hazard = float(
            torch.nn.functional.softplus(model.total_hazard_head(global_state)[0, 0])
            if has_legal_mark
            else 0.0
        )

    if bool(masks["ring_system_grow"][0].any()):
        raise ProductionSuccessorKernelError(
            "ring_system_grow has legal mass despite the production macro being disabled"
        )

    marks: list[ScoredRewriteMark] = []
    for family_name, table_name in _TABLE_FAMILIES:
        family_index = MARK_RULE_TO_INDEX[family_name]
        family_normalizer = float(action_log_z[0, family_index])
        table_mask = masks[table_name][0]
        table_logits = logits[table_name][0]
        coordinates = torch.nonzero(table_mask, as_tuple=False)
        for raw_coordinate in coordinates:
            coordinate = tuple(int(value) for value in raw_coordinate)
            executor_rule_name, action = _coordinate_action(
                model,
                state,
                batch,
                family_name=family_name,
                table_name=table_name,
                coordinate=coordinate,
            )
            raw_logit = float(table_logits[coordinate])
            log_probability = (
                float(family_log_probabilities[family_index])
                + raw_logit
                - family_normalizer
            )
            if not isfinite(log_probability):
                raise ProductionSuccessorKernelError(
                    f"legal {family_name} coordinate {coordinate} has non-finite log probability"
                )
            marks.append(
                ScoredRewriteMark(
                    family_name=family_name,
                    table_name=table_name,
                    executor_rule_name=executor_rule_name,
                    action=action,
                    coordinate=coordinate,
                    log_probability=log_probability,
                )
            )

    law = FactorizedMarkedLaw(
        source_key=canonical_state_key(state),
        marks=tuple(marks),
        total_hazard=total_hazard,
        family_log_probabilities=tuple(
            float(value) for value in family_log_probabilities
        ),
        enabled_families=tuple(
            MARK_RULE_NAMES[index]
            for index, value in enumerate(enabled[0])
            if bool(value)
        ),
    )
    if marks and abs(law.total_probability - 1.0) > normalization_tolerance:
        raise ProductionSuccessorKernelError(
            "factorized marked law is not normalized: "
            f"{law.total_probability:.12g} over {len(marks)} marks"
        )
    if not marks and total_hazard != 0.0:
        raise ProductionSuccessorKernelError(
            "a terminal marked fiber has nonzero total hazard"
        )
    return law


def canonical_successor_result(
    model: FactorizedTraceletRateModel,
    state: MolecularGraph,
    time: float,
    *,
    identity: KernelIdentity | None = None,
    system: RewriteSystem | None = None,
    prepared_batch: FactorizedMarkBatch | None = None,
) -> SuccessorKernelResult:
    """Execute and aggregate the complete marked law into molecular successors."""

    marked_law = enumerate_factorized_marked_law(
        model,
        state,
        time,
        prepared_batch=prepared_batch,
    )
    runtime = system or de_novo_rewrite_system()
    source_key = marked_law.source_key
    successor_states: dict[str, MolecularGraph] = {}
    successor_marks: dict[str, list[ScoredRewriteMark]] = {}
    virtual_marks: list[ScoredRewriteMark] = []
    marks_by_family = {name: 0 for name in MARK_RULE_NAMES}
    productive_by_family = {name: 0 for name in MARK_RULE_NAMES}

    for mark in marked_law.marks:
        marks_by_family[mark.family_name] += 1
        try:
            successor = runtime.apply(
                state,
                mark.executor_rule_name,
                mark.action,
            )
        except Exception as error:
            raise ProductionSuccessorKernelError(
                "the model's legal mask admitted an action rejected by the production executor: "
                f"{mark.family_name}/{mark.executor_rule_name} {mark.action!r}"
            ) from error
        key = canonical_state_key(successor)
        if key == source_key:
            virtual_marks.append(mark)
            continue
        productive_by_family[mark.family_name] += 1
        successor_states.setdefault(key, successor)
        successor_marks.setdefault(key, []).append(mark)

    ordered_keys = tuple(sorted(successor_marks))
    identity = identity or _default_kernel_identity(model)
    if not ordered_keys:
        batch = SuccessorBatch(
            source_key=source_key,
            successors=(),
            identity=identity,
            virtual_mass=float(sum(mark.probability for mark in virtual_marks)),
        )
        diagnostics = SuccessorKernelDiagnostics(
            raw_mark_count=len(marked_law.marks),
            productive_mark_count=0,
            virtual_self_mark_count=len(virtual_marks),
            canonical_successor_count=0,
            raw_productive_mass=0.0,
            virtual_self_mass=batch.virtual_mass,
            alias_multiplicities=(),
            marks_by_family=marks_by_family,
            productive_marks_by_family=productive_by_family,
        )
        validate_successor_batch(batch)
        return SuccessorKernelResult(batch, diagnostics, marked_law)

    key_to_group = {key: index for index, key in enumerate(ordered_keys)}
    productive_marks = tuple(
        mark for key in ordered_keys for mark in successor_marks[key]
    )
    candidate_log_probability = torch.tensor(
        [mark.log_probability for mark in productive_marks],
        dtype=torch.float64,
    )
    candidate_to_example = torch.zeros(
        len(productive_marks),
        dtype=torch.long,
    )
    candidate_to_successor = torch.tensor(
        [
            key_to_group[key]
            for key in ordered_keys
            for _ in successor_marks[key]
        ],
        dtype=torch.long,
    )
    successor_log_probability = segmented_successor_logprobs(
        candidate_log_probability,
        candidate_to_example,
        candidate_to_successor,
        n_examples=1,
        n_successors=len(ordered_keys),
    )
    successors = tuple(
        CanonicalSuccessor(
            key=key,
            state=successor_states[key],
            probability=float(successor_log_probability[index].exp()),
            alias_count=len(successor_marks[key]),
        )
        for index, key in enumerate(ordered_keys)
    )
    virtual_mass = float(sum(mark.probability for mark in virtual_marks))
    raw_productive_mass = float(
        sum(mark.probability for mark in productive_marks)
    )
    batch = SuccessorBatch(
        source_key=source_key,
        successors=successors,
        identity=identity,
        virtual_mass=virtual_mass,
    )
    diagnostics = SuccessorKernelDiagnostics(
        raw_mark_count=len(marked_law.marks),
        productive_mark_count=len(productive_marks),
        virtual_self_mark_count=len(virtual_marks),
        canonical_successor_count=len(successors),
        raw_productive_mass=raw_productive_mass,
        virtual_self_mass=virtual_mass,
        alias_multiplicities=tuple(
            len(successor_marks[key]) for key in ordered_keys
        ),
        marks_by_family=marks_by_family,
        productive_marks_by_family=productive_by_family,
    )
    validate_successor_batch(batch)
    if abs(raw_productive_mass + virtual_mass - marked_law.total_probability) > 2e-5:
        raise ProductionSuccessorKernelError(
            "productive and virtual mass do not reconstruct the marked law"
        )
    return SuccessorKernelResult(batch, diagnostics, marked_law)


class FactorizedCanonicalSuccessorKernel:
    """Canonical molecular jump kernel at one fixed model time."""

    def __init__(
        self,
        model: FactorizedTraceletRateModel,
        *,
        time: float,
        identity: KernelIdentity | None = None,
        system: RewriteSystem | None = None,
    ) -> None:
        self.model = model
        self.time = float(time)
        self._identity = identity or _default_kernel_identity(model)
        self.system = system or de_novo_rewrite_system()

    def successors(self, state: MolecularGraph) -> SuccessorBatch:
        return canonical_successor_result(
            self.model,
            state,
            self.time,
            identity=self._identity,
            system=self.system,
        ).batch

    def result(self, state: MolecularGraph) -> SuccessorKernelResult:
        return canonical_successor_result(
            self.model,
            state,
            self.time,
            identity=self._identity,
            system=self.system,
        )

    def identity(self) -> KernelIdentity:
        return self._identity


__all__ = [
    "FactorizedCanonicalSuccessorKernel",
    "FactorizedMarkedLaw",
    "ProductionSuccessorKernelError",
    "ScoredRewriteMark",
    "SuccessorKernelDiagnostics",
    "SuccessorKernelResult",
    "canonical_successor_result",
    "enumerate_factorized_marked_law",
]
