"""Deterministic semantic-axis labels for the RingCore-V1 validation panel.

The checkpoint leaderboard balances its secondary metric over *joint semantic
cells*.  Those cells describe the immutable teacher trace, not a checkpoint's
predictions, so their labels must be fixed before any snapshot is scored.

This module deliberately keeps two concepts separate:

* the six categorical labels used to define a reasonably populated cell; and
* exact audit metadata, such as signed atom-count and graph-cycle-rank deltas.

In particular, ``split_unit`` is the categorical ``held_scaffold``.  The
partition algorithm and version are bound in the label-contract provenance,
not placed in each cell and not upgraded to a topology-disjoint claim.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from compose_v4.chem.molecular_graph import (
    ELEMENTS,
    MolecularGraph,
    contract_scars,
    is_element,
    molecular_graph_to_smiles,
)
from compose_v4.data.scaffold_partition import (
    SCAFFOLD_KEY_ALGORITHM,
    SCAFFOLD_KEY_VERSION,
)
from compose_v4.experiments.ringcore_successor_leaderboard import (
    SEMANTIC_CELL_AXES,
    SEMANTIC_CELL_ENCODER_VERSION,
    encode_semantic_cell,
    stable_json_sha256,
)
from compose_v4.rewrite.action_codec import ActionCodecError, canonical_family

SEMANTIC_AXIS_LABELER_SCHEMA = "compose.ringcore.validation_semantic_axes"
SEMANTIC_AXIS_LABELER_VERSION = 1
SEMANTIC_CELL_CENSUS_SCHEMA = "compose.ringcore.semantic_cell_census"
SEMANTIC_CELL_CENSUS_VERSION = 1
VALIDATION_PARTITION = "validation"
SPLIT_UNIT_LABEL = "held_scaffold"
PARTITION_SALT = "ringcore-v1"

_CNOF = frozenset({"C", "N", "O", "F"})
_LAYER_ALIASES = {
    "corruption": "general_corruption",
    "general_corruption": "general_corruption",
    "cycle_ops": "cycle_operations",
    "cycle_operations": "cycle_operations",
    "mmp_analogue": "mmp_analogue",
}
_EVIDENCE_ORIGIN = {
    "general_corruption": "synthetic_general_corruption",
    "cycle_operations": "synthetic_cycle_operations",
    # The endpoints provide observed analogue evidence.  The executable path
    # between them is compiled and must not be described as an observed route.
    "mmp_analogue": "real_mmp_analogue_endpoints",
}
_TEACHER_CAPABILITY = {
    "atom_insert": "cardinality_primitive",
    "atom_delete": "cardinality_primitive",
    "atom_restate": "atom_state",
    "bond_reorder": "bond_state",
    "bond_reroute": "attachment_reroute",
    "cycle_insert": "cycle_topology_rewrite",
    "cycle_attach": "cycle_topology_rewrite",
    "ring_system_restate": "ring_state",
    "ring_system_delete": "ring_topology_macro",
}
SEMANTIC_DIAGNOSTIC_AXES = (
    "exact_element_subset",
    "aromaticity_transition",
    "charge_transition",
)


class SemanticAxisLabelError(ValueError):
    """One trace cannot be labeled without guessing or changing semantics."""


@dataclass(frozen=True)
class SemanticTraceContext:
    """Immutable trace information needed by the pure labeler.

    Deltas and chemistry strata are defined on the complete trace endpoints,
    not the current teacher jump.  This matters for compiled analogue paths:
    the first deletion in a net-growth analogue must not be mislabeled as a
    shrinkage example.
    """

    layer: str
    partition: str
    path_length: int
    progress_index: int
    trace_source: MolecularGraph
    trace_target: MolecularGraph
    trace_rule_names: tuple[str, ...]
    teacher_rule_name: str | None

    @property
    def terminal(self) -> bool:
        return self.progress_index == self.path_length


@dataclass(frozen=True)
class SemanticAxisAssignment:
    """Panel fields plus exact, non-cell audit deltas for one sampled row."""

    terminal: bool
    axis_items: tuple[tuple[str, str], ...] | None
    semantic_cell_id: str | None
    atom_count_delta: int | None
    graph_cycle_rank_delta: int | None
    diagnostic_items: tuple[tuple[str, str], ...] | None

    def __post_init__(self) -> None:
        if self.terminal:
            if (
                self.axis_items is not None
                or self.semantic_cell_id is not None
                or self.atom_count_delta is not None
                or self.graph_cycle_rank_delta is not None
                or self.diagnostic_items is not None
            ):
                raise ValueError(
                    "terminal assignment must have null axes, deltas, and diagnostics"
                )
            return
        if (
            self.axis_items is None
            or self.semantic_cell_id is None
            or self.atom_count_delta is None
            or self.graph_cycle_rank_delta is None
            or self.diagnostic_items is None
        ):
            raise ValueError("nonterminal assignment is incomplete")
        if tuple(axis for axis, _value in self.axis_items) != SEMANTIC_CELL_AXES:
            raise ValueError("semantic assignment axes are absent or reordered")
        if (
            tuple(axis for axis, _value in self.diagnostic_items)
            != SEMANTIC_DIAGNOSTIC_AXES
        ):
            raise ValueError("semantic diagnostics are absent or reordered")
        if encode_semantic_cell(dict(self.axis_items)) != self.semantic_cell_id:
            raise ValueError("semantic assignment cell id disagrees with its axes")

    @property
    def axis_values(self) -> dict[str, str] | None:
        """Return a fresh JSON-safe mapping in the frozen axis order."""

        return None if self.axis_items is None else dict(self.axis_items)

    @property
    def diagnostic_values(self) -> dict[str, str] | None:
        """Return fine chemistry labels kept out of equal-cell weighting."""

        return (
            None
            if self.diagnostic_items is None
            else dict(self.diagnostic_items)
        )

    def to_panel_fields(self) -> dict[str, object]:
        """Fields consumed directly by ``ringcore_validation_panel`` rows."""

        return {
            "semantic_axis_values": self.axis_values,
            "semantic_cell_id": self.semantic_cell_id,
        }


def _normalize_layer(layer: str) -> str:
    try:
        return _LAYER_ALIASES[str(layer)]
    except KeyError:
        raise SemanticAxisLabelError(
            f"unknown RingCore-V1 validation layer {layer!r}"
        ) from None


def _visible_graph(state: MolecularGraph) -> MolecularGraph:
    visible = contract_scars(state)
    if visible is None:
        raise SemanticAxisLabelError(
            "state contains a degree-three-or-higher SCAR that cannot be "
            "contracted into a semantic molecular graph"
        )
    real = is_element(visible.atom_types)
    inactive = ~real
    if bool(np.any(visible.bonds[inactive, :] != 0)):
        raise SemanticAxisLabelError("semantic graph has a bond incident to padding")
    if bool(np.any(visible.formal_charges[inactive] != 0)):
        raise SemanticAxisLabelError("semantic graph has charge on padding")
    return visible


def graph_cycle_rank(state: MolecularGraph) -> int:
    """Return exact graph cycle rank ``|E| - |V| + c``.

    Real atoms are selected only through the authoritative ``is_element``
    predicate.  SCARs are first contracted according to production read-out
    semantics; bond order is irrelevant because every nonzero undirected bond
    contributes one edge.
    """

    visible = _visible_graph(state)
    real_slots = tuple(
        int(slot) for slot in np.flatnonzero(is_element(visible.atom_types))
    )
    if not real_slots:
        return 0
    real = set(real_slots)
    edge_count = sum(
        1
        for left_index, left in enumerate(real_slots)
        for right in real_slots[left_index + 1 :]
        if int(visible.bonds[left, right]) != 0
    )
    unseen = set(real_slots)
    component_count = 0
    while unseen:
        component_count += 1
        start = unseen.pop()
        stack = [start]
        while stack:
            left = stack.pop()
            neighbors = {
                int(right)
                for right in np.flatnonzero(visible.bonds[left] != 0)
                if int(right) in real
            }
            fresh = neighbors & unseen
            unseen.difference_update(fresh)
            stack.extend(fresh)
    rank = edge_count - len(real_slots) + component_count
    if rank < 0:
        raise SemanticAxisLabelError("graph cycle rank is unexpectedly negative")
    return int(rank)


def _direction(value: int, *, positive: str, negative: str) -> str:
    if value > 0:
        return positive
    if value < 0:
        return negative
    return "same"


def _path_scale(path_length: int) -> str:
    if path_length <= 2:
        return "precise_local_1_2"
    if path_length <= 6:
        return "ordinary_lead_optimization_3_6"
    return "longer_compositional_gt_6"


def _canonical_families(rule_names: Sequence[str]) -> tuple[str, ...]:
    try:
        return tuple(canonical_family(rule) for rule in rule_names)
    except ActionCodecError as exc:
        raise SemanticAxisLabelError(str(exc)) from exc


def _capability_regime(teacher_family: str) -> str:
    try:
        return _TEACHER_CAPABILITY[teacher_family]
    except KeyError:
        raise SemanticAxisLabelError(
            f"teacher family has no semantic capability class: {teacher_family!r}"
        ) from None


def _charge_stratum(state: MolecularGraph) -> str:
    visible = _visible_graph(state)
    real = is_element(visible.atom_types)
    if not bool(real.any()):
        return "null"
    charges = visible.formal_charges[real]
    if not bool(np.any(charges != 0)):
        return "uncharged"
    net = int(charges.sum())
    if net == 0:
        return "zwitterionic"
    return "net_positive" if net > 0 else "net_negative"


def _aromatic_stratum(state: MolecularGraph) -> str:
    visible = _visible_graph(state)
    if visible.n_real_atoms == 0:
        return "null"
    smiles = molecular_graph_to_smiles(visible)
    if smiles is None:
        raise SemanticAxisLabelError(
            "semantic graph cannot be sanitized for aromatic-context labeling"
        )
    from rdkit import Chem

    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise SemanticAxisLabelError(
            "semantic graph cannot be reparsed for aromatic-context labeling"
        )
    return (
        "aromatic"
        if any(atom.GetIsAromatic() for atom in molecule.GetAtoms())
        else "nonaromatic"
    )


def _element_symbols(
    source: MolecularGraph,
    target: MolecularGraph,
) -> frozenset[str]:
    indices: set[int] = set()
    for state in (source, target):
        visible = _visible_graph(state)
        indices.update(
            int(index) for index in visible.atom_types[is_element(visible.atom_types)]
        )
    return frozenset(ELEMENTS[index] for index in indices)


def _chemistry_charge_stratum(
    source: MolecularGraph,
    target: MolecularGraph,
) -> str:
    """Coarse selection stratum; fine chemistry remains a separate diagnostic.

    The complete 33,404-trace validation census showed that crossing exact
    element subsets, aromaticity transitions, and charge transitions yielded
    1,776 cells, most supported by fewer than ten rows. Equal weighting would
    therefore amplify singleton chemistry combinations. The selection cell
    keeps only the production-support distinction needed for broad-organic
    coverage; the exact tails are retained by ``_chemistry_diagnostics``.
    """

    symbols = _element_symbols(source, target)
    return (
        "cnof_only_any_aromaticity_any_charge"
        if symbols <= _CNOF
        else "expanded_organic_any_aromaticity_any_charge"
    )


def _chemistry_diagnostics(
    source: MolecularGraph,
    target: MolecularGraph,
) -> tuple[tuple[str, str], ...]:
    symbols = _element_symbols(source, target)
    exact_elements = "_".join(
        element for element in ELEMENTS if element in symbols
    )
    values = {
        "exact_element_subset": exact_elements or "null",
        "aromaticity_transition": (
            f"{_aromatic_stratum(source)}_to_{_aromatic_stratum(target)}"
        ),
        "charge_transition": (
            f"{_charge_stratum(source)}_to_{_charge_stratum(target)}"
        ),
    }
    return tuple((axis, values[axis]) for axis in SEMANTIC_DIAGNOSTIC_AXES)


def _validate_context(context: SemanticTraceContext) -> str:
    layer = _normalize_layer(context.layer)
    if context.partition != VALIDATION_PARTITION:
        raise SemanticAxisLabelError(
            "RingCore-V1 checkpoint-selection labels require validation rows"
        )
    if type(context.path_length) is not int or context.path_length < 0:
        raise SemanticAxisLabelError("path_length must be a nonnegative integer")
    if (
        type(context.progress_index) is not int
        or not 0 <= context.progress_index <= context.path_length
    ):
        raise SemanticAxisLabelError("progress_index lies outside the trace")
    if len(context.trace_rule_names) != context.path_length:
        raise SemanticAxisLabelError(
            "trace rule count disagrees with path_length"
        )
    if context.trace_source.n_atoms != context.trace_target.n_atoms:
        raise SemanticAxisLabelError(
            "trace endpoints disagree on persistent-slot capacity"
        )
    if context.terminal:
        if context.teacher_rule_name is not None:
            raise SemanticAxisLabelError("terminal row carries a teacher rule")
    else:
        expected_rule = context.trace_rule_names[context.progress_index]
        if context.teacher_rule_name != expected_rule:
            raise SemanticAxisLabelError(
                "teacher rule disagrees with the immutable trace progress"
            )
    return layer


def label_validation_semantic_axes(
    context: SemanticTraceContext,
) -> SemanticAxisAssignment:
    """Label one immutable validation row without checkpoint-dependent inputs."""

    layer = _validate_context(context)
    if context.terminal:
        return SemanticAxisAssignment(
            terminal=True,
            axis_items=None,
            semantic_cell_id=None,
            atom_count_delta=None,
            graph_cycle_rank_delta=None,
            diagnostic_items=None,
        )

    source = _visible_graph(context.trace_source)
    target = _visible_graph(context.trace_target)
    atom_delta = target.n_real_atoms - source.n_real_atoms
    cycle_delta = graph_cycle_rank(target) - graph_cycle_rank(source)
    _canonical_families(context.trace_rule_names)
    assert context.teacher_rule_name is not None
    try:
        teacher_family = canonical_family(context.teacher_rule_name)
    except ActionCodecError as exc:
        raise SemanticAxisLabelError(str(exc)) from exc
    cardinality = _direction(
        atom_delta,
        positive="grow",
        negative="shrink",
    )
    topology = _direction(
        cycle_delta,
        positive="increase",
        negative="decrease",
    )
    axes = {
        "capability_regime": _capability_regime(teacher_family),
        "evidence_origin": _EVIDENCE_ORIGIN[layer],
        "path_scale": _path_scale(context.path_length),
        "cardinality_topology_delta": (
            f"cardinality_{cardinality};cycle_rank_{topology}"
        ),
        "chemistry_charge_stratum": _chemistry_charge_stratum(source, target),
        "split_unit": SPLIT_UNIT_LABEL,
    }
    ordered = tuple((axis, axes[axis]) for axis in SEMANTIC_CELL_AXES)
    return SemanticAxisAssignment(
        terminal=False,
        axis_items=ordered,
        semantic_cell_id=encode_semantic_cell(axes),
        atom_count_delta=atom_delta,
        graph_cycle_rank_delta=cycle_delta,
        diagnostic_items=_chemistry_diagnostics(source, target),
    )


def context_from_path_record(
    record: Any,
    *,
    progress_index: int,
    layer: str | None = None,
    partition: str | None = None,
) -> SemanticTraceContext:
    """Adapt a production ``PathRecord`` without replay or mutable metadata."""

    try:
        path = record.path
        trace = path.trace
        address = record.corpus_address
        path_length = int(path.path_length)
        rule_names = tuple(step.rule_name for step in trace.steps)
    except (AttributeError, TypeError) as exc:
        raise SemanticAxisLabelError(
            "record does not expose the production PathRecord/trace contract"
        ) from exc
    addressed_layer = None if address is None else str(address.layer)
    addressed_partition = None if address is None else str(address.partition)
    resolved_layer = layer if layer is not None else addressed_layer
    resolved_partition = partition if partition is not None else addressed_partition
    if resolved_layer is None or resolved_partition is None:
        raise SemanticAxisLabelError(
            "layer and partition require an immutable address or explicit values"
        )
    if (
        type(progress_index) is not int
        or not 0 <= progress_index <= path_length
    ):
        raise SemanticAxisLabelError("progress_index lies outside the trace")
    if (
        addressed_layer is not None
        and _normalize_layer(addressed_layer) != _normalize_layer(resolved_layer)
    ):
        raise SemanticAxisLabelError("explicit layer disagrees with packed address")
    if addressed_partition is not None and addressed_partition != resolved_partition:
        raise SemanticAxisLabelError(
            "explicit partition disagrees with packed address"
        )
    teacher_rule = (
        None if progress_index == path_length else rule_names[progress_index]
    )
    return SemanticTraceContext(
        layer=resolved_layer,
        partition=resolved_partition,
        path_length=path_length,
        progress_index=progress_index,
        trace_source=trace.source,
        trace_target=trace.target,
        trace_rule_names=rule_names,
        teacher_rule_name=teacher_rule,
    )


def semantic_axis_labeler_contract() -> dict[str, object]:
    """Machine-readable frozen label semantics, independent of any census."""

    return {
        "schema": SEMANTIC_AXIS_LABELER_SCHEMA,
        "schema_version": SEMANTIC_AXIS_LABELER_VERSION,
        "semantic_cell_encoder_version": SEMANTIC_CELL_ENCODER_VERSION,
        "required_axes": list(SEMANTIC_CELL_AXES),
        "row_scope": {
            "partition": VALIDATION_PARTITION,
            "terminal": "semantic_axis_values=null and semantic_cell_id=null",
            "nonterminal": "all six axes required",
            "trace_granularity": (
                "path scale, endpoint deltas, and chemistry use the immutable "
                "complete trace; capability uses the row-local teacher family"
            ),
        },
        "layers": {
            "accepted_aliases": dict(sorted(_LAYER_ALIASES.items())),
            "evidence_origin": dict(sorted(_EVIDENCE_ORIGIN.items())),
            "mmp_qualification": (
                "real analogue endpoints; compiled executable route is not "
                "claimed to be an observed route"
            ),
        },
        "path_scale": {
            "precise_local_1_2": "1 <= path_length <= 2",
            "ordinary_lead_optimization_3_6": "3 <= path_length <= 6",
            "longer_compositional_gt_6": "path_length > 6",
        },
        "cardinality_topology_delta": {
            "granularity": "endpoint direction categories",
            "cardinality": ["shrink", "same", "grow"],
            "graph_cycle_rank": ["decrease", "same", "increase"],
            "graph_cycle_rank_formula": "|E|-|V|+c",
            "real_atom_predicate": "is_element",
            "scar_policy": "contract_scars before graph statistics",
            "exact_signed_deltas_retained_outside_cell": True,
        },
        "chemistry_charge_stratum": {
            "selection_categories": [
                "cnof_only_any_aromaticity_any_charge",
                "expanded_organic_any_aromaticity_any_charge",
            ],
            "selection_reason": (
                "full validation-corpus sparsity audit; exact chemistry "
                "cross-products are not equal-weighted"
            ),
            "separate_diagnostic_axes": list(SEMANTIC_DIAGNOSTIC_AXES),
            "exact_element_aromaticity_and_charge_retained": True,
        },
        "capability_regime": {
            "teacher_family_mapping": dict(sorted(_TEACHER_CAPABILITY.items())),
            "derivation": "row-local canonical teacher family",
        },
        "split_unit": {
            "cell_label": SPLIT_UNIT_LABEL,
            "partition_unit": "scaffold key",
            "scaffold_key_algorithm": SCAFFOLD_KEY_ALGORITHM,
            "scaffold_key_version": SCAFFOLD_KEY_VERSION,
            "partition_salt": PARTITION_SALT,
            "claim_scope": (
                "held scaffold-key partition only; no topology-, series-, or "
                "transformation-disjoint claim"
            ),
        },
        "result_dependent_merging_forbidden": True,
    }


def semantic_axis_labeler_contract_sha256() -> str:
    return stable_json_sha256(semantic_axis_labeler_contract())


def validate_labeler_against_leaderboard_config(
    config: Mapping[str, object],
) -> str:
    """Bind the labeler to the frozen leaderboard axes and current layers."""

    panels = config.get("panels")
    if not isinstance(panels, Mapping):
        raise SemanticAxisLabelError("leaderboard config lacks panels")
    cells = panels.get("semantic_cells")
    if not isinstance(cells, Mapping):
        raise SemanticAxisLabelError("leaderboard config lacks semantic cells")
    if cells.get("encoder_version") != SEMANTIC_CELL_ENCODER_VERSION:
        raise SemanticAxisLabelError("leaderboard encoder version disagrees")
    if tuple(cells.get("required_axes") or ()) != SEMANTIC_CELL_AXES:
        raise SemanticAxisLabelError("leaderboard axes disagree with labeler")
    validation = config.get("validation_data")
    if not isinstance(validation, Mapping):
        raise SemanticAxisLabelError("leaderboard config lacks validation data")
    if validation.get("partition") != VALIDATION_PARTITION:
        raise SemanticAxisLabelError("leaderboard selection partition is not validation")
    layers = validation.get("layers")
    if not isinstance(layers, Mapping):
        raise SemanticAxisLabelError("leaderboard validation layers are absent")
    expected_layers = frozenset(_EVIDENCE_ORIGIN)
    if frozenset(str(layer) for layer in layers) != expected_layers:
        raise SemanticAxisLabelError(
            "leaderboard validation layers disagree with labeler"
        )
    for layer, payload in layers.items():
        if not isinstance(payload, Mapping) or int(payload.get("records", 0)) <= 0:
            raise SemanticAxisLabelError(
                f"leaderboard layer {layer!r} has no frozen records"
            )
    return semantic_axis_labeler_contract_sha256()


@dataclass
class BoundedSemanticCellCensus:
    """Streaming nonempty-cell census with an explicit memory bound."""

    maximum_nonempty_cells: int
    total_rows: int = 0
    terminal_rows: int = 0
    nonterminal_rows: int = 0
    total_importance_weight: float = 0.0
    terminal_importance_weight: float = 0.0
    nonterminal_importance_weight: float = 0.0
    _cell_counts: Counter[str] = field(default_factory=Counter, init=False)
    _cell_weights: Counter[str] = field(default_factory=Counter, init=False)
    _cell_axes: dict[str, tuple[tuple[str, str], ...]] = field(
        default_factory=dict,
        init=False,
    )
    _axis_counts: dict[str, Counter[str]] = field(
        default_factory=lambda: {
            axis: Counter() for axis in SEMANTIC_CELL_AXES
        },
        init=False,
    )
    _exact_delta_counts: Counter[str] = field(
        default_factory=Counter,
        init=False,
    )
    _diagnostic_counts: dict[str, Counter[str]] = field(
        default_factory=lambda: {
            axis: Counter() for axis in SEMANTIC_DIAGNOSTIC_AXES
        },
        init=False,
    )

    def __post_init__(self) -> None:
        if (
            type(self.maximum_nonempty_cells) is not int
            or self.maximum_nonempty_cells <= 0
        ):
            raise ValueError("maximum_nonempty_cells must be a positive integer")

    def add(
        self,
        assignment: SemanticAxisAssignment,
        *,
        importance_weight: float = 1.0,
    ) -> None:
        weight = float(importance_weight)
        if not np.isfinite(weight) or weight <= 0.0:
            raise ValueError("importance_weight must be finite and positive")
        self.total_rows += 1
        self.total_importance_weight += weight
        if assignment.terminal:
            self.terminal_rows += 1
            self.terminal_importance_weight += weight
            return
        assert assignment.semantic_cell_id is not None
        assert assignment.axis_items is not None
        assert assignment.atom_count_delta is not None
        assert assignment.graph_cycle_rank_delta is not None
        assert assignment.diagnostic_items is not None
        cell = assignment.semantic_cell_id
        if cell not in self._cell_counts:
            if len(self._cell_counts) >= self.maximum_nonempty_cells:
                raise SemanticAxisLabelError(
                    "semantic-cell census exceeded its explicit nonempty-cell bound"
                )
            self._cell_axes[cell] = assignment.axis_items
        elif self._cell_axes[cell] != assignment.axis_items:
            raise SemanticAxisLabelError(
                "one semantic-cell id mapped to two axis assignments"
            )
        self.nonterminal_rows += 1
        self.nonterminal_importance_weight += weight
        self._cell_counts[cell] += 1
        self._cell_weights[cell] += weight
        for axis, value in assignment.axis_items:
            self._axis_counts[axis][value] += 1
        for axis, value in assignment.diagnostic_items:
            self._diagnostic_counts[axis][value] += 1
        delta_key = (
            f"atoms:{assignment.atom_count_delta:+d};"
            f"cycle_rank:{assignment.graph_cycle_rank_delta:+d}"
        )
        self._exact_delta_counts[delta_key] += 1

    def extend(
        self,
        assignments: Iterable[SemanticAxisAssignment],
    ) -> None:
        for assignment in assignments:
            self.add(assignment)

    def payload(self) -> dict[str, object]:
        body: dict[str, object] = {
            "schema": SEMANTIC_CELL_CENSUS_SCHEMA,
            "schema_version": SEMANTIC_CELL_CENSUS_VERSION,
            "labeler_contract_sha256": semantic_axis_labeler_contract_sha256(),
            "maximum_nonempty_cells": self.maximum_nonempty_cells,
            "counts": {
                "rows": self.total_rows,
                "terminal_rows": self.terminal_rows,
                "nonterminal_rows": self.nonterminal_rows,
                "nonempty_cells": len(self._cell_counts),
            },
            "importance_weight_sums": {
                "all": self.total_importance_weight,
                "terminal": self.terminal_importance_weight,
                "nonterminal": self.nonterminal_importance_weight,
            },
            "axis_marginal_row_counts": {
                axis: dict(sorted(counts.items()))
                for axis, counts in self._axis_counts.items()
            },
            "diagnostic_marginal_row_counts": {
                axis: dict(sorted(counts.items()))
                for axis, counts in self._diagnostic_counts.items()
            },
            "exact_endpoint_delta_row_counts": dict(
                sorted(self._exact_delta_counts.items())
            ),
            "cells": {
                cell: {
                    "rows": self._cell_counts[cell],
                    "importance_weight_sum": self._cell_weights[cell],
                    "axis_values": dict(self._cell_axes[cell]),
                }
                for cell in sorted(self._cell_counts)
            },
        }
        body["census_sha256"] = stable_json_sha256(body)
        return body


__all__ = [
    "BoundedSemanticCellCensus",
    "SEMANTIC_AXIS_LABELER_SCHEMA",
    "SEMANTIC_AXIS_LABELER_VERSION",
    "SEMANTIC_CELL_CENSUS_SCHEMA",
    "SEMANTIC_CELL_CENSUS_VERSION",
    "SEMANTIC_DIAGNOSTIC_AXES",
    "SPLIT_UNIT_LABEL",
    "SemanticAxisAssignment",
    "SemanticAxisLabelError",
    "SemanticTraceContext",
    "context_from_path_record",
    "graph_cycle_rank",
    "label_validation_semantic_axes",
    "semantic_axis_labeler_contract",
    "semantic_axis_labeler_contract_sha256",
    "validate_labeler_against_leaderboard_config",
]
