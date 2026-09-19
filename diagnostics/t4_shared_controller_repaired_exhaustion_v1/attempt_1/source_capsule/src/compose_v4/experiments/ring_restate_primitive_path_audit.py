"""Charge-clean primitive-path audit for packed ring-system-restatement teachers.

This module is intentionally checkpoint independent.  It does not enumerate a
model fiber, assign rates, mutate operator support, or authorize training.  It
answers one structural question for every addressed packed teacher:

* does the production ``ring_system_restate`` executor reproduce the stored
  exact successor;
* does the authoritative micro lowering reproduce that same successor through
  valid, connected, charge-policy-preserving intermediates; and
* is that lowering already contained in the *currently declared* primitive-7
  marked support?

The last distinction matters.  ``lower_ring_system_restate`` emits executable
``bond_reorder`` micro instructions, while the current primitive-7 O0 basis
deliberately excludes bond reorders on cyclic edges.  A validity-closed micro
lowering is therefore not automatically a legal path in the declared
primitive-7 learned process.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from types import MappingProxyType
from typing import Any, Iterable, Mapping

import networkx as nx
import numpy as np

from compose_v4.chem.aromaticity import (
    perceived_aromatic_ring_count,
    resonance_invariant_bond_classes,
)
from compose_v4.chem.molecular_graph import (
    BOND_AROMATIC,
    BOND_SINGLE,
    ELEMENT_TO_IDX,
    MolecularGraph,
    is_element,
)
from compose_v4.chem.persistent_state_identity import (
    persistent_slot_state_sha256,
)
from compose_v4.chem.state import is_connected_or_null, is_valid_state
from compose_v4.data.charge_policy import charge_policy_preserved
from compose_v4.data.packed_charge_policy_audit import (
    resolve_unified_manifest_shards,
)
from compose_v4.data.packed_trace_store import read_addressed_packed_shard
from compose_v4.data.representability_overlay import (
    check_unlisted,
    excluded_keys,
    load_overlay,
    trace_key,
)
from compose_v4.experiments.editing_operator_support import (
    _cycle_edges,
    _primitive_candidate_is_declared,
)
from compose_v4.experiments.editing_t1_panel import (
    validate_charge_policy_exclusions,
)
from compose_v4.rewrite.action_codec import encode_action
from compose_v4.rewrite.kernel import (
    canonical_state_key,
    de_novo_rewrite_system,
)
from compose_v4.rewrite.trace import RewriteStep
from compose_v4.rewrite.tracelets import (
    RingSystemRestate,
    lower_ring_system_restate,
)


RING_RESTATE_PRIMITIVE_PATH_AUDIT_SCHEMA = "compose.editing.ring_restate_primitive_path_audit"
RING_RESTATE_PRIMITIVE_PATH_AUDIT_SCHEMA_VERSION = 1
RING_RESTATE_PRIMITIVE_PATH_AUDIT_STATUS = (
    "CHECKPOINT_INDEPENDENT_OPERATOR_EVIDENCE_NO_PRODUCTION_MUTATION"
)
_BUDGETS = (8, 16)
_PARTITION_ORDER = {"train": 0, "validation": 1, "test": 2}
_IMPLEMENTATION_SOURCES = (
    "src/compose_v4/experiments/ring_restate_primitive_path_audit.py",
    "src/compose_v4/experiments/editing_operator_support.py",
    "src/compose_v4/rewrite/tracelets.py",
    "src/compose_v4/rewrite/kernel.py",
    "src/compose_v4/rewrite/action_codec.py",
    "src/compose_v4/data/packed_trace_store.py",
    "src/compose_v4/data/charge_policy.py",
    "src/compose_v4/data/representability_overlay.py",
    "src/compose_v4/chem/aromaticity.py",
    "src/compose_v4/chem/state.py",
    "src/compose_v4/chem/persistent_state_identity.py",
)


class RingRestatePrimitivePathAuditError(RuntimeError):
    """The packed audit could not establish complete, exact coverage."""


@dataclass(frozen=True)
class RingRestateAuditInputs:
    """Immutable local paths required by the whole-corpus audit."""

    transfer_root: Path
    unified_manifest: Path
    representability_overlay: Path
    charge_policy_audit: Path
    charge_policy_exclusions: Path

    @classmethod
    def from_transfer_root(
        cls,
        transfer_root: Path,
        *,
        charge_policy_audit: Path,
        charge_policy_exclusions: Path,
    ) -> "RingRestateAuditInputs":
        root = Path(transfer_root)
        return cls(
            transfer_root=root,
            unified_manifest=root / "UNIFIED_PACKED_MANIFEST.json",
            representability_overlay=root / "REPRESENTABILITY_OVERLAY.json",
            charge_policy_audit=Path(charge_policy_audit),
            charge_policy_exclusions=Path(charge_policy_exclusions),
        )


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _stable_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def _load_json_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(Path(path).read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RingRestatePrimitivePathAuditError(
            f"cannot read JSON object {path}: {error}"
        ) from error
    if not isinstance(payload, dict):
        raise RingRestatePrimitivePathAuditError(f"expected object-valued JSON at {path}")
    return payload


def implementation_provenance(
    *,
    repo_root: Path | None = None,
) -> Mapping[str, object]:
    """Hash every source that defines decoding, lowering, and audit semantics."""

    root = Path(repo_root) if repo_root is not None else Path(__file__).resolve().parents[3]
    sources: dict[str, str] = {}
    for relative in _IMPLEMENTATION_SOURCES:
        path = root / relative
        if not path.is_file():
            raise RingRestatePrimitivePathAuditError(
                f"ring-restatement audit implementation source is absent: {path}"
            )
        sources[relative] = _sha256_file(path)
    return MappingProxyType(
        {
            "sources": MappingProxyType(sources),
            "implementation_sha256": _stable_sha256(sources),
        }
    )


def _same_exact_state(left: MolecularGraph, right: MolecularGraph) -> bool:
    return bool(
        left.n_atoms == right.n_atoms
        and np.array_equal(left.atom_types, right.atom_types)
        and np.array_equal(left.formal_charges, right.formal_charges)
        and np.array_equal(left.implicit_h_counts, right.implicit_h_counts)
        and np.array_equal(left.bonds, right.bonds)
    )


def _affected_ring_system(
    state: MolecularGraph,
    action: RingSystemRestate,
) -> tuple[tuple[int, ...], tuple[tuple[int, int], ...]]:
    """Recover the exact non-bridge component containing all changed edges."""

    real = tuple(int(slot) for slot in np.flatnonzero(is_element(state.atom_types)))
    graph = nx.Graph()
    graph.add_nodes_from(real)
    graph.add_edges_from(
        (left, right)
        for offset, left in enumerate(real)
        for right in real[offset + 1 :]
        if int(state.bonds[left, right]) != 0
    )
    bridges = {frozenset(edge) for edge in nx.bridges(graph)}
    ring_graph = graph.copy()
    ring_graph.remove_edges_from(tuple(tuple(edge) for edge in bridges))
    changed_edges = {frozenset((int(change.a), int(change.b))) for change in action.changes}
    matches: list[tuple[tuple[int, ...], tuple[tuple[int, int], ...]]] = []
    for vertices in nx.connected_components(ring_graph):
        component = ring_graph.subgraph(vertices)
        component_edges = {frozenset((int(left), int(right))) for left, right in component.edges()}
        if changed_edges and changed_edges.issubset(component_edges):
            matches.append(
                (
                    tuple(sorted(int(vertex) for vertex in vertices)),
                    tuple(
                        sorted(
                            (min(int(left), int(right)), max(int(left), int(right)))
                            for left, right in component.edges()
                        )
                    ),
                )
            )
    if len(matches) != 1:
        raise RingRestatePrimitivePathAuditError(
            "ring-system restatement does not address exactly one non-bridge component"
        )
    return matches[0]


def _event_strata(
    source: MolecularGraph,
    successor: MolecularGraph,
    action: RingSystemRestate,
) -> Mapping[str, object]:
    system_atoms, system_edges = _affected_ring_system(source, action)
    source_aromatic_view = resonance_invariant_bond_classes(source)
    successor_aromatic_view = resonance_invariant_bond_classes(successor)
    source_aromatic_rings = perceived_aromatic_ring_count(source)
    successor_aromatic_rings = perceived_aromatic_ring_count(successor)
    aromatic_delta = successor_aromatic_rings - source_aromatic_rings
    if aromatic_delta > 0:
        direction = "aromatizing"
    elif aromatic_delta < 0:
        direction = "dearomatizing"
    else:
        direction = "aromaticity_neutral"
    source_aromatic_edges = sum(
        int(source_aromatic_view[left, right]) == BOND_AROMATIC for left, right in system_edges
    )
    successor_aromatic_edges = sum(
        int(successor_aromatic_view[left, right]) == BOND_AROMATIC for left, right in system_edges
    )
    carbon = int(ELEMENT_TO_IDX["C"])
    cycle_rank = len(system_edges) - len(system_atoms) + 1
    return MappingProxyType(
        {
            "direction": direction,
            "aromatic_ring_count_before": int(source_aromatic_rings),
            "aromatic_ring_count_after": int(successor_aromatic_rings),
            "aromatic_ring_count_delta": int(aromatic_delta),
            "system_atom_count": len(system_atoms),
            "system_edge_count": len(system_edges),
            "system_cycle_rank": int(cycle_rank),
            "heterocycle": any(int(source.atom_types[slot]) != carbon for slot in system_atoms),
            "source_system_saturated": all(
                int(source.bonds[left, right]) == BOND_SINGLE for left, right in system_edges
            ),
            "target_system_saturated": all(
                int(successor.bonds[left, right]) == BOND_SINGLE for left, right in system_edges
            ),
            "source_system_aromatic_edge_count": int(source_aromatic_edges),
            "target_system_aromatic_edge_count": int(successor_aromatic_edges),
        }
    )


def _declared_primitive7_membership(
    state: MolecularGraph,
    step: RewriteStep,
) -> tuple[bool, str | None]:
    """Use the current O0 basis semantics without enumerating an irrelevant fiber.

    Every current ring-restatement lowering is a bond reorder.  Cyclic
    bond-reorder actions are rejected unconditionally by
    ``editing_operator_support._direct_candidate_allowed``.  Detect that exact
    reason first; use the complete declared-fiber membership checker for any
    future lowering instruction that is not covered by this shortcut.
    """

    if step.rule_name == "bond_reorder":
        edge = frozenset((int(step.action.a), int(step.action.b)))
        if edge in _cycle_edges(state):
            return False, "current_primitive7_excludes_cyclic_bond_reorder"
    if _primitive_candidate_is_declared(state, step):
        return True, None
    return False, "instruction_absent_from_current_primitive7_fiber"


def audit_ring_restate_teacher(
    source: MolecularGraph,
    stored_successor: MolecularGraph,
    action: RingSystemRestate,
) -> Mapping[str, object]:
    """Audit one exact packed teacher and return a compact deterministic result."""

    runtime = de_novo_rewrite_system()
    failures: list[dict[str, object]] = []

    def fail(code: str, detail: str, **context: object) -> None:
        failures.append({"code": code, "detail": detail, **context})

    for name, state in (("source", source), ("stored_successor", stored_successor)):
        if not is_valid_state(state):
            fail(f"{name}_invalid", f"{name} fails the authoritative validity predicate")
        if not is_connected_or_null(state):
            fail(f"{name}_disconnected", f"{name} fails the connectedness predicate")
    if not charge_policy_preserved(source, stored_successor):
        fail(
            "stored_teacher_charge_policy_violation",
            "stored teacher successor violates the source-relative charge policy",
        )

    macro_successor: MolecularGraph | None = None
    try:
        macro_successor = runtime.apply(source, "ring_system_restate", action)
    except Exception as error:  # keep auditing the complete corpus after one failure.
        fail("macro_execution_failed", f"{type(error).__name__}: {error}")
    if macro_successor is not None:
        if not _same_exact_state(macro_successor, stored_successor):
            fail(
                "macro_exact_endpoint_mismatch",
                "production macro endpoint differs from the packed exact successor",
            )
        try:
            macro_key = canonical_state_key(macro_successor)
            stored_key = canonical_state_key(stored_successor)
        except Exception as error:
            fail("macro_canonicalization_failed", f"{type(error).__name__}: {error}")
        else:
            if macro_key != stored_key:
                fail(
                    "macro_canonical_endpoint_mismatch",
                    f"macro canonical key {macro_key!r} != stored {stored_key!r}",
                )

    instructions: tuple[tuple[str, Any], ...] = ()
    try:
        instructions = lower_ring_system_restate(source, action)
    except Exception as error:
        fail("authoritative_lowering_failed", f"{type(error).__name__}: {error}")

    current = source
    primitive_families: list[str] = []
    declared_membership: list[bool] = []
    declared_failure_reasons: Counter[str] = Counter()
    intermediate_checks: list[dict[str, object]] = []
    for step_index, (rule_name, primitive_action) in enumerate(instructions):
        step = RewriteStep(str(rule_name), primitive_action)
        primitive_families.append(step.rule_name)
        declared, reason = _declared_primitive7_membership(current, step)
        declared_membership.append(declared)
        if reason is not None:
            declared_failure_reasons[reason] += 1
        try:
            next_state = runtime.apply(current, step.rule_name, step.action)
        except Exception as error:
            fail(
                "primitive_execution_failed",
                f"{type(error).__name__}: {error}",
                step_index=step_index,
                rule_name=step.rule_name,
            )
            break
        valid = is_valid_state(next_state)
        connected = is_connected_or_null(next_state)
        charge_from_source = charge_policy_preserved(source, next_state)
        charge_consecutive = charge_policy_preserved(current, next_state)
        intermediate_checks.append(
            {
                "step_index": step_index,
                "rule_name": step.rule_name,
                "valid": bool(valid),
                "connected": bool(connected),
                "charge_policy_preserved_from_teacher_source": bool(charge_from_source),
                "charge_policy_preserved_consecutively": bool(charge_consecutive),
                "declared_primitive7_member": bool(declared),
                "declared_primitive7_failure_reason": reason,
            }
        )
        if not valid:
            fail(
                "primitive_intermediate_invalid",
                "primitive intermediate fails the authoritative validity predicate",
                step_index=step_index,
            )
        if not connected:
            fail(
                "primitive_intermediate_disconnected",
                "primitive intermediate fails connectedness",
                step_index=step_index,
            )
        if not charge_from_source or not charge_consecutive:
            fail(
                "primitive_intermediate_charge_policy_violation",
                "primitive intermediate violates the editing charge policy",
                step_index=step_index,
                source_relative=bool(charge_from_source),
                consecutive=bool(charge_consecutive),
            )
        current = next_state

    lowering_completed = len(intermediate_checks) == len(instructions) and bool(instructions)
    if instructions and lowering_completed:
        if not _same_exact_state(current, stored_successor):
            fail(
                "primitive_exact_endpoint_mismatch",
                "authoritative primitive lowering differs from the packed exact successor",
            )
        if macro_successor is not None and not _same_exact_state(current, macro_successor):
            fail(
                "primitive_macro_exact_endpoint_mismatch",
                "primitive endpoint differs from the production macro endpoint",
            )
        try:
            primitive_key = canonical_state_key(current)
            stored_key = canonical_state_key(stored_successor)
        except Exception as error:
            fail("primitive_canonicalization_failed", f"{type(error).__name__}: {error}")
        else:
            if primitive_key != stored_key:
                fail(
                    "primitive_canonical_endpoint_mismatch",
                    f"primitive canonical key {primitive_key!r} != stored {stored_key!r}",
                )

    strata: Mapping[str, object] = MappingProxyType({})
    try:
        strata = _event_strata(source, stored_successor, action)
    except Exception as error:
        fail("stratification_failed", f"{type(error).__name__}: {error}")

    source_digest = persistent_slot_state_sha256(source)
    stored_digest = persistent_slot_state_sha256(stored_successor)
    action_sha256 = _stable_sha256(encode_action("ring_system_restate", action))
    return MappingProxyType(
        {
            "source_state_sha256": source_digest,
            "stored_successor_state_sha256": stored_digest,
            "action_sha256": action_sha256,
            "change_count": len(action.changes),
            "lowering_length": len(instructions),
            "primitive_family_sequence": tuple(primitive_families),
            "lowering_completed": bool(lowering_completed),
            "all_primitive_intermediates_valid": bool(
                intermediate_checks and all(row["valid"] for row in intermediate_checks)
            ),
            "all_primitive_intermediates_connected": bool(
                intermediate_checks and all(row["connected"] for row in intermediate_checks)
            ),
            "all_primitive_intermediates_charge_policy_preserving": bool(
                intermediate_checks
                and all(
                    row["charge_policy_preserved_from_teacher_source"]
                    and row["charge_policy_preserved_consecutively"]
                    for row in intermediate_checks
                )
            ),
            "all_lowering_steps_in_declared_primitive7": bool(
                declared_membership and all(declared_membership)
            ),
            "declared_primitive7_step_count": int(sum(declared_membership)),
            "declared_primitive7_failure_reasons": dict(sorted(declared_failure_reasons.items())),
            "intermediate_checks": tuple(intermediate_checks),
            "strata": strata,
            "failures": tuple(failures),
        }
    )


def _counter(counter: Counter[Any]) -> dict[str, int]:
    return {
        str(key): int(value)
        for key, value in sorted(counter.items(), key=lambda item: str(item[0]))
    }


def _sequence_key(sequence: Iterable[str]) -> str:
    return " -> ".join(str(value) for value in sequence)


class _Aggregate:
    def __init__(self) -> None:
        self.events = 0
        self.hard_failures = 0
        self.lowering_length: Counter[int] = Counter()
        self.change_count: Counter[int] = Counter()
        self.primitive_sequence: Counter[str] = Counter()
        self.direction: Counter[str] = Counter()
        self.heterocycle: Counter[str] = Counter()
        self.source_saturation: Counter[str] = Counter()
        self.target_saturation: Counter[str] = Counter()
        self.system_cycle_rank: Counter[int] = Counter()
        self.declared_membership: Counter[str] = Counter()
        self.declared_steps = 0
        self.total_steps = 0
        self.failure_codes: Counter[str] = Counter()
        self.budget_fit: Counter[int] = Counter()
        self.valid_path_events = 0
        self.connected_path_events = 0
        self.charge_clean_path_events = 0
        self.exact_endpoint_events = 0

    def add(self, result: Mapping[str, object]) -> None:
        self.events += 1
        failures = tuple(result["failures"])
        if failures:
            self.hard_failures += 1
            for failure in failures:
                self.failure_codes[str(failure["code"])] += 1
        length = int(result["lowering_length"])
        self.lowering_length[length] += 1
        self.change_count[int(result["change_count"])] += 1
        self.primitive_sequence[_sequence_key(result["primitive_family_sequence"])] += 1
        strata = result["strata"]
        if strata:
            self.direction[str(strata["direction"])] += 1
            self.heterocycle[str(bool(strata["heterocycle"])).lower()] += 1
            self.source_saturation[str(bool(strata["source_system_saturated"])).lower()] += 1
            self.target_saturation[str(bool(strata["target_system_saturated"])).lower()] += 1
            self.system_cycle_rank[int(strata["system_cycle_rank"])] += 1
        declared = bool(result["all_lowering_steps_in_declared_primitive7"])
        self.declared_membership["fully_declared" if declared else "not_fully_declared"] += 1
        self.declared_steps += int(result["declared_primitive7_step_count"])
        self.total_steps += length
        for budget in _BUDGETS:
            if length <= budget:
                self.budget_fit[budget] += 1
        self.valid_path_events += int(bool(result["all_primitive_intermediates_valid"]))
        self.connected_path_events += int(bool(result["all_primitive_intermediates_connected"]))
        self.charge_clean_path_events += int(
            bool(result["all_primitive_intermediates_charge_policy_preserving"])
        )
        endpoint_failure_codes = {
            "macro_exact_endpoint_mismatch",
            "macro_canonical_endpoint_mismatch",
            "primitive_exact_endpoint_mismatch",
            "primitive_macro_exact_endpoint_mismatch",
            "primitive_canonical_endpoint_mismatch",
            "macro_execution_failed",
            "authoritative_lowering_failed",
            "primitive_execution_failed",
        }
        self.exact_endpoint_events += int(
            not any(str(failure["code"]) in endpoint_failure_codes for failure in failures)
        )

    def to_json(self) -> dict[str, object]:
        denominator = max(self.events, 1)
        return {
            "teacher_events": self.events,
            "teacher_events_with_hard_failures": self.hard_failures,
            "hard_failure_fraction": self.hard_failures / denominator,
            "lowering_length_histogram": _counter(self.lowering_length),
            "change_count_histogram": _counter(self.change_count),
            "primitive_family_sequence_histogram": _counter(self.primitive_sequence),
            "direction_histogram": _counter(self.direction),
            "heterocycle_histogram": _counter(self.heterocycle),
            "source_system_saturation_histogram": _counter(self.source_saturation),
            "target_system_saturation_histogram": _counter(self.target_saturation),
            "system_cycle_rank_histogram": _counter(self.system_cycle_rank),
            "declared_primitive7_event_membership": _counter(self.declared_membership),
            "declared_primitive7_step_count": self.declared_steps,
            "total_lowering_step_count": self.total_steps,
            "declared_primitive7_step_fraction": (
                self.declared_steps / self.total_steps if self.total_steps else 0.0
            ),
            "failure_code_counts": _counter(self.failure_codes),
            "valid_primitive_path_fraction": self.valid_path_events / denominator,
            "connected_primitive_path_fraction": (self.connected_path_events / denominator),
            "charge_policy_preserving_primitive_path_fraction": (
                self.charge_clean_path_events / denominator
            ),
            "exact_macro_and_primitive_endpoint_fraction": (
                self.exact_endpoint_events / denominator
            ),
            "budget_fit": {
                str(budget): {
                    "events": int(self.budget_fit[budget]),
                    "fraction": self.budget_fit[budget] / denominator,
                }
                for budget in _BUDGETS
            },
        }

    def to_stratum_json(self) -> dict[str, object]:
        """Compact path-cost and integrity summary for one chemistry stratum."""

        denominator = max(self.events, 1)
        return {
            "teacher_events": self.events,
            "teacher_events_with_hard_failures": self.hard_failures,
            "hard_failure_fraction": self.hard_failures / denominator,
            "lowering_length_histogram": _counter(self.lowering_length),
            "system_cycle_rank_histogram": _counter(self.system_cycle_rank),
            "exact_macro_and_primitive_endpoint_fraction": (
                self.exact_endpoint_events / denominator
            ),
            "valid_primitive_path_fraction": self.valid_path_events / denominator,
            "connected_primitive_path_fraction": (self.connected_path_events / denominator),
            "charge_policy_preserving_primitive_path_fraction": (
                self.charge_clean_path_events / denominator
            ),
            "fully_declared_primitive7_event_fraction": (
                self.declared_membership["fully_declared"] / denominator
            ),
            "declared_primitive7_step_fraction": (
                self.declared_steps / self.total_steps if self.total_steps else 0.0
            ),
            "budget_fit": {
                str(budget): {
                    "events": int(self.budget_fit[budget]),
                    "fraction": self.budget_fit[budget] / denominator,
                }
                for budget in _BUDGETS
            },
        }


def _event_identity(
    *,
    address: Any,
    progress_index: int,
    result: Mapping[str, object],
) -> dict[str, object]:
    return {
        "packed_shard_content_sha256": address.packed_shard_content_sha256,
        "entry_index": int(address.entry_index),
        "trace_id": address.trace_id,
        "progress_index": int(progress_index),
        "source_state_sha256": result["source_state_sha256"],
        "action_sha256": result["action_sha256"],
        "stored_successor_state_sha256": result["stored_successor_state_sha256"],
    }


def _chemistry_stratum_keys(
    result: Mapping[str, object],
) -> tuple[str, ...]:
    """Return overlapping, explicitly named chemistry strata for one teacher."""

    strata = result["strata"]
    if not strata:
        return ()
    direction = str(strata["direction"])
    ring_chemistry = "heterocycle" if bool(strata["heterocycle"]) else "carbocycle"

    def electronics(prefix: str) -> str:
        if int(strata[f"{prefix}_system_aromatic_edge_count"]) > 0:
            return "aromatic"
        if bool(strata[f"{prefix}_system_saturated"]):
            return "saturated"
        return "unsaturated_nonaromatic"

    source_electronics = electronics("source")
    target_electronics = electronics("target")
    return (
        f"direction/{direction}",
        f"ring_chemistry/{ring_chemistry}",
        f"source_electronics/{source_electronics}",
        f"target_electronics/{target_electronics}",
        f"direction_x_ring_chemistry/{direction}|{ring_chemistry}",
    )


def audit_unified_ring_restate_teachers(
    inputs: RingRestateAuditInputs,
    *,
    progress_callback: Any | None = None,
) -> dict[str, object]:
    """Audit every addressable ring-restatement teacher in the unified corpus."""

    for name, path in (
        ("transfer root", inputs.transfer_root),
        ("unified manifest", inputs.unified_manifest),
        ("representability overlay", inputs.representability_overlay),
        ("charge-policy audit", inputs.charge_policy_audit),
        ("charge-policy exclusions", inputs.charge_policy_exclusions),
    ):
        if name == "transfer root":
            if not Path(path).is_dir():
                raise RingRestatePrimitivePathAuditError(f"{name} is absent: {path}")
        elif not Path(path).is_file():
            raise RingRestatePrimitivePathAuditError(f"{name} is absent: {path}")

    manifest, shards = resolve_unified_manifest_shards(
        inputs.unified_manifest,
        audit_root=inputs.transfer_root / "edit_packed_v1",
        mmp_root=inputs.transfer_root / "mmp_packed_v1",
    )
    representability = load_overlay(inputs.representability_overlay)
    charge_audit = _load_json_object(inputs.charge_policy_audit)
    charge_exclusions = _load_json_object(inputs.charge_policy_exclusions)
    charge_audit_sha256 = _sha256_file(inputs.charge_policy_audit)
    charge_exclusions_sha256 = _sha256_file(inputs.charge_policy_exclusions)
    charge_index = validate_charge_policy_exclusions(
        audit=charge_audit,
        audit_file_sha256=charge_audit_sha256,
        exclusions=charge_exclusions,
        exclusions_file_sha256=charge_exclusions_sha256,
    )

    global_aggregate = _Aggregate()
    selection_aggregate = _Aggregate()
    partition_aggregates: defaultdict[str, _Aggregate] = defaultdict(_Aggregate)
    layer_aggregates: defaultdict[str, _Aggregate] = defaultdict(_Aggregate)
    chemistry_strata_all: defaultdict[str, _Aggregate] = defaultdict(_Aggregate)
    chemistry_strata_selection: defaultdict[str, _Aggregate] = defaultdict(_Aggregate)
    trace_counts: Counter[str] = Counter()
    teacher_counts: Counter[str] = Counter()
    per_shard: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    audited_event_identities: list[dict[str, object]] = []
    seen_charge_addresses: set[tuple[str, int]] = set()
    seen_representability_keys: set[tuple[str, str, str]] = set()
    audit_cache: dict[tuple[str, str, str], Mapping[str, object]] = {}
    cache_hits = 0

    for shard_index, declared in enumerate(shards):
        allowed_representability = excluded_keys(
            representability,
            declared.manifest_layer,
            declared.partition,
        )
        shard_trace_counts: Counter[str] = Counter()
        shard_teacher_counts: Counter[str] = Counter()
        observed_entries = 0
        for addressed in read_addressed_packed_shard(
            declared.path,
            verify_fraction=0.0,
        ):
            observed_entries += 1
            address = addressed.address
            if (
                address.layer != declared.envelope_layer
                or address.partition != declared.partition
                or address.packed_shard_name != declared.path.name
            ):
                raise RingRestatePrimitivePathAuditError(
                    "packed trace address disagrees with its unified-manifest lane"
                )
            trace_counts["raw"] += 1
            shard_trace_counts["raw"] += 1
            key = trace_key(addressed.trace)
            representability_excluded = check_unlisted(
                addressed.trace,
                key,
                allowed_representability,
                layer=declared.manifest_layer,
                partition=declared.partition,
            )
            if representability_excluded:
                trace_counts["representability_excluded"] += 1
                shard_trace_counts["representability_excluded"] += 1
                seen_representability_keys.add((declared.manifest_layer, declared.partition, key))
            charge_address = (
                address.packed_shard_content_sha256,
                int(address.entry_index),
            )
            excluded_trace_id = charge_index.get(charge_address)
            charge_excluded = excluded_trace_id is not None
            if charge_excluded:
                if excluded_trace_id != address.trace_id:
                    raise RingRestatePrimitivePathAuditError(
                        "charge exclusion trace ID disagrees with packed address"
                    )
                seen_charge_addresses.add(charge_address)
                trace_counts["charge_policy_excluded"] += 1
                shard_trace_counts["charge_policy_excluded"] += 1
            if representability_excluded and charge_excluded:
                trace_counts["overlay_intersection"] += 1
                shard_trace_counts["overlay_intersection"] += 1

            ring_progress = tuple(
                progress_index
                for progress_index, step in enumerate(addressed.trace.steps)
                if step.rule_name == "ring_system_restate"
            )
            teacher_counts["raw"] += len(ring_progress)
            shard_teacher_counts["raw"] += len(ring_progress)
            if representability_excluded:
                teacher_counts["representability_excluded"] += len(ring_progress)
                shard_teacher_counts["representability_excluded"] += len(ring_progress)
            if charge_excluded:
                teacher_counts["charge_policy_excluded"] += len(ring_progress)
                shard_teacher_counts["charge_policy_excluded"] += len(ring_progress)
            if representability_excluded or charge_excluded:
                continue

            trace_counts["audited_eligible"] += 1
            shard_trace_counts["audited_eligible"] += 1
            for progress_index in ring_progress:
                teacher_counts["audited"] += 1
                shard_teacher_counts["audited"] += 1
                source = addressed.path.state_at(progress_index)
                stored_successor = addressed.path.state_at(progress_index + 1)
                action = addressed.trace.steps[progress_index].action
                if not isinstance(action, RingSystemRestate):
                    raise RingRestatePrimitivePathAuditError(
                        "ring_system_restate teacher decoded to the wrong action type"
                    )
                source_digest = persistent_slot_state_sha256(source)
                target_digest = persistent_slot_state_sha256(stored_successor)
                action_digest = _stable_sha256(encode_action("ring_system_restate", action))
                cache_key = (source_digest, action_digest, target_digest)
                result = audit_cache.get(cache_key)
                if result is None:
                    result = audit_ring_restate_teacher(
                        source,
                        stored_successor,
                        action,
                    )
                    audit_cache[cache_key] = result
                else:
                    cache_hits += 1
                identity = _event_identity(
                    address=address,
                    progress_index=progress_index,
                    result=result,
                )
                audited_event_identities.append(identity)
                global_aggregate.add(result)
                partition_aggregates[declared.partition].add(result)
                layer_aggregates[declared.manifest_layer].add(result)
                for stratum_key in _chemistry_stratum_keys(result):
                    chemistry_strata_all[stratum_key].add(result)
                    if declared.partition in {"train", "validation"}:
                        chemistry_strata_selection[stratum_key].add(result)
                if declared.partition in {"train", "validation"}:
                    selection_aggregate.add(result)
                if result["failures"]:
                    failures.append(
                        {
                            "address": identity,
                            "layer": declared.manifest_layer,
                            "partition": declared.partition,
                            "failures": list(result["failures"]),
                        }
                    )
        per_shard.append(
            {
                "manifest_layer": declared.manifest_layer,
                "envelope_layer": declared.envelope_layer,
                "partition": declared.partition,
                "relative_path": declared.relative_path,
                "packed_shard_content_sha256": (_sha256_file(declared.path)),
                "observed_entries": observed_entries,
                "trace_counts": _counter(shard_trace_counts),
                "ring_system_restate_teacher_counts": _counter(shard_teacher_counts),
            }
        )
        if progress_callback is not None:
            progress_callback(
                {
                    "shard_index": shard_index + 1,
                    "shards_total": len(shards),
                    **per_shard[-1],
                }
            )

    if seen_charge_addresses != set(charge_index):
        missing = set(charge_index) - seen_charge_addresses
        unexpected = seen_charge_addresses - set(charge_index)
        raise RingRestatePrimitivePathAuditError(
            "whole-corpus scan and charge exclusion index disagree: "
            f"missing={len(missing)}, unexpected={len(unexpected)}"
        )
    expected_representability_keys = {
        (
            str(entry["layer"]),
            str(entry["partition"]),
            str(entry["trace_key"]),
        )
        for entry in representability["exclusions"]
    }
    if seen_representability_keys != expected_representability_keys:
        raise RingRestatePrimitivePathAuditError(
            "whole-corpus scan and representability overlay disagree"
        )
    expected_raw_traces = int(manifest["totals"]["entries"])
    if trace_counts["raw"] != expected_raw_traces:
        raise RingRestatePrimitivePathAuditError(
            f"unified trace census drifted: {trace_counts['raw']} != {expected_raw_traces}"
        )
    audited_event_identities.sort(
        key=lambda row: (
            row["packed_shard_content_sha256"],
            row["entry_index"],
            row["progress_index"],
        )
    )
    all_hard_checks_pass = not failures and teacher_counts["audited"] > 0
    implementation = implementation_provenance()
    report_body: dict[str, object] = {
        "schema": RING_RESTATE_PRIMITIVE_PATH_AUDIT_SCHEMA,
        "schema_version": RING_RESTATE_PRIMITIVE_PATH_AUDIT_SCHEMA_VERSION,
        "status": RING_RESTATE_PRIMITIVE_PATH_AUDIT_STATUS,
        "production_support_mutated": False,
        "training_authorized": False,
        "checkpoint_dependency": "none",
        "method": {
            "teacher_source": (
                "every addressed packed ring_system_restate teacher in the unified V1 corpus"
            ),
            "macro_executor": "de_novo_rewrite_system.apply",
            "authoritative_lowering": "lower_ring_system_restate",
            "exact_endpoint_identity": "persistent slot arrays and canonical molecular key",
            "validity": "is_valid_state",
            "connectedness": "is_connected_or_null",
            "charge_policy": (
                "charge_policy_preserved both source-to-intermediate and consecutive-intermediate"
            ),
            "declared_primitive7_membership": (
                "editing_operator_support._primitive_candidate_is_declared "
                "with the exact cyclic-bond exclusion short-circuit"
            ),
            "selection_partitions": ["train", "validation"],
            "test_partition_role": (
                "reported separately for completeness; not used for the V2 operator choice"
            ),
            "edit_budgets": list(_BUDGETS),
        },
        "inputs": {
            "unified_manifest_sha256": _sha256_file(inputs.unified_manifest),
            "unified_manifest_checksum": manifest["manifest_checksum"],
            "representability_overlay_sha256": _sha256_file(inputs.representability_overlay),
            "representability_effective_corpus_checksum": representability[
                "effective_corpus_checksum"
            ],
            "charge_policy_audit_file_sha256": charge_audit_sha256,
            "charge_policy_exclusions_file_sha256": charge_exclusions_sha256,
            "charge_policy_exclusion_payload_sha256": charge_exclusions["payload_sha256"],
            "charge_policy_source_input_inventory_sha256": charge_exclusions[
                "source_input_inventory_sha256"
            ],
            "declared_shards": len(shards),
        },
        "implementation": {
            "sources": dict(implementation["sources"]),
            "implementation_sha256": implementation["implementation_sha256"],
        },
        "coverage": {
            "trace_counts": _counter(trace_counts),
            "ring_system_restate_teacher_counts": _counter(teacher_counts),
            "unique_exact_teacher_cases": len(audit_cache),
            "reused_exact_teacher_case_count": cache_hits,
            "audited_event_identity_sha256": _stable_sha256(audited_event_identities),
            "charge_exclusion_addresses_seen": len(seen_charge_addresses),
            "representability_exclusions_seen": len(seen_representability_keys),
        },
        "all_partitions": global_aggregate.to_json(),
        "selection_partitions_train_validation": selection_aggregate.to_json(),
        "by_partition": {
            partition: partition_aggregates[partition].to_json()
            for partition in sorted(
                partition_aggregates,
                key=_PARTITION_ORDER.__getitem__,
            )
        },
        "by_layer": {
            layer: layer_aggregates[layer].to_json() for layer in sorted(layer_aggregates)
        },
        "by_chemistry_stratum_all_partitions": {
            key: chemistry_strata_all[key].to_stratum_json() for key in sorted(chemistry_strata_all)
        },
        "by_chemistry_stratum_selection_partitions": {
            key: chemistry_strata_selection[key].to_stratum_json()
            for key in sorted(chemistry_strata_selection)
        },
        "o0_operator_decision": {
            "decision_scope": "imminent bounded editing pilots only",
            "production_operator_freeze": "still pending bounded pilot evidence",
            "recommended_bounded_pilot_basis": ("primitive7_plus_ring_system_restate"),
            "evidence_population": {
                "partitions": ["train", "validation"],
                "teacher_events": selection_aggregate.events,
                "test_partition_used_for_decision": False,
            },
            "current_primitive7": {
                "authoritative_lowering_paths_fully_in_support": (
                    selection_aggregate.declared_membership["fully_declared"]
                ),
                "authoritative_lowering_steps_in_support": (selection_aggregate.declared_steps),
                "authoritative_lowering_steps_total": (selection_aggregate.total_steps),
                "blocking_semantic": (
                    "cyclic bond_reorder is masked by the current primitive7 O0 basis"
                ),
                "conclusion": (
                    "current primitive7 does not realize any audited "
                    "authoritative ring-system-restatement lowering"
                ),
            },
            "primitive7_plus_ring_system_restate": {
                "committed_event_cost": 1,
                "certified_exact_teacher_endpoints": (selection_aggregate.exact_endpoint_events),
                "certified_valid_connected_charge_clean_path_events": min(
                    selection_aggregate.valid_path_events,
                    selection_aggregate.connected_path_events,
                    selection_aggregate.charge_clean_path_events,
                ),
                "conclusion": (
                    "ring_system_restate supplies certified one-step support "
                    "for every audited selection-partition teacher endpoint"
                ),
            },
            "hypothetical_widened_cyclic_bond_reorder_basis": {
                "status": "separate_unimplemented_untested_operator_change",
                "conditional_reachability_statement": (
                    "if the bond_reorder family were widened to admit the exact "
                    "audited cyclic reorder instructions, every audited "
                    "teacher endpoint would be reachable by its authoritative "
                    "2-to-20-step lowering"
                ),
                "lowering_length_minimum": min(selection_aggregate.lowering_length),
                "lowering_length_maximum": max(selection_aggregate.lowering_length),
                "fits_budget_8": {
                    "events": selection_aggregate.budget_fit[8],
                    "fraction": (selection_aggregate.budget_fit[8] / selection_aggregate.events),
                },
                "fits_budget_16": {
                    "events": selection_aggregate.budget_fit[16],
                    "fraction": (selection_aggregate.budget_fit[16] / selection_aggregate.events),
                },
                "candidate_implication": (
                    "the widened basis would expose state-dependent scalar "
                    "cyclic-reorder choices at successive intermediates, "
                    "whereas restate exposes coordinated complete marks"
                ),
                "candidate_count_boundary": (
                    "this teacher-path audit does not enumerate or compare "
                    "complete candidate-fiber sizes"
                ),
            },
            "budget_implication": (
                "the macro consumes one committed edit; its certified "
                "primitive-equivalent cost is the audited lowering length, so "
                "dropping the macro in favor of a hypothetical widened basis "
                "would spend 2-to-20 edit events before any other objective "
                "repair"
            ),
            "pilot_rationale": [
                "the current packed corpus contains direct restatement teachers",
                "all audited macro actions and lowerings are exact and validity closed",
                "none of their authoritative lowerings is in current primitive7 support",
                "retaining restate avoids simultaneously changing support and recompiling supervision before the bounded capacity pilot",
            ],
            "claim_boundary": (
                "this supports retaining restate for bounded pilots; it does "
                "not establish learned superiority over a separately "
                "implemented and trained widened cyclic-bond-reorder basis"
            ),
        },
        "per_shard": per_shard,
        "failures": failures,
        "all_hard_checks_pass": all_hard_checks_pass,
        "interpretation_boundary": {
            "establishes": [
                "production macro versus packed exact endpoint equality",
                "authoritative lowering endpoint equality",
                "validity, connectedness, and charge preservation of every intermediate",
                "lowering-length and registered-budget coverage",
                "membership of authoritative lowering steps in current primitive7 support",
            ],
            "does_not_establish": [
                "learned probability or calibration",
                "whether another non-authoritative primitive7 path reaches the same endpoint",
                "whether a macro is easier to learn than a widened cyclic bond-reorder family",
                "complete candidate-fiber size or normalization burden under either basis",
                "final operator choice without bounded pilot evidence",
            ],
        },
    }
    report = {
        **report_body,
        "artifact_sha256": _stable_sha256(report_body),
    }
    return report


__all__ = [
    "RING_RESTATE_PRIMITIVE_PATH_AUDIT_SCHEMA",
    "RING_RESTATE_PRIMITIVE_PATH_AUDIT_SCHEMA_VERSION",
    "RING_RESTATE_PRIMITIVE_PATH_AUDIT_STATUS",
    "RingRestateAuditInputs",
    "RingRestatePrimitivePathAuditError",
    "audit_ring_restate_teacher",
    "audit_unified_ring_restate_teachers",
    "implementation_provenance",
]
