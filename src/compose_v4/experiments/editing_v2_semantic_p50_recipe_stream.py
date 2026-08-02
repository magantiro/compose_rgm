"""Prospective semantic Editing-V2 P50 recipe and exact-stream compiler.

This module freezes the bounded Stage-A capability pilot without launching it.
It consumes the strict semantic Active8 source, an exact semantic Gate0 PASS,
and an exact semantic T1 GO.  It then classifies the complete admitted train
stream once, builds the deterministic equal-cell 3,200-address schedule, gives
every scheduled row an explicit deterministic time, and records the complete-
trace closure required by the successor-fiber cache.

The compiled artifact is deliberately non-authorizing.  A separate promotion
step can produce the authorizing schema only after five purpose-specific,
physical binding receipts have been validated.  Test fixtures can exercise the
promotion schema, but this module never fabricates production receipts or
launches training.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from compose_v4.data.editing_corpus_contract import ACTIVE8_FAMILIES
from compose_v4.data.editing_v2_semantic_capability_cells import (
    SemanticCapabilityCellRegistry,
    classify_verified_structural_transition,
    load_semantic_capability_cell_registry,
)
from compose_v4.data.successor_fiber_cache import SuccessorFiberCacheAddress
from compose_v4.experiments.editing_stage_a_semantic_cell_scheduler import (
    StageASemanticCellScheduler,
)
from compose_v4.experiments.editing_v2_semantic_p50_source_inventory import (
    VerifiedSemanticP50SourceInventory,
)
from compose_v4.experiments.editing_v2_semantic_t1_decision import (
    DECISION_GO_STATUS,
    validate_semantic_gate_zero_evidence_receipt,
    validate_semantic_t1_capacity_decision,
)

PREPARED_SCHEMA = "compose.editing_v2.semantic_p50_prepared_recipe"
SCHEMA_VERSION = 1

PREPARED_STATUS = "PREPARED_EXACT_STREAM_PENDING_PHYSICAL_BINDINGS_NO_AUTHORITY"

OPTIMIZER_STEPS = 50
BATCH_SIZE = 64
SCHEDULED_EXAMPLES = OPTIMIZER_STEPS * BATCH_SIZE
SEED = 31
GRADIENT_OPPORTUNITY_FRACTION = 0.8
MAXIMUM_FAMILY_NLL_REGRESSION_NATS = 0.25
MAXIMUM_CELL_NLL_REGRESSION_NATS = 0.25
TIME_ALGORITHM = "sha256_counter_open_unit_interval_53bit_v1"
SAMPLING_POLICY = "stage_a_equal_semantic_cell_capability_exception_v1"
POLICY_RELATIVE_PATH = "configs/editing_v2_semantic_p50_recipe_policy_v1.json"
POLICY_SCHEMA = "compose.editing_v2.semantic_p50_recipe_policy"

REQUIRED_BINDING_PURPOSES = (
    "stream_union_successor_cache",
    "validation_baseline",
    "trainer_runtime",
    "execution_environment",
    "launch_projection",
)

_NO_AUTHORITY = {
    "training_authorized": False,
    "bounded_p50_authorized": False,
    "p500_authorized": False,
    "long_training_authorized": False,
    "checkpoint_selection_authorized": False,
    "final_test_selection_authorized": False,
}
_HEX = frozenset("0123456789abcdef")
_POLICY_FIELDS = {
    "schema",
    "schema_version",
    "policy_id",
    "status",
    *_NO_AUTHORITY,
    "scientific_scope",
    "active_families",
    "optimization",
    "objective",
    "sampling",
    "time_derivation",
    "cache",
    "thresholds",
    "required_physical_binding_purposes",
    "policy_sha256",
}
_EXPECTED_OPTIMIZATION = {
    "initialization": "scratch_from_t1_bound_initial_model_state",
    "optimizer_steps": OPTIMIZER_STEPS,
    "batch_size": BATCH_SIZE,
    "scheduled_nonterminal_examples": SCHEDULED_EXAMPLES,
    "optimizer": "adamw",
    "learning_rate": 0.001,
    "weight_decay": 0.0,
    "scheduler": "constant",
    "gradient_clip_norm": 10.0,
    "seed": SEED,
    "dtype": "float32",
    "mixed_precision": False,
    "deterministic_algorithms_required": True,
    "resume": False,
}
_EXPECTED_OBJECTIVE = {
    "name": "balanced_semantic_cell_productive_identity",
    "unit": "productive_embedded_canonical_successor",
    "hazard_included": False,
    "hazard_weight": 0.0,
    "terminal_rows": "excluded",
    "importance_correction": "none",
    "path_position_coefficient": 1.0,
}
_EXPECTED_TIME_DERIVATION = {
    "algorithm": TIME_ALGORITHM,
    "seed": SEED,
    "support": "strict_open_unit_interval",
    "serialized_value": "python_float_hex_v1",
}
_EXPECTED_SAMPLING = {
    "policy_id": SAMPLING_POLICY,
    "semantic_cell_probability": "equal_round_robin",
    "within_cell_probability": "deterministic_uniform_cycle",
    "lane_probability": "not_a_sampling_level_preserved_as_audit_dimension",
    "source_group_probability": "explicit_stage_a_exception_not_used",
    "source_group_id_in_stream": None,
    "production_hierarchy_claim_authorized": False,
    "rationale": "P50 tests load-bearing capability acquisition; production-law calibration remains a later separately frozen stage",
}
_EXPECTED_CACHE = {
    "coverage_mode": "complete_trace_closure_of_planned_address_union",
    "source_population": "full_admitted_semantic_active8_corpus",
    "closure_only_rows_schedulable": False,
    "terminal_rows_schedulable": False,
}
_EXPECTED_THRESHOLDS = {
    "gradient_opportunity_fraction": GRADIENT_OPPORTUNITY_FRACTION,
    "gradient_floor_rule": "max_1_ceil_fraction_times_planned_optimizer_step_opportunities",
    "zero_planned_family_or_cell_opportunities_allowed": False,
    "maximum_family_final_minus_baseline_successor_nll_nats": (MAXIMUM_FAMILY_NLL_REGRESSION_NATS),
    "maximum_cell_final_minus_baseline_successor_nll_nats": (MAXIMUM_CELL_NLL_REGRESSION_NATS),
    "baseline_partition_role": "validation",
    "baseline_definition": "exact_pre_update_scratch_evaluation_on_the_frozen_validation_stream",
    "baseline_values_inspected_when_thresholds_frozen": False,
}


class SemanticP50RecipeStreamError(ValueError):
    """P50 prerequisites, stream, closure, or physical bindings disagree."""


def _canonical_bytes(value: object, *, newline: bool = False) -> bytes:
    try:
        encoded = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise SemanticP50RecipeStreamError(
            "semantic P50 artifact is not finite canonical JSON"
        ) from error
    return encoded + (b"\n" if newline else b"")


def _sha(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _sha_sequence(values: Iterable[object]) -> str:
    """Hash a canonical JSON array without materializing a second full array."""

    digest = hashlib.sha256(b"[")
    for index, value in enumerate(values):
        if index:
            digest.update(b",")
        digest.update(_canonical_bytes(value))
    digest.update(b"]")
    return digest.hexdigest()


def _file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def _require_sha(value: object, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _HEX for character in value)
    ):
        raise SemanticP50RecipeStreamError(f"{field} must be a lowercase SHA-256")
    return value


def _require_text(value: object, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value.strip() != value
        or any(ord(character) < 32 for character in value)
    ):
        raise SemanticP50RecipeStreamError(f"{field} must be normalized nonempty text")
    return value


def _address_payload(address: SuccessorFiberCacheAddress) -> dict[str, object]:
    return asdict(address)


def _write_canonical(path: Path, payload: Mapping[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    encoded = _canonical_bytes(payload, newline=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def load_semantic_p50_recipe_policy(
    path: Path | None = None,
) -> tuple[dict[str, Any], str]:
    """Load and validate the prospective policy that defines this compiler."""

    selected = (
        Path(path)
        if path is not None
        else Path(__file__).resolve().parents[3] / POLICY_RELATIVE_PATH
    )
    try:
        raw = selected.read_bytes()
        policy = json.loads(raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SemanticP50RecipeStreamError(
            f"semantic P50 recipe policy is unreadable: {selected}"
        ) from error
    if not isinstance(policy, dict):
        raise SemanticP50RecipeStreamError("semantic P50 recipe policy is not an object")
    body = dict(policy)
    supplied = body.pop("policy_sha256", None)
    optimization = policy.get("optimization")
    objective = policy.get("objective")
    sampling = policy.get("sampling")
    time_derivation = policy.get("time_derivation")
    cache = policy.get("cache")
    thresholds = policy.get("thresholds")
    if not all(
        isinstance(item, Mapping)
        for item in (
            optimization,
            objective,
            sampling,
            time_derivation,
            cache,
            thresholds,
        )
    ):
        raise SemanticP50RecipeStreamError("semantic P50 recipe policy sections are incomplete")
    assert isinstance(optimization, Mapping)
    assert isinstance(objective, Mapping)
    assert isinstance(sampling, Mapping)
    assert isinstance(time_derivation, Mapping)
    assert isinstance(cache, Mapping)
    assert isinstance(thresholds, Mapping)
    if (
        set(policy) != _POLICY_FIELDS
        or policy.get("schema") != POLICY_SCHEMA
        or policy.get("schema_version") != SCHEMA_VERSION
        or policy.get("status") != "FROZEN_PROSPECTIVE_RECIPE_NO_TRAINING_AUTHORITY"
        or supplied != _sha(body)
        or any(policy.get(key) is not value for key, value in _NO_AUTHORITY.items())
        or tuple(policy.get("active_families", ())) != ACTIVE8_FAMILIES
        or dict(optimization) != _EXPECTED_OPTIMIZATION
        or dict(objective) != _EXPECTED_OBJECTIVE
        or dict(sampling) != _EXPECTED_SAMPLING
        or dict(time_derivation) != _EXPECTED_TIME_DERIVATION
        or dict(cache) != _EXPECTED_CACHE
        or dict(thresholds) != _EXPECTED_THRESHOLDS
        or tuple(policy.get("required_physical_binding_purposes", ())) != REQUIRED_BINDING_PURPOSES
    ):
        raise SemanticP50RecipeStreamError(
            "semantic P50 recipe policy identity or supported projection disagrees"
        )
    return policy, hashlib.sha256(raw).hexdigest()


@dataclass(frozen=True, slots=True)
class SemanticP50Candidate:
    """One exact train transition eligible for the Stage-A scheduler."""

    address: SuccessorFiberCacheAddress
    family: str
    semantic_cell_id: str
    data_lane: str
    assignment_sha256: str

    def __post_init__(self) -> None:
        if self.address.is_terminal or self.address.partition != "train":
            raise SemanticP50RecipeStreamError(
                "semantic P50 candidates must be nonterminal train addresses"
            )
        if self.family not in ACTIVE8_FAMILIES:
            raise SemanticP50RecipeStreamError("semantic P50 candidate is outside Active8")
        components = self.semantic_cell_id.rsplit(":", 2)
        if len(components) != 3 or components[1] != self.family or not components[2]:
            raise SemanticP50RecipeStreamError(
                "semantic P50 cell identity disagrees with its family"
            )
        _require_text(self.semantic_cell_id, field="semantic_cell_id")
        _require_text(self.data_lane, field="data_lane")
        if self.address.layer != self.data_lane:
            raise SemanticP50RecipeStreamError(
                "semantic P50 candidate lane differs from its exact address"
            )
        _require_sha(self.assignment_sha256, field="assignment_sha256")

    def as_payload(self) -> dict[str, object]:
        return {
            "address": _address_payload(self.address),
            "family": self.family,
            "semantic_cell_id": self.semantic_cell_id,
            "data_lane": self.data_lane,
            "assignment_sha256": self.assignment_sha256,
        }


@dataclass(frozen=True, slots=True)
class SemanticP50Prerequisites:
    """Exact source, Gate0, T1, registry, and scratch identities."""

    source_inventory_file_sha256: str
    source_inventory_sha256: str
    process_identity_sha256: str
    model_runtime_identity_sha256: str
    active8_policy_sha256: str
    operator_capability_fingerprint: str
    decision_source_implementation_sha256: str
    gate_zero_evidence_file_sha256: str
    gate_zero_evidence_sha256: str
    t1_decision_file_sha256: str
    t1_decision_sha256: str
    t1_selected_model_state_sha256: str
    scratch_initial_model_state_sha256: str
    capability_registry_sha256: str
    classifier_implementation_sha256: str

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            value = getattr(self, name)
            if name == "operator_capability_fingerprint":
                if (
                    not isinstance(value, str)
                    or len(value) != 16
                    or any(character not in _HEX for character in value)
                ):
                    raise SemanticP50RecipeStreamError(
                        "operator_capability_fingerprint must be 16 lowercase hex characters"
                    )
            else:
                _require_sha(value, field=name)

    def as_payload(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class SemanticP50CandidateInventory:
    """Complete classified train inventory plus exact prerequisite identities."""

    prerequisites: SemanticP50Prerequisites
    candidates: tuple[SemanticP50Candidate, ...]
    candidate_inventory_sha256: str = ""

    def __post_init__(self) -> None:
        normalized = tuple(sorted(self.candidates, key=lambda item: item.address))
        if not normalized:
            raise SemanticP50RecipeStreamError("semantic P50 candidate inventory is empty")
        if normalized != self.candidates:
            object.__setattr__(self, "candidates", normalized)
        addresses = tuple(candidate.address for candidate in normalized)
        if len(set(addresses)) != len(addresses):
            raise SemanticP50RecipeStreamError(
                "semantic P50 candidate inventory contains duplicate addresses"
            )
        expected = _sha_sequence(
            normalized_candidate.as_payload() for normalized_candidate in normalized
        )
        if self.candidate_inventory_sha256:
            if self.candidate_inventory_sha256 != expected:
                raise SemanticP50RecipeStreamError(
                    "semantic P50 candidate inventory self-hash disagrees"
                )
        else:
            object.__setattr__(self, "candidate_inventory_sha256", expected)


def validate_semantic_p50_prerequisite_relationships(
    *,
    source: VerifiedSemanticP50SourceInventory,
    gate_zero: Mapping[str, Any],
    gate_zero_file_sha256: str,
    t1_decision: Mapping[str, Any],
    t1_decision_file_sha256: str,
    registry: SemanticCapabilityCellRegistry,
) -> SemanticP50Prerequisites:
    """Validate cross-artifact identities after each physical chain was reopened."""

    if not isinstance(source, VerifiedSemanticP50SourceInventory):
        raise TypeError("semantic P50 requires a verified semantic source inventory")
    if not isinstance(gate_zero, Mapping) or not isinstance(t1_decision, Mapping):
        raise TypeError("semantic P50 Gate0 and T1 prerequisites must be mappings")
    if not isinstance(registry, SemanticCapabilityCellRegistry):
        raise TypeError("semantic P50 requires the verified capability registry")

    source_identity = source.index.identity_payload()
    runtime = gate_zero.get("model_runtime_identity")
    capability = gate_zero.get("capability_registry")
    if not isinstance(runtime, Mapping) or not isinstance(capability, Mapping):
        raise SemanticP50RecipeStreamError(
            "semantic Gate0 lacks model-runtime or capability-registry identity"
        )
    required_relationships = (
        gate_zero.get("structural_result") == "PASS",
        gate_zero.get("decision_source_identity") == source_identity,
        gate_zero.get("decision_source_inventory_sha256") == source.binding.source_inventory_sha256,
        runtime.get("identity_sha256") == source.binding.model_runtime_identity_sha256,
        runtime.get("process_identity_sha256") == source.binding.process_identity_sha256,
        gate_zero.get("active8_policy", {}).get("policy_sha256")
        == source.binding.active8_policy_sha256,
        capability.get("registry_sha256") == registry.registry_sha256,
        capability.get("classifier_implementation_sha256")
        == registry.classifier_implementation_sha256,
        t1_decision.get("status") == DECISION_GO_STATUS,
        t1_decision.get("bounded_p50_authorized") is True,
        t1_decision.get("failed_checks") == [],
        t1_decision.get("decision_source_inventory_sha256")
        == source.binding.source_inventory_sha256,
        t1_decision.get("gate_zero_evidence_file_sha256") == gate_zero_file_sha256,
        tuple(t1_decision.get("required_families", ())) == ACTIVE8_FAMILIES,
        t1_decision.get("initial_model_state_sha256") == runtime.get("initial_model_state_sha256"),
    )
    if not all(required_relationships):
        raise SemanticP50RecipeStreamError(
            "semantic source, Gate0, T1, registry, or scratch identity disagrees"
        )
    return SemanticP50Prerequisites(
        source_inventory_file_sha256=source.binding.source_inventory_file_sha256,
        source_inventory_sha256=source.binding.source_inventory_sha256,
        process_identity_sha256=source.binding.process_identity_sha256,
        model_runtime_identity_sha256=source.binding.model_runtime_identity_sha256,
        active8_policy_sha256=source.binding.active8_policy_sha256,
        operator_capability_fingerprint=(source.binding.operator_capability_fingerprint),
        decision_source_implementation_sha256=(
            source.binding.decision_source_implementation_sha256
        ),
        gate_zero_evidence_file_sha256=_require_sha(
            gate_zero_file_sha256, field="gate_zero_evidence_file_sha256"
        ),
        gate_zero_evidence_sha256=_require_sha(
            gate_zero.get("evidence_sha256"), field="gate_zero_evidence_sha256"
        ),
        t1_decision_file_sha256=_require_sha(
            t1_decision_file_sha256, field="t1_decision_file_sha256"
        ),
        t1_decision_sha256=_require_sha(
            t1_decision.get("decision_sha256"), field="t1_decision_sha256"
        ),
        t1_selected_model_state_sha256=_require_sha(
            t1_decision.get("selected_model_state_sha256"),
            field="t1_selected_model_state_sha256",
        ),
        scratch_initial_model_state_sha256=_require_sha(
            t1_decision.get("initial_model_state_sha256"),
            field="scratch_initial_model_state_sha256",
        ),
        capability_registry_sha256=registry.registry_sha256,
        classifier_implementation_sha256=registry.classifier_implementation_sha256,
    )


def build_semantic_p50_candidate_inventory(
    *,
    source: VerifiedSemanticP50SourceInventory,
    gate_zero_evidence_path: Path,
    t1_decision_path: Path,
    repo_root: Path,
    registry: SemanticCapabilityCellRegistry | None = None,
) -> SemanticP50CandidateInventory:
    """Reopen prerequisites and classify the admitted train stream exactly once."""

    selected_registry = registry or load_semantic_capability_cell_registry()
    gate_path = Path(gate_zero_evidence_path).resolve()
    t1_path = Path(t1_decision_path).resolve()
    gate_zero = validate_semantic_gate_zero_evidence_receipt(gate_path)
    gate_file_sha = _file_sha(gate_path)
    try:
        t1_payload = json.loads(t1_path.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SemanticP50RecipeStreamError("semantic T1 decision is unreadable") from error
    t1_decision = validate_semantic_t1_capacity_decision(
        t1_payload,
        decision_path=t1_path,
        repo_root=Path(repo_root),
        expected_gate_zero_evidence_file_sha256=gate_file_sha,
        require_p50_go=True,
    )
    prerequisites = validate_semantic_p50_prerequisite_relationships(
        source=source,
        gate_zero=gate_zero,
        gate_zero_file_sha256=gate_file_sha,
        t1_decision=t1_decision,
        t1_decision_file_sha256=_file_sha(t1_path),
        registry=selected_registry,
    )

    candidates: list[SemanticP50Candidate] = []
    assignment_stream = hashlib.sha256()
    cell_counts: Counter[str] = Counter()
    family_counts: Counter[str] = Counter()
    lane_counts: Counter[str] = Counter()
    for trace in source.index.iter_accepted_traces_for_partition("train"):
        for transition in source.index.accepted_transitions_for(trace):
            assignment = classify_verified_structural_transition(
                source.index, transition, registry=selected_registry
            )
            assignment_stream.update(_canonical_bytes(assignment.as_payload(), newline=True))
            address = SuccessorFiberCacheAddress.from_packed_trace(
                transition.addressed_trace.address,
                progress_index=transition.step_index,
            )
            candidate = SemanticP50Candidate(
                address=address,
                family=assignment.model_family,
                semantic_cell_id=assignment.capability_cell_id,
                data_lane=assignment.data_lane,
                assignment_sha256=assignment.assignment_sha256,
            )
            candidates.append(candidate)
            cell_counts[candidate.semantic_cell_id] += 1
            family_counts[candidate.family] += 1
            lane_counts[candidate.data_lane] += 1

    expected_cells = {
        str(row["capability_cell_id"]): int(row["teacher_count"])
        for row in gate_zero["capability_cell_counts"]
    }
    expected_families = {
        str(key): int(value)
        for key, value in gate_zero["decision_eligible_teacher_counts_by_family"].items()
    }
    expected_lanes = {
        str(key): int(value)
        for key, value in gate_zero["decision_eligible_teacher_counts_by_lane"].items()
    }
    if (
        assignment_stream.hexdigest() != gate_zero["structural_assignment_inventory_sha256"]
        or dict(cell_counts) != {key: value for key, value in expected_cells.items() if value}
        or dict(family_counts) != expected_families
        or dict(lane_counts) != expected_lanes
    ):
        raise SemanticP50RecipeStreamError(
            "semantic P50 reclassification differs from exact Gate0 evidence"
        )
    ordered = tuple(sorted(candidates, key=lambda item: item.address))
    return SemanticP50CandidateInventory(
        prerequisites=prerequisites,
        candidates=ordered,
    )


def semantic_p50_time_hex(
    *, stream_index: int, address: SuccessorFiberCacheAddress
) -> str:
    """Derive the frozen deterministic P50 time coordinate for one row."""

    if type(stream_index) is not int or stream_index < 0:
        raise SemanticP50RecipeStreamError(
            "semantic P50 time stream_index must be a nonnegative integer"
        )
    digest = hashlib.sha256(
        _canonical_bytes(
            {
                "algorithm": TIME_ALGORITHM,
                "seed": SEED,
                "stream_index": stream_index,
                "address": _address_payload(address),
            }
        )
    ).digest()
    numerator = (int.from_bytes(digest[:8], "big") >> 11) + 1
    value = numerator / ((1 << 53) + 1)
    if not 0.0 < value < 1.0:
        raise AssertionError("open-unit-interval time derivation escaped its support")
    return value.hex()


def _opportunity_rows(
    rows: Sequence[Mapping[str, Any]], *, field: str, values: Sequence[str]
) -> tuple[list[dict[str, object]], dict[str, int]]:
    steps: dict[str, set[int]] = {value: set() for value in values}
    draws: Counter[str] = Counter()
    for row in rows:
        value = str(row[field])
        if value not in steps:
            raise SemanticP50RecipeStreamError(f"scheduled row contains unknown {field}: {value!r}")
        steps[value].add(int(row["optimizer_step"]))
        draws[value] += 1
    reports: list[dict[str, object]] = []
    floors: dict[str, int] = {}
    for value in values:
        opportunity_count = len(steps[value])
        if opportunity_count == 0 or draws[value] == 0:
            raise SemanticP50RecipeStreamError(
                f"required {field} {value!r} has zero planned optimization exposure"
            )
        floor = max(1, math.ceil(GRADIENT_OPPORTUNITY_FRACTION * opportunity_count))
        floors[value] = floor
        reports.append(
            {
                field: value,
                "scheduled_draw_count": draws[value],
                "optimizer_step_opportunity_count": opportunity_count,
                "minimum_observed_nonzero_gradient_update_count": floor,
            }
        )
    return reports, floors


def compile_semantic_p50_prepared_recipe(
    inventory: SemanticP50CandidateInventory,
    *,
    policy_path: Path | None = None,
) -> dict[str, Any]:
    """Compile the exact 50x64 stream and cache closure without authority."""

    if not isinstance(inventory, SemanticP50CandidateInventory):
        raise TypeError("semantic P50 compiler requires a candidate inventory")
    policy, policy_file_sha256 = load_semantic_p50_recipe_policy(policy_path)
    by_cell: dict[str, list[SuccessorFiberCacheAddress]] = defaultdict(list)
    candidate_by_address: dict[SuccessorFiberCacheAddress, SemanticP50Candidate] = {}
    for candidate in inventory.candidates:
        by_cell[candidate.semantic_cell_id].append(candidate.address)
        candidate_by_address[candidate.address] = candidate
    cells = tuple(sorted(by_cell))
    scheduler = StageASemanticCellScheduler(
        declared_cells=cells,
        examples_by_cell=by_cell,
        seed=SEED,
    )
    scheduled, cursor = scheduler.take(scheduler.initial_cursor(), SCHEDULED_EXAMPLES)
    if cursor.next_stream_index != SCHEDULED_EXAMPLES:
        raise AssertionError("semantic P50 scheduler returned an incomplete stream")

    rows: list[dict[str, Any]] = []
    for example in scheduled.examples:
        candidate = candidate_by_address[example.address]
        if candidate.semantic_cell_id != example.semantic_cell_id:
            raise SemanticP50RecipeStreamError(
                "semantic scheduler cell differs from candidate identity"
            )
        row_body = {
            "stream_index": example.stream_index,
            "optimizer_step": example.stream_index // BATCH_SIZE,
            "batch_offset": example.stream_index % BATCH_SIZE,
            "address": _address_payload(example.address),
            "family": candidate.family,
            "semantic_cell_id": candidate.semantic_cell_id,
            "data_lane": candidate.data_lane,
            "source_group_id": None,
            "time_hex": semantic_p50_time_hex(
                stream_index=example.stream_index, address=example.address
            ),
            "identity_coefficient": 1.0,
            "importance_correction": None,
            "path_position_coefficient": 1.0,
        }
        rows.append({**row_body, "row_sha256": _sha(row_body)})
    if len(rows) != SCHEDULED_EXAMPLES:
        raise AssertionError("semantic P50 stream length disagrees")

    requested_addresses = tuple(
        sorted({SuccessorFiberCacheAddress(**row["address"]) for row in rows})
    )
    requested_set = set(requested_addresses)
    touched_trace_representatives = {address.trace_key: address for address in requested_addresses}
    closure_addresses: list[SuccessorFiberCacheAddress] = []
    for _, representative in sorted(touched_trace_representatives.items()):
        closure_addresses.extend(
            SuccessorFiberCacheAddress(
                **{
                    **_address_payload(representative),
                    "progress_index": progress_index,
                }
            )
            for progress_index in range(representative.path_length + 1)
        )
    closure_addresses = sorted(set(closure_addresses))
    closure_rows = [
        {
            "address": _address_payload(address),
            "requested_for_training": address in requested_set,
            "closure_only": address not in requested_set,
            "schedulable": address in requested_set and not address.is_terminal,
            "terminal": address.is_terminal,
        }
        for address in closure_addresses
    ]
    if (
        not closure_rows
        or any(row["terminal"] and row["schedulable"] for row in closure_rows)
        or any(row["closure_only"] and row["schedulable"] for row in closure_rows)
        or any(address.is_terminal for address in requested_addresses)
    ):
        raise SemanticP50RecipeStreamError(
            "complete-trace closure leaked a terminal or closure-only row into scheduling"
        )
    for trace_key, representative in touched_trace_representatives.items():
        indices = {
            address.progress_index
            for address in closure_addresses
            if address.trace_key == trace_key
        }
        if indices != set(range(representative.path_length + 1)):
            raise SemanticP50RecipeStreamError(
                "semantic P50 cache closure is missing trace progress rows"
            )

    family_reports, family_floors = _opportunity_rows(rows, field="family", values=ACTIVE8_FAMILIES)
    cell_reports, cell_floors = _opportunity_rows(rows, field="semantic_cell_id", values=cells)
    lane_counts = Counter(str(row["data_lane"]) for row in rows)
    ordered_address_stream_sha256 = _sha([row["address"] for row in rows])
    ordered_training_stream_sha256 = _sha(rows)
    requested_union_sha256 = _sha([_address_payload(address) for address in requested_addresses])
    closure_inventory_sha256 = _sha(closure_rows)
    stream_contract = {
        "optimizer_steps": OPTIMIZER_STEPS,
        "batch_size": BATCH_SIZE,
        "scheduled_nonterminal_examples": SCHEDULED_EXAMPLES,
        "scheduler": scheduler.identity_payload(),
        "scheduler_terminal_cursor": cursor.to_payload(),
        "time_derivation": dict(policy["time_derivation"]),
        "sampling_policy": {
            **dict(policy["sampling"]),
            "scientific_scope": "stage_a_capability_acquisition_not_production_law_calibration",
        },
        "objective": dict(policy["objective"]),
    }
    thresholds = policy["thresholds"]
    validation_contract = {
        "partition_role": thresholds["baseline_partition_role"],
        "baseline": thresholds["baseline_definition"],
        "baseline_artifact_required_before_authority": True,
        "required_families": list(ACTIVE8_FAMILIES),
        "required_nonempty_semantic_cells": list(cells),
        "maximum_family_final_minus_baseline_successor_nll_nats": (
            thresholds["maximum_family_final_minus_baseline_successor_nll_nats"]
        ),
        "maximum_cell_final_minus_baseline_successor_nll_nats": (
            thresholds["maximum_cell_final_minus_baseline_successor_nll_nats"]
        ),
        "baseline_values_inspected_when_thresholds_frozen": thresholds[
            "baseline_values_inspected_when_thresholds_frozen"
        ],
    }
    body: dict[str, Any] = {
        "schema": PREPARED_SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "status": PREPARED_STATUS,
        **_NO_AUTHORITY,
        "scientific_scope": "scratch_active8_stage_a_capability_pilot",
        "recipe_policy_file_sha256": policy_file_sha256,
        "recipe_policy_sha256": policy["policy_sha256"],
        "prerequisites": inventory.prerequisites.as_payload(),
        "candidate_inventory_sha256": inventory.candidate_inventory_sha256,
        "candidate_count": len(inventory.candidates),
        "required_families": list(ACTIVE8_FAMILIES),
        "declared_nonempty_semantic_cells": list(cells),
        "recipe": {
            **dict(policy["optimization"]),
            "t1_selected_checkpoint_used_for_initialization": False,
        },
        "stream_contract": stream_contract,
        "ordered_stream_rows": rows,
        "ordered_address_stream_sha256": ordered_address_stream_sha256,
        "ordered_training_stream_sha256": ordered_training_stream_sha256,
        "requested_address_union": [_address_payload(address) for address in requested_addresses],
        "requested_address_union_sha256": requested_union_sha256,
        "complete_trace_closure_inventory": closure_rows,
        "complete_trace_closure_inventory_sha256": closure_inventory_sha256,
        "cache_contract": {
            **dict(policy["cache"]),
            "touched_trace_count": len(touched_trace_representatives),
            "requested_unique_nonterminal_address_count": len(requested_addresses),
            "closure_record_count": len(closure_rows),
            "closure_only_record_count": sum(bool(row["closure_only"]) for row in closure_rows),
        },
        "planned_family_exposure": family_reports,
        "planned_semantic_cell_exposure": cell_reports,
        "minimum_nonzero_gradient_updates_by_family": family_floors,
        "minimum_nonzero_gradient_updates_by_semantic_cell": cell_floors,
        "scheduled_lane_counts": dict(sorted(lane_counts.items())),
        "validation_contract": validation_contract,
        "required_physical_binding_purposes": list(REQUIRED_BINDING_PURPOSES),
        "unresolved_physical_bindings": list(REQUIRED_BINDING_PURPOSES),
    }
    return {**body, "prepared_recipe_sha256": _sha(body)}


def write_semantic_p50_prepared_recipe(path: Path, payload: Mapping[str, Any]) -> None:
    """Publish one canonical prepared recipe after self-hash validation."""

    body = dict(payload)
    supplied = body.pop("prepared_recipe_sha256", None)
    if (
        payload.get("schema") != PREPARED_SCHEMA
        or payload.get("schema_version") != SCHEMA_VERSION
        or payload.get("status") != PREPARED_STATUS
        or supplied != _sha(body)
        or any(payload.get(key) is not value for key, value in _NO_AUTHORITY.items())
    ):
        raise SemanticP50RecipeStreamError("prepared semantic P50 recipe is invalid")
    _write_canonical(Path(path), payload)


def authorize_semantic_p50_recipe(
    *,
    prepared: Mapping[str, Any],
    binding_paths: Sequence[Path],
) -> dict[str, Any]:
    """Refuse promotion until every bound domain artifact has a strict reopener.

    A generic self-hashed receipt is not evidence that a successor cache covers
    the exact address union, that a baseline was evaluated on the sealed
    validation role, or that a runtime and launch projection implement the
    frozen recipe. Promotion therefore remains unavailable until those five
    purpose-specific validators exist and this function invokes them on their
    physical artifacts.
    """

    raise SemanticP50RecipeStreamError(
        "semantic P50 authorization is not implemented: prepared recipes remain "
        "non-authorizing until domain-specific physical reopeners validate the "
        "successor cache, validation baseline, trainer runtime, execution "
        "environment, and launch projection"
    )


__all__ = [
    "BATCH_SIZE",
    "MAXIMUM_CELL_NLL_REGRESSION_NATS",
    "MAXIMUM_FAMILY_NLL_REGRESSION_NATS",
    "OPTIMIZER_STEPS",
    "POLICY_RELATIVE_PATH",
    "POLICY_SCHEMA",
    "PREPARED_SCHEMA",
    "REQUIRED_BINDING_PURPOSES",
    "SCHEDULED_EXAMPLES",
    "SEED",
    "SemanticP50Candidate",
    "SemanticP50CandidateInventory",
    "SemanticP50Prerequisites",
    "SemanticP50RecipeStreamError",
    "authorize_semantic_p50_recipe",
    "build_semantic_p50_candidate_inventory",
    "compile_semantic_p50_prepared_recipe",
    "load_semantic_p50_recipe_policy",
    "semantic_p50_time_hex",
    "validate_semantic_p50_prerequisite_relationships",
    "write_semantic_p50_prepared_recipe",
]
