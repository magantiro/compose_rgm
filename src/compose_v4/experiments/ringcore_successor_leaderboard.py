"""Fail-closed preparation and metric semantics for the RingCore-V1 leaderboard.

This module deliberately does *not* run the 32 checkpoints and does not select a
winner.  It provides the small, locally testable pieces that must be correct
before an expensive evaluator is allowed to start:

* bind the frozen protocol to the complete 32-snapshot inventory;
* verify each artifact by name, byte count, SHA-256, and recovery step;
* reconstruct the model from ``current_state_dict`` rather than the embedded
  validation-selected ``best_state_dict``;
* score a teacher molecular successor from the one production pushforward; and
* aggregate validation rows with the frozen production and semantic-cell laws.

The legacy ``scripts/ring_core_checkpoint_selection.py`` predates the production
pushforward and is not an evidence path.  In particular, best-effort alias
enumeration or silently skipped examples are forbidden here.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from compose_v4.experiments.checkpoint_evaluator import file_sha256
from compose_v4.experiments.production_successor_kernel import (
    SuccessorKernelResult,
)
from compose_v4.experiments.successor_kernel import validate_successor_batch
from compose_v4.model.factorized_tracelet_rate_model import MARK_RULE_NAMES

LEADERBOARD_SCHEMA = "compose.ringcore.successor_leaderboard"
LEADERBOARD_SCHEMA_VERSION = 1
PRIMARY_METRIC = "production_weighted_canonical_successor_nll"
SECONDARY_METRIC = "balanced_semantic_cell_canonical_successor_nll"
PRODUCTION_PANEL_ID = "production_law"
VALIDATION_PARTITION = "validation"
CURRENT_STATE_SOURCE = "current_state_dict"
SEMANTIC_CELL_ENCODER_VERSION = "joint_semantic_axes_v1"
SEMANTIC_CELL_AXES = (
    "capability_regime",
    "evidence_origin",
    "path_scale",
    "cardinality_topology_delta",
    "chemistry_charge_stratum",
    "split_unit",
)
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_SNAPSHOT = re.compile(r"checkpoint\.step([1-9][0-9]*)\.pt\Z")


class SuccessorLeaderboardError(ValueError):
    """The leaderboard contract or one proposed score is not scientifically valid."""


@dataclass(frozen=True)
class SnapshotEvaluationSpec:
    """One immutable snapshot that must be scored as its current training state."""

    step: int
    name: str
    sha256: str
    bytes: int
    state_source: str = CURRENT_STATE_SOURCE
    partition: str = VALIDATION_PARTITION

    def path_under(self, run_directory: str | Path) -> Path:
        return Path(run_directory) / self.name


@dataclass(frozen=True)
class CurrentSnapshotLoad:
    """Auditable description of the weights installed in a reconstructed model."""

    step: int
    checkpoint_sha256: str
    checkpoint_bytes: int
    state_source: str
    checkpoint_kind: str


@dataclass(frozen=True)
class TeacherSuccessor:
    """Identity and sampling-law information for one nonterminal validation draw."""

    panel_id: str
    draw_index: int
    partition: str
    family: str
    semantic_cell_id: str
    teacher_successor_key: str
    importance_weight: float
    teacher_mark_log_probability: float | None = None


@dataclass(frozen=True)
class SuccessorRowMetrics:
    """All checkpoint-dependent and support-dependent diagnostics for one draw."""

    panel_id: str
    draw_index: int
    partition: str
    family: str
    semantic_cell_id: str
    importance_weight: float
    canonical_successor_nll: float
    canonical_successor_probability: float
    canonical_successor_rank: int
    canonical_successor_mrr: float
    uniform_canonical_successor_nll: float
    learned_minus_uniform_log_likelihood: float
    training_target_nll: float | None
    selected_mark_minus_successor_nll_gap: float | None
    raw_legal_mark_count: int
    productive_mark_count: int
    canonical_successor_count: int
    teacher_successor_alias_multiplicity: int
    maximum_alias_multiplicity: int
    virtual_self_mass: float
    productive_mass: float
    teacher_family_probability: float
    family_choice_rank: int
    family_choice_top1: bool
    family_choice_top3: bool
    family_probabilities: tuple[float, ...]


def stable_json_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def load_json_object(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text())
    if not isinstance(payload, dict):
        raise SuccessorLeaderboardError(f"{path} must contain a JSON object")
    return payload


def _expected_steps(config: Mapping[str, Any]) -> tuple[int, ...]:
    run = config.get("run") or {}
    contract = run.get("snapshot_steps") or {}
    start = int(contract.get("start", 0))
    stop = int(contract.get("stop", 0))
    interval = int(contract.get("interval", 0))
    if start <= 0 or stop < start or interval <= 0:
        raise SuccessorLeaderboardError("invalid snapshot step contract")
    steps = tuple(range(start, stop + 1, interval))
    if steps[-1] != stop:
        raise SuccessorLeaderboardError("snapshot stop is not interval-aligned")
    if len(steps) != int(run.get("snapshot_count", -1)):
        raise SuccessorLeaderboardError("snapshot count disagrees with the step contract")
    return steps


def validate_leaderboard_config(config: Mapping[str, Any]) -> None:
    """Reject stale family-balanced, test-selecting, or best-state protocols."""

    if config.get("schema") != LEADERBOARD_SCHEMA:
        raise SuccessorLeaderboardError("unexpected leaderboard schema")
    if config.get("schema_version") != LEADERBOARD_SCHEMA_VERSION:
        raise SuccessorLeaderboardError("unexpected leaderboard schema version")
    if config.get("status") != "FROZEN_BEFORE_CANONICAL_SUCCESSOR_SCORING":
        raise SuccessorLeaderboardError("leaderboard was not frozen before scoring")

    run = config.get("run") or {}
    frozen_hash = str(run.get("frozen_inventory_sha256", ""))
    if _SHA256.fullmatch(frozen_hash) is None:
        raise SuccessorLeaderboardError("run lacks a full frozen inventory SHA-256")
    if run.get("snapshot_model_state") != CURRENT_STATE_SOURCE:
        raise SuccessorLeaderboardError("snapshots must be scored from current_state_dict")
    if run.get("embedded_best_state_forbidden") is not True:
        raise SuccessorLeaderboardError("embedded best-state scoring is not forbidden")
    _expected_steps(config)

    validation = config.get("validation_data") or {}
    if validation.get("partition") != VALIDATION_PARTITION:
        raise SuccessorLeaderboardError("checkpoint selection must use validation")
    if validation.get("test_partition_forbidden_for_selection") is not True:
        raise SuccessorLeaderboardError("test selection is not explicitly forbidden")

    metric_ids = (config.get("reported_metrics") or {}).get("metric_ids") or {}
    if metric_ids.get("primary") != PRIMARY_METRIC:
        raise SuccessorLeaderboardError("primary metric is not production-law successor NLL")
    if metric_ids.get("secondary") != SECONDARY_METRIC:
        raise SuccessorLeaderboardError(
            "secondary metric must balance joint semantic cells, not families"
        )
    selection = config.get("selection_rule") or {}
    if tuple(selection.get("metric_order") or ()) != (
        PRIMARY_METRIC,
        SECONDARY_METRIC,
    ):
        raise SuccessorLeaderboardError(
            "selection metric order must place production-law NLL before "
            "balanced semantic-cell NLL"
        )
    if selection.get("secondary_applies_only_after_primary_statistical_tie") is not True:
        raise SuccessorLeaderboardError(
            "semantic-cell NLL is not constrained to a secondary tie-break role"
        )
    cells = (config.get("panels") or {}).get("semantic_cells") or {}
    if cells.get("encoder_version") != SEMANTIC_CELL_ENCODER_VERSION:
        raise SuccessorLeaderboardError("semantic-cell encoder version is not frozen")
    if cells.get("panel_field") != "semantic_cell_id":
        raise SuccessorLeaderboardError("semantic-cell panel field is not frozen")
    if tuple(cells.get("required_axes") or ()) != SEMANTIC_CELL_AXES:
        raise SuccessorLeaderboardError("semantic-cell axes are absent or reordered")
    production_panel = (config.get("panels") or {}).get("production_law") or {}
    if (
        production_panel.get(
            "minimum_active_family_nonterminal_examples_before_selection"
        )
        != 64
    ):
        raise SuccessorLeaderboardError(
            "production-panel active-family minimum must remain frozen at 64"
        )

    bootstrap = (config.get("statistics") or {}).get("paired_bootstrap") or {}
    expected_bootstrap = {
        "schema_version": 1,
        "seed": 2026072903,
        "replicates": 10000,
        "chunk_replicates": 128,
        "resampling_unit": "fixed validation panel draw",
        "estimator": "self-normalized importance-weighted paired mean difference",
        "exact_draw_id_alignment_required": True,
        "checkpoint_order_or_ranking_performed": False,
    }
    bootstrap_mismatch = {
        key: {"expected": expected, "observed": bootstrap.get(key)}
        for key, expected in expected_bootstrap.items()
        if bootstrap.get(key) != expected
    }
    if bootstrap_mismatch:
        raise SuccessorLeaderboardError(
            f"paired-bootstrap contract mismatch: {bootstrap_mismatch}"
        )
    for interval_name in ("two_sided_interval", "one_sided_lower_bound"):
        interval = bootstrap.get(interval_name) or {}
        if interval.get("confidence") != 0.95 or interval.get("method") != "percentile":
            raise SuccessorLeaderboardError(
                f"paired-bootstrap {interval_name} is not frozen at 95% percentile"
            )

    execution = config.get("execution_contract") or {}
    required_execution = {
        "partition": VALIDATION_PARTITION,
        "snapshot_state_source": CURRENT_STATE_SOURCE,
        "all_32_snapshots_required": True,
        "same_panel_for_every_snapshot": True,
        "skipped_nonterminal_rows_allowed": False,
        "best_checkpoint_field_emitted_before_all_gates": False,
        "ranking_performed_by_preparation_or_scoring_code": False,
        "selection_performed_by_preparation_or_scoring_code": False,
        "future_ranking_requires_all_snapshot_provenance_and_hard_gates": True,
        "future_selection_requires_all_snapshot_provenance_and_hard_gates": True,
        "independent_alias_enumeration_for_evidence_forbidden": True,
    }
    mismatch = {
        key: {"expected": expected, "observed": execution.get(key)}
        for key, expected in required_execution.items()
        if execution.get(key) != expected
    }
    if mismatch:
        raise SuccessorLeaderboardError(
            f"leaderboard execution contract mismatch: {mismatch}"
        )


def inventory_self_hash(inventory: Mapping[str, Any]) -> str:
    """Recompute the immutable freezer hash (the hash field excludes itself)."""

    payload = dict(inventory)
    payload.pop("inventory_sha256", None)
    return stable_json_sha256(payload)


def encode_semantic_cell(axis_values: Mapping[str, str]) -> str:
    """Encode one complete joint semantic cell without result-dependent merging.

    The per-axis *labelers* still need to be implemented against the packed
    validation records.  Once labels are supplied, this identity function is
    fixed: all six axes are required, extras are rejected, and key ordering
    cannot alter the id.
    """

    observed = set(axis_values)
    expected = set(SEMANTIC_CELL_AXES)
    if observed != expected:
        raise SuccessorLeaderboardError(
            "semantic-cell axes mismatch: "
            f"missing={sorted(expected - observed)}, unexpected={sorted(observed - expected)}"
        )
    normalized = {}
    for axis in SEMANTIC_CELL_AXES:
        value = str(axis_values[axis]).strip()
        if not value:
            raise SuccessorLeaderboardError(
                f"semantic-cell axis {axis!r} has an empty label"
            )
        normalized[axis] = value
    digest = stable_json_sha256(
        {
            "encoder_version": SEMANTIC_CELL_ENCODER_VERSION,
            "axis_values": normalized,
        }
    )[:20]
    return f"{SEMANTIC_CELL_ENCODER_VERSION}:{digest}"


def validate_inventory(
    config: Mapping[str, Any],
    inventory: Mapping[str, Any],
    *,
    require_exact_self_hash: bool,
) -> None:
    """Validate the snapshot table and optionally require the exact immutable JSON.

    A compact projection may be useful for display, but it is not the immutable
    scientific artifact. Execution requires the exact freezer output and sets
    ``require_exact_self_hash=True``.
    """

    validate_leaderboard_config(config)
    if inventory.get("artifact_kind") != "frozen_training_run_inventory":
        raise SuccessorLeaderboardError("unexpected inventory artifact kind")
    if inventory.get("source_run_label") != (config.get("run") or {}).get(
        "run_label"
    ):
        raise SuccessorLeaderboardError("inventory run label disagrees with protocol")
    claimed = str(inventory.get("inventory_sha256", ""))
    expected = str((config.get("run") or {}).get("frozen_inventory_sha256", ""))
    if claimed != expected:
        raise SuccessorLeaderboardError(
            "inventory identity disagrees with frozen leaderboard protocol"
        )
    if require_exact_self_hash and inventory_self_hash(inventory) != claimed:
        raise SuccessorLeaderboardError(
            "inventory is only a compact projection; execution requires the exact "
            "immutable inventory whose self-hash matches the protocol"
        )

    verification = inventory.get("verification") or {}
    required_verification = {
        "passed": True,
        "checkpoint_selection_performed": False,
        "test_metrics_used_for_selection": False,
        "snapshot_names_complete": True,
        "snapshot_payload_steps_match_names": True,
    }
    mismatch = {
        key: verification.get(key)
        for key, expected_value in required_verification.items()
        if verification.get(key) != expected_value
    }
    if mismatch:
        raise SuccessorLeaderboardError(
            f"frozen inventory verification is incomplete: {mismatch}"
        )

    expected_steps = _expected_steps(config)
    contract = inventory.get("snapshot_contract") or {}
    if tuple(int(step) for step in contract.get("expected_steps") or ()) != expected_steps:
        raise SuccessorLeaderboardError("inventory expected steps disagree with protocol")
    if tuple(int(step) for step in contract.get("observed_steps") or ()) != expected_steps:
        raise SuccessorLeaderboardError("inventory does not contain every expected step")

    rows = inventory.get("snapshots") or ()
    if len(rows) != len(expected_steps):
        raise SuccessorLeaderboardError("snapshot inventory is incomplete")
    names: set[str] = set()
    digests: set[str] = set()
    for expected_step, row in zip(expected_steps, rows):
        if not isinstance(row, Mapping):
            raise SuccessorLeaderboardError("snapshot inventory rows must be mappings")
        name = str(row.get("name", ""))
        match = _SNAPSHOT.fullmatch(name)
        if match is None or int(match.group(1)) != expected_step:
            raise SuccessorLeaderboardError(
                f"snapshot name/step mismatch at expected step {expected_step}"
            )
        if int(row.get("completed_steps", -1)) != expected_step:
            raise SuccessorLeaderboardError(
                f"snapshot payload step mismatch at {expected_step}"
            )
        digest = str(row.get("sha256", ""))
        if _SHA256.fullmatch(digest) is None:
            raise SuccessorLeaderboardError(
                f"snapshot {name} lacks a full SHA-256"
            )
        if int(row.get("bytes", 0)) <= 0:
            raise SuccessorLeaderboardError(f"snapshot {name} has no bytes")
        if name in names or digest in digests:
            raise SuccessorLeaderboardError("snapshot names and hashes must be unique")
        names.add(name)
        digests.add(digest)


def prepare_snapshot_specs(
    config: Mapping[str, Any],
    inventory: Mapping[str, Any],
) -> tuple[SnapshotEvaluationSpec, ...]:
    """Return all 32 current-state validation specs, or fail without a partial list."""

    validate_inventory(
        config,
        inventory,
        require_exact_self_hash=True,
    )
    specs = tuple(
        SnapshotEvaluationSpec(
            step=int(row["completed_steps"]),
            name=str(row["name"]),
            sha256=str(row["sha256"]),
            bytes=int(row["bytes"]),
        )
        for row in inventory["snapshots"]
    )
    if any(spec.state_source != CURRENT_STATE_SOURCE for spec in specs):
        raise SuccessorLeaderboardError("a snapshot spec does not use current_state_dict")
    if any(spec.partition != VALIDATION_PARTITION for spec in specs):
        raise SuccessorLeaderboardError("a snapshot spec does not use validation")
    return specs


def verify_snapshot_artifact(spec: SnapshotEvaluationSpec, path: str | Path) -> Path:
    """Resolve and hash one snapshot before deserialization."""

    resolved = Path(path)
    if resolved.name != spec.name:
        raise SuccessorLeaderboardError(
            f"snapshot path {resolved.name!r} does not match inventory {spec.name!r}"
        )
    if not resolved.is_file():
        raise SuccessorLeaderboardError(f"snapshot is absent: {resolved}")
    observed_bytes = resolved.stat().st_size
    if observed_bytes != spec.bytes:
        raise SuccessorLeaderboardError(
            f"snapshot byte count {observed_bytes} != frozen {spec.bytes}"
        )
    observed_sha = file_sha256(resolved)
    if observed_sha != spec.sha256:
        raise SuccessorLeaderboardError(
            f"snapshot SHA-256 {observed_sha} != frozen {spec.sha256}"
        )
    return resolved


def validate_current_snapshot_payload(
    payload: object,
    spec: SnapshotEvaluationSpec,
    *,
    expected_run_identity_sha256: str | None = None,
) -> Mapping[str, Any]:
    """Return exactly ``current_state_dict`` after validating recovery identity."""

    if not isinstance(payload, Mapping):
        raise SuccessorLeaderboardError("snapshot payload must be a mapping")
    required = {
        "checkpoint_kind",
        "completed_steps",
        "current_state_dict",
        "best_state_dict",
        "history",
    }
    missing = sorted(required - payload.keys())
    if missing:
        raise SuccessorLeaderboardError(
            f"snapshot lacks exact-recovery fields: {missing}"
        )
    if payload["checkpoint_kind"] != "exact_training_recovery":
        raise SuccessorLeaderboardError("snapshot is not an exact recovery checkpoint")
    if int(payload["completed_steps"]) != spec.step:
        raise SuccessorLeaderboardError(
            "snapshot completed_steps disagrees with its frozen step"
        )
    history = payload["history"]
    if not isinstance(history, Sequence) or not history:
        raise SuccessorLeaderboardError("snapshot history is absent")
    last = history[-1]
    if not isinstance(last, Mapping) or int(float(last.get("step", -1))) != spec.step:
        raise SuccessorLeaderboardError("snapshot history does not end at its step")
    if expected_run_identity_sha256 is not None:
        if payload.get("provenance_sha256") != expected_run_identity_sha256:
            raise SuccessorLeaderboardError(
                "snapshot provenance does not match the frozen run identity"
            )
    current = payload["current_state_dict"]
    if not isinstance(current, Mapping) or not current:
        raise SuccessorLeaderboardError("current_state_dict is empty")
    return current


def _load_production_checkpoint(path: Path, *, expected_scope_hash: str | None):
    """Indirection kept patchable in tests; production reconstruction remains authoritative."""

    try:
        from evaluate_tracelet_rollouts import (  # noqa: PLC0415
            load_factorized_rollout_checkpoint,
        )
    except ImportError:
        from scripts.evaluate_tracelet_rollouts import (  # noqa: PLC0415
            load_factorized_rollout_checkpoint,
        )
    return load_factorized_rollout_checkpoint(
        path,
        expected_scope_hash=expected_scope_hash,
    )


def load_current_snapshot_model(
    path: str | Path,
    spec: SnapshotEvaluationSpec,
    *,
    config: Mapping[str, Any],
    inventory: Mapping[str, Any],
):
    """Load an inventory-bound snapshot and strictly install its CURRENT weights.

    The caller cannot supply a free-floating hash, scope, or provenance string:
    all three are derived from the exact self-hashed inventory after the complete
    32-snapshot contract passes.
    """

    frozen_specs = prepare_snapshot_specs(config, inventory)
    if spec not in frozen_specs:
        raise SuccessorLeaderboardError(
            "snapshot spec is not one of the 32 exact-inventory-bound validation specs"
        )
    metadata = inventory.get("checkpoint_metadata") or {}
    expected_scope_hash = metadata.get("corpus_scope_hash")
    if not isinstance(expected_scope_hash, str) or not expected_scope_hash:
        raise SuccessorLeaderboardError(
            "exact inventory lacks the frozen broad-organic corpus scope hash"
        )
    manifest = inventory.get("manifest_summary") or {}
    expected_run_identity_sha256 = manifest.get("run_identity_sha256")
    if (
        not isinstance(expected_run_identity_sha256, str)
        or _SHA256.fullmatch(expected_run_identity_sha256) is None
    ):
        raise SuccessorLeaderboardError(
            "exact inventory lacks the frozen run identity SHA-256"
        )

    resolved = verify_snapshot_artifact(spec, path)
    import torch  # noqa: PLC0415

    payload = torch.load(resolved, map_location="cpu", weights_only=False)
    current = validate_current_snapshot_payload(
        payload,
        spec,
        expected_run_identity_sha256=expected_run_identity_sha256,
    )
    model, _metadata = _load_production_checkpoint(
        resolved,
        expected_scope_hash=expected_scope_hash,
    )
    # The production rollout loader intentionally reconstructs a recovery payload
    # from best_state_dict.  Leaderboard semantics require the opposite: strict
    # replacement with the step-N current state.
    model.load_state_dict(current, strict=True)
    model.eval()
    return model, CurrentSnapshotLoad(
        step=spec.step,
        checkpoint_sha256=spec.sha256,
        checkpoint_bytes=spec.bytes,
        state_source=CURRENT_STATE_SOURCE,
        checkpoint_kind=str(payload["checkpoint_kind"]),
    )


def _finite_nonnegative(value: float, *, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise SuccessorLeaderboardError(f"{name} must be finite and nonnegative")
    return result


def score_teacher_successor(
    result: SuccessorKernelResult,
    teacher: TeacherSuccessor,
) -> SuccessorRowMetrics:
    """Score one nonterminal teacher under the productive molecular jump law."""

    if teacher.partition != VALIDATION_PARTITION:
        raise SuccessorLeaderboardError("leaderboard rows must come from validation")
    if teacher.draw_index < 0:
        raise SuccessorLeaderboardError("panel draw_index must be nonnegative")
    if teacher.family not in MARK_RULE_NAMES:
        raise SuccessorLeaderboardError(f"unknown teacher family {teacher.family!r}")
    if not teacher.semantic_cell_id.strip():
        raise SuccessorLeaderboardError("semantic_cell_id is empty")
    weight = _finite_nonnegative(
        teacher.importance_weight,
        name="importance_weight",
    )
    if weight == 0.0:
        raise SuccessorLeaderboardError("nonterminal importance_weight must be positive")

    validate_successor_batch(result.batch)
    if result.batch.is_terminal:
        raise SuccessorLeaderboardError("nonterminal teacher has a terminal successor row")
    matches = [
        successor
        for successor in result.batch.successors
        if successor.key == teacher.teacher_successor_key
    ]
    if len(matches) != 1:
        raise SuccessorLeaderboardError(
            "teacher successor must be present exactly once after canonical grouping"
        )
    target = matches[0]
    probability = float(target.probability)
    if not math.isfinite(probability) or probability <= 0.0:
        raise SuccessorLeaderboardError("teacher successor has no finite positive mass")
    nll = -math.log(probability)

    probabilities = [float(row.probability) for row in result.batch.successors]
    rank = 1 + sum(value > probability for value in probabilities)
    support_size = result.batch.support_size
    uniform_nll = math.log(support_size)
    learned_advantage = math.log(probability) + uniform_nll

    family_probabilities = tuple(
        math.exp(value) for value in result.marked_law.family_log_probabilities
    )
    if len(family_probabilities) != len(MARK_RULE_NAMES):
        raise SuccessorLeaderboardError("family probability vector has the wrong width")
    if not all(math.isfinite(value) and value >= 0.0 for value in family_probabilities):
        raise SuccessorLeaderboardError("family probabilities are not finite")
    if abs(sum(family_probabilities) - 1.0) > 2e-5:
        raise SuccessorLeaderboardError("family probabilities are not normalized")
    family_index = MARK_RULE_NAMES.index(teacher.family)
    family_rank = 1 + sum(
        value > family_probabilities[family_index] for value in family_probabilities
    )

    training_target_nll = None
    nll_gap = None
    if teacher.teacher_mark_log_probability is not None:
        mark_logp = float(teacher.teacher_mark_log_probability)
        if not math.isfinite(mark_logp) or mark_logp > 1e-7:
            raise SuccessorLeaderboardError(
                "teacher mark log probability must be finite and at most zero"
            )
        training_target_nll = -mark_logp
        nll_gap = training_target_nll - nll
        if nll_gap < -1e-6:
            raise SuccessorLeaderboardError(
                "canonical successor has less mass than its selected teacher mark"
            )

    diagnostics = result.diagnostics
    maximum_alias = max(diagnostics.alias_multiplicities, default=0)
    return SuccessorRowMetrics(
        panel_id=teacher.panel_id,
        draw_index=int(teacher.draw_index),
        partition=teacher.partition,
        family=teacher.family,
        semantic_cell_id=teacher.semantic_cell_id,
        importance_weight=weight,
        canonical_successor_nll=nll,
        canonical_successor_probability=probability,
        canonical_successor_rank=rank,
        canonical_successor_mrr=1.0 / rank,
        uniform_canonical_successor_nll=uniform_nll,
        learned_minus_uniform_log_likelihood=learned_advantage,
        training_target_nll=training_target_nll,
        selected_mark_minus_successor_nll_gap=nll_gap,
        raw_legal_mark_count=int(diagnostics.raw_mark_count),
        productive_mark_count=int(diagnostics.productive_mark_count),
        canonical_successor_count=support_size,
        teacher_successor_alias_multiplicity=int(target.alias_count),
        maximum_alias_multiplicity=int(maximum_alias),
        virtual_self_mass=_finite_nonnegative(
            diagnostics.virtual_self_mass,
            name="virtual_self_mass",
        ),
        productive_mass=_finite_nonnegative(
            diagnostics.raw_productive_mass,
            name="productive_mass",
        ),
        teacher_family_probability=family_probabilities[family_index],
        family_choice_rank=family_rank,
        family_choice_top1=family_rank == 1,
        family_choice_top3=family_rank <= 3,
        family_probabilities=family_probabilities,
    )


def _weighted_mean(rows: Sequence[SuccessorRowMetrics], field: str) -> float:
    denominator = sum(row.importance_weight for row in rows)
    if denominator <= 0.0:
        raise SuccessorLeaderboardError(f"no positive weight for {field}")
    numerator = sum(
        row.importance_weight * float(getattr(row, field)) for row in rows
    )
    result = numerator / denominator
    if not math.isfinite(result):
        raise SuccessorLeaderboardError(f"non-finite aggregate {field}")
    return result


def aggregate_validation_rows(
    rows: Sequence[SuccessorRowMetrics],
    *,
    required_semantic_cells: Sequence[str],
    expected_nonterminal_draw_indices: Sequence[int],
) -> dict[str, Any]:
    """Aggregate one checkpoint's fixed nonterminal validation panel.

    No row can be skipped: duplicate draw identities, missing frozen cells, test
    rows, and non-finite values all abort the whole checkpoint score.
    """

    if not rows:
        raise SuccessorLeaderboardError("no nonterminal validation rows to aggregate")
    if any(row.partition != VALIDATION_PARTITION for row in rows):
        raise SuccessorLeaderboardError("a non-validation row entered selection metrics")
    if any(row.panel_id != PRODUCTION_PANEL_ID for row in rows):
        raise SuccessorLeaderboardError(
            "a row came from a different panel than the frozen production_law panel"
        )
    identities = [(row.panel_id, row.draw_index) for row in rows]
    if len(set(identities)) != len(identities):
        raise SuccessorLeaderboardError("panel draw identities are not unique")
    expected_indices = tuple(int(index) for index in expected_nonterminal_draw_indices)
    if not expected_indices:
        raise SuccessorLeaderboardError(
            "frozen nonterminal panel-index census is empty"
        )
    if any(index < 0 for index in expected_indices):
        raise SuccessorLeaderboardError(
            "frozen nonterminal panel indices must be nonnegative"
        )
    if len(set(expected_indices)) != len(expected_indices):
        raise SuccessorLeaderboardError(
            "frozen nonterminal panel indices are duplicated"
        )
    observed_indices = {row.draw_index for row in rows}
    required_indices = set(expected_indices)
    if observed_indices != required_indices:
        raise SuccessorLeaderboardError(
            "nonterminal panel census mismatch: "
            f"missing={sorted(required_indices - observed_indices)}, "
            f"unexpected={sorted(observed_indices - required_indices)}"
        )
    required = tuple(str(cell) for cell in required_semantic_cells)
    if not required or any(not cell for cell in required):
        raise SuccessorLeaderboardError("required semantic-cell census is empty")
    if len(set(required)) != len(required):
        raise SuccessorLeaderboardError("required semantic cells are duplicated")

    by_cell: dict[str, list[SuccessorRowMetrics]] = defaultdict(list)
    by_family: dict[str, list[SuccessorRowMetrics]] = defaultdict(list)
    for row in rows:
        by_cell[row.semantic_cell_id].append(row)
        by_family[row.family].append(row)
    observed = set(by_cell)
    expected = set(required)
    if observed != expected:
        raise SuccessorLeaderboardError(
            "semantic-cell census mismatch: "
            f"missing={sorted(expected - observed)}, unexpected={sorted(observed - expected)}"
        )

    per_cell = {
        cell: {
            "examples": len(by_cell[cell]),
            "canonical_successor_nll": _weighted_mean(
                by_cell[cell],
                "canonical_successor_nll",
            ),
        }
        for cell in required
    }
    secondary = sum(
        metrics["canonical_successor_nll"] for metrics in per_cell.values()
    ) / len(per_cell)
    primary = _weighted_mean(rows, "canonical_successor_nll")

    per_family = {
        family: {
            "examples": len(family_rows),
            "canonical_successor_nll": _weighted_mean(
                family_rows,
                "canonical_successor_nll",
            ),
            "canonical_successor_probability": _weighted_mean(
                family_rows,
                "canonical_successor_probability",
            ),
            "canonical_successor_mrr": _weighted_mean(
                family_rows,
                "canonical_successor_mrr",
            ),
            "canonical_successor_rank": _weighted_mean(
                family_rows,
                "canonical_successor_rank",
            ),
            "uniform_canonical_successor_nll": _weighted_mean(
                family_rows,
                "uniform_canonical_successor_nll",
            ),
            "learned_minus_uniform_log_likelihood": _weighted_mean(
                family_rows,
                "learned_minus_uniform_log_likelihood",
            ),
            "raw_legal_mark_count": _weighted_mean(
                family_rows,
                "raw_legal_mark_count",
            ),
            "productive_mark_count": _weighted_mean(
                family_rows,
                "productive_mark_count",
            ),
            "canonical_successor_count": _weighted_mean(
                family_rows,
                "canonical_successor_count",
            ),
            "teacher_successor_alias_multiplicity": _weighted_mean(
                family_rows,
                "teacher_successor_alias_multiplicity",
            ),
            "maximum_alias_multiplicity": _weighted_mean(
                family_rows,
                "maximum_alias_multiplicity",
            ),
            "virtual_self_mass": _weighted_mean(
                family_rows,
                "virtual_self_mass",
            ),
            "productive_mass": _weighted_mean(
                family_rows,
                "productive_mass",
            ),
            "teacher_family_probability": _weighted_mean(
                family_rows,
                "teacher_family_probability",
            ),
            "family_choice_top1": _weighted_mean(
                family_rows,
                "family_choice_top1",
            ),
            "family_choice_top3": _weighted_mean(
                family_rows,
                "family_choice_top3",
            ),
        }
        for family, family_rows in sorted(by_family.items())
    }
    for family, family_rows in sorted(by_family.items()):
        mark_presence = [
            row.training_target_nll is not None for row in family_rows
        ]
        if any(mark_presence) and not all(mark_presence):
            raise SuccessorLeaderboardError(
                f"family {family} mixes present and absent training-target scores"
            )
        if all(mark_presence):
            per_family[family]["training_target_nll"] = _weighted_mean(
                family_rows,
                "training_target_nll",
            )

    total_weight = sum(row.importance_weight for row in rows)
    teacher_family_mass = {
        family: sum(
            row.importance_weight for row in rows if row.family == family
        )
        / total_weight
        for family in MARK_RULE_NAMES
    }
    predicted_family_mass = {
        family: sum(
            row.importance_weight * row.family_probabilities[index]
            for row in rows
        )
        / total_weight
        for index, family in enumerate(MARK_RULE_NAMES)
    }
    family_tv = 0.5 * sum(
        abs(teacher_family_mass[family] - predicted_family_mass[family])
        for family in MARK_RULE_NAMES
    )
    family_kl = sum(
        teacher_family_mass[family]
        * math.log(
            teacher_family_mass[family]
            / max(predicted_family_mass[family], 1e-300)
        )
        for family in MARK_RULE_NAMES
        if teacher_family_mass[family] > 0.0
    )

    alias_histogram = Counter(
        row.teacher_successor_alias_multiplicity for row in rows
    )
    result: dict[str, Any] = {
        PRIMARY_METRIC: primary,
        SECONDARY_METRIC: secondary,
        "family_balanced_canonical_successor_nll_diagnostic": sum(
            metrics["canonical_successor_nll"]
            for metrics in per_family.values()
        )
        / len(per_family),
        "per_semantic_cell": per_cell,
        "per_family": per_family,
        "nonterminal_examples": len(rows),
        "fraction_states_with_any_aliased_successor": sum(
            row.maximum_alias_multiplicity > 1 for row in rows
        )
        / len(rows),
        "fraction_teacher_successors_aliased": sum(
            row.teacher_successor_alias_multiplicity > 1 for row in rows
        )
        / len(rows),
        "teacher_alias_multiplicity_histogram": {
            str(key): value for key, value in sorted(alias_histogram.items())
        },
        "family_mass_calibration_tv": family_tv,
        "family_mass_calibration_kl": family_kl,
        "mean_raw_legal_mark_count": _weighted_mean(
            rows,
            "raw_legal_mark_count",
        ),
        "mean_productive_mark_count": _weighted_mean(
            rows,
            "productive_mark_count",
        ),
        "mean_canonical_successor_count": _weighted_mean(
            rows,
            "canonical_successor_count",
        ),
        "mean_virtual_self_mass": _weighted_mean(rows, "virtual_self_mass"),
        "mean_productive_mass": _weighted_mean(rows, "productive_mass"),
    }
    rows_with_mark = [
        row for row in rows if row.training_target_nll is not None
    ]
    if rows_with_mark:
        result["training_target_nll"] = _weighted_mean(
            rows_with_mark,
            "training_target_nll",
        )
        result["selected_mark_minus_successor_nll_gap"] = _weighted_mean(
            rows_with_mark,
            "selected_mark_minus_successor_nll_gap",
        )
    return result


def readiness_summary(
    config: Mapping[str, Any],
    inventory: Mapping[str, Any],
) -> dict[str, Any]:
    """Machine-readable local readiness without pretending remote inputs exist."""

    validate_inventory(config, inventory, require_exact_self_hash=True)
    specs = prepare_snapshot_specs(config, inventory)
    exact_hash_matches = inventory_self_hash(inventory) == inventory.get(
        "inventory_sha256"
    )
    semantic_status = (
        ((config.get("panels") or {}).get("semantic_cells") or {}).get(
            "execution_status"
        )
    )
    blockers = []
    if not exact_hash_matches:
        blockers.append("exact immutable inventory self-hash does not match")
    if semantic_status != "FROZEN_READY":
        blockers.append(
            "semantic-axis labelers and the validation cell census are not frozen"
        )
    blockers.extend(
        (
            "fixed production-law and family-forensics panel artifacts are not built",
            "all-32 production-kernel scoring runner and hard-gate integration are not implemented",
            "remote snapshot files are not locally available for SHA/payload/capability verification",
        )
    )
    return {
        "schema": "compose.ringcore.successor_leaderboard.readiness",
        "schema_version": 1,
        "ready_to_execute": not blockers,
        "snapshot_specs_prepared": len(specs),
        "snapshot_steps": [spec.step for spec in specs],
        "snapshot_state_source": CURRENT_STATE_SOURCE,
        "selection_partition": VALIDATION_PARTITION,
        "primary_metric": PRIMARY_METRIC,
        "secondary_metric": SECONDARY_METRIC,
        "selection_safety": {
            "all_snapshot_specs_require_exact_inventory_self_hash": True,
            "aggregation_accepts_only_production_law_panel": True,
            "aggregation_requires_exact_nonterminal_draw_census": True,
            "automatic_ranking_code_present": False,
            "automatic_selection_code_present": False,
            "balanced_family_nll_is_diagnostic_only": True,
            "current_state_dict_required": True,
            "future_ranking_requires_all_snapshot_provenance_and_hard_gates": True,
            "future_selection_requires_all_snapshot_provenance_and_hard_gates": True,
            "fixed_panel_schema_and_validator_present": True,
            "panel_metric_vectors_are_validation_and_current_snapshot_bound": True,
            "paired_bootstrap_is_exact_panel_bound": True,
            "expansion_primitives_perform_no_ranking_or_selection": True,
            "semantic_cell_nll_role": "secondary_after_primary_statistical_tie",
            "test_partition_input_accepted": False,
            "validation_partition_required": True,
        },
        "local_exact_inventory_self_hash_matches": exact_hash_matches,
        "blockers": blockers,
        "explicit_non_actions": [
            "no checkpoint evaluated",
            "no checkpoint ranked",
            "no checkpoint selected",
            "no test metrics used",
            "no Modal job launched",
        ],
    }


__all__ = [
    "CURRENT_STATE_SOURCE",
    "CurrentSnapshotLoad",
    "LEADERBOARD_SCHEMA",
    "PRIMARY_METRIC",
    "PRODUCTION_PANEL_ID",
    "SECONDARY_METRIC",
    "SEMANTIC_CELL_AXES",
    "SEMANTIC_CELL_ENCODER_VERSION",
    "SnapshotEvaluationSpec",
    "SuccessorLeaderboardError",
    "SuccessorRowMetrics",
    "TeacherSuccessor",
    "VALIDATION_PARTITION",
    "aggregate_validation_rows",
    "encode_semantic_cell",
    "inventory_self_hash",
    "load_current_snapshot_model",
    "load_json_object",
    "prepare_snapshot_specs",
    "readiness_summary",
    "score_teacher_successor",
    "stable_json_sha256",
    "validate_current_snapshot_payload",
    "validate_inventory",
    "validate_leaderboard_config",
    "verify_snapshot_artifact",
]
