"""Immutable validation panels and non-selecting leaderboard statistics.

The module contains no checkpoint loader, model forward pass, ranking, or
selection function.  It freezes checkpoint-independent evaluation inputs and
provides paired statistical primitives over already aligned per-draw metrics.

Two panel kinds are deliberately distinct:

``production_law``
    The exact prefix of the frozen validation sampling stream. It contains
    terminal rows for hazard reporting and nonterminal rows for the primary and
    semantic-cell secondary successor metrics.

``family_forensics``
    A sparse deterministic scan of the validation stream retaining nonterminal
    rows until each active family reaches its frozen target. It is for
    capability gates and diagnostics, never the production-law selector.
"""

from __future__ import annotations

import copy
import math
import os
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np

from compose_v4.experiments.ringcore_successor_leaderboard import (
    CURRENT_STATE_SOURCE,
    PRODUCTION_PANEL_ID,
    SEMANTIC_CELL_AXES,
    SuccessorLeaderboardError,
    encode_semantic_cell,
    prepare_snapshot_specs,
    stable_json_sha256,
    validate_inventory,
    validate_leaderboard_config,
)
from compose_v4.model.factorized_tracelet_rate_model import MARK_RULE_NAMES

PANEL_SCHEMA = "compose.ringcore.fixed_validation_panel"
PANEL_SCHEMA_VERSION = 1
PANEL_STATUS = "FROZEN_FIXED_VALIDATION_PANEL"
FAMILY_FORENSICS_PANEL_ID = "family_forensics"
PANEL_KINDS = (PRODUCTION_PANEL_ID, FAMILY_FORENSICS_PANEL_ID)
STATE_DIGEST_SCHEMA = "persistent_slot_state_v1"
BOOTSTRAP_SCHEMA_VERSION = 1
METRIC_VECTOR_SCHEMA_VERSION = 1
VALIDATION_PARTITION = "validation"
METRIC_VECTOR_SOURCE_KINDS = (
    "snapshot_current_state",
    "uniform_canonical_successor",
)
PANEL_METRIC_IDS = (
    "canonical_successor_nll",
    "canonical_successor_log_likelihood",
)

_ROW_KEYS = {
    "row_index",
    "stream_draw_index",
    "partition",
    "layer",
    "record_key",
    "path_length",
    "progress_index",
    "time",
    "importance_weight",
    "terminal",
    "teacher_family",
    "teacher_rule_name",
    "teacher_successor_key",
    "teacher_action_sha256",
    "semantic_axis_values",
    "semantic_cell_id",
    "source_state_key",
    "source_state_sha256",
    "exact_state_ref",
    "row_sha256",
}
_PANEL_KEYS = {
    "schema",
    "schema_version",
    "status",
    "panel_kind",
    "partition",
    "leaderboard_protocol_sha256",
    "frozen_inventory_sha256",
    "source_run_label",
    "seed",
    "provenance",
    "draw_contract",
    "rows",
    "artifact_sha256",
}
_CENSUS_KEYS = {
    "row_count",
    "nonterminal_row_count",
    "terminal_row_count",
    "nonterminal_draw_indices",
    "semantic_cell_census",
    "family_nonterminal_census",
}
_STATE_REF_KEYS = {
    "shard_name",
    "shard_sha256",
    "record_index",
    "progress_index",
    "state_digest_schema",
}


class ValidationPanelError(SuccessorLeaderboardError):
    """A fixed panel is mutable, incomplete, off-protocol, or not validation."""


@dataclass(frozen=True)
class PairedBootstrapResult:
    """A paired interval over one aligned validation panel; never a ranking."""

    schema_version: int
    interval_kind: str
    panel_kind: str
    panel_artifact_sha256: str
    family_filter: str | None
    metric_id: str
    left_source_kind: str
    right_source_kind: str
    left_vector_sha256: str
    right_vector_sha256: str
    estimator: str
    difference_direction: str
    observation_count: int
    draw_id_census_sha256: str
    seed: int
    replicates: int
    confidence: float
    observed_difference: float
    lower_bound: float
    upper_bound: float | None
    contains_zero: bool | None
    ranking_performed: bool = False
    selection_performed: bool = False


@dataclass(frozen=True)
class ProductionPanelExpansionDecision:
    """Whether the production stream must extend to its preregistered prefix."""

    current_draws: int
    target_draws: int
    expand: bool
    reasons: tuple[str, ...]
    ranking_performed: bool = False
    selection_performed: bool = False


@dataclass(frozen=True)
class FamilyForensicsExpansionDecision:
    """Per-family target counts; does not pass/fail or choose a checkpoint."""

    current_counts: tuple[tuple[str, int], ...]
    target_counts: tuple[tuple[str, int], ...]
    statistically_ambiguous_families: tuple[str, ...]
    expand: bool
    ranking_performed: bool = False
    selection_performed: bool = False


@dataclass(frozen=True)
class PanelMetricVector:
    """One per-draw metric vector bound to validation and an allowed law."""

    schema_version: int
    partition: str
    panel_kind: str
    panel_artifact_sha256: str
    source_kind: str
    snapshot_step: int | None
    snapshot_name: str | None
    snapshot_sha256: str | None
    state_source: str
    metric_id: str
    family_filter: str | None
    values: tuple[tuple[int, float], ...]
    vector_sha256: str


def _is_sha256(value: object) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    return all(character in "0123456789abcdef" for character in value)


def _require_nonnegative_int(value: object, *, field: str) -> int:
    if type(value) is not int or value < 0:
        raise ValidationPanelError(f"{field} must be a nonnegative integer")
    return value


def _require_finite_number(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValidationPanelError(f"{field} must be a finite JSON number")
    result = float(value)
    if not math.isfinite(result):
        raise ValidationPanelError(f"{field} must be a finite JSON number")
    return result


def _row_self_hash(row: Mapping[str, Any]) -> str:
    payload = dict(row)
    payload.pop("row_sha256", None)
    return stable_json_sha256(payload)


def panel_artifact_self_hash(panel: Mapping[str, Any]) -> str:
    payload = dict(panel)
    payload.pop("artifact_sha256", None)
    return stable_json_sha256(payload)


def _protocol_provenance(
    config: Mapping[str, Any],
    inventory: Mapping[str, Any],
) -> dict[str, Any]:
    validation = config["validation_data"]
    return {
        "unified_manifest_checksum": validation["unified_manifest_checksum"],
        "representability_overlay_checksum": validation[
            "representability_overlay_checksum"
        ],
        "layer_weights_hash": validation["layer_weights_hash"],
        "operator_registry_hash": validation["operator_registry_hash"],
        "checkpoint_metadata_sha256": inventory["checkpoint_metadata_sha256"],
        "state_schema_sha256": inventory["state_schema_sha256"],
        "state_digest_schema": STATE_DIGEST_SCHEMA,
    }


def _derived_census(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    nonterminal = [row for row in rows if not bool(row["terminal"])]
    return {
        "row_count": len(rows),
        "nonterminal_row_count": len(nonterminal),
        "terminal_row_count": len(rows) - len(nonterminal),
        "nonterminal_draw_indices": [
            int(row["stream_draw_index"]) for row in nonterminal
        ],
        "semantic_cell_census": dict(
            sorted(Counter(str(row["semantic_cell_id"]) for row in nonterminal).items())
        ),
        "family_nonterminal_census": dict(
            sorted(Counter(str(row["teacher_family"]) for row in nonterminal).items())
        ),
    }


def _seal_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    sealed = []
    for raw in rows:
        row = copy.deepcopy(dict(raw))
        if "row_sha256" in row:
            raise ValidationPanelError(
                "unsealed input rows must not provide their own row_sha256"
            )
        row["row_sha256"] = _row_self_hash(row)
        sealed.append(row)
    return sealed


def seal_validation_panel(
    *,
    panel_kind: str,
    rows: Sequence[Mapping[str, Any]],
    draw_contract: Mapping[str, Any],
    config: Mapping[str, Any],
    inventory: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind rows to the frozen protocol/inventory and return a self-hashed artifact."""

    validate_leaderboard_config(config)
    validate_inventory(config, inventory, require_exact_self_hash=True)
    if panel_kind not in PANEL_KINDS:
        raise ValidationPanelError(f"unknown panel kind {panel_kind!r}")
    panel_spec = config["panels"][panel_kind]
    sealed_rows = _seal_rows(rows)
    contract = copy.deepcopy(dict(draw_contract))
    overlap = set(contract) & _CENSUS_KEYS
    if overlap:
        raise ValidationPanelError(
            f"draw contract attempts to override derived census fields: {sorted(overlap)}"
        )
    contract.update(_derived_census(sealed_rows))
    artifact: dict[str, Any] = {
        "schema": PANEL_SCHEMA,
        "schema_version": PANEL_SCHEMA_VERSION,
        "status": PANEL_STATUS,
        "panel_kind": panel_kind,
        "partition": VALIDATION_PARTITION,
        "leaderboard_protocol_sha256": stable_json_sha256(config),
        "frozen_inventory_sha256": inventory["inventory_sha256"],
        "source_run_label": inventory["source_run_label"],
        "seed": int(panel_spec["seed"]),
        "provenance": _protocol_provenance(config, inventory),
        "draw_contract": contract,
        "rows": sealed_rows,
    }
    artifact["artifact_sha256"] = panel_artifact_self_hash(artifact)
    validate_validation_panel_artifact(artifact, config=config, inventory=inventory)
    return artifact


def _validate_state_ref(value: object, *, progress_index: int) -> None:
    if not isinstance(value, Mapping) or set(value) != _STATE_REF_KEYS:
        raise ValidationPanelError("exact_state_ref has the wrong schema")
    shard_name = value["shard_name"]
    if not isinstance(shard_name, str) or not shard_name:
        raise ValidationPanelError("exact_state_ref shard_name is empty")
    pure = PurePosixPath(shard_name)
    if pure.is_absolute() or ".." in pure.parts:
        raise ValidationPanelError("exact_state_ref shard_name is unsafe")
    if not _is_sha256(value["shard_sha256"]):
        raise ValidationPanelError("exact_state_ref lacks a shard SHA-256")
    _require_nonnegative_int(
        value["record_index"],
        field="exact_state_ref record_index",
    )
    state_progress = _require_nonnegative_int(
        value["progress_index"],
        field="exact_state_ref progress_index",
    )
    if state_progress != progress_index:
        raise ValidationPanelError(
            "exact_state_ref progress disagrees with the panel row"
        )
    if value["state_digest_schema"] != STATE_DIGEST_SCHEMA:
        raise ValidationPanelError("exact_state_ref state digest schema drifted")


def _validate_row(
    row: object,
    *,
    expected_row_index: int,
    active_families: set[str],
    layers: set[str],
) -> None:
    if not isinstance(row, Mapping) or set(row) != _ROW_KEYS:
        missing = sorted(_ROW_KEYS - set(row if isinstance(row, Mapping) else ()))
        extra = sorted(set(row if isinstance(row, Mapping) else ()) - _ROW_KEYS)
        raise ValidationPanelError(
            f"panel row schema mismatch: missing={missing}, extra={extra}"
        )
    row_index = _require_nonnegative_int(row["row_index"], field="panel row index")
    if row_index != expected_row_index:
        raise ValidationPanelError("panel row indices are not contiguous")
    _require_nonnegative_int(
        row["stream_draw_index"],
        field="stream draw index",
    )
    if row["partition"] != VALIDATION_PARTITION:
        raise ValidationPanelError("panel rows must be validation only")
    if row["layer"] not in layers:
        raise ValidationPanelError(f"unknown validation layer {row['layer']!r}")
    for key in ("record_key", "source_state_key"):
        if not isinstance(row[key], str) or not row[key]:
            raise ValidationPanelError(f"panel row {key} is empty")
    if not _is_sha256(row["source_state_sha256"]):
        raise ValidationPanelError("source_state_sha256 is invalid")
    path_length = _require_nonnegative_int(
        row["path_length"],
        field="path length",
    )
    progress = _require_nonnegative_int(
        row["progress_index"],
        field="progress index",
    )
    if progress > path_length:
        raise ValidationPanelError("path/progress indices are invalid")
    time = _require_finite_number(row["time"], field="sampled model time")
    weight = _require_finite_number(
        row["importance_weight"],
        field="importance weight",
    )
    if not 0.0 <= time < 1.0:
        raise ValidationPanelError("sampled model time must lie in [0,1)")
    if weight <= 0.0:
        raise ValidationPanelError("importance weight must be finite and positive")
    _validate_state_ref(row["exact_state_ref"], progress_index=progress)

    if type(row["terminal"]) is not bool:
        raise ValidationPanelError("terminal must be a JSON boolean")
    terminal = row["terminal"]
    teacher_fields = (
        "teacher_family",
        "teacher_rule_name",
        "teacher_successor_key",
        "teacher_action_sha256",
        "semantic_axis_values",
        "semantic_cell_id",
    )
    if terminal:
        if progress != path_length:
            raise ValidationPanelError("terminal row is not at path end")
        if any(row[field] is not None for field in teacher_fields):
            raise ValidationPanelError("terminal row carries successor supervision")
    else:
        if progress >= path_length:
            raise ValidationPanelError("nonterminal row is at or beyond path end")
        if row["teacher_family"] not in active_families:
            raise ValidationPanelError(
                f"nonterminal row has inactive family {row['teacher_family']!r}"
            )
        if row["teacher_family"] not in MARK_RULE_NAMES:
            raise ValidationPanelError("teacher family is not a model family")
        for field in ("teacher_rule_name", "teacher_successor_key"):
            if not isinstance(row[field], str) or not row[field]:
                raise ValidationPanelError(f"nonterminal row {field} is empty")
        if not _is_sha256(row["teacher_action_sha256"]):
            raise ValidationPanelError("teacher_action_sha256 is invalid")
        axes = row["semantic_axis_values"]
        if not isinstance(axes, Mapping) or set(axes) != set(SEMANTIC_CELL_AXES):
            raise ValidationPanelError(
                "semantic axis values do not cover the frozen axes"
            )
        if row["semantic_cell_id"] != encode_semantic_cell(axes):
            raise ValidationPanelError("semantic cell id does not match its axes")

    if not _is_sha256(row["row_sha256"]):
        raise ValidationPanelError("row_sha256 is invalid")
    if _row_self_hash(row) != row["row_sha256"]:
        raise ValidationPanelError("panel row self-hash does not match")


def validate_validation_panel_artifact(
    panel: Mapping[str, Any],
    *,
    config: Mapping[str, Any],
    inventory: Mapping[str, Any],
) -> None:
    """Validate exact provenance, census, row identities, and panel-kind law."""

    validate_leaderboard_config(config)
    validate_inventory(config, inventory, require_exact_self_hash=True)
    if set(panel) != _PANEL_KEYS:
        raise ValidationPanelError(
            "validation-panel top-level schema has missing or extra fields"
        )
    if panel.get("schema") != PANEL_SCHEMA:
        raise ValidationPanelError("unexpected validation-panel schema")
    if panel.get("schema_version") != PANEL_SCHEMA_VERSION:
        raise ValidationPanelError("unexpected validation-panel schema version")
    if panel.get("status") != PANEL_STATUS:
        raise ValidationPanelError("validation panel is not frozen")
    kind = panel.get("panel_kind")
    if kind not in PANEL_KINDS:
        raise ValidationPanelError(f"unknown panel kind {kind!r}")
    if panel.get("partition") != VALIDATION_PARTITION:
        raise ValidationPanelError("validation panel cannot use test or train data")
    if panel.get("leaderboard_protocol_sha256") != stable_json_sha256(config):
        raise ValidationPanelError("validation panel protocol hash drifted")
    if panel.get("frozen_inventory_sha256") != inventory["inventory_sha256"]:
        raise ValidationPanelError("validation panel inventory identity drifted")
    if panel.get("source_run_label") != inventory["source_run_label"]:
        raise ValidationPanelError("validation panel run label drifted")
    if panel.get("seed") != config["panels"][kind]["seed"]:
        raise ValidationPanelError("validation panel seed drifted")
    if panel.get("provenance") != _protocol_provenance(config, inventory):
        raise ValidationPanelError("validation panel corpus provenance drifted")
    if not _is_sha256(panel.get("artifact_sha256")):
        raise ValidationPanelError("validation panel lacks an artifact SHA-256")
    if panel_artifact_self_hash(panel) != panel["artifact_sha256"]:
        raise ValidationPanelError("validation panel self-hash does not match")

    rows = panel.get("rows")
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        raise ValidationPanelError("validation panel rows must be a sequence")
    active = set(config["panels"]["family_forensics"]["active_families"])
    layers = set(config["validation_data"]["layers"])
    for index, row in enumerate(rows):
        _validate_row(
            row,
            expected_row_index=index,
            active_families=active,
            layers=layers,
        )
    stream_indices = [int(row["stream_draw_index"]) for row in rows]
    if len(set(stream_indices)) != len(stream_indices):
        raise ValidationPanelError("stream draw indices are duplicated")

    contract = panel.get("draw_contract")
    if not isinstance(contract, Mapping):
        raise ValidationPanelError("validation panel lacks a draw contract")
    census = _derived_census(rows)
    for key, value in census.items():
        if contract.get(key) != value:
            raise ValidationPanelError(f"draw census mismatch for {key}")

    if kind == PRODUCTION_PANEL_ID:
        _validate_production_contract(contract, config=config, stream_indices=stream_indices)
    else:
        _validate_forensics_contract(
            contract,
            config=config,
            rows=rows,
            stream_indices=stream_indices,
        )


def _validate_production_contract(
    contract: Mapping[str, Any],
    *,
    config: Mapping[str, Any],
    stream_indices: Sequence[int],
) -> None:
    expected_keys = _CENSUS_KEYS | {
        "stage",
        "requested_stream_draws",
        "stream_draws_consumed",
    }
    if set(contract) != expected_keys:
        raise ValidationPanelError("production draw contract has extra or missing fields")
    spec = config["panels"]["production_law"]
    requested = int(contract.get("requested_stream_draws", -1))
    stage = contract.get("stage")
    allowed = {
        "initial": int(spec["initial_draws"]),
        "expanded": int(spec["expanded_draws"]),
    }
    if stage not in allowed or requested != allowed[stage]:
        raise ValidationPanelError("production panel stage/draw count is off-protocol")
    if int(contract.get("stream_draws_consumed", -1)) != requested:
        raise ValidationPanelError("production panel did not consume its exact prefix")
    if list(stream_indices) != list(range(requested)):
        raise ValidationPanelError(
            "production panel rows are not the exact deterministic stream prefix"
        )


def _validate_forensics_contract(
    contract: Mapping[str, Any],
    *,
    config: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    stream_indices: Sequence[int],
) -> None:
    expected_keys = _CENSUS_KEYS | {
        "requested_nonterminal_examples_by_family",
        "stream_draws_consumed",
    }
    if set(contract) != expected_keys:
        raise ValidationPanelError("forensics draw contract has extra or missing fields")
    spec = config["panels"]["family_forensics"]
    active = tuple(spec["active_families"])
    targets = contract.get("requested_nonterminal_examples_by_family")
    if not isinstance(targets, Mapping) or set(targets) != set(active):
        raise ValidationPanelError("forensics family targets do not cover active families")
    initial = int(spec["initial_minimum_nonterminal_examples_per_family"])
    expanded = int(spec["expanded_minimum_nonterminal_examples_per_ambiguous_family"])
    if any(int(targets[family]) not in (initial, expanded) for family in active):
        raise ValidationPanelError("forensics family target is not initial or expanded")
    if any(bool(row["terminal"]) for row in rows):
        raise ValidationPanelError("family-forensics panel contains terminal rows")
    observed = Counter(str(row["teacher_family"]) for row in rows)
    if any(observed.get(family, 0) != int(targets[family]) for family in active):
        raise ValidationPanelError("forensics panel does not meet exact family targets")
    consumed = int(contract.get("stream_draws_consumed", -1))
    if not 0 < consumed <= int(spec["maximum_stream_draws"]):
        raise ValidationPanelError("forensics stream draw budget is invalid")
    if stream_indices != sorted(stream_indices):
        raise ValidationPanelError("forensics rows are not in deterministic stream order")
    if stream_indices and stream_indices[-1] >= consumed:
        raise ValidationPanelError("forensics row lies beyond consumed stream prefix")


def write_validation_panel(
    panel: Mapping[str, Any],
    path: str | Path,
    *,
    config: Mapping[str, Any],
    inventory: Mapping[str, Any],
) -> Path:
    """Validate then immutably write one fixed validation panel."""

    validate_validation_panel_artifact(panel, config=config, inventory=inventory)
    destination = Path(path)
    text = (
        __import__("json").dumps(panel, indent=2, sort_keys=True, allow_nan=False)
        + "\n"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(text)
        try:
            # A hard link publishes the already-complete temporary inode while
            # failing atomically if another process has frozen this path. Using
            # os.replace here would allow a concurrent writer to overwrite a
            # different supposedly immutable panel.
            os.link(temporary, destination)
        except FileExistsError:
            if destination.read_text() != text:
                raise FileExistsError(
                    "fixed validation panel already exists with different "
                    f"content: {destination}"
                ) from None
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def _metric_vector_payload(vector: PanelMetricVector) -> dict[str, Any]:
    payload = asdict(vector)
    payload.pop("vector_sha256", None)
    return payload


def metric_vector_self_hash(vector: PanelMetricVector) -> str:
    return stable_json_sha256(_metric_vector_payload(vector))


def _panel_metric_rows(
    panel: Mapping[str, Any],
    *,
    family_filter: str | None,
) -> list[Mapping[str, Any]]:
    return [
        row
        for row in panel["rows"]
        if not bool(row["terminal"])
        and (family_filter is None or row["teacher_family"] == family_filter)
    ]


def bind_panel_metric_vector(
    values: Mapping[int, float],
    *,
    metric_id: str,
    source_kind: str,
    panel: Mapping[str, Any],
    config: Mapping[str, Any],
    inventory: Mapping[str, Any],
    family_filter: str | None = None,
    snapshot_step: int | None = None,
) -> PanelMetricVector:
    """Create a self-hashed validation vector for one allowed probability law."""

    validate_validation_panel_artifact(panel, config=config, inventory=inventory)
    if metric_id not in PANEL_METRIC_IDS:
        raise ValidationPanelError(f"metric id {metric_id!r} is not panel-approved")
    if source_kind not in METRIC_VECTOR_SOURCE_KINDS:
        raise ValidationPanelError(f"metric-vector source {source_kind!r} is invalid")
    active = set(config["panels"]["family_forensics"]["active_families"])
    if family_filter is not None and family_filter not in active:
        raise ValidationPanelError(f"metric-vector family {family_filter!r} is inactive")
    rows = _panel_metric_rows(panel, family_filter=family_filter)
    ids = tuple(int(row["stream_draw_index"]) for row in rows)
    if any(type(draw) is not int or draw < 0 for draw in values):
        raise ValidationPanelError(
            "metric-vector draw ids must be nonnegative integers"
        )
    if not ids or set(values) != set(ids):
        raise ValidationPanelError(
            "metric vector does not match the exact nonterminal panel draw census"
        )
    ordered_values = tuple(
        (
            draw,
            _require_finite_number(
                values[draw],
                field=f"metric value for draw {draw}",
            ),
        )
        for draw in ids
    )

    snapshot_name = None
    snapshot_sha256 = None
    state_source = "not_applicable"
    if source_kind == "snapshot_current_state":
        if snapshot_step is None:
            raise ValidationPanelError("snapshot metric vector lacks a step")
        matches = [
            spec
            for spec in prepare_snapshot_specs(config, inventory)
            if spec.step == int(snapshot_step)
        ]
        if len(matches) != 1:
            raise ValidationPanelError(
                "metric-vector snapshot step is absent from the frozen inventory"
            )
        snapshot = matches[0]
        snapshot_name = snapshot.name
        snapshot_sha256 = snapshot.sha256
        state_source = CURRENT_STATE_SOURCE
    elif snapshot_step is not None:
        raise ValidationPanelError(
            "uniform canonical-successor vector cannot carry a snapshot step"
        )

    provisional = PanelMetricVector(
        schema_version=METRIC_VECTOR_SCHEMA_VERSION,
        partition=VALIDATION_PARTITION,
        panel_kind=str(panel["panel_kind"]),
        panel_artifact_sha256=str(panel["artifact_sha256"]),
        source_kind=source_kind,
        snapshot_step=None if snapshot_step is None else int(snapshot_step),
        snapshot_name=snapshot_name,
        snapshot_sha256=snapshot_sha256,
        state_source=state_source,
        metric_id=metric_id,
        family_filter=family_filter,
        values=ordered_values,
        vector_sha256="",
    )
    vector = PanelMetricVector(
        **{
            **_metric_vector_payload(provisional),
            "vector_sha256": metric_vector_self_hash(provisional),
        }
    )
    validate_panel_metric_vector(
        vector,
        panel=panel,
        config=config,
        inventory=inventory,
    )
    return vector


def validate_panel_metric_vector(
    vector: PanelMetricVector,
    *,
    panel: Mapping[str, Any],
    config: Mapping[str, Any],
    inventory: Mapping[str, Any],
) -> None:
    """Refuse off-panel, test, best-state, incomplete, or mutated metric vectors."""

    validate_validation_panel_artifact(panel, config=config, inventory=inventory)
    if vector.schema_version != METRIC_VECTOR_SCHEMA_VERSION:
        raise ValidationPanelError("metric-vector schema version drifted")
    if vector.partition != VALIDATION_PARTITION:
        raise ValidationPanelError("metric vectors must be validation only")
    if (
        vector.panel_kind != panel["panel_kind"]
        or vector.panel_artifact_sha256 != panel["artifact_sha256"]
    ):
        raise ValidationPanelError("metric vector is bound to a different panel")
    if vector.metric_id not in PANEL_METRIC_IDS:
        raise ValidationPanelError("metric vector has an unapproved metric id")
    active = set(config["panels"]["family_forensics"]["active_families"])
    if vector.family_filter is not None and vector.family_filter not in active:
        raise ValidationPanelError("metric vector has an inactive family filter")
    rows = _panel_metric_rows(panel, family_filter=vector.family_filter)
    ids = tuple(int(row["stream_draw_index"]) for row in rows)
    if tuple(draw for draw, _value in vector.values) != ids:
        raise ValidationPanelError("metric vector draw order/census drifted")
    if not all(math.isfinite(float(value)) for _draw, value in vector.values):
        raise ValidationPanelError("metric vector contains non-finite values")

    if vector.source_kind == "snapshot_current_state":
        matches = [
            spec
            for spec in prepare_snapshot_specs(config, inventory)
            if spec.step == vector.snapshot_step
        ]
        if len(matches) != 1:
            raise ValidationPanelError("metric vector snapshot is not frozen")
        snapshot = matches[0]
        if (
            vector.snapshot_name != snapshot.name
            or vector.snapshot_sha256 != snapshot.sha256
            or vector.state_source != CURRENT_STATE_SOURCE
        ):
            raise ValidationPanelError(
                "metric vector is not bound to the frozen current snapshot state"
            )
    elif vector.source_kind == "uniform_canonical_successor":
        if any(
            value is not None
            for value in (
                vector.snapshot_step,
                vector.snapshot_name,
                vector.snapshot_sha256,
            )
        ) or vector.state_source != "not_applicable":
            raise ValidationPanelError(
                "uniform metric vector carries checkpoint provenance"
            )
    else:
        raise ValidationPanelError("metric vector has an invalid source kind")
    if not _is_sha256(vector.vector_sha256):
        raise ValidationPanelError("metric vector lacks a SHA-256")
    if metric_vector_self_hash(vector) != vector.vector_sha256:
        raise ValidationPanelError("metric vector self-hash does not match")


def _bootstrap_contract(config: Mapping[str, Any]) -> Mapping[str, Any]:
    validate_leaderboard_config(config)
    return config["statistics"]["paired_bootstrap"]


def paired_weighted_bootstrap(
    left: PanelMetricVector,
    right: PanelMetricVector,
    *,
    panel: Mapping[str, Any],
    interval_kind: str,
    config: Mapping[str, Any],
    inventory: Mapping[str, Any],
) -> PairedBootstrapResult:
    """Paired LEFT-minus-RIGHT interval over an exact self-hashed panel census.

    Draw ids and importance weights are derived from ``panel``. A caller may
    provide values, but cannot choose a subset, substitute weights, or detach the
    interval from validation provenance.
    """

    contract = _bootstrap_contract(config)
    validate_panel_metric_vector(
        left,
        panel=panel,
        config=config,
        inventory=inventory,
    )
    validate_panel_metric_vector(
        right,
        panel=panel,
        config=config,
        inventory=inventory,
    )
    if interval_kind not in ("two_sided", "one_sided_lower"):
        raise ValidationPanelError(f"unknown bootstrap interval kind {interval_kind!r}")
    if left.metric_id != right.metric_id:
        raise ValidationPanelError(
            "paired bootstrap vectors use different metric ids"
        )
    if left.family_filter != right.family_filter:
        raise ValidationPanelError(
            "paired bootstrap vectors use different family filters"
        )
    if tuple(draw for draw, _value in left.values) != tuple(
        draw for draw, _value in right.values
    ):
        raise ValidationPanelError(
            "paired bootstrap requires exact left/right draw-id alignment"
        )
    ids = tuple(draw for draw, _value in left.values)
    selected_rows = _panel_metric_rows(panel, family_filter=left.family_filter)
    left_values = np.asarray([float(value) for _draw, value in left.values], dtype=np.float64)
    right_values = np.asarray(
        [float(value) for _draw, value in right.values],
        dtype=np.float64,
    )
    weight_values = np.asarray(
        [float(row["importance_weight"]) for row in selected_rows],
        dtype=np.float64,
    )
    if not np.isfinite(left_values).all() or not np.isfinite(right_values).all():
        raise ValidationPanelError("paired bootstrap values are non-finite")
    if not np.isfinite(weight_values).all() or bool((weight_values <= 0.0).any()):
        raise ValidationPanelError("paired bootstrap weights must be finite and positive")
    differences = left_values - right_values
    observed = float(np.sum(weight_values * differences) / np.sum(weight_values))

    seed = int(contract["seed"])
    replicates = int(contract["replicates"])
    chunk_size = int(contract["chunk_replicates"])
    rng = np.random.default_rng(seed)
    bootstrap = np.empty(replicates, dtype=np.float64)
    n = len(ids)
    offset = 0
    while offset < replicates:
        take = min(chunk_size, replicates - offset)
        sampled = rng.integers(0, n, size=(take, n))
        sampled_weights = weight_values[sampled]
        sampled_differences = differences[sampled]
        bootstrap[offset : offset + take] = np.sum(
            sampled_weights * sampled_differences,
            axis=1,
        ) / np.sum(sampled_weights, axis=1)
        offset += take

    if interval_kind == "two_sided":
        interval = contract["two_sided_interval"]
        confidence = float(interval["confidence"])
        alpha = 1.0 - confidence
        lower, upper = np.quantile(
            bootstrap,
            (alpha / 2.0, 1.0 - alpha / 2.0),
            method="linear",
        )
        upper_bound: float | None = float(upper)
        contains_zero: bool | None = bool(lower <= 0.0 <= upper)
    else:
        interval = contract["one_sided_lower_bound"]
        confidence = float(interval["confidence"])
        lower = np.quantile(
            bootstrap,
            1.0 - confidence,
            method="linear",
        )
        upper_bound = None
        contains_zero = None
    return PairedBootstrapResult(
        schema_version=BOOTSTRAP_SCHEMA_VERSION,
        interval_kind=interval_kind,
        panel_kind=str(panel["panel_kind"]),
        panel_artifact_sha256=str(panel["artifact_sha256"]),
        family_filter=left.family_filter,
        metric_id=left.metric_id,
        left_source_kind=left.source_kind,
        right_source_kind=right.source_kind,
        left_vector_sha256=left.vector_sha256,
        right_vector_sha256=right.vector_sha256,
        estimator=str(contract["estimator"]),
        difference_direction="left_minus_right",
        observation_count=n,
        draw_id_census_sha256=stable_json_sha256(
            {
                "panel_artifact_sha256": panel["artifact_sha256"],
                "family_filter": left.family_filter,
                "draw_ids": ids,
            }
        ),
        seed=seed,
        replicates=replicates,
        confidence=confidence,
        observed_difference=observed,
        lower_bound=float(lower),
        upper_bound=upper_bound,
        contains_zero=contains_zero,
    )


def production_panel_expansion_decision(
    *,
    panel: Mapping[str, Any],
    primary_interval: PairedBootstrapResult | None,
    inconclusive_hard_gates: Sequence[str],
    config: Mapping[str, Any],
    inventory: Mapping[str, Any],
) -> ProductionPanelExpansionDecision:
    """Apply frozen expansion triggers without identifying or ordering checkpoints."""

    validate_leaderboard_config(config)
    validate_validation_panel_artifact(panel, config=config, inventory=inventory)
    if panel["panel_kind"] != PRODUCTION_PANEL_ID:
        raise ValidationPanelError(
            "production expansion requires the production_law panel"
        )
    spec = config["panels"]["production_law"]
    initial = int(spec["initial_draws"])
    expanded = int(spec["expanded_draws"])
    current_draws = int(panel["draw_contract"]["requested_stream_draws"])
    if int(current_draws) not in (initial, expanded):
        raise ValidationPanelError("current production panel size is off-protocol")
    active = tuple(config["panels"]["family_forensics"]["active_families"])
    census = panel["draw_contract"]["family_nonterminal_census"]
    active_family_nonterminal_counts = {
        family: int(census.get(family, 0)) for family in active
    }
    if primary_interval is not None:
        if primary_interval.interval_kind != "two_sided":
            raise ValidationPanelError(
                "production expansion requires a two-sided interval"
            )
        if primary_interval.panel_artifact_sha256 != panel["artifact_sha256"]:
            raise ValidationPanelError(
                "primary interval was not computed on this exact production panel"
            )
        if primary_interval.panel_kind != PRODUCTION_PANEL_ID:
            raise ValidationPanelError(
                "primary interval was not computed on production_law"
            )
        if primary_interval.family_filter is not None:
            raise ValidationPanelError(
                "primary interval must cover the whole nonterminal production panel"
            )
        if primary_interval.metric_id != "canonical_successor_nll":
            raise ValidationPanelError(
                "primary expansion interval must compare canonical-successor NLL"
            )
        if (
            primary_interval.left_source_kind != "snapshot_current_state"
            or primary_interval.right_source_kind != "snapshot_current_state"
        ):
            raise ValidationPanelError(
                "primary expansion interval must compare two current snapshots"
            )

    reasons: list[str] = []
    if primary_interval is None:
        reasons.append("primary_paired_interval_unavailable")
    elif primary_interval.contains_zero:
        reasons.append("primary_paired_interval_contains_zero")
    sparse = sorted(
        family
        for family, count in active_family_nonterminal_counts.items()
        if int(count)
        < int(
            spec["minimum_active_family_nonterminal_examples_before_selection"]
        )
    )
    threshold = int(
        spec["minimum_active_family_nonterminal_examples_before_selection"]
    )
    reasons.extend(
        f"active_family_below_{threshold}:{family}" for family in sparse
    )
    reasons.extend(
        f"hard_gate_numerically_inconclusive:{gate}"
        for gate in sorted({str(gate) for gate in inconclusive_hard_gates})
        if gate
    )
    can_expand = current_draws == initial
    return ProductionPanelExpansionDecision(
        current_draws=int(current_draws),
        target_draws=expanded if reasons and can_expand else int(current_draws),
        expand=bool(reasons and can_expand),
        reasons=tuple(reasons),
    )


def family_forensics_expansion_decision(
    *,
    panel: Mapping[str, Any],
    one_sided_intervals: Mapping[str, PairedBootstrapResult],
    config: Mapping[str, Any],
    inventory: Mapping[str, Any],
) -> FamilyForensicsExpansionDecision:
    """Request more rows for incomplete/ambiguous families without a gate verdict."""

    validate_leaderboard_config(config)
    validate_validation_panel_artifact(panel, config=config, inventory=inventory)
    if panel["panel_kind"] != FAMILY_FORENSICS_PANEL_ID:
        raise ValidationPanelError(
            "family expansion requires the family_forensics panel"
        )
    spec = config["panels"]["family_forensics"]
    active = tuple(spec["active_families"])
    if not set(one_sided_intervals).issubset(active):
        raise ValidationPanelError("one-sided intervals contain an inactive family")
    initial = int(spec["initial_minimum_nonterminal_examples_per_family"])
    expanded = int(spec["expanded_minimum_nonterminal_examples_per_ambiguous_family"])
    census = panel["draw_contract"]["family_nonterminal_census"]
    current_counts = {family: int(census.get(family, 0)) for family in active}
    ambiguous: list[str] = []
    targets: dict[str, int] = {}
    for family in active:
        count = int(current_counts[family])
        if count not in (initial, expanded):
            raise ValidationPanelError(f"forensics count for {family} is off-protocol")
        interval = one_sided_intervals.get(family)
        if interval is None:
            ambiguous.append(family)
            targets[family] = max(count, expanded)
            continue
        if interval.interval_kind != "one_sided_lower":
            raise ValidationPanelError(
                f"family interval for {family} is not a one-sided lower bound"
            )
        if interval.panel_artifact_sha256 != panel["artifact_sha256"]:
            raise ValidationPanelError(
                f"family interval for {family} uses a different panel"
            )
        if (
            interval.panel_kind != FAMILY_FORENSICS_PANEL_ID
            or interval.family_filter != family
        ):
            raise ValidationPanelError(
                f"family interval for {family} has the wrong panel/family filter"
            )
        if interval.metric_id != "canonical_successor_log_likelihood":
            raise ValidationPanelError(
                f"family interval for {family} must compare canonical log likelihood"
            )
        if (
            interval.left_source_kind != "snapshot_current_state"
            or interval.right_source_kind != "uniform_canonical_successor"
        ):
            raise ValidationPanelError(
                f"family interval for {family} must be learned minus uniform"
            )
        lower_value = float(interval.lower_bound)
        if not math.isfinite(lower_value):
            raise ValidationPanelError(f"lower bound for {family} is non-finite")
        if lower_value <= 0.0:
            ambiguous.append(family)
            targets[family] = max(count, expanded)
        else:
            targets[family] = count
    return FamilyForensicsExpansionDecision(
        current_counts=tuple((family, int(current_counts[family])) for family in active),
        target_counts=tuple((family, targets[family]) for family in active),
        statistically_ambiguous_families=tuple(ambiguous),
        expand=any(targets[family] > current_counts[family] for family in active),
    )


def bootstrap_result_dict(result: PairedBootstrapResult) -> dict[str, Any]:
    return asdict(result)


__all__ = [
    "BOOTSTRAP_SCHEMA_VERSION",
    "FAMILY_FORENSICS_PANEL_ID",
    "METRIC_VECTOR_SCHEMA_VERSION",
    "PANEL_SCHEMA",
    "PANEL_SCHEMA_VERSION",
    "PANEL_STATUS",
    "STATE_DIGEST_SCHEMA",
    "FamilyForensicsExpansionDecision",
    "PairedBootstrapResult",
    "PanelMetricVector",
    "ProductionPanelExpansionDecision",
    "ValidationPanelError",
    "bind_panel_metric_vector",
    "bootstrap_result_dict",
    "family_forensics_expansion_decision",
    "metric_vector_self_hash",
    "paired_weighted_bootstrap",
    "panel_artifact_self_hash",
    "production_panel_expansion_decision",
    "seal_validation_panel",
    "validate_panel_metric_vector",
    "validate_validation_panel_artifact",
    "write_validation_panel",
]
