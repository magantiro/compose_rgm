"""Build immutable RingCore-V1 validation panels from the production draw law.

This module does not load or score a checkpoint.  It materializes the exact
checkpoint-independent rows consumed by the successor leaderboard:

* ``production_law`` is an exact deterministic stream prefix, terminals included;
* ``family_forensics`` is a deterministic sparse scan retaining nonterminal rows
  until every active family reaches its frozen target.

The draw stream is the real ``FactorizedMarkDataset`` over the real
``HierarchicalMarkSampler``.  No sampling law is reconstructed in experiment
code.  Every retained row is tied to an immutable packed-trace address and to
the exact persistent-slot source state.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Any

from compose_v4.chem.persistent_state_identity import (
    persistent_slot_state_sha256,
)
from compose_v4.data.production_edit_corpus import LayeredEditCorpus
from compose_v4.experiments.factorized_mark_conditional import (
    FactorizedMarkDataset,
)
from compose_v4.experiments.factorized_successor_training import (
    rewrite_action_codec_sha256,
)
from compose_v4.experiments.hierarchical_sampler import PRODUCTION_LAYERS
from compose_v4.experiments.ringcore_semantic_axes import (
    context_from_path_record,
    label_validation_semantic_axes,
    semantic_axis_labeler_contract_sha256,
    validate_labeler_against_leaderboard_config,
)
from compose_v4.experiments.ringcore_successor_leaderboard import (
    PRODUCTION_PANEL_ID,
    stable_json_sha256,
    validate_inventory,
    validate_leaderboard_config,
)
from compose_v4.experiments.ringcore_validation_panel import (
    FAMILY_FORENSICS_PANEL_ID,
    STATE_DIGEST_SCHEMA,
    ValidationPanelError,
    seal_validation_panel,
)
from compose_v4.rewrite.action_codec import canonical_family
from compose_v4.rewrite.kernel import canonical_state_key

PANEL_SAMPLER_SOURCE_PATHS = (
    "src/compose_v4/data/production_edit_corpus.py",
    "src/compose_v4/experiments/factorized_mark_conditional.py",
    "src/compose_v4/experiments/hierarchical_sampler.py",
    "src/compose_v4/experiments/tracelet_conditional.py",
    "src/compose_v4/rewrite/progress.py",
)
_LAYER_TO_CONFIG = {
    "corruption": "general_corruption",
    "general_corruption": "general_corruption",
    "cycle_ops": "cycle_operations",
    "cycle_operations": "cycle_operations",
    "mmp_analogue": "mmp_analogue",
}


def panel_sampler_implementation_sha256(
    *,
    repository_root: Path | None = None,
) -> str:
    """Hash every production source that determines one panel draw."""

    root = (
        Path(repository_root)
        if repository_root is not None
        else Path(__file__).resolve().parents[3]
    )
    digest = hashlib.sha256()
    for relative in PANEL_SAMPLER_SOURCE_PATHS:
        path = root / relative
        if not path.is_file():
            raise ValidationPanelError(f"panel sampler source is absent: {relative}")
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _sampling_contract(config: Mapping[str, Any]) -> Mapping[str, Any]:
    validation = config["validation_data"]
    contract = validation.get("record_sampling")
    if not isinstance(contract, Mapping):
        raise ValidationPanelError("validation record_sampling must be a frozen object")
    return contract


def _configured_layer_weights(config: Mapping[str, Any]) -> dict[str, float]:
    layers = config["validation_data"]["layers"]
    return {str(layer): float(payload["trace_draw_weight"]) for layer, payload in layers.items()}


def _observed_sampler_weights(corpus: LayeredEditCorpus) -> dict[str, float]:
    sampler = corpus.sampler
    layers = tuple(str(layer) for layer in sampler._layers)
    probabilities = tuple(float(value) for value in sampler._layer_probs)
    return dict(zip(layers, probabilities, strict=True))


def validate_panel_corpus_sampler(
    corpus: LayeredEditCorpus,
    *,
    config: Mapping[str, Any],
) -> None:
    """Fail before drawing if the corpus sampler is not the frozen V1 law."""

    if not isinstance(corpus, LayeredEditCorpus):
        raise ValidationPanelError("panel construction requires a production LayeredEditCorpus")
    corpus.verify_alignment()
    contract = _sampling_contract(config)
    expected_hash = contract.get("implementation_sha256")
    observed_hash = panel_sampler_implementation_sha256()
    if expected_hash != observed_hash:
        raise ValidationPanelError("panel sampler implementation changed after protocol freeze")
    sampler = corpus.sampler
    expected_bins = tuple(int(value) for value in contract["curriculum_bin_edges"])
    if tuple(sampler.path_length_bins) != expected_bins:
        raise ValidationPanelError(
            "corpus sampler path-length bins disagree with the panel protocol"
        )
    if float(sampler.cold_element_floor) != float(contract["cold_element_floor"]):
        raise ValidationPanelError(
            "corpus sampler cold-element floor disagrees with the panel protocol"
        )
    expected_weights = _configured_layer_weights(config)
    observed_weights = _observed_sampler_weights(corpus)
    if set(observed_weights) != set(expected_weights):
        raise ValidationPanelError("corpus sampler layers disagree with the panel protocol")
    if any(
        abs(observed_weights[layer] - expected_weights[layer]) > 1e-12 for layer in expected_weights
    ):
        raise ValidationPanelError("corpus sampler weights disagree with the panel protocol")
    ordered_bounds = tuple(corpus.layer_bounds)
    if ordered_bounds != PRODUCTION_LAYERS:
        raise ValidationPanelError("corpus records are not in canonical production-layer order")
    for record_index, tag in enumerate(sampler.tags):
        if tag.layer != corpus.layer_of(record_index):
            raise ValidationPanelError("corpus sampler tags are not aligned with record order")
    configured_counts = {
        layer: int(payload["records"])
        for layer, payload in config["validation_data"]["layers"].items()
    }
    observed_counts = {
        layer: int(upper - lower) for layer, (lower, upper) in corpus.layer_bounds.items()
    }
    if observed_counts != configured_counts:
        raise ValidationPanelError(
            "corpus validation record census disagrees with the panel protocol"
        )


def _dataset(
    corpus: LayeredEditCorpus,
    *,
    seed: int,
    length: int,
    config: Mapping[str, Any],
) -> FactorizedMarkDataset:
    validation = config["validation_data"]
    time = validation["time_sampling"]
    progress = validation["progress_sampling"]
    ordinary = tuple(float(value) for value in time["ordinary_interval"])
    if ordinary != (0.01, 0.99):
        raise ValidationPanelError("FactorizedMarkDataset ordinary-time interval drifted")
    return FactorizedMarkDataset(
        corpus.records,
        start_index=0,
        length=int(length),
        seed=int(seed),
        late_time_fraction=float(time["late_time_fraction"]),
        operational_horizon=float(time["operational_horizon"]),
        progress_stratification_fraction=float(progress["stratification_fraction"]),
        record_index_sampler=corpus.sampler,
    )


def _normalized_layer(layer: str) -> str:
    try:
        return _LAYER_TO_CONFIG[str(layer)]
    except KeyError:
        raise ValidationPanelError(
            f"packed trace uses an unknown validation layer {layer!r}"
        ) from None


def _record_key(address: Any) -> str:
    return (
        f"{address.layer}/{address.partition}/{address.packed_shard_name}:"
        f"{address.entry_index}:{address.trace_id}"
    )


def _state_ref(address: Any, *, progress_index: int) -> dict[str, Any]:
    shard_name = str(
        PurePosixPath(
            str(address.layer),
            str(address.partition),
            str(address.packed_shard_name),
        )
    )
    return {
        "shard_name": shard_name,
        "shard_sha256": address.packed_shard_content_sha256,
        "record_index": int(address.entry_index),
        "progress_index": int(progress_index),
        "state_digest_schema": STATE_DIGEST_SCHEMA,
    }


def _unsealed_row(
    *,
    dataset: FactorizedMarkDataset,
    stream_draw_index: int,
    row_index: int,
) -> dict[str, Any]:
    example = dataset[int(stream_draw_index)]
    return _unsealed_row_from_example(
        dataset=dataset,
        example=example,
        stream_draw_index=stream_draw_index,
        row_index=row_index,
    )


def _unsealed_row_from_example(
    *,
    dataset: FactorizedMarkDataset,
    example: Any,
    stream_draw_index: int,
    row_index: int,
) -> dict[str, Any]:
    if example.record_index is None or example.progress_index is None:
        raise ValidationPanelError("production dataset omitted record/progress identity")
    record = dataset.records[int(example.record_index)]
    address = record.corpus_address
    if address is None:
        raise ValidationPanelError("validation panel cannot contain an unaddressed trace")
    if address.partition != "validation":
        raise ValidationPanelError("validation panel draw resolved outside validation")
    progress_index = int(example.progress_index)
    path = record.path
    if address.path_length != path.path_length:
        raise ValidationPanelError("packed address and decoded path length disagree")
    if persistent_slot_state_sha256(example.state) != persistent_slot_state_sha256(
        path.state_at(progress_index)
    ):
        raise ValidationPanelError("sampled state does not equal its addressed progress state")
    layer = _normalized_layer(address.layer)
    assignment = label_validation_semantic_axes(
        context_from_path_record(
            record,
            progress_index=progress_index,
            layer=layer,
            partition="validation",
        )
    )
    terminal = progress_index == path.path_length
    if terminal != assignment.terminal:
        raise ValidationPanelError("semantic labeler terminal status disagrees with the path")
    teacher_family = None
    teacher_rule_name = None
    teacher_successor_key = None
    teacher_action_sha256 = None
    if not terminal:
        step = path.trace.steps[progress_index]
        teacher_rule_name = str(step.rule_name)
        teacher_family = canonical_family(teacher_rule_name)
        if example.teacher_rule_name != teacher_rule_name:
            raise ValidationPanelError("sampled teacher rule disagrees with immutable trace")
        if example.teacher_action != step.action:
            raise ValidationPanelError("sampled teacher action disagrees with immutable trace")
        teacher_successor_key = canonical_state_key(path.state_at(progress_index + 1))
        teacher_action_sha256 = rewrite_action_codec_sha256(
            teacher_rule_name,
            step.action,
        )
    elif (
        example.teacher_rule_name is not None
        or example.teacher_action is not None
        or example.teacher_rate != 0.0
    ):
        raise ValidationPanelError("terminal dataset draw carries teacher supervision")
    source_state_key = canonical_state_key(example.state)
    return {
        "row_index": int(row_index),
        "stream_draw_index": int(stream_draw_index),
        "partition": "validation",
        "layer": layer,
        "record_key": _record_key(address),
        "path_length": int(path.path_length),
        "progress_index": progress_index,
        "time": float(example.time),
        "importance_weight": float(example.importance_weight),
        "terminal": terminal,
        "teacher_family": teacher_family,
        "teacher_rule_name": teacher_rule_name,
        "teacher_successor_key": teacher_successor_key,
        "teacher_action_sha256": teacher_action_sha256,
        "semantic_axis_values": assignment.axis_values,
        "semantic_cell_id": assignment.semantic_cell_id,
        "source_state_key": source_state_key,
        "source_state_sha256": persistent_slot_state_sha256(example.state),
        "exact_state_ref": _state_ref(
            address,
            progress_index=progress_index,
        ),
    }


def _preflight(
    corpus: LayeredEditCorpus,
    *,
    config: Mapping[str, Any],
    inventory: Mapping[str, Any],
) -> None:
    validate_leaderboard_config(config)
    validate_inventory(config, inventory, require_exact_self_hash=True)
    contract_sha256 = validate_labeler_against_leaderboard_config(config)
    if contract_sha256 != semantic_axis_labeler_contract_sha256():
        raise ValidationPanelError("semantic labeler contract changed during panel construction")
    validate_panel_corpus_sampler(corpus, config=config)


def build_production_validation_panel(
    corpus: LayeredEditCorpus,
    *,
    stage: str,
    config: Mapping[str, Any],
    inventory: Mapping[str, Any],
) -> dict[str, Any]:
    """Build and seal one exact production-law prefix."""

    _preflight(corpus, config=config, inventory=inventory)
    spec = config["panels"][PRODUCTION_PANEL_ID]
    allowed = {
        "initial": int(spec["initial_draws"]),
        "expanded": int(spec["expanded_draws"]),
    }
    if stage not in allowed:
        raise ValidationPanelError(f"unknown production panel stage {stage!r}")
    draws = allowed[stage]
    dataset = _dataset(
        corpus,
        seed=int(spec["seed"]),
        length=draws,
        config=config,
    )
    rows = tuple(
        _unsealed_row(
            dataset=dataset,
            stream_draw_index=draw_index,
            row_index=draw_index,
        )
        for draw_index in range(draws)
    )
    return seal_validation_panel(
        panel_kind=PRODUCTION_PANEL_ID,
        rows=rows,
        draw_contract={
            "stage": stage,
            "requested_stream_draws": draws,
            "stream_draws_consumed": draws,
        },
        config=config,
        inventory=inventory,
    )


def build_family_forensics_validation_panel(
    corpus: LayeredEditCorpus,
    *,
    requested_nonterminal_examples_by_family: Mapping[str, int],
    config: Mapping[str, Any],
    inventory: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the deterministic sparse all-family capability panel."""

    _preflight(corpus, config=config, inventory=inventory)
    spec = config["panels"][FAMILY_FORENSICS_PANEL_ID]
    active = tuple(str(value) for value in spec["active_families"])
    targets = {
        str(family): int(value)
        for family, value in requested_nonterminal_examples_by_family.items()
    }
    if set(targets) != set(active):
        raise ValidationPanelError("forensics targets must exactly cover active families")
    initial = int(spec["initial_minimum_nonterminal_examples_per_family"])
    expanded = int(spec["expanded_minimum_nonterminal_examples_per_ambiguous_family"])
    if any(target not in (initial, expanded) for target in targets.values()):
        raise ValidationPanelError("forensics targets must use frozen initial/expanded counts")
    maximum = int(spec["maximum_stream_draws"])
    dataset = _dataset(
        corpus,
        seed=int(spec["seed"]),
        length=maximum,
        config=config,
    )
    counts = dict.fromkeys(active, 0)
    rows: list[dict[str, Any]] = []
    stream_draws_consumed = 0
    for draw_index in range(maximum):
        example = dataset[draw_index]
        family = (
            None
            if example.teacher_rule_name is None
            else canonical_family(str(example.teacher_rule_name))
        )
        stream_draws_consumed = draw_index + 1
        if family is None or family not in targets or counts[family] >= targets[family]:
            continue
        candidate = _unsealed_row_from_example(
            dataset=dataset,
            example=example,
            stream_draw_index=draw_index,
            row_index=len(rows),
        )
        if candidate["teacher_family"] != family:
            raise ValidationPanelError("forensics prefilter family disagrees with sealed row")
        rows.append(candidate)
        counts[family] += 1
        if all(counts[family] == targets[family] for family in active):
            break
    shortfalls = {
        family: targets[family] - counts[family]
        for family in active
        if counts[family] != targets[family]
    }
    if shortfalls:
        raise ValidationPanelError(
            f"forensics stream budget exhausted before family targets: {shortfalls}"
        )
    return seal_validation_panel(
        panel_kind=FAMILY_FORENSICS_PANEL_ID,
        rows=tuple(rows),
        draw_contract={
            "requested_nonterminal_examples_by_family": targets,
            "stream_draws_consumed": stream_draws_consumed,
        },
        config=config,
        inventory=inventory,
    )


def panel_build_identity(
    *,
    config: Mapping[str, Any],
    inventory: Mapping[str, Any],
    panel: Mapping[str, Any],
) -> dict[str, Any]:
    """Compact identity suitable for a durable audit manifest."""

    validate_leaderboard_config(config)
    validate_inventory(config, inventory, require_exact_self_hash=True)
    return {
        "leaderboard_protocol_sha256": stable_json_sha256(config),
        "inventory_sha256": inventory["inventory_sha256"],
        "semantic_axis_labeler_contract_sha256": (semantic_axis_labeler_contract_sha256()),
        "panel_sampler_implementation_sha256": (panel_sampler_implementation_sha256()),
        "panel_kind": panel["panel_kind"],
        "panel_artifact_sha256": panel["artifact_sha256"],
        "row_count": panel["draw_contract"]["row_count"],
        "nonterminal_row_count": panel["draw_contract"]["nonterminal_row_count"],
        "terminal_row_count": panel["draw_contract"]["terminal_row_count"],
    }


__all__ = [
    "PANEL_SAMPLER_SOURCE_PATHS",
    "build_family_forensics_validation_panel",
    "build_production_validation_panel",
    "panel_build_identity",
    "panel_sampler_implementation_sha256",
    "validate_panel_corpus_sampler",
]
