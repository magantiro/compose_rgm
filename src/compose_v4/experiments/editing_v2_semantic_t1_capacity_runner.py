"""Deterministic cache-only semantic Editing-V2 T1 capacity runner.

The GPU path consumes only the exact prepared-input artifact and the strict
successor-coordinate cache.  Chemistry execution and canonical grouping happen
before this runner.  The objective is the productive embedded canonical-
successor NLL; the total-hazard head is frozen, excluded from the optimizer,
and required to remain bit-identical.
"""

from __future__ import annotations

import copy
import hashlib
import io
import json
import math
import os
import random
import tempfile
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import torch
from torch import Tensor

from compose_v4.data.immutable_artifact import write_bytes_if_absent
from compose_v4.experiments.editing_p50_gate import state_dict_semantic_sha256
from compose_v4.experiments.editing_v2_semantic_development_cell_roles import (
    load_semantic_development_cell_roles,
)
from compose_v4.experiments.editing_v2_semantic_t1_capacity_policy import (
    SemanticT1CapacityPolicyError,
    load_semantic_t1_capacity_policy,
    semantic_t1_evaluation_schedule,
    semantic_t1_threshold_stop_allowed,
)
from compose_v4.experiments.editing_v2_semantic_t1_checkpoint import (
    CHECKPOINT_SCHEMA,
    CHECKPOINT_SCHEMA_VERSION,
    CHECKPOINT_STATUS,
)
from compose_v4.experiments.editing_v2_semantic_t1_decision import (
    RESULT_FILENAME,
    build_semantic_t1_capacity_result,
    semantic_t1_runner_implementation_sha256,
)
from compose_v4.experiments.editing_v2_semantic_t1_panel_cache import (
    SemanticT1PanelArtifact,
)
from compose_v4.experiments.editing_v2_semantic_t1_prepared_inputs import (
    LoadedSemanticT1PreparedInputs,
    SemanticT1PreparedInputError,
    authenticate_semantic_t1_prepared_inputs,
)
from compose_v4.experiments.editing_v2_semantic_t1_successor_cache import (
    SemanticT1SuccessorCache,
)
from compose_v4.experiments.factorized_mark_conditional import (
    FactorizedMarkCollator,
    FactorizedMarkExample,
)
from compose_v4.experiments.factorized_successor_training import (
    factorized_successor_identity_loss,
    forward_compiled_successor_partitions,
    forward_teacher_successor_batch,
)
from compose_v4.model.factorized_tracelet_rate_model import (
    FactorizedTraceletRateModel,
)

SELECTED_CHECKPOINT_FILENAME = "SEMANTIC_T1_SELECTED_CHECKPOINT.pt"
FAILURE_DIAGNOSTICS_FILENAME = "SEMANTIC_T1_FAILURE_DIAGNOSTICS.json"
FAILURE_DIAGNOSTICS_STATUS = (
    "BLOCKED_FROZEN_DIAGNOSTIC_SCOPES_NOT_IMPLEMENTED_NO_DOWNSTREAM_AUTHORITY"
)
NO_AUTHORITY = {
    "training_authorized": False,
    "bounded_p50_authorized": False,
    "long_training_authorized": False,
    "checkpoint_selection_authorized": False,
    "final_test_selection_authorized": False,
}
TOTAL_HAZARD_PREFIX = "total_hazard_head."
FAMILY_ROUTE_PREFIXES = ("family_head.",)
ACTION_ROUTE_PREFIXES: Mapping[str, tuple[str, ...]] = {
    "atom_insert": ("grow_root_head.", "grow_query.", "grow_option."),
    "atom_delete": ("delete_head.",),
    "atom_restate": ("restate_head.",),
    "bond_reorder": ("reorder_head.",),
    "bond_reroute": ("graft_head.",),
    "cycle_insert": ("cycle_close_head.",),
    "cycle_attach": ("cycle_open_head.",),
    "ring_system_restate": ("ring_restate_head.",),
}


class SemanticT1CapacityRunnerError(RuntimeError):
    """T1 execution or its recovery identity failed closed."""


class SemanticT1ResultBuilder(Protocol):
    """Build one result mapping from the runner's five evidence projections."""

    def __call__(
        self,
        *,
        provenance: Mapping[str, Any],
        run_integrity: Mapping[str, Any],
        evaluation_trajectory: Sequence[Mapping[str, Any]],
        entry_metrics: Sequence[Mapping[str, Any]],
        gradient_evidence: Sequence[Mapping[str, Any]],
    ) -> Mapping[str, Any]: ...


def _canonical_bytes(value: object, *, newline: bool = False) -> bytes:
    try:
        raw = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise SemanticT1CapacityRunnerError(
            "semantic T1 result metadata is not finite canonical JSON"
        ) from error
    return raw + (b"\n" if newline else b"")


def _sha(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def _torch_bytes(value: object) -> bytes:
    output = io.BytesIO()
    torch.save(value, output)
    return output.getvalue()


def optimizer_state_semantic_sha256(value: Mapping[str, Any]) -> str:
    """Hash optimizer tensors and metadata independently of pickle bytes."""

    digest = hashlib.sha256()
    digest.update(b"compose.semantic_t1.optimizer_state.v1\0")

    def update(item: object) -> None:
        if item is None:
            digest.update(b"n")
        elif type(item) is bool:
            digest.update(b"b1" if item else b"b0")
        elif type(item) is int:
            encoded = str(item).encode("ascii")
            digest.update(b"i" + len(encoded).to_bytes(8, "big") + encoded)
        elif type(item) is float:
            if not math.isfinite(item):
                raise SemanticT1CapacityRunnerError("optimizer metadata contains a nonfinite float")
            encoded = item.hex().encode("ascii")
            digest.update(b"f" + len(encoded).to_bytes(8, "big") + encoded)
        elif isinstance(item, str):
            encoded = item.encode("utf-8")
            digest.update(b"s" + len(encoded).to_bytes(8, "big") + encoded)
        elif isinstance(item, Tensor):
            if item.device.type == "meta" or item.layout != torch.strided:
                raise SemanticT1CapacityRunnerError(
                    "optimizer tensor is not materialized dense data"
                )
            tensor = item.detach().cpu().contiguous()
            if (tensor.is_floating_point() or tensor.is_complex()) and not bool(
                torch.isfinite(tensor).all()
            ):
                raise SemanticT1CapacityRunnerError("optimizer tensor contains nonfinite values")
            dtype = str(tensor.dtype).encode("ascii")
            shape = _canonical_bytes(list(tensor.shape))
            raw = tensor.reshape(-1).view(torch.uint8).numpy().tobytes()
            digest.update(b"t")
            for block in (dtype, shape, raw):
                digest.update(len(block).to_bytes(8, "big"))
                digest.update(block)
        elif isinstance(item, Mapping):
            digest.update(b"m" + len(item).to_bytes(8, "big"))
            encoded_items: list[tuple[bytes, object]] = []
            for key, nested in item.items():
                if type(key) is int:
                    key_bytes = b"i" + str(key).encode("ascii")
                elif isinstance(key, str):
                    key_bytes = b"s" + key.encode("utf-8")
                else:
                    raise SemanticT1CapacityRunnerError(
                        "optimizer mapping key must be int or string"
                    )
                encoded_items.append((key_bytes, nested))
            for key_bytes, nested in sorted(encoded_items, key=lambda pair: pair[0]):
                digest.update(len(key_bytes).to_bytes(8, "big"))
                digest.update(key_bytes)
                update(nested)
        elif isinstance(item, (list, tuple)):
            digest.update(b"l" + len(item).to_bytes(8, "big"))
            for nested in item:
                update(nested)
        else:
            raise SemanticT1CapacityRunnerError(
                f"unsupported optimizer metadata type: {type(item).__name__}"
            )

    update(value)
    return digest.hexdigest()


def _atomic_write(path: Path, payload: bytes) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if destination.read_bytes() != payload:
            raise SemanticT1CapacityRunnerError(
                f"immutable T1 artifact already exists with different bytes: {destination}"
            )
        return
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def _clone_state_dict(model: FactorizedTraceletRateModel) -> dict[str, Tensor]:
    return {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}


def _hazard_state(model: FactorizedTraceletRateModel) -> dict[str, Tensor]:
    state = {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.state_dict().items()
        if name.startswith(TOTAL_HAZARD_PREFIX)
    }
    if not state:
        raise SemanticT1CapacityRunnerError("model has no total-hazard parameters")
    return state


def _assert_hazard_identity(
    model: FactorizedTraceletRateModel, expected: Mapping[str, Tensor]
) -> None:
    observed = _hazard_state(model)
    if observed.keys() != expected.keys() or any(
        not torch.equal(observed[name], expected[name]) for name in expected
    ):
        raise SemanticT1CapacityRunnerError(
            "total-hazard parameters changed under the hazard-free T1 objective"
        )


def _rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": (None if not torch.cuda.is_available() else torch.cuda.get_rng_state_all()),
    }


def _restore_rng_state(value: Mapping[str, Any]) -> None:
    random.setstate(value["python"])
    np.random.set_state(value["numpy"])
    torch.set_rng_state(value["torch_cpu"])
    if value["torch_cuda"] is not None:
        if not torch.cuda.is_available():
            raise SemanticT1CapacityRunnerError(
                "checkpoint contains CUDA RNG state but CUDA is unavailable"
            )
        torch.cuda.set_rng_state_all(value["torch_cuda"])


@dataclass(frozen=True, slots=True)
class SemanticT1RuntimeInputs:
    prepared: LoadedSemanticT1PreparedInputs
    panel: SemanticT1PanelArtifact
    cache: SemanticT1SuccessorCache
    capacity_policy: Mapping[str, Any]

    def __post_init__(self) -> None:
        try:
            authenticated = authenticate_semantic_t1_prepared_inputs(
                self.prepared,
                panel=self.panel,
                cache=self.cache,
            )
        except SemanticT1PreparedInputError as error:
            raise SemanticT1CapacityRunnerError(
                f"prepared semantic T1 inputs failed strict authentication: {error}"
            ) from error
        if authenticated is not self.prepared:
            raise SemanticT1CapacityRunnerError(
                "prepared-input authentication returned a different runtime object"
            )
        artifact = self.prepared.artifact
        if (
            artifact["capacity_policy_sha256"] != self.capacity_policy.get("policy_sha256")
            or artifact["cache_completion_sha256"] != self.cache.completion.get("completion_sha256")
            or artifact["cache_manifest_sha256"] != self.cache.manifest.get("manifest_sha256")
            or artifact["panel_artifact_sha256"]
            != self.cache.manifest.get("panel_binding", {}).get("panel_artifact_sha256")
            or artifact.get("cache_source_revision_sha256")
            != self.cache.manifest.get("source_revision", {}).get("source_revision_sha256")
            or artifact.get("semantic_model_process_contract")
            != self.cache.manifest.get("semantic_model_process_contract")
            or artifact.get("decision_source_inventory_sha256")
            != self.cache.completion.get("decision_source_inventory_sha256")
            or artifact.get("initial_model_state_sha256")
            != self.cache.completion.get("initial_model_state_sha256")
        ):
            raise SemanticT1CapacityRunnerError(
                "prepared inputs, successor cache, and capacity policy disagree"
            )
        cell_roles = load_semantic_development_cell_roles()
        observed_cells = {
            str(entry["capability_cell_id"]) for entry in self.prepared.artifact["entries"]
        }
        missing_cells = sorted(cell_roles.required_cell_set - observed_cells)
        unexpected_cells = sorted(observed_cells - cell_roles.required_cell_set)
        if missing_cells or unexpected_cells:
            raise SemanticT1CapacityRunnerError(
                "semantic T1 runner requires the exact frozen required editing cells; "
                f"missing={missing_cells}, unexpected={unexpected_cells}"
            )

    @property
    def entries(self) -> tuple[Mapping[str, Any], ...]:
        return tuple(self.prepared.artifact["entries"])


def semantic_t1_address_stream(
    runtime: SemanticT1RuntimeInputs,
    *,
    draw_count: int,
) -> tuple[str, ...]:
    """Draw family, nonempty cell, and unique entry uniformly by SHA counter."""

    if type(draw_count) is not int or draw_count < 0:
        raise ValueError("draw_count must be a nonnegative integer")
    policy = runtime.capacity_policy
    families = tuple(policy["required_families"])
    entries_by_cell: dict[str, dict[str, tuple[str, ...]]] = {}
    for family in families:
        grouped: defaultdict[str, list[str]] = defaultdict(list)
        for entry in runtime.entries:
            if entry["model_family"] == family:
                grouped[str(entry["capability_cell_id"])].append(str(entry["panel_entry_sha256"]))
        if not grouped:
            raise SemanticT1CapacityRunnerError(
                f"semantic T1 panel has no entry for required family {family!r}"
            )
        entries_by_cell[family] = {
            cell: tuple(sorted(values)) for cell, values in sorted(grouped.items())
        }
    seed_material = ":".join(
        (
            str(policy["policy_sha256"]),
            str(runtime.prepared.artifact["artifact_sha256"]),
            str(policy["optimization"]["seed"]),
        )
    )

    def choose(values: Sequence[str], *, counter: int, level: str) -> str:
        digest = hashlib.sha256(f"{seed_material}:{counter}:{level}".encode()).digest()
        return values[int.from_bytes(digest[:8], "big") % len(values)]

    stream: list[str] = []
    for counter in range(draw_count):
        family = choose(families, counter=counter, level="family")
        cells = tuple(entries_by_cell[family])
        cell = choose(cells, counter=counter, level="cell")
        stream.append(
            choose(
                entries_by_cell[family][cell],
                counter=counter,
                level="entry",
            )
        )
    return tuple(stream)


def _collator(model: FactorizedTraceletRateModel) -> FactorizedMarkCollator:
    return FactorizedMarkCollator.from_capabilities(
        model.operator_capabilities,
        use_aromatic_bond_view=True,
        ring_catalog=model.ring_catalog,
    )


def _batch_for_ids(
    runtime: SemanticT1RuntimeInputs,
    model: FactorizedTraceletRateModel,
    collator: FactorizedMarkCollator,
    panel_ids: Sequence[str],
) -> tuple[Any, tuple[Any, ...], tuple[Any, ...], tuple[Mapping[str, Any], ...]]:
    entries_by_id = {str(entry["panel_entry_sha256"]): entry for entry in runtime.entries}
    entries = tuple(entries_by_id[panel_id] for panel_id in panel_ids)
    examples = [
        FactorizedMarkExample(
            state=runtime.prepared.states_by_panel_entry_sha256[panel_id],
            time=float.fromhex(str(entry["support_time_hex"])),
            teacher_action=None,
            teacher_rule_name=str(entry["model_family"]),
            teacher_rate=1.0,
            importance_weight=1.0,
        )
        for panel_id, entry in zip(panel_ids, entries, strict=True)
    ]
    batch = collator(examples)
    fibers = tuple(
        runtime.cache.record_for_panel_entry_sha256(panel_id).teacher_fiber
        for panel_id in panel_ids
    )
    if any(fiber is None for fiber in fibers):
        raise SemanticT1CapacityRunnerError(
            "semantic T1 training minibatch contains a terminal cache row"
        )
    partitions = tuple(
        runtime.prepared.partitions_by_panel_entry_sha256[panel_id] for panel_id in panel_ids
    )
    return batch, fibers, partitions, entries


@dataclass(frozen=True, slots=True)
class _MaterializedSemanticT1Panel:
    batch: Any
    panel_ids: tuple[str, ...]
    fibers: tuple[Any, ...]
    partitions: tuple[Any, ...]
    entries: tuple[Mapping[str, Any], ...]
    index_by_panel_id: Mapping[str, int]


def _index_factorized_batch(batch: Any, indices: Sequence[int]) -> Any:
    """Select arbitrary repeated panel rows without rerunning legal enumeration."""

    resolved = tuple(int(index) for index in indices)
    if not resolved or min(resolved) < 0 or max(resolved) >= batch.batch_size:
        raise SemanticT1CapacityRunnerError("materialized T1 batch index is invalid")
    tensor_indices = torch.tensor(resolved, dtype=torch.long)

    def tensor(value: Tensor) -> Tensor:
        return value.index_select(0, tensor_indices.to(value.device))

    def optional_tensor(value: Tensor | None) -> Tensor | None:
        return None if value is None else tensor(value)

    def aligned_tuple(value: Sequence[Any] | None) -> tuple[Any, ...] | None:
        return None if value is None else tuple(value[index] for index in resolved)

    sparse = batch.ring_grow_support_sparse
    selected_sparse = None
    if sparse is not None:
        rows = sparse.index_rows()
        selected_sparse = type(sparse).from_index_rows(
            tuple(rows[index] for index in resolved),
            width=sparse.width,
        )
    return replace(
        batch,
        states=tuple(batch.states[index] for index in resolved),
        atom_types=tensor(batch.atom_types),
        formal_charges=tensor(batch.formal_charges),
        implicit_h_counts=tensor(batch.implicit_h_counts),
        bonds=tensor(batch.bonds),
        neural_bonds=tensor(batch.neural_bonds),
        times=tensor(batch.times),
        atom_topology=tensor(batch.atom_topology),
        closure_topology=tensor(batch.closure_topology),
        ring_system_topology=tensor(batch.ring_system_topology),
        atom_delete_mask=tensor(batch.atom_delete_mask),
        cycle_edge_mask=tensor(batch.cycle_edge_mask),
        cyclic_pair_mask=tensor(batch.cyclic_pair_mask),
        graft_mask=tensor(batch.graft_mask),
        graft_remove_neighbors=tensor(batch.graft_remove_neighbors),
        graft_successor_groups=tensor(batch.graft_successor_groups),
        teacher_actions=aligned_tuple(batch.teacher_actions),
        teacher_rule_names=aligned_tuple(batch.teacher_rule_names),
        teacher_rates=tensor(batch.teacher_rates),
        importance_weights=tensor(batch.importance_weights),
        ring_restate_actions=aligned_tuple(batch.ring_restate_actions),
        ring_restate_successor_group_ids=aligned_tuple(batch.ring_restate_successor_group_ids),
        ring_restate_successor_group_descriptors=aligned_tuple(
            batch.ring_restate_successor_group_descriptors
        ),
        ring_restate_successor_group_multiplicities=aligned_tuple(
            batch.ring_restate_successor_group_multiplicities
        ),
        atom_delete_admission_mask=optional_tensor(batch.atom_delete_admission_mask),
        atom_restate_admission_mask=optional_tensor(batch.atom_restate_admission_mask),
        cycle_close_admission_mask=optional_tensor(batch.cycle_close_admission_mask),
        cycle_open_admission_mask=optional_tensor(batch.cycle_open_admission_mask),
        ring_grow_support_mask=optional_tensor(batch.ring_grow_support_mask),
        ring_grow_support_sparse=selected_sparse,
        ring_delete_actions=aligned_tuple(batch.ring_delete_actions),
        ring_grow_support_is_exact=(
            tensor(batch.ring_grow_support_is_exact)
            if isinstance(batch.ring_grow_support_is_exact, Tensor)
            else batch.ring_grow_support_is_exact
        ),
        ring_grow_enablement_is_exact=(
            tensor(batch.ring_grow_enablement_is_exact)
            if isinstance(batch.ring_grow_enablement_is_exact, Tensor)
            else batch.ring_grow_enablement_is_exact
        ),
        ring_topology_local_support_log_mass=optional_tensor(
            batch.ring_topology_local_support_log_mass
        ),
        ring_teacher_semantic_certificates=aligned_tuple(batch.ring_teacher_semantic_certificates),
        property_condition_values=optional_tensor(batch.property_condition_values),
        property_condition_mask=optional_tensor(batch.property_condition_mask),
    )


def _materialize_panel(
    runtime: SemanticT1RuntimeInputs,
    model: FactorizedTraceletRateModel,
    collator: FactorizedMarkCollator,
) -> _MaterializedSemanticT1Panel:
    panel_ids = tuple(str(entry["panel_entry_sha256"]) for entry in runtime.entries)
    batch, fibers, partitions, entries = _batch_for_ids(
        runtime,
        model,
        collator,
        panel_ids,
    )
    return _MaterializedSemanticT1Panel(
        batch=batch,
        panel_ids=panel_ids,
        fibers=fibers,
        partitions=partitions,
        entries=entries,
        index_by_panel_id={panel_id: index for index, panel_id in enumerate(panel_ids)},
    )


def _batch_from_materialized_panel(
    panel: _MaterializedSemanticT1Panel,
    panel_ids: Sequence[str],
) -> tuple[Any, tuple[Any, ...], tuple[Any, ...], tuple[Mapping[str, Any], ...]]:
    try:
        indices = tuple(panel.index_by_panel_id[panel_id] for panel_id in panel_ids)
    except KeyError as error:
        raise SemanticT1CapacityRunnerError(
            "T1 address stream references a panel entry outside the materialized panel"
        ) from error
    return (
        _index_factorized_batch(panel.batch, indices),
        tuple(panel.fibers[index] for index in indices),
        tuple(panel.partitions[index] for index in indices),
        tuple(panel.entries[index] for index in indices),
    )


def _metric_rows(
    panel: _MaterializedSemanticT1Panel,
    model: FactorizedTraceletRateModel,
    *,
    batch_size: int,
) -> list[dict[str, Any]]:
    model_was_training = model.training
    model.eval()
    rows: list[dict[str, Any]] = []
    ids = panel.panel_ids
    with torch.no_grad():
        for start in range(0, len(ids), batch_size):
            selected_ids = ids[start : start + batch_size]
            batch, _fibers, partitions, entries = _batch_from_materialized_panel(
                panel, selected_ids
            )
            prediction = forward_compiled_successor_partitions(
                model,
                batch.to(model.device),
                partitions,
            )
            for entry, keys, values in zip(
                entries,
                prediction.successor_keys,
                prediction.successor_log_probabilities,
                strict=True,
            ):
                target_index = keys.index(str(entry["successor_canonical_key"]))
                target_log_probability = float(values[target_index].detach().cpu())
                target_probability = math.exp(target_log_probability)
                maximum = float(values.max().detach().cpu())
                rank = 1 + int((values > values[target_index]).sum().detach().cpu())
                rows.append(
                    {
                        "panel_entry_sha256": entry["panel_entry_sha256"],
                        "model_family": entry["model_family"],
                        "capability_cell_id": entry["capability_cell_id"],
                        "teacher_successor_probability": target_probability,
                        "teacher_successor_nll": -target_log_probability,
                        "teacher_successor_rank": rank,
                        "teacher_successor_top1": target_log_probability >= maximum,
                        "canonical_successor_count": len(keys),
                    }
                )
    model.train(model_was_training)
    return rows


def summarize_semantic_t1_metrics(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if not rows:
        raise SemanticT1CapacityRunnerError("cannot summarize an empty T1 panel")

    def aggregate(selected: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        probabilities = [float(item["teacher_successor_probability"]) for item in selected]
        nlls = [float(item["teacher_successor_nll"]) for item in selected]
        top1 = [bool(item["teacher_successor_top1"]) for item in selected]
        return {
            "entry_count": len(selected),
            "mean_teacher_successor_probability": sum(probabilities) / len(probabilities),
            "minimum_teacher_successor_probability": min(probabilities),
            "mean_teacher_successor_nll": sum(nlls) / len(nlls),
            "maximum_teacher_successor_nll": max(nlls),
            "teacher_successor_top1_fraction": sum(top1) / len(top1),
            "every_entry_teacher_successor_top1": all(top1),
        }

    by_family: dict[str, Any] = {}
    by_cell: dict[str, Any] = {}
    for family in sorted({str(item["model_family"]) for item in rows}):
        selected = [item for item in rows if item["model_family"] == family]
        by_family[family] = aggregate(selected)
    for cell in sorted({str(item["capability_cell_id"]) for item in rows}):
        selected = [item for item in rows if item["capability_cell_id"] == cell]
        by_cell[cell] = aggregate(selected)
    overall = aggregate(rows)
    return {
        "overall": overall,
        "by_family": by_family,
        "by_nonempty_semantic_cell": by_cell,
        "per_entry": [dict(item) for item in rows],
    }


def semantic_t1_threshold_checks(
    metrics: Mapping[str, Any],
    *,
    thresholds: Mapping[str, Any],
    gradient_evidence: Mapping[str, Mapping[str, Any]],
) -> dict[str, bool]:
    families = metrics["by_family"]
    cells = metrics["by_nonempty_semantic_cell"]
    entries = metrics["per_entry"]
    return {
        "family_top1": all(
            value["teacher_successor_top1_fraction"]
            >= thresholds["minimum_unique_state_teacher_successor_top1"]
            for value in families.values()
        ),
        "family_probability": all(
            value["mean_teacher_successor_probability"]
            >= thresholds["minimum_unique_state_teacher_successor_probability"]
            for value in families.values()
        ),
        "family_nll": all(
            value["mean_teacher_successor_nll"]
            <= thresholds["maximum_unique_state_teacher_successor_nll"]
            for value in families.values()
        ),
        "cell_top1": all(
            value["teacher_successor_top1_fraction"]
            >= thresholds["minimum_nonempty_cell_teacher_successor_top1"]
            for value in cells.values()
        ),
        "cell_probability": all(
            value["mean_teacher_successor_probability"]
            >= thresholds["minimum_nonempty_cell_teacher_successor_probability"]
            for value in cells.values()
        ),
        "cell_nll": all(
            value["mean_teacher_successor_nll"]
            <= thresholds["maximum_nonempty_cell_teacher_successor_nll"]
            for value in cells.values()
        ),
        "every_entry_top1": (
            not thresholds["require_every_unique_entry_teacher_successor_top1"]
            or all(bool(entry["teacher_successor_top1"]) for entry in entries)
        ),
        "every_entry_probability": all(
            float(entry["teacher_successor_probability"])
            >= thresholds["minimum_every_unique_entry_teacher_successor_probability"]
            for entry in entries
        ),
        "family_route_gradient": all(
            bool(value["family_route_finite_nonzero_seen"]) for value in gradient_evidence.values()
        ),
        "action_route_gradient": all(
            bool(value["action_route_finite_nonzero_seen"]) for value in gradient_evidence.values()
        ),
    }


def semantic_t1_failing_families(
    metrics: Mapping[str, Any],
    *,
    thresholds: Mapping[str, Any],
    gradient_evidence: Mapping[str, Mapping[str, Any]],
) -> dict[str, tuple[str, ...]]:
    """Identify only the families that fail one or more frozen capacity gates."""

    entries = tuple(metrics["per_entry"])
    cells = metrics["by_nonempty_semantic_cell"]
    result: dict[str, tuple[str, ...]] = {}
    for family, family_metrics in sorted(metrics["by_family"].items()):
        reasons: set[str] = set()
        if (
            family_metrics["teacher_successor_top1_fraction"]
            < thresholds["minimum_unique_state_teacher_successor_top1"]
        ):
            reasons.add("family_top1")
        if (
            family_metrics["mean_teacher_successor_probability"]
            < thresholds["minimum_unique_state_teacher_successor_probability"]
        ):
            reasons.add("family_probability")
        if (
            family_metrics["mean_teacher_successor_nll"]
            > thresholds["maximum_unique_state_teacher_successor_nll"]
        ):
            reasons.add("family_nll")
        family_entries = [entry for entry in entries if entry["model_family"] == family]
        family_cells = {str(entry["capability_cell_id"]) for entry in family_entries}
        if any(
            cells[cell]["teacher_successor_top1_fraction"]
            < thresholds["minimum_nonempty_cell_teacher_successor_top1"]
            for cell in family_cells
        ):
            reasons.add("cell_top1")
        if any(
            cells[cell]["mean_teacher_successor_probability"]
            < thresholds["minimum_nonempty_cell_teacher_successor_probability"]
            for cell in family_cells
        ):
            reasons.add("cell_probability")
        if any(
            cells[cell]["mean_teacher_successor_nll"]
            > thresholds["maximum_nonempty_cell_teacher_successor_nll"]
            for cell in family_cells
        ):
            reasons.add("cell_nll")
        if thresholds["require_every_unique_entry_teacher_successor_top1"] and any(
            not bool(entry["teacher_successor_top1"]) for entry in family_entries
        ):
            reasons.add("every_entry_top1")
        if any(
            float(entry["teacher_successor_probability"])
            < thresholds["minimum_every_unique_entry_teacher_successor_probability"]
            for entry in family_entries
        ):
            reasons.add("every_entry_probability")
        family_gradient = gradient_evidence.get(family, {})
        if not bool(family_gradient.get("family_route_finite_nonzero_seen")):
            reasons.add("family_route_gradient")
        if not bool(family_gradient.get("action_route_finite_nonzero_seen")):
            reasons.add("action_route_gradient")
        if reasons:
            result[family] = tuple(sorted(reasons))
    return result


def build_semantic_t1_failure_diagnostics(
    *,
    capacity_policy: Mapping[str, Any],
    selected_metrics: Mapping[str, Any],
    gradient_evidence: Mapping[str, Mapping[str, Any]],
    result_sha256: str,
) -> dict[str, Any]:
    """Fail closed instead of silently skipping the frozen scope-order audit."""

    optimization = capacity_policy["optimization"]
    scopes = tuple(optimization["failure_diagnostic_scope_order"])
    if optimization["failure_diagnostics_only_for_failing_families"] is not True:
        raise SemanticT1CapacityRunnerError(
            "semantic T1 diagnostic policy no longer restricts work to failures"
        )
    failures = semantic_t1_failing_families(
        selected_metrics,
        thresholds=capacity_policy["thresholds"],
        gradient_evidence=gradient_evidence,
    )
    if not failures:
        raise SemanticT1CapacityRunnerError(
            "failure diagnostics were requested without a failing family"
        )
    body = {
        "schema": "compose.editing_v2.semantic_t1_failure_diagnostics",
        "schema_version": 1,
        "status": FAILURE_DIAGNOSTICS_STATUS,
        **NO_AUTHORITY,
        "capacity_policy_sha256": capacity_policy["policy_sha256"],
        "capacity_result_sha256": result_sha256,
        "failure_diagnostics_only_for_failing_families": True,
        "failure_diagnostic_scope_order": list(scopes),
        "failing_families": [
            {"family": family, "failed_gates": list(reasons)}
            for family, reasons in failures.items()
        ],
        "diagnostic_training_executed": False,
        "diagnostic_checkpoint_selected": False,
        "next_safe_action": (
            "implement and prospectively validate the frozen ordered diagnostic "
            "scopes for only these failing families before interpreting scope cause"
        ),
    }
    return {**body, "diagnostics_sha256": _sha(body)}


def _criterion(metrics: Mapping[str, Any], *, step: int) -> tuple[float, float, int]:
    rows = metrics["per_entry"]
    minimum_probability = min(float(item["teacher_successor_probability"]) for item in rows)
    mean_nll = sum(float(item["teacher_successor_nll"]) for item in rows) / len(rows)
    return (minimum_probability, -mean_nll, -step)


def _build_and_publish_capacity_result(
    *,
    output_root: Path,
    provenance: Mapping[str, Any],
    run_integrity: Mapping[str, Any],
    evaluation_trajectory: Sequence[Mapping[str, Any]],
    entry_metrics: Sequence[Mapping[str, Any]],
    gradient_evidence: Sequence[Mapping[str, Any]],
    result_builder: SemanticT1ResultBuilder | None = None,
    result_filename: str = RESULT_FILENAME,
) -> tuple[dict[str, Any], Path]:
    """Build and immutably publish either the legacy or an injected result."""

    if not isinstance(result_filename, str) or not result_filename:
        raise SemanticT1CapacityRunnerError(
            "semantic T1 result filename must be one nonempty relative filename"
        )
    filename = Path(result_filename)
    if filename.is_absolute() or filename.name != result_filename or result_filename in {".", ".."}:
        raise SemanticT1CapacityRunnerError(
            "semantic T1 result filename must be one nonempty relative filename"
        )
    builder = build_semantic_t1_capacity_result if result_builder is None else result_builder
    built = builder(
        provenance=provenance,
        run_integrity=run_integrity,
        evaluation_trajectory=evaluation_trajectory,
        entry_metrics=entry_metrics,
        gradient_evidence=gradient_evidence,
    )
    if not isinstance(built, Mapping):
        raise SemanticT1CapacityRunnerError("semantic T1 result builder returned a non-mapping")
    result = dict(built)
    result_path = Path(output_root) / result_filename
    write_bytes_if_absent(result_path, _canonical_bytes(result, newline=True))
    return result, result_path


def _route_gradient_evidence(
    model: FactorizedTraceletRateModel,
    prediction: Any,
    batch: Any,
    entries: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, float | bool]]:
    named = dict(model.named_parameters())
    family_parameters = tuple(
        parameter
        for name, parameter in named.items()
        if name.startswith(FAMILY_ROUTE_PREFIXES) and parameter.requires_grad
    )
    result: dict[str, dict[str, float | bool]] = {}
    log_probabilities = prediction.selected_productive_successor_log_probability
    for family in sorted({str(entry["model_family"]) for entry in entries}):
        indices = [index for index, entry in enumerate(entries) if entry["model_family"] == family]
        family_loss = -log_probabilities[indices].mean()
        action_parameters = tuple(
            parameter
            for name, parameter in named.items()
            if name.startswith(ACTION_ROUTE_PREFIXES[family]) and parameter.requires_grad
        )
        if not family_parameters or not action_parameters:
            raise SemanticT1CapacityRunnerError(
                f"T1 gradient route parameter surface is absent for {family}"
            )
        parameters = (*family_parameters, *action_parameters)
        gradients = torch.autograd.grad(
            family_loss,
            parameters,
            retain_graph=True,
            allow_unused=True,
        )
        family_gradients = gradients[: len(family_parameters)]
        action_gradients = gradients[len(family_parameters) :]

        def evidence(values: Sequence[Tensor | None]) -> tuple[bool, float]:
            finite = all(value is None or bool(torch.isfinite(value).all()) for value in values)
            norm = math.sqrt(
                sum(float(value.detach().norm()) ** 2 for value in values if value is not None)
            )
            return finite and norm > 0.0, norm

        family_seen, family_norm = evidence(family_gradients)
        action_seen, action_norm = evidence(action_gradients)
        result[family] = {
            "family_route_finite_nonzero_seen": family_seen,
            "family_route_gradient_norm": family_norm,
            "action_route_finite_nonzero_seen": action_seen,
            "action_route_gradient_norm": action_norm,
        }
    return result


def _merge_gradient_evidence(
    cumulative: dict[str, dict[str, Any]],
    observed: Mapping[str, Mapping[str, Any]],
) -> None:
    for family, evidence in observed.items():
        target = cumulative[family]
        target["sampled_optimizer_steps"] += 1
        for route in ("family_route", "action_route"):
            seen = bool(evidence[f"{route}_finite_nonzero_seen"])
            target[f"{route}_finite_nonzero_seen"] |= seen
            target[f"{route}_nonzero_steps"] += int(seen)
            target[f"{route}_cumulative_gradient_norm"] += float(evidence[f"{route}_gradient_norm"])


def _optimizer_configuration(
    model: FactorizedTraceletRateModel,
    optimizer: torch.optim.Optimizer,
) -> dict[str, Any]:
    """Return the exact serializable optimizer surface bound by recovery."""

    names_by_parameter_id = {id(parameter): name for name, parameter in model.named_parameters()}

    def normalized(value: object) -> object:
        if value is None or type(value) in (bool, int, float, str):
            return value
        if isinstance(value, (tuple, list)):
            return [normalized(item) for item in value]
        raise SemanticT1CapacityRunnerError(
            f"optimizer configuration contains unsupported metadata {type(value).__name__}"
        )

    groups: list[dict[str, Any]] = []
    for group in optimizer.param_groups:
        parameters = group.get("params")
        if not isinstance(parameters, list) or any(
            id(parameter) not in names_by_parameter_id for parameter in parameters
        ):
            raise SemanticT1CapacityRunnerError(
                "optimizer contains a parameter outside the named model surface"
            )
        groups.append(
            {
                "parameter_names": [
                    names_by_parameter_id[id(parameter)] for parameter in parameters
                ],
                "hyperparameters": {
                    str(name): normalized(value)
                    for name, value in sorted(group.items())
                    if name != "params"
                },
            }
        )
    return {
        "class": f"{type(optimizer).__module__}.{type(optimizer).__qualname__}",
        "defaults": {
            str(name): normalized(value) for name, value in sorted(optimizer.defaults.items())
        },
        "parameter_groups": groups,
    }


def _checkpoint_identity(
    runtime: SemanticT1RuntimeInputs,
    *,
    provenance: Mapping[str, Any],
    model: FactorizedTraceletRateModel,
    optimizer: torch.optim.Optimizer,
) -> dict[str, Any]:
    environment = provenance.get("execution_environment")
    if not isinstance(environment, Mapping):
        raise SemanticT1CapacityRunnerError(
            "T1 checkpoint provenance lacks an execution environment"
        )
    environment = dict(environment)
    environment_body = dict(environment)
    environment_sha256 = environment_body.pop("environment_sha256", None)
    source_revision = runtime.prepared.artifact.get("source_revision")
    runner_source_revision_sha256 = provenance.get("runner_source_revision_sha256")
    runner_implementation_sha256 = provenance.get("runner_implementation_sha256")
    expected_runner_implementation_sha256 = semantic_t1_runner_implementation_sha256(
        repo_root=Path(__file__).resolve().parents[3]
    )
    parameter = next(model.parameters())
    device = parameter.device
    optimization = dict(runtime.capacity_policy.get("optimization", {}))
    deterministic = torch.are_deterministic_algorithms_enabled()
    if (
        not isinstance(source_revision, Mapping)
        or runner_source_revision_sha256 != source_revision.get("source_revision_sha256")
        or runner_implementation_sha256 != expected_runner_implementation_sha256
        or environment_sha256 != _sha(environment_body)
        or environment.get("dtype") != str(parameter.dtype).removeprefix("torch.")
        or environment.get("accelerator_class") != ("gpu" if device.type == "cuda" else device.type)
        or optimization.get("deterministic_algorithms_required") is not True
        or deterministic is not True
    ):
        raise SemanticT1CapacityRunnerError(
            "T1 checkpoint implementation, source, environment, device, dtype, "
            "or determinism identity disagrees"
        )
    optimizer_configuration = _optimizer_configuration(model, optimizer)
    return {
        "capacity_policy_sha256": runtime.capacity_policy["policy_sha256"],
        "optimization_policy": optimization,
        "optimization_policy_sha256": _sha(optimization),
        "prepared_input_artifact_sha256": runtime.prepared.artifact["artifact_sha256"],
        "cache_completion_sha256": runtime.cache.completion["completion_sha256"],
        "cache_manifest_sha256": runtime.cache.manifest["manifest_sha256"],
        "initial_model_state_sha256": runtime.cache.completion["initial_model_state_sha256"],
        "runner_implementation_sha256": runner_implementation_sha256,
        "runner_source_revision_sha256": runner_source_revision_sha256,
        "execution_environment": environment,
        "execution_environment_sha256": environment_sha256,
        "model_device": str(device),
        "model_device_type": device.type,
        "model_device_index": device.index,
        "model_dtype": str(parameter.dtype),
        "deterministic_algorithms_enabled": deterministic,
        "optimizer_configuration": optimizer_configuration,
        "optimizer_configuration_sha256": _sha(optimizer_configuration),
    }


def _write_checkpoint(
    path: Path,
    *,
    runtime: SemanticT1RuntimeInputs,
    model: FactorizedTraceletRateModel,
    optimizer: torch.optim.Optimizer,
    completed_steps: int,
    selected_step: int,
    selected_criterion: tuple[float, float, int],
    selected_state: Mapping[str, Tensor],
    trajectory: Sequence[Mapping[str, Any]],
    gradient_evidence: Mapping[str, Mapping[str, Any]],
    stream_sha256: str,
    resume_count: int,
    checkpoint_identity: Mapping[str, Any],
) -> dict[str, Any]:
    model_state = _clone_state_dict(model)
    optimizer_state = optimizer.state_dict()
    payload = {
        "schema": CHECKPOINT_SCHEMA,
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "status": CHECKPOINT_STATUS,
        **NO_AUTHORITY,
        "identity": dict(checkpoint_identity),
        "completed_steps": completed_steps,
        "selected_step": selected_step,
        "selected_criterion": selected_criterion,
        "model_state": model_state,
        "model_state_sha256": state_dict_semantic_sha256(model_state),
        "optimizer_state": optimizer_state,
        "optimizer_state_sha256": optimizer_state_semantic_sha256(optimizer_state),
        "selected_model_state": dict(selected_state),
        "selected_model_state_sha256": state_dict_semantic_sha256(selected_state),
        "trajectory": [dict(item) for item in trajectory],
        "gradient_evidence": copy.deepcopy(dict(gradient_evidence)),
        "rng_state": _rng_state(),
        "stream_sha256": stream_sha256,
        "torch_version": str(torch.__version__),
        "resume_count": resume_count,
    }
    encoded = _torch_bytes(payload)
    _atomic_write(Path(path), encoded)
    return {
        "path": str(Path(path)),
        "file_sha256": hashlib.sha256(encoded).hexdigest(),
        "file_bytes": len(encoded),
        "completed_steps": completed_steps,
        "model_state_sha256": payload["model_state_sha256"],
        "optimizer_state_sha256": payload["optimizer_state_sha256"],
        "resume_count": resume_count,
    }


def _load_checkpoint(
    path: Path,
    *,
    runtime: SemanticT1RuntimeInputs,
    model: FactorizedTraceletRateModel,
    optimizer: torch.optim.Optimizer,
    expected_stream_sha256: str,
    expected_file_sha256: str,
    expected_checkpoint_identity: Mapping[str, Any],
) -> dict[str, Any]:
    if (
        not isinstance(expected_file_sha256, str)
        or len(expected_file_sha256) != 64
        or any(character not in "0123456789abcdef" for character in expected_file_sha256)
        or _file_sha(path) != expected_file_sha256
    ):
        raise SemanticT1CapacityRunnerError(
            "T1 checkpoint physical SHA-256 differs from the trusted receipt"
        )
    try:
        payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    except (OSError, RuntimeError, ValueError) as error:
        raise SemanticT1CapacityRunnerError(f"cannot load T1 checkpoint: {path}") from error
    if (
        not isinstance(payload, dict)
        or payload.get("schema") != CHECKPOINT_SCHEMA
        or payload.get("schema_version") != CHECKPOINT_SCHEMA_VERSION
        or payload.get("status") != CHECKPOINT_STATUS
        or payload.get("identity") != dict(expected_checkpoint_identity)
        or payload.get("stream_sha256") != expected_stream_sha256
        or payload.get("torch_version") != str(torch.__version__)
        or type(payload.get("resume_count")) is not int
        or payload["resume_count"] < 0
        or payload.get("model_state_sha256")
        != state_dict_semantic_sha256(payload.get("model_state", {}))
        or payload.get("selected_model_state_sha256")
        != state_dict_semantic_sha256(payload.get("selected_model_state", {}))
        or payload.get("optimizer_state_sha256")
        != optimizer_state_semantic_sha256(payload.get("optimizer_state", {}))
        or any(payload.get(name) is not expected for name, expected in NO_AUTHORITY.items())
    ):
        raise SemanticT1CapacityRunnerError("T1 checkpoint identity disagrees")
    model.load_state_dict(payload["model_state"], strict=True)
    optimizer.load_state_dict(payload["optimizer_state"])
    _restore_rng_state(payload["rng_state"])
    return payload


def run_semantic_t1_capacity(
    model: FactorizedTraceletRateModel,
    runtime: SemanticT1RuntimeInputs,
    *,
    output_directory: Path,
    provenance: Mapping[str, Any],
    resume_checkpoint_path: Path | None = None,
    resume_checkpoint_file_sha256: str | None = None,
    result_builder: SemanticT1ResultBuilder | None = None,
    result_filename: str = RESULT_FILENAME,
) -> dict[str, Any]:
    """Run or resume the frozen joint Active8 unique-state capacity test."""

    policy = runtime.capacity_policy
    optimization = policy["optimization"]
    thresholds = policy["thresholds"]
    if (
        not isinstance(model, FactorizedTraceletRateModel)
        or str(next(model.parameters()).dtype) != f"torch.{optimization['dtype']}"
        or state_dict_semantic_sha256(model.state_dict())
        != runtime.cache.completion["initial_model_state_sha256"]
        or tuple(policy["required_families"]) != tuple(ACTION_ROUTE_PREFIXES)
    ):
        raise SemanticT1CapacityRunnerError(
            "T1 model or Active8 identity differs from the frozen scratch state"
        )
    device = next(model.parameters()).device
    if device.type != "cuda" and optimization["accelerator_class"] == "gpu":
        raise SemanticT1CapacityRunnerError("semantic T1 policy requires a CUDA GPU")
    torch.use_deterministic_algorithms(True)
    seed = int(optimization["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    trainable: list[Tensor] = []
    hazard_initial = _hazard_state(model)
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(not name.startswith(TOTAL_HAZARD_PREFIX))
        if parameter.requires_grad:
            trainable.append(parameter)
    optimizer = torch.optim.AdamW(
        trainable,
        lr=float(optimization["learning_rate"]),
        weight_decay=float(optimization["weight_decay"]),
    )
    if any(
        parameter.requires_grad
        for name, parameter in model.named_parameters()
        if name.startswith(TOTAL_HAZARD_PREFIX)
    ):
        raise SemanticT1CapacityRunnerError("hazard entered the T1 optimizer scope")
    checkpoint_identity = _checkpoint_identity(
        runtime,
        provenance=provenance,
        model=model,
        optimizer=optimizer,
    )

    maximum_steps = int(optimization["maximum_optimizer_steps"])
    batch_size = int(optimization["batch_size"])
    stream = semantic_t1_address_stream(runtime, draw_count=maximum_steps * batch_size)
    stream_sha256 = _sha(list(stream))
    families = tuple(policy["required_families"])
    gradient_evidence: dict[str, dict[str, Any]] = {
        family: {
            "sampled_optimizer_steps": 0,
            "family_route_finite_nonzero_seen": False,
            "family_route_nonzero_steps": 0,
            "family_route_cumulative_gradient_norm": 0.0,
            "action_route_finite_nonzero_seen": False,
            "action_route_nonzero_steps": 0,
            "action_route_cumulative_gradient_norm": 0.0,
        }
        for family in families
    }
    completed_steps = 0
    trajectory: list[dict[str, Any]] = []
    selected_step = 0
    selected_state = _clone_state_dict(model)
    selected_criterion = (float("-inf"), float("-inf"), 0)
    collator = _collator(model)
    materialized_panel = _materialize_panel(runtime, model, collator)

    resume_count = 0
    if (resume_checkpoint_path is None) is not (resume_checkpoint_file_sha256 is None):
        raise SemanticT1CapacityRunnerError(
            "resume checkpoint path and trusted physical SHA-256 are both required"
        )
    if resume_checkpoint_path is not None:
        assert resume_checkpoint_file_sha256 is not None
        recovered = _load_checkpoint(
            resume_checkpoint_path,
            runtime=runtime,
            model=model,
            optimizer=optimizer,
            expected_stream_sha256=stream_sha256,
            expected_file_sha256=resume_checkpoint_file_sha256,
            expected_checkpoint_identity=checkpoint_identity,
        )
        completed_steps = int(recovered["completed_steps"])
        selected_step = int(recovered["selected_step"])
        selected_criterion = tuple(recovered["selected_criterion"])
        selected_state = recovered["selected_model_state"]
        trajectory = [dict(item) for item in recovered["trajectory"]]
        gradient_evidence = copy.deepcopy(recovered["gradient_evidence"])
        resume_count = int(recovered["resume_count"]) + 1
        if not 0 <= completed_steps <= maximum_steps:
            raise SemanticT1CapacityRunnerError("checkpoint step is out of range")

    output_root = Path(output_directory)
    checkpoint_receipts: list[dict[str, Any]] = []
    report_points = {int(value) for value in optimization["report_points"]}
    try:
        evaluation_schedule = semantic_t1_evaluation_schedule(optimization)
    except SemanticT1CapacityPolicyError as error:
        raise SemanticT1CapacityRunnerError(
            f"semantic T1 evaluation policy is invalid: {error}"
        ) from error
    evaluation_steps = frozenset(evaluation_schedule)
    stop_reason = "MAXIMUM_OPTIMIZER_STEPS_REACHED"
    final_checks: dict[str, bool] | None = None

    model.train()
    while completed_steps <= maximum_steps:
        if completed_steps in evaluation_steps:
            rows = _metric_rows(
                materialized_panel,
                model,
                batch_size=batch_size,
            )
            metrics = summarize_semantic_t1_metrics(rows)
            checks = semantic_t1_threshold_checks(
                metrics,
                thresholds=thresholds,
                gradient_evidence=gradient_evidence,
            )
            criterion = _criterion(metrics, step=completed_steps)
            if criterion > selected_criterion:
                selected_criterion = criterion
                selected_step = completed_steps
                selected_state = _clone_state_dict(model)
            trajectory.append(
                {
                    "step": completed_steps,
                    "minimum_entry_teacher_successor_probability": criterion[0],
                    "mean_teacher_successor_nll": -criterion[1],
                    "all_threshold_checks_pass": all(checks.values()),
                    "threshold_checks": checks,
                    "model_state_sha256": state_dict_semantic_sha256(model.state_dict()),
                }
            )
            final_checks = checks
            try:
                threshold_stop_allowed = semantic_t1_threshold_stop_allowed(
                    optimizer_step=completed_steps,
                    threshold_checks=checks,
                )
            except SemanticT1CapacityPolicyError as error:
                raise SemanticT1CapacityRunnerError(
                    f"semantic T1 threshold-stop state is invalid: {error}"
                ) from error
            if threshold_stop_allowed:
                stop_reason = "ALL_FROZEN_THRESHOLDS_PASS"
                break
        if completed_steps == maximum_steps:
            break

        start = completed_steps * batch_size
        selected_ids = stream[start : start + batch_size]
        batch, fibers, _partitions, entries = _batch_from_materialized_panel(
            materialized_panel, selected_ids
        )
        device_batch = batch.to(model.device)
        optimizer.zero_grad(set_to_none=True)
        prediction = forward_teacher_successor_batch(
            model,
            device_batch,
            fibers,
        )
        loss = factorized_successor_identity_loss(prediction, device_batch)
        if not bool(torch.isfinite(loss)):
            raise SemanticT1CapacityRunnerError(
                f"nonfinite T1 loss before optimizer step {completed_steps + 1}"
            )
        observed_routes = _route_gradient_evidence(model, prediction, device_batch, entries)
        if any(
            not evidence["family_route_finite_nonzero_seen"]
            or not evidence["action_route_finite_nonzero_seen"]
            for evidence in observed_routes.values()
        ):
            failed = sorted(
                family
                for family, evidence in observed_routes.items()
                if not evidence["family_route_finite_nonzero_seen"]
                or not evidence["action_route_finite_nonzero_seen"]
            )
            raise SemanticT1CapacityRunnerError(
                f"zero or nonfinite T1 family/action route gradient: {failed}"
            )
        _merge_gradient_evidence(gradient_evidence, observed_routes)
        loss.backward()
        for name, parameter in model.named_parameters():
            if name.startswith(TOTAL_HAZARD_PREFIX):
                if parameter.grad is not None:
                    raise SemanticT1CapacityRunnerError(
                        f"frozen hazard parameter received a gradient: {name}"
                    )
            elif parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all()):
                raise SemanticT1CapacityRunnerError(
                    f"nonfinite gradient before T1 step {completed_steps + 1}: {name}"
                )
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            trainable, float(optimization["gradient_clip_norm"])
        )
        if not bool(torch.isfinite(gradient_norm)):
            raise SemanticT1CapacityRunnerError("T1 gradient norm is nonfinite")
        optimizer.step()
        completed_steps += 1
        _assert_hazard_identity(model, hazard_initial)

        if completed_steps in report_points:
            checkpoint_receipts.append(
                _write_checkpoint(
                    output_root / "checkpoints" / f"step_{completed_steps:04d}.pt",
                    runtime=runtime,
                    model=model,
                    optimizer=optimizer,
                    completed_steps=completed_steps,
                    selected_step=selected_step,
                    selected_criterion=selected_criterion,
                    selected_state=selected_state,
                    trajectory=trajectory,
                    gradient_evidence=gradient_evidence,
                    stream_sha256=stream_sha256,
                    resume_count=resume_count,
                    checkpoint_identity=checkpoint_identity,
                )
            )

    assert final_checks is not None
    terminal_state = _clone_state_dict(model)
    terminal_state_sha256 = state_dict_semantic_sha256(terminal_state)
    model.load_state_dict(selected_state, strict=True)
    selected_state_sha256 = state_dict_semantic_sha256(model.state_dict())
    selected_rows = _metric_rows(materialized_panel, model, batch_size=batch_size)
    selected_metrics = summarize_semantic_t1_metrics(selected_rows)
    selected_checks = semantic_t1_threshold_checks(
        selected_metrics,
        thresholds=thresholds,
        gradient_evidence=gradient_evidence,
    )
    _assert_hazard_identity(model, hazard_initial)
    selected_checkpoint = {
        "schema": CHECKPOINT_SCHEMA,
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "status": CHECKPOINT_STATUS,
        **NO_AUTHORITY,
        "identity": checkpoint_identity,
        "selected_step": selected_step,
        "model_state": _clone_state_dict(model),
        "model_state_sha256": selected_state_sha256,
        "stream_sha256": stream_sha256,
        "torch_version": str(torch.__version__),
    }
    selected_bytes = _torch_bytes(selected_checkpoint)
    selected_path = output_root / SELECTED_CHECKPOINT_FILENAME
    _atomic_write(selected_path, selected_bytes)

    trajectory_for_result = [
        {
            "step": int(point["step"]),
            "minimum_entry_teacher_successor_probability": point[
                "minimum_entry_teacher_successor_probability"
            ],
            "mean_entry_canonical_successor_nll": point["mean_teacher_successor_nll"],
            "model_state_sha256": point["model_state_sha256"],
        }
        for point in trajectory
    ]
    family_order = tuple(policy["required_families"])
    entry_metrics = sorted(
        (
            {
                "panel_entry_sha256": row["panel_entry_sha256"],
                "family": row["model_family"],
                "semantic_cell_id": row["capability_cell_id"],
                "teacher_successor_probability": row["teacher_successor_probability"],
                "canonical_successor_nll": row["teacher_successor_nll"],
                "teacher_successor_rank": row["teacher_successor_rank"],
                "teacher_successor_top1": row["teacher_successor_top1"],
            }
            for row in selected_metrics["per_entry"]
        ),
        key=lambda row: (
            family_order.index(row["family"]),
            row["semantic_cell_id"],
            row["panel_entry_sha256"],
        ),
    )
    gradient_rows = [
        {
            "family": family,
            "family_gate_gradient_finite": True,
            "family_gate_cumulative_l2": gradient_evidence[family][
                "family_route_cumulative_gradient_norm"
            ],
            "family_gate_nonzero_update_steps": gradient_evidence[family][
                "family_route_nonzero_steps"
            ],
            "action_route_gradient_finite": True,
            "action_route_cumulative_l2": gradient_evidence[family][
                "action_route_cumulative_gradient_norm"
            ],
            "action_route_nonzero_update_steps": gradient_evidence[family][
                "action_route_nonzero_steps"
            ],
        }
        for family in family_order
    ]
    run_integrity = {
        "optimizer_steps_completed": completed_steps,
        "evaluation_steps": [int(point["step"]) for point in trajectory],
        "selected_step": selected_step,
        "termination_reason": (
            "all_thresholds_passed_early"
            if stop_reason == "ALL_FROZEN_THRESHOLDS_PASS"
            else "maximum_optimizer_steps_reached"
        ),
        "resume_requested": resume_count > 0,
        "resume_count": resume_count,
        "abort_triggered": False,
        "abort_reasons": [],
        "nonfinite_event_count": 0,
        "unsupported_teacher_count": 0,
        "missing_candidate_count": 0,
        "provenance_drift_detected": False,
        "stream_identity_drift_detected": False,
        "deterministic_algorithms_enabled": (torch.are_deterministic_algorithms_enabled()),
        "dtype": "float32",
        "mixed_precision": False,
        "hazard_included": False,
        "address_stream_sha256": stream_sha256,
        "selected_model_state_sha256": selected_state_sha256,
        "selected_checkpoint_file_sha256": hashlib.sha256(selected_bytes).hexdigest(),
        "optimizer_state_sha256": optimizer_state_semantic_sha256(optimizer.state_dict()),
    }
    result, result_path = _build_and_publish_capacity_result(
        output_root=output_root,
        provenance=provenance,
        run_integrity=run_integrity,
        evaluation_trajectory=trajectory_for_result,
        entry_metrics=entry_metrics,
        gradient_evidence=gradient_rows,
        result_builder=result_builder,
        result_filename=result_filename,
    )
    failure_diagnostics_path: Path | None = None
    failure_diagnostics: dict[str, Any] | None = None
    if not all(selected_checks.values()):
        failure_diagnostics = build_semantic_t1_failure_diagnostics(
            capacity_policy=policy,
            selected_metrics=selected_metrics,
            gradient_evidence=gradient_evidence,
            result_sha256=result["result_sha256"],
        )
        failure_diagnostics_path = output_root / FAILURE_DIAGNOSTICS_FILENAME
        write_bytes_if_absent(
            failure_diagnostics_path,
            _canonical_bytes(failure_diagnostics, newline=True),
        )
    return {
        "result": result,
        "result_path": result_path,
        "selected_checkpoint_path": selected_path,
        "selected_checkpoint_file_sha256": hashlib.sha256(selected_bytes).hexdigest(),
        "terminal_model_state_sha256": terminal_state_sha256,
        "selected_threshold_checks": selected_checks,
        "terminal_threshold_checks": final_checks,
        "recovery_checkpoints": checkpoint_receipts,
        "hazard_initial_state_sha256": state_dict_semantic_sha256(hazard_initial),
        "failure_diagnostics": failure_diagnostics,
        "failure_diagnostics_path": failure_diagnostics_path,
    }


def load_capacity_policy_for_runner(path: Path) -> dict[str, Any]:
    """Named runner boundary around the frozen policy validator."""

    return load_semantic_t1_capacity_policy(path)


__all__ = [
    "FAILURE_DIAGNOSTICS_FILENAME",
    "RESULT_FILENAME",
    "SELECTED_CHECKPOINT_FILENAME",
    "SemanticT1CapacityRunnerError",
    "SemanticT1ResultBuilder",
    "SemanticT1RuntimeInputs",
    "build_semantic_t1_failure_diagnostics",
    "load_capacity_policy_for_runner",
    "optimizer_state_semantic_sha256",
    "run_semantic_t1_capacity",
    "semantic_t1_address_stream",
    "semantic_t1_evaluation_schedule",
    "semantic_t1_failing_families",
    "semantic_t1_runner_implementation_sha256",
    "semantic_t1_threshold_stop_allowed",
    "semantic_t1_threshold_checks",
    "summarize_semantic_t1_metrics",
]
