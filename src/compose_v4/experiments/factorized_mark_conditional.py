"""Deterministic, prefetched training for the dense marked-rewrite generator."""

from __future__ import annotations

import copy
import json
from collections import OrderedDict
from contextlib import nullcontext
from dataclasses import dataclass, field, replace
from math import cos, exp, pi
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Iterator, Mapping

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset

from compose_v4.chem.molecular_graph import MolecularGraph
from compose_v4.experiments import zero_mixture_instrumentation as _zmi
from compose_v4.experiments.cnof_conditional import PathRecord
from compose_v4.experiments.training_support_cache import (
    ShardedTrainingSupportCache,
)
from compose_v4.experiments.tracelet_conditional import _sample_tracelet_progress
from compose_v4.model.factorized_tracelet_rate_model import (
    _CYCLE_OP_EXECUTOR_TO_FAMILY,
    ChemistryStateFeatures,
    FactorizedMarkBatch,
    FactorizedTraceletRateModel,
    MARK_RULE_NAMES,
    MARK_RULE_TO_INDEX,
    RingTeacherSemanticCertificate,
    SparseBinaryRows,
    factorized_mark_bregman_loss,
    prepare_factorized_mark_batch,
)
from compose_v4.rewrite.ring_system_fiber import (
    clear_semantic_ring_state_caches,
    warm_ring_system_candidate_indices,
)
from compose_v4.rewrite.tracelets import RingSystemGrow
from compose_v4.rewrite.typed_ring_catalog import TypedRingCatalog


@dataclass(frozen=True)
class FactorizedMarkExample:
    state: MolecularGraph
    time: float
    teacher_action: Any | None
    teacher_rule_name: str | None
    teacher_rate: float
    importance_weight: float
    property_condition_values: tuple[float, ...] | None = None
    property_condition_mask: tuple[bool, ...] | None = None
    ring_grow_support_indices: tuple[int, ...] | None = None
    ring_grow_support_width: int = 0
    ring_grow_support_is_exact: bool = False
    ring_grow_enablement_is_exact: bool = False
    ring_topology_local_support_log_mass: float | None = None
    ring_teacher_semantic_certificate: RingTeacherSemanticCertificate | None = None

    @property
    def ring_grow_support_mask(self) -> tuple[bool, ...] | None:
        """Dense compatibility view; transport and caches use sparse indices."""

        if self.ring_grow_support_indices is None:
            return None
        selected = set(self.ring_grow_support_indices)
        return tuple(index in selected for index in range(self.ring_grow_support_width))


class FactorizedMarkDataset(Dataset[FactorizedMarkExample]):
    """Index-deterministic examples safe under arbitrary worker prefetching.

    Example ``i`` is a pure function of ``(seed, i)``.  A resumed run starts at
    ``completed_steps * batch_size`` and therefore sees exactly the same future
    training stream without serializing worker RNG or prefetch queues.
    """

    def __init__(
        self,
        records: tuple[PathRecord, ...],
        *,
        start_index: int,
        length: int,
        seed: int,
        late_time_fraction: float,
        operational_horizon: float,
        progress_stratification_fraction: float,
        ring_catalog: TypedRingCatalog | None = None,
        ring_electronic_mode: str = "factorized_local",
        support_cache_limit: int = 2048,
        support_cache_reset_interval: int = 2048,
        training_support_cache: ShardedTrainingSupportCache | None = None,
        require_cached_support: bool = False,
        target_property_conditions: Mapping[str, tuple[float, ...]] | None = None,
        condition_dropout_probability: float = 0.0,
        ring_family_mass_mode: str = "boolean",
        record_index_sampler: object | None = None,
    ) -> None:
        if not records:
            raise ValueError("factorized mark training records must be non-empty")
        if start_index < 0 or length < 0:
            raise ValueError("dataset indices and length must be non-negative")
        if not 0.0 <= late_time_fraction <= 1.0:
            raise ValueError("late-time fraction must lie in [0, 1]")
        if operational_horizon <= 0.0:
            raise ValueError("operational horizon must be positive")
        if not 0.0 <= progress_stratification_fraction <= 1.0:
            raise ValueError("progress stratification must lie in [0, 1]")
        if support_cache_limit <= 0 or support_cache_reset_interval <= 0:
            raise ValueError("support cache limits and reset intervals must be positive")
        if training_support_cache is not None and ring_catalog is None:
            raise ValueError("training support cache requires a ring catalog")
        if require_cached_support and training_support_cache is None:
            raise ValueError("required training support cache was not provided")
        if not 0.0 <= condition_dropout_probability <= 1.0:
            raise ValueError("condition dropout probability must lie in [0, 1]")
        if ring_family_mass_mode not in {
            "boolean",
            "catalog_topology_local_support",
        }:
            raise ValueError("unknown ring family mass mode")
        condition_widths = (
            set()
            if target_property_conditions is None
            else {len(values) for values in target_property_conditions.values()}
        )
        if target_property_conditions is not None and (
            condition_widths != {next(iter(condition_widths), 0)}
            or next(iter(condition_widths), 0) <= 0
        ):
            raise ValueError("target property conditions must share one positive width")
        if (
            training_support_cache is not None
            and start_index + length > training_support_cache.total_rows
        ):
            raise ValueError("training dataset extends beyond its support cache")
        self.records = records
        self.start_index = int(start_index)
        self.length = int(length)
        self.seed = int(seed)
        self.late_time_fraction = float(late_time_fraction)
        self.operational_horizon = float(operational_horizon)
        self.progress_stratification_fraction = float(progress_stratification_fraction)
        self.ring_catalog = ring_catalog
        self.ring_electronic_mode = str(ring_electronic_mode)
        self.support_cache_limit = int(support_cache_limit)
        self.support_cache_reset_interval = int(support_cache_reset_interval)
        self.training_support_cache = training_support_cache
        self.require_cached_support = bool(require_cached_support)
        self.target_property_conditions = target_property_conditions
        self.condition_dropout_probability = float(condition_dropout_probability)
        self.ring_family_mass_mode = str(ring_family_mass_mode)
        # Optional hierarchical record sampler (layer -> curriculum bin -> example). None -> uniform draw
        # over records, byte-identical to the de-novo path. Must expose a stateless ``draw(rng) -> int``.
        if record_index_sampler is not None and len(getattr(record_index_sampler, "tags", records)) \
                != len(records):
            raise ValueError("record_index_sampler tags must align 1:1 with the training records")
        self.record_index_sampler = record_index_sampler
        self._ring_support_model: FactorizedTraceletRateModel | None = None
        self._ring_support_examples_since_reset = 0

    def _ring_chemistry_model(self) -> FactorizedTraceletRateModel:
        if (
            self._ring_support_model is not None
            and self._ring_support_examples_since_reset >= self.support_cache_reset_interval
        ):
            self._ring_support_model.clear_ring_candidate_caches()
            clear_semantic_ring_state_caches()
            self._ring_support_examples_since_reset = 0
        if self._ring_support_model is None:
            # This is a chemistry oracle only. Its parameters are never read,
            # and the forked RNG prevents worker initialization from changing
            # the deterministic training stream.
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(0)
                self._ring_support_model = FactorizedTraceletRateModel(
                    self.ring_catalog,
                    hidden_dim=1,
                    message_passing_steps=1,
                    mark_dim=1,
                    ring_electronic_mode=self.ring_electronic_mode,
                    ring_candidate_cache_limit=self.support_cache_limit,
                )
        return self._ring_support_model

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> FactorizedMarkExample:
        if not 0 <= index < self.length:
            raise IndexError(index)
        absolute_index = self.start_index + int(index)
        rng = np.random.default_rng(np.random.SeedSequence((self.seed, absolute_index)))
        if self.record_index_sampler is not None:
            record = self.records[int(self.record_index_sampler.draw(rng))]
        else:
            record = self.records[int(rng.integers(len(self.records)))]
        if rng.random() < self.late_time_fraction:
            operational_time = float(rng.uniform(0.0, self.operational_horizon))
            time = 1.0 - exp(-operational_time)
        else:
            time = float(rng.uniform(0.01, 0.99))
        progress, importance_weight = _sample_tracelet_progress(
            record.path,
            time=time,
            rng=rng,
            stratification_fraction=self.progress_stratification_fraction,
        )
        if progress < record.path.path_length:
            step = record.path.trace.steps[progress]
            # cycle ops record EXECUTOR names (bond_insert/bond_delete) that the model scores under the
            # cycle_insert/cycle_attach slots; accept them (the rule_name stays the executor name, which
            # _teacher_action_score dispatches on). Everything else must be a dense family.
            if (
                step.rule_name not in MARK_RULE_TO_INDEX
                and step.rule_name not in _CYCLE_OP_EXECUTOR_TO_FAMILY
            ):
                raise ValueError(
                    f"compiled teacher uses unsupported dense family: {step.rule_name}"
                )
            teacher_action = step.action
            teacher_rule_name = step.rule_name
            teacher_rate = record.path.operational_jump_rate(progress)
        else:
            teacher_action = None
            teacher_rule_name = None
            teacher_rate = 0.0
        state = record.path.state_at(progress)
        property_condition_values = None
        property_condition_mask = None
        if self.target_property_conditions is not None:
            raw_values = self.target_property_conditions.get(record.target_key)
            if raw_values is None:
                raise KeyError(f"missing target property condition: {record.target_key}")
            property_condition_values = tuple(float(value) for value in raw_values)
            if not np.isfinite(property_condition_values).all():
                raise ValueError("target property condition contains a non-finite value")
            dropout_rng = np.random.default_rng(
                np.random.SeedSequence((self.seed, absolute_index, 0xC0D17))
            )
            observed = bool(
                dropout_rng.random() >= self.condition_dropout_probability
            )
            property_condition_mask = (observed,) * len(property_condition_values)
        ring_grow_support_mask = None
        ring_grow_support_indices = None
        ring_grow_support_width = 0
        ring_grow_support_is_exact = False
        ring_grow_enablement_is_exact = False
        ring_topology_local_support_log_mass = None
        ring_teacher_semantic_certificate = None
        chemistry_model = None
        if self.ring_catalog is not None:
            cached_support = (
                None
                if self.training_support_cache is None
                else (
                    self.training_support_cache.require(absolute_index)
                    if self.require_cached_support
                    else self.training_support_cache.get(absolute_index)
                )
            )
            if cached_support is not None:
                ring_grow_support_indices = cached_support.indices
                ring_grow_support_width = int(cached_support.width)
                ring_grow_support_is_exact = bool(cached_support.support_is_exact)
                ring_grow_enablement_is_exact = bool(cached_support.enablement_is_exact)
                ring_teacher_semantic_certificate = cached_support.teacher_semantic_certificate
            else:
                chemistry_model = self._ring_chemistry_model()
                if teacher_rule_name == "ring_system_grow":
                    ring_grow_support_mask = chemistry_model._ring_grow_support(state)
                    ring_grow_support_is_exact = True
                else:
                    ring_grow_support_mask = chemistry_model._ring_grow_enablement_certificate(
                        state
                    )
                ring_grow_enablement_is_exact = True
                ring_grow_support_indices = tuple(
                    int(index)
                    for index, supported in enumerate(ring_grow_support_mask)
                    if supported
                )
                ring_grow_support_width = len(ring_grow_support_mask)
            if (
                teacher_rule_name == "ring_system_grow"
                and ring_teacher_semantic_certificate is None
            ):
                if not isinstance(teacher_action, RingSystemGrow):
                    raise TypeError("ring-system teacher has the wrong action type")
                if cached_support is not None and self.require_cached_support:
                    raise RuntimeError(
                        "required training support row lacks its exact semantic teacher certificate"
                    )
                if chemistry_model is None:
                    chemistry_model = self._ring_chemistry_model()
                ring_teacher_semantic_certificate = (
                    chemistry_model.ring_teacher_semantic_certificate(
                        state,
                        teacher_action,
                    )
                )
            if chemistry_model is not None:
                self._ring_support_examples_since_reset += 1
        if self.ring_family_mass_mode == "catalog_topology_local_support":
            if self.ring_catalog is None:
                raise RuntimeError("ring topology mass requires a ring catalog")
            if chemistry_model is None:
                chemistry_model = self._ring_chemistry_model()
                self._ring_support_examples_since_reset += 1
            ring_topology_local_support_log_mass = (
                chemistry_model._ring_topology_local_support_log_mass(state)
            )
        return FactorizedMarkExample(
            state=state,
            time=time,
            teacher_action=teacher_action,
            teacher_rule_name=teacher_rule_name,
            teacher_rate=teacher_rate,
            importance_weight=importance_weight,
            property_condition_values=property_condition_values,
            property_condition_mask=property_condition_mask,
            ring_grow_support_indices=ring_grow_support_indices,
            ring_grow_support_width=ring_grow_support_width,
            ring_grow_support_is_exact=ring_grow_support_is_exact,
            ring_grow_enablement_is_exact=ring_grow_enablement_is_exact,
            ring_topology_local_support_log_mass=(
                ring_topology_local_support_log_mass
            ),
            ring_teacher_semantic_certificate=ring_teacher_semantic_certificate,
        )


@dataclass(frozen=True)
class FactorizedMarkCollator:
    use_aromatic_bond_view: bool
    ring_catalog: TypedRingCatalog | None = None
    chemistry_feature_cache_limit: int = 2048
    compute_ring_grow_support: bool = True
    compute_ring_restates: bool = False
    compute_cyclic_graft: bool = False
    compute_ring_opening: bool = False
    _chemistry_feature_cache: OrderedDict[
        tuple[int, bool, bool, tuple[bytes, bytes, bytes, bytes]],
        ChemistryStateFeatures,
    ] = field(default_factory=OrderedDict, init=False, repr=False, compare=False)

    def __call__(self, examples: list[FactorizedMarkExample]) -> FactorizedMarkBatch:
        support_rows = tuple(example.ring_grow_support_indices for example in examples)
        topology_masses = tuple(
            example.ring_topology_local_support_log_mass for example in examples
        )
        if any(value is None for value in topology_masses) and not all(
            value is None for value in topology_masses
        ):
            raise ValueError("factorized examples mix topology masses and missing values")
        has_precomputed_ring_support = bool(support_rows) and all(
            row is not None for row in support_rows
        )
        condition_values = tuple(
            example.property_condition_values for example in examples
        )
        condition_masks = tuple(example.property_condition_mask for example in examples)
        has_conditions = all(values is not None for values in condition_values) and all(
            mask is not None for mask in condition_masks
        )
        if not has_conditions and any(
            values is not None or mask is not None
            for values, mask in zip(condition_values, condition_masks)
        ):
            raise ValueError("property-conditioned examples are only partially populated")
        batch = prepare_factorized_mark_batch(
            tuple(example.state for example in examples),
            tuple(example.time for example in examples),
            tuple(example.teacher_action for example in examples),
            tuple(example.teacher_rule_name for example in examples),
            tuple(example.teacher_rate for example in examples),
            tuple(example.importance_weight for example in examples),
            use_aromatic_bond_view=self.use_aromatic_bond_view,
            ring_catalog=self.ring_catalog,
            chemistry_feature_cache=self._chemistry_feature_cache,
            chemistry_feature_cache_limit=self.chemistry_feature_cache_limit,
            compute_ring_grow_support=self.compute_ring_grow_support
            and not has_precomputed_ring_support,
            compute_ring_restates=self.compute_ring_restates,
            compute_cyclic_graft=self.compute_cyclic_graft,
            compute_ring_opening=self.compute_ring_opening,
            property_condition_values=(
                tuple(values for values in condition_values if values is not None)
                if has_conditions
                else None
            ),
            property_condition_mask=(
                tuple(mask for mask in condition_masks if mask is not None)
                if has_conditions
                else None
            ),
        )
        if has_precomputed_ring_support:
            widths = {example.ring_grow_support_width for example in examples}
            if len(widths) != 1:
                raise ValueError("precomputed ring support rows have inconsistent widths")
            batch = replace(
                batch,
                ring_grow_support_mask=None,
                ring_grow_support_sparse=SparseBinaryRows.from_index_rows(
                    tuple(row for row in support_rows if row is not None),
                    width=widths.pop(),
                ),
                ring_grow_support_is_exact=torch.tensor(
                    tuple(example.ring_grow_support_is_exact for example in examples),
                    dtype=torch.bool,
                ),
                ring_grow_enablement_is_exact=torch.tensor(
                    tuple(example.ring_grow_enablement_is_exact for example in examples),
                    dtype=torch.bool,
                ),
            )
        batch = replace(
            batch,
            ring_teacher_semantic_certificates=tuple(
                example.ring_teacher_semantic_certificate for example in examples
            ),
            ring_topology_local_support_log_mass=(
                None
                if all(value is None for value in topology_masses)
                else torch.tensor(
                    tuple(
                        float(value) for value in topology_masses if value is not None
                    ),
                    dtype=batch.times.dtype,
                )
            ),
        )
        return batch


def factorized_mark_loader(
    records: tuple[PathRecord, ...],
    *,
    steps: int,
    batch_size: int,
    start_step: int,
    seed: int,
    workers: int,
    late_time_fraction: float,
    operational_horizon: float,
    progress_stratification_fraction: float,
    use_aromatic_bond_view: bool,
    pin_memory: bool,
    ring_catalog: TypedRingCatalog | None = None,
    prefetch_factor: int = 2,
    ring_electronic_mode: str = "factorized_local",
    training_support_cache: ShardedTrainingSupportCache | None = None,
    require_cached_support: bool = False,
    target_property_conditions: Mapping[str, tuple[float, ...]] | None = None,
    condition_dropout_probability: float = 0.0,
    ring_family_mass_mode: str = "boolean",
    compute_ring_grow_support: bool = True,
    compute_ring_restates: bool = False,
    compute_cyclic_graft: bool = False,
    compute_ring_opening: bool = False,
    record_index_sampler: object | None = None,
) -> DataLoader[FactorizedMarkBatch]:
    if not 0 <= start_step <= steps:
        raise ValueError("start step lies outside the training horizon")
    if prefetch_factor <= 0:
        raise ValueError("prefetch factor must be positive")
    remaining_steps = steps - start_step
    if ring_catalog is not None:
        warm_ring_system_candidate_indices(ring_catalog)
    _zmi.bump("edit_dataset_constructions")
    _zmi.bump("edit_dataloader_constructions")
    dataset = FactorizedMarkDataset(
        records,
        start_index=start_step * batch_size,
        length=remaining_steps * batch_size,
        seed=seed,
        late_time_fraction=late_time_fraction,
        operational_horizon=operational_horizon,
        progress_stratification_fraction=progress_stratification_fraction,
        ring_catalog=ring_catalog,
        ring_electronic_mode=ring_electronic_mode,
        training_support_cache=training_support_cache,
        require_cached_support=require_cached_support,
        target_property_conditions=target_property_conditions,
        condition_dropout_probability=condition_dropout_probability,
        ring_family_mass_mode=ring_family_mass_mode,
        record_index_sampler=record_index_sampler,
    )
    options: dict[str, Any] = {}
    if workers > 0:
        options.update(
            persistent_workers=True,
            prefetch_factor=prefetch_factor,
        )
    loader_generator = torch.Generator(device="cpu")
    loader_generator.manual_seed(int(seed))
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        collate_fn=FactorizedMarkCollator(
            use_aromatic_bond_view,
            ring_catalog,
            compute_ring_grow_support=compute_ring_grow_support,
            compute_ring_restates=compute_ring_restates,
            compute_cyclic_graft=compute_cyclic_graft,
            compute_ring_opening=compute_ring_opening,
        ),
        pin_memory=pin_memory,
        drop_last=True,
        generator=loader_generator,
        **options,
    )


def sample_factorized_mark_batch(
    records: tuple[PathRecord, ...],
    *,
    batch_size: int,
    seed: int,
    late_time_fraction: float,
    operational_horizon: float,
    progress_stratification_fraction: float = 0.0,
    use_aromatic_bond_view: bool = True,
    workers: int = 0,
    ring_catalog: TypedRingCatalog | None = None,
    ring_electronic_mode: str = "factorized_local",
    target_property_conditions: Mapping[str, tuple[float, ...]] | None = None,
    condition_dropout_probability: float = 0.0,
    ring_family_mass_mode: str = "boolean",
) -> FactorizedMarkBatch:
    if workers < 0:
        raise ValueError("evaluation workers must be non-negative")
    if ring_catalog is not None:
        warm_ring_system_candidate_indices(ring_catalog)
    dataset = FactorizedMarkDataset(
        records,
        start_index=0,
        length=batch_size,
        seed=seed,
        late_time_fraction=late_time_fraction,
        operational_horizon=operational_horizon,
        progress_stratification_fraction=progress_stratification_fraction,
        ring_catalog=ring_catalog,
        ring_electronic_mode=ring_electronic_mode,
        target_property_conditions=target_property_conditions,
        condition_dropout_probability=condition_dropout_probability,
        ring_family_mass_mode=ring_family_mass_mode,
    )
    collator = FactorizedMarkCollator(use_aromatic_bond_view, ring_catalog)
    if workers == 0:
        return collator([dataset[index] for index in range(batch_size)])
    loader = DataLoader(
        dataset,
        batch_size=min(batch_size, 32),
        shuffle=False,
        num_workers=workers,
        collate_fn=collator,
        pin_memory=False,
        drop_last=False,
    )
    return _concatenate_factorized_mark_batches(tuple(loader))


def attach_property_conditions(
    batch: FactorizedMarkBatch,
    records: tuple[PathRecord, ...],
    *,
    seed: int,
    target_property_conditions: Mapping[str, tuple[float, ...]],
    condition_dropout_probability: float = 0.0,
    start_index: int = 0,
) -> FactorizedMarkBatch:
    """Attach endpoint conditions without rebuilding cached chemistry support."""

    if not records:
        raise ValueError("property-condition attachment requires records")
    if start_index < 0:
        raise ValueError("start_index must be non-negative")
    if not 0.0 <= condition_dropout_probability <= 1.0:
        raise ValueError("condition dropout probability must lie in [0, 1]")
    widths = {len(values) for values in target_property_conditions.values()}
    if len(widths) != 1 or next(iter(widths), 0) <= 0:
        raise ValueError("target property conditions must share one positive width")

    values: list[tuple[float, ...]] = []
    masks: list[tuple[bool, ...]] = []
    width = next(iter(widths))
    for offset in range(batch.batch_size):
        absolute_index = start_index + offset
        rng = np.random.default_rng(np.random.SeedSequence((seed, absolute_index)))
        record = records[int(rng.integers(len(records)))]
        row = target_property_conditions.get(record.target_key)
        if row is None:
            raise KeyError(f"missing target property condition: {record.target_key}")
        numeric_row = tuple(float(value) for value in row)
        if not np.isfinite(numeric_row).all():
            raise ValueError("target property condition contains a non-finite value")
        dropout_rng = np.random.default_rng(
            np.random.SeedSequence((seed, absolute_index, 0xC0D17))
        )
        observed = bool(dropout_rng.random() >= condition_dropout_probability)
        values.append(numeric_row if observed else (0.0,) * width)
        masks.append((observed,) * width)

    return replace(
        batch,
        property_condition_values=torch.tensor(
            values,
            dtype=batch.times.dtype,
            device=batch.times.device,
        ),
        property_condition_mask=torch.tensor(
            masks,
            dtype=torch.bool,
            device=batch.times.device,
        ),
    )


def _concatenate_factorized_mark_batches(
    batches: tuple[FactorizedMarkBatch, ...],
) -> FactorizedMarkBatch:
    if not batches:
        raise ValueError("cannot concatenate an empty batch sequence")

    def tensors(name: str) -> Tensor:
        return torch.cat([getattr(batch, name) for batch in batches], dim=0)

    support_masks = tuple(batch.ring_grow_support_mask for batch in batches)
    support_sparse = tuple(batch.ring_grow_support_sparse for batch in batches)
    if all(mask is not None for mask in support_masks):
        ring_grow_support_mask = torch.cat(
            [mask for mask in support_masks if mask is not None],
            dim=0,
        )
        ring_grow_support_sparse = None
    elif all(
        mask is not None or sparse is not None
        for mask, sparse in zip(support_masks, support_sparse)
    ):
        sparse_batches = tuple(
            sparse if sparse is not None else SparseBinaryRows.from_dense(mask)
            for mask, sparse in zip(support_masks, support_sparse)
            if sparse is not None or mask is not None
        )
        widths = {item.width for item in sparse_batches}
        if len(widths) != 1:
            raise ValueError("factorized ring support widths do not align")
        ring_grow_support_mask = None
        ring_grow_support_sparse = SparseBinaryRows.from_index_rows(
            tuple(row for item in sparse_batches for row in item.index_rows()),
            width=widths.pop(),
        )
    else:
        ring_grow_support_mask = None
        ring_grow_support_sparse = None
    delete_actions = tuple(batch.ring_delete_actions for batch in batches)
    ring_delete_actions = (
        None
        if any(actions is None for actions in delete_actions)
        else tuple(
            action_group
            for actions in delete_actions
            if actions is not None
            for action_group in actions
        )
    )
    property_values = tuple(batch.property_condition_values for batch in batches)
    property_masks = tuple(batch.property_condition_mask for batch in batches)
    if all(values is None and mask is None for values, mask in zip(property_values, property_masks)):
        concatenated_property_values = None
        concatenated_property_masks = None
    elif all(values is not None and mask is not None for values, mask in zip(property_values, property_masks)):
        concatenated_property_values = torch.cat(
            [values for values in property_values if values is not None], dim=0
        )
        concatenated_property_masks = torch.cat(
            [mask for mask in property_masks if mask is not None], dim=0
        )
    else:
        raise ValueError("factorized batches mix conditioned and unconditioned rows")
    topology_masses = tuple(
        batch.ring_topology_local_support_log_mass for batch in batches
    )
    if all(values is None for values in topology_masses):
        concatenated_topology_mass = None
    elif all(values is not None for values in topology_masses):
        concatenated_topology_mass = torch.cat(
            [values for values in topology_masses if values is not None],
            dim=0,
        )
    else:
        raise ValueError("factorized batches mix topology masses and missing values")

    return FactorizedMarkBatch(
        states=tuple(state for batch in batches for state in batch.states),
        atom_types=tensors("atom_types"),
        formal_charges=tensors("formal_charges"),
        implicit_h_counts=tensors("implicit_h_counts"),
        bonds=tensors("bonds"),
        neural_bonds=tensors("neural_bonds"),
        times=tensors("times"),
        atom_topology=tensors("atom_topology"),
        closure_topology=tensors("closure_topology"),
        ring_system_topology=tensors("ring_system_topology"),
        atom_delete_mask=tensors("atom_delete_mask"),
        cycle_edge_mask=tensors("cycle_edge_mask"),
        cyclic_pair_mask=tensors("cyclic_pair_mask"),
        graft_mask=tensors("graft_mask"),
        graft_remove_neighbors=tensors("graft_remove_neighbors"),
        graft_successor_groups=tensors("graft_successor_groups"),
        teacher_actions=tuple(action for batch in batches for action in batch.teacher_actions),
        teacher_rule_names=tuple(name for batch in batches for name in batch.teacher_rule_names),
        teacher_rates=tensors("teacher_rates"),
        importance_weights=tensors("importance_weights"),
        ring_restate_actions=tuple(
            actions for batch in batches for actions in batch.ring_restate_actions
        ),
        ring_grow_support_mask=ring_grow_support_mask,
        ring_grow_support_sparse=ring_grow_support_sparse,
        ring_delete_actions=ring_delete_actions,
        ring_grow_support_is_exact=torch.cat(
            [
                batch.ring_grow_support_is_exact
                if isinstance(batch.ring_grow_support_is_exact, Tensor)
                else torch.full(
                    (batch.batch_size,),
                    bool(batch.ring_grow_support_is_exact),
                    dtype=torch.bool,
                )
                for batch in batches
            ],
            dim=0,
        ),
        ring_grow_enablement_is_exact=torch.cat(
            [
                batch.ring_grow_enablement_is_exact
                if isinstance(batch.ring_grow_enablement_is_exact, Tensor)
                else torch.full(
                    (batch.batch_size,),
                    bool(batch.ring_grow_enablement_is_exact),
                    dtype=torch.bool,
                )
                for batch in batches
            ],
            dim=0,
        ),
        ring_topology_local_support_log_mass=concatenated_topology_mass,
        ring_teacher_semantic_certificates=tuple(
            certificate
            for batch in batches
            for certificate in (
                batch.ring_teacher_semantic_certificates
                if getattr(batch, "ring_teacher_semantic_certificates", None) is not None
                else (None,) * batch.batch_size
            )
        ),
        property_condition_values=concatenated_property_values,
        property_condition_mask=concatenated_property_masks,
    )


@torch.no_grad()
def factorized_mark_metrics(
    model: FactorizedTraceletRateModel,
    batch: FactorizedMarkBatch,
    *,
    use_bf16: bool,
    microbatch_size: int | None = None,
) -> dict[str, float]:
    """Evaluate a frozen batch without making GPU memory scale with its size."""

    if microbatch_size is not None and microbatch_size <= 0:
        raise ValueError("evaluation microbatch size must be positive")
    device = model.device
    resolved_microbatch_size = min(
        batch.batch_size,
        batch.batch_size if microbatch_size is None else microbatch_size,
    )
    loss_sum = 0.0
    teacher_probability_sum = 0.0
    teacher_family_probability_sum = 0.0
    family_hits_sum = 0
    family_top3_hits_sum = 0
    nonterminal_count = 0
    terminal_hazard_sum = 0.0
    terminal_count = 0
    predicted_hazard_sum = 0.0
    teacher_hazard_sum = 0.0
    hazard_absolute_error_sum = 0.0
    weighted_hazard_absolute_error_sum = 0.0
    importance_weight_sum = 0.0
    family_counts = [0 for _ in MARK_RULE_NAMES]
    family_hits = [0 for _ in MARK_RULE_NAMES]
    family_top3_hits = [0 for _ in MARK_RULE_NAMES]
    family_teacher_probability_sums = [0.0 for _ in MARK_RULE_NAMES]
    family_teacher_mark_probability_sums = [0.0 for _ in MARK_RULE_NAMES]
    for start in range(0, batch.batch_size, resolved_microbatch_size):
        cpu_batch = batch.subbatch(
            start,
            min(start + resolved_microbatch_size, batch.batch_size),
        )
        device_batch = cpu_batch.to(
            device,
            non_blocking=device.type == "cuda",
        )
        context = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if use_bf16 and device.type == "cuda"
            else nullcontext()
        )
        with context:
            prediction = model.forward_mark_batch(device_batch)
            loss = factorized_mark_bregman_loss(prediction, device_batch)
        teacher_family = torch.tensor(
            [
                MARK_RULE_TO_INDEX[_CYCLE_OP_EXECUTOR_TO_FAMILY.get(name, name)]
                if name is not None
                else -1
                for name in device_batch.teacher_rule_names
            ],
            dtype=torch.long,
            device=device,
        )
        nonterminal = teacher_family >= 0
        terminal = ~nonterminal
        teacher_rates = device_batch.teacher_rates.to(device)
        importance_weights = device_batch.importance_weights.to(device)
        hazard_absolute_error = (prediction.total_hazard - teacher_rates).abs()
        predicted_hazard_sum += float(prediction.total_hazard.sum())
        teacher_hazard_sum += float(teacher_rates.sum())
        hazard_absolute_error_sum += float(hazard_absolute_error.sum())
        weighted_hazard_absolute_error_sum += float(
            (hazard_absolute_error * importance_weights).sum()
        )
        importance_weight_sum += float(importance_weights.sum())
        current_nonterminal_count = int(nonterminal.sum())
        current_terminal_count = int(terminal.sum())
        loss_sum += float(loss) * device_batch.batch_size
        if current_nonterminal_count:
            teacher_probability_sum += float(
                prediction.selected_mark_log_probability[nonterminal].exp().sum()
            )
            teacher_family_probability = prediction.family_log_probabilities.gather(
                1,
                teacher_family.clamp_min(0).unsqueeze(1),
            ).squeeze(1).exp()
            teacher_family_probability_sum += float(
                teacher_family_probability[nonterminal].sum()
            )
            family_top3 = prediction.family_log_probabilities.topk(
                k=min(3, len(MARK_RULE_NAMES)),
                dim=-1,
            ).indices
            family_hits_sum += int(
                (
                    prediction.family_log_probabilities.argmax(dim=-1)[nonterminal]
                    == teacher_family[nonterminal]
                ).sum()
            )
            family_top3_hits_sum += int(
                (
                    family_top3[nonterminal]
                    == teacher_family[nonterminal].unsqueeze(1)
                ).any(dim=1).sum()
            )
            predicted_family = prediction.family_log_probabilities.argmax(dim=-1)
            teacher_mark_probability = prediction.selected_mark_log_probability.exp()
            for family_index in range(len(MARK_RULE_NAMES)):
                selected_family = nonterminal & (teacher_family == family_index)
                family_count = int(selected_family.sum())
                if not family_count:
                    continue
                family_counts[family_index] += family_count
                family_hits[family_index] += int(
                    (predicted_family[selected_family] == family_index).sum()
                )
                family_top3_hits[family_index] += int(
                    (family_top3[selected_family] == family_index).any(dim=1).sum()
                )
                family_teacher_probability_sums[family_index] += float(
                    teacher_family_probability[selected_family].sum()
                )
                family_teacher_mark_probability_sums[family_index] += float(
                    teacher_mark_probability[selected_family].sum()
                )
            nonterminal_count += current_nonterminal_count
        if current_terminal_count:
            terminal_hazard_sum += float(prediction.total_hazard[terminal].sum())
            terminal_count += current_terminal_count
    metrics = {
        "factorized_gm_loss": loss_sum / batch.batch_size,
        "mean_teacher_mark_probability": (
            teacher_probability_sum / nonterminal_count if nonterminal_count else 0.0
        ),
        "mean_teacher_family_probability": (
            teacher_family_probability_sum / nonterminal_count if nonterminal_count else 0.0
        ),
        "family_accuracy": (family_hits_sum / nonterminal_count if nonterminal_count else 0.0),
        "family_top3_accuracy": (
            family_top3_hits_sum / nonterminal_count if nonterminal_count else 0.0
        ),
        "mean_terminal_hazard": (terminal_hazard_sum / terminal_count if terminal_count else 0.0),
        "mean_predicted_hazard": predicted_hazard_sum / batch.batch_size,
        "mean_teacher_hazard": teacher_hazard_sum / batch.batch_size,
        "mean_absolute_hazard_error": hazard_absolute_error_sum / batch.batch_size,
        "importance_weighted_mean_absolute_hazard_error": (
            weighted_hazard_absolute_error_sum / importance_weight_sum
            if importance_weight_sum
            else 0.0
        ),
    }
    represented_family_indices = [
        index for index, count in enumerate(family_counts) if count
    ]
    metrics["balanced_family_accuracy"] = (
        sum(
            family_hits[index] / family_counts[index]
            for index in represented_family_indices
        )
        / len(represented_family_indices)
        if represented_family_indices
        else 0.0
    )
    metrics["balanced_family_top3_accuracy"] = (
        sum(
            family_top3_hits[index] / family_counts[index]
            for index in represented_family_indices
        )
        / len(represented_family_indices)
        if represented_family_indices
        else 0.0
    )
    metrics["represented_families"] = float(len(represented_family_indices))
    for family_index, family_name in enumerate(MARK_RULE_NAMES):
        count = family_counts[family_index]
        metrics[f"teacher_examples_{family_name}"] = float(count)
        metrics[f"family_accuracy_{family_name}"] = (
            family_hits[family_index] / count if count else 0.0
        )
        metrics[f"family_top3_accuracy_{family_name}"] = (
            family_top3_hits[family_index] / count if count else 0.0
        )
        metrics[f"mean_teacher_family_probability_{family_name}"] = (
            family_teacher_probability_sums[family_index] / count if count else 0.0
        )
        metrics[f"mean_teacher_mark_probability_{family_name}"] = (
            family_teacher_mark_probability_sums[family_index] / count
            if count
            else 0.0
        )
    return metrics


def train_factorized_mark_model(
    model: FactorizedTraceletRateModel,
    train_records: tuple[PathRecord, ...],
    validation_batch: FactorizedMarkBatch,
    *,
    dry_launch: bool = False,
    dry_launch_output: str | None = None,
    steps: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    seed: int,
    workers: int,
    late_time_fraction: float,
    operational_horizon: float,
    progress_stratification_fraction: float,
    use_aromatic_bond_view: bool,
    use_bf16: bool,
    ring_catalog: TypedRingCatalog | None = None,
    data_prefetch_factor: int = 2,
    ring_electronic_mode: str = "factorized_local",
    evaluation_points: int = 10,
    evaluation_interval: int | None = None,
    warmup_steps: int = 0,
    schedule_steps: int | None = None,
    minimum_learning_rate_fraction: float = 1.0,
    early_stopping_patience: int = 0,
    early_stopping_min_relative_delta: float = 0.0,
    progress_callback: Callable[[dict[str, float]], None] | None = None,
    checkpoint_interval: int = 0,
    checkpoint_callback: Callable[[dict[str, object]], None] | None = None,
    resume_state: dict[str, object] | None = None,
    profile_timing: bool = False,
    evaluation_batch_size: int | None = None,
    initial_validation_metrics: dict[str, float] | None = None,
    training_support_cache: ShardedTrainingSupportCache | None = None,
    require_cached_support: bool = False,
    target_property_conditions: Mapping[str, tuple[float, ...]] | None = None,
    condition_dropout_probability: float = 0.0,
    trainable_parameter_scope: str = "all",
    record_index_sampler: object | None = None,
) -> tuple[list[dict[str, float]], dict[str, float]]:
    if steps <= 0 or batch_size <= 0 or learning_rate <= 0.0:
        raise ValueError("steps, batch size, and learning rate must be positive")
    if workers < 0:
        raise ValueError("data workers must be non-negative")
    if data_prefetch_factor <= 0:
        raise ValueError("data prefetch factor must be positive")
    if evaluation_interval is not None and evaluation_interval <= 0:
        raise ValueError("evaluation interval must be positive")
    resolved_schedule_steps = steps if schedule_steps is None else int(schedule_steps)
    if resolved_schedule_steps <= 0:
        raise ValueError("learning-rate schedule horizon must be positive")
    if not 0 <= warmup_steps <= steps:
        raise ValueError("warmup steps must lie in [0, steps]")
    if not 0.0 < minimum_learning_rate_fraction <= 1.0:
        raise ValueError("minimum learning-rate fraction must lie in (0, 1]")
    if early_stopping_patience < 0:
        raise ValueError("early-stopping patience must be non-negative")
    if evaluation_batch_size is not None and evaluation_batch_size <= 0:
        raise ValueError("evaluation batch size must be positive")
    if not 0.0 <= early_stopping_min_relative_delta < 1.0:
        raise ValueError("early-stopping minimum relative delta must lie in [0, 1)")
    configure_factorized_trainable_parameters(
        model,
        scope=trainable_parameter_scope,
    )
    optimizer = torch.optim.AdamW(
        factorized_adamw_parameter_groups(model, weight_decay=weight_decay),
        lr=learning_rate,
    )
    if resume_state is None:
        start_step = 0
        model.eval()
        best_metrics = (
            factorized_mark_metrics(
                model,
                validation_batch,
                use_bf16=use_bf16,
                microbatch_size=evaluation_batch_size,
            )
            if initial_validation_metrics is None
            else {str(key): float(value) for key, value in initial_validation_metrics.items()}
        )
        best_metrics["selected_step"] = 0.0
        best_state = _clone_model_state(model)
        history: list[dict[str, float]] = []
        evaluations_without_improvement = 0
        early_stopping_reference_loss = float(best_metrics["factorized_gm_loss"])
    else:
        required = {
            "completed_steps",
            "current_state_dict",
            "optimizer_state_dict",
            "best_state_dict",
            "best_metrics",
            "history",
            "optimizer_kind",
            "evaluations_without_improvement",
        }
        missing = sorted(required - resume_state.keys())
        if missing:
            raise ValueError(f"factorized recovery state is missing: {missing}")
        if resume_state["optimizer_kind"] != "adamw_decoupled_v1":
            raise ValueError("factorized recovery checkpoint uses an incompatible optimizer")
        start_step = int(resume_state["completed_steps"])
        model.load_state_dict(resume_state["current_state_dict"])  # type: ignore[arg-type]
        optimizer.load_state_dict(resume_state["optimizer_state_dict"])  # type: ignore[arg-type]
        torch_rng_state = resume_state.get("torch_rng_state")
        if isinstance(torch_rng_state, Tensor):
            torch.set_rng_state(torch_rng_state.detach().cpu().to(torch.uint8))
        cuda_rng_states = resume_state.get("cuda_rng_states")
        if cuda_rng_states is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(
                [state.detach().cpu().to(torch.uint8) for state in cuda_rng_states]  # type: ignore[union-attr]
            )
        best_state = {
            name: value.detach().clone()
            for name, value in resume_state["best_state_dict"].items()  # type: ignore[union-attr]
        }
        best_metrics = {
            str(key): float(value)
            for key, value in resume_state["best_metrics"].items()  # type: ignore[union-attr]
        }
        history = [
            {str(key): float(value) for key, value in row.items()}
            for row in resume_state["history"]  # type: ignore[union-attr]
        ]
        evaluations_without_improvement = int(resume_state["evaluations_without_improvement"])
        early_stopping_reference_loss = float(
            resume_state.get(
                "early_stopping_reference_loss",
                best_metrics["factorized_gm_loss"],
            )
        )

    if (
        resume_state is not None
        and early_stopping_patience > 0
        and evaluations_without_improvement >= early_stopping_patience
    ):
        # The recovery callback runs before the terminal early-stop break.  A
        # retry from that exact checkpoint must therefore be a no-op, not an
        # extra evaluation interval that can alter the selected model.
        model.load_state_dict(best_state)
        model.eval()
        return history, best_metrics

    loader = factorized_mark_loader(
        train_records,
        steps=steps,
        batch_size=batch_size,
        start_step=start_step,
        seed=seed,
        workers=workers,
        late_time_fraction=late_time_fraction,
        operational_horizon=operational_horizon,
        progress_stratification_fraction=progress_stratification_fraction,
        use_aromatic_bond_view=use_aromatic_bond_view,
        pin_memory=model.device.type == "cuda",
        ring_catalog=ring_catalog,
        prefetch_factor=data_prefetch_factor,
        ring_electronic_mode=ring_electronic_mode,
        training_support_cache=training_support_cache,
        require_cached_support=require_cached_support,
        target_property_conditions=target_property_conditions,
        condition_dropout_probability=condition_dropout_probability,
        ring_family_mass_mode=model.ring_family_mass_mode,
        compute_ring_grow_support=model.enable_ring_grow_macro,
        compute_ring_restates=model.enable_ring_restates,
        compute_cyclic_graft=model.enable_cyclic_graft,
        compute_ring_opening=model.enable_ring_opening,
        record_index_sampler=record_index_sampler,
    )
    timing_loop_started = perf_counter()
    iterator: Iterator[FactorizedMarkBatch] = iter(loader)
    cumulative_data_wait = 0.0
    cumulative_update_time = 0.0
    maximum_data_wait = 0.0
    data_wait_history: list[float] = []
    resolved_evaluation_interval = (
        evaluation_interval
        if evaluation_interval is not None
        else max(steps // evaluation_points, 1)
    )
    model.train()
    for step in range(start_step, steps):
        completed_steps = step + 1
        current_learning_rate = cosine_warmup_learning_rate(
            base_learning_rate=learning_rate,
            completed_step=completed_steps,
            total_steps=resolved_schedule_steps,
            warmup_steps=warmup_steps,
            minimum_fraction=minimum_learning_rate_fraction,
        )
        for parameter_group in optimizer.param_groups:
            parameter_group["lr"] = current_learning_rate
        started = perf_counter()
        cpu_batch = next(iterator)
        loaded_at = perf_counter()
        batch = cpu_batch.to(model.device, non_blocking=model.device.type == "cuda")
        transferred_at = perf_counter()
        optimizer.zero_grad(set_to_none=True)
        context = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if use_bf16 and model.device.type == "cuda"
            else nullcontext()
        )
        with context:
            prediction = model.forward_mark_batch(batch)
            loss = factorized_mark_bregman_loss(prediction, batch)
        if dry_launch:
            # CPU dry-launch (owner mandate §1): exit after the FIRST real training forward + one real
            # editing-validation forward, BEFORE backward/optimizer -- proving the zero-mixture path reaches
            # the first editing batch with a finite loss and zero optimizer updates on the real code path.
            _zmi.bump("edit_training_batches_emitted")
            _zmi.bump("model_forward_calls")
            _zmi.bump("gm_loss_calls")
            model.eval()
            val_metrics = factorized_mark_metrics(
                model, validation_batch, use_bf16=use_bf16, microbatch_size=evaluation_batch_size,
            )
            _zmi.bump("edit_validation_batches_emitted")
            _zmi.bump("model_forward_calls")
            _zmi.bump("gm_loss_calls")
            ok, zmi_report = _zmi.zero_mixture_ok()
            result = {
                "phase": "dry_launch_first_batch",
                "train_gm_loss": float(loss),
                "train_loss_finite": bool(torch.isfinite(loss)),
                "validation_gm_loss": float(val_metrics.get("factorized_gm_loss", float("nan"))),
                "validation_loss_finite": bool(
                    torch.isfinite(torch.tensor(float(val_metrics.get("factorized_gm_loss", float("nan")))))
                ),
                "train_batch_size": int(batch.batch_size),
                "validation_batch_size": int(validation_batch.batch_size),
                "zero_mixture_ok": ok,
                "zero_mixture_report": zmi_report,
            }
            if dry_launch_output:
                Path(dry_launch_output).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
            print(json.dumps({"phase": "dry_launch_first_batch",
                              "train_gm_loss": result["train_gm_loss"],
                              "validation_gm_loss": result["validation_gm_loss"],
                              "zero_mixture_ok": ok}, sort_keys=True), flush=True)
            return result
        if profile_timing:
            _synchronize(model.device)
        forwarded_at = perf_counter()
        _zmi.bump("backward_calls")
        loss.backward()
        if profile_timing:
            _synchronize(model.device)
        backward_at = perf_counter()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        _zmi.bump("optimizer_steps")
        _zmi.bump("optimizer_step_calls")
        optimizer.step()
        if profile_timing:
            _synchronize(model.device)
        optimized_at = perf_counter()
        data_wait = loaded_at - started
        update_time = optimized_at - started
        cumulative_data_wait += data_wait
        cumulative_update_time += update_time
        maximum_data_wait = max(maximum_data_wait, data_wait)
        data_wait_history.append(data_wait)
        evaluated = (
            step == 0
            or completed_steps % resolved_evaluation_interval == 0
            or completed_steps == steps
        )
        should_early_stop = False
        if evaluated:
            model.eval()
            metrics = factorized_mark_metrics(
                model,
                validation_batch,
                use_bf16=use_bf16,
                microbatch_size=evaluation_batch_size,
            )
            metrics["step"] = float(completed_steps)
            metrics["train_batch_loss"] = float(loss.detach())
            metrics["learning_rate"] = float(current_learning_rate)
            current_validation_loss = float(metrics["factorized_gm_loss"])
            improved = current_validation_loss < best_metrics["factorized_gm_loss"]
            if improved:
                best_metrics = {
                    key: value
                    for key, value in metrics.items()
                    if key not in {"step", "train_batch_loss"}
                }
                best_metrics["selected_step"] = float(completed_steps)
                best_state = _clone_model_state(model)
            material_threshold = early_stopping_reference_loss * (
                1.0 - early_stopping_min_relative_delta
            )
            materially_improved = current_validation_loss < material_threshold
            if materially_improved:
                early_stopping_reference_loss = current_validation_loss
                evaluations_without_improvement = 0
            elif completed_steps >= max(warmup_steps, 1):
                evaluations_without_improvement += 1
            metrics["materially_improved"] = float(materially_improved)
            metrics["early_stopping_reference_loss"] = float(early_stopping_reference_loss)
            metrics["evaluations_without_improvement"] = float(evaluations_without_improvement)
            should_early_stop = (
                early_stopping_patience > 0
                and evaluations_without_improvement >= early_stopping_patience
            )
            metrics["early_stopped"] = float(should_early_stop)
            history.append(metrics)
            if progress_callback is not None:
                observed_updates = completed_steps - start_step
                timing_metrics = {
                    "timing/profile_synchronized": float(profile_timing),
                    "timing/data_wait_seconds": data_wait,
                    "timing/cumulative_data_wait_seconds": cumulative_data_wait,
                    "timing/mean_data_wait_seconds": (cumulative_data_wait / observed_updates),
                    "timing/p95_data_wait_seconds": float(np.percentile(data_wait_history, 95)),
                    "timing/max_data_wait_seconds": maximum_data_wait,
                }
                if profile_timing:
                    elapsed = max(optimized_at - timing_loop_started, 1e-12)
                    timing_metrics.update(
                        {
                            "timing/transfer_seconds": transferred_at - loaded_at,
                            "timing/forward_seconds": forwarded_at - transferred_at,
                            "timing/backward_seconds": backward_at - forwarded_at,
                            "timing/optimizer_seconds": optimized_at - backward_at,
                            "timing/update_seconds": update_time,
                            "timing/data_wait_fraction": (
                                cumulative_data_wait / max(cumulative_update_time, 1e-12)
                            ),
                            "timing/updates_per_second": observed_updates / elapsed,
                            "timing/examples_per_second": (observed_updates * batch_size / elapsed),
                        }
                    )
                progress_callback(
                    {
                        **metrics,
                        **timing_metrics,
                    }
                )
            model.train()
        should_checkpoint = checkpoint_callback is not None and (
            completed_steps == steps
            or should_early_stop
            or (checkpoint_interval > 0 and completed_steps % checkpoint_interval == 0)
        )
        if should_checkpoint:
            checkpoint_callback(
                _factorized_recovery_state(
                    model=model,
                    optimizer=optimizer,
                    completed_steps=completed_steps,
                    best_state=best_state,
                    best_metrics=best_metrics,
                    history=history,
                    evaluations_without_improvement=(evaluations_without_improvement),
                    early_stopping_reference_loss=early_stopping_reference_loss,
                )
            )
        if should_early_stop:
            break
    model.load_state_dict(best_state)
    model.eval()
    return history, best_metrics


def _clone_model_state(model: nn.Module) -> dict[str, Tensor]:
    return {name: value.detach().clone() for name, value in model.state_dict().items()}


def factorized_adamw_parameter_groups(
    model: nn.Module,
    *,
    weight_decay: float,
) -> list[dict[str, object]]:
    """Partition dense weights from lookup and scale parameters for AdamW.

    Mark dictionaries are large, sparsely activated embedding tables.  Coupled
    Adam L2 regularization normalizes their otherwise tiny decay gradients and
    can collapse an inactive entry by roughly one learning-rate unit per step.
    Decoupled decay avoids that pathology, and embeddings, biases, normalization
    scales, and other vector parameters receive no decay at all.
    """

    if weight_decay < 0.0:
        raise ValueError("weight decay must be non-negative")
    no_decay_ids: set[int] = set()
    for module in model.modules():
        if isinstance(module, (nn.Embedding, nn.LayerNorm)):
            no_decay_ids.update(id(parameter) for parameter in module.parameters(recurse=False))

    decay: list[nn.Parameter] = []
    no_decay: list[nn.Parameter] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        excluded = id(parameter) in no_decay_ids or parameter.ndim < 2 or name.endswith(".bias")
        (no_decay if excluded else decay).append(parameter)

    assigned = {id(parameter) for parameter in (*decay, *no_decay)}
    expected = {id(parameter) for parameter in model.parameters() if parameter.requires_grad}
    if assigned != expected or len(decay) + len(no_decay) != len(expected):
        raise RuntimeError("optimizer parameter partition is incomplete or duplicated")
    return [
        {"params": decay, "weight_decay": float(weight_decay)},
        {"params": no_decay, "weight_decay": 0.0},
    ]


_CHEMISTRY_MARK_PARAMETER_PREFIXES = (
    "grow_root_head.",
    "grow_query.",
    "grow_option.",
    "restate_head.",
    "reorder_head.",
    "ring_system_atom_head.",
    "ring_system_role_head.",
    "ring_system_global_category_pair",
    "ring_system_adjacent_category_pair",
)

_RING_TOPOLOGY_PARAMETER_PREFIXES = (
    "ring_topology_group_head.",
)


def configure_factorized_trainable_parameters(
    model: FactorizedTraceletRateModel,
    *,
    scope: str,
) -> tuple[str, ...]:
    """Select a controlled training surface for a factorized rate model.

    ``chemistry_marks_only`` repairs atom/bond marks and coordinated ring
    electronics while keeping the encoder, total hazard, family law, Graft,
    ring topology/template choice, and placement rates exactly frozen.  It is
    intentionally narrower than generic fine-tuning so a chemistry pilot
    cannot recreate the family-rate and topology drift seen in the coupled
    P1/P2 experiment.

    ``ring_topology_only`` trains only the shared topology-and-cycle group
    head introduced above the complete-template selector.  It leaves family
    timing, exact template residuals, placements, electronics, and the graph
    encoder frozen, isolating the diagnosed fused-ring probability seam.

    ``chemistry_and_topology`` is the union of the two above: it repairs the
    atom/bond/ring-electronic marks and the shared topology-cycle group head
    together, while keeping the encoder, total hazard, family law, Graft,
    exact template residuals, and placement frozen.  This composes the two
    independently rollout-validated fixes (chemistry marks and ring-topology
    mix) without unfreezing the family/Graft rates that caused the coupled
    P1/P2 drift.
    """

    valid_scopes = {
        "all",
        "chemistry_marks_only",
        "ring_topology_only",
        "chemistry_and_topology",
    }
    if scope not in valid_scopes:
        raise ValueError(f"unknown trainable parameter scope: {scope}")
    selected: list[str] = []
    for name, parameter in model.named_parameters():
        trainable = (
            scope == "all"
            or (
                scope == "chemistry_marks_only"
                and name.startswith(_CHEMISTRY_MARK_PARAMETER_PREFIXES)
            )
            or (
                scope == "ring_topology_only"
                and name.startswith(_RING_TOPOLOGY_PARAMETER_PREFIXES)
            )
            or (
                scope == "chemistry_and_topology"
                and name.startswith(
                    _CHEMISTRY_MARK_PARAMETER_PREFIXES
                    + _RING_TOPOLOGY_PARAMETER_PREFIXES
                )
            )
        )
        parameter.requires_grad_(trainable)
        if trainable:
            selected.append(name)
    if not selected:
        raise RuntimeError("trainable parameter scope selected no parameters")
    return tuple(selected)


def cosine_warmup_learning_rate(
    *,
    base_learning_rate: float,
    completed_step: int,
    total_steps: int,
    warmup_steps: int,
    minimum_fraction: float,
) -> float:
    """Deterministic linear-warmup/cosine-decay schedule, safe on resume."""

    if base_learning_rate <= 0.0 or total_steps <= 0:
        raise ValueError("learning rate and total steps must be positive")
    if completed_step < 1:
        raise ValueError("completed step must be positive")
    if not 0 <= warmup_steps <= total_steps:
        raise ValueError("warmup steps must lie in [0, total_steps]")
    if not 0.0 < minimum_fraction <= 1.0:
        raise ValueError("minimum fraction must lie in (0, 1]")
    if warmup_steps > 0 and completed_step <= warmup_steps:
        return base_learning_rate * completed_step / warmup_steps
    decay_steps = max(total_steps - warmup_steps, 1)
    progress = min(
        max((completed_step - warmup_steps) / decay_steps, 0.0),
        1.0,
    )
    cosine_fraction = 0.5 * (1.0 + cos(pi * progress))
    scale = minimum_fraction + (1.0 - minimum_fraction) * cosine_fraction
    return base_learning_rate * scale


def _factorized_recovery_state(
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    completed_steps: int,
    best_state: dict[str, Tensor],
    best_metrics: dict[str, float],
    history: list[dict[str, float]],
    evaluations_without_improvement: int,
    early_stopping_reference_loss: float,
) -> dict[str, object]:
    return {
        "optimizer_kind": "adamw_decoupled_v1",
        "completed_steps": int(completed_steps),
        "current_state_dict": _clone_model_state(model),
        "optimizer_state_dict": copy.deepcopy(optimizer.state_dict()),
        "torch_rng_state": torch.get_rng_state().detach().cpu(),
        "cuda_rng_states": (
            [state.detach().cpu() for state in torch.cuda.get_rng_state_all()]
            if torch.cuda.is_available()
            else None
        ),
        "best_state_dict": {name: value.detach().clone() for name, value in best_state.items()},
        "best_metrics": copy.deepcopy(best_metrics),
        "history": copy.deepcopy(history),
        "evaluations_without_improvement": int(evaluations_without_improvement),
        "early_stopping_reference_loss": float(early_stopping_reference_loss),
    }


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


__all__ = [
    "FactorizedMarkCollator",
    "FactorizedMarkDataset",
    "FactorizedMarkExample",
    "factorized_mark_loader",
    "factorized_mark_metrics",
    "factorized_adamw_parameter_groups",
    "configure_factorized_trainable_parameters",
    "cosine_warmup_learning_rate",
    "sample_factorized_mark_batch",
    "train_factorized_mark_model",
]
