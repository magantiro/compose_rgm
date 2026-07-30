"""RNG-neutral joins from the mark-training stream to successor fibers.

The existing :class:`FactorizedMarkDataset` remains the sole owner of record,
time, and progress sampling. This module observes the selected immutable packed
address and joins support-only successor metadata after that draw. A cache miss
or exact-state mismatch aborts before tensorization or optimization.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, fields

import torch
from torch.utils.data import Dataset

from compose_v4.chem.persistent_state_identity import (
    persistent_slot_state_sha256,
)
from compose_v4.data.sharded_successor_fiber_cache import (
    ShardedSuccessorFiberCache,
)
from compose_v4.data.successor_fiber_cache import (
    SuccessorFiberCacheAddress,
    SuccessorFiberCacheRecord,
)
from compose_v4.experiments.factorized_mark_conditional import (
    FactorizedMarkDataset,
    FactorizedMarkExample,
)
from compose_v4.experiments.factorized_successor_training import (
    StateProductiveSupport,
    TeacherSuccessorFiber,
)
from compose_v4.model.factorized_tracelet_rate_model import FactorizedMarkBatch

SuccessorSemanticCellKey = tuple[str, int, int]


class FactorizedSuccessorDataError(RuntimeError):
    """A sampled row cannot be joined exactly to frozen successor support."""


@dataclass(frozen=True)
class FactorizedSuccessorExample:
    """One unchanged mark example plus its exact support-only cache row."""

    mark_example: FactorizedMarkExample
    cache_record: SuccessorFiberCacheRecord
    semantic_cell_id: str | None = None

    def __post_init__(self) -> None:
        if self.semantic_cell_id is not None and (
            not isinstance(self.semantic_cell_id, str)
            or not self.semantic_cell_id
        ):
            raise ValueError("semantic_cell_id must be null or nonempty text")


@dataclass(frozen=True)
class FactorizedSuccessorBatch:
    """Accelerator tensors with aligned immutable successor metadata."""

    mark_batch: FactorizedMarkBatch
    fibers: tuple[TeacherSuccessorFiber | None, ...]
    state_supports: tuple[StateProductiveSupport, ...]
    cache_addresses: tuple[SuccessorFiberCacheAddress, ...]
    semantic_cell_ids: tuple[str | None, ...]

    def __post_init__(self) -> None:
        batch_size = self.mark_batch.batch_size
        lengths = {
            len(self.fibers),
            len(self.state_supports),
            len(self.cache_addresses),
            len(self.semantic_cell_ids),
        }
        if lengths != {batch_size}:
            raise ValueError(
                "successor batch metadata is not aligned with the mark batch"
            )
        for fiber, support, address in zip(
            self.fibers,
            self.state_supports,
            self.cache_addresses,
            strict=True,
        ):
            if fiber is None:
                if not address.is_terminal:
                    raise ValueError(
                        "nonterminal successor batch row lacks a teacher fiber"
                    )
            else:
                if address.is_terminal:
                    raise ValueError(
                        "terminal successor batch row carries a teacher fiber"
                    )
                if fiber.state_support != support:
                    raise ValueError(
                        "successor batch fiber/support metadata disagrees"
                    )

    @property
    def batch_size(self) -> int:
        return self.mark_batch.batch_size

    def subbatch(self, start: int, stop: int) -> FactorizedSuccessorBatch:
        if not 0 <= start < stop <= self.batch_size:
            raise ValueError(
                f"invalid successor subbatch bounds {(start, stop)} for "
                f"{self.batch_size}"
            )
        return FactorizedSuccessorBatch(
            mark_batch=self.mark_batch.subbatch(start, stop),
            fibers=self.fibers[start:stop],
            state_supports=self.state_supports[start:stop],
            cache_addresses=self.cache_addresses[start:stop],
            semantic_cell_ids=self.semantic_cell_ids[start:stop],
        )

    def to(
        self,
        device: torch.device,
        *,
        non_blocking: bool = False,
    ) -> FactorizedSuccessorBatch:
        return FactorizedSuccessorBatch(
            mark_batch=self.mark_batch.to(
                device,
                non_blocking=non_blocking,
            ),
            fibers=self.fibers,
            state_supports=self.state_supports,
            cache_addresses=self.cache_addresses,
            semantic_cell_ids=self.semantic_cell_ids,
        )

    def pin_memory(self) -> FactorizedSuccessorBatch:
        return FactorizedSuccessorBatch(
            mark_batch=self.mark_batch.pin_memory(),
            fibers=self.fibers,
            state_supports=self.state_supports,
            cache_addresses=self.cache_addresses,
            semantic_cell_ids=self.semantic_cell_ids,
        )


class FactorizedSuccessorDataset(Dataset[FactorizedSuccessorExample]):
    """Exact post-sampling cache join with no fallback or online compilation."""

    def __init__(
        self,
        mark_dataset: FactorizedMarkDataset,
        successor_cache: ShardedSuccessorFiberCache,
        *,
        semantic_cell_ids: Mapping[SuccessorSemanticCellKey, str] | None = None,
        require_semantic_cell_ids: bool = False,
    ) -> None:
        if not isinstance(mark_dataset, FactorizedMarkDataset):
            raise TypeError(
                "successor dataset requires the production FactorizedMarkDataset"
            )
        missing_addresses = tuple(
            index
            for index, record in enumerate(mark_dataset.records)
            if record.corpus_address is None
        )
        if missing_addresses:
            raise FactorizedSuccessorDataError(
                "successor training forbids raw/replayed records without immutable "
                f"packed addresses; missing indices={missing_addresses[:20]}"
            )
        if semantic_cell_ids is not None:
            invalid = tuple(
                key
                for key, value in semantic_cell_ids.items()
                if (
                    not isinstance(key, tuple)
                    or len(key) != 3
                    or not isinstance(value, str)
                    or not value
                )
            )
            if invalid:
                raise ValueError(
                    "semantic-cell sidecar contains malformed keys or values"
                )
        self.mark_dataset = mark_dataset
        self.successor_cache = successor_cache
        self.semantic_cell_ids = semantic_cell_ids
        self.require_semantic_cell_ids = bool(require_semantic_cell_ids)

    def __len__(self) -> int:
        return len(self.mark_dataset)

    def __getitem__(self, index: int) -> FactorizedSuccessorExample:
        # This is the only sampling call. Everything below is a deterministic
        # exact-address lookup and validation of the already-selected row.
        mark_example = self.mark_dataset[index]
        record_index = mark_example.record_index
        progress_index = mark_example.progress_index
        if type(record_index) is not int or type(progress_index) is not int:
            raise FactorizedSuccessorDataError(
                "mark dataset did not expose its selected record/progress"
            )
        if not 0 <= record_index < len(self.mark_dataset.records):
            raise FactorizedSuccessorDataError(
                "sampled record index lies outside the source corpus"
            )
        path_record = self.mark_dataset.records[record_index]
        address = path_record.corpus_address
        if address is None:  # Guard remains local even after constructor audit.
            raise FactorizedSuccessorDataError(
                "sampled record has no immutable packed address"
            )
        if not 0 <= progress_index <= address.path_length:
            raise FactorizedSuccessorDataError(
                "sampled progress lies outside the packed trace"
            )
        if path_record.path.path_length != address.path_length:
            raise FactorizedSuccessorDataError(
                "packed address and sampled path have different lengths"
            )
        cache_record = self.successor_cache.require(
            address,
            progress_index=progress_index,
            source_state=mark_example.state,
        )

        is_terminal = progress_index == address.path_length
        if is_terminal:
            if (
                mark_example.teacher_action is not None
                or mark_example.teacher_rule_name is not None
                or mark_example.teacher_rate != 0.0
                or cache_record.teacher_fiber is not None
            ):
                raise FactorizedSuccessorDataError(
                    "terminal sampled row carries jump supervision"
                )
        else:
            if (
                mark_example.teacher_action is None
                or mark_example.teacher_rule_name is None
                or mark_example.teacher_rate <= 0.0
                or cache_record.teacher_fiber is None
            ):
                raise FactorizedSuccessorDataError(
                    "nonterminal sampled row lacks jump supervision"
                )
            next_state = path_record.path.state_at(progress_index + 1)
            next_digest = persistent_slot_state_sha256(next_state)
            if cache_record.target_state_sha256 != next_digest:
                raise FactorizedSuccessorDataError(
                    "cached teacher target disagrees with the exact next path state"
                )

        cell_key = (
            address.packed_shard_content_sha256,
            address.entry_index,
            progress_index,
        )
        semantic_cell_id = (
            None
            if self.semantic_cell_ids is None
            else self.semantic_cell_ids.get(cell_key)
        )
        if self.require_semantic_cell_ids and semantic_cell_id is None:
            raise FactorizedSuccessorDataError(
                "sampled row is absent from the frozen semantic-cell sidecar"
            )
        return FactorizedSuccessorExample(
            mark_example=mark_example,
            cache_record=cache_record,
            semantic_cell_id=semantic_cell_id,
        )


@dataclass(frozen=True)
class FactorizedSuccessorCollator:
    """Delegate existing tensorization and align support-only metadata."""

    mark_collator: Callable[[list[FactorizedMarkExample]], FactorizedMarkBatch]

    def __call__(
        self,
        examples: list[FactorizedSuccessorExample],
    ) -> FactorizedSuccessorBatch:
        if not examples:
            raise ValueError("cannot collate an empty successor batch")
        mark_examples = [example.mark_example for example in examples]
        mark_batch = self.mark_collator(mark_examples)
        if mark_batch.batch_size != len(examples):
            raise FactorizedSuccessorDataError(
                "mark collator changed successor batch cardinality"
            )
        for batch_state, example in zip(
            mark_batch.states,
            examples,
            strict=True,
        ):
            record = example.cache_record
            if (
                persistent_slot_state_sha256(batch_state)
                != record.source_state_sha256
            ):
                raise FactorizedSuccessorDataError(
                    "collated mark state disagrees with cached exact state"
                )
        return FactorizedSuccessorBatch(
            mark_batch=mark_batch,
            fibers=tuple(
                example.cache_record.teacher_fiber for example in examples
            ),
            state_supports=tuple(
                example.cache_record.state_support for example in examples
            ),
            cache_addresses=tuple(
                example.cache_record.address for example in examples
            ),
            semantic_cell_ids=tuple(
                example.semantic_cell_id for example in examples
            ),
        )


def assert_successor_wrapper_is_rng_neutral(
    mark_dataset: FactorizedMarkDataset,
    successor_dataset: FactorizedSuccessorDataset,
    *,
    indices: tuple[int, ...],
) -> None:
    """Audit that the wrapper preserves every scientific field of the draw."""

    for index in indices:
        expected = mark_dataset[index]
        observed = successor_dataset[index].mark_example
        state_matches = (
            persistent_slot_state_sha256(expected.state)
            == persistent_slot_state_sha256(observed.state)
        )
        metadata_matches = all(
            getattr(expected, descriptor.name)
            == getattr(observed, descriptor.name)
            for descriptor in fields(FactorizedMarkExample)
            if descriptor.name != "state"
        )
        if not state_matches or not metadata_matches:
            raise FactorizedSuccessorDataError(
                f"successor wrapper perturbed mark example at index {index}"
            )


__all__ = [
    "FactorizedSuccessorBatch",
    "FactorizedSuccessorCollator",
    "FactorizedSuccessorDataError",
    "FactorizedSuccessorDataset",
    "FactorizedSuccessorExample",
    "SuccessorSemanticCellKey",
    "assert_successor_wrapper_is_rng_neutral",
]
