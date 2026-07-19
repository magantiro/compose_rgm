"""Content-addressed sparse support shards for deterministic GM training rows."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path
import tempfile
from time import monotonic, sleep
from typing import Mapping

import torch
from torch import Tensor

from compose_v4.model.factorized_tracelet_rate_model import (
    RingTeacherSemanticCertificate,
)


TRAINING_SUPPORT_CACHE_FORMAT_VERSION = 1
# Bump this only when the set of executable rewrite actions changes.  Pure
# implementation accelerations and neural-model changes deliberately leave it
# untouched so their caches remain reusable.
RING_SUPPORT_SEMANTICS_VERSION = 1


def training_support_signature_fingerprint(
    signature: Mapping[str, object],
) -> str:
    """Return a stable content address for one scientific sampling stream."""

    serialized = json.dumps(
        dict(signature),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def training_support_cache_root(
    base: str | Path,
    signature: Mapping[str, object],
) -> Path:
    """Place incompatible streams in independent immutable namespaces."""

    fingerprint = training_support_signature_fingerprint(signature)
    return Path(base) / f"training_support_v{TRAINING_SUPPORT_CACHE_FORMAT_VERSION}_{fingerprint}"


def training_support_shard_path(
    root: str | Path,
    *,
    start: int,
    stop: int,
) -> Path:
    if start < 0 or stop <= start:
        raise ValueError("training support shard bounds are invalid")
    return Path(root) / f"rows_{start:012d}_{stop:012d}.pt"


@dataclass(frozen=True)
class TrainingSupportRow:
    """All expensive chemistry support for one deterministic training index."""

    indices: tuple[int, ...]
    width: int
    support_is_exact: bool
    enablement_is_exact: bool
    teacher_semantic_certificate: RingTeacherSemanticCertificate | None = None

    def __post_init__(self) -> None:
        if self.width < 0:
            raise ValueError("training support width must be non-negative")
        normalized = tuple(int(value) for value in self.indices)
        if normalized != tuple(sorted(set(normalized))):
            raise ValueError("training support indices must be sorted and unique")
        if any(index < 0 or index >= self.width for index in normalized):
            raise ValueError("training support index lies outside the vocabulary")


@dataclass(frozen=True)
class TrainingSupportShard:
    """CSR-encoded support rows for one contiguous range of stream indices."""

    start: int
    stop: int
    width: int
    indptr: Tensor
    indices: Tensor
    support_is_exact: Tensor
    enablement_is_exact: Tensor
    teacher_semantic_certificates: tuple[RingTeacherSemanticCertificate | None, ...]
    teacher_certificates_complete: bool

    def __post_init__(self) -> None:
        rows = int(self.stop) - int(self.start)
        if self.start < 0 or rows <= 0 or self.width < 0:
            raise ValueError("training support shard metadata is invalid")
        if self.indptr.ndim != 1 or self.indptr.numel() != rows + 1:
            raise ValueError("training support indptr has the wrong shape")
        if self.indices.ndim != 1:
            raise ValueError("training support indices must be one-dimensional")
        if self.support_is_exact.shape != (rows,):
            raise ValueError("training support exactness flags have the wrong shape")
        if self.enablement_is_exact.shape != (rows,):
            raise ValueError("training enablement flags have the wrong shape")
        if len(self.teacher_semantic_certificates) != rows:
            raise ValueError("training semantic-certificate rows have the wrong shape")
        indptr = self.indptr.detach().cpu().to(torch.int64)
        indices = self.indices.detach().cpu().to(torch.int64)
        if int(indptr[0]) != 0 or int(indptr[-1]) != int(indices.numel()):
            raise ValueError("training support CSR boundaries are invalid")
        if bool(torch.any(indptr[1:] < indptr[:-1])):
            raise ValueError("training support CSR boundaries are not monotone")
        if indices.numel() and (int(indices.min()) < 0 or int(indices.max()) >= int(self.width)):
            raise ValueError("training support CSR index lies outside the vocabulary")
        for row in range(rows):
            left = int(indptr[row])
            right = int(indptr[row + 1])
            values = indices[left:right]
            if values.numel() > 1 and bool(torch.any(values[1:] <= values[:-1])):
                raise ValueError("training support CSR rows must be sorted and unique")

    @classmethod
    def from_rows(
        cls,
        rows: tuple[TrainingSupportRow, ...],
        *,
        start: int,
    ) -> "TrainingSupportShard":
        if not rows:
            raise ValueError("cannot create an empty training support shard")
        widths = {int(row.width) for row in rows}
        if len(widths) != 1:
            raise ValueError("training support rows have inconsistent widths")
        offsets = [0]
        flat = []
        for row in rows:
            flat.extend(int(index) for index in row.indices)
            offsets.append(len(flat))
        return cls(
            start=int(start),
            stop=int(start) + len(rows),
            width=widths.pop(),
            indptr=torch.tensor(offsets, dtype=torch.int64),
            indices=torch.tensor(flat, dtype=torch.int32),
            support_is_exact=torch.tensor(
                tuple(bool(row.support_is_exact) for row in rows),
                dtype=torch.bool,
            ),
            enablement_is_exact=torch.tensor(
                tuple(bool(row.enablement_is_exact) for row in rows),
                dtype=torch.bool,
            ),
            teacher_semantic_certificates=tuple(row.teacher_semantic_certificate for row in rows),
            teacher_certificates_complete=True,
        )

    def row(self, absolute_index: int) -> TrainingSupportRow:
        if not self.start <= absolute_index < self.stop:
            raise IndexError(absolute_index)
        local = int(absolute_index) - int(self.start)
        left = int(self.indptr[local])
        right = int(self.indptr[local + 1])
        return TrainingSupportRow(
            indices=tuple(int(value) for value in self.indices[left:right].detach().cpu().tolist()),
            width=int(self.width),
            support_is_exact=bool(self.support_is_exact[local]),
            enablement_is_exact=bool(self.enablement_is_exact[local]),
            teacher_semantic_certificate=self.teacher_semantic_certificates[local],
        )


def _atomic_torch_save(payload: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def save_training_support_shard(
    shard: TrainingSupportShard,
    path: str | Path,
    *,
    signature: Mapping[str, object],
) -> None:
    """Atomically publish one complete immutable support shard."""

    fingerprint = training_support_signature_fingerprint(signature)
    _atomic_torch_save(
        {
            "format_version": TRAINING_SUPPORT_CACHE_FORMAT_VERSION,
            "signature": dict(signature),
            "signature_fingerprint": fingerprint,
            "start": int(shard.start),
            "stop": int(shard.stop),
            "width": int(shard.width),
            "indptr": shard.indptr.detach().cpu().to(torch.int64),
            "indices": shard.indices.detach().cpu().to(torch.int32),
            "support_is_exact": shard.support_is_exact.detach().cpu().bool(),
            "enablement_is_exact": shard.enablement_is_exact.detach().cpu().bool(),
            "teacher_semantic_certificates": shard.teacher_semantic_certificates,
        },
        Path(path),
    )


def load_training_support_shard(
    path: str | Path,
    *,
    signature: Mapping[str, object],
    expected_start: int | None = None,
    expected_stop: int | None = None,
) -> TrainingSupportShard:
    """Load and validate one shard before exposing any cached support."""

    payload = torch.load(Path(path), map_location="cpu", weights_only=False, mmap=True)
    fingerprint = training_support_signature_fingerprint(signature)
    if int(payload.get("format_version", -1)) != TRAINING_SUPPORT_CACHE_FORMAT_VERSION:
        raise ValueError("training support cache format mismatch")
    if payload.get("signature_fingerprint") != fingerprint:
        raise ValueError("training support cache/config mismatch")
    if payload.get("signature") != dict(signature):
        raise ValueError("training support cache signature payload mismatch")
    start = int(payload["start"])
    stop = int(payload["stop"])
    if expected_start is not None and start != int(expected_start):
        raise ValueError("training support shard starts at the wrong index")
    if expected_stop is not None and stop != int(expected_stop):
        raise ValueError("training support shard stops at the wrong index")
    return TrainingSupportShard(
        start=start,
        stop=stop,
        width=int(payload["width"]),
        indptr=payload["indptr"],
        indices=payload["indices"],
        support_is_exact=payload["support_is_exact"],
        enablement_is_exact=payload["enablement_is_exact"],
        teacher_semantic_certificates=tuple(
            payload.get("teacher_semantic_certificates", (None,) * (stop - start))
        ),
        teacher_certificates_complete="teacher_semantic_certificates" in payload,
    )


@dataclass
class ShardedTrainingSupportCache:
    """Lazy mmap-backed reader safe to copy into persistent data workers."""

    base: Path
    signature: Mapping[str, object]
    total_rows: int
    shard_size: int
    shard_cache_limit: int = 4
    wait_timeout_seconds: float = 0.0
    poll_interval_seconds: float = 0.25
    _loaded: OrderedDict[int, TrainingSupportShard] = field(
        default_factory=OrderedDict,
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        self.base = Path(self.base)
        self.signature = dict(self.signature)
        if self.total_rows <= 0 or self.shard_size <= 0 or self.shard_cache_limit <= 0:
            raise ValueError("training support cache dimensions must be positive")
        if self.wait_timeout_seconds < 0.0 or self.poll_interval_seconds <= 0.0:
            raise ValueError("training support cache wait settings are invalid")

    @property
    def root(self) -> Path:
        return training_support_cache_root(self.base, self.signature)

    def bounds(self, absolute_index: int) -> tuple[int, int]:
        if not 0 <= absolute_index < self.total_rows:
            raise IndexError(absolute_index)
        start = (int(absolute_index) // int(self.shard_size)) * int(self.shard_size)
        return start, min(start + int(self.shard_size), int(self.total_rows))

    def path_for_index(self, absolute_index: int) -> Path:
        start, stop = self.bounds(absolute_index)
        return training_support_shard_path(self.root, start=start, stop=stop)

    def get(self, absolute_index: int) -> TrainingSupportRow | None:
        start, stop = self.bounds(absolute_index)
        shard = self._loaded.get(start)
        if shard is None:
            path = training_support_shard_path(self.root, start=start, stop=stop)
            if not path.is_file():
                return None
            shard = load_training_support_shard(
                path,
                signature=self.signature,
                expected_start=start,
                expected_stop=stop,
            )
            self._loaded[start] = shard
            if len(self._loaded) > self.shard_cache_limit:
                self._loaded.popitem(last=False)
        else:
            self._loaded.move_to_end(start)
        return shard.row(absolute_index)

    def require(self, absolute_index: int) -> TrainingSupportRow:
        row = self.get(absolute_index)
        deadline = monotonic() + float(self.wait_timeout_seconds)
        while row is None and monotonic() < deadline:
            sleep(min(float(self.poll_interval_seconds), max(deadline - monotonic(), 0.0)))
            row = self.get(absolute_index)
        if row is None:
            raise FileNotFoundError(
                f"missing compiled training support for row {absolute_index}: "
                f"{self.path_for_index(absolute_index)}"
            )
        shard_start, _ = self.bounds(absolute_index)
        shard = self._loaded.get(shard_start)
        if shard is None or not shard.teacher_certificates_complete:
            raise FileNotFoundError(
                "compiled training support lacks exact semantic teacher "
                f"certificates for row {absolute_index}: "
                f"{self.path_for_index(absolute_index)}"
            )
        return row


__all__ = [
    "RING_SUPPORT_SEMANTICS_VERSION",
    "ShardedTrainingSupportCache",
    "TRAINING_SUPPORT_CACHE_FORMAT_VERSION",
    "TrainingSupportRow",
    "TrainingSupportShard",
    "load_training_support_shard",
    "save_training_support_shard",
    "training_support_cache_root",
    "training_support_shard_path",
    "training_support_signature_fingerprint",
]
