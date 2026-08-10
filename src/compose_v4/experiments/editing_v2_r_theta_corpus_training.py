"""Resumable, checkpointed training of R_theta on the frozen COMPOSE corpus.

WHAT THIS IS FOR
----------------
The profiler measured a step; this runs a claim-bearing training job. The
difference is entirely in what happens when something stops: a run that cannot
resume exactly is a run whose result cannot be attributed to the frozen inputs.

MEASURED on an A10G at batch 32: 634 ms per step, 50.45 examples/s, so one
144,870-example epoch is ~48 minutes. A container wall shorter than the run is
therefore normal, not exceptional, and the loop must always end on a
checkpoint rather than on a kill.

WHAT A CHECKPOINT CARRIES
-------------------------
Model, optimizer AND RNG state, plus the step counter, the selected-checkpoint
state and its criterion, the metric trajectory, and the per-family gradient
evidence. Resuming from model weights alone would silently restart the
optimizer moments and the sampling RNG, which changes the trajectory while
every count still looks right.

Each state carries its own semantic digest and the file carries a physical one,
so a resume verifies the bytes it is about to train from rather than trusting
the path it found them at.

SELECTION IS NOT THE LAST STEP
------------------------------
The checkpoint that ships is chosen by held-out ``(minimum teacher-successor
probability, -mean NLL)`` -- the same criterion the T1 runner uses -- evaluated
on the frozen panel, never by final training loss. The panel is NOT distributed
like the training draw (cycle_insert is 36.33% of it against 6.50% of the law),
so both the panel-native mean and the deployment-weighted mean are reported and
neither is presented as "the" number.
"""

from __future__ import annotations

import copy
import hashlib
import io
import json
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from compose_v4.experiments.editing_p50_gate import state_dict_semantic_sha256
from compose_v4.experiments.factorized_successor_training import (
    factorized_successor_identity_loss,
    forward_teacher_successor_batch,
)

CHECKPOINT_SCHEMA = "compose.editing_v2.r_theta_corpus_checkpoint"
CHECKPOINT_SCHEMA_VERSION = 1
CHECKPOINT_STATUS = "RECOVERABLE_R_THETA_STATE_NO_DOWNSTREAM_AUTHORITY"
CHECKPOINT_FILENAME = "R_THETA_CHECKPOINT.pt"

#: Frozen hazard head. It must receive no gradient and must not move.
TOTAL_HAZARD_PREFIX = "total_hazard_head."

#: A per-cell mean over fewer than this many entries is noise, not a gate.
THIN_CELL_ENTRIES = 30


class RThetaTrainingError(RuntimeError):
    """The corpus training run cannot proceed safely."""


def _torch_bytes(payload: Mapping[str, Any]) -> bytes:
    buffer = io.BytesIO()
    torch.save(payload, buffer)
    return buffer.getvalue()


def _atomic_write(path: Path, payload: bytes) -> None:
    """Write via a temp file and rename, so a kill cannot leave a torn file.

    A half-written checkpoint that still loads is worse than none: it resumes
    into a state that never existed.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(dir=str(path.parent), suffix=".partial")
    try:
        with os.fdopen(handle, "wb") as sink:
            sink.write(payload)
            sink.flush()
            os.fsync(sink.fileno())
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def _rng_state() -> dict[str, Any]:
    return {
        "torch": torch.get_rng_state(),
        "torch_cuda": (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        ),
    }


def _restore_rng(state: Mapping[str, Any]) -> None:
    torch.set_rng_state(state["torch"])
    if state.get("torch_cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def _clone_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}


@dataclass(frozen=True, slots=True)
class RunIdentity:
    """Every frozen input the result is attributed to.

    Carried into the checkpoint and verified on resume, so a run cannot be
    continued against a different corpus, law, panel or initialization and
    still present itself as the same experiment.
    """

    initialization_seed: int
    initial_model_state_sha256: str
    library_sha256: str
    split_sha256: str
    sampling_law_sha256: str
    manifest_sha256: str
    packed_store_records_sha256: str
    eval_panel_sha256: str

    def payload(self) -> dict[str, Any]:
        return {
            "initialization_seed": self.initialization_seed,
            "initial_model_state_sha256": self.initial_model_state_sha256,
            "library_sha256": self.library_sha256,
            "split_sha256": self.split_sha256,
            "sampling_law_sha256": self.sampling_law_sha256,
            "manifest_sha256": self.manifest_sha256,
            "packed_store_records_sha256": self.packed_store_records_sha256,
            "eval_panel_sha256": self.eval_panel_sha256,
        }

    def sha256(self) -> str:
        return hashlib.sha256(
            json.dumps(self.payload(), sort_keys=True).encode()
        ).hexdigest()


def write_checkpoint(
    path: Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    identity: RunIdentity,
    completed_steps: int,
    selected_step: int,
    selected_criterion: tuple[float, float, int] | None,
    selected_state: Mapping[str, torch.Tensor] | None,
    trajectory: Sequence[Mapping[str, Any]],
    stream_sha256: str,
    resume_count: int,
) -> dict[str, Any]:
    """Persist everything a resume needs, atomically, with digests."""

    model_state = _clone_state(model)
    optimizer_state = optimizer.state_dict()
    chosen = dict(selected_state) if selected_state is not None else {}
    payload = {
        "schema": CHECKPOINT_SCHEMA,
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "status": CHECKPOINT_STATUS,
        "identity": identity.payload(),
        "identity_sha256": identity.sha256(),
        "completed_steps": int(completed_steps),
        "selected_step": int(selected_step),
        "selected_criterion": list(selected_criterion) if selected_criterion else None,
        "model_state": model_state,
        "model_state_sha256": state_dict_semantic_sha256(model_state),
        "optimizer_state": optimizer_state,
        "selected_model_state": chosen,
        "selected_model_state_sha256": (
            state_dict_semantic_sha256(chosen) if chosen else None
        ),
        "trajectory": [dict(item) for item in trajectory],
        # Without this a resume silently restarts the sampling and dropout
        # streams, which changes the trajectory while every count still agrees.
        "rng_state": _rng_state(),
        "stream_sha256": stream_sha256,
        "torch_version": str(torch.__version__),
        "resume_count": int(resume_count),
    }
    encoded = _torch_bytes(payload)
    _atomic_write(Path(path), encoded)
    return {
        "path": str(path),
        "file_sha256": hashlib.sha256(encoded).hexdigest(),
        "file_bytes": len(encoded),
        "completed_steps": int(completed_steps),
        "model_state_sha256": payload["model_state_sha256"],
        "selected_step": int(selected_step),
        "resume_count": int(resume_count),
    }


def load_checkpoint(
    path: Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    identity: RunIdentity,
    expected_stream_sha256: str,
    expected_file_sha256: str | None = None,
) -> dict[str, Any]:
    """Restore a run, refusing anything that is not the same experiment."""

    path = Path(path)
    encoded = path.read_bytes()
    observed = hashlib.sha256(encoded).hexdigest()
    if expected_file_sha256 is not None and observed != expected_file_sha256:
        raise RThetaTrainingError(
            f"checkpoint file digest {observed} is not the expected {expected_file_sha256}"
        )
    payload = torch.load(io.BytesIO(encoded), map_location="cpu", weights_only=False)
    if payload.get("schema") != CHECKPOINT_SCHEMA:
        raise RThetaTrainingError("checkpoint schema disagrees")
    if payload.get("identity_sha256") != identity.sha256():
        raise RThetaTrainingError(
            "checkpoint was written against different frozen inputs; refusing "
            "to continue it as the same run"
        )
    if payload.get("stream_sha256") != expected_stream_sha256:
        raise RThetaTrainingError("checkpoint was written against a different draw stream")
    model.load_state_dict(payload["model_state"], strict=True)
    optimizer.load_state_dict(payload["optimizer_state"])
    _restore_rng(payload["rng_state"])
    return payload


def evaluate_panel(
    model: torch.nn.Module,
    *,
    entry_ids: Sequence[str],
    build_batch,
    batch_size: int,
) -> list[dict[str, Any]]:
    """Score the held-out panel through the TEACHER path.

    Checkpoint selection needs the teacher-successor probability and NLL, and
    both come from ``forward_teacher_successor_batch``, which reads a median of
    ONE alias. Rank and top-1 would need full successor partitions, which the
    chunk library does not store; they are diagnostics, not selection inputs.
    """

    was_training = model.training
    model.eval()
    rows: list[dict[str, Any]] = []
    try:
        with torch.no_grad():
            for start in range(0, len(entry_ids), batch_size):
                selected = list(entry_ids[start : start + batch_size])
                batch, fibers, entries = build_batch(selected)
                prediction = forward_teacher_successor_batch(model, batch, fibers)
                log_probabilities = (
                    prediction.selected_productive_successor_log_probability.detach()
                    .cpu()
                    .tolist()
                )
                for entry, value in zip(entries, log_probabilities, strict=True):
                    rows.append(
                        {
                            "entry_id": entry.entry_id,
                            "model_family": entry.model_family,
                            "capability_cell_id": entry.capability_cell_id,
                            "teacher_successor_log_probability": float(value),
                            "teacher_successor_probability": math.exp(float(value)),
                            "teacher_successor_nll": -float(value),
                        }
                    )
    finally:
        model.train(was_training)
    return rows


def summarize_panel(
    rows: Sequence[Mapping[str, Any]],
    *,
    deployment_family_share: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    """Aggregate panel rows, reporting BOTH means rather than choosing one.

    The panel is not distributed like the training draw, so its plain mean is
    not an estimate of performance under the sampling law. Reporting only one
    of the two invites reading it as the other.
    """

    if not rows:
        raise RThetaTrainingError("cannot summarize an empty panel")

    def aggregate(selected: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        nlls = [float(item["teacher_successor_nll"]) for item in selected]
        probabilities = [float(item["teacher_successor_probability"]) for item in selected]
        return {
            "entries": len(selected),
            "mean_nll": sum(nlls) / len(nlls),
            "maximum_nll": max(nlls),
            "minimum_probability": min(probabilities),
            "mean_probability": sum(probabilities) / len(probabilities),
        }

    by_family: dict[str, list[Mapping[str, Any]]] = {}
    by_cell: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        by_family.setdefault(str(row["model_family"]), []).append(row)
        by_cell.setdefault(str(row["capability_cell_id"]), []).append(row)

    families = {name: aggregate(items) for name, items in sorted(by_family.items())}
    cells = {name: aggregate(items) for name, items in sorted(by_cell.items())}
    overall = aggregate(rows)

    deployment_mean_nll = None
    if deployment_family_share:
        weighted = 0.0
        covered = 0.0
        for name, share in deployment_family_share.items():
            if name in families:
                weighted += share * families[name]["mean_nll"]
                covered += share
        if covered > 0:
            deployment_mean_nll = weighted / covered

    return {
        "panel_native": overall,
        # Reweighted to the training draw. Answers a different question than
        # panel_native, and the artifact says so rather than implying one.
        "deployment_weighted_mean_nll": deployment_mean_nll,
        "by_family": families,
        "by_capability_cell": cells,
        "thin_cells": sorted(
            name for name, value in cells.items() if value["entries"] < THIN_CELL_ENTRIES
        ),
    }


def selection_criterion(summary: Mapping[str, Any], *, step: int) -> tuple[float, float, int]:
    """Higher is better, lexicographically.

    Minimum probability first: a checkpoint that has abandoned some transition
    entirely is worse than one that is uniformly mediocre, and a mean hides
    exactly that.
    """

    native = summary["panel_native"]
    return (
        float(native["minimum_probability"]),
        -float(native["mean_nll"]),
        -int(step),
    )


def assert_training_invariants(
    model: torch.nn.Module,
    *,
    hazard_before: Mapping[str, torch.Tensor],
    step: int,
) -> None:
    """Refuse to continue on any violation, rather than record and proceed."""

    moved = []
    for name, before in hazard_before.items():
        current = dict(model.named_parameters())[name].detach()
        if not torch.equal(before.to(current.device), current):
            moved.append(name)
    if moved:
        raise RThetaTrainingError(
            f"frozen hazard parameters changed by step {step}: {sorted(moved)}"
        )


__all__ = [
    "CHECKPOINT_FILENAME",
    "CHECKPOINT_SCHEMA",
    "CHECKPOINT_SCHEMA_VERSION",
    "THIN_CELL_ENTRIES",
    "TOTAL_HAZARD_PREFIX",
    "RThetaTrainingError",
    "RunIdentity",
    "assert_training_invariants",
    "evaluate_panel",
    "load_checkpoint",
    "selection_criterion",
    "summarize_panel",
    "write_checkpoint",
]
