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
The checkpoint that ships is chosen on the held-out reserve, never by final
training loss, and the number it is chosen by is the REFERENCE-LAW-WEIGHTED
mean NLL, subject to no family or capability-cell collapse.

Reference-law-weighted, not "training-weighted". The claim is not "evaluate
this way because SGD happened to sample this way". It is that the sampling law
DEFINES the molecular reference process R_theta is meant to estimate: the
compiled library is what chemistry is AVAILABLE, and the law is what the model
is supposed to LEARN. A stratum the law assigns zero mass therefore contributes
zero to the selection number.

That distinction is load-bearing here. MEASURED: the law gives
``atom_delete|synthetic`` and ``atom_insert|synthetic`` exactly zero draw, and
``atom_restate|synthetic`` 0.4%, because those families' real supply already
meets their targets. Scoring the model on them would select partly for
imitating a synthetic teacher policy we deliberately chose not to train --
recreating, on the lane axis, the population mismatch the support-stratified
reserve was built to remove.

FOUR VIEWS, ONE RESERVE, ONE VOTE
---------------------------------
    reference_law_weighted   the selection number
    by_family / by_capability_cell
                             selection GATES; a good weighted mean must not
                             hide a collapsing operator or capability cell
    by_support_band          unsupported / low / medium / well-supported,
                             the generalization regimes
    zero_mass_stratum        compiled-but-undrawn chemistry. Scientifically
                             interesting, diagnostic only, never a vote.

Reporting them separately is what lets one metric avoid answering contradictory
questions. The externally mined 16-shard panel is retained as a cross-cohort
test and is likewise never a selection input.
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

#: Support bands, keyed on how many POST-SPLIT training sources share the
#: entry's Murcko scaffold. Band "0" is unsupported under this split and this
#: scaffold definition -- not a claim that the chemistry is novel.
SUPPORT_BAND_ORDER = ("0", "1-4", "5-24", "25+")

#: A capability degrading by more than this many nats from its FIRST
#: measurement counts as collapsed. Loose enough not to fire on ordinary
#: step-to-step movement, tight enough that a genuinely abandoned operator
#: cannot hide inside it.
COLLAPSE_TOLERANCE_NATS = 0.5


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
    collapse_baseline: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    """Persist everything a resume needs, atomically, with digests.

    ``collapse_baseline`` is run state, not a derived quantity. Collapse is
    defined against the FIRST evaluation, so a resume that recaptured it from a
    mid-training summary would compare each capability against an already
    degraded reference and the gate would stop detecting anything.
    """

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
        "collapse_baseline": (dict(collapse_baseline)
                              if collapse_baseline is not None else None),
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
    lane_by_entry_id: Mapping[str, str] | None = None,
    support_band_by_entry_id: Mapping[str, str] | None = None,
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
                # The joint NLL cannot say WHICH half is wrong. The scorer
                # already returns both factors of
                #     P(y|x) = P(F|x) . P(y|x,F)
                # and discarding them means a +4 nat gap cannot be attributed
                # to the family gate or to site selection -- which imply
                # different fixes. Keep them.
                family_terms = (
                    prediction.teacher_family_log_probability.detach().cpu().tolist()
                )
                identity_terms = (
                    prediction.selected_within_teacher_family_log_probability.detach()
                    .cpu()
                    .tolist()
                )
                for entry, value, family_term, identity_term in zip(
                    entries, log_probabilities, family_terms, identity_terms, strict=True
                ):
                    lane = (lane_by_entry_id or {}).get(entry.entry_id)
                    rows.append(
                        {
                            "entry_id": entry.entry_id,
                            "model_family": entry.model_family,
                            "capability_cell_id": entry.capability_cell_id,
                            # The stratum key the reference law is indexed by.
                            # Carried here rather than rejoined downstream, so a
                            # row can never be weighted by a lane it does not
                            # have.
                            "lane": lane,
                            "stratum": (None if lane is None
                                        else f"{entry.model_family}|{lane}"),
                            "support_band": (support_band_by_entry_id or {}).get(
                                entry.entry_id),
                            "teacher_successor_log_probability": float(value),
                            "teacher_successor_probability": math.exp(float(value)),
                            "teacher_successor_nll": -float(value),
                            "family_nll": -float(family_term),
                            "identity_nll": -float(identity_term),
                        }
                    )
    finally:
        model.train(was_training)
    return rows


def summarize_panel(
    rows: Sequence[Mapping[str, Any]],
    *,
    deployment_family_share: Mapping[str, float] | None = None,
    reference_law_stratum_share: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    """Aggregate reserve rows into four views, only one of which votes.

    The reserve is stratified to the LIBRARY's composition, but R_theta is
    trained under the LAW's, and the two differ sharply on the lane axis --
    42.8% synthetic available against 21.0% drawn, with two strata at exactly
    zero. A plain mean is therefore not an estimate of the reference process;
    it is an estimate over chemistry that includes capabilities the law
    deliberately declines to train.

    ``reference_law_stratum_share`` supplies the law's realized draw share per
    ``"family|lane"``. Strata it does not mention, or gives zero, are excluded
    from the selection number and reported under ``zero_mass_stratum`` instead
    -- visible as a stress diagnostic, never a vote.
    """

    if not rows:
        raise RThetaTrainingError("cannot summarize an empty panel")

    def aggregate(selected: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        nlls = [float(item["teacher_successor_nll"]) for item in selected]
        probabilities = [float(item["teacher_successor_probability"]) for item in selected]
        measured = {
            "entries": len(selected),
            "mean_nll": sum(nlls) / len(nlls),
            "maximum_nll": max(nlls),
            "minimum_probability": min(probabilities),
            "mean_probability": sum(probabilities) / len(probabilities),
        }
        # P(y|x) = P(F|x) . P(y|x,F). A joint mean cannot say WHICH factor moved,
        # and the two imply different fixes: a family-gate drift is an allocation
        # problem across families, a within-family drift is the candidate scorer.
        # The scorer already returns both, so carrying them costs nothing and
        # not carrying them cost a whole diagnostic pass when atom_restate
        # degraded.
        family_terms = [float(item["family_nll"]) for item in selected
                        if item.get("family_nll") is not None]
        identity_terms = [float(item["identity_nll"]) for item in selected
                          if item.get("identity_nll") is not None]
        if family_terms:
            measured["mean_family_nll"] = sum(family_terms) / len(family_terms)
        if identity_terms:
            measured["mean_identity_nll"] = sum(identity_terms) / len(identity_terms)
        return measured

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

    # ---- view 1: the selection number -------------------------------------
    by_stratum: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        if row.get("stratum"):
            by_stratum.setdefault(str(row["stratum"]), []).append(row)
    strata = {name: aggregate(items) for name, items in sorted(by_stratum.items())}

    reference_law_weighted = None
    weighted_strata: dict[str, dict[str, Any]] = {}
    zero_mass: dict[str, dict[str, Any]] = {}
    if reference_law_stratum_share:
        positive = {k: float(v) for k, v in reference_law_stratum_share.items()
                    if float(v) > 0}
        # Renormalize over strata the reserve can actually measure. Dropping a
        # stratum without renormalizing would quietly shrink the weighted mean
        # toward zero rather than reweight it.
        usable = {k: v for k, v in positive.items() if k in strata}
        scale = sum(usable.values())
        if scale > 0:
            reference_law_weighted = sum(
                (weight / scale) * strata[name]["mean_nll"]
                for name, weight in usable.items()
            )
            weighted_strata = {
                name: {"law_weight": weight,
                       "normalized_weight": weight / scale,
                       **strata[name]}
                for name, weight in sorted(usable.items())
            }
        for name, measured in sorted(strata.items()):
            if positive.get(name, 0.0) <= 0:
                zero_mass[name] = {"law_weight": positive.get(name, 0.0), **measured}

    # ---- view 3: generalization regimes ------------------------------------
    by_band: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        if row.get("support_band"):
            by_band.setdefault(str(row["support_band"]), []).append(row)
    bands = {
        name: aggregate(by_band[name])
        for name in SUPPORT_BAND_ORDER if name in by_band
    }

    return {
        # Not a selection input. Kept because a reader who sees only the
        # weighted number cannot tell how much the weighting moved it.
        "panel_native": overall,
        "deployment_weighted_mean_nll": deployment_mean_nll,
        # THE selection number. Weighted by the law that defines the reference
        # process, not by what the reserve happens to contain.
        "reference_law_weighted_mean_nll": reference_law_weighted,
        "reference_law_weighted_strata": weighted_strata,
        "reference_law_mass_measured": sum(
            item["normalized_weight"] for item in weighted_strata.values()
        ) if weighted_strata else None,
        # Selection GATES: a good weighted mean must not hide a dead operator.
        "by_family": families,
        "by_capability_cell": cells,
        # Generalization regimes.
        "by_support_band": bands,
        # Diagnostic only. Compiled chemistry the law declines to train; scoring
        # it would select for imitating a teacher policy we chose not to learn.
        "zero_mass_stratum": zero_mass,
        "thin_cells": sorted(
            name for name, value in cells.items() if value["entries"] < THIN_CELL_ENTRIES
        ),
    }


def capability_floor(summary: Mapping[str, Any]) -> float:
    """The weakest capability the checkpoint still supports.

    Collapse means a whole FAMILY or CAPABILITY CELL has died, not that one
    example is hard. The previous criterion led on the minimum probability over
    all entries, which is a single-row order statistic: over 15,031 rows it is
    dominated by whichever entry happens to be hardest, and it moves between
    checkpoints for reasons that have nothing to do with capability.

    So the floor is the smallest MEAN probability over families and non-thin
    capability cells. Thin cells are excluded because a mean over fewer than
    THIN_CELL_ENTRIES entries would fire or not fire at random -- they are
    reported by ``summarize_panel`` and gated deliberately, not trusted here.
    """

    candidates = [
        value["mean_probability"] for value in summary["by_family"].values()
    ] + [
        value["mean_probability"]
        for value in summary["by_capability_cell"].values()
        if value["entries"] >= THIN_CELL_ENTRIES
    ]
    if not candidates:
        raise RThetaTrainingError(
            "no family or non-thin capability cell to floor selection on")
    return float(min(candidates))


def capability_baseline(summary: Mapping[str, Any]) -> dict[str, float]:
    """Per-capability mean NLL at the first evaluation, for collapse detection.

    Captured once and compared against, rather than tracking each capability's
    best-so-far. Best-so-far is the stricter reading and was rejected: normal
    step-to-step fluctuation would trip it constantly, and a gate that fires on
    noise stops being a gate.
    """

    baseline = {f"family:{name}": float(value["mean_nll"])
                for name, value in summary["by_family"].items()}
    baseline.update({
        f"cell:{name}": float(value["mean_nll"])
        for name, value in summary["by_capability_cell"].items()
        if value["entries"] >= THIN_CELL_ENTRIES
    })
    return baseline


def collapsed_capabilities(
    summary: Mapping[str, Any],
    *,
    baseline: Mapping[str, float] | None,
    tolerance_nats: float = COLLAPSE_TOLERANCE_NATS,
) -> tuple[str, ...]:
    """Which capabilities are worse than they were before training began.

    Collapse is defined against the FIRST evaluation, so it means "this
    operator has actively degraded relative to the untrained model" -- an
    unambiguous fact, not a threshold picked to make the run pass. An absolute
    probability floor was rejected because families differ in branching factor
    from 59 to 420 median legal marks, so one number cannot mean the same thing
    across them.

    Thin cells never gate: a mean over fewer than THIN_CELL_ENTRIES entries
    would fire or not fire at random. They are reported by ``summarize_panel``
    and excluded from the baseline above.
    """

    if not baseline:
        return ()
    current = {f"family:{name}": float(value["mean_nll"])
               for name, value in summary["by_family"].items()}
    current.update({
        f"cell:{name}": float(value["mean_nll"])
        for name, value in summary["by_capability_cell"].items()
        if value["entries"] >= THIN_CELL_ENTRIES
    })
    return tuple(sorted(
        name for name, reference in baseline.items()
        if name in current and current[name] > reference + tolerance_nats
    ))


def selection_criterion(
    summary: Mapping[str, Any],
    *,
    step: int,
    baseline: Mapping[str, float] | None = None,
    tolerance_nats: float = COLLAPSE_TOLERANCE_NATS,
) -> tuple[int, float, int]:
    """Higher is better, lexicographically:

        (eligible, -reference-law-weighted NLL, -step)

    LOWEST reference-law-weighted NLL, SUBJECT TO no capability collapse. The
    gate is a predicate, not a quantity: it decides which checkpoints may
    compete, and among those the weighted NLL alone decides which wins.

    An earlier version led on the capability floor itself, which quietly made
    "maximize the weakest capability" a second objective -- a checkpoint with a
    trivially better floor would beat one with substantially better matched
    NLL. Capability checks exist to stop an aggregate hiding a dead operator,
    not to become the thing being optimized.

    Ties break toward the EARLIER step: if two checkpoints are indistinguishable
    on the reserve, the one that got there with less training is the one to keep.

    If nothing is eligible the ordering still returns a best-of-bad rather than
    failing, since a run must end on some checkpoint -- but eligibility is the
    leading term, so any eligible checkpoint beats every ineligible one, and the
    trajectory records which is which.
    """

    weighted = summary.get("reference_law_weighted_mean_nll")
    if weighted is None:
        raise RThetaTrainingError(
            "selection needs reference_law_weighted_mean_nll; pass "
            "reference_law_stratum_share to summarize_panel rather than "
            "selecting on a population the law does not describe")
    collapsed = collapsed_capabilities(
        summary, baseline=baseline, tolerance_nats=tolerance_nats)
    return (0 if collapsed else 1, -float(weighted), -int(step))


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
    "SUPPORT_BAND_ORDER",
    "THIN_CELL_ENTRIES",
    "TOTAL_HAZARD_PREFIX",
    "RThetaTrainingError",
    "RunIdentity",
    "assert_training_invariants",
    "COLLAPSE_TOLERANCE_NATS",
    "capability_baseline",
    "capability_floor",
    "collapsed_capabilities",
    "evaluate_panel",
    "load_checkpoint",
    "selection_criterion",
    "summarize_panel",
    "write_checkpoint",
]
