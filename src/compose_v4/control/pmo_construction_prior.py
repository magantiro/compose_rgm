"""Production seam that puts the learned construction prior on the scored PMO path.

The library seam already exists: ``DynamicProgramOptimizer.construction_prior`` is
read by ``_mutate`` and threaded into the construction draw as ``successor_prior``,
and ``PmoPopulationController`` accepts it on both ``__init__`` and ``restore``.
What was missing is the hop this repository has been burned by three times -- a
mechanism that is built, tested and completely unreachable because no scored entry
point ever supplies it.  This module is that hop.

Why a SPEC and not the object
-----------------------------
``run_program_campaign`` records ``optimizer_kwargs_sha256 = identity(optimizer_kwargs)``,
which is ``json.dumps`` -- a live ``LearnedSuccessorPrior`` is not serializable, so
passing the object would raise before the first round.  Worse, it would leave the two
arms of a matched A/B sharing a run identity.  A :class:`ConstructionPriorSpec` payload
is JSON, so it hashes, it distinguishes the arms in the durable manifest, and it
survives ``restore`` -- which matters because ``run_program_campaign`` rebuilds the
optimizer from a snapshot for every already-complete round, and an arm parameter that
``restore`` drops silently rebuilds the OTHER arm from this arm's state.

What OFF is
-----------
ABSENT.  ``construction_prior_spec=None`` means the controller never assigns
``self.construction_prior``, so the class attribute ``None`` stands and the draw is the
historical ``rng.integers``/``rng.permutation`` path byte-identically.  A "uniform prior
object" is NOT an off switch: any law object consumes ``rng.random`` where the unlawed
draw consumes ``rng.permutation``, so it reproduces the SUPPORT and not the DRAWS.

Task independence
-----------------
Nothing here reads an oracle, a task identity, a target or an objective value.  The
model is a task-blind editing law ``Q_theta(y | x, t)``; one spec serves every benchmark.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# ---- The qualifying checkpoint ---------------------------------------------

#: The provisional editing checkpoint the construction-prior evidence was measured on.
CHECKPOINT_NAME = "ringcore_a7546e2_best.pt"
CHECKPOINT_SHA256 = "24117dfeaee91729bb4ebccb5eb5b993b4a6218605045e51917823d086b4c1e4"

#: The corpus scope the checkpoint was trained under.  ``load_factorized_rollout_checkpoint``
#: refuses a checkpoint whose persisted ``corpus_scope_hash`` differs, so a cross-scope
#: model fails loudly instead of sampling silently off-scope.
CORPUS_SCOPE_HASH = "3721d69851110fdd"

#: MANDATORY LABEL.  ``configs/ringcore_v1_checkpoint_selection.json`` forbids this
#: checkpoint for frozen results: it was selected by hazard-inclusive GM loss, an
#: explicitly forbidden criterion.  The owner authorized it as a DEVELOPMENT
#: QUALIFICATION only.  It rides in the spec, the contract, the run receipt and the
#: result artifact so no number produced with it can be quoted as a publication number
#: without the label travelling with it.
CHECKPOINT_STATUS = "DEVELOPMENT_ONLY_CONSTRUCTION_PRIOR_QUALIFICATION"

#: The operating point the zero-oracle evidence was measured at.  Floor 0.05 keeps every
#: legal candidate strictly supported, so the prior is a RE-RANKING and never a filter;
#: it is the same floor the T4 ``BridgeRegionLaw`` uses for the same reason.
DEFAULT_FLOOR = 0.05
DEFAULT_TEMPERATURE = 1.0
DEFAULT_CACHE_ENTRIES = 4096


class ConstructionPriorUnavailable(RuntimeError):
    """The declared construction prior could not be built exactly as specified.

    Raised rather than degraded.  Silently falling back to the uniform draw would
    run the OFF arm under the ON arm's name, which is the one failure this whole
    experiment cannot survive.
    """


@dataclass(frozen=True)
class ConstructionPriorSpec:
    """A JSON-serializable declaration of one construction prior.

    Every field is part of the arm identity: two specs that differ anywhere produce
    different ``optimizer_kwargs_sha256`` values and therefore different run identities.
    """

    checkpoint_path: str
    checkpoint_sha256: str = CHECKPOINT_SHA256
    corpus_scope_hash: str = CORPUS_SCOPE_HASH
    status: str = CHECKPOINT_STATUS
    floor: float = DEFAULT_FLOOR
    temperature: float = DEFAULT_TEMPERATURE
    cache_entries: int = DEFAULT_CACHE_ENTRIES

    def payload(self) -> dict:
        """The exact dictionary that rides in ``optimizer_kwargs``.

        ``checkpoint_path`` is deliberately EXCLUDED: it is a machine-local mount
        path, so including it would make the arm identity depend on where the file
        happens to sit.  The checkpoint is pinned by its ``sha256`` instead, which is
        what actually determines the law.
        """

        return {
            "schema_version": "pmo_construction_prior_spec_v1",
            "checkpoint_sha256": self.checkpoint_sha256,
            "corpus_scope_hash": self.corpus_scope_hash,
            "status": self.status,
            "floor": float(self.floor),
            "temperature": float(self.temperature),
        }

    @classmethod
    def from_payload(cls, payload: dict, *, checkpoint_path=None) -> ConstructionPriorSpec:
        if payload.get("schema_version") != "pmo_construction_prior_spec_v1":
            raise ConstructionPriorUnavailable(
                f"unexpected construction prior spec schema {payload.get('schema_version')!r}"
            )
        resolved = resolve_checkpoint_path() if checkpoint_path is None else checkpoint_path
        return cls(
            checkpoint_path=str(resolved),
            checkpoint_sha256=str(payload["checkpoint_sha256"]),
            corpus_scope_hash=str(payload["corpus_scope_hash"]),
            status=str(payload["status"]),
            floor=float(payload["floor"]),
            temperature=float(payload["temperature"]),
        )


# ---- Where the checkpoint lives ---------------------------------------------

#: The checkpoint PATH is environment, not identity.  It is deliberately kept out of
#: :meth:`ConstructionPriorSpec.payload` so an arm's ``optimizer_kwargs_sha256`` cannot
#: depend on which machine mounted the file where; the bytes are pinned by sha256
#: instead, which is what determines the law.  Search order is explicit override first,
#: then the Modal artifact volume, then the local durable backup.
CHECKPOINT_ENVIRONMENT_VARIABLE = "COMPOSE_CONSTRUCTION_PRIOR_CHECKPOINT"
CHECKPOINT_SEARCH_PATHS = (
    f"/artifacts/construction_prior/{CHECKPOINT_NAME}",
    f"/Users/rmaganti/compose_ckpt_backup/{CHECKPOINT_NAME}",
    f"/Users/rmaganti/compose_fragment_ckpt/{CHECKPOINT_NAME}",
)


def resolve_checkpoint_path() -> Path:
    """Locate the qualifying checkpoint, or say exactly where it was looked for."""

    override = os.environ.get(CHECKPOINT_ENVIRONMENT_VARIABLE)
    searched = ([override] if override else []) + list(CHECKPOINT_SEARCH_PATHS)
    for candidate in searched:
        path = Path(candidate)
        if path.exists():
            return path
    raise ConstructionPriorUnavailable(
        "construction-prior checkpoint not found; searched " + ", ".join(searched)
    )


# ---- Model construction -----------------------------------------------------

#: Process-level cache keyed on the checkpoint's CONTENT hash, never its path.
#: ``run_program_campaign`` rebuilds the optimizer once per already-complete round
#: when resuming, and reloading a 24 MB torch checkpoint per round would dominate a
#: resume.  Keyed on content so two specs naming the same bytes share one model and
#: two specs naming different bytes can never share one.
_MODEL_CACHE: dict[str, Any] = {}


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_prior_model(spec: ConstructionPriorSpec):
    """Reconstruct the editing model the spec names, verifying its bytes first."""

    path = Path(spec.checkpoint_path)
    if not path.exists():
        raise ConstructionPriorUnavailable(
            f"declared construction-prior checkpoint is absent: {path}"
        )
    observed = _file_sha256(path)
    if observed != spec.checkpoint_sha256:
        raise ConstructionPriorUnavailable(
            f"construction-prior checkpoint sha256 {observed} != declared "
            f"{spec.checkpoint_sha256}; refusing to run an arm under a model it did "
            "not declare"
        )
    cached = _MODEL_CACHE.get(observed)
    if cached is not None:
        return cached
    try:
        from evaluate_tracelet_rollouts import (
            load_factorized_rollout_checkpoint,
        )
    except ImportError:  # pragma: no cover - depends on how `scripts` is mounted
        from scripts.evaluate_tracelet_rollouts import (
            load_factorized_rollout_checkpoint,
        )
    model, _metadata = load_factorized_rollout_checkpoint(
        path, expected_scope_hash=spec.corpus_scope_hash
    )
    model.eval()
    _MODEL_CACHE[observed] = model
    return model


def build_construction_prior(spec: ConstructionPriorSpec):
    """Build the live prior object the construction draw consults.

    Raises :class:`ConstructionPriorUnavailable` rather than returning ``None``: a
    ``None`` here would run the control arm while every artifact claimed the treated
    arm, which is strictly worse than a crash before the first charged call.
    """

    from compose_v4.control.learned_successor_prior import (
        LearnedSuccessorPrior,
    )

    return LearnedSuccessorPrior(
        load_prior_model(spec),
        floor=spec.floor,
        temperature=spec.temperature,
        cache_entries=spec.cache_entries,
    )


__all__ = [
    "CHECKPOINT_ENVIRONMENT_VARIABLE",
    "CHECKPOINT_NAME",
    "CHECKPOINT_SHA256",
    "CHECKPOINT_STATUS",
    "CORPUS_SCOPE_HASH",
    "ConstructionPriorSpec",
    "ConstructionPriorUnavailable",
    "build_construction_prior",
    "load_prior_model",
    "resolve_checkpoint_path",
]
