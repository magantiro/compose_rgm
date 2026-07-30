"""The one entry point every paper experiment consumes. Piece A1.1: construction and provenance.

Why one evaluator
-----------------
Every experiment needs the same things: a model reconstructed from a checkpoint, capability flags resolved,
slot-addressed state handled, legal successors enumerated, marks aggregated into canonical molecular
successors, a budget convention, metrics, provenance. If each experiment did that itself, the paper's
numbers would not be comparable, and the failure would be invisible -- two experiments can both look
correct while measuring different kernels. So the derivation lives here exactly once.

This module does NOT re-derive the process. It composes production primitives that DEFINE the trained
process (checkpoint reconstruction, the executor, legal-action enumeration, ``canonical_state_key``,
capability resolution) and adds only the experiment-facing surface. Reimplementing any of those would mean
evaluating a different stochastic process than the one that was optimized.

Validation, not reporting
-------------------------
Capability flags are not merely echoed. They are checked against the configuration the REGISTRY declares
for the lineage an experiment depends on, and the resulting support signature is what cross-arm fairness is
asserted on. A mismatch is a hard failure: a silently different support turns a controlled comparison into
two unrelated measurements.

Note what is deliberately NOT done. Checking a flag against the checkpoint metadata it was derived from
would be tautological -- the production loader preserves the historical ``corrupted_prior_mix`` fallback
for old editing metadata and reads explicit support flags for new checkpoints, so such a check can never fail and would
give false assurance. Two checks that CAN fail are used instead: the whole-ring macro must be off (the
loader defaults it to True when the key is absent, so a pre-RingCore checkpoint is caught), and the flags
must match the registry's declared production configuration (so an editing experiment cannot be handed a
de-novo checkpoint).

Evaluator surface
-----------------
Construction, capability resolution, support signature and the versioned output schema live here.
``EvaluationContext.successor_kernel`` is the sole experiment-facing constructor for the production
factorized canonical-successor kernel.  The implementation itself lives in one focused module, but
experiments never construct it independently.
"""
from __future__ import annotations

import hashlib
import json
import platform
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from compose_v4.data.charge_policy import CHARGE_POLICY_VERSION
from compose_v4.experiments.successor_kernel import KernelIdentity, SupportSignature

# Bump ONLY additively: consumers of an earlier version must keep working. A field may be added; a field
# may not change meaning or disappear.
OUTPUT_SCHEMA = "compose.experiments.evaluation"
OUTPUT_SCHEMA_VERSION = 1

# Capability flags that determine legal successor support. Order is fixed so the signature is stable.
CAPABILITY_FLAGS = (
    "enable_ring_restates",
    "enable_cyclic_graft",
    "enable_heteroatom_scan",
    "enable_ring_opening",
    "enable_cycle_ops",
    "enable_ring_system_delete",
)

# RingCore-V1 replaced the whole-ring macro with compositional cycle_close/cycle_open. The macro must be
# OFF: leaving it on would change the legal support away from the production configuration.
REQUIRED_MACRO_STATE = {"enable_ring_grow_macro": False}


class EvaluationError(RuntimeError):
    """The evaluator refused to run: bad checkpoint, capability mismatch, or unknown experiment."""


def _historical_optional_bool(
    payload: dict[str, Any],
    key: str,
    *,
    default: bool,
) -> bool:
    """Read an optional checkpoint Boolean without accepting truthy malformed metadata."""

    if key not in payload:
        return default
    value = payload[key]
    if type(value) is not bool:
        raise EvaluationError(
            f"checkpoint metadata {key!r} must be a literal Boolean when present"
        )
    return value


@dataclass(frozen=True)
class CheckpointProvenance:
    """Everything needed to say which artifact produced a number."""

    path: str
    sha256: str
    bytes: int
    checkpoint_kind: str | None
    organic_vocabulary: bool
    corrupted_prior_mix: bool
    enable_ring_restates: bool
    enable_cycle_ops: bool
    enable_ring_system_delete: bool
    corpus_scope_hash: str | None
    max_atoms: int | None
    bond_representation: str | None
    rate_factorization: str | None


@dataclass(frozen=True)
class EvaluationContext:
    """A constructed, validated evaluation target: model + resolved support + provenance."""

    experiment_id: str
    seed: int
    model: Any
    checkpoint: CheckpointProvenance
    support_signature: SupportSignature
    registry_protocol_hash: str
    capability_flags: dict[str, bool] = field(default_factory=dict)

    def kernel_identity(self, implementation: str) -> KernelIdentity:
        return KernelIdentity(
            implementation=implementation,
            support_signature=self.support_signature,
            checkpoint_sha256=self.checkpoint.sha256,
        )

    def successor_kernel(self, *, time: float):
        """Construct the one production molecular kernel experiments consume."""
        from compose_v4.experiments.production_successor_kernel import (  # noqa: PLC0415
            FactorizedCanonicalSuccessorKernel,
        )

        return FactorizedCanonicalSuccessorKernel(
            self.model,
            time=float(time),
            identity=self.kernel_identity(
                "factorized_ringcore_segmented_pushforward"
            ),
        )


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _operator_registry_hash() -> str | None:
    """The production operator-registry hash, or None if the identity module is unavailable.

    Imported lazily and tolerantly: ``scripts/`` is not always on the path for a library consumer, and a
    missing hash must degrade to "unknown" rather than crash an evaluation. It is still recorded, so a
    None here is visible in the output rather than silently absent.
    """
    try:
        from ring_core_identity import recompute_operator_registry_hash  # noqa: PLC0415
    except Exception:
        try:
            from scripts.ring_core_identity import (  # noqa: PLC0415
                recompute_operator_registry_hash,
            )
        except Exception:
            return None
    try:
        return str(recompute_operator_registry_hash())
    except Exception:
        return None


def resolve_capability_flags(model: Any) -> dict[str, bool]:
    """Read the capability flags actually set on the constructed model."""
    resolved = {}
    for flag in CAPABILITY_FLAGS:
        value = getattr(model, flag, None)
        if value is None:
            raise EvaluationError(
                f"model does not expose capability flag {flag!r}; the evaluator must report the flags "
                "that were actually applied, not assume defaults"
            )
        resolved[flag] = bool(value)
    return resolved


def validate_capabilities(
    model: Any, *, required: dict[str, bool] | None = None
) -> dict[str, bool]:
    """Check the resolved flags against what the REGISTRY requires for this experiment.

    Deliberately NOT checked against the checkpoint's own metadata. The production loader derives every
    capability flag from those same metadata keys (legacy ``corrupted_prior_mix`` metadata supplies the
    historical fallback, while new support flags are read directly), so comparing the resolved flag
    to the key it came from is tautological -- it can never fail and would give false assurance.

    The check that can actually fail compares against the registry's declared production configuration.
    That catches the realistic mistake: handing an editing experiment a de-novo or pre-RingCore
    checkpoint, which would silently score a different legal support.
    """
    flags = resolve_capability_flags(model)

    # Non-tautological on its own: the loader DEFAULTS this to True when the key is absent, so an older
    # checkpoint predating RingCore-V1 arrives with the macro on and is refused here.
    for name, expected in REQUIRED_MACRO_STATE.items():
        actual = getattr(model, name, None)
        if actual is not None and bool(actual) != expected:
            raise EvaluationError(
                f"{name} is {bool(actual)} but production RingCore-V1 requires {expected}: the "
                "whole-ring macro is superseded by compositional cycle_close/cycle_open, and enabling it "
                "changes the legal successor support"
            )

    if required:
        combined = dict(flags)
        macro = getattr(model, "enable_ring_grow_macro", None)
        if macro is not None:
            combined["enable_ring_grow_macro"] = bool(macro)
        mismatched = {
            name: {"required": expected, "actual": combined.get(name)}
            for name, expected in required.items()
            if combined.get(name) != bool(expected)
        }
        if mismatched:
            raise EvaluationError(
                "loaded capabilities do not match the configuration the registry requires for this "
                f"experiment: {mismatched}. This checkpoint would score a different legal support than "
                "the declared production configuration."
            )
    return flags


def read_checkpoint_provenance(path: Path) -> CheckpointProvenance:
    """Metadata + content hash of a checkpoint, without constructing a model."""
    import torch  # noqa: PLC0415  -- heavy import, only needed here

    if not path.is_file():
        raise EvaluationError(f"checkpoint not found: {path}")
    if path.stat().st_size == 0:
        raise EvaluationError(f"checkpoint is empty: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise EvaluationError(f"checkpoint payload is not a mapping: {path}")
    from compose_v4.experiments.p50_completion import (  # noqa: PLC0415
        P50CompletionError,
        validate_p50_completion_member,
    )

    try:
        validate_p50_completion_member(path, payload)
    except P50CompletionError as error:
        raise EvaluationError(str(error)) from error
    corrupted_prior_mix = _historical_optional_bool(
        payload,
        "corrupted_prior_mix",
        default=False,
    )
    return CheckpointProvenance(
        path=str(path),
        sha256=file_sha256(path),
        bytes=path.stat().st_size,
        checkpoint_kind=payload.get("checkpoint_kind"),
        organic_vocabulary=bool(payload.get("organic_vocabulary", False)),
        corrupted_prior_mix=corrupted_prior_mix,
        enable_ring_restates=_historical_optional_bool(
            payload,
            "enable_ring_restates",
            default=corrupted_prior_mix,
        ),
        # The gate persists "enable_cycle_ops" (checkpoint_metadata), NOT "cycle_op_mix".
        enable_cycle_ops=bool(payload.get("enable_cycle_ops", False)),
        # Historical checkpoints implicitly enabled the family.
        enable_ring_system_delete=_historical_optional_bool(
            payload,
            "enable_ring_system_delete",
            default=True,
        ),
        corpus_scope_hash=payload.get("corpus_scope_hash"),
        max_atoms=(
            None if payload.get("max_atoms") is None else int(payload["max_atoms"])
        ),
        bond_representation=(
            None
            if payload.get("bond_representation") is None
            else str(payload["bond_representation"])
        ),
        rate_factorization=(
            None
            if payload.get("rate_factorization") is None
            else str(payload["rate_factorization"])
        ),
    )


def build_support_signature(
    model: Any, flags: dict[str, bool], *, provenance: CheckpointProvenance
) -> SupportSignature:
    """Assemble the full signature that cross-arm comparability is asserted on."""
    # The vocabulary's identity is its (element, valence) CLASS table: that table is what determines
    # which atoms are representable, so it distinguishes ORGANIC (15 classes) from CNOF (4) and is
    # support-determining. An empty value here would let two different vocabularies compare as equal
    # arms, so it is asserted non-empty by the tests rather than silently tolerated.
    vocabulary = getattr(model, "atom_vocabulary", None)
    elements: tuple[str, ...] = ()
    classes = getattr(vocabulary, "classes", None) if vocabulary is not None else None
    if classes:
        elements = tuple(
            f"{pair[0]}:{pair[1]}" if isinstance(pair, (tuple, list)) and len(pair) >= 2 else str(pair)
            for pair in classes
        )
    elif vocabulary is not None:
        for attribute in ("elements", "symbols", "atom_types"):
            candidate = getattr(vocabulary, attribute, None)
            if candidate:
                elements = tuple(str(item) for item in candidate)
                break
    return SupportSignature(
        operator_registry_hash=_operator_registry_hash(),
        capability_flags=tuple((name, flags[name]) for name in CAPABILITY_FLAGS),
        element_vocabulary=elements,
        charge_vocabulary=(-2, -1, 0, 1, 2),
        charge_policy=CHARGE_POLICY_VERSION,
        bond_vocabulary=("none", "single", "double", "triple", "aromatic"),
        aromaticity_policy=provenance.bond_representation,
        max_atoms=provenance.max_atoms,
        valence_policy="declared_element_valence_classes_plus_hydrogen_budget",
        canonicalizer_version="canonical_state_key",
        executor_version=_operator_registry_hash(),
        persistent_slot_schema="slot_stable_v1",
        atom_insert_arity_support=(0, 1),
        embedded_jump_chain_policy="fixed_step_embedded_jump_chain",
        ringcore_configuration=(
            "ringcore_v1_compositional_cycle_ops"
            if flags["enable_cycle_ops"]
            else "cycle_ops_disabled"
        ),
    )


def build_context(
    *,
    checkpoint: Path,
    registry: dict[str, Any],
    experiment_id: str,
    seed: int,
    expected_scope_hash: str | None = None,
) -> EvaluationContext:
    """Construct and validate an evaluation target. Fails loudly on every mismatch."""
    from compose_v4.experiments.registry import experiment as registry_experiment  # noqa: PLC0415

    spec = registry_experiment(registry, experiment_id)  # raises on an unknown id
    if seed not in registry["protocol"]["seeds"]:
        raise EvaluationError(
            f"seed {seed} is not one of the frozen protocol seeds {registry['protocol']['seeds']}"
        )

    required = spec.get("checkpoint")
    if required == "none":
        raise EvaluationError(
            f"experiment {experiment_id} declares checkpoint: none and must not be given one"
        )
    provenance = read_checkpoint_provenance(Path(checkpoint))
    model = _construct_model(Path(checkpoint), expected_scope_hash=expected_scope_hash)
    # The registry declares the production capability configuration for the lineage this experiment
    # depends on; an experiment on the editing prior must not silently accept a de-novo checkpoint.
    lineage = (registry.get("provenance") or {}).get(required) or {}
    flags = validate_capabilities(model, required=lineage.get("required_capabilities"))
    signature = build_support_signature(model, flags, provenance=provenance)

    return EvaluationContext(
        experiment_id=experiment_id,
        seed=seed,
        model=model,
        checkpoint=provenance,
        support_signature=signature,
        registry_protocol_hash=registry["protocol"]["protocol_freeze"]["content_hash"],
        capability_flags=flags,
    )


def _construct_model(checkpoint: Path, *, expected_scope_hash: str | None):
    """Delegate to the production loader; never reconstruct the model independently."""
    try:
        from evaluate_tracelet_rollouts import (  # noqa: PLC0415
            load_factorized_rollout_checkpoint,
        )
    except ImportError:
        try:
            from scripts.evaluate_tracelet_rollouts import (  # noqa: PLC0415
                load_factorized_rollout_checkpoint,
            )
        except ImportError as error:
            raise EvaluationError(
                "the production checkpoint loader is unavailable; add `scripts` to PYTHONPATH. The "
                "evaluator must not reconstruct the model itself."
            ) from error
    try:
        model, _ = load_factorized_rollout_checkpoint(
            checkpoint, expected_scope_hash=expected_scope_hash
        )
    except Exception as error:
        raise EvaluationError(f"could not load checkpoint {checkpoint}: {error}") from error
    return model


def evaluation_envelope(
    context: EvaluationContext,
    *,
    metrics: dict[str, Any] | None = None,
    per_sample_records: str | None = None,
    registry_path: str | None = None,
) -> dict[str, Any]:
    """The stable, additive-only output schema every experiment writes."""
    return {
        "schema": OUTPUT_SCHEMA,
        "schema_version": OUTPUT_SCHEMA_VERSION,
        "experiment_id": context.experiment_id,
        "seed": context.seed,
        "checkpoint": asdict(context.checkpoint),
        "capability_flags": dict(context.capability_flags),
        "support_signature": asdict(context.support_signature),
        "registry": {
            "path": registry_path,
            "protocol_content_hash": context.registry_protocol_hash,
        },
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
        },
        "metrics": dict(metrics or {}),
        "per_sample_records": per_sample_records,
    }


def write_envelope(envelope: dict[str, Any], path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(envelope, indent=2, sort_keys=True) + "\n")
    return path
