"""Exact scratch runtime for the frozen Editing-V2 semantic model/process.

Gate 0, T1, and later bounded pilots must not reconstruct the semantic model
with independent collections of Boolean flags.  This module is the typed
construction boundary: it consumes the validated semantic model/process
contract, constructs the deterministic broad-organic scratch model, and
returns the exact architecture and initial-state identity used by downstream
artifacts.

The factory contains no optimizer, panel policy, numeric threshold, sampling
law, or training authority.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch

from compose_v4.chem.molecular_graph import ORGANIC_VOCABULARY
from compose_v4.experiments.editing_gate_zero_runtime import (
    build_production_ringcore_catalog,
)
from compose_v4.experiments.editing_gate_zero_semantic_contract import (
    FrozenGateZeroSemanticContract,
    validate_gate_zero_semantic_contract,
)
from compose_v4.experiments.editing_p50_gate import state_dict_semantic_sha256
from compose_v4.model.factorized_tracelet_rate_model import (
    ATOM_DELETE_ACTION_SEMANTICS,
    LEGACY_ATOM_DELETE_ACTION_SEMANTICS,
    FactorizedTraceletRateModel,
)
from compose_v4.rewrite.typed_ring_catalog import (
    TypedRingCatalog,
    ring_catalog_fingerprint,
)

ATOM_DELETE_ACTION_SEMANTICS_FIELD = "atom_delete_action_semantics"


class SemanticScratchRuntimeError(RuntimeError):
    """The requested scratch runtime differs from the frozen semantic model."""


def model_identity_atom_delete_action_semantics(
    model_identity: Mapping[str, Any],
) -> str:
    """Return the atom-delete mode a persisted ``model_identity`` reconstructs with.

    An ABSENT key means the historical legacy mode, so a checkpoint or contract
    written before Process V2 reconstructs byte-identically -- absence is the only
    reading that keeps historical artifacts loadable.  A PRESENT key must name a
    declared mode: a stale or unknown string raises here rather than silently
    degrading to legacy, which would reconstruct a differently supported model
    under the persisted identity's name.
    """

    if not isinstance(model_identity, Mapping):
        raise TypeError("model identity must be a mapping")
    if ATOM_DELETE_ACTION_SEMANTICS_FIELD not in model_identity:
        return LEGACY_ATOM_DELETE_ACTION_SEMANTICS
    value = model_identity[ATOM_DELETE_ACTION_SEMANTICS_FIELD]
    if value not in ATOM_DELETE_ACTION_SEMANTICS:
        raise SemanticScratchRuntimeError(
            "persisted model identity names an unknown atom-delete action semantics: "
            f"{value!r}; expected one of {ATOM_DELETE_ACTION_SEMANTICS}"
        )
    return str(value)


@dataclass(frozen=True, slots=True)
class SemanticScratchModelConfig:
    """Typed architecture and initialization coordinates for one scratch model."""

    initialization_seed: int
    max_atoms: int
    hidden_dim: int
    message_passing_steps: int
    mark_dim: int
    dtype: str
    atom_vocabulary_class_count: int
    catalog_fingerprint: str

    def __post_init__(self) -> None:
        for field in (
            "initialization_seed",
            "max_atoms",
            "hidden_dim",
            "message_passing_steps",
            "mark_dim",
            "atom_vocabulary_class_count",
        ):
            value = getattr(self, field)
            if type(value) is not int or value <= 0:
                raise ValueError(f"semantic scratch {field} must be a positive integer")
        if self.dtype != "torch.float32":
            raise ValueError("current semantic scratch runtime requires torch.float32")
        if (
            not isinstance(self.catalog_fingerprint, str)
            or len(self.catalog_fingerprint) != 16
            or any(
                character not in "0123456789abcdef"
                for character in self.catalog_fingerprint
            )
        ):
            raise ValueError(
                "semantic scratch catalog_fingerprint must be 16 lowercase hex characters"
            )


@dataclass(frozen=True, slots=True)
class SemanticScratchArchitecture:
    """Observed architecture identity after exact model construction."""

    max_atoms: int
    hidden_dim: int
    message_passing_steps: int
    mark_dim: int
    dtype: str
    parameter_dtypes: tuple[str, ...]
    atom_vocabulary_class_count: int
    catalog_fingerprint: str
    operator_capability_fingerprint: str

    def as_payload(self) -> dict[str, object]:
        return {
            "max_atoms": self.max_atoms,
            "hidden_dim": self.hidden_dim,
            "message_passing_steps": self.message_passing_steps,
            "mark_dim": self.mark_dim,
            "dtype": self.dtype,
            "parameter_dtypes": list(self.parameter_dtypes),
            "atom_vocabulary_class_count": self.atom_vocabulary_class_count,
            "catalog_fingerprint": self.catalog_fingerprint,
            "operator_capability_fingerprint": (self.operator_capability_fingerprint),
        }


@dataclass(frozen=True, slots=True)
class SemanticScratchRuntime:
    """One deterministic model plus its exact frozen semantic identities."""

    model: FactorizedTraceletRateModel
    config: SemanticScratchModelConfig
    architecture: SemanticScratchArchitecture
    semantic_model_identity: dict[str, Any]
    semantic_model_process_contract_sha256: str
    process_identity_sha256: str
    initial_model_state_sha256: str


def _semantic_identity(
    contract: FrozenGateZeroSemanticContract,
) -> tuple[dict[str, Any], str, str]:
    if not isinstance(contract, FrozenGateZeroSemanticContract):
        raise TypeError(
            "semantic scratch runtime requires a validated model/process contract"
        )
    payload = validate_gate_zero_semantic_contract(contract.payload)
    if payload != contract.payload:
        raise SemanticScratchRuntimeError(
            "semantic model/process contract changed during validation"
        )
    return (
        dict(payload["model_identity"]),
        str(payload["contract_sha256"]),
        str(payload["process_identity_sha256"]),
    )


def semantic_runtime_model_identity(
    model: FactorizedTraceletRateModel,
) -> dict[str, Any]:
    """Return the model identity a constructed semantic model persists as.

    This is the WRITE side of the same contract
    :func:`model_identity_atom_delete_action_semantics` reads.  The atom-delete
    key is emitted only when the model is not on the legacy mode, so a Process-V1
    model persists exactly the field set it always did and a Process-V2 model
    persists the mode its expanded delete fiber is defined by.
    """

    capabilities = model.operator_capabilities
    identity: dict[str, Any] = {
        "mark_dim": model.mark_dim,
        "operator_capability_fingerprint": capabilities.fingerprint(),
        "compute_ring_grow_support": capabilities.compute_ring_grow_support,
        "compute_ring_restates": capabilities.compute_ring_restates,
        "compute_cyclic_graft": capabilities.compute_cyclic_graft,
        "compute_ring_opening": capabilities.compute_ring_opening,
        "compute_ring_system_delete": capabilities.compute_ring_system_delete,
        "enable_cycle_ops": model.enable_cycle_ops,
        "use_aromatic_bond_view": True,
        "editing_process_semantics": capabilities.editing_process_semantics,
        "atom_restate_action_semantics": (capabilities.atom_restate_action_semantics),
        "ring_restate_scorer_mode": capabilities.ring_restate_scorer_mode,
        "cycle_close_action_semantics": (capabilities.cycle_close_action_semantics),
        "cycle_open_action_semantics": capabilities.cycle_open_action_semantics,
        "cycle_open_scorer_mode": model.cycle_open_scorer_mode,
    }
    if capabilities.atom_delete_action_semantics != LEGACY_ATOM_DELETE_ACTION_SEMANTICS:
        identity[ATOM_DELETE_ACTION_SEMANTICS_FIELD] = (
            capabilities.atom_delete_action_semantics
        )
    return identity


def _require_model_identity(
    model: FactorizedTraceletRateModel,
    expected: Mapping[str, Any],
) -> None:
    observed = semantic_runtime_model_identity(model)
    # The atom-delete mode is compared on BOTH sides under the absent-means-legacy reading, so a
    # historical identity that never carried the key still matches a legacy-delete model exactly, and a
    # Process-V2 identity can never be satisfied by a legacy-delete model.
    expected_identity = dict(expected)
    expected_identity[ATOM_DELETE_ACTION_SEMANTICS_FIELD] = (
        model_identity_atom_delete_action_semantics(expected_identity)
    )
    observed[ATOM_DELETE_ACTION_SEMANTICS_FIELD] = (
        model.operator_capabilities.atom_delete_action_semantics
    )
    if observed != expected_identity:
        mismatches = {
            field: {
                "expected": expected_identity.get(field),
                "observed": observed.get(field),
            }
            for field in sorted(set(expected_identity) | set(observed))
            if expected_identity.get(field) != observed.get(field)
        }
        raise SemanticScratchRuntimeError(
            f"constructed model differs from semantic identity: {mismatches}"
        )


def build_model_from_semantic_model_identity(
    model_identity: Mapping[str, Any],
    *,
    catalog: TypedRingCatalog,
    hidden_dim: int,
    message_passing_steps: int,
    mark_dim: int,
    initialization_seed: int,
) -> FactorizedTraceletRateModel:
    """Reconstruct the exact semantic model one persisted identity describes.

    One reconstruction path, shared by the production scratch runtime and by any
    caller that has to reload a persisted identity, so a save/load round trip
    cannot drift from the runtime it is supposed to reproduce.
    """

    # Isolate deterministic initialization from the caller's RNG stream.  This
    # produces the same CPU state as manual_seed while keeping orchestration and
    # later sampling streams independent of model construction.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(initialization_seed)
        model = FactorizedTraceletRateModel(
            catalog,
            hidden_dim=hidden_dim,
            message_passing_steps=message_passing_steps,
            mark_dim=mark_dim,
            enable_ring_restates=bool(model_identity["compute_ring_restates"]),
            enable_cyclic_graft=bool(model_identity["compute_cyclic_graft"]),
            enable_heteroatom_scan=True,
            enable_ring_opening=bool(model_identity["compute_ring_opening"]),
            enable_cycle_ops=bool(model_identity["enable_cycle_ops"]),
            cycle_open_scorer_mode=str(model_identity["cycle_open_scorer_mode"]),
            editing_process_semantics=str(model_identity["editing_process_semantics"]),
            atom_restate_action_semantics=str(
                model_identity["atom_restate_action_semantics"]
            ),
            ring_restate_scorer_mode=str(model_identity["ring_restate_scorer_mode"]),
            cycle_close_action_semantics=str(
                model_identity["cycle_close_action_semantics"]
            ),
            cycle_open_action_semantics=str(
                model_identity["cycle_open_action_semantics"]
            ),
            # Restored from the persisted identity; absent means the legacy mode, so a semantic
            # identity written before Process V2 reconstructs exactly the model it always did.
            atom_delete_action_semantics=(
                model_identity_atom_delete_action_semantics(model_identity)
            ),
            enable_ring_grow_macro=bool(model_identity["compute_ring_grow_support"]),
            enable_ring_system_delete=bool(
                model_identity["compute_ring_system_delete"]
            ),
            atom_vocabulary=ORGANIC_VOCABULARY,
        ).to(dtype=torch.float32)
    model.eval()
    return model


def build_semantic_scratch_runtime(
    config: SemanticScratchModelConfig,
    semantic_contract: FrozenGateZeroSemanticContract,
) -> SemanticScratchRuntime:
    """Build the exact deterministic semantic Editing-V2 scratch runtime."""

    if not isinstance(config, SemanticScratchModelConfig):
        raise TypeError("config must be SemanticScratchModelConfig")
    model_identity, contract_sha256, process_sha256 = _semantic_identity(
        semantic_contract
    )
    if config.mark_dim != model_identity["mark_dim"]:
        raise SemanticScratchRuntimeError(
            "scratch mark dimension differs from semantic model identity"
        )
    if config.atom_vocabulary_class_count != len(ORGANIC_VOCABULARY.classes):
        raise SemanticScratchRuntimeError(
            "scratch broad-organic vocabulary width differs from production"
        )

    catalog = build_production_ringcore_catalog(max_atoms=config.max_atoms)
    if ring_catalog_fingerprint(catalog) != config.catalog_fingerprint:
        raise SemanticScratchRuntimeError(
            "scratch RingCore catalog differs from the requested architecture"
        )

    model = build_model_from_semantic_model_identity(
        model_identity,
        catalog=catalog,
        hidden_dim=config.hidden_dim,
        message_passing_steps=config.message_passing_steps,
        mark_dim=config.mark_dim,
        initialization_seed=config.initialization_seed,
    )
    _require_model_identity(model, model_identity)

    state = model.state_dict()
    architecture = SemanticScratchArchitecture(
        max_atoms=config.max_atoms,
        hidden_dim=model.hidden_dim,
        message_passing_steps=model.message_passing_steps,
        mark_dim=model.mark_dim,
        dtype=str(next(model.parameters()).dtype),
        parameter_dtypes=tuple(
            sorted({str(tensor.dtype) for tensor in state.values()})
        ),
        atom_vocabulary_class_count=len(model.atom_vocabulary.classes),
        catalog_fingerprint=ring_catalog_fingerprint(model.ring_catalog),
        operator_capability_fingerprint=(model.operator_capabilities.fingerprint()),
    )
    expected_architecture = SemanticScratchArchitecture(
        max_atoms=config.max_atoms,
        hidden_dim=config.hidden_dim,
        message_passing_steps=config.message_passing_steps,
        mark_dim=config.mark_dim,
        dtype=config.dtype,
        parameter_dtypes=(config.dtype,),
        atom_vocabulary_class_count=config.atom_vocabulary_class_count,
        catalog_fingerprint=config.catalog_fingerprint,
        operator_capability_fingerprint=str(
            model_identity["operator_capability_fingerprint"]
        ),
    )
    if architecture != expected_architecture:
        raise SemanticScratchRuntimeError(
            "constructed scratch architecture differs from the exact runtime config"
        )
    return SemanticScratchRuntime(
        model=model,
        config=config,
        architecture=architecture,
        semantic_model_identity=model_identity,
        semantic_model_process_contract_sha256=contract_sha256,
        process_identity_sha256=process_sha256,
        initial_model_state_sha256=state_dict_semantic_sha256(state),
    )


__all__ = [
    "ATOM_DELETE_ACTION_SEMANTICS_FIELD",
    "SemanticScratchArchitecture",
    "SemanticScratchModelConfig",
    "SemanticScratchRuntime",
    "SemanticScratchRuntimeError",
    "build_model_from_semantic_model_identity",
    "build_semantic_scratch_runtime",
    "model_identity_atom_delete_action_semantics",
    "semantic_runtime_model_identity",
]
