"""Reconstruct the exact Process-V2 scratch model an Active8 decision runs on.

WHY A SEPARATE, TINY MODULE
---------------------------
A decision is only reproducible if the model it was decided against is.  The
policy says *what* is admitted; the map/reduce says *where*; this says *which
model*, and it does so by going through the frozen Process-V2 model/process
contract rather than by assembling a collection of Boolean flags at the call
site.  ``build_gate_zero_process_v2_contract()`` is the one place the V2 model
identity is declared, and ``build_semantic_scratch_runtime`` is the one typed
construction boundary, so a caller that wanted a Process-V2 model any other way
would be constructing a second definition of the same thing.

THE ARCHITECTURE IS READ, NOT RETYPED
-------------------------------------
:func:`load_process_v2_active8_model_config` projects the ``model`` block of the
frozen decision-runtime contract.  The values there -- seed, width, depth, mark
dimension, vocabulary width and RingCore catalog fingerprint -- are settled
policy, and restating them in code is the duplication the repository's registry
rule forbids.

WHAT THE RUNTIME IDENTITY IS FOR
--------------------------------
Every Active8 task receipt names a ``model_runtime_identity_sha256``, and the
reducer requires every task to name the same one.  That identity therefore has
to move whenever anything that could change a decision changes: the constructed
architecture, the semantic model identity, the frozen contract, the process
identity, the initial parameter state, the admission policy, and the enumeration
time the marked law is taken at.  All of them are inside the hashed body.

It authorizes nothing.  Constructing a model is not permission to decide with it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from compose_v4.data.editing_v2_process_v2_active8_policy import (
    ProcessV2Active8Policy,
    ProcessV2ProductionCandidateChecker,
    build_process_v2_active8_policy,
    validate_process_v2_active8_policy,
)
from compose_v4.data.editing_v2_process_v2_schema import canonical_sha256
from compose_v4.experiments.editing_gate_zero_semantic_contract import (
    PROCESS_V2_CONTRACT_SCHEMA,
    FrozenGateZeroSemanticContract,
    GateZeroSemanticContractError,
    build_gate_zero_process_v2_contract,
    load_gate_zero_semantic_contract,
)
from compose_v4.experiments.editing_v2_semantic_runtime import (
    SemanticScratchModelConfig,
    SemanticScratchRuntime,
    build_semantic_scratch_runtime,
)
from compose_v4.model.factorized_tracelet_rate_model import (
    PROCESS_V2_ATOM_DELETE_ACTION_SEMANTICS,
    PROCESS_V2_EDITING_PROCESS_SEMANTICS,
)

RUNTIME_IDENTITY_SCHEMA = "compose.data.editing_v2_process_v2_active8_model_runtime"
RUNTIME_IDENTITY_SCHEMA_VERSION = 1

#: The frozen contract that declares the Process-V2 decision architecture.
PRODUCTION_RUNTIME_CONTRACT_PATH = (
    "configs/editing_v2_process_v2_active8_decision_runtime.json"
)
PRODUCTION_MODEL_PROCESS_CONTRACT_PATH = (
    "configs/editing_gate_zero_semantic_model_process_v2.json"
)

DEFAULT_CANDIDATE_TIME = 0.5
DEFAULT_CANDIDATE_CACHE_SIZE = 4096

_MODEL_CONFIG_FIELDS: tuple[str, ...] = (
    "initialization_seed",
    "max_atoms",
    "hidden_dim",
    "message_passing_steps",
    "mark_dim",
    "dtype",
    "atom_vocabulary_class_count",
    "catalog_fingerprint",
)


class ProcessV2Active8RuntimeError(RuntimeError):
    """The Process-V2 Active8 model runtime could not be reconstructed exactly."""


@dataclass(frozen=True)
class ProcessV2Active8Runtime:
    """One deterministic Process-V2 model, its policy, checker and identity."""

    scratch: SemanticScratchRuntime
    policy: ProcessV2Active8Policy
    checker: ProcessV2ProductionCandidateChecker
    candidate_time: float
    candidate_cache_size: int
    identity: dict[str, Any]

    @property
    def model(self) -> Any:
        return self.scratch.model

    @property
    def identity_sha256(self) -> str:
        return str(self.identity["identity_sha256"])


def process_v2_model_process_contract(
    *, repo_root: Path | None = None
) -> FrozenGateZeroSemanticContract:
    """Return the frozen Process-V2 model/process contract.

    Read from the committed artifact when it is present, so a decision names the
    exact bytes it was made under; rebuilt from the production identity API when
    it is not, which is the same body by construction.  Either way the loader
    validates it, so an edited artifact is refused rather than adopted.
    """

    if repo_root is not None:
        path = Path(repo_root) / PRODUCTION_MODEL_PROCESS_CONTRACT_PATH
        if path.is_file():
            contract = load_gate_zero_semantic_contract(path)
            if contract.payload["schema"] != PROCESS_V2_CONTRACT_SCHEMA:
                raise ProcessV2Active8RuntimeError(
                    f"{PRODUCTION_MODEL_PROCESS_CONTRACT_PATH} is not the Process-V2 "
                    "model/process contract"
                )
            return contract
    try:
        payload = build_gate_zero_process_v2_contract()
    except GateZeroSemanticContractError as error:
        raise ProcessV2Active8RuntimeError(
            "the frozen Process-V2 model/process contract cannot be constructed"
        ) from error
    return FrozenGateZeroSemanticContract(
        source=Path(PRODUCTION_MODEL_PROCESS_CONTRACT_PATH),
        payload=payload,
        file_sha256=canonical_sha256(payload),
    )


def load_process_v2_active8_model_config(
    *, repo_root: Path, contract_relative_path: str = PRODUCTION_RUNTIME_CONTRACT_PATH
) -> SemanticScratchModelConfig:
    """Project the frozen decision-runtime contract's architecture block."""

    path = Path(repo_root) / contract_relative_path
    try:
        payload = json.loads(path.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ProcessV2Active8RuntimeError(
            f"the Process-V2 decision-runtime contract is unreadable: {path}"
        ) from error
    model = payload.get("model") if isinstance(payload, dict) else None
    if not isinstance(model, dict) or not set(_MODEL_CONFIG_FIELDS) <= set(model):
        raise ProcessV2Active8RuntimeError(
            f"{contract_relative_path} declares no complete Process-V2 model block"
        )
    return SemanticScratchModelConfig(
        initialization_seed=int(model["initialization_seed"]),
        max_atoms=int(model["max_atoms"]),
        hidden_dim=int(model["hidden_dim"]),
        message_passing_steps=int(model["message_passing_steps"]),
        mark_dim=int(model["mark_dim"]),
        dtype=str(model["dtype"]),
        atom_vocabulary_class_count=int(model["atom_vocabulary_class_count"]),
        catalog_fingerprint=str(model["catalog_fingerprint"]),
    )


def process_v2_active8_model_runtime_identity(
    scratch: SemanticScratchRuntime,
    *,
    policy: ProcessV2Active8Policy,
    candidate_time: float,
    candidate_cache_size: int,
) -> dict[str, Any]:
    """The self-hashed descriptor every Active8 receipt names."""

    if not isinstance(scratch, SemanticScratchRuntime):
        raise ProcessV2Active8RuntimeError(
            "a Process-V2 Active8 runtime identity requires a constructed scratch runtime"
        )
    validate_process_v2_active8_policy(policy)
    if policy.process_identity_sha256 != scratch.process_identity_sha256:
        raise ProcessV2Active8RuntimeError(
            "the constructed model and the Active8 policy bind different process identities"
        )
    body = {
        "schema": RUNTIME_IDENTITY_SCHEMA,
        "schema_version": RUNTIME_IDENTITY_SCHEMA_VERSION,
        "architecture": scratch.architecture.as_payload(),
        "semantic_model_identity": dict(scratch.semantic_model_identity),
        "semantic_model_process_contract_sha256": (
            scratch.semantic_model_process_contract_sha256
        ),
        "process_identity_sha256": scratch.process_identity_sha256,
        "initial_model_state_sha256": scratch.initial_model_state_sha256,
        "policy_sha256": policy.policy_sha256,
        # The marked law is a function of time, so the enumeration time is part
        # of the identity of what was enumerated.
        "candidate_time_hex": float(candidate_time).hex(),
        "candidate_cache_size": int(candidate_cache_size),
    }
    return {**body, "identity_sha256": canonical_sha256(body)}


def validate_process_v2_active8_model_runtime_identity(value: object) -> dict[str, Any]:
    """Refuse a runtime descriptor that does not describe a Process-V2 model.

    A self-hash proves a descriptor is intact, not that it describes the right
    model.  A Process-V1 runtime identity is perfectly self-consistent, so the
    only thing separating it from a Process-V2 one is what it says about the
    process semantics and the atom-delete fiber -- which is precisely the
    expansion Process V2 exists for.  Both are required here by name.
    """

    if not isinstance(value, dict):
        raise ProcessV2Active8RuntimeError("a model runtime identity must be an object")
    identity = dict(value)
    declared = identity.get("identity_sha256")
    body = {key: item for key, item in identity.items() if key != "identity_sha256"}
    if not isinstance(declared, str) or declared != canonical_sha256(body):
        raise ProcessV2Active8RuntimeError(
            "the model runtime identity self-hash disagrees with its body"
        )
    if (
        identity.get("schema") != RUNTIME_IDENTITY_SCHEMA
        or identity.get("schema_version") != RUNTIME_IDENTITY_SCHEMA_VERSION
    ):
        raise ProcessV2Active8RuntimeError(
            "the model runtime identity is not a Process-V2 Active8 runtime descriptor"
        )
    model_identity = identity.get("semantic_model_identity")
    if not isinstance(model_identity, dict):
        raise ProcessV2Active8RuntimeError(
            "the model runtime identity carries no semantic model identity"
        )
    observed = {
        "editing_process_semantics": model_identity.get("editing_process_semantics"),
        "atom_delete_action_semantics": model_identity.get(
            "atom_delete_action_semantics"
        ),
    }
    expected = {
        "editing_process_semantics": PROCESS_V2_EDITING_PROCESS_SEMANTICS,
        "atom_delete_action_semantics": PROCESS_V2_ATOM_DELETE_ACTION_SEMANTICS,
    }
    if observed != expected:
        raise ProcessV2Active8RuntimeError(
            "the model runtime identity describes another editing process: "
            f"expected {expected}, observed {observed}"
        )
    return identity


def build_process_v2_active8_runtime(
    *,
    model_config: SemanticScratchModelConfig,
    repo_root: Path | None = None,
    policy: ProcessV2Active8Policy | None = None,
    candidate_time: float = DEFAULT_CANDIDATE_TIME,
    candidate_cache_size: int = DEFAULT_CANDIDATE_CACHE_SIZE,
) -> ProcessV2Active8Runtime:
    """Build the exact Process-V2 model, checker and runtime identity together.

    Returned as one object because the three are only meaningful together: a
    checker bound to a different model, or an identity computed from a different
    architecture, is exactly the drift the identity exists to catch.
    """

    selected = validate_process_v2_active8_policy(
        policy or build_process_v2_active8_policy()
    )
    contract = process_v2_model_process_contract(repo_root=repo_root)
    scratch = build_semantic_scratch_runtime(model_config, contract)
    checker = ProcessV2ProductionCandidateChecker(
        scratch.model,
        policy=selected,
        cache_size=candidate_cache_size,
        time=candidate_time,
    )
    identity = process_v2_active8_model_runtime_identity(
        scratch,
        policy=selected,
        candidate_time=candidate_time,
        candidate_cache_size=candidate_cache_size,
    )
    return ProcessV2Active8Runtime(
        scratch=scratch,
        policy=selected,
        checker=checker,
        candidate_time=float(candidate_time),
        candidate_cache_size=int(candidate_cache_size),
        identity=identity,
    )


def runtime_identity_resolver(runtime: ProcessV2Active8Runtime):
    """A resolver a worker can call with the live model and be answered exactly.

    The worker re-resolves the identity from the model it was handed rather than
    trusting the plan, so a model swapped after planning is refused.  The
    resolver therefore checks object identity: a different model object cannot be
    answered with this runtime's descriptor.
    """

    def resolve(model: Any) -> dict[str, Any]:
        if model is not runtime.scratch.model:
            raise ProcessV2Active8RuntimeError(
                "the Process-V2 Active8 runtime was asked to describe another model"
            )
        return dict(runtime.identity)

    return resolve


__all__ = [
    "DEFAULT_CANDIDATE_CACHE_SIZE",
    "DEFAULT_CANDIDATE_TIME",
    "PRODUCTION_MODEL_PROCESS_CONTRACT_PATH",
    "PRODUCTION_RUNTIME_CONTRACT_PATH",
    "RUNTIME_IDENTITY_SCHEMA",
    "RUNTIME_IDENTITY_SCHEMA_VERSION",
    "ProcessV2Active8Runtime",
    "ProcessV2Active8RuntimeError",
    "build_process_v2_active8_runtime",
    "validate_process_v2_active8_model_runtime_identity",
    "load_process_v2_active8_model_config",
    "process_v2_active8_model_runtime_identity",
    "process_v2_model_process_contract",
    "runtime_identity_resolver",
]
