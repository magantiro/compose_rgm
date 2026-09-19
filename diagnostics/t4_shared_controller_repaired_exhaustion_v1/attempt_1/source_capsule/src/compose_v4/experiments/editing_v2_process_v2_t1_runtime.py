"""Minimal Process-V2 T1 prepared-input and runtime boundary.

The Process-V2 Active8 and Gate-0 chain deliberately stops before molecular
successor compilation.  This module performs that one missing CPU operation for
the bounded, unique-state T1 panel:

* reopen the exact persistent-slot source and target states through the
  Process-V2 cache/rebind path;
* enumerate the complete productive canonical-successor partition on CPU;
* verify the selected teacher action and exact successor are present;
* publish restart-safe plan, leaf, prepared-input, and completion artifacts; and
* expose a small runtime view consumed by the existing successor-level T1
  optimization core.

It does not train, score, select a checkpoint, grant P50 authority, or rebuild
any upstream corpus artifact.  The map boundary is the already-authenticated
Active8 source chunk, so each selected chunk is reopened at most once per leaf.

The one-pass compiler and validators protect against incomplete, stale, or
post-seal-corrupted artifacts.  Content hashes do not establish correctness
against a fully resealed trusted-root replacement or a deterministic compiler
defect; those require separate bounded independent-oracle tests.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from types import MappingProxyType, SimpleNamespace
from typing import Any

import torch

from compose_v4.chem.molecular_graph import ORGANIC_VOCABULARY
from compose_v4.chem.persistent_state_identity import persistent_slot_state_sha256
from compose_v4.data.editing_v2_process_v2_active8_plan import (
    MODEL_RUNTIME_FIELDS,
    model_runtime_descriptor,
)
from compose_v4.data.editing_v2_process_v2_schema import (
    ProcessV2SchemaError,
    authority_false_block,
    canonical_bytes,
    canonical_sha256,
    require_authority_false,
    verify_self_hash,
)
from compose_v4.data.immutable_artifact import ImmutableArtifactError, write_bytes_if_absent
from compose_v4.experiments.editing_gate_zero_semantic_contract import (
    load_gate_zero_semantic_contract,
)
from compose_v4.experiments.editing_p50_gate import state_dict_semantic_sha256
from compose_v4.experiments.editing_v2_process_v2_contract_chain import (
    ACTIVE8_DECISION_RUNTIME,
    DEVELOPMENT_CELL_ROLES,
    GATE_ZERO_MODEL_PROCESS_V2,
    T1_CAPACITY_POLICY,
    load_process_v2_chain_artifact,
)
from compose_v4.experiments.editing_v2_process_v2_t1_panel import (
    ProcessV2T1PanelError,
    ProcessV2T1Source,
    resolve_process_v2_t1_entries,
    validate_process_v2_t1_panel,
)
from compose_v4.experiments.editing_v2_semantic_runtime import (
    SemanticScratchModelConfig,
    SemanticScratchRuntime,
    build_semantic_scratch_runtime,
    model_identity_atom_delete_action_semantics,
)
from compose_v4.experiments.editing_v2_semantic_t1_prepared_inputs import (
    compiled_successor_map_from_payload,
    compiled_successor_map_payload,
)
from compose_v4.experiments.factorized_successor_training import (
    CompiledStateSuccessorMap,
    SuccessorTrainingError,
    TeacherSuccessorFiber,
    compile_state_successor_map,
    require_exact_successor_action_identity,
    rewrite_action_codec_sha256,
    teacher_successor_fiber_from_exact_digest,
)
from compose_v4.model.contextual_ring_restate_rate_model import (
    CONTEXTUAL_RING_RESTATE_SCORER_MODE,
    ContextualRingRestateFactorizedTraceletRateModel,
)
from compose_v4.rewrite import action_codec_v4
from compose_v4.rewrite.trace_shard import decode_state, encode_state

PLAN_SCHEMA = "compose.editing_v2.process_v2_t1_prepared_plan"
PLAN_SCHEMA_VERSION = 2
PLAN_STATUS = "PROCESS_V2_T1_PREPARED_PLAN_NO_DOWNSTREAM_AUTHORITY"
PLAN_FILENAME = "PROCESS_V2_T1_PREPARED_PLAN.json"
LEAF_SCHEMA = "compose.editing_v2.process_v2_t1_prepared_leaf"
LEAF_SCHEMA_VERSION = 1
LEAF_STATUS = "PROCESS_V2_T1_PREPARED_LEAF_NO_DOWNSTREAM_AUTHORITY"
LEAF_FILENAME = "PROCESS_V2_T1_PREPARED_LEAF.json"
PREPARED_SCHEMA = "compose.editing_v2.process_v2_t1_prepared_inputs"
PREPARED_SCHEMA_VERSION = 3
PREPARED_STATUS = "PROCESS_V2_T1_EXACT_SUCCESSOR_INPUTS_NO_DOWNSTREAM_AUTHORITY"
PREPARED_FILENAME = "PROCESS_V2_T1_PREPARED_INPUTS.json"
COMPLETION_SCHEMA = "compose.editing_v2.process_v2_t1_prepared_completion"
COMPLETION_SCHEMA_VERSION = 1
COMPLETION_STATUS = "PROCESS_V2_T1_PREPARED_COMPLETE_NO_DOWNSTREAM_AUTHORITY"
COMPLETION_FILENAME = "PROCESS_V2_T1_PREPARED_COMPLETE.json"
TASKS_DIRNAME = "tasks"

_PREPARED_ENTRY_FIELDS = frozenset(
    {
        "panel_entry_sha256",
        "model_family",
        "capability_cell_id",
        "support_time_hex",
        "source_state_sha256",
        "target_state_sha256",
        "successor_canonical_key",
        "teacher_action_sha256",
        "objective_coefficient",
        "raw_mark_count",
        "canonical_successor_count",
        "production_successor_alias_multiplicity",
        "exact_state",
        "successor_partition",
        "productive_alias_count",
        "virtual_alias_count",
        "model_scores_or_probabilities_stored",
        "hazard_included",
        "entry_sha256",
    }
)

_IMPLEMENTATION_FILES = (
    "src/compose_v4/experiments/editing_v2_process_v2_t1_runtime.py",
    "src/compose_v4/experiments/editing_v2_process_v2_t1_panel.py",
    "src/compose_v4/experiments/factorized_successor_training.py",
    "src/compose_v4/experiments/editing_v2_semantic_t1_prepared_inputs.py",
    "src/compose_v4/experiments/editing_v2_semantic_runtime.py",
    "src/compose_v4/model/contextual_ring_restate_rate_model.py",
    "src/compose_v4/model/factorized_tracelet_rate_model.py",
    "src/compose_v4/model/relational_reroute_rate_model.py",
)

# Exact score-free T1 preparation produced before the scorer-only Process-V2
# repair.  These values are an allowlist, not a general stale-artifact escape.
# The bridge below accepts no other predecessor and still validates every
# canonical entry, successor partition, and physical file hash.
_SCORE_REVISION_PREDECESSOR = MappingProxyType(
    {
        "prepared_completion_sha256": (
            "a93c0b65b08c6b7726deeae727213b7145b88407830f95563c5060b86c4fc865"
        ),
        "prepared_completion_file_sha256": (
            "7d46cce0aaa0cf1534f9ae49a63690f7c5cef5dbc57a74c4735516ba2ff5de66"
        ),
        "prepared_artifact_sha256": (
            "39c264c7b6f370b983d46301ab299721e6b2ded4e3eb350ef8f27b76e1f910f2"
        ),
        "prepared_file_sha256": (
            "3b46192fba2f83fe1ff2d82b4804c4145d222f16b7a76bba43486156f294c16f"
        ),
        "prepared_implementation_sha256": (
            "867bc894d14c3b2cb80f4ddd5f97eeb0cf7252ee4d04001090799a573fd871c1"
        ),
        "initial_model_state_sha256": (
            "f721684555a7f2ec1fafba422a5cb73334c19f832738aae1fa1ce82168efcfb6"
        ),
        "model_identity_sha256": (
            "f34b0207da8d1df50686b97e7e691574bf09025a33e7cfcac525475c5147fc52"
        ),
        "semantic_model_process_contract_sha256": (
            "971a124a3d790e6d5435be5651d9d3f324ea7407cfd0e82944290ffaad138d3a"
        ),
    }
)

_SCORE_INDEPENDENT_MODEL_RUNTIME_FIELDS = (
    "atom_vocabulary_class_count",
    "catalog_fingerprint",
    "dtype",
    "hidden_dim",
    "initialization_seed",
    "mark_dim",
    "max_atoms",
    "message_passing_steps",
    "operator_capability_fingerprint",
)


class ProcessV2T1RuntimeError(RuntimeError):
    """A Process-V2 T1 prerequisite or exact successor partition is invalid."""


_RING_RESTATE_SCORE_REVISION_FIELD = "ring_restate_context_scorer_mode"


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _require_sha(value: object, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ProcessV2T1RuntimeError(f"{field} must be a full lowercase SHA-256")
    return value


def _load_canonical(path: Path, *, label: str) -> tuple[dict[str, Any], bytes]:
    try:
        raw = Path(path).read_bytes()
        value = json.loads(raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ProcessV2T1RuntimeError(f"{label} is unreadable: {path}") from error
    if not isinstance(value, dict) or raw != canonical_bytes(value) + b"\n":
        raise ProcessV2T1RuntimeError(f"{label} is not canonical newline-framed JSON")
    return value, raw


def _implementation_sha256(repo_root: Path) -> str:
    root = Path(repo_root).resolve()
    rows: list[dict[str, str]] = []
    for relative in _IMPLEMENTATION_FILES:
        path = root / relative
        if not path.is_file():
            raise ProcessV2T1RuntimeError(f"T1 implementation source is absent: {relative}")
        rows.append({"path": relative, "file_sha256": _file_sha256(path)})
    return canonical_sha256(rows)


def _validate_source_revision(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    revision = dict(value)
    expected_fields = {
        "schema",
        "schema_version",
        "commit",
        "tree",
        "image_revision_sha256",
        "serialized_source_inventory_sha256",
        "source_revision_sha256",
    }
    if set(revision) != expected_fields:
        raise ProcessV2T1RuntimeError("T1 source revision fields disagree")
    body = {key: item for key, item in revision.items() if key != "source_revision_sha256"}
    if (
        revision["schema"] != "compose.editing_v2.process_v2_t1_authenticated_image_revision"
        or revision["schema_version"] != 1
        or any(
            not isinstance(revision[field], str)
            or len(revision[field]) != 40
            or any(character not in "0123456789abcdef" for character in revision[field])
            for field in ("commit", "tree")
        )
        or any(
            not isinstance(revision[field], str)
            or len(revision[field]) != 64
            or any(character not in "0123456789abcdef" for character in revision[field])
            for field in (
                "image_revision_sha256",
                "serialized_source_inventory_sha256",
                "source_revision_sha256",
            )
        )
        or revision["source_revision_sha256"] != canonical_sha256(body)
    ):
        raise ProcessV2T1RuntimeError("T1 source revision is not an authenticated image binding")
    return revision


def load_process_v2_t1_capacity_policy(
    path: Path, *, repo_root: Path
) -> tuple[dict[str, Any], str]:
    """Load only the current Process-V2 sparse capacity protocol."""

    source = Path(path).resolve()
    expected = (Path(repo_root) / T1_CAPACITY_POLICY).resolve()
    if source != expected:
        raise ProcessV2T1RuntimeError("T1 capacity policy path is not the Process-V2 contract")
    policy = load_process_v2_chain_artifact(T1_CAPACITY_POLICY, repo_root=repo_root)
    if (
        policy.get("schema") != "compose.editing_v2.process_v2_t1_capacity_policy"
        or tuple(policy.get("required_families", ()))
        != (
            "atom_insert",
            "atom_delete",
            "atom_restate",
            "bond_reorder",
            "bond_reroute",
            "cycle_insert",
            "cycle_attach",
            "ring_system_restate",
        )
        or policy.get("panel_kind") != "unique_state_single_target_canonical_successor_capacity"
        or policy.get("hazard_included") is not False
        or policy.get("training_authorized") is not False
        or policy.get("bounded_p50_authorized") is not False
        or policy.get("optimization", {}).get("trajectory_evaluation")
        != "step_zero_and_report_points_only"
    ):
        raise ProcessV2T1RuntimeError("T1 capacity policy is outside Process-V2 T1 scope")
    return {**policy, "policy_sha256": policy["contract_sha256"]}, _file_sha256(source)


def _required_editing_cell_ids(roles: Mapping[str, Any]) -> set[str]:
    values = roles.get("required_cell_ids")
    if (
        not isinstance(values, list)
        or not values
        or any(not isinstance(value, str) or not value for value in values)
        or len(values) != len(set(values))
    ):
        raise ProcessV2T1RuntimeError("T1 development role artifact lacks required cell ids")
    return set(values)


def _scratch_runtime_from_descriptor(
    descriptor: Mapping[str, Any], *, repo_root: Path
) -> tuple[SemanticScratchRuntime, dict[str, Any]]:
    runtime_contract = load_process_v2_chain_artifact(ACTIVE8_DECISION_RUNTIME, repo_root=repo_root)
    semantic_path = Path(repo_root) / GATE_ZERO_MODEL_PROCESS_V2
    semantic = load_gate_zero_semantic_contract(semantic_path)
    expected_parent = runtime_contract["parents"]["semantic_model_process"]
    if (
        semantic.file_sha256 != expected_parent["physical"]["sha256"]
        or semantic.sha256 != expected_parent["semantic"]["sha256"]
    ):
        raise ProcessV2T1RuntimeError("Process-V2 semantic model contract bytes disagree")
    config = SemanticScratchModelConfig(
        initialization_seed=int(descriptor["initialization_seed"]),
        max_atoms=int(descriptor["max_atoms"]),
        hidden_dim=int(descriptor["hidden_dim"]),
        message_passing_steps=int(descriptor["message_passing_steps"]),
        mark_dim=int(descriptor["mark_dim"]),
        dtype=str(descriptor["dtype"]),
        atom_vocabulary_class_count=int(descriptor["atom_vocabulary_class_count"]),
        catalog_fingerprint=str(descriptor["catalog_fingerprint"]),
    )
    runtime = build_semantic_scratch_runtime(config, semantic)
    observed = model_runtime_descriptor(runtime)
    if observed != dict(descriptor) or observed[
        "initial_model_state_sha256"
    ] != state_dict_semantic_sha256(runtime.model.state_dict()):
        raise ProcessV2T1RuntimeError("scratch model differs from its bound descriptor")
    binding = {
        "active8_decision_runtime_file_sha256": _file_sha256(
            Path(repo_root) / ACTIVE8_DECISION_RUNTIME
        ),
        "active8_decision_runtime_sha256": runtime_contract["contract_sha256"],
        "semantic_model_process_file_sha256": semantic.file_sha256,
        "semantic_model_process_sha256": semantic.sha256,
        "model_runtime": observed,
    }
    return runtime, binding


def _scratch_runtime_for_score_revision(
    predecessor_descriptor: Mapping[str, Any],
    *,
    repo_root: Path,
    materialized_state: Mapping[str, Any] | None = None,
) -> tuple[SemanticScratchRuntime, dict[str, Any], dict[str, Any]]:
    """Build the current scorer on one exact predecessor's support geometry.

    ``materialized_state`` supplies the two state dicts instead of re-deriving
    them from ``initialization_seed``.  It changes WHERE the weights come from
    and nothing else: every assertion below still runs, so the frozen
    ``initial_model_state_sha256`` is still enforced -- it is simply verified
    against loaded bytes rather than against a reconstruction.

    That is a STRICTER check, not a weaker one.  Seed reconstruction proves the
    running environment can regenerate the state; loading and verifying proves
    the state actually in use IS the frozen one.

    It exists because the reconstruction is only satisfiable on the platform
    that produced the constant.  PyTorch's CPU ``normal_`` fill is vectorized
    per architecture and reproducible only on the same platform, so a macOS
    arm64 host on the identical pinned versions rebuilds this architecture --
    118 parameters, zero shape mismatches, same seed, same ring catalog -- with
    different numbers.  Pinning cannot fix that; it is not a version
    difference.  Overwriting the expected constant instead would be the
    ``neutralize_catalog_drift()`` antipattern and would split-brain the corpus,
    since the already-compiled rows were scored under the frozen state.
    """

    predecessor = dict(predecessor_descriptor)
    if tuple(sorted(predecessor)) != MODEL_RUNTIME_FIELDS:
        raise ProcessV2T1RuntimeError(
            "score-revision predecessor model descriptor fields disagree"
        )
    frozen = _SCORE_REVISION_PREDECESSOR
    for field in (
        "initial_model_state_sha256",
        "model_identity_sha256",
        "semantic_model_process_contract_sha256",
    ):
        if predecessor[field] != frozen[field]:
            raise ProcessV2T1RuntimeError(
                f"score-revision predecessor {field} is not allowlisted"
            )

    semantic_path = Path(repo_root) / GATE_ZERO_MODEL_PROCESS_V2
    semantic = load_gate_zero_semantic_contract(semantic_path)
    runtime_contract = load_process_v2_chain_artifact(
        ACTIVE8_DECISION_RUNTIME, repo_root=repo_root
    )
    expected_parent = runtime_contract["parents"]["semantic_model_process"]
    if (
        semantic.file_sha256 != expected_parent["physical"]["sha256"]
        or semantic.sha256 != expected_parent["semantic"]["sha256"]
    ):
        raise ProcessV2T1RuntimeError("Process-V2 semantic model contract bytes disagree")

    config = SemanticScratchModelConfig(
        initialization_seed=int(predecessor["initialization_seed"]),
        max_atoms=int(predecessor["max_atoms"]),
        hidden_dim=int(predecessor["hidden_dim"]),
        message_passing_steps=int(predecessor["message_passing_steps"]),
        mark_dim=int(predecessor["mark_dim"]),
        dtype=str(predecessor["dtype"]),
        atom_vocabulary_class_count=int(predecessor["atom_vocabulary_class_count"]),
        catalog_fingerprint=str(predecessor["catalog_fingerprint"]),
    )
    predecessor_runtime = build_semantic_scratch_runtime(config, semantic)
    if materialized_state is not None:
        predecessor_runtime.model.load_state_dict(
            materialized_state["predecessor"], strict=True
        )
    predecessor_identity = dict(predecessor_runtime.semantic_model_identity)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(config.initialization_seed)
        model = ContextualRingRestateFactorizedTraceletRateModel(
            predecessor_runtime.model.ring_catalog,
            hidden_dim=config.hidden_dim,
            message_passing_steps=config.message_passing_steps,
            mark_dim=config.mark_dim,
            enable_ring_restates=bool(
                predecessor_identity["compute_ring_restates"]
            ),
            enable_cyclic_graft=bool(predecessor_identity["compute_cyclic_graft"]),
            enable_heteroatom_scan=True,
            enable_ring_opening=bool(predecessor_identity["compute_ring_opening"]),
            enable_cycle_ops=bool(predecessor_identity["enable_cycle_ops"]),
            cycle_open_scorer_mode=str(
                predecessor_identity["cycle_open_scorer_mode"]
            ),
            editing_process_semantics=str(
                predecessor_identity["editing_process_semantics"]
            ),
            atom_restate_action_semantics=str(
                predecessor_identity["atom_restate_action_semantics"]
            ),
            ring_restate_scorer_mode=str(
                predecessor_identity["ring_restate_scorer_mode"]
            ),
            cycle_close_action_semantics=str(
                predecessor_identity["cycle_close_action_semantics"]
            ),
            cycle_open_action_semantics=str(
                predecessor_identity["cycle_open_action_semantics"]
            ),
            atom_delete_action_semantics=(
                model_identity_atom_delete_action_semantics(predecessor_identity)
            ),
            enable_ring_grow_macro=bool(
                predecessor_identity["compute_ring_grow_support"]
            ),
            enable_ring_system_delete=bool(
                predecessor_identity["compute_ring_system_delete"]
            ),
            atom_vocabulary=ORGANIC_VOCABULARY,
        ).to(dtype=torch.float32)
    if materialized_state is not None:
        model.load_state_dict(materialized_state["revised"], strict=True)
    predecessor_state = predecessor_runtime.model.state_dict()
    if set(predecessor_state) - {"graft_relation_head.weight"} == set(predecessor_state):
        raise ProcessV2T1RuntimeError(
            "relational score predecessor lacks its declared residual parameter"
        )
    base_state = {
        name: value
        for name, value in predecessor_state.items()
        if name != "graft_relation_head.weight"
    }
    base_state_sha256 = state_dict_semantic_sha256(base_state)
    relational_state_sha256 = state_dict_semantic_sha256(predecessor_state)
    if (
        base_state_sha256 != frozen["initial_model_state_sha256"]
        or bool(predecessor_state["graft_relation_head.weight"].count_nonzero())
    ):
        raise ProcessV2T1RuntimeError(
            "relational score revision is not one zero-initialized residual over base"
        )
    current_state = model.state_dict()
    if any(
        name not in current_state or not torch.equal(value, current_state[name])
        for name, value in predecessor_state.items()
    ):
        raise ProcessV2T1RuntimeError(
            "ring-restatement score revision changed a predecessor parameter"
        )
    added_parameters = tuple(sorted(set(current_state) - set(predecessor_state)))
    if added_parameters != ("ring_restate_context_head.weight",) or bool(
        current_state[added_parameters[0]].count_nonzero()
    ):
        raise ProcessV2T1RuntimeError(
            "ring-restatement score revision is not one zero-initialized residual"
        )
    current_identity = {
        **predecessor_identity,
        _RING_RESTATE_SCORE_REVISION_FIELD: CONTEXTUAL_RING_RESTATE_SCORER_MODE,
    }
    runtime = replace(
        predecessor_runtime,
        model=model.eval(),
        semantic_model_identity=current_identity,
        initial_model_state_sha256=state_dict_semantic_sha256(current_state),
    )
    current = model_runtime_descriptor(runtime)
    if any(
        current[field] != predecessor[field]
        for field in _SCORE_INDEPENDENT_MODEL_RUNTIME_FIELDS
    ):
        raise ProcessV2T1RuntimeError(
            "score-revision repair changed T1 support or tensor geometry"
        )
    if current["initial_model_state_sha256"] != state_dict_semantic_sha256(
        runtime.model.state_dict()
    ):
        raise ProcessV2T1RuntimeError("current score-revision scratch state is unstable")
    if all(current[field] == predecessor[field] for field in set(current)):
        raise ProcessV2T1RuntimeError("score-revision bridge did not observe a scorer revision")

    binding = {
        "active8_decision_runtime_file_sha256": _file_sha256(
            Path(repo_root) / ACTIVE8_DECISION_RUNTIME
        ),
        "active8_decision_runtime_sha256": runtime_contract["contract_sha256"],
        "semantic_model_process_file_sha256": semantic.file_sha256,
        "semantic_model_process_sha256": semantic.sha256,
        "model_runtime": current,
    }
    bridge = {
        "predecessor_prepared_completion_sha256": frozen[
            "prepared_completion_sha256"
        ],
        "predecessor_initial_model_state_sha256": predecessor[
            "initial_model_state_sha256"
        ],
        "current_initial_model_state_sha256": current[
            "initial_model_state_sha256"
        ],
        "score_revision": {
            "schema": "compose.editing_v2.process_v2_t1_score_revision",
            "schema_version": 1,
            _RING_RESTATE_SCORE_REVISION_FIELD: (
                CONTEXTUAL_RING_RESTATE_SCORER_MODE
            ),
            "added_parameters": list(added_parameters),
        },
        "support_geometry_sha256": canonical_sha256(
            {field: current[field] for field in _SCORE_INDEPENDENT_MODEL_RUNTIME_FIELDS}
        ),
    }
    containment_body = {
        "schema": "compose.editing_v2.process_v2_score_revision_containment",
        "schema_version": 1,
        "base_initial_model_state_sha256": base_state_sha256,
        "relational_initial_model_state_sha256": relational_state_sha256,
        "current_initial_model_state_sha256": current[
            "initial_model_state_sha256"
        ],
        "support_geometry_sha256": bridge["support_geometry_sha256"],
        "implementation_sha256": canonical_sha256(
            [
                {
                    "path": relative,
                    "file_sha256": _file_sha256(Path(repo_root) / relative),
                }
                for relative in (
                    "src/compose_v4/experiments/editing_v2_process_v2_t1_runtime.py",
                    "src/compose_v4/model/factorized_tracelet_rate_model.py",
                    "src/compose_v4/model/relational_reroute_rate_model.py",
                    "src/compose_v4/model/contextual_ring_restate_rate_model.py",
                )
            ]
        ),
        "zero_initialized_residuals": [
            {
                "affected_family": "bond_reroute",
                "parameter": "graft_relation_head.weight",
                "predecessor": "base",
                "successor": "relational",
                "zero_initialized": True,
            },
            {
                "affected_family": "ring_system_restate",
                "parameter": "ring_restate_context_head.weight",
                "predecessor": "relational",
                "successor": "current",
                "zero_initialized": True,
            },
        ],
    }
    containment = {
        **containment_body,
        "receipt_sha256": canonical_sha256(containment_body),
    }
    return runtime, binding, {
        **bridge,
        "score_revision_containment": containment,
        "score_revision_rebind_sha256": canonical_sha256(bridge),
    }


def load_materialized_scorer_state(directory: Path) -> dict[str, Any]:
    """Load and authenticate scorer weights materialized on the frozen platform.

    Returns the two state dicts for ``materialized_state``.  The receipt is
    checked against the frozen constant BEFORE the bytes are handed to the
    builder, so unauthenticated weights cannot reach a compile even if the
    builder's own assertions were ever loosened.  See
    ``modal_apps/materialize_process_v2_predecessor_state_app.py``.
    """

    directory = Path(directory)
    receipt = json.loads((directory / "MATERIALIZATION_RECEIPT.json").read_text())
    frozen = _SCORE_REVISION_PREDECESSOR["initial_model_state_sha256"]
    if receipt.get("base_state_sha256") != frozen:
        raise ProcessV2T1RuntimeError(
            f"materialized scorer receipt names base state "
            f"{receipt.get('base_state_sha256')}, not the frozen {frozen}"
        )
    predecessor = torch.load(
        directory / "predecessor_state.pt", map_location="cpu", weights_only=True
    )
    revised = torch.load(
        directory / "revised_state.pt", map_location="cpu", weights_only=True
    )
    base_state = {
        name: value
        for name, value in predecessor.items()
        if name != "graft_relation_head.weight"
    }
    observed = state_dict_semantic_sha256(base_state)
    if observed != frozen:
        raise ProcessV2T1RuntimeError(
            f"materialized scorer bytes hash {observed}, not the frozen {frozen}"
        )
    return {"predecessor": predecessor, "revised": revised, "receipt": receipt}


def build_process_v2_score_revised_scratch_runtime(
    source: ProcessV2T1Source,
    *,
    materialized_state: Mapping[str, Any] | None = None,
) -> tuple[SemanticScratchRuntime, dict[str, Any], dict[str, Any]]:
    """Construct the P50 scorer and its exact zero-residual containment receipt."""

    bound = source.plan.get("binding", {}).get("model_runtime")
    if not isinstance(bound, Mapping) or tuple(sorted(bound)) != MODEL_RUNTIME_FIELDS:
        raise ProcessV2T1RuntimeError("Active8 plan lacks its model runtime descriptor")
    runtime, binding, bridge = _scratch_runtime_for_score_revision(
        bound, repo_root=source.repo_root, materialized_state=materialized_state
    )
    if runtime.process_identity_sha256 != source.contracts.process_identity_sha256:
        raise ProcessV2T1RuntimeError("score-revised scratch process identity changed")
    return runtime, binding, dict(bridge["score_revision_containment"])


def build_process_v2_t1_scratch_runtime(
    source: ProcessV2T1Source,
) -> tuple[SemanticScratchRuntime, dict[str, Any]]:
    """Construct the current score model on Active8's frozen support geometry.

    Active8 admission enumerates candidates and stores no model scores or
    probabilities.  Its runtime descriptor therefore owns the architecture and
    legal-support geometry, while Gate 0 owns the current scoring-model
    contract.  A scorer-only model revision may change the model identity and
    zero-initialized state hash without forcing molecular re-enumeration, but it
    may not change any field that determines tensor geometry or legal support.
    """

    bound = source.plan.get("binding", {}).get("model_runtime")
    if not isinstance(bound, Mapping) or tuple(sorted(bound)) != MODEL_RUNTIME_FIELDS:
        raise ProcessV2T1RuntimeError("Active8 plan lacks its model runtime descriptor")
    config = SemanticScratchModelConfig(
        initialization_seed=int(bound["initialization_seed"]),
        max_atoms=int(bound["max_atoms"]),
        hidden_dim=int(bound["hidden_dim"]),
        message_passing_steps=int(bound["message_passing_steps"]),
        mark_dim=int(bound["mark_dim"]),
        dtype=str(bound["dtype"]),
        atom_vocabulary_class_count=int(bound["atom_vocabulary_class_count"]),
        catalog_fingerprint=str(bound["catalog_fingerprint"]),
    )
    semantic_path = Path(source.repo_root) / GATE_ZERO_MODEL_PROCESS_V2
    semantic = load_gate_zero_semantic_contract(semantic_path)
    runtime = build_semantic_scratch_runtime(config, semantic)
    observed = model_runtime_descriptor(runtime)
    support_fields = (
        "atom_vocabulary_class_count",
        "catalog_fingerprint",
        "dtype",
        "hidden_dim",
        "initialization_seed",
        "mark_dim",
        "max_atoms",
        "message_passing_steps",
        "operator_capability_fingerprint",
    )
    if any(observed[field] != bound[field] for field in support_fields):
        raise ProcessV2T1RuntimeError(
            "the current T1 model changed Active8 architecture or legal support"
        )
    runtime_contract = load_process_v2_chain_artifact(
        ACTIVE8_DECISION_RUNTIME, repo_root=source.repo_root
    )
    expected_parent = runtime_contract["parents"]["semantic_model_process"]
    if (
        semantic.file_sha256 != expected_parent["physical"]["sha256"]
        or semantic.sha256 != expected_parent["semantic"]["sha256"]
    ):
        raise ProcessV2T1RuntimeError("Process-V2 semantic model contract bytes disagree")
    binding = {
        "active8_decision_runtime_file_sha256": _file_sha256(
            Path(source.repo_root) / ACTIVE8_DECISION_RUNTIME
        ),
        "active8_decision_runtime_sha256": runtime_contract["contract_sha256"],
        "semantic_model_process_file_sha256": semantic.file_sha256,
        "semantic_model_process_sha256": semantic.sha256,
        "model_runtime": observed,
    }
    if runtime.process_identity_sha256 != source.contracts.process_identity_sha256:
        raise ProcessV2T1RuntimeError("scratch model differs from the Active8-bound runtime")
    return runtime, binding


def build_process_v2_t1_prepared_plan(
    panel: Mapping[str, Any],
    *,
    source: ProcessV2T1Source,
    source_revision: Mapping[str, Any],
) -> dict[str, Any]:
    """Freeze tasks under a launcher-authenticated serialized image revision."""

    return _build_process_v2_t1_prepared_plan(
        panel,
        source=source,
        source_revision=_validate_source_revision(source_revision),
    )


def _build_process_v2_t1_prepared_plan(
    panel: Mapping[str, Any],
    *,
    source: ProcessV2T1Source,
    source_revision: Mapping[str, Any],
    implementation_sha256: str | None = None,
) -> dict[str, Any]:
    """Pure plan construction from an already authenticated revision receipt."""

    try:
        validated_panel = validate_process_v2_t1_panel(panel, source=source)
    except ProcessV2T1PanelError as error:
        raise ProcessV2T1RuntimeError(f"T1 panel is invalid: {error}") from error
    revision = _validate_source_revision(source_revision)
    _runtime, model_binding = build_process_v2_t1_scratch_runtime(source)
    entries_by_task: dict[str, list[dict[str, Any]]] = {}
    for entry in validated_panel["entries"]:
        identity = str(entry["task_identity_sha256"])
        entries_by_task.setdefault(identity, []).append(dict(entry))
    active8_tasks = {str(task["task_identity_sha256"]): dict(task) for task in source.plan["tasks"]}
    if set(entries_by_task) - set(active8_tasks):
        raise ProcessV2T1RuntimeError("T1 panel names an Active8 task outside its plan")
    panel_bytes = canonical_bytes(validated_panel) + b"\n"
    base = {
        "source_revision": revision,
        "implementation_sha256": (
            _implementation_sha256(source.repo_root)
            if implementation_sha256 is None
            else _require_sha(implementation_sha256, field="T1 implementation")
        ),
        "process_identity_sha256": source.contracts.process_identity_sha256,
        "active8_completion_sha256": source.index.active8_completion_sha256,
        "active8_sentinel_sha256": source.index.active8_sentinel_sha256,
        "active8_plan_sha256": source.plan["plan_sha256"],
        "active8_run_identity_sha256": source.plan["run_identity_sha256"],
        "gate_zero_decision_sha256": source.decision["decision_sha256"],
        "panel_sha256": validated_panel["panel_sha256"],
        "panel_file_sha256": hashlib.sha256(panel_bytes).hexdigest(),
        "panel_file_bytes": len(panel_bytes),
        "panel_entry_inventory_sha256": canonical_sha256(
            sorted(str(entry["panel_entry_sha256"]) for entry in validated_panel["entries"])
        ),
        "support_time_hex": validated_panel["support_time_hex"],
        "model_binding": model_binding,
    }
    build_identity_sha256 = canonical_sha256(base)
    tasks: list[dict[str, Any]] = []
    for task_index, active8_identity in enumerate(sorted(entries_by_task)):
        entries = sorted(
            entries_by_task[active8_identity], key=lambda item: item["panel_entry_sha256"]
        )
        body = {
            "task_index": task_index,
            "active8_task_identity_sha256": active8_identity,
            "active8_chunk_file_sha256": active8_tasks[active8_identity]["chunk_file_sha256"],
            "panel_entry_sha256s": [entry["panel_entry_sha256"] for entry in entries],
            "panel_entry_count": len(entries),
        }
        tasks.append(
            {
                **body,
                "task_identity_sha256": canonical_sha256(
                    {"build_identity_sha256": build_identity_sha256, "task": body}
                ),
            }
        )
    body = {
        "schema": PLAN_SCHEMA,
        "schema_version": PLAN_SCHEMA_VERSION,
        "status": PLAN_STATUS,
        **authority_false_block(),
        **base,
        "build_identity_sha256": build_identity_sha256,
        "task_count": len(tasks),
        "task_inventory_sha256": canonical_sha256(tasks),
        "selected_entry_count": len(validated_panel["entries"]),
        "tasks": tasks,
    }
    with_run = {**body, "run_identity_sha256": canonical_sha256(body)}
    return {**with_run, "plan_sha256": canonical_sha256(with_run)}


def _validate_process_v2_t1_prepared_plan(
    value: object,
    *,
    panel: Mapping[str, Any],
    source: ProcessV2T1Source,
    implementation_sha256: str | None,
) -> dict[str, Any]:
    """Rebuild a plan from its exact inputs and require byte identity."""

    if not isinstance(value, Mapping):
        raise ProcessV2T1RuntimeError("T1 prepared plan must be an object")
    plan = dict(value)
    expected_fields = {
        "schema",
        "schema_version",
        "status",
        *authority_false_block(),
        "source_revision",
        "implementation_sha256",
        "process_identity_sha256",
        "active8_completion_sha256",
        "active8_sentinel_sha256",
        "active8_plan_sha256",
        "active8_run_identity_sha256",
        "gate_zero_decision_sha256",
        "panel_sha256",
        "panel_file_sha256",
        "panel_file_bytes",
        "panel_entry_inventory_sha256",
        "support_time_hex",
        "model_binding",
        "build_identity_sha256",
        "task_count",
        "task_inventory_sha256",
        "selected_entry_count",
        "tasks",
        "run_identity_sha256",
        "plan_sha256",
    }
    if set(plan) != expected_fields:
        raise ProcessV2T1RuntimeError("T1 prepared plan field set disagrees")
    try:
        verify_self_hash(plan, field="plan_sha256", label="the T1 prepared plan")
        require_authority_false(plan, label="the T1 prepared plan")
    except ProcessV2SchemaError as error:
        raise ProcessV2T1RuntimeError(str(error)) from error
    if (
        plan["schema"] != PLAN_SCHEMA
        or plan["schema_version"] != PLAN_SCHEMA_VERSION
        or plan["status"] != PLAN_STATUS
        or plan["task_count"] != len(plan["tasks"])
        or plan["task_inventory_sha256"] != canonical_sha256(plan["tasks"])
        or plan["selected_entry_count"]
        != sum(int(task["panel_entry_count"]) for task in plan["tasks"])
    ):
        raise ProcessV2T1RuntimeError("T1 prepared plan identity or census disagrees")
    rebuilt = _build_process_v2_t1_prepared_plan(
        panel,
        source=source,
        source_revision=_validate_source_revision(plan["source_revision"]),
        implementation_sha256=implementation_sha256,
    )
    if canonical_bytes(rebuilt) != canonical_bytes(plan):
        raise ProcessV2T1RuntimeError("T1 prepared plan differs from exact recomputation")
    return plan


def validate_process_v2_t1_prepared_plan(
    value: object,
    *,
    panel: Mapping[str, Any],
    source: ProcessV2T1Source,
) -> dict[str, Any]:
    """Rebuild a current plan from its exact inputs and require byte identity."""

    return _validate_process_v2_t1_prepared_plan(
        value,
        panel=panel,
        source=source,
        implementation_sha256=None,
    )


def validate_process_v2_t1_leaf_reuse_plan(
    value: object,
    *,
    panel: Mapping[str, Any],
    source: ProcessV2T1Source,
    expected_plan_sha256: str,
) -> dict[str, Any]:
    """Authenticate one exact historical plan without claiming current compilation."""

    expected = _require_sha(expected_plan_sha256, field="expected reuse plan")
    if not isinstance(value, Mapping) or value.get("plan_sha256") != expected:
        raise ProcessV2T1RuntimeError("T1 leaf reuse plan is not the explicitly authorized plan")
    implementation = _require_sha(
        value.get("implementation_sha256"), field="historical T1 implementation"
    )
    return _validate_process_v2_t1_prepared_plan(
        value,
        panel=panel,
        source=source,
        implementation_sha256=implementation,
    )


def authenticate_process_v2_t1_prepared_plan_for_worker(
    value: object,
    *,
    panel: Mapping[str, Any],
    source: ProcessV2T1Source,
) -> dict[str, Any]:
    """Authenticate coordinator-validated inputs without rescanning the corpus.

    The coordinator builds the panel from the complete eligible decision stream
    and freezes its canonical bytes in the plan.  A map worker must authenticate
    those exact bytes and the plan's derivable identities, but must not repeat
    the global panel-selection scan before compiling its assigned states.
    """

    if not isinstance(value, Mapping) or not isinstance(panel, Mapping):
        raise ProcessV2T1RuntimeError("T1 worker plan and panel must be objects")
    plan = dict(value)
    panel_value = dict(panel)
    expected_plan_fields = {
        "schema",
        "schema_version",
        "status",
        *authority_false_block(),
        "source_revision",
        "implementation_sha256",
        "process_identity_sha256",
        "active8_completion_sha256",
        "active8_sentinel_sha256",
        "active8_plan_sha256",
        "active8_run_identity_sha256",
        "gate_zero_decision_sha256",
        "panel_sha256",
        "panel_file_sha256",
        "panel_file_bytes",
        "panel_entry_inventory_sha256",
        "support_time_hex",
        "model_binding",
        "build_identity_sha256",
        "task_count",
        "task_inventory_sha256",
        "selected_entry_count",
        "tasks",
        "run_identity_sha256",
        "plan_sha256",
    }
    if set(plan) != expected_plan_fields:
        raise ProcessV2T1RuntimeError("T1 worker plan field set disagrees")
    panel_bytes = canonical_bytes(panel_value) + b"\n"
    try:
        verify_self_hash(plan, field="plan_sha256", label="the T1 prepared plan")
        require_authority_false(plan, label="the T1 prepared plan")
        verify_self_hash(panel_value, field="panel_sha256", label="the T1 panel")
        require_authority_false(panel_value, label="the T1 panel")
    except ProcessV2SchemaError as error:
        raise ProcessV2T1RuntimeError(str(error)) from error
    source_revision = _validate_source_revision(plan.get("source_revision", {}))
    without_plan_sha = {key: item for key, item in plan.items() if key != "plan_sha256"}
    without_run_identity = {
        key: item for key, item in without_plan_sha.items() if key != "run_identity_sha256"
    }
    tasks = plan.get("tasks")
    if not isinstance(tasks, list):
        raise ProcessV2T1RuntimeError("T1 worker plan tasks must be a list")
    expected_task_fields = {
        "task_index",
        "active8_task_identity_sha256",
        "active8_chunk_file_sha256",
        "panel_entry_sha256s",
        "panel_entry_count",
        "task_identity_sha256",
    }
    active8_tasks = {str(task["task_identity_sha256"]): task for task in source.plan["tasks"]}
    task_ids: list[str] = []
    entry_ids: list[str] = []
    for index, raw_task in enumerate(tasks):
        if not isinstance(raw_task, Mapping) or set(raw_task) != expected_task_fields:
            raise ProcessV2T1RuntimeError("T1 worker task field set disagrees")
        task = dict(raw_task)
        active8_identity = str(task["active8_task_identity_sha256"])
        active8_task = active8_tasks.get(active8_identity)
        identifiers = task["panel_entry_sha256s"]
        task_body = {key: item for key, item in task.items() if key != "task_identity_sha256"}
        if (
            type(task["task_index"]) is not int
            or task["task_index"] != index
            or active8_task is None
            or task["active8_chunk_file_sha256"] != active8_task["chunk_file_sha256"]
            or not isinstance(identifiers, list)
            or not identifiers
            or identifiers != sorted(identifiers)
            or len(identifiers) != len(set(identifiers))
            or type(task["panel_entry_count"]) is not int
            or task["panel_entry_count"] != len(identifiers)
            or task["task_identity_sha256"]
            != canonical_sha256(
                {
                    "build_identity_sha256": plan.get("build_identity_sha256"),
                    "task": task_body,
                }
            )
        ):
            raise ProcessV2T1RuntimeError("T1 worker task identity or census disagrees")
        task_ids.append(_require_sha(task["task_identity_sha256"], field="task identity"))
        entry_ids.extend(_require_sha(item, field="panel entry") for item in identifiers)
    base_keys = {
        "source_revision",
        "implementation_sha256",
        "process_identity_sha256",
        "active8_completion_sha256",
        "active8_sentinel_sha256",
        "active8_plan_sha256",
        "active8_run_identity_sha256",
        "gate_zero_decision_sha256",
        "panel_sha256",
        "panel_file_sha256",
        "panel_file_bytes",
        "panel_entry_inventory_sha256",
        "support_time_hex",
        "model_binding",
    }
    base = {key: plan.get(key) for key in base_keys}
    if (
        plan.get("schema") != PLAN_SCHEMA
        or plan.get("schema_version") != PLAN_SCHEMA_VERSION
        or plan.get("status") != PLAN_STATUS
        or plan.get("implementation_sha256") != _implementation_sha256(source.repo_root)
        or plan.get("process_identity_sha256") != source.contracts.process_identity_sha256
        or plan.get("active8_completion_sha256") != source.index.active8_completion_sha256
        or plan.get("active8_sentinel_sha256") != source.index.active8_sentinel_sha256
        or plan.get("active8_plan_sha256") != source.plan["plan_sha256"]
        or plan.get("active8_run_identity_sha256") != source.plan["run_identity_sha256"]
        or plan.get("gate_zero_decision_sha256") != source.decision["decision_sha256"]
        or plan.get("panel_sha256") != panel_value.get("panel_sha256")
        or plan.get("panel_file_sha256") != hashlib.sha256(panel_bytes).hexdigest()
        or plan.get("panel_file_bytes") != len(panel_bytes)
        or plan.get("panel_entry_inventory_sha256") != canonical_sha256(sorted(entry_ids))
        or plan.get("build_identity_sha256") != canonical_sha256(base)
        or plan.get("run_identity_sha256") != canonical_sha256(without_run_identity)
        or plan.get("task_count") != len(tasks)
        or plan.get("task_inventory_sha256") != canonical_sha256(tasks)
        or plan.get("selected_entry_count") != len(entry_ids)
        or len(task_ids) != len(set(task_ids))
        or len(entry_ids) != len(set(entry_ids))
        or source_revision != plan.get("source_revision")
    ):
        raise ProcessV2T1RuntimeError("T1 worker plan, panel, or source binding disagrees")
    panel_entries = panel_value.get("entries")
    if not isinstance(panel_entries, list) or sorted(
        str(entry.get("panel_entry_sha256"))
        for entry in panel_entries
        if isinstance(entry, Mapping)
    ) != sorted(entry_ids):
        raise ProcessV2T1RuntimeError("T1 worker panel entry inventory disagrees")
    return plan


def write_process_v2_t1_prepared_plan(
    plan: Mapping[str, Any],
    *,
    output_root: Path,
    panel: Mapping[str, Any],
    source: ProcessV2T1Source,
) -> Path:
    validated = validate_process_v2_t1_prepared_plan(plan, panel=panel, source=source)
    path = Path(output_root) / validated["run_identity_sha256"] / PLAN_FILENAME
    try:
        write_bytes_if_absent(path, canonical_bytes(validated) + b"\n")
    except ImmutableArtifactError as error:
        raise ProcessV2T1RuntimeError(str(error)) from error
    return path


def _entry_from_exact_transition(
    panel_entry: Mapping[str, Any],
    *,
    addressed: Any,
    scratch_runtime: SemanticScratchRuntime,
    support_time: float,
) -> dict[str, Any]:
    progress = int(panel_entry["progress_index"])
    source_state = addressed.path.state_at(progress)
    target_state = addressed.path.state_at(progress + 1)
    step = addressed.trace.steps[progress]
    observed_action_sha256 = rewrite_action_codec_sha256(
        step.rule_name,
        step.action,
        schema_version=action_codec_v4.SCHEMA_VERSION,
    )
    if (
        observed_action_sha256 != panel_entry["action_sha256"]
        or persistent_slot_state_sha256(source_state) != panel_entry["source_state_sha256"]
        or persistent_slot_state_sha256(target_state) != panel_entry["target_state_sha256"]
    ):
        raise ProcessV2T1RuntimeError("T1 selected action or exact state changed after paneling")
    try:
        partition = compile_state_successor_map(
            scratch_runtime.model,
            source_state,
            time=support_time,
        )
        require_exact_successor_action_identity(
            partition,
            target_state_sha256=str(panel_entry["target_state_sha256"]),
            action_sha256=str(panel_entry["action_sha256"]),
        )
        teacher = teacher_successor_fiber_from_exact_digest(
            partition, str(panel_entry["target_state_sha256"])
        )
    except (SuccessorTrainingError, ValueError) as error:
        raise ProcessV2T1RuntimeError(
            "T1 teacher is absent from its exact Process-V2 successor partition"
        ) from error
    state_payload = encode_state(source_state)
    if encode_state(decode_state(state_payload)) != state_payload:
        raise ProcessV2T1RuntimeError("T1 exact-state encoding is not byte-stable")
    partition_payload = compiled_successor_map_payload(partition)
    productive_alias_count = sum(len(group.marks) for group in partition.successor_groups)
    virtual_alias_count = len(partition.virtual_marks)
    body = {
        "panel_entry_sha256": panel_entry["panel_entry_sha256"],
        "model_family": panel_entry["model_family"],
        "capability_cell_id": panel_entry["capability_cell_id"],
        "support_time_hex": float(support_time).hex(),
        "source_state_sha256": panel_entry["source_state_sha256"],
        "target_state_sha256": panel_entry["target_state_sha256"],
        "successor_canonical_key": panel_entry["canonical_successor_key"],
        "teacher_action_sha256": panel_entry["action_sha256"],
        "objective_coefficient": 1,
        "raw_mark_count": productive_alias_count + virtual_alias_count,
        "canonical_successor_count": len(partition.successor_groups),
        "production_successor_alias_multiplicity": len(teacher.aliases),
        "exact_state": state_payload,
        "successor_partition": partition_payload,
        "productive_alias_count": productive_alias_count,
        "virtual_alias_count": virtual_alias_count,
        "model_scores_or_probabilities_stored": False,
        "hazard_included": False,
    }
    return {**body, "entry_sha256": canonical_sha256(body)}


def compile_process_v2_t1_prepared_leaf(
    plan: Mapping[str, Any],
    *,
    task_identity_sha256: str,
    panel: Mapping[str, Any],
    source: ProcessV2T1Source,
    scratch_runtime: SemanticScratchRuntime,
    progress_callback: Callable[[Mapping[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Compile exact successor partitions for one selected Active8 chunk."""

    validated = validate_process_v2_t1_prepared_plan(plan, panel=panel, source=source)
    return _compile_validated_process_v2_t1_prepared_leaf(
        validated,
        task_identity_sha256=task_identity_sha256,
        panel=panel,
        source=source,
        scratch_runtime=scratch_runtime,
        progress_callback=progress_callback,
    )


def compile_authenticated_process_v2_t1_prepared_leaf(
    plan: Mapping[str, Any],
    *,
    task_identity_sha256: str,
    panel: Mapping[str, Any],
    source: ProcessV2T1Source,
    scratch_runtime: SemanticScratchRuntime,
    progress_callback: Callable[[Mapping[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Compile one worker leaf from a coordinator-validated bound panel."""

    validated = authenticate_process_v2_t1_prepared_plan_for_worker(
        plan, panel=panel, source=source
    )
    return _compile_validated_process_v2_t1_prepared_leaf(
        validated,
        task_identity_sha256=task_identity_sha256,
        panel=panel,
        source=source,
        scratch_runtime=scratch_runtime,
        progress_callback=progress_callback,
    )


def _compile_validated_process_v2_t1_prepared_leaf(
    validated: Mapping[str, Any],
    *,
    task_identity_sha256: str,
    panel: Mapping[str, Any],
    source: ProcessV2T1Source,
    scratch_runtime: SemanticScratchRuntime,
    progress_callback: Callable[[Mapping[str, Any]], None] | None,
) -> dict[str, Any]:
    task = next(
        (
            candidate
            for candidate in validated["tasks"]
            if candidate["task_identity_sha256"] == task_identity_sha256
        ),
        None,
    )
    if task is None:
        raise ProcessV2T1RuntimeError("T1 prepared task is absent from the plan")
    descriptor = model_runtime_descriptor(scratch_runtime)
    if descriptor != validated["model_binding"]["model_runtime"]:
        raise ProcessV2T1RuntimeError("T1 leaf scratch runtime differs from the plan")
    panel_by_id = {str(entry["panel_entry_sha256"]): dict(entry) for entry in panel["entries"]}
    try:
        selected = [panel_by_id[str(item)] for item in task["panel_entry_sha256s"]]
    except KeyError as error:
        raise ProcessV2T1RuntimeError("T1 prepared task names an absent panel entry") from error
    if any(
        entry["task_identity_sha256"] != task["active8_task_identity_sha256"] for entry in selected
    ):
        raise ProcessV2T1RuntimeError("T1 prepared task crosses Active8 source chunks")
    try:
        resolved = resolve_process_v2_t1_entries(source, selected)
    except ProcessV2T1PanelError as error:
        raise ProcessV2T1RuntimeError(f"exact T1 states could not be reopened: {error}") from error
    addressed_by_entry: dict[str, Any] = {}
    for item in resolved:
        for entry in item.panel_entries:
            identifier = str(entry["panel_entry_sha256"])
            if identifier in addressed_by_entry:
                raise ProcessV2T1RuntimeError("T1 exact entry was reopened twice")
            addressed_by_entry[identifier] = item.addressed_trace
    if set(addressed_by_entry) != set(task["panel_entry_sha256s"]):
        raise ProcessV2T1RuntimeError("T1 leaf did not reopen its complete exact entry set")
    support_time = float.fromhex(str(validated["support_time_hex"]))
    started_at = time.monotonic()
    if progress_callback is not None:
        progress_callback(
            {
                "phase": "process_v2_t1_prepare_task_start",
                "task_identity_sha256": task["task_identity_sha256"],
                "entry_count": len(selected),
            }
        )
    entries: list[dict[str, Any]] = []
    for entry_index, entry in enumerate(selected, start=1):
        entries.append(
            _entry_from_exact_transition(
                entry,
                addressed=addressed_by_entry[str(entry["panel_entry_sha256"])],
                scratch_runtime=scratch_runtime,
                support_time=support_time,
            )
        )
        if progress_callback is not None:
            progress_callback(
                {
                    "phase": "process_v2_t1_prepare_entry_complete",
                    "task_identity_sha256": task["task_identity_sha256"],
                    "entry_index": entry_index,
                    "entry_count": len(selected),
                    "elapsed_seconds": time.monotonic() - started_at,
                }
            )
    entries.sort(key=lambda item: item["panel_entry_sha256"])
    body = {
        "schema": LEAF_SCHEMA,
        "schema_version": LEAF_SCHEMA_VERSION,
        "status": LEAF_STATUS,
        **authority_false_block(),
        "plan_sha256": validated["plan_sha256"],
        "run_identity_sha256": validated["run_identity_sha256"],
        "build_identity_sha256": validated["build_identity_sha256"],
        "task_identity_sha256": task["task_identity_sha256"],
        "active8_task_identity_sha256": task["active8_task_identity_sha256"],
        "panel_sha256": validated["panel_sha256"],
        "initial_model_state_sha256": descriptor["initial_model_state_sha256"],
        "entry_count": len(entries),
        "entry_inventory_sha256": canonical_sha256(entries),
        "entries": entries,
        "model_scores_or_probabilities_stored": False,
        "hazard_included": False,
    }
    leaf = {**body, "leaf_sha256": canonical_sha256(body)}
    return validate_process_v2_t1_prepared_leaf(leaf, plan=validated)


def validate_process_v2_t1_prepared_leaf(
    value: object, *, plan: Mapping[str, Any]
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ProcessV2T1RuntimeError("T1 prepared leaf must be an object")
    leaf = dict(value)
    expected_fields = {
        "schema",
        "schema_version",
        "status",
        *authority_false_block(),
        "plan_sha256",
        "run_identity_sha256",
        "build_identity_sha256",
        "task_identity_sha256",
        "active8_task_identity_sha256",
        "panel_sha256",
        "initial_model_state_sha256",
        "entry_count",
        "entry_inventory_sha256",
        "entries",
        "model_scores_or_probabilities_stored",
        "hazard_included",
        "leaf_sha256",
    }
    if set(leaf) != expected_fields:
        raise ProcessV2T1RuntimeError("T1 prepared leaf field set disagrees")
    try:
        verify_self_hash(leaf, field="leaf_sha256", label="a T1 prepared leaf")
        require_authority_false(leaf, label="a T1 prepared leaf")
    except ProcessV2SchemaError as error:
        raise ProcessV2T1RuntimeError(str(error)) from error
    task = next(
        (
            candidate
            for candidate in plan["tasks"]
            if candidate["task_identity_sha256"] == leaf["task_identity_sha256"]
        ),
        None,
    )
    if (
        task is None
        or leaf["schema"] != LEAF_SCHEMA
        or leaf["schema_version"] != LEAF_SCHEMA_VERSION
        or leaf["status"] != LEAF_STATUS
        or leaf["plan_sha256"] != plan["plan_sha256"]
        or leaf["run_identity_sha256"] != plan["run_identity_sha256"]
        or leaf["build_identity_sha256"] != plan["build_identity_sha256"]
        or leaf["panel_sha256"] != plan["panel_sha256"]
        or leaf["active8_task_identity_sha256"] != task["active8_task_identity_sha256"]
        or leaf["initial_model_state_sha256"]
        != plan["model_binding"]["model_runtime"]["initial_model_state_sha256"]
        or leaf["entry_count"] != len(leaf["entries"])
        or leaf["entry_inventory_sha256"] != canonical_sha256(leaf["entries"])
        or leaf["model_scores_or_probabilities_stored"] is not False
        or leaf["hazard_included"] is not False
    ):
        raise ProcessV2T1RuntimeError("T1 prepared leaf identity or census disagrees")
    observed_ids: list[str] = []
    for raw in leaf["entries"]:
        if not isinstance(raw, Mapping) or set(raw) != _PREPARED_ENTRY_FIELDS:
            raise ProcessV2T1RuntimeError("T1 prepared entry field set disagrees")
        entry = dict(raw)
        try:
            verify_self_hash(entry, field="entry_sha256", label="a T1 prepared entry")
            state = decode_state(entry["exact_state"])
            partition = compiled_successor_map_from_payload(entry["successor_partition"])
            teacher = teacher_successor_fiber_from_exact_digest(
                partition, str(entry["target_state_sha256"])
            )
            require_exact_successor_action_identity(
                partition,
                target_state_sha256=str(entry["target_state_sha256"]),
                action_sha256=str(entry["teacher_action_sha256"]),
            )
        except (ProcessV2SchemaError, SuccessorTrainingError, TypeError, ValueError) as error:
            raise ProcessV2T1RuntimeError("T1 prepared entry is internally invalid") from error
        productive = sum(len(group.marks) for group in partition.successor_groups)
        virtual = len(partition.virtual_marks)
        if (
            encode_state(state) != entry["exact_state"]
            or persistent_slot_state_sha256(state) != entry["source_state_sha256"]
            or partition.source_state_sha256 != entry["source_state_sha256"]
            or partition.source_key == entry["successor_canonical_key"]
            or teacher.target_key != entry["successor_canonical_key"]
            or entry["objective_coefficient"] != 1
            or entry["raw_mark_count"] != productive + virtual
            or entry["canonical_successor_count"] != len(partition.successor_groups)
            or entry["production_successor_alias_multiplicity"] != len(teacher.aliases)
            or entry["productive_alias_count"] != productive
            or entry["virtual_alias_count"] != virtual
            or entry["model_scores_or_probabilities_stored"] is not False
            or entry["hazard_included"] is not False
        ):
            raise ProcessV2T1RuntimeError("T1 prepared entry state or successor census disagrees")
        observed_ids.append(_require_sha(entry["panel_entry_sha256"], field="panel entry"))
    if observed_ids != sorted(task["panel_entry_sha256s"]) or len(set(observed_ids)) != len(
        observed_ids
    ):
        raise ProcessV2T1RuntimeError("T1 prepared leaf entry inventory disagrees")
    return leaf


def write_process_v2_t1_prepared_leaf(
    leaf: Mapping[str, Any], *, plan: Mapping[str, Any], run_root: Path
) -> Path:
    validated = validate_process_v2_t1_prepared_leaf(leaf, plan=plan)
    path = Path(run_root) / TASKS_DIRNAME / str(validated["task_identity_sha256"]) / LEAF_FILENAME
    try:
        write_bytes_if_absent(path, canonical_bytes(validated) + b"\n")
    except ImmutableArtifactError as error:
        raise ProcessV2T1RuntimeError(str(error)) from error
    return path


def _load_prepared_leaf(path: Path, *, plan: Mapping[str, Any]) -> dict[str, Any]:
    leaf, _raw = _load_canonical(path, label="a T1 prepared leaf")
    return validate_process_v2_t1_prepared_leaf(leaf, plan=plan)


def _load_reusable_prepared_leaf(path: Path, *, plan: Mapping[str, Any]) -> dict[str, Any]:
    """Authenticate an immutable leaf envelope without decoding its chemistry again."""

    leaf, _raw = _load_canonical(path, label="a reusable T1 prepared leaf")
    expected_fields = {
        "schema",
        "schema_version",
        "status",
        *authority_false_block(),
        "plan_sha256",
        "run_identity_sha256",
        "build_identity_sha256",
        "task_identity_sha256",
        "active8_task_identity_sha256",
        "panel_sha256",
        "initial_model_state_sha256",
        "entry_count",
        "entry_inventory_sha256",
        "entries",
        "model_scores_or_probabilities_stored",
        "hazard_included",
        "leaf_sha256",
    }
    if set(leaf) != expected_fields:
        raise ProcessV2T1RuntimeError("reusable T1 prepared leaf field set disagrees")
    try:
        verify_self_hash(leaf, field="leaf_sha256", label="a reusable T1 prepared leaf")
        require_authority_false(leaf, label="a reusable T1 prepared leaf")
    except ProcessV2SchemaError as error:
        raise ProcessV2T1RuntimeError(str(error)) from error
    task = next(
        (
            candidate
            for candidate in plan["tasks"]
            if candidate["task_identity_sha256"] == leaf["task_identity_sha256"]
        ),
        None,
    )
    entries = leaf.get("entries")
    observed_ids = [
        str(entry.get("panel_entry_sha256"))
        for entry in entries
        if isinstance(entry, Mapping)
    ] if isinstance(entries, list) else []
    if (
        task is None
        or leaf["schema"] != LEAF_SCHEMA
        or leaf["schema_version"] != LEAF_SCHEMA_VERSION
        or leaf["status"] != LEAF_STATUS
        or leaf["plan_sha256"] != plan["plan_sha256"]
        or leaf["run_identity_sha256"] != plan["run_identity_sha256"]
        or leaf["build_identity_sha256"] != plan["build_identity_sha256"]
        or leaf["panel_sha256"] != plan["panel_sha256"]
        or leaf["active8_task_identity_sha256"] != task["active8_task_identity_sha256"]
        or leaf["initial_model_state_sha256"]
        != plan["model_binding"]["model_runtime"]["initial_model_state_sha256"]
        or not isinstance(entries, list)
        or leaf["entry_count"] != len(entries)
        or leaf["entry_inventory_sha256"] != canonical_sha256(entries)
        or leaf["model_scores_or_probabilities_stored"] is not False
        or leaf["hazard_included"] is not False
        or len(observed_ids) != len(entries)
        or observed_ids != sorted(task["panel_entry_sha256s"])
    ):
        raise ProcessV2T1RuntimeError("reusable T1 prepared leaf identity or census disagrees")
    return leaf


def _require_complete_panel_entry_inventory(
    observed_ids: Sequence[str], expected_ids: Sequence[str]
) -> None:
    """Compare membership independently of the runtime entry ordering."""

    if sorted(observed_ids) != sorted(expected_ids) or len(set(observed_ids)) != len(
        observed_ids
    ):
        raise ProcessV2T1RuntimeError("T1 prepared reduction omits or repeats a panel entry")


def _build_process_v2_t1_prepared_inputs(
    validated: Mapping[str, Any],
    *,
    source: ProcessV2T1Source,
    run_root: Path,
    source_revision: Mapping[str, Any],
    reusable_leaves: bool,
) -> dict[str, Any]:
    """Reduce authenticated leaves into one score-independent runtime input."""

    leaves: list[dict[str, Any]] = []
    for task in validated["tasks"]:
        task_root = Path(run_root) / TASKS_DIRNAME / str(task["task_identity_sha256"])
        leaf_path = task_root / LEAF_FILENAME
        if not leaf_path.is_file():
            raise ProcessV2T1RuntimeError(f"T1 prepared leaf is absent: {leaf_path}")
        leaf = (
            _load_reusable_prepared_leaf(leaf_path, plan=validated)
            if reusable_leaves
            else _load_prepared_leaf(leaf_path, plan=validated)
        )
        leaves.append(leaf)
    if len(leaves) != validated["task_count"]:
        raise ProcessV2T1RuntimeError("T1 prepared leaf count differs from the plan")
    entries = [dict(entry) for leaf in leaves for entry in leaf["entries"]]
    entries.sort(
        key=lambda item: (
            item["model_family"],
            item["capability_cell_id"],
            item["panel_entry_sha256"],
        )
    )
    observed_ids = [str(entry["panel_entry_sha256"]) for entry in entries]
    expected_ids = [
        str(identifier) for task in validated["tasks"] for identifier in task["panel_entry_sha256s"]
    ]
    _require_complete_panel_entry_inventory(observed_ids, expected_ids)
    leaf_receipts = [
        {
            "task_identity_sha256": leaf["task_identity_sha256"],
            "leaf_sha256": leaf["leaf_sha256"],
            "entry_count": leaf["entry_count"],
        }
        for leaf in sorted(leaves, key=lambda item: item["task_identity_sha256"])
    ]
    manifest_body = {
        "plan_sha256": validated["plan_sha256"],
        "run_identity_sha256": validated["run_identity_sha256"],
        "panel_sha256": validated["panel_sha256"],
        "leaf_inventory_sha256": canonical_sha256(leaf_receipts),
        "entry_inventory_sha256": canonical_sha256(observed_ids),
        "entry_count": len(entries),
    }
    manifest_sha256 = canonical_sha256(manifest_body)
    body = {
        "schema": PREPARED_SCHEMA,
        "schema_version": PREPARED_SCHEMA_VERSION,
        "status": PREPARED_STATUS,
        **authority_false_block(),
        "source_revision": _validate_source_revision(source_revision),
        "implementation_sha256": _implementation_sha256(source.repo_root),
        "leaf_source_revision": validated["source_revision"],
        "leaf_implementation_sha256": validated["implementation_sha256"],
        "leaf_reuse": {
            "reused_precomputed_leaves": reusable_leaves,
            "leaf_plan_sha256": validated["plan_sha256"],
            "leaf_run_identity_sha256": validated["run_identity_sha256"],
            "reduction_rule": "complete_panel_entry_identity_multiset_v1",
        },
        "process_identity_sha256": validated["process_identity_sha256"],
        "active8_completion_sha256": validated["active8_completion_sha256"],
        "active8_sentinel_sha256": validated["active8_sentinel_sha256"],
        "gate_zero_decision_sha256": validated["gate_zero_decision_sha256"],
        "panel_sha256": validated["panel_sha256"],
        "panel_entry_inventory_sha256": validated["panel_entry_inventory_sha256"],
        "plan_sha256": validated["plan_sha256"],
        "run_identity_sha256": validated["run_identity_sha256"],
        "build_identity_sha256": validated["build_identity_sha256"],
        "manifest_sha256": manifest_sha256,
        "model_binding": validated["model_binding"],
        "support_time_hex": validated["support_time_hex"],
        "state_encoding": "compose.rewrite.trace.encoded_state_v2_exact_slots",
        "successor_partition_contract": {
            "productive_groups_complete": True,
            "productive_aliases_disjoint": True,
            "virtual_aliases_disjoint": True,
            "self_events_excluded_from_embedded_jump_chain": True,
            "model_scores_or_probabilities_stored": False,
            "hazard_included": False,
        },
        "leaf_count": len(leaves),
        "leaf_inventory_sha256": canonical_sha256(leaf_receipts),
        "entry_count": len(entries),
        "entry_inventory_sha256": canonical_sha256(entries),
        "entries": entries,
    }
    artifact = {**body, "artifact_sha256": canonical_sha256(body)}
    return validate_process_v2_t1_prepared_inputs(
        artifact,
        expected_plan=validated,
        validate_entry_payloads=not reusable_leaves,
    )


def build_process_v2_t1_prepared_inputs(
    plan: Mapping[str, Any],
    *,
    panel: Mapping[str, Any],
    source: ProcessV2T1Source,
    run_root: Path,
) -> dict[str, Any]:
    """Reduce current exact task leaves into one score-independent runtime input."""

    validated = validate_process_v2_t1_prepared_plan(plan, panel=panel, source=source)
    return _build_process_v2_t1_prepared_inputs(
        validated,
        source=source,
        run_root=run_root,
        source_revision=validated["source_revision"],
        reusable_leaves=False,
    )


def build_reused_process_v2_t1_prepared_inputs(
    plan: Mapping[str, Any],
    *,
    panel: Mapping[str, Any],
    source: ProcessV2T1Source,
    run_root: Path,
    source_revision: Mapping[str, Any],
    expected_plan_sha256: str,
) -> dict[str, Any]:
    """Reduce one explicitly authorized complete leaf run without chemistry recomputation."""

    validated = validate_process_v2_t1_leaf_reuse_plan(
        plan,
        panel=panel,
        source=source,
        expected_plan_sha256=expected_plan_sha256,
    )
    return _build_process_v2_t1_prepared_inputs(
        validated,
        source=source,
        run_root=run_root,
        source_revision=source_revision,
        reusable_leaves=True,
    )


def validate_process_v2_t1_prepared_inputs(
    value: object,
    *,
    expected_plan: Mapping[str, Any] | None = None,
    repo_root: Path | None = None,
    validate_entry_payloads: bool = True,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ProcessV2T1RuntimeError("T1 prepared inputs must be an object")
    artifact = dict(value)
    expected_fields = {
        "schema",
        "schema_version",
        "status",
        *authority_false_block(),
        "source_revision",
        "implementation_sha256",
        "leaf_source_revision",
        "leaf_implementation_sha256",
        "leaf_reuse",
        "process_identity_sha256",
        "active8_completion_sha256",
        "active8_sentinel_sha256",
        "gate_zero_decision_sha256",
        "panel_sha256",
        "panel_entry_inventory_sha256",
        "plan_sha256",
        "run_identity_sha256",
        "build_identity_sha256",
        "manifest_sha256",
        "model_binding",
        "support_time_hex",
        "state_encoding",
        "successor_partition_contract",
        "leaf_count",
        "leaf_inventory_sha256",
        "entry_count",
        "entry_inventory_sha256",
        "entries",
        "artifact_sha256",
    }
    if set(artifact) != expected_fields:
        raise ProcessV2T1RuntimeError("T1 prepared input field set disagrees")
    try:
        verify_self_hash(artifact, field="artifact_sha256", label="the T1 prepared inputs")
        require_authority_false(artifact, label="the T1 prepared inputs")
    except ProcessV2SchemaError as error:
        raise ProcessV2T1RuntimeError(str(error)) from error
    if (
        artifact["schema"] != PREPARED_SCHEMA
        or artifact["schema_version"] != PREPARED_SCHEMA_VERSION
        or artifact["status"] != PREPARED_STATUS
        or artifact["state_encoding"] != "compose.rewrite.trace.encoded_state_v2_exact_slots"
        or artifact["entry_count"] != len(artifact["entries"])
        or artifact["entry_inventory_sha256"] != canonical_sha256(artifact["entries"])
        or artifact["leaf_count"] <= 0
    ):
        raise ProcessV2T1RuntimeError("T1 prepared input identity or census disagrees")
    _validate_source_revision(artifact["source_revision"])
    _validate_source_revision(artifact["leaf_source_revision"])
    leaf_reuse = artifact["leaf_reuse"]
    if (
        not isinstance(leaf_reuse, Mapping)
        or set(leaf_reuse)
        != {
            "reused_precomputed_leaves",
            "leaf_plan_sha256",
            "leaf_run_identity_sha256",
            "reduction_rule",
        }
        or type(leaf_reuse["reused_precomputed_leaves"]) is not bool
        or leaf_reuse["leaf_plan_sha256"] != artifact["plan_sha256"]
        or leaf_reuse["leaf_run_identity_sha256"] != artifact["run_identity_sha256"]
        or leaf_reuse["reduction_rule"] != "complete_panel_entry_identity_multiset_v1"
    ):
        raise ProcessV2T1RuntimeError("T1 prepared leaf provenance disagrees")
    _require_sha(
        artifact["leaf_implementation_sha256"], field="leaf implementation"
    )
    if repo_root is not None and artifact["implementation_sha256"] != _implementation_sha256(
        repo_root
    ):
        raise ProcessV2T1RuntimeError("T1 prepared input implementation is stale")
    if expected_plan is not None:
        comparisons = {
            "leaf_source_revision": expected_plan["source_revision"],
            "leaf_implementation_sha256": expected_plan["implementation_sha256"],
            "process_identity_sha256": expected_plan["process_identity_sha256"],
            "active8_completion_sha256": expected_plan["active8_completion_sha256"],
            "active8_sentinel_sha256": expected_plan["active8_sentinel_sha256"],
            "gate_zero_decision_sha256": expected_plan["gate_zero_decision_sha256"],
            "panel_sha256": expected_plan["panel_sha256"],
            "panel_entry_inventory_sha256": expected_plan["panel_entry_inventory_sha256"],
            "plan_sha256": expected_plan["plan_sha256"],
            "run_identity_sha256": expected_plan["run_identity_sha256"],
            "build_identity_sha256": expected_plan["build_identity_sha256"],
            "model_binding": expected_plan["model_binding"],
            "support_time_hex": expected_plan["support_time_hex"],
        }
        if any(artifact.get(name) != expected for name, expected in comparisons.items()):
            raise ProcessV2T1RuntimeError("T1 prepared inputs bind another plan")
    observed: list[str] = []
    observed_sources: set[str] = set()
    for entry in artifact["entries"]:
        if not isinstance(entry, Mapping) or set(entry) != _PREPARED_ENTRY_FIELDS:
            raise ProcessV2T1RuntimeError("T1 prepared entry field set disagrees")
        # Validate the entry body directly without pretending the synthetic
        # envelope belongs to a plan task.
        if validate_entry_payloads:
            try:
                verify_self_hash(entry, field="entry_sha256", label="a T1 prepared entry")
                state = decode_state(entry["exact_state"])
                partition = compiled_successor_map_from_payload(entry["successor_partition"])
                teacher = teacher_successor_fiber_from_exact_digest(
                    partition, str(entry["target_state_sha256"])
                )
                require_exact_successor_action_identity(
                    partition,
                    target_state_sha256=str(entry["target_state_sha256"]),
                    action_sha256=str(entry["teacher_action_sha256"]),
                )
            except (ProcessV2SchemaError, SuccessorTrainingError, TypeError, ValueError) as error:
                raise ProcessV2T1RuntimeError("T1 prepared entry is internally invalid") from error
            productive = sum(len(group.marks) for group in partition.successor_groups)
            virtual = len(partition.virtual_marks)
            if (
                encode_state(state) != entry["exact_state"]
                or persistent_slot_state_sha256(state) != entry["source_state_sha256"]
                or partition.source_state_sha256 != entry["source_state_sha256"]
                or teacher.target_key != entry["successor_canonical_key"]
                or entry["raw_mark_count"] != productive + virtual
                or entry["canonical_successor_count"] != len(partition.successor_groups)
                or entry["production_successor_alias_multiplicity"] != len(teacher.aliases)
                or entry["productive_alias_count"] != productive
                or entry["virtual_alias_count"] != virtual
                or entry["model_scores_or_probabilities_stored"] is not False
                or entry["hazard_included"] is not False
            ):
                raise ProcessV2T1RuntimeError(
                    "T1 prepared entry state or successor census disagrees"
                )
        observed.append(str(entry["panel_entry_sha256"]))
        source_sha256 = str(entry["source_state_sha256"])
        if source_sha256 in observed_sources:
            raise ProcessV2T1RuntimeError("T1 prepared inputs repeat an exact source state")
        observed_sources.add(source_sha256)
    if len(set(observed)) != len(observed):
        raise ProcessV2T1RuntimeError("T1 prepared inputs repeat a panel entry")
    return artifact


def _build_prepared_completion(
    artifact: Mapping[str, Any],
    *,
    artifact_raw: bytes,
) -> dict[str, Any]:
    if artifact_raw != canonical_bytes(artifact) + b"\n":
        raise ProcessV2T1RuntimeError("T1 prepared completion received different bytes")
    body = {
        "schema": COMPLETION_SCHEMA,
        "schema_version": COMPLETION_SCHEMA_VERSION,
        "status": COMPLETION_STATUS,
        **authority_false_block(),
        "plan_sha256": artifact["plan_sha256"],
        "run_identity_sha256": artifact["run_identity_sha256"],
        "process_identity_sha256": artifact["process_identity_sha256"],
        "active8_completion_sha256": artifact["active8_completion_sha256"],
        "gate_zero_decision_sha256": artifact["gate_zero_decision_sha256"],
        "panel_sha256": artifact["panel_sha256"],
        "manifest_sha256": artifact["manifest_sha256"],
        "prepared_filename": PREPARED_FILENAME,
        "prepared_file_sha256": hashlib.sha256(artifact_raw).hexdigest(),
        "prepared_file_bytes": len(artifact_raw),
        "prepared_artifact_sha256": artifact["artifact_sha256"],
        "initial_model_state_sha256": artifact["model_binding"]["model_runtime"][
            "initial_model_state_sha256"
        ],
        "entry_count": artifact["entry_count"],
        "entry_inventory_sha256": artifact["entry_inventory_sha256"],
        "training_launched": False,
    }
    return {**body, "completion_sha256": canonical_sha256(body)}


def publish_process_v2_t1_prepared_inputs(
    plan: Mapping[str, Any],
    *,
    panel: Mapping[str, Any],
    source: ProcessV2T1Source,
    run_root: Path,
) -> Path:
    """Atomically publish prepared bytes, then their completion receipt."""

    validated_plan = validate_process_v2_t1_prepared_plan(plan, panel=panel, source=source)
    artifact = build_process_v2_t1_prepared_inputs(
        validated_plan,
        panel=panel,
        source=source,
        run_root=run_root,
    )
    root = Path(run_root)
    artifact_path = root / PREPARED_FILENAME
    try:
        write_bytes_if_absent(artifact_path, canonical_bytes(artifact) + b"\n")
    except ImmutableArtifactError as error:
        raise ProcessV2T1RuntimeError(str(error)) from error
    raw = artifact_path.read_bytes()
    if raw != canonical_bytes(artifact) + b"\n":
        raise ProcessV2T1RuntimeError("published T1 prepared bytes changed on reopen")
    completion = _build_prepared_completion(
        artifact,
        artifact_raw=raw,
    )
    completion_path = root / COMPLETION_FILENAME
    try:
        write_bytes_if_absent(completion_path, canonical_bytes(completion) + b"\n")
    except ImmutableArtifactError as error:
        raise ProcessV2T1RuntimeError(str(error)) from error
    return completion_path


def publish_reused_process_v2_t1_prepared_inputs(
    plan: Mapping[str, Any],
    *,
    panel: Mapping[str, Any],
    source: ProcessV2T1Source,
    leaf_run_root: Path,
    output_root: Path,
    source_revision: Mapping[str, Any],
    expected_plan_sha256: str,
) -> Path:
    """Publish a corrected reduction of one exact, already-complete leaf run."""

    artifact = build_reused_process_v2_t1_prepared_inputs(
        plan,
        panel=panel,
        source=source,
        run_root=leaf_run_root,
        source_revision=source_revision,
        expected_plan_sha256=expected_plan_sha256,
    )
    root = Path(output_root)
    artifact_path = root / PREPARED_FILENAME
    try:
        write_bytes_if_absent(artifact_path, canonical_bytes(artifact) + b"\n")
    except ImmutableArtifactError as error:
        raise ProcessV2T1RuntimeError(str(error)) from error
    raw = artifact_path.read_bytes()
    if raw != canonical_bytes(artifact) + b"\n":
        raise ProcessV2T1RuntimeError("published reused T1 prepared bytes changed on reopen")
    completion = _build_prepared_completion(artifact, artifact_raw=raw)
    completion_path = root / COMPLETION_FILENAME
    try:
        write_bytes_if_absent(completion_path, canonical_bytes(completion) + b"\n")
    except ImmutableArtifactError as error:
        raise ProcessV2T1RuntimeError(str(error)) from error
    return completion_path


def _load_prepared_completion(
    completion_path: Path, *, repo_root: Path, score_revision_rebind: bool = False
) -> tuple[dict[str, Any], dict[str, Any], str, str]:
    completion, completion_raw = _load_canonical(
        completion_path, label="the T1 prepared completion"
    )
    expected_fields = {
        "schema",
        "schema_version",
        "status",
        *authority_false_block(),
        "plan_sha256",
        "run_identity_sha256",
        "process_identity_sha256",
        "active8_completion_sha256",
        "gate_zero_decision_sha256",
        "panel_sha256",
        "manifest_sha256",
        "prepared_filename",
        "prepared_file_sha256",
        "prepared_file_bytes",
        "prepared_artifact_sha256",
        "initial_model_state_sha256",
        "entry_count",
        "entry_inventory_sha256",
        "training_launched",
        "completion_sha256",
    }
    if set(completion) != expected_fields:
        raise ProcessV2T1RuntimeError("T1 prepared completion field set disagrees")
    try:
        verify_self_hash(completion, field="completion_sha256", label="the T1 prepared completion")
        require_authority_false(completion, label="the T1 prepared completion")
    except ProcessV2SchemaError as error:
        raise ProcessV2T1RuntimeError(str(error)) from error
    if (
        completion["schema"] != COMPLETION_SCHEMA
        or completion["schema_version"] != COMPLETION_SCHEMA_VERSION
        or completion["status"] != COMPLETION_STATUS
        or completion["prepared_filename"] != PREPARED_FILENAME
        or completion["training_launched"] is not False
    ):
        raise ProcessV2T1RuntimeError("T1 prepared completion identity disagrees")
    artifact_path = Path(completion_path).resolve().parent / PREPARED_FILENAME
    artifact, artifact_raw = _load_canonical(artifact_path, label="the T1 prepared inputs")
    artifact = validate_process_v2_t1_prepared_inputs(
        artifact,
        repo_root=None if score_revision_rebind else repo_root,
    )
    if score_revision_rebind:
        frozen = _SCORE_REVISION_PREDECESSOR
        observed_completion_file_sha256 = hashlib.sha256(completion_raw).hexdigest()
        observed_prepared_file_sha256 = hashlib.sha256(artifact_raw).hexdigest()
        predecessor_runtime = artifact["model_binding"]["model_runtime"]
        if (
            completion["completion_sha256"] != frozen["prepared_completion_sha256"]
            or observed_completion_file_sha256
            != frozen["prepared_completion_file_sha256"]
            or artifact["artifact_sha256"] != frozen["prepared_artifact_sha256"]
            or observed_prepared_file_sha256 != frozen["prepared_file_sha256"]
            or artifact["implementation_sha256"]
            != frozen["prepared_implementation_sha256"]
        ):
            raise ProcessV2T1RuntimeError(
                "T1 score-revision input is not the exact allowlisted predecessor"
            )
        bound_runtime_ok = artifact["model_binding"]["model_runtime"] == predecessor_runtime
        scratch_process_ok = True
    else:
        scratch, binding = _scratch_runtime_from_descriptor(
            artifact["model_binding"]["model_runtime"], repo_root=repo_root
        )
        bound_runtime_ok = artifact["model_binding"] == binding
        scratch_process_ok = (
            scratch.process_identity_sha256 == artifact["process_identity_sha256"]
        )
    if (
        completion["prepared_file_sha256"] != hashlib.sha256(artifact_raw).hexdigest()
        or completion["prepared_file_bytes"] != len(artifact_raw)
        or completion["prepared_artifact_sha256"] != artifact["artifact_sha256"]
        or completion["process_identity_sha256"] != artifact["process_identity_sha256"]
        or completion["active8_completion_sha256"] != artifact["active8_completion_sha256"]
        or completion["gate_zero_decision_sha256"] != artifact["gate_zero_decision_sha256"]
        or completion["panel_sha256"] != artifact["panel_sha256"]
        or completion["manifest_sha256"] != artifact["manifest_sha256"]
        or completion["initial_model_state_sha256"]
        != artifact["model_binding"]["model_runtime"]["initial_model_state_sha256"]
        or completion["entry_count"] != artifact["entry_count"]
        or completion["entry_inventory_sha256"] != artifact["entry_inventory_sha256"]
        or not bound_runtime_ok
        or not scratch_process_ok
    ):
        raise ProcessV2T1RuntimeError(
            "T1 completion, prepared bytes, and current scratch runtime disagree"
        )
    return (
        completion,
        artifact,
        hashlib.sha256(completion_raw).hexdigest(),
        hashlib.sha256(artifact_raw).hexdigest(),
    )


@dataclass(frozen=True)
class LoadedProcessV2T1PreparedInputs:
    artifact: Mapping[str, Any]
    states_by_panel_entry_sha256: Mapping[str, Any]
    partitions_by_panel_entry_sha256: Mapping[str, CompiledStateSuccessorMap]


@dataclass(frozen=True)
class ProcessV2T1CacheView:
    completion: Mapping[str, Any]
    manifest: Mapping[str, Any]
    teacher_fibers_by_panel_entry_sha256: Mapping[str, TeacherSuccessorFiber]

    def record_for_panel_entry_sha256(self, identifier: str) -> Any:
        try:
            fiber = self.teacher_fibers_by_panel_entry_sha256[identifier]
        except KeyError as error:
            raise ProcessV2T1RuntimeError(
                "T1 runtime requested a teacher outside the prepared panel"
            ) from error
        return SimpleNamespace(teacher_fiber=fiber)


@dataclass(frozen=True)
class ProcessV2T1RuntimeInputs:
    """Duck-typed runtime consumed by the existing successor-level T1 core."""

    prepared: LoadedProcessV2T1PreparedInputs
    cache: ProcessV2T1CacheView
    capacity_policy: Mapping[str, Any]

    @property
    def entries(self) -> tuple[Mapping[str, Any], ...]:
        return tuple(self.prepared.artifact["entries"])


def load_process_v2_t1_runtime_inputs(
    completion_path: Path,
    *,
    capacity_policy_path: Path,
    repo_root: Path,
    score_revision_rebind: bool = False,
) -> tuple[ProcessV2T1RuntimeInputs, SemanticScratchRuntime, dict[str, Any]]:
    """Reopen CPU-prepared partitions and reconstruct the bound scratch model."""

    (
        completion,
        artifact,
        completion_file_sha256,
        prepared_file_sha256,
    ) = _load_prepared_completion(
        completion_path,
        repo_root=repo_root,
        score_revision_rebind=score_revision_rebind,
    )
    bridge: dict[str, Any] | None = None
    if score_revision_rebind:
        scratch, binding, bridge = _scratch_runtime_for_score_revision(
            artifact["model_binding"]["model_runtime"], repo_root=repo_root
        )
        if scratch.process_identity_sha256 != artifact["process_identity_sha256"]:
            raise ProcessV2T1RuntimeError(
                "current score-revision scratch process identity changed"
            )
    else:
        scratch, binding = _scratch_runtime_from_descriptor(
            artifact["model_binding"]["model_runtime"], repo_root=repo_root
        )
    policy, policy_file_sha256 = load_process_v2_t1_capacity_policy(
        capacity_policy_path, repo_root=repo_root
    )
    if not score_revision_rebind and binding != artifact["model_binding"]:
        raise ProcessV2T1RuntimeError("T1 scratch runtime differs from prepared inputs")
    states: dict[str, Any] = {}
    partitions: dict[str, CompiledStateSuccessorMap] = {}
    fibers: dict[str, TeacherSuccessorFiber] = {}
    observed_cells: set[str] = set()
    family_counts: dict[str, int] = {}
    for entry in artifact["entries"]:
        identifier = str(entry["panel_entry_sha256"])
        state = decode_state(entry["exact_state"])
        partition = compiled_successor_map_from_payload(entry["successor_partition"])
        fiber = teacher_successor_fiber_from_exact_digest(
            partition, str(entry["target_state_sha256"])
        )
        states[identifier] = state
        partitions[identifier] = partition
        fibers[identifier] = fiber
        observed_cells.add(str(entry["capability_cell_id"]))
        family = str(entry["model_family"])
        family_counts[family] = family_counts.get(family, 0) + 1
    roles = load_process_v2_chain_artifact(DEVELOPMENT_CELL_ROLES, repo_root=repo_root)
    required_cells = _required_editing_cell_ids(roles)
    if observed_cells != required_cells:
        raise ProcessV2T1RuntimeError(
            "T1 prepared inputs do not exactly cover the required Process-V2 cells"
        )
    minimums = policy.get("panel_cardinality", {}).get("minimum_entries_by_family", {})
    maximums = policy.get("panel_cardinality", {}).get("maximum_entries_by_family", {})
    if any(
        not int(minimums[family]) <= family_counts.get(family, 0) <= int(maximums[family])
        for family in policy["required_families"]
    ):
        raise ProcessV2T1RuntimeError("T1 prepared family counts violate the capacity policy")
    prepared = LoadedProcessV2T1PreparedInputs(
        artifact=MappingProxyType(dict(artifact)),
        states_by_panel_entry_sha256=MappingProxyType(states),
        partitions_by_panel_entry_sha256=MappingProxyType(partitions),
    )
    manifest = {
        "manifest_sha256": artifact["manifest_sha256"],
        "source_revision": artifact["source_revision"],
        "semantic_model_process_contract": {
            "file_sha256": binding["semantic_model_process_file_sha256"],
            "contract_sha256": binding["semantic_model_process_sha256"],
        },
        "panel_binding": {"panel_artifact_sha256": artifact["panel_sha256"]},
    }
    cache_completion = {
        **dict(completion),
        "initial_model_state_sha256": binding["model_runtime"][
            "initial_model_state_sha256"
        ],
        "manifest_sha256": artifact["manifest_sha256"],
    }
    cache = ProcessV2T1CacheView(
        completion=MappingProxyType(cache_completion),
        manifest=MappingProxyType(manifest),
        teacher_fibers_by_panel_entry_sha256=MappingProxyType(fibers),
    )
    runtime = ProcessV2T1RuntimeInputs(
        prepared=prepared,
        cache=cache,
        capacity_policy=MappingProxyType(dict(policy)),
    )
    panel_entry_metadata = [
        {
            "panel_entry_sha256": str(entry["panel_entry_sha256"]),
            "family": str(entry["model_family"]),
            "semantic_cell_id": str(entry["capability_cell_id"]),
        }
        for entry in artifact["entries"]
    ]
    provenance = {
        "capacity_policy_file_sha256": policy_file_sha256,
        "capacity_policy_sha256": policy["policy_sha256"],
        "prepared_completion_file_sha256": completion_file_sha256,
        "prepared_completion_sha256": completion["completion_sha256"],
        "prepared_input_file_sha256": prepared_file_sha256,
        "prepared_input_artifact_sha256": artifact["artifact_sha256"],
        "panel_sha256": artifact["panel_sha256"],
        "panel_entry_inventory_sha256": artifact["panel_entry_inventory_sha256"],
        "panel_entry_binding_count": len(panel_entry_metadata),
        "panel_entry_metadata_sha256": canonical_sha256(panel_entry_metadata),
        "gate_zero_decision_sha256": artifact["gate_zero_decision_sha256"],
        "active8_completion_sha256": artifact["active8_completion_sha256"],
        "process_identity_sha256": artifact["process_identity_sha256"],
        "initial_model_state_sha256": binding["model_runtime"][
            "initial_model_state_sha256"
        ],
        "leaf_source_revision_sha256": artifact["leaf_source_revision"][
            "source_revision_sha256"
        ],
        "leaf_implementation_sha256": artifact["leaf_implementation_sha256"],
        "leaf_reuse": dict(artifact["leaf_reuse"]),
    }
    if score_revision_rebind:
        if bridge is None:
            raise ProcessV2T1RuntimeError("T1 score-revision bridge identity is absent")
        provenance["score_revision_rebind"] = bridge
    return runtime, scratch, provenance


__all__ = [
    "COMPLETION_FILENAME",
    "LEAF_FILENAME",
    "PLAN_FILENAME",
    "PREPARED_FILENAME",
    "LoadedProcessV2T1PreparedInputs",
    "ProcessV2T1CacheView",
    "ProcessV2T1RuntimeError",
    "ProcessV2T1RuntimeInputs",
    "authenticate_process_v2_t1_prepared_plan_for_worker",
    "build_process_v2_score_revised_scratch_runtime",
    "build_process_v2_t1_prepared_inputs",
    "build_process_v2_t1_prepared_plan",
    "build_process_v2_t1_scratch_runtime",
    "build_reused_process_v2_t1_prepared_inputs",
    "compile_authenticated_process_v2_t1_prepared_leaf",
    "compile_process_v2_t1_prepared_leaf",
    "load_process_v2_t1_capacity_policy",
    "load_process_v2_t1_runtime_inputs",
    "publish_process_v2_t1_prepared_inputs",
    "publish_reused_process_v2_t1_prepared_inputs",
    "validate_process_v2_t1_leaf_reuse_plan",
    "validate_process_v2_t1_prepared_inputs",
    "validate_process_v2_t1_prepared_leaf",
    "validate_process_v2_t1_prepared_plan",
    "write_process_v2_t1_prepared_leaf",
    "write_process_v2_t1_prepared_plan",
]
