"""Prospective V2 validation audit for component-factored cycle opening.

The V1 audit correctly failed, but it also established that RDKit's
whole-molecule resonance supplier is not complete over independently varying,
slot-labeled conjugated groups.  This module preserves that negative result and
performs a new comparison against the independent complete global MILP oracle.

The V1 receipt is reused only as a self-hashed ledger of the exact admitted
validation source set.  Molecular states are decoded from the same physically
pinned persistent-slot shard.  No state is reconstructed from SMILES, and no
train, controller-validation, or test data are read.
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
from collections import Counter, defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import scipy
from rdkit import rdBase

from compose_v4.chem.persistent_state_identity import persistent_slot_state_sha256
from compose_v4.data.packed_trace_store import read_frozen_source_addressed_packed_shard
from compose_v4.experiments.aromatic_cycle_open_global_oracle import (
    GLOBAL_MILP_ORACLE_STATUS,
    enumerate_global_milp_kekule_aliases,
)
from compose_v4.experiments.aromatic_cycle_open_semantics import (
    AromaticCycleOpenAliasOverflow,
    enumerate_component_factored_kekule_assignments,
)
from compose_v4.experiments.editing_cycle_open_component_equivalence import (
    NON_AUTHORIZING_STATUS as V1_NON_AUTHORIZING_STATUS,
)
from compose_v4.experiments.editing_cycle_open_component_equivalence import (
    RECEIPT_SCHEMA as V1_RECEIPT_SCHEMA,
)
from compose_v4.experiments.editing_cycle_open_component_equivalence import (
    RECEIPT_SCHEMA_VERSION as V1_RECEIPT_SCHEMA_VERSION,
)
from compose_v4.experiments.editing_cycle_open_component_equivalence import (
    RESULT_SCHEMA as V1_RESULT_SCHEMA,
)
from compose_v4.experiments.editing_cycle_open_component_equivalence import (
    RESULT_SCHEMA_VERSION as V1_RESULT_SCHEMA_VERSION,
)
from compose_v4.experiments.editing_cycle_open_component_equivalence import (
    _bridge_edges,
    _semantic_aromatic_edges,
    compare_one_edge,
    file_sha256,
    semantic_sha256,
)
from compose_v4.rewrite.action_codec import canonical_family
from compose_v4.rewrite.kernel import canonical_state_key

CONTRACT_SCHEMA = "compose.editing.cycle_open_global_equivalence_contract"
CONTRACT_SCHEMA_VERSION = 2
PLAN_SCHEMA = "compose.editing.cycle_open_global_equivalence_plan"
PLAN_SCHEMA_VERSION = 2
SOURCE_RECORD_SCHEMA = "compose.editing.cycle_open_global_equivalence_source"
SOURCE_RECORD_SCHEMA_VERSION = 2
RECEIPT_SCHEMA = "compose.editing.cycle_open_global_equivalence_receipt"
RECEIPT_SCHEMA_VERSION = 2
RESULT_SCHEMA = "compose.editing.cycle_open_global_equivalence_result"
RESULT_SCHEMA_VERSION = 2
NON_AUTHORIZING_STATUS = (
    "VALIDATION_ONLY_COMPLETE_GLOBAL_CYCLE_OPEN_EQUIVALENCE_NO_TRAINING_AUTHORITY"
)
PINNED_CONTRACT_FILE_SHA256 = (
    "715ac7f904f3d5b73723e820984d35d4d6660a8b7b430316c77fa16bcac0de15"
)

_CONTRACT_FIELDS = {
    "comparison_policy",
    "contract_sha256",
    "edge_scope",
    "equivalence_gate",
    "exact_validation_input",
    "excluded_partitions",
    "oracle_policy",
    "partitions",
    "runtime_policy",
    "schema",
    "schema_version",
    "source_scope",
    "status",
    "training_authorized",
    "v1_negative_evidence",
}
_COMPARISON_POLICY = {
    "component_resolver": (
        "complete_degree_constrained_assignments_per_resonance_invariant_aromatic_component"
    ),
    "equivalence_fields": [
        "admission",
        "typed_rejection",
        "source_identity",
        "selected_edge",
        "semantic_aromatic_class",
        "global_alias_cardinality",
        "global_forced_single_cardinality",
        "canonical_product_identity",
        "fixed_coordinate_exact_successor_identity",
        "inverse_bond_order",
    ],
    "exact_representative_policy": (
        "lexicographically_minimum_persistent_slot_state_within_the_unique_canonical_product_group"
    ),
    "work_policy": "record_actual_executions_separately_from_implied_global_cardinality",
}
_ORACLE_POLICY = {
    "assignment_domain": (
        "all_slot_labeled_binary_single_double_assignments_with_exact_source_double_incidence"
    ),
    "identity": GLOBAL_MILP_ORACLE_STATUS,
    "maximum_assignments": 4096,
    "overflow_policy": "typed_incomplete_comparison_blocks_gate",
    "postsolve_checks": [
        "binary_closeness",
        "exact_integer_incidence",
        "production_state_validity",
        "connectedness",
        "canonical_source_identity",
    ],
    "solver": "scipy.optimize.milp_global_problem_with_iterative_no_good_constraints",
}
_RUNTIME_POLICY = {
    "python": "3.14.2",
    "numpy": "2.4.2",
    "scipy": "1.17.1",
    "rdkit": "2025.09.6",
    "omp_num_threads": "1",
}


class CycleOpenGlobalEquivalenceError(RuntimeError):
    """The prospective V2 audit cannot establish its declared evidence."""


def canonical_json_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise CycleOpenGlobalEquivalenceError(
            "global equivalence value is not finite deterministic JSON"
        ) from error


def pretty_json_bytes(value: object) -> bytes:
    try:
        return (
            json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise CycleOpenGlobalEquivalenceError(
            "global equivalence artifact is not finite deterministic JSON"
        ) from error


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def load_json(path: str | Path, *, description: str) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise CycleOpenGlobalEquivalenceError(
            f"cannot load {description}: {path}"
        ) from error
    if not isinstance(value, dict):
        raise CycleOpenGlobalEquivalenceError(f"{description} must be a JSON object")
    return value


def load_contract(path: str | Path) -> dict[str, Any]:
    if file_sha256(path) != PINNED_CONTRACT_FILE_SHA256:
        raise CycleOpenGlobalEquivalenceError(
            "global equivalence contract physical bytes disagree"
        )
    contract = load_json(path, description="global equivalence contract")
    contract_hash = contract.get("contract_sha256")
    body = {key: value for key, value in contract.items() if key != "contract_sha256"}
    if not _is_sha256(contract_hash) or semantic_sha256(body) != contract_hash:
        raise CycleOpenGlobalEquivalenceError(
            "global equivalence contract self-hash disagrees"
        )
    if (
        set(contract) != _CONTRACT_FIELDS
        or contract.get("schema") != CONTRACT_SCHEMA
        or contract.get("schema_version") != CONTRACT_SCHEMA_VERSION
        or contract.get("status") != NON_AUTHORIZING_STATUS
        or contract.get("training_authorized") is not False
        or contract.get("partitions") != ["validation"]
        or contract.get("excluded_partitions")
        != [
            "train",
            "controller_validation",
            "test",
        ]
        or contract.get("equivalence_gate")
        != "zero_mismatches_zero_oracle_overflows_complete_declared_edge_coverage"
        or contract.get("source_scope")
        != "all_distinct_exact_source_states_in_the_self_hashed_v1_admitted_cycle_attach_source_ledger"
        or contract.get("edge_scope")
        != "all_resonance_invariant_aromatic_edges_in_each_declared_source"
        or contract.get("comparison_policy") != _COMPARISON_POLICY
    ):
        raise CycleOpenGlobalEquivalenceError(
            "global equivalence contract scope, status, or gate disagrees"
        )
    if contract.get("oracle_policy") != _ORACLE_POLICY:
        raise CycleOpenGlobalEquivalenceError(
            "global equivalence oracle policy disagrees"
        )
    for section in ("v1_negative_evidence", "exact_validation_input"):
        values = contract.get(section)
        if not isinstance(values, Mapping) or any(
            not _is_sha256(value)
            for key, value in values.items()
            if key.endswith("sha256")
        ):
            raise CycleOpenGlobalEquivalenceError(
                f"global equivalence {section} identity is malformed"
            )
    if contract.get("runtime_policy") != _RUNTIME_POLICY:
        raise CycleOpenGlobalEquivalenceError(
            "global equivalence runtime policy disagrees"
        )
    return contract


def validate_v1_receipt(
    value: Mapping[str, Any],
    *,
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    receipt = dict(value)
    receipt_hash = receipt.pop("receipt_sha256", None)
    expected = contract["v1_negative_evidence"]
    if (
        receipt_hash != semantic_sha256(receipt)
        or receipt_hash != expected["receipt_sha256"]
    ):
        raise CycleOpenGlobalEquivalenceError(
            "V1 source-ledger receipt self-hash disagrees"
        )
    receipt["receipt_sha256"] = receipt_hash
    task = receipt.get("task")
    source_map = receipt.get("source_trace_identity_sha256s")
    reconciliations = receipt.get("reconciliations")
    if (
        receipt.get("schema") != V1_RECEIPT_SCHEMA
        or receipt.get("schema_version") != V1_RECEIPT_SCHEMA_VERSION
        or receipt.get("status") != V1_NON_AUTHORIZING_STATUS
        or receipt.get("training_authorized") is not False
        or receipt.get("plan_sha256") != expected["plan_sha256"]
        or receipt.get("implementation_sha256") != expected["implementation_sha256"]
        or not isinstance(task, Mapping)
        or task.get("packed_shard_content_sha256")
        != contract["exact_validation_input"]["packed_shard_content_sha256"]
        or not isinstance(source_map, Mapping)
        or not source_map
        or not all(
            isinstance(digest, str) and isinstance(identities, list)
            for digest, identities in source_map.items()
        )
        or not isinstance(reconciliations, Mapping)
        or not reconciliations
        or not all(value is True for value in reconciliations.values())
    ):
        raise CycleOpenGlobalEquivalenceError(
            "V1 source ledger schema, identity, authority, or reconciliation disagrees"
        )
    return receipt


def validate_v1_result(
    value: Mapping[str, Any],
    *,
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    result = dict(value)
    result_hash = result.pop("result_sha256", None)
    expected = contract["v1_negative_evidence"]
    if (
        result_hash != semantic_sha256(result)
        or result_hash != expected["result_sha256"]
    ):
        raise CycleOpenGlobalEquivalenceError("V1 negative result self-hash disagrees")
    result["result_sha256"] = result_hash
    totals = result.get("totals")
    nonpassing = result.get("nonpassing_source_records")
    gate = result.get("equivalence_gate")
    if (
        result.get("schema") != V1_RESULT_SCHEMA
        or result.get("schema_version") != V1_RESULT_SCHEMA_VERSION
        or result.get("status") != V1_NON_AUTHORIZING_STATUS
        or result.get("training_authorized") is not False
        or result.get("plan_sha256") != expected["plan_sha256"]
        or result.get("implementation_sha256") != expected["implementation_sha256"]
        or result.get("receipt_sha256s") != [expected["receipt_sha256"]]
        or not isinstance(totals, Mapping)
        or totals.get("mismatch_edge_count") != expected["mismatch_edge_count"]
        or totals.get("semantic_aromatic_edge_count")
        != expected["semantic_aromatic_edge_count"]
        or not isinstance(nonpassing, list)
        or len(nonpassing) != expected["nonpassing_source_count"]
        or not isinstance(gate, Mapping)
        or gate.get("passed") is not False
    ):
        raise CycleOpenGlobalEquivalenceError(
            "V1 negative result identity, authority, or failure census disagrees"
        )
    return result


def git_revision(*, repo_root: str | Path) -> dict[str, Any]:
    root = Path(repo_root)
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise CycleOpenGlobalEquivalenceError("cannot bind V2 audit to Git") from error
    return {"commit": commit, "tree_dirty": bool(dirty)}


def implementation_identity(*, repo_root: str | Path) -> dict[str, Any]:
    root = Path(repo_root)
    sources = (
        "scripts/audit_editing_cycle_open_global_equivalence.py",
        "src/compose_v4/experiments/editing_cycle_open_global_equivalence.py",
        "src/compose_v4/experiments/aromatic_cycle_open_global_oracle.py",
        "src/compose_v4/experiments/aromatic_cycle_open_semantics.py",
        "src/compose_v4/experiments/editing_cycle_open_component_equivalence.py",
        "src/compose_v4/chem/aromaticity.py",
        "src/compose_v4/chem/molecular_graph.py",
        "src/compose_v4/chem/persistent_state_identity.py",
        "src/compose_v4/chem/state.py",
        "src/compose_v4/data/packed_trace_store.py",
        "src/compose_v4/rewrite/action_codec.py",
        "src/compose_v4/rewrite/kernel.py",
        "src/compose_v4/rewrite/operators.py",
    )
    hashes = {relative: file_sha256(root / relative) for relative in sources}
    return {"sources": hashes, "implementation_sha256": semantic_sha256(hashes)}


def runtime_identity() -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "rdkit": rdBase.rdkitVersion,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_count": os.cpu_count(),
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        "milp_decision_precision": "float64_with_exact_binary_and_integer_incidence_recheck",
    }


def build_plan(
    contract: Mapping[str, Any],
    *,
    contract_path: str | Path,
    v1_receipt_path: str | Path,
    v1_result_path: str | Path,
    packed_shard_path: str | Path,
    repo_root: str | Path,
) -> dict[str, Any]:
    revision = git_revision(repo_root=repo_root)
    if revision["tree_dirty"]:
        raise CycleOpenGlobalEquivalenceError(
            "V2 audit requires a clean committed tree"
        )
    expected = contract["v1_negative_evidence"]
    exact = contract["exact_validation_input"]
    runtime = runtime_identity()
    observed_runtime_policy = {
        key: runtime[key]
        for key in ("python", "numpy", "scipy", "rdkit", "omp_num_threads")
    }
    if observed_runtime_policy != contract["runtime_policy"]:
        raise CycleOpenGlobalEquivalenceError(
            f"V2 audit runtime disagrees with the frozen policy: {observed_runtime_policy}"
        )
    observed_hashes = {
        "contract_file_sha256": file_sha256(contract_path),
        "v1_receipt_file_sha256": file_sha256(v1_receipt_path),
        "v1_result_file_sha256": file_sha256(v1_result_path),
        "packed_shard_content_sha256": file_sha256(packed_shard_path),
        "packed_manifest_sha256": file_sha256(
            Path(packed_shard_path).with_name("shard_0000.jsonl.manifest.json")
        ),
        "packed_provenance_overlay_sha256": file_sha256(
            Path(str(packed_shard_path) + ".provenance.json")
        ),
    }
    expected_material_hashes = {
        "v1_receipt_file_sha256": expected["receipt_file_sha256"],
        "v1_result_file_sha256": expected["result_file_sha256"],
        "packed_shard_content_sha256": exact["packed_shard_content_sha256"],
        "packed_manifest_sha256": exact["packed_manifest_sha256"],
        "packed_provenance_overlay_sha256": exact["packed_provenance_overlay_sha256"],
    }
    if {
        key: value
        for key, value in observed_hashes.items()
        if key != "contract_file_sha256"
    } != expected_material_hashes:
        raise CycleOpenGlobalEquivalenceError(
            f"V2 audit physical inputs disagree: {observed_hashes}"
        )
    body = {
        "schema": PLAN_SCHEMA,
        "schema_version": PLAN_SCHEMA_VERSION,
        "status": NON_AUTHORIZING_STATUS,
        "training_authorized": False,
        "contract_sha256": contract["contract_sha256"],
        "input_paths": {
            "contract": str(Path(contract_path).resolve()),
            "v1_receipt": str(Path(v1_receipt_path).resolve()),
            "v1_result": str(Path(v1_result_path).resolve()),
            "packed_shard": str(Path(packed_shard_path).resolve()),
        },
        "input_file_sha256s": observed_hashes,
        "code_revision": revision,
        "implementation": implementation_identity(repo_root=repo_root),
        "runtime": runtime,
        "seed_policy": "none_complete_deterministic_feasibility_enumeration",
        "partitions": ["validation"],
        "excluded_partitions": ["train", "controller_validation", "test"],
        "maximum_assignments": contract["oracle_policy"]["maximum_assignments"],
    }
    return {**body, "plan_sha256": semantic_sha256(body)}


def _trace_identity(address: object) -> str:
    return semantic_sha256(
        {
            "packed_shard_content_sha256": address.packed_shard_content_sha256,
            "entry_index": address.entry_index,
            "trace_id": address.trace_id,
        }
    )


def _audit_one_source(
    state: object,
    *,
    maximum_assignments: int,
) -> dict[str, Any]:
    source_digest = persistent_slot_state_sha256(state)
    components = enumerate_component_factored_kekule_assignments(state)
    aromatic_edges = _semantic_aromatic_edges(state)
    bridge_edges = _bridge_edges(state)
    try:
        global_result = enumerate_global_milp_kekule_aliases(
            state,
            maximum_assignments=maximum_assignments,
        )
        oracle_overflow = None
    except AromaticCycleOpenAliasOverflow as error:
        global_result = None
        oracle_overflow = error
    edge_records = tuple(
        compare_one_edge(
            state,
            edge,
            maximum_oracle_structures=maximum_assignments,
            exhaustive_enumeration=(
                global_result.enumeration if global_result is not None else None
            ),
            oracle_already_overflowed=oracle_overflow is not None,
        )
        for edge in aromatic_edges
    )
    outcomes = Counter(str(record["outcome"]) for record in edge_records)
    counts = {
        "aromatic_component_count": len(components),
        "semantic_aromatic_edge_count": len(aromatic_edges),
        "semantic_aromatic_bridge_edge_count": sum(
            edge in bridge_edges for edge in aromatic_edges
        ),
        "complete_edge_comparison_count": sum(
            bool(record["comparison_complete"]) for record in edge_records
        ),
        "equivalent_edge_count": outcomes["equivalent"],
        "mismatch_edge_count": outcomes["mismatch"],
        "oracle_overflow_edge_count": outcomes["exhaustive_oracle_overflow"],
        "factored_executed_product_count": sum(
            int(record["work"]["factored_executed_products"]) for record in edge_records
        ),
        "oracle_executed_product_count": sum(
            int(record["work"]["oracle_executed_products"] or 0)
            for record in edge_records
        ),
    }
    semantic_fields = sorted(
        {field for record in edge_records for field in record["field_equivalence"]}
    )
    counts.update(
        {
            f"semantic_field_mismatch__{field}": sum(
                not bool(record["field_equivalence"].get(field, True))
                for record in edge_records
                if record["comparison_complete"]
            )
            for field in semantic_fields
        }
    )
    reconciliations = {
        "edge_partition": (
            counts["equivalent_edge_count"]
            + counts["mismatch_edge_count"]
            + counts["oracle_overflow_edge_count"]
            == counts["semantic_aromatic_edge_count"]
        ),
        "complete_partition": (
            counts["complete_edge_comparison_count"]
            == counts["equivalent_edge_count"] + counts["mismatch_edge_count"]
        ),
        "oracle_alias_cardinality_or_typed_overflow": (
            oracle_overflow is not None
            or (
                global_result is not None
                and global_result.diagnostics.preserving_alias_count
                == prod_or_zero(
                    tuple(len(component.bond_orders) for component in components)
                )
            )
        ),
    }
    if not all(reconciliations.values()):
        raise CycleOpenGlobalEquivalenceError(
            f"V2 source comparison denominators disagree: {reconciliations}"
        )
    body = {
        "schema": SOURCE_RECORD_SCHEMA,
        "schema_version": SOURCE_RECORD_SCHEMA_VERSION,
        "status": NON_AUTHORIZING_STATUS,
        "training_authorized": False,
        "exact_source_state_sha256": source_digest,
        "source_canonical_key_sha256": semantic_sha256(canonical_state_key(state)),
        "component_edge_counts": [len(component.edges) for component in components],
        "component_assignment_counts": [
            len(component.bond_orders) for component in components
        ],
        "oracle_diagnostics": (
            {
                "completed_source_count": 1,
                "overflow_source_count": 0,
                "semantic_aromatic_edge_count": (
                    global_result.diagnostics.semantic_aromatic_edge_count
                ),
                "constrained_vertex_count_on_completed_sources": (
                    global_result.diagnostics.constrained_vertex_count
                ),
                "feasible_assignment_count_on_completed_sources": (
                    global_result.diagnostics.feasible_assignment_count
                ),
                "preserving_alias_count_on_completed_sources": (
                    global_result.diagnostics.preserving_alias_count
                ),
                "solver_call_count_on_completed_sources": (
                    global_result.diagnostics.solver_call_count
                ),
                "observed_assignments_at_overflow": 0,
            }
            if global_result is not None
            else {
                "completed_source_count": 0,
                "overflow_source_count": 1,
                "semantic_aromatic_edge_count": len(aromatic_edges),
                "constrained_vertex_count_on_completed_sources": 0,
                "feasible_assignment_count_on_completed_sources": 0,
                "preserving_alias_count_on_completed_sources": 0,
                "solver_call_count_on_completed_sources": 0,
                "observed_assignments_at_overflow": int(
                    oracle_overflow.observed_structures
                ),
            }
        ),
        "counts": counts,
        "outcomes": dict(sorted(outcomes.items())),
        "edge_record_sha256s": [
            record["edge_record_sha256"] for record in edge_records
        ],
        "nonpassing_edge_records": [
            record for record in edge_records if not bool(record["semantic_equivalent"])
        ],
        "reconciliations": reconciliations,
    }
    return {**body, "source_record_sha256": semantic_sha256(body)}


def prod_or_zero(values: tuple[int, ...]) -> int:
    if not values:
        return 0
    result = 1
    for value in values:
        result *= value
    return result


def audit_exact_validation_shard(
    contract: Mapping[str, Any],
    plan: Mapping[str, Any],
    v1_receipt: Mapping[str, Any],
    *,
    packed_shard_path: str | Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    exact = contract["exact_validation_input"]
    expected_sources = {
        str(digest): tuple(str(identity) for identity in identities)
        for digest, identities in v1_receipt["source_trace_identity_sha256s"].items()
    }
    expected_identity_to_source: dict[str, str] = {}
    for source_digest, identities in expected_sources.items():
        if not _is_sha256(source_digest) or not identities:
            raise CycleOpenGlobalEquivalenceError(
                "V1 source ledger contains a malformed source"
            )
        for identity in identities:
            if not _is_sha256(identity) or identity in expected_identity_to_source:
                raise CycleOpenGlobalEquivalenceError(
                    "V1 source ledger contains a malformed or duplicate trace identity"
                )
            expected_identity_to_source[identity] = source_digest

    source_states: dict[str, object] = {}
    observed_identities: dict[str, set[str]] = defaultdict(set)
    rows = read_frozen_source_addressed_packed_shard(
        packed_shard_path,
        expected_shard_sha256=exact["packed_shard_content_sha256"],
        expected_manifest_sha256=exact["packed_manifest_sha256"],
        expected_overlay_sha256=exact["packed_provenance_overlay_sha256"],
        verify_fraction=0.0,
    )
    physical_traces = 0
    physical_cycle_attach_rows = 0
    for addressed in rows:
        physical_traces += 1
        identity = _trace_identity(addressed.address)
        for progress_index, step in enumerate(addressed.trace.steps):
            if canonical_family(step.rule_name) != "cycle_attach":
                continue
            physical_cycle_attach_rows += 1
            expected_source_digest = expected_identity_to_source.get(identity)
            if expected_source_digest is None:
                continue
            state = addressed.path.state_at(progress_index)
            observed_source_digest = persistent_slot_state_sha256(state)
            if observed_source_digest != expected_source_digest:
                raise CycleOpenGlobalEquivalenceError(
                    "V1 trace identity resolves to a different exact source state"
                )
            previous = source_states.setdefault(observed_source_digest, state)
            if persistent_slot_state_sha256(previous) != observed_source_digest:
                raise CycleOpenGlobalEquivalenceError(
                    "source-state deduplication changed identity"
                )
            observed_identities[observed_source_digest].add(identity)

    observed_source_map = {
        digest: tuple(sorted(identities))
        for digest, identities in observed_identities.items()
    }
    normalized_expected_map = {
        digest: tuple(sorted(identities))
        for digest, identities in expected_sources.items()
    }
    if observed_source_map != normalized_expected_map or set(source_states) != set(
        expected_sources
    ):
        raise CycleOpenGlobalEquivalenceError(
            "exact shard does not reproduce the V1 admitted validation source ledger"
        )

    source_records = [
        _audit_one_source(
            source_states[digest],
            maximum_assignments=int(plan["maximum_assignments"]),
        )
        for digest in sorted(source_states)
    ]
    receipt_body = {
        "schema": RECEIPT_SCHEMA,
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "status": NON_AUTHORIZING_STATUS,
        "training_authorized": False,
        "plan_sha256": plan["plan_sha256"],
        "implementation_sha256": plan["implementation"]["implementation_sha256"],
        "v1_receipt_sha256": v1_receipt["receipt_sha256"],
        "counts": {
            "physical_traces_scanned": physical_traces,
            "physical_cycle_attach_rows_scanned": physical_cycle_attach_rows,
            "admitted_source_trace_identities": len(expected_identity_to_source),
            "unique_exact_sources": len(source_states),
        },
        "source_trace_identity_map_sha256": semantic_sha256(normalized_expected_map),
        "source_records": source_records,
        "reconciliations": {
            "source_identity_set": set(source_states) == set(expected_sources),
            "source_trace_identity_map": observed_source_map == normalized_expected_map,
            "source_record_count": len(source_records) == len(source_states),
        },
    }
    receipt = {**receipt_body, "receipt_sha256": semantic_sha256(receipt_body)}

    totals: Counter[str] = Counter()
    oracle_totals: Counter[str] = Counter()
    for record in source_records:
        totals.update({key: int(value) for key, value in record["counts"].items()})
        oracle_totals.update(
            {key: int(value) for key, value in record["oracle_diagnostics"].items()}
        )
    complete = (
        len(source_records) > 0
        and totals["semantic_aromatic_edge_count"] > 0
        and totals["oracle_overflow_edge_count"] == 0
        and totals["mismatch_edge_count"] == 0
        and totals["complete_edge_comparison_count"]
        == totals["semantic_aromatic_edge_count"]
    )
    result_body = {
        "schema": RESULT_SCHEMA,
        "schema_version": RESULT_SCHEMA_VERSION,
        "status": NON_AUTHORIZING_STATUS,
        "training_authorized": False,
        "plan_sha256": plan["plan_sha256"],
        "implementation_sha256": plan["implementation"]["implementation_sha256"],
        "receipt_sha256": receipt["receipt_sha256"],
        "unique_source_count": len(source_records),
        "totals": dict(sorted(totals.items())),
        "oracle_totals": dict(sorted(oracle_totals.items())),
        "maximum_preserving_alias_count": max(
            int(
                record["oracle_diagnostics"][
                    "preserving_alias_count_on_completed_sources"
                ]
            )
            for record in source_records
        ),
        "source_record_sha256s": [
            record["source_record_sha256"] for record in source_records
        ],
        "nonpassing_source_records": [
            {
                "exact_source_state_sha256": record["exact_source_state_sha256"],
                "source_record_sha256": record["source_record_sha256"],
                "mismatch_edge_count": record["counts"]["mismatch_edge_count"],
                "oracle_overflow_edge_count": record["counts"][
                    "oracle_overflow_edge_count"
                ],
                "nonpassing_edge_records": record["nonpassing_edge_records"],
            }
            for record in source_records
            if record["counts"]["mismatch_edge_count"] > 0
            or record["counts"]["oracle_overflow_edge_count"] > 0
        ],
        "equivalence_gate": {
            "passed": complete,
            "required": (
                "zero mismatches, zero oracle overflows, and complete coverage of every "
                "semantic aromatic edge in the declared validation source set"
            ),
        },
    }
    result = {**result_body, "result_sha256": semantic_sha256(result_body)}
    return receipt, result


def atomic_write_if_absent(path: str | Path, payload: bytes) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if destination.read_bytes() != payload:
            raise CycleOpenGlobalEquivalenceError(
                f"immutable output already exists with different bytes: {destination}"
            )
        return
    temporary = destination.with_name(f".{destination.name}.tmp.{os.getpid()}")
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


__all__ = [
    "CONTRACT_SCHEMA",
    "CONTRACT_SCHEMA_VERSION",
    "NON_AUTHORIZING_STATUS",
    "PLAN_SCHEMA",
    "PLAN_SCHEMA_VERSION",
    "RECEIPT_SCHEMA",
    "RECEIPT_SCHEMA_VERSION",
    "RESULT_SCHEMA",
    "RESULT_SCHEMA_VERSION",
    "SOURCE_RECORD_SCHEMA",
    "SOURCE_RECORD_SCHEMA_VERSION",
    "CycleOpenGlobalEquivalenceError",
    "atomic_write_if_absent",
    "audit_exact_validation_shard",
    "build_plan",
    "canonical_json_bytes",
    "git_revision",
    "implementation_identity",
    "load_contract",
    "load_json",
    "pretty_json_bytes",
    "prod_or_zero",
    "runtime_identity",
    "validate_v1_receipt",
    "validate_v1_result",
]
