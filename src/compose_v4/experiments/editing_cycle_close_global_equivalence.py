"""Prospective validation-only global equivalence audit for semantic closing.

This audit compares the component-factored cycle-close prototype against a
structurally independent complete global MILP enumeration. It reads only the
exact persistent-slot validation sources already frozen by the completed
cycle-open V2 audit. It cannot authorize training or alter the production
process.
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import scipy
from rdkit import rdBase

from compose_v4.chem.molecular_graph import MolecularGraph, is_element
from compose_v4.chem.persistent_state_identity import persistent_slot_state_sha256
from compose_v4.data.packed_trace_store import read_frozen_source_addressed_packed_shard
from compose_v4.experiments.aromatic_cycle_close_semantics import (
    AromaticCycleCloseRejectionCode,
    prepare_component_factored_cycle_close_context,
    resolve_component_factored_cycle_close,
)
from compose_v4.experiments.aromatic_cycle_open_global_oracle import (
    GLOBAL_MILP_ORACLE_STATUS,
    enumerate_global_milp_kekule_aliases,
)
from compose_v4.experiments.aromatic_cycle_open_semantics import (
    AromaticCycleOpenAliasOverflow,
)
from compose_v4.experiments.editing_cycle_open_component_equivalence import (
    file_sha256,
    semantic_sha256,
)
from compose_v4.rewrite.action_codec import canonical_family
from compose_v4.rewrite.kernel import canonical_state_key
from compose_v4.rewrite.operators import (
    BondInsert,
    apply_bond_insert,
    is_valid_bond_insert,
)

CONTRACT_SCHEMA = "compose.editing.cycle_close_global_equivalence_contract"
CONTRACT_SCHEMA_VERSION = 1
PLAN_SCHEMA = "compose.editing.cycle_close_global_equivalence_plan"
PLAN_SCHEMA_VERSION = 1
SOURCE_RECORD_SCHEMA = "compose.editing.cycle_close_global_equivalence_source"
SOURCE_RECORD_SCHEMA_VERSION = 1
RECEIPT_SCHEMA = "compose.editing.cycle_close_global_equivalence_receipt"
RECEIPT_SCHEMA_VERSION = 1
RESULT_SCHEMA = "compose.editing.cycle_close_global_equivalence_result"
RESULT_SCHEMA_VERSION = 1
NON_AUTHORIZING_STATUS = (
    "VALIDATION_ONLY_COMPLETE_GLOBAL_CYCLE_CLOSE_EQUIVALENCE_NO_TRAINING_AUTHORITY"
)
PINNED_CONTRACT_FILE_SHA256 = (
    "0982d99117ad86f703262e9be91154c6e991f7b3d31a687583da3aa472c7b16f"
)
SOURCE_RECEIPT_SCHEMA = "compose.editing.cycle_open_global_equivalence_receipt"
SOURCE_RECEIPT_SCHEMA_VERSION = 2
SOURCE_RESULT_SCHEMA = "compose.editing.cycle_open_global_equivalence_result"
SOURCE_RESULT_SCHEMA_VERSION = 2
SOURCE_STATUS = (
    "VALIDATION_ONLY_COMPLETE_GLOBAL_CYCLE_OPEN_EQUIVALENCE_NO_TRAINING_AUTHORITY"
)


class CycleCloseGlobalEquivalenceError(RuntimeError):
    """The declared cycle-close equivalence evidence cannot be established."""


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
        raise CycleCloseGlobalEquivalenceError(
            "cycle-close audit value is not finite deterministic JSON"
        ) from error


def pretty_json_bytes(value: object) -> bytes:
    return json.dumps(value, indent=2, sort_keys=True, allow_nan=False).encode() + b"\n"


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
        raise CycleCloseGlobalEquivalenceError(
            f"cannot load {description}: {path}"
        ) from error
    if not isinstance(value, dict):
        raise CycleCloseGlobalEquivalenceError(f"{description} must be an object")
    return value


def load_contract(path: str | Path) -> dict[str, Any]:
    if file_sha256(path) != PINNED_CONTRACT_FILE_SHA256:
        raise CycleCloseGlobalEquivalenceError(
            "cycle-close equivalence contract physical bytes disagree"
        )
    contract = load_json(path, description="cycle-close equivalence contract")
    claimed = contract.get("contract_sha256")
    body = {key: value for key, value in contract.items() if key != "contract_sha256"}
    if not _is_sha256(claimed) or semantic_sha256(body) != claimed:
        raise CycleCloseGlobalEquivalenceError(
            "cycle-close equivalence contract self-hash disagrees"
        )
    if (
        contract.get("schema") != CONTRACT_SCHEMA
        or contract.get("schema_version") != CONTRACT_SCHEMA_VERSION
        or contract.get("status") != NON_AUTHORIZING_STATUS
        or contract.get("training_authorized") is not False
        or contract.get("partitions") != ["validation"]
        or contract.get("excluded_partitions")
        != ["train", "controller_validation", "test"]
        or contract.get("source_scope")
        != "all_1707_distinct_exact_validation_sources_in_the_completed_cycle_open_v2_receipt"
        or contract.get("candidate_scope")
        != "all_undirected_nonbonded_endpoint_pair_and_order_coordinates_admitted_by_raw_bond_insert_on_at_least_one_complete_global_preserving_alias"
        or contract.get("equivalence_gate")
        != "zero_mismatches_zero_oracle_overflows_complete_declared_candidate_coverage"
        or contract.get("oracle_policy", {}).get("identity")
        != GLOBAL_MILP_ORACLE_STATUS
    ):
        raise CycleCloseGlobalEquivalenceError(
            "cycle-close equivalence contract scope or authority disagrees"
        )
    return contract


def validate_source_evidence(
    receipt: Mapping[str, Any],
    result: Mapping[str, Any],
    *,
    contract: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    receipt_value = dict(receipt)
    receipt_hash = receipt_value.pop("receipt_sha256", None)
    result_value = dict(result)
    result_hash = result_value.pop("result_sha256", None)
    expected = contract["exact_validation_input"]
    if (
        receipt_hash != semantic_sha256(receipt_value)
        or receipt_hash != expected["completed_cycle_open_receipt_sha256"]
        or result_hash != semantic_sha256(result_value)
        or result_hash != expected["completed_cycle_open_result_sha256"]
    ):
        raise CycleCloseGlobalEquivalenceError(
            "completed cycle-open source evidence self-hash disagrees"
        )
    receipt_value["receipt_sha256"] = receipt_hash
    result_value["result_sha256"] = result_hash
    source_records = receipt_value.get("source_records")
    if (
        receipt_value.get("schema") != SOURCE_RECEIPT_SCHEMA
        or receipt_value.get("schema_version") != SOURCE_RECEIPT_SCHEMA_VERSION
        or receipt_value.get("status") != SOURCE_STATUS
        or receipt_value.get("training_authorized") is not False
        or not isinstance(source_records, list)
        or len(source_records) != expected["expected_unique_exact_sources"]
        or len(
            {
                record.get("exact_source_state_sha256")
                for record in source_records
                if isinstance(record, Mapping)
            }
        )
        != len(source_records)
        or result_value.get("schema") != SOURCE_RESULT_SCHEMA
        or result_value.get("schema_version") != SOURCE_RESULT_SCHEMA_VERSION
        or result_value.get("status") != SOURCE_STATUS
        or result_value.get("training_authorized") is not False
        or result_value.get("equivalence_gate", {}).get("passed") is not True
        or result_value.get("unique_source_count") != len(source_records)
    ):
        raise CycleCloseGlobalEquivalenceError(
            "completed cycle-open source evidence is incomplete or malformed"
        )
    return receipt_value, result_value


def git_revision(*, repo_root: str | Path) -> dict[str, Any]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise CycleCloseGlobalEquivalenceError("cannot bind audit to Git") from error
    return {"commit": commit, "tree_dirty": bool(dirty)}


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
    }


def implementation_identity(*, repo_root: str | Path) -> dict[str, Any]:
    root = Path(repo_root)
    sources = (
        "scripts/audit_editing_cycle_close_global_equivalence.py",
        "src/compose_v4/experiments/editing_cycle_close_global_equivalence.py",
        "src/compose_v4/experiments/aromatic_cycle_close_semantics.py",
        "src/compose_v4/experiments/aromatic_cycle_open_global_oracle.py",
        "src/compose_v4/experiments/aromatic_cycle_open_semantics.py",
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


def build_plan(
    contract: Mapping[str, Any],
    *,
    contract_path: str | Path,
    source_receipt_path: str | Path,
    source_result_path: str | Path,
    packed_shard_path: str | Path,
    repo_root: str | Path,
) -> dict[str, Any]:
    revision = git_revision(repo_root=repo_root)
    if revision["tree_dirty"]:
        raise CycleCloseGlobalEquivalenceError("audit requires a clean committed tree")
    exact = contract["exact_validation_input"]
    observed = {
        "contract_file_sha256": file_sha256(contract_path),
        "completed_cycle_open_receipt_file_sha256": file_sha256(source_receipt_path),
        "completed_cycle_open_result_file_sha256": file_sha256(source_result_path),
        "packed_shard_content_sha256": file_sha256(packed_shard_path),
        "packed_manifest_sha256": file_sha256(
            Path(packed_shard_path).with_name("shard_0000.jsonl.manifest.json")
        ),
        "packed_provenance_overlay_sha256": file_sha256(
            Path(str(packed_shard_path) + ".provenance.json")
        ),
    }
    for key, value in observed.items():
        if key == "contract_file_sha256":
            continue
        if value != exact[key]:
            raise CycleCloseGlobalEquivalenceError(
                f"cycle-close audit physical input {key} disagrees"
            )
    runtime = runtime_identity()
    policy = {
        key: runtime[key]
        for key in ("python", "numpy", "scipy", "rdkit", "omp_num_threads")
    }
    if policy != contract["runtime_policy"]:
        raise CycleCloseGlobalEquivalenceError(
            f"cycle-close audit runtime disagrees: {policy}"
        )
    body = {
        "schema": PLAN_SCHEMA,
        "schema_version": PLAN_SCHEMA_VERSION,
        "status": NON_AUTHORIZING_STATUS,
        "training_authorized": False,
        "contract_sha256": contract["contract_sha256"],
        "input_paths": {
            "contract": str(Path(contract_path).resolve()),
            "source_receipt": str(Path(source_receipt_path).resolve()),
            "source_result": str(Path(source_result_path).resolve()),
            "packed_shard": str(Path(packed_shard_path).resolve()),
        },
        "input_file_sha256s": observed,
        "code_revision": revision,
        "implementation": implementation_identity(repo_root=repo_root),
        "runtime": runtime,
        "partitions": ["validation"],
        "excluded_partitions": ["train", "controller_validation", "test"],
        "seed_policy": "none_complete_deterministic_enumeration",
        "maximum_assignments": contract["oracle_policy"]["maximum_assignments"],
    }
    return {**body, "plan_sha256": semantic_sha256(body)}


def _exact_state_identity(state: MolecularGraph) -> tuple:
    return (
        tuple(int(value) for value in state.atom_types),
        tuple(int(value) for value in state.formal_charges),
        tuple(int(value) for value in state.implicit_h_counts),
        tuple(int(value) for value in state.bonds.reshape(-1)),
    )


def _candidate_actions(aliases: tuple[MolecularGraph, ...]) -> tuple[BondInsert, ...]:
    if not aliases:
        return ()
    first = aliases[0]
    real = tuple(int(index) for index in np.flatnonzero(is_element(first.atom_types)))
    return tuple(
        action
        for offset, left in enumerate(real)
        for right in real[offset + 1 :]
        if int(first.bonds[left, right]) == 0
        for order in (1, 2, 3)
        if any(
            is_valid_bond_insert(alias, action := BondInsert(left, right, order))
            for alias in aliases
        )
    )


def _oracle_resolution(
    state: MolecularGraph,
    action: BondInsert,
    aliases: tuple[MolecularGraph, ...],
) -> dict[str, Any]:
    source_key = canonical_state_key(state)
    products: list[MolecularGraph] = []
    invalid_alias_count = 0
    for alias in aliases:
        if not is_valid_bond_insert(alias, action):
            invalid_alias_count += 1
            continue
        products.append(apply_bond_insert(alias, action))
    groups: dict[str, list[MolecularGraph]] = {}
    for successor in products:
        groups.setdefault(canonical_state_key(successor), []).append(successor)
    product_keys = tuple(sorted(groups))
    if invalid_alias_count:
        rejection = AromaticCycleCloseRejectionCode.ALIAS_EXECUTION_DISAGREEMENT.value
    elif not product_keys:
        rejection = AromaticCycleCloseRejectionCode.EMPTY_CANONICAL_PRODUCT_GROUP.value
    elif len(product_keys) != 1:
        rejection = AromaticCycleCloseRejectionCode.AMBIGUOUS_CANONICAL_PRODUCT.value
    else:
        rejection = None
    admitted = rejection is None
    successor_identity = None
    if admitted:
        successor_identity = min(
            _exact_state_identity(successor) for successor in groups[product_keys[0]]
        )
    return {
        "admitted": admitted,
        "rejection_code": rejection,
        "source_key_sha256": semantic_sha256(source_key),
        "canonical_product_key_sha256s": [semantic_sha256(key) for key in product_keys],
        "successor_exact_identity_sha256": (
            None if successor_identity is None else semantic_sha256(successor_identity)
        ),
        "executed_product_count": len(products),
        "invalid_alias_count": invalid_alias_count,
    }


def _comparison_record(
    state: MolecularGraph,
    action: BondInsert,
    aliases: tuple[MolecularGraph, ...],
    *,
    context: object,
) -> dict[str, Any]:
    factored = resolve_component_factored_cycle_close(
        state,
        action,
        context=context,
    )
    oracle = _oracle_resolution(state, action, aliases)
    factored_product_hashes = [
        semantic_sha256(key) for key in factored.canonical_product_keys
    ]
    factored_successor_hash = (
        None
        if factored.successor is None
        else semantic_sha256(_exact_state_identity(factored.successor))
    )
    fields = {
        "admitted": factored.admitted == oracle["admitted"],
        "rejection_code": (
            None if factored.rejection_code is None else factored.rejection_code.value
        )
        == oracle["rejection_code"],
        "source_key_sha256": semantic_sha256(factored.source_key)
        == oracle["source_key_sha256"],
        "canonical_product_key_sha256s": factored_product_hashes
        == oracle["canonical_product_key_sha256s"],
        "successor_exact_identity_sha256": factored_successor_hash
        == oracle["successor_exact_identity_sha256"],
    }
    body = {
        "action": {"a": action.a, "b": action.b, "order": action.order},
        "semantic_equivalent": all(fields.values()),
        "field_equivalence": fields,
        "factored": {
            "admitted": factored.admitted,
            "rejection_code": (
                None
                if factored.rejection_code is None
                else factored.rejection_code.value
            ),
            "canonical_product_key_sha256s": factored_product_hashes,
            "successor_exact_identity_sha256": factored_successor_hash,
            "affected_component_count": factored.affected_component_count,
            "enumerated_affected_assignment_count": (
                factored.enumerated_affected_assignment_count
            ),
            "executed_product_count": factored.executed_product_count,
        },
        "oracle": oracle,
    }
    return {**body, "comparison_sha256": semantic_sha256(body)}


def audit_one_source(
    state: MolecularGraph,
    *,
    maximum_assignments: int,
) -> dict[str, Any]:
    source_digest = persistent_slot_state_sha256(state)
    try:
        global_result = enumerate_global_milp_kekule_aliases(
            state,
            maximum_assignments=maximum_assignments,
        )
    except AromaticCycleOpenAliasOverflow as error:
        body = {
            "schema": SOURCE_RECORD_SCHEMA,
            "schema_version": SOURCE_RECORD_SCHEMA_VERSION,
            "status": NON_AUTHORIZING_STATUS,
            "training_authorized": False,
            "exact_source_state_sha256": source_digest,
            "oracle_overflow": True,
            "observed_assignments_at_overflow": error.observed_structures,
            "counts": {"candidate_count": 0, "mismatch_count": 0},
            "comparison_sha256s": [],
            "nonpassing_comparisons": [],
        }
        return {**body, "source_record_sha256": semantic_sha256(body)}

    aliases = global_result.enumeration.aliases
    context = prepare_component_factored_cycle_close_context(state)
    comparisons = tuple(
        _comparison_record(state, action, aliases, context=context)
        for action in _candidate_actions(aliases)
    )
    outcomes = Counter(
        "equivalent" if item["semantic_equivalent"] else "mismatch"
        for item in comparisons
    )
    admitted = sum(bool(item["oracle"]["admitted"]) for item in comparisons)
    ambiguous = sum(
        item["oracle"]["rejection_code"]
        == AromaticCycleCloseRejectionCode.AMBIGUOUS_CANONICAL_PRODUCT.value
        for item in comparisons
    )
    body = {
        "schema": SOURCE_RECORD_SCHEMA,
        "schema_version": SOURCE_RECORD_SCHEMA_VERSION,
        "status": NON_AUTHORIZING_STATUS,
        "training_authorized": False,
        "exact_source_state_sha256": source_digest,
        "source_canonical_key_sha256": semantic_sha256(canonical_state_key(state)),
        "oracle_overflow": False,
        "oracle_diagnostics": {
            "semantic_aromatic_edge_count": (
                global_result.diagnostics.semantic_aromatic_edge_count
            ),
            "feasible_assignment_count": (
                global_result.diagnostics.feasible_assignment_count
            ),
            "preserving_alias_count": global_result.diagnostics.preserving_alias_count,
            "solver_call_count": global_result.diagnostics.solver_call_count,
        },
        "counts": {
            "candidate_count": len(comparisons),
            "equivalent_count": outcomes["equivalent"],
            "mismatch_count": outcomes["mismatch"],
            "admitted_count": admitted,
            "ambiguous_rejection_count": ambiguous,
            "other_rejection_count": len(comparisons) - admitted - ambiguous,
            "factored_executed_product_count": sum(
                int(item["factored"]["executed_product_count"]) for item in comparisons
            ),
            "oracle_executed_product_count": sum(
                int(item["oracle"]["executed_product_count"]) for item in comparisons
            ),
        },
        "comparison_sha256s": [item["comparison_sha256"] for item in comparisons],
        "nonpassing_comparisons": [
            item for item in comparisons if not item["semantic_equivalent"]
        ],
    }
    return {**body, "source_record_sha256": semantic_sha256(body)}


def audit_exact_validation_shard(
    contract: Mapping[str, Any],
    plan: Mapping[str, Any],
    source_receipt: Mapping[str, Any],
    *,
    packed_shard_path: str | Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    expected = {
        str(record["exact_source_state_sha256"])
        for record in source_receipt["source_records"]
    }
    states: dict[str, MolecularGraph] = {}
    rows = read_frozen_source_addressed_packed_shard(
        packed_shard_path,
        expected_shard_sha256=contract["exact_validation_input"][
            "packed_shard_content_sha256"
        ],
        expected_manifest_sha256=contract["exact_validation_input"][
            "packed_manifest_sha256"
        ],
        expected_overlay_sha256=contract["exact_validation_input"][
            "packed_provenance_overlay_sha256"
        ],
        verify_fraction=0.0,
    )
    physical_traces = 0
    physical_cycle_attach_rows = 0
    for addressed in rows:
        physical_traces += 1
        for progress_index, step in enumerate(addressed.trace.steps):
            if canonical_family(step.rule_name) != "cycle_attach":
                continue
            physical_cycle_attach_rows += 1
            state = addressed.path.state_at(progress_index)
            digest = persistent_slot_state_sha256(state)
            if digest in expected:
                states.setdefault(digest, state)
    if set(states) != expected:
        raise CycleCloseGlobalEquivalenceError(
            "packed shard does not reproduce the frozen validation source set"
        )

    source_records = [
        audit_one_source(
            states[digest],
            maximum_assignments=int(plan["maximum_assignments"]),
        )
        for digest in sorted(states)
    ]
    receipt_body = {
        "schema": RECEIPT_SCHEMA,
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "status": NON_AUTHORIZING_STATUS,
        "training_authorized": False,
        "plan_sha256": plan["plan_sha256"],
        "implementation_sha256": plan["implementation"]["implementation_sha256"],
        "source_receipt_sha256": source_receipt["receipt_sha256"],
        "counts": {
            "physical_traces_scanned": physical_traces,
            "physical_cycle_attach_rows_scanned": physical_cycle_attach_rows,
            "unique_exact_sources": len(states),
        },
        "source_records": source_records,
        "reconciliations": {
            "source_set": set(states) == expected,
            "source_record_count": len(source_records) == len(expected),
        },
    }
    receipt = {**receipt_body, "receipt_sha256": semantic_sha256(receipt_body)}

    totals: Counter[str] = Counter()
    overflow_sources = 0
    maximum_aliases = 0
    for record in source_records:
        totals.update({key: int(value) for key, value in record["counts"].items()})
        overflow_sources += int(record["oracle_overflow"])
        maximum_aliases = max(
            maximum_aliases,
            int(record.get("oracle_diagnostics", {}).get("preserving_alias_count", 0)),
        )
    passed = (
        bool(source_records)
        and totals["candidate_count"] > 0
        and totals["mismatch_count"] == 0
        and overflow_sources == 0
        and totals["equivalent_count"] == totals["candidate_count"]
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
        "oracle_overflow_source_count": overflow_sources,
        "maximum_preserving_alias_count": maximum_aliases,
        "source_record_sha256s": [
            record["source_record_sha256"] for record in source_records
        ],
        "nonpassing_source_records": [
            record
            for record in source_records
            if record["oracle_overflow"] or record["counts"]["mismatch_count"]
        ],
        "equivalence_gate": {
            "passed": passed,
            "required": (
                "zero mismatches, zero oracle overflows, and complete coverage of every "
                "declared cycle-close coordinate in the frozen validation source set"
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
            raise CycleCloseGlobalEquivalenceError(
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
    "CycleCloseGlobalEquivalenceError",
    "atomic_write_if_absent",
    "audit_exact_validation_shard",
    "audit_one_source",
    "build_plan",
    "canonical_json_bytes",
    "implementation_identity",
    "load_contract",
    "load_json",
    "pretty_json_bytes",
    "runtime_identity",
    "validate_source_evidence",
]
