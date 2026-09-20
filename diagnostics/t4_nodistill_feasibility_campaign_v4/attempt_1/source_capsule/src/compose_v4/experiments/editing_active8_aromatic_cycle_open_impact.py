"""Non-authorizing impact audit for semantic aromatic cycle opening.

The audit reads immutable, exact persistent-slot states from the frozen Active8
validation shards. It neither executes the historical teacher action
nor rebuilds any corpus object.  A row is in the impact envelope when its
accepted teacher is ``bond_delete`` and the selected edge is aromatic in the
resonance-invariant molecular view.  The prospective edge-anchored resolver is
then applied only to characterize how its successor differs from the stored
historical successor.

The map/reduce boundary is one physical gzip shard per receipt.  Receipts carry
the exact source bindings and the identity sets needed for an exact global
unique-trace and unique-source reduction.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, Callable

import networkx as nx
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

from compose_v4.chem.aromaticity import resonance_invariant_bond_classes
from compose_v4.chem.molecular_graph import (
    BOND_AROMATIC,
    BOND_DOUBLE,
    BOND_NULL,
    BOND_SINGLE,
    BOND_TRIPLE,
    IDX_TO_ELEMENT,
    MolecularGraph,
    is_element,
    molecular_graph_to_smiles,
)
from compose_v4.chem.persistent_state_identity import (
    persistent_slot_state_sha256,
)
from compose_v4.data.active8_trace_inventory import Active8TraceAdmission
from compose_v4.experiments.aromatic_cycle_open_semantics import (
    PROTOTYPE_STATUS,
    AromaticCycleOpenAliasOverflow,
    AromaticCycleOpenResolution,
    resolve_edge_anchored_cycle_open,
)
from compose_v4.rewrite.action_codec import canonical_family
from compose_v4.rewrite.kernel import canonical_state_key
from compose_v4.rewrite.operators import BondDelete


CONTRACT_SCHEMA = "compose.editing.active8_aromatic_cycle_open_impact_contract"
CONTRACT_SCHEMA_VERSION = 1
PLAN_SCHEMA = "compose.editing.active8_aromatic_cycle_open_impact_plan"
PLAN_SCHEMA_VERSION = 1
RECEIPT_SCHEMA = "compose.editing.active8_aromatic_cycle_open_shard_receipt"
RECEIPT_SCHEMA_VERSION = 1
RESULT_SCHEMA = "compose.editing.active8_aromatic_cycle_open_impact_result"
RESULT_SCHEMA_VERSION = 1
PINNED_AUDIT_CONTRACT_FILE_SHA256 = (
    "7961c6ab99a127ca2b5fd30e4c24107f160c9bb393ac668f38146c7ff0dcf97c"
)

NON_AUTHORIZING_STATUS = "FROZEN_ACTIVE8_AROMATIC_CYCLE_OPEN_IMPACT_AUDIT_NO_TRAINING_AUTHORITY"
OVERFLOW_REASON_CODE = "kekule_alias_enumeration_overflow"
_PARTITION_ORDER = {"train": 0, "validation": 1}
_RAW_BOND_LABELS = {
    BOND_NULL: "none",
    BOND_SINGLE: "single",
    BOND_DOUBLE: "double",
    BOND_TRIPLE: "triple",
    BOND_AROMATIC: "aromatic",
}

_CONTRACT_FIELDS = {
    "schema",
    "schema_version",
    "status",
    "training_authorized",
    "partitions",
    "excluded_partitions",
    "executor_rule",
    "model_family",
    "aromatic_edge_predicate",
    "resolver",
    "parent_active8_identity",
    "map_reduce_policy",
    "contract_sha256",
}
_PARENT_IDENTITY_FIELDS = {
    "inventory_manifest_path",
    "inventory_manifest_file_sha256",
    "inventory_sha256",
    "effective_source_corpus_cache_sha256",
    "support_contract_sha256",
    "unified_packed_manifest_sha256",
}
_PARENT_SHA_FIELDS = _PARENT_IDENTITY_FIELDS - {"inventory_manifest_path"}
_RESOLVER_FIELDS = {
    "prototype_status",
    "maximum_aliases",
    "overflow_policy",
    "aromatic_edge_policy",
    "canonical_product_policy",
    "exact_representative_policy",
}
_MAP_REDUCE_FIELDS = {
    "unit",
    "receipt_publication",
    "reducer_coverage",
    "final_test_excluded",
}


class AromaticCycleOpenImpactAuditError(RuntimeError):
    """The requested audit cannot establish its immutable evidence boundary."""


def canonical_json_bytes(value: object) -> bytes:
    """Return the deterministic JSON encoding used for all semantic hashes."""

    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise AromaticCycleOpenImpactAuditError(
            "audit value is not finite deterministic JSON"
        ) from error


def pretty_json_bytes(value: object) -> bytes:
    """Return stable human-readable artifact bytes."""

    try:
        return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    except (TypeError, ValueError) as error:
        raise AromaticCycleOpenImpactAuditError(
            "audit artifact is not finite deterministic JSON"
        ) from error


def semantic_sha256(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    try:
        with Path(path).open("rb") as handle:
            while block := handle.read(1 << 20):
                digest.update(block)
    except OSError as error:
        raise AromaticCycleOpenImpactAuditError(
            f"cannot hash required audit input: {path}"
        ) from error
    return digest.hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _require_exact_fields(
    value: object,
    expected: set[str],
    *,
    field: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != expected:
        observed = sorted(value) if isinstance(value, Mapping) else type(value).__name__
        raise AromaticCycleOpenImpactAuditError(
            f"{field} fields disagree: expected={sorted(expected)}, observed={observed}"
        )
    return value


def _validate_self_hash(payload: Mapping[str, Any], *, field: str) -> None:
    observed = payload.get(field)
    if not _is_sha256(observed):
        raise AromaticCycleOpenImpactAuditError(f"{field} must be a lowercase SHA-256")
    body = {key: value for key, value in payload.items() if key != field}
    if semantic_sha256(body) != observed:
        raise AromaticCycleOpenImpactAuditError(f"{field} self-hash disagrees")


def load_audit_contract(path: str | Path) -> dict[str, Any]:
    """Load the frozen impact contract and verify every pinned identity."""

    if file_sha256(path) != PINNED_AUDIT_CONTRACT_FILE_SHA256:
        raise AromaticCycleOpenImpactAuditError(
            "aromatic cycle-open impact contract physical bytes disagree"
        )
    try:
        payload = json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise AromaticCycleOpenImpactAuditError(
            f"cannot load aromatic cycle-open impact contract: {path}"
        ) from error
    contract = dict(_require_exact_fields(payload, _CONTRACT_FIELDS, field="audit contract"))
    if (
        contract["schema"] != CONTRACT_SCHEMA
        or contract["schema_version"] != CONTRACT_SCHEMA_VERSION
        or contract["status"] != NON_AUTHORIZING_STATUS
        or contract["training_authorized"] is not False
    ):
        raise AromaticCycleOpenImpactAuditError(
            "audit contract schema, status, or authority is invalid"
        )
    if contract["partitions"] != ["validation"]:
        raise AromaticCycleOpenImpactAuditError(
            "this pinned impact audit must remain explicitly validation-only"
        )
    excluded = contract["excluded_partitions"]
    if excluded != ["train", "controller_validation", "test"]:
        raise AromaticCycleOpenImpactAuditError(
            "validation-only audit must explicitly exclude train, "
            "controller-validation, and final-test partitions"
        )
    if contract["executor_rule"] != "bond_delete" or contract["model_family"] != "cycle_attach":
        raise AromaticCycleOpenImpactAuditError(
            "impact audit executor/family ontology is not the frozen Active8 mapping"
        )
    resolver = _require_exact_fields(
        contract["resolver"], _RESOLVER_FIELDS, field="contract.resolver"
    )
    if (
        resolver["prototype_status"] != PROTOTYPE_STATUS
        or type(resolver["maximum_aliases"]) is not int
        or resolver["maximum_aliases"] <= 0
        or resolver["overflow_policy"] != "typed_failure_no_truncation"
    ):
        raise AromaticCycleOpenImpactAuditError(
            "audit resolver policy is incompatible with the prototype"
        )
    parent = _require_exact_fields(
        contract["parent_active8_identity"],
        _PARENT_IDENTITY_FIELDS,
        field="contract.parent_active8_identity",
    )
    if (
        not isinstance(parent["inventory_manifest_path"], str)
        or not parent["inventory_manifest_path"].startswith("/artifacts/")
        or any(not _is_sha256(parent[field]) for field in _PARENT_SHA_FIELDS)
    ):
        raise AromaticCycleOpenImpactAuditError(
            "parent Active8 manifest path or SHA-256 identities are malformed"
        )
    policy = _require_exact_fields(
        contract["map_reduce_policy"],
        _MAP_REDUCE_FIELDS,
        field="contract.map_reduce_policy",
    )
    if (
        policy["unit"] != "one_complete_physical_packed_shard"
        or policy["receipt_publication"] != "immutable_no_overwrite"
        or policy["reducer_coverage"] != "exact_planned_task_set"
        or policy["final_test_excluded"] is not True
    ):
        raise AromaticCycleOpenImpactAuditError("map/reduce policy is not fail closed")
    _validate_self_hash(contract, field="contract_sha256")
    return contract


def validate_parent_admission(
    contract: Mapping[str, Any], admission: Active8TraceAdmission
) -> None:
    """Require the exact Active8 source frozen in the impact contract."""

    parent = contract["parent_active8_identity"]
    observed = {
        "inventory_manifest_path": str(admission.manifest_path),
        "inventory_manifest_file_sha256": admission.manifest_file_sha256,
        "inventory_sha256": admission.inventory_sha256,
        "effective_source_corpus_cache_sha256": (admission.effective_source_corpus_cache_sha256),
        "support_contract_sha256": admission.support_contract_sha256,
        "unified_packed_manifest_sha256": admission.unified_packed_manifest_sha256,
    }
    if observed != parent:
        raise AromaticCycleOpenImpactAuditError(
            f"Active8 admission identity disagrees with the contract: {observed}"
        )


def active8_partition_census(
    admission: Active8TraceAdmission,
) -> dict[str, dict[str, int]]:
    """Report exact available partition support from the pinned admission."""

    census: dict[str, Counter[str]] = {}
    for metadata in admission.shard_metadata_by_digest.values():
        partition = metadata.get("partition")
        counts = metadata.get("counts")
        family_rows = metadata.get("accepted_nonterminal_rows_by_family")
        if (
            not isinstance(partition, str)
            or not isinstance(counts, Mapping)
            or not isinstance(family_rows, Mapping)
        ):
            raise AromaticCycleOpenImpactAuditError(
                "Active8 shard metadata cannot establish its partition census"
            )
        row = census.setdefault(partition, Counter())
        row["source_shards"] += 1
        for field in (
            "traces",
            "accepted_traces",
            "excluded_traces",
            "accepted_nonterminal_rows",
            "accepted_terminal_rows",
        ):
            count = counts.get(field)
            if type(count) is not int or count < 0:
                raise AromaticCycleOpenImpactAuditError(
                    f"Active8 shard partition count is malformed: {field}"
                )
            row[field] += count
        cycle_attach_rows = family_rows.get("cycle_attach")
        if type(cycle_attach_rows) is not int or cycle_attach_rows < 0:
            raise AromaticCycleOpenImpactAuditError(
                "Active8 shard lacks a cycle_attach teacher-row census"
            )
        row["cycle_attach_teacher_rows"] += cycle_attach_rows
    return {partition: dict(sorted(counts.items())) for partition, counts in sorted(census.items())}


def _require_partition_admission(
    contract: Mapping[str, Any],
    admission: Active8TraceAdmission,
) -> dict[str, dict[str, int]]:
    census = active8_partition_census(admission)
    missing = [
        partition
        for partition in contract["partitions"]
        if partition not in census
        or census[partition].get("source_shards", 0) == 0
        or census[partition].get("accepted_traces", 0) == 0
    ]
    if missing:
        raise AromaticCycleOpenImpactAuditError(
            "BLOCKED: pinned Active8 admission lacks required admitted "
            f"partition(s) {missing}; exact_available_partition_census={census}"
        )
    if set(census) != set(contract["partitions"]):
        raise AromaticCycleOpenImpactAuditError(
            "pinned Active8 admission contains roles outside this explicitly "
            f"bounded diagnostic: exact_available_partition_census={census}"
        )
    return census


def implementation_identity(*, repo_root: str | Path | None = None) -> dict[str, Any]:
    """Bind the full local code surface used by this diagnostic."""

    root = Path(repo_root) if repo_root is not None else Path(__file__).resolve().parents[3]
    sources = (
        "modal_apps/audit_editing_active8_aromatic_cycle_open_impact.py",
        "scripts/audit_editing_active8_aromatic_cycle_open_impact.py",
        "src/compose_v4/experiments/editing_active8_aromatic_cycle_open_impact.py",
        "src/compose_v4/experiments/aromatic_cycle_open_semantics.py",
        "src/compose_v4/chem/aromaticity.py",
        "src/compose_v4/chem/molecular_graph.py",
        "src/compose_v4/chem/persistent_state_identity.py",
        "src/compose_v4/data/active8_trace_inventory.py",
        "src/compose_v4/data/packed_charge_policy_audit.py",
        "src/compose_v4/data/packed_trace_store.py",
        "src/compose_v4/rewrite/action_codec.py",
        "src/compose_v4/rewrite/kernel.py",
        "src/compose_v4/rewrite/operators.py",
    )
    hashes = {relative: file_sha256(root / relative) for relative in sources}
    return {
        "sources": hashes,
        "implementation_sha256": semantic_sha256(hashes),
    }


def _task_body(
    *,
    task_index: int,
    declared: object,
    digest: str,
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    counts = metadata.get("counts")
    family_rows = metadata.get("accepted_nonterminal_rows_by_family")
    if not isinstance(counts, Mapping) or not isinstance(family_rows, Mapping):
        raise AromaticCycleOpenImpactAuditError("Active8 shard metadata lacks exact counts")
    required_counts = (
        "traces",
        "accepted_traces",
        "accepted_nonterminal_rows",
        "accepted_terminal_rows",
    )
    if any(type(counts.get(field)) is not int for field in required_counts):
        raise AromaticCycleOpenImpactAuditError("Active8 shard metadata has malformed counts")
    cycle_attach_rows = family_rows.get("cycle_attach")
    if type(cycle_attach_rows) is not int:
        raise AromaticCycleOpenImpactAuditError("Active8 shard metadata lacks cycle_attach count")
    body = {
        "task_index": int(task_index),
        "manifest_layer": str(declared.manifest_layer),
        "layer": str(declared.envelope_layer),
        "partition": str(declared.partition),
        "relative_path": str(declared.relative_path),
        "packed_shard_name": str(declared.path.name),
        "packed_shard_content_sha256": digest,
        "packed_manifest_sha256": metadata["packed_manifest_sha256"],
        "packed_provenance_overlay_sha256": metadata["packed_provenance_overlay_sha256"],
        "expected_counts": {field: int(counts[field]) for field in required_counts},
        "expected_cycle_attach_teacher_rows": int(cycle_attach_rows),
    }
    body["task_sha256"] = semantic_sha256(body)
    body["receipt_filename"] = f"task_{task_index:05d}.{body['task_sha256']}.json"
    return body


def build_audit_plan(
    contract: Mapping[str, Any],
    admission: Active8TraceAdmission,
    declared_shards: Sequence[object],
    *,
    unified_manifest_file_sha256: str,
    inputs: Mapping[str, str],
    code_revision: Mapping[str, Any],
    implementation: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the deterministic one-physical-shard map plan."""

    validate_parent_admission(contract, admission)
    partition_census = _require_partition_admission(contract, admission)
    parent = contract["parent_active8_identity"]
    if unified_manifest_file_sha256 != parent["unified_packed_manifest_sha256"]:
        raise AromaticCycleOpenImpactAuditError(
            "unified packed manifest bytes disagree with the frozen contract"
        )
    if not isinstance(inputs, Mapping) or not inputs:
        raise AromaticCycleOpenImpactAuditError("plan inputs must record exact paths")
    if code_revision.get("tree_dirty") is not False or not code_revision.get("commit"):
        raise AromaticCycleOpenImpactAuditError(
            "authoritative audit plan requires a clean committed tree"
        )
    implementation_sha256 = implementation.get("implementation_sha256")
    if not _is_sha256(implementation_sha256):
        raise AromaticCycleOpenImpactAuditError("implementation identity lacks a full SHA-256")

    selected = tuple(
        shard for shard in declared_shards if shard.partition in contract["partitions"]
    )
    if not selected:
        raise AromaticCycleOpenImpactAuditError("audit plan selected no shards")
    if any(shard.partition == "test" for shard in selected):
        raise AromaticCycleOpenImpactAuditError("final-test shard entered audit plan")
    ordered = tuple(
        sorted(
            selected,
            key=lambda shard: (
                _PARTITION_ORDER[str(shard.partition)],
                str(shard.envelope_layer),
                str(shard.relative_path),
            ),
        )
    )
    tasks = []
    for task_index, shard in enumerate(ordered):
        digest = admission.expected_source_digest(
            packed_shard_name=shard.path.name,
            layer=shard.envelope_layer,
            partition=shard.partition,
        )
        metadata = admission.shard_metadata_by_digest[digest]
        tasks.append(
            _task_body(
                task_index=task_index,
                declared=shard,
                digest=digest,
                metadata=metadata,
            )
        )
    for partition in contract["partitions"]:
        admission.assert_partition_shards(
            partition,
            (
                (task["layer"], task["partition"], task["packed_shard_name"])
                for task in tasks
                if task["partition"] == partition
            ),
        )

    body = {
        "schema": PLAN_SCHEMA,
        "schema_version": PLAN_SCHEMA_VERSION,
        "status": NON_AUTHORIZING_STATUS,
        "training_authorized": False,
        "contract_sha256": contract["contract_sha256"],
        "parent_active8_identity": dict(parent),
        "inputs": dict(sorted(inputs.items())),
        "code_revision": dict(code_revision),
        "implementation": dict(implementation),
        "partitions": list(contract["partitions"]),
        "excluded_partitions": list(contract["excluded_partitions"]),
        "active8_partition_census": partition_census,
        "task_count": len(tasks),
        "tasks": tasks,
    }
    return {**body, "plan_sha256": semantic_sha256(body)}


def validate_audit_plan(
    payload: object,
    *,
    expected_contract_sha256: str | None = None,
) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise AromaticCycleOpenImpactAuditError("audit plan must be an object")
    plan = dict(payload)
    _validate_self_hash(plan, field="plan_sha256")
    if (
        plan.get("schema") != PLAN_SCHEMA
        or plan.get("schema_version") != PLAN_SCHEMA_VERSION
        or plan.get("status") != NON_AUTHORIZING_STATUS
        or plan.get("training_authorized") is not False
        or plan.get("partitions") != ["validation"]
        or plan.get("excluded_partitions") != ["train", "controller_validation", "test"]
    ):
        raise AromaticCycleOpenImpactAuditError(
            "audit plan schema, status, scope, or authority is invalid"
        )
    if expected_contract_sha256 is not None and plan.get("contract_sha256") != (
        expected_contract_sha256
    ):
        raise AromaticCycleOpenImpactAuditError("audit plan names another contract")
    partition_census = plan.get("active8_partition_census")
    if (
        not isinstance(partition_census, Mapping)
        or set(partition_census) != {"validation"}
        or not isinstance(partition_census["validation"], Mapping)
        or partition_census["validation"].get("source_shards", 0) <= 0
        or partition_census["validation"].get("accepted_traces", 0) <= 0
    ):
        raise AromaticCycleOpenImpactAuditError(
            "audit plan does not report the exact admitted validation census"
        )
    tasks = plan.get("tasks")
    if not isinstance(tasks, list) or plan.get("task_count") != len(tasks):
        raise AromaticCycleOpenImpactAuditError("audit plan task census is invalid")
    for index, task in enumerate(tasks):
        if not isinstance(task, Mapping) or task.get("task_index") != index:
            raise AromaticCycleOpenImpactAuditError("audit plan task order is invalid")
        if task.get("partition") not in plan["partitions"]:
            raise AromaticCycleOpenImpactAuditError("audit plan contains an unauthorized partition")
        task_body = {
            key: value
            for key, value in task.items()
            if key not in {"task_sha256", "receipt_filename"}
        }
        if semantic_sha256(task_body) != task.get("task_sha256"):
            raise AromaticCycleOpenImpactAuditError("audit task self-hash disagrees")
        expected_filename = f"task_{index:05d}.{task['task_sha256']}.json"
        if task.get("receipt_filename") != expected_filename:
            raise AromaticCycleOpenImpactAuditError("audit task receipt filename disagrees")
    return plan


def _exact_trace_identity(address: object) -> str:
    return semantic_sha256(
        {
            "packed_shard_content_sha256": address.packed_shard_content_sha256,
            "entry_index": address.entry_index,
            "trace_id": address.trace_id,
        }
    )


def _increment_nested(counter: Counter[str], key: str) -> None:
    counter[str(key)] += 1


Resolver = Callable[..., AromaticCycleOpenResolution]


def _state_graph(state: MolecularGraph) -> nx.Graph:
    real = tuple(int(slot) for slot in np.flatnonzero(is_element(state.atom_types)))
    graph = nx.Graph()
    graph.add_nodes_from(real)
    graph.add_edges_from(
        (left, right)
        for offset, left in enumerate(real)
        for right in real[offset + 1 :]
        if int(state.bonds[left, right]) != BOND_NULL
    )
    return graph


def _semantic_aromatic_cycle_edges(
    state: MolecularGraph,
    perceived: np.ndarray,
) -> tuple[tuple[tuple[int, int], ...], tuple[tuple[int, int], ...]]:
    """Return perceived aromatic cycle edges and any anomalous aromatic bridges."""

    graph = _state_graph(state)
    bridges = {tuple(sorted((int(left), int(right)))) for left, right in nx.bridges(graph)}
    aromatic_edges = tuple(
        (left, right)
        for left in sorted(int(node) for node in graph.nodes())
        for right in sorted(int(node) for node in graph.nodes())
        if left < right and int(perceived[left, right]) == BOND_AROMATIC
    )
    cycle_edges = tuple(edge for edge in aromatic_edges if edge not in bridges)
    noncycle_edges = tuple(edge for edge in aromatic_edges if edge in bridges)
    return cycle_edges, noncycle_edges


def _source_topology_descriptor(
    state: MolecularGraph,
    *,
    semantic_aromatic_cycle_edges: Sequence[tuple[int, int]],
) -> dict[str, Any]:
    """Describe topology without treating an SSSR basis as graph cycle rank."""

    graph = _state_graph(state)
    aromatic_graph = nx.Graph()
    aromatic_graph.add_edges_from(semantic_aromatic_cycle_edges)
    aromatic_components = tuple(
        tuple(sorted(int(node) for node in component))
        for component in sorted(
            nx.connected_components(aromatic_graph),
            key=lambda component: tuple(sorted(int(node) for node in component)),
        )
    )
    aromatic_system_sizes = tuple(len(component) for component in aromatic_components)
    aromatic_system_cycle_ranks = tuple(
        int(
            aromatic_graph.subgraph(component).number_of_edges()
            - aromatic_graph.subgraph(component).number_of_nodes()
            + 1
        )
        for component in aromatic_components
    )

    smiles = molecular_graph_to_smiles(state)
    molecule = Chem.MolFromSmiles(smiles) if smiles is not None else None
    if molecule is None:
        raise AromaticCycleOpenImpactAuditError(
            "accepted exact source cannot be serialized for ring taxonomy"
        )
    atom_rings = tuple(
        frozenset(int(atom) for atom in ring) for ring in molecule.GetRingInfo().AtomRings()
    )
    bond_rings = tuple(
        tuple(int(bond) for bond in ring) for ring in molecule.GetRingInfo().BondRings()
    )
    if len(atom_rings) != len(bond_rings):
        raise AromaticCycleOpenImpactAuditError(
            "RDKit atom and bond SSSR bases disagree for an accepted source"
        )
    aromatic_ring_sizes = []
    aromatic_atom_rings = []
    for atom_ring, bond_ring in zip(atom_rings, bond_rings, strict=True):
        bonds = [molecule.GetBondWithIdx(index) for index in bond_ring]
        if bonds and all(bond.GetIsAromatic() for bond in bonds):
            aromatic_atom_rings.append(atom_ring)
            aromatic_ring_sizes.append(len(atom_ring))

    fused = any(
        len(left & right) >= 2
        for index, left in enumerate(aromatic_atom_rings)
        for right in aromatic_atom_rings[index + 1 :]
    )
    bridged = bool(rdMolDescriptors.CalcNumBridgeheadAtoms(molecule) > 0)
    isolated = any(rank == 1 for rank in aromatic_system_cycle_ranks)
    heteroaromatic = any(
        IDX_TO_ELEMENT[int(state.atom_types[slot])] != "C"
        for component in aromatic_components
        for slot in component
    )
    cycle_rank = int(
        graph.number_of_edges() - graph.number_of_nodes() + nx.number_connected_components(graph)
    )
    return {
        "active_atom_count": int(state.n_real_atoms),
        "graph_cycle_rank": cycle_rank,
        "heteroaromatic": heteroaromatic,
        "aromatic_system_count": len(aromatic_components),
        "aromatic_system_atom_sizes": sorted(aromatic_system_sizes),
        "aromatic_system_cycle_ranks": sorted(aromatic_system_cycle_ranks),
        "aromatic_sssr_ring_sizes": sorted(aromatic_ring_sizes),
        "topology_flags": {
            "isolated_aromatic_system": isolated,
            "fused_aromatic_rings": fused,
            "bridged_ring_source": bridged,
        },
        "ring_size_semantics": "RDKit_symmetrized_SSSR_aromatic_rings",
    }


def _cached_resolution(
    cache: dict[tuple[str, int, int], tuple[str, AromaticCycleOpenResolution | None]],
    *,
    source: MolecularGraph,
    source_digest: str,
    edge: tuple[int, int],
    maximum_aliases: int,
    resolver: Resolver,
) -> tuple[str, AromaticCycleOpenResolution | None]:
    key = (source_digest, int(edge[0]), int(edge[1]))
    if key not in cache:
        try:
            cache[key] = (
                "resolved",
                resolver(
                    source,
                    BondDelete(*edge),
                    maximum_aliases=maximum_aliases,
                ),
            )
        except AromaticCycleOpenAliasOverflow:
            cache[key] = (OVERFLOW_REASON_CODE, None)
    return cache[key]


def _source_reachability_record(
    source: MolecularGraph,
    *,
    source_digest: str,
    maximum_aliases: int,
    resolver: Resolver,
    resolution_cache: dict[tuple[str, int, int], tuple[str, AromaticCycleOpenResolution | None]],
) -> dict[str, Any]:
    perceived = resonance_invariant_bond_classes(source)
    cycle_edges, noncycle_edges = _semantic_aromatic_cycle_edges(source, perceived)
    edge_records = []
    outcomes: Counter[str] = Counter()
    canonical_successors: set[str] = set()
    for edge in cycle_edges:
        resolution_status, resolution = _cached_resolution(
            resolution_cache,
            source=source,
            source_digest=source_digest,
            edge=edge,
            maximum_aliases=maximum_aliases,
            resolver=resolver,
        )
        edge_record: dict[str, Any] = {
            "edge": [int(edge[0]), int(edge[1])],
        }
        if resolution_status == OVERFLOW_REASON_CODE:
            outcome = OVERFLOW_REASON_CODE
            edge_record["outcome"] = outcome
        else:
            if resolution is None or not resolution.semantic_aromatic_edge:
                raise AromaticCycleOpenImpactAuditError(
                    "semantic aromatic reachability edge disagrees with resolver"
                )
            outcome = (
                "admitted"
                if resolution.admitted
                else (
                    resolution.rejection_code.value if resolution.rejection_code is not None else ""
                )
            )
            if not outcome:
                raise AromaticCycleOpenImpactAuditError(
                    "source reachability resolver rejected without a typed reason"
                )
            edge_record.update(
                {
                    "outcome": outcome,
                    "enumerated_alias_count": resolution.enumerated_alias_count,
                    "forced_single_alias_count": resolution.forced_single_alias_count,
                    "executed_product_count": resolution.executed_product_count,
                }
            )
            if resolution.admitted:
                if resolution.successor is None:
                    raise AromaticCycleOpenImpactAuditError(
                        "admitted source reachability edge has no successor"
                    )
                canonical_key = canonical_state_key(resolution.successor)
                canonical_successors.add(canonical_key)
                edge_record.update(
                    {
                        "successor_exact_state_sha256": (
                            persistent_slot_state_sha256(resolution.successor)
                        ),
                        "successor_canonical_key_sha256": semantic_sha256(canonical_key),
                    }
                )
        outcomes[outcome] += 1
        edge_records.append(edge_record)

    admitted_edges = outcomes["admitted"]
    descriptor = _source_topology_descriptor(
        source,
        semantic_aromatic_cycle_edges=cycle_edges,
    )
    return {
        "exact_source_state_sha256": source_digest,
        "semantic_aromatic_cycle_edge_count": len(cycle_edges),
        "semantic_aromatic_noncycle_edge_count": len(noncycle_edges),
        "admitted_semantic_aromatic_cycle_edge_count": admitted_edges,
        "admitted_canonical_successor_count": len(canonical_successors),
        "zero_admitted_semantic_aromatic_cycle_edges": admitted_edges == 0,
        "has_semantic_aromatic_cycle_edge": bool(cycle_edges),
        "resolver_outcomes": dict(sorted(outcomes.items())),
        "edge_records": edge_records,
        "topology": descriptor,
    }


def audit_one_shard(
    task: Mapping[str, Any],
    addressed_traces: Iterable[object],
    admission: Active8TraceAdmission,
    *,
    contract: Mapping[str, Any],
    plan_sha256: str,
    implementation_sha256: str,
    resolver: Resolver = resolve_edge_anchored_cycle_open,
) -> dict[str, Any]:
    """Audit exactly one complete physical shard and return a sealed receipt."""

    if task.get("partition") not in contract["partitions"] or task.get("partition") == "test":
        raise AromaticCycleOpenImpactAuditError(
            "shard task is outside the authorized audit partitions"
        )
    if not _is_sha256(plan_sha256) or not _is_sha256(implementation_sha256):
        raise AromaticCycleOpenImpactAuditError(
            "shard receipt requires exact plan and implementation identities"
        )
    maximum_aliases = int(contract["resolver"]["maximum_aliases"])
    counts: Counter[str] = Counter()
    family_rows: Counter[str] = Counter()
    raw_orders: Counter[str] = Counter()
    resolver_outcomes: Counter[str] = Counter()
    exact_comparisons: Counter[str] = Counter()
    canonical_comparisons: Counter[str] = Counter()
    affected_trace_ids: set[str] = set()
    affected_source_digests: set[str] = set()
    cycle_open_teacher_sources: dict[str, MolecularGraph] = {}
    resolution_cache: dict[
        tuple[str, int, int], tuple[str, AromaticCycleOpenResolution | None]
    ] = {}
    observed_digest: str | None = None

    for addressed in addressed_traces:
        address = addressed.address
        counts["physical_traces_scanned"] += 1
        if (
            address.packed_shard_name != task["packed_shard_name"]
            or address.layer != task["layer"]
            or address.partition != task["partition"]
            or address.packed_shard_content_sha256 != task["packed_shard_content_sha256"]
        ):
            raise AromaticCycleOpenImpactAuditError(
                "decoded packed address disagrees with its exact shard task"
            )
        if observed_digest is None:
            observed_digest = address.packed_shard_content_sha256
        elif observed_digest != address.packed_shard_content_sha256:
            raise AromaticCycleOpenImpactAuditError(
                "one shard stream yielded multiple physical digests"
            )
        if not admission.is_accepted(address):
            counts["excluded_traces_scanned"] += 1
            continue
        counts["accepted_traces_scanned"] += 1
        if len(addressed.trace.steps) != address.path_length:
            raise AromaticCycleOpenImpactAuditError(
                "accepted packed trace path length disagrees with its address"
            )
        counts["accepted_nonterminal_teacher_rows"] += address.path_length

        trace_is_affected = False
        for progress_index, step in enumerate(addressed.trace.steps):
            family = canonical_family(step.rule_name)
            family_rows[family] += 1
            if family != contract["model_family"]:
                continue
            counts["accepted_cycle_attach_teacher_rows"] += 1
            if step.rule_name != contract["executor_rule"] or not isinstance(
                step.action, BondDelete
            ):
                raise AromaticCycleOpenImpactAuditError(
                    "cycle_attach teacher is not the frozen BondDelete executor action"
                )

            source = addressed.path.state_at(progress_index)
            source_digest = persistent_slot_state_sha256(source)
            cycle_open_teacher_sources.setdefault(source_digest, source)
            left, right = sorted((int(step.action.a), int(step.action.b)))
            perceived = resonance_invariant_bond_classes(source)
            if int(perceived[left, right]) != BOND_AROMATIC:
                counts["nonaromatic_cycle_attach_teacher_rows"] += 1
                continue

            counts["perceived_aromatic_bond_delete_teacher_rows"] += 1
            trace_is_affected = True
            affected_source_digests.add(source_digest)
            raw_order = int(source.bonds[left, right])
            try:
                raw_label = _RAW_BOND_LABELS[raw_order]
            except KeyError as error:
                raise AromaticCycleOpenImpactAuditError(
                    f"affected teacher has unknown raw bond class {raw_order}"
                ) from error
            _increment_nested(raw_orders, raw_label)

            resolution_status, resolution = _cached_resolution(
                resolution_cache,
                source=source,
                source_digest=source_digest,
                edge=(left, right),
                maximum_aliases=maximum_aliases,
                resolver=resolver,
            )
            if resolution_status == OVERFLOW_REASON_CODE:
                _increment_nested(resolver_outcomes, OVERFLOW_REASON_CODE)
                continue
            if resolution is None:
                raise AromaticCycleOpenImpactAuditError(
                    "resolved affected teacher has no resolver result"
                )
            if resolution.prototype_status != contract["resolver"]["prototype_status"]:
                raise AromaticCycleOpenImpactAuditError(
                    "resolver returned another semantic prototype identity"
                )
            if not resolution.semantic_aromatic_edge:
                raise AromaticCycleOpenImpactAuditError(
                    "perceived aromatic teacher was not aromatic to the resolver"
                )
            if not resolution.admitted:
                if resolution.rejection_code is None:
                    raise AromaticCycleOpenImpactAuditError(
                        "resolver rejected a teacher without a typed reason"
                    )
                _increment_nested(resolver_outcomes, resolution.rejection_code.value)
                continue
            if resolution.successor is None:
                raise AromaticCycleOpenImpactAuditError(
                    "admitted resolver result has no exact successor"
                )
            _increment_nested(resolver_outcomes, "admitted")
            historical = addressed.path.state_at(progress_index + 1)
            exact_key = (
                "same"
                if persistent_slot_state_sha256(resolution.successor)
                == persistent_slot_state_sha256(historical)
                else "different"
            )
            canonical_key = (
                "same"
                if canonical_state_key(resolution.successor) == canonical_state_key(historical)
                else "different"
            )
            _increment_nested(exact_comparisons, exact_key)
            _increment_nested(canonical_comparisons, canonical_key)

        if trace_is_affected:
            counts["traces_with_perceived_aromatic_bond_delete_teacher_in_shard"] += 1
            affected_trace_ids.add(_exact_trace_identity(address))

    source_reachability_records = [
        _source_reachability_record(
            cycle_open_teacher_sources[source_digest],
            source_digest=source_digest,
            maximum_aliases=maximum_aliases,
            resolver=resolver,
            resolution_cache=resolution_cache,
        )
        for source_digest in sorted(cycle_open_teacher_sources)
    ]
    counts["unique_cycle_open_teacher_source_states_in_shard"] = len(source_reachability_records)

    admission.assert_complete_source_shard(
        packed_shard_name=str(task["packed_shard_name"]),
        layer=str(task["layer"]),
        partition=str(task["partition"]),
        observed_digest=observed_digest,
        observed_entries=counts["physical_traces_scanned"],
    )
    expected = task["expected_counts"]
    reconciliations = {
        "physical_traces": counts["physical_traces_scanned"] == expected["traces"],
        "accepted_plus_excluded_traces": (
            counts["accepted_traces_scanned"] + counts["excluded_traces_scanned"]
            == counts["physical_traces_scanned"]
        ),
        "accepted_traces": counts["accepted_traces_scanned"] == expected["accepted_traces"],
        "accepted_terminal_rows": (
            expected["accepted_terminal_rows"] == counts["accepted_traces_scanned"]
        ),
        "accepted_nonterminal_rows": counts["accepted_nonterminal_teacher_rows"]
        == expected["accepted_nonterminal_rows"],
        "family_rows": (sum(family_rows.values()) == counts["accepted_nonterminal_teacher_rows"]),
        "cycle_attach_rows": counts["accepted_cycle_attach_teacher_rows"]
        == task["expected_cycle_attach_teacher_rows"],
        "aromatic_plus_nonaromatic_cycle_attach": (
            counts["perceived_aromatic_bond_delete_teacher_rows"]
            + counts["nonaromatic_cycle_attach_teacher_rows"]
            == counts["accepted_cycle_attach_teacher_rows"]
        ),
        "cycle_open_source_reachability_records": (
            counts["unique_cycle_open_teacher_source_states_in_shard"]
            == len(cycle_open_teacher_sources)
        ),
        "affected_trace_identity_records": (
            counts["traces_with_perceived_aromatic_bond_delete_teacher_in_shard"]
            == len(affected_trace_ids)
        ),
    }
    if not all(reconciliations.values()):
        raise AromaticCycleOpenImpactAuditError(
            f"shard audit denominators do not reconcile: {reconciliations}"
        )
    if sum(resolver_outcomes.values()) != counts["perceived_aromatic_bond_delete_teacher_rows"]:
        raise AromaticCycleOpenImpactAuditError(
            "resolver outcomes do not partition affected aromatic teachers"
        )
    admitted = resolver_outcomes["admitted"]
    if (
        sum(exact_comparisons.values()) != admitted
        or sum(canonical_comparisons.values()) != admitted
    ):
        raise AromaticCycleOpenImpactAuditError(
            "successor comparisons do not cover every admitted resolution"
        )

    body = {
        "schema": RECEIPT_SCHEMA,
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "status": NON_AUTHORIZING_STATUS,
        "training_authorized": False,
        "contract_sha256": contract["contract_sha256"],
        "plan_sha256": plan_sha256,
        "implementation_sha256": implementation_sha256,
        "task": dict(task),
        "counts": dict(sorted(counts.items())),
        "accepted_teacher_rows_by_family": dict(sorted(family_rows.items())),
        "raw_deleted_bond_class": dict(sorted(raw_orders.items())),
        "resolver_outcomes": dict(sorted(resolver_outcomes.items())),
        "prototype_vs_historical_exact_digest": dict(sorted(exact_comparisons.items())),
        "prototype_vs_historical_canonical_key": dict(sorted(canonical_comparisons.items())),
        "affected_trace_identity_sha256s": sorted(affected_trace_ids),
        "affected_exact_source_state_sha256s": sorted(affected_source_digests),
        "cycle_open_source_reachability": source_reachability_records,
        "reconciliations": reconciliations,
    }
    return {**body, "receipt_sha256": semantic_sha256(body)}


def validate_shard_receipt(
    value: object,
    *,
    expected_contract_sha256: str,
    expected_plan_sha256: str,
    expected_task: Mapping[str, Any],
    expected_implementation_sha256: str,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise AromaticCycleOpenImpactAuditError("shard receipt must be an object")
    receipt = dict(value)
    _validate_self_hash(receipt, field="receipt_sha256")
    if (
        receipt.get("schema") != RECEIPT_SCHEMA
        or receipt.get("schema_version") != RECEIPT_SCHEMA_VERSION
        or receipt.get("status") != NON_AUTHORIZING_STATUS
        or receipt.get("training_authorized") is not False
        or receipt.get("contract_sha256") != expected_contract_sha256
        or receipt.get("plan_sha256") != expected_plan_sha256
        or receipt.get("implementation_sha256") != expected_implementation_sha256
        or receipt.get("task") != dict(expected_task)
    ):
        raise AromaticCycleOpenImpactAuditError(
            "shard receipt schema, identity, task, or authority disagrees"
        )
    reconciliations = receipt.get("reconciliations")
    if (
        not isinstance(reconciliations, Mapping)
        or not reconciliations
        or not all(value is True for value in reconciliations.values())
    ):
        raise AromaticCycleOpenImpactAuditError(
            "shard receipt has a failed denominator reconciliation"
        )
    source_records = receipt.get("cycle_open_source_reachability")
    if not isinstance(source_records, list):
        raise AromaticCycleOpenImpactAuditError("shard receipt lacks source reachability records")
    source_digests = []
    for record in source_records:
        if not isinstance(record, Mapping):
            raise AromaticCycleOpenImpactAuditError("source reachability record must be an object")
        source_digest = record.get("exact_source_state_sha256")
        if not _is_sha256(source_digest):
            raise AromaticCycleOpenImpactAuditError(
                "source reachability record lacks an exact-state SHA-256"
            )
        edge_records = record.get("edge_records")
        outcomes = record.get("resolver_outcomes")
        if not isinstance(edge_records, list) or not isinstance(outcomes, Mapping):
            raise AromaticCycleOpenImpactAuditError(
                "source reachability edge evidence is malformed"
            )
        if record.get("semantic_aromatic_cycle_edge_count") != len(edge_records):
            raise AromaticCycleOpenImpactAuditError(
                "source reachability edge census does not reconcile"
            )
        outcome_count = 0
        for count in outcomes.values():
            if type(count) is not int or count < 0:
                raise AromaticCycleOpenImpactAuditError(
                    "source reachability outcome count is malformed"
                )
            outcome_count += count
        if outcome_count != len(edge_records):
            raise AromaticCycleOpenImpactAuditError(
                "source reachability outcomes do not partition its edges"
            )
        admitted = int(outcomes.get("admitted", 0))
        if record.get("admitted_semantic_aromatic_cycle_edge_count") != admitted:
            raise AromaticCycleOpenImpactAuditError(
                "source reachability admitted-edge count disagrees"
            )
        if record.get("zero_admitted_semantic_aromatic_cycle_edges") != (admitted == 0):
            raise AromaticCycleOpenImpactAuditError(
                "source reachability zero-admitted flag disagrees"
            )
        source_digests.append(source_digest)
    if source_digests != sorted(set(source_digests)):
        raise AromaticCycleOpenImpactAuditError(
            "source reachability records are not uniquely digest-sorted"
        )
    return receipt


def _sum_mapping(target: Counter[str], value: object, *, field: str) -> None:
    if not isinstance(value, Mapping):
        raise AromaticCycleOpenImpactAuditError(f"{field} must be an object")
    for key, count in value.items():
        if type(count) is not int or count < 0:
            raise AromaticCycleOpenImpactAuditError(f"{field}.{key} must be a nonnegative integer")
        target[str(key)] += count


def _source_reachability_summary(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    semantic_edge_histogram: Counter[str] = Counter()
    admitted_edge_histogram: Counter[str] = Counter()
    outcomes: Counter[str] = Counter()
    topology_flags: Counter[str] = Counter()
    aromatic_system_sizes: Counter[str] = Counter()
    aromatic_system_cycle_ranks: Counter[str] = Counter()
    aromatic_ring_sizes: Counter[str] = Counter()
    counts: Counter[str] = Counter()

    for record in records:
        semantic_edges = int(record["semantic_aromatic_cycle_edge_count"])
        admitted_edges = int(record["admitted_semantic_aromatic_cycle_edge_count"])
        counts["unique_cycle_open_teacher_source_states"] += 1
        counts["sources_with_semantic_aromatic_cycle_edges"] += int(semantic_edges > 0)
        counts["sources_without_semantic_aromatic_cycle_edges"] += int(semantic_edges == 0)
        counts["sources_with_at_least_one_admitted_edge"] += int(admitted_edges > 0)
        counts["zero_admitted_sources"] += int(admitted_edges == 0)
        counts["zero_admitted_sources_with_semantic_aromatic_edges"] += int(
            semantic_edges > 0 and admitted_edges == 0
        )
        counts["semantic_aromatic_cycle_edges"] += semantic_edges
        counts["semantic_aromatic_noncycle_edges"] += int(
            record["semantic_aromatic_noncycle_edge_count"]
        )
        counts["admitted_semantic_aromatic_cycle_edges"] += admitted_edges
        counts["admitted_canonical_successors_summed_over_sources"] += int(
            record["admitted_canonical_successor_count"]
        )
        semantic_edge_histogram[str(semantic_edges)] += 1
        admitted_edge_histogram[str(admitted_edges)] += 1
        _sum_mapping(
            outcomes,
            record["resolver_outcomes"],
            field="source_reachability.resolver_outcomes",
        )
        topology = record["topology"]
        if topology["heteroaromatic"]:
            counts["heteroaromatic_sources"] += 1
        else:
            counts["nonheteroaromatic_sources"] += 1
        for label, present in topology["topology_flags"].items():
            if present:
                topology_flags[str(label)] += 1
        for size in topology["aromatic_system_atom_sizes"]:
            aromatic_system_sizes[str(int(size))] += 1
        for rank in topology["aromatic_system_cycle_ranks"]:
            aromatic_system_cycle_ranks[str(int(rank))] += 1
        for size in topology["aromatic_sssr_ring_sizes"]:
            aromatic_ring_sizes[str(int(size))] += 1

    return {
        "counts": dict(sorted(counts.items())),
        "semantic_aromatic_cycle_edge_count_histogram": dict(
            sorted(semantic_edge_histogram.items(), key=lambda item: int(item[0]))
        ),
        "admitted_edge_count_histogram": dict(
            sorted(admitted_edge_histogram.items(), key=lambda item: int(item[0]))
        ),
        "resolver_outcomes": dict(sorted(outcomes.items())),
        "overlapping_topology_flag_counts": dict(sorted(topology_flags.items())),
        "aromatic_system_atom_size_counts": dict(
            sorted(aromatic_system_sizes.items(), key=lambda item: int(item[0]))
        ),
        "aromatic_system_cycle_rank_counts": dict(
            sorted(aromatic_system_cycle_ranks.items(), key=lambda item: int(item[0]))
        ),
        "aromatic_sssr_ring_size_counts": dict(
            sorted(aromatic_ring_sizes.items(), key=lambda item: int(item[0]))
        ),
    }


def reduce_shard_receipts(
    plan: Mapping[str, Any],
    receipts: Sequence[Mapping[str, Any]],
    *,
    contract: Mapping[str, Any],
    implementation_sha256: str,
) -> dict[str, Any]:
    """Reduce the exact planned receipt set into one non-authorizing result."""

    validated_plan = validate_audit_plan(plan, expected_contract_sha256=contract["contract_sha256"])
    if validated_plan.get("parent_active8_identity") != dict(contract["parent_active8_identity"]):
        raise AromaticCycleOpenImpactAuditError("reducer plan names another Active8 parent")
    tasks = validated_plan["tasks"]
    if len(receipts) != len(tasks):
        raise AromaticCycleOpenImpactAuditError("reducer receipt count differs from the exact plan")

    totals: Counter[str] = Counter()
    family_rows: Counter[str] = Counter()
    raw_orders: Counter[str] = Counter()
    outcomes: Counter[str] = Counter()
    exact_comparisons: Counter[str] = Counter()
    canonical_comparisons: Counter[str] = Counter()
    trace_ids: set[str] = set()
    source_ids: set[str] = set()
    reachability_by_source: dict[str, Mapping[str, Any]] = {}
    strata: dict[tuple[str, str], Counter[str]] = {}
    stratum_reachability: dict[tuple[str, str], dict[str, Mapping[str, Any]]] = {}
    receipt_summaries = []

    for task, raw_receipt in zip(tasks, receipts, strict=True):
        receipt = validate_shard_receipt(
            raw_receipt,
            expected_contract_sha256=contract["contract_sha256"],
            expected_plan_sha256=validated_plan["plan_sha256"],
            expected_task=task,
            expected_implementation_sha256=implementation_sha256,
        )
        _sum_mapping(totals, receipt["counts"], field="receipt.counts")
        _sum_mapping(
            family_rows,
            receipt["accepted_teacher_rows_by_family"],
            field="receipt.accepted_teacher_rows_by_family",
        )
        _sum_mapping(
            raw_orders,
            receipt["raw_deleted_bond_class"],
            field="receipt.raw_deleted_bond_class",
        )
        _sum_mapping(
            outcomes,
            receipt["resolver_outcomes"],
            field="receipt.resolver_outcomes",
        )
        _sum_mapping(
            exact_comparisons,
            receipt["prototype_vs_historical_exact_digest"],
            field="receipt.prototype_vs_historical_exact_digest",
        )
        _sum_mapping(
            canonical_comparisons,
            receipt["prototype_vs_historical_canonical_key"],
            field="receipt.prototype_vs_historical_canonical_key",
        )
        trace_ids.update(receipt["affected_trace_identity_sha256s"])
        source_ids.update(receipt["affected_exact_source_state_sha256s"])
        stratum = strata.setdefault((task["partition"], task["layer"]), Counter())
        _sum_mapping(stratum, receipt["counts"], field="receipt.counts")
        stratum_sources = stratum_reachability.setdefault((task["partition"], task["layer"]), {})
        for record in receipt["cycle_open_source_reachability"]:
            source_digest = record["exact_source_state_sha256"]
            prior = reachability_by_source.get(source_digest)
            if prior is not None and prior != record:
                raise AromaticCycleOpenImpactAuditError(
                    "one exact source has conflicting reachability records"
                )
            reachability_by_source[source_digest] = record
            stratum_prior = stratum_sources.get(source_digest)
            if stratum_prior is not None and stratum_prior != record:
                raise AromaticCycleOpenImpactAuditError(
                    "one exact source has conflicting stratum reachability records"
                )
            stratum_sources[source_digest] = record
        receipt_summaries.append(
            {
                "task_index": task["task_index"],
                "partition": task["partition"],
                "layer": task["layer"],
                "packed_shard_content_sha256": task["packed_shard_content_sha256"],
                "receipt_filename": task["receipt_filename"],
                "receipt_sha256": receipt["receipt_sha256"],
            }
        )

    affected_rows = totals["perceived_aromatic_bond_delete_teacher_rows"]
    if sum(outcomes.values()) != affected_rows:
        raise AromaticCycleOpenImpactAuditError(
            "reduced resolver outcomes do not partition affected rows"
        )
    admitted = outcomes["admitted"]
    if (
        sum(exact_comparisons.values()) != admitted
        or sum(canonical_comparisons.values()) != admitted
    ):
        raise AromaticCycleOpenImpactAuditError(
            "reduced successor comparisons do not cover admitted rows"
        )

    stratum_rows = [
        {
            "partition": partition,
            "layer": layer,
            "counts": dict(sorted(counts.items())),
        }
        for (partition, layer), counts in sorted(
            strata.items(), key=lambda item: (_PARTITION_ORDER[item[0][0]], item[0][1])
        )
    ]
    source_records = [
        reachability_by_source[source_digest] for source_digest in sorted(reachability_by_source)
    ]
    source_reachability_strata = [
        {
            "partition": partition,
            "layer": layer,
            "summary": _source_reachability_summary(
                [records[source_digest] for source_digest in sorted(records)]
            ),
        }
        for (partition, layer), records in sorted(
            stratum_reachability.items(),
            key=lambda item: (_PARTITION_ORDER[item[0][0]], item[0][1]),
        )
    ]
    identity_sets = {
        "affected_trace_identity_sha256s": sorted(trace_ids),
        "affected_exact_source_state_sha256s": sorted(source_ids),
        "cycle_open_teacher_exact_source_state_sha256s": sorted(reachability_by_source),
    }
    body = {
        "schema": RESULT_SCHEMA,
        "schema_version": RESULT_SCHEMA_VERSION,
        "status": NON_AUTHORIZING_STATUS,
        "training_authorized": False,
        "contract_sha256": contract["contract_sha256"],
        "plan_sha256": validated_plan["plan_sha256"],
        "implementation_sha256": implementation_sha256,
        "scope": {
            "partitions": list(contract["partitions"]),
            "excluded_partitions": list(contract["excluded_partitions"]),
            "census_unit": "accepted_nonterminal_teacher_row",
            "aromaticity_source": "resonance_invariant_bond_classes",
            "historical_executor_replayed": False,
            "corpus_rebuilt": False,
            "active8_partition_census": validated_plan["active8_partition_census"],
            "full_admitted_train_impact_available": False,
            "limitation": (
                "the pinned Active8 parent is validation-only; full admitted-train "
                "impact and rematerialization counts remain unavailable"
            ),
        },
        "denominators": {
            "accepted_nonterminal_teacher_rows": totals["accepted_nonterminal_teacher_rows"],
            "accepted_cycle_attach_teacher_rows": totals["accepted_cycle_attach_teacher_rows"],
            "perceived_aromatic_bond_delete_teacher_rows": affected_rows,
            "nonaromatic_cycle_attach_teacher_rows": totals[
                "nonaromatic_cycle_attach_teacher_rows"
            ],
            "traces_with_perceived_aromatic_bond_delete_teacher": totals[
                "traces_with_perceived_aromatic_bond_delete_teacher_in_shard"
            ],
            "cycle_open_teacher_source_occurrences": totals["accepted_cycle_attach_teacher_rows"],
            "perceived_aromatic_teacher_source_occurrences": affected_rows,
        },
        "unique_counts": {
            "affected_traces": len(trace_ids),
            "affected_exact_source_states": len(source_ids),
            "cycle_open_teacher_exact_source_states": len(reachability_by_source),
        },
        "counts": dict(sorted(totals.items())),
        "accepted_teacher_rows_by_family": dict(sorted(family_rows.items())),
        "raw_deleted_bond_class": dict(sorted(raw_orders.items())),
        "resolver_outcomes": dict(sorted(outcomes.items())),
        "prototype_vs_historical_exact_digest": dict(sorted(exact_comparisons.items())),
        "prototype_vs_historical_canonical_key": dict(sorted(canonical_comparisons.items())),
        "partition_layer_strata": stratum_rows,
        "source_reachability": {
            "scope": (
                "every unique accepted exact source state containing at least "
                "one cycle_attach teacher"
            ),
            "edge_definition": (
                "resonance-invariant perceived aromatic bonds that are not graph bridges"
            ),
            "topology_semantics": {
                "isolated_aromatic_system": (
                    "source has an aromatic connected component with graph cycle rank one"
                ),
                "fused_aromatic_rings": (
                    "at least two RDKit symmetrized-SSSR aromatic rings share two or more atoms"
                ),
                "bridged_ring_source": ("RDKit reports at least one bridgehead atom in the source"),
                "flags_overlap": True,
                "ring_size": "RDKit symmetrized-SSSR aromatic ring size",
                "aromatic_system_size": (
                    "atom count in a resonance-invariant aromatic-edge component"
                ),
            },
            "threshold_applied": False,
            "summary": _source_reachability_summary(source_records),
            "by_partition_layer": source_reachability_strata,
            "records": source_records,
        },
        "identity_sets_sha256": semantic_sha256(identity_sets),
        "receipts": receipt_summaries,
    }
    return {**body, "result_sha256": semantic_sha256(body)}


def load_json_artifact(path: str | Path, *, description: str) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise AromaticCycleOpenImpactAuditError(f"cannot load {description}: {path}") from error
    if not isinstance(value, dict):
        raise AromaticCycleOpenImpactAuditError(f"{description} must be an object")
    return value


__all__ = [
    "AromaticCycleOpenImpactAuditError",
    "NON_AUTHORIZING_STATUS",
    "active8_partition_census",
    "audit_one_shard",
    "build_audit_plan",
    "canonical_json_bytes",
    "file_sha256",
    "implementation_identity",
    "load_audit_contract",
    "load_json_artifact",
    "pretty_json_bytes",
    "reduce_shard_receipts",
    "semantic_sha256",
    "validate_audit_plan",
    "validate_parent_admission",
    "validate_shard_receipt",
]
