"""A2.2b: exhaustive audit of the frozen ``carbon_6_slots`` exact graph.

The structural graph already exists.  This module does not change its support.
It rebuilds that graph through ``enumerable_ringcore``, assigns a deterministic
index to every production canonical key, then classifies every raw action in the
exact benchmark's declared action language.

Each raw candidate receives exactly one disposition:

``included_productive``
    Executes to a different canonical state inside the closed slice.
``virtual_self``
    Executes successfully but canonicalizes to the source molecule.
``excluded_executor_invalid``
    The production executor rejects the raw Cartesian-language proposal.
``excluded_vocabulary_boundary``
    A production null-root action leaves the carbon-only vocabulary.
``excluded_outside_closed_slice``
    Executes productively but is absent from the supposedly closed graph.  This
    disposition is represented explicitly and is a hard failure if nonzero.

The only probability law here is a diagnostic fixed marked law, uniform over
in-slice executable marks.  It exists solely to verify production segmented
aggregation and productive-versus-virtual mass accounting.  It is not the E6
solver kernel and no learned checkpoint is used.

The independent dictionary aggregator belongs in tests and is never imported
by this result-producing module.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from rdkit import rdBase

from compose_v4.chem.molecular_graph import MolecularGraph, NULL_IDX
from compose_v4.experiments.enumerable_ringcore import (
    NULL_KEY,
    Candidate,
    ReachableGraph,
    build_reachable_graph,
    graph_fingerprint,
)
from compose_v4.experiments.registry import load_registry
from compose_v4.model.segmented_successor import segmented_successor_logprobs
from compose_v4.rewrite.action_codec import (
    canonical_family,
    canonical_json,
    encode_action,
    public_operator_name,
)
from compose_v4.rewrite.factorized_fiber import _factorized_candidates
from compose_v4.rewrite.fiber import _candidate_actions
from compose_v4.rewrite.kernel import (
    InvalidRewrite,
    canonical_state_key,
    de_novo_rewrite_system,
)

A2_2B_CONTRACT_SCHEMA = "compose.experiments.e6_a2_2b_graph_audit_contract"
A2_2B_CONTRACT_VERSION = 1
A2_2B_ARTIFACT_SCHEMA = "compose.experiments.e6_a2_2b_graph_audit"
A2_2B_ARTIFACT_VERSION = 1
A2_2B_ARTIFACT_STATUS = "GRAPH_AUDIT_COMPLETE_NO_CONTROL_RESULT"

INCLUDED_PRODUCTIVE = "included_productive"
VIRTUAL_SELF = "virtual_self"
EXCLUDED_EXECUTOR_INVALID = "excluded_executor_invalid"
EXCLUDED_VOCABULARY_BOUNDARY = "excluded_vocabulary_boundary"
EXCLUDED_OUTSIDE_CLOSED_SLICE = "excluded_outside_closed_slice"
ACTION_DISPOSITIONS = (
    INCLUDED_PRODUCTIVE,
    VIRTUAL_SELF,
    EXCLUDED_EXECUTOR_INVALID,
    EXCLUDED_VOCABULARY_BOUNDARY,
    EXCLUDED_OUTSIDE_CLOSED_SLICE,
)

_ROOT = Path(__file__).resolve().parents[3]
_IMPLEMENTATION_SOURCES = (
    "src/compose_v4/chem/molecular_graph.py",
    "src/compose_v4/chem/state.py",
    "src/compose_v4/experiments/e6_graph_audit.py",
    "src/compose_v4/experiments/enumerable_ringcore.py",
    "src/compose_v4/experiments/registry.py",
    "src/compose_v4/model/segmented_successor.py",
    "src/compose_v4/rewrite/action_codec.py",
    "src/compose_v4/rewrite/factorized_fiber.py",
    "src/compose_v4/rewrite/fiber.py",
    "src/compose_v4/rewrite/kernel.py",
    "src/compose_v4/rewrite/operators.py",
)
_DICTIONARY_ORACLE_TEST_SOURCE = "tests/test_e6_graph_audit.py"
_CONTRACT_FIELDS = {
    "schema",
    "schema_version",
    "status",
    "experiment_id",
    "milestone",
    "paper_claim_authorized",
    "checkpoint_required",
    "benchmark",
    "enumeration_contract",
    "action_dispositions",
    "diagnostic_mark_law",
    "numeric_tolerances",
    "required_invariants",
    "scope_caveat",
    "contract_sha256",
}
_ARTIFACT_FIELDS = {
    "schema",
    "schema_version",
    "status",
    "experiment_id",
    "milestone",
    "paper_claim_authorized",
    "checkpoint_used",
    "control_solver_run",
    "scope_caveat",
    "benchmark",
    "provenance",
    "state_index",
    "state_rows",
    "aggregate",
    "invariants",
    "dictionary_oracle_policy",
    "graph_audit_hash",
    "artifact_sha256",
}


class E6GraphAuditError(RuntimeError):
    """The graph contract, exhaustive classification, or provenance failed."""


@dataclass(frozen=True)
class ActionClassification:
    """One raw action after production execution and boundary classification."""

    raw_index: int
    executor_rule: str
    model_family: str
    public_operator: str
    action_json: str
    disposition: str
    successor_key: str | None

    def hash_payload(self) -> dict[str, Any]:
        return {
            "raw_index": self.raw_index,
            "executor_rule": self.executor_rule,
            "model_family": self.model_family,
            "public_operator": self.public_operator,
            "action": json.loads(self.action_json),
            "disposition": self.disposition,
            "successor_key": self.successor_key,
        }


@dataclass(frozen=True)
class AuditedState:
    """Complete classification and segmented law for one canonical source."""

    source_index: int
    source_key: str
    actions: tuple[ActionClassification, ...]
    successor_probabilities: dict[str, float]
    alias_multiplicities: dict[str, int]
    raw_productive_mass: float
    virtual_self_mass: float


def _canonical_json_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise E6GraphAuditError("A2.2b metadata is not finite canonical JSON") from error


def stable_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def contract_self_hash(contract: Mapping[str, Any]) -> str:
    body = dict(contract)
    body.pop("contract_sha256", None)
    return stable_sha256(body)


def artifact_self_hash(artifact: Mapping[str, Any]) -> str:
    body = dict(artifact)
    body.pop("artifact_sha256", None)
    return stable_sha256(body)


def implementation_sha256(repo_root: str | Path = _ROOT) -> str:
    root = Path(repo_root)
    digest = hashlib.sha256()
    for relative in sorted(_IMPLEMENTATION_SOURCES):
        path = root / relative
        if not path.is_file():
            raise E6GraphAuditError(f"A2.2b implementation source is absent: {relative}")
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _stable_registry_path(
    registry_path: str | Path,
    *,
    repo_root: str | Path = _ROOT,
) -> str:
    """Use one provenance name for the same registry passed relatively or absolutely."""

    path = Path(registry_path).resolve()
    root = Path(repo_root).resolve()
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return str(path)


def _dictionary_oracle_policy(
    repo_root: str | Path = _ROOT,
) -> dict[str, Any]:
    source = Path(repo_root) / _DICTIONARY_ORACLE_TEST_SOURCE
    if not source.is_file():
        raise E6GraphAuditError(
            f"fresh dictionary-oracle source is absent: {_DICTIONARY_ORACLE_TEST_SOURCE}"
        )
    return {
        "role": "fresh_test_only_oracle",
        "used_to_produce_audit_counts_or_hashes": False,
        "exhaustive_equivalence_required_before_review": True,
        "source_path": _DICTIONARY_ORACLE_TEST_SOURCE,
        "source_file_sha256": file_sha256(source),
    }


def _benchmark_candidate(contract: Mapping[str, Any]) -> Candidate:
    benchmark = contract["benchmark"]
    return Candidate(
        candidate_id=str(benchmark["candidate_id"]),
        seed_smiles=str(benchmark["seed_smiles"]),
        elements=tuple(str(value) for value in benchmark["elements"]),
        max_hydrogens=int(benchmark["max_hydrogens"]),
        horizon=benchmark["horizon"],
    )


def _validate_contract_binding(
    contract: Mapping[str, Any],
    *,
    registry_path: str | Path,
) -> None:
    if set(contract) != _CONTRACT_FIELDS:
        raise E6GraphAuditError("A2.2b contract has missing or unknown top-level fields")
    if (
        contract["schema"] != A2_2B_CONTRACT_SCHEMA
        or contract["schema_version"] != A2_2B_CONTRACT_VERSION
        or contract["experiment_id"] != "E6"
        or contract["milestone"] != "A2.2b"
        or contract["paper_claim_authorized"] is not False
        or contract["checkpoint_required"] is not False
        or contract["action_dispositions"] != list(ACTION_DISPOSITIONS)
    ):
        raise E6GraphAuditError("A2.2b contract identity or non-claiming status is invalid")
    if contract.get("contract_sha256") != contract_self_hash(contract):
        raise E6GraphAuditError("A2.2b contract self-hash does not match")

    registry = load_registry(registry_path)
    selected = registry["exact_sizing"]["selected"]
    benchmark = contract["benchmark"]
    expected = {
        "candidate_id": benchmark["candidate_id"],
        "n_states": benchmark["expected_n_states"],
        "n_edges": benchmark["expected_n_edges"],
        "graph_fingerprint": benchmark["expected_structural_graph_fingerprint"],
    }
    observed = {
        "candidate_id": selected["candidate_id"],
        "n_states": selected["n_states"],
        "n_edges": selected["n_edges"],
        "graph_fingerprint": selected["graph_fingerprint"],
    }
    if observed != expected:
        raise E6GraphAuditError(
            f"registry-selected exact graph drifted from A2.2b: {observed} != {expected}"
        )
    enumeration = contract["enumeration_contract"]
    if enumeration["declared_executor_rules"] != [
        "atom_insert",
        "atom_delete",
        "atom_restate",
        "bond_insert",
        "bond_delete",
        "bond_reorder",
    ]:
        raise E6GraphAuditError("A2.2b declared executor-rule language changed")
    if "bond_reroute" not in enumeration["outside_declared_language"]:
        raise E6GraphAuditError("A2.2b must state that graft lies outside this exact slice")
    law = contract["diagnostic_mark_law"]
    if law["is_solver_kernel"] is not False:
        raise E6GraphAuditError("the A2.2b diagnostic marked law must not be called the solver kernel")


def load_a2_2b_contract(
    path: str | Path,
    *,
    registry_path: str | Path,
) -> dict[str, Any]:
    try:
        payload = json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise E6GraphAuditError(f"cannot read A2.2b contract: {path}") from error
    if not isinstance(payload, dict):
        raise E6GraphAuditError("A2.2b contract must be an object")
    _validate_contract_binding(payload, registry_path=registry_path)
    return payload


def deterministic_state_index(graph: ReachableGraph) -> tuple[str, ...]:
    """Canonical keys in the sole state-index order used downstream."""

    keys = tuple(sorted(graph.states))
    if len(keys) != len(set(keys)):
        raise E6GraphAuditError("canonical state index contains duplicates")
    return keys


def _successor_elements(key: str) -> set[str]:
    if key == NULL_KEY:
        return set()
    from rdkit import Chem  # noqa: PLC0415

    molecule = Chem.MolFromSmiles(key, sanitize=False)
    if molecule is None:
        raise E6GraphAuditError(f"canonical successor key is not parseable: {key!r}")
    return {atom.GetSymbol() for atom in molecule.GetAtoms()}


def _raw_candidates(
    state: MolecularGraph,
    source_key: str,
    candidate: Candidate,
) -> tuple[tuple[str, Any], ...]:
    if source_key == NULL_KEY:
        return tuple(
            _factorized_candidates(
                state,
                allow_bond_reroute=False,
            )
        )
    return tuple(_candidate_actions(state, candidate.spec()))


def _segmented_uniform_productive_law(
    classifications: Sequence[ActionClassification],
) -> tuple[dict[str, float], dict[str, int]]:
    productive = [
        action
        for action in classifications
        if action.disposition == INCLUDED_PRODUCTIVE
    ]
    if not productive:
        return {}, {}
    keys = tuple(sorted({str(action.successor_key) for action in productive}))
    key_to_index = {key: index for index, key in enumerate(keys)}
    log_probabilities = segmented_successor_logprobs(
        torch.zeros(len(productive), dtype=torch.float64),
        torch.zeros(len(productive), dtype=torch.long),
        torch.tensor(
            [key_to_index[str(action.successor_key)] for action in productive],
            dtype=torch.long,
        ),
        n_examples=1,
        n_successors=len(keys),
    )
    multiplicities = Counter(str(action.successor_key) for action in productive)
    return (
        {
            key: float(log_probabilities[index].exp())
            for index, key in enumerate(keys)
        },
        dict(sorted(multiplicities.items())),
    )


def audit_state_actions(
    state: MolecularGraph,
    *,
    source_index: int,
    source_key: str,
    candidate: Candidate,
    state_index: Mapping[str, int],
    system: Any = None,
) -> AuditedState:
    """Classify every raw candidate and aggregate productive aliases."""

    runtime = system if system is not None else de_novo_rewrite_system()
    allowed_elements = set(candidate.elements)
    actions: list[ActionClassification] = []
    seen_actions: set[str] = set()
    for raw_index, (rule_name, action) in enumerate(
        _raw_candidates(state, source_key, candidate)
    ):
        encoded = encode_action(rule_name, action)
        action_json = canonical_json(encoded)
        if action_json in seen_actions:
            raise E6GraphAuditError(
                f"source {source_key!r} enumerated the same raw action more than once"
            )
        seen_actions.add(action_json)
        family = canonical_family(rule_name)
        successor_key: str | None = None
        try:
            successor = runtime.apply(state, rule_name, action)
        except InvalidRewrite:
            disposition = EXCLUDED_EXECUTOR_INVALID
        else:
            successor_key = canonical_state_key(successor)
            if successor_key == source_key:
                disposition = VIRTUAL_SELF
            elif not _successor_elements(successor_key) <= allowed_elements:
                disposition = EXCLUDED_VOCABULARY_BOUNDARY
            elif successor_key not in state_index:
                disposition = EXCLUDED_OUTSIDE_CLOSED_SLICE
            else:
                disposition = INCLUDED_PRODUCTIVE
        actions.append(
            ActionClassification(
                raw_index=raw_index,
                executor_rule=rule_name,
                model_family=family,
                public_operator=public_operator_name(family),
                action_json=action_json,
                disposition=disposition,
                successor_key=successor_key,
            )
        )

    probabilities, aliases = _segmented_uniform_productive_law(actions)
    productive_count = sum(
        action.disposition == INCLUDED_PRODUCTIVE for action in actions
    )
    virtual_count = sum(action.disposition == VIRTUAL_SELF for action in actions)
    legal_count = productive_count + virtual_count
    if legal_count <= 0:
        raise E6GraphAuditError(f"source {source_key!r} has no in-slice executable mark")
    return AuditedState(
        source_index=source_index,
        source_key=source_key,
        actions=tuple(actions),
        successor_probabilities=probabilities,
        alias_multiplicities=aliases,
        raw_productive_mass=productive_count / legal_count,
        virtual_self_mass=virtual_count / legal_count,
    )


def _state_row(audited: AuditedState, state: MolecularGraph) -> dict[str, Any]:
    disposition_counts = Counter(action.disposition for action in audited.actions)
    family_dispositions: dict[str, Counter[str]] = defaultdict(Counter)
    for action in audited.actions:
        family_dispositions[action.model_family][action.disposition] += 1
    productive_count = disposition_counts[INCLUDED_PRODUCTIVE]
    virtual_count = disposition_counts[VIRTUAL_SELF]
    canonical_count = len(audited.alias_multiplicities)
    alias_histogram = Counter(audited.alias_multiplicities.values())
    normalization_residual = abs(sum(audited.successor_probabilities.values()) - 1.0)
    outgoing = tuple(sorted(audited.successor_probabilities))
    action_stream = [action.hash_payload() for action in audited.actions]
    law_rows = [
        {
            "successor_key": key,
            "probability": audited.successor_probabilities[key],
            "alias_multiplicity": audited.alias_multiplicities[key],
        }
        for key in outgoing
    ]
    body = {
        "state_index": audited.source_index,
        "canonical_key": audited.source_key,
        "active_atom_count": int(state.n_real_atoms),
        "slot_count": len(state.atom_types),
        "at_slot_capacity": not bool(np.any(state.atom_types == NULL_IDX)),
        "raw_action_count": len(audited.actions),
        "disposition_counts": {
            disposition: disposition_counts[disposition]
            for disposition in ACTION_DISPOSITIONS
        },
        "family_disposition_counts": {
            family: {
                disposition: counts[disposition]
                for disposition in ACTION_DISPOSITIONS
            }
            for family, counts in sorted(family_dispositions.items())
        },
        "in_slice_executable_mark_count": productive_count + virtual_count,
        "productive_mark_count": productive_count,
        "virtual_self_mark_count": virtual_count,
        "canonical_out_edge_count": canonical_count,
        "raw_mark_to_edge_compression": productive_count - canonical_count,
        "alias_multiplicity_histogram": {
            str(value): count for value, count in sorted(alias_histogram.items())
        },
        "maximum_alias_multiplicity": max(
            audited.alias_multiplicities.values(),
            default=0,
        ),
        "raw_productive_mass": audited.raw_productive_mass,
        "virtual_self_mass": audited.virtual_self_mass,
        "executable_mass_reconstruction_residual": abs(
            audited.raw_productive_mass + audited.virtual_self_mass - 1.0
        ),
        "segmented_successor_normalization_residual": normalization_residual,
        "outgoing_successor_index_hash": stable_sha256(outgoing),
        "segmented_law_hash": stable_sha256(law_rows),
        "action_stream_sha256": stable_sha256(action_stream),
    }
    return {**body, "row_sha256": stable_sha256(body)}


def _aggregate_rows(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    disposition_totals = Counter()
    family_totals: dict[str, Counter[str]] = defaultdict(Counter)
    alias_histogram = Counter()
    for row in rows:
        disposition_totals.update(row["disposition_counts"])
        for family, counts in row["family_disposition_counts"].items():
            family_totals[family].update(counts)
        alias_histogram.update(
            {
                int(multiplicity): int(count)
                for multiplicity, count in row["alias_multiplicity_histogram"].items()
            }
        )
    total_raw = sum(disposition_totals.values())
    productive = disposition_totals[INCLUDED_PRODUCTIVE]
    virtual = disposition_totals[VIRTUAL_SELF]
    canonical_edges = sum(int(row["canonical_out_edge_count"]) for row in rows)
    return {
        "state_count": len(rows),
        "raw_action_count": total_raw,
        "disposition_counts": {
            disposition: disposition_totals[disposition]
            for disposition in ACTION_DISPOSITIONS
        },
        "family_disposition_counts": {
            family: {
                disposition: counts[disposition]
                for disposition in ACTION_DISPOSITIONS
            }
            for family, counts in sorted(family_totals.items())
        },
        "in_slice_executable_mark_count": productive + virtual,
        "productive_mark_count": productive,
        "virtual_self_mark_count": virtual,
        "canonical_directed_edge_count": canonical_edges,
        "raw_productive_mark_to_canonical_edge_compression": productive - canonical_edges,
        "alias_multiplicity_histogram": {
            str(multiplicity): count
            for multiplicity, count in sorted(alias_histogram.items())
        },
        "maximum_alias_multiplicity": max(alias_histogram, default=0),
        "states_with_any_aliased_successor": sum(
            any(int(value) > 1 for value in row["alias_multiplicity_histogram"])
            for row in rows
        ),
        "states_at_slot_capacity": sum(bool(row["at_slot_capacity"]) for row in rows),
        "maximum_segmented_normalization_residual": max(
            float(row["segmented_successor_normalization_residual"])
            for row in rows
        ),
        "maximum_executable_mass_reconstruction_residual": max(
            float(row["executable_mass_reconstruction_residual"])
            for row in rows
        ),
    }


def audit_reachable_graph(
    graph: ReachableGraph,
    *,
    candidate: Candidate,
    contract: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, bool]]:
    """Audit one already-built graph without selecting or changing it."""

    keys = deterministic_state_index(graph)
    index = {key: position for position, key in enumerate(keys)}
    audited_states = [
        audit_state_actions(
            graph.states[key],
            source_index=position,
            source_key=key,
            candidate=candidate,
            state_index=index,
        )
        for position, key in enumerate(keys)
    ]
    rows = [
        _state_row(audited, graph.states[audited.source_key])
        for audited in audited_states
    ]
    reconstructed_edges = {
        (audited.source_key, successor_key)
        for audited in audited_states
        for successor_key in audited.successor_probabilities
    }
    aggregate = _aggregate_rows(rows)
    tolerance = float(contract["numeric_tolerances"]["normalization"])
    expected = contract["benchmark"]
    invariants = {
        "reachable_search_closed": graph.stop_reason == "closed",
        "deterministic_sorted_canonical_state_index": (
            keys == tuple(sorted(keys)) and len(keys) == len(set(keys))
        ),
        "structural_graph_fingerprint_unchanged": (
            graph_fingerprint(graph)
            == expected["expected_structural_graph_fingerprint"]
        ),
        "every_raw_action_has_exactly_one_disposition": all(
            row["raw_action_count"] == sum(row["disposition_counts"].values())
            for row in rows
        ),
        "no_productive_action_leaves_closed_slice": (
            aggregate["disposition_counts"][EXCLUDED_OUTSIDE_CLOSED_SLICE] == 0
        ),
        "every_structural_edge_has_at_least_one_productive_mark": all(
            multiplicity > 0
            for audited in audited_states
            for multiplicity in audited.alias_multiplicities.values()
        ),
        "reconstructed_edge_set_equals_structural_edge_set": (
            reconstructed_edges == graph.edges
        ),
        "segmented_successor_rows_normalized": (
            aggregate["maximum_segmented_normalization_residual"] <= tolerance
        ),
        "raw_mark_alias_compression_reported": (
            aggregate["raw_productive_mark_to_canonical_edge_compression"]
            == aggregate["productive_mark_count"] - graph.n_edges
        ),
        "productive_and_virtual_mass_reconstruct_executable_mass": (
            aggregate["maximum_executable_mass_reconstruction_residual"]
            <= tolerance
        ),
    }
    state_index_payload = {
        "ordering": "ascending production canonical_state_key",
        "count": len(keys),
        "keys": list(keys),
        "state_index_sha256": stable_sha256(list(keys)),
    }
    return state_index_payload, rows, invariants


def run_a2_2b_graph_audit(
    contract: Mapping[str, Any],
    *,
    registry_path: str | Path,
    repo_root: str | Path = _ROOT,
) -> dict[str, Any]:
    """Rebuild and exhaustively audit the frozen selected exact graph."""

    _validate_contract_binding(contract, registry_path=registry_path)
    candidate = _benchmark_candidate(contract)
    registry = load_registry(registry_path)
    exact_sizing = registry["exact_sizing"]
    graph = build_reachable_graph(
        candidate,
        state_cap=int(exact_sizing["N_max"]) + 1,
        edge_cap=int(exact_sizing["E_max"]) + 1,
        deadline_seconds=None,
    )
    state_index, rows, invariants = audit_reachable_graph(
        graph,
        candidate=candidate,
        contract=contract,
    )
    benchmark = contract["benchmark"]
    invariants["registry_selected_identity_matches_contract"] = True
    invariants["state_count_unchanged"] = graph.n_states == benchmark["expected_n_states"]
    invariants["edge_count_unchanged"] = graph.n_edges == benchmark["expected_n_edges"]

    aggregate = _aggregate_rows(rows)
    graph_audit_payload = {
        "schema": "compose.experiments.e6_a2_2b_graph_audit_identity",
        "schema_version": 1,
        "contract_sha256": contract["contract_sha256"],
        "structural_graph_fingerprint": graph_fingerprint(graph),
        "state_index_sha256": state_index["state_index_sha256"],
        "state_row_hashes": [row["row_sha256"] for row in rows],
        "aggregate": aggregate,
    }
    graph_audit_hash = stable_sha256(graph_audit_payload)
    invariants["graph_audit_hash_separate_from_structural_fingerprint"] = (
        graph_audit_hash != graph_fingerprint(graph)
    )
    failed = sorted(name for name, passed in invariants.items() if not passed)
    if failed:
        raise E6GraphAuditError(f"A2.2b invariants failed: {failed}")

    registry_path = Path(registry_path)
    body: dict[str, Any] = {
        "schema": A2_2B_ARTIFACT_SCHEMA,
        "schema_version": A2_2B_ARTIFACT_VERSION,
        "status": A2_2B_ARTIFACT_STATUS,
        "experiment_id": "E6",
        "milestone": "A2.2b",
        "paper_claim_authorized": False,
        "checkpoint_used": None,
        "control_solver_run": False,
        "scope_caveat": contract["scope_caveat"],
        "benchmark": {
            "candidate_id": candidate.candidate_id,
            "seed_smiles": candidate.seed_smiles,
            "declared_elements": list(candidate.elements),
            "max_hydrogens": candidate.max_hydrogens,
            "horizon": candidate.horizon,
            "structural_graph_fingerprint": graph_fingerprint(graph),
            "n_states": graph.n_states,
            "n_edges": graph.n_edges,
            "stop_reason": graph.stop_reason,
            "declared_executor_rules": contract["enumeration_contract"][
                "declared_executor_rules"
            ],
            "outside_declared_language": contract["enumeration_contract"][
                "outside_declared_language"
            ],
            "slot_capacity_policy": contract["enumeration_contract"][
                "slot_capacity_policy"
            ],
            "diagnostic_mark_law": contract["diagnostic_mark_law"],
        },
        "provenance": {
            "contract_sha256": contract["contract_sha256"],
            "registry_path": _stable_registry_path(
                registry_path,
                repo_root=repo_root,
            ),
            "registry_file_sha256": file_sha256(registry_path),
            "registry_protocol_content_hash": registry["protocol"][
                "protocol_freeze"
            ]["content_hash"],
            "implementation_sha256": implementation_sha256(repo_root),
            "implementation_sources": list(_IMPLEMENTATION_SOURCES),
            "numpy_version": np.__version__,
            "torch_version": torch.__version__,
            "rdkit_version": rdBase.rdkitVersion,
        },
        "state_index": state_index,
        "state_rows": rows,
        "aggregate": aggregate,
        "invariants": dict(sorted(invariants.items())),
        "dictionary_oracle_policy": _dictionary_oracle_policy(repo_root),
        "graph_audit_hash": graph_audit_hash,
    }
    return {**body, "artifact_sha256": stable_sha256(body)}


def validate_a2_2b_artifact(
    artifact: Mapping[str, Any],
    *,
    contract: Mapping[str, Any],
    registry_path: str | Path,
    repo_root: str | Path = _ROOT,
) -> None:
    """Fail closed on artifact drift, tampering, or weakened scope."""

    _validate_contract_binding(contract, registry_path=registry_path)
    if set(artifact) != _ARTIFACT_FIELDS:
        raise E6GraphAuditError("A2.2b artifact has missing or unknown top-level fields")
    if (
        artifact["schema"] != A2_2B_ARTIFACT_SCHEMA
        or artifact["schema_version"] != A2_2B_ARTIFACT_VERSION
        or artifact["status"] != A2_2B_ARTIFACT_STATUS
        or artifact["experiment_id"] != "E6"
        or artifact["milestone"] != "A2.2b"
        or artifact["paper_claim_authorized"] is not False
        or artifact["checkpoint_used"] is not None
        or artifact["control_solver_run"] is not False
    ):
        raise E6GraphAuditError("A2.2b artifact identity or non-result status is invalid")
    if artifact_self_hash(artifact) != artifact["artifact_sha256"]:
        raise E6GraphAuditError("A2.2b artifact self-hash does not match")
    registry_path = Path(registry_path)
    registry = load_registry(registry_path)
    expected_provenance = {
        "contract_sha256": contract["contract_sha256"],
        "registry_path": _stable_registry_path(
            registry_path,
            repo_root=repo_root,
        ),
        "registry_file_sha256": file_sha256(registry_path),
        "registry_protocol_content_hash": registry["protocol"][
            "protocol_freeze"
        ]["content_hash"],
        "implementation_sha256": implementation_sha256(repo_root),
        "implementation_sources": list(_IMPLEMENTATION_SOURCES),
        "numpy_version": np.__version__,
        "torch_version": torch.__version__,
        "rdkit_version": rdBase.rdkitVersion,
    }
    if artifact["provenance"] != expected_provenance:
        raise E6GraphAuditError("A2.2b artifact provenance does not match current sources")
    if not artifact["invariants"] or not all(artifact["invariants"].values()):
        raise E6GraphAuditError("A2.2b artifact contains an absent or failed invariant")
    if artifact["dictionary_oracle_policy"] != _dictionary_oracle_policy(repo_root):
        raise E6GraphAuditError("A2.2b artifact weakens the dictionary-oracle boundary")
    state_index = artifact["state_index"]
    if (
        state_index["count"] != len(state_index["keys"])
        or state_index["keys"] != sorted(state_index["keys"])
        or state_index["state_index_sha256"] != stable_sha256(state_index["keys"])
    ):
        raise E6GraphAuditError("A2.2b canonical state index is inconsistent")
    for row in artifact["state_rows"]:
        body = dict(row)
        claimed_row_hash = body.pop("row_sha256", None)
        if claimed_row_hash != stable_sha256(body):
            raise E6GraphAuditError("A2.2b state-row self-hash does not match")
    if _aggregate_rows(artifact["state_rows"]) != artifact["aggregate"]:
        raise E6GraphAuditError("A2.2b aggregate does not reconstruct from state rows")
    graph_audit_payload = {
        "schema": "compose.experiments.e6_a2_2b_graph_audit_identity",
        "schema_version": 1,
        "contract_sha256": contract["contract_sha256"],
        "structural_graph_fingerprint": artifact["benchmark"][
            "structural_graph_fingerprint"
        ],
        "state_index_sha256": artifact["state_index"]["state_index_sha256"],
        "state_row_hashes": [row["row_sha256"] for row in artifact["state_rows"]],
        "aggregate": artifact["aggregate"],
    }
    if stable_sha256(graph_audit_payload) != artifact["graph_audit_hash"]:
        raise E6GraphAuditError("A2.2b graph-audit hash does not match its audited rows")


def freeze_a2_2b_artifact(
    artifact: Mapping[str, Any],
    output_path: str | Path,
    *,
    contract: Mapping[str, Any],
    registry_path: str | Path,
    repo_root: str | Path = _ROOT,
) -> None:
    """Freeze canonical evidence bytes once without overwrite."""

    validate_a2_2b_artifact(
        artifact,
        contract=contract,
        registry_path=registry_path,
        repo_root=repo_root,
    )
    encoded = _canonical_json_bytes(artifact) + b"\n"
    path = Path(output_path)
    if path.exists():
        if path.read_bytes() != encoded:
            raise E6GraphAuditError(f"immutable A2.2b artifact collision: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        handle = path.open("xb")
    except FileExistsError as error:
        raise E6GraphAuditError(f"immutable A2.2b artifact collision: {path}") from error
    with handle:
        handle.write(encoded)


__all__ = [
    "A2_2B_ARTIFACT_SCHEMA",
    "A2_2B_ARTIFACT_STATUS",
    "A2_2B_ARTIFACT_VERSION",
    "A2_2B_CONTRACT_SCHEMA",
    "A2_2B_CONTRACT_VERSION",
    "ACTION_DISPOSITIONS",
    "EXCLUDED_EXECUTOR_INVALID",
    "EXCLUDED_OUTSIDE_CLOSED_SLICE",
    "EXCLUDED_VOCABULARY_BOUNDARY",
    "E6GraphAuditError",
    "INCLUDED_PRODUCTIVE",
    "VIRTUAL_SELF",
    "ActionClassification",
    "AuditedState",
    "artifact_self_hash",
    "audit_reachable_graph",
    "audit_state_actions",
    "contract_self_hash",
    "deterministic_state_index",
    "file_sha256",
    "freeze_a2_2b_artifact",
    "implementation_sha256",
    "load_a2_2b_contract",
    "run_a2_2b_graph_audit",
    "stable_sha256",
    "validate_a2_2b_artifact",
]
