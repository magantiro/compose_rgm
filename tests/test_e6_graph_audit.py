"""A2.2b exact-graph classification and independent dictionary verification."""

from __future__ import annotations

import copy
import json
from collections import Counter, defaultdict
from pathlib import Path

import pytest
from rdkit import Chem

from compose_v4.experiments.e6_graph_audit import (
    EXCLUDED_EXECUTOR_INVALID,
    EXCLUDED_OUTSIDE_CLOSED_SLICE,
    EXCLUDED_VOCABULARY_BOUNDARY,
    INCLUDED_PRODUCTIVE,
    VIRTUAL_SELF,
    E6GraphAuditError,
    artifact_self_hash,
    audit_state_actions,
    contract_self_hash,
    deterministic_state_index,
    load_a2_2b_contract,
    stable_sha256,
    validate_a2_2b_artifact,
)
from compose_v4.experiments.enumerable_ringcore import (
    NULL_KEY,
    Candidate,
    build_reachable_graph,
    graph_fingerprint,
)
from compose_v4.rewrite.factorized_fiber import _factorized_candidates
from compose_v4.rewrite.fiber import _candidate_actions
from compose_v4.rewrite.kernel import (
    InvalidRewrite,
    canonical_state_key,
    de_novo_rewrite_system,
)

_ROOT = Path(__file__).resolve().parent.parent
_CONTRACT_PATH = _ROOT / "configs/exact_control_a2_2b_graph_audit_v1.json"
_REGISTRY_PATH = _ROOT / "configs/experiment_registry.yaml"
_ARTIFACT_PATH = (
    _ROOT / "diagnostics/exactness/e6_carbon_6_slots_a2_2b_graph_audit_v1_2026-07-30.json"
)


@pytest.fixture(scope="module")
def contract():
    return load_a2_2b_contract(_CONTRACT_PATH, registry_path=_REGISTRY_PATH)


@pytest.fixture(scope="module")
def candidate(contract):
    benchmark = contract["benchmark"]
    return Candidate(
        candidate_id=benchmark["candidate_id"],
        seed_smiles=benchmark["seed_smiles"],
        elements=tuple(benchmark["elements"]),
        max_hydrogens=benchmark["max_hydrogens"],
        horizon=benchmark["horizon"],
    )


@pytest.fixture(scope="module")
def full_graph(candidate):
    return build_reachable_graph(
        candidate,
        state_cap=20_001,
        edge_cap=2_000_001,
        deadline_seconds=None,
    )


def _elements(key: str) -> set[str]:
    if key == NULL_KEY:
        return set()
    molecule = Chem.MolFromSmiles(key, sanitize=False)
    assert molecule is not None
    return {atom.GetSymbol() for atom in molecule.GetAtoms()}


def _fresh_raw_candidates(state, source_key, candidate):
    if source_key == NULL_KEY:
        return tuple(
            _factorized_candidates(
                state,
                allow_bond_reroute=False,
            )
        )
    return tuple(_candidate_actions(state, candidate.spec()))


def _fresh_dictionary_oracle(state, source_key, candidate, state_keys):
    """Fresh loops: execute, classify, group, and normalize with no segmented call."""

    system = de_novo_rewrite_system()
    dispositions = Counter()
    productive = defaultdict(int)
    allowed = set(candidate.elements)
    raw_count = 0
    for rule_name, action in _fresh_raw_candidates(state, source_key, candidate):
        raw_count += 1
        try:
            successor = system.apply(state, rule_name, action)
        except InvalidRewrite:
            dispositions[EXCLUDED_EXECUTOR_INVALID] += 1
            continue
        successor_key = canonical_state_key(successor)
        if successor_key == source_key:
            dispositions[VIRTUAL_SELF] += 1
        elif not _elements(successor_key) <= allowed:
            dispositions[EXCLUDED_VOCABULARY_BOUNDARY] += 1
        elif successor_key not in state_keys:
            dispositions[EXCLUDED_OUTSIDE_CLOSED_SLICE] += 1
        else:
            dispositions[INCLUDED_PRODUCTIVE] += 1
            productive[successor_key] += 1

    productive_total = sum(productive.values())
    legal_total = productive_total + dispositions[VIRTUAL_SELF]
    probabilities = {key: count / productive_total for key, count in sorted(productive.items())}
    return {
        "raw_count": raw_count,
        "dispositions": dispositions,
        "aliases": dict(sorted(productive.items())),
        "probabilities": probabilities,
        "raw_productive_mass": productive_total / legal_total,
        "virtual_self_mass": dispositions[VIRTUAL_SELF] / legal_total,
    }


def test_contract_is_self_hashed_registry_bound_and_non_claiming(contract):
    assert contract["contract_sha256"] == contract_self_hash(contract)
    assert contract["paper_claim_authorized"] is False
    assert contract["checkpoint_required"] is False
    assert contract["diagnostic_mark_law"]["is_solver_kernel"] is False


def test_contract_tampering_fails_closed(tmp_path):
    payload = json.loads(_CONTRACT_PATH.read_text())
    payload["benchmark"]["expected_n_edges"] += 1
    path = tmp_path / "tampered.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(E6GraphAuditError, match="self-hash|drifted"):
        load_a2_2b_contract(path, registry_path=_REGISTRY_PATH)


def test_state_index_is_sorted_canonical_and_deterministic(full_graph, contract):
    first = deterministic_state_index(full_graph)
    second = deterministic_state_index(full_graph)
    assert first == second == tuple(sorted(full_graph.states))
    assert len(first) == contract["benchmark"]["expected_n_states"]
    assert first[0] == NULL_KEY
    assert stable_sha256(list(first)) == (
        "4fe916579503e2b4750556e4dff479127a812d33b0df1f4f1dc2db23036d676c"
    )


def test_structural_graph_identity_is_unchanged(full_graph, contract):
    assert full_graph.stop_reason == "closed"
    assert full_graph.n_states == contract["benchmark"]["expected_n_states"]
    assert full_graph.n_edges == contract["benchmark"]["expected_n_edges"]
    assert (
        graph_fingerprint(full_graph)
        == (contract["benchmark"]["expected_structural_graph_fingerprint"])
    )


def test_null_boundary_accounts_for_every_production_root_mark(
    full_graph,
    candidate,
):
    keys = deterministic_state_index(full_graph)
    audited = audit_state_actions(
        full_graph.states[NULL_KEY],
        source_index=0,
        source_key=NULL_KEY,
        candidate=candidate,
        state_index={key: index for index, key in enumerate(keys)},
    )
    counts = Counter(action.disposition for action in audited.actions)
    assert len(audited.actions) == 4
    assert counts[INCLUDED_PRODUCTIVE] == 1
    assert counts[EXCLUDED_VOCABULARY_BOUNDARY] == 3
    assert audited.successor_probabilities == {"C": pytest.approx(1.0)}


def test_exhaustive_segmented_rows_match_fresh_dictionary_oracle(
    full_graph,
    candidate,
    contract,
):
    """All 967 rows: production segmented aggregation versus fresh dict loops."""

    keys = deterministic_state_index(full_graph)
    state_index = {key: index for index, key in enumerate(keys)}
    tolerance = contract["numeric_tolerances"]["production_vs_dictionary_oracle"]
    maximum_residual = 0.0
    for index, key in enumerate(keys):
        state = full_graph.states[key]
        audited = audit_state_actions(
            state,
            source_index=index,
            source_key=key,
            candidate=candidate,
            state_index=state_index,
        )
        reference = _fresh_dictionary_oracle(
            state,
            key,
            candidate,
            set(keys),
        )
        observed_dispositions = Counter(action.disposition for action in audited.actions)
        assert len(audited.actions) == reference["raw_count"], key
        assert observed_dispositions == reference["dispositions"], key
        assert audited.alias_multiplicities == reference["aliases"], key
        assert set(audited.successor_probabilities) == set(reference["probabilities"]), key
        for successor_key, probability in audited.successor_probabilities.items():
            residual = abs(probability - reference["probabilities"][successor_key])
            maximum_residual = max(maximum_residual, residual)
            assert residual <= tolerance, (key, successor_key, residual)
        assert audited.raw_productive_mass == pytest.approx(
            reference["raw_productive_mass"],
            abs=tolerance,
        )
        assert audited.virtual_self_mass == pytest.approx(
            reference["virtual_self_mass"],
            abs=tolerance,
        )
    assert maximum_residual <= tolerance


def test_historical_artifact_is_intact_but_refused_as_current(contract):
    artifact = json.loads(_ARTIFACT_PATH.read_text())
    with pytest.raises(E6GraphAuditError, match="provenance.*current sources"):
        validate_a2_2b_artifact(artifact, contract=contract, registry_path=_REGISTRY_PATH)
    assert artifact["artifact_sha256"] == artifact_self_hash(artifact)
    assert artifact["benchmark"]["structural_graph_fingerprint"] == ("84121ff86cbc1ba8")
    assert len(artifact["graph_audit_hash"]) == 64
    assert artifact["graph_audit_hash"] != (artifact["benchmark"]["structural_graph_fingerprint"])
    assert all(artifact["invariants"].values())
    assert artifact["paper_claim_authorized"] is False
    assert artifact["checkpoint_used"] is None
    assert artifact["control_solver_run"] is False


def test_fresh_graph_audit_validates_without_mocking_provenance(current_e6_inputs):
    root, _, contract, artifact = current_e6_inputs
    validate_a2_2b_artifact(
        artifact,
        contract=contract,
        registry_path=root / "configs/experiment_registry.yaml",
        repo_root=root,
    )
    assert artifact["artifact_sha256"] == artifact_self_hash(artifact)
    assert all(artifact["invariants"].values())


def test_artifact_tampering_fails_closed(contract):
    artifact = json.loads(_ARTIFACT_PATH.read_text())
    tampered = copy.deepcopy(artifact)
    tampered["aggregate"]["raw_action_count"] += 1
    with pytest.raises(E6GraphAuditError, match="self-hash"):
        validate_a2_2b_artifact(
            tampered,
            contract=contract,
            registry_path=_REGISTRY_PATH,
        )


def test_audit_has_no_silent_disposition_or_boundary_loss():
    artifact = json.loads(_ARTIFACT_PATH.read_text())
    aggregate = artifact["aggregate"]
    assert aggregate["raw_action_count"] == sum(aggregate["disposition_counts"].values())
    assert aggregate["disposition_counts"][EXCLUDED_OUTSIDE_CLOSED_SLICE] == 0
    assert aggregate["canonical_directed_edge_count"] == 14_432
    assert aggregate["productive_mark_count"] >= 14_432
    assert aggregate["raw_productive_mark_to_canonical_edge_compression"] == (
        aggregate["productive_mark_count"] - 14_432
    )
    assert aggregate["virtual_self_mark_count"] > 0
    assert aggregate["disposition_counts"][EXCLUDED_EXECUTOR_INVALID] > 0
    assert aggregate["disposition_counts"][EXCLUDED_VOCABULARY_BOUNDARY] == 3


def test_old_exactness_implementations_are_not_reused():
    forbidden = (
        "exact_doob_enumerable_benchmark",
        "doob_guidance_ground_truth",
        "e0_toy_h_exactness",
        "exact_doob_enumerable_cap4",
        "exact_doob_enumerable_cap5",
    )
    for relative in (
        "src/compose_v4/experiments/e6_graph_audit.py",
        "scripts/run_e6_a2_2b_graph_audit.py",
        "configs/exact_control_a2_2b_graph_audit_v1.json",
    ):
        text = (_ROOT / relative).read_text()
        assert not any(item in text for item in forbidden), relative


def test_result_producer_never_imports_a_dictionary_oracle():
    for relative in (
        "src/compose_v4/experiments/e6_graph_audit.py",
        "scripts/run_e6_a2_2b_graph_audit.py",
    ):
        text = (_ROOT / relative).read_text()
        assert "reference_successor_kernel" not in text
        assert "_fresh_dictionary_oracle" not in text
