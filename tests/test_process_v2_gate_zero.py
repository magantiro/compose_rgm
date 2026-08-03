"""Process-V2 Gate 0, driven through a structural stand-in for the decision index.

The stand-in satisfies :class:`ProcessV2Active8Index` and inherits nothing, which
is the whole point of expressing the seam as a protocol: Gate 0 can be built and
proven before the concrete index exists, and the concrete index cannot acquire a
V1 base class by satisfying it.

What these tests are careful about.  Every fixture value that must agree with a
frozen policy is DERIVED from that policy -- the capability cells and their role
partition come from the loaded Process-V2 contract, the executor-to-family alias
comes from ``action_codec_v4.canonical_family``, the strata come from the
contract's own bins -- so no expectation here is a transcription that could
silently drift.  And every derived value has a paired negative case in which the
fixture deliberately disagrees with the policy, because a check whose input is
built to satisfy it cannot fail on its own.
"""

from __future__ import annotations

import ast
import json
import shutil
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

import pytest

from compose_v4.data.editing_corpus_contract import ACTIVE8_FAMILIES
from compose_v4.data.editing_v2_process_v2_active8_interfaces import (
    ACTIVE8_CENSUS_FIELDS,
    ACTIVE8_EXCLUDED,
    UPSTREAM_REJECTED,
    ProcessV2Active8Index,
    ProcessV2TraceKey,
)
from compose_v4.data.editing_v2_process_v2_schema import (
    AUTHORITY_FIELDS,
    ProcessV2SchemaError,
    require_authority_false,
    require_no_granted_authority,
)
from compose_v4.experiments import editing_v2_process_v2_gate_zero as gate_zero
from compose_v4.rewrite.action_codec_v4 import ACTIVE8_EXECUTOR_RULES, canonical_family
from compose_v4.rewrite.editing_v2_process_identity import editing_process_v2_identity

REPO_ROOT = Path(__file__).resolve().parents[1]

#: family -> the one executor rule that aliases to it, read from the frozen codec.
_EXECUTOR_FOR_FAMILY = {canonical_family(rule): rule for rule in ACTIVE8_EXECUTOR_RULES}

_V1_DECISION_SOURCE = "compose_v4.data.editing_v2_semantic_active8_decision_source"


def _sha(seed: str) -> str:
    """A stable 64-hex identity for a fixture object."""

    import hashlib

    return hashlib.sha256(seed.encode()).hexdigest()


# ---- The structural stand-in ----


class StandInIndex:
    """A decision index keyed by the full trace key, inheriting nothing."""

    def __init__(
        self,
        rows: list[dict[str, Any]],
        transitions: Mapping[ProcessV2TraceKey, list[dict[str, Any]]],
        *,
        process_identity_sha256: str | None = None,
        refuse: frozenset[str] = frozenset(),
    ) -> None:
        self._rows = list(rows)
        self._transitions = {key: list(value) for key, value in transitions.items()}
        self._refuse = refuse
        self.requested_keys: list[ProcessV2TraceKey] = []
        self.process_identity_sha256 = process_identity_sha256 or str(
            editing_process_v2_identity()["process_identity_sha256"]
        )
        self.completion_sha256 = _sha("stand-in-completion")

    def counts(self) -> Mapping[str, int]:
        accepted = sum(1 for row in self._rows if row["rejection_category"] is None)
        upstream = sum(1 for row in self._rows if row["rejection_category"] == UPSTREAM_REJECTED)
        excluded = sum(1 for row in self._rows if row["rejection_category"] == ACTIVE8_EXCLUDED)
        return {
            "source_entries": len(self._rows),
            "upstream_rejected_entries": upstream,
            "active8_accepted_entries": accepted,
            "active8_excluded_entries": excluded,
        }

    def rejected_traces_by_code(self) -> Mapping[str, int]:
        codes: dict[str, int] = {}
        for row in self._rows:
            category = row["rejection_category"]
            if category is not None:
                codes[category] = codes.get(category, 0) + 1
        return codes

    def identity(self) -> Mapping[str, Any]:
        return {"schema": "test.stand_in_index", "schema_version": 1}

    def iter_resolved_traces(self) -> Iterator[Mapping[str, Any]]:
        yield from self._rows

    def accepted_transitions_for(
        self, trace_key: ProcessV2TraceKey
    ) -> Iterator[Mapping[str, Any]]:
        self.requested_keys.append(trace_key)
        yield from self._transitions.get(trace_key, ())

    def validate_accepted_transition(self, transition: Mapping[str, Any]) -> None:
        if transition["assignment_sha256"] in self._refuse:
            raise ValueError("this transition is not what the index published")


# ---- Fixture construction, driven by the frozen contract ----


@pytest.fixture(scope="module", name="contract")
def _contract() -> gate_zero.FrozenProcessV2GateZeroContract:
    return gate_zero.load_process_v2_gate_zero_contract(repo_root=REPO_ROOT)


def _already_a_view(transition, **_kwargs):
    """The stand-in publishes structural views, so the adapter is the identity.

    Deliberate and narrow: the real adapter needs exact persistent-slot states
    and a real ActionV4 record, which a synthetic cell-by-cell fixture cannot
    fabricate without rebuilding the chain it exists to avoid. These tests drive
    the DECISION logic; the adapter is proven by the end-to-end chain, and by
    the guard below that the production default is the real one.
    """

    return transition


def test_the_default_transition_view_is_the_real_adapter() -> None:
    """Without this, substituting the view could quietly become the normal path."""

    import inspect

    default = inspect.signature(gate_zero.build_process_v2_gate_zero_evidence).parameters[
        "view"
    ].default
    assert default is gate_zero._structural_view


def _transition(
    contract: gate_zero.FrozenProcessV2GateZeroContract,
    cell_id: str,
    *,
    progress_index: int = 0,
    **overrides: Any,
) -> dict[str, Any]:
    _namespace, family, context = cell_id.split(":")
    raw, successors, aliases = 4, 2, 1
    transition = {
        "assignment_sha256": _sha(f"{cell_id}|{progress_index}"),
        "action_sha256": _sha(f"action|{cell_id}|{progress_index}"),
        "canonical_successor_count": successors,
        "canonical_successor_count_stratum": gate_zero.stratum_for(
            successors, contract.strata_bins["canonical_successor_count"]
        ),
        "capability_cell_id": cell_id,
        "data_lane": contract.data_lanes[0],
        "executor_rule": _EXECUTOR_FOR_FAMILY[family],
        "family_context": context,
        "matching_mark_count": 1,
        "model_family": family,
        "partition_role": "train",
        "progress_index": progress_index,
        "raw_mark_count": raw,
        "raw_mark_count_stratum": gate_zero.stratum_for(
            raw, contract.strata_bins["raw_mark_count"]
        ),
        "source_state_sha256": _sha(f"source|{cell_id}"),
        "successor_alias_multiplicity": aliases,
        "successor_alias_multiplicity_stratum": gate_zero.stratum_for(
            aliases, contract.strata_bins["successor_alias_multiplicity"]
        ),
        "target_state_sha256": _sha(f"target|{cell_id}"),
        "terminal": False,
    }
    transition.update(overrides)
    return transition


def _row(
    seed: str, *, role: str, category: str | None = None, transitions: int = 0
) -> dict[str, Any]:
    return {
        "v1_task_identity_sha256": _sha(f"task|{seed}"),
        "entry_index": 7,
        "trace_id": f"trace-{seed}",
        "partition_role": role,
        "rejection_category": category,
        "decision_sha256": _sha(f"decision|{seed}"),
        "accepted_transition_count": transitions,
    }


def _complete_fixture(
    contract: gate_zero.FrozenProcessV2GateZeroContract,
    *,
    omit_cells: frozenset[str] = frozenset(),
    sealed_seeds: tuple[str, ...] | None = None,
    override: Mapping[str, Mapping[str, Any]] | None = None,
    refuse: frozenset[str] = frozenset(),
) -> StandInIndex:
    """One train trace per registered capability cell, plus sealed and rejected rows."""

    rows: list[dict[str, Any]] = []
    transitions: dict[ProcessV2TraceKey, list[dict[str, Any]]] = {}
    for cell_id in contract.registered_cell_ids:
        if cell_id in omit_cells:
            continue
        row = _row(cell_id, role="train", transitions=1)
        rows.append(row)
        key = (row["v1_task_identity_sha256"], row["entry_index"], row["trace_id"])
        transitions[key] = [
            _transition(contract, cell_id, **dict((override or {}).get(cell_id, {})))
        ]
    seeds = sealed_seeds if sealed_seeds is not None else contract.sealed_roles
    for seed, role in zip(seeds, contract.sealed_roles, strict=True):
        rows.append(_row(seed, role=role, transitions=0))
    rows.append(_row("upstream", role="train", category=UPSTREAM_REJECTED))
    rows.append(_row("excluded", role="train", category=ACTIVE8_EXCLUDED))
    return StandInIndex(rows, transitions, refuse=refuse)


# ---- Acceptance 11: eight families, every required cell ----


def test_gate_zero_sees_all_eight_families_and_every_required_cell(contract) -> None:
    """Including ``atom_delete:connected_nonleaf_death``, the Process-V2 addition."""

    index = _complete_fixture(contract)
    evidence = gate_zero.build_process_v2_gate_zero_evidence(index, view=_already_a_view, contract=contract)

    assert evidence["active8_families"] == list(ACTIVE8_FAMILIES)
    families = evidence["decision_eligible_teacher_counts_by_family"]
    assert set(families) == set(ACTIVE8_FAMILIES)
    assert all(count > 0 for count in families.values())
    executors = evidence["decision_eligible_teacher_counts_by_executor_rule"]
    assert set(executors) == set(ACTIVE8_EXECUTOR_RULES)
    assert all(count > 0 for count in executors.values())

    rows = {row["capability_cell_id"]: row for row in evidence["capability_cell_counts"]}
    connected_nonleaf = f"{contract.namespace}:atom_delete:connected_nonleaf_death"
    assert connected_nonleaf in contract.required_cell_ids
    assert rows[connected_nonleaf]["teacher_count"] > 0
    assert evidence["empty_required_editing_cell_ids"] == []
    assert evidence["counts"]["observed_required_editing_cells"] == 17
    assert evidence["counts"]["registered_capability_cells"] == 22
    assert evidence["structural_result"] == "PASS"


def test_an_empty_connected_nonleaf_death_cell_fails_the_gate(contract) -> None:
    """The negative control for the test above: the cell is load-bearing."""

    connected_nonleaf = f"{contract.namespace}:atom_delete:connected_nonleaf_death"
    index = _complete_fixture(contract, omit_cells=frozenset({connected_nonleaf}))
    evidence = gate_zero.build_process_v2_gate_zero_evidence(index, view=_already_a_view, contract=contract)

    assert evidence["empty_required_editing_cell_ids"] == [connected_nonleaf]
    assert (
        evidence["checks"]["all_required_editing_cells_have_decision_eligible_teachers"]
        is False
    )
    assert evidence["structural_result"] == "FAIL"
    # atom_delete still has its other required context, so the FAIL is the CELL's,
    # not the family's -- a family-level check alone would have missed it.
    assert evidence["decision_eligible_teacher_counts_by_family"]["atom_delete"] > 0
    assert evidence["checks"]["all_active8_families_have_decision_eligible_teachers"] is True


def test_a_missing_family_fails_even_when_every_other_cell_is_present(contract) -> None:
    dropped = frozenset(
        cell for cell in contract.registered_cell_ids if ":ring_system_restate:" in cell
    )
    index = _complete_fixture(contract, omit_cells=dropped)
    evidence = gate_zero.build_process_v2_gate_zero_evidence(index, view=_already_a_view, contract=contract)

    assert evidence["decision_eligible_teacher_counts_by_family"]["ring_system_restate"] == 0
    assert evidence["decision_eligible_teacher_counts_by_executor_rule"][
        "ring_system_restate"
    ] == 0
    assert evidence["checks"]["all_active8_families_have_decision_eligible_teachers"] is False
    assert evidence["structural_result"] == "FAIL"


# ---- Acceptance 12: sealed nontraining roles ----


def test_sealed_roles_cannot_move_a_gate_zero_threshold_or_result(contract) -> None:
    """More validation traces must change the seal and nothing else."""

    lean = _complete_fixture(contract)
    rich = _complete_fixture(contract)
    for extra, role in enumerate(contract.sealed_roles):
        rich._rows.append(_row(f"extra-{extra}", role=role, transitions=0))

    left = gate_zero.build_process_v2_gate_zero_evidence(lean, view=_already_a_view, contract=contract)
    right = gate_zero.build_process_v2_gate_zero_evidence(rich, view=_already_a_view, contract=contract)

    assert left["structural_result"] == right["structural_result"] == "PASS"
    assert left["checks"] == right["checks"]
    assert left["capability_cell_counts"] == right["capability_cell_counts"]
    assert (
        left["decision_eligible_teacher_counts_by_family"]
        == right["decision_eligible_teacher_counts_by_family"]
    )
    assert left["decision_eligible_teacher_counts_by_lane"] == (
        right["decision_eligible_teacher_counts_by_lane"]
    )
    decision_counts = [
        key
        for key in left["counts"]
        if key.startswith(("decision_eligible", "structural", "unique"))
    ]
    assert decision_counts
    assert all(left["counts"][key] == right["counts"][key] for key in decision_counts)
    assert left["structural_assignment_inventory_sha256"] == (
        right["structural_assignment_inventory_sha256"]
    )

    # The seals moved, which is how we know the extra traces were observed at all
    # rather than skipped: a seal that never changes proves nothing.
    assert left["sealed_nondecision_role_inventory_sha256"] != (
        right["sealed_nondecision_role_inventory_sha256"]
    )
    assert set(left["sealed_nondecision_role_inventory_sha256"]) == set(contract.sealed_roles)


def test_gate_zero_never_opens_a_sealed_trace(contract) -> None:
    """The mechanism behind the test above, asserted directly on the index."""

    index = _complete_fixture(contract)
    gate_zero.build_process_v2_gate_zero_evidence(index, view=_already_a_view, contract=contract)

    sealed_ids = {_sha(f"task|{role}") for role in contract.sealed_roles}
    assert len(sealed_ids) == len(contract.sealed_roles)
    assert index.requested_keys
    assert not [key for key in index.requested_keys if key[0] in sealed_ids]
    assert len(index.requested_keys) == 22


def test_a_trace_in_an_undeclared_partition_role_is_refused(contract) -> None:
    index = _complete_fixture(contract)
    index._rows.append(_row("rogue", role="pretraining", transitions=0))
    with pytest.raises(gate_zero.ProcessV2GateZeroError, match="undeclared role"):
        gate_zero.build_process_v2_gate_zero_evidence(index, view=_already_a_view, contract=contract)


# ---- Acceptance 13: PASS and FAIL are both nonauthorizing ----


@pytest.mark.parametrize("expected", ["PASS", "FAIL"])
def test_both_results_publish_completely_nonauthorizing_artifacts(
    contract, tmp_path: Path, expected: str
) -> None:
    omit = (
        frozenset()
        if expected == "PASS"
        else frozenset({f"{contract.namespace}:atom_delete:connected_nonleaf_death"})
    )
    index = _complete_fixture(contract, omit_cells=omit)
    output = tmp_path / "run"
    published = gate_zero.run_process_v2_gate_zero(
        index,
        view=_already_a_view,
        repo_root=REPO_ROOT,
        artifact_root=tmp_path,
        output_directory=output,
        contract=contract,
    )
    assert published["evidence"]["structural_result"] == expected
    assert published["decision"]["structural_result"] == expected
    assert published["completion"]["structural_result"] == expected

    for name, payload in published.items():
        require_no_granted_authority(payload, label=name)
        require_authority_false(payload, label=name)
        assert all(payload[field] is False for field in AUTHORITY_FIELDS)
    assert published["decision"]["next_authorized_stage"] is None
    assert "authoriz" in published["decision"]["required_next_action"] or (
        published["decision"]["required_next_action"].startswith("repair_")
    )
    # A FAIL is published, not withheld.
    for filename in (
        gate_zero.EVIDENCE_FILENAME,
        gate_zero.DECISION_FILENAME,
        gate_zero.COMPLETION_FILENAME,
    ):
        assert (output / filename).is_file()

    reloaded = gate_zero.load_process_v2_gate_zero_artifacts(
        output_directory=output, index=index, contract=contract, view=_already_a_view
    )
    assert reloaded["evidence"] == published["evidence"]


def test_an_index_whose_identity_grants_authority_is_refused(contract) -> None:
    """The one path by which a grant can reach the evidence: the index descriptor.

    ``decision_index_identity`` is the only block Gate 0 copies verbatim from an
    object it does not own, so the depth-first authority guard on the finished
    evidence is load-bearing rather than decorative. Removing that guard leaves
    this the only failing test.
    """

    class _Granting(StandInIndex):
        def identity(self) -> Mapping[str, Any]:
            return {"schema": "test.stand_in_index", "grants": {"t1_authorized": True}}

    complete = _complete_fixture(contract)
    index = _Granting(complete._rows, complete._transitions)
    with pytest.raises(gate_zero.ProcessV2GateZeroError, match="grants authority"):
        gate_zero.build_process_v2_gate_zero_evidence(index, view=_already_a_view, contract=contract)


def test_a_granted_authority_field_anywhere_in_the_evidence_is_refused() -> None:
    """The guard is applied at every depth, so a nested grant is refused too."""

    payload = {**dict.fromkeys(AUTHORITY_FIELDS, False), "nested": {"t1_authorized": True}}
    with pytest.raises(ProcessV2SchemaError):
        require_no_granted_authority(payload, label="fixture")


def test_a_tampered_published_artifact_is_refused(contract, tmp_path: Path) -> None:
    index = _complete_fixture(contract)
    output = tmp_path / "run"
    gate_zero.run_process_v2_gate_zero(
        index,
        view=_already_a_view,
        repo_root=REPO_ROOT,
        artifact_root=tmp_path,
        output_directory=output,
        contract=contract,
    )
    target = output / gate_zero.EVIDENCE_FILENAME
    payload = json.loads(target.read_text())
    payload["counts"]["decision_eligible_transitions"] += 1
    target.write_bytes(gate_zero._bytes(payload, newline=True))
    with pytest.raises(gate_zero.ProcessV2GateZeroError, match="differs from the recomputed"):
        gate_zero.load_process_v2_gate_zero_artifacts(
            view=_already_a_view,
            output_directory=output, index=index, contract=contract
        )


def test_the_output_directory_must_lie_inside_the_artifact_root(
    contract, tmp_path: Path
) -> None:
    index = _complete_fixture(contract)
    with pytest.raises(gate_zero.ProcessV2GateZeroError, match="outside artifact_root"):
        gate_zero.run_process_v2_gate_zero(
            index,
            view=_already_a_view,
            repo_root=REPO_ROOT,
            artifact_root=tmp_path / "inside",
            output_directory=tmp_path / "elsewhere",
            contract=contract,
        )


# ---- Per-teacher structural requirements ----


@pytest.mark.parametrize(
    ("overrides", "requirement"),
    [
        ({"raw_mark_count": 0, "raw_mark_count_stratum": None}, "has_positive_raw_mark_count"),
        (
            {"canonical_successor_count": 0, "canonical_successor_count_stratum": None},
            "has_positive_canonical_successor_count",
        ),
        ({"matching_mark_count": 2}, "matches_exactly_one_action_v4_mark"),
        ({"successor_alias_multiplicity": 9}, "has_exact_successor_alias_aggregation"),
        ({"raw_mark_count_stratum": "marks_065_plus"}, "declares_the_frozen_evidence_strata"),
        ({"executor_rule": "cycle_open"}, "declares_the_frozen_executor_family_alias"),
    ],
)
def test_a_teacher_violating_a_frozen_requirement_fails_the_gate(
    contract, overrides: dict[str, Any], requirement: str
) -> None:
    """Each frozen per-teacher requirement, proven by a teacher that breaks it."""

    cell_id = f"{contract.namespace}:atom_insert:one_neighbor_birth"
    index = _complete_fixture(contract, override={cell_id: overrides})
    evidence = gate_zero.build_process_v2_gate_zero_evidence(index, view=_already_a_view, contract=contract)

    assert evidence["teacher_requirement_failure_counts"][requirement] == 1
    assert evidence["checks"][f"every_decision_eligible_teacher_{requirement}"] is False
    assert evidence["structural_result"] == "FAIL"
    receipts = [
        receipt
        for receipt in evidence["classification_failure_receipts"]
        if receipt["failure_type"] == f"teacher_{requirement}"
    ]
    assert len(receipts) == 1
    assert receipts[0]["capability_cell_id"] == cell_id
    # A defective teacher is still counted, so its cell does not silently vanish
    # from the census and the defect is visible as a defect rather than a gap.
    rows = {row["capability_cell_id"]: row for row in evidence["capability_cell_counts"]}
    assert rows[cell_id]["teacher_count"] == 1


def test_every_frozen_requirement_has_a_published_check(contract) -> None:
    """Without this a requirement could be declared and never evaluated."""

    index = _complete_fixture(contract)
    evidence = gate_zero.build_process_v2_gate_zero_evidence(index, view=_already_a_view, contract=contract)
    for requirement in gate_zero.TEACHER_REQUIREMENTS:
        assert f"every_decision_eligible_teacher_{requirement}" in evidence["checks"]
    assert set(evidence["teacher_requirement_failure_counts"]) == set(
        gate_zero.TEACHER_REQUIREMENTS
    )


def test_a_transition_the_index_refuses_becomes_a_typed_receipt(contract) -> None:
    cell_id = f"{contract.namespace}:bond_reroute:single_atom_pendant_acyclic_source"
    refused = _sha(f"{cell_id}|0")
    index = _complete_fixture(contract, refuse=frozenset({refused}))
    evidence = gate_zero.build_process_v2_gate_zero_evidence(index, view=_already_a_view, contract=contract)

    assert evidence["classification_failure_count"] == 1
    assert evidence["checks"]["classification_failure_count_is_zero"] is False
    assert evidence["structural_result"] == "FAIL"
    receipt = evidence["classification_failure_receipts"][0]
    assert receipt["failure_type"] == "index_refused_its_own_transition"
    assert receipt["capability_cell_id"] == cell_id
    # Refused transitions stay in the transition census and out of the assignments.
    counts = evidence["counts"]
    assert counts["structural_assignments"] + counts["classification_failures"] == (
        counts["decision_eligible_transitions"]
    )


def test_a_cell_outside_the_frozen_registry_becomes_a_typed_receipt(contract) -> None:
    cell_id = f"{contract.namespace}:atom_insert:one_neighbor_birth"
    index = _complete_fixture(
        contract, override={cell_id: {"capability_cell_id": "not:a:cell"}}
    )
    evidence = gate_zero.build_process_v2_gate_zero_evidence(index, view=_already_a_view, contract=contract)
    assert evidence["classification_failure_count"] == 1
    assert evidence["classification_failure_receipts"][0]["failure_type"] == (
        "capability_cell_outside_the_frozen_registry"
    )
    assert evidence["structural_result"] == "FAIL"


def test_receipts_are_bounded_by_the_frozen_limit(contract) -> None:
    """The contract's limit is honoured while the COUNT stays complete."""

    override = {cell: {"matching_mark_count": 3} for cell in contract.registered_cell_ids}
    index = _complete_fixture(contract, override=override)
    evidence = gate_zero.build_process_v2_gate_zero_evidence(index, view=_already_a_view, contract=contract)
    assert evidence["teacher_requirement_failure_counts"][
        "matches_exactly_one_action_v4_mark"
    ] == 22
    assert len(evidence["classification_failure_receipts"]) <= contract.failure_receipt_limit


# ---- Stream and census invariants ----


def test_a_declared_transition_count_that_disagrees_is_refused(contract) -> None:
    index = _complete_fixture(contract)
    index._rows[0]["accepted_transition_count"] = 2
    with pytest.raises(gate_zero.ProcessV2GateZeroError, match="declares 2 accepted"):
        gate_zero.build_process_v2_gate_zero_evidence(index, view=_already_a_view, contract=contract)


def test_a_repeated_trace_key_is_refused(contract) -> None:
    index = _complete_fixture(contract)
    index._rows.append(dict(index._rows[0]))
    with pytest.raises(gate_zero.ProcessV2GateZeroError, match="repeats the key"):
        gate_zero.build_process_v2_gate_zero_evidence(index, view=_already_a_view, contract=contract)


def test_a_rejected_trace_declaring_transitions_is_refused(contract) -> None:
    index = _complete_fixture(contract)
    for row in index._rows:
        if row["rejection_category"] == UPSTREAM_REJECTED:
            row["accepted_transition_count"] = 1
    with pytest.raises(gate_zero.ProcessV2GateZeroError, match="absence of"):
        gate_zero.build_process_v2_gate_zero_evidence(index, view=_already_a_view, contract=contract)


def test_the_two_rejection_categories_stay_separate_in_the_census(contract) -> None:
    """An unevaluated trace and a rejected one are different facts."""

    index = _complete_fixture(contract)
    evidence = gate_zero.build_process_v2_gate_zero_evidence(index, view=_already_a_view, contract=contract)
    assert evidence["counts"]["upstream_rejected_traces"] == 1
    assert evidence["counts"]["active8_excluded_traces"] == 1
    assert evidence["rejected_traces_by_code"] == {UPSTREAM_REJECTED: 1, ACTIVE8_EXCLUDED: 1}
    assert set(evidence["active8_census"]) == set(ACTIVE8_CENSUS_FIELDS)
    assert evidence["checks"]["active8_census_identity_reconciles"] is True


def test_a_census_that_does_not_reconcile_fails_rather_than_raises(contract) -> None:
    class _Skewed(StandInIndex):
        def counts(self) -> Mapping[str, int]:
            base = dict(super().counts())
            base["source_entries"] += 5
            return base

    complete = _complete_fixture(contract)
    index = _Skewed(complete._rows, complete._transitions)
    evidence = gate_zero.build_process_v2_gate_zero_evidence(index, view=_already_a_view, contract=contract)
    assert evidence["checks"]["active8_census_identity_reconciles"] is False
    assert evidence["checks"]["resolved_trace_census_matches"] is False
    assert evidence["structural_result"] == "FAIL"


def test_a_terminal_transition_is_counted_and_fails_the_gate(contract) -> None:
    cell_id = f"{contract.namespace}:atom_restate:element_identity_change"
    index = _complete_fixture(contract, override={cell_id: {"terminal": True}})
    evidence = gate_zero.build_process_v2_gate_zero_evidence(index, view=_already_a_view, contract=contract)
    assert evidence["counts"]["terminal_assignments"] == 1
    assert evidence["checks"]["terminal_assignment_count_is_zero"] is False
    assert evidence["structural_result"] == "FAIL"


def test_an_index_process_identity_that_is_not_the_live_v2_one_fails(contract) -> None:
    complete = _complete_fixture(contract)
    index = StandInIndex(
        complete._rows, complete._transitions, process_identity_sha256="0" * 64
    )
    evidence = gate_zero.build_process_v2_gate_zero_evidence(index, view=_already_a_view, contract=contract)
    assert evidence["checks"]["process_identity_matches_live_process_v2"] is False
    assert evidence["structural_result"] == "FAIL"


def test_an_object_that_does_not_satisfy_the_protocol_is_refused(contract) -> None:
    class _NotAnIndex:
        pass

    with pytest.raises(gate_zero.ProcessV2GateZeroError, match="ProcessV2Active8Index"):
        gate_zero.build_process_v2_gate_zero_evidence(_NotAnIndex(), contract=contract)


def test_the_stand_in_satisfies_the_protocol_without_inheriting_it(contract) -> None:
    index = _complete_fixture(contract)
    assert isinstance(index, ProcessV2Active8Index)
    assert [cls.__name__ for cls in type(index).__mro__] == ["StandInIndex", "object"]


# ---- Contract binding ----


def test_the_contract_binds_the_live_process_v2_identity_and_its_typed_parents(
    contract,
) -> None:
    live = str(editing_process_v2_identity()["process_identity_sha256"])
    assert contract.process_identity_sha256 == live
    assert contract.payload["process_identity"]["process_identity_sha256"] == live
    assert contract.payload["schema_version"] == 3
    assert contract.payload["contract_id"] == gate_zero.CONTRACT_ID
    # The parents are loaded objects, not restated hashes.
    assert contract.capability_cells["contract_id"] == "editing_v2_process_v2_capability_cells"
    assert contract.development_cell_roles["contract_id"] == (
        "editing_v2_process_v2_development_cell_roles"
    )
    assert contract.decision_runtime["contract_id"] == (
        "editing_v2_process_v2_active8_decision_runtime"
    )
    assert contract.semantic_model_process["schema"].endswith(".process_v2")
    assert len(contract.registered_cell_ids) == 22


def test_the_contract_loader_refuses_the_v1_gate_zero_contract_in_place(
    tmp_path: Path,
) -> None:
    """Substituting the V1 body at the V2 path must not load as Process V2.

    Mutated on a copy: the V2 path is overwritten with the frozen V1 contract's
    own bytes, which is the substitution a reader who greps by filename would
    never notice.
    """

    root = tmp_path / "checkout"
    root.mkdir()
    (root / "src").symlink_to(REPO_ROOT / "src")
    shutil.copytree(REPO_ROOT / "configs", root / "configs")
    v1_contract = REPO_ROOT / "configs/editing_v2_semantic_gate_zero_structural_v1.json"
    (root / gate_zero.CONTRACT_RELATIVE_PATH).write_bytes(v1_contract.read_bytes())

    with pytest.raises(gate_zero.ProcessV2GateZeroError):
        gate_zero.load_process_v2_gate_zero_contract(repo_root=root)

    # Control: the untouched copy still loads, so the refusal above is the
    # substitution and not the copy.
    shutil.rmtree(root / "configs")
    shutil.copytree(REPO_ROOT / "configs", root / "configs")
    assert gate_zero.load_process_v2_gate_zero_contract(repo_root=root).sha256


def test_stratum_for_returns_none_outside_every_frozen_bin(contract) -> None:
    bins = contract.strata_bins["successor_alias_multiplicity"]
    assert gate_zero.stratum_for(1, bins) == "aliases_001"
    assert gate_zero.stratum_for(3, bins) == "aliases_002_004"
    assert gate_zero.stratum_for(500, bins) == "aliases_005_plus"
    assert gate_zero.stratum_for(0, bins) is None


# ---- The import boundary this runner exists to hold ----


def _import_edges(path: Path) -> set[str]:
    edges: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(), filename=str(path))):
        if isinstance(node, ast.Import):
            edges.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            edges.add(node.module)
    return edges


def _transitive_compose_modules(entry: Path) -> set[str]:
    """Every ``compose_v4`` module statically reachable from ``entry``."""

    seen: set[str] = set()
    stack = [entry]
    while stack:
        path = stack.pop()
        for module in _import_edges(path):
            if not module.startswith("compose_v4.") or module in seen:
                continue
            seen.add(module)
            candidate = REPO_ROOT / "src" / Path(*module.split(".")).with_suffix(".py")
            if candidate.is_file():
                stack.append(candidate)
    return seen


def test_the_runner_does_not_import_the_v1_concrete_decision_source() -> None:
    """Directly or transitively: a V1 loader refuses a V2 payload by construction."""

    entry = Path(gate_zero.__file__)
    assert _V1_DECISION_SOURCE not in _import_edges(entry)
    reachable = _transitive_compose_modules(entry)
    assert reachable, "the static import scan found nothing, so it proves nothing"
    assert _V1_DECISION_SOURCE not in reachable
    assert "compose_v4.data.editing_v2_semantic_active8_source_adapter" not in reachable
    assert "compose_v4.experiments.editing_v2_semantic_gate_zero" not in reachable
    # The scan can see the V1 module when it is really there: negative control.
    v1_entry = REPO_ROOT / "src/compose_v4/experiments/editing_v2_semantic_gate_zero.py"
    assert _V1_DECISION_SOURCE in _import_edges(v1_entry)


def test_the_runner_has_no_function_local_compose_imports() -> None:
    """A lazy import would defeat the static closure the test above relies on."""

    entry = Path(gate_zero.__file__)
    tree = ast.parse(entry.read_text(), filename=str(entry))
    nested = [
        node
        for parent in ast.walk(tree)
        if isinstance(parent, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        for node in ast.walk(parent)
        if isinstance(node, (ast.Import, ast.ImportFrom))
    ]
    assert not [
        node
        for node in nested
        if any(
            (alias.name if isinstance(node, ast.Import) else (node.module or "")).startswith(
                "compose_v4"
            )
            for alias in node.names
        )
    ]


# ---- The pure calculations this runner shares with the V1 runner ----


def test_the_shared_pure_calculations_agree_with_the_v1_runner(tmp_path: Path) -> None:
    """A differential pin against the production V1 functions, not a restatement.

    These are the calculations the brief calls reusable verbatim.  They are not
    imported from the V1 module -- that module imports the V1 concrete decision
    source -- so this test drives BOTH implementations and requires equality,
    which is what keeps the two from drifting apart silently.
    """

    from compose_v4.experiments import editing_v2_semantic_gate_zero as v1

    sample = {"b": [1, {"a": None}], "a": "é"}
    assert gate_zero._bytes(sample) == v1._canonical_bytes(sample)
    assert gate_zero._bytes(sample, newline=True) == v1._canonical_bytes(sample, newline=True)
    assert gate_zero._sha(sample) == v1._sha(sample)

    target = tmp_path / "bytes.bin"
    target.write_bytes(b"process-v2")
    assert gate_zero._file_sha(target) == v1._file_sha(target)

    v1_row = v1._empty_cell_row("ns:atom_delete:leaf_death", "atom_delete", "leaf_death")
    v2_row = gate_zero._empty_cell_row(
        "ns:atom_delete:leaf_death", "atom_delete", "leaf_death"
    )
    # The V2 row adds the executor-rule histogram the V1 row has no need for; every
    # field the V1 row declares must still be present and identical.
    assert set(v1_row) <= set(v2_row)
    assert {key: v2_row[key] for key in v1_row} == v1_row
    assert set(v2_row) - set(v1_row) == {"executor_rule_counts"}


def test_the_decision_rule_matches_the_v1_runner_on_both_outcomes() -> None:
    from compose_v4.experiments import editing_v2_semantic_gate_zero as v1

    for result, expected_prefix in (("PASS", "record_separate"), ("FAIL", "repair_")):
        evidence = {
            "structural_result": result,
            "evidence_sha256": _sha(result),
            "contract_sha256": _sha("contract"),
            "decision_index_completion_sha256": _sha("completion"),
            "decision_source_inventory_sha256": _sha("completion"),
        }
        v1_decision = v1._structural_decision_from_verified_evidence(evidence)
        v2_decision = gate_zero.process_v2_gate_zero_decision(evidence)
        assert v1_decision["structural_result"] == v2_decision["structural_result"] == result
        assert v1_decision["next_authorized_stage"] is v2_decision["next_authorized_stage"] is None
        assert v2_decision["required_next_action"].startswith(expected_prefix)
        assert v1_decision["required_next_action"].startswith(expected_prefix)
