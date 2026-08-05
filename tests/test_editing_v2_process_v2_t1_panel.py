"""The bounded Process-V2 bridge from Active8/Gate 0 to T1 inputs."""

from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from compose_v4.data.editing_process_v2_rebind import (
    mounted_process_v2_artifact_path,
)
from compose_v4.data.editing_v2_process_v2_active8_plan import (
    write_process_v2_active8_plan,
)
from compose_v4.data.editing_v2_process_v2_active8_reduce import (
    COMPLETION_FILENAME,
    validate_process_v2_active8_completion,
)
from compose_v4.data.editing_v2_process_v2_gate_zero import run_gate_zero
from compose_v4.data.editing_v2_process_v2_gate_zero import (
    resolve_gate_zero_eligible_stream,
)
from compose_v4.data.editing_v2_process_v2_schema import canonical_bytes, canonical_sha256
from compose_v4.experiments.editing_v2_process_v2_contract_chain import (
    T1_PANEL_POLICY,
    load_process_v2_chain_artifact,
)
from compose_v4.experiments.editing_v2_process_v2_t1_panel import (
    PANEL_FILENAME,
    ProcessV2T1PanelError,
    ProcessV2T1Source,
    _validate_gate_zero_pass,
    build_process_v2_t1_panel,
    iter_process_v2_t1_candidates,
    load_process_v2_t1_panel,
    open_process_v2_t1_source,
    resolve_process_v2_t1_entries,
    validate_process_v2_t1_panel,
    write_process_v2_t1_panel,
)

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "tests") not in sys.path:
    sys.path.insert(0, str(ROOT / "tests"))

from test_editing_v2_process_v2_active8_pipeline import _Stage  # noqa: E402
from test_editing_v2_process_v2_gate_zero import (  # noqa: E402
    _build_run as _build_standin_run,
    _contracts as _standin_contracts,
    _digest as _standin_digest,
    _identity as _standin_identity,
)


def _stage_evidence(tmp_path: Path):
    stage = _Stage(tmp_path / "artifacts")
    write_process_v2_active8_plan(
        stage.plan, artifact_root=stage.artifact_root, repo_root=ROOT
    )
    stage.run()
    stage.reduce()
    active8_root = mounted_process_v2_artifact_path(
        str(stage.plan["run_artifact_root"]),
        artifact_root=stage.artifact_root,
        field="plan.run_artifact_root",
    )
    gate_zero_root = tmp_path / "gate-zero"
    decision = run_gate_zero(active8_root, gate_zero_root=gate_zero_root, repo_root=ROOT)
    # This production-shaped cache fixture covers only atom deletion and cycle
    # closing.  It is useful for exact-address testing but correctly fails the
    # eight-family Gate 0; it must never be used as launch authority.
    assert decision["decision"] == "FAIL"
    contracts, index = resolve_gate_zero_eligible_stream(active8_root, repo_root=ROOT)
    completion = validate_process_v2_active8_completion(
        json.loads((active8_root / COMPLETION_FILENAME).read_bytes())
    )
    policy = load_process_v2_chain_artifact(T1_PANEL_POLICY, repo_root=ROOT)
    source = ProcessV2T1Source(
        active8_run_root=active8_root,
        artifact_root=stage.artifact_root,
        repo_root=ROOT,
        contracts=contracts,
        index=index,
        decision=decision,
        decision_file_sha256=canonical_sha256(decision),
        completion=completion,
        plan=stage.plan,
        policy=policy,
        policy_file_sha256=canonical_sha256("fixture-policy-file"),
    )
    return stage, source, gate_zero_root / "DECISION.json"


def _unique_standin_rows(rows):
    rewritten = []
    for index, row in enumerate(rows):
        body = {key: value for key, value in row.items() if key != "assignment_sha256"}
        evidence = dict(body["candidate_evidence"])
        token = f"{row['task_identity_sha256']}:{row['trace_id']}:{index}"
        evidence["source_state_sha256"] = _standin_identity(f"source:{token}")
        evidence["target_state_sha256"] = _standin_identity(f"target:{token}")
        evidence["source_canonical_key"] = f"SOURCE-{token}"
        evidence["canonical_successor_key"] = f"TARGET-{token}"
        body["candidate_evidence"] = evidence
        rewritten.append({**body, "assignment_sha256": _standin_digest(body)})
    return rewritten


def _pass_source(tmp_path: Path):
    contracts = _standin_contracts()
    active8_root = tmp_path / "active8"
    _build_standin_run(active8_root, contracts=contracts, mutate=_unique_standin_rows)
    gate_zero_root = tmp_path / "gate-zero"
    decision = run_gate_zero(active8_root, gate_zero_root=gate_zero_root, repo_root=ROOT)
    assert decision["decision"] == "PASS"
    bound_contracts, index = resolve_gate_zero_eligible_stream(active8_root, repo_root=ROOT)
    decision_path = gate_zero_root / "DECISION.json"
    loaded, decision_file_sha256 = _validate_gate_zero_pass(
        decision_path, contracts=bound_contracts, index=index
    )
    policy = load_process_v2_chain_artifact(T1_PANEL_POLICY, repo_root=ROOT)
    completion = validate_process_v2_active8_completion(
        json.loads((active8_root / COMPLETION_FILENAME).read_bytes())
    )
    # Selection and provenance do not open a molecular chunk.  These stand-in
    # plan fields exercise only that bounded half; exact resolution uses the
    # independent production-shaped fixture above.
    plan = {
        "plan_sha256": _standin_identity("standin-plan"),
        "run_identity_sha256": _standin_identity("standin-run"),
        "binding_sha256": _standin_identity("standin-binding"),
        "tasks": [],
    }
    return ProcessV2T1Source(
        active8_run_root=active8_root,
        artifact_root=tmp_path,
        repo_root=ROOT,
        contracts=bound_contracts,
        index=index,
        decision=loaded,
        decision_file_sha256=decision_file_sha256,
        completion=completion,
        plan=plan,
        policy=policy,
        policy_file_sha256=canonical_sha256("standin-policy-file"),
    )


@pytest.fixture(scope="module")
def stage_evidence(tmp_path_factory):
    return _stage_evidence(tmp_path_factory.mktemp("process-v2-t1"))


@pytest.fixture()
def pass_source(tmp_path: Path):
    return _pass_source(tmp_path)


def test_the_authenticated_stream_opens_only_train_decision_shards(
    pass_source, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = pass_source
    import compose_v4.data.editing_v2_process_v2_gate_zero as gate_zero

    original = gate_zero._read_decision_shard
    opened: list[Path] = []

    def recording(path: Path):
        opened.append(Path(path))
        return original(path)

    monkeypatch.setattr(gate_zero, "_read_decision_shard", recording)
    candidates = tuple(iter_process_v2_t1_candidates(source))

    assert len(candidates) == source.decision["accounting"]["transitions"]
    assert candidates
    assert {candidate["partition_role"] for candidate in candidates} == {"train"}
    eligible = set(source.index.eligible_task_identities)
    assert {path.parent.name for path in opened} == eligible
    assert not eligible.intersection(
        str(shard["task_identity_sha256"])
        for shard in source.index.shards
        if shard["partition_role"] in source.contracts.sealed_roles
    )


def test_a_resealed_gate_zero_fail_cannot_feed_t1(
    stage_evidence, tmp_path: Path
) -> None:
    _stage, source, _decision_path = stage_evidence
    body = {
        key: value for key, value in source.decision.items() if key != "decision_sha256"
    }
    body["decision"] = "FAIL"
    failed = {**body, "decision_sha256": canonical_sha256(body)}
    failed_path = tmp_path / "FAILED_DECISION.json"
    failed_path.write_bytes(canonical_bytes(failed) + b"\n")

    with pytest.raises(ProcessV2T1PanelError, match="Gate-0 PASS"):
        open_process_v2_t1_source(
            source.active8_run_root,
            gate_zero_decision_path=failed_path,
            artifact_root=source.artifact_root,
            repo_root=ROOT,
        )


def test_selected_address_reopens_the_exact_persistent_slot_states(stage_evidence) -> None:
    _stage, source, _decision_path = stage_evidence
    candidate = next(iter(iter_process_v2_t1_candidates(source)))
    resolved = resolve_process_v2_t1_entries(source, (candidate,))

    assert len(resolved) == 1
    assert resolved[0].panel_entries == (candidate,)
    assert resolved[0].addressed_trace.address.trace_id == candidate["trace_id"]


def test_small_genuine_fixture_exercises_panel_provenance_without_relaxing_production(
    pass_source, tmp_path: Path
) -> None:
    """Test mechanics under fixture cardinality; the real public source stays frozen.

    The module fixture deliberately has one discovered example per capability
    cell, not the production minimum of 64 per family.  A copied, explicitly
    test-only source lowers only this test's cardinality and receives a distinct
    policy identity.  The next assertion proves the untouched production source
    still refuses.
    """

    source = pass_source
    with pytest.raises(ProcessV2T1PanelError, match="frozen minimum"):
        build_process_v2_t1_panel(source, scratch_dir=tmp_path)

    required_by_family: dict[str, int] = {}
    for cell in source.contracts.required_cell_ids:
        _namespace, family, _context = cell.split(":", 2)
        required_by_family[family] = required_by_family.get(family, 0) + 1
    policy = dict(source.policy)
    policy["minimum_entries_by_family"] = dict.fromkeys(
        source.contracts.active_families, 1
    )
    policy["maximum_entries_by_family"] = required_by_family
    policy["contract_sha256"] = canonical_sha256(
        {key: value for key, value in policy.items() if key != "contract_sha256"}
    )
    fixture_source = replace(
        source,
        policy=policy,
        policy_file_sha256=canonical_sha256("fixture-cardinality-policy"),
    )
    panel = build_process_v2_t1_panel(fixture_source, scratch_dir=tmp_path)
    validated = validate_process_v2_t1_panel(panel, source=fixture_source)
    path = write_process_v2_t1_panel(
        validated, output_root=tmp_path / "panel", source=fixture_source
    )

    assert path.name == PANEL_FILENAME
    assert json.loads(path.read_bytes()) == validated
    assert load_process_v2_t1_panel(path, source=fixture_source) == validated
    assert set(validated["family_counts"]) == set(source.contracts.active_families)
    assert set(source.contracts.required_cell_ids).issubset(
        validated["capability_cell_counts"]
    )


def test_a_panel_entry_with_a_stale_exact_state_hash_is_refused(stage_evidence) -> None:
    _stage, source, _decision_path = stage_evidence
    candidate = next(iter(iter_process_v2_t1_candidates(source)))
    changed = {**candidate, "source_state_sha256": "f" * 64}

    with pytest.raises(ProcessV2T1PanelError, match="exact cached states"):
        resolve_process_v2_t1_entries(source, (changed,))
