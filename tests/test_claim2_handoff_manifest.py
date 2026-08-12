"""``handoff.json`` must be true, not merely present.

Handoff acceptance gate 2 is "handoff.json validates" and gate 3 is "all
frozen-input hashes match the main branch".  A manifest whose hashes were typed
rather than computed passes neither, and a wrong digest is worse than a missing
one because it looks verified.  So every digit in the manifest is recomputed
here from the file it describes.

This test caught a fabricated digest tail during development.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
HANDOFF = REPO / "docs" / "workstreams" / "claim2-trajectory" / "handoff.json"


@pytest.fixture(scope="module")
def manifest() -> dict:
    return json.loads(HANDOFF.read_text())


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_every_created_artifact_hash_is_the_real_file_hash(manifest):
    for entry in manifest["artifacts_created"]:
        path = REPO / entry["path"]
        assert path.exists(), f"{entry['path']} is declared but missing"
        assert file_sha256(path) == entry["sha256"], f"{entry['path']} digest is wrong"


def test_every_frozen_input_hash_is_the_real_file_hash(manifest):
    for entry in manifest["frozen_inputs"]:
        if not entry.get("path"):
            continue
        path = REPO / entry["path"]
        assert path.exists(), f"{entry['path']} is declared but missing"
        assert file_sha256(path) == entry["sha256"], f"{entry['path']} digest is wrong"


def test_inner_content_digests_match_their_artifacts(manifest):
    """The envelope and panel carry their own content hash; both must agree."""
    for entry in manifest["artifacts_created"]:
        payload = None
        for field in ("envelope_sha256", "panel_sha256"):
            if field in entry:
                payload = payload or json.loads((REPO / entry["path"]).read_text())
                assert payload[field] == entry[field], f"{entry['path']}:{field} disagrees"


def test_declared_frozen_sampling_law_matches_the_files_own_claim(manifest):
    entry = next(
        e for e in manifest["frozen_inputs"] if e["object"] == "empirical_family_sampling_law"
    )
    law = json.loads((REPO / entry["path"]).read_text())
    assert law["frozen_sha256"] == entry["declared_frozen_sha256"]


def test_declared_r_theta_identity_matches_the_decision_record(manifest):
    entry = next(
        e for e in manifest["frozen_inputs"] if e["object"] == "r_theta_decision_record"
    )
    record = json.loads((REPO / entry["path"]).read_text())
    assert record["identity_sha256"] == entry["identity_sha256"]
    assert record["selected_model_state_sha256"] == entry["selected_model_state_sha256"]
    assert record["selected_step"] == entry["selected_step"]
    assert record["run"] == entry["run"]


def test_declared_process_identity_matches_the_capability_cell_registry(manifest):
    entry = next(
        e for e in manifest["frozen_inputs"] if e["object"] == "process_v2_chemistry"
    )
    cells = json.loads((REPO / entry["path"]).read_text())
    assert cells["process_identity"]["process_identity_sha256"] == entry["process_identity_sha256"]


def test_held_out_status_is_explicit_and_consistent(manifest):
    """Gate 4: held-out-open status must be explicit and must not contradict itself."""
    assert manifest["held_out_opened"] is False
    assert manifest["held_out_gate"]["reserve_source_list_materialized"] is False
    assert manifest["modal_runs_launched"] == 0
    assert manifest["gpu_used"] is False
    assert manifest["results"]["trajectories_generated"] == 0
    for entry in manifest["artifacts_created"]:
        assert entry["status"] in {"DESIGN_ONLY", "SMOKE_HELD_IN"}


def test_no_confirmatory_panel_file_exists_on_this_branch():
    """The strongest possible check that the reserve was not opened."""
    assert not (REPO / "diagnostics" / "claim2_trajectory_confirmatory_panel.json").exists()


def test_manifest_declares_no_existing_file_was_modified(manifest):
    assert manifest["existing_files_modified"] == []


def test_every_gate_verdict_is_one_of_the_allowed_values(manifest):
    for gate in manifest["gate_verdicts"]:
        assert gate["verdict"] in {"PASS", "FAIL", "INCONCLUSIVE"}
        if gate["verdict"] == "INCONCLUSIVE":
            assert gate.get("resolved_by"), f"{gate['gate']} must say what resolves it"


def test_smoke_plan_is_costed_and_not_authorized(manifest):
    plan = manifest["smoke_plan"]
    assert plan["authorized"] is False
    assert plan["gpu"] == "none"
    assert plan["expected_container_hours"] > 0
    assert plan["worst_case_container_hours"] >= plan["expected_container_hours"]
    assert plan["what_it_cannot_resolve"]
    # The stated worst case must be the arithmetic the app enforces.
    expected = 1 + 3 * plan["seeds"] * (plan["horizon"] - 1) + 1
    assert plan["worst_case_enumerations_per_source"] == expected
    assert (
        plan["worst_case_total_enumerations"]
        == expected * plan["sources"]
    )


def test_smoke_command_matches_the_apps_actual_entrypoint(manifest):
    """A command in a handoff that does not run is worse than no command."""
    app = (REPO / "modal_apps" / "claim2_trajectory_characterization_app.py").read_text()
    command = manifest["smoke_plan"]["command"]
    assert "modal_apps/claim2_trajectory_characterization_app.py" in command
    for flag in ("--sources", "--seeds", "--horizon", "--kernel-budget"):
        assert flag in command
        parameter = flag.lstrip("-").replace("-", "_")
        assert f"{parameter}:" in app, f"entrypoint has no {parameter} parameter"


def test_budget_in_the_smoke_command_covers_the_declared_worst_case(manifest):
    plan = manifest["smoke_plan"]
    tokens = plan["command"].split()
    budget = int(tokens[tokens.index("--kernel-budget") + 1])
    assert budget >= plan["worst_case_enumerations_per_source"]
