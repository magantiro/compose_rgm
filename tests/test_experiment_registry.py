"""The frozen registry must load, and every way of weakening the freeze must fail loudly.

The registry is a pre-commitment device, so the tests that matter are the ones proving it cannot be
quietly degraded: a v1 file must not be read as v2, protocol drift must be detected, and an experiment
must not accumulate so many primary metrics that a story can be assembled afterwards.
"""
from __future__ import annotations

import copy
from pathlib import Path

import pytest
import yaml

from compose_v4.experiments.registry import (
    MAX_PRIMARY_METRICS,
    REGISTRY_SCHEMA_VERSION,
    RegistryViolation,
    SEMANTIC_PANEL_IDS,
    experiment,
    load_registry,
    protocol_content_hash,
    verify_protocol_freeze,
)

_REGISTRY_PATH = Path(__file__).resolve().parent.parent / "configs" / "experiment_registry.yaml"


@pytest.fixture()
def registry():
    return load_registry(_REGISTRY_PATH)


def _write(tmp_path: Path, payload: dict) -> Path:
    path = tmp_path / "registry.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=True))
    return path


# ---- the real artifact --------------------------------------------------------------------------------


def test_the_committed_registry_loads_and_validates():
    loaded = load_registry(_REGISTRY_PATH)
    assert loaded["schema_version"] == REGISTRY_SCHEMA_VERSION
    assert set(loaded["experiments"]) == {"E1", "E2", "E3", "E4", "E5", "E6", "E7"}


def test_every_experiment_declares_one_to_three_primary_metrics(registry):
    for key, spec in registry["experiments"].items():
        assert 1 <= len(spec["primary_metrics"]) <= MAX_PRIMARY_METRICS, key


def test_every_panel_id_is_semantic_not_a_figure_number(registry):
    for key, spec in registry["experiments"].items():
        for panel in spec["panel_ids"]:
            assert panel in SEMANTIC_PANEL_IDS, f"{key}: {panel}"
            assert "figure" not in panel.lower(), f"{key} hard-codes a figure number: {panel}"


def test_provenance_names_the_live_run_and_records_the_dead_one(registry):
    editing = registry["provenance"]["editing_prior"]
    assert editing["run_id"] == "compose-v4-ringcore-v1-scientific-a7546e2-v1"
    superseded = editing["supersedes"]
    assert superseded["run_id"] == "compose-v4-ringcore-v1-scientific-nitya-6d067e9-v1"
    # The relaunch fixed infrastructure only; asserting this keeps the claim honest in the artifact.
    assert superseded["scientific_contract_changed"] is False


def test_exact_sizing_bounds_both_states_and_edges(registry):
    """State count alone is not tractability -- a small dense graph can be harder than a larger sparse one."""
    sizing = registry["exact_sizing"]
    assert sizing["N_min"] == 500
    assert sizing["N_max"] == 20000
    assert sizing["E_max"] == 2000000
    assert sizing["selection"] == "smallest_qualifying_candidate"
    required = set(sizing["non_degeneracy_required"])
    assert {"atom_birth", "atom_death", "cycle_change", "multi_path_terminal_event"} <= required


def test_learned_controller_freezes_the_encoder_policy(registry):
    controller = registry["learned_controller"]
    assert controller["encoder_policy"] == "frozen_base_with_trainable_adapter"
    assert "NEVER write back" in controller["secondary_ablation_constraint"]
    # The bootstrapped target must use canonical successors, not a mark-level surrogate.
    bootstrapped = next(
        e for e in controller["target_estimators"] if e["id"] == "bootstrapped_backward"
    )
    assert "CANONICAL" in bootstrapped["definition"]


def test_task_generation_forbids_checkpoint_informed_construction(registry):
    generation = registry["protocol"]["task_generation"]
    assert generation["builder_takes_no_checkpoint_argument"] is True
    assert generation["development_partition"] == "validation"
    assert generation["final_partition"] == "test"


def test_freeze_record_is_provenance_and_states_the_structural_guarantee(registry):
    freeze = registry["protocol"]["protocol_freeze"]
    assert isinstance(freeze["highest_existing_snapshot_step"], int)
    assert "no checkpoint outputs" in freeze["structural_guarantee"].lower()


def test_accessor_rejects_an_unknown_experiment(registry):
    with pytest.raises(RegistryViolation, match="unknown experiment id"):
        experiment(registry, "E99")


def test_accessor_returns_the_requested_experiment(registry):
    assert experiment(registry, "E6")["checkpoint"] == "none"


# ---- ways of weakening the freeze, each must fail ------------------------------------------------------


def test_a_v1_registry_is_rejected_not_partially_read(tmp_path, registry):
    stale = copy.deepcopy(registry)
    stale["schema_version"] = 1
    with pytest.raises(RegistryViolation, match="schema_version"):
        load_registry(_write(tmp_path, stale))


def test_protocol_drift_is_detected(tmp_path, registry):
    """Editing any frozen protocol field must invalidate the recorded content hash."""
    drifted = copy.deepcopy(registry)
    drifted["protocol"]["multi_objective"]["source_similarity_floor"] = 0.1
    with pytest.raises(RegistryViolation, match="DRIFTED"):
        verify_protocol_freeze(drifted)


def test_drift_detection_is_not_vacuous(registry):
    """The recorded hash must actually match the committed protocol, or the check proves nothing."""
    protocol = registry["protocol"]
    assert protocol["protocol_freeze"]["content_hash"] == protocol_content_hash(protocol)


def test_too_many_primary_metrics_is_rejected(tmp_path, registry):
    bloated = copy.deepcopy(registry)
    bloated["experiments"]["E6"]["primary_metrics"] = ["a", "b", "c", "d"]
    bloated["protocol"]["protocol_freeze"]["content_hash"] = protocol_content_hash(
        bloated["protocol"]
    )
    with pytest.raises(RegistryViolation, match="primary metrics"):
        load_registry(_write(tmp_path, bloated))


def test_a_metric_cannot_be_both_primary_and_secondary(tmp_path, registry):
    confused = copy.deepcopy(registry)
    spec = confused["experiments"]["E6"]
    spec["secondary_metrics"] = list(spec["primary_metrics"])
    with pytest.raises(RegistryViolation, match="both primary and secondary"):
        load_registry(_write(tmp_path, confused))


def test_a_missing_protocol_block_is_rejected(tmp_path, registry):
    stripped = copy.deepcopy(registry)
    del stripped["protocol"]["multi_objective"]
    with pytest.raises(RegistryViolation, match="missing required blocks"):
        load_registry(_write(tmp_path, stripped))


def test_incomplete_compute_accounting_is_rejected(tmp_path, registry):
    """Dropping a cost field is how an unfair controller comparison sneaks in."""
    partial = copy.deepcopy(registry)
    partial["protocol"]["compute_accounting"]["required_per_arm"] = ["oracle_calls"]
    partial["protocol"]["protocol_freeze"]["content_hash"] = protocol_content_hash(
        partial["protocol"]
    )
    with pytest.raises(RegistryViolation, match="every cost field"):
        load_registry(_write(tmp_path, partial))


def test_a_figure_number_panel_is_rejected(tmp_path, registry):
    hardcoded = copy.deepcopy(registry)
    hardcoded["experiments"]["E6"]["panel_ids"] = ["Figure 3"]
    with pytest.raises(RegistryViolation, match="unknown panel ids"):
        load_registry(_write(tmp_path, hardcoded))


def test_missing_task_artifacts_are_rejected(tmp_path, registry):
    incomplete = copy.deepcopy(registry)
    incomplete["experiments"]["E3"]["task_artifacts"] = {"final": "configs/tasks/x.final.json"}
    with pytest.raises(RegistryViolation, match="development and final"):
        load_registry(_write(tmp_path, incomplete))


def test_a_missing_registry_file_fails_loudly(tmp_path):
    with pytest.raises(RegistryViolation, match="not found"):
        load_registry(tmp_path / "absent.yaml")


def test_malformed_yaml_fails_loudly(tmp_path):
    path = tmp_path / "registry.yaml"
    path.write_text("experiments: [unclosed\n")
    with pytest.raises(RegistryViolation, match="not valid YAML"):
        load_registry(path)
