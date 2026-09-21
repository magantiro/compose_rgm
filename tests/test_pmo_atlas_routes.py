"""Guards for the answer-known PMO teacher-route atlas.

These tests drive the PRODUCTION executor and the PRODUCTION loader; none of
them recomputes an expectation from the code under test.  Each guard has a
matching negative so that a mutation at the corresponding call site turns a
named test red.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from compose_v4.experiments.pmo_atlas_routes import (
    ATLAS_ARTIFACTS,
    DEVELOPMENT_INFORMED_LABEL,
    PMO_MAX_ACTIVE_ATOMS,
    PMO_SLOTS,
    AtlasIntegrityError,
    AtlasRegimeError,
    assert_development_informed,
    assert_not_scored_consumable,
    lineage_census,
    load_atlas,
    payload_sha256,
    replay_route,
    route_checkpoints,
)

REPO_ROOT = Path(__file__).resolve().parents[1]

_ALL_INPUTS_PRESENT = all(
    (REPO_ROOT / artifact.relative_path).is_file() for artifact in ATLAS_ARTIFACTS
)
requires_atlas = pytest.mark.skipif(
    not _ALL_INPUTS_PRESENT, reason="frozen atlas inputs are not vendored here"
)


def _dossier():
    return load_atlas(REPO_ROOT)


# ---- Integrity ----


@requires_atlas
def test_every_frozen_input_verifies():
    dossier = _dossier()
    assert len(dossier.artifact_hashes) == len(ATLAS_ARTIFACTS)
    for artifact in ATLAS_ARTIFACTS:
        assert dossier.artifact_hashes[artifact.relative_path] == artifact.file_sha256


def _mirror(tmp_path: Path) -> Path:
    for artifact in ATLAS_ARTIFACTS:
        source = REPO_ROOT / artifact.relative_path
        target = tmp_path / artifact.relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())
    return tmp_path


@requires_atlas
def test_load_refuses_an_input_whose_bytes_moved(tmp_path):
    """The FILE-hash guard alone must fire.

    The tampered payload is re-hashed so that it is internally consistent.  The
    payload-hash guard therefore cannot catch it, and only the declared
    ``file_sha256`` comparison stands between the loader and drifted content.
    Without this, the two guards share a sink and bypassing either one is
    invisible.
    """

    root = _mirror(tmp_path)
    drifted = root / ATLAS_ARTIFACTS[0].relative_path
    document = json.loads(drifted.read_text())
    document["payload"]["schema_version"] = "tampered"
    document["payload_sha256"] = payload_sha256(document["payload"])
    drifted.write_text(json.dumps(document))

    with pytest.raises(AtlasIntegrityError):
        load_atlas(root)


@requires_atlas
def test_load_refuses_a_payload_that_disagrees_with_its_own_hash(tmp_path):
    """The PAYLOAD-hash guard alone must fire, with the file hash disabled."""

    root = _mirror(tmp_path)
    drifted = root / ATLAS_ARTIFACTS[0].relative_path
    document = json.loads(drifted.read_text())
    document["payload"]["schema_version"] = "tampered"
    drifted.write_text(json.dumps(document))

    with pytest.raises(AtlasIntegrityError):
        load_atlas(root, verify=False)


@requires_atlas
def test_dossier_covers_the_declared_eleven_tasks():
    dossier = _dossier()
    assert len(dossier.tasks) == 11
    assert "perindopril_mpo" in dossier.tasks


# ---- Replay ----


@requires_atlas
def test_every_recorded_route_replays_exactly():
    """The whole dossier, through the production executor, with no exceptions."""

    dossier = _dossier()
    replays = [replay_route(route) for route in dossier.routes]
    failures = [r.program_id for r in replays if not r.exact]
    assert failures == [], failures
    assert len(replays) == len(dossier.routes)


@requires_atlas
def test_replay_reports_a_corrupted_action_as_inexact():
    dossier = _dossier()
    route = dossier.spines()[0]
    actions = list(copy.deepcopy(route.actions))
    # A legal-looking but wrong deletion target: the executor decides.
    actions[0] = {**actions[0], "payload": {"v": PMO_SLOTS - 1}}
    broken = type(route)(**{**route.__dict__, "actions": tuple(actions)})
    replay = replay_route(broken)
    assert not replay.exact


@requires_atlas
def test_replay_detects_an_interior_divergence_not_only_the_endpoint():
    """A route whose endpoint is right but whose interior record is wrong fails.

    This is the guard that distinguishes exact replay from endpoint recovery.
    """

    dossier = _dossier()
    route = dossier.spines()[0]
    states = list(copy.deepcopy(route.recorded_states))
    assert len(states) >= 3
    # Swap two distinct interior records: the endpoint is untouched.
    states[1], states[2] = states[2], states[1]
    tampered = type(route)(**{**route.__dict__, "recorded_states": tuple(states)})
    replay = replay_route(tampered)
    assert replay.endpoint_matches_record
    assert replay.intermediate_key_mismatches
    assert not replay.exact


# ---- Checkpoints ----


@requires_atlas
def test_checkpoints_are_committed_48_slot_states_inside_support():
    dossier = _dossier()
    for route in dossier.spines():
        checkpoints = route_checkpoints(route)
        assert checkpoints
        for checkpoint in checkpoints:
            assert checkpoint.state["n_slots"] == PMO_SLOTS
            assert 1 <= checkpoint.heavy_atoms <= PMO_MAX_ACTIVE_ATOMS
            assert checkpoint.smiles


@requires_atlas
def test_the_anchor_checkpoint_is_the_recorded_endpoint():
    dossier = _dossier()
    for route in dossier.spines():
        anchor = [c for c in route_checkpoints(route) if c.label == "anchor"]
        assert len(anchor) == 1
        assert anchor[0].step_index == route.primitive_steps
        assert anchor[0].remaining_steps == 0
        assert anchor[0].smiles == route.recorded_endpoint_smiles


@requires_atlas
def test_checkpoints_are_ordered_and_strictly_advance():
    dossier = _dossier()
    route = dossier.spines()[0]
    checkpoints = route_checkpoints(route)
    indices = [c.step_index for c in checkpoints]
    assert indices == sorted(set(indices))


def test_checkpoint_fraction_outside_the_unit_interval_is_refused():
    class _Stub:
        actions = ({"executor_rule": "atom_delete"},)
        program_id = "stub"

    with pytest.raises(ValueError):
        route_checkpoints(_Stub(), positions=(("bad", 1.5),))


# ---- Information regime ----


@requires_atlas
def test_every_route_carries_the_development_informed_label():
    dossier = _dossier()
    assert dossier.regime == DEVELOPMENT_INFORMED_LABEL
    assert all(route.regime == DEVELOPMENT_INFORMED_LABEL for route in dossier.routes)


def test_a_payload_without_the_regime_label_is_refused():
    assert_development_informed({"information_regime": DEVELOPMENT_INFORMED_LABEL})
    with pytest.raises(AtlasRegimeError):
        assert_development_informed({})
    with pytest.raises(AtlasRegimeError):
        assert_development_informed({"information_regime": "held_out"})


def test_prohibited_consumers_are_refused_and_diagnostics_are_allowed():
    for consumer in ("scored_no_prescreen_run", "route_prior_fit", "proposal_library"):
        with pytest.raises(AtlasRegimeError):
            assert_not_scored_consumable(consumer)
    assert_not_scored_consumable("atlas_transport_diagnostic")


# ---- Census ----


@requires_atlas
def test_the_census_separates_programs_from_independent_lineages():
    """The dossier's breadth claim rests on lineages and sources, not programs."""

    dossier = _dossier()
    census = lineage_census(dossier)
    assert set(census) == set(dossier.tasks)
    assert sum(entry["programs"] for entry in census.values()) == len(dossier.routes)
    # Every task section is compiled from a single start molecule.
    assert all(entry["distinct_sources"] == 1 for entry in census.values())


@requires_atlas
def test_the_dossier_is_dominated_by_one_shared_source_molecule():
    """Lineage poverty is a property of the data and must stay visible."""

    dossier = _dossier()
    shared = "COc1ccccc1CNS(=O)(=O)c1cc(C(=O)N2CCCCCC2)cs1"
    tasks_on_shared = {r.task for r in dossier.routes if r.source_smiles == shared}
    assert len(tasks_on_shared) == 9
    assert len({r.source_smiles for r in dossier.routes}) == 3
