"""Guards for TEST C: the blind arm, its blindness, and the evaluator's ladder.

Two properties carry the experiment and are therefore tested at their CALL
SITES rather than at their definitions:

* the optimizer under test never sees an answer-known molecule, and
* a productivity probe runs the SAME campaign geometry Test B ran, so the
  measured top ten is comparable to the teacher ladder rather than merely
  similar in spirit.

The geometry guard compares kwargs captured from the LIVE Test-B driver against
kwargs captured from the LIVE probe driver.  Neither side is transcribed, so a
drift in either one turns this red.
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from compose_v4.experiments import pmo_atlas_discovery as discovery
from compose_v4.experiments.pmo_atlas_routes import load_atlas

REPO_ROOT = Path(__file__).resolve().parents[1]
GEOMETRY_KEYS = (
    "library",
    "rounds",
    "queries_per_round",
    "hierarchy",
    "fit_model",
    "stagnation_rounds",
    "bootstrap_rounds",
    "initialization_mode",
    "initial_parent_fraction",
    "optimizer_type",
    "initial_batch_fn",
)


def _atlas():
    return load_atlas(REPO_ROOT)


def _stub_oracle():
    oracle = types.SimpleNamespace()
    oracle.ledger = lambda: {"distinct_molecules_evaluated": 0}
    return oracle


def _checkpoint(smiles: str, state: dict) -> types.SimpleNamespace:
    return types.SimpleNamespace(
        program_id="probe",
        task="qed",
        label="anchor",
        step_index=1,
        fraction=1.0,
        remaining_steps=0,
        smiles=smiles,
        state=state,
        heavy_atoms=1,
    )


def _row(smiles: str, state: dict | None, index: int = 3, score: float = 0.5):
    return discovery.TrajectoryRow(
        index=index, endpoint=smiles, score=score, role="candidate", state=state
    )


# ---- Blind initialization ----


def test_the_blind_initialization_is_the_pinned_task_independent_bank():
    payload = discovery.load_blind_initialization(REPO_ROOT)
    assert payload["count"] == discovery.BLIND_INITIALIZATION_COUNT
    assert payload["source_sha256"] == discovery.BLIND_INITIALIZATION_SOURCE_SHA256
    assert all("score" not in row and "task" not in row for row in payload["candidates"])


def _tamper(tmp_path, *, relock: bool):
    """Write a modified initialization, optionally with a CONSISTENT lock.

    Relocking matters: an inconsistent tamper is caught by the lock guard AND
    by the file-hash guard, so it cannot show which one is load-bearing.
    """

    from compose_v4.control.docking_value import identity

    source = REPO_ROOT / discovery.BLIND_INITIALIZATION_PATH
    target = tmp_path / discovery.BLIND_INITIALIZATION_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = json.loads(source.read_text())
    payload["candidates"][0]["endpoint"] = "CCO"
    if relock:
        body = {key: value for key, value in payload.items() if key != "lock_sha256"}
        payload["lock_sha256"] = identity(body)
    target.write_text(json.dumps(payload))
    return target


def test_an_initialization_whose_bytes_moved_is_refused(tmp_path):
    # The lock is recomputed, so ONLY the file-hash guard can refuse this.
    _tamper(tmp_path, relock=True)
    with pytest.raises(discovery.OptimizerBlindnessError, match="moved"):
        discovery.load_blind_initialization(tmp_path)


def test_an_initialization_whose_lock_disagrees_with_its_content_is_refused(
    tmp_path, monkeypatch
):
    # The pinned digest is pointed at the tampered bytes, so ONLY the lock guard
    # can refuse this. Without both tests one guard hides behind the other.
    from compose_v4.experiments.pmo_atlas_routes import file_sha256

    target = _tamper(tmp_path, relock=False)
    monkeypatch.setattr(
        discovery, "BLIND_INITIALIZATION_SHA256", file_sha256(target)
    )
    with pytest.raises(discovery.OptimizerBlindnessError, match="lock"):
        discovery.load_blind_initialization(tmp_path)


# ---- Blindness at the call site ----


def test_an_atlas_molecule_in_the_initialization_is_refused():
    dossier = _atlas()
    atlas = discovery.atlas_molecules(dossier)
    answer = min(atlas)
    initialization = {"candidates": [{"endpoint": answer, "source_id": "leak"}]}
    with pytest.raises(discovery.OptimizerBlindnessError, match="atlas molecules"):
        discovery.assert_optimizer_blind(
            initialization=initialization,
            optimizer_kwargs={},
            library=(),
            atlas=atlas,
        )


def test_a_clean_initialization_passes_the_blindness_guard():
    dossier = _atlas()
    atlas = discovery.atlas_molecules(dossier)
    payload = discovery.load_blind_initialization(REPO_ROOT)
    evidence = discovery.assert_optimizer_blind(
        initialization=payload, optimizer_kwargs={}, library=(), atlas=atlas
    )
    assert evidence["leaked"] == []
    assert evidence["initialization_molecules"] > 0


def test_a_program_library_is_refused_in_a_blind_run():
    with pytest.raises(discovery.OptimizerBlindnessError, match="library"):
        discovery.assert_optimizer_blind(
            initialization={"candidates": []},
            optimizer_kwargs={},
            library=({"plan": 1},),
            atlas=frozenset({"CCO"}),
        )


def test_answer_known_optimizer_kwargs_are_refused():
    with pytest.raises(discovery.OptimizerBlindnessError, match="answer-known key"):
        discovery.assert_optimizer_blind(
            initialization={"candidates": []},
            optimizer_kwargs={"destination": "CCO"},
            library=(),
            atlas=frozenset({"CCO"}),
        )


def test_the_blind_run_driver_calls_the_blindness_guard_before_any_charged_call(
    tmp_path, monkeypatch
):
    import pmo_atlas_blind_search as driver

    called: list[str] = []

    def _tripwire(**kwargs):
        called.append("guard")
        raise discovery.OptimizerBlindnessError("tripwire")

    monkeypatch.setattr(driver, "assert_optimizer_blind", _tripwire)
    monkeypatch.setattr(
        driver, "run_program_campaign", lambda **kwargs: pytest.fail("campaign ran unguarded")
    )
    payload = discovery.load_blind_initialization(REPO_ROOT)
    with pytest.raises(discovery.OptimizerBlindnessError, match="tripwire"):
        driver._run_one(
            REPO_ROOT,
            tmp_path / "run",
            "qed",
            payload,
            frozenset(),
            _stub_oracle(),
            16,
            16,
            1,
        )
    assert called == ["guard"]


# ---- Probe seeds ----


def test_a_probe_seed_from_the_atlas_is_refused():
    import pmo_atlas_entry_probe as probe

    atlas = discovery.atlas_molecules(_atlas())
    answer = min(atlas)
    row = _row(answer, {"slots": 48})
    with pytest.raises(probe.ProbeSeedError, match="atlas molecule"):
        probe.assert_probe_seed_is_blind(row, (row,), atlas)


def test_a_probe_seed_outside_the_blind_trajectory_is_refused():
    import pmo_atlas_entry_probe as probe

    row = _row("CCO", {"slots": 48})
    other = _row("CCC", {"slots": 48}, index=4)
    with pytest.raises(probe.ProbeSeedError, match="not in the blind trajectory"):
        probe.assert_probe_seed_is_blind(row, (other,), frozenset())


def test_a_probe_seed_without_a_state_is_refused_rather_than_reparsed():
    import pmo_atlas_entry_probe as probe

    row = _row("CCO", None)
    with pytest.raises(probe.ProbeSeedError, match="48-slot state"):
        probe.assert_probe_seed_is_blind(row, (row,), frozenset())


# ---- Campaign geometry parity with Test B ----


def _capture(module, name="run_program_campaign"):
    seen: dict = {}

    def _record(**kwargs):
        seen.update(kwargs)
        return {"history": []}

    return seen, _record


def test_the_probe_runs_the_same_campaign_geometry_as_the_test_b_driver(tmp_path, monkeypatch):
    import pmo_atlas_entry_probe as probe
    import pmo_atlas_local_lift as test_b

    state = json.loads(
        (REPO_ROOT / "diagnostics/parent_edit_cycles/prepared/init_20260921.json").read_text()
    )["candidates"][0]["state"]
    smiles = "CC1CCC(C)(C(=O)N2CCCC(C)(C(=O)[O-])C2)O1"

    b_seen, b_record = _capture(test_b)
    monkeypatch.setattr(test_b, "run_program_campaign", b_record)
    test_b._run_one(
        REPO_ROOT,
        tmp_path / "b",
        "qed",
        _checkpoint(smiles, state),
        _stub_oracle(),
        64,
        16,
        20260921,
    )

    c_seen, c_record = _capture(probe)
    monkeypatch.setattr(probe, "run_program_campaign", c_record)
    probe._probe(
        REPO_ROOT,
        tmp_path / "c",
        "qed",
        _row(smiles, state),
        ["blind_best_score"],
        _stub_oracle(),
        64,
        16,
        20260921,
    )

    assert b_seen and c_seen
    for key in GEOMETRY_KEYS:
        assert b_seen[key] == c_seen[key], f"campaign geometry drifted on {key!r}"
    assert b_seen["optimizer_kwargs"] == c_seen["optimizer_kwargs"]
    assert b_seen["config"] == c_seen["config"]
    # The seed molecule is the ONLY thing a probe changes.
    assert c_seen["initialization"]["candidates"][0]["endpoint"] == smiles
    assert len(c_seen["initialization"]["candidates"]) == 1


def test_the_shared_geometry_helper_reproduces_the_test_b_driver(tmp_path, monkeypatch):
    import pmo_atlas_local_lift as test_b

    from compose_v4.experiments.pmo_population_v1 import configuration

    state = json.loads(
        (REPO_ROOT / "diagnostics/parent_edit_cycles/prepared/init_20260921.json").read_text()
    )["candidates"][0]["state"]
    b_seen, b_record = _capture(test_b)
    monkeypatch.setattr(test_b, "run_program_campaign", b_record)
    test_b._run_one(
        REPO_ROOT, tmp_path / "b", "qed", _checkpoint("CCO", state), _stub_oracle(), 64, 16, 7
    )
    helper = discovery.local_lift_kwargs(
        repo_root=REPO_ROOT,
        config=configuration(7),
        rounds=b_seen["rounds"],
        queries_per_round=16,
        optimizer_kwargs=b_seen["optimizer_kwargs"],
    )
    for key in GEOMETRY_KEYS:
        assert helper[key] == b_seen[key], f"shared helper drifted on {key!r}"


# ---- Structural diagnostic ----


def test_nearest_approach_reports_the_maximum_not_an_average():
    rows = (
        _row("CCO", None, index=1),
        _row("c1ccccc1", None, index=2),
        _row("c1ccccc1O", None, index=3),
    )
    approach = discovery.nearest_approach(rows, ["c1ccccc1O"])
    assert approach["max_similarity"] == pytest.approx(1.0)
    assert approach["at_charged_call"] == 3
    assert approach["nearest_blind_molecule"] == "c1ccccc1O"
    values = [point["max_similarity"] for point in approach["running_max_curve"]]
    assert values == sorted(values), "the approach curve must be a running maximum"


def test_nearest_approach_refuses_an_empty_reference_set():
    with pytest.raises(ValueError, match="empty reference set"):
        discovery.nearest_approach((_row("CCO", None),), [])


# ---- Ladder ----


def _ladder():
    return (
        discovery.LadderRung("qed", "early", 0.1, 0.20),
        discovery.LadderRung("qed", "near_anchor", 0.5, 0.50),
        discovery.LadderRung("qed", "anchor", 1.0, 0.90),
    )


def test_a_measurement_below_the_lowest_rung_is_not_rounded_up():
    verdict = discovery.classify_rung(0.05, _ladder())
    assert verdict["rung_reached"] == "below_early"
    assert verdict["rungs_reached"] == 0


def test_a_measurement_reaches_only_the_rung_it_clears():
    assert discovery.classify_rung(0.20, _ladder())["rung_reached"] == "early"
    assert discovery.classify_rung(0.89, _ladder())["rung_reached"] == "near_anchor"
    assert discovery.classify_rung(0.95, _ladder())["rung_reached"] == "anchor"


def test_the_ladder_is_loaded_per_task_from_the_test_b_runs():
    payload = json.loads(
        (REPO_ROOT / "diagnostics/pmo_atlas_v1/test_b_local_lift.json").read_text()
    )["payload"]
    ladder = discovery.load_ladder(payload)
    assert set(ladder) == set(payload["tasks_probed"])
    assert all(len(rungs) == 3 for rungs in ladder.values())


# ---- Lift reduction ----


def test_a_run_that_only_preserves_its_seed_scores_no_lift():
    rows = [{"endpoint": "CCO", "score": 0.9}]
    summary = discovery.summarize_lift(rows, "CCO")
    assert summary["top_ten_new_mean"] is None
    assert summary["distinct_new_above_seed"] == 0
    assert summary["best_overall"] == pytest.approx(0.9)


def test_the_top_ten_excludes_the_seed_itself():
    rows = [{"endpoint": "CCO", "score": 1.0}] + [
        {"endpoint": "N" + "C" * n + "O", "score": 0.1} for n in range(1, 12)
    ]
    summary = discovery.summarize_lift(rows, "CCO")
    assert summary["top_ten_new_mean"] == pytest.approx(0.1)
    assert summary["seed_score"] == pytest.approx(1.0)


# ---- Trajectory provenance ----


def test_the_parent_comes_from_the_executed_trace_not_the_label(tmp_path):
    source = json.loads(
        (REPO_ROOT / "diagnostics/parent_edit_cycles/prepared/init_20260921.json").read_text()
    )
    parent_state = source["candidates"][0]["state"]
    child_state = source["candidates"][1]["state"]
    parent_smiles = discovery._state_smiles(parent_state)
    child_smiles = discovery._state_smiles(child_state)

    run = tmp_path / "run"
    (run / "campaign" / "round_0001").mkdir(parents=True)
    (run / "campaign" / "round_0001" / "pending.json").write_text(
        json.dumps(
            {
                "batch": {
                    "candidates": [
                        {
                            "endpoint": child_smiles,
                            "provenance": {
                                "parent_endpoint": "CCO",
                                "parent_measured_score": 0.4,
                                "planner_channel": "shallow_program_channel",
                            },
                            "trace": {"states": [parent_state, child_state], "actions": [{}, {}]},
                        }
                    ]
                }
            }
        )
    )
    (run / "oracle" / "query_000000").mkdir(parents=True)
    (run / "oracle" / "query_000000" / "result.json").write_text(
        json.dumps({"index": 0, "endpoint": child_smiles, "score": 0.7, "status": "complete"})
    )
    rows = discovery.read_blind_trajectory(run)
    assert len(rows) == 1
    assert rows[0].parent_endpoint == parent_smiles
    assert rows[0].parent_label_disagrees is True
    assert rows[0].program_primitives == 2


# ---- Route reconstruction ----


def _chain_row(index, endpoint, parent, score=0.5, role="candidate"):
    return discovery.TrajectoryRow(
        index=index,
        endpoint=endpoint,
        score=score,
        role=role,
        parent_endpoint=parent,
        state={"slots": 48},
    )


def test_ancestry_walks_back_to_an_initialization_molecule():
    rows = (
        _chain_row(1, "A", None, role="initialization"),
        _chain_row(2, "B", "A"),
        _chain_row(3, "C", "B"),
    )
    chain = discovery.ancestry(rows, "C")
    assert [step["endpoint"] for step in chain] == ["A", "B", "C"]
    assert chain[0]["role"] == "initialization"


def test_ancestry_stops_at_a_parent_outside_the_trajectory_rather_than_inventing_one():
    rows = (_chain_row(2, "B", "GHOST"),)
    chain = discovery.ancestry(rows, "B")
    assert chain[0]["note"].startswith("parent is outside")
    assert chain[-1]["endpoint"] == "B"


def test_ancestry_terminates_on_a_cycle():
    rows = (_chain_row(1, "A", "B"), _chain_row(2, "B", "A"))
    chain = discovery.ancestry(rows, "B")
    assert len(chain) == 2
