"""The macro-option mechanism is CONSUMED by the production path, not merely present.

A validated repair behind a flag is inert until a caller passes it, and "the module has
tests" hides that completely.  These drive the real `propose_batch` / `selection` /
`_allocate` on a real controller and require the production path to reach both
protection hooks.

Synthetic scorer only; zero benchmark oracle calls.
"""

import json
import shutil
from pathlib import Path

import pytest

from compose_v4.control.docking_value import identity
from compose_v4.control.dynamic_program_synthesis_v21 import (
    initial_dynamic_program_batch_v21,
)
from compose_v4.control.pmo_macro_option import MacroOption, MacroOptionStage, option_identity
from compose_v4.control.pmo_macro_option_controller import (
    MACRO_OPTION_CHANNEL_TAG,
    MacroOptionController,
    assert_macro_option_protection_is_consumed,
)
from compose_v4.control.program_campaign import ProgramQueryLedger, run_program_campaign
from compose_v4.control.program_task import ProgramTask
from compose_v4.experiments.pmo_macro_option_arms import (
    ValleySimilarityScorer,
    assert_production_batch_geometry,
    configuration,
    load_initialization,
    load_jump_checkpoint,
)

ROOT = Path(__file__).resolve().parents[1]
SEED = 20260922
# Production charges exactly what the controller allocates; see the guard.
QUERIES_PER_ROUND = configuration(SEED).candidates_per_batch


def _bootstrap(tmp_path, *, count=2, budget=56, rounds=6):
    """Real campaign rounds, so a bridge is CHARGED and a window really opens.

    Faking the charge with a bare `note_charged` would leave the bridge outside the
    archive, and then no continuation could be materialized from it -- the tests would
    pass the consumption probe and prove nothing about the leg ever being offered.

    `queries_per_round` MUST equal `candidates_per_batch`, which is what production does.
    An earlier version of this fixture charged 4 of the 8 the controller allocates, and
    `select_parent_edits` then dropped the reserved leg after `_allocate` had chosen it --
    so no bridge was ever charged and every downstream test failed for a reason that had
    nothing to do with the mechanism. The guard makes that unreproducible.
    """
    assert_production_batch_geometry(configuration(SEED), QUERIES_PER_ROUND)
    initialized = load_initialization(ROOT, count=count, seed=SEED)
    from rdkit import Chem

    origin = Chem.MolFromSmiles(initialized["candidates"][0]["endpoint"])
    scorer = ValleySimilarityScorer(origin_heavy_atoms=origin.GetNumHeavyAtoms())
    task = ProgramTask("synthetic_valley_similarity", scorer.protocol, "pmo")
    ledger = ProgramQueryLedger(tmp_path / "oracle", task, scorer, budget=budget)
    kwargs = {
        "jump_checkpoint": load_jump_checkpoint(ROOT),
        "enable_online_memory": False,
        "enable_macro_options": True,
        "macro_option_protection": True,
        "macro_option_settings": {
            "max_options": 1,
            "synthesis_attempts_per_origin": 3,
            "declaration_wall_seconds": 90.0,
        },
    }
    campaign = run_program_campaign(
        output=tmp_path / "campaign",
        task=task,
        config=configuration(SEED),
        initialization=initialized,
        library=(),
        ledger=ledger,
        rounds=rounds,
        queries_per_round=QUERIES_PER_ROUND,
        hierarchy=None,
        fit_model=None,
        stagnation_rounds=None,
        bootstrap_rounds=1,
        initialization_mode="all_scored_pool",
        initial_parent_fraction=0.2,
        optimizer_type=MacroOptionController,
        optimizer_kwargs=kwargs,
        initial_batch_fn=initial_dynamic_program_batch_v21,
    )
    return campaign, kwargs, scorer, task


def _eligibility(task):
    return task.endpoint_evaluator()


def _restore_with_open_window(folder, kwargs, **overrides):
    """Restore the published round whose registry holds a genuinely open window.

    The bridge is in the archive because the campaign charged it, so a continuation can
    actually be materialized from it. If no published round ever holds an open window
    that is a finding about the mechanism, not a fixture inconvenience, so it fails loud.
    """
    for path in sorted((folder / "campaign").glob("round_*/complete.json")):
        snapshot = json.loads(path.read_text())["snapshot"]
        state = snapshot.get("macro_options") or {}
        records = (state.get("registry") or {}).get("records") or []
        open_ids = [
            row["option"]["option_id"]
            for row in records
            if row["status"] == "bridge_charged"
        ]
        if not open_ids:
            continue
        controller = MacroOptionController.restore(snapshot, **{**kwargs, **overrides})
        if controller.option_registry.protected_bridges():
            option_id = min(controller.option_registry.protected_bridges().values())
            return controller, option_id, controller.option_registry.options[option_id]
    raise AssertionError(
        "no published round held an open protection window: the mechanism never charged "
        "a declared bridge, so nothing downstream can be tested"
    )


@pytest.fixture(scope="module")
def bootstrapped(tmp_path_factory):
    folder = tmp_path_factory.mktemp("macro_wiring")
    yield (folder, *_bootstrap(folder))
    shutil.rmtree(folder, ignore_errors=True)


# ---- Consumption ----


def test_the_production_path_reaches_both_protection_hooks(bootstrapped):
    folder, _campaign, kwargs, _, task = bootstrapped
    controller, option_id, option = _restore_with_open_window(folder, kwargs)
    # The bridge is in the archive because the campaign charged it, so the floor has
    # something real to lift.
    assert controller._archive_entry_for(option.bridge_endpoint) is not None
    eligibility = _eligibility(task)
    snapshot = json.loads(json.dumps(controller.snapshot()))
    # A fresh controller per probe: the check calls its closure once for each hook, so a
    # closure that leaves a pending batch behind would refuse on the second call.
    consumed = assert_macro_option_protection_is_consumed(
        lambda: MacroOptionController.restore(snapshot, **kwargs).propose_batch(eligibility)
    )
    assert consumed["consumed"] == {"parent_mass_floor": True, "reserved_slot": True}
    assert option_id in controller.option_registry.options
    assert option.protects_a_single_crossing


def test_the_consumption_check_fails_when_protection_is_off(bootstrapped):
    """The check must be capable of reporting NOT consumed, or a pass proves nothing.

    The unprotected arm is built by flipping the flag in the published snapshot and
    re-deriving its identity, NOT by resuming a protected run as an unprotected one --
    `restore` refuses that outright, and it is right to, because it is the silent arm
    change the whole matched comparison exists to prevent.

    The flipped snapshot is one a genuinely unprotected run could have produced. Both
    arms declare the same options, reserve stage 0 identically and open the window on the
    same `note_charged`; protection changes only `selection` and the CONTINUATION
    reservation, both of which come after the state this snapshot records.
    """
    folder, _campaign, kwargs, _, task = bootstrapped
    protected, _, _ = _restore_with_open_window(folder, kwargs)
    snapshot = json.loads(json.dumps(protected.snapshot()))
    snapshot["macro_options"]["protection_enabled"] = False
    snapshot["snapshot_id"] = identity(
        {k: v for k, v in snapshot.items() if k != "snapshot_id"}
    )
    controller = MacroOptionController.restore(
        snapshot, **{**kwargs, "macro_option_protection": False}
    )
    assert controller.option_registry.protected_bridges(), (
        "the negative control must hold the same open window as the positive one, or it "
        "reports NOT consumed for the wrong reason"
    )
    unprotected = {**kwargs, "macro_option_protection": False}
    with pytest.raises(ValueError, match="NOT consumed"):
        assert_macro_option_protection_is_consumed(
            lambda: MacroOptionController.restore(snapshot, **unprotected).propose_batch(
                _eligibility(task)
            )
        )


def test_nothing_runs_at_all_when_macro_options_are_disabled(bootstrapped):
    _folder, campaign, kwargs, _, task = bootstrapped
    plain = json.loads(json.dumps(campaign["snapshot"]))
    plain.pop("macro_options")
    plain["snapshot_id"] = identity(
        {k: v for k, v in plain.items() if k != "snapshot_id"}
    )
    controller = MacroOptionController.restore(
        plain, **{**kwargs, "enable_macro_options": False}
    )
    before = controller.option_rng.bit_generator.state
    batch = controller.propose_batch(_eligibility(task))
    assert not controller.option_registry.options
    assert not controller.options_declared
    # ABSENT is the only byte-identical OFF: the dedicated stream must be untouched, so
    # the disabled controller is its parent exactly.
    assert controller.option_rng.bit_generator.state == before
    tags = {
        row["provenance"].get("entry_channel")
        for row in batch["eligible_pool"]["candidates"]
    }
    assert MACRO_OPTION_CHANNEL_TAG not in tags


# ---- Provenance for the downstream ablation ----


def test_every_offered_leg_carries_its_channel_tag_at_synthesis_time(bootstrapped):
    folder, _campaign, kwargs, _, task = bootstrapped
    controller, option_id, option = _restore_with_open_window(folder, kwargs)
    batch = controller.propose_batch(_eligibility(task))
    legs = [
        row
        for row in batch["eligible_pool"]["candidates"]
        if row["provenance"].get("entry_channel") == MACRO_OPTION_CHANNEL_TAG
    ]
    assert legs, "a protected option offered no leg into the pool"
    for leg in legs:
        provenance = leg["provenance"]
        assert provenance["macro_option_id"] in controller.option_registry.options
        assert isinstance(provenance["macro_option_stage"], int)
        assert isinstance(provenance["macro_option_protected"], bool)
        assert provenance["macro_option_total_primitives"] > option.realization_ceiling
    assert any(leg["provenance"]["macro_option_id"] == option_id for leg in legs)


def test_a_protected_leg_is_selected_for_the_oracle_not_merely_offered(bootstrapped):
    folder, _campaign, kwargs, _, task = bootstrapped
    controller, _, _ = _restore_with_open_window(folder, kwargs)
    batch = controller.propose_batch(_eligibility(task))
    selected = [
        row
        for row in batch["candidates"]
        if row["provenance"].get("entry_channel") == MACRO_OPTION_CHANNEL_TAG
    ]
    reservation = batch["allocation"]["macro_option_reservation"]
    assert selected, "a reserved continuation was offered and then dropped"
    assert reservation["protection_enabled"] is True
    assert reservation["reserved_added"] + reservation["reserved_already_chosen"] >= 1


# ---- Resume ----


def test_a_resume_refuses_to_change_the_arm(bootstrapped):
    folder, _campaign, kwargs, _, _ = bootstrapped
    controller, _, _ = _restore_with_open_window(folder, kwargs)
    snapshot = controller.snapshot()
    with pytest.raises(ValueError, match="arm flags changed"):
        MacroOptionController.restore(
            snapshot, **{**kwargs, "macro_option_protection": False}
        )


def test_a_resume_carries_the_open_window_so_the_bridge_is_not_stranded(bootstrapped):
    folder, _campaign, kwargs, _, _ = bootstrapped
    controller, option_id, option = _restore_with_open_window(folder, kwargs)
    snapshot = json.loads(json.dumps(controller.snapshot()))
    resumed = MacroOptionController.restore(snapshot, **kwargs)
    assert resumed.option_registry.protected_bridges() == {
        option.bridge_endpoint: option_id
    }
    assert resumed.options_declared is True
    assert resumed.option_registry.construction(option_id)["stages"]
    assert resumed.option_rng.bit_generator.state == controller.option_rng.bit_generator.state


def test_a_snapshot_with_options_cannot_resume_into_a_controller_without_them(bootstrapped):
    folder, _campaign, kwargs, _, _ = bootstrapped
    controller, _, _ = _restore_with_open_window(folder, kwargs)
    snapshot = controller.snapshot()
    with pytest.raises(ValueError, match="arm flags changed"):
        MacroOptionController.restore(
            snapshot, **{**kwargs, "enable_macro_options": False}
        )


# ---- Declaration refuses what needs no staging ----


def test_a_declared_option_always_exceeds_the_realization_ceiling(bootstrapped):
    _folder, campaign, _kwargs, _, _ = bootstrapped
    report = campaign["snapshot"]["macro_options"]["diagnostics"]["declaration"]
    ceiling = report["realization_ceiling"]
    assert report["options"], "the campaign declared no macro option"
    for option in report["options"]:
        assert option["total_primitives"] > ceiling
        assert option["stages"] and 2 <= len(option["stages"]) <= 4
        assert option["protection_source"] == "chosen_ordering_trough_width"


def test_a_stored_leg_that_no_longer_reproduces_its_endpoint_is_refused(bootstrapped):
    # Admissibility is a hard filter: a leg whose executed endpoint moved is never
    # offered under the declared option's name.
    folder, _campaign, kwargs, _, task = bootstrapped
    controller, option_id, _ = _restore_with_open_window(folder, kwargs)
    stored = controller.option_registry.construction(option_id)["stages"][1]
    stored["endpoint"] = "C"
    with pytest.raises(ValueError, match="no longer reproduces its endpoint"):
        controller._materialize_stage(option_id, 1, _eligibility(task))


def test_a_macro_inside_the_ceiling_cannot_be_declared():
    with pytest.raises(ValueError, match="realization ceiling"):
        MacroOption(
            option_id=option_identity("CCO", ["CCN", "CCCN"]),
            origin_endpoint="CCO",
            stages=(
                MacroOptionStage(0, "CCO", "CCN", 3),
                MacroOptionStage(1, "CCN", "CCCN", 4),
            ),
            protection_rounds=1,
            declared_at_round=0,
            realization_ceiling=23,
        )


# ---- The geometry the reservation depends on ----


def test_production_pmo_charges_exactly_what_it_allocates():
    """The PREMISE the macro-option reservation rests on, read from production source.

    The reservation sits in `_allocate`. It is the binding gate only because production
    sets `candidates_per_batch == QUERIES_PER_ROUND`, which sends `prepare_query_batch`
    down its `len(candidates) <= count` branch so `lock_query_subset` discards nothing.
    If production ever stops doing that, a second discard appears downstream of the
    reservation and this mechanism silently stops binding -- so the premise is asserted
    against the production module rather than restated as a comment here.
    """
    from compose_v4.experiments import pmo_population_v1

    assert (
        pmo_population_v1.configuration().candidates_per_batch
        == pmo_population_v1.QUERIES_PER_ROUND
    )


def test_the_harness_refuses_a_geometry_where_the_reservation_is_not_the_binding_gate():
    # MEASURED: at 8 allocated against 4 charged, the reserved stage-0 leg was generated,
    # reserved and chosen by `_allocate`, then dropped by `select_parent_edits`, so no
    # bridge was charged across an entire campaign and nothing raised.
    config = configuration(SEED)
    assert_production_batch_geometry(config, config.candidates_per_batch)
    for mismatched in (config.candidates_per_batch - 4, config.candidates_per_batch + 1):
        with pytest.raises(ValueError, match="charge exactly what the controller allocates"):
            assert_production_batch_geometry(config, mismatched)


def test_a_declared_leg_reaches_the_charged_batch_under_the_matched_geometry(bootstrapped):
    """Not merely allocated: LOCKED into a batch the ledger charged.

    `_allocate` choosing a reserved candidate proves nothing on its own -- that is exactly
    what happened while the fixture geometry was wrong. The evidence is the published
    pending batch, which is what the campaign hands the ledger.
    """
    folder, _campaign, _kwargs, _, _ = bootstrapped
    locked = []
    for path in sorted((folder / "campaign").glob("round_*/pending.json")):
        batch = json.loads(path.read_text())["batch"]
        locked += [
            row["endpoint"]
            for row in batch["candidates"]
            if row["provenance"].get("entry_channel") == MACRO_OPTION_CHANNEL_TAG
        ]
    assert locked, "no declared macro-option leg was ever locked into a charged batch"
