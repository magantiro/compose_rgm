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
    configuration,
    load_initialization,
    load_jump_checkpoint,
)

ROOT = Path(__file__).resolve().parents[1]
SEED = 20260922


def _bootstrap(tmp_path, *, count=2, budget=6):
    """One bootstrap round of the real campaign, so the controller has a real archive."""
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
        rounds=1,
        queries_per_round=2,
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


def _restore(campaign, kwargs, **overrides):
    return MacroOptionController.restore(campaign["snapshot"], **{**kwargs, **overrides})


def _eligibility(task):
    return task.endpoint_evaluator()


def _open_a_window(controller):
    """Declare from the real archive, then charge the first leg through the registry."""
    report = controller._declare_macro_options()
    assert report["options"], "declaration produced no macro option to protect"
    option_id = report["options"][0]["option_id"]
    option = controller.option_registry.options[option_id]
    controller.option_registry.advance_round(controller.batches)
    controller.option_registry.note_charged(option.bridge_endpoint, controller.batches)
    return option_id, option


@pytest.fixture(scope="module")
def bootstrapped(tmp_path_factory):
    folder = tmp_path_factory.mktemp("macro_wiring")
    result = _bootstrap(folder)
    yield result
    shutil.rmtree(folder, ignore_errors=True)


# ---- Consumption ----


def test_the_production_path_reaches_both_protection_hooks(bootstrapped):
    campaign, kwargs, _, task = bootstrapped
    controller = _restore(campaign, kwargs)
    option_id, option = _open_a_window(controller)
    # The bridge must be in the archive for the floor to have anything to lift.
    assert controller.option_registry.protected_bridges()
    eligibility = _eligibility(task)
    consumed = assert_macro_option_protection_is_consumed(
        lambda: controller.propose_batch(eligibility)
    )
    assert consumed["consumed"] == {"parent_mass_floor": True, "reserved_slot": True}
    assert option_id in controller.option_registry.options
    assert option.protects_a_single_crossing


def test_the_consumption_check_fails_when_protection_is_off(bootstrapped):
    # The check must be capable of reporting NOT consumed, or a pass proves nothing.
    campaign, kwargs, _, task = bootstrapped
    controller = _restore(campaign, kwargs, macro_option_protection=False)
    _open_a_window(controller)
    with pytest.raises(ValueError, match="NOT consumed"):
        assert_macro_option_protection_is_consumed(
            lambda: controller.propose_batch(_eligibility(task))
        )


def test_nothing_runs_at_all_when_macro_options_are_disabled(bootstrapped):
    campaign, kwargs, _, task = bootstrapped
    controller = _restore(campaign, kwargs, enable_macro_options=False)
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
    campaign, kwargs, _, task = bootstrapped
    controller = _restore(campaign, kwargs)
    option_id, option = _open_a_window(controller)
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
    campaign, kwargs, _, task = bootstrapped
    controller = _restore(campaign, kwargs)
    _open_a_window(controller)
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
    campaign, kwargs, _, _ = bootstrapped
    controller = _restore(campaign, kwargs)
    _open_a_window(controller)
    snapshot = controller.snapshot()
    with pytest.raises(ValueError, match="arm flags changed"):
        MacroOptionController.restore(
            snapshot, **{**kwargs, "macro_option_protection": False}
        )


def test_a_resume_carries_the_open_window_so_the_bridge_is_not_stranded(bootstrapped):
    campaign, kwargs, _, _ = bootstrapped
    controller = _restore(campaign, kwargs)
    option_id, option = _open_a_window(controller)
    snapshot = json.loads(json.dumps(controller.snapshot()))
    resumed = MacroOptionController.restore(snapshot, **kwargs)
    assert resumed.option_registry.protected_bridges() == {
        option.bridge_endpoint: option_id
    }
    assert resumed.options_declared is True
    assert resumed.option_registry.construction(option_id)["stages"]
    assert resumed.option_rng.bit_generator.state == controller.option_rng.bit_generator.state


def test_a_snapshot_with_options_cannot_resume_into_a_controller_without_them(bootstrapped):
    campaign, kwargs, _, _ = bootstrapped
    controller = _restore(campaign, kwargs)
    _open_a_window(controller)
    snapshot = controller.snapshot()
    with pytest.raises(ValueError, match="arm flags changed"):
        MacroOptionController.restore(
            snapshot, **{**kwargs, "enable_macro_options": False}
        )


# ---- Declaration refuses what needs no staging ----


def test_a_declared_option_always_exceeds_the_realization_ceiling(bootstrapped):
    campaign, kwargs, _, _ = bootstrapped
    controller = _restore(campaign, kwargs)
    report = controller._declare_macro_options()
    ceiling = report["realization_ceiling"]
    assert report["options"]
    for option in report["options"]:
        assert option["total_primitives"] > ceiling
        assert option["stages"] and 2 <= len(option["stages"]) <= 4
        assert option["protection_source"] == "chosen_ordering_trough_width"


def test_a_stored_leg_that_no_longer_reproduces_its_endpoint_is_refused(bootstrapped):
    # Admissibility is a hard filter: a leg whose executed endpoint moved is never
    # offered under the declared option's name.
    campaign, kwargs, _, task = bootstrapped
    controller = _restore(campaign, kwargs)
    option_id, _ = _open_a_window(controller)
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
