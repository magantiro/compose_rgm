"""Runtime correctness of the macro-option arm end to end, on a synthetic scorer.

The three checks this repository requires of any runtime change, all driven through the
real `run_program_campaign` rather than a transcription of it:

  * RESTORE -- a campaign resumed from its own committed round snapshots rebuilds the
    option registry with its windows intact, so a preempted run does not strand a bridge
    whose oracle call is already spent.
  * BUDGET TERMINATION -- the ledger binds, and options cannot spend past it.
  * WORKER SERIALIZATION -- the snapshot survives the JSON round trip the campaign
    performs when it publishes each round, and restores from the published bytes.

Zero benchmark oracle calls: the scorer is the declared synthetic `valley_similarity_v1`.
"""

import json
from pathlib import Path

import pytest
from rdkit import Chem

from compose_v4.control.dynamic_program_synthesis_v21 import (
    initial_dynamic_program_batch_v21,
)
from compose_v4.control.pmo_macro_option_controller import (
    MACRO_OPTION_CHANNEL_TAG,
    MacroOptionController,
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
SEED = 20260923
# Production charges exactly what the controller allocates, so `_allocate` is the only
# discard and the macro-option reservation is the binding gate.
QUERIES_PER_ROUND = configuration(SEED).candidates_per_batch
SETTINGS = {
    "max_options": 1,
    "synthesis_attempts_per_origin": 3,
    "declaration_wall_seconds": 90.0,
}


def _campaign(folder, *, rounds, budget, protection=True, ledger=None):
    assert_production_batch_geometry(configuration(SEED), QUERIES_PER_ROUND)
    initialized = load_initialization(ROOT, count=2, seed=SEED)
    origin = Chem.MolFromSmiles(initialized["candidates"][0]["endpoint"])
    scorer = ValleySimilarityScorer(origin_heavy_atoms=origin.GetNumHeavyAtoms())
    task = ProgramTask("synthetic_valley_similarity", scorer.protocol, "pmo")
    ledger = ledger or ProgramQueryLedger(folder / "oracle", task, scorer, budget=budget)
    kwargs = {
        "jump_checkpoint": load_jump_checkpoint(ROOT),
        "enable_online_memory": False,
        "enable_macro_options": True,
        "macro_option_protection": protection,
        "macro_option_settings": SETTINGS,
    }
    result = run_program_campaign(
        output=folder / "campaign",
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
    return result, ledger, kwargs, task


@pytest.fixture(scope="module")
def short_run(tmp_path_factory):
    folder = tmp_path_factory.mktemp("macro_runtime")
    return (folder, *_campaign(folder, rounds=5, budget=48))


def test_the_option_registry_survives_the_published_json_round_trip(short_run):
    folder, result, _ledger, kwargs, _task = short_run
    published = sorted((folder / "campaign").glob("round_*/complete.json"))
    assert published, "the campaign published no round snapshot to resume from"
    snapshot = json.loads(published[-1].read_text())["snapshot"]
    # The bytes on disk, not the in-memory object: this is what a restarted worker reads.
    resumed = MacroOptionController.restore(snapshot, **kwargs)
    assert resumed.enable_macro_options is True
    assert resumed.macro_option_protection is True
    assert resumed.option_registry.report()["declared"] == len(
        result["snapshot"]["macro_options"]["registry"]["records"]
    )


def test_a_resumed_campaign_keeps_its_declared_options_and_does_not_redeclare(short_run):
    folder, result, ledger, _kwargs, _task = short_run
    before = result["snapshot"]["macro_options"]["registry"]["records"]
    # Re-entering the SAME output directory is the campaign's own resume path.
    resumed, _, _, _ = _campaign(folder, rounds=5, budget=48, ledger=ledger)
    after = resumed["snapshot"]["macro_options"]["registry"]["records"]
    assert [row["option"]["option_id"] for row in after] == [
        row["option"]["option_id"] for row in before
    ]
    assert resumed["snapshot"]["macro_options"]["options_declared"] is True


def test_a_resume_does_not_strand_a_charged_bridge(short_run):
    folder, _result, _ledger, kwargs, _task = short_run
    published = sorted((folder / "campaign").glob("round_*/complete.json"))
    for path in published:
        snapshot = json.loads(path.read_text())["snapshot"]
        controller = MacroOptionController.restore(snapshot, **kwargs)
        for option_id, option in controller.option_registry.options.items():
            status = controller.option_registry.status(option_id)
            charged = controller.option_registry.report()["options"]
            row = next(r for r in charged if r["option_id"] == option_id)
            if status == "bridge_charged":
                # A charged intermediate must come back with a window and a stored
                # construction, or its already-spent oracle call buys nothing.
                assert row["protected_through_round"] is not None
                assert controller.option_registry.construction(option_id)["stages"]
                assert row["frontier"] >= 0
            assert row["trough_width_rounds"] == 1
            assert option.protects_a_single_crossing


def test_the_ledger_budget_binds_and_options_cannot_spend_past_it(tmp_path):
    budget = 20
    result, ledger, _, _ = _campaign(tmp_path, rounds=20, budget=budget)
    assert len(ledger.rows) <= budget
    assert ledger.remaining >= 0
    assert result["oracle_calls"] == len(ledger.rows)
    # Every charged row is a real reservation with a receipt; nothing an option touched
    # bypassed the ledger.
    assert all(row["status"] == "complete" and row["receipt_id"] for row in ledger.rows)


def test_every_charged_option_leg_is_an_ordinary_ledger_row(short_run):
    folder, _result, ledger, _kwargs, _task = short_run
    charged = {row["endpoint"] for row in ledger.rows}
    tagged = set()
    for path in sorted((folder / "campaign").glob("round_*/pending.json")):
        batch = json.loads(path.read_text())["batch"]
        for candidate in batch["candidates"]:
            if candidate["provenance"].get("entry_channel") == MACRO_OPTION_CHANNEL_TAG:
                tagged.add(candidate["endpoint"])
    assert tagged, "no macro-option leg was ever locked into a charged batch"
    # An option leg is charged through the same ledger at the same price; there is no
    # separate accounting path and no uncharged shortcut.
    assert tagged <= charged
