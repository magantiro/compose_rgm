"""Real executor, deliberately synthetic task labels; no benchmark oracle calls."""

from dataclasses import replace

import pytest
from rdkit import Chem

from compose_v4.control.adaptive_program_optimizer import ProgramSearchConfig
from compose_v4.control.edit_program import extract_program
from compose_v4.control.edit_program_policy import ProgramEntry
from compose_v4.control.parent_edit_model import ParentEditFeatures, ParentEditModel
from compose_v4.control.program_campaign import ProgramQueryLedger, run_program_campaign
from compose_v4.control.program_task import ProgramTask, initialization_lock, pmo_top_ten_auc
from compose_v4.rewrite.trace_shard import encode_state
from tests.test_edit_program import ring_and_carbonyl


def test_three_scored_learning_rounds_charge_initialization_and_resume(tmp_path):
    source, stages = ring_and_carbonyl()
    program, _ = extract_program(source, stages)
    library = (ProgramEntry(program, ("fixture",)),)
    task = ProgramTask("synthetic_size_fixture", "fixture:no-benchmark-oracle", "pmo")
    initialized = initialization_lock(
        [{"state": encode_state(source), "source_id": "fixture"}],
        count=1,
        seed=7,
        source_sha256="a" * 64,
    )
    ledger = ProgramQueryLedger(
        tmp_path / "queries",
        task,
        lambda s: Chem.MolFromSmiles(s).GetNumHeavyAtoms() / 40,
        budget=7,
    )
    config = replace(
        ProgramSearchConfig.parent_edit_recipe(seed=871, score_direction="maximize"),
        require_broad_runtime=False,
        candidates_per_batch=4,
        attempts_per_batch=48,
    )
    updates = []

    def fit(search, task):
        features, rows = ParentEditFeatures(), []
        for entry in search.entries.values():
            for receipt, observation in search.observations.items():
                if observation["endpoint"] == entry["endpoint"]:
                    rows.append(
                        {
                            "features": features(entry, parent_state=entry["source_state"]),
                            "endpoint": entry["endpoint"],
                            "receipt_id": receipt,
                            "utility": task.utility(observation["score"]),
                            "oracle_protocol": task.oracle_protocol,
                        }
                    )
        updates.append(len(rows))
        return ParentEditModel.fit(
            rows, oracle_protocol=task.oracle_protocol, input_sha256="b" * 64
        )

    arguments = {
        "output": tmp_path / "run",
        "task": task,
        "config": config,
        "initialization": initialized,
        "library": library,
        "ledger": ledger,
        "rounds": 3,
        "queries_per_round": 2,
        "fit_model": fit,
        "fit_model_id": "c" * 64,
    }
    result = run_program_campaign(**arguments)
    assert len(result["history"]) == 3
    assert len(updates) == 2 and updates[1] > updates[0]
    assert result["oracle_calls"] <= 7 and ledger.rows[0]["role"] == "initialization"
    assert result["pmo_top10_auc_with_flat_tail"] is not None
    before = len(ledger.rows)
    repeated = run_program_campaign(**arguments)
    assert repeated["snapshot"] == result["snapshot"] and len(ledger.rows) == before


def test_failed_oracle_stays_charged_and_cannot_retry(tmp_path):
    task = ProgramTask("fixture", "fixture", "pmo")

    def failure(_):
        raise RuntimeError("synthetic oracle interruption")

    ledger = ProgramQueryLedger(tmp_path, task, failure, budget=2)
    with pytest.raises(RuntimeError, match="interruption"):
        ledger.query("CC", lock_id="fixture-lock", role="initialization")
    assert ledger.remaining == 1
    with pytest.raises(RuntimeError, match="blocks"):
        ledger.query("CCC", lock_id="fixture-lock", role="candidate")
    with pytest.raises(RuntimeError, match="failed query remains charged"):
        ProgramQueryLedger(tmp_path, task, failure, budget=2)


def test_pmo_logging_trapezoids_and_tail_are_not_final_best():
    assert pmo_top_ten_auc([0.2, 0.3, 0.4], budget=5, frequency=2) == pytest.approx(0.105)
    assert pmo_top_ten_auc([0.2, 0.3, 0.4], budget=5, frequency=2, finish=True) == pytest.approx(
        0.225
    )
    with pytest.raises(ValueError, match="invalid unique-query"):
        pmo_top_ten_auc([0.2] * 6, budget=5)
