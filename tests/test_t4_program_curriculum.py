"""Oracle-free coverage/accounting checks for the bounded curriculum driver."""

from compose_v4.control.edit_program import extract_program
from compose_v4.control.edit_program_graph import compile_program_graph, execute_program_graph
from compose_v4.experiments import t4_program_curriculum as curriculum
from tests.test_edit_program import ring_and_carbonyl


class Volume:
    def reload(self):
        pass

    def commit(self):
        pass


def test_curriculum_preserves_all_cells_counts_failures_and_does_not_invent_other_curves(
    tmp_path, monkeypatch
):
    source, stages = ring_and_carbonyl()
    program, binding = extract_program(source, stages)
    _, trace = execute_program_graph(source, compile_program_graph(program), binding)
    rows = []
    cells = ("jak2_1", "fa7_0", "braf_1", "5ht1b_0")
    for cell in cells:
        rows.append(
            {
                "cell": cell,
                "original_seed": "CC",
                "smiles": "CC",
                "role": "seed_control",
                "ds": -1.0,
            }
        )
        for index in range(8):
            rows.append(
                {
                    "cell": cell,
                    "original_seed": "CC",
                    "smiles": trace["endpoint"],
                    "trace": trace,
                    "role": "candidate",
                    "ds": None if index == 0 else -2.0 - index,
                }
            )
    lock = {
        "take": rows,
        "cells": cells,
        "source_registry": {"fixture": True},
        "compute": {"fixture": True},
    }
    monkeypatch.setattr(curriculum, "validate_task", lambda *_: lock)
    monkeypatch.setattr(
        curriculum, "property_scorer", lambda _: lambda _: {"oracle_eligible": True}
    )
    monkeypatch.setattr(
        curriculum.subprocess, "check_output", lambda *_args, **_kwargs: "unit-fixture-no-oracle"
    )
    calls = []

    def parallel(tasks):
        calls.extend(tasks)
        # Like real worker receipts, outcomes omit the input execution trace.
        return [
            {**{k: v for k, v in rows[t["index"]].items() if k != "trace"}, "index": t["index"]}
            for t in reversed(tasks)
        ]

    task = {"run_id": "fixture", "files_sha256": {curriculum.LOCK: "fixture"}}
    result = curriculum.run_remote(task, tmp_path, tmp_path, Volume(), lambda _: None, parallel)
    assert len(calls) == result["new_oracle_calls"] == 36
    assert len(result["cells"]) == 4
    for cell in result["cells"]:
        assert cell["program_curve"][0]["best_eligible_ds"] is None
        assert cell["program_curve"][-1]["calls_including_seed_control"] == 9
        assert cell["best"]["ds"] == -9
        assert cell["broad_control_curve"] is None and cell["adaptive_curve"] is None
    # Completed result reuse invokes no second fake oracle batch.
    assert (
        curriculum.run_remote(task, tmp_path, tmp_path, Volume(), lambda _: None, parallel)
        == result
    )
    assert len(calls) == 36
