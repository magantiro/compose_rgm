"""A resume may raise an authorized budget, never lower it, and never change the task.

A charged oracle call cannot be un-charged, so growth is safe; a DECREASE would make an
already-spent run read as under budget and could mask an overspend.
"""

from __future__ import annotations

import pytest

from compose_v4.control.program_campaign import ProgramQueryLedger
from compose_v4.control.program_task import ProgramTask


def _task(name: str = "celecoxib_rediscovery") -> ProgramTask:
    return ProgramTask(name, "protocol-abc", "pmo")


def _ledger(folder, budget: int, task: ProgramTask | None = None) -> ProgramQueryLedger:
    return ProgramQueryLedger(folder, task or _task(), lambda smiles: 0.5, budget=budget)


def test_budget_may_be_raised_on_resume(tmp_path):
    first = _ledger(tmp_path, 250)
    first.query("CCO", lock_id="l0", role="test")
    assert first.remaining == 249

    resumed = _ledger(tmp_path, 1000)
    # the charged call survives the extension -- it is not re-spent and not forgotten
    assert len(resumed.rows) == 1
    assert resumed.remaining == 999


def test_budget_may_not_be_lowered_on_resume(tmp_path):
    _ledger(tmp_path, 1000)
    with pytest.raises(ValueError, match="may not decrease"):
        _ledger(tmp_path, 250)


def test_task_may_not_change_even_when_budget_grows(tmp_path):
    _ledger(tmp_path, 250)
    with pytest.raises(ValueError, match="task changed"):
        _ledger(tmp_path, 1000, task=_task("gsk3b"))


def test_extension_history_is_recorded(tmp_path):
    import json

    _ledger(tmp_path, 250)
    _ledger(tmp_path, 1000)
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["budget"] == 1000
    assert manifest["budget_history"] == [250, 1000]
