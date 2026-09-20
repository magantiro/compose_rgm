"""The Modal worker may only select result keys that `execute_task` actually returns.

This guards a defect that reached the launch line: `execute_task`'s AUC field was renamed
from `auc_top10_development_1000` (a hardcoded 1000-call denominator) to
`auc_top10_at_budget` + `auc_budget`, and the worker's key selection was not updated.
The worker writes `result.json` BEFORE selecting, so the data would have survived -- but
the `KeyError` lands after the whole budget is charged, is caught, written as
`failure.json` and re-raised, and retries are disabled.  Every task would have run its
full 250 calls and then recorded as a failure.

The free-oracle gate could not catch it: the gate calls `execute_task` directly, so it
never executes the worker's post-processing.  Hence a static contract test.
"""

from __future__ import annotations

import ast
import inspect
import re
from pathlib import Path

from compose_v4.experiments import pmo_population_v1 as population

ROOT = Path(__file__).resolve().parents[1]
WORKER = ROOT / "modal_apps/pmo_population_v1_app.py"
# Keys the worker attaches to the result itself before selecting from it.
INJECTED = {"run_id", "contract_payload_sha256", "automatic_retries"}


def _returned_keys() -> set[str]:
    tree = ast.parse(inspect.getsource(population.execute_task).lstrip())
    keys: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Return) and isinstance(node.value, ast.Dict):
            keys |= {k.value for k in node.value.keys if isinstance(k, ast.Constant)}
    return keys


def _selected_keys() -> set[str]:
    match = re.search(
        r"return \{key: result\[key\] for key in\s*\((.*?)\)\}", WORKER.read_text(), re.DOTALL
    )
    assert match, "worker no longer selects result keys in the expected form"
    return set(re.findall(r'"([^"]+)"', match.group(1)))


def test_worker_selects_only_keys_execute_task_returns():
    returned, selected = _returned_keys(), _selected_keys()
    assert returned, "could not recover execute_task's returned keys"
    missing = selected - returned - INJECTED
    assert not missing, (
        f"worker would KeyError after charging the full budget on: {sorted(missing)}; "
        f"execute_task returns {sorted(returned)}"
    )


def test_the_auc_reading_carries_the_budget_it_was_computed_at():
    """A bare AUC is uninterpretable: 250-call and 10000-call readings are different metrics."""
    returned = _returned_keys()
    assert "auc_budget" in returned, "the AUC reading must state its own denominator"
    assert "auc_top10_development_1000" not in returned, (
        "a hardcoded 1000-call denominator understates a shorter run and is not "
        "comparable to a published 10000-call figure"
    )
    assert "auc_budget" in _selected_keys(), "the worker must propagate the AUC denominator"
