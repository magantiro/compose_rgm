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
INJECTED = {
    "run_id",
    "contract_payload_sha256",
    "automatic_retries",
    "oracle_positive_control_passed",
}


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


# ---- the oracle positive control must stay wired ---------------------------


def test_the_worker_scores_through_the_call_time_pinned_oracle():
    """`evaluate` must go through `AssetPinnedOracle`, never a bare `tdc.Oracle`.

    A bare oracle resolves its pickled asset against whatever the working directory
    happens to be at CALL time.  The PMO gsk3b run that this guard exists for scored
    250 distinct molecules at exactly 0.0 because the worker restored the working
    directory after CONSTRUCTING the oracle, and PyTDC's bare `except` turned the
    resulting `FileNotFoundError` into `default_property`.
    """
    source = WORKER.read_text()
    assert "AssetPinnedOracle(" in source, "the worker no longer pins the asset root"
    assert re.search(r"evaluate=lambda smiles: float\(scorer\(smiles\)\)", source), (
        "the worker must score through the pinned wrapper, not the raw oracle"
    )
    assert not re.search(r"evaluate=lambda smiles: float\(oracle\(smiles\)\)", source), (
        "scoring through the raw oracle reintroduces the cwd-dependent asset load"
    )


def test_the_worker_calls_a_positive_control_before_charging_the_budget():
    """Construction is not evidence; only a CALL against known values is.

    The environment smoke that was in place throughout the defect records
    `oracle_called: False` by design, so it could never have caught this.  The gate
    must therefore be an assertion that calls the oracle, and it must sit before
    `execute_task` spends anything.
    """
    source = WORKER.read_text()
    assert "assert_positive_control(" in source, "the worker no longer runs a positive control"
    control_at = source.index("assert_positive_control(scorer")
    execute_at = source.index("result = execute_task(")
    assert control_at < execute_at, (
        "the positive control must run before the first charged oracle call"
    )


def test_the_positive_control_failure_leaves_the_task_runnable():
    """A control failure must not consume the task's one no-retry attempt."""
    source = WORKER.read_text()
    control_at = source.index("assert_positive_control(scorer")
    started_at = source.index('_write_json(started, {')
    assert control_at < started_at, (
        "the control must run before started.json, or a failed control burns the attempt"
    )
