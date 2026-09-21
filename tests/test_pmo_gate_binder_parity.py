"""The live-parent support gate must bind through the binder production binds through.

The gate's own docstring makes scoring conditional on its receipt: "Scoring is
prohibited until the resulting receipt passes the non-root support floor."  A receipt
only carries that authority if the gate exercised the deployed binder.  When the
controller's jump lane moved from the width-4 beam (``bind_joint_plan``) to the
complete realizer, the gate kept the beam and its sealed decision silently stopped
gating the runtime it was cited for.  No test failed, because nothing compared them.

The expectation here is DERIVED FROM THE CONTROLLER, never restated.  A binder call is
identified structurally -- a call whose second positional argument is the plan latent --
and the callee is resolved through the module's own ``from ... import`` bindings.  Two
independent paths must agree:

* static: the gate's resolved binder target and keyword-argument names equal the
  controller's;
* runtime: the function object each module actually holds is the same object.

Naming either binder in this file would reintroduce exactly the duplicated constant
that let the drift through, so neither is named: swap either side's binder and both
paths go red.
"""

from __future__ import annotations

import ast
import importlib
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_CONTROLLER_MODULE = "compose_v4.control.pmo_population_controller"
_GATE_MODULE = "compose_v4.experiments.pmo_population_live_parent_gate"


def _module_path(dotted: str) -> Path:
    return _REPO_ROOT / "src" / (dotted.replace(".", "/") + ".py")


def _import_bindings(tree: ast.Module) -> dict[str, str]:
    """Local name -> fully qualified target for every ``from X import Y`` in the file."""

    bindings: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            for alias in node.names:
                bindings[alias.asname or alias.name] = f"{node.module}.{alias.name}"
    return bindings


def _binder_calls(dotted: str) -> list[dict[str, object]]:
    """Every plan-binding call in a module, keyed by what it resolves to.

    Structural identification: a binder is called as ``f(<parent>, plan, ...)``.  This
    is read off the source rather than matched against a name, so the test cannot
    agree with a stale constant the way the drift it exists to catch did.
    """

    tree = ast.parse(_module_path(dotted).read_text())
    bindings = _import_bindings(tree)
    calls = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or len(node.args) < 2:
            continue
        second = node.args[1]
        if not (isinstance(second, ast.Name) and second.id == "plan"):
            continue
        if not isinstance(node.func, ast.Name):
            continue
        target = bindings.get(node.func.id)
        if target is None or not target.startswith("compose_v4."):
            # A module-local helper; its own body carries the imported call.
            continue
        calls.append(
            {
                "line": node.lineno,
                "local_name": node.func.id,
                "target": target,
                "keywords": frozenset(kw.arg for kw in node.keywords if kw.arg),
            }
        )
    return calls


def _sole_controller_binder() -> dict[str, object]:
    calls = _binder_calls(_CONTROLLER_MODULE)
    assert len(calls) == 1, (
        "expected exactly one plan-binding call in the controller's jump lane; "
        f"found {[(row['line'], row['target']) for row in calls]}. "
        "If the lane legitimately gained a second binder call, this test's derivation "
        "needs updating -- do not weaken it to accept a mismatch."
    )
    return calls[0]


def test_gate_binds_through_the_controllers_binder() -> None:
    """The gate's binder target must equal the controller's, derived from the source."""

    controller = _sole_controller_binder()
    gate_calls = _binder_calls(_GATE_MODULE)
    assert gate_calls, (
        "the support gate makes no resolvable plan-binding call; its receipt cannot "
        "certify support for a binder it never exercises"
    )
    drifted = [row for row in gate_calls if row["target"] != controller["target"]]
    assert not drifted, (
        "the gate binds through a different function than production does: gate "
        f"{[(row['line'], row['target']) for row in drifted]} vs controller "
        f"{controller['target']} (line {controller['line']}). The gate's sealed "
        "decision would certify a binder the runtime no longer uses."
    )


def test_gate_passes_the_controllers_binder_parameters() -> None:
    """Keyword NAMES must match; values may differ where the lane budget differs.

    The controller narrows ``seconds_cap`` by the jump lane's remaining wall time,
    which the gate has no analogue for, so only the parameter set is compared.  A
    parameter the controller sets and the gate omits would silently leave the gate on
    a library default -- a different configuration measured under the gate's name.
    """

    controller = _sole_controller_binder()
    for row in _binder_calls(_GATE_MODULE):
        assert row["keywords"] == controller["keywords"], (
            f"gate binder call at line {row['line']} passes {sorted(row['keywords'])} "
            f"while the controller passes {sorted(controller['keywords'])}"
        )


def test_gate_and_controller_hold_the_same_binder_object() -> None:
    """Runtime path, independent of the source-text one: same function object.

    Catches a drift the static check cannot -- an alias or a same-named wrapper in
    another module resolving to different behaviour.
    """

    controller = _sole_controller_binder()
    gate_calls = _binder_calls(_GATE_MODULE)
    controller_fn = getattr(
        importlib.import_module(_CONTROLLER_MODULE), str(controller["local_name"])
    )
    gate_module = importlib.import_module(_GATE_MODULE)
    for row in gate_calls:
        gate_fn = getattr(gate_module, str(row["local_name"]), None)
        assert gate_fn is controller_fn, (
            f"the gate's {row['local_name']} is not the controller's "
            f"{controller['local_name']}: {gate_fn!r} vs {controller_fn!r}"
        )


def test_contract_declares_the_binder_the_controller_calls() -> None:
    """The sealed contract's declared binder must be the one the call site uses.

    ``configs/pmo_population_controller_v1.json`` names the jump lane's binder and its
    call site.  That string is semantics, so the mechanical re-pin tool will not move
    it -- which means a binder swap that forgot the contract would leave the contract
    authorizing a runtime that no longer exists, in the same way the gate was left
    certifying one.  The expectation is again read from the controller's source.
    """

    import json

    controller = _sole_controller_binder()
    contract = json.loads(
        (_REPO_ROOT / "configs" / "pmo_population_controller_v1.json").read_text()
    )
    realization = contract["payload"]["realization"]
    assert realization["binder"] == controller["target"], (
        f"contract declares binder {realization['binder']} but the controller calls "
        f"{controller['target']} at line {controller['line']}"
    )
    assert realization["call_site"].endswith("_generate_jump_pool"), (
        "the contract's declared call site moved; the derivation in this test reads "
        "the whole controller module and would no longer be pinned to the jump lane"
    )


def test_gate_decision_config_matches_the_receipt_it_cites() -> None:
    """The sealed decision's numbers must come from the receipt it points at.

    ``reseal_pmo_population_contracts.py`` re-hashes every key already present in
    ``implementation_sha256`` but treats ``support_observed`` as ordinary semantics and
    only guards it against CHANGE.  So a re-seal after a binder swap re-pins the gate
    SOURCE while leaving support numbers measured under the previous binder -- a config
    that validates and binds a stale measurement.  This closes that by requiring the
    numbers, the cited receipt, and the binder to agree with each other and with the
    controller.
    """

    import hashlib
    import json

    config = json.loads(
        (_REPO_ROOT / "configs" / "pmo_population_live_parent_gate_v2.json").read_text()
    )["payload"]
    cited = config["inputs"]["exhaustive_result"]
    receipt_path = _REPO_ROOT / cited
    assert receipt_path.exists(), f"the decision cites a receipt that is not on disk: {cited}"
    digest = hashlib.sha256(receipt_path.read_bytes()).hexdigest()
    assert digest == config["input_sha256"]["exhaustive_result"], (
        f"{cited} does not hash to the value the decision pins"
    )

    receipt = json.loads(receipt_path.read_text())["payload"]
    for key, value in config["support_observed"].items():
        assert receipt["support"][key] == value, (
            f"decision reports {key}={value} but its cited receipt says "
            f"{receipt['support'][key]}; the decision was not re-measured"
        )

    controller = _sole_controller_binder()
    assert receipt["binder"]["function"] == controller["target"], (
        "the cited receipt was produced by a binder the controller no longer calls"
    )
    assert config["binder_observed"]["function"] == controller["target"], (
        "the decision records a binder the controller no longer calls"
    )
    floor = config["parent_policy"]["minimum_supported_non_root_parents"]
    assert config["support_observed"]["supported_parent_count"] >= floor, (
        "the decision records a PASS below its own support floor"
    )
