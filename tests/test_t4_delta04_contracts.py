"""The delta=0.4 arms must differ from delta=0.6 ONLY in the benchmark threshold.

The scientific claim under test is "one frozen state-adaptive controller across
all T4 tasks".  If a controller setting drifted between the two thresholds that
claim would be false, and the drift would be invisible in any summary -- the
contracts are large and nobody diffs them by eye.

So the guard enumerates every LEAF of both payloads and fails on any difference
outside an explicit allow-list.  The leaf set is derived from the payloads, not
written down, so a controller field added later is covered automatically.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TARGETS = ("5ht1b", "braf", "fa7", "jak2", "parp1")

#: Leaf paths (or their prefixes) that are ALLOWED to differ between thresholds.
#: Everything here is a benchmark input, a per-arm identity, or prose.
ALLOWED_DIFFERENCES = (
    "delta",
    "delta_provenance",
    "runtime_inputs_sha256",
    "claim_boundary",
    "experimental_setting",
)


def _payload(relative: str) -> dict:
    return json.loads((ROOT / relative).read_text())["payload"]


def _leaves(value, prefix: str = "") -> dict[str, object]:
    """Flatten a payload to {dotted path: scalar}."""

    if isinstance(value, dict):
        out: dict[str, object] = {}
        for key, item in value.items():
            out.update(_leaves(item, f"{prefix}.{key}" if prefix else str(key)))
        return out
    if isinstance(value, list):
        out = {}
        for index, item in enumerate(value):
            out.update(_leaves(item, f"{prefix}[{index}]"))
        return out
    return {prefix: value}


def _allowed(path: str) -> bool:
    root = path.split(".", 1)[0].split("[", 1)[0]
    return root in ALLOWED_DIFFERENCES


def _arms(target: str) -> tuple[dict, dict]:
    return (
        _payload(f"configs/t4_unified_controller_{target}_d06_v1.json"),
        _payload(f"configs/t4_unified_controller_{target}_d04_v1.json"),
    )


@pytest.mark.parametrize("target", TARGETS)
def test_only_the_threshold_and_identity_differ(target: str) -> None:
    d06, d04 = _arms(target)
    left, right = _leaves(d06), _leaves(d04)

    assert set(left) - set(right) == set() or all(
        _allowed(path) for path in set(left) - set(right)
    ), f"{target}: leaves removed outside the allow-list"
    assert all(
        _allowed(path) for path in set(right) - set(left)
    ), f"{target}: leaves added outside the allow-list"

    unexplained = sorted(
        path
        for path in set(left) & set(right)
        if left[path] != right[path] and not _allowed(path)
    )
    assert not unexplained, (
        f"{target}: {len(unexplained)} controller settings differ between the two "
        f"thresholds: {unexplained[:10]}"
    )


@pytest.mark.parametrize("target", TARGETS)
def test_the_threshold_actually_moved(target: str) -> None:
    d06, d04 = _arms(target)
    assert d06["delta"] == 0.6
    assert d04["delta"] == 0.4


@pytest.mark.parametrize("target", TARGETS)
def test_executable_delta_agrees_with_the_prose(target: str) -> None:
    """A shipped contract once carried delta 0.4 while its prose said 0.6.

    `delta` is executable -- `Fiber(cell["smiles"], contract["delta"], ...)` --
    and `claim_boundary` is prose, so the two can disagree silently and the run
    is then not the experiment the boundary describes.
    """

    for payload in _arms(target):
        assert f"delta {payload['delta']}" in payload["claim_boundary"], (
            f"{target}: executable delta={payload['delta']} is not the delta the "
            f"claim_boundary declares: {payload['claim_boundary'][:160]}"
        )


@pytest.mark.parametrize("target", TARGETS)
def test_each_arm_pins_its_own_wrapper_and_not_its_sibling(target: str) -> None:
    d06, d04 = _arms(target)
    assert (
        f"modal_apps/t4_unified_controller_{target}_app.py"
        in d06["runtime_inputs_sha256"]
    )
    assert (
        f"modal_apps/t4_unified_controller_{target}_d04_app.py"
        in d04["runtime_inputs_sha256"]
    )
    assert (
        f"modal_apps/t4_unified_controller_{target}_app.py"
        not in d04["runtime_inputs_sha256"]
    ), f"{target}: the d04 arm pins the d06 wrapper, so it would launch the wrong arm"


@pytest.mark.parametrize("target", TARGETS)
def test_every_pin_addresses_the_current_tree(target: str) -> None:
    """Stale pins that address nothing are a recorded failure mode here."""

    import hashlib

    for payload in _arms(target):
        for relative, expected in payload["runtime_inputs_sha256"].items():
            path = ROOT / relative
            assert path.exists(), f"{target}: pinned input is missing: {relative}"
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
            assert actual == expected, f"{target}: stale pin for {relative}"


@pytest.mark.parametrize("target", TARGETS)
def test_generated_arms_are_not_self_authorizing(target: str) -> None:
    _, d04 = _arms(target)
    assert d04["scored_launch_authorized"] is False
    assert d04["modal_launch_authorized"] is False


def test_the_guard_can_actually_fail() -> None:
    """Negative control: a drifted controller setting must be caught.

    Without this the allow-list could be wrong in the permissive direction and
    every assertion above would still pass.
    """

    d06, d04 = _arms("braf")
    drifted = json.loads(json.dumps(d04))
    drifted["batch"] = d06["batch"] + 1
    left, right = _leaves(d06), _leaves(drifted)
    unexplained = [
        path
        for path in set(left) & set(right)
        if left[path] != right[path] and not _allowed(path)
    ]
    assert unexplained == ["batch"]
