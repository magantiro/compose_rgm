"""Replicates 2 and 3 may differ from replicate 1 ONLY in the stochastic seed.

The claim is that three replicates measure the SAME frozen policy under
different randomness. If a controller setting moved between replicate 1 and the
new arms, the spread would be confounded with a configuration change and the
mean of the three would not mean anything.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TARGETS = ("5ht1b", "braf", "fa7", "jak2", "parp1")
TAGS = ("d04", "d06")
REPLICATE_STRIDE = 8_000_000_029
CHARGED_CALLS_PER_CELL = 250
ARMS = [(t, g) for t in TARGETS for g in TAGS]

#: Allowed to differ between a base arm (replicate 1) and its replicate arm.
ALLOWED = (
    "cells",
    "claim_boundary",
    "replicate_policy",
    "runtime_inputs_sha256",
    "schema_version",
    "status",
    "total_charged_call_ceiling",
)


def _payload(relative: str) -> dict:
    return json.loads((ROOT / relative).read_text())["payload"]


def _base(target: str, tag: str) -> dict:
    return _payload(f"configs/t4_unified_controller_{target}_{tag}_v1.json")


def _replicate(target: str, tag: str) -> dict:
    return _payload(f"configs/t4_unified_controller_{target}_{tag}_r23_v1.json")


def _leaves(value, prefix: str = "") -> dict[str, object]:
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
    return path.split(".", 1)[0].split("[", 1)[0] in ALLOWED


@pytest.mark.parametrize("target,tag", ARMS)
def test_no_controller_setting_moved(target: str, tag: str) -> None:
    left, right = _leaves(_base(target, tag)), _leaves(_replicate(target, tag))
    unexplained = sorted(
        path
        for path in set(left) & set(right)
        if left[path] != right[path] and not _allowed(path)
    )
    assert not unexplained, (
        f"{target} {tag}: controller settings differ between replicate 1 and "
        f"replicates 2/3: {unexplained[:10]}"
    )


@pytest.mark.parametrize("target,tag", ARMS)
def test_seeds_follow_the_recorded_derivation(target: str, tag: str) -> None:
    base = {row["cell"]: row["controller_seed"] for row in _base(target, tag)["cells"]}
    for row in _replicate(target, tag)["cells"]:
        expected = base[row["source_cell"]] + REPLICATE_STRIDE * (row["replicate"] - 1)
        assert row["controller_seed"] == expected, (
            f"{row['cell']}: seed {row['controller_seed']} is not "
            f"{row['source_cell']} + {REPLICATE_STRIDE}*({row['replicate']}-1)"
        )
        assert row["replicate_1_controller_seed"] == base[row["source_cell"]]


@pytest.mark.parametrize("target,tag", ARMS)
def test_replicate_one_is_not_rerun(target: str, tag: str) -> None:
    """A replicate arm must not contain replicate 1's seed for any cell."""

    base_seeds = {row["controller_seed"] for row in _base(target, tag)["cells"]}
    for row in _replicate(target, tag)["cells"]:
        assert row["replicate"] in (2, 3)
        assert row["controller_seed"] not in base_seeds


@pytest.mark.parametrize("target,tag", ARMS)
def test_ceiling_is_derived_from_the_cells(target: str, tag: str) -> None:
    payload = _replicate(target, tag)
    assert payload["charged_calls_per_cell"] == CHARGED_CALLS_PER_CELL
    assert payload["total_charged_call_ceiling"] == sum(
        CHARGED_CALLS_PER_CELL for _ in payload["cells"]
    )


def test_the_whole_panel_is_exactly_the_authorized_budget() -> None:
    """30 cells x 2 replicates x 250 = 15,000, recomputed rather than asserted."""

    cells = sum(len(_replicate(t, g)["cells"]) for t, g in ARMS)
    total = sum(_replicate(t, g)["total_charged_call_ceiling"] for t, g in ARMS)
    assert cells == 60, f"expected 30 cells x 2 replicates, got {cells}"
    assert total == 15_000, f"expected 15,000 charged calls, got {total}"


@pytest.mark.parametrize("target,tag", ARMS)
def test_within_a_cell_the_two_replicates_differ(target: str, tag: str) -> None:
    by_source: dict[str, set[int]] = {}
    for row in _replicate(target, tag)["cells"]:
        by_source.setdefault(row["source_cell"], set()).add(row["controller_seed"])
    for source, seeds in by_source.items():
        assert len(seeds) == 2, f"{source}: replicates share a seed"


def test_historical_cross_cell_seed_sharing_is_preserved_not_silently_broken() -> None:
    """braf_0 and parp1_0 share a base seed historically; that survives the stride.

    Recorded as `inherited_cross_cell_seed_sharing` in the replication manifest
    and deliberately preserved, because a replicate clones the method rather than
    repairing it. Asserting global seed uniqueness would look like a stronger
    guard and would in fact demand a change nobody authorized.
    """

    braf = {r["source_cell"]: r["controller_seed"] for r in _replicate("braf", "d06")["cells"]
            if r["replicate"] == 2}
    parp1 = {r["source_cell"]: r["controller_seed"] for r in _replicate("parp1", "d06")["cells"]
             if r["replicate"] == 2}
    assert braf["braf_0"] == parp1["parp1_0"]


@pytest.mark.parametrize("target,tag", ARMS)
def test_arms_are_not_self_authorizing(target: str, tag: str) -> None:
    payload = _replicate(target, tag)
    assert payload["scored_launch_authorized"] is False
    assert payload["modal_launch_authorized"] is False


@pytest.mark.parametrize("target,tag", ARMS)
def test_every_pin_addresses_the_current_tree(target: str, tag: str) -> None:
    payload = _replicate(target, tag)
    expected_wrapper = f"modal_apps/t4_unified_controller_{target}_{tag}_r23_app.py"
    assert expected_wrapper in payload["runtime_inputs_sha256"]
    for relative, digest in payload["runtime_inputs_sha256"].items():
        path = ROOT / relative
        assert path.exists(), f"missing pinned input {relative}"
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest, (
            f"stale pin for {relative}"
        )


def test_the_guard_can_actually_fail() -> None:
    """Negative control: a drifted setting and a wrong seed must both be caught."""

    base, replicate = _base("fa7", "d06"), json.loads(json.dumps(_replicate("fa7", "d06")))
    replicate["parents"] = base["parents"] + 1
    left, right = _leaves(base), _leaves(replicate)
    assert [
        p for p in set(left) & set(right)
        if left[p] != right[p] and not _allowed(p)
    ] == ["parents"]

    row = replicate["cells"][0]
    assert row["controller_seed"] != row["replicate_1_controller_seed"] + 1
