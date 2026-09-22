"""The canonical T4 controller: one frozen configuration, three arms, no target routing.

These tests exist to make three claims checkable rather than asserted:

1. the three arms differ ONLY in their declared proposal vocabulary and the presence
   of the adaptive rule, and agree exactly on everything else;
2. no runtime module can branch on a target identity, and the check that says so is
   capable of failing;
3. a selected row produced by the REAL proposal pipeline carries every key the app's
   `run_cell` subscripts off it -- derived from `run_cell`'s own source by AST, so the
   question comes from the code under test and the answer comes from a production run.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from compose_v4.control.docking_value import identity
from compose_v4.experiments import t4_canonical_controller as canonical
from compose_v4.experiments.t4_canonical_controller import (
    ARMS,
    SHARED_CONTROLLER,
    TargetNameRoutingError,
    assert_no_target_name_routing,
    canonical_cells,
    controller_identity,
    load_seeds,
    should_expand_support,
    target_name_routing_findings,
)
from compose_v4.experiments.t4_integrated_route_fiber import validate_expert_vocabulary
from compose_v4.experiments.t4_matched_pilot import unseal

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "modal_apps/t4_canonical_shared_controller_app.py"
SEEDS = ROOT / "docs/GENMOL_T4_SEEDS.json"
CONTRACTS = {
    arm: ROOT / f"configs/t4_canonical_shared_controller_{arm.lower()}_v1.json" for arm in ARMS
}


def _contracts() -> dict[str, dict]:
    missing = [arm for arm, path in CONTRACTS.items() if not path.exists()]
    if missing:
        pytest.skip(f"arm contracts not built yet: {missing}")
    return {arm: unseal(path) for arm, path in CONTRACTS.items()}


# ---- One controller -----------------------------------------------------------------


def test_the_three_arms_agree_on_the_shared_controller():
    """The identity is computed over each SEALED payload, not over the source dict.

    Comparing the builder's input to itself would pass however the contracts were
    written; comparing the three artefacts is what makes "one controller" checkable.
    """

    contracts = _contracts()
    identities = {arm: controller_identity(payload) for arm, payload in contracts.items()}
    assert len(set(identities.values())) == 1, identities


def test_every_shared_field_is_byte_identical_across_arms():
    contracts = _contracts()
    for field in SHARED_CONTROLLER:
        values = {arm: json.dumps(payload[field], sort_keys=True)
                  for arm, payload in contracts.items()}
        assert len(set(values.values())) == 1, f"{field} differs across arms: {values}"


def test_arms_differ_only_in_vocabulary_and_the_adaptive_rule():
    contracts = _contracts()
    allowed_to_differ = {
        "arm", "arm_name", "arm_isolates", "cells", "cell_identity_sha256", "deltas",
        "experts", "adaptive_support_expansion", "expansion_trigger_min_eligible",
        "support_expansion", "total_charged_call_ceiling",
    }
    keys = set().union(*(set(payload) for payload in contracts.values()))
    for key in sorted(keys - allowed_to_differ):
        values = {arm: json.dumps(payload.get(key), sort_keys=True)
                  for arm, payload in contracts.items()}
        assert len(set(values.values())) == 1, f"{key} differs across arms: {values}"


def test_a_cell_shared_by_two_arms_has_identical_inputs():
    contracts = _contracts()
    seen: dict[str, dict] = {}
    for payload in contracts.values():
        for cell in payload["cells"]:
            previous = seen.setdefault(cell["cell"], cell)
            assert previous == cell, f"{cell['cell']} differs between arms"
    # The delta=0.6 cells really are shared, or this test proves nothing.
    shared = {row["cell"] for row in contracts["A"]["cells"]} & {
        row["cell"] for row in contracts["C"]["cells"]
    }
    assert len(shared) == 15, shared


def test_controller_seed_is_a_function_of_the_inputs_only():
    """Same seed molecule and delta -> same RNG seed, whichever arm asks."""

    seeds = load_seeds(SEEDS)
    a = {row["cell"]: row["controller_seed"] for row in canonical_cells(seeds, (0.6,))}
    c = {row["cell"]: row["controller_seed"] for row in canonical_cells(seeds, (0.6, 0.4))}
    for cell, value in a.items():
        assert c[cell] == value
    # and the two deltas of one seed are DIFFERENT streams
    assert c["fa7_0_d06"] != c["fa7_0_d04"]


def test_the_adaptive_rule_is_present_exactly_when_declared():
    for arm, payload in _contracts().items():
        adaptive = payload["adaptive_support_expansion"]
        assert isinstance(adaptive, bool)
        assert ("support_expansion" in payload) is adaptive, arm
        assert (payload["expansion_trigger_min_eligible"] > 0) is adaptive, arm


def test_budget_ceiling_matches_the_cell_count():
    total = 0
    for payload in _contracts().values():
        expected = len(payload["cells"]) * payload["charged_calls_per_cell"]
        assert payload["total_charged_call_ceiling"] == expected
        total += expected
    assert total == 15000, total


def test_every_arm_vocabulary_validates():
    for payload in _contracts().values():
        assert validate_expert_vocabulary(tuple(payload["experts"]))


def test_cells_cover_the_published_panel():
    contracts = _contracts()
    seeds = load_seeds(SEEDS)
    by_smiles = {row["smiles"] for row in seeds}
    for arm, payload in contracts.items():
        assert {row["smiles"] for row in payload["cells"]} == by_smiles, arm
    assert len(contracts["C"]["cells"]) == 30
    assert len(contracts["A"]["cells"]) == len(contracts["B"]["cells"]) == 15


# ---- No target-name routing ---------------------------------------------------------


def _runtime_modules() -> list[Path]:
    payload = next(iter(_contracts().values()))
    return [
        ROOT / relative
        for relative in sorted(payload["runtime_inputs_sha256"])
        if relative.endswith(".py")
    ]


def test_no_runtime_module_branches_on_a_target_name():
    scanned = assert_no_target_name_routing(_runtime_modules())
    assert len(scanned) >= 10, scanned


def test_the_routing_check_can_fail(tmp_path):
    """A control that proves the scan is load-bearing rather than vacuously green."""

    injected = tmp_path / "injected.py"
    injected.write_text(
        "def choose(target, state):\n"
        "    if target == 'braf':\n"
        "        return 'wide'\n"
        "    return 'narrow'\n"
    )
    with pytest.raises(TargetNameRoutingError):
        assert_no_target_name_routing([injected])


@pytest.mark.parametrize(
    "snippet",
    [
        "x = 1 if target == 'fa7' else 2\n",
        "while target != 'jak2':\n    break\n",
        "assert target != '5ht1b'\n",
        "y = [v for v in rows if v == 'parp1_0']\n",
        "z = target in {'braf', 'fa7'}\n",
    ],
)
def test_the_routing_check_catches_every_branching_construct(tmp_path, snippet):
    path = tmp_path / "snippet.py"
    path.write_text(snippet)
    findings = target_name_routing_findings(path.read_text(), filename=str(path))
    assert findings, snippet


def test_the_routing_check_permits_target_names_as_data(tmp_path):
    """Irrelevant-perturbation control: without it, "everything refused" would be
    indistinguishable from a harness that refuses everything."""

    path = tmp_path / "data.py"
    path.write_text(
        "BOXES = {'braf': [1, 2, 3], 'fa7': [4, 5, 6]}\n"
        "def label(target, index):\n"
        "    return f'{target}_{index}'\n"
        "COMMENT = 'measured on fa7_0'\n"
    )
    assert assert_no_target_name_routing([path], allow_data_assignment_to=("BOXES",))


def test_the_receptor_table_is_the_only_place_a_target_name_is_data():
    """If the exemption widened, the guarantee would quietly weaken."""

    source = Path(canonical.__file__).read_text()
    tree = ast.parse(source)
    assigned = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and any(
                    token in ast.unparse(node.value).lower() for token in canonical.TARGET_TOKENS
                ):
                    assigned.add(target.id)
    assert assigned == {"T4_RECEPTORS"}, assigned


# ---- The dry pass: every key run_cell subscripts off a selected row ------------------


def _run_cell_source() -> str:
    text = APP.read_text()
    for node in ast.walk(ast.parse(text)):
        if isinstance(node, ast.FunctionDef) and node.name == "run_cell":
            return ast.get_source_segment(text, node) or ""
    raise AssertionError("run_cell not found in the canonical app")


def _row_keys_run_cell_requires() -> set[str]:
    """Every `row["..."]` subscript `run_cell` performs on a SELECTED row.

    Derived from the code under test so it cannot drift, while the values it is
    checked against come from a real pipeline run -- the source supplies the question
    and the production path supplies the answer.
    """

    tree = ast.parse(_run_cell_source())

    def _subscripts(node) -> set[str]:
        found = set()
        for child in ast.walk(node):
            if (
                isinstance(child, ast.Subscript)
                and isinstance(child.value, ast.Name)
                and child.value.id == "row"
                and isinstance(child.slice, ast.Constant)
                and isinstance(child.slice.value, str)
            ):
                found.add(child.slice.value)
        return found

    keys: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.For, ast.comprehension)):
            continue
        names = {child.id for child in ast.walk(node.iter) if isinstance(child, ast.Name)}
        binds_row = any(
            isinstance(child, ast.Name) and child.id == "row" for child in ast.walk(node.target)
        )
        if "selected" in names and binds_row:
            body = node.body if isinstance(node, ast.For) else [node.iter]
            for statement in body:
                keys |= _subscripts(statement)
    assert keys, "no loop over `selected` binding `row` was found in run_cell"
    return keys


@pytest.mark.parametrize("experts", [("shallow",), ("shallow", "anchored_replacement", "structured")])
def test_dry_pass_a_real_selected_row_carries_every_key_run_cell_needs(experts):
    """Drive the real round body for both arm vocabularies, then check the rows.

    Four launches of a predecessor arm died on exactly this class of defect -- a key
    the round body subscripts that the record shape does not carry -- each time only
    discoverable by launching.
    """

    import numpy as np

    from compose_v4.control.fiber_control import ProgramValue, SearchState
    from compose_v4.experiments.t4_fiber_campaign import Fiber, expand
    from compose_v4.experiments.t4_integrated_route_fiber import (
        attach_features,
        expert_census,
        merge_expert_pools,
        select_batch,
    )

    parent = "CC(C)CCN(C)C(=O)c1ccccc1"
    fiber = Fiber(parent, 0.2, support="compose_valid")
    pools = {
        expert: expand(
            parent, -7.5, fiber, np.random.default_rng(20260922 + index),
            draws=40, multi_region=True, horizon=3, proposal_lane=expert,
        )
        for index, expert in enumerate(experts)
    }
    assert any(pools.values()), "no lane produced anything; this test would prove little"

    state = SearchState(archive={parent: -7.5}, budget=8, rounds=0)
    merged = merge_expert_pools(pools, experts=experts)
    fresh = [row for row in merged if row["smiles"] not in state.archive]
    candidates = attach_features(fresh, state, fiber, experts=experts)
    assert candidates
    expert_census(candidates, experts=experts)
    selected = select_batch(
        candidates, ProgramValue(penalty=1.0), state, np.random.default_rng(1),
        round_index=1, batch=8, exploration=2, expert_floor_rounds=2, experts=experts,
    )
    assert selected

    required = _row_keys_run_cell_requires()
    assert "proposal_experts" in required, (
        "run_cell no longer requires proposal_experts; this guard would stop catching "
        "the defect that killed a predecessor launch"
    )
    for row in selected:
        missing = sorted(required - set(row))
        assert not missing, f"selected row is missing {missing}"

    # And the lock must serialize, which is where a numpy feature array bites. This
    # drives the SAME function the app writes locks with, not a transcription of it.
    json.dumps(canonical.jsonable(candidates), sort_keys=True)


def test_expansion_records_normalize_into_a_selectable_row():
    """The union path arm C takes: raw `expand` records carry no `proposal_experts`."""

    import numpy as np

    from compose_v4.control.fiber_control import ProgramValue, SearchState
    from compose_v4.experiments.t4_fiber_campaign import Fiber, expand
    from compose_v4.experiments.t4_integrated_route_fiber import (
        attach_features,
        select_batch,
    )
    from compose_v4.experiments.t4_support_expansion import normalize_expansion_records

    experts = ("shallow", "anchored_replacement", "structured")
    parent = "CC(C)CCN(C)C(=O)c1ccccc1"
    fiber = Fiber(parent, 0.2, support="compose_valid")
    ladder = expand(parent, -7.5, fiber, np.random.default_rng(7), draws=40,
                    multi_region=True, horizon=3, proposal_lane="shallow")
    assert ladder
    assert "proposal_experts" not in ladder[0], (
        "expand now supplies proposal_experts itself; this test's premise has moved"
    )
    state = SearchState(archive={parent: -7.5}, budget=8, rounds=0)
    fresh = normalize_expansion_records(row for row in ladder if row["smiles"] != parent)
    candidates = attach_features(fresh, state, fiber, experts=experts)
    selected = select_batch(
        candidates, ProgramValue(penalty=1.0), state, np.random.default_rng(1),
        round_index=1, batch=8, exploration=2, expert_floor_rounds=2, experts=experts,
    )
    assert selected
    required = _row_keys_run_cell_requires()
    for row in selected:
        assert not sorted(required - set(row))


# ---- The trigger --------------------------------------------------------------------


def test_the_trigger_reads_only_the_eligible_count():
    assert should_expand_support(distinct_eligible=0, minimum=4)
    assert should_expand_support(distinct_eligible=3, minimum=4)
    assert not should_expand_support(distinct_eligible=4, minimum=4)
    assert not should_expand_support(distinct_eligible=99, minimum=4)
    # At 1 it is exactly the shipped "the pool came back empty" condition.
    assert should_expand_support(distinct_eligible=0, minimum=1)
    assert not should_expand_support(distinct_eligible=1, minimum=1)


def test_the_trigger_signature_admits_nothing_cell_specific():
    import inspect

    parameters = set(inspect.signature(should_expand_support).parameters)
    assert parameters == {"distinct_eligible", "minimum"}, parameters


# ---- Contract / runtime binding -----------------------------------------------------


def test_every_pinned_runtime_input_matches_the_working_tree():
    from compose_v4.experiments.continuation_profile import sha256_file

    for arm, payload in _contracts().items():
        for relative, expected in payload["runtime_inputs_sha256"].items():
            actual = sha256_file(ROOT / relative)
            assert actual == expected, f"{arm}: {relative} moved"


def test_the_app_pins_itself_and_the_controller_module():
    payload = next(iter(_contracts().values()))
    pinned = set(payload["runtime_inputs_sha256"])
    assert "modal_apps/t4_canonical_shared_controller_app.py" in pinned
    assert "src/compose_v4/experiments/t4_canonical_controller.py" in pinned
    assert "src/compose_v4/experiments/t4_support_expansion.py" in pinned


def test_contract_payload_hash_verifies():
    for arm, path in CONTRACTS.items():
        if not path.exists():
            pytest.skip("contracts not built")
        envelope = json.loads(path.read_text())
        assert identity(envelope["payload"]) == envelope["payload_sha256"], arm


def test_no_route_distilled_expert_anywhere_in_the_arms():
    """The exclusion is a scientific decision; a test keeps it from drifting back."""

    for arm, payload in _contracts().items():
        assert "route_complete_region" not in payload["experts"], arm
        assert "route_complete_region" not in payload["proposal"], arm
        assert "route_distilled" not in json.dumps(payload["runtime_inputs_sha256"]), arm
