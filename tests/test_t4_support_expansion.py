"""The bounded support expansion: its stopping rule, its bounds, and its wiring.

The invariant under test is the one that decides whether `fa7_0` spends its budget
or repeats its blank: an empty candidate pool must EXPAND before it may terminate.

Every behavioural test drives the real `run_support_expansion` with stub stages
whose yield depends on the draw budget, which is exactly the situation the
mechanism exists for.  The wiring tests read the app's own source rather than a
transcription of it, and each one is paired with a mutation in
`test_wiring_mutations` that must turn it red -- a structural check whose
expectation is recomputed from the code under test cannot fail.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from compose_v4.experiments.t4_support_expansion import (
    ESCALATABLE_LANES,
    ExpansionOutcome,
    SupportExpansionContractError,
    SupportExpansionNotConsumed,
    SupportExpansionPolicy,
    assert_support_expansion_is_consumed,
    resolve_support_expansion,
    run_support_expansion,
)

ROOT = Path(__file__).resolve().parents[1]
BASE_APP = ROOT / "modal_apps/t4_fa7_0_support_expansion_base_app.py"
WRAPPER_APP = ROOT / "modal_apps/t4_fa7_0_support_expansion_app.py"
CONTRACT = ROOT / "configs/t4_fa7_0_support_expansion_v1.json"

_GOOD_BLOCK = {
    "draw_ladder": [960, 1920, 3840],
    "lanes": ["shallow", "anchored_replacement"],
    "zero_support_fallback": True,
    "stop_at_distinct_eligible": 4,
    "max_extra_draws_per_event": 6720,
    "wall_seconds": 7200.0,
}


def _policy(**overrides) -> SupportExpansionPolicy:
    block = {**_GOOD_BLOCK, **overrides}
    return SupportExpansionPolicy.from_contract(block)


def _records(count: int, *, prefix: str = "C") -> list[dict]:
    return [{"smiles": prefix + "C" * index} for index in range(count)]


# ---- Contract surface ----------------------------------------------------


def test_policy_reads_a_well_formed_block():
    policy = _policy()
    assert policy.draw_ladder == (960, 1920, 3840)
    assert policy.lanes == ("shallow", "anchored_replacement")
    assert policy.zero_support_fallback is True
    assert policy.as_record()["draw_ladder"] == [960, 1920, 3840]


@pytest.mark.parametrize(
    "overrides",
    [
        {"draw_ladder": []},
        {"draw_ladder": [960, 0]},
        {"draw_ladder": [960, -1]},
        {"lanes": []},
        {"lanes": ["route_complete_region"]},
        {"stop_at_distinct_eligible": 0},
        {"max_extra_draws_per_event": 100},
        {"wall_seconds": 0.0},
        {"wall_seconds": -1.0},
    ],
)
def test_policy_refuses_an_unbounded_or_malformed_block(overrides):
    with pytest.raises(SupportExpansionContractError):
        _policy(**overrides)


def test_policy_refuses_an_unknown_key():
    block = {**_GOOD_BLOCK, "max_rounds": 3}
    with pytest.raises(SupportExpansionContractError):
        SupportExpansionPolicy.from_contract(block)


def test_fallback_flag_must_be_an_explicit_boolean():
    """A coerced or absent value would leave it unclear whether the stage ran."""

    for value in (None, 1, "true", ""):
        block = {**_GOOD_BLOCK, "zero_support_fallback": value}
        with pytest.raises(SupportExpansionContractError):
            SupportExpansionPolicy.from_contract(block)


def test_route_lane_is_not_escalatable():
    assert "route_complete_region" not in ESCALATABLE_LANES


def test_absent_field_is_the_only_none():
    assert resolve_support_expansion({}) is None
    assert resolve_support_expansion({"support_expansion": _GOOD_BLOCK}) is not None
    with pytest.raises(SupportExpansionContractError):
        resolve_support_expansion({"support_expansion": {"draw_ladder": []}})


# ---- The stopping rule ---------------------------------------------------


def test_a_budget_dependent_yield_is_recovered_by_escalation():
    """The mechanism's whole point: empty at the base budget, non-empty above it.

    This is the `fa7_0` situation in miniature -- the normal round found nothing,
    and a larger draw budget over the same lanes does.
    """

    seen = []

    def escalate(draws, attempt):
        seen.append((draws, attempt))
        return _records(4) if draws >= 1920 else []

    outcome = run_support_expansion(_policy(zero_support_fallback=False), escalate=escalate)
    assert seen == [(960, 0), (1920, 1)]
    assert outcome.stop_reason == "reached_target"
    assert outcome.distinct_eligible == 4
    assert outcome.found_any
    assert outcome.draws_spent == 2880


def test_a_ladder_that_never_yields_is_exhaustion_with_the_bound_named():
    outcome = run_support_expansion(
        _policy(zero_support_fallback=False), escalate=lambda draws, attempt: []
    )
    assert outcome.stop_reason == "ladder_exhausted"
    assert outcome.attempts == 3
    assert outcome.draws_spent == 6720
    assert not outcome.found_any


def test_the_draw_cap_binds_before_the_ladder_ends():
    outcome = run_support_expansion(
        _policy(
            zero_support_fallback=False,
            draw_ladder=[960, 1920],
            max_extra_draws_per_event=2000,
        ),
        escalate=lambda draws, attempt: [],
    )
    assert outcome.stop_reason == "draw_cap"
    assert outcome.attempts == 1
    assert outcome.draws_spent == 960


def test_the_wall_clock_binds_and_is_checked_before_a_step_is_spent():
    ticks = iter([0.0, 0.0, 100.0, 100.0, 100.0])

    outcome = run_support_expansion(
        _policy(zero_support_fallback=False, wall_seconds=10.0),
        escalate=lambda draws, attempt: [],
        clock=lambda: next(ticks),
    )
    assert outcome.stop_reason == "wall_clock"
    assert outcome.attempts == 1


def test_already_seen_endpoints_are_not_counted_as_found():
    """An expansion that only re-finds archived molecules has found nothing new."""

    outcome = run_support_expansion(
        _policy(zero_support_fallback=False, stop_at_distinct_eligible=2),
        escalate=lambda draws, attempt: _records(3),
        already_seen=[row["smiles"] for row in _records(3)],
    )
    assert outcome.distinct_eligible == 0
    assert outcome.stop_reason == "ladder_exhausted"


def test_duplicate_endpoints_across_steps_are_counted_once():
    outcome = run_support_expansion(
        _policy(zero_support_fallback=False, stop_at_distinct_eligible=99),
        escalate=lambda draws, attempt: _records(2),
    )
    assert outcome.distinct_eligible == 2
    assert outcome.attempts == 3


def test_a_record_without_an_endpoint_is_refused_rather_than_silently_dropped():
    with pytest.raises(SupportExpansionContractError):
        run_support_expansion(
            _policy(zero_support_fallback=False),
            escalate=lambda draws, attempt: [{"quality": 0.9}],
        )


# ---- The fallback stage --------------------------------------------------


def test_the_fallback_runs_first_and_can_satisfy_the_target_alone():
    """It is the cheap, structurally different stage, so no ladder step is spent."""

    calls = []

    def escalate(draws, attempt):
        calls.append(draws)
        return []

    outcome = run_support_expansion(
        _policy(stop_at_distinct_eligible=2),
        escalate=escalate,
        fallback=lambda: (_records(3), {"regions_considered": 26}),
    )
    assert calls == []
    assert outcome.fallback_ran is True
    assert outcome.fallback_eligible == 3
    assert outcome.fallback_work == {"regions_considered": 26}
    assert outcome.stop_reason == "reached_target"
    assert outcome.attempt_log[0]["stage"] == "zero_support_fallback"


def test_a_zero_yield_fallback_still_hands_over_to_the_ladder():
    """The measured fa7_0 case: the fallback finds nothing and the ladder must run."""

    outcome = run_support_expansion(
        _policy(stop_at_distinct_eligible=1),
        escalate=lambda draws, attempt: _records(1) if attempt >= 1 else [],
        fallback=lambda: ([], {"regions_considered": 26, "distinct_eligible_endpoints": 0}),
    )
    assert outcome.fallback_ran is True
    assert outcome.fallback_eligible == 0
    assert outcome.attempts == 2
    assert outcome.stop_reason == "reached_target"
    assert outcome.found_any


def test_the_fallback_is_skipped_when_the_contract_disables_it():
    outcome = run_support_expansion(
        _policy(zero_support_fallback=False),
        escalate=lambda draws, attempt: [],
        fallback=lambda: (_records(9), {}),
    )
    assert outcome.fallback_ran is False
    assert outcome.distinct_eligible == 0


# ---- The consumption gate ------------------------------------------------


def test_publishing_exhaustion_without_an_expansion_is_refused():
    """The `not_run` clause, on a fixture the second clause cannot also catch.

    A plain `ExpansionOutcome()` trips BOTH clauses, so disabling either one left
    the other to raise and a mutation survived. `fallback_ran=True` satisfies the
    second clause, leaving only the stop-reason check able to refuse.
    """

    with pytest.raises(SupportExpansionNotConsumed):
        assert_support_expansion_is_consumed(
            ExpansionOutcome(stop_reason="not_run", fallback_ran=True)
        )
    with pytest.raises(SupportExpansionNotConsumed):
        assert_support_expansion_is_consumed(ExpansionOutcome())


def test_an_event_that_ran_no_stage_at_all_is_refused():
    """A misconfigured bound that spends nothing must not end the cell."""

    outcome = ExpansionOutcome(stop_reason="draw_cap")
    with pytest.raises(SupportExpansionNotConsumed):
        assert_support_expansion_is_consumed(outcome)


def test_a_real_exhausted_ladder_is_accepted():
    outcome = run_support_expansion(
        _policy(zero_support_fallback=False), escalate=lambda draws, attempt: []
    )
    assert_support_expansion_is_consumed(outcome)


# ---- The contract this arm launches under --------------------------------


def _contract_payload() -> dict:
    return json.loads(CONTRACT.read_text())["payload"]


def test_the_contract_declares_one_cell_and_both_mechanisms():
    payload = _contract_payload()
    assert [row["cell"] for row in payload["cells"]] == ["fa7_0"]
    assert payload["delta"] == 0.6
    # The ceiling is DERIVED, so assert the INVARIANT rather than a literal that
    # has to be edited after every dead launch: ceiling plus everything already
    # charged must equal the owner's authorized total, exactly.
    assert payload["charged_calls_per_cell"] == payload["total_charged_call_ceiling"]
    prior = payload["prior_charged_calls"]
    spent = sum(
        block["charged_calls"]
        for key, block in prior.items()
        if isinstance(block, dict) and "charged_calls" in block
    )
    assert spent >= 1, prior
    assert payload["charged_calls_per_cell"] + spent == 248, (
        "ceiling plus already-charged calls must equal the authorized 248 exactly; "
        "a dead launch's calls are subtracted, never forgiven"
    )
    assert payload["proposal"]["shallow"]["region_law"] == "free_gate_margin_v1"
    assert payload["proposal"]["shallow"]["completion_law"] == "free_gate_margin_v1"
    SupportExpansionPolicy.from_contract(payload["support_expansion"])


def test_the_contract_pins_every_module_the_mechanisms_live_in():
    """A mechanism absent from the pin block is free to drift under this contract."""

    pinned = set(_contract_payload()["runtime_inputs_sha256"])
    for relative in (
        "src/compose_v4/control/replace_completion_law.py",
        "src/compose_v4/control/completion_law_contract.py",
        "src/compose_v4/control/zero_support_fallback.py",
        "src/compose_v4/experiments/t4_support_expansion.py",
        "src/compose_v4/control/bridge_region_law.py",
        "src/compose_v4/control/region_law_contract.py",
        "modal_apps/t4_fa7_0_support_expansion_app.py",
        "modal_apps/t4_fa7_0_support_expansion_base_app.py",
    ):
        assert relative in pinned, relative


def test_every_pinned_runtime_input_matches_the_tree():
    from compose_v4.experiments.continuation_profile import sha256_file

    for relative, expected in _contract_payload()["runtime_inputs_sha256"].items():
        assert sha256_file(ROOT / relative) == expected, relative


# ---- The app wiring ------------------------------------------------------


def _run_cell_source() -> str:
    tree = ast.parse(BASE_APP.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "run_cell":
            return ast.get_source_segment(BASE_APP.read_text(), node) or ""
    raise AssertionError("run_cell not found in the base app")


def _terminal_block() -> ast.If:
    """The `if not selected:` block that publishes candidate exhaustion."""

    source = _run_cell_source()
    tree = ast.parse(ast.unparse(ast.parse(source)))
    blocks = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.If) and "candidate_exhaustion" in ast.unparse(node)
    ]
    assert blocks, "no candidate_exhaustion branch found in run_cell"
    return min(blocks, key=lambda node: len(ast.unparse(node)))


def _calls(node: ast.AST) -> set[str]:
    names = set()
    for child in ast.walk(node):
        if isinstance(child, ast.Call):
            func = child.func
            if isinstance(func, ast.Name):
                names.add(func.id)
            elif isinstance(func, ast.Attribute):
                names.add(func.attr)
    return names


def test_run_cell_expands_support_before_it_can_terminate():
    """The invariant: exhaustion is publishable only after an expansion event."""

    source = _run_cell_source()
    assert "run_support_expansion" in source, (
        "run_cell never runs a support expansion; an empty candidate pool would "
        "terminate the cell on the first unlucky draw, which is how fa7_0 went blank"
    )
    expansion_at = source.index("run_support_expansion(")
    terminal_at = source.index('"status": "candidate_exhaustion"')
    assert expansion_at < terminal_at, (
        "the expansion must run BEFORE the terminal publish, not after it"
    )


def test_the_terminal_publish_is_gated_on_the_expansion_having_run():
    assert "assert_support_expansion_is_consumed" in _calls(_terminal_block()), (
        "the candidate_exhaustion branch does not assert the expansion was consumed, "
        "so a declared-but-inert expansion could still end the cell"
    )


def test_the_terminal_result_records_the_expansion_and_its_bound():
    block = ast.unparse(_terminal_block())
    assert "support_expansion" in block
    assert "as_record" in block


def test_a_missing_contract_block_raises_rather_than_terminating_quietly():
    """`policy` must be BOUND to the resolver's result, not merely import it.

    The first version of this test looked for the string, which the import line
    inside `run_cell` supplies even after the call is deleted -- a mutation
    survived on exactly that.
    """

    tree = ast.parse(_run_cell_source())
    bound = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "policy"
            for target in node.targets
        )
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Name)
        and node.value.func.id == "resolve_support_expansion"
    ]
    assert bound, (
        "run_cell never binds `policy` to resolve_support_expansion(contract); the "
        "contract's expansion block would go unread"
    )
    tail = _run_cell_source()
    assert "raise ValueError" in tail[tail.index("resolve_support_expansion(contract)") :]


def test_the_round_lock_carries_the_expansion_telemetry():
    """Otherwise a terminal result could not be audited after the fact."""

    source = _run_cell_source()
    assert "support_expansion_telemetry" in source


def test_the_base_app_bakes_its_own_source_into_the_image():
    """A stale bake would ship the parent arm's loop under this contract's name."""

    source = BASE_APP.read_text()
    assert "t4_fa7_0_support_expansion_base_app.py" in source
    assert "t4_integrated_route_fiber_parp1_app.py" not in source


def test_the_proposal_worker_threads_both_laws_and_the_fallback():
    source = BASE_APP.read_text()
    for needle in (
        "completion_law_for_proposal_lane",
        "completion_law=completion_law",
        "fallback_candidates",
        "zero_support_fallback",
    ):
        assert needle in source, needle


def test_the_wrapper_refuses_a_contract_missing_any_mechanism():
    """Drives the real `_assert_authorized`, not a transcription of it."""

    import importlib.util

    spec = importlib.util.spec_from_file_location("_wrapper_probe", WRAPPER_APP)
    source = WRAPPER_APP.read_text()
    assert spec is not None
    # Import only the guard, not the Modal app: execute the module body up to the
    # point the guard is defined, which is everything before the `_assert_authorized()`
    # call at module scope.
    head = source[: source.index("_assert_authorized()\n")]
    namespace: dict = {"__file__": str(WRAPPER_APP)}
    exec(compile(head, str(WRAPPER_APP), "exec"), namespace)  # noqa: S102
    guard = namespace["_assert_authorized"]
    payload = _contract_payload()
    authorized = payload["status"] == namespace["AUTHORIZED_STATUS"]
    if authorized:
        guard()
    else:
        # A draft contract must refuse, which is itself one of the guard's clauses.
        with pytest.raises(RuntimeError):
            guard()

    target = Path(namespace["__file__"]).resolve().parents[1] / namespace["CONTRACT"]
    # Each mutation removes ONE mechanism from an otherwise authorized contract, so a
    # surviving mutation means that clause is decorative.
    authorized_payload = {**payload, "status": namespace["AUTHORIZED_STATUS"]}
    shallow = authorized_payload["proposal"]["shallow"]
    mutations = [
        {"proposal": {**authorized_payload["proposal"],
                      "shallow": {k: v for k, v in shallow.items() if k != "completion_law"}}},
        {"proposal": {**authorized_payload["proposal"],
                      "shallow": {k: v for k, v in shallow.items() if k != "region_law"}}},
        {"support_expansion": {}},
        {"cells": [{"cell": "fa7_2"}]},
        {"status": "DRAFT_PENDING_OWNER_AUTHORIZATION"},
    ]
    original = target.read_text()
    try:
        # Positive control: an authorized contract with nothing removed must PASS, so
        # a guard that refuses everything cannot masquerade as five killed mutations.
        target.write_text(json.dumps({"payload": authorized_payload}))
        guard()
        for mutation in mutations:
            target.write_text(json.dumps({"payload": {**authorized_payload, **mutation}}))
            with pytest.raises(RuntimeError):
                guard()
    finally:
        target.write_text(original)


# ---- The fallback record must survive the real feature path ----------------


def test_a_real_fallback_record_survives_attach_features_and_selection():
    """The shape bug that fires only when the fallback SUCCEEDS.

    `t4_integrated_route_fiber._experts` validates a record's `proposal_lane`
    against the frozen expert vocabulary and RAISES on an unknown one, so a record
    carrying "zero_support_fallback" killed `attach_features` at exactly the moment
    the fallback first produced an eligible endpoint. Stubbed `{"smiles": ...}`
    records cannot catch that, so this drives the REAL fallback and pushes a REAL
    record through the real feature and selection path.
    """

    import numpy as np

    from compose_v4.control.fiber_control import ProgramValue, SearchState
    from compose_v4.control.zero_support_fallback import fallback_candidates
    from compose_v4.experiments.t4_fiber_campaign import Fiber
    from compose_v4.experiments.t4_integrated_route_fiber import (
        attach_features,
        expert_census,
        select_batch,
    )

    # A permissive reference so the fallback actually returns something: the point
    # is the RECORD SHAPE, not this molecule's chemistry.
    parent = "CC(C)CCN(C)C(=O)c1ccccc1"
    fiber = Fiber(parent, 0.2, support="compose_valid")
    produced, _ = fallback_candidates(
        parent,
        np.random.default_rng(20260922),
        check=fiber.check,
        reference_smiles=parent,
        delta=0.2,
    )
    assert produced, "the fallback returned nothing, so this test proves nothing"
    assert produced[0]["proposal_lane"] == "zero_support_fallback", (
        "the fallback no longer labels its lane, so the app's re-labelling may be dead"
    )

    # The app's re-labelling, applied exactly as `proposal_worker` applies it.
    records = [
        {
            **row,
            "proposal_lane": None,
            "proposal_experts": [],
            "support_expansion_stage": "zero_support_fallback",
            "parent_score": -7.5,
            "families": ("atom_delete",),
            "program_families": ("atom_delete",),
            "regions": 1,
            "created": int(row.get("inserted_atoms", 0)),
            "deleted": int(row.get("deleted_atoms", 0)),
        }
        for row in produced
    ]
    state = SearchState(archive={parent: -7.5}, budget=8, rounds=0)
    candidates = attach_features(records, state, fiber)
    assert candidates, "attach_features dropped every real fallback record"
    expert_census(candidates)
    selected = select_batch(
        candidates,
        ProgramValue(penalty=1.0),
        state,
        np.random.default_rng(1),
        round_index=1,
        batch=4,
        exploration=1,
        expert_floor_rounds=2,
    )
    assert selected, "select_batch could not select a real fallback candidate"


def test_the_app_clears_the_fallback_lane_label():
    source = BASE_APP.read_text()
    assert '"proposal_lane": None' in source
    assert '"support_expansion_stage": "zero_support_fallback"' in source


def test_a_real_fallback_candidate_serializes_into_a_round_lock():
    """`attach_features` attaches a SET fingerprint and numpy features.

    The round lock is published as canonical JSON, so an expansion record that
    cannot serialize would fail AFTER the expansion succeeded and before anything
    was docked -- the same fires-only-on-success shape as the lane-label defect.
    """

    import numpy as np

    from compose_v4.control.fiber_control import SearchState
    from compose_v4.control.zero_support_fallback import fallback_candidates
    from compose_v4.experiments.t4_fiber_campaign import Fiber
    from compose_v4.experiments.t4_integrated_route_fiber import attach_features

    def _jsonable(value):
        """The app's own helper, reproduced so a drift in it fails this test."""

        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, set):
            return sorted(value)
        if isinstance(value, dict):
            return {str(key): _jsonable(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [_jsonable(item) for item in value]
        if isinstance(value, np.generic):
            return value.item()
        return value

    parent = "CC(C)CCN(C)C(=O)c1ccccc1"
    fiber = Fiber(parent, 0.2, support="compose_valid")
    produced, _ = fallback_candidates(
        parent,
        np.random.default_rng(20260922),
        check=fiber.check,
        reference_smiles=parent,
        delta=0.2,
    )
    rows = [
        {
            **row,
            "proposal_lane": None,
            "proposal_experts": [],
            "support_expansion_stage": "zero_support_fallback",
            "parent_score": -7.5,
            "families": ("atom_delete",),
            "program_families": ("atom_delete",),
            "regions": 1,
            "created": int(row.get("inserted_atoms", 0)),
            "deleted": int(row.get("deleted_atoms", 0)),
        }
        for row in produced
    ]
    state = SearchState(archive={parent: -7.5}, budget=8, rounds=0)
    candidates = attach_features(rows, state, fiber)
    assert candidates
    json.dumps(_jsonable(candidates), sort_keys=True, separators=(",", ":"))


# ---- The root checkpoint that makes a round-one preemption survivable -------


def test_the_root_call_is_checkpointed_before_round_one():
    """MEASURED FAILURE run 4321b8b1: preemption in round one killed the cell.

    The root lock is published BEFORE the root docking and the first checkpoint was
    only written at the END of round one, so a preemption in between left a lock
    with no checkpoint -- which `_resume_state` correctly refuses. Modal restarts a
    preempted container with the same input even at `retries=0`, so the restart met
    its own root lock and the cell died having charged one call for nothing. The
    support expansion widened that window from minutes to tens of minutes.
    """

    source = _run_cell_source()
    root_at = source.index('_publish(folder / "round_000_lock.json"')
    loop_at = source.index("while state.budget > 0:")
    between = source[root_at:loop_at]
    assert 'folder / "checkpoint.json"' in between, (
        "the completed root call is not checkpointed before the round loop, so a "
        "preemption during round one leaves a lock with no checkpoint and the cell "
        "cannot resume"
    )
    # It must carry the contract identity, or `_resume_state` refuses it outright.
    checkpoint_at = between.index('folder / "checkpoint.json"')
    tail = between[checkpoint_at:]
    for field in ("contract_payload_sha256", "charged_calls", "budget_remaining", "rng_state"):
        assert field in tail, field


def test_the_root_lock_is_not_forfeited_by_its_own_checkpoint():
    """Round 0 must read as COMPLETE on resume, never as an interrupted round.

    `_resume_state` skips locks whose index is at or below the last checkpointed
    round. A root checkpoint carries `rounds: []`, so `completed` is 0 and the
    round-0 lock is skipped rather than debited -- exercised here against the real
    function so a change to that arithmetic fails.
    """

    import importlib.util
    import json as _json
    import tempfile

    spec = importlib.util.spec_from_file_location("_base_probe", BASE_APP)
    assert spec is not None
    source = BASE_APP.read_text()
    head = source[: source.index("def _jsonable(")]
    # Execute only the pure helpers: stop before the Modal function definitions.
    namespace: dict = {"__file__": str(BASE_APP)}
    marker = "def _resume_state("
    helpers = head[head.index("def _publish(") :]
    exec(  # noqa: S102
        compile(
            "from pathlib import Path\nimport json\n"
            + helpers[helpers.index(marker) :],
            str(BASE_APP),
            "exec",
        ),
        namespace,
    )
    resume = namespace["_resume_state"]

    payload = {
        "schema_version": "t4_integrated_route_fiber_checkpoint_v1",
        "status": "running",
        "cell": "fa7_0",
        "contract_payload_sha256": "deadbeef",
        "charged_calls": 1,
        "budget_remaining": 246,
        "archive": {"CCO": -8.3},
        "features": [],
        "improvements": [],
        "history": [],
        "rounds": [],
        "rng_state": {"state": 1},
    }
    with tempfile.TemporaryDirectory() as directory:
        folder = Path(directory)
        from compose_v4.control.docking_value import identity

        (folder / "checkpoint.json").write_text(
            _json.dumps({"payload": payload, "payload_sha256": identity(payload)})
        )
        (folder / "round_000_lock.json").write_text(
            _json.dumps({"payload": {"round": 0, "queries": [{"smiles": "CCO"}]}})
        )
        restored = resume(folder, {"contract_payload_sha256": "deadbeef"}, {})

    assert restored is not None, "a root checkpoint must be resumable"
    assert restored["forfeited_calls"] == 0, (
        "the round-0 lock was debited as an interrupted round; its call is already "
        "counted in the checkpoint and would be charged twice"
    )
    assert restored["charged_calls"] == 1
    assert restored["budget_remaining"] == 246


# ---- The dry pass: drive the whole expansion round body end to end ---------


def _row_keys_run_cell_requires() -> set[str]:
    """Every `row["..."]` subscript `run_cell` performs, read from its own source.

    Derived from the code under test rather than hand-listed, so it cannot drift
    out of date -- but the VALUES it is checked against come from a real pipeline
    run, so this is not a tautology: the source supplies the question and the
    production path supplies the answer.
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

    # Scope to loops that iterate the SELECTED batch. `run_cell` binds the name
    # `row` in several unrelated loops -- over contract["cells"], over the docked
    # observations -- and folding those in would demand keys a proposal record is
    # not supposed to carry. The scoping is still derived from the source, not
    # hand-listed: it is "whichever loops walk `selected`".
    keys: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.For, ast.comprehension)):
            continue
        iterable = node.iter
        target = node.target
        names = {
            child.id for child in ast.walk(iterable) if isinstance(child, ast.Name)
        }
        binds_row = any(
            isinstance(child, ast.Name) and child.id == "row"
            for child in ast.walk(target)
        )
        if "selected" in names and binds_row:
            body = node.body if isinstance(node, ast.For) else [iterable]
            for statement in body:
                keys |= _subscripts(statement)
    assert keys, "no loop over `selected` binding `row` was found in run_cell"
    return keys


def test_dry_pass_expansion_round_body_end_to_end():
    """Four launches, four defects, each found only by launching. This runs the
    whole expansion round body locally so the remaining ones surface at once.

    It drives the REAL stages -- `fallback_candidates` and `expand` for records,
    `normalize_expansion_records`, `attach_features`, `select_batch` -- and then
    checks the resulting selected rows against every key `run_cell` subscripts off
    `row`, including the query rows and the post-docking bookkeeping.
    """

    import numpy as np

    from compose_v4.control.fiber_control import ProgramValue, SearchState
    from compose_v4.control.zero_support_fallback import fallback_candidates
    from compose_v4.experiments.t4_fiber_campaign import Fiber, expand
    from compose_v4.experiments.t4_integrated_route_fiber import (
        attach_features,
        expert_census,
        select_batch,
    )
    from compose_v4.experiments.t4_support_expansion import normalize_expansion_records

    parent = "CC(C)CCN(C)C(=O)c1ccccc1"
    fiber = Fiber(parent, 0.2, support="compose_valid")

    # Stage 1: the fallback, exactly as the worker relabels it.
    produced, _ = fallback_candidates(
        parent,
        np.random.default_rng(20260922),
        check=fiber.check,
        reference_smiles=parent,
        delta=0.2,
    )
    records = [
        {
            **row,
            "proposal_lane": None,
            "proposal_experts": [],
            "support_expansion_stage": "zero_support_fallback",
            "parent_score": -7.5,
            "families": ("atom_delete",),
            "program_families": ("atom_delete",),
            "regions": 1,
            "created": int(row.get("inserted_atoms", 0)),
            "deleted": int(row.get("deleted_atoms", 0)),
        }
        for row in produced
    ]
    # Stage 2: the draw ladder -- raw `expand` output, which carries NO
    # `proposal_experts`. This is the record shape that killed run 727d9db5.
    ladder = expand(
        parent,
        -7.5,
        fiber,
        np.random.default_rng(7),
        draws=40,
        multi_region=True,
        horizon=3,
        proposal_lane="shallow",
    )
    assert ladder, "the ladder stage produced nothing, so this test proves little"
    assert "proposal_experts" not in ladder[0], (
        "expand now supplies proposal_experts itself; this test's premise has moved"
    )
    records.extend(ladder)

    state = SearchState(archive={parent: -7.5}, budget=8, rounds=0)
    fresh = normalize_expansion_records(
        row for row in records if row["smiles"] not in state.archive
    )
    candidates = attach_features(fresh, state, fiber)
    assert candidates
    expert_census(candidates)
    selected = select_batch(
        candidates,
        ProgramValue(penalty=1.0),
        state,
        np.random.default_rng(1),
        round_index=1,
        batch=8,
        exploration=2,
        expert_floor_rounds=2,
    )
    assert selected, "select_batch returned nothing from a real expansion pool"

    # Every key run_cell subscripts off a row must be present on every selected row.
    required = _row_keys_run_cell_requires()
    assert "proposal_experts" in required, (
        "the key that killed run 727d9db5 is no longer required by run_cell; "
        "this guard would no longer catch it"
    )
    for row in selected:
        missing = sorted(required - set(row))
        assert not missing, f"selected expansion row is missing {missing}"

    # And the query rows and round lock must build and serialize.
    queries = [
        {
            "query_id": f"run_fa7_0_r001_q{index:02d}",
            "smiles": row["smiles"],
            "selection_kind": row["selection_kind"],
            "proposal_experts": row["proposal_experts"],
            "parent": row["parent"],
            "parent_score": row["parent_score"],
        }
        for index, row in enumerate(selected)
    ]
    assert queries
    json.dumps(queries, sort_keys=True)


def test_the_app_normalizes_expansion_records_before_selection():
    """The library fix is inert unless run_cell calls it on the expansion pool."""

    tree = ast.parse(_run_cell_source())
    bound = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "fresh"
            for target in node.targets
        )
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Name)
        and node.value.func.id == "normalize_expansion_records"
    ]
    assert bound, (
        "run_cell does not bind `fresh` to normalize_expansion_records(...); the "
        "expansion pool would reach select_batch without proposal_experts and the "
        "round would die building its query rows, discarding what it just found"
    )
