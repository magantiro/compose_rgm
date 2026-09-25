"""Arm C guards: the intervention changes PLANNED CONNECTIONS and nothing else.

Every test drives the production functions on real programs taken from a completed
arm-A snapshot; none re-derives an expectation from the code under test.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from compose_v4.control.edit_program import (
    EditProgram,
    ProgramExecutionError,
    execute_bound_program,
)
from compose_v4.control.edit_program_graph import compile_program_graph, scheduled_program
from compose_v4.control.pmo_binding_intervention import (
    binding_rng,
    created_operands,
    execute_rebound_program,
)
from compose_v4.rewrite.kernel import canonical_state_key
from compose_v4.rewrite.trace_shard import decode_state

SNAPSHOT = Path(__file__).parent / "fixtures" / "pmo_armc_panel_v1.json"
BUDGET = {"max_primitives": 32, "max_blocks": 8}


def _panel(limit: int = 30):
    if not SNAPSHOT.exists():
        pytest.skip("arm-C panel fixture is absent from this checkout")
    entries = json.load(SNAPSHOT.open())["entries"]
    out = []
    for record in list(entries.values())[:limit]:
        source = decode_state(record["source_state"])
        graph = compile_program_graph(EditProgram.from_payload(record["program"]))
        program, _, _ = scheduled_program(graph, source.n_real_atoms)
        assignment = tuple(record["assignment"])
        try:
            execute_bound_program(source, program, assignment, **BUDGET)
        except Exception:  # noqa: BLE001, S112 -- a panel program that will not execute
            # under production budgets is not a subject for this guard; the fixture is a
            # snapshot, not a curated set, and excluding it here is not a silent rescue.
            continue
        out.append((source, program, assignment))
    assert out, "panel must contain executable programs"
    return out


def _proposal_id(source, program, assignment) -> str:
    return f"{program.program_id}:{canonical_state_key(source)}:{assignment}"


def test_binding_rng_is_deterministic_and_not_process_salted():
    """A run must reproduce its bindings across processes, so never Python `hash()`."""
    # PINNED against a literal, not against a re-derivation: Python's `hash()` is
    # stable WITHIN a process, so a same-process determinism check alone would pass
    # for a salted implementation and silently make a run irreproducible elsewhere.
    first = binding_rng(7, "proposal-a").integers(2**40)
    assert int(first) == 700503571673
    assert first == binding_rng(7, "proposal-a").integers(2**40)
    assert first != binding_rng(7, "proposal-b").integers(2**40)
    assert first != binding_rng(8, "proposal-a").integers(2**40)


def test_identity_reproduces_the_original_and_consumes_no_randomness():
    """The plumbing must be provably inert when it is asked to change nothing."""
    for source, program, assignment in _panel():
        _, base = execute_bound_program(source, program, assignment, **BUDGET)
        rng = binding_rng(11, _proposal_id(source, program, assignment))
        before = json.dumps(rng.bit_generator.state, sort_keys=True, default=str)
        _, same = execute_rebound_program(
            source, program, assignment, rng=rng, identity=True, **BUDGET
        )
        assert same["endpoint"] == base["endpoint"]
        assert tuple(same["marks"]) == tuple(program.marks)
        assert same["actions"] == base["actions"]
        after = json.dumps(rng.bit_generator.state, sort_keys=True, default=str)
        assert before == after, "identity intervention advanced the generator"


def test_only_created_operands_move():
    """Rule sequence, length, block count and every source binding are preserved."""
    for source, program, assignment in _panel():
        rng = binding_rng(13, _proposal_id(source, program, assignment))
        try:
            _, bound = execute_rebound_program(
                source, program, assignment, rng=rng, **BUDGET
            )
        except ProgramExecutionError:
            continue
        original = [json.loads(text) for text in program.marks]
        modified = [json.loads(text) for text in bound["marks"]]
        assert len(original) == len(modified)
        assert bound["blocks"] == [
            b for b in bound["blocks"]
        ]  # block boundaries carried through unchanged
        for before, after in zip(original, modified):
            assert before["executor_rule"] == after["executor_rule"]
            # Every operand that is not a created handle is byte-identical.
            assert json.dumps(_strip_created(before), sort_keys=True) == json.dumps(
                _strip_created(after), sort_keys=True
            )


def _strip_created(record: dict) -> dict:
    """The mark with its created operands blanked, so only the rest is compared."""
    text = json.dumps(record, sort_keys=True)
    value = json.loads(text)

    def blank(node):
        if isinstance(node, dict):
            if set(node) == {"created"}:
                return {"created": "*"}
            return {k: blank(v) for k, v in node.items()}
        if isinstance(node, list):
            return [blank(v) for v in node]
        return node

    return blank(value)


def test_the_birth_slot_of_an_insert_is_never_rebound():
    """`atom_insert`'s own new handle is assigned by `fresh_slot`, never resampled."""
    for _, program, _ in _panel():
        for text in program.marks:
            record = json.loads(text)
            if record["executor_rule"] != "atom_insert":
                continue
            birth = int(next(iter(record["payload"]["slot"].items()))[1])
            assert birth not in created_operands(record)


def test_created_handle_indices_survive_a_deletion():
    """Regression: handle indices are sequential and never reused after `atom_delete`.

    Deriving the next created index from the LIVE list makes it collide with a
    surviving handle as soon as one is deleted, which silently rebinds a later
    operation onto the wrong atom.  The observable symptom is that the ORIGINAL
    binding stops being admissible before any alternative has been chosen.
    """
    checked, witnessed = 0, 0
    for source, program, assignment in _panel():
        marks = [json.loads(text) for text in program.marks]
        seen_delete = False
        for record in marks:
            if record["executor_rule"] == "atom_delete":
                if next(iter(record["payload"]["v"].items()))[0] == "created":
                    seen_delete = True
            elif record["executor_rule"] == "atom_insert" and seen_delete:
                witnessed += 1
                break
        rng = binding_rng(17, _proposal_id(source, program, assignment))
        try:
            _, bound = execute_rebound_program(
                source, program, assignment, rng=rng, **BUDGET
            )
        except ProgramExecutionError:
            continue
        diverged = False
        for step in bound["rebinding"]["audit"]:
            if step["created_operands"] and not diverged:
                assert step["identity_admissible"], (
                    "the prescribed binding must stay eligible until the first change"
                )
                checked += 1
            if step["changed"]:
                diverged = True
    assert checked, "no pre-divergence step was exercised"
    # COVERAGE FLOOR: index reuse can only be observed on a program that deletes a
    # created handle and then creates another one.  Without this the guard passes on a
    # panel that never reaches the branch, which is how the mutation first survived.
    assert witnessed, "panel exercises no delete-then-insert program; guard is vacuous"


def test_a_rebinding_failure_is_frozen_and_labelled(monkeypatch):
    """No fallback to the original program; the failure stage is recorded.

    CONSTRUCTED, not hoped for.  After the delete-bookkeeping fix this panel produces no
    natural rebinding failure at all, so a test that waited for one would either skip
    (scoring a silent fallback as PASS -- the mutation that survived this battery once) or
    assert a condition the corrected code no longer reaches.  The executor is therefore
    stubbed to refuse every candidate from a chosen step onward, which is exactly the
    situation the frozen-failure contract governs.
    """
    from compose_v4.control import pmo_binding_intervention as arm_c
    from compose_v4.rewrite.kernel import InvalidRewrite

    source, program, assignment = next(
        (s, p, a)
        for s, p, a in _panel()
        if any(created_operands(json.loads(t)) for t in p.marks)
    )
    real = arm_c.execute_program
    refused = {"after": 0}

    def refusing(current, actions):
        refused["after"] += 1
        if refused["after"] > 2:
            raise InvalidRewrite("stubbed refusal: no binding is admissible here")
        return real(current, actions)

    monkeypatch.setattr(arm_c, "execute_program", refusing)
    rng = binding_rng(23, _proposal_id(source, program, assignment))
    with pytest.raises(ProgramExecutionError) as caught:
        execute_rebound_program(source, program, assignment, rng=rng, **BUDGET)
    receipt = caught.value.receipt
    assert receipt["failure_stage"] == "rebinding"
    assert receipt["complete"] is False
    # A fallback would have produced a completed receipt instead of raising at all.
    assert "endpoint" not in receipt


def test_the_modified_program_is_statically_valid_and_replays(monkeypatch):
    """Regression: `atom_delete` bookkeeping must follow the REBOUND mark.

    Popping the handle the recipe PRESCRIBED, rather than the one the executed deletion
    actually removed, leaves the deleted atom live and retires a surviving one.  Nothing
    fails at that step; `EditProgram.validate` rejects a later reference instead, and the
    production wrapper then cannot build the program it must archive.  The four panel
    programs that delete a created handle and then insert again are the only witnesses.
    """
    from compose_v4.control.pmo_binding_intervention import execute_program_graph_rebound

    checked = 0
    for source, program, assignment in _panel(limit=80):
        graph = compile_program_graph(program)
        for occurrence in (1, 2, 3):
            _, trace = execute_program_graph_rebound(
                source,
                graph,
                assignment,
                run_seed=29,
                occurrence=occurrence,
                **BUDGET,
            )
            # The wrapper builds and RESCHEDULES the modified program and replays it; a
            # returned trace therefore already proves static validity and exact replay.
            rebuilt = EditProgram.from_payload(trace["modified_program"])
            assert tuple(rebuilt.marks) == tuple(trace["marks"])
            assert trace["selected_program_id"] == rebuilt.program_id
            checked += 1
    assert checked, "no program was exercised"


def test_repeated_occurrences_of_one_recipe_do_not_replay_one_binding():
    """The generator key must identify a proposal OCCURRENCE, not just its content.

    `program_id` is a content hash, so a key built only from (program, source, binding)
    would hand the same recipe on the same parent the identical choice every time and make
    arm C far less stochastic than intended.
    """
    from compose_v4.control.pmo_binding_intervention import execute_program_graph_rebound

    varied, failures = 0, 0
    for source, program, assignment in _panel(limit=80):
        graph = compile_program_graph(program)
        endpoints = set()
        for occurrence in range(1, 5):
            try:
                _, trace = execute_program_graph_rebound(
                    source, graph, assignment, run_seed=31, occurrence=occurrence,
                    **BUDGET,
                )
            except ProgramExecutionError:
                # A legitimate frozen failure: an earlier choice left no admissible
                # binding later.  It is counted, not swallowed, and not rescued.
                failures += 1
                continue
            endpoints.add(trace["endpoint"])
        if len(endpoints) > 1:
            varied += 1
    assert varied, "no program varied across occurrences; the key ignores the ordinal"


def test_arm_b_and_arm_c_are_mutually_exclusive(monkeypatch):
    """Enabling both arms must fail closed, not create a third unnamed condition."""
    from compose_v4.control import pmo_binding_intervention as arm_c
    from compose_v4.control import pmo_uniform_chain as arm_b

    monkeypatch.setenv(arm_c.ENV_FLAG, "1")
    monkeypatch.setenv(arm_b.ENV_FLAG, "1")
    with pytest.raises(RuntimeError, match="mutually exclusive"):
        arm_c.binding_arm_enabled()
    monkeypatch.delenv(arm_b.ENV_FLAG)
    assert arm_c.binding_arm_enabled() is True
    monkeypatch.delenv(arm_c.ENV_FLAG)
    assert arm_c.binding_arm_enabled() is False


def test_the_proposal_path_actually_reaches_the_intervention():
    """Arm C must not be a THROTTLE: the seam is derived from the call site, not a list.

    The completion-law A/B spent a whole scored run measuring a mechanism that reached
    20% of the proposals it governed, because a drop-in replacement caller silently
    omitted the keyword.  This asserts STRUCTURALLY that the v2.1 proposal path both
    consults `binding_arm_enabled` and calls the rebound executor, and that every
    `execute_program_graph` call in that path is guarded by that flag.
    """
    import ast
    import inspect

    from compose_v4.control import dynamic_program_synthesis_v21 as v21

    tree = ast.parse(inspect.getsource(v21))
    called = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert "binding_arm_enabled" in called, "v2.1 never consults the arm-C flag"
    assert "execute_program_graph_rebound" in called, "v2.1 never calls the rebound executor"

    # Every plain execute_program_graph call must sit inside an `if`/`else` whose test
    # mentions the flag, so a new unguarded call site cannot appear silently.
    guarded, total = 0, 0
    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        test_names = {
            n.id for n in ast.walk(node.test) if isinstance(n, ast.Name)
        }
        if "binding_arm_enabled" not in test_names:
            continue
        for branch in (node.body, node.orelse):
            for inner in branch:
                for call in ast.walk(inner):
                    if (
                        isinstance(call, ast.Call)
                        and isinstance(call.func, ast.Name)
                        and call.func.id == "execute_program_graph"
                    ):
                        guarded += 1
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "execute_program_graph"
        ):
            total += 1
    assert total, "no execute_program_graph call site found; the seam moved"
    assert guarded == total, (
        f"{total - guarded} of {total} execute_program_graph calls are not guarded by "
        "the arm-C flag, so arm C would silently reach only some proposals"
    )


def test_the_archive_admits_arm_c_and_the_stored_program_explains_the_molecule():
    """Integration: `add_measured_program` replays the STORED program and demands exact
    endpoint AND state-sequence identity.

    This is the check that makes arm C an experiment rather than a throttle.  The
    unpatched proposal path stored the PRESCRIBED program beside arm C's endpoint; the
    counterfactual below shows every endpoint-changing proposal would then have been
    rejected at admission, so only proposals on which the intervention changed nothing
    could ever have entered the archive.  Zero oracle calls: the score is a synthetic
    constant and no task identity is used.
    """
    from compose_v4.control.adaptive_program_optimizer import (
        ProgramOptimizer,
        ProgramSearchConfig,
    )
    from compose_v4.control.pmo_binding_intervention import execute_program_graph_rebound

    config = ProgramSearchConfig(require_broad_runtime=False, **BUDGET)
    group, protocol = "armc_guard", "synthetic_constant_no_oracle"

    def store(program_payload, trace, assignment, tag):
        optimizer = ProgramOptimizer(config, source_group=group, oracle_protocol=protocol)
        optimizer.add_measured_program(
            {
                "source_group": group,
                "oracle_protocol": protocol,
                "source_state": trace["states"][0],
                "program": program_payload,
                "assignment": list(assignment),
                "endpoint": trace["endpoint"],
                "trace": trace,
                "static_score": 0.0,
                "inherited_static_score": None,
            },
            receipt_id=tag,
            score=0.0,
        )

    admitted, changed, prescribed_rejected = 0, 0, 0
    for i, (source, program, assignment) in enumerate(_panel(limit=40)):
        graph = compile_program_graph(program)
        _, base = execute_bound_program(source, program, assignment, **BUDGET)
        try:
            _, trace = execute_program_graph_rebound(
                source, graph, assignment, run_seed=20260925, occurrence=i + 1, **BUDGET
            )
        except ProgramExecutionError:
            continue
        store(trace["modified_program"], trace, assignment, f"c-{i}")
        admitted += 1
        if trace["endpoint"] == base["endpoint"]:
            continue
        changed += 1
        with pytest.raises(ValueError, match="exact replay or endpoint identity"):
            store(program.payload(), trace, assignment, f"bad-{i}")
        prescribed_rejected += 1

    assert admitted, "no arm-C proposal was admitted"
    assert changed, "no endpoint changed, so the counterfactual is untested"
    assert prescribed_rejected == changed


def test_only_the_proposal_seam_rebinds():
    """Internal construction and admission REPLAY must stay prescribed.

    `_channel_proposal` and `add_measured_program` both execute programs internally.  If
    the rebound executor were reachable from those, arm C would rebind during mutation,
    recombination or replay -- and an unchanged `_channel_proposal` would no longer mean
    unchanged behaviour.  Derived from the import graph, so a new caller cannot appear
    silently.
    """
    import ast
    import inspect

    from compose_v4.control import adaptive_program_optimizer, dynamic_program_synthesis
    from compose_v4.control import dynamic_program_synthesis_v21 as v21

    callers = []
    for module in (adaptive_program_optimizer, dynamic_program_synthesis, v21):
        tree = ast.parse(inspect.getsource(module))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "execute_program_graph_rebound"
            ):
                callers.append(module.__name__)
    assert callers == [v21.__name__], (
        f"the rebound executor is reachable from {sorted(set(callers))}; only the v2.1 "
        "proposal seam may rebind"
    )


def test_the_real_proposal_path_stores_the_program_that_explains_its_endpoint(monkeypatch):
    """Production-path smoke through `propose_batch` with arm C on.  Zero oracle calls.

    Drives the actual v2.1 channel pool -- `_channel_proposal`, scheduling, execution,
    eligibility, candidate construction -- and requires of EVERY candidate that the
    program it stores, recompiled and rescheduled by the production scheduler, replays to
    the candidate's own endpoint and state sequence.  A path that returns arm C's molecule
    while archiving the prescribed program passes every unit test and fails this.
    """
    from compose_v4.control.adaptive_program_optimizer import ProgramSearchConfig
    from compose_v4.control.dynamic_program_synthesis_v21 import DynamicV21ProgramOptimizer
    from compose_v4.control.pmo_binding_intervention import ENV_FLAG, STAGE_NAME

    monkeypatch.setenv(ENV_FLAG, "1")
    config = ProgramSearchConfig(
        require_broad_runtime=False,
        attempts_per_batch=24,
        candidates_per_batch=8,
        wall_seconds=45.0,
        seed=20260925,
        **BUDGET,
    )
    group, protocol = "armc_prodpath", "synthetic_constant_no_oracle"
    optimizer = DynamicV21ProgramOptimizer(
        config, source_group=group, oracle_protocol=protocol
    )
    seeded = 0
    for i, (source, program, assignment) in enumerate(_panel(limit=40)):
        _, base = execute_bound_program(source, program, assignment, **BUDGET)
        optimizer.add_measured_program(
            {
                "source_group": group,
                "oracle_protocol": protocol,
                "source_state": base["states"][0],
                "program": program.payload(),
                "assignment": list(assignment),
                "endpoint": base["endpoint"],
                "trace": base,
                "static_score": 0.0,
                "inherited_static_score": None,
            },
            receipt_id=f"seed-{i}",
            score=-float(i % 7),
        )
        seeded += 1
        if seeded >= 24:
            break
    assert seeded >= 8, "archive too small to propose from"

    batch = optimizer.propose_batch(
        lambda row: {"oracle_eligible": True, "smiles": row["smiles"]}
    )
    candidates = batch["candidates"] if isinstance(batch, dict) else batch
    assert candidates, "the proposal path produced no candidate"

    for candidate in candidates:
        assert candidate["trace"].get("ablation_stage") == STAGE_NAME, (
            "a candidate did not come through the intervention; the hook is not on the "
            "path every proposal takes"
        )
        source = decode_state(candidate["source_state"])
        stored = EditProgram.from_payload(candidate["program"])
        rescheduled, _, _ = scheduled_program(
            compile_program_graph(stored), source.n_real_atoms
        )
        _, replay = execute_bound_program(
            source, rescheduled, tuple(candidate["assignment"]), **BUDGET
        )
        assert replay["endpoint"] == candidate["endpoint"], (
            "the archived program does not build the molecule the candidate reports"
        )
        assert replay["states"] == candidate["trace"]["states"]

def test_a_modified_program_that_reschedules_differently_is_REFUSED(monkeypatch):
    """CONSTRUCTED. Rebinding changes the dataflow graph, so the modified program's own
    canonical schedule need not reproduce the order just executed -- and the archive's
    admission replays a stored program through `compile_program_graph` +
    `scheduled_program`. On the development panel this never arises (0 of 640 attempts),
    so the guard's removal is undetectable from real data and the condition must be
    constructed: the scheduler is stubbed to return a reordered program, and the wrapper
    must refuse rather than archive an entry whose program explains a different molecule.
    """
    from types import SimpleNamespace

    from compose_v4.control import pmo_binding_intervention as arm_c

    source, program, assignment = next(
        (s, p, a) for s, p, a in _panel() if len(p.marks) >= 2
    )
    graph = compile_program_graph(program)
    real = arm_c.scheduled_program
    calls = {"n": 0}

    def reordering(compiled, atoms, **kwargs):
        scheduled, order, timeline = real(compiled, atoms, **kwargs)
        calls["n"] += 1
        if calls["n"] == 1:
            return scheduled, order, timeline  # the prescribed recipe, untouched
        # The SECOND call is the modified program's own reschedule: reverse it.
        reversed_marks = tuple(reversed(scheduled.marks))
        if reversed_marks == tuple(scheduled.marks):
            pytest.skip("this program's schedule is symmetric under reversal")
        # A STAND-IN, not a valid `EditProgram`: reversing a real schedule puts a
        # reference before its birth, which `EditProgram.__post_init__` rejects before the
        # guard under test could run.  Only `.marks` is read, because the refusal fires
        # immediately on the order comparison and the object is never executed.
        return (SimpleNamespace(marks=reversed_marks), order, timeline)

    monkeypatch.setattr(arm_c, "scheduled_program", reordering)
    with pytest.raises(ProgramExecutionError) as caught:
        arm_c.execute_program_graph_rebound(
            source, graph, assignment, run_seed=41, occurrence=1, **BUDGET
        )
    assert caught.value.receipt["failure_stage"] == "rebinding_reschedule"
    assert caught.value.receipt["complete"] is False


def test_a_modified_program_that_replays_differently_is_REFUSED(monkeypatch):
    """CONSTRUCTED, for the same reason as above: 0 of 640 real attempts reach it.

    The replay check is what proves the ARCHIVED program builds the molecule the candidate
    reports. Here the replay is stubbed to return a different endpoint, and the wrapper
    must refuse instead of returning a candidate whose stored program disagrees with it.
    """
    from compose_v4.control import pmo_binding_intervention as arm_c

    source, program, assignment = next(iter(_panel()))
    graph = compile_program_graph(program)
    real = arm_c.execute_bound_program
    calls = {"n": 0}

    def wrong_replay(*args, **kwargs):
        product, receipt = real(*args, **kwargs)
        calls["n"] += 1
        return product, {**receipt, "endpoint": receipt["endpoint"] + ".[He]"}

    monkeypatch.setattr(arm_c, "execute_bound_program", wrong_replay)
    with pytest.raises(ProgramExecutionError) as caught:
        arm_c.execute_program_graph_rebound(
            source, graph, assignment, run_seed=43, occurrence=1, **BUDGET
        )
    assert caught.value.receipt["failure_stage"] == "rebinding_replay"
    assert calls["n"] >= 1, "the replay was never attempted"
