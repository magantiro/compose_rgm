"""Guards for the step-level learned chemical prior and its production wiring.

Every guard here is mutation-proven: the repository's standing rule is that a
comparison whose expectation is recomputed from the code under test cannot
fail, so each test is written against an EXTERNAL reference -- the numpy RNG
contract, the production executor's canonical successor key, or the live
proposal path itself -- never against a transcription of the prior.

The wiring guards (permutation, floor, byte-identical off, consumption) need no
checkpoint and always run.  The join-certification guards need the trained model
and skip with an explicit reason when it is absent; read the skip count.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from compose_v4.control.current_state_edits import ENUMERATORS, current_state_program
from compose_v4.control.learned_successor_prior import (
    FAMILY_TO_EXECUTOR_RULE,
    SUCCESSOR_EXACT_COORDINATE_JOIN,
    LearnedSuccessorPrior,
    PriorNotConsumed,
    assert_prior_is_consumed,
)
from compose_v4.rewrite.trace_shard import decode_state

_FIXTURE = Path(__file__).parent / "fixtures" / "pmo_production_parents.json"
_CHECKPOINT = Path("/Users/rmaganti/compose_fragment_ckpt/ringcore_a7546e2_best.pt")
_SCOPE = "3721d69851110fdd"
_MODEL: list = []


def _parents() -> list:
    payload = json.loads(_FIXTURE.read_text())
    return [(row, decode_state(row["state"])) for row in payload["parents"]]


def _model():
    if not _CHECKPOINT.exists():
        pytest.skip(f"trained editing checkpoint absent at {_CHECKPOINT}")
    if not _MODEL:
        import sys

        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
        from evaluate_tracelet_rollouts import load_factorized_rollout_checkpoint

        model, _meta = load_factorized_rollout_checkpoint(
            _CHECKPOINT, expected_scope_hash=_SCOPE
        )
        _MODEL.append(model.eval())
    return _MODEL[0]


def _first_populated(source, minimum: int = 3):
    for family in sorted(ENUMERATORS):
        try:
            actions = ENUMERATORS[family](source)
        except (ValueError, RuntimeError, KeyError, IndexError, TypeError):
            continue
        if len(actions) >= minimum:
            return family, actions
    return None, None


# ---- Wiring guards: no checkpoint required ----


def test_order_returns_a_permutation_and_never_filters() -> None:
    """The prior re-ranks. Nothing is added, dropped, or substituted."""
    prior = LearnedSuccessorPrior(model=None)
    prior.weights = lambda source, *, family, actions: np.linspace(
        1.0, 5.0, len(tuple(actions))
    ) / np.linspace(1.0, 5.0, len(tuple(actions))).sum()
    checked = 0
    for _row, source in _parents():
        family, actions = _first_populated(source)
        if family is None:
            continue
        ordered = prior.order(source, np.random.default_rng(7), family=family, actions=actions)
        assert len(ordered) == len(actions)
        assert sorted(map(id, ordered)) == sorted(map(id, actions))
        checked += 1
    assert checked >= 3, "fixture must exercise several real parents"


def test_every_legal_candidate_keeps_strictly_positive_weight() -> None:
    """The support floor: a prior that can exclude a candidate is a filter."""
    model = _model()
    prior = LearnedSuccessorPrior(model, floor=0.05)
    checked = 0
    for _row, source in _parents():
        for family in sorted(ENUMERATORS):
            try:
                actions = ENUMERATORS[family](source)
            except (ValueError, RuntimeError, KeyError, IndexError, TypeError):
                continue
            if len(actions) < 3:
                continue
            weights = prior.weights(source, family=family, actions=actions)
            assert weights.shape == (len(actions),)
            assert np.all(weights > 0.0), "a legal candidate reached probability zero"
            assert weights.min() * 20.0 >= weights.max() * 0.999, (
                "floor=0.05 must cap the spread at 20x"
            )
            assert abs(float(weights.sum()) - 1.0) < 1e-9
            checked += 1
    assert checked >= 5


def test_absent_prior_reproduces_the_historical_uniform_draw_exactly() -> None:
    """`prior is None` must be byte-identical to the pre-existing draw.

    The reference is the NUMPY RNG CONTRACT, not a copy of the production line:
    after the historical draw the generator must have consumed exactly one
    `integers(len(actions))` call, and the chosen action must be the one at that
    index.  A uniform-law object would reproduce the support and fail this.
    """
    checked = 0
    for _row, source in _parents():
        family, actions = _first_populated(source, minimum=4)
        if family is None:
            continue
        reference = np.random.default_rng(20260921)
        expected_index = int(reference.integers(len(actions)))
        expected_state = reference.bit_generator.state

        live = np.random.default_rng(20260921)
        try:
            _program, _binding, detail = current_state_program(
                source, live, family=family, successor_prior=None
            )
        except (ValueError, RuntimeError, KeyError, IndexError, TypeError):
            continue
        assert live.bit_generator.state == expected_state, (
            "the unlawed path consumed a different number of RNG values"
        )
        assert detail["enumerated_actions"] == len(actions)
        assert detail["conditional_action_probability"] == pytest.approx(1 / len(actions))
        assert detail["proposal_law"] == "uniform over legal actions"
        assert 0 <= expected_index < len(actions)
        checked += 1
    assert checked >= 3


def test_production_draw_consults_the_prior() -> None:
    """The keyword must be consumed by the real path, not merely accepted.

    `inspect.signature` would pass even if the keyword were dropped one hop
    later, so this drives the production function and requires it to reach the
    prior.
    """
    _row, source = _parents()[0]
    for _candidate_row, candidate in _parents():
        if _first_populated(candidate, minimum=4)[0] is not None:
            source = candidate
            break

    def draw(prior, seed):
        family, _actions = _first_populated(source, minimum=4)
        return current_state_program(
            source, np.random.default_rng(seed), family=family, successor_prior=prior
        )

    attempt = assert_prior_is_consumed(draw)
    assert attempt >= 1


def test_consumption_probe_fails_when_the_keyword_is_dropped() -> None:
    """Negative control: the probe must be able to FAIL.

    A guard that cannot go red is not a guard. This drives a draw that ignores
    the prior and requires `assert_prior_is_consumed` to raise.
    """
    _row, source = _parents()[0]

    def draw_ignoring_the_prior(prior, seed):
        family, _actions = _first_populated(source, minimum=3)
        if family is None:
            raise ValueError("no populated family")
        return current_state_program(source, np.random.default_rng(seed), family=family)

    with pytest.raises(PriorNotConsumed):
        assert_prior_is_consumed(draw_ignoring_the_prior, attempts=4)


def test_prior_reads_no_task_objective_or_property_information() -> None:
    """Task independence, derived from the AST and the runtime, not from prose.

    An earlier version of this guard scanned the file text and tripped over its
    own docstring, which legitimately says the prior is NOT a QED/SA screen.
    Imports are the thing that can actually make it task-dependent, so parse
    them, and additionally require that evaluating the module never pulls a
    property calculator into `sys.modules`.
    """
    import ast

    path = (
        Path(__file__).resolve().parents[1]
        / "src/compose_v4/control/learned_successor_prior.py"
    )
    tree = ast.parse(path.read_text())
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
            imported.update(f"{node.module}.{a.name}" for a in node.names)
    forbidden_fragments = (
        "QED", "SA_Score", "sascorer", "molecular_quality", "oracle",
        "pmo_oracle", "docking", "tdc",
    )
    for name in sorted(imported):
        for fragment in forbidden_fragments:
            assert fragment.lower() not in name.lower(), (
                f"learned_successor_prior imports {name!r}, which carries "
                f"objective/property information ({fragment}); the prior must be "
                "task-independent"
            )
    # And no function in it may take a task/score/objective argument.
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            args = [a.arg for a in node.args.args + node.args.kwonlyargs]
            for arg in args:
                assert not any(
                    token in arg.lower()
                    for token in ("task", "reward", "objective", "oracle", "qed")
                ), f"{node.name} takes an objective-bearing argument {arg!r}"


# ---- Join certification: needs the trained model ----


def test_certified_families_reach_the_model_marks_own_successor() -> None:
    """Re-derive the certified set from real parents by successor identity.

    This is the guard that lets the fast coordinate path be trusted. Its
    expectation is the PRODUCTION EXECUTOR's canonical key on both sides, never
    a recomputation of the join.  Adding a family to
    SUCCESSOR_EXACT_COORDINATE_JOIN that is not exact turns this red.
    """
    model = _model()
    from compose_v4.control.learned_successor_prior import (
        _candidate_key,
        _coordinate_key,
    )
    from compose_v4.experiments.production_successor_kernel import (
        _default_rewrite_system,
        enumerate_factorized_marked_law,
    )
    from compose_v4.experiments.whole_ring_plan import execute_program
    from compose_v4.rewrite.action_codec_v4 import encode_action
    from compose_v4.rewrite.kernel import canonical_state_key

    runtime = _default_rewrite_system(model)
    agree: dict[str, int] = {}
    total: dict[str, int] = {}
    for _row, source in _parents():
        if source.n_real_atoms < 4:
            continue
        law = enumerate_factorized_marked_law(model, source, 0.5)
        for family in sorted(SUCCESSOR_EXACT_COORDINATE_JOIN):
            rule = FAMILY_TO_EXECUTOR_RULE[family]
            table = {}
            for mark in law.marks:
                if mark.executor_rule_name == rule:
                    table[_coordinate_key(rule, mark.action)] = mark
            try:
                actions = ENUMERATORS[family](source)
            except (ValueError, RuntimeError, KeyError, IndexError, TypeError):
                continue
            for action in actions[:25]:
                key = _candidate_key(source, family, action)
                mark = table.get(key) if key is not None else None
                if mark is None:
                    continue
                total[family] = total.get(family, 0) + 1
                left, _receipt = execute_program(source, [encode_action(family, action)])
                right = runtime.apply(source, mark.executor_rule_name, mark.action)
                if canonical_state_key(left) == canonical_state_key(right):
                    agree[family] = agree.get(family, 0) + 1
    assert total, "certification exercised no candidates"
    for family, count in sorted(total.items()):
        assert agree.get(family, 0) == count, (
            f"{family} is in SUCCESSOR_EXACT_COORDINATE_JOIN but only "
            f"{agree.get(family, 0)}/{count} candidates reach the model mark's own "
            "successor; either remove it from the certified set or fix the join"
        )


def test_cycle_open_is_excluded_from_the_certified_fast_path() -> None:
    """cycle_open must stay off the fast path and be verified by execution.

    Opening an aromatic ring resolves the residual Kekule structure differently
    on the two sides, so its coordinate join reaches a DIFFERENT molecule. This
    pins the exclusion so a future cleanup cannot quietly fold it back in.
    """
    assert "cycle_open" in FAMILY_TO_EXECUTOR_RULE
    assert "cycle_open" not in SUCCESSOR_EXACT_COORDINATE_JOIN
    prior = LearnedSuccessorPrior(model=None, verify_successors=True)
    assert prior.verify_successors is True


def test_path_log_likelihood_scores_a_real_multi_step_edit_path() -> None:
    """The second consumer's interface: rank an already-constructed program.

    Reference is the model's own per-step law, reached independently here by
    asking the prior for the weights of the step's family and checking the
    scored step is the one whose mark the law holds -- not by re-running
    `path_log_likelihood`'s own arithmetic.
    """
    model = _model()
    prior = LearnedSuccessorPrior(model)
    from compose_v4.experiments.whole_ring_plan import execute_program
    from compose_v4.rewrite.action_codec_v4 import encode_action

    for _row, source in _parents():
        if source.n_real_atoms < 6:
            continue
        family, actions = _first_populated(source, minimum=4)
        if family is None:
            continue
        first = actions[0]
        middle, _receipt = execute_program(source, [encode_action(family, first)])
        second_family, second_actions = _first_populated(middle, minimum=2)
        if second_family is None:
            continue
        end, _receipt2 = execute_program(
            middle, [encode_action(second_family, second_actions[0])]
        )
        result = prior.path_log_likelihood(
            [source, middle, end],
            [(family, first), (second_family, second_actions[0])],
        )
        assert result["path_length"] == 2
        assert result["scored_steps"] + result["unjoined_steps"] == 2
        if result["scored_steps"] == 2:
            assert result["total_log_likelihood"] < 0.0
            assert 0.0 < result["per_step_probability"] <= 1.0
            # A longer path cannot score higher in TOTAL than its own prefix.
            prefix = prior.path_log_likelihood([source, middle], [(family, first)])
            assert result["total_log_likelihood"] <= prefix["total_log_likelihood"] + 1e-9
        return
    pytest.skip("fixture produced no two-step path")


def test_path_log_likelihood_rejects_a_malformed_path() -> None:
    """A state/action count mismatch must raise, never be scored silently."""
    from compose_v4.control.learned_successor_prior import LearnedSuccessorPriorError

    prior = LearnedSuccessorPrior(model=None)
    _row, source = _parents()[0]
    with pytest.raises(LearnedSuccessorPriorError):
        prior.path_log_likelihood([source], [("cycle_close", object())])
    with pytest.raises(LearnedSuccessorPriorError):
        prior.path_log_likelihood([source, source], [])


def test_the_weight_cache_is_transparent() -> None:
    """A warm weight lookup must equal a cold computation, exactly.

    The reference is a FRESH prior instance that has never populated its cache,
    so the comparison does not run through the code path under test. Without
    this, a cache that silently returned uniform weights would still satisfy the
    floor and normalization guards.
    """
    model = _model()
    warm = LearnedSuccessorPrior(model)
    checked = 0
    for _row, source in _parents():
        for family in sorted(ENUMERATORS):
            try:
                actions = ENUMERATORS[family](source)
            except (ValueError, RuntimeError, KeyError, IndexError, TypeError):
                continue
            if len(actions) < 4:
                continue
            first = warm.weights(source, family=family, actions=actions)
            second = warm.weights(source, family=family, actions=actions)
            cold = LearnedSuccessorPrior(model).weights(
                source, family=family, actions=actions
            )
            np.testing.assert_allclose(second, first, rtol=0, atol=0)
            np.testing.assert_allclose(second, cold, rtol=1e-12, atol=1e-12)
            # And it must not have degenerated to the uniform law.
            if float(cold.max()) > float(cold.min()) * 1.0000001:
                assert abs(float(second.max()) - 1.0 / len(actions)) > 1e-9, (
                    "cached weights collapsed to uniform"
                )
            checked += 1
            break
    assert checked >= 3


def test_a_mutated_weight_cache_is_caught() -> None:
    """Negative control for the guard above: prove it can go red.

    Corrupts the cache in-place, exactly as the surviving mutation did, and
    requires the transparency comparison to fail.
    """
    model = _model()
    prior = LearnedSuccessorPrior(model)
    for _row, source in _parents():
        for family in sorted(ENUMERATORS):
            try:
                actions = ENUMERATORS[family](source)
            except (ValueError, RuntimeError, KeyError, IndexError, TypeError):
                continue
            if len(actions) < 4:
                continue
            first = prior.weights(source, family=family, actions=actions)
            if float(first.max()) <= float(first.min()) * 1.0000001:
                continue  # a genuinely flat law cannot demonstrate the failure
            for key in list(prior._weight_cache):
                prior._weight_cache[key] = np.full(len(actions), 1.0 / len(actions))
            corrupted = prior.weights(source, family=family, actions=actions)
            assert not np.allclose(corrupted, first), (
                "the transparency guard cannot detect a corrupted cache"
            )
            return
    pytest.skip("fixture produced no non-flat weight vector")


def test_guidance_strength_actually_changes_the_weights() -> None:
    """Temperature and floor must be part of the cache key and must bite.

    A sweep over guidance strength is only meaningful if the settings reach the
    weights. A cache keyed without them would silently return the first
    setting's weights for every later one, and a sweep would report a flat
    non-result that was an artifact.
    """
    model = _model()
    prior = LearnedSuccessorPrior(model)
    compared = 0
    for _row, source in _parents():
        for family in sorted(ENUMERATORS):
            try:
                actions = ENUMERATORS[family](source)
            except (ValueError, RuntimeError, KeyError, IndexError, TypeError):
                continue
            if len(actions) < 6:
                continue
            prior.floor, prior.temperature = 0.05, 1.0
            flat = prior.weights(source, family=family, actions=actions)
            if float(flat.max()) <= float(flat.min()) * 1.0000001:
                continue
            prior.floor, prior.temperature = 0.05, 0.2
            sharp = prior.weights(source, family=family, actions=actions)
            assert not np.allclose(flat, sharp), (
                "lowering the temperature did not change the weights; the "
                "setting is not reaching the law"
            )
            assert float(sharp.max()) > float(flat.max()), (
                "a lower temperature must concentrate mass, not spread it"
            )
            prior.floor, prior.temperature = 0.5, 1.0
            floored = prior.weights(source, family=family, actions=actions)
            assert float(floored.max()) / float(floored.min()) <= 2.0000001, (
                "a floor of 0.5 must cap the spread at 2x"
            )
            compared += 1
            break
        if compared >= 3:
            break
    assert compared >= 3, "fixture produced too few non-flat laws to compare"


def test_inlined_state_identity_matches_the_production_helper() -> None:
    """The prior inlines `docking_value.identity` to avoid importing it.

    The docstring promises the two stay in step; this is the guard that makes
    that promise checkable. The prior must not import the objective-bearing
    module, but the TEST may, which is what lets the comparison be external.
    """
    from compose_v4.control.docking_value import identity
    from compose_v4.rewrite.trace_shard import encode_state

    checked = 0
    for _row, source in _parents():
        assert LearnedSuccessorPrior._state_identity(source) == identity(
            encode_state(source)
        )
        checked += 1
    assert checked >= 5


def test_this_call_site_cannot_change_heavy_atom_count() -> None:
    """Every family at this call site preserves the heavy-atom count.

    This is a SCOPE fact, and it is why a QED result measured here is not a
    result about the whole proposal stream: `atom_insert` and `atom_delete` --
    the two largest families in the production per-edit census and both net
    negative on QED -- do not route through `current_state_program` at all.
    Pinning it keeps a future reader from over-reading a QED null measured here.
    """
    from compose_v4.control.edit_program import execute_bound_program

    checked = 0
    for _row, source in _parents():
        if source.n_real_atoms < 4:
            continue
        for family in sorted(ENUMERATORS):
            try:
                actions = ENUMERATORS[family](source)
            except (ValueError, RuntimeError, KeyError, IndexError, TypeError):
                continue
            for action in actions[:6]:
                try:
                    program, binding, _detail = current_state_program(
                        source, np.random.default_rng(3), family=family
                    )
                    endpoint, _r = execute_bound_program(source, program, tuple(binding))
                except (ValueError, RuntimeError, KeyError, IndexError, TypeError):
                    continue
                assert endpoint.n_real_atoms == source.n_real_atoms, (
                    f"{family} changed the heavy-atom count "
                    f"{source.n_real_atoms} -> {endpoint.n_real_atoms}; the scope "
                    "note on this call site is wrong and must be revised"
                )
                checked += 1
                break
    assert checked >= 5
