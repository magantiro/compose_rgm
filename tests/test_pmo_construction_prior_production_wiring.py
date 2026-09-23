"""The construction prior must reach the SCORED PMO entry point, not just the library.

``tests/test_construction_prior_wiring.py`` proves the LIBRARY seam: a prior handed to
``DynamicProgramOptimizer`` reaches the construction draw.  That seam was complete,
mutation-tested and still completely inert in production, because no scored entry point
supplied it -- the same shape as the region law behind its opt-in keyword and
``donor_program`` outside the scored import closure.

These tests cover the five PRODUCTION hops that close it, and they are written to fail
for the specific reason each hop can break:

1. ``pmo_online_memory.memory_channel_proposal`` threads ``optimizer.construction_prior``.
   This is the hop that matters most and is the easiest to miss: that function
   INTERCEPTS the shallow lane whenever the online memory is warm, so a prior wired only
   into ``DynamicProgramOptimizer._mutate`` is bypassed exactly in the arm it is being
   measured in, on a fraction that shrinks as the memory warms.
2. ``PmoPopulationController.__init__`` builds the prior from a JSON spec.
3. ``PmoPopulationController.restore`` FORWARDS that spec -- an accepted-and-dropped
   arm parameter rebuilds the other arm from this arm's snapshot, silently.
4. ``pmo_population_v1.execute_task`` puts the spec into ``optimizer_kwargs``, and omits
   the key entirely when there is no prior, so OFF keeps the historical run identity.
5. The Modal worker reads the spec from the ARM'S OWN sealed contract payload and from
   nowhere else.

Hops 3-5 are checked by walking the real source AST rather than by executing a Modal
container or a 250-call campaign.  A signature check would pass on every defect this
file exists to catch, so the assertions are about what the code PASSES, never about
what it accepts.
"""

from __future__ import annotations

import ast
import inspect
import json
from pathlib import Path

import numpy as np
import pytest

from compose_v4.chem.molecular_graph import smiles_to_molecular_graph
from compose_v4.chem.state import pad_molecular_graph
from compose_v4.control.learned_successor_prior import (
    PriorNotConsumed,
    assert_prior_is_consumed,
)

ROOT = Path(__file__).resolve().parents[1]
SOURCE_SMILES = "CC(C)Cc1ccc(cc1)C(C)C(=O)O"


def _source():
    # 48 slots: the PMO proposal path pads to 48 and refuses anything else.
    return pad_molecular_graph(smiles_to_molecular_graph(SOURCE_SMILES), 48)


# ---- Hop 1: the warm-memory lane, driven for real ---------------------------


class _StubConfig:
    max_primitives = 32
    max_blocks = 8


class _WarmMemory:
    """The smallest object satisfying ``memory_channel_proposal``'s warm branch.

    ``region_law`` returns None, which is a real production state (a memory can be
    warm and still decline to produce a law) and keeps this test about the SUCCESSOR
    prior rather than about the region law.
    """

    warm = True

    def __init__(self):
        self.cost = {"syntheses": 0}

    def region_law(self):
        return _NullRegionLaw()


class _NullRegionLaw:
    """A region law that defers entirely, so only the successor prior is under test."""

    def order(self, source, rng, *, family, actions):
        return tuple(actions)


class _StubOptimizer:
    def __init__(self, rng, prior):
        self.shallow_rng = rng
        self.config = _StubConfig()
        self.construction_prior = prior


def _entry(source):
    from compose_v4.rewrite.trace_shard import encode_state

    return {"trace": {"states": [encode_state(source)]}}


def test_the_warm_memory_lane_threads_the_construction_prior():
    """The lane arm B actually takes must consult the prior.

    Driven through the REAL ``memory_channel_proposal`` with a warm memory, so this
    fails if the ``successor_prior=`` keyword is dropped from its synthesis call --
    which is precisely the state the code shipped in.
    """

    from compose_v4.control.pmo_online_memory import memory_channel_proposal

    source = _source()
    entry = _entry(source)

    def draw(prior, seed):
        rng = np.random.default_rng(np.random.SeedSequence([seed, 41]))
        optimizer = _StubOptimizer(rng, prior)

        def fallback(channel, entry_):  # must NOT be taken on the warm shallow lane
            raise AssertionError(
                "memory_channel_proposal delegated to the fallback on a warm shallow "
                "lane; this test would then be measuring the wrong function"
            )

        memory_channel_proposal(
            optimizer, "shallow_program_channel", entry, fallback, _WarmMemory()
        )

    assert assert_prior_is_consumed(draw, attempts=24) >= 1


def test_the_warm_memory_lane_consumption_probe_can_fail():
    """Negative control: the probe must report an unwired draw, not pass vacuously."""

    with pytest.raises(PriorNotConsumed):
        assert_prior_is_consumed(lambda prior, seed: None, attempts=3)


def test_the_cold_memory_lane_still_delegates_untouched():
    """A cold memory must reach the unmodified production fallback.

    If threading the prior had been implemented by taking the warm branch always, the
    arm-A no-op would be gone and every historical run's RNG stream would have moved.
    """

    from compose_v4.control.pmo_online_memory import memory_channel_proposal

    taken = []

    def fallback(channel, entry_):
        taken.append(channel)
        return ("source", "program", "binding", {})

    class _Cold:
        warm = False

    memory_channel_proposal(
        _StubOptimizer(np.random.default_rng(0), None),
        "shallow_program_channel",
        _entry(_source()),
        fallback,
        _Cold(),
    )
    assert taken == ["shallow_program_channel"]


# ---- Hops 2-3: the controller builds and forwards a SPEC --------------------


def _controller_ast():
    from compose_v4.control import pmo_population_controller

    return ast.parse(inspect.getsource(pmo_population_controller))


def _function(tree, class_name, method_name):
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for child in node.body:
                if isinstance(child, ast.FunctionDef) and child.name == method_name:
                    return child
    raise AssertionError(f"{class_name}.{method_name} not found")


def test_restore_forwards_the_construction_prior_spec_to_the_constructor():
    """Accepting an arm parameter is NOT passing it.

    ``run_program_campaign`` hands ``optimizer_kwargs`` to both the constructor and
    ``restore``, and rebuilds the optimizer from a snapshot once per already-complete
    round.  A ``restore`` that accepts the spec and drops it out of
    ``constructor_kwargs`` resumes the treated arm as the control arm with nothing in
    any artifact to show the swap -- strictly worse than the ``TypeError`` an
    unaccepted keyword raises.  A signature-parity check passes on exactly that defect,
    so this walks the forwarded KEYS.
    """

    restore = _function(_controller_ast(), "PmoPopulationController", "restore")
    forwarded = set()
    for node in ast.walk(restore):
        if isinstance(node, ast.keyword) and node.arg == "constructor_kwargs":
            assert isinstance(node.value, ast.Dict), "constructor_kwargs must be a literal"
            for key in node.value.keys:
                if isinstance(key, ast.Constant):
                    forwarded.add(key.value)
    assert "construction_prior_spec" in forwarded, (
        "PmoPopulationController.restore accepts construction_prior_spec but does not "
        f"forward it; it forwards {sorted(forwarded)}"
    )
    accepted = {arg.arg for arg in restore.args.kwonlyargs}
    assert "construction_prior_spec" in accepted


def test_the_controller_builds_a_prior_from_a_spec_and_refuses_two_sources():
    """The spec hop, driven through the real constructor with a stubbed builder."""

    from compose_v4.control import pmo_construction_prior, pmo_population_controller

    built = {}
    sentinel = object()

    def _fake_build(spec):
        built["spec"] = spec
        return sentinel

    original = pmo_construction_prior.build_construction_prior
    pmo_construction_prior.build_construction_prior = _fake_build
    try:
        payload = pmo_construction_prior.ConstructionPriorSpec(
            checkpoint_path="/nowhere/model.pt"
        ).payload()
        controller = _minimal_controller(
            pmo_population_controller, construction_prior_spec=payload
        )
        assert controller.construction_prior is sentinel
        assert built["spec"].checkpoint_sha256 == payload["checkpoint_sha256"]
        assert controller.construction_prior_spec == payload

        with pytest.raises(ValueError, match="OR as a spec"):
            _minimal_controller(
                pmo_population_controller,
                construction_prior=object(),
                construction_prior_spec=payload,
            )
    finally:
        pmo_construction_prior.build_construction_prior = original


def test_an_unflagged_controller_has_no_prior():
    """ABSENT is the only byte-identical off; a uniform law object is not one."""

    from compose_v4.control import pmo_population_controller

    controller = _minimal_controller(pmo_population_controller)
    assert controller.construction_prior is None
    assert controller.construction_prior_spec is None


def _minimal_controller(module, **kwargs):
    from compose_v4.control.adaptive_program_optimizer import ProgramSearchConfig

    config = ProgramSearchConfig.program_only_recipe(seed=7, score_direction="maximize")
    checkpoint = {"schema_version": module.CHECKPOINT_SCHEMA}
    return module.PmoPopulationController(
        config,
        source_group="test",
        oracle_protocol="test",
        hierarchy=None,
        jump_checkpoint=checkpoint,
        **kwargs,
    )


# ---- Hop 4: the scored entry point supplies it ------------------------------


def test_execute_task_puts_the_spec_in_optimizer_kwargs_and_omits_it_when_absent():
    """``optimizer_kwargs`` is what the controller is built from AND what is hashed.

    The key must be OMITTED rather than set to None when there is no prior: a present
    ``None`` would move ``optimizer_kwargs_sha256`` for every unflagged run, breaking
    comparability with every trajectory recorded before this seam existed.
    """

    from compose_v4.experiments import pmo_population_v1

    tree = ast.parse(inspect.getsource(pmo_population_v1))
    execute = next(
        node for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "execute_task"
    )
    accepted = {arg.arg for arg in execute.args.kwonlyargs}
    assert "construction_prior_spec" in accepted

    assigned_under_guard = False
    for node in ast.walk(execute):
        if not isinstance(node, ast.If):
            continue
        test_names = {
            n.id for n in ast.walk(node.test) if isinstance(n, ast.Name)
        }
        if "construction_prior_spec" not in test_names:
            continue
        for inner in ast.walk(node):
            if (
                isinstance(inner, ast.Subscript)
                and isinstance(inner.slice, ast.Constant)
                and inner.slice.value == "construction_prior_spec"
            ):
                assigned_under_guard = True
    assert assigned_under_guard, (
        "execute_task must add 'construction_prior_spec' to optimizer_kwargs only "
        "inside a guard on the argument being present, so an unflagged run's "
        "optimizer_kwargs_sha256 is unchanged"
    )

    # And it must actually be the dict handed to run_program_campaign.
    campaign_call = next(
        node for node in ast.walk(execute)
        if isinstance(node, ast.Call)
        and getattr(node.func, "id", None) == "run_program_campaign"
    )
    passed = {
        kw.arg: kw.value for kw in campaign_call.keywords
    }
    assert getattr(passed["optimizer_kwargs"], "id", None) == "optimizer_kwargs", (
        "run_program_campaign must receive the same optimizer_kwargs the guard "
        "populated, not a separately built literal"
    )


# ---- Hop 5: the worker reads it from the ARM'S OWN sealed payload -----------


def test_the_worker_reads_the_prior_only_from_the_arm_contract():
    """A caller must not be able to select a law the owner did not authorize.

    The arm's separately sealed payload is the single source, exactly as it already is
    for ``enable_online_memory``.  Reading it from ``spec`` -- the spawn argument --
    would let any caller turn the prior on or off without an authorization.
    """

    source = (ROOT / "modal_apps/pmo_population_v1_app.py").read_text()
    tree = ast.parse(source)
    run_task = next(
        node for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "run_task"
    )
    execute_call = next(
        node for node in ast.walk(run_task)
        if isinstance(node, ast.Call)
        and getattr(node.func, "id", None) == "execute_task"
    )
    keyword = next(
        (kw for kw in execute_call.keywords if kw.arg == "construction_prior_spec"), None
    )
    assert keyword is not None, "the worker never passes construction_prior_spec"
    expression = ast.unparse(keyword.value)
    assert "contract_envelope" in expression, (
        f"the prior must come from the sealed arm payload, not from {expression!r}"
    )
    assert "spec[" not in expression and "spec.get" not in expression, (
        f"the prior must not be selectable from the spawn spec: {expression!r}"
    )


def test_every_whitelisted_arm_contract_exists_and_seals_itself():
    """A baked arm whose payload does not match its own seal cannot be authorized."""

    import hashlib

    source = (ROOT / "modal_apps/pmo_population_v1_app.py").read_text()
    tree = ast.parse(source)
    arms = next(
        node.value for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(getattr(t, "id", None) == "AB_CONTRACTS" for t in node.targets)
    )
    relatives = {
        key.value: value.value
        for key, value in zip(arms.keys, arms.values)
        if isinstance(key, ast.Constant) and isinstance(value, ast.Constant)
    }
    assert {"B_cp_off", "B_cp_on"} <= set(relatives)
    for arm, relative in relatives.items():
        path = ROOT / relative
        assert path.exists(), f"{arm}: baked contract {relative} is absent"
        envelope = json.loads(path.read_text())
        digest = hashlib.sha256(
            json.dumps(
                envelope["payload"], sort_keys=True, separators=(",", ":"), allow_nan=False
            ).encode()
        ).hexdigest()
        assert digest == envelope["payload_sha256"], f"{arm}: payload broke its own seal"
        assert envelope["payload"]["arm"]["name"] == arm


def test_the_two_construction_prior_arms_differ_in_exactly_the_prior():
    """The matched claim, checked rather than asserted in prose."""

    off = json.loads(
        (ROOT / "configs/pmo_ab_b_construction_prior_off_contract_v1.json").read_text()
    )["payload"]
    on = json.loads(
        (ROOT / "configs/pmo_ab_b_construction_prior_on_contract_v1.json").read_text()
    )["payload"]

    differing = sorted(k for k in set(off) | set(on) if off.get(k) != on.get(k))
    assert differing == ["arm"], f"arms are NOT matched; they differ in {differing}"
    assert off["budget"] == on["budget"]
    assert off["tasks"] == on["tasks"] == ["celecoxib_rediscovery"]
    assert off["implementation_sha256"] == on["implementation_sha256"]

    arm_differing = sorted(
        k for k in set(off["arm"]) | set(on["arm"]) if off["arm"].get(k) != on["arm"].get(k)
    )
    assert set(arm_differing) <= {"construction_prior", "name", "comparison", "what_differs"}
    assert off["arm"].get("construction_prior") is None
    assert on["arm"]["construction_prior"]["schema_version"] == (
        "pmo_construction_prior_spec_v1"
    )
    assert off["arm"]["enable_online_memory"] is on["arm"]["enable_online_memory"] is True


def test_the_treated_arm_carries_the_mandatory_development_only_label():
    """The checkpoint is forbidden for frozen results, so the label must travel."""

    from compose_v4.control.pmo_construction_prior import CHECKPOINT_STATUS

    on = json.loads(
        (ROOT / "configs/pmo_ab_b_construction_prior_on_contract_v1.json").read_text()
    )["payload"]
    assert on["arm"]["construction_prior"]["status"] == CHECKPOINT_STATUS
    assert CHECKPOINT_STATUS == "DEVELOPMENT_ONLY_CONSTRUCTION_PRIOR_QUALIFICATION"
    assert on["checkpoint_publication_restriction"]["may_become_a_published_number"] is False
