"""Realizing a globally sampled ring plan through the production sampler.

Three things must hold or arm C measures nothing:

* the restriction the plan imposes must actually REACH the sampler -- a plan
  whose exclusion mask is dropped one hop later is the inert-repair failure
  this repository has now paid for three times, so the stub sampler ASSERTS
  what it was handed rather than ignoring it;
* the realized signature must be read back from the EXECUTED action, never
  echoed from the request, or "the plan was realized" becomes a restatement of
  the specification (the ``rings_closed`` failure, one layer up);
* an unrealizable system must be recorded and the plan must continue, so the
  realization rate is a property of the host and not of the ordering.

The ring-grow action used here is a real, executable ``RingSystemGrow``
(hexane -> cyclohexane) applied through the real de-novo runtime, so the
signature readback runs the production code rather than a fixture that agrees
with it by construction.
"""

from __future__ import annotations

import numpy as np
import pytest

from compose_v4.chem.molecular_graph import (
    ELEMENT_TO_IDX,
    molecular_graph_to_smiles,
    smiles_to_molecular_graph,
)
from compose_v4.chem.state import pad_molecular_graph
from compose_v4.experiments.denovo_ring_plan import (
    ARMS,
    UNREALIZED_NO_SUPPORT,
    UNREALIZED_SIGNATURE_ABSENT,
    CatalogSignatureIndex,
    initial_frozen_time,
    install_one_ring_system,
    realize_ring_plan,
    restricted_ring_sampling,
    sample_denovo_arm,
)
from compose_v4.model.factorized_tracelet_rate_model import MARK_RULE_NAMES
from compose_v4.rewrite.kernel import de_novo_rewrite_system
from compose_v4.rewrite.tracelets import AtomPayload, RingBond, RingSystemGrow

CARBON = int(ELEMENT_TO_IDX["C"])


# ---- Real executable fixtures ----


def _hexane():
    return pad_molecular_graph(smiles_to_molecular_graph("CCCCCC"), 40)


def _cyclohexane_grow() -> RingSystemGrow:
    """A real ring-system grow: hexane's carbon path closed into cyclohexane."""

    return RingSystemGrow(
        system_atoms=(0, 1, 2, 3, 4, 5),
        interface_atoms=(),
        scaffold_bonds=tuple(RingBond(index, index + 1, 1) for index in range(5)),
        bond_reorders=(),
        # The closure removes one hydrogen from each endpoint, so the payload
        # carries the PRE-closure count: the executor restates, then inserts.
        atom_payloads=tuple(
            AtomPayload(index, CARBON, 0, 3 if index in (0, 5) else 2)
            for index in range(6)
        ),
        atom_insertions=(),
        bond_insertions=(RingBond(0, 5, 1),),
    )


def test_the_fixture_action_really_executes() -> None:
    state = _hexane()
    successor = de_novo_rewrite_system().apply(
        state, "ring_system_grow", _cyclohexane_grow()
    )
    assert molecular_graph_to_smiles(successor) == "C1CCCCC1"


class _Template:
    def __init__(self, target_bonds, source_bonds) -> None:
        self.span = 1 + max(
            max((a for a, _b, _o in target_bonds), default=0),
            max((b for _a, b, _o in target_bonds), default=0),
        )
        self.target_bonds = tuple(target_bonds)
        self.source_bonds = tuple(source_bonds)


def _ring_template(size: int):
    """A single-ring template of ``size`` atoms, needing ``size`` host atoms."""

    target = tuple(
        (index, (index + 1) % size, 1) for index in range(size)
    )
    source = tuple((index, index + 1, 1) for index in range(size - 1))
    return _Template(target, source)


def _index(sizes=(3, 5, 6, 6)) -> CatalogSignatureIndex:
    return CatalogSignatureIndex.build([_ring_template(size) for size in sizes])


# ---- The catalog index ----


def test_index_groups_templates_by_signature_and_costs_them_by_host() -> None:
    index = _index()
    assert index.template_count == 4
    assert index.indices[(6,)] == (2, 3)
    assert index.indices[(3,)] == (0,)
    assert index.host_atoms[(3,)] == 3
    assert index.host_atoms[(6,)] == 6


def test_excluded_for_keeps_exactly_the_matching_templates() -> None:
    index = _index()
    assert index.excluded_for((6,)) == (0, 1)
    assert index.excluded_for((3,)) == (1, 2, 3)


def test_excluded_for_refuses_a_signature_the_catalog_cannot_express() -> None:
    with pytest.raises(KeyError, match="absent from the catalog"):
        _index().excluded_for((9,))


def test_order_plan_spends_the_host_on_the_most_demanding_system_first() -> None:
    index = _index()
    assert index.order_plan(((3,), (6,), (5,))) == ((6,), (5,), (3,))
    # Deterministic under repeats and under input order.
    assert index.order_plan(((6,), (3,), (6,))) == ((6,), (6,), (3,))
    assert index.order_plan(((3,), (6,), (5,))) == index.order_plan(((5,), (3,), (6,)))


# ---- The restriction ----


class _Model:
    """Records what restriction the production sampler would have seen."""

    def __init__(self, action=None, rule_name="ring_system_grow") -> None:
        self._action = action
        self._rule_name = rule_name
        self.seen: list[tuple[tuple[str, ...], tuple[int, ...]]] = []

    def sample_rewrite_mark(self, state, time, rng):
        self.seen.append(
            (
                tuple(getattr(self, "disabled_sampling_rule_names", ())),
                tuple(getattr(self, "excluded_sampling_ring_template_indices", ())),
            )
        )

        class _Sampled:
            total_hazard = 1.0
            rule_name = self._rule_name
            action = self._action

        return _Sampled()


def test_restriction_sets_then_restores_the_production_attributes() -> None:
    model = _Model()
    model.disabled_sampling_rule_names = ("atom_delete",)
    model.excluded_sampling_ring_template_indices = (7,)
    with restricted_ring_sampling(
        model, only_ring_grow=True, excluded_template_indices=(1, 2)
    ):
        assert "ring_system_grow" not in model.disabled_sampling_rule_names
        assert set(model.disabled_sampling_rule_names) == set(MARK_RULE_NAMES) - {
            "ring_system_grow"
        }
        assert model.excluded_sampling_ring_template_indices == (1, 2)
    assert model.disabled_sampling_rule_names == ("atom_delete",)
    assert model.excluded_sampling_ring_template_indices == (7,)


def test_restriction_restores_even_when_the_draw_raises() -> None:
    # A stranded restriction would silently convert every later trajectory in
    # the same worker into a ring-only sampler.
    model = _Model()
    with pytest.raises(RuntimeError, match="boom"), restricted_ring_sampling(
        model, disable_ring_grow=True
    ):
        raise RuntimeError("boom")
    assert model.disabled_sampling_rule_names == ()
    assert model.excluded_sampling_ring_template_indices == ()


def test_restriction_refuses_to_both_require_and_forbid_ring_growth() -> None:
    with pytest.raises(ValueError, match="both require and forbid"), restricted_ring_sampling(
        _Model(), only_ring_grow=True, disable_ring_grow=True
    ):
        pass


def test_disable_ring_grow_names_exactly_the_ring_family() -> None:
    model = _Model()
    with restricted_ring_sampling(model, disable_ring_grow=True):
        assert model.disabled_sampling_rule_names == ("ring_system_grow",)


# ---- Installation ----


def test_installation_hands_the_sampler_the_plan_restriction() -> None:
    """The consumption check: a dropped exclusion makes the plan inert."""

    model = _Model(action=_cyclohexane_grow())
    index = _index()
    install_one_ring_system(
        model,
        _hexane(),
        rng=np.random.default_rng(0),
        time_value=0.05,
        index=index,
        signature=(6,),
    )
    (disabled, excluded), = model.seen
    assert set(disabled) == set(MARK_RULE_NAMES) - {"ring_system_grow"}
    assert excluded == index.excluded_for((6,))


def test_an_unpinned_installation_restricts_the_family_but_not_the_templates() -> None:
    model = _Model(action=_cyclohexane_grow())
    install_one_ring_system(
        model,
        _hexane(),
        rng=np.random.default_rng(0),
        time_value=0.05,
        signature=None,
    )
    (disabled, excluded), = model.seen
    assert set(disabled) == set(MARK_RULE_NAMES) - {"ring_system_grow"}
    assert excluded == ()


def test_the_realized_signature_is_read_from_the_action_not_the_request() -> None:
    """A field that echoes the request cannot witness that it was met."""

    model = _Model(action=_cyclohexane_grow())
    successor, realized, reason = install_one_ring_system(
        model,
        _hexane(),
        rng=np.random.default_rng(0),
        time_value=0.05,
        index=_index(),
        # Ask for a five-ring; the sampler hands back a six-ring.
        signature=(5,),
    )
    assert reason is None
    assert molecular_graph_to_smiles(successor) == "C1CCCCC1"
    assert realized == (6,)


def test_a_signature_outside_the_catalog_is_refused_before_any_draw() -> None:
    model = _Model(action=_cyclohexane_grow())
    successor, realized, reason = install_one_ring_system(
        model,
        _hexane(),
        rng=np.random.default_rng(0),
        time_value=0.05,
        index=_index(),
        signature=(9,),
    )
    assert (successor, realized, reason) == (None, None, UNREALIZED_SIGNATURE_ABSENT)
    assert model.seen == []


def test_an_empty_support_is_reported_as_a_host_failure() -> None:
    model = _Model(action=None)
    successor, realized, reason = install_one_ring_system(
        model,
        _hexane(),
        rng=np.random.default_rng(0),
        time_value=0.05,
        index=_index(),
        signature=(6,),
    )
    assert (successor, realized, reason) == (None, None, UNREALIZED_NO_SUPPORT)


def test_a_non_ring_draw_raises_rather_than_being_executed_as_one() -> None:
    model = _Model(action=_cyclohexane_grow(), rule_name="atom_insert")
    with pytest.raises(RuntimeError, match="not 'ring_system_grow'"):
        install_one_ring_system(
            model,
            _hexane(),
            rng=np.random.default_rng(0),
            time_value=0.05,
            index=_index(),
            signature=(6,),
        )


def test_a_pinned_signature_without_an_index_is_a_programming_error() -> None:
    with pytest.raises(ValueError, match="needs a catalog index"):
        install_one_ring_system(
            _Model(),
            _hexane(),
            rng=np.random.default_rng(0),
            time_value=0.05,
            signature=(6,),
        )


# ---- Whole-plan realization ----


class _SequenceModel:
    """Returns a scripted sequence of draws, one per installation attempt."""

    def __init__(self, actions) -> None:
        self._actions = list(actions)
        self.calls = 0

    def sample_rewrite_mark(self, state, time, rng):
        action = self._actions[self.calls] if self.calls < len(self._actions) else None
        self.calls += 1

        class _Sampled:
            total_hazard = 1.0
            rule_name = "ring_system_grow"

        _Sampled.action = action
        return _Sampled()


def test_realization_continues_past_a_system_the_host_cannot_carry() -> None:
    # The first (most demanding) system fails; the plan must still attempt the
    # rest, or the realization rate would measure the ordering instead.
    model = _SequenceModel([None, _cyclohexane_grow()])
    state, outcome = realize_ring_plan(
        model,
        _hexane(),
        ((6,), (5,)),
        rng=np.random.default_rng(0),
        time_value=0.05,
        index=_index(),
    )
    assert model.calls == 2
    assert outcome.order == ((6,), (5,))
    assert outcome.unrealized == (((6,), UNREALIZED_NO_SUPPORT),)
    assert outcome.realized == ((6,),)
    assert outcome.events == 1
    assert outcome.fully_realized is False
    assert molecular_graph_to_smiles(state) == "C1CCCCC1"


def test_a_fully_realized_plan_reports_no_unrealized_systems() -> None:
    model = _SequenceModel([_cyclohexane_grow()])
    _state, outcome = realize_ring_plan(
        model,
        _hexane(),
        ((6,),),
        rng=np.random.default_rng(0),
        time_value=0.05,
        index=_index(),
    )
    assert outcome.fully_realized is True
    assert outcome.to_json()["realized"] == [[6]]
    assert outcome.to_json()["events"] == 1


def test_an_unpinned_plan_realizes_only_the_count() -> None:
    model = _Model(action=_cyclohexane_grow())
    _state, outcome = realize_ring_plan(
        model,
        _hexane(),
        ((5,),),
        rng=np.random.default_rng(0),
        time_value=0.05,
        index=_index(),
        pin_signatures=False,
    )
    assert model.seen[0][1] == ()
    assert outcome.realized == ((6,),)


# ---- Arms ----


def test_arm_labels_are_the_three_the_experiment_declares() -> None:
    assert ARMS == ("A", "B", "C")


def test_an_unknown_arm_is_refused() -> None:
    with pytest.raises(ValueError, match="unknown arm"):
        sample_denovo_arm(_Model(), arm="D", rng=np.random.default_rng(0), source_prior=None)


def test_a_planned_arm_without_its_prior_is_refused() -> None:
    for arm in ("B", "C"):
        with pytest.raises(ValueError, match="catalog index and a plan prior"):
            sample_denovo_arm(
                _Model(), arm=arm, rng=np.random.default_rng(0), source_prior=object()
            )


def test_the_plan_is_conditioned_at_the_time_the_first_ordinary_event_would_be() -> None:
    # Read off the production loop's own frozen_time at operational_time == 0.
    assert initial_frozen_time(0.1) == pytest.approx(1.0 - np.exp(-0.05))


def test_a_prepared_initial_state_and_a_source_prior_are_mutually_exclusive() -> None:
    # Silently preferring the prior would make a planned arm emit the control's
    # molecules while still reporting itself as planned.
    from compose_v4.experiments.tracelet_conditional import sample_tracelet_ancestral

    with pytest.raises(ValueError, match="either a source prior or a prepared"):
        sample_tracelet_ancestral(
            _Model(),
            rng=np.random.default_rng(0),
            n_slots=40,
            source_prior=object(),
            initial_state=_hexane(),
        )


def test_the_production_sampler_still_reads_both_restriction_attributes() -> None:
    """Drift alarm on the only wiring this mechanism has.

    The restriction is expressed entirely through two attributes the shipped
    sampler consults.  If a future edit stops consulting either, every planned
    arm silently becomes the control, so the names are asserted against the
    production source rather than against a copy of it.
    """

    import inspect

    from compose_v4.model.factorized_tracelet_rate_model import (
        FactorizedTraceletRateModel,
    )

    conditioned = inspect.getsource(
        FactorizedTraceletRateModel.sample_rewrite_mark_conditioned
    )
    family = inspect.getsource(FactorizedTraceletRateModel._sample_action_from_family)
    assert "disabled_sampling_rule_names" in conditioned
    assert "excluded_sampling_ring_template_indices" in family


class _TerminalPrior:
    """A source prior that always hands back the same real hexane state."""

    def sample(self, rng, n_slots):
        return _hexane()


class _FixedPlanPrior:
    def __init__(self, plan) -> None:
        self._plan = plan
        self.asked: list[int] = []

    def sample_with_provenance(self, rng, heavy_atoms):
        self.asked.append(int(heavy_atoms))
        return self._plan, {"requested_bin": "x", "drawn_bin": "x", "bin_fallback": False}


class _PlanThenTerminalModel:
    """Installs one ring, then terminates the ordinary process immediately."""

    def __init__(self) -> None:
        self.seen: list[tuple[tuple[str, ...], tuple[int, ...]]] = []
        self.calls = 0

    def sample_rewrite_mark(self, state, time, rng):
        self.seen.append(
            (
                tuple(getattr(self, "disabled_sampling_rule_names", ())),
                tuple(getattr(self, "excluded_sampling_ring_template_indices", ())),
            )
        )
        self.calls += 1
        first = self.calls == 1

        class _Sampled:
            total_hazard = 1.0 if first else 0.0
            rule_name = "ring_system_grow" if first else "<TERMINAL>"
            action = _cyclohexane_grow() if first else None

        return _Sampled()


def _arm(arm: str, model, plan_prior):
    return sample_denovo_arm(
        model,
        arm=arm,
        rng=np.random.default_rng(0),
        source_prior=_TerminalPrior(),
        index=_index(),
        plan_prior=plan_prior,
        n_slots=40,
        operational_horizon=0.2,
        time_step=0.1,
        max_events=8,
    )


def test_arm_c_pins_the_sizes_and_arm_b_pins_only_the_count() -> None:
    """The single line that separates the two planned arms.

    If arm C ever stopped pinning signatures it would silently become arm B
    while still reporting itself as C, which is precisely the comparison the
    experiment exists to make.
    """

    index = _index()
    model_c = _PlanThenTerminalModel()
    record_c = _arm("C", model_c, _FixedPlanPrior(((6,),)))
    assert model_c.seen[0][1] == index.excluded_for((6,))

    model_b = _PlanThenTerminalModel()
    record_b = _arm("B", model_b, _FixedPlanPrior(((6,),)))
    assert model_b.seen[0][1] == ()

    for record in (record_b, record_c):
        assert record["plan_events"] == 1
        assert record["ring_plan"]["realized"] == [[6]]


def test_arm_a_runs_the_production_path_and_carries_no_plan() -> None:
    plan_prior = _FixedPlanPrior(((6,),))
    record = _arm("A", _PlanThenTerminalModel(), plan_prior)
    assert record["ring_plan"] is None
    assert record["plan_events"] == 0
    # Arm A must not consult the plan prior at all.
    assert plan_prior.asked == []


def test_a_planned_arm_asks_the_prior_for_the_size_it_actually_has() -> None:
    # Hexane is six heavy atoms; drawing a plan for some other size would
    # decorrelate the ring skeleton from the molecule the tree prior produced.
    plan_prior = _FixedPlanPrior(((6,),))
    _arm("C", _PlanThenTerminalModel(), plan_prior)
    assert plan_prior.asked == [6]


def test_the_continuation_runs_with_ring_growth_disabled() -> None:
    # Otherwise the ordinary process would add rings on top of the plan and the
    # realized marginal would not be p(R).
    model = _PlanThenTerminalModel()
    _arm("C", model, _FixedPlanPrior(((6,),)))
    assert len(model.seen) > 1
    for disabled, _excluded in model.seen[1:]:
        assert disabled == ("ring_system_grow",)
