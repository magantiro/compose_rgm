import numpy as np

from compose_v4.chem.molecular_graph import smiles_to_molecular_graph
from compose_v4.chem.state import pad_molecular_graph
from compose_v4.rewrite.causal_trace import CausalTraceCTMC
from compose_v4.rewrite.kernel import canonical_state_key
from compose_v4.rewrite.tracelet_compiler import compile_null_to_target_tracelets


def _trace(smiles: str):
    target = pad_molecular_graph(smiles_to_molecular_graph(smiles), 16)
    return target, compile_null_to_target_tracelets(
        target,
        typed_ring_payloads=True,
    )


def test_causal_teacher_exposes_independent_branch_frontier() -> None:
    target, trace = _trace("CC(C)C")
    causal = CausalTraceCTMC(trace)
    completed = frozenset((0, 1))
    assert causal.frontier(completed) == (2, 3)

    left = causal.state_for_order((0, 1, 2, 3))
    right = causal.state_for_order((0, 1, 3, 2))
    assert canonical_state_key(left) == canonical_state_key(target)
    assert canonical_state_key(right) == canonical_state_key(target)


def test_causal_frontier_rates_preserve_remaining_total_hazard() -> None:
    _, trace = _trace("CC(C)C")
    causal = CausalTraceCTMC(trace)
    completed, _, state = causal.sample_ideal(2, np.random.default_rng(7))
    sample = causal._sample_from_ideal(0.5, completed, state, operational=True)
    assert len(sample.frontier_steps) == 2
    assert sum(sample.teacher_rates) == causal.path_length - len(completed)
    assert sample.teacher_rates[0] == sample.teacher_rates[1]


def test_causal_progress_retains_binomial_marginal() -> None:
    _, trace = _trace("CC(C)C")
    causal = CausalTraceCTMC(trace)
    rng = np.random.default_rng(11)
    draws = np.asarray([causal.sample_progress(0.4, rng) for _ in range(20_000)])
    assert abs(float(draws.mean()) - causal.path_length * 0.4) < 0.04


def test_ring_transactions_remain_indivisible_causal_events() -> None:
    _, trace = _trace("c1ccccc1-c2ccccc2")
    causal = CausalTraceCTMC(trace)
    assert [step.rule_name for step in trace.steps] == [
        "cycle_insert",
        "cycle_attach",
    ]
    assert causal.dependencies == (frozenset(), frozenset((0,)))
    assert causal.frontier(frozenset()) == (0,)
    assert causal.frontier(frozenset((0,))) == (1,)
