"""Synthetic graph-edit intermediates are not automatically returned candidates."""

import numpy as np
import pytest

from compose_v4.chem.molecular_graph import smiles_to_molecular_graph
from compose_v4.chem.state import pad_molecular_graph
from compose_v4.control.option_continuation import (
    EXECUTABLE_PRODUCT_GATE,
    LEGACY_PRODUCT_GATE,
    OptionContinuationKernel,
    OptionState,
)
from compose_v4.control.region_rewrite import Lineage, RewriteContext
from compose_v4.experiments.t4_endpoint_selection import acceptable_endpoint
from compose_v4.experiments.t4_task_search import PreparationConfig
from compose_v4.gates.med_chem_gate import is_executable, is_valid
from compose_v4.rewrite.kernel import canonical_state_key, editing_v2_semantic_rewrite_system
from compose_v4.rewrite.operators import BondReorder, CycleCloseEdge


@pytest.mark.parametrize(
    "smiles,marks",
    [
        ("C=CCC", (("bond_reorder", BondReorder(1, 2, 2)), ("bond_reorder", BondReorder(0, 1, 1)))),
        (
            "CCCCCCCCCCC",
            (("cycle_close", CycleCloseEdge(0, 10, 1)), ("cycle_close", CycleCloseEdge(0, 5, 1))),
        ),
    ],
)
def test_two_step_paths_cross_unacceptable_intermediates(smiles, marks):
    graph = pad_molecular_graph(smiles_to_molecular_graph(smiles), 48)
    system = editing_v2_semantic_rewrite_system()
    intermediate = system.apply(graph, *marks[0])
    final = system.apply(intermediate, *marks[1])
    intermediate_key = canonical_state_key(intermediate)
    assert is_executable(intermediate_key) and not is_valid(intermediate_key)
    assert is_valid(canonical_state_key(final))

    def law(state):
        family, action = marks[0] if canonical_state_key(state) == smiles else marks[1]
        return (family,), (action,), (1.0,)

    node = OptionState(
        graph,
        graph,
        RewriteContext(frozenset(), frozenset(range(graph.n_real_atoms)), (), "pendant", 0),
        Lineage.initial(range(graph.n_real_atoms)),
        "generic",
        0,
        2,
        "synthetic",
    )
    old = OptionContinuationKernel(law, system, max_executor_applications=2)
    assert old.row(node).successors == ()
    for lazy in (False, True):
        kernel = OptionContinuationKernel(
            law, system, max_executor_applications=2, product_gate=EXECUTABLE_PRODUCT_GATE
        )
        current = node
        for expected in (intermediate, final):
            current = (
                kernel.lazy_row(current).sample(np.random.default_rng(5))
                if lazy
                else kernel.row(current).successors[0]
            )
            assert canonical_state_key(current.graph) == canonical_state_key(expected)
        assert current.remaining == 0
        assert kernel.work.executor_applications == 2
    # Deliberately passing scalar fixture properties cannot bypass the final screen.
    assert not acceptable_endpoint(
        {"smiles": intermediate_key, "qed": 0.8, "sa": 3.0, "sim": 0.6, "v": 0.0}
    )


def test_gate_policy_is_explicit_and_legacy_serialization_unchanged():
    assert "product_gate" not in PreparationConfig().payload()
    assert PreparationConfig().product_gate == LEGACY_PRODUCT_GATE
    config = PreparationConfig(product_gate=EXECUTABLE_PRODUCT_GATE)
    assert PreparationConfig(**config.payload()) == config
    with pytest.raises(ValueError, match="product gate"):
        PreparationConfig(product_gate="silently_relaxed")
    with pytest.raises(ValueError, match="nonempty smiles"):
        acceptable_endpoint({"qed": 0.8, "sa": 3.0, "sim": 0.6, "v": 0.0})


def test_pathwise_mode_keeps_executor_and_frozen_context_guards():
    from dataclasses import replace

    from test_option_continuation import source

    node = source(horizon=1)
    for action, context in (
        (BondReorder(0, 2, 2), node.context),  # no existing bond
        (
            BondReorder(0, 1, 2),
            RewriteContext(
                frozenset(range(4)),
                frozenset(),
                (),
                "pendant",
                1,
            ),
        ),
    ):
        kernel = OptionContinuationKernel(
            lambda _, a=action: (("bond_reorder",), (a,), (1.0,)),
            editing_v2_semantic_rewrite_system(),
            max_executor_applications=1,
            product_gate=EXECUTABLE_PRODUCT_GATE,
        )
        assert kernel.row(replace(node, context=context)).successors == ()


def test_terminal_and_oracle_selection_reject_unacceptable_completion(monkeypatch):
    from compose_v4.control import molecular_task_search as hierarchy
    from compose_v4.experiments import t4_task_search as module

    # Artificial labels/properties isolate gate placement, not predictor quality.
    class SyntheticValue:
        def __init__(self):
            self.payload = {"snapshot_sha256": "synthetic-no-training"}

        def desirability(self, smiles, feasible):
            return float(feasible)

        def predict(self, smiles):
            return np.full(len(smiles), -10.0)

    monkeypatch.setattr(module.DockingValue, "fit", lambda *_a, **_k: SyntheticValue())
    monkeypatch.setattr(
        module,
        "calculate_properties",
        lambda *_a, **_k: {
            "qed": 0.8,
            "sa": 3.0,
            "sim": 0.6,
            "v": 0.0,
        },
    )
    graph = pad_molecular_graph(smiles_to_molecular_graph("C=CCC"), 48)
    # A fixed synthetic WHERE draw isolates endpoint behavior from region sampling.
    regions = hierarchy.enumerate_regions("C=CCC")
    largest = max(regions, key=lambda r: r.size)
    monkeypatch.setattr(hierarchy, "enumerate_regions", lambda _: [largest])
    from compose_v4.rewrite.trace_shard import encode_state

    warm = {
        "schema_version": "t4_exact_archive_v1",
        "round": 0,
        "oracle_attempts": 0,
        "archive": [{"smiles": "C=CCC", "state": encode_state(graph), "ds": None}],
    }
    config = PreparationConfig(
        lineages=1,
        primitive_budget=1,
        max_rollouts=2,
        initial_rollouts=1,
        rollouts_per_decision=1,
        product_gate=EXECUTABLE_PRODUCT_GATE,
    )
    result = module.prepare(
        warm,
        source_sha256="synthetic",
        input_sha256={},
        config=config,
        enumerate_law=lambda _: (("bond_reorder",), (BondReorder(1, 2, 2),), (1.0,)),
        system=editing_v2_semantic_rewrite_system(),
    )
    assert result["pool"] and result["terminal_value_evaluations"]
    assert all(r["value"] == 0 for r in result["terminal_value_evaluations"])
    assert all(not r["oracle_eligible"] for r in result["pool"])
    assert all(r["med_chem_exclusion_reasons"] for r in result["pool"])
    assert result["take"] == [] and result["new_oracle_calls"] == 0
