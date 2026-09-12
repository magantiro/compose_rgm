import json
from pathlib import Path

import numpy as np
import pytest

from compose_v4.chem.molecular_graph import ELEMENT_TO_IDX, smiles_to_molecular_graph
from compose_v4.chem.state import pad_molecular_graph
from compose_v4.control.molecular_task_search import MolecularHierarchy
from compose_v4.control.option_continuation import EXECUTABLE_PRODUCT_GATE, OptionContinuationKernel
from compose_v4.experiments.t4_winner_refinement import (
    contract_at,
    property_scorer,
    replicate_summary,
    sample_stream,
    select_candidates,
    winner_at,
)
from compose_v4.rewrite.kernel import editing_v2_semantic_rewrite_system
from compose_v4.rewrite.operators import AtomInsert

ROOT = Path(__file__).resolve().parents[1]


def test_selection_is_unique_balanced_and_ignores_task_scores():
    def row(smiles, count, group, eligible=True):
        return {
            "smiles": smiles,
            "primitive_count": count,
            "group": group,
            "oracle_eligible": eligible,
            "predicted_score": -1000.0,
        }

    rows = [
        row("CCN", 2, "options/generic"),
        row("CCO", 1, "primitive/local"),
        row("CCF", 1, "primitive/grow"),
        row("CCN", 1, "primitive/local"),
        row("CCC", 1, "primitive/local"),
        row("CCCl", 3, "options/generic", False),
    ]
    selected = select_candidates(rows, "CCC", limit=3, multi_quota=1)
    assert len(selected) == 3
    assert selected[0]["smiles"] == "CCN"
    assert len({r["smiles"] for r in selected}) == 3
    changed = select_candidates(
        [{**r, "predicted_score": 10000.0} for r in rows], "CCC", limit=3, multi_quota=1
    )
    assert [r["smiles"] for r in selected] == [r["smiles"] for r in changed]


def test_repeats_report_selection_and_independent_repeats_separately():
    winner = [{"ds": d} for d in [-13.6, -13.0, -13.2]]
    candidate = [{"ds": d} for d in [-14.0, -12.0, -12.1]]
    result = replicate_summary(winner, candidate)
    assert result["mean_difference"] > 0
    assert result["repeat_only_mean_difference"] == pytest.approx(1.05)
    assert (
        replicate_summary(winner, [{"ds": None}, *candidate[1:]])["status"]
        == "inconclusive_oracle_failure"
    )


def test_real_hierarchy_stream_replays_exactly_and_retains_region_metadata():
    graph = pad_molecular_graph(smiles_to_molecular_graph("CCCC"), 48)

    def law(state):
        # Explicit model-free engineering reference, not an R_theta estimate.
        slot = int(np.flatnonzero(state.atom_types == 0)[0])
        choices = [
            AtomInsert(slot, ELEMENT_TO_IDX["C"], 0, 3, ((int(i), 1),))
            for i in np.flatnonzero(state.implicit_h_counts > 0)
        ]
        return ("atom_insert",) * len(choices), tuple(choices), (1 / len(choices),) * len(choices)

    def run():
        kernel = OptionContinuationKernel(
            law,
            editing_v2_semantic_rewrite_system(),
            max_executor_applications=1000,
            product_gate=EXECUTABLE_PRODUCT_GATE,
        )
        hierarchy = MolecularHierarchy(
            kernel, generic_horizon=1, ring_options=(), lazy_applicability=True
        )
        return sample_stream(graph, hierarchy, seed=5, options=2, budget=2, progress={})

    products, detail = run()
    assert detail["status"] == "complete"
    assert products and products[-1]["primitive_count"] == 2
    assert all(r["bundles"][-1]["region"]["r_release"] > 0 for r in products)
    assert (products, detail) == run()
    json.dumps(products, allow_nan=False)


def test_recipe_and_original_seed_gate_are_bound():
    c = contract_at(ROOT)
    graph = winner_at(ROOT, c)
    assert graph.n_real_atoms == 30
    seed = json.loads((ROOT / "docs/GENMOL_T4_SEEDS.json").read_text())[0]["smiles"]
    report = json.loads((ROOT / c["winner"]["path"]).read_text())
    row = property_scorer(seed)({"smiles": report["target"]})
    assert row["oracle_eligible"]
    assert row["sim"] < 1.0
    assert c["compute"]["oracle_call_limit"] == 3 + 14 + 2
