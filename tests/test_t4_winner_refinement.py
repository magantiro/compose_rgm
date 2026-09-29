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
    reused_winner_controls,
    sample_stream,
    select_candidates,
    winner_at,
)
from compose_v4.rewrite.kernel import editing_v2_semantic_rewrite_system
from compose_v4.rewrite.operators import AtomInsert

ROOT = Path(__file__).resolve().parents[1]


def test_resume_reuses_only_the_exact_completed_controls():
    from compose_v4.rewrite.kernel import canonical_state_key
    from compose_v4.rewrite.trace_shard import encode_state

    contract = contract_at(ROOT)
    state = winner_at(ROOT, contract)
    seed = json.loads((ROOT / "docs/GENMOL_T4_SEEDS.json").read_text())[0]["smiles"]
    winner = property_scorer(seed)(
        {"smiles": canonical_state_key(state), "state": encode_state(state)}
    )
    rows = reused_winner_controls(ROOT, contract, winner)
    assert len(rows) == 3 and all(row["completed_at_utc"] for row in rows)
    with pytest.raises(ValueError, match="different winner"):
        reused_winner_controls(ROOT, contract, {**winner, "sim": 0.9})
    with pytest.raises(ValueError, match="seed schedule"):
        reused_winner_controls(ROOT, {**contract, "docking": {"seeds": [1, 2, 3]}}, winner)


def test_qualified_inference_receipt_is_bound_to_its_source_revision():
    from compose_v4.experiments.continuation_profile import verify_file
    from compose_v4.experiments.t4_matched_pilot import unseal

    c = contract_at(ROOT)
    config = c["qualified_inference"]
    package = unseal(ROOT / "diagnostics/pmo_inference_speed/package_manifest.json")
    receipt = unseal(ROOT / config["qualification"]["path"])
    assert receipt["status"] == "pass"
    assert receipt["export"]["manifest_sha256"] == config["package"]["manifest_sha256"]
    assert (
        package["provenance"]["reference_inputs"]["checkpoint"]["sha256"]
        == c["expected_input_sha256"]["r_theta_checkpoint"]
    )
    later_repaired_registry = "src/compose_v4/data/editing_v2_process_v2_policy_registry.py"
    for path, digest in package["provenance"]["dependency_sources"].items():
        expected = receipt["cache_sources"].get(path, digest)
        if path == later_repaired_registry:
            # The September 10 package pins the earlier registry source. The
            # September 13 repair changed it, so the current checkout must
            # refuse the historical package rather than silently requalify it.
            assert expected == "1c7b0da5c87842c8d5e4e034a4db57c1a3455fb86b3d3713e7447a6a36b430eb"
            with pytest.raises(ValueError, match="input identity mismatch"):
                verify_file(ROOT / path, expected)
        else:
            verify_file(ROOT / path, expected)
    for path, digest in config["extra_model_sources"].items():
        verify_file(ROOT / path, digest)


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
