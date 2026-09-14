from __future__ import annotations

import gzip
import json
from pathlib import Path

from compose_v4.control.docking_value import identity
from compose_v4.experiments.pmo_dependency_region_program import (
    dependency_region_program,
)
from tools.pmo_dependency_region_program_v2 import _load_contract, sha256_file


def _source_corpus() -> dict:
    path = (
        "diagnostics/pmo_exact_program_segmentation/attempt_2/"
        "training_exact_segmented_corpus.json.gz"
    )
    with gzip.open(path, "rt") as handle:
        return json.load(handle)["payload"]


def test_one_exact_trace_has_macro_free_control_and_relative_created_references():
    route = next(
        row
        for row in _source_corpus()["routes"]
        if row["segmentation"]["dependency_edges"]
    )
    program = dependency_region_program(tuple(route["states"]), tuple(route["actions"]))

    assert program["exact_replay"] is True
    assert program["cross_component_created_dependency_edges"] == 0
    assert program["cross_component_cycle_dependency_edges"] == 0
    assert [row["emission_index"] for row in program["emissions"]] == list(
        range(len(route["actions"]))
    )
    for component in program["components"]:
        emitted = [
            row
            for row in program["emissions"]
            if row["component_index"] == component["component_index"]
        ]
        assert emitted[0]["control_before"] == "open"
        assert emitted[-1]["control_after"] == "close"
        assert all(row["control_before"] == "continue" for row in emitted[1:])
    references = [
        reference
        for emission in program["emissions"]
        for reference in emission["created_handle_references"]
    ]
    assert references
    assert all(reference["created_backreference"] >= 1 for reference in references)
    assert "slot" not in json.dumps(program["emissions"])
    assert "macro" not in json.dumps(program["components"])


def test_contract_is_self_hashed_and_preserves_v1_artifacts():
    contract = _load_contract(Path.cwd())
    source = contract["source_v1"]

    assert contract["outputs"]["runtime_checkpoint"] is None
    assert contract["representation"]["runtime_maximum_primitives"] == 32
    assert contract["representation"]["runtime_maximum_components"] == 8
    assert source["result_sha256"] == sha256_file(Path(source["result_path"]))
    assert source["training_corpus_sha256"] == sha256_file(
        Path(source["training_corpus_path"])
    )


def test_sealed_v2_result_passes_complete_support_and_reuses_panel():
    output = Path("diagnostics/pmo_dependency_region_program_v2/attempt_1")
    result_envelope = json.loads((output / "result.json").read_text())
    with gzip.open(
        output / "training_dependency_region_corpus.json.gz", "rt"
    ) as handle:
        corpus_envelope = json.load(handle)

    result = result_envelope["payload"]
    corpus = corpus_envelope["payload"]
    assert result_envelope["payload_sha256"] == identity(result)
    assert corpus_envelope["payload_sha256"] == identity(corpus)
    assert result["decision"] == (
        "dependency_region_representation_gate_passed_"
        "autoregressive_decoder_comparison_enabled"
    )
    assert all(result["gates"].values())
    overall = result["representation"]["overall"]
    assert overall["exact_replay_routes"] == 184
    assert overall["exact_replay_precision"] == 1.0
    assert overall["runtime_length_routes"] == 106
    assert overall["complete_representation_routes"] == 106
    assert overall["complete_representation_coverage_runtime_length"] == 1.0
    assert overall["cross_component_created_dependency_edges"] == 0
    assert overall["cross_component_cycle_dependency_edges"] == 0
    assert {
        fold: (row["runtime_length_routes"], row["complete_representation_routes"])
        for fold, row in result["representation"]["held_out_folds"].items()
    } == {"0": (33, 33), "1": (25, 25), "2": (48, 48)}
    panel = result["negative_panel_reference"]
    assert panel["attempts"] == 576
    assert panel["unique_complete"] == 559
    assert panel["new_proposals"] == 0
    assert corpus["negative_panel_reference"] == panel
    assert corpus["runtime_checkpoint"] is False
    assert corpus["model_fit"] is False
    assert corpus["new_oracle_calls"] == 0
