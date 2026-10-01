"""Small fixed-panel selector tests. No benchmark artifacts or network required."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy

import pytest

from compose_v4.experiments.fragments import selection_ablation
from compose_v4.experiments.fragments.selection_ablation import _pick, replay


def panel():
    offers = [{"draw": index, "status": "failed"} for index in range(8)]
    offers[0] = {
        "draw": 0,
        "status": "model_supported",
        "endpoint": "CCO",
        "mean_log_mark": -2.0,
    }
    offers[3] = {
        "draw": 3,
        "status": "model_supported",
        "endpoint": "CCN",
        "mean_log_mark": -0.5,
    }
    return {"offered": offers, "model_supported_count": 2, "selected_smiles": "CCN"}


def result(task="motif_extension"):
    return {
        "task": task,
        "cells": [
            {
                "prompt": "EXAMPLE",
                "seed": 2,
                "attempts": [
                    {"attempt_index": 0, "panel": panel()},
                    {
                        "attempt_index": 1,
                        "panel": {
                            "offered": [{"draw": index, "status": "failed"} for index in range(8)],
                            "model_supported_count": 0,
                            "selected_smiles": None,
                        },
                    },
                ],
            }
        ],
    }


def test_same_attempted_panels_are_replayed_without_filling_empty_slots():
    first = replay(result(), seed=11)
    assert first == replay(result(), seed=11)
    cell = first["cells"][0]
    assert cell["attempts"] == 2
    for law in ("reference", "uniform"):
        assert cell["selections"][law][0]["endpoint"] in {"CCO", "CCN"}
        assert cell["selections"][law][1] == {
            "attempt_index": 1,
            "endpoint": "",
            "probability": None,
        }


def test_native_scores_only_change_the_learned_selector():
    offers = [("CCO", -4.0), ("CCN", 0.0)]
    learned = _pick(offers, law="reference", emitted=set(), task="motif_extension", u=0.5)
    uniform = _pick(offers, law="uniform", emitted=set(), task="motif_extension", u=0.5)
    assert learned[0] == "CCN"
    assert uniform[0] == "CCN"
    reversed_scores = [("CCO", 0.0), ("CCN", -4.0)]
    assert (
        _pick(reversed_scores, law="reference", emitted=set(), task="motif_extension", u=0.5)[0]
        == "CCO"
    )
    assert (
        _pick(reversed_scores, law="uniform", emitted=set(), task="motif_extension", u=0.5)
        == uniform
    )


def test_linker_novelty_is_controller_weight_not_reference_score():
    offers = [("CCO", 0.0), ("CCN", -0.2)]
    fresh = _pick(offers, law="reference", emitted={"CCO"}, task="linker_design", u=0.5)
    no_novelty = _pick(offers, law="reference", emitted={"CCO"}, task="motif_extension", u=0.5)
    assert fresh[0] == "CCN"
    assert no_novelty[0] == "CCO"


@pytest.mark.parametrize("defect", ["short", "duplicate", "unsupported_selection", "count"])
def test_malformed_panel_fails_closed(defect):
    value = deepcopy(result())
    broken = value["cells"][0]["attempts"][0]["panel"]
    if defect == "short":
        broken["offered"].pop()
    elif defect == "duplicate":
        broken["offered"][3]["endpoint"] = "CCO"
    elif defect == "unsupported_selection":
        broken["selected_smiles"] = "CCC"
    else:
        broken["model_supported_count"] = 3
    with pytest.raises(ValueError):
        replay(value, seed=11)


def test_replay_requires_bound_generation_and_never_overwrites(tmp_path, monkeypatch):
    checkpoint = "a" * 64
    source = result()
    source.update(schema="compose_fragment_generation_v1", checkpoint_sha256=checkpoint)
    input_path = tmp_path / "result.json"
    input_path.write_text(json.dumps(source))
    provenance = {
        "mode": "local_generation",
        "configuration": {"task": source["task"]},
        "result_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
    }
    (tmp_path / "provenance.json").write_text(json.dumps(provenance))
    monkeypatch.setattr(
        selection_ablation,
        "load_registry",
        lambda root: {"assets": {"checkpoint": {"sha256": checkpoint}}},
    )
    monkeypatch.setattr(selection_ablation, "core_runtime_identity", lambda: {"test": True})
    output = tmp_path / "comparison.json"
    selection_ablation.run(
        root=tmp_path, input_path=input_path, output_path=output, seed=11, assets=None
    )
    assert json.loads(output.read_text())["reference_checkpoint_sha256"] == checkpoint
    with pytest.raises(FileExistsError, match="overwrite"):
        selection_ablation.run(
            root=tmp_path, input_path=input_path, output_path=output, seed=11, assets=None
        )
    input_path.write_text(input_path.read_text() + " ")
    with pytest.raises(ValueError, match="bound shared-reference"):
        selection_ablation.run(
            root=tmp_path,
            input_path=input_path,
            output_path=tmp_path / "tampered.json",
            seed=11,
            assets=None,
        )
