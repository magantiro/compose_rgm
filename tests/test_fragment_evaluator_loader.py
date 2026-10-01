from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from compose_v4.benchmark.fragment_official_metrics import official_prompt_metrics
from compose_v4.experiments.fragments.assets import load_registry
from compose_v4.experiments.fragments.evaluator import load_evaluator


@pytest.fixture
def evaluator_files(tmp_path):
    sources = {
        "evaluator_mol": "def calculate_average_tanimoto(*args, **kwargs): return 0.5\n",
        "evaluator_metrics": (
            "from ..utils.mol import calculate_average_tanimoto\n"
            "from ..preprocess.preprocess_tokenize import custom_decode_sequence\n"
            "def evaluate_smiles(*args, **kwargs): return {'value': calculate_average_tanimoto()}\n"
        ),
    }
    paths, hashes = {}, {}
    for name, source in sources.items():
        paths[name] = tmp_path / (name + ".py")
        paths[name].write_text(source)
        hashes[name] = hashlib.sha256(source.encode()).hexdigest()
    return paths, hashes


def test_loader_uses_explicit_files_and_private_relative_imports(evaluator_files):
    paths, hashes = evaluator_files
    evaluator = load_evaluator(paths, hashes)
    assert evaluator.evaluate_smiles() == {"value": 0.5}
    assert evaluator.source_sha256 == hashes
    module = sys.modules[evaluator.evaluate_smiles.__module__]
    assert module.__name__.startswith("_compose_fragment_evaluator_")
    with pytest.raises(NotImplementedError, match="already_smiles=True"):
        module.custom_decode_sequence(None)


def test_cached_evaluator_is_reverified(evaluator_files):
    paths, hashes = evaluator_files
    load_evaluator(paths, hashes)
    paths["evaluator_mol"].write_text("raise RuntimeError('must not execute')\n")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        load_evaluator(paths, hashes)


def test_all_hashes_checked_before_any_source_executes(evaluator_files):
    paths, hashes = evaluator_files
    source = "raise AssertionError('executed before both identities checked')\n"
    paths["evaluator_mol"].write_text(source)
    hashes["evaluator_mol"] = hashlib.sha256(source.encode()).hexdigest()
    hashes["evaluator_metrics"] = "0" * 64
    with pytest.raises(ValueError, match="evaluator_metrics.py"):
        load_evaluator(paths, hashes)


def test_failed_import_cleans_private_namespace(evaluator_files):
    paths, hashes = evaluator_files
    source = "raise ImportError('invalid evaluator dependency')\n"
    paths["evaluator_mol"].write_text(source)
    hashes["evaluator_mol"] = hashlib.sha256(source.encode()).hexdigest()
    before = {name for name in sys.modules if name.startswith("_compose_fragment_evaluator_")}
    with pytest.raises(ImportError, match="invalid evaluator dependency"):
        load_evaluator(paths, hashes)
    assert before == {
        name for name in sys.modules if name.startswith("_compose_fragment_evaluator_")
    }


def test_metric_adapter_keeps_attempt_denominator_and_upstream_flags():
    calls = []

    def evaluate(samples, tokenizer, **kwargs):
        calls.append((samples, tokenizer.decode("CCO"), kwargs))
        return {"validity": 50, "uniqueness": 100, "quality": 0, "diversity": 0}

    metrics = official_prompt_metrics(
        ["CCO", ""], evaluator=SimpleNamespace(evaluate_smiles=evaluate), expected_samples=2
    )
    assert metrics["validity"] == 50.0
    assert calls[0][0] == ["CCO", ""]
    assert calls[0][1] == "CCO"
    assert calls[0][2]["already_smiles"] is True
    for malformed in (["CCO"], ["CC.O", ""], ["OCC", ""]):
        with pytest.raises(ValueError):
            official_prompt_metrics(
                malformed, evaluator=SimpleNamespace(evaluate_smiles=evaluate), expected_samples=2
            )
    assert len(calls) == 1


@pytest.mark.external_artifact
def test_pinned_upstream_evaluator_handles_duplicate_and_failed_slots():
    root = Path(__file__).resolve().parents[1]
    registry = load_registry(root)
    assets = Path(os.environ.get("COMPOSE_FRAGMENT_ASSETS", root / "local_assets/fragments"))
    names = ("evaluator_mol", "evaluator_metrics")
    paths = {name: assets / registry["assets"][name]["path"] for name in names}
    absent = [str(path) for path in paths.values() if not path.is_file()]
    if absent:
        pytest.skip(
            f"pinned evaluator files absent: {absent}. See experiments/fragments/GENERATION.md"
        )
    hashes = {name: registry["assets"][name]["sha256"] for name in names}
    evaluator = load_evaluator(paths, hashes)
    metrics = official_prompt_metrics(
        ["CCO", "CCO", "CCN", ""],
        evaluator=evaluator,
        expected_samples=4,
    )
    assert metrics == pytest.approx(
        {
            "validity": 75.0,
            "uniqueness": 200.0 / 3.0,
            "diversity": 0.667,
            "quality": 0.0,
        }
    )
    assert official_prompt_metrics([""], evaluator=evaluator, expected_samples=1) == {
        "validity": 0.0,
        "uniqueness": 0.0,
        "diversity": 0.0,
        "quality": 0.0,
    }
    # A passing control catches upstream's broad exception-to-zero behavior.
    ibuprofen = "CC(C)Cc1ccc(C(C)C(=O)O)cc1"
    metrics = official_prompt_metrics(
        [ibuprofen, ibuprofen, ""], evaluator=evaluator, expected_samples=3
    )
    assert metrics == pytest.approx(
        {"validity": 200.0 / 3.0, "uniqueness": 50.0, "diversity": 0.0, "quality": 100.0 / 3.0}
    )
