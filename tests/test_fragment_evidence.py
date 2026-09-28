"""Portable fragment reductions must not depend on private worktrees or RDKit."""

from __future__ import annotations

import copy
import json
import os
import shutil
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from compose_v4.experiments.fragments.evidence import (
    EvidenceError,
    contained_path,
    load_evidence,
    read_json,
    sha256,
)
from compose_v4.experiments.fragments.reduction import recompute_intervals, reduce_evidence
from compose_v4.experiments.fragments.reporting import json_text, markdown, provenance, publish

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def evidence():
    return load_evidence(ROOT)


@pytest.fixture
def changed(evidence):
    return replace(
        evidence,
        artifacts=copy.deepcopy(evidence.artifacts),
        manifest=copy.deepcopy(evidence.manifest),
    )


def test_locked_benchmark_and_sample_sd(evidence):
    result = reduce_evidence(evidence)
    benchmark = result["benchmark"]
    motif = benchmark["motif_extension"]["metrics"]
    assert motif["quality"]["mean"] == pytest.approx(42.63333333333333)
    assert motif["validity"]["mean"] == pytest.approx(99.96666666666667)
    assert motif["validity"]["sample_sd"] == pytest.approx(0.05773502691896)
    assert benchmark["superstructure_generation"]["metrics"]["quality"][
        "sample_sd"
    ] == pytest.approx(3.08922859842604)
    assert set(benchmark) == {
        "motif_extension",
        "scaffold_decoration",
        "linker_design",
        "superstructure_generation",
    }
    assert result["aliases"] == {"scaffold_morphing": "linker_design"}
    assert "scaffold_morphing" not in benchmark


def test_all_ablations_and_prefixes_are_reduced(evidence):
    result = reduce_evidence(evidence)
    expected = {
        "motif_extension": 5.4333333333,
        "scaffold_decoration": 2.2333333333,
        "linker_design": 7.5333333333,
        "superstructure_generation": 13.6333333333,
    }
    for task, difference in expected.items():
        comparison = result["comparisons"][task]
        assert comparison["intervals"]["quality"]["mean_difference"] == pytest.approx(difference)
        assert len(comparison["per_prompt"]) == 10
        assert comparison["arms"]["deployed"]["attempts"] == 3000
    for task, budgets in result["prefixes"].items():
        assert set(budgets) == {"1", "2", "4", "8"}
        assert budgets["1"]["deployed"] == budgets["1"]["uniform"]
        assert (
            budgets["4"]["deployed"]["metrics"]["quality"]["mean"]
            > budgets["8"]["uniform"]["metrics"]["quality"]["mean"]
        )
        assert len(budgets["8"]["deployed"]["per_seed"]) == 3
    # The copied interval source contains a different, older linker experiment.
    assert result["comparisons"]["linker_design"]["intervals"]["quality"][
        "percentile_95"
    ] == pytest.approx([3.5666666667, 11.5])


@pytest.mark.parametrize("role", ["motif_selection", "decoration_selection", "linker_selection"])
def test_duplicate_or_missing_prompt_seed_prefix_fails(changed, role):
    artifact = changed.artifacts[role]
    rows = artifact["rows"] if role == "motif_selection" else artifact["prefix"]["rows"]
    rows[-1] = copy.deepcopy(rows[0])
    with pytest.raises(EvidenceError, match="duplicate"):
        reduce_evidence(changed)


def test_superstructure_duplicate_rows_fail(changed):
    rows = changed.artifacts["superstructure_uniform"]["rows"]
    rows[-1] = copy.deepcopy(rows[0])
    with pytest.raises(EvidenceError, match="duplicate"):
        reduce_evidence(changed)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1, 101, True])
def test_invalid_metric_fails(changed, value):
    changed.artifacts["motif_selection"]["rows"][0]["arms"]["learned"]["metrics"]["validity"] = (
        value
    )
    with pytest.raises(EvidenceError, match="finite value"):
        reduce_evidence(changed)


def test_no_silent_exclusion_or_aggregation_drift(changed):
    changed.artifacts["linker_selection"]["population"]["total_attempts"] = 2999
    with pytest.raises(EvidenceError, match="population"):
        reduce_evidence(changed)


def test_source_means_must_agree_with_rows(changed):
    changed.artifacts["superstructure_learned"]["official_mean"]["quality"] = 40
    with pytest.raises(EvidenceError, match="disagrees"):
        reduce_evidence(changed)


def test_wrong_generation_seed_panel_fails(changed):
    changed.manifest["generation_seeds"]["motif_extension"] = [0, 1, 2]
    with pytest.raises(EvidenceError, match="population"):
        reduce_evidence(changed)


def test_morphing_cannot_become_an_independent_experiment(changed):
    changed.manifest["aliases"] = {}
    with pytest.raises(EvidenceError, match="morphing"):
        reduce_evidence(changed)


def test_bad_interval_lineage_fails(changed):
    changed.artifacts["motif_intervals"]["input_sha256"][
        "diagnostics/fragment_common_panel_motif_v1/result.json"
    ] = "0" * 64
    with pytest.raises(EvidenceError, match="different motif"):
        reduce_evidence(changed)


def test_imported_superstructure_contract_identity_is_checked(changed):
    changed.artifacts["superstructure_learned"]["contract_payload_sha256"] = "0" * 64
    with pytest.raises(EvidenceError, match="different contract"):
        reduce_evidence(changed)


@pytest.mark.parametrize("payload", ['{"a": 1, "a": 2}', '{"a": NaN}', "[]"])
def test_malformed_json_fails_with_path(tmp_path, payload):
    path = tmp_path / "bad.json"
    path.write_text(payload)
    with pytest.raises(EvidenceError, match="bad.json"):
        read_json(path)


def test_manifest_paths_cannot_escape_root(tmp_path):
    for value in ("../outside.json", "/absolute.json"):
        with pytest.raises(EvidenceError, match="inside"):
            contained_path(tmp_path, value)
    (tmp_path / "escape").symlink_to(tmp_path.parent, target_is_directory=True)
    with pytest.raises(EvidenceError, match="escapes"):
        contained_path(tmp_path, "escape/outside.json")


def _export(evidence, destination):
    for spec in evidence.manifest["artifacts"].values():
        path = destination / spec["path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / spec["path"], path)
    manifest = destination / "experiments/fragments/manifest.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(evidence.manifest_path, manifest)
    package = Path("src/compose_v4/experiments/fragments")
    shutil.copytree(
        ROOT / package, destination / package, ignore=shutil.ignore_patterns("__pycache__")
    )
    for path in ("src/compose_v4/__init__.py", "src/compose_v4/experiments/__init__.py"):
        original = ROOT / path
        if original.is_file():
            shutil.copyfile(original, destination / path)
    return destination


def test_missing_and_corrupt_artifacts_fail_closed(evidence, tmp_path):
    export = _export(evidence, tmp_path / "source")
    path = export / evidence.manifest["artifacts"]["superstructure_uniform"]["path"]
    path.write_text("{}")
    with pytest.raises(EvidenceError, match="SHA-256 mismatch"):
        load_evidence(export)
    path.unlink()
    with pytest.raises(EvidenceError, match="missing superstructure_uniform; expected SHA-256"):
        load_evidence(export)


def test_stdlib_only_export_reproduces_without_worktrees(evidence, tmp_path):
    export = _export(evidence, tmp_path / "source")
    assert not (export / ".worktrees").exists()
    assert not (export / ".git").exists()
    environment = {**os.environ, "PYTHONPATH": str(export / "src"), "PYTHONDONTWRITEBYTECODE": "1"}
    expected = reduce_evidence(evidence)
    for name in ("first", "second"):
        command = [
            sys.executable,
            "-S",
            "-m",
            "compose_v4.experiments.fragments",
            "--root",
            str(export),
            "tables",
            "--output",
            str(tmp_path / name),
        ]
        run = subprocess.run(
            command,
            cwd=tmp_path,
            env=environment,
            text=True,
            capture_output=True,
            timeout=30,
            check=False,
        )
        assert run.returncode == 0, run.stderr
        assert json.loads((tmp_path / name / "tables.json").read_text()) == expected
        receipt = json.loads((tmp_path / name / "provenance.json").read_text())
        assert receipt["code_revision_status"] == "source_export_without_git"
        assert receipt["numpy"] is None
        for filename, digest in receipt["output_sha256"].items():
            assert sha256(tmp_path / name / filename) == digest
    assert (tmp_path / "first/tables.json").read_bytes() == (
        tmp_path / "second/tables.json"
    ).read_bytes()
    assert (tmp_path / "first/tables.md").read_bytes() == (
        tmp_path / "second/tables.md"
    ).read_bytes()


@pytest.mark.parametrize("existing", ["directory", "file", "symlink"])
def test_publishing_never_overwrites(evidence, tmp_path, existing):
    target = tmp_path / "report"
    if existing == "directory":
        target.mkdir()
    elif existing == "file":
        target.write_text("user-owned")
    else:
        target.symlink_to(tmp_path / "missing")
    report = reduce_evidence(evidence)
    with pytest.raises(FileExistsError, match="refusing"):
        publish(target, report, {})
    assert os.path.lexists(target)
    if existing == "file":
        assert target.read_text() == "user-owned"


def test_render_failure_does_not_publish_partial_report(evidence, tmp_path, monkeypatch):
    from compose_v4.experiments.fragments import reporting

    def broken(_):
        raise ValueError("render interrupted")

    monkeypatch.setattr(reporting, "markdown", broken)
    with pytest.raises(ValueError, match="interrupted"):
        publish(tmp_path / "report", reduce_evidence(evidence), {})
    assert list(tmp_path.iterdir()) == []


def test_wrong_numpy_is_not_silently_accepted(evidence, monkeypatch):
    import numpy as np

    monkeypatch.setattr(np, "__version__", "wrong-version")
    with pytest.raises(EvidenceError, match="requires numpy==1.26.4"):
        recompute_intervals(reduce_evidence(evidence))


def test_receipt_and_human_report_describe_scope(evidence):
    report = reduce_evidence(evidence)
    receipt = provenance(evidence, report)
    assert receipt["manifest_sha256"] == sha256(evidence.manifest_path)
    assert set(receipt["implementation_sha256"]) == {
        "__init__.py",
        "__main__.py",
        "evidence.py",
        "reduction.py",
        "reporting.py",
    }
    text = markdown(report)
    assert "99.967" in text
    assert "No new molecules" in text
    assert "not an independent experiment" in text
    assert json.loads(json_text(report)) == report
