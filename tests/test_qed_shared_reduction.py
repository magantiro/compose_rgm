"""Complete-panel and candidate-metric checks for QED reduction."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import pytest
from rdkit import Chem
from rdkit.Chem import QED

from compose_v4.experiments.qed_shared_reduction import reduce_qed_sources
from compose_v4.experiments.qed_shared_smc import QEDSMCConfig, candidate_seed
from compose_v4.experiments.qed_source_support import audit_qed_source

SOURCES = ("CCO", "CCN")
CONFIG = QEDSMCConfig(horizon=1, particles=2, candidates=1)


def _write_result(directory: Path, index: int, source: str) -> Path:
    represented = audit_qed_source(source, max_active_atoms=40).represented
    assert represented is not None
    path = directory / f"source_{index:04d}.json"
    candidate = {
        "index": 0,
        "seed": candidate_seed(represented, 0),
        "status": "EXTINCT_NO_HIT",
        "smiles": source,
        "qed": float(QED.qed(Chem.MolFromSmiles(source))),
        "similarity_to_source": 1.0,
        "success": False,
        "steps": 1,
        "resamples": 0,
    }
    path.write_text(
        json.dumps(
            {
                "schema_version": "compose.qed.shared_result.v1",
                "test_index": index,
                "source_split_sha256": "split",
                "reference_manifest_sha256": "reference-manifest",
                "value_assets_manifest_sha256": "value-assets",
                "value_metadata_sha256": "value-metadata",
                "code_sha256": {"smc": "code"},
                "result": {
                    "source_original": source,
                    "source_represented": represented,
                    "reference": {"checkpoint_sha256": "shared-checkpoint"},
                    "value_head_source_split_sha256": "split",
                    "configuration": asdict(CONFIG),
                    "success": False,
                    "candidates": [candidate],
                },
            }
        )
    )
    return path


def _reduce(directory: Path) -> dict:
    return reduce_qed_sources(
        directory,
        SOURCES,
        source_split_sha256="split",
        checkpoint_sha256="shared-checkpoint",
        config=CONFIG,
    )


def test_complete_panel_recounts_failed_slots(tmp_path: Path) -> None:
    for index, source in enumerate(SOURCES):
        _write_result(tmp_path, index, source)
    reduced = _reduce(tmp_path)
    assert reduced["sources"] == 2
    assert reduced["successful_sources"] == 0
    assert reduced["failed_output_slots"] == 2
    assert len(reduced["input_sha256"]) == 2


def test_incomplete_panel_or_metric_drift_fails(tmp_path: Path) -> None:
    first = _write_result(tmp_path, 0, SOURCES[0])
    with pytest.raises(ValueError, match="incomplete"):
        _reduce(tmp_path)
    _write_result(tmp_path, 1, SOURCES[1])
    record = json.loads(first.read_text())
    record["result"]["candidates"][0]["qed"] += 0.01
    first.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="benchmark metrics"):
        _reduce(tmp_path)


def test_mixed_model_identity_fails(tmp_path: Path) -> None:
    for index, source in enumerate(SOURCES):
        _write_result(tmp_path, index, source)
    second = tmp_path / "source_0001.json"
    record = json.loads(second.read_text())
    record["value_metadata_sha256"] = "other-value"
    second.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="mix model or code identities"):
        _reduce(tmp_path)
