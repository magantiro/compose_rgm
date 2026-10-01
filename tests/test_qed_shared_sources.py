import json
from pathlib import Path

import pytest

from compose_v4.experiments.qed_shared_sources import load_qed_source_roles

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "experiments/qed/shared_sources.json"


def test_qed_shared_split_is_disjoint_at_model_identity() -> None:
    roles = load_qed_source_roles(ROOT, MANIFEST)
    assert len(roles.train) == 1023
    assert len(roles.validation) == 128
    assert len(roles.test) == 800
    assert roles.excluded_train_indices == (115,)
    assert roles.train_input_indices[114:117] == (114, 116, 117)


def test_qed_shared_split_refuses_undeclared_overlap(tmp_path: Path) -> None:
    manifest = json.loads(MANIFEST.read_text())
    manifest["excluded_train_rows"] = []
    altered = tmp_path / "split.json"
    altered.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="declared train exclusions"):
        load_qed_source_roles(ROOT, altered)


def test_qed_shared_split_refuses_source_hash_drift(tmp_path: Path) -> None:
    manifest = json.loads(MANIFEST.read_text())
    manifest["train"]["sha256"] = "0" * 64
    altered = tmp_path / "split.json"
    altered.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="train source hash mismatch"):
        load_qed_source_roles(ROOT, altered)
