import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from compose_v4.chem.molecular_graph import smiles_to_molecular_graph
from compose_v4.chem.state import pad_molecular_graph
from compose_v4.experiments.hphi_region_features import input_dim
from compose_v4.experiments.qed_shared_value import QEDSharedValueHead, QEDValueAssets


class TinyReference:
    reference = SimpleNamespace(max_active_atoms=40)
    config = SimpleNamespace(persistent_slots=48)

    def __init__(self) -> None:
        self.encode_calls = 0

    def identity(self) -> dict[str, str]:
        return {"checkpoint_sha256": "shared-model"}

    def encode(self, _state) -> np.ndarray:
        self.encode_calls += 1
        return np.zeros(256, dtype=np.float64)


def make_value() -> QEDSharedValueHead:
    budget_max = 4
    width = input_dim(budget_max)
    head = torch.nn.Linear(width, 1)
    with torch.no_grad():
        head.weight.zero_()
        head.bias.zero_()
    return QEDSharedValueHead(
        TinyReference(),
        head,
        np.zeros(width),
        np.ones(width),
        {
            "schema_version": "compose.qed.shared_value.v1",
            "reference": TinyReference().identity(),
            "feature_schema": "region_features_v1",
            "target_semantics": "terminal_region",
            "budget_max": budget_max,
            "source_split_sha256": "split-hash",
            "guidance_target": {"region": [0.9, 0.4], "qualified": True},
        },
    )


def test_value_head_uses_exact_boundary_and_fitted_head() -> None:
    bound = make_value().for_source("CCO")
    state = pad_molecular_graph(smiles_to_molecular_graph("CCO"), 48)
    assert bound.value(state, 4, (0.9, 0.4)) == pytest.approx(0.5)
    assert bound.value(state, 4, (0.9, 0.4)) == pytest.approx(0.5)
    assert bound.owner.reference.encode_calls == 1
    assert bound.value(state, 0, (0.9, 0.4)) == 0.0
    assert bound.value(state, 4, (0.1, 0.4)) == pytest.approx(0.5)
    assert bound.value(state, 0, (0.1, 0.4)) == 1.0


def test_value_head_refuses_hitting_target_metadata() -> None:
    value = make_value()
    metadata = dict(value.metadata)
    metadata["target_semantics"] = "ever_hit_region"
    with pytest.raises(ValueError, match="terminal region"):
        QEDSharedValueHead(value.reference, value.head, value.mean, value.scale, metadata)


def test_value_head_refuses_reference_identity_mismatch() -> None:
    value = make_value()
    metadata = dict(value.metadata)
    metadata["reference"] = {"checkpoint_sha256": "other-model"}
    with pytest.raises(ValueError, match="another reference"):
        QEDSharedValueHead(value.reference, value.head, value.mean, value.scale, metadata)


def test_value_head_refuses_bad_normalization() -> None:
    value = make_value()
    scale = value.scale.copy()
    scale[0] = 0
    with pytest.raises(ValueError, match="normalization"):
        QEDSharedValueHead(value.reference, value.head, value.mean, scale, value.metadata)


def test_value_head_loader_checks_files_and_split(tmp_path: Path) -> None:
    value = make_value()
    head_path = tmp_path / "head.pt"
    traced = torch.jit.trace(value.head, torch.zeros((1, input_dim(value.budget_max))))
    traced.save(str(head_path))
    norm_path = tmp_path / "norm.json"
    norm_path.write_text(json.dumps({"mean": value.mean.tolist(), "scale": value.scale.tolist()}))
    metadata_path = tmp_path / "metadata.json"
    metadata_path.write_text(json.dumps(value.metadata))

    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    assets = QEDValueAssets(
        head_path,
        digest(head_path),
        norm_path,
        digest(norm_path),
        metadata_path,
        digest(metadata_path),
    )
    loaded = QEDSharedValueHead.load(
        value.reference, assets, expected_source_split_sha256="split-hash"
    )
    assert loaded.budget_max == value.budget_max
    with pytest.raises(ValueError, match="another source split"):
        QEDSharedValueHead.load(value.reference, assets, expected_source_split_sha256="wrong-split")
    bad_assets = QEDValueAssets(
        head_path, "0" * 64, norm_path, digest(norm_path), metadata_path, digest(metadata_path)
    )
    with pytest.raises(ValueError, match="asset hash mismatch"):
        QEDSharedValueHead.load(
            value.reference, bad_assets, expected_source_split_sha256="split-hash"
        )


def test_value_head_loader_refuses_unqualified_guidance(tmp_path: Path) -> None:
    value = make_value()
    head_path = tmp_path / "head.pt"
    torch.jit.trace(value.head, torch.zeros((1, input_dim(value.budget_max)))).save(str(head_path))
    norm_path = tmp_path / "normalization.json"
    norm_path.write_text(json.dumps({"mean": value.mean.tolist(), "scale": value.scale.tolist()}))
    metadata_path = tmp_path / "metadata.json"
    metadata = dict(value.metadata)
    metadata["guidance_target"] = {"region": [0.9, 0.4], "qualified": False}
    metadata_path.write_text(json.dumps(metadata))
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    assets = QEDValueAssets(
        head_path,
        digest(head_path),
        norm_path,
        digest(norm_path),
        metadata_path,
        digest(metadata_path),
    )
    with pytest.raises(ValueError, match="did not qualify"):
        QEDSharedValueHead.load(value.reference, assets, expected_source_split_sha256="split-hash")


def test_value_asset_manifest_uses_fixed_names(tmp_path: Path) -> None:
    manifest = {
        "schema_version": "compose.qed.shared_value_assets.v1",
        "head": {"path": "head.pt", "sha256": "a" * 64},
        "normalization": {"path": "normalization.json", "sha256": "b" * 64},
        "metadata": {"path": "metadata.json", "sha256": "c" * 64},
    }
    (tmp_path / "assets.json").write_text(json.dumps(manifest))
    assets = QEDValueAssets.from_directory(tmp_path)
    assert assets.head == tmp_path / "head.pt"
    assert assets.metadata_sha256 == "c" * 64
    manifest["head"]["path"] = "../other.pt"
    (tmp_path / "assets.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="invalid head"):
        QEDValueAssets.from_directory(tmp_path)
