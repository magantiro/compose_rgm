"""Path-neutral fragment exports preserve the sampling inputs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from tools import export_fragment_assets as export


def _artifact(schema: str, *, source: bool, inputs: bool) -> dict:
    artifact = {"schema": schema, "counts": [[3, 0, 7]], "source_sha256": "x" * 64}
    if source:
        artifact["source"] = "/Users/example/research/guacamol/train.smiles"
    if inputs:
        artifact["input_sha256"] = {
            "/Users/example/research/guacamol/train.smiles": "a" * 64,
            "/Users/example/project/src/compose_v4/benchmark/prior.py": "b" * 64,
        }
    return artifact


def test_normalization_changes_only_path_provenance():
    original = _artifact(export.SCHEMAS["mass_prior"], source=True, inputs=True)
    clean = export.normalize(original, "mass_prior")
    assert clean["source"] == "guacamol/train.smiles"
    assert clean["input_sha256"] == {
        "guacamol/train.smiles": "a" * 64,
        "src/compose_v4/benchmark/prior.py": "b" * 64,
    }
    assert export.scientific_payload(clean) == export.scientific_payload(original)
    assert original["source"].startswith("/Users/")


def test_unrecognized_path_and_colliding_labels_fail():
    with pytest.raises(ValueError, match="no portable source label"):
        export.portable_label("/Users/example/private/table.csv")
    asset = _artifact(export.SCHEMAS["joint_prior"], source=False, inputs=True)
    asset["input_sha256"]["/home/other/guacamol/train.smiles"] = "c" * 64
    with pytest.raises(ValueError, match="source labels collide"):
        export.normalize(asset, "joint_prior")


def test_nested_data_directory_keeps_source_root():
    path = "/Users/example/project/src/compose_v4/data/scaffold_partition.py"
    assert export.portable_label(path) == "src/compose_v4/data/scaffold_partition.py"


def test_path_neutral_export_is_idempotent():
    original = _artifact(export.SCHEMAS["mass_prior"], source=True, inputs=True)
    clean = export.normalize(original, "mass_prior")
    assert export.normalize(clean, "mass_prior") == clean
    with pytest.raises(ValueError, match="unrecognized portable source label"):
        export.portable_label("../private/table.csv")


def test_export_rejects_wrong_source_and_never_replaces_output(tmp_path: Path):
    original = _artifact(export.SCHEMAS["region_catalog"], source=True, inputs=False)
    source = tmp_path / "input.json"
    source.write_text(json.dumps(original))
    target = tmp_path / "output.json"
    expected = hashlib.sha256(source.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="source identity mismatch"):
        export.export_one(source, target, key="region_catalog", expected="0" * 64)
    result = export.export_one(source, target, key="region_catalog", expected=expected)
    assert result["scientific_payload_equal"] is True
    assert b"/Users/" not in target.read_bytes()
    target.write_text("unrelated data")
    with pytest.raises(FileExistsError, match="refusing to replace"):
        export.export_one(source, target, key="region_catalog", expected=expected)


def test_export_requires_distinct_source_and_output(tmp_path: Path):
    with pytest.raises(ValueError, match="must not overwrite"):
        export.export_all(tmp_path, tmp_path)


def test_export_writes_path_free_receipt(tmp_path: Path, monkeypatch):
    source_dir = tmp_path / "inputs"
    output_dir = tmp_path / "outputs"
    source_dir.mkdir()
    assets = {}
    for key in export.ASSET_KEYS:
        artifact = _artifact(
            export.SCHEMAS[key],
            source=key != "joint_prior",
            inputs=key in {"joint_prior", "mass_prior"},
        )
        path = source_dir / f"{key}.json"
        path.write_text(json.dumps(artifact))
        assets[key] = {"path": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    manifest = tmp_path / "assets.json"
    manifest.write_text(json.dumps({"schema": "compose_fragment_assets_v1", "assets": assets}))
    monkeypatch.setattr(export, "MANIFEST", manifest)

    receipt = export.export_all(source_dir, output_dir)
    assert receipt == export.export_all(source_dir, output_dir)
    assert receipt["source_manifest_sha256"] == hashlib.sha256(manifest.read_bytes()).hexdigest()
    assert receipt["implementation_sha256"] == export.sha256(Path(export.__file__))
    assert {row["asset"] for row in receipt["assets"]} == set(export.ASSET_KEYS)
    assert b"/Users/" not in (output_dir / "export_manifest.json").read_bytes()
    assert all(row["scientific_payload_equal"] for row in receipt["assets"])

    (output_dir / "export_manifest.json").write_text("different export")
    with pytest.raises(FileExistsError, match="refusing to replace different export"):
        export.export_all(source_dir, output_dir)

    output_dir.rename(tmp_path / "other-output")
    changed = json.loads(manifest.read_text())
    changed["assets"]["mass_prior"]["sha256"] = "0" * 64
    manifest.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="mass_prior: source identity mismatch"):
        export.export_all(source_dir, output_dir)
    assert not output_dir.exists()


def test_bundled_completion_priors_are_portable_and_hash_pinned():
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((root / "experiments/fragments/assets.json").read_text())
    for key in ("joint_prior", "mass_prior"):
        entry = manifest["assets"][key]
        path = root / "local_assets/fragments" / entry["path"]
        raw = path.read_bytes()
        assert hashlib.sha256(raw).hexdigest() == entry["sha256"]
        assert b"/Users/" not in raw
        assert b"/home/" not in raw
        payload = json.loads(raw)
        assert export.normalize(payload, key) == payload
