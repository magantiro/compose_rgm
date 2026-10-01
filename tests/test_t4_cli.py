from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from compose_v4.experiments.t4.__main__ import read_config
from tests.test_t4_route_proposals import LEAD

ROOT = Path(__file__).resolve().parents[1]


def configuration(tmp_path):
    tools = tmp_path / "tools"
    tools.mkdir()
    data = b"non-executable test content, validation must not execute it"
    digest = hashlib.sha256(data).hexdigest()
    for name in ("obabel", "qvina", "receptor"):
        path = tools / name
        path.write_bytes(data)
        path.chmod(0o700)
    return {
        "schema_version": "compose.t4.config.v1",
        "search": {"lead": LEAD, "delta": 0.4, "seed": 7, "lanes": ["shallow"]},
        "reference": None,
        "route_expert": None,
        "docking": {
            "obabel": "tools/obabel",
            "obabel_sha256": digest,
            "qvina": "tools/qvina",
            "qvina_sha256": digest,
            "receptor": "tools/receptor",
            "receptor_sha256": digest,
            "center": [1, 2, 3],
            "size": [10, 11, 12],
            "seed": 42,
        },
    }


def test_validate_from_source_export_needs_no_git_cloud_or_docking(tmp_path):
    config = tmp_path / "config.json"
    config.write_text(json.dumps(configuration(tmp_path)))
    export = tmp_path / "source"
    shutil.copytree(
        ROOT / "src", export / "src", ignore=shutil.ignore_patterns("__pycache__", "*.pyc")
    )
    result = subprocess.run(
        [sys.executable, "-m", "compose_v4.experiments.t4", "validate", "--config", str(config)],
        cwd=export,
        env={**os.environ, "PYTHONPATH": str(export / "src"), "OMP_NUM_THREADS": "1"},
        capture_output=True,
        text=True,
        timeout=45,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["status"] == "files_verified" and payload["evaluations"] == 0
    assert payload["reference"] is None
    assert "executables not run" in payload["note"]
    assert not (export / "docking").exists()


@pytest.mark.parametrize(
    "field,value,message",
    [
        ("schema_version", "unknown", "config.v1"),
        ("reference", {"path": "missing.pt"}, "reference expects exactly"),
        (
            "route_expert",
            {"path": "missing.json", "sha256": "a" * 64},
            "route_expert is required exactly",
        ),
    ],
)
def test_cli_rejects_bad_config_before_accessing_assets(tmp_path, field, value, message):
    payload = configuration(tmp_path)
    payload[field] = value
    path = tmp_path / "config.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match=message):
        read_config(path)


def test_example_settings_parse_without_loading_assets():
    _, config, docking = read_config(ROOT / "experiments/t4/example.json")
    assert config.guidance.mode == "active" and config.guidance.strength > 0
    assert config.guidance.on_unsupported == "preserve_mass"
    assert config.budget == 250
    assert docking.seed == 20260918


def test_config_errors_name_the_input_file(tmp_path):
    payload = configuration(tmp_path)
    payload["search"]["unknown_field"] = True
    path = tmp_path / "malformed.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="malformed.json.*unknown_field"):
        read_config(path)


def test_unreadable_json_names_the_input_file(tmp_path):
    path = tmp_path / "broken.json"
    path.write_text("{not json}")
    with pytest.raises(ValueError, match="broken.json"):
        read_config(path)


@pytest.mark.parametrize(
    "mode,asset",
    [
        ("active", None),
        (
            "off",
            {
                "path": "reference.pt",
                "sha256": "a" * 64,
                "catalog_path": "catalog.json",
                "catalog_sha256": "b" * 64,
                "catalog_fingerprint": "fixture",
            },
        ),
    ],
)
def test_reference_configuration_cannot_silently_be_ignored(tmp_path, mode, asset):
    payload = configuration(tmp_path)
    payload["search"]["guidance"] = {"mode": mode, "strength": 0.25 if mode == "active" else 0}
    payload["reference"] = asset
    path = tmp_path / "reference-config.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="reference asset is required"):
        read_config(path)
