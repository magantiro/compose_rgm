"""PMO panel preparation is deterministic and does not touch an oracle."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from compose_v4.experiments.pmo_panel import cell_config, derived_seed, load_panel
from tools.prepare_pmo_panel import main

ROOT = Path(__file__).resolve().parents[1]
TASK_DIR = ROOT / "experiments/pmo"
PANEL = TASK_DIR / "panel.json"
TEMPLATE = TASK_DIR / "example.json"
CHECKPOINT = ROOT / "local_assets/fragments/r_theta_nll.pt"
CATALOG = ROOT / "local_assets/fragments/catalog.json"


def _panel() -> dict:
    audit = json.loads((TASK_DIR / "assets/oracle_asset_audit.json").read_text())
    return load_panel(PANEL, oracle_tasks=audit["tasks"])


def test_task_registry_covers_the_22_objective_protocol():
    panel = _panel()
    indices = panel["seed_indices"]
    assert len(indices) == 23
    assert panel["excluded_from_22_objective_mean"] == ["valsartan_smarts"]
    assert derived_seed("albuterol_similarity", 20260923, 2, indices) == 20261925
    assert derived_seed("gsk3b", 20260923, 2, indices) == 20268925
    assert derived_seed("scaffold_hop", 20260923, 2, indices) == 20279925
    assert (
        len({derived_seed(task, 20260923, rep, indices) for task in indices for rep in range(3)})
        == 69
    )
    with pytest.raises(ValueError, match="replicate"):
        derived_seed("albuterol_similarity", 20260923, 1000, indices)
    with pytest.raises(ValueError, match="unknown PMO task"):
        derived_seed("not_an_oracle", 20260923, 0, indices)


def test_registry_rejects_missing_task_and_bad_index_type(tmp_path):
    audit = json.loads((TASK_DIR / "assets/oracle_asset_audit.json").read_text())
    value = json.loads(PANEL.read_text())
    value["seed_indices"]["gsk3b"] = "8"
    path = tmp_path / "bad-panel.json"
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="seed indices"):
        load_panel(path, oracle_tasks=audit["tasks"])
    value["seed_indices"].pop("gsk3b")
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="seed indices"):
        load_panel(path, oracle_tasks=audit["tasks"])


def test_arm_and_reference_modes_change_only_their_declared_fields(tmp_path):
    template = json.loads(TEMPLATE.read_text())
    configs = {}
    for arm in ("structured", "uniform_chain", "created_atom_rebinding"):
        for mode in ("off", "shadow", "active"):
            configs[arm, mode] = cell_config(
                template,
                template_dir=TEMPLATE.parent,
                task="albuterol_similarity",
                seed=20261925,
                arm=arm,
                guidance_mode=mode,
                strength=0.25,
                oracle_python=tmp_path / "oracle-python",
                oracle_assets=None,
                checkpoint=CHECKPOINT,
                catalog=CATALOG,
            )
    structured = configs["structured", "active"]
    assert structured["guidance"]["strength"] == 0.25
    assert structured["reference"]["path"] == str(CHECKPOINT)
    assert structured["reference"]["catalog_path"] == str(CATALOG)
    for arm in ("uniform_chain", "created_atom_rebinding"):
        variant = configs[arm, "active"]
        assert variant | {"arm": "structured"} == structured
    for mode in ("off", "shadow"):
        variant = configs["structured", mode]
        assert variant["guidance"]["strength"] == 0
        assert (variant["reference"] is None) == (mode == "off")
        assert (
            variant | {"guidance": structured["guidance"], "reference": structured["reference"]}
            == structured
        )
    assert template["reference"]["path"].startswith("../")


def test_preparer_fails_before_oracle_use_for_missing_assets_and_excluded_task(tmp_path):
    oracle_python = tmp_path / "oracle-python"
    oracle_python.write_text("unused test executable\n")
    oracle_python.chmod(0o700)
    common = [
        "--output",
        str(tmp_path / "panel"),
        "--oracle-python",
        str(oracle_python),
        "--base-seed",
        "20260923",
        "--replicates",
        "1",
        "--strength",
        "0.25",
    ]
    with pytest.raises(ValueError, match="require --oracle-assets"):
        main([*common, "--tasks", "gsk3b"])
    with pytest.raises(SystemExit, match="2"):
        main([*common, "--tasks", "valsartan_smarts"])
    assert not (tmp_path / "panel").exists()


@pytest.mark.external_artifact
def test_preparer_writes_matched_configs_without_oracle_calls(tmp_path, capsys):
    if not CHECKPOINT.exists():
        pytest.skip("hash-pinned NLL reference weights are not installed")
    with CHECKPOINT.open("rb") as stream:
        is_pointer = stream.read(64).startswith(b"version https://git-lfs.github.com/spec/v1")
    if is_pointer:
        pytest.skip("hash-pinned NLL reference weights are not installed")
    oracle_python = tmp_path / "oracle-python"
    oracle_python.write_text("unused test executable\n")
    oracle_python.chmod(0o700)
    output = tmp_path / "panel"
    argv = [
        "--output",
        str(output),
        "--oracle-python",
        str(oracle_python),
        "--base-seed",
        "20260923",
        "--replicates",
        "1",
        "--strength",
        "0.25",
        "--tasks",
        "albuterol_similarity",
        "--arms",
        "structured",
        "uniform_chain",
        "--guidance-modes",
        "off",
        "shadow",
        "active",
    ]
    main(argv)
    summary = json.loads(capsys.readouterr().out)
    manifest = json.loads((output / "manifest.json").read_text())
    assert summary["configs"] == len(manifest["configurations"]) == 6
    assert summary["oracle_calls"] == manifest["oracle_calls"] == 0
    assert {row["seed"] for row in manifest["configurations"]} == {20261923}
    assert {row["arm"] for row in manifest["configurations"]} == {"structured", "uniform_chain"}
    assert {row["guidance_mode"] for row in manifest["configurations"]} == {
        "off",
        "shadow",
        "active",
    }
    assert all((output / row["path"]).is_file() for row in manifest["configurations"])
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        main(argv)
