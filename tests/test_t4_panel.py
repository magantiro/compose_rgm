"""A complete T4 panel can be prepared without scoring a molecule."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from compose_v4.experiments.t4.__main__ import read_config
from compose_v4.experiments.t4_panel import (
    cell_config,
    guidance_arm_config,
    load_panel,
    sha256,
)
from tools import prepare_t4_panel

ROOT = Path(__file__).resolve().parents[1]
PANEL = ROOT / "experiments/t4/targets.json"
TEMPLATE = ROOT / "experiments/t4/example.json"
CHECKPOINT = ROOT / "local_assets/fragments/r_theta_nll.pt"
REFERENCE_SHA256 = json.loads((ROOT / "experiments/reference/model.json").read_text())[
    "checkpoint"
]["sha256"]


def test_complete_panel_builds_both_thresholds_without_assets_or_docking(tmp_path):
    panel, leads = load_panel(PANEL)
    template = json.loads(TEMPLATE.read_text())
    assert len(leads) == 15
    assert len(panel["targets"]) == 5
    expert_paths = []
    for target in panel["targets"].values():
        path = PANEL.parent / target["route_expert"]
        assert sha256(path) == target["route_expert_sha256"]
        expert_paths.append(path)
    assert len(set(expert_paths)) == 5

    seen_seeds = set()
    for lead in leads:
        target = panel["targets"][lead["target"]]
        for delta in (0.4, 0.6):
            config = cell_config(
                template,
                lead,
                target,
                delta=delta,
                replicate=0,
                base_seed=20260918,
                strength=0.25,
                obabel=tmp_path / "obabel",
                obabel_sha256="0" * 64,
                qvina=tmp_path / "qvina02",
                qvina_sha256=panel["qvina_sha256"],
                receptor=tmp_path / f"{lead['target']}.pdbqt",
                checkpoint=CHECKPOINT,
                checkpoint_sha256=REFERENCE_SHA256,
                route_expert=PANEL.parent / target["route_expert"],
            )
            path = tmp_path / f"cell_{lead['idx']}_{delta}.json"
            path.write_text(json.dumps(config))
            _, search, docking = read_config(path)
            assert search.lead == lead["smiles"]
            assert search.delta == delta
            assert search.budget == 250
            assert search.guidance.mode == "active"
            assert search.guidance.strength == 0.25
            assert docking.center == tuple(target["center"])
            assert docking.size == tuple(target["size"])
            assert search.seed == 20260918 + 1000 * lead["idx"]
            if delta == 0.4:
                assert search.seed not in seen_seeds
                seen_seeds.add(search.seed)
    assert len(seen_seeds) == 15


def test_panel_rejects_changed_leads(tmp_path):
    panel = json.loads(PANEL.read_text())
    panel["lead_file"] = "leads.json"
    (tmp_path / "leads.json").write_text("[]")
    path = tmp_path / "targets.json"
    path.write_text(json.dumps(panel))
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        load_panel(path)


def test_guidance_arms_preserve_cell_inputs_and_change_only_reference_selection(tmp_path):
    panel, leads = load_panel(PANEL)
    lead = leads[0]
    template = json.loads(TEMPLATE.read_text())
    base = cell_config(
        template,
        lead,
        panel["targets"][lead["target"]],
        delta=0.4,
        replicate=1,
        base_seed=20260918,
        strength=0.25,
        obabel=tmp_path / "obabel",
        obabel_sha256="0" * 64,
        qvina=tmp_path / "qvina02",
        qvina_sha256=panel["qvina_sha256"],
        receptor=tmp_path / "receptor.pdbqt",
        checkpoint=tmp_path / "checkpoint.pt",
        checkpoint_sha256="1" * 64,
        route_expert=PANEL.parent / panel["targets"][lead["target"]]["route_expert"],
    )
    for mode in ("off", "shadow", "active"):
        config = guidance_arm_config(base, mode)
        path = tmp_path / f"{mode}.json"
        path.write_text(json.dumps(config))
        _, search, docking = read_config(path)
        assert search.guidance.mode == mode
        assert search.seed == base["search"]["seed"]
        assert docking.seed == base["docking"]["seed"]
        assert config["docking"] == base["docking"]
        assert config["route_expert"] == base["route_expert"]
        assert config["search"] | {"guidance": base["search"]["guidance"]} == base["search"]
        assert (config["reference"] is None) == (mode == "off")
        if mode != "off":
            assert config["reference"] == base["reference"]
    with pytest.raises(ValueError, match="unknown T4 guidance arm"):
        guidance_arm_config(base, "learned")


def test_preparer_writes_three_matched_arms_without_docking(tmp_path, monkeypatch):
    output = tmp_path / "panel"
    monkeypatch.setattr(prepare_t4_panel, "verified_file", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prepare_t4_panel.py",
            "--output",
            str(output),
            "--obabel",
            str(tmp_path / "obabel"),
            "--qvina",
            str(tmp_path / "qvina02"),
            "--receptors",
            str(tmp_path / "receptors"),
            "--checkpoint",
            str(tmp_path / "checkpoint.pt"),
            "--base-seed",
            "20260918",
            "--replicates",
            "1",
            "--strength",
            "0.25",
            "--guidance-modes",
            "off",
            "shadow",
            "active",
        ],
    )
    (tmp_path / "obabel").write_bytes(b"test binary")
    prepare_t4_panel.main()
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["docking_calls"] == 0
    assert len(manifest["configurations"]) == 90
    assert set(manifest["guidance_modes"]) == {"off", "shadow", "active"}
    for lead in range(15):
        for delta in (4, 6):
            rows = [
                row
                for row in manifest["configurations"]
                if row["lead_index"] == lead and int(row["delta"] * 10) == delta
            ]
            assert len(rows) == 3
            configs = {
                row["guidance_mode"]: json.loads((output / row["path"]).read_text()) for row in rows
            }
            assert len({config["search"]["seed"] for config in configs.values()}) == 1
            assert len({config["docking"]["seed"] for config in configs.values()}) == 1
            assert configs["off"]["reference"] is None
            assert configs["active"]["reference"] == configs["shadow"]["reference"]
            for arm in ("off", "shadow"):
                assert (
                    configs[arm]["search"] | {"guidance": configs["active"]["search"]["guidance"]}
                    == configs["active"]["search"]
                )


def test_panel_rejects_incomplete_target_definition(tmp_path):
    panel = json.loads(PANEL.read_text())
    del panel["targets"]["fa7"]["docking_seed"]
    path = tmp_path / "targets.json"
    path.write_text(json.dumps(panel))
    with pytest.raises(ValueError, match="incomplete target definitions"):
        load_panel(path)


def test_cell_config_requires_explicit_supported_strength():
    panel, leads = load_panel(PANEL)
    template = json.loads(TEMPLATE.read_text())
    target = panel["targets"][leads[0]["target"]]
    paths = {
        "obabel": Path("/tmp/obabel"),
        "obabel_sha256": "0" * 64,
        "qvina": Path("/tmp/qvina"),
        "qvina_sha256": panel["qvina_sha256"],
        "receptor": Path("/tmp/receptor"),
        "checkpoint": Path("/tmp/checkpoint"),
        "checkpoint_sha256": "1" * 64,
        "route_expert": Path("/tmp/route_expert"),
    }
    with pytest.raises(ValueError, match="reference strength"):
        cell_config(
            template,
            leads[0],
            target,
            delta=0.6,
            replicate=0,
            base_seed=13,
            strength=0.0,
            **paths,
        )
    with pytest.raises(ValueError, match="nonnegative replicate"):
        cell_config(
            template,
            leads[0],
            target,
            delta=0.6,
            replicate=-1,
            base_seed=13,
            strength=0.25,
            **paths,
        )
