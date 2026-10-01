"""Build reproducible T4 cell configurations without running docking."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path

from compose_v4.experiments.t4.__main__ import read_config
from compose_v4.experiments.t4_quickvina import verified_file


def sha256(path: Path) -> str:
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load_panel(path: Path) -> tuple[dict, list[dict]]:
    """Check the task panel and its original lead transcription."""
    path = Path(path).resolve()
    panel = json.loads(path.read_text())
    if panel.get("schema_version") != "compose.t4.targets.v1":
        raise ValueError(f"unsupported T4 target panel: {path}")
    targets = panel.get("targets")
    if not isinstance(targets, dict) or set(targets) != {"5ht1b", "braf", "fa7", "jak2", "parp1"}:
        raise ValueError(f"T4 target panel has the wrong target set: {path}")
    required = {
        "center",
        "size",
        "receptor_sha256",
        "docking_seed",
        "route_expert",
        "route_expert_sha256",
    }
    if any(not isinstance(target, dict) or set(target) != required for target in targets.values()):
        raise ValueError(f"T4 target panel has incomplete target definitions: {path}")
    if not isinstance(panel.get("lead_file"), str) or not isinstance(panel.get("lead_sha256"), str):
        raise TypeError(f"T4 target panel has no string lead file identity: {path}")
    leads_path = (path.parent / panel["lead_file"]).resolve()
    verified_file(leads_path, panel["lead_sha256"])
    leads = json.loads(leads_path.read_text())
    if (
        not isinstance(leads, list)
        or len(leads) != 15
        or any(not isinstance(row, dict) for row in leads)
        or [row.get("idx") for row in leads] != list(range(15))
        or any(type(row["idx"]) is not int for row in leads)
        or any(
            row.get("target") not in targets
            or not isinstance(row.get("smiles"), str)
            or not row["smiles"]
            for row in leads
        )
        or any(sum(row["target"] == target for row in leads) != 3 for target in targets)
    ):
        raise ValueError(f"T4 panel must contain three leads per target: {leads_path}")
    return panel, leads


def cell_config(
    template: dict,
    lead: dict,
    target: dict,
    *,
    delta: float,
    replicate: int,
    base_seed: int,
    strength: float,
    obabel: Path,
    obabel_sha256: str,
    qvina: Path,
    receptor: Path,
    checkpoint: Path,
    checkpoint_sha256: str,
    route_expert: Path,
    qvina_sha256: str,
) -> dict:
    """Construct one active-reference config from fixed panel inputs."""
    if delta not in (0.4, 0.6) or type(replicate) is not int or replicate < 0:
        raise ValueError("T4 cell needs delta 0.4/0.6 and a nonnegative replicate")
    if type(base_seed) is not int or base_seed < 0:
        raise ValueError("T4 cell needs a nonnegative base seed")
    if type(strength) is not float or not 0 < strength <= 1:
        raise ValueError("T4 reference strength must be chosen in (0, 1]")
    config = deepcopy(template)
    config["search"]["lead"] = lead["smiles"]
    config["search"]["delta"] = delta
    config["search"]["seed"] = base_seed + 1000 * lead["idx"] + replicate
    config["search"]["guidance"]["mode"] = "active"
    config["search"]["guidance"]["strength"] = strength
    config["reference"] = {
        **template["reference"],
        "path": str(checkpoint.resolve()),
        "sha256": checkpoint_sha256,
        "catalog_path": str(
            (Path(__file__).resolve().parents[3] / "local_assets/fragments/catalog.json").resolve()
        ),
    }
    config["route_expert"] = {
        "path": str(route_expert.resolve()),
        "sha256": target["route_expert_sha256"],
    }
    config["docking"].update(
        {
            "obabel": str(obabel.resolve()),
            "obabel_sha256": obabel_sha256,
            "qvina": str(qvina.resolve()),
            "qvina_sha256": qvina_sha256,
            "receptor": str(receptor.resolve()),
            "receptor_sha256": target["receptor_sha256"],
            "center": target["center"],
            "size": target["size"],
            "seed": target["docking_seed"],
        }
    )
    return config


def guidance_arm_config(active_config: dict, mode: str) -> dict:
    """Change only the reference-selection arm of one prepared T4 cell."""
    if mode not in ("off", "shadow", "active"):
        raise ValueError(f"unknown T4 guidance arm: {mode!r}")
    if active_config["search"]["guidance"]["mode"] != "active":
        raise ValueError("T4 guidance arms require an active-reference base config")
    if active_config["reference"] is None:
        raise ValueError("T4 guidance arms require a reference asset in the base config")
    config = deepcopy(active_config)
    config["search"]["guidance"]["mode"] = mode
    if mode != "active":
        config["search"]["guidance"]["strength"] = 0.0
    if mode == "off":
        config["reference"] = None
    return config


def verify_cell_config(path: Path) -> None:
    """Exercise the same schema and source-state checks as the T4 runner."""
    read_config(path)
