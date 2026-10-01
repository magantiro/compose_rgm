"""Deterministic PMO task, seed and arm configurations without oracle calls."""

from __future__ import annotations

import copy
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from compose_v4.control.reference_guidance import GuidanceConfig

ARMS = ("structured", "uniform_chain", "created_atom_rebinding")
GUIDANCE_MODES = ("off", "shadow", "active")


def load_panel(path: Path, *, oracle_tasks: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the fixed PMO objective and seed registry."""
    value = json.loads(path.read_text())
    expected = {
        "schema_version",
        "seed_indices",
        "excluded_from_22_objective_mean",
        "seed_policy_source",
    }
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError(f"invalid PMO panel registry fields: {path}")
    if value["schema_version"] != "compose.pmo.panel.v1":
        raise ValueError(f"unsupported PMO panel registry: {path}")
    indices = value["seed_indices"]
    if (
        not isinstance(indices, dict)
        or set(indices) != set(oracle_tasks)
        or any(type(index) is not int for index in indices.values())
        or sorted(indices.values()) != list(range(len(indices)))
    ):
        raise ValueError(f"PMO seed indices must cover the oracle registry exactly: {path}")
    if value["excluded_from_22_objective_mean"] != ["valsartan_smarts"]:
        raise ValueError(f"PMO 22-objective exclusion changed: {path}")
    if len(indices) != 23:
        raise ValueError(
            f"PMO panel requires 23 declared objectives, including the exclusion: {path}"
        )
    provenance = value["seed_policy_source"]
    if (
        not isinstance(provenance, dict)
        or set(provenance) != {"revision", "path", "sha256"}
        or not all(isinstance(item, str) and item for item in provenance.values())
    ):
        raise ValueError(f"PMO seed-policy provenance is incomplete: {path}")
    return value


def derived_seed(task: str, base_seed: int, replicate: int, indices: Mapping[str, int]) -> int:
    """Preserve the published controller's stable task index derivation."""
    if task not in indices:
        raise ValueError(f"unknown PMO task: {task}")
    if type(base_seed) is not int or base_seed < 0:
        raise ValueError("base_seed must be a nonnegative integer")
    if type(replicate) is not int or not 0 <= replicate < 1000:
        raise ValueError("replicate must be an integer in [0, 1000)")
    return base_seed + 1000 * indices[task] + replicate


def cell_config(
    template: Mapping[str, Any],
    *,
    template_dir: Path,
    task: str,
    seed: int,
    arm: str,
    guidance_mode: str,
    strength: float,
    oracle_python: Path,
    oracle_assets: Path | None,
    checkpoint: Path,
    catalog: Path,
) -> dict[str, Any]:
    """Copy one campaign template while keeping non-intervention fields fixed."""
    if arm not in ARMS or guidance_mode not in GUIDANCE_MODES:
        raise ValueError(f"unsupported PMO proposal arm or guidance mode: {arm}, {guidance_mode}")
    config = copy.deepcopy(dict(template))
    config["task"] = task
    config["seed"] = seed
    config["arm"] = arm
    config["guidance"]["mode"] = guidance_mode
    config["guidance"]["strength"] = strength if guidance_mode == "active" else 0.0
    GuidanceConfig(**config["guidance"])
    for name in ("jump_plans", "initialization"):
        config[name]["path"] = str((template_dir / config[name]["path"]).resolve())
    if guidance_mode == "off":
        config["reference"] = None
    else:
        config["reference"]["path"] = str(checkpoint.resolve())
        config["reference"]["catalog_path"] = str(catalog.resolve())
    config["oracle"]["python"] = str(oracle_python.resolve())
    config["oracle"]["asset_root"] = None if oracle_assets is None else str(oracle_assets.resolve())
    return config


__all__ = ["ARMS", "GUIDANCE_MODES", "cell_config", "derived_seed", "load_panel"]
