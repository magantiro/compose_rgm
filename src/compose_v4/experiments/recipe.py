"""Frozen command recipes for tracelet Generator Matching experiments."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_tracelet_recipe(path: Path) -> dict[str, Any]:
    recipe = json.loads(path.read_text())
    if not isinstance(recipe, dict):
        raise ValueError("recipe must contain a JSON object")
    if not isinstance(recipe.get("name"), str) or not recipe["name"]:
        raise ValueError("recipe requires a non-empty name")
    arguments = recipe.get("arguments")
    if not isinstance(arguments, dict) or not arguments:
        raise ValueError("recipe requires a non-empty arguments object")
    unknown_values = [
        key
        for key, value in arguments.items()
        if not isinstance(value, (str, int, float, bool)) or value is None
    ]
    if unknown_values:
        raise ValueError(
            f"recipe arguments must be scalar JSON values: {sorted(unknown_values)}"
        )
    return recipe


def build_tracelet_recipe_argv(
    recipe: dict[str, Any],
    *,
    smiles_file: Path,
    quality_reference_file: Path | None = None,
) -> tuple[str, ...]:
    """Materialize a recipe into the existing trainer's command-line API."""

    arguments = recipe.get("arguments")
    if not isinstance(arguments, dict):
        raise ValueError("recipe requires an arguments object")
    argv = [str(smiles_file)]
    for name, value in arguments.items():
        flag = f"--{name.replace('_', '-')}"
        if isinstance(value, bool):
            if value:
                argv.append(flag)
            continue
        argv.extend((flag, str(value)))
    if quality_reference_file is not None:
        argv.extend(("--quality-reference-file", str(quality_reference_file)))
    return tuple(argv)


__all__ = ["build_tracelet_recipe_argv", "load_tracelet_recipe"]
