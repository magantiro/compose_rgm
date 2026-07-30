"""Frozen command recipes for tracelet Generator Matching experiments."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

_BOOLEAN_OPTIONAL_ARGUMENTS = frozenset(
    {
        # argparse.BooleanOptionalAction in train_tracelet_cnof_gate.py.
        # False is an explicit support choice, not the absence of a store_true flag.
        "enable_ring_restates",
        "enable_ring_system_delete",
    }
)


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
    if arguments.get("factorized_training_objective") == "canonical_successor":
        if arguments.get("skip_rollouts") is not True:
            raise ValueError(
                "canonical-successor editing recipes must freeze skip_rollouts=true"
            )
        missing_capabilities = [
            name
            for name in (
                "enable_ring_restates",
                "enable_ring_system_delete",
            )
            if type(arguments.get(name)) is not bool
        ]
        if missing_capabilities:
            raise ValueError(
                "canonical-successor editing recipes must explicitly freeze "
                f"Boolean capabilities: {missing_capabilities}"
            )
    argv = [str(smiles_file)]
    for name, value in arguments.items():
        flag = f"--{name.replace('_', '-')}"
        if isinstance(value, bool):
            if value:
                argv.append(flag)
            elif name in _BOOLEAN_OPTIONAL_ARGUMENTS:
                argv.append(f"--no-{name.replace('_', '-')}")
            continue
        argv.extend((flag, str(value)))
    if quality_reference_file is not None:
        argv.extend(("--quality-reference-file", str(quality_reference_file)))
    return tuple(argv)


__all__ = ["build_tracelet_recipe_argv", "load_tracelet_recipe"]
