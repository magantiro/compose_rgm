from __future__ import annotations

from pathlib import Path

from compose_v4.experiments.recipe import (
    build_tracelet_recipe_argv,
    load_tracelet_recipe,
)


def test_frozen_tree_fcd_recipe_materializes_expected_command() -> None:
    root = Path(__file__).resolve().parents[1]
    recipe = load_tracelet_recipe(root / "recipes" / "tree_fcd_transfer_stage1.json")
    argv = build_tracelet_recipe_argv(
        recipe,
        smiles_file=Path("train.smi"),
        quality_reference_file=Path("heldout.smi"),
    )

    assert argv[0] == "train.smi"
    assert "--source-prior" in argv
    assert argv[argv.index("--source-prior") + 1] == "carbon_tree"
    assert "--ring-electronic-mode" in argv
    assert argv[argv.index("--ring-electronic-mode") + 1] == "factorized_local"
    assert argv[argv.index("--tree-couplings-per-target") + 1] == "4"
    assert "--quality-metrics" in argv
    assert "--include-fcd" in argv
    assert argv[-2:] == ("--quality-reference-file", "heldout.smi")
