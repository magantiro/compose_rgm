from __future__ import annotations

from pathlib import Path

import pytest

from compose_v4.experiments.recipe import (
    build_tracelet_recipe_argv,
    load_tracelet_recipe,
)


def test_boolean_optional_false_is_not_dropped_from_recipe_argv() -> None:
    argv = build_tracelet_recipe_argv(
        {
            "arguments": {
                "enable_ring_restates": False,
                "enable_ring_system_delete": False,
                "quality_metrics": False,
            }
        },
        smiles_file=Path("train.smi"),
    )

    assert "--no-enable-ring-restates" in argv
    assert "--no-enable-ring-system-delete" in argv
    assert "--quality-metrics" not in argv


def test_canonical_successor_recipe_requires_no_rollouts_and_explicit_support() -> None:
    base = {
        "factorized_training_objective": "canonical_successor",
        "skip_rollouts": True,
        "enable_ring_restates": True,
        "enable_ring_system_delete": False,
    }
    argv = build_tracelet_recipe_argv(
        {"arguments": base},
        smiles_file=Path("train.smi"),
    )
    assert "--skip-rollouts" in argv
    assert "--enable-ring-restates" in argv
    assert "--no-enable-ring-system-delete" in argv

    for missing in ("skip_rollouts", "enable_ring_restates"):
        arguments = dict(base)
        arguments.pop(missing)
        with pytest.raises(ValueError):
            build_tracelet_recipe_argv(
                {"arguments": arguments},
                smiles_file=Path("train.smi"),
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


def test_stage3_recipe_has_early_selection_and_throughput_gates() -> None:
    root = Path(__file__).resolve().parents[1]
    recipe = load_tracelet_recipe(
        root / "recipes" / "tree_fcd_transfer_stage3_flexible_graft.json"
    )
    arguments = recipe["arguments"]

    assert arguments["warmup_steps"] == 500
    assert arguments["evaluation_every"] == 250
    assert arguments["early_stopping_patience"] == 6
    assert arguments["early_stopping_min_relative_delta"] == 0.001
    assert arguments["recovery_every"] == 500
    assert arguments["data_workers"] == 24
    assert arguments["data_prefetch_factor"] == 2


def test_stage3_integration_smoke_recipe_is_tiny_but_semantically_matched() -> None:
    root = Path(__file__).resolve().parents[1]
    production = load_tracelet_recipe(
        root / "recipes" / "tree_fcd_transfer_stage3_flexible_graft.json"
    )["arguments"]
    smoke = load_tracelet_recipe(
        root / "recipes" / "tree_fcd_transfer_stage3_integration_smoke.json"
    )["arguments"]

    assert smoke["train_size"] == 32
    assert smoke["validation_size"] == 4
    assert smoke["test_size"] == 4
    assert smoke["steps"] == 2
    assert smoke["fast_split"] is True
    assert smoke["early_stopping_patience"] == 0
    assert "skip_rollouts" not in smoke
    for key in (
        "training_backend",
        "rate_factorization",
        "ring_proposals",
        "ring_electronic_mode",
        "source_prior",
        "tree_size_prior",
        "tree_transport",
        "bond_representation",
        "teacher_ordering",
    ):
        assert smoke[key] == production[key]


def test_frozen_cnof_manifest_flag_survives_recipe_materialization() -> None:
    argv = build_tracelet_recipe_argv(
        {
            "arguments": {
                "train_size": 50000,
                "frozen_cnof_split_manifest": "frozen_split.json",
            }
        },
        smiles_file=Path("/guacamol/source.smiles"),
    )
    assert argv[:3] == ("/guacamol/source.smiles", "--train-size", "50000")
    assert argv[-2:] == ("--frozen-cnof-split-manifest", "frozen_split.json")
