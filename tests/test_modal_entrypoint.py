from __future__ import annotations

import ast
from pathlib import Path


def test_modal_entrypoint_uses_frozen_recipe_and_persistent_artifacts() -> None:
    root = Path(__file__).resolve().parents[1]
    path = root / "modal_apps" / "train_tracelet_gm.py"
    source = path.read_text()
    tree = ast.parse(source)

    function_names = {
        node.name for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    assert {
        "_materialize_remote_recipe",
        "_run_remote",
        "smoke_stage",
        "preflight_stage",
        "compile_stage",
        "train_stage",
        "evaluate_stage",
        "pipeline_stage",
        "main",
    } <= function_names
    assert "tree_fcd_transfer_stage1.json" in source
    assert "compose-v4-artifacts" in source
    assert "guacamol_heldout_val_5000_seed0.smiles" in source
    assert "train_stage.spawn" in source
    assert "evaluate_stage.spawn" in source
    assert "pipeline_stage.spawn" in source
    assert "compile_paths_only=True" in source
    assert "require_path_cache=True" in source
    assert "compiled_proposal_shard_saved" in source
    assert "compiled_path_shard_saved" in source
    assert "subprocess.Popen" in source
    assert "artifact_volume.commit()" in source
    assert "source_sha256" in source
