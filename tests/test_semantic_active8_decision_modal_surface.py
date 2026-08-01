"""Static and behavioral safety tests for the semantic Active8 Modal app."""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
APP_PATH = ROOT / "modal_apps" / "run_semantic_active8_decisions_app.py"


def _function(tree: ast.Module, name: str) -> ast.FunctionDef:
    return next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )


def _load_launcher():
    spec = importlib.util.spec_from_file_location(
        "semantic_active8_decision_launcher", APP_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_modal_surface_is_cpu_only_restart_safe_and_non_authorizing() -> None:
    source = APP_PATH.read_text()
    tree = ast.parse(source)
    assert "gpu=" not in source
    assert "MAX_MAP_CONTAINERS = 100" in source
    worker_source = ast.get_source_segment(source, _function(tree, "decide_one_chunk"))
    assert worker_source is not None
    assert "_worker_model_and_checker" in worker_source
    assert "plan_artifact_path" in worker_source
    assert "_load_json_exact" in worker_source
    assert "model_runtime_identity_resolver" in worker_source
    assert "artifact_volume.reload()" in worker_source
    assert "artifact_volume.commit()" in worker_source
    cache_source = ast.get_source_segment(
        source, _function(tree, "_worker_model_and_checker")
    )
    assert cache_source is not None
    assert "ProductionSemanticExactCandidateChecker" in cache_source
    assert "build_exact_semantic_model" in cache_source

    driver_source = ast.get_source_segment(source, _function(tree, "driver"))
    assert driver_source is not None
    assert "semantic_migration_completion" in driver_source
    assert "semantic_chunk_cache_plan" in driver_source
    assert "semantic_chunk_cache_global_completion" in driver_source
    assert "completed_semantic_active8_decision_task_ids" in driver_source
    assert "write_semantic_active8_decision_plan" in driver_source
    assert "decide_one_chunk.starmap" in driver_source
    assert "reduce_decisions.remote" in driver_source

    main_source = ast.get_source_segment(source, _function(tree, "main"))
    assert main_source is not None
    assert "expected_commit" in main_source
    assert "driver.remote" in main_source
    assert "driver.spawn" not in main_source
    assert '"training_launched": False' in main_source
    assert '"gate_zero_authorized": False' in main_source
    assert '"t1_authorized": False' in main_source
    assert '"bounded_p50_authorized": False' in main_source


def test_runtime_contract_is_exact_semantic_parent_and_self_hashed() -> None:
    launcher = _load_launcher()
    loaded = launcher._imports(ROOT)
    runtime, semantic = launcher._load_runtime_contract(root=ROOT, loaded=loaded)
    assert runtime["training_authorized"] is False
    assert runtime["gate_zero_authorized"] is False
    assert runtime["t1_authorized"] is False
    assert runtime["bounded_p50_authorized"] is False
    assert runtime["model"] == {
        "initialization_seed": 20260730,
        "max_atoms": 40,
        "hidden_dim": 256,
        "message_passing_steps": 6,
        "mark_dim": 32,
        "dtype": "torch.float32",
        "atom_vocabulary_class_count": 15,
        "catalog_fingerprint": "639ff6078c32d43c",
        "candidate_time": 0.5,
        "candidate_cache_size": 4096,
    }
    assert (
        runtime["semantic_model_process_contract"]["contract_sha256"] == semantic.sha256
    )
    assert runtime["semantic_model_process_contract"][
        "file_sha256"
    ] == launcher._file_sha256(ROOT / launcher.SEMANTIC_CONTRACT_SOURCE)


def test_source_revision_rejects_dirty_or_wrong_commit_and_hashes_all_inputs(
    monkeypatch,
) -> None:
    launcher = _load_launcher()

    def clean_git(_root, *arguments):
        if arguments == ("rev-parse", "HEAD"):
            return "a" * 40
        if arguments == ("rev-parse", "HEAD^{tree}"):
            return "b" * 40
        if arguments == ("status", "--porcelain=v1", "--untracked-files=all"):
            return ""
        raise AssertionError(arguments)

    monkeypatch.setattr(launcher, "_git", clean_git)
    revision = launcher.local_source_revision(expected_commit="a" * 40, repo_root=ROOT)
    assert revision["worktree_clean"] is True
    assert set(revision["serialized_sources"]) == set(
        launcher._serialized_source_paths(ROOT)
    )
    assert launcher.RUNTIME_CONTRACT_SOURCE in revision["serialized_sources"]
    assert launcher.SEMANTIC_CONTRACT_SOURCE in revision["serialized_sources"]
    assert (
        "src/compose_v4/data/editing_v2_semantic_active8_decision_mapreduce.py"
        in revision["serialized_sources"]
    )
    launcher._validate_remote_source_revision(
        revision, loaded=launcher._imports(ROOT), remote_root=ROOT
    )

    with pytest.raises(RuntimeError, match="exact clean committed worktree"):
        launcher.local_source_revision(expected_commit="c" * 40, repo_root=ROOT)

    def dirty_git(root, *arguments):
        if arguments == ("status", "--porcelain=v1", "--untracked-files=all"):
            return "?? untracked.py"
        return clean_git(root, *arguments)

    monkeypatch.setattr(launcher, "_git", dirty_git)
    with pytest.raises(RuntimeError, match="exact clean committed worktree"):
        launcher.local_source_revision(expected_commit="a" * 40, repo_root=ROOT)


def test_exact_live_model_identity_binds_semantic_capabilities_seed_and_state(
    monkeypatch,
) -> None:
    launcher = _load_launcher()
    loaded = launcher._imports(ROOT)
    runtime, semantic = launcher._load_runtime_contract(root=ROOT, loaded=loaded)
    revision_body = {
        "schema": launcher.SOURCE_REVISION_SCHEMA,
        "schema_version": launcher.SOURCE_REVISION_SCHEMA_VERSION,
        "commit": "a" * 40,
        "tree": "b" * 40,
        "worktree_clean": True,
        "serialized_sources": {
            relative: launcher._file_sha256(ROOT / relative)
            for relative in launcher._serialized_source_paths(ROOT)
        },
        "runtime_contract_sha256": runtime["runtime_contract_sha256"],
        "semantic_model_process_contract_sha256": semantic.sha256,
    }
    revision = {
        **revision_body,
        "source_revision_sha256": launcher._sha256(revision_body),
    }
    monkeypatch.setattr(launcher, "_software_identity", lambda value: value["software"])
    model_a, identity_a = launcher.build_exact_semantic_model(
        runtime=runtime,
        semantic=semantic,
        source_revision=revision,
        loaded=loaded,
    )
    model_b, identity_b = launcher.build_exact_semantic_model(
        runtime=runtime,
        semantic=semantic,
        source_revision=revision,
        loaded=loaded,
    )
    assert identity_a == identity_b
    assert identity_a["initialization_seed"] == 20260730
    assert len(identity_a["initial_model_state_sha256"]) == 64
    assert (
        identity_a["architecture"]["operator_capability_fingerprint"]
        == semantic.payload["model_identity"]["operator_capability_fingerprint"]
    )
    assert identity_a["architecture"]["dtype"] == "torch.float32"
    assert model_a is not model_b
