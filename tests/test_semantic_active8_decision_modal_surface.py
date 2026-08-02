"""Static and behavioral safety tests for the semantic Active8 Modal app."""

from __future__ import annotations

import ast
import importlib.util
import json
from pathlib import Path

import pytest

from compose_v4.data.semantic_active8_chunk_cache_mapreduce import (
    SemanticActive8ChunkCacheReductionWitness,
)

ROOT = Path(__file__).resolve().parents[1]
APP_PATH = ROOT / "modal_apps" / "run_semantic_active8_decisions_app.py"


def _function(tree: ast.Module, name: str) -> ast.FunctionDef:
    return next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name
    )


def _load_launcher():
    spec = importlib.util.spec_from_file_location("semantic_active8_decision_launcher", APP_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_modal_surface_is_cpu_only_restart_safe_and_non_authorizing() -> None:
    source = APP_PATH.read_text()
    tree = ast.parse(source)
    assert "gpu=" not in source
    assert "MODAL_VOLUME_V1_MAX_CONCURRENT_WRITERS = 5" in source
    assert "MAX_MAP_CONTAINERS = MODAL_VOLUME_V1_MAX_CONCURRENT_WRITERS" in source
    launcher = _load_launcher()
    assert launcher.MODAL_VOLUME_V1_MAX_CONCURRENT_WRITERS == 5
    assert launcher.MAX_MAP_CONTAINERS == 5
    worker_source = ast.get_source_segment(source, _function(tree, "decide_one_chunk"))
    assert worker_source is not None
    assert "_worker_model_and_checker" in worker_source
    assert "plan_artifact_path" in worker_source
    assert "_load_json_exact" in worker_source
    assert "model_runtime_identity_resolver" in worker_source
    assert "artifact_volume.reload()" in worker_source
    assert "artifact_volume.commit()" in worker_source
    worker_decorator = ast.get_source_segment(
        source, _function(tree, "decide_one_chunk").decorator_list[0]
    )
    assert worker_decorator is not None
    assert "max_containers=MAX_MAP_CONTAINERS" in worker_decorator
    cache_source = ast.get_source_segment(source, _function(tree, "_worker_model_and_checker"))
    assert cache_source is not None
    assert "ProductionSemanticExactCandidateChecker" in cache_source
    assert "build_exact_semantic_model" in cache_source

    driver_source = ast.get_source_segment(source, _function(tree, "driver"))
    assert driver_source is not None
    assert "semantic_migration_completion" in driver_source
    assert "semantic_chunk_cache_plan" in driver_source
    assert "semantic_chunk_cache_global_completion" in driver_source
    assert "chunk_cache_witness=cache_witness" in driver_source
    assert "completed_semantic_active8_decision_task_ids" in driver_source
    assert "write_semantic_active8_decision_plan" in driver_source
    assert "decide_one_chunk.starmap" in driver_source
    assert "reduce_decisions.remote" in driver_source

    reducer_source = ast.get_source_segment(source, _function(tree, "reduce_decisions"))
    assert reducer_source is not None
    assert reducer_source.index("artifact_volume.reload()") < reducer_source.index(
        "reduce_semantic_active8_decisions"
    )
    assert "os.link(" not in source
    assert ".link_to(" not in source

    main_source = ast.get_source_segment(source, _function(tree, "main"))
    assert main_source is not None
    assert "expected_commit" in main_source
    assert "driver.remote" in main_source
    assert "driver.spawn" not in main_source
    assert '"training_launched": False' in main_source
    assert '"gate_zero_authorized": False' in main_source
    assert '"t1_authorized": False' in main_source
    assert '"bounded_p50_authorized": False' in main_source


def test_chunk_inputs_reduce_once_and_preserve_exact_caller_pointer(
    tmp_path: Path,
) -> None:
    launcher = _load_launcher()
    cache_plan = {"schema": "fixture.cache-plan", "tasks": []}
    pointer = {
        "schema": "fixture.cache-pointer",
        "completion_sha256": "a" * 64,
    }
    witness = SemanticActive8ChunkCacheReductionWitness(
        reduction={**pointer, "completion": {"schema": "fixture.completion"}},
        validated_caches=(),
    )
    plan_path = tmp_path / "PLAN.json"
    pointer_path = tmp_path / "GLOBAL_COMPLETE.json"
    plan_path.write_text(
        json.dumps(cache_plan, sort_keys=True, separators=(",", ":")) + "\n"
    )
    pointer_path.write_text(
        json.dumps(pointer, sort_keys=True, separators=(",", ":")) + "\n"
    )
    calls = 0

    def reduce_once(observed_plan, *, artifact_root):
        nonlocal calls
        calls += 1
        assert observed_plan == cache_plan
        assert artifact_root == launcher.ARTIFACT_ROOT
        return witness

    loaded = {"reduce_semantic_active8_chunk_caches_with_witness": reduce_once}
    observed_plan, observed_witness = launcher._load_verified_chunk_inputs(
        plan_path=plan_path,
        global_completion_path=pointer_path,
        loaded=loaded,
    )
    assert observed_plan == cache_plan
    assert observed_witness is witness
    assert calls == 1

    pointer_path.write_text(
        json.dumps(
            {**pointer, "completion_sha256": "b" * 64},
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    )
    with pytest.raises(
        RuntimeError,
        match="caller chunk-cache GLOBAL_COMPLETE differs from strict reduction",
    ):
        launcher._load_verified_chunk_inputs(
            plan_path=plan_path,
            global_completion_path=pointer_path,
            loaded=loaded,
        )
    assert calls == 2


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
    assert runtime["semantic_model_process_contract"]["contract_sha256"] == semantic.sha256
    assert runtime["semantic_model_process_contract"]["file_sha256"] == launcher._file_sha256(
        ROOT / launcher.SEMANTIC_CONTRACT_SOURCE
    )


def test_image_copy_inventory_matches_the_hashed_source_inventory() -> None:
    launcher = _load_launcher()
    source = APP_PATH.read_text()
    tree = ast.parse(source)
    expected = {launcher.LAUNCHER_SOURCE, launcher.MIGRATION_LAUNCHER_SOURCE}
    for source_directory in launcher.IMAGE_SOURCE_DIRECTORIES:
        expected.update(
            path.relative_to(ROOT).as_posix()
            for path in (ROOT / source_directory).rglob("*")
            if path.is_file() and path.suffix != ".pyc" and "__pycache__" not in path.parts
        )
    assert launcher.IMAGE_SOURCE_DIRECTORIES == ("src", "configs")
    assert set(launcher._serialized_source_paths(ROOT)) == expected
    assert (ROOT / launcher.MIGRATION_LAUNCHER_SOURCE).is_file()
    mounted_files = {
        ast.get_source_segment(source, node.args[0])
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "add_local_file"
        and node.args
    }
    assert "ROOT / LAUNCHER_SOURCE" in mounted_files
    assert "ROOT / MIGRATION_LAUNCHER_SOURCE" in mounted_files


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
        if arguments == ("ls-files",):
            return "\n".join(launcher._serialized_source_paths(ROOT))
        raise AssertionError(arguments)

    monkeypatch.setattr(launcher, "_git", clean_git)
    revision = launcher.local_source_revision(expected_commit="a" * 40, repo_root=ROOT)
    assert revision["worktree_clean"] is True
    assert revision["execution_source_revision"]["commit"] == "a" * 40
    assert revision["execution_source_revision"]["tree"] == "b" * 40
    assert set(revision["serialized_sources"]) == set(launcher._serialized_source_paths(ROOT))
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

    def clean_git(_root, *arguments):
        return {
            ("rev-parse", "HEAD"): "a" * 40,
            ("rev-parse", "HEAD^{tree}"): "b" * 40,
            ("status", "--porcelain=v1", "--untracked-files=all"): "",
            ("ls-files",): "\n".join(launcher._serialized_source_paths(ROOT)),
        }[arguments]

    monkeypatch.setattr(launcher, "_git", clean_git)
    revision = launcher.local_source_revision(expected_commit="a" * 40, repo_root=ROOT)
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
    assert identity_a["producer_source_revision_sha256"] == revision["source_revision_sha256"]
    assert (
        identity_a["execution_source_revision_sha256"]
        == revision["execution_source_revision"]["source_revision_sha256"]
    )
    assert model_a is not model_b
