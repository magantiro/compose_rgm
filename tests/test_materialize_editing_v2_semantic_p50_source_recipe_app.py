"""Focused contracts for the semantic P50 source/recipe Modal producer."""

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from modal_apps import (
    materialize_editing_v2_semantic_p50_source_recipe_app as source_recipe_app,
)

SHA = "a" * 64


def test_image_copy_inventory_matches_the_hashed_source_inventory() -> None:
    root = Path(source_recipe_app.__file__).resolve().parents[1]
    expected = {source_recipe_app.LAUNCHER_SOURCE}
    for source_directory in source_recipe_app.IMAGE_SOURCE_DIRECTORIES:
        expected.update(
            path.relative_to(root).as_posix()
            for path in (root / source_directory).rglob("*")
            if path.is_file() and path.suffix != ".pyc" and "__pycache__" not in path.parts
        )
    assert source_recipe_app.IMAGE_SOURCE_DIRECTORIES == ("src", "configs")
    assert set(source_recipe_app._serialized_source_paths(root)) == expected


def _source_revision() -> dict[str, object]:
    hashes = {"source.py": SHA}
    body: dict[str, object] = {
        "schema": source_recipe_app.SOURCE_REVISION_SCHEMA,
        "schema_version": source_recipe_app.SCHEMA_VERSION,
        "commit": "1" * 40,
        "tree": "2" * 40,
        "worktree_clean": True,
        "serialized_source_hashes": hashes,
        "serialized_source_hashes_sha256": source_recipe_app._sha(hashes),
    }
    return {**body, "source_revision_sha256": source_recipe_app._sha(body)}


def _inputs() -> dict[str, dict[str, str]]:
    return {
        name: {
            "artifact_path": f"/artifacts/semantic-p50/{name}.json",
            "file_sha256": format(index + 1, "064x"),
        }
        for index, name in enumerate(source_recipe_app._INPUT_NAMES)
    }


def _source_identity() -> dict[str, str]:
    return {
        "inventory_sha256": "1" * 64,
        "process_identity_sha256": "2" * 64,
        "model_runtime_identity_sha256": "3" * 64,
        "policy_sha256": "4" * 64,
        "decision_source_implementation_sha256": "5" * 64,
    }


def _gate_zero() -> dict[str, dict[str, object]]:
    return {
        "evidence": {"evidence_sha256": "6" * 64, "structural_result": "PASS"},
        "decision": {"decision_sha256": "7" * 64},
        "completion": {"completion_sha256": "8" * 64},
    }


def _t1() -> dict[str, str]:
    return {
        "decision_sha256": "9" * 64,
        "status": "GO_BOUNDED_P50_SEMANTIC_T1_CAPACITY",
        "initial_model_state_sha256": "b" * 64,
        "selected_model_state_sha256": "c" * 64,
    }


def _policy() -> dict[str, str]:
    return {"policy_sha256": "d" * 64}


def _request(**overrides: object) -> dict[str, object]:
    arguments: dict[str, object] = {
        "source_revision": _source_revision(),
        "input_records": _inputs(),
        "source_identity": _source_identity(),
        "gate_zero_artifacts": _gate_zero(),
        "t1_decision": _t1(),
        "recipe_policy": _policy(),
        "recipe_policy_file_sha256": "e" * 64,
        "capability_registry_sha256": "f" * 64,
    }
    arguments.update(overrides)
    return source_recipe_app.build_run_request(**arguments)  # type: ignore[arg-type]


def _function_sources() -> tuple[str, dict[str, str]]:
    source = Path(source_recipe_app.__file__).read_text()
    tree = ast.parse(source)
    functions = {
        node.name: ast.get_source_segment(source, node)
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    assert all(value is not None for value in functions.values())
    return source, functions  # type: ignore[return-value]


def test_request_is_deterministic_content_addressed_and_nonauthorizing() -> None:
    first = _request()
    second = _request()
    assert first == second
    assert first["run_identity_sha256"] == source_recipe_app._sha(
        {key: value for key, value in first.items() if key != "run_identity_sha256"}
    )
    assert all(first[field] is False for field in source_recipe_app._NO_AUTHORITY)
    assert list(first["inputs"]) == list(source_recipe_app._INPUT_NAMES)  # type: ignore[arg-type]
    assert first["runtime"]["device"] == "cpu"  # type: ignore[index]
    assert first["runtime"]["torch_version"]  # type: ignore[index]
    assert first["runtime"]["rdkit_version"]  # type: ignore[index]

    changed_inputs = _inputs()
    changed_inputs["t1_decision"]["file_sha256"] = "0" * 64
    assert (
        _request(input_records=changed_inputs)["run_identity_sha256"]
        != first["run_identity_sha256"]
    )
    assert (
        _request(recipe_policy_file_sha256="0" * 64)["run_identity_sha256"]
        != first["run_identity_sha256"]
    )


def test_request_rejects_missing_input_and_escaping_path() -> None:
    inputs = _inputs()
    inputs.pop("t1_decision")
    with pytest.raises(ValueError, match="input records disagree"):
        _request(input_records=inputs)

    inputs = _inputs()
    inputs["t1_decision"]["artifact_path"] = "/artifacts/../escape.json"
    with pytest.raises(ValueError, match="normalized path below /artifacts"):
        _request(input_records=inputs)


def test_physical_input_snapshot_rejects_midscan_mutation(tmp_path: Path) -> None:
    supplied: dict[str, str] = {}
    mounted: dict[str, Path] = {}
    for index, name in enumerate(source_recipe_app._INPUT_NAMES):
        path = tmp_path / f"{name}.json"
        path.write_text(f'{{"index":{index}}}\n', encoding="utf-8")
        supplied[name] = f"/artifacts/semantic-p50/{name}.json"
        mounted[name] = path
    snapshot = source_recipe_app._input_records(supplied, mounted)
    source_recipe_app._require_input_records_unchanged(mounted, snapshot)

    mounted["t1_decision"].write_text('{"changed":true}\n', encoding="utf-8")
    with pytest.raises(RuntimeError, match="changed during compilation"):
        source_recipe_app._require_input_records_unchanged(mounted, snapshot)


def test_canonical_loader_rejects_same_bytes_symlink(tmp_path: Path) -> None:
    target = tmp_path / "target.json"
    target.write_text('{"value":1}\n', encoding="utf-8")
    link = tmp_path / "link.json"
    link.symlink_to(target)
    with pytest.raises(RuntimeError, match="is absent"):
        source_recipe_app._load_canonical(link, field="symlinked artifact")


def test_local_source_revision_requires_exact_clean_commit() -> None:
    with (
        patch.object(
            source_recipe_app,
            "_git",
            side_effect=["1" * 40, "2" * 40, "", "source.py"],
        ),
        patch.object(
            source_recipe_app,
            "_serialized_source_hashes",
            return_value={"source.py": SHA},
        ),
    ):
        revision = source_recipe_app.local_source_revision(
            expected_commit="1" * 40, repo_root=Path("/fixture")
        )
    assert revision["commit"] == "1" * 40

    with (
        patch.object(
            source_recipe_app,
            "_git",
            side_effect=["1" * 40, "2" * 40, "?? dirty.py"],
        ),
        pytest.raises(RuntimeError, match="clean committed worktree"),
    ):
        source_recipe_app.local_source_revision(
            expected_commit="1" * 40, repo_root=Path("/fixture")
        )

    with (
        patch.object(
            source_recipe_app,
            "_git",
            side_effect=["1" * 40, "2" * 40, "", "another.py"],
        ),
        patch.object(
            source_recipe_app,
            "_serialized_source_hashes",
            return_value={"source.py": SHA},
        ),
        pytest.raises(RuntimeError, match="must be Git-tracked"),
    ):
        source_recipe_app.local_source_revision(
            expected_commit="1" * 40, repo_root=Path("/fixture")
        )


def test_prepared_recipe_must_keep_all_five_bindings_unresolved() -> None:
    purposes = (
        "stream_union_successor_cache",
        "validation_baseline",
        "trainer_runtime",
        "execution_environment",
        "launch_projection",
    )
    recipe_stream = SimpleNamespace(
        PREPARED_SCHEMA="prepared",
        PREPARED_STATUS="pending",
        REQUIRED_BINDING_PURPOSES=purposes,
        SCHEDULED_EXAMPLES=3200,
        OPTIMIZER_STEPS=50,
        BATCH_SIZE=64,
    )
    prerequisites = {"source_inventory_sha256": "1" * 64}
    prepared = {
        "schema": "prepared",
        "status": "pending",
        **source_recipe_app._NO_AUTHORITY,
        "prerequisites": prerequisites,
        "required_physical_binding_purposes": list(purposes),
        "unresolved_physical_bindings": list(purposes),
        "ordered_stream_rows": [{} for _ in range(3200)],
        "recipe": {"optimizer_steps": 50, "batch_size": 64},
        "stream_contract": {"scheduled_nonterminal_examples": 3200},
    }
    source_recipe_app._validate_prepared_recipe(
        prepared,
        recipe_stream=recipe_stream,
        expected_prerequisites=prerequisites,
    )

    with pytest.raises(RuntimeError, match="crosses its frozen boundary"):
        source_recipe_app._validate_prepared_recipe(
            {**prepared, "unresolved_physical_bindings": list(purposes[:-1])},
            recipe_stream=recipe_stream,
            expected_prerequisites=prerequisites,
        )
    with pytest.raises(RuntimeError, match="crosses its frozen boundary"):
        source_recipe_app._validate_prepared_recipe(
            {**prepared, "training_authorized": True},
            recipe_stream=recipe_stream,
            expected_prerequisites=prerequisites,
        )


def test_driver_strictly_reopens_in_order_before_compilation() -> None:
    source, functions = _function_sources()
    driver = functions["_driver_impl"]
    required_in_order = (
        'loaded["resolve_source"]',
        "load_semantic_gate_zero_structural_artifacts",
        "validate_semantic_t1_capacity_decision",
        "publish_semantic_p50_source_inventory",
        "load_semantic_p50_source_inventory",
        "build_semantic_p50_candidate_inventory",
        "compile_semantic_p50_prepared_recipe",
        "write_semantic_p50_prepared_recipe",
    )
    offsets = [driver.index(fragment) for fragment in required_in_order]
    assert offsets == sorted(offsets)
    assert driver.index("if completion_path.exists()") < driver.index(
        "build_semantic_p50_candidate_inventory"
    )
    assert driver.count("_require_input_records_unchanged") >= 4
    assert "candidates.prerequisites.as_payload() != expected_prerequisites" in driver
    assert "require_p50_go=True" in driver
    assert 'structural_result") != "PASS"' in driver
    assert "authorize_semantic_p50_recipe" not in source


def test_modal_surface_is_single_container_cpu_only_and_has_no_training_path() -> None:
    source, functions = _function_sources()
    tree = ast.parse(source)
    remote = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "materialize_source_recipe"
    )
    decorator = ast.get_source_segment(source, remote.decorator_list[0])
    assert decorator is not None
    assert "gpu=" not in decorator
    assert "cpu=8.0" in decorator
    assert "max_containers=1" in decorator
    assert ".starmap(" not in source
    assert "torch.optim" not in source
    assert "cuda" not in source
    assert '"training_launched": False' in functions["_driver_result"]
    assert '"training_authorized": False' in source
    assert '"bounded_p50_authorized": False' in source


def test_serialized_revision_covers_launcher_core_sources_and_configs() -> None:
    paths = source_recipe_app._serialized_source_paths(source_recipe_app.ROOT)
    assert source_recipe_app.LAUNCHER_SOURCE in paths
    assert "src/compose_v4/experiments/editing_v2_semantic_p50_recipe_stream.py" in paths
    assert "src/compose_v4/experiments/editing_v2_semantic_p50_source_inventory.py" in paths
    assert "configs/editing_v2_semantic_p50_recipe_policy_v1.json" in paths
    assert len(paths) == len(set(paths))
