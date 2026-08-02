"""Focused contracts for the semantic P50 validation-baseline producer."""

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from modal_apps import (
    materialize_editing_v2_semantic_p50_validation_baseline_app as baseline_app,
)

SHA = "a" * 64


def test_image_copy_inventory_matches_the_hashed_source_inventory() -> None:
    root = Path(baseline_app.__file__).resolve().parents[1]
    expected = {baseline_app.LAUNCHER_SOURCE}
    for source_directory in baseline_app.IMAGE_SOURCE_DIRECTORIES:
        expected.update(
            path.relative_to(root).as_posix()
            for path in (root / source_directory).rglob("*")
            if path.is_file() and path.suffix != ".pyc" and "__pycache__" not in path.parts
        )
    assert baseline_app.IMAGE_SOURCE_DIRECTORIES == ("src", "configs")
    assert set(baseline_app._serialized_source_paths(root)) == expected


def _source_revision() -> dict[str, object]:
    hashes = {"source.py": SHA}
    body: dict[str, object] = {
        "schema": baseline_app.SOURCE_REVISION_SCHEMA,
        "schema_version": baseline_app.SCHEMA_VERSION,
        "commit": "1" * 40,
        "tree": "2" * 40,
        "worktree_clean": True,
        "serialized_source_hashes": hashes,
        "serialized_source_hashes_sha256": baseline_app._sha(hashes),
    }
    return {**body, "source_revision_sha256": baseline_app._sha(body)}


def _input_records() -> dict[str, dict[str, str]]:
    return {
        name: {
            "artifact_path": f"/artifacts/validation-baseline/{name}.json",
            "file_sha256": format(index + 1, "064x"),
        }
        for index, name in enumerate(baseline_app._INPUT_NAMES)
    }


def _request(**overrides: object) -> dict[str, object]:
    arguments: dict[str, object] = {
        "source_revision": _source_revision(),
        "input_records": _input_records(),
        "prepared_recipe_sha256": "b" * 64,
        "source_inventory_sha256": "c" * 64,
        "successor_cache_completion_sha256": "d" * 64,
        "scratch_initial_model_state_sha256": "e" * 64,
    }
    arguments.update(overrides)
    return baseline_app.build_run_request(**arguments)  # type: ignore[arg-type]


def _function_sources() -> tuple[str, dict[str, str]]:
    source = Path(baseline_app.__file__).read_text()
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
    assert first["run_identity_sha256"] == baseline_app._sha(
        {key: value for key, value in first.items() if key != "run_identity_sha256"}
    )
    assert all(first[field] is False for field in baseline_app._NO_AUTHORITY)
    assert list(first["inputs"]) == list(baseline_app._INPUT_NAMES)  # type: ignore[arg-type]
    assert first["evaluation"] == {  # type: ignore[comparison-overlap]
        "accelerator_class": "gpu",
        "modal_gpu_type": "A10G",
        "dtype": "torch.float32",
        "mixed_precision": False,
        "batch_size": 64,
        "deterministic_algorithms_required": True,
        "cudnn_benchmark": False,
        "cudnn_deterministic": True,
        "cudnn_tf32_allowed": False,
        "cuda_matmul_tf32_allowed": False,
        "cublas_workspace_config": ":4096:8",
        "nvidia_tf32_override": "0",
        "objective": "productive_embedded_canonical_successor_nll",
        "model_state": "untouched_scratch_initial_state",
    }

    changed = _input_records()
    changed["successor_cache_completion"]["file_sha256"] = "0" * 64
    assert _request(input_records=changed)["run_identity_sha256"] != first["run_identity_sha256"]
    assert (
        _request(scratch_initial_model_state_sha256="0" * 64)["run_identity_sha256"]
        != first["run_identity_sha256"]
    )


def test_request_rejects_missing_input_and_escaping_path() -> None:
    inputs = _input_records()
    inputs.pop("source_inventory")
    with pytest.raises(ValueError, match="input records disagree"):
        _request(input_records=inputs)


def test_physical_input_snapshot_rejects_one_mutated_input(tmp_path: Path) -> None:
    artifact_root = tmp_path / "artifacts"
    input_root = artifact_root / "inputs"
    input_root.mkdir(parents=True)
    addresses: dict[str, str] = {}
    for index, name in enumerate(baseline_app._INPUT_NAMES):
        path = input_root / f"{name}.json"
        path.write_bytes(f"input-{index}".encode())
        addresses[name] = f"/artifacts/inputs/{name}.json"

    snapshot = baseline_app._snapshot_physical_inputs(
        addresses,
        artifact_root=artifact_root,
    )
    baseline_app._require_unchanged_physical_inputs(
        snapshot,
        input_addresses=addresses,
        artifact_root=artifact_root,
    )

    (input_root / "t1_decision.json").write_bytes(b"mutated")
    with pytest.raises(RuntimeError, match="changed after their initial snapshot"):
        baseline_app._require_unchanged_physical_inputs(
            snapshot,
            input_addresses=addresses,
            artifact_root=artifact_root,
        )

    inputs = _input_records()
    inputs["source_inventory"]["artifact_path"] = "/artifacts/../escape.json"
    with pytest.raises(ValueError, match="normalized path below /artifacts"):
        _request(input_records=inputs)


def test_source_revision_is_local_git_plus_remote_byte_attestation() -> None:
    with (
        patch.object(
            baseline_app,
            "_git",
            side_effect=["1" * 40, "2" * 40, "", "source.py"],
        ),
        patch.object(
            baseline_app,
            "_serialized_source_hashes",
            return_value={"source.py": SHA},
        ),
    ):
        revision = baseline_app.local_source_revision(
            expected_commit="1" * 40, repo_root=Path("/fixture")
        )
    assert revision == _source_revision()

    with (
        patch.object(
            baseline_app,
            "_git",
            side_effect=["1" * 40, "2" * 40, "?? dirty.py"],
        ),
        pytest.raises(RuntimeError, match="clean committed worktree"),
    ):
        baseline_app.local_source_revision(expected_commit="1" * 40, repo_root=Path("/fixture"))

    with patch.object(
        baseline_app,
        "_serialized_source_hashes",
        return_value={"source.py": SHA},
    ):
        assert (
            baseline_app._validate_source_revision(_source_revision(), remote_root=Path("/remote"))
            == _source_revision()
        )

    _, functions = _function_sources()
    assert "_git(" not in functions["_validate_source_revision"]
    assert "_serialized_source_hashes(remote_root)" in functions["_validate_source_revision"]


def test_source_revision_rejects_ignored_serialized_file() -> None:
    with (
        patch.object(
            baseline_app,
            "_git",
            side_effect=["1" * 40, "2" * 40, "", "tracked.py"],
        ),
        patch.object(
            baseline_app,
            "_serialized_source_hashes",
            return_value={"ignored.py": SHA},
        ),
        pytest.raises(RuntimeError, match="serialized source/config must be Git-tracked"),
    ):
        baseline_app.local_source_revision(
            expected_commit="1" * 40,
            repo_root=Path("/fixture"),
        )


def test_exact_reopen_order_includes_recipe_source_scratch_and_cache() -> None:
    _, functions = _function_sources()
    opener = functions["_open_exact_prerequisites"]
    fragments = (
        "load_semantic_p50_prepared_recipe",
        'loaded["load_source"]',
        'loaded["validate_gate_zero"]',
        'loaded["validate_t1"]',
        'loaded["validate_prerequisites"]',
        "_scratch_runtime(",
        "expected_semantic_p50_successor_cache_identity_from_reopened_prerequisites",
        "open_semantic_p50_successor_cache",
    )
    offsets = [opener.index(fragment) for fragment in fragments]
    assert offsets == sorted(offsets)
    assert "require_p50_go=True" in opener
    assert "scratch_initial_model_state_sha256" in opener


def test_cpu_and_gpu_stages_reopen_every_published_scientific_artifact() -> None:
    _, functions = _function_sources()
    cpu = functions["_prepare_inventory_impl"]
    gpu = functions["_evaluate_baseline_impl"]

    assert cpu.index("_open_exact_prerequisites") < cpu.index(
        "materialize_semantic_p50_validation_inventory"
    )
    assert cpu.index("materialize_semantic_p50_validation_inventory") < cpu.index(
        "open_semantic_p50_validation_inventory"
    )

    ordered_gpu_fragments = (
        "_open_exact_prerequisites",
        "open_semantic_p50_validation_inventory",
        "_validation_states",
        "_evaluate_scratch",
        "build_semantic_p50_validation_evaluation_environment_receipt",
        "open_semantic_p50_validation_evaluation_environment_receipt",
        "materialize_semantic_p50_validation_baseline",
        "open_semantic_p50_validation_baseline",
    )
    offsets = [gpu.index(fragment) for fragment in ordered_gpu_fragments]
    assert offsets == sorted(offsets)
    assert '"training_launched": False' in gpu
    assert '"training_authorized": False' in gpu
    assert '"bounded_p50_authorized": False' in gpu


def test_cpu_and_gpu_drivers_rehash_the_initial_input_snapshot() -> None:
    _, functions = _function_sources()
    request_builder = functions["_request_for_opened"]
    cpu = functions["_prepare_inventory_impl"]
    gpu = functions["_evaluate_baseline_impl"]

    assert "input_records=input_snapshot" in request_builder
    assert "_file_sha(" not in request_builder

    cpu_snapshot = cpu.index("_snapshot_physical_inputs")
    cpu_open = cpu.index("_open_exact_prerequisites")
    cpu_request = cpu.index("_request_for_opened")
    cpu_inventory_open = cpu.index("open_semantic_p50_validation_inventory")
    cpu_checks = [
        index
        for index in range(len(cpu))
        if cpu.startswith("_require_unchanged_physical_inputs", index)
    ]
    assert cpu_snapshot < cpu_open < cpu_checks[0] < cpu_request
    assert cpu_inventory_open < cpu_checks[-1]
    assert len(cpu_checks) == 2

    gpu_snapshot = gpu.index("_snapshot_physical_inputs")
    gpu_open = gpu.index("_open_exact_prerequisites")
    gpu_request = gpu.index("_request_for_opened")
    gpu_evaluate = gpu.index("_evaluate_scratch")
    gpu_baseline_open = gpu.index("open_semantic_p50_validation_baseline")
    gpu_completion_open = gpu.index("producer completion differs after immutable publication")
    gpu_checks = [
        index
        for index in range(len(gpu))
        if gpu.startswith("_require_unchanged_physical_inputs", index)
    ]
    assert gpu_snapshot < gpu_open < gpu_checks[0] < gpu_request
    assert gpu_checks[1] < gpu_evaluate
    assert gpu_baseline_open < gpu_checks[2]
    assert gpu_completion_open < gpu_checks[3]
    assert len(gpu_checks) == 4


def test_evaluator_uses_frozen_times_cached_fibers_and_production_forward_only() -> None:
    source, functions = _function_sources()
    evaluator = functions["_evaluate_scratch"]
    assert "float.fromhex(candidate.time_hex)" in evaluator
    assert "record.teacher_fiber" in evaluator
    assert 'loaded["forward_teacher_successor_batch"]' in evaluator
    assert "selected_productive_successor_log_probability" in evaluator
    assert 'loaded["state_dict_semantic_sha256"]' in evaluator
    assert "torch.no_grad()" in evaluator
    assert "torch.use_deterministic_algorithms(True)" in evaluator
    assert "torch.optim" not in source
    assert ".backward(" not in source
    assert "editing_v2_semantic_p50_runner" not in source
    assert "authorize_semantic_p50_recipe" not in source


def test_validation_state_recovery_rejects_missing_and_duplicate_addresses() -> None:
    address = "address"
    candidate = SimpleNamespace(address=address, source_state_sha256="state-sha")
    inventory = SimpleNamespace(candidates=(candidate,))
    state = object()
    packed = object()
    transition = SimpleNamespace(
        step_index=0,
        addressed_trace=SimpleNamespace(
            address=packed,
            path=SimpleNamespace(state_at=lambda index: state),
        ),
    )

    class AddressClass:
        @staticmethod
        def from_packed_trace(_packed, *, progress_index):
            assert progress_index == 0
            return address

    class Index:
        def __init__(self, repeats: int):
            self.repeats = repeats

        def iter_accepted_traces_for_partition(self, partition):
            assert partition == "validation"
            return (SimpleNamespace(),)

        def accepted_transitions_for(self, _trace):
            return (transition,) * self.repeats

    loaded = {
        "SuccessorFiberCacheAddress": AddressClass,
        "persistent_slot_state_sha256": lambda value: "state-sha" if value is state else "wrong",
    }
    with patch.object(baseline_app, "_address_sha", return_value="address-sha"):
        recovered = baseline_app._validation_states(
            inventory,
            source=SimpleNamespace(index=Index(1)),
            loaded=loaded,
        )
    assert recovered == {"address-sha": state}

    with (
        patch.object(baseline_app, "_address_sha", return_value="address-sha"),
        pytest.raises(RuntimeError, match="repeats an inventory address"),
    ):
        baseline_app._validation_states(
            inventory,
            source=SimpleNamespace(index=Index(2)),
            loaded=loaded,
        )

    with pytest.raises(RuntimeError, match="omits an inventory state"):
        baseline_app._validation_states(
            inventory,
            source=SimpleNamespace(index=Index(0)),
            loaded=loaded,
        )


def test_modal_surface_has_bounded_cpu_and_gpu_roles_without_mapping() -> None:
    source, _ = _function_sources()
    tree = ast.parse(source)
    nodes = {
        node.name: node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name in {"prepare_validation_inventory", "evaluate_validation_baseline"}
    }
    cpu_decorator = ast.get_source_segment(
        source, nodes["prepare_validation_inventory"].decorator_list[0]
    )
    gpu_decorator = ast.get_source_segment(
        source, nodes["evaluate_validation_baseline"].decorator_list[0]
    )
    assert cpu_decorator is not None and gpu_decorator is not None
    assert "gpu=" not in cpu_decorator
    assert "cpu=8.0" in cpu_decorator
    assert "max_containers=1" in cpu_decorator
    assert "gpu=GPU_TYPE" in gpu_decorator
    assert "cpu=8.0" in gpu_decorator
    assert "max_containers=1" in gpu_decorator
    assert ".starmap(" not in source


def test_baseline_gpu_and_numerics_match_the_p50_execution_surface() -> None:
    source, functions = _function_sources()
    evaluator = functions["_evaluate_scratch"]
    assert baseline_app.GPU_TYPE == "A10G"
    assert 'GPU_TYPE = "L4"' not in source
    assert '"NVIDIA_TF32_OVERRIDE": "0"' in source
    assert '"CUBLAS_WORKSPACE_CONFIG": ":4096:8"' in source
    assert "torch.use_deterministic_algorithms(True)" in evaluator
    assert "torch.backends.cudnn.benchmark = False" in evaluator
    assert "torch.backends.cudnn.deterministic = True" in evaluator
    assert "torch.backends.cudnn.allow_tf32 = False" in evaluator
    assert "torch.backends.cuda.matmul.allow_tf32 = False" in evaluator


def test_serialized_revision_covers_launcher_evaluator_sources_and_configs() -> None:
    paths = baseline_app._serialized_source_paths(baseline_app.ROOT)
    assert baseline_app.LAUNCHER_SOURCE in paths
    assert "src/compose_v4/experiments/editing_v2_semantic_p50_validation_baseline.py" in paths
    assert "src/compose_v4/experiments/factorized_successor_training.py" in paths
    assert baseline_app.SEMANTIC_MODEL_PROCESS_SOURCE in paths
    assert "configs/editing_v2_semantic_p50_recipe_policy_v1.json" in paths
    assert "configs/experiment_registry.yaml" in paths
    assert "configs/lipid_reactions/README.md" in paths
    assert len(paths) == len(set(paths))
