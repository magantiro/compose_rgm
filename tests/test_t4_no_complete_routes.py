from __future__ import annotations

from dataclasses import replace

import numpy as np

from compose_v4.control.adaptive_program_optimizer import ProgramSearchConfig
from compose_v4.control.dynamic_program_synthesis import (
    GENERIC_MODULES,
    DynamicProgramOptimizer,
    compile_generic_module,
    initial_dynamic_program_batch,
    synthesize_dynamic_program,
)
from compose_v4.control.edit_program_graph import (
    compile_program_graph,
    execute_program_graph,
)
from compose_v4.experiments.editing_v2_evaluation_semantics import (
    production_state_from_smiles,
)
from compose_v4.experiments.t4_no_complete_routes import (
    SOURCE_LIBRARY,
    is_complete_route,
    partition_library,
    recovered_improvement,
)


def test_remote_app_packages_source_library_for_partition_preflight():
    from modal_apps import t4_no_complete_routes_app

    assert t4_no_complete_routes_app.SOURCE_LIBRARY == SOURCE_LIBRARY


def _row(label, *, extra=False):
    blocks = [{"label": label, "stop": 1}]
    if extra:
        blocks.append({"label": "suffix", "stop": 2})
    return {"program": {"blocks": blocks}}


def test_partition_removes_only_single_complete_transformation_blocks():
    named = _row("pendant_benzene")
    complete = _row("compiled_complete_transformation")
    multi = _row("compiled_complete_transformation", extra=True)
    retained, removed = partition_library([named, complete, multi])

    assert retained == [named, multi]
    assert removed == [complete]
    assert is_complete_route(complete) is True
    assert is_complete_route(multi) is False


def test_recovered_improvement_is_lower_is_better_and_abstains_when_undefined():
    assert recovered_improvement(-5.8, -13.3, -11.8) == 0.8
    assert recovered_improvement(-9.4, -11.0, -11.0) == 1.0
    assert recovered_improvement(-8.0, -8.0, -8.1) is None
    assert recovered_improvement(-8.0, -10.0, None) is None


def test_generic_segment_module_is_parameterized_and_exactly_replayable():
    source = production_state_from_smiles("CCO", max_atoms=48)
    product, stage = compile_generic_module(
        source, np.random.default_rng(7), "segment_grow"
    )

    assert stage["name"] == "segment_grow"
    assert 1 <= stage["parameters"]["length"] <= 8
    assert stage["endpoint"] != "CCO"
    assert product.n_real_atoms == source.n_real_atoms + stage["parameters"]["length"]


def test_dynamic_program_uses_only_generic_modules_and_no_intermediate_oracle():
    source = production_state_from_smiles("CCOC(=O)NCC", max_atoms=48)
    _, program, binding, trace, metadata = synthesize_dynamic_program(
        source, np.random.default_rng(19), max_modules=3
    )
    _, replay = execute_program_graph(
        source,
        compile_program_graph(program),
        binding,
        max_primitives=32,
        max_blocks=8,
    )

    assert metadata["initial_stored_complete_routes"] == 0
    assert metadata["source_library_rows_loaded"] == 0
    assert metadata["intermediate_task_evaluations"] == 0
    assert 1 <= metadata["completed_module_count"] <= 3
    assert all(row["family"] in GENERIC_MODULES for row in metadata["modules"])
    assert replay == trace


def test_dynamic_cold_start_has_empty_route_archive_and_is_deterministic():
    source = production_state_from_smiles("CCOC(=O)NCC", max_atoms=48)
    config = replace(
        ProgramSearchConfig.program_only_recipe(seed=23),
        attempts_per_batch=4,
        candidates_per_batch=4,
        wall_seconds=20,
    )
    kwargs = {
        "source_group": "test-source",
        "oracle_protocol": "test-oracle",
        "eligibility": lambda row: {**row, "oracle_eligible": True},
    }
    first = initial_dynamic_program_batch(source, (), config, **kwargs)
    second = initial_dynamic_program_batch(source, (), config, **kwargs)

    assert first["batch_id"] == second["batch_id"]
    assert first["initial_route_archive"] == []
    assert first["source_library_rows_loaded"] == 0
    assert first["candidates"]


def test_dynamic_optimizer_restore_preserves_dynamic_class():
    optimizer = DynamicProgramOptimizer(
        ProgramSearchConfig.program_only_recipe(seed=31),
        source_group="test-source",
        oracle_protocol="test-oracle",
        hierarchy=None,
    )

    restored = DynamicProgramOptimizer.restore(optimizer.snapshot(), hierarchy=None)

    assert isinstance(restored, DynamicProgramOptimizer)
