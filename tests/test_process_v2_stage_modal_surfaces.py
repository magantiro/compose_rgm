"""The two unlaunched Process-V2 Modal surfaces, held to what they must refuse.

Neither app has ever been executed and neither is authorized to be.  What is
testable locally is the surface: container bounds, restart behaviour, exact
inputs, the refusals, and the property that importing either module launches
nothing.  Those are exactly the properties whose absence is invisible until a
remote job is already running, which is why they are pinned here.

Every assertion drives the production object.  The container bound is read off
the real decorator, the wave geometry comes from the production
``run_bounded_map``, the dirty-tree refusal runs the real ``local_image_revision``
with Git stubbed, and the restart behaviour runs the real driver body.
"""

from __future__ import annotations

import ast
import json
import subprocess
import sys
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from compose_v4.data.editing_v2_process_v2_chunk_cache import (
    PLATFORM_MAX_MAP_CONTAINERS,
    ProcessV2ConcurrencyError,
    plan_submission_waves,
    validate_map_container_bound,
)
from compose_v4.data.editing_v2_process_v2_schema import (
    AUTHORITY_FIELDS,
    require_authority_false,
    require_no_granted_authority,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
for _extra in (str(REPO_ROOT), str(REPO_ROOT / "tests")):
    if _extra not in sys.path:
        sys.path.insert(0, _extra)

import test_process_v2_gate_zero as gate_zero_fixture  # noqa: E402

from modal_apps import run_process_v2_active8_decisions_app as active8_app  # noqa: E402
from modal_apps import run_process_v2_gate_zero_app as gate_zero_app  # noqa: E402

APPS = (gate_zero_app, active8_app)
APP_PATHS = tuple(Path(module.__file__) for module in APPS)
_LAUNCH_CALLS = ("remote", "spawn", "starmap", "map", "update_autoscaler")


def _function_source(path: Path, name: str) -> str:
    source = path.read_text()
    tree = ast.parse(source, filename=str(path))
    node = next(
        item
        for item in tree.body
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and item.name == name
    )
    segment = ast.get_source_segment(source, node)
    assert segment is not None
    return segment


def _decorator_source(path: Path, name: str) -> str:
    source = path.read_text()
    tree = ast.parse(source, filename=str(path))
    node = next(
        item
        for item in tree.body
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and item.name == name
    )
    decorators = [ast.get_source_segment(source, item) for item in node.decorator_list]
    return "\n".join(text for text in decorators if text)


# ---- Nothing launches on import ----


@pytest.mark.parametrize("path", APP_PATHS, ids=lambda path: path.name)
def test_no_launch_call_appears_at_module_scope(path: Path) -> None:
    """A launch written at module scope would fire the moment anyone imports it."""

    source = path.read_text()
    tree = ast.parse(source, filename=str(path))
    inside_functions = {
        id(call)
        for parent in ast.walk(tree)
        if isinstance(parent, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda))
        for call in ast.walk(parent)
        if isinstance(call, ast.Call)
    }
    module_scope_launches = [
        ast.get_source_segment(source, call)
        for call in ast.walk(tree)
        if isinstance(call, ast.Call)
        and isinstance(call.func, ast.Attribute)
        and call.func.attr in _LAUNCH_CALLS
        and id(call) not in inside_functions
    ]
    assert not module_scope_launches, module_scope_launches
    # And the calls that do exist are the ones we expect to exist, so this test
    # cannot pass by the launcher having become inert.
    assert any(f".{name}(" in source for name in _LAUNCH_CALLS)


@pytest.mark.parametrize("path", APP_PATHS, ids=lambda path: path.name)
def test_importing_the_app_in_a_clean_process_launches_nothing(path: Path) -> None:
    """Imported for real, with every Modal launch surface booby-trapped.

    A subprocess so the trap cannot be defeated by this session having already
    imported the module.
    """

    program = f"""
import sys
sys.path.insert(0, {str(REPO_ROOT)!r})
sys.path.insert(0, {str(REPO_ROOT / "src")!r})
import modal

fired = []


def _trap(name):
    def _raise(*args, **kwargs):
        fired.append(name)
        raise AssertionError("a Modal launch fired during import: " + name)

    return _raise


installed = []
for _name in {list(_LAUNCH_CALLS)!r}:
    for _target in (modal.Function, modal.App):
        if hasattr(_target, _name):
            try:
                setattr(_target, _name, _trap(_name))
            except (AttributeError, TypeError):
                continue
            if getattr(_target, _name).__name__ == "_raise":
                installed.append(_target.__name__ + "." + _name)

# Without this the trap could silently fail to install and the import below
# would prove nothing at all.
assert installed, "no Modal launch surface could be trapped"

from modal_apps import {path.stem}  # noqa: F401

print("IMPORTED_WITHOUT_LAUNCH", len(fired), len(installed))
"""
    result = subprocess.run(
        [sys.executable, "-c", program], capture_output=True, text=True, timeout=300
    )
    assert result.returncode == 0, result.stderr
    fired, installed = (
        result.stdout.split("IMPORTED_WITHOUT_LAUNCH ", 1)[1].split()[:2]
    )
    assert fired == "0"
    assert int(installed) >= len(_LAUNCH_CALLS)


def test_the_description_entrypoints_execute_nothing(capsys) -> None:
    """``describe`` prints the exact command and roots and never claims a run."""

    for module in APPS:
        source = _function_source(Path(module.__file__), "describe")
        assert not [name for name in _LAUNCH_CALLS if f".{name}(" in source]
        assert "_git(" not in source

    gate_zero_app.describe.info.raw_f()
    payload = json.loads(capsys.readouterr().out)
    assert payload["executed"] is False
    assert payload["status"] == "DESCRIPTION_ONLY_NOTHING_WAS_EXECUTED"
    assert payload["would_run_command"].startswith("modal run --detach")
    assert "run_process_v2_gate_zero_app.py::main" in payload["would_run_command"]
    assert payload["would_write_artifact_root"].endswith("/<run_identity_sha256>")
    assert payload["execution_shape"] == "one_bounded_streaming_job_not_a_map_reduce"
    require_authority_false(payload, label="the Gate-0 launch description")
    assert all(payload[field] is False for field in AUTHORITY_FIELDS)

    active8_app.describe.info.raw_f()
    payload = json.loads(capsys.readouterr().out)
    assert payload["executed"] is False
    assert payload["max_map_containers"] == PLATFORM_MAX_MAP_CONTAINERS
    assert set(payload["would_bind_stage_modules"]) == set(active8_app.ACTIVE8_STAGE_SYMBOLS)
    require_authority_false(payload, label="the Active8 launch description")


# ---- Container bounds ----


def test_gate_zero_is_one_bounded_streaming_job_not_a_fan_out() -> None:
    path = Path(gate_zero_app.__file__)
    decorator = _decorator_source(path, "run_gate_zero")
    assert "max_containers=GATE_ZERO_MAX_CONTAINERS" in decorator
    assert gate_zero_app.GATE_ZERO_MAX_CONTAINERS == 1
    assert "gpu=" not in decorator
    source = path.read_text()
    tree = ast.parse(source, filename=str(path))
    # No map/reduce machinery: no wave planner, no fan-out submission, no reducer.
    called = {
        node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, (ast.Attribute, ast.Name))
    }
    assert not called & {"plan_submission_waves", "run_bounded_map", "starmap", "map"}
    remote_functions = {
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and any(
            "app.function" in (ast.get_source_segment(source, decorator) or "")
            for decorator in node.decorator_list
        )
    }
    assert remote_functions == {"run_gate_zero"}
    assert not [name for name in called if name.startswith("reduce")]


def test_active8_map_is_capped_at_forty_in_the_decorator_and_at_submission() -> None:
    """Both mechanisms, because the decorator alone left the bound cosmetic."""

    path = Path(active8_app.__file__)
    assert PLATFORM_MAX_MAP_CONTAINERS == 40
    assert active8_app.DEFAULT_MAX_MAP_CONTAINERS == PLATFORM_MAX_MAP_CONTAINERS
    assert "max_containers=PLATFORM_MAX_MAP_CONTAINERS" in _decorator_source(
        path, "decide_one_chunk"
    )
    driver = _function_source(path, "driver")
    assert "validate_map_container_bound(max_map_containers)" in driver
    assert "run_bounded_map(" in driver
    assert "set_autoscaler=decide_one_chunk.update_autoscaler" in driver
    # The reducer and the driver are serialized: only one may write completion.
    for name in ("reduce_decisions", "driver"):
        assert "max_containers=1" in _decorator_source(path, name)


def test_the_bound_is_refused_never_clamped() -> None:
    assert validate_map_container_bound(PLATFORM_MAX_MAP_CONTAINERS) == 40
    for bad in (0, -1, PLATFORM_MAX_MAP_CONTAINERS + 1, True, "40", 4.0):
        with pytest.raises(ProcessV2ConcurrencyError):
            validate_map_container_bound(bad)


def test_the_submission_geometry_never_exceeds_the_bound() -> None:
    tasks = [f"{index:064x}" for index in range(97)]
    for bound in (1, 20, PLATFORM_MAX_MAP_CONTAINERS):
        waves = plan_submission_waves(tasks, bound)
        assert all(len(wave) <= bound for wave in waves)
        assert [task for wave in waves for task in wave] == sorted(tasks)


def test_describe_refuses_a_fan_out_outside_the_platform_bound() -> None:
    with pytest.raises(ProcessV2ConcurrencyError):
        active8_app.describe.info.raw_f(max_map_containers=PLATFORM_MAX_MAP_CONTAINERS + 1)


# ---- Exact inputs ----


def _revision(seed: str) -> dict[str, Any]:
    return {"implementation_sha256": gate_zero_fixture._sha(seed)}


def _gate_zero_request(**overrides: Any) -> dict[str, Any]:
    arguments: dict[str, Any] = {
        "image_revision": {"image_revision_sha256": gate_zero_fixture._sha("image")},
        "gate_zero_revision": _revision("gate-zero"),
        "decision_index_revision": _revision("index"),
        "inputs": {
            "cache_run_artifact_root": "/artifacts/editing_v2/cache/g1",
            "rebind_run_artifact_root": "/artifacts/editing_v2/rebind/r1",
            "active8_run_artifact_root": "/artifacts/editing_v2/active8/a1",
        },
        "contract_file_sha256": gate_zero_fixture._sha("contract-file"),
        "contract_sha256": gate_zero_fixture._sha("contract"),
        "output_artifact_prefix": gate_zero_app.OUTPUT_ARTIFACT_PREFIX,
    }
    arguments.update(overrides)
    return gate_zero_app.build_run_request(**arguments)


def test_the_gate_zero_run_identity_is_content_addressed_and_nonauthorizing() -> None:
    first = _gate_zero_request()
    assert first == _gate_zero_request()
    require_no_granted_authority(first, label="request")
    require_authority_false(first, label="request")
    moved = _gate_zero_request(decision_index_revision=_revision("other-index"))
    assert moved["run_identity_sha256"] != first["run_identity_sha256"]
    moved = _gate_zero_request(contract_sha256=gate_zero_fixture._sha("other-contract"))
    assert moved["run_identity_sha256"] != first["run_identity_sha256"]


@pytest.mark.parametrize(
    "bad",
    ["/tmp/elsewhere/x", "artifacts/x/y", "/artifacts/../etc/passwd", "/artifacts/x/", ""],
)
def test_an_input_outside_the_artifact_root_is_refused(bad: str) -> None:
    for module in APPS:
        with pytest.raises(RuntimeError, match="below /artifacts|resolves outside"):
            module.artifact_path(bad, field="input")


def test_the_active8_run_identity_excludes_the_fan_out() -> None:
    """Worker count is a schedule; letting it in would address another run."""

    request = active8_app.build_run_request(
        image_revision={"image_revision_sha256": gate_zero_fixture._sha("image")},
        input_binding_revision=_revision("inputs"),
        active8_revision=_revision("active8"),
        inputs={
            "cache_run_artifact_root": "/artifacts/editing_v2/cache/g1",
            "rebind_run_artifact_root": "/artifacts/editing_v2/rebind/r1",
        },
        output_artifact_prefix=active8_app.OUTPUT_ARTIFACT_PREFIX,
    )
    flattened = json.dumps(request)
    assert "max_map_containers" not in flattened
    assert "container" not in flattened
    require_no_granted_authority(request, label="request")


# ---- Dirty tree and revision refusals ----


@pytest.mark.parametrize("module", APPS, ids=lambda module: Path(module.__file__).name)
def test_a_dirty_worktree_is_refused(module) -> None:
    commit, tree = "1" * 40, "2" * 40
    with (
        patch.object(module, "_git", side_effect=[commit, tree, "?? scratch.py"]),
        pytest.raises(RuntimeError, match="clean committed worktree"),
    ):
        module.local_image_revision(expected_commit=commit, repo_root=REPO_ROOT)


@pytest.mark.parametrize("module", APPS, ids=lambda module: Path(module.__file__).name)
def test_an_unexpected_commit_is_refused(module) -> None:
    with (
        patch.object(module, "_git", side_effect=["1" * 40, "2" * 40, ""]),
        pytest.raises(RuntimeError, match="clean committed worktree"),
    ):
        module.local_image_revision(expected_commit="3" * 40, repo_root=REPO_ROOT)
    with pytest.raises(RuntimeError, match="full lowercase Git commit"):
        module.local_image_revision(expected_commit="not-a-commit", repo_root=REPO_ROOT)


@pytest.mark.parametrize("module", APPS, ids=lambda module: Path(module.__file__).name)
def test_a_container_whose_files_differ_is_refused(module, tmp_path: Path) -> None:
    """The image revision is owner-computed remotely, never trusted as supplied."""

    remote = tmp_path / "remote"
    (remote / "modal_apps").mkdir(parents=True)
    for directory in module.IMAGE_SOURCE_DIRECTORIES:
        (remote / directory).mkdir()
        (remote / directory / "only.txt").write_text("original")
    launcher = remote / module.LAUNCHER_SOURCE
    launcher.write_text("# launcher\n")

    with patch.object(module, "_git", side_effect=["1" * 40, "2" * 40, "", "\n".join(
        [module.LAUNCHER_SOURCE]
        + [f"{directory}/only.txt" for directory in module.IMAGE_SOURCE_DIRECTORIES]
    )]):
        revision = module.local_image_revision(expected_commit="1" * 40, repo_root=remote)
    assert module.validate_remote_image_revision(revision, remote_root=remote) == revision

    launcher.write_text("# a different launcher\n")
    with pytest.raises(RuntimeError, match="differs in this container"):
        module.validate_remote_image_revision(revision, remote_root=remote)


def test_the_absent_active8_stage_is_refused_by_exact_name() -> None:
    """Today the stage does not exist, and the refusal must say which names."""

    with pytest.raises(RuntimeError) as error:
        active8_app.resolve_active8_stage(remote_root=REPO_ROOT)
    message = str(error.value)
    assert "editing_v2_process_v2_active8_policy" in message
    assert "editing_v2_process_v2_active8_mapreduce" in message

    with pytest.raises(RuntimeError, match="is not present in this tree"):
        gate_zero_app.resolve_decision_index_factory(remote_root=REPO_ROOT)


# ---- Volume visibility ----


@pytest.mark.parametrize(
    ("module", "name", "before"),
    [
        (active8_app, "decide_one_chunk", "execute_process_v2_active8_decision_task"),
        (active8_app, "reduce_decisions", "reduce_process_v2_active8_decisions"),
    ],
)
def test_a_worker_reloads_before_reading_another_containers_output(
    module, name: str, before: str
) -> None:
    segment = _function_source(Path(module.__file__), name)
    assert "artifact_volume.reload()" in segment
    assert segment.index("artifact_volume.reload()") < segment.index(before)
    assert segment.index("artifact_volume.reload()") < segment.index("artifact_volume.commit()")


def test_the_active8_driver_reloads_before_reading_before_scanning_and_after_reducing() -> None:
    segment = _function_source(Path(active8_app.__file__), "driver")
    reloads = [
        index
        for index in range(len(segment))
        if segment.startswith("artifact_volume.reload()", index)
    ]
    assert len(reloads) >= 3
    scan_at = segment.index("completed_process_v2_active8_task_ids")
    reduce_at = segment.index("reduce_decisions.remote(")
    assert any(index < scan_at for index in reloads)
    assert any(index > reduce_at for index in reloads)
    assert segment[reduce_at:].index("artifact_volume.reload()") < segment[reduce_at:].index(
        "report = {"
    )


def test_the_gate_zero_remote_function_reloads_around_its_publication() -> None:
    segment = _function_source(Path(gate_zero_app.__file__), "run_gate_zero")
    assert "reload_volume=artifact_volume.reload" in segment
    assert "commit_volume=artifact_volume.commit" in segment
    driver = _function_source(Path(gate_zero_app.__file__), "gate_zero_driver")
    assert driver.count("reload_volume()") == 2
    assert driver.index("reload_volume()") < driver.index("require_committed_inputs(")
    assert driver.index("commit_volume()") < driver.rindex("reload_volume()")


# ---- Restart behaviour, driven through the real Gate-0 driver ----


class _Recorder:
    def __init__(self, result: Any = None) -> None:
        self.calls = 0
        self.result = result

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        self.calls += 1
        return self.result


def _drive_gate_zero(tmp_path: Path, monkeypatch, *, contract, index) -> dict[str, Any]:
    upstream = _Recorder(
        {
            "cache_completion_sha256": gate_zero_fixture._sha("cache"),
            "rebind_completion_sha256": gate_zero_fixture._sha("rebind"),
            "admitted_source_identity": {"schema": "test.admitted", "schema_version": 1},
        }
    )
    monkeypatch.setattr(gate_zero_app, "validate_remote_image_revision", _Recorder({}))
    monkeypatch.setattr(gate_zero_app, "validate_implementation_revision", _Recorder({}))
    monkeypatch.setattr(gate_zero_app, "require_committed_inputs", upstream)
    monkeypatch.setattr(
        gate_zero_app, "resolve_decision_index_factory", _Recorder(lambda *a, **k: index)
    )
    reload_calls = _Recorder()
    commit_calls = _Recorder()
    report = gate_zero_app.gate_zero_driver(
        cache_run_artifact_root="/artifacts/editing_v2/cache/g1",
        rebind_run_artifact_root="/artifacts/editing_v2/rebind/r1",
        active8_run_artifact_root="/artifacts/editing_v2/active8/a1",
        output_artifact_prefix="/artifacts/editing_v2/process_v2_gate_zero",
        image_revision={"image_revision_sha256": gate_zero_fixture._sha("image")},
        gate_zero_revision=_revision("gate-zero"),
        decision_index_revision=_revision("index"),
        artifact_root=tmp_path,
        remote_root=REPO_ROOT,
        reload_volume=reload_calls,
        commit_volume=commit_calls,
    )
    assert upstream.calls == 1, "the driver must require the committed upstream generations"
    assert reload_calls.calls == 2 and commit_calls.calls == 1
    return report


@pytest.fixture(name="contract", scope="module")
def _contract():
    from compose_v4.experiments.editing_v2_process_v2_gate_zero import (
        load_process_v2_gate_zero_contract,
    )

    return load_process_v2_gate_zero_contract(repo_root=REPO_ROOT)


def test_an_interrupted_run_reuses_its_verified_artifacts(
    tmp_path: Path, monkeypatch, contract
) -> None:
    """Reuse is a revalidation, never a file-exists check."""

    index = gate_zero_fixture._complete_fixture(contract)
    first = _drive_gate_zero(tmp_path, monkeypatch, contract=contract, index=index)
    assert first["reused_existing_run"] is False
    assert first["structural_result"] == "PASS"

    second = _drive_gate_zero(tmp_path, monkeypatch, contract=contract, index=index)
    assert second["reused_existing_run"] is True
    assert second["run_artifact_root"] == first["run_artifact_root"]
    assert second["evidence_sha256"] == first["evidence_sha256"]
    assert second["completion_sha256"] == first["completion_sha256"]


def test_a_partial_run_root_is_refused_rather_than_completed_in_place(
    tmp_path: Path, monkeypatch, contract
) -> None:
    index = gate_zero_fixture._complete_fixture(contract)
    report = _drive_gate_zero(tmp_path, monkeypatch, contract=contract, index=index)
    run_root = tmp_path / Path(report["run_artifact_root"]).relative_to("/artifacts")
    (run_root / "COMPLETE.json").unlink()

    with pytest.raises(RuntimeError, match="partial artifact set"):
        _drive_gate_zero(tmp_path, monkeypatch, contract=contract, index=index)


def test_the_gate_zero_report_is_nonauthorizing_on_a_failing_gate(
    tmp_path: Path, monkeypatch, contract
) -> None:
    omitted = f"{contract.namespace}:atom_delete:connected_nonleaf_death"
    index = gate_zero_fixture._complete_fixture(contract, omit_cells=frozenset({omitted}))
    report = _drive_gate_zero(tmp_path, monkeypatch, contract=contract, index=index)
    assert report["structural_result"] == "FAIL"
    assert report["executed"] is True
    require_no_granted_authority(report, label="the Gate-0 run report")
    require_authority_false(report, label="the Gate-0 run report")
    assert report["training_launched"] is False


def test_a_stale_decision_index_is_refused(tmp_path: Path, monkeypatch, contract) -> None:
    """An index computed under another process identity is not evidence."""

    complete = gate_zero_fixture._complete_fixture(contract)
    stale = gate_zero_fixture.StandInIndex(
        complete._rows, complete._transitions, process_identity_sha256="0" * 64
    )
    with pytest.raises(RuntimeError, match="not the live Process-V2 identity"):
        _drive_gate_zero(tmp_path, monkeypatch, contract=contract, index=stale)
