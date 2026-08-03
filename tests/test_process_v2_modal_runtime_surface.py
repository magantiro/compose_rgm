"""The Process-V2 Modal runtime surface: Git, concurrency, visibility, streaming.

Four defects lived in this surface, and each is pinned here by the behaviour
that used to be wrong rather than by a comment saying it is now right.

**Planning could not run remotely.**  The remote driver called ``build_plan``,
which called ``repository_process_v2_rebind_source_revision``, which shells out
to ``git rev-parse HEAD``.  The image has source files and no ``.git``.
``test_planning_fails_on_the_real_image_surface_the_old_way`` constructs that
exact surface, a directory tree holding the serialized sources with no ``.git``,
and shows the old call raising there; the next test plans successfully on the
same tree with Git made unreachable outright.

**``--max-map-containers`` was cosmetic.**  Validated, printed, and then ignored
while the decorator capped at 20 and one ``starmap`` submitted everything.  The
tests below drive the production ``run_bounded_map`` and assert the submission
geometry for 1, 20 and 40, and the refusal of 0 and 41.

**Volume visibility was implicit.**  Reload calls and their ordering are
asserted against each remote function body.

**The admitted-source resolver built a global proof dictionary.**  It is now a
streaming join whose peak memory does not follow the corpus, whose evidence
digests are invariant to sharding, and whose rejection ledger is a deterministic
sorted file of complete reason-coded records.
"""

from __future__ import annotations

import ast
import gzip
import importlib.util
import json
import shutil
import subprocess
import sys
import tracemalloc
from pathlib import Path

import pytest

from compose_v4.data.editing_process_v2_admitted_source import (
    DEFAULT_REJECTION_LEDGER_FILENAME,
    REFUSAL_STATUS,
    ProcessV2AdmittedSourceError,
    resolve_process_v2_admitted_source,
    write_admitted_source_refusal_report,
)
from compose_v4.data.editing_process_v2_rebind import (
    ProcessV2RebindError,
    repository_process_v2_rebind_source_revision,
)
from compose_v4.data.editing_v2_process_v2_chunk_cache import (
    CACHE_IMPLEMENTATION_FILES,
    DEFAULT_CACHE_MAP_CONTAINERS,
    PLATFORM_MAX_MAP_CONTAINERS,
    ProcessV2ConcurrencyError,
    build_cache_implementation_revision,
    plan_submission_waves,
    run_bounded_map,
    validate_map_container_bound,
)
from compose_v4.data.editing_v2_process_v2_schema import canonical_sha256

_REPO_ROOT = Path(__file__).resolve().parents[1]
for _extra in (str(_REPO_ROOT / "tests"), str(_REPO_ROOT / "scripts")):
    if _extra not in sys.path:
        sys.path.insert(0, _extra)

import plan_process_v2_rebind as plan_driver  # noqa: E402
import test_editing_process_v2_rebind as v1_fixture  # noqa: E402

REBIND_APP = _REPO_ROOT / "modal_apps" / "run_process_v2_rebind_app.py"
CACHE_APP = _REPO_ROOT / "modal_apps" / "build_process_v2_chunk_cache_app.py"
# Exactly what each image serializes, which is what an image surface must hold.
IMAGE_DIRECTORIES = ("src", "configs")
IMAGE_FILES = (
    "scripts/plan_process_v2_rebind.py",
    "modal_apps/run_process_v2_rebind_app.py",
    "modal_apps/build_process_v2_chunk_cache_app.py",
)


def _load(path: Path):
    spec = importlib.util.spec_from_file_location(f"surface_{path.stem}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _function(path: Path, name: str) -> tuple[str, str]:
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    node = next(
        item
        for item in tree.body
        if isinstance(item, ast.FunctionDef) and item.name == name
    )
    segment = ast.get_source_segment(source, node)
    assert segment is not None
    return source, segment


def _decorator(path: Path, name: str) -> str:
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    node = next(
        item
        for item in tree.body
        if isinstance(item, ast.FunctionDef) and item.name == name
    )
    segment = ast.get_source_segment(source, node.decorator_list[0])
    assert segment is not None
    return segment


# ---- 1. The image surface has no .git, and planning must not need one ---------


@pytest.fixture(scope="module")
def image_surface(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The real serialized image tree: source files, and deliberately no .git."""

    root = tmp_path_factory.mktemp("image_surface") / "compose"
    for directory in IMAGE_DIRECTORIES:
        shutil.copytree(
            _REPO_ROOT / directory,
            root / directory,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )
    for relative in IMAGE_FILES:
        (root / relative).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(_REPO_ROOT / relative, root / relative)
    assert not (root / ".git").exists()
    return root


def test_planning_fails_on_the_real_image_surface_the_old_way(image_surface: Path) -> None:
    """The regression witness: the removed call cannot work in the image."""

    with pytest.raises(ProcessV2RebindError, match="cannot establish the rebind Git identity"):
        repository_process_v2_rebind_source_revision(repo_root=image_surface)


def test_planning_succeeds_on_the_image_surface_with_git_unreachable(
    image_surface: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The supplied revision is enough; Git is not merely unused, it is absent.

    ``subprocess.run`` is replaced with a hard failure for the duration, so a
    single surviving Git call anywhere on the planning path fails the test
    rather than silently succeeding on the developer's checkout.
    """

    payload = v1_fixture._build_v1_payload(tmp_path / "artifacts")
    revision = v1_fixture._source_revision()
    process_identity, _builder = v1_fixture._pinned_identities()

    def refuse(*_args: object, **_kwargs: object):
        raise AssertionError("remote planning invoked a subprocess")

    monkeypatch.setattr(subprocess, "run", refuse)
    plan = plan_driver.build_plan(
        artifact_root=payload.artifact_root,
        v1_payload_root_artifact_path=v1_fixture.PAYLOAD_ARTIFACT_PATH,
        expected_process_identity_sha256=process_identity["process_identity_sha256"],
        output_artifact_prefix="/artifacts/rebind_fixture",
        entries_per_task=1,
        source_revision=revision,
        repo_root=image_surface,
    )
    assert plan["source_revision"]["source_revision_sha256"] == revision["source_revision_sha256"]
    assert plan["expected_task_count"] == len(plan["tasks"])


def test_a_supplied_revision_that_does_not_match_the_image_is_refused(
    image_surface: Path,
) -> None:
    revision = dict(v1_fixture._source_revision())
    files = dict(revision["implementation_files"])
    files[next(iter(files))] = "0" * 64
    revision["implementation_files"] = files
    with pytest.raises(ProcessV2RebindError, match="stale or malformed"):
        plan_driver.resolve_source_revision(revision, repo_root=image_surface)


@pytest.mark.parametrize("app_path", [REBIND_APP, CACHE_APP])
def test_no_remote_body_can_reach_git(app_path: Path) -> None:
    """Git appears only under the local entrypoint's own call graph."""

    source = app_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    remote = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and any(
            isinstance(decorator, ast.Call)
            and isinstance(decorator.func, ast.Attribute)
            and decorator.func.attr == "function"
            for decorator in node.decorator_list
        )
    ]
    assert remote, f"{app_path.name} declares no remote functions"
    for node in remote:
        segment = ast.get_source_segment(source, node)
        assert segment is not None
        for forbidden in ("subprocess", "_git(", "repository_process_v2_rebind_source_revision"):
            assert forbidden not in segment, f"{node.name} reaches {forbidden}"
    # And the only Git helper is called from the local entrypoint alone.
    local = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "main"
    )
    local_segment = ast.get_source_segment(source, local)
    assert local_segment is not None
    assert "local_image_revision(" in local_segment or "local_source_revision(" in local_segment


def test_the_remote_cross_check_binds_the_source_and_image_revisions(
    image_surface: Path,
) -> None:
    """Two identities meet remotely, and pairing the wrong pair is refused."""

    rebind = _load(REBIND_APP)
    revision = v1_fixture._source_revision()
    image_revision = {"commit": revision["commit"], "tree": revision["tree"]}
    validated = rebind.validate_remote_source_revision(
        revision, image_revision, remote_root=image_surface
    )
    assert validated["source_revision_sha256"] == revision["source_revision_sha256"]

    with pytest.raises(RuntimeError, match="different trees"):
        rebind.validate_remote_source_revision(
            revision, {"commit": "c" * 40, "tree": "d" * 40}, remote_root=image_surface
        )


def test_each_remote_body_revalidates_the_supplied_revision() -> None:
    for path, names, validator in (
        (
            REBIND_APP,
            ("prove_one_range", "reduce_rebind", "driver"),
            "validate_remote_source_revision",
        ),
        (
            CACHE_APP,
            ("build_one_source_cache", "reduce_chunk_cache", "driver"),
            "validate_remote_source_revision",
        ),
    ):
        for name in names:
            _source, segment = _function(path, name)
            assert validator in segment, f"{path.name}:{name} does not revalidate its revision"


def test_the_cache_app_separates_the_narrow_and_broad_revisions(
    image_surface: Path,
) -> None:
    """The artifact-addressing revision is the library's, not the image's.

    The image revision hashes every file under ``src`` and ``configs``, so
    passing it in as the scientific ``source_revision`` -- which the app used to
    do -- relocated every cached byte whenever anything anywhere moved.
    """

    cache = _load(CACHE_APP)
    narrow = build_cache_implementation_revision(repo_root=image_surface)
    assert set(narrow["implementation_files"]) == set(CACHE_IMPLEMENTATION_FILES)
    assert "commit" not in narrow and "serialized_sources" not in narrow

    sources = {
        relative: cache._file_sha256(image_surface / relative)
        for relative in cache.serialized_source_paths(image_surface)
    }
    body = {
        "schema": cache.IMAGE_REVISION_SCHEMA,
        "schema_version": cache.IMAGE_REVISION_SCHEMA_VERSION,
        "commit": "a" * 40,
        "tree": "b" * 40,
        "worktree_clean": True,
        "serialized_sources": sources,
    }
    image_revision = {**body, "image_revision_sha256": cache._sha256(body)}
    assert (
        cache.validate_remote_source_revision(
            narrow, image_revision, remote_root=image_surface
        )
        == narrow
    )

    # The relationship between the two, stated as the structural fact it is:
    # the image inventory is a superset of the narrow one and agrees on every
    # shared file. That is what makes the explicit cross-check in the app
    # currently unreachable -- both sides rehash the same image -- and it is
    # kept there so a future divergence in either file set becomes loud rather
    # than silent. Recorded here rather than tested through a stub, because a
    # test of an unreachable branch proves nothing about the shipped path.
    assert set(CACHE_IMPLEMENTATION_FILES) <= set(sources)
    assert all(
        sources[relative] == narrow["implementation_files"][relative]
        for relative in CACHE_IMPLEMENTATION_FILES
    )

    # Both halves refuse a claim that does not match the image.
    drifted = {**sources, CACHE_IMPLEMENTATION_FILES[0]: "0" * 64}
    drifted_body = {**body, "serialized_sources": drifted}
    with pytest.raises(RuntimeError, match="differs in the image"):
        cache.validate_remote_source_revision(
            narrow,
            {**drifted_body, "image_revision_sha256": cache._sha256(drifted_body)},
            remote_root=image_surface,
        )
    forged_files = {**narrow["implementation_files"], CACHE_IMPLEMENTATION_FILES[0]: "1" * 64}
    forged_body = {
        "schema": narrow["schema"],
        "schema_version": narrow["schema_version"],
        "implementation_files": forged_files,
        "implementation_files_sha256": canonical_sha256(forged_files),
    }
    with pytest.raises(RuntimeError, match="modules that are actually present"):
        cache.validate_remote_source_revision(
            {**forged_body, "cache_implementation_sha256": canonical_sha256(forged_body)},
            image_revision,
            remote_root=image_surface,
        )


def test_the_rebind_launcher_takes_no_v1_payload_root() -> None:
    """Defect #10: the raw root was a bare unchecked positional. It is gone.

    Read off the real signature, not off a comment, so reintroducing the
    argument fails here rather than in review.
    """

    source = REBIND_APP.read_text(encoding="utf-8")
    tree = ast.parse(source)
    entrypoint = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "main"
    )
    arguments = [argument.arg for argument in entrypoint.args.args]
    assert arguments[0] == "cache_run_artifact_root"
    assert "v1_payload_root" not in arguments
    # `entries_per_task` is the cache's chunk size, so it is not a launcher dial
    # either: the chunk boundary is the task boundary.
    assert "entries_per_task" not in arguments

    driver_source = ast.get_source_segment(
        source, next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "driver")
    )
    assert driver_source is not None
    assert "build_cache_fed_plan" in driver_source
    assert "v1_payload_root" not in driver_source


def test_every_remote_rebind_body_requires_the_production_geometry() -> None:
    """An oracle-geometry plan cannot be mapped, reduced, or planned remotely."""

    for name in ("prove_one_range", "reduce_rebind", "driver"):
        _source, segment = _function(REBIND_APP, name)
        assert "require_production_source_geometry" in segment, name


# ---- 2. Real bounded parallelism ----------------------------------------------


@pytest.mark.parametrize(
    ("bound", "tasks", "expected"),
    [
        (1, 20, [1] * 20),
        (20, 20, [20]),
        (40, 20, [20]),
        (40, 100, [40, 40, 20]),
        (20, 100, [20, 20, 20, 20, 20]),
        (1, 3, [1, 1, 1]),
    ],
)
def test_the_requested_bound_changes_the_submission_geometry(
    bound: int, tasks: int, expected: list[int]
) -> None:
    ids = [f"{index:064x}" for index in range(tasks)]
    assert [len(wave) for wave in plan_submission_waves(ids, bound)] == expected


def test_run_bounded_map_submits_waves_and_lowers_the_autoscaler() -> None:
    submitted: list[tuple[str, ...]] = []
    autoscaler: list[dict[str, int]] = []
    ids = [f"{index:064x}" for index in range(45)]

    results = run_bounded_map(
        ids,
        max_map_containers=20,
        submit=lambda wave: (submitted.append(wave) or [f"done:{task}" for task in wave]),
        set_autoscaler=lambda **kwargs: autoscaler.append(kwargs),
    )

    assert [len(wave) for wave in submitted] == [20, 20, 5]
    assert max(len(wave) for wave in submitted) <= 20
    assert autoscaler == [{"max_containers": 20}]
    assert len(results) == 45
    # Deterministic and exhaustive: every task submitted exactly once, sorted.
    assert sorted(task for wave in submitted for task in wave) == sorted(ids)


@pytest.mark.parametrize("bad", [0, -1, 41, 1000, 1.0, "20", None, True])
def test_zero_and_over_forty_are_refused_not_clamped(bad: object) -> None:
    with pytest.raises(ProcessV2ConcurrencyError, match=r"\[1, 40\]"):
        validate_map_container_bound(bad)


def test_a_wave_that_loses_a_result_is_a_failure_not_a_silent_gap() -> None:
    with pytest.raises(ProcessV2ConcurrencyError, match="lost or repeated"):
        run_bounded_map(
            [f"{index:064x}" for index in range(4)],
            max_map_containers=2,
            submit=lambda wave: list(wave)[:-1],
        )


def test_a_repeated_task_identity_is_refused() -> None:
    with pytest.raises(ProcessV2ConcurrencyError, match="repeated task identity"):
        plan_submission_waves(["a" * 64, "a" * 64], 2)


def test_both_launchers_cap_at_the_one_shared_platform_ceiling() -> None:
    rebind, cache = _load(REBIND_APP), _load(CACHE_APP)
    assert rebind.PLATFORM_MAX_MAP_CONTAINERS == PLATFORM_MAX_MAP_CONTAINERS == 40
    assert cache.PLATFORM_MAX_MAP_CONTAINERS is PLATFORM_MAX_MAP_CONTAINERS
    assert cache.DEFAULT_CACHE_MAP_CONTAINERS == DEFAULT_CACHE_MAP_CONTAINERS == 20
    assert "max_containers=PLATFORM_MAX_MAP_CONTAINERS" in _decorator(REBIND_APP, "prove_one_range")
    assert "max_containers=PLATFORM_MAX_MAP_CONTAINERS" in _decorator(
        CACHE_APP, "build_one_source_cache"
    )
    # Reducers, publishers and drivers are serialized.
    for path, name in (
        (REBIND_APP, "reduce_rebind"),
        (REBIND_APP, "driver"),
        (CACHE_APP, "reduce_chunk_cache"),
        (CACHE_APP, "driver"),
    ):
        assert "max_containers=1" in _decorator(path, name), f"{path.name}:{name} is not serialized"


def test_each_driver_uses_the_bounded_map_and_not_a_single_starmap() -> None:
    for path, function_name in (
        (REBIND_APP, "prove_one_range"),
        (CACHE_APP, "build_one_source_cache"),
    ):
        _source, driver_segment = _function(path, "driver")
        assert "run_bounded_map(" in driver_segment
        assert "update_autoscaler" in driver_segment
        assert f"{function_name}.starmap" in driver_segment
        # The starmap is inside the wave callback, never over the whole set.
        assert "for task_id in wave" in driver_segment
        assert "for task_id in missing" not in driver_segment


def test_no_stage_of_either_launcher_requests_a_gpu() -> None:
    for path in (REBIND_APP, CACHE_APP):
        assert "gpu=" not in path.read_text(encoding="utf-8")


# ---- 3. Volume visibility boundaries ------------------------------------------


@pytest.mark.parametrize(
    ("path", "name", "before"),
    [
        (REBIND_APP, "prove_one_range", "execute_process_v2_rebind_task("),
        (REBIND_APP, "reduce_rebind", "reduce_process_v2_rebind("),
        (CACHE_APP, "build_one_source_cache", "execute_process_v2_chunk_cache_task("),
        (CACHE_APP, "reduce_chunk_cache", "reduce_process_v2_chunk_cache("),
    ],
)
def test_a_worker_reloads_before_reading_another_containers_output(
    path: Path, name: str, before: str
) -> None:
    _source, segment = _function(path, name)
    assert "artifact_volume.reload()" in segment
    assert segment.index("artifact_volume.reload()") < segment.index(before)
    assert "artifact_volume.commit()" in segment
    assert segment.index("artifact_volume.reload()") < segment.index("artifact_volume.commit()")


@pytest.mark.parametrize(
    ("path", "scan", "reducer"),
    [
        (REBIND_APP, "completed_process_v2_rebind_task_ids(", "reduce_rebind.remote("),
        (
            CACHE_APP,
            "completed_process_v2_chunk_cache_task_ids(",
            "reduce_chunk_cache.remote(",
        ),
    ],
)
def test_the_driver_reloads_before_scanning_and_after_the_reducer(
    path: Path, scan: str, reducer: str
) -> None:
    _source, segment = _function(path, "driver")
    reloads = [
        index
        for index in range(len(segment))
        if segment.startswith("artifact_volume.reload()", index)
    ]
    assert len(reloads) >= 3, "driver needs a reload before reading, before scanning, and after reducing"
    scan_at = segment.index(scan)
    reduce_at = segment.index(reducer)
    assert any(index < scan_at for index in reloads), "no reload before the reusable-task scan"
    assert any(index > reduce_at for index in reloads), "no reload after the reducer returns"
    # And the post-reducer reload precedes whatever resolves completion.
    tail = segment[reduce_at:]
    assert tail.index("artifact_volume.reload()") < tail.index("return {")


# ---- 4. The streaming admission join ------------------------------------------


def _resolved(tmp_path: Path, *, entries_per_task: int, tasks=None, ledger: str | None = None):
    payload = v1_fixture._build_v1_payload(
        tmp_path / "artifacts", tasks=tasks or v1_fixture._V1_TASKS_WITH_EXCLUSION
    )
    plan = v1_fixture._plan_for(payload, entries_per_task=entries_per_task)
    v1_fixture._execute_all(payload, plan)
    v1_fixture.reduce_process_v2_rebind(
        plan, artifact_root=payload.artifact_root, repo_root=v1_fixture.ROOT
    )
    source = resolve_process_v2_admitted_source(
        plan,
        artifact_root=payload.artifact_root,
        repo_root=v1_fixture.ROOT,
        rejection_ledger_artifact_path=ledger,
    )
    return payload, plan, source


def test_the_evidence_digests_survive_a_reshard_while_the_address_moves(
    tmp_path: Path,
) -> None:
    _fine_payload, fine_plan, fine = _resolved(tmp_path / "fine", entries_per_task=1)
    _coarse_payload, coarse_plan, coarse = _resolved(tmp_path / "coarse", entries_per_task=8)

    assert fine_plan["entries_per_task"] != coarse_plan["entries_per_task"]
    assert fine_plan["run_identity_sha256"] != coarse_plan["run_identity_sha256"]
    assert len(fine_plan["tasks"]) > len(coarse_plan["tasks"])

    # Semantic evidence is invariant to the schedule.
    assert fine.decision_semantic_sha256 == coarse.decision_semantic_sha256
    assert fine.admitted_evidence_sha256 == coarse.admitted_evidence_sha256
    assert (
        fine.rejection_ledger["rejection_semantic_sha256"]
        == coarse.rejection_ledger["rejection_semantic_sha256"]
    )
    assert (
        fine.rejection_ledger["rejection_stream_sha256"]
        == coarse.rejection_ledger["rejection_stream_sha256"]
    )
    assert dict(fine.counts) == dict(coarse.counts)

    # The physical execution identity is exactly where the schedule shows up.
    assert (
        fine.physical_execution_identity["physical_execution_sha256"]
        != coarse.physical_execution_identity["physical_execution_sha256"]
    )
    assert fine.physical_execution_identity["entries_per_task"] == 1
    assert coarse.physical_execution_identity["entries_per_task"] == 8
    assert (
        fine.physical_execution_identity["range_task_count"]
        > coarse.physical_execution_identity["range_task_count"]
    )
    # Task identity and run identity are separate fields, not one blended hash.
    assert set(fine.physical_execution_identity) >= {
        "run_identity_sha256",
        "plan_sha256",
        "range_task_inventory_sha256",
        "physical_execution_sha256",
    }


def test_a_different_corpus_does_change_the_evidence_digests(tmp_path: Path) -> None:
    """The negative half of invariance: the digests are not simply constant."""

    _clean_payload, _clean_plan, clean = _resolved(
        tmp_path / "clean", entries_per_task=1, tasks=v1_fixture._V1_TASKS
    )
    _mixed_payload, _mixed_plan, mixed = _resolved(
        tmp_path / "mixed", entries_per_task=1, tasks=v1_fixture._V1_TASKS_WITH_EXCLUSION
    )
    assert clean.counts["rejected_entries"] == 0 < mixed.counts["rejected_entries"]
    assert clean.decision_semantic_sha256 != mixed.decision_semantic_sha256
    assert clean.admitted_evidence_sha256 != mixed.admitted_evidence_sha256
    assert (
        clean.rejection_ledger["rejection_semantic_sha256"]
        != mixed.rejection_ledger["rejection_semantic_sha256"]
    )


def test_the_rejection_ledger_is_sorted_complete_and_reason_coded(tmp_path: Path) -> None:
    payload, plan, source = _resolved(tmp_path, entries_per_task=1)
    ledger_path = f"{plan['run_artifact_root']}/{DEFAULT_REJECTION_LEDGER_FILENAME}"
    republished = resolve_process_v2_admitted_source(
        plan,
        artifact_root=payload.artifact_root,
        repo_root=v1_fixture.ROOT,
        rejection_ledger_artifact_path=ledger_path,
    )
    ledger = republished.rejection_ledger
    assert ledger["ledger_artifact_path"] == ledger_path
    assert ledger["rejected_entries"] == source.counts["rejected_entries"] > 0
    assert ledger["rejection_stream_sha256"] == source.rejection_ledger["rejection_stream_sha256"]
    assert ledger["ledger_file_sha256"] is not None
    assert source.rejection_ledger["ledger_file_sha256"] is None

    on_disk = payload.artifact_root / ledger_path.removeprefix("/artifacts/")
    with gzip.open(on_disk, "rb") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    assert len(rows) == ledger["rejected_entries"]
    keys = [(row["v1_task_identity_sha256"], row["entry_index"]) for row in rows]
    assert keys == sorted(keys)
    for row in rows:
        # A complete record, not a hash of one.
        assert row["exclusion_code"]
        assert row["trace_id"] and row["v1_record_sha256"]
        assert "detail" in row and "unsupported_teacher_steps" in row
        assert "step_index" in row and "path_length" in row


def test_the_published_ledger_is_byte_stable_across_two_resolutions(tmp_path: Path) -> None:
    payload, plan, _source = _resolved(tmp_path, entries_per_task=1)
    ledger_path = f"{plan['run_artifact_root']}/{DEFAULT_REJECTION_LEDGER_FILENAME}"
    first = resolve_process_v2_admitted_source(
        plan,
        artifact_root=payload.artifact_root,
        repo_root=v1_fixture.ROOT,
        rejection_ledger_artifact_path=ledger_path,
    ).rejection_ledger
    on_disk = payload.artifact_root / ledger_path.removeprefix("/artifacts/")
    original = on_disk.read_bytes()
    second = resolve_process_v2_admitted_source(
        plan,
        artifact_root=payload.artifact_root,
        repo_root=v1_fixture.ROOT,
        rejection_ledger_artifact_path=ledger_path,
    ).rejection_ledger
    assert on_disk.read_bytes() == original
    assert first == second


def test_resolution_memory_grows_far_slower_than_the_global_dictionary_it_replaced(
    tmp_path: Path,
) -> None:
    """No global proof dictionary: peak memory tracks the range, not the corpus.

    Both arms are production code.  The streaming arm is
    ``resolve_process_v2_admitted_source``; the comparison arm materialises the
    whole overlay index, which is the shape the resolver used to build and then
    rebuild.  Absolute peaks are dominated by a fixed ~1.3 MB of executor and
    validation machinery, so the discriminating quantity is the *slope*: how
    much peak each arm adds when the corpus grows.  Measured over twelvefold
    growth the eager slope is about 3x the streaming one, stable to two decimal
    places across repeats, so the threshold is set at 2x.
    """

    from compose_v4.data.editing_process_v2_admitted_source import _overlay_index

    lane, role, names = v1_fixture._V1_TASKS_WITH_EXCLUSION[1]
    peaks: dict[int, tuple[int, int]] = {}
    entries: dict[int, int] = {}
    for multiplier in (2, 32):
        tasks = (
            v1_fixture._V1_TASKS_WITH_EXCLUSION[0],
            (lane, role, names * multiplier),
        )
        payload, plan, _built = _resolved(
            tmp_path / f"m{multiplier}", entries_per_task=1, tasks=tasks
        )
        tracemalloc.start()
        source = resolve_process_v2_admitted_source(
            plan, artifact_root=payload.artifact_root, repo_root=v1_fixture.ROOT
        )
        streaming_peak = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()

        tracemalloc.start()
        index = _overlay_index(plan, artifact_root=payload.artifact_root)
        eager = {identity: dict(index[identity]) for identity in index}
        eager_peak = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()

        peaks[multiplier] = (streaming_peak, eager_peak)
        entries[multiplier] = sum(len(decisions) for decisions in eager.values())
        assert entries[multiplier] == int(source.counts["source_entries"])

    assert entries[32] >= 10 * entries[2]
    streaming_slope = peaks[32][0] - peaks[2][0]
    eager_slope = peaks[32][1] - peaks[2][1]
    assert streaming_slope > 0 and eager_slope > 0
    assert eager_slope >= 2 * streaming_slope


def test_one_v1_task_keeps_at_most_one_range_resident(tmp_path: Path) -> None:
    """The mechanism behind the slope, asserted directly on the production view."""

    from compose_v4.data.editing_process_v2_admitted_source import (
        _LazyTaskDecisions,
        _overlay_index,
    )

    payload, plan, _source = _resolved(tmp_path, entries_per_task=1)
    index = _overlay_index(plan, artifact_root=payload.artifact_root)
    identity = str(plan["v1_payload_binding"]["v1_tasks"][1]["v1_task_identity_sha256"])
    view = index[identity]
    assert isinstance(view, _LazyTaskDecisions)

    ranges = sum(
        1 for task in plan["tasks"] if str(task["v1_task_identity_sha256"]) == identity
    )
    assert ranges > 1
    seen = 0
    for _entry_index, _decision in view.stream():
        seen += 1
        # One range's decisions, never the task's and never the corpus's.
        assert len(view._window) <= int(plan["entries_per_task"])
    assert seen == int(plan["v1_payload_binding"]["v1_tasks"][1]["v1_entries"])


def test_an_integrity_refusal_reports_outside_the_run_namespace(tmp_path: Path) -> None:
    payload, plan, _source = _resolved(tmp_path, entries_per_task=1)
    run_root = payload.artifact_root / str(plan["run_artifact_root"]).removeprefix("/artifacts/")
    diagnostics = tmp_path / "diagnostics"

    error = ProcessV2AdmittedSourceError("the V2 admission overlay is invalid")
    report_path = write_admitted_source_refusal_report(
        diagnostic_root=diagnostics,
        plan=plan,
        stage="resolve_process_v2_admitted_source",
        error=error,
        detail={"v1_task_identity_sha256": "0" * 64},
    )
    assert diagnostics not in run_root.parents and run_root not in report_path.parents
    report = json.loads(report_path.read_bytes())
    assert report["status"] == REFUSAL_STATUS
    assert report["run_identity_sha256"] == plan["run_identity_sha256"]
    assert report["training_authorized"] is False
    assert report_path.name == f"{report['refusal_sha256']}.json"
    # Republishing the same refusal is a no-op, not a collision.
    assert (
        write_admitted_source_refusal_report(
            diagnostic_root=diagnostics,
            plan=plan,
            stage="resolve_process_v2_admitted_source",
            error=error,
            detail={"v1_task_identity_sha256": "0" * 64},
        )
        == report_path
    )
