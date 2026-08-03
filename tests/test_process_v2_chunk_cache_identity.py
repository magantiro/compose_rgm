"""The chunk cache's own revision, and what a resealed artifact cannot hide.

Two defects are pinned here.

**The cache hashed no implementation revision at all.**  ``plan_process_v2_chunk
_cache`` used to accept any mapping carrying a ``source_revision_sha256`` key
and fold that string into ``cache_physical_identity_sha256`` without ever
re-deriving it, so any caller could mint a revision and get a well-formed,
self-consistent plan.  The revision is now owner-computed from
``CACHE_IMPLEMENTATION_FILES`` and re-derived at every boundary.

**The launcher's image revision was being used as the scientific one.**  It
hashes every file under ``src`` and ``configs``, so editing anything anywhere
relocated every cached byte.  The two are now separate, and the narrow one
deliberately excludes both the Git commit and the rebind module.

**A resealed artifact is self-consistent by construction.**  Editing a nested
revision, a chunk count or a census and recomputing the self-hash produces a
document that validates against itself.  Only re-deriving each declared value
from what it addresses can refuse it, and that is what these tests exercise:
every mutation below is resealed before it is offered.

**The revision covered storing and not reading, and the test could not say so.**
``CACHE_IMPLEMENTATION_FILES`` named the modules that write a chunk.  Two trees
differing only in ``rewrite/progress.py`` therefore read the same committed
cache, published the identical ``cache_implementation_sha256``, both validated
clean, and handed their callers different decoded rows.  The test that should
have caught it asserted ``set(revision["implementation_files"]) ==
set(CACHE_IMPLEMENTATION_FILES)`` -- an expectation recomputed from the constant
under test, which ``build_cache_implementation_revision`` satisfies by
construction, so deleting ``packed_trace_store.py``, ``trace_shard.py`` or the
cache module *itself* from the list left the whole suite green.  The replacement
derives the decode closure by running the real decode and requires the revision
to hash every module in it, then proves per module that editing it moves the
cache address.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

from compose_v4.data.editing_v2_process_v2_chunk_cache import (
    CACHE_IMPLEMENTATION_FILES,
    COMPLETION_FILENAME,
    MANIFEST_FILENAME,
    SOURCE_DIRNAME,
    ProcessV2ChunkCacheError,
    build_cache_implementation_revision,
    cache_implementation_sha256,
    chunk_targets_for_source,
    execute_process_v2_chunk_cache_task,
    load_committed_process_v2_chunk_cache_completion,
    open_process_v2_chunk_cache,
    plan_process_v2_chunk_cache,
    read_process_v2_chunk_target,
    reduce_process_v2_chunk_cache,
    validate_cache_implementation_revision,
    validate_process_v2_chunk_cache_completion,
    validate_process_v2_chunk_cache_manifest,
    validate_process_v2_chunk_cache_plan,
    validate_process_v2_chunk_cache_source,
    write_process_v2_chunk_cache_plan,
)
from compose_v4.data.editing_v2_process_v2_schema import canonical_bytes, canonical_sha256
from compose_v4.rewrite import action_codec, action_codec_v3, action_codec_v4, operators

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "tests") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "tests"))

import test_editing_v2_process_v2_completion_binder as binder_fixture  # noqa: E402

_register_extra_fixture_traces = binder_fixture._register_extra_fixture_traces


# ---- Fixtures -----------------------------------------------------------------


def _built_cache(tmp_path: Path, *, records_per_chunk: int = 2):
    payload, _completion, expectation = binder_fixture.build_migration_run(tmp_path / "artifacts")
    binding = binder_fixture.bind(payload, expectation)
    plan = plan_process_v2_chunk_cache(
        binding,
        output_artifact_prefix="/artifacts/chunk_cache_identity_fixture",
        records_per_chunk=records_per_chunk,
    )
    write_process_v2_chunk_cache_plan(plan, artifact_root=payload.artifact_root)
    for task in plan["tasks"]:
        execute_process_v2_chunk_cache_task(
            plan, task["task_identity_sha256"], artifact_root=payload.artifact_root
        )
    completion = reduce_process_v2_chunk_cache(plan, artifact_root=payload.artifact_root)
    return payload, plan, completion


def _run_root(payload, plan) -> Path:
    return payload.artifact_root / str(plan["run_artifact_root"]).removeprefix("/artifacts/")


def _reseal(body: dict[str, Any], *, field: str) -> dict[str, Any]:
    stripped = {key: item for key, item in body.items() if key != field}
    return {**stripped, field: canonical_sha256(stripped)}


def _multi_chunk_source(payload, plan) -> Path:
    """A published source holding at least two chunks, or fail loudly.

    The chunking-rule and inventory mutations are only expressible on a source
    that actually has a non-final chunk; silently exercising a one-chunk source
    would make both tests pass without testing anything.
    """

    root = _run_root(payload, plan) / SOURCE_DIRNAME
    for task in plan["tasks"]:
        source_dir = root / str(task["task_identity_sha256"])
        manifest = json.loads((source_dir / MANIFEST_FILENAME).read_bytes())
        if int(manifest["chunk_count"]) >= 2:
            return source_dir
    raise AssertionError("the fixture published no multi-chunk source cache")


def _rewrite_manifest(source_dir: Path, mutate) -> dict[str, Any]:
    """Mutate a published manifest and reseal it, exactly as a forger would."""

    manifest = json.loads((source_dir / MANIFEST_FILENAME).read_bytes())
    mutate(manifest)
    resealed = _reseal(manifest, field="manifest_sha256")
    (source_dir / MANIFEST_FILENAME).write_bytes(canonical_bytes(resealed) + b"\n")
    return resealed


# ---- The decode closure, derived by running the decode ------------------------
#
# Nothing below reads ``CACHE_IMPLEMENTATION_FILES``.  The closure is derived
# three ways out of the production code itself, and their union is what the
# narrow revision has to hash:
#
#   1. every ``compose_v4`` source whose code *executes* while
#      ``read_process_v2_chunk_target`` turns cached lines into
#      ``SemanticPackedRowRead`` rows;
#   2. every ``compose_v4`` module defining a type -- base classes included --
#      reachable in the decoded row object graph, which catches a module that
#      supplies only the row's shape; and
#   3. every ``compose_v4`` module defining a payload class the frozen V4
#      persistence surface can construct, which catches a family whose traces
#      this fixture happens not to carry.
#
# ``codec_implementation_hash()`` is not a substitute for any of it: it hashes a
# *declared* surface, so a codec whose declared rules are unchanged but whose
# decoding differs leaves it fixed.

_SRC_ROOT = _REPO_ROOT / "src"
# A floor on the derivation.  A derivation that silently collapsed to a handful
# of modules would satisfy the coverage assertion vacuously; these are modules
# the measured escape ran through, and they are independent of the constant.
_DECODE_CLOSURE_WITNESSES = (
    "src/compose_v4/data/editing_v2_process_v2_chunk_cache.py",
    "src/compose_v4/data/packed_trace_store.py",
    "src/compose_v4/rewrite/action_codec_v4.py",
    "src/compose_v4/rewrite/progress.py",
    "src/compose_v4/rewrite/trace.py",
    "src/compose_v4/rewrite/trace_shard.py",
)
_DECODE_CLOSURE: tuple[str, ...] | None = None


def _repository_source(filename: object) -> str | None:
    """The repository-relative path of a ``compose_v4`` source, or ``None``."""

    if not isinstance(filename, str) or not filename.endswith(".py"):
        return None
    path = Path(filename)
    if not path.is_absolute() or not path.is_relative_to(_SRC_ROOT):
        return None
    return str(path.relative_to(_REPO_ROOT))


def _module_source(module: object) -> str | None:
    if not isinstance(module, str):
        return None
    return _repository_source(getattr(sys.modules.get(module), "__file__", None))


def _executed_sources(call):
    """Run ``call``, reporting every ``compose_v4`` source that executed in it."""

    executed: set[str] = set()

    def tracer(frame, event, argument):
        source = _repository_source(frame.f_code.co_filename)
        if source is not None:
            executed.add(source)
        return None

    previous = sys.gettrace()
    sys.settrace(tracer)
    try:
        result = call()
    finally:
        sys.settrace(previous)
    return executed, result


def _row_type_sources(rows) -> set[str]:
    """Every module defining a type the decoded rows are actually built from."""

    found: set[str] = set()
    seen: set[int] = set()

    def walk(value: object, depth: int) -> None:
        if depth > 12 or id(value) in seen:
            return
        seen.add(id(value))
        for klass in type(value).__mro__:
            source = _module_source(getattr(klass, "__module__", None))
            if source is not None:
                found.add(source)
        if isinstance(value, (str, bytes, bytearray, int, float, type(None))):
            return
        if isinstance(value, dict):
            for key, item in value.items():
                walk(key, depth + 1)
                walk(item, depth + 1)
            return
        if isinstance(value, (list, tuple, set, frozenset)):
            for item in value:
                walk(item, depth + 1)
            return
        attributes = getattr(value, "__dict__", None)
        if isinstance(attributes, dict):
            for item in attributes.values():
                walk(item, depth + 1)

    for row in rows:
        walk(row, 0)
    return found


def _persisted_payload_sources() -> set[str]:
    """Every module defining a payload the frozen V4 surface can decode into."""

    semantic = {
        action_codec_v4.SEMANTIC_ATOM_RESTATE_RULE: action_codec_v4.SEMANTIC_ATOM_RESTATE_PAYLOAD,
        action_codec_v4.SEMANTIC_CYCLE_CLOSE_RULE: action_codec_v4.SEMANTIC_CYCLE_CLOSE_PAYLOAD,
        action_codec_v3.SEMANTIC_CYCLE_OPEN_RULE: action_codec_v3.SEMANTIC_CYCLE_OPEN_PAYLOAD,
    }
    found: set[str] = set()
    for rule in action_codec_v4.supported_executor_rules():
        name = semantic.get(rule) or action_codec._EXECUTOR_TO_PAYLOAD[rule]
        payload = action_codec._PAYLOAD_TYPES.get(name) or getattr(operators, name)
        source = _module_source(payload.__module__)
        assert source is not None, f"payload {name!r} resolves outside the repository"
        found.add(source)
    return found


def _derive_decode_closure(root: Path) -> tuple[str, ...]:
    # The extra traces are registered here rather than through the autouse
    # fixture, so the derivation is identical whichever module asks for it.
    pristine = dict(binder_fixture.v1_fixture._FIXTURE_TRACES)
    binder_fixture.v1_fixture._FIXTURE_TRACES.update(binder_fixture._EXTRA_TRACES)
    try:
        payload, plan, _completion = _built_cache(root)
    finally:
        binder_fixture.v1_fixture._FIXTURE_TRACES.clear()
        binder_fixture.v1_fixture._FIXTURE_TRACES.update(pristine)
    sources = _run_root(payload, plan) / SOURCE_DIRNAME

    def decode() -> list[Any]:
        decoded: list[Any] = []
        for task in plan["tasks"]:
            source_dir = sources / str(task["task_identity_sha256"])
            manifest = validate_process_v2_chunk_cache_source(source_dir)
            for target in chunk_targets_for_source(
                manifest, source_artifact_path=str(task["output_artifact_path"])
            ):
                decoded.extend(read_process_v2_chunk_target(source_dir, target=target))
        return decoded

    executed, rows = _executed_sources(decode)
    assert rows, "the closure derivation decoded no rows"
    return tuple(sorted(executed | _row_type_sources(rows) | _persisted_payload_sources()))


def decode_closure(root: Path) -> tuple[str, ...]:
    """The derived decode closure, computed once and reused across modules."""

    global _DECODE_CLOSURE
    if _DECODE_CLOSURE is None:
        _DECODE_CLOSURE = _derive_decode_closure(root)
    return _DECODE_CLOSURE


# ---- 1. The narrow revision is owner-computed ---------------------------------


def test_the_narrow_revision_hashes_every_module_the_decode_path_uses(
    tmp_path: Path,
) -> None:
    """The revision must cover reading, not only storing.

    The expectation is the closure derived by *running* the decode, so it does
    not move when ``CACHE_IMPLEMENTATION_FILES`` moves.  That is the whole point:
    the assertion this replaced compared the revision's keys against the constant
    the revision is built from, which no edit to the constant can break.
    """

    closure = decode_closure(tmp_path)
    missing_witnesses = [name for name in _DECODE_CLOSURE_WITNESSES if name not in closure]
    assert not missing_witnesses, (
        "the derivation lost modules the decode demonstrably runs through, so its "
        f"coverage result would be vacuous: {missing_witnesses}"
    )

    revision = build_cache_implementation_revision()
    uncovered = sorted(set(closure) - set(revision["implementation_files"]))
    assert not uncovered, (
        "the narrow revision does not hash every module the decode path uses, so "
        "two trees that decode a cached row differently address the same cache: "
        f"{uncovered}"
    )


def test_the_narrow_revision_names_only_behaviour_affecting_modules() -> None:
    revision = build_cache_implementation_revision()
    # No Git object: a commit that touches nothing behaviour-affecting must not
    # relocate a single cached byte.
    assert not any(
        key in revision for key in ("commit", "tree", "worktree_clean", "serialized_sources")
    )
    # The rebind consumes this cache and cannot change what it holds, so its
    # bytes are deliberately outside the revision.
    assert "src/compose_v4/data/editing_process_v2_rebind.py" not in CACHE_IMPLEMENTATION_FILES
    assert revision["cache_implementation_sha256"] == cache_implementation_sha256()


def test_the_narrow_revision_moves_when_a_named_module_moves(tmp_path: Path) -> None:
    """A copy of the tree with one named module edited must hash differently."""

    surface = tmp_path / "surface"
    for relative in CACHE_IMPLEMENTATION_FILES:
        target = surface / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((_REPO_ROOT / relative).read_bytes())
    assert build_cache_implementation_revision(repo_root=surface) == (
        build_cache_implementation_revision()
    )

    edited = surface / CACHE_IMPLEMENTATION_FILES[0]
    edited.write_bytes(edited.read_bytes() + b"\n# behaviour-affecting edit\n")
    moved = build_cache_implementation_revision(repo_root=surface)
    assert moved["cache_implementation_sha256"] != cache_implementation_sha256()


def test_editing_any_decode_path_module_moves_the_cache_address(tmp_path: Path) -> None:
    """The escape, restated as a per-module property of the shipped revision.

    Every module the decode runs through gets edited on a faithful copy of the
    tree, one at a time; each edit has to move ``cache_implementation_sha256``.
    A module the revision does not hash cannot move it, which is exactly how two
    trees that decode differently came to share one cache address.  Deleting
    ``packed_trace_store.py``, ``trace_shard.py`` or this cache module itself
    from ``CACHE_IMPLEMENTATION_FILES`` used to leave every test green and now
    fails here, because the loop is over the derived closure and not over the
    constant being checked.
    """

    closure = decode_closure(tmp_path)
    surface = tmp_path / "mutation_surface"
    for relative in sorted(set(closure) | set(CACHE_IMPLEMENTATION_FILES)):
        target = surface / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((_REPO_ROOT / relative).read_bytes())
    baseline = cache_implementation_sha256(repo_root=surface)
    assert baseline == cache_implementation_sha256()

    unhashed: list[str] = []
    for relative in closure:
        edited = surface / relative
        pristine = edited.read_bytes()
        edited.write_bytes(pristine + b"\n# behaviour-affecting edit\n")
        try:
            if cache_implementation_sha256(repo_root=surface) == baseline:
                unhashed.append(relative)
        finally:
            edited.write_bytes(pristine)
    assert not unhashed, (
        "editing these decode-path modules leaves the cache address fixed: "
        f"{unhashed}"
    )
    assert cache_implementation_sha256(repo_root=surface) == baseline


def test_a_caller_supplied_revision_is_recomputed_and_matched() -> None:
    """Supplying a revision adds a requirement; it never becomes the authority."""

    live = build_cache_implementation_revision()
    assert validate_cache_implementation_revision(live) == live

    forged_files = dict(live["implementation_files"])
    forged_files[CACHE_IMPLEMENTATION_FILES[0]] = "0" * 64
    forged_body = {
        "schema": live["schema"],
        "schema_version": live["schema_version"],
        "implementation_files": forged_files,
        "implementation_files_sha256": canonical_sha256(forged_files),
    }
    # Internally perfect, and still refused: the only revision that counts is the
    # one recomputed from the modules that are actually present.
    forged = _reseal(
        {**forged_body, "cache_implementation_sha256": ""}, field="cache_implementation_sha256"
    )
    assert forged["cache_implementation_sha256"] == canonical_sha256(forged_body)
    with pytest.raises(ProcessV2ChunkCacheError, match="modules that are actually present"):
        validate_cache_implementation_revision(forged)


def test_a_revision_with_an_extra_or_missing_field_is_refused() -> None:
    live = build_cache_implementation_revision()
    with pytest.raises(ProcessV2ChunkCacheError, match="fields disagree"):
        validate_cache_implementation_revision({**live, "commit": "a" * 40})
    with pytest.raises(ProcessV2ChunkCacheError, match="fields disagree"):
        validate_cache_implementation_revision(
            {key: item for key, item in live.items() if key != "implementation_files_sha256"}
        )


# ---- 2. Acceptance test 11: resealed mutations fail ---------------------------


def test_a_resealed_nested_revision_fails_the_plan(tmp_path: Path) -> None:
    """Defect #6: edit the nested revision, reseal both layers, still refused."""

    _payload, plan, _completion = _built_cache(tmp_path)
    revision = dict(plan["cache_implementation_revision"])
    files = dict(revision["implementation_files"])
    files[CACHE_IMPLEMENTATION_FILES[-1]] = "1" * 64
    revision["implementation_files"] = files
    revision["implementation_files_sha256"] = canonical_sha256(files)
    revision = _reseal(revision, field="cache_implementation_sha256")

    resealed = _reseal(
        {
            **plan,
            "cache_implementation_revision": revision,
            "cache_implementation_sha256": revision["cache_implementation_sha256"],
        },
        field="plan_sha256",
    )
    # The forged plan is self-consistent at every level it declares.
    assert resealed["plan_sha256"] == canonical_sha256(
        {key: item for key, item in resealed.items() if key != "plan_sha256"}
    )
    assert resealed["cache_implementation_sha256"] == revision["cache_implementation_sha256"]
    with pytest.raises(ProcessV2ChunkCacheError, match="modules that are actually present"):
        validate_process_v2_chunk_cache_plan(resealed)


def test_a_resealed_plan_that_keeps_the_old_digest_fails_its_addressing(
    tmp_path: Path,
) -> None:
    """The second, independent refusal: the run address is derived, not copied."""

    _payload, plan, _completion = _built_cache(tmp_path)
    resealed = _reseal(
        {**plan, "cache_implementation_sha256": "2" * 64}, field="plan_sha256"
    )
    with pytest.raises(ProcessV2ChunkCacheError, match="does not address its own revision"):
        validate_process_v2_chunk_cache_plan(resealed)


def test_a_resealed_chunk_count_fails_the_source_manifest(tmp_path: Path) -> None:
    """Defect #7: ``chunk_count`` is re-derived from the chunks it counts."""

    payload, plan, _completion = _built_cache(tmp_path)
    source_dir = _run_root(payload, plan) / SOURCE_DIRNAME / plan["tasks"][0][
        "task_identity_sha256"
    ]

    def bump(manifest: dict[str, Any]) -> None:
        manifest["chunk_count"] = int(manifest["chunk_count"]) + 1

    resealed = _rewrite_manifest(source_dir, bump)
    assert resealed["manifest_sha256"] == canonical_sha256(
        {key: item for key, item in resealed.items() if key != "manifest_sha256"}
    )
    with pytest.raises(ProcessV2ChunkCacheError, match="chunks and lists"):
        validate_process_v2_chunk_cache_manifest(resealed)
    with pytest.raises(ProcessV2ChunkCacheError, match="chunks and lists"):
        validate_process_v2_chunk_cache_source(source_dir)


def test_a_resealed_chunk_inventory_fails_its_inventory_hash(tmp_path: Path) -> None:
    payload, plan, _completion = _built_cache(tmp_path, records_per_chunk=1)
    source_dir = _multi_chunk_source(payload, plan)

    def shrink(manifest: dict[str, Any]) -> None:
        # A consistent-looking smaller cache: drop the last chunk, fix the count
        # and the entry census, and leave the inventory hash addressing the old
        # list. The inventory hash is the value that cannot be talked around.
        chunks = manifest["chunks"][:-1]
        manifest["chunks"] = chunks
        manifest["chunk_count"] = len(chunks)
        manifest["entries"] = int(chunks[-1]["entry_stop"])

    resealed = _rewrite_manifest(source_dir, shrink)
    with pytest.raises(ProcessV2ChunkCacheError, match="inventory hash"):
        validate_process_v2_chunk_cache_manifest(resealed)


def test_a_resealed_entry_census_fails_its_chunk_cover(tmp_path: Path) -> None:
    """``entries`` is re-derived from the address space the chunks cover.

    Found by mutation: bypassing this check left every focused suite green, so
    a manifest could declare a census its own chunks do not add up to -- which
    is exactly how a cache silently reduces a corpus.
    """

    payload, plan, _completion = _built_cache(tmp_path)
    source_dir = _run_root(payload, plan) / SOURCE_DIRNAME / plan["tasks"][0][
        "task_identity_sha256"
    ]

    def inflate(manifest: dict[str, Any]) -> None:
        manifest["entries"] = int(manifest["entries"]) + 1

    resealed = _rewrite_manifest(source_dir, inflate)
    with pytest.raises(ProcessV2ChunkCacheError, match="does not cover its entries"):
        validate_process_v2_chunk_cache_manifest(resealed)
    with pytest.raises(ProcessV2ChunkCacheError, match="does not cover its entries"):
        validate_process_v2_chunk_cache_source(source_dir)


def test_a_resealed_chunk_partition_fails_the_chunking_rule(tmp_path: Path) -> None:
    """A contiguous address space is not enough; the chunk size is a rule."""

    payload, plan, _completion = _built_cache(tmp_path, records_per_chunk=1)
    source_dir = _multi_chunk_source(payload, plan)

    def relabel(manifest: dict[str, Any]) -> None:
        manifest["records_per_chunk"] = 3

    resealed = _rewrite_manifest(source_dir, relabel)
    with pytest.raises(ProcessV2ChunkCacheError, match="the chunking rule requires"):
        validate_process_v2_chunk_cache_manifest(resealed)


def test_a_resealed_manifest_revision_fails_the_source(tmp_path: Path) -> None:
    payload, plan, _completion = _built_cache(tmp_path)
    source_dir = _run_root(payload, plan) / SOURCE_DIRNAME / plan["tasks"][0][
        "task_identity_sha256"
    ]

    def relabel(manifest: dict[str, Any]) -> None:
        manifest["cache_implementation_sha256"] = "3" * 64

    resealed = _rewrite_manifest(source_dir, relabel)
    with pytest.raises(ProcessV2ChunkCacheError, match="different chunk-cache implementation"):
        validate_process_v2_chunk_cache_manifest(resealed)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("chunk_count", 999, "census disagrees"),
        ("entries", 999, "census disagrees"),
        ("source_count", 999, "census disagrees"),
        ("cache_implementation_sha256", "4" * 64, "different implementation revision"),
    ],
)
def test_a_resealed_completion_census_is_refused(
    tmp_path: Path, field: str, value: object, message: str
) -> None:
    _payload, _plan, completion = _built_cache(tmp_path)
    resealed = _reseal({**completion, field: value}, field="completion_sha256")
    with pytest.raises(ProcessV2ChunkCacheError, match=message):
        validate_process_v2_chunk_cache_completion(resealed)


def test_a_resealed_completion_inventory_row_is_refused(tmp_path: Path) -> None:
    _payload, _plan, completion = _built_cache(tmp_path)
    inventory = [dict(row) for row in completion["source_inventory"]]
    inventory[0]["entries"] = int(inventory[0]["entries"]) + 1
    resealed = _reseal(
        {
            **completion,
            "source_inventory": inventory,
            "source_inventory_sha256": canonical_sha256(inventory),
            "entries": int(completion["entries"]) + 1,
        },
        field="completion_sha256",
    )
    # Every declared census now reconciles inside the completion itself; the
    # refusal comes from the published manifests it claims to summarise.
    assert validate_process_v2_chunk_cache_completion(resealed)


def test_an_inflated_inventory_row_is_refused_at_the_opening_boundary(
    tmp_path: Path,
) -> None:
    payload, _plan, completion = _built_cache(tmp_path)
    inventory = [dict(row) for row in completion["source_inventory"]]
    inventory[0]["entries"] = int(inventory[0]["entries"]) + 1
    resealed = _reseal(
        {
            **completion,
            "source_inventory": inventory,
            "source_inventory_sha256": canonical_sha256(inventory),
            "entries": int(completion["entries"]) + 1,
        },
        field="completion_sha256",
    )
    with pytest.raises(ProcessV2ChunkCacheError, match="disagrees with the committed completion"):
        open_process_v2_chunk_cache(resealed, artifact_root=payload.artifact_root)


# ---- 3. The completion is the committed marker --------------------------------


def test_an_uncommitted_generation_cannot_be_opened(tmp_path: Path) -> None:
    """Acceptance test 15, first half: absent is not partial."""

    payload, plan, completion = _built_cache(tmp_path)
    run_root = _run_root(payload, plan)
    (run_root / COMPLETION_FILENAME).unlink()
    with pytest.raises(ProcessV2ChunkCacheError, match="not committed"):
        load_committed_process_v2_chunk_cache_completion(
            str(plan["run_artifact_root"]), artifact_root=payload.artifact_root
        )
    # Every source cache is still present and individually valid, which is
    # exactly the state a killed reducer leaves behind. It is still not a
    # generation.
    for task in plan["tasks"]:
        assert validate_process_v2_chunk_cache_source(
            run_root / SOURCE_DIRNAME / task["task_identity_sha256"]
        )
    reduce_process_v2_chunk_cache(plan, artifact_root=payload.artifact_root)
    reopened = load_committed_process_v2_chunk_cache_completion(
        str(plan["run_artifact_root"]), artifact_root=payload.artifact_root
    )
    assert reopened == completion


def test_a_refused_reduction_publishes_no_committed_generation(tmp_path: Path) -> None:
    payload, _completion, expectation = binder_fixture.build_migration_run(
        tmp_path / "artifacts"
    )
    binding = binder_fixture.bind(payload, expectation)
    plan = plan_process_v2_chunk_cache(
        binding,
        output_artifact_prefix="/artifacts/chunk_cache_identity_fixture",
        records_per_chunk=2,
    )
    write_process_v2_chunk_cache_plan(plan, artifact_root=payload.artifact_root)
    for task in plan["tasks"][:-1]:
        execute_process_v2_chunk_cache_task(
            plan, task["task_identity_sha256"], artifact_root=payload.artifact_root
        )
    with pytest.raises(ProcessV2ChunkCacheError):
        reduce_process_v2_chunk_cache(plan, artifact_root=payload.artifact_root)
    run_root = _run_root(payload, plan)
    assert not (run_root / COMPLETION_FILENAME).exists()
    with pytest.raises(ProcessV2ChunkCacheError, match="not committed"):
        load_committed_process_v2_chunk_cache_completion(
            str(plan["run_artifact_root"]), artifact_root=payload.artifact_root
        )


def test_the_opening_boundary_orders_sources_and_targets(tmp_path: Path) -> None:
    payload, _plan, completion = _built_cache(tmp_path)
    generation = open_process_v2_chunk_cache(completion, artifact_root=payload.artifact_root)
    cells = [
        (str(manifest["data_lane"]), str(manifest["split"]))
        for manifest in generation.source_manifests
    ]
    assert cells == sorted(cells)
    targets = generation.targets()
    keys = [(target.data_lane, target.split, target.entry_start) for target in targets]
    assert keys == sorted(keys)
    assert len(targets) == int(completion["chunk_count"])
    assert sum(target.row_count for target in targets) == int(completion["entries"])
