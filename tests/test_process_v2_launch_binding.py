"""The narrow implementation revision a Process-V2 launcher binds.

The defect this module exists to prevent is measured history, not theory: a
hand-written implementation file list was wrong in both directions, and its guard
could not fail because the guard's expectation was built from the constant it was
checking.  So the tests below never read the closure to build an expectation.
They name the modules a naive scan would MISS and require them to be present, and
they mutate a real file and require the identity to move.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from compose_v4.data.editing_v2_process_v2_launch_binding import (
    IMPLEMENTATION_REVISION_SCHEMA,
    ProcessV2LaunchBindingError,
    build_implementation_revision,
    implementation_closure,
    require_tracked,
    validate_implementation_revision,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
GATE_ZERO = "compose_v4.experiments.editing_v2_process_v2_gate_zero"


def test_the_closure_contains_the_edges_a_naive_scan_drops() -> None:
    """Two edge shapes silently vanish from a careless import scan.

    ``from compose_v4.rewrite import action_codec_v3 as v3`` records the PACKAGE
    as its module, so a scan that only takes ``node.module`` never reaches the
    codec that decides what a mark decodes to.  And an import written inside a
    function is an edge exactly as a top-level one is; four modules in this
    closure are reachable only that way.
    """

    closure = set(implementation_closure((GATE_ZERO,), repo_root=REPO_ROOT))
    assert "src/compose_v4/experiments/editing_v2_process_v2_gate_zero.py" in closure
    # `from package import module`, reached only through action_codec_v4.
    assert "src/compose_v4/rewrite/action_codec_v3.py" in closure
    # Imported inside a function of editing_v2_process_identity.
    assert "src/compose_v4/rewrite/process_v2_atom_delete.py" in closure
    assert "src/compose_v4/data/charge_policy.py" in closure
    # Imported inside a function of rewrite/operators.
    assert "src/compose_v4/rewrite/semantic_cycle_open.py" in closure
    assert len(closure) > 20


def test_the_closure_stops_at_the_package_boundary() -> None:
    closure = implementation_closure((GATE_ZERO,), repo_root=REPO_ROOT)
    assert all(path.startswith("src/compose_v4/") for path in closure)
    assert all(path.endswith(".py") for path in closure)
    assert list(closure) == sorted(closure)


def test_a_revision_is_deterministic_and_self_hashed() -> None:
    first = build_implementation_revision((GATE_ZERO,), repo_root=REPO_ROOT)
    second = build_implementation_revision((GATE_ZERO,), repo_root=REPO_ROOT)
    assert first == second
    assert first["schema"] == IMPLEMENTATION_REVISION_SCHEMA
    assert first["entry_modules"] == [GATE_ZERO]
    assert set(first["implementation_files"]) == set(
        implementation_closure((GATE_ZERO,), repo_root=REPO_ROOT)
    )
    assert validate_implementation_revision(first, repo_root=REPO_ROOT) == first


def _mirror(tmp_path: Path) -> Path:
    root = tmp_path / "tree"
    (root / "src").mkdir(parents=True)
    shutil.copytree(REPO_ROOT / "src" / "compose_v4", root / "src" / "compose_v4")
    return root


def test_editing_any_closure_member_moves_the_identity(tmp_path: Path) -> None:
    """Sampled across the three edge shapes, on a real tree copy.

    The samples are chosen for how they enter the closure, not for convenience:
    the entry module itself, a module reachable only through the
    ``from package import module`` form, and one reachable only through a
    function-local import.
    """

    root = _mirror(tmp_path)
    baseline = build_implementation_revision((GATE_ZERO,), repo_root=root)
    assert baseline["implementation_sha256"] == (
        build_implementation_revision((GATE_ZERO,), repo_root=REPO_ROOT)["implementation_sha256"]
    )
    for relative in (
        "src/compose_v4/experiments/editing_v2_process_v2_gate_zero.py",
        "src/compose_v4/rewrite/action_codec_v3.py",
        "src/compose_v4/rewrite/process_v2_atom_delete.py",
    ):
        target = root / relative
        original = target.read_bytes()
        target.write_bytes(original + b"\n# behaviour-affecting edit\n")
        moved = build_implementation_revision((GATE_ZERO,), repo_root=root)
        assert moved["implementation_sha256"] != baseline["implementation_sha256"], relative
        with pytest.raises(ProcessV2LaunchBindingError, match=relative):
            validate_implementation_revision(baseline, repo_root=root)
        target.write_bytes(original)
    assert build_implementation_revision((GATE_ZERO,), repo_root=root) == baseline


def test_a_module_outside_the_closure_does_not_move_the_identity(tmp_path: Path) -> None:
    """The negative control: without it the closure could be 'all of src'."""

    root = _mirror(tmp_path)
    baseline = build_implementation_revision((GATE_ZERO,), repo_root=root)
    outside = root / "src/compose_v4/eval/molecular_quality.py"
    assert outside.is_file()
    assert outside.relative_to(root).as_posix() not in baseline["implementation_files"]
    outside.write_bytes(outside.read_bytes() + b"\n# unrelated edit\n")
    assert build_implementation_revision((GATE_ZERO,), repo_root=root) == baseline


def test_the_revision_carries_no_commit() -> None:
    """A commit touching no behaviour must not relocate a content-addressed artifact."""

    revision = build_implementation_revision((GATE_ZERO,), repo_root=REPO_ROOT)
    assert "commit" not in revision
    assert "tree" not in revision
    assert not [key for key in revision if "commit" in key or "git" in key]


# ---- Refusals ----


def _package(root: Path, name: str, body: str) -> None:
    path = root / "src" / "compose_v4" / f"{name}.py"
    path.parent.mkdir(parents=True, exist_ok=True)
    (path.parent / "__init__.py").touch()
    path.write_text(body)


def test_an_absent_entry_module_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ProcessV2LaunchBindingError, match="no source file"):
        implementation_closure(("compose_v4.not_a_module",), repo_root=tmp_path)


def test_an_entry_module_outside_the_package_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ProcessV2LaunchBindingError, match="outside the compose_v4 package"):
        implementation_closure(("numpy",), repo_root=tmp_path)


def test_a_dynamic_import_in_the_closure_is_refused(tmp_path: Path) -> None:
    """A dynamic edge is invisible to a static scan, so the closure would lie."""

    _package(tmp_path, "entry", "from compose_v4 import lazy_leaf\n")
    _package(
        tmp_path,
        "lazy_leaf",
        "import importlib\n\n\ndef go():\n    importlib.import_module('x')\n",
    )
    with pytest.raises(ProcessV2LaunchBindingError, match="importlib"):
        implementation_closure(("compose_v4.entry",), repo_root=tmp_path)

    _package(tmp_path, "lazy_leaf", "def go():\n    return __import__('x')\n")
    with pytest.raises(ProcessV2LaunchBindingError, match="__import__"):
        implementation_closure(("compose_v4.entry",), repo_root=tmp_path)


def test_a_relative_import_in_the_closure_is_refused(tmp_path: Path) -> None:
    _package(tmp_path, "entry", "from compose_v4 import sibling\n")
    _package(tmp_path, "sibling", "from . import other\n")
    with pytest.raises(ProcessV2LaunchBindingError, match="relative import"):
        implementation_closure(("compose_v4.entry",), repo_root=tmp_path)


def test_a_revision_with_the_wrong_field_set_is_refused() -> None:
    revision = dict(build_implementation_revision((GATE_ZERO,), repo_root=REPO_ROOT))
    revision["extra"] = 1
    with pytest.raises(ProcessV2LaunchBindingError, match="field set disagrees"):
        validate_implementation_revision(revision, repo_root=REPO_ROOT)


def test_a_resealed_revision_over_a_shortened_file_set_is_refused() -> None:
    """Dropping a file and resealing must not validate.

    This is the exact shape that survived before: a guard comparing the payload
    to a constant the payload was built from.
    """

    from compose_v4.data.editing_v2_process_v2_schema import canonical_sha256

    revision = build_implementation_revision((GATE_ZERO,), repo_root=REPO_ROOT)
    files = dict(revision["implementation_files"])
    files.pop("src/compose_v4/rewrite/action_codec_v3.py")
    body = {
        "schema": revision["schema"],
        "schema_version": revision["schema_version"],
        "entry_modules": revision["entry_modules"],
        "implementation_files": files,
        "implementation_files_sha256": canonical_sha256(files),
    }
    resealed = {**body, "implementation_sha256": canonical_sha256(body)}
    with pytest.raises(ProcessV2LaunchBindingError, match="does not describe this tree"):
        validate_implementation_revision(resealed, repo_root=REPO_ROOT)


def test_require_tracked_names_every_untracked_input() -> None:
    require_tracked(["a.py"], tracked=["a.py", "b.py"])
    with pytest.raises(ProcessV2LaunchBindingError, match=r"\['c.py'\]"):
        require_tracked(["a.py", "c.py"], tracked=["a.py"])
