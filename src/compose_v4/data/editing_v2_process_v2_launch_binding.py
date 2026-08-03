"""The narrow implementation revision a Process-V2 launcher binds.

WHY A CLOSURE AND NOT A HAND-WRITTEN LIST
-----------------------------------------
The chunk cache learned this the expensive way: a hand-written
``CACHE_IMPLEMENTATION_FILES`` was wrong in BOTH directions -- it named a module
that does not exist and missed five that decide what a cached row decodes to --
and its guard could not fail, because the guard's expectation was built from the
constant it was checking.  Two trees differing only in ``rewrite/progress.py``
published the identical revision and returned different rows.

So this module does not accept a list.  It takes ENTRY modules and derives the
closure by parsing the import graph, including imports written inside functions,
which are edges exactly as top-level ones are.  A caller cannot shorten the
closure without deleting an import, and deleting an import changes behaviour.

WHAT THE CLOSURE DOES NOT COVER, STATED RATHER THAN ASSUMED
-----------------------------------------------------------
A dynamic import (``importlib``, ``__import__``) is invisible to a static scan,
so :func:`implementation_closure` REFUSES any closure member that imports
``importlib`` or calls ``__import__``.  The claim is then sound rather than
optimistic: either every edge is statically visible, or the build fails and says
which module broke the assumption.

The revision deliberately carries no Git commit.  A commit that touches no
behaviour-affecting module must not relocate a content-addressed artifact; the
commit is the launcher's separate, broader image identity.

NOTHING HERE LAUNCHES ANYTHING.  It computes and validates hashes.
"""

from __future__ import annotations

import ast
import hashlib
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from compose_v4.data.editing_v2_process_v2_schema import canonical_sha256

IMPLEMENTATION_REVISION_SCHEMA = (
    "compose.editing_v2.process_v2.launch_implementation_revision"
)
IMPLEMENTATION_REVISION_SCHEMA_VERSION = 1

_PACKAGE = "compose_v4"
_SOURCE_DIRNAME = "src"
_REVISION_FIELDS = frozenset(
    {
        "schema",
        "schema_version",
        "entry_modules",
        "implementation_files",
        "implementation_files_sha256",
        "implementation_sha256",
    }
)


class ProcessV2LaunchBindingError(RuntimeError):
    """A launcher's implementation binding is incomplete, absent, or disagrees."""


def _module_file(module: str, *, repo_root: Path) -> Path | None:
    base = Path(repo_root) / _SOURCE_DIRNAME / Path(*module.split("."))
    for candidate in (base.with_suffix(".py"), base / "__init__.py"):
        if candidate.is_file():
            return candidate
    return None


def _parse(path: Path) -> ast.Module:
    try:
        return ast.parse(Path(path).read_text(), filename=str(path))
    except (OSError, SyntaxError) as error:
        raise ProcessV2LaunchBindingError(f"cannot parse {path}") from error


def _edges(tree: ast.Module) -> set[str]:
    """Every module name imported anywhere in the file, nesting included.

    ``from a.b import c`` contributes both ``a.b`` and ``a.b.c``, because the
    ``from package import module`` form is how ``action_codec_v4`` reaches
    ``action_codec_v3`` and a scan that only recorded ``a.b`` would drop it.
    """

    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                raise ProcessV2LaunchBindingError(
                    "a relative import cannot be resolved by module name; the closure "
                    "would silently omit its target"
                )
            module = node.module or ""
            if module:
                names.add(module)
                names.update(f"{module}.{alias.name}" for alias in node.names)
    return names


def _require_static_imports(path: Path, tree: ast.Module) -> None:
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id == "__import__":
                raise ProcessV2LaunchBindingError(
                    f"{path} calls __import__, so its edges are not statically visible "
                    "and the derived implementation closure would be incomplete"
                )
    if any(name == "importlib" or name.startswith("importlib.") for name in _edges(tree)):
        raise ProcessV2LaunchBindingError(
            f"{path} imports importlib, so its edges are not statically visible and the "
            "derived implementation closure would be incomplete"
        )


def implementation_closure(
    entry_modules: Sequence[str], *, repo_root: Path
) -> tuple[str, ...]:
    """Every ``compose_v4`` source file reachable from ``entry_modules``.

    Returned as repository-relative POSIX paths, sorted, entries included.

    Raises:
        ProcessV2LaunchBindingError: if an entry module is absent, if any closure
            member uses a relative or dynamic import, or if the closure is empty.
    """

    root = Path(repo_root)
    pending: list[str] = []
    for module in entry_modules:
        if not module.startswith(f"{_PACKAGE}."):
            raise ProcessV2LaunchBindingError(
                f"entry module {module!r} is outside the {_PACKAGE} package"
            )
        if _module_file(module, repo_root=root) is None:
            raise ProcessV2LaunchBindingError(
                f"entry module {module!r} has no source file under {root / _SOURCE_DIRNAME}"
            )
        pending.append(module)

    visited: set[str] = set()
    files: set[Path] = set()
    while pending:
        module = pending.pop()
        if module in visited:
            continue
        visited.add(module)
        path = _module_file(module, repo_root=root)
        if path is None:
            continue
        files.add(path)
        tree = _parse(path)
        _require_static_imports(path, tree)
        for name in _edges(tree):
            if name.startswith(f"{_PACKAGE}.") and name not in visited:
                pending.append(name)
    if not files:
        raise ProcessV2LaunchBindingError("the derived implementation closure is empty")
    return tuple(sorted(path.relative_to(root).as_posix() for path in files))


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def build_implementation_revision(
    entry_modules: Sequence[str], *, repo_root: Path
) -> dict[str, Any]:
    """Hash the derived closure into one deterministic non-authorizing revision."""

    root = Path(repo_root)
    entries = tuple(sorted(entry_modules))
    if not entries:
        raise ProcessV2LaunchBindingError("an implementation revision needs an entry module")
    files = {
        relative: _file_sha256(root / relative)
        for relative in implementation_closure(entries, repo_root=root)
    }
    body: dict[str, Any] = {
        "schema": IMPLEMENTATION_REVISION_SCHEMA,
        "schema_version": IMPLEMENTATION_REVISION_SCHEMA_VERSION,
        "entry_modules": list(entries),
        "implementation_files": files,
        "implementation_files_sha256": canonical_sha256(files),
    }
    return {**body, "implementation_sha256": canonical_sha256(body)}


def validate_implementation_revision(
    revision: Mapping[str, Any], *, repo_root: Path
) -> dict[str, Any]:
    """Re-derive the revision from the tree in hand and require exact equality.

    Owner-computed at every boundary: a supplied revision is provenance at most
    and is never trusted, so a container running different code fails here rather
    than publishing under an identity it did not earn.  No Git is called, which
    is why this runs where ``.git`` does not exist.
    """

    if not isinstance(revision, Mapping) or set(revision) != _REVISION_FIELDS:
        raise ProcessV2LaunchBindingError(
            "the implementation revision field set disagrees; expected "
            f"{sorted(_REVISION_FIELDS)}"
        )
    entries = revision["entry_modules"]
    if not isinstance(entries, list) or not all(isinstance(name, str) for name in entries):
        raise ProcessV2LaunchBindingError("entry_modules must be a list of module names")
    expected = build_implementation_revision(entries, repo_root=repo_root)
    if dict(revision) != expected:
        supplied = revision.get("implementation_files")
        observed = expected["implementation_files"]
        detail = "the file set differs"
        if isinstance(supplied, Mapping) and set(supplied) == set(observed):
            moved = sorted(key for key in observed if supplied[key] != observed[key])
            detail = f"these files differ: {moved}" if moved else "the envelope differs"
        raise ProcessV2LaunchBindingError(
            f"the supplied implementation revision does not describe this tree; {detail}"
        )
    return expected


def require_tracked(paths: Iterable[str], *, tracked: Iterable[str]) -> None:
    """Every serialized path must be Git-tracked, so a launch cannot ship a stray file."""

    known = set(tracked)
    untracked = sorted(path for path in paths if path not in known)
    if untracked:
        raise ProcessV2LaunchBindingError(
            f"these launch inputs are not tracked in Git and cannot be bound: {untracked}"
        )


__all__ = [
    "IMPLEMENTATION_REVISION_SCHEMA",
    "IMPLEMENTATION_REVISION_SCHEMA_VERSION",
    "ProcessV2LaunchBindingError",
    "build_implementation_revision",
    "implementation_closure",
    "require_tracked",
    "validate_implementation_revision",
]
