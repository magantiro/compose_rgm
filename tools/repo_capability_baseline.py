"""Enumerate every public symbol reachable from every ``compose_v4`` module.

This exists to make a STRUCTURAL refactor falsifiable. The repository has found
six separate cases of a real, tested mechanism that no production caller reached;
moving a module out of its caller's path would create a seventh silently. The
guard against that is a mechanical before/after comparison of the importable
public surface, not a reading of the diff.

Two independent views are recorded per module, because each catches what the
other cannot:

``import`` view
    The module is actually imported and ``vars(module)`` is filtered to names not
    beginning with an underscore. This is the real answer to "can a caller still
    write ``from compose_v4.x import y``", and it sees names created by star
    imports, re-exports and decorators. It is blind to a module that cannot be
    imported in this environment at all.

``ast`` view
    The module source is parsed and top-level ``def``/``class``/assignment and
    ``import ... as`` bindings are collected. This never executes anything, so it
    covers modules whose dependencies are missing, and it is stable under a
    dependency upgrade. It is blind to dynamically created names.

The comparison rule the caller is expected to apply is a SUPERSET check: every
``(module, symbol)`` pair present at baseline must still be present afterwards,
or be listed as an intentional, justified break. A symbol that moved must remain
importable from its old path through a shim.

Determinism: output is sorted at every level and carries the interpreter and
library versions it was produced under, because an ``import`` view is only
comparable against another ``import`` view taken in the same environment.

Usage::

    python3 tools/repo_capability_baseline.py --out diagnostics/repo_hygiene/x.json
    python3 tools/repo_capability_baseline.py --compare BASELINE.json --out AFTER.json
"""

from __future__ import annotations

import argparse
import ast
import importlib
import json
import pathlib
import signal
import sys
import traceback
from typing import Any

# ---- Discovery ----

MODULE_IMPORT_TIMEOUT_SECONDS = 60


def repo_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parents[1]


def discover_modules(package_root: pathlib.Path) -> list[str]:
    """Return every importable ``compose_v4`` module name, sorted.

    A directory without ``__init__.py`` is still walked: this repository has
    subpackages that rely on namespace-package semantics, and skipping them
    would understate the baseline surface, which is the one direction of error
    this tool must not make.
    """
    names: list[str] = []
    for path in sorted(package_root.rglob("*.py")):
        relative = path.relative_to(package_root.parent)
        parts = list(relative.with_suffix("").parts)
        if parts[-1] == "__init__":
            parts = parts[:-1]
            if not parts:
                continue
        names.append(".".join(parts))
    names.append(package_root.name)
    return sorted(set(names))


# ---- AST view ----


def ast_public_symbols(path: pathlib.Path) -> tuple[list[str], str | None]:
    """Top-level public bindings a reader would expect to be importable."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (OSError, SyntaxError) as exc:
        return [], f"{type(exc).__name__}: {exc}"

    found: set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            found.add(node.name)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    found.add(target.id)
        elif isinstance(node, ast.AnnAssign):
            if isinstance(node.target, ast.Name):
                found.add(node.target.id)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            # A re-export is part of the public surface: callers import through it.
            for alias in node.names:
                if alias.name == "*":
                    continue
                bound = alias.asname or alias.name.split(".")[0]
                found.add(bound)
    return sorted(n for n in found if not n.startswith("_")), None


# ---- Import view ----


class _Timeout(Exception):
    pass


def _alarm(_signum: int, _frame: Any) -> None:
    raise _Timeout(f"import exceeded {MODULE_IMPORT_TIMEOUT_SECONDS}s")


def import_public_symbols(name: str) -> tuple[list[str] | None, str | None]:
    """Import ``name`` and return its public names, or the failure reason.

    A failure is recorded rather than raised. Some modules legitimately cannot
    import in a given environment (an optional dependency, a missing asset), and
    the comparison only requires that the set of importable modules does not
    SHRINK.
    """
    previous = signal.signal(signal.SIGALRM, _alarm)
    signal.alarm(MODULE_IMPORT_TIMEOUT_SECONDS)
    try:
        module = importlib.import_module(name)
    except BaseException as exc:  # noqa: BLE001 - a failure is data here, not an error
        head = str(exc).strip().splitlines()
        detail = head[0] if head else ""
        return None, f"{type(exc).__name__}: {detail[:200]}"
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)
    return sorted(n for n in vars(module) if not n.startswith("_")), None


# ---- Report ----


def build_report(root: pathlib.Path) -> dict[str, Any]:
    package_root = root / "src" / "compose_v4"
    src = str(root / "src")
    if src not in sys.path:
        sys.path.insert(0, src)

    modules = discover_modules(package_root)
    entries: dict[str, Any] = {}
    for name in modules:
        relative = name.replace(".", "/")
        path = package_root.parent / f"{relative}.py"
        if not path.exists():
            path = package_root.parent / relative / "__init__.py"
        ast_symbols, ast_error = ast_public_symbols(path)
        import_symbols, import_error = import_public_symbols(name)
        entries[name] = {
            "ast_symbols": ast_symbols,
            "ast_error": ast_error,
            "import_ok": import_symbols is not None,
            "import_symbols": import_symbols if import_symbols is not None else [],
            "import_error": import_error,
        }

    importable = sorted(n for n, e in entries.items() if e["import_ok"])
    return {
        "schema_version": 1,
        "interpreter": sys.version.split()[0],
        "library_versions": _library_versions(),
        "module_count": len(entries),
        "importable_module_count": len(importable),
        "ast_symbol_pair_count": sum(len(e["ast_symbols"]) for e in entries.values()),
        "import_symbol_pair_count": sum(len(e["import_symbols"]) for e in entries.values()),
        "modules": {name: entries[name] for name in sorted(entries)},
    }


def _library_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for name in ("numpy", "rdkit", "torch", "networkx", "scipy"):
        try:
            versions[name] = str(getattr(importlib.import_module(name), "__version__", "unknown"))
        except BaseException:  # noqa: BLE001 - absence is the datum
            versions[name] = "absent"
    return versions


def compare(baseline: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    """Return the pairs present at baseline and missing afterwards, per view."""

    def pairs(report: dict[str, Any], key: str) -> set[tuple[str, str]]:
        out: set[tuple[str, str]] = set()
        for module, entry in report["modules"].items():
            for symbol in entry[key]:
                out.add((module, symbol))
        return out

    result: dict[str, Any] = {}
    for key in ("ast_symbols", "import_symbols"):
        before = pairs(baseline, key)
        now = pairs(after, key)
        lost = sorted(f"{m}.{s}" for m, s in before - now)
        result[key] = {
            "baseline_pairs": len(before),
            "after_pairs": len(now),
            "lost": lost,
            "lost_count": len(lost),
            "gained_count": len(now - before),
            "is_superset": not lost,
        }

    before_importable = {n for n, e in baseline["modules"].items() if e["import_ok"]}
    after_importable = {n for n, e in after["modules"].items() if e["import_ok"]}
    lost_modules = sorted(before_importable - after_importable)
    result["importable_modules"] = {
        "baseline": len(before_importable),
        "after": len(after_importable),
        "lost": lost_modules,
        "is_superset": not lost_modules,
    }
    result["environments_match"] = (
        baseline["interpreter"] == after["interpreter"]
        and baseline["library_versions"] == after["library_versions"]
    )
    result["verdict"] = (
        "CAPABILITY_PRESERVED"
        if result["ast_symbols"]["is_superset"]
        and result["import_symbols"]["is_superset"]
        and result["importable_modules"]["is_superset"]
        else "CAPABILITY_LOST"
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, help="where to write the report JSON")
    parser.add_argument("--compare", help="baseline report to diff the new one against")
    args = parser.parse_args()

    root = repo_root()
    report = build_report(root)
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        f"modules={report['module_count']} importable={report['importable_module_count']} "
        f"ast_pairs={report['ast_symbol_pair_count']} "
        f"import_pairs={report['import_symbol_pair_count']} -> {out}"
    )

    if args.compare:
        baseline = json.loads(pathlib.Path(args.compare).read_text(encoding="utf-8"))
        diff = compare(baseline, report)
        diff_path = out.with_name(out.stem + "_vs_baseline.json")
        diff_path.write_text(json.dumps(diff, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps({k: v for k, v in diff.items() if k != "modules"}, indent=2)[:4000])
        print(f"-> {diff_path}")
        return 0 if diff["verdict"] == "CAPABILITY_PRESERVED" else 1
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:  # noqa: BLE001 - surface the traceback, never a bare verdict
        traceback.print_exc()
        sys.exit(2)
