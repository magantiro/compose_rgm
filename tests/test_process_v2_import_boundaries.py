"""Import boundaries the Process-V2 chain depends on, proven by scanning source.

Three boundaries are recorded here.  Each is a property of the *import graph*,
which no behavioural test can observe: a module that imports the wrong thing
still computes the right answer until the day the import's own validation
refuses, and by then the layers above it have been built on the assumption that
the boundary held.

1. **The reference successor kernel stays a test oracle.**  Enforcement already
   exists in ``tests/test_reference_successor_kernel.py``, which owns the strict
   rule that no production module may so much as *mention* the oracle.  That gate
   is not duplicated and not weakened here.  Recorded finding: it fails at this
   commit, on a false positive, because it is a substring scan and
   ``src/compose_v4/rewrite/editing_v2_process_identity.py`` names the oracle's
   *path* in the content-addressed file list whose hashes define the V1 process
   identity.  That is a hashed path literal, not an edge.  What this module adds
   is the typed version of the same boundary: an import-graph check that no
   production module, and in particular no Process-V2 module, actually imports
   the oracle.  Repairing the substring gate belongs to the module that owns it.

2. **No Process-V2 module imports a V1 loader that revalidates V1 schema or live
   identity.**  This boundary currently FAILS, deliberately and visibly: see the
   expected-failure test at the bottom.

3. **The frozen schema module stays a leaf.**  It is the module every Process-V2
   layer imports, so an edge out of it is an edge every layer acquires.

Scanning, not convention: the whole point is that these hold at every commit,
including commits written by someone who has not read this file.
"""

from __future__ import annotations

import ast
import inspect
import sys
from pathlib import Path

import pytest

from compose_v4.data.editing_v2_semantic_active8_source_adapter import (
    resolve_editing_v2_semantic_active8_sources,
)

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "tests") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "tests"))

import test_reference_successor_kernel as oracle_boundary  # noqa: E402

# ---- Which modules are Process-V2 modules ----

# The scanned roots.  ``scripts`` and ``modal_apps`` are included because a
# driver that imports a live-V1 loader breaks the boundary exactly as a library
# module would; the boundary is about what runs, not about where it lives.
_SCANNED_ROOTS = ("src", "scripts", "modal_apps")

# A Process-V2 module is named as one.  Naming is how this repository already
# separates the two process lineages, and a rule anyone can apply by reading a
# path is a rule that survives a new file being added by someone who has not
# read this test.
_PROCESS_V2_NAME_MARKER = "process_v2"

FROZEN_SCHEMA_RELATIVE_PATH = "src/compose_v4/data/editing_v2_process_v2_schema.py"

# The V1 loaders a Process-V2 module may not import, declared as
# ``(module, symbol)``.  Membership is not asserted by fiat: the test below
# proves the declared loader really does revalidate the *live* V1 identity, so
# this list stays evidence-based rather than becoming a style rule.
V1_LOADERS_THAT_REVALIDATE_LIVE_V1_IDENTITY = {
    (
        "compose_v4.data.editing_v2_semantic_active8_source_adapter",
        "resolve_editing_v2_semantic_active8_sources",
    ),
}


def _process_v2_modules() -> tuple[Path, ...]:
    return tuple(
        sorted(
            path
            for root in _SCANNED_ROOTS
            if (_REPO_ROOT / root).is_dir()
            for path in (_REPO_ROOT / root).rglob("*.py")
            if _PROCESS_V2_NAME_MARKER in path.name
        )
    )


def _module_name(path: Path) -> str:
    """The dotted name a file is imported under, for resolving relative imports."""

    # ``src/compose_v4/data/x.py`` imports as ``compose_v4.data.x``; a script or
    # an app imports by its bare module name.  Both drop exactly the root.
    parts = list(path.relative_to(_REPO_ROOT).with_suffix("").parts)[1:]
    return ".".join(parts)


def _imported_symbols(path: Path) -> set[tuple[str, str]]:
    """Every ``(module, symbol)`` the file imports, relative imports resolved.

    ``import a.b`` contributes ``("a.b", "")`` so a plain module import is still
    an edge; a boundary that only looked at ``from`` imports would miss it.
    """

    module_name = _module_name(path)
    package = module_name.rsplit(".", 1)[0] if "." in module_name else ""
    edges: set[tuple[str, str]] = set()
    for node in ast.walk(ast.parse(path.read_text(), filename=str(path))):
        if isinstance(node, ast.Import):
            for alias in node.names:
                edges.add((alias.name, ""))
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = package.split(".")
                base = base[: len(base) - node.level + 1] if node.level > 1 else base
                target = ".".join(filter(None, [".".join(base), node.module or ""]))
            else:
                target = node.module or ""
            for alias in node.names:
                edges.add((target, alias.name))
    return edges


# ---- The module set itself ----


def test_the_process_v2_module_set_is_discovered_and_is_not_empty() -> None:
    """Without this every boundary below could pass by scanning nothing.

    A discovery rule that silently stops matching turns three enforced gates
    into three green no-ops, which is worse than having no gate at all because
    the suite then reports the boundary as held.
    """

    modules = _process_v2_modules()
    relative = {str(path.relative_to(_REPO_ROOT)) for path in modules}
    assert len(modules) >= 6
    # The two modules the boundaries below are actually about must be in the set.
    assert FROZEN_SCHEMA_RELATIVE_PATH in relative
    assert "src/compose_v4/data/editing_v2_process_v2_active8_source.py" in relative
    # Every scanned root that contributes a module contributes a real file.
    assert all(path.is_file() for path in modules)


# ---- 1. The reference successor kernel stays a test oracle ----


def test_the_owning_oracle_gate_exists_and_covers_the_process_v2_directories() -> None:
    """Without this the Process-V2 chain relies on a gate it never checks exists.

    ``tests/test_reference_successor_kernel.py`` owns the strict text rule and is
    not duplicated here.  What is asserted is that it still exists and still
    scans every directory a Process-V2 module lives in, so deleting, renaming or
    narrowing it fails here rather than silently removing the enforcement the
    Process-V2 modules inherit.

    The gate is deliberately not invoked.  It fails at this commit on a false
    positive (see this module's docstring), and importing a pre-existing failure
    into this file would report a defect that is not this boundary's.
    """

    gate = oracle_boundary.test_no_production_module_imports_the_oracle
    gate_source = inspect.getsource(gate)
    assert "reference_successor_kernel" in gate_source

    # The gate names its scanned roots in its own body; a narrowing that dropped
    # a directory a Process-V2 module lives in would leave that module
    # unenforced while the gate still passed.
    covered = {
        str(path.relative_to(_REPO_ROOT)).split("/", 1)[0] for path in _process_v2_modules()
    }
    assert covered
    for root in sorted(covered):
        assert f'"{root}"' in gate_source, root


def test_no_production_module_has_an_import_edge_to_the_reference_oracle() -> None:
    """Without this the only enforcement of a real boundary is a substring scan.

    The rule that matters is that no reported number can come from the test
    oracle, and that is a property of the import graph.  A substring scan is
    both too strict, as this commit demonstrates, and too weak, because it
    cannot tell an ``import`` from a string.  This check is typed: it parses
    every production module and looks for an actual edge.

    It is also the record of the current false positive.  The one production
    file that mentions the oracle mentions it exactly once, as a path in the
    hashed implementation-file list that defines the V1 process identity, and
    has no edge.  If that mention ever becomes a real import, this fails.
    """

    oracle_module = "compose_v4.experiments.reference_successor_kernel"
    scanned = 0
    mentions: dict[str, int] = {}
    edges_found: list[str] = []
    for root in _SCANNED_ROOTS:
        directory = _REPO_ROOT / root
        if not directory.is_dir():
            continue
        for path in directory.rglob("*.py"):
            if path.name == "reference_successor_kernel.py":
                continue
            scanned += 1
            text = path.read_text(errors="replace")
            if "reference_successor_kernel" not in text:
                continue
            relative = str(path.relative_to(_REPO_ROOT))
            mentions[relative] = text.count("reference_successor_kernel")
            for module, symbol in _imported_symbols(path):
                if module.startswith(oracle_module):
                    edges_found.append(f"{relative} imports {module}.{symbol}")

    assert scanned > 100, f"the production scan visited only {scanned} modules"
    assert not edges_found, (
        "the reference aggregator is a test oracle; a production import edge would let a "
        f"reported number come from it: {sorted(edges_found)}"
    )
    # Measured: the one textual mention, and the reason the owning substring gate
    # is red at this commit while the real boundary holds.
    assert mentions == {"src/compose_v4/rewrite/editing_v2_process_identity.py": 1}, (
        "the set of production files that textually mention the test oracle has moved; "
        "recheck whether tests/test_reference_successor_kernel.py is now green and whether "
        f"this module's reason for not invoking it still holds. Observed: {mentions}"
    )


def test_no_process_v2_module_mentions_the_reference_oracle_at_all() -> None:
    """Without this the Process-V2 chain could acquire the oracle unnoticed.

    The Process-V2 module set is new and has no legitimate reason to name the
    oracle in any form, so it is held to the strict rule rather than to the
    typed one.  Stated separately from the production-wide check so a future
    false positive elsewhere cannot mask a real one here.
    """

    offenders = [
        str(path.relative_to(_REPO_ROOT))
        for path in _process_v2_modules()
        if "reference_successor_kernel" in path.read_text(errors="replace")
    ]
    assert not offenders, offenders


# ---- 2. No Process-V2 module imports a live-V1 loader ----


def test_the_declared_v1_loader_really_does_revalidate_the_live_v1_identity() -> None:
    """Without this the boundary below would be a naming rule, not a finding.

    Evidence, so the expected-failure test has a checkable premise: the declared
    loader reads the *live* Editing-V2 process identity and requires the
    completion it is validating to equal it.  A payload sealed under a
    superseded identity is therefore refused by construction, no matter how
    intact it is, which is precisely why a Process-V2 module must not route the
    historical payload through it.

    This is a source-level assertion rather than a behavioural one: making the
    loader refuse requires a complete V1 migration completion, which belongs to
    the V1 loader's own tests and is not this workstream's to build.
    """

    for module, symbol in V1_LOADERS_THAT_REVALIDATE_LIVE_V1_IDENTITY:
        assert module.startswith("compose_v4.data.")
        assert symbol == "resolve_editing_v2_semantic_active8_sources"

    source = inspect.getsource(resolve_editing_v2_semantic_active8_sources)
    assert "live_process_identity = editing_v2_process_identity()" in source
    assert 'live_process_identity["process_identity_sha256"]' in source


@pytest.mark.xfail(
    strict=True,
    reason=(
        "runnable-chain specification Section 5, known real defect: "
        "src/compose_v4/data/editing_v2_process_v2_active8_source.py imports the V1 loader "
        "resolve_editing_v2_semantic_active8_sources, which revalidates the LIVE V1 process "
        "identity and therefore rejects the exact historical payload by construction. The "
        "module is a provenance-and-census join that does not compute Process-V2 candidate "
        "decisions and is to be replaced by a distinct Process-V2 Active8 stage in Wave 2. "
        "The boundary is recorded now and becomes a real gate the moment that replacement "
        "lands: strict xfail turns the fix into a reported failure, which is the signal to "
        "delete this marker."
    ),
)
def test_no_process_v2_module_imports_a_v1_loader_that_revalidates_live_v1_identity() -> None:
    """Without this the boundary is invisible until a real payload is refused.

    A Process-V2 module that routes the historical-pinned payload through a
    live-V1 loader passes every unit test whose resolvers are monkeypatched, and
    fails only against the real artifact, at the point where a remote stage is
    already running.
    """

    offenders: list[str] = []
    for path in _process_v2_modules():
        edges = _imported_symbols(path)
        for module, symbol in V1_LOADERS_THAT_REVALIDATE_LIVE_V1_IDENTITY:
            if (module, symbol) in edges or (module, "*") in edges or (module, "") in edges:
                offenders.append(f"{path.relative_to(_REPO_ROOT)} imports {module}.{symbol}")
    assert not offenders, (
        "a Process-V2 module must not import a V1 loader that revalidates V1 schema or live "
        f"identity; found: {sorted(offenders)}"
    )


# ---- 3. The frozen schema module stays a leaf ----


def test_the_frozen_schema_module_imports_nothing_from_experiments_and_stays_a_leaf() -> None:
    """Without this an edge added to the shared schema is acquired by every layer.

    Every Process-V2 layer imports this module, so an import it takes on is an
    import all of them take on.  An edge into ``experiments`` would pull the
    test oracle's package into the chain that publishes evidence; an edge into
    any ``compose_v4`` package at all is what would let such an edge appear
    transitively without this direct check noticing.  It is a leaf today, and
    the leaf property is what keeps the ``experiments`` check sufficient.
    """

    schema_path = _REPO_ROOT / FROZEN_SCHEMA_RELATIVE_PATH
    edges = _imported_symbols(schema_path)
    assert edges, "the frozen schema module imports nothing at all, so the scan found no file"

    from_experiments = sorted(
        f"{module}.{symbol}" if symbol else module
        for module, symbol in edges
        if module.startswith("compose_v4.experiments")
    )
    assert not from_experiments, from_experiments

    from_compose = sorted(
        f"{module}.{symbol}" if symbol else module
        for module, symbol in edges
        if module.split(".", 1)[0] == "compose_v4"
    )
    assert not from_compose, (
        "the frozen shared schema must stay a leaf; every Process-V2 layer imports it, so "
        f"these edges would become every layer's edges: {from_compose}"
    )
