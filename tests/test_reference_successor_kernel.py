"""The independent aggregation oracle, driven through the REAL executor on real molecules.

Two things are checked here:

  1. the oracle groups aliases correctly and normalizes over productive mass only -- verified on states
     built by the production chemistry stack, not on synthetic graphs, because alias structure is a
     property of real molecular symmetry;
  2. the oracle stays an oracle: a scan test fails if any non-test module imports it, which is what keeps
     it from silently becoming the source of a reported number.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from compose_v4.chem.molecular_graph import MolecularGraph, smiles_to_molecular_graph
from compose_v4.experiments.reference_successor_kernel import (
    compare_against_reference,
    reference_successor_batch,
    reference_successor_probabilities,
)
from compose_v4.experiments.successor_kernel import validate_successor_batch
from compose_v4.rewrite.kernel import canonical_state_key

_ROOT = Path(__file__).resolve().parent.parent
_ORACLE_MODULE = "compose_v4.experiments.reference_successor_kernel"
_ORACLE_PATH = Path("src/compose_v4/experiments/reference_successor_kernel.py")


def _module_name_for_path(path: Path) -> str:
    """Return the import name a repository Python file would have when imported."""
    relative = path.relative_to(_ROOT)
    parts = list(relative.with_suffix("").parts)
    if parts and parts[0] == "src":
        parts.pop(0)
    if parts and parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def _resolve_import_from(node: ast.ImportFrom, *, importer: str) -> str:
    """Resolve an ImportFrom base sufficiently to identify the frozen oracle."""
    if node.level == 0:
        return node.module or ""
    package = importer.rsplit(".", 1)[0] if "." in importer else ""
    package_parts = package.split(".") if package else []
    parents_to_drop = node.level - 1
    if parents_to_drop > len(package_parts):
        return ""
    prefix = package_parts[: len(package_parts) - parents_to_drop]
    suffix = node.module.split(".") if node.module else []
    return ".".join((*prefix, *suffix))


def _is_oracle_module(name: str) -> bool:
    return name == _ORACLE_MODULE or name.startswith(f"{_ORACLE_MODULE}.")


def _oracle_import_edges(source: str, *, importer: str, filename: str) -> tuple[str, ...]:
    """Find static imports of the test oracle, including literal dynamic imports."""
    try:
        tree = ast.parse(source, filename=filename)
    except SyntaxError as error:
        raise AssertionError(
            f"cannot verify the oracle-import boundary because {filename} does not parse: {error}"
        ) from error

    importlib_aliases = {"importlib"}
    import_module_aliases: set[str] = set()
    findings: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "importlib":
                    importlib_aliases.add(alias.asname or alias.name)
                if _is_oracle_module(alias.name):
                    findings.append(f"line {node.lineno}: import {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            base = _resolve_import_from(node, importer=importer)
            if base == "importlib":
                for alias in node.names:
                    if alias.name == "import_module":
                        import_module_aliases.add(alias.asname or alias.name)
            if _is_oracle_module(base):
                findings.append(f"line {node.lineno}: from {base} import ...")
                continue
            for alias in node.names:
                candidate = f"{base}.{alias.name}" if base else alias.name
                if _is_oracle_module(candidate):
                    findings.append(f"line {node.lineno}: from {base} import {alias.name}")
        elif isinstance(node, ast.Call) and node.args:
            is_dynamic_import = (
                isinstance(node.func, ast.Name)
                and node.func.id in {"__import__", *import_module_aliases}
            ) or (
                isinstance(node.func, ast.Attribute)
                and node.func.attr == "import_module"
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id in importlib_aliases
            )
            module_arg = node.args[0]
            if (
                is_dynamic_import
                and isinstance(module_arg, ast.Constant)
                and isinstance(module_arg.value, str)
                and _is_oracle_module(module_arg.value)
            ):
                findings.append(f"line {node.lineno}: literal dynamic import {module_arg.value}")
    return tuple(findings)


class _StubSystem:
    """Minimal executor stand-in: maps (rule, action) to an already-built successor state.

    The production executor is exercised in the integration test below; this stub isolates the GROUPING
    logic, which is the part the oracle reimplements and therefore the part under test here.
    """

    def __init__(self, table):
        self._table = table

    def apply(self, state, rule_name, action):
        return self._table[(rule_name, action)]


def _mol(smiles: str) -> MolecularGraph:
    return smiles_to_molecular_graph(smiles)


# ---- grouping and normalization ------------------------------------------------------------------------


def test_aliases_to_the_same_molecule_are_merged_and_mass_summed():
    """Three marks reaching one molecule must yield ONE successor holding the summed mass."""
    source, target = _mol("CCO"), _mol("CCC")
    system = _StubSystem({("r1", "a"): target, ("r2", "b"): target, ("r3", "c"): target})
    batch = reference_successor_batch(
        source, [("r1", "a", 0.2), ("r2", "b", 0.3), ("r3", "c", 0.5)], system=system
    )
    validate_successor_batch(batch)
    assert batch.support_size == 1
    assert batch.successors[0].alias_count == 3
    assert batch.successors[0].probability == pytest.approx(1.0)


def test_distinct_successors_split_mass_proportionally():
    source, a, b = _mol("CCO"), _mol("CCC"), _mol("CCN")
    system = _StubSystem({("r1", "a"): a, ("r2", "b"): b})
    probs = reference_successor_probabilities(
        source, [("r1", "a", 3.0), ("r2", "b", 1.0)], system=system
    )
    assert probs[canonical_state_key(a)] == pytest.approx(0.75)
    assert probs[canonical_state_key(b)] == pytest.approx(0.25)


def test_self_transition_becomes_virtual_mass_and_is_excluded_from_the_law():
    """A mark returning the same molecule advances the proposal clock but is not a molecular jump."""
    source, other = _mol("CCO"), _mol("CCC")
    system = _StubSystem({("self", "s"): _mol("CCO"), ("jump", "j"): other})
    batch = reference_successor_batch(
        source, [("self", "s", 0.6), ("jump", "j", 0.4)], system=system
    )
    validate_successor_batch(batch)
    assert batch.support_size == 1
    assert batch.virtual_mass == pytest.approx(0.6)
    # Conditioned on jumping: the surviving successor carries all the productive mass, NOT 0.4.
    assert batch.successors[0].probability == pytest.approx(1.0)


def test_declared_history_aware_wrapper_routes_immediate_return_to_virtual_mass():
    source, previous, forward = _mol("CCO"), _mol("CC"), _mol("CCC")
    system = _StubSystem({("back", "b"): previous, ("fwd", "f"): forward})
    batch = reference_successor_batch(
        source,
        [("back", "b", 0.7), ("fwd", "f", 0.3)],
        system=system,
        previous_state_key=canonical_state_key(previous),
    )
    assert batch.virtual_mass == pytest.approx(0.7)
    assert batch.keys == (canonical_state_key(forward),)


def test_state_only_base_kernel_keeps_a_possible_return_as_a_legal_successor():
    source, previous, forward = _mol("CCO"), _mol("CC"), _mol("CCC")
    system = _StubSystem({("back", "b"): previous, ("fwd", "f"): forward})
    batch = reference_successor_batch(
        source,
        [("back", "b", 0.7), ("fwd", "f", 0.3)],
        system=system,
    )
    assert batch.virtual_mass == pytest.approx(0.0)
    assert batch.probability_of(canonical_state_key(previous)) == pytest.approx(0.7)
    assert batch.probability_of(canonical_state_key(forward)) == pytest.approx(0.3)


def test_all_mass_virtual_yields_a_terminal_batch():
    source = _mol("CCO")
    system = _StubSystem({("self", "s"): _mol("CCO")})
    batch = reference_successor_batch(source, [("self", "s", 1.0)], system=system)
    assert batch.is_terminal and batch.virtual_mass == pytest.approx(1.0)
    validate_successor_batch(batch)


def test_zero_mass_marks_are_dropped_not_counted_as_aliases():
    source, target = _mol("CCO"), _mol("CCC")
    system = _StubSystem({("r1", "a"): target, ("r2", "b"): target})
    batch = reference_successor_batch(source, [("r1", "a", 1.0), ("r2", "b", 0.0)], system=system)
    assert batch.successors[0].alias_count == 1


def test_negative_mass_is_rejected():
    source = _mol("CCO")
    system = _StubSystem({("r1", "a"): _mol("CCC")})
    with pytest.raises(ValueError, match="finite and nonnegative"):
        reference_successor_batch(source, [("r1", "a", -0.5)], system=system)


# ---- the comparison helper ------------------------------------------------------------------------------


def test_comparison_reports_agreement_as_empty():
    assert compare_against_reference({"a": 0.5, "b": 0.5}, {"a": 0.5, "b": 0.5}) == []


def test_comparison_catches_a_wrong_grouping_not_just_a_wrong_total():
    """The failure this oracle exists for: totals match, grouping does not."""
    produced = {"a": 1.0}
    reference = {"a": 0.5, "b": 0.5}
    problems = compare_against_reference(produced, reference)
    assert any("missing successor" in p for p in problems)
    assert any("differs by" in p for p in problems)
    assert sum(produced.values()) == pytest.approx(sum(reference.values()))


def test_comparison_reports_extra_successors():
    problems = compare_against_reference({"a": 0.5, "z": 0.5}, {"a": 1.0})
    assert any("extra successor 'z'" in p for p in problems)


# ---- integration with the REAL production executor ------------------------------------------------------


def test_real_executor_grouping_on_a_symmetric_molecule():
    """Drive the actual production executor, so the oracle is exercised against real semantics.

    Benzene is chosen because its symmetry means several distinct marks land on the same molecule -- the
    alias structure that makes a mark-level law differ from a molecular one.
    """
    from compose_v4.rewrite.factorized_fiber import _factorized_candidates
    from compose_v4.rewrite.kernel import de_novo_rewrite_system

    state = _mol("c1ccccc1")
    system = de_novo_rewrite_system()
    candidates = _factorized_candidates(state, allow_bond_reroute=False)
    marks = [(name, action, 1.0) for name, action in list(candidates)[:12]]
    assert marks, "benzene must expose legal edits under the declared de-novo system"
    batch = reference_successor_batch(state, marks, system=system)
    validate_successor_batch(batch)
    # This fixture must actually exercise many-to-one mark aggregation.
    assert batch.support_size < len(marks)
    assert any(successor.alias_count > 1 for successor in batch.successors)
    assert sum(successor.alias_count for successor in batch.successors) == len(marks)


# ---- the oracle must stay an oracle ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "source,importer",
    (
        ("import compose_v4.experiments.reference_successor_kernel\n", "scripts.direct"),
        (
            "from compose_v4.experiments.reference_successor_kernel import reference_successor_batch\n",
            "scripts.from_direct",
        ),
        (
            "from compose_v4.experiments import reference_successor_kernel\n",
            "scripts.from_parent",
        ),
        ("from . import reference_successor_kernel\n", "compose_v4.experiments.relative"),
    ),
)
def test_oracle_boundary_scanner_detects_static_import_edges(source: str, importer: str):
    assert _oracle_import_edges(source, importer=importer, filename="attacker.py")


@pytest.mark.parametrize(
    "source",
    (
        "import importlib\nimportlib.import_module("
        + "'compose_v4.experiments.reference_successor_kernel')\n",
        "import importlib as il\nil.import_module("
        + "'compose_v4.experiments.reference_successor_kernel')\n",
        "from importlib import import_module as load\nload("
        + "'compose_v4.experiments.reference_successor_kernel')\n",
        "__import__('compose_v4.experiments.reference_successor_kernel')\n",
    ),
)
def test_oracle_boundary_scanner_detects_literal_dynamic_imports(source: str):
    assert _oracle_import_edges(source, importer="scripts.dynamic", filename="attacker.py")


def test_oracle_boundary_scanner_ignores_a_hashed_inventory_path():
    source = """
IMPLEMENTATION_FILES = (
    "src/compose_v4/experiments/reference_successor_kernel.py",
)
"""
    assert not _oracle_import_edges(
        source, importer="compose_v4.rewrite.identity", filename="safe.py"
    )


def test_oracle_boundary_scanner_fails_closed_on_unparseable_python():
    with pytest.raises(AssertionError, match="does not parse"):
        _oracle_import_edges("def broken(:\n", importer="scripts.broken", filename="broken.py")


def test_no_production_module_imports_the_oracle():
    """Enforcement, not substring matching: production must not import the test oracle."""
    offenders: list[str] = []
    for directory in ("src", "scripts", "modal_apps"):
        root = _ROOT / directory
        if not root.is_dir():
            continue
        for path in root.rglob("*.py"):
            relative = path.relative_to(_ROOT)
            if relative == _ORACLE_PATH:
                continue
            findings = _oracle_import_edges(
                path.read_text(errors="strict"),
                importer=_module_name_for_path(path),
                filename=str(relative),
            )
            offenders.extend(f"{relative}: {finding}" for finding in findings)
    assert not offenders, (
        "the reference aggregator is a TEST ORACLE and must not be imported by production or result-"
        f"producing code; found: {offenders}"
    )
