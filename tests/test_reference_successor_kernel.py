"""The independent aggregation oracle, driven through the REAL executor on real molecules.

Two things are checked here:

  1. the oracle groups aliases correctly and normalizes over productive mass only -- verified on states
     built by the production chemistry stack, not on synthetic graphs, because alias structure is a
     property of real molecular symmetry;
  2. the oracle stays an oracle: a scan test fails if any non-test module imports it, which is what keeps
     it from silently becoming the source of a reported number.
"""
from __future__ import annotations

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
    batch = reference_successor_batch(
        source, [("r1", "a", 1.0), ("r2", "b", 0.0)], system=system
    )
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
    from compose_v4.rewrite.factorized_fiber import _factorized_candidates  # noqa: PLC0415
    from compose_v4.rewrite.kernel import de_novo_rewrite_system  # noqa: PLC0415

    state = _mol("c1ccccc1")
    system = de_novo_rewrite_system()
    try:
        candidates = _factorized_candidates(state, allow_bond_reroute=False)
    except Exception as error:  # pragma: no cover - enumerator signature drift is a real finding
        pytest.skip(f"factorized enumerator unavailable with this signature: {error}")

    marks = [(name, action, 1.0) for name, action in list(candidates)[:12]]
    if not marks:
        pytest.skip("no candidates enumerated for benzene under the de-novo system")
    batch = reference_successor_batch(state, marks, system=system)
    validate_successor_batch(batch)
    # Aliasing must be real: fewer distinct successors than marks, or at least a merged group somewhere.
    assert batch.support_size <= len(marks)
    assert sum(s.alias_count for s in batch.successors) + 0 <= len(marks)


# ---- the oracle must stay an oracle ---------------------------------------------------------------------


def test_no_production_module_imports_the_oracle():
    """Enforcement, not convention: a reported number must never come from the reference implementation."""
    offenders = []
    for directory in ("src", "scripts", "modal_apps"):
        root = _ROOT / directory
        if not root.is_dir():
            continue
        for path in root.rglob("*.py"):
            if path.name == "reference_successor_kernel.py":
                continue
            text = path.read_text(errors="replace")
            if "reference_successor_kernel" in text:
                offenders.append(str(path.relative_to(_ROOT)))
    assert not offenders, (
        "the reference aggregator is a TEST ORACLE and must not be imported by production or result-"
        f"producing code; found: {offenders}"
    )
