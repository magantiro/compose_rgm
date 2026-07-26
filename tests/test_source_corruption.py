"""The corrupted-molecule source prior builds valid, mark-family-only traces from a real molecule.

Native GM (source-prior swap): both the trim direction (real molecule as source) and the grow
direction (corrupted source) must replay through valid connected molecules to their target, use only
trainable mark families (never bond_insert/bond_delete, which the mark model rejects mid-training),
and cover substitution (atom_restate) beyond bare insertion.
"""

from __future__ import annotations

import numpy as np

from compose_v4.chem.molecular_graph import smiles_to_molecular_graph
from compose_v4.chem.state import (
    is_connected_or_null,
    is_valid_state,
    pad_molecular_graph,
)
from compose_v4.rewrite.kernel import canonical_state_key, de_novo_rewrite_system
from compose_v4.rewrite.source_corruption import FORBIDDEN_FAMILIES, make_edit_pair
from compose_v4.rewrite.trace import execute_trace

_LEADS = (
    "COc1cc(C(=O)Nc2ccccc2Oc2ccccc2)on1",
    "C=CCNC(=O)Nc1ccc(F)c(NC(=O)OC)c1",
    "COC(=O)c1ccc(CNc2ccn(C)n2)[nH]1",
    "COc1ccccc1N1CCN(c2ccc(=O)n(CC(=O)NC3CC3)n2)CC1",
)

_MARK_RULE_NAMES = {
    "atom_insert", "atom_delete", "atom_restate", "bond_reorder", "bond_reroute",
    "cycle_insert", "cycle_attach", "ring_system_grow", "ring_system_delete", "ring_system_restate",
}


def _pairs(seed_offset: int = 0):
    system = de_novo_rewrite_system()
    out = []
    for i, smi in enumerate(_LEADS):
        target = pad_molecular_graph(smiles_to_molecular_graph(smi), 40)
        trim, grow = make_edit_pair(
            target, 5, system=system, rng=np.random.default_rng(i + seed_offset)
        )
        out.append((trim, grow, system))
    return out


def test_both_directions_replay_valid_and_only_mark_families() -> None:
    made = 0
    for trim, grow, system in _pairs():
        for trace in (trim, grow):
            if trace is None:
                continue
            made += 1
            _final, states = execute_trace(trace.source, trace.steps, system=system, return_states=True)
            for state in states:
                assert is_valid_state(state) and is_connected_or_null(state)
            assert canonical_state_key(states[-1]) == canonical_state_key(trace.target)
            for step in trace.steps:
                assert step.rule_name in _MARK_RULE_NAMES        # trainable mark family
                assert step.rule_name not in FORBIDDEN_FAMILIES  # never bond_insert/bond_delete
    assert made >= 2


def test_trim_source_is_the_real_molecule() -> None:
    # The crucial property: the trim direction puts the REAL molecule as the source (progress 0), so
    # real leads are in-distribution and the model does not want to terminate at a complete molecule.
    system = de_novo_rewrite_system()
    target = pad_molecular_graph(smiles_to_molecular_graph(_LEADS[0]), 40)
    trim, _grow = make_edit_pair(target, 5, system=system, rng=np.random.default_rng(0))
    assert trim is not None
    assert canonical_state_key(trim.source) == canonical_state_key(target)


def test_operator_coverage_beyond_insertion() -> None:
    # Across leads the corruption should exercise substitution/bond-order, not only atom delete/insert,
    # or the model never learns bioisostere edits.
    families: set[str] = set()
    for trim, grow, _system in _pairs(seed_offset=7):
        for trace in (trim, grow):
            if trace is not None:
                families.update(s.rule_name for s in trace.steps)
    assert "atom_restate" in families or "bond_reorder" in families
