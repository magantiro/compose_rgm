from __future__ import annotations

import numpy as np

from compose_v4.chem.molecular_graph import MolecularGraph
from compose_v4.experiments.pmo_legal_action_policy import (
    RankerConfig,
    action_features,
    enumerate_rule_successors,
    fit_diagonal_contrastive_ranker,
    rank_of_teacher,
    rank_successor_keys,
    score_features,
)
from compose_v4.rewrite.kernel import (
    canonical_state_key,
    editing_v2_semantic_rewrite_system,
)
from compose_v4.rewrite.operators import AtomInsert


def _methane() -> MolecularGraph:
    atom_types = np.zeros(48, dtype=np.int32)
    charges = np.zeros(48, dtype=np.int32)
    hydrogens = np.zeros(48, dtype=np.int32)
    bonds = np.zeros((48, 48), dtype=np.int32)
    atom_types[0] = 2
    hydrogens[0] = 4
    return MolecularGraph(atom_types, charges, hydrogens, bonds)


def test_atom_insert_fiber_contains_exact_ethane_successor():
    source = _methane()
    system = editing_v2_semantic_rewrite_system()
    teacher = system.apply(source, "atom_insert", AtomInsert(1, 2, 0, 3, ((0, 1),)))
    candidates = enumerate_rule_successors(source, "atom_insert")
    teacher_key = canonical_state_key(teacher)
    assert rank_of_teacher(candidates, teacher_key) is not None
    assert len({candidate.successor_key for candidate in candidates}) == len(candidates)


def test_action_features_and_ranker_are_finite():
    source = _methane()
    candidates = enumerate_rule_successors(source, "atom_insert")
    features = [action_features(source, candidate) for candidate in candidates[:3]]
    differences = [(features[0] - feature, 1.0) for feature in features[1:]]
    checkpoint = fit_diagonal_contrastive_ranker(differences, RankerConfig())
    scores = [score_features(checkpoint, feature) for feature in features]
    assert np.isfinite(scores).all()
    assert scores[0] > max(scores[1:])


def test_compact_successor_rank_matches_candidate_rank():
    source = _methane()
    candidates = enumerate_rule_successors(source, "atom_insert")
    scores = {
        candidate.successor_key: float(index)
        for index, candidate in enumerate(candidates)
    }
    teacher_key = candidates[-1].successor_key
    assert rank_successor_keys(
        tuple(candidate.successor_key for candidate in candidates),
        teacher_key,
        scores,
    ) == rank_of_teacher(candidates, teacher_key, scores)
