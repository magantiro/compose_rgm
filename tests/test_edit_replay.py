from dataclasses import asdict

import numpy as np
import pytest

from compose_v4.chem.molecular_graph import MolecularGraph, smiles_to_molecular_graph
from compose_v4.chem.state import pad_molecular_graph
from compose_v4.control.donor_program import PendantCut, compile_transplant, pendant_cuts
from compose_v4.control.edit_replay import EditReplay, cut_features, fit_replay
from compose_v4.rewrite.trace_shard import encode_state


def graph(smiles):
    return pad_molecular_graph(smiles_to_molecular_graph(smiles), 48)


def observation():
    source, donor = graph("CCO"), graph("CC")
    left = next(c for c in pendant_cuts(source) if c.component == (2,))
    right = next(c for c in pendant_cuts(donor) if c.component == (1,))
    return {
        "parent_smiles": "CCO",
        "parent_score": 0.4,
        "smiles": "CCC",
        "score": 0.8,
        "source": encode_state(source),
        "donor": encode_state(donor),
        "donor_index": 0,
        "source_cut": asdict(left),
        "donor_cut": asdict(right),
        "origin": "fixture",
    }


def test_replay_transfers_edit_to_a_new_molecule_without_copying_endpoint():
    policy = EditReplay(fit_replay([observation()]))
    new = graph("CCCO")
    cuts, probabilities = policy.distribution(new)
    assert probabilities.sum() == pytest.approx(1)
    assert (probabilities > 0).all()
    selected = cuts[int(probabilities[:, 0].argmax())]
    assert selected.component == (3,)
    result = compile_transplant(new, graph("CC"), selected, PendantCut(0, 1, (1,)))
    assert result["status"] == "compiled"
    assert result["smiles"] == "CCCC"
    assert result["smiles"] != observation()["smiles"]


def test_features_are_slot_permutation_invariant():
    source = graph("CCO")
    cut = next(c for c in pendant_cuts(source) if c.component == (2,))
    order = np.random.default_rng(31).permutation(48)
    inverse = np.argsort(order)
    changed = MolecularGraph(
        source.atom_types[order],
        source.formal_charges[order],
        source.implicit_h_counts[order],
        source.bonds[np.ix_(order, order)],
    )
    moved = PendantCut(
        int(inverse[cut.anchor]),
        int(inverse[cut.root]),
        tuple(sorted(map(int, inverse[list(cut.component)]))),
    )
    assert cut_features(source, cut) == cut_features(changed, moved)


def test_duplicate_experience_does_not_multiply_policy_mass_and_bad_labels_fail():
    row = observation()
    bank = fit_replay([row, {**row, "origin": "second_receipt"}])
    assert bank["unique_edges"] == 1
    assert bank["weights"] == [1]
    assert bank["entries"][0]["origins"] == ["fixture", "second_receipt"]
    with pytest.raises(ValueError, match="conflicting"):
        fit_replay([row, {**row, "score": 0.7}])
