from dataclasses import asdict

import numpy as np
import pytest

from compose_v4.chem.molecular_graph import MolecularGraph, smiles_to_molecular_graph
from compose_v4.chem.state import pad_molecular_graph
from compose_v4.control.donor_program import PendantCut, pendant_cuts
from compose_v4.control.edit_chooser import EditChooser, EditFeatures, edit_kernel, fit_chooser
from compose_v4.experiments.pmo_edit_chooser import assign_splits, ranking_metrics, reconcile
from compose_v4.rewrite.trace_shard import encode_state


def graph(smiles):
    return pad_molecular_graph(smiles_to_molecular_graph(smiles), 48)


def row(donor="CC", score=0.8):
    source, other = graph("CCO"), graph(donor)
    a = next(c for c in pendant_cuts(source) if c.component == (2,))
    b = next(c for c in pendant_cuts(other) if c.component == (1,))
    return {
        "row_id": donor,
        "parent_smiles": "CCO",
        "parent_score": 0.4,
        "smiles": "CCC" if donor == "CC" else "CCN",
        "score": score,
        "source": encode_state(source),
        "donor": encode_state(other),
        "source_cut": asdict(a),
        "donor_cut": asdict(b),
        "status": "compiled",
        "origin": "fixture",
        "wave": "earlier",
    }


def test_model_learns_beneficial_and_damaging_complete_edits_and_keeps_support():
    rows = [
        row(),
        row("CN", 0.1),
        {**row(), "status": "search_unresolved", "score": None, "row_id": "failure"},
    ]
    cache = EditFeatures()
    features = [cache(r) for r in rows]
    model = EditChooser(fit_chooser(rows, features, input_identity="fixture"))
    values, completion = model.predict(features, batch_size=1)
    batched_values, batched_completion = model.predict(features, batch_size=32)
    np.testing.assert_allclose(values, batched_values, rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(completion, batched_completion, rtol=1e-12, atol=1e-14)
    assert values[0] > values[1]
    assert model.payload["training_counts"]["scored"] == 2
    assert model.payload["training_counts"]["improved"] == 1
    assert np.all((0 <= completion) & (completion <= 1))
    base = np.array([0.7, 0.2, 0.1])
    distribution = model.distribution(values, base)
    assert distribution.sum() == pytest.approx(1)
    assert (distribution >= 0.2 * base).all()
    with pytest.raises(ValueError, match="fabricated"):
        fit_chooser([{**rows[2], "score": 0.0}], [features[2]], input_identity="bad")
    with pytest.raises(ValueError, match="strictly positive"):
        model.distribution(values, [1, 0, 0])


def test_features_and_kernel_ignore_persistent_slot_permutation():
    r = row()
    source = graph("CCO")
    order = np.random.default_rng(13).permutation(48)
    inverse = np.argsort(order)
    moved = MolecularGraph(
        source.atom_types[order],
        source.formal_charges[order],
        source.implicit_h_counts[order],
        source.bonds[np.ix_(order, order)],
    )
    a = r["source_cut"]
    cut = PendantCut(
        int(inverse[a["anchor"]]),
        int(inverse[a["root"]]),
        tuple(sorted(int(inverse[i]) for i in a["component"])),
    )
    changed = {**r, "source": encode_state(moved), "source_cut": asdict(cut)}
    cache = EditFeatures()
    assert cache(r) == cache(changed)
    features = [cache(r), cache(row("CN", 0.1))]
    matrix = edit_kernel(features, features)
    np.testing.assert_allclose(matrix, matrix.T)
    np.testing.assert_allclose(np.diag(matrix), 1, atol=1e-14)
    assert np.linalg.eigvalsh(matrix).min() >= -1e-12


def test_dedup_preserves_negative_outcomes_and_conflicts_fail():
    r = row(score=0.1)
    rows = reconcile([r, {**r, "origin": "duplicate"}])
    assert len(rows) == 1 and len(rows[0]["origins"]) == 2
    assert rows[0]["score"] == 0.1
    with pytest.raises(ValueError, match="conflicting"):
        reconcile([r, {**r, "score": 0.5}])


def test_group_split_removes_cross_role_molecule_labels():
    rows = [
        dict(row(), parent_smiles=f"parent-{i}", smiles=f"product-{i}", row_id=str(i))
        for i in range(30)
    ]
    split = assign_splits(rows)
    cal = next(r for r in split if r["split"] == "calibration")
    train = next(r for r in split if r["split"] == "train")
    train["smiles"] = cal["smiles"]
    split = assign_splits(rows)
    assert train["split"] == "excluded"
    labeled = lambda role: {
        s for r in split if r["split"] == role for s in (r["parent_smiles"], r["smiles"])
    }
    assert not labeled("train") & labeled("calibration")


def test_ranking_reports_precision_coverage_and_singletons_separately():
    rows = [row(), row("CN", 0.1), {**row(), "parent_smiles": "different"}]
    metrics = ranking_metrics(rows, [1, 0, 100])
    assert metrics["eligible_pools"] == 1
    assert metrics["singleton_pools"] == 1
    assert metrics["improvement_precision"] == 1
    assert metrics["improver_parent_coverage"] == 1
    assert metrics["selected_gain"] == pytest.approx(0.4)
    assert metrics["best_available_gain"] == pytest.approx(0.4)
