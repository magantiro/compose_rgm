import copy

import numpy as np
import pytest

from compose_v4.experiments.pmo_chronological import (
    edit_kernels,
    evaluate_stream,
    kernel_prediction,
    state_features,
)


def test_fixed_chemical_kernels_are_psd_and_no_change_has_zero_difference():
    kernel, counts = state_features(["CCO", "CCN", "c1ccccc1"])
    rows = [
        {"parent_index": 0, "product_index": q, "option": "grow", "scale": "local"}
        for q in range(3)
    ]
    assert counts.shape == (3, 5)
    for matrix in [kernel, *edit_kernels(kernel, rows).values()]:
        assert np.allclose(matrix, matrix.T)
        assert np.linalg.eigvalsh(matrix).min() >= -1e-10
        assert np.isfinite(matrix).all()
    with pytest.raises(ValueError, match="disjoint"):
        kernel_prediction(kernel, [0.1, 0.2, 0.3], [0, 1], [1, 2])


def test_chronological_fit_cannot_see_later_labels():
    # Synthetic diagnostic fixture only, not molecular evidence or oracle labels.
    values = np.linspace(0.1, 0.8, 40)
    stream = {
        "case": {"arm": "fixture"},
        "scores": values.tolist(),
        "exclusions": [],
        "rows": [
            {
                "id": str(i),
                "query": i + 1,
                "parent_index": 0,
                "product_index": i,
                "parent_query": 1,
                "parent_score": values[0],
                "score": values[i],
                "option": "grow" if i % 2 else "restate",
                "scale": "local",
            }
            for i in range(4, 40)
        ],
    }
    gram = np.exp(-((values[:, None] - values[None, :]) ** 2))
    counts = np.zeros((40, 5))
    first = evaluate_stream(stream, gram, counts)
    changed = copy.deepcopy(stream)
    changed["scores"][30:] = [0.0] * 10
    for row in changed["rows"]:
        if row["query"] > 30:
            row["score"] = 0.0
    second = evaluate_stream(changed, gram, counts)
    assert [r for r in first["predictions"] if r["query"] <= 30] == [
        r for r in second["predictions"] if r["query"] <= 30
    ]
    assert all(r["fit_through_query"] < r["query"] for r in first["predictions"])
    assert [s for s in first["snapshots"] if s["cutoff"] <= 30] == [
        s for s in second["snapshots"] if s["cutoff"] <= 30
    ]
