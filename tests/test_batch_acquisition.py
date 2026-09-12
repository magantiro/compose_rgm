import numpy as np
import pytest

from compose_v4.control.batch_acquisition import probability_of_optimality, select_qpo_batch


def candidates():
    return [
        {
            "candidate_id": f"candidate-{i}",
            "canonical_smiles": "C" * (i + 1),
            "locked": True,
            "complete": True,
        }
        for i in range(3)
    ]


def test_joint_qpo_counts_and_unique_batch():
    draws = np.asarray([[3, 1, 0], [0, 3, 1], [0, 1, 3], [2, 2, 0]], dtype=float)
    assert probability_of_optimality(draws) == pytest.approx([0.375, 0.375, 0.25])
    result = select_qpo_batch(candidates(), draws, batch_size=2)
    assert result["selected_candidate_ids"] == ["candidate-1", "candidate-0"]
    assert len(set(result["selected_candidate_ids"])) == 2


def test_acquisition_rejects_unlocked_and_duplicate_molecules():
    rows = candidates()
    rows[0]["locked"] = False
    with pytest.raises(ValueError, match="locked complete"):
        select_qpo_batch(rows, np.ones((2, 3)), batch_size=1)
    rows = candidates()
    rows[1]["canonical_smiles"] = rows[0]["canonical_smiles"]
    with pytest.raises(ValueError, match="unique"):
        select_qpo_batch(rows, np.ones((2, 3)), batch_size=1)
