"""The optional content law leaves the train-only joint mass model unchanged."""

import numpy as np

from compose_v4.benchmark.joint_mass_pendant_policy import JointMassPendantSampler


def test_uniform_content_is_recorded_without_changing_default_receipt():
    catalog = {
        "schema": "split_first_training_pendant_catalog_v1",
        "source_sha256": "fixture",
        "split": {"partition": "train"},
        "selection_uses_qed_sa": False,
        "benchmark_prompts_used_for_selection": False,
        "entries": [
            {
                "context": "C:0:0",
                "rooted_smiles": smiles,
                "heavy_atoms": 1,
                "ring_count": 0,
                "occurrences": 1,
                "source_rows": [index],
                "source_balanced_weight": weight,
            }
            for index, (smiles, weight) in enumerate((("[1*]C", 1.0), ("[1*]N", 9.0)))
        ],
    }
    prior = {
        "schema": "split_first_training_decoration_mass_prior_v1",
        "source_sha256": "fixture",
        "split": {"partition": "train"},
        "quality_labels_used": False,
        "benchmark_prompts_used_to_fit": False,
        "rows_with_murcko_decorations": 1,
        "counts": [[20, 1, 1, 1]],
    }
    frozen = JointMassPendantSampler(catalog, prior)
    broader = JointMassPendantSampler(catalog, prior, content_allocation="uniform_within_cell")
    assert frozen.mass_counts == broader.mass_counts
    _, old_receipt = frozen.sample(("C:0:0",), 20, np.random.default_rng(1))
    _, new_receipt = broader.sample(("C:0:0",), 20, np.random.default_rng(1))
    assert "content_allocation" not in old_receipt
    assert new_receipt["content_allocation"] == "uniform_within_cell"
    assert old_receipt["mass_probability"] == new_receipt["mass_probability"] == 1.0
