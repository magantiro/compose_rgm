"""Frozen support denominators, gate failures and completed-attempt resume."""

from copy import deepcopy

import numpy as np
import pytest
from run_fragment_training_linker_support import restore_completed_attempt, support_summary


def rows():
    rng = np.random.default_rng(0)
    return [
        {
            "drug": f"fixture_{i // 2}",
            "attempt_index": i % 2,
            "panel": {
                "rng_state_before": deepcopy(rng.bit_generator.state),
                "rng_state_after": deepcopy(rng.bit_generator.state),
                "output_count": 1,
                "selected_smiles": "CC",
                "offered_count": 8,
                "offered": [{"draw": draw, "status": "model_supported"} for draw in range(8)],
                "exact_compiled_count": 8,
                "model_supported_count": 8,
            },
            "selected_valid_connected": True,
            "selected_exact_core_path_fidelity": True,
        }
        for i in range(20)
    ]


def test_support_gate_requires_coverage_every_prompt_and_perfect_output_fidelity():
    complete = rows()
    assert support_summary(complete)["support_pass"]
    missing = deepcopy(complete)
    for index in (0, 2):
        missing[index]["panel"]["output_count"] = 0
        missing[index]["panel"]["selected_smiles"] = None
    summary = support_summary(missing)
    assert summary["outputs"] == 18
    assert summary["output_coverage"] == 0.9
    assert summary["offered_candidate_draws"] == 160
    assert summary["support_pass"]
    missing[3]["panel"]["output_count"] = 0
    assert not support_summary(missing)["support_pass"]
    absent_prompt = deepcopy(complete)
    for index in (0, 1):
        absent_prompt[index]["panel"]["output_count"] = 0
    assert not support_summary(absent_prompt)["checks"]["every_prompt_emits"]
    complete[0]["selected_exact_core_path_fidelity"] = False
    assert not support_summary(complete)["support_pass"]


def test_short_or_duplicated_support_census_is_not_a_result():
    with pytest.raises(ValueError, match="exactly two"):
        support_summary(rows()[:-1])
    repeated = rows()
    repeated[-1] = deepcopy(repeated[0])
    with pytest.raises(ValueError, match="exactly two"):
        support_summary(repeated)
    malformed = rows()
    malformed[0]["panel"]["offered"] = malformed[0]["panel"]["offered"][:-1]
    with pytest.raises(ValueError, match="eight recorded"):
        support_summary(malformed)


def test_completed_attempt_resume_checks_rng_identity_and_complete_draws():
    row = rows()[0]
    rng = np.random.default_rng(0)
    restore_completed_attempt(row, drug="fixture_0", attempt_index=0, rng=rng)
    with pytest.raises(ValueError, match="identity/RNG/draw"):
        restore_completed_attempt(row, drug="other", attempt_index=0, rng=rng)
    rng.random()
    with pytest.raises(ValueError, match="identity/RNG/draw"):
        restore_completed_attempt(row, drug="fixture_0", attempt_index=0, rng=rng)
