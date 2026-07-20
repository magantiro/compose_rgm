"""Representation ablation on the LiON leave-library-out cross-chemistry transfer.

Asserts that `scripts/qualify_lion_representation_transfer.py` computed BOTH a
baseline (Morgan + RDKit descriptors) and a RICHER RDKit-descriptor representation
through the identical leave-library-out + Michael-included positive-control
protocol, and that the recorded numbers are finite and in-range.

Deliberately does NOT assert richer > baseline: the outcome may be a null result
(no lift), which is itself informative.  It only requires that both representations
are evaluated and the Michael leave-out delta is recorded.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
DIAG = REPO_ROOT / "diagnostics/lion_representation_transfer.json"

pytestmark = pytest.mark.skipif(not DIAG.exists(), reason="representation-transfer diagnostic not generated")


def _load() -> dict:
    return json.loads(DIAG.read_text())


def _in_corr_range(x) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(x) and -1.0 <= x <= 1.0


def test_both_representations_present() -> None:
    d = _load()
    reps = d["representations"]
    assert set(reps) == {"baseline_morgan_rdkit", "richer_rdkit_descriptors"}
    for rep in reps.values():
        assert "per_library" in rep and rep["per_library"]
        assert isinstance(rep["feature_dim"], int) and rep["feature_dim"] > 0


def test_richer_representation_is_actually_richer() -> None:
    d = _load()
    reps = d["representations"]
    base_dim = reps["baseline_morgan_rdkit"]["feature_dim"]
    rich_dim = reps["richer_rdkit_descriptors"]["feature_dim"]
    # richer must strictly add feature columns onto the baseline
    assert rich_dim > base_dim
    assert reps["richer_rdkit_descriptors"]["added_descriptors"], "no added descriptors recorded"


def test_michael_leaveout_present_and_in_range_for_both() -> None:
    d = _load()
    assert d["michael_addition_library"] is not None
    for rep in d["representations"].values():
        assert _in_corr_range(rep["michael_leaveout_spearman"])
        # leave-library-out per-library Spearmans must be finite correlations
        for lib in rep["per_library"].values():
            assert _in_corr_range(lib["spearman"])
            assert math.isfinite(lib["top_decile_enrichment"])
            assert lib["train_lipids"] >= 30 and lib["held_out_lipids"] >= 10


def test_positive_control_present_for_both() -> None:
    d = _load()
    for rep in d["representations"].values():
        pc = rep["positive_control"]
        assert _in_corr_range(pc["michael_subset_spearman"])
        assert _in_corr_range(pc["overall_spearman"])


def test_delta_recorded_and_consistent() -> None:
    d = _load()
    reps = d["representations"]
    b = reps["baseline_morgan_rdkit"]["michael_leaveout_spearman"]
    r = reps["richer_rdkit_descriptors"]["michael_leaveout_spearman"]
    delta = d["michael_leaveout_delta"]
    assert delta is not None and math.isfinite(delta)
    # delta must equal richer - baseline (to rounding)
    assert abs(delta - (r - b)) <= 1e-2
    assert d["representation_lifts_transfer"] in {
        "lifts", "lowers", "no meaningful lift (null result)",
        "indeterminate (Michael leave-out Spearman missing for a representation)",
    }
