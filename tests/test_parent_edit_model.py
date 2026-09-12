"""Completed-program learning/task boundaries on explicitly synthetic labels."""

import copy

import numpy as np
import pytest

from compose_v4.control.parent_edit_model import ParentEditModel, select_parent_edits
from compose_v4.control.program_task import (
    ProgramTask,
    archive_top_k,
    initialization_lock,
    predicted_archive_gains,
)
from compose_v4.rewrite.trace_shard import encode_state
from tests.test_edit_program import graph


def rows():
    return [
        {
            "features": [float(i), 1.0],
            "utility": float(i),
            "endpoint": f"fixture:{i}",
            "receipt_id": f"fixture:receipt:{i}",
            "oracle_protocol": "fixture",
        }
        for i in range(4)
    ]


def test_model_is_domain_bound_reproducible_and_uses_finite_observations_only():
    one = ParentEditModel.fit(rows(), oracle_protocol="fixture", input_sha256="a" * 64)
    two = ParentEditModel.fit(rows(), oracle_protocol="fixture", input_sha256="a" * 64)
    assert one.payload == two.payload
    mean, deviation = one.predict([[0, 1], [3, 1]], oracle_protocol="fixture")
    assert mean[1] > mean[0] and np.all(deviation >= 0)
    with pytest.raises(ValueError, match="domains"):
        one.predict([[0, 1]], oracle_protocol="other-target")
    malformed = rows()
    malformed[0]["utility"] = float("nan")
    with pytest.raises(ValueError, match="measured outcomes"):
        ParentEditModel.fit(malformed, oracle_protocol="fixture", input_sha256="a" * 64)
    damaged = copy.deepcopy(one.payload)
    damaged["members"][0]["mean"] += 1
    with pytest.raises(ValueError, match="identity"):
        ParentEditModel(damaged)


def test_duplicate_program_representations_do_not_multiply_endpoint_mass():
    training = rows()
    a = ParentEditModel.fit(training, oracle_protocol="fixture", input_sha256="a" * 64)
    b = ParentEditModel.fit(
        training + [training[-1]], oracle_protocol="fixture", input_sha256="b" * 64
    )
    probes = [r["features"] for r in training]
    assert np.allclose(
        a.predict(probes, oracle_protocol="fixture")[0],
        b.predict(probes, oracle_protocol="fixture")[0],
    )


def test_pmo_does_not_inherit_t4_gates_and_initialization_has_no_free_labels():
    pmo = ProgramTask("fixture_pmo", "fixture", "pmo")
    t4 = ProgramTask("fixture_t4", "fixture", "t4", "CCCCCC", 0.4)
    assert pmo.utility(0.3) == 0.3 and t4.utility(-9) == 9
    assert pmo.top_k == 10 and t4.top_k == 1
    assert pmo.endpoint_evaluator()({"smiles": "CC"})["oracle_eligible"]
    assert not t4.endpoint_evaluator()({"smiles": "CC"})["oracle_eligible"]
    with pytest.raises(ValueError, match="inherit"):
        ProgramTask("pmo", "fixture", "pmo", "CC", 0.4)
    records = [{"state": encode_state(graph(s)), "source_id": s} for s in ("CC", "CCC", "CCO")]
    locked = initialization_lock(records, count=2, seed=3, source_sha256="a" * 64)
    assert locked == initialization_lock(
        list(reversed(records)), count=2, seed=3, source_sha256="a" * 64
    )
    assert locked["new_oracle_calls"] == 0 and locked["count"] == 2
    with pytest.raises(ValueError, match="never task labels"):
        initialization_lock([{**records[0], "score": 1}], count=1, seed=3, source_sha256="a" * 64)


def test_best_k_accounts_for_unique_endpoints_and_acquisition_is_not_parent_gain():
    assert archive_top_k([("a", 1), ("a", 3), ("b", 1)], k=2) == 1.5
    assert archive_top_k([], k=10) is None
    assert np.allclose(predicted_archive_gains([8, 12], [10], k=1), [0, 2])
    ids = [str(i) for i in range(8)]
    with pytest.raises(ValueError, match="unqualified"):
        select_parent_edits(ids, range(8), [4], k=1, count=4, seed=1, mode="learned")
    chosen = select_parent_edits(
        ids, range(8), [4], k=1, count=4, seed=1, mode="learned", diagnostic=True
    )
    assert len(chosen["audit_indices"]) == 1
    assert "7" in chosen["selected_ids"] and not chosen["uncertainty_used"]


def test_real_mutation_features_separate_constructor_contraction_from_parent_growth():
    import json
    from pathlib import Path

    from compose_v4.control.parent_edit_model import ParentEditFeatures

    path = Path("diagnostics/t4_second_generation/attempt_2/braf_1_score_blind_batch.json")
    if not path.exists():
        pytest.skip("documented BRAF development pool absent")
    pool = json.loads(path.read_text())
    archive = json.loads(
        Path("diagnostics/t4_program_curriculum/attempt_1/braf_1_archive.json").read_text()
    )["optimizer"]
    candidate = next(
        c
        for c in pool["candidates"]
        if c["provenance"]["program_size"]["delta_from_measured_parent"] == 1
    )
    parent = archive["entries"][candidate["provenance"]["entry_id"]]
    features = ParentEditFeatures()
    base = features(candidate, parent_state=parent["trace"]["states"][-1])
    extended = features.with_mutation_context(
        candidate, parent_state=parent["trace"]["states"][-1], parent_record=parent
    )
    assert base[-16:][4] < 0  # Original construction shrinks the benchmark seed.
    assert extended[len(base)] == pytest.approx(1 / 40)  # Actual selected parent grows.
    assert extended[-2] == 0 and extended[-1] == 0  # No fabricated executed parent-to-child trace.
