from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest


_MODULE_PATH = Path(__file__).resolve().parents[1] / "modal_apps" / "audit_editing_corpus.py"
_SPEC = importlib.util.spec_from_file_location("audit_editing_corpus", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def test_integrated_progress_probabilities_are_normalized() -> None:
    for path_length in range(17):
        probabilities = _MODULE.integrated_progress_probabilities(path_length)
        assert probabilities.shape == (path_length + 1,)
        assert np.all(probabilities >= 0.0)
        assert float(probabilities.sum()) == pytest.approx(1.0, abs=1e-12)


def test_integrated_binomial_mean_matches_time_law() -> None:
    expected_normalized_t = 0.5
    expected_operational_t = 1.0 - (
        1.0 - np.exp(-_MODULE.OPERATIONAL_HORIZON)
    ) / _MODULE.OPERATIONAL_HORIZON
    expected_t = (
        (1.0 - _MODULE.LATE_TIME_FRACTION) * expected_normalized_t
        + _MODULE.LATE_TIME_FRACTION * expected_operational_t
    )
    path_length = 16
    probabilities = _MODULE.integrated_progress_probabilities(path_length)
    observed_mean = float(
        np.dot(np.arange(path_length + 1), probabilities)
    )
    assert observed_mean == pytest.approx(path_length * expected_t, abs=1e-11)


def test_teacher_rate_coefficients_respect_sequence_order() -> None:
    forward, _, _ = _MODULE.expected_mark_coefficients(
        ("atom_delete", "atom_insert")
    )
    reverse, _, _ = _MODULE.expected_mark_coefficients(
        ("atom_insert", "atom_delete")
    )
    assert forward["atom_delete"] > forward["atom_insert"]
    assert reverse["atom_insert"] > reverse["atom_delete"]
    assert sum(forward.values()) == pytest.approx(sum(reverse.values()))


def test_trace_key_matches_frozen_multi_neighbor_exclusion() -> None:
    trace = {
        "metadata": {},
        "trace_id": "record-level-id-must-not-override-decoded-trace-key",
        "steps": [
            {
                "action": {
                    "executor_rule": "atom_insert",
                    "payload_type": "AtomInsert",
                    "payload": {"neighbors": [[5, 1], [16, 1]]},
                }
            }
        ],
    }
    expected = hashlib_sha(
        ("atom_insert", "((5, 1), (16, 1))")
    )
    assert _MODULE._trace_key_from_wire(trace) == expected


def hashlib_sha(parts: tuple[str, ...]) -> str:
    import hashlib

    digest = hashlib.sha256()
    for part in parts:
        digest.update(part.encode())
    return "sha:" + digest.hexdigest()[:24]
