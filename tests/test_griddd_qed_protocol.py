"""Parity and protocol tests for the GrIDDD QED runner.

Code-level only, per the frozen protocol: probabilities finite and normalized,
seeds reproducible and process-independent, resume lossless. NO benchmark
outcome is inspected.
"""

from __future__ import annotations

import gzip
import json
import os
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import QED

from modal_apps import griddd_qed_app
from modal_apps.griddd_qed_app import (
    HORIZON,
    N_REPLICATES,
    PROTOCOL_VERSION,
    replicate_seed,
    resume_from_partial,
)

# --------------------------------------------------------------------------
# The frozen seed manifest
# --------------------------------------------------------------------------


def test_seed_is_deterministic_across_processes():
    """The whole point of sha256 over Python's process-salted hash()."""
    a = replicate_seed("qed", "CCO", 7)
    code = (
        "from modal_apps.griddd_qed_app import replicate_seed; "
        "print(replicate_seed('qed', 'CCO', 7))"
    )
    for salt in ("1", "37"):
        result = subprocess.run(
            [sys.executable, "-c", code],
            env={**os.environ, "PYTHONHASHSEED": salt},
            check=True,
            capture_output=True,
            text=True,
        )
        assert int(result.stdout.strip()) == a
    assert isinstance(a, int) and 0 <= a < 2**64


def test_seed_varies_across_replicates_and_sources():
    seeds = {replicate_seed("qed", "CCO", r) for r in range(N_REPLICATES)}
    assert len(seeds) == N_REPLICATES, "replicates must not collide"
    assert replicate_seed("qed", "CCO", 0) != replicate_seed("qed", "CCC", 0)


def test_seed_depends_on_protocol_version(monkeypatch):
    """A protocol change must not silently reuse the old seed stream."""
    payload_now = replicate_seed("qed", "CCO", 0)
    assert PROTOCOL_VERSION == "griddd-jin-qed-v1"
    monkeypatch.setattr(griddd_qed_app, "PROTOCOL_VERSION", "fixture-alternative-v1")
    assert payload_now != replicate_seed("qed", "CCO", 0)


def test_same_seed_gives_the_same_draw():
    """Identical seeds must reproduce identical sampling, for the ablation."""
    w = np.array([0.1, 0.5, 0.4])
    a = np.random.default_rng(replicate_seed("qed", "CCO", 3))
    b = np.random.default_rng(replicate_seed("qed", "CCO", 3))
    draws_a = [int(a.choice(3, p=w)) for _ in range(50)]
    draws_b = [int(b.choice(3, p=w)) for _ in range(50)]
    assert draws_a == draws_b


# --------------------------------------------------------------------------
# The policy weights
# --------------------------------------------------------------------------


@pytest.fixture
def sample_fixture(monkeypatch):
    """Exercise the real local sampler, mocking only its model and property inputs."""
    from compose_v4.experiments import production_successor_kernel

    keys = ("CCO", "CCN", "CCC")
    recorded = []
    original_rng = np.random.default_rng

    class RecordingRng:
        def __init__(self, seed):
            self.delegate = original_rng(seed)

        def choice(self, size, *, p):
            recorded.append(np.asarray(p).copy())
            return self.delegate.choice(size, p=p)

    def sample(reference, qualities):
        recorded.clear()
        successors = tuple(
            SimpleNamespace(key=key, probability=probability)
            for key, probability in zip(keys, reference, strict=True)
        )
        quality = dict(zip(keys, qualities, strict=True))
        monkeypatch.setattr(griddd_qed_app, "_runtime", lambda: {"model": object()})
        monkeypatch.setattr(
            production_successor_kernel,
            "canonical_successor_result",
            lambda *args: SimpleNamespace(batch=SimpleNamespace(successors=successors)),
        )
        monkeypatch.setattr(QED, "qed", lambda molecule: quality[Chem.MolToSmiles(molecule)])
        monkeypatch.setattr(np.random, "default_rng", RecordingRng)
        with pytest.warns(UserWarning, match="executing locally"):
            result = griddd_qed_app.run_source.local(
                {"index": 0, "smiles": keys[0], "replicates": 1}
            )
        return result, tuple(recorded)

    return sample


def test_policy_weights_are_finite_and_normalized(sample_fixture):
    result, draws = sample_fixture([0.5, 0.3, 0.2], [0.7, 0.9, 0.1])
    assert len(draws) == HORIZON
    assert result["n_candidates"] == 1
    p = draws[0]
    assert p == pytest.approx([0.35 / 0.64, 0.27 / 0.64, 0.02 / 0.64])
    assert np.all(np.isfinite(p))
    assert p == pytest.approx(p.clip(0), abs=0)
    assert float(p.sum()) == pytest.approx(1.0)
    assert all(np.array_equal(draw, p) for draw in draws)


def test_qed_zero_candidates_get_zero_mass_not_removed_support(sample_fixture):
    """A QED-0 successor must be unreachable, but must not corrupt the law."""
    result, draws = sample_fixture([0.5, 0.3, 0.2], [0.0, 0.8, 0.0])
    assert result["candidates"] == ["CCN"]
    assert len(draws) == HORIZON
    assert all(np.array_equal(p, [0.0, 1.0, 0.0]) for p in draws)


def test_zero_denominator_triggers_the_preregistered_fallback(sample_fixture):
    """If every successor has QED 0, fall back to R_theta -- not a constant."""
    result, draws = sample_fixture([0.5, 0.3, 0.2], [0.0, 0.0, 0.0])
    assert result["zero_denominator_events"] == HORIZON
    assert len(draws) == HORIZON
    assert all(np.array_equal(p, [0.5, 0.3, 0.2]) for p in draws)


def test_zero_reference_mass_stops_without_an_unrecorded_draw(sample_fixture):
    result, draws = sample_fixture([0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
    assert not draws
    assert result["zero_denominator_events"] == 1
    assert result["per_replicate"][0]["dead_end"] == "zero_reference_mass"
    assert result["n_candidates"] == 1


def test_policy_is_a_tilt_not_a_reranking(sample_fixture):
    """R_theta must still matter: equal QED must leave R_theta ordering intact."""
    result, draws = sample_fixture([0.7, 0.2, 0.1], [0.5, 0.5, 0.5])
    assert result["zero_denominator_events"] == 0
    assert len(draws) == HORIZON
    assert all(p == pytest.approx([0.7, 0.2, 0.1], abs=1e-12) for p in draws)


# --------------------------------------------------------------------------
# Preemption safety
# --------------------------------------------------------------------------


def _write(path, results):
    path.write_bytes(gzip.compress(json.dumps({"results": results}).encode()))


def test_resume_is_lossless(tmp_path):
    tasks = [{"index": i, "smiles": "C"} for i in range(800)]
    p = tmp_path / "full.partial.json.gz"
    _write(p, [{"index": i, "status": "OK"} for i in range(317)])
    done, pending = resume_from_partial(tasks, p)
    assert len(done) == 317 and len(pending) == 483
    assert {r["index"] for r in done} | {t["index"] for t in pending} == set(range(800))


def test_torn_partial_degrades_to_a_full_run(tmp_path):
    p = tmp_path / "torn.partial.json.gz"
    good = gzip.compress(json.dumps({"results": [{"index": 0}]}).encode())
    p.write_bytes(good[: len(good) // 2])
    done, pending = resume_from_partial([{"index": i} for i in range(50)], p)
    assert done == [] and len(pending) == 50


def test_horizon_and_replicates_match_the_frozen_protocol():
    assert HORIZON == 6
    assert N_REPLICATES == 20
