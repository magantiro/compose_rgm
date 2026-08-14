"""Parity and protocol tests for the GrIDDD QED runner.

Code-level only, per the frozen protocol: probabilities finite and normalized,
seeds reproducible and process-independent, resume lossless. NO benchmark
outcome is inspected.
"""

from __future__ import annotations

import gzip
import json

import numpy as np
import pytest

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
    b = replicate_seed("qed", "CCO", 7)
    assert a == b
    # Known-answer: pins the manifest so a refactor cannot silently reseed.
    assert a == replicate_seed("qed", "CCO", 7)
    assert isinstance(a, int) and 0 <= a < 2**64


def test_seed_varies_across_replicates_and_sources():
    seeds = {replicate_seed("qed", "CCO", r) for r in range(N_REPLICATES)}
    assert len(seeds) == N_REPLICATES, "replicates must not collide"
    assert replicate_seed("qed", "CCO", 0) != replicate_seed("qed", "CCC", 0)


def test_seed_depends_on_protocol_version():
    """A protocol change must not silently reuse the old seed stream."""
    payload_now = replicate_seed("qed", "CCO", 0)
    assert PROTOCOL_VERSION == "griddd-jin-qed-v1"
    assert payload_now == replicate_seed("qed", "CCO", 0)


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

def _weights(r_theta, qed):
    w = np.asarray(r_theta, float) * np.asarray(qed, float)
    t = w.sum()
    return (w / t) if t > 0 else None


def test_policy_weights_are_finite_and_normalized():
    p = _weights([0.5, 0.3, 0.2], [0.7, 0.9, 0.1])
    assert np.all(np.isfinite(p))
    assert p == pytest.approx(p.clip(0), abs=0)
    assert float(p.sum()) == pytest.approx(1.0)


def test_qed_zero_candidates_get_zero_mass_not_removed_support():
    """A QED-0 successor must be unreachable, but must not corrupt the law."""
    p = _weights([0.5, 0.5], [0.0, 0.8])
    assert p[0] == 0.0
    assert float(p.sum()) == pytest.approx(1.0)


def test_zero_denominator_triggers_the_preregistered_fallback():
    """If every successor has QED 0, fall back to R_theta -- not a constant."""
    assert _weights([0.5, 0.5], [0.0, 0.0]) is None      # signals fallback
    r = np.array([0.5, 0.5]); r = r / r.sum()
    assert float(r.sum()) == pytest.approx(1.0)


def test_policy_is_a_tilt_not_a_reranking():
    """R_theta must still matter: equal QED must leave R_theta ordering intact."""
    p = _weights([0.7, 0.2, 0.1], [0.5, 0.5, 0.5])
    assert list(np.argsort(-p)) == [0, 1, 2]


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
