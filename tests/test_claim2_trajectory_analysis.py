"""The Claim-2 shard -> frontier pipeline, exercised end to end without Modal.

A synthetic shard with real molecules proves the whole analysis path works --
per-trajectory metrics, per-source averaging, the frontier, the paired
bootstrap, the instrument checks and the power floor -- before any
container-hour is spent producing a real one.

The instrument checks get the most attention here, because a run whose arms
never diverged, or whose kernel disagreed with the production one, must be
refused rather than reported.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

_SPEC = importlib.util.spec_from_file_location(
    "analyse_claim2_trajectories", REPO / "scripts" / "analyse_claim2_trajectories.py"
)
analysis = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(analysis)

from compose_v4.experiments.claim2_trajectory_metrics import (  # noqa: E402
    DescriptorEnvelope,
    calibrate_envelope,
    descriptor_vector,
)
from compose_v4.experiments.claim2_transport_laws import ARMS  # noqa: E402

#: Real molecules, so every RDKit metric is exercised for real.
CHAIN = [
    "CNC(=O)c1ccccc1",
    "CNC(=O)c1ccccc1C",
    "CNC(=O)c1ccccc1CC",
    "CNC(=O)c1ccc(O)cc1CC",
    "CNC(=O)c1ccc(O)cc1CCN",
    "CNC(=O)c1ccc(O)cc1CCNC",
    "CNC(=O)c1ccc(OC)cc1CCNC",
]
FAMILIES = [
    ["atom_insert"],
    ["atom_insert"],
    ["atom_restate"],
    ["atom_insert"],
    ["atom_insert"],
    ["atom_restate"],
]


@pytest.fixture(scope="module")
def envelope() -> DescriptorEnvelope:
    vectors = [descriptor_vector(smiles) for smiles in CHAIN] * 30
    return calibrate_envelope(vectors, calibration_source="test", status="TEST")


def make_trajectory(arm: str, seed: int, length: int = 6, cycle: bool = False):
    if cycle:
        states = [CHAIN[0], CHAIN[1], CHAIN[0], CHAIN[1], CHAIN[0], CHAIN[1], CHAIN[0]]
        families = [["atom_insert"]] * 6
    else:
        states = CHAIN[: length + 1]
        families = FAMILIES[:length]
    return {
        "arm": arm,
        "seed": seed,
        "states": states,
        "families": families,
        "cells": [[f"{f[0]}:{f[0]}"] for f in families],
        "support_sizes": [40] * len(families),
        "committed_edits": len(states) - 1,
        "stop_reason": "horizon",
    }


def make_shard(index: int, *, cycling_arm: str | None = None, agrees: bool = True):
    trajectories = []
    for arm in ARMS:
        for seed in (0, 1):
            trajectories.append(
                make_trajectory(arm, seed, cycle=(arm == cycling_arm))
            )
    return {
        "schema": "compose.claim2.trajectory_shard",
        "status": "SMOKE_HELD_IN",
        "index": index,
        "source": CHAIN[0] if index == 0 else f"{CHAIN[0]}.{index}",
        "start_key": CHAIN[0],
        "support_band": ("0", "1-4", "5-24", "25+")[index % 4],
        "size_band": "medium",
        "horizon": 6,
        "seeds": [0, 1],
        "arms": list(ARMS),
        "kernel_agreement": {
            "agrees": agrees,
            "keys_match": agrees,
            "alias_counts_match": agrees,
            "max_absolute_probability_difference": 0.0 if agrees else 0.4,
        },
        "state_divergence": {
            CHAIN[0]: {
                "support_size": 40,
                "pairwise": {"a|b": 0.31, "a|c": 0.22, "b|c": 0.18},
                "minimum": 0.18,
                "degenerate": False,
            }
        },
        "trajectories": trajectories,
        "kernel_calls": 30,
        "budget_exhausted": False,
    }


# ---- per-trajectory metrics ----------------------------------------------


def test_trajectory_metrics_on_a_real_forward_walk(envelope):
    row = analysis.trajectory_metrics(make_trajectory("r_theta", 0), envelope, 6)
    assert row["committed_edits"] == 6
    assert row["endpoint_tanimoto_distance"] > 0.0
    assert row["heavy_atom_change"] > 0
    assert row["unique_state_fraction"] == pytest.approx(1.0)
    assert row["any_state_revisit_rate"] == pytest.approx(0.0)
    assert row["multi_family"] is True
    assert row["envelope_retention"] == pytest.approx(1.0)
    assert not row["early_dead_end"]


def test_trajectory_metrics_detect_a_two_cycle(envelope):
    row = analysis.trajectory_metrics(make_trajectory("r_theta", 0, cycle=True), envelope, 6)
    assert row["immediate_reversal_rate"] > 0.5
    assert row["any_state_revisit_rate"] > 0.5
    assert row["unique_state_fraction"] < 0.5
    # A pure two-cycle ends where it started: no structural movement at all.
    assert row["endpoint_tanimoto_distance"] == pytest.approx(0.0)


# ---- per-source aggregation ----------------------------------------------


def test_seeds_are_averaged_within_source_before_anything_else(envelope):
    """Seeds are repeated measures; the source is the statistical unit."""
    shards = [make_shard(i) for i in range(4)]
    by_arm = analysis.per_source_by_arm(shards, envelope)
    for arm in ARMS:
        assert len(by_arm[arm]) == 4, "one entry per SOURCE, not per trajectory"
        for row in by_arm[arm].values():
            assert row["seeds"] == 2


def test_endpoint_uniqueness_is_measured_per_source(envelope):
    """Identical seeds land identically, so uniqueness is 1/seeds."""
    by_arm = analysis.per_source_by_arm([make_shard(0)], envelope)
    assert by_arm["r_theta"][CHAIN[0]]["endpoint_uniqueness"] == pytest.approx(0.5)


# ---- instrument checks ----------------------------------------------------


def test_instrument_checks_pass_on_a_healthy_run():
    checks = analysis.instrument_checks([make_shard(i) for i in range(4)])
    assert checks["kernel_cross_check_disagreements"] == []
    assert checks["arm_divergence_min"] == pytest.approx(0.18)
    assert checks["degenerate_states"] == 0
    assert checks["budget_exhausted_sources"] == 0


def test_instrument_checks_surface_a_kernel_disagreement():
    """If this kernel is not the production kernel, nothing downstream is real."""
    checks = analysis.instrument_checks(
        [make_shard(0), make_shard(1, agrees=False)]
    )
    assert len(checks["kernel_cross_check_disagreements"]) == 1
    assert checks["kernel_cross_check_disagreements"][0]["source"].endswith(".1")


def test_instrument_checks_flag_a_degenerate_single_successor_state():
    """Three arms over a one-successor support are one process with three labels."""
    shard = make_shard(0)
    shard["state_divergence"] = {
        "x": {"support_size": 1, "pairwise": {"a|b": 0.0}, "minimum": 0.0, "degenerate": True}
    }
    checks = analysis.instrument_checks([shard])
    assert checks["degenerate_states"] == 1
    assert checks["singleton_support_states"] == 1
    assert checks["degenerate_state_fraction"] == pytest.approx(1.0)


# ---- the full pipeline ----------------------------------------------------


def _run(tmp_path: Path, shards, envelope, extra_argv=()):
    directory = tmp_path / "shards"
    directory.mkdir()
    for shard in shards:
        (directory / f"source-{shard['index']:04d}.json").write_text(json.dumps(shard))
    envelope_path = tmp_path / "envelope.json"
    payload = envelope.to_json()
    payload["envelope_sha256"] = "0" * 64
    envelope_path.write_text(json.dumps(payload))
    out = tmp_path / "analysis.json"
    argv = sys.argv
    sys.argv = [
        "analyse_claim2_trajectories.py",
        "--shards", str(directory),
        "--envelope", str(envelope_path),
        "--out", str(out),
        *extra_argv,
    ]
    try:
        code = analysis.main()
    finally:
        sys.argv = argv
    return code, json.loads(out.read_text())


def test_pipeline_refuses_a_verdict_on_an_underpowered_smoke(tmp_path, envelope):
    """Eight sources measure cost and sanity. They do not decide a claim."""
    code, result = _run(tmp_path, [make_shard(i) for i in range(8)], envelope)
    assert code == 0
    assert result["source_count"] == 8
    assert result["verdict_emitted"] is False
    for comparison in result["comparisons"].values():
        assert comparison["verdict"] == "UNDERPOWERED_NO_VERDICT"


def test_pipeline_emits_a_verdict_once_the_source_floor_is_met(tmp_path, envelope):
    code, result = _run(tmp_path, [make_shard(i) for i in range(24)], envelope)
    assert code == 0
    assert result["verdict_emitted"] is True
    for comparison in result["comparisons"].values():
        assert comparison["verdict"] in {
            "dominates", "dominated_by", "incomparable", "unresolved"
        }


def test_pipeline_marks_a_disagreeing_kernel_invalid_and_withholds_the_verdict(
    tmp_path, envelope
):
    shards = [make_shard(i) for i in range(24)]
    shards[3]["kernel_agreement"]["agrees"] = False
    code, result = _run(tmp_path, shards, envelope)
    assert code == 1
    assert result["status"] == "INVALID_INSTRUMENT"
    assert result["verdict_emitted"] is False


def test_pipeline_detects_a_cycling_arm_through_the_health_metrics(tmp_path, envelope):
    """The comparison must be able to come out against R_theta."""
    shards = [make_shard(i, cycling_arm="r_theta") for i in range(24)]
    _code, result = _run(tmp_path, shards, envelope)
    reference = result["per_source"]["r_theta"]
    uniform = result["per_source"]["uniform_canonical"]
    any_source = next(iter(reference))
    assert reference[any_source]["any_state_revisit_rate"] > uniform[any_source][
        "any_state_revisit_rate"
    ]
    assert result["frontier"]["r_theta"]["mobility"] < result["frontier"][
        "uniform_canonical"
    ]["mobility"]
    comparison = result["comparisons"]["r_theta_vs_uniform_canonical"]
    assert comparison["verdict"] in {"dominated_by", "incomparable"}


def test_pipeline_stratifies_by_support_band(tmp_path, envelope):
    _code, result = _run(tmp_path, [make_shard(i) for i in range(24)], envelope)
    assert set(result["by_support_band"]) == {"0", "1-4", "5-24", "25+"}
    for band in result["by_support_band"].values():
        for arm in ARMS:
            assert band[arm]["sources"] == 6


def test_pipeline_records_the_banned_statistics_in_its_own_output(tmp_path, envelope):
    """The artifact carries the list, so a later reader cannot reintroduce one."""
    _code, result = _run(tmp_path, [make_shard(i) for i in range(8)], envelope)
    banned = result["instrument_checks"]["banned_statistics"]
    assert "reference_log_likelihood_of_own_trajectories" in banned
    assert "validity_rate_by_arm" in banned
