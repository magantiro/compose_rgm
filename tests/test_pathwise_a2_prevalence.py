"""Stage A2 analysis, driven to completion on synthetic shards before launch.

The point of these tests is that A2 CAN FAIL, and can fail on each criterion
separately. A gate that only says PASS is not a gate, and this lane has already
spent one run learning that lesson properly.

Each failure mode below is a real way the pathwise programme closes:
  * too few excursions            -> V3
  * a mask that strangles support -> V4a
  * a mask that empties           -> V4b
  * events on one or two molecules-> V5a / V5b
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from compose_v4.experiments.pathwise_reversible_families import (
    clogp_corridor,
    corridor_excursions,
    state_is_feasible,
)

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts/analyse_pathwise_corridor_prevalence.py"

LOW, HIGH = clogp_corridor()

#: biphenyl, cLogP 3.354 -- mid-corridor.
INSIDE = "c1ccc(-c2ccccc2)cc1"
#: cLogP 6.257 -- well above the upper bound, not a boundary graze.
OUTSIDE = "CCCCCCCCc1ccc(-c2ccccc2)cc1"


def test_corridor_bounds_are_the_frozen_iqr():
    """Guard the one number A2 must not move."""
    assert (round(LOW, 4), round(HIGH, 4)) == (2.3689, 4.4522)


def test_inside_and_outside_fixtures_are_what_the_tests_assume():
    assert state_is_feasible("B_physchem_corridor", INSIDE)
    assert not state_is_feasible("B_physchem_corridor", OUTSIDE)


# --------------------------------------------------------------- excursions


def test_excursion_records_depth_and_duration():
    trajectory = [INSIDE, OUTSIDE, OUTSIDE, INSIDE]
    runs = corridor_excursions("B_physchem_corridor", trajectory)
    assert len(runs) == 1
    assert runs[0]["duration"] == 2
    assert runs[0]["start_index"] == 1
    assert runs[0]["max_depth"] > 0
    assert runs[0]["returned"] is True


def test_excursion_not_returned_when_it_runs_to_the_endpoint():
    runs = corridor_excursions("B_physchem_corridor", [INSIDE, INSIDE, OUTSIDE])
    assert len(runs) == 1
    assert runs[0]["returned"] is False


def test_two_separate_excursions_are_counted_separately():
    trajectory = [INSIDE, OUTSIDE, INSIDE, OUTSIDE, INSIDE]
    runs = corridor_excursions("B_physchem_corridor", trajectory)
    assert [r["duration"] for r in runs] == [1, 1]


# ------------------------------------------------------------ shard builder


def build_shard(index: int, *, events: int, rollouts: int = 6,
                retention: float = 0.5, empty: bool = False) -> dict:
    rolls = []
    for r in range(rollouts):
        if r < events:
            trajectory = [INSIDE, OUTSIDE, INSIDE]      # leaves and returns
        else:
            trajectory = [INSIDE, INSIDE, INSIDE]       # never leaves
        audit = {
            "states": len(trajectory),
            "source_feasible": True,
            "endpoint_feasible": True,
            "any_violation": r < events,
            "violation_count": 1 if r < events else 0,
            "first_violation_index": 1 if r < events else None,
            "returned": r < events,
            "endpoint_valid_path_invalid": r < events,
        }
        rolls.append({
            "rollout": r,
            "trajectory": trajectory,
            "dead_end_step": None,
            "audit": audit,
            "excursions": corridor_excursions("B_physchem_corridor", trajectory),
            "clogp": [3.0, 5.2, 3.0] if r < events else [3.0, 3.0, 3.0],
            "terminal_utility": [0.1, 0.2],
        })
    census = [{
        "step": i, "candidates": 10,
        "kept": 0 if empty else max(1, int(round(retention * 10))),
        "retention": 0.0 if empty else retention,
        "retained_reference_mass": retention,
        "empty_after_mask": empty,
        "state_feasible": True,
    } for i in range(6)]
    return {
        "schema": "compose.pathwise.a2_source",
        "status": "SMOKE_HELD_IN",
        "index": index,
        "source": INSIDE,
        "start_key": INSIDE,
        "start_clogp": 3.0,
        "corridor": [LOW, HIGH],
        "horizon": 6,
        "goal": "B",
        "rollouts_requested": rollouts,
        "rollouts_completed": rollouts,
        "violating_rollouts": events,
        "endpoint_valid_path_invalid_events": events,
        "source_has_event": bool(events),
        "rollouts": rolls,
        "mask_census": census,
        "kernel_calls": 35,
        "kernel_call_budget": 200,
        "seconds": 600.0,
    }


def run_analyser(tmp_path: Path, shards: list[dict]) -> dict:
    shard_dir = tmp_path / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    for shard in shards:
        (shard_dir / f"{shard['index']:03d}.json").write_text(json.dumps(shard))
    out = tmp_path / "a2.json"
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "--shards", str(shard_dir),
         "--out", str(out)],
        capture_output=True, text=True, cwd=str(REPO), check=False)
    assert proc.returncode == 0, proc.stderr
    return json.loads(out.read_text())


# ------------------------------------------------------------- the verdicts


def test_a2_passes_when_the_phenomenon_is_frequent_and_spread(tmp_path):
    shards = [build_shard(i, events=3) for i in range(12)]  # 36 events, 12 sources
    report = run_analyser(tmp_path, shards)
    assert report["verdict"] == "PASS"
    assert report["failed_criteria"] is None
    assert report["source_level"]["sources_with_at_least_one_event"] == 12


def test_a2_FAILS_when_there_are_too_few_events(tmp_path):
    shards = [build_shard(i, events=1) for i in range(12)]  # 12 events < 20
    report = run_analyser(tmp_path, shards)
    assert report["verdict"] == "FAIL"
    assert "V3_event_yield" in report["failed_criteria"]


def test_a2_FAILS_when_events_concentrate_on_one_molecule(tmp_path):
    """The source-spread criterion doing its job: 24 events, but 22 of them
    from a single source."""
    shards = [build_shard(0, events=6, rollouts=22)]
    shards += [build_shard(1, events=1), build_shard(2, events=1)]
    shards += [build_shard(i, events=0) for i in range(3, 12)]
    report = run_analyser(tmp_path, shards)
    assert report["verdict"] == "FAIL"
    assert "V5a_source_spread" in report["failed_criteria"]


def test_a2_FAILS_when_one_source_dominates_the_event_count(tmp_path):
    """V5b: enough distinct sources, but one carries most of the mass."""
    shards = [build_shard(0, events=30, rollouts=30)]
    shards += [build_shard(i, events=1) for i in range(1, 6)]
    shards += [build_shard(i, events=0) for i in range(6, 12)]
    report = run_analyser(tmp_path, shards)
    assert report["verdict"] == "FAIL"
    assert "V5b_no_single_source_dominates" in report["failed_criteria"]


def test_a2_FAILS_when_the_mask_strangles_the_support(tmp_path):
    shards = [build_shard(i, events=3, retention=0.02) for i in range(12)]
    report = run_analyser(tmp_path, shards)
    assert report["verdict"] == "FAIL"
    assert "V4a_median_retention" in report["failed_criteria"]


def test_a2_FAILS_when_the_mask_empties_the_support(tmp_path):
    shards = [build_shard(i, events=3, empty=True) for i in range(12)]
    report = run_analyser(tmp_path, shards)
    assert report["verdict"] == "FAIL"
    assert "V4b_mask_empty_rare" in report["failed_criteria"]


# ------------------------------------------------------- statistical hygiene


def test_intervals_are_clustered_by_source_not_by_trajectory(tmp_path):
    shards = [build_shard(i, events=3) for i in range(12)]
    report = run_analyser(tmp_path, shards)
    boot = report["source_level"]["bootstrap_per_source_event_rate"]
    assert boot["n_sources"] == 12
    assert boot["unit"] == "source (rollouts kept together)"
    # trajectory level must NOT carry an interval
    assert "ci95_low" not in json.dumps(report["trajectory_level"])


def test_bootstrap_widens_when_sources_disagree(tmp_path):
    """A phenomenon carried by half the molecules must show a wider interval
    than one present in all of them -- that is the whole reason for clustering."""
    uniform = run_analyser(tmp_path / "u", [build_shard(i, events=3)
                                            for i in range(12)])
    split = run_analyser(tmp_path / "s",
                         [build_shard(i, events=6) for i in range(6)]
                         + [build_shard(i, events=0) for i in range(6, 12)])
    wide = split["source_level"]["bootstrap_fraction_of_sources_with_event"]
    narrow = uniform["source_level"]["bootstrap_fraction_of_sources_with_event"]
    assert (wide["ci95_high"] - wide["ci95_low"]) > (
        narrow["ci95_high"] - narrow["ci95_low"])


def test_report_carries_the_developmental_framing_verbatim(tmp_path):
    report = run_analyser(tmp_path, [build_shard(i, events=3) for i in range(12)])
    assert "developmental follow-up, not independent confirmation" in report["framing"]
    assert report["stage_a_verdict_unchanged"] == "FAIL"


def test_analyser_refuses_an_all_void_run(tmp_path):
    shard_dir = tmp_path / "shards"
    shard_dir.mkdir()
    (shard_dir / "000.json").write_text(json.dumps(
        {"status": "INVALID_INSTRUMENT", "index": 0,
         "error": "start key outside corridor"}))
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "--shards", str(shard_dir),
         "--out", str(tmp_path / "o.json")],
        capture_output=True, text=True, cwd=str(REPO), check=False)
    assert proc.returncode != 0
    assert "INVALID_INSTRUMENT" in proc.stderr


# ------------------------------------------------------------------- panel


@pytest.mark.skipif(not (REPO / "diagnostics/pathwise_a2_panel.json").exists(),
                    reason="A2 panel not built")
def test_a2_panel_is_disjoint_from_every_earlier_source_set():
    panel = json.loads((REPO / "diagnostics/pathwise_a2_panel.json").read_text())
    a2 = {r["source"] for r in panel["sources"]}
    assert len(a2) == len(panel["sources"]), "duplicate sources in the A2 panel"

    stage_a = json.loads(
        (REPO / "diagnostics/pathwise_constraints_smoke_panel.json").read_text())
    assert not (a2 & {r["source"] for r in stage_a["sources"]})

    cohort = json.loads(
        (REPO / "diagnostics/retarget_calibration_cohort.json").read_text())
    assert not (a2 & {r["source"] for r in cohort["sources"]})


@pytest.mark.skipif(not (REPO / "diagnostics/pathwise_a2_panel.json").exists(),
                    reason="A2 panel not built")
def test_every_a2_source_starts_inside_the_corridor():
    """Applicability: a source outside the corridor can never leave and return,
    and would silently drop out of the denominator as it did in stage A."""
    panel = json.loads((REPO / "diagnostics/pathwise_a2_panel.json").read_text())
    for row in panel["sources"]:
        assert state_is_feasible("B_physchem_corridor", row["source"]), row["source"]
