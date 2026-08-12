"""End-to-end dry run of the pathwise pipeline with a synthetic kernel.

The frozen R_theta only runs on Modal, so the analysis script would otherwise
be first exercised by the run it is meant to interpret. Here it is driven to
completion against shards built from the local arm policies over a hand-made
successor graph, before any Modal spend.

Two fixtures, and the second one is the important one: a graph where the motif
is NEVER violated. The analyser must return G1 FAIL there. A gate that can only
say PASS is not a gate.
"""

from __future__ import annotations

import json
import random
import subprocess
import sys
from pathlib import Path

from compose_v4.experiments.pathwise_arms import STAGE_A, ARM_BUILDERS, ArmContext
from compose_v4.experiments.pathwise_constraints import (
    derive_protected_motif,
    mask_successors,
    path_violation_summary,
)

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts/analyse_pathwise_constraints.py"

SOURCE = "CCN1CCN(c2ccccc2)CC1"
MOTIF = derive_protected_motif(SOURCE)

KEEP_1 = "CCN1CCN(c2ccccc2Cl)CC1"
KEEP_2 = "CCN1CCN(c2ccc(Cl)cc2Cl)CC1"
BREAK = "CCN1CCN(C2CCCCC2)CC1"
RESTORED = "CCN1CCN(c2ccccc2Br)CC1"

#: The constraint BITES: the best endpoint is only reachable through BREAK.
BITING_GRAPH = {
    SOURCE: [(KEEP_1, 0.5), (BREAK, 0.5)],
    KEEP_1: [(KEEP_2, 1.0)],
    KEEP_2: [],
    BREAK: [(RESTORED, 1.0)],
    RESTORED: [],
}
#: The constraint is VACUOUS: nothing on offer ever breaks the ring.
VACUOUS_GRAPH = {
    SOURCE: [(KEEP_1, 0.5), (RESTORED, 0.5)],
    KEEP_1: [(KEEP_2, 1.0)],
    KEEP_2: [],
    RESTORED: [],
}
UTILITY = {SOURCE: 0.0, KEEP_1: 1.0, KEEP_2: 2.0, BREAK: 3.0, RESTORED: 9.0}


def build_shard(index: int, graph: dict) -> dict:
    ctx = ArmContext(
        successors=lambda key: list(graph.get(key, [])),
        utility=lambda key: (UTILITY[key], UTILITY[key]),
        motif_smarts=MOTIF.smarts,
        rng=random.Random(index),
        horizon=2,
        rollouts=6,
        shortlist=3,
    )
    arms = {}
    for name in STAGE_A:
        result = ARM_BUILDERS[name](ctx, SOURCE)
        landing = result["trajectory"][-1]
        audit = path_violation_summary(MOTIF.smarts, result["trajectory"])
        arms[name] = {
            "landing": landing,
            "trajectory": result["trajectory"],
            "edits": len(result["trajectory"]) - 1,
            "dead_end_step": result.get("dead_end_step"),
            "completed": result.get("dead_end_step") is None,
            "selection_failed": result.get("selection_failed"),
            "rollouts_offered": result.get("rollouts_offered"),
            "rollouts_admissible": result.get("rollouts_admissible"),
            "overrides": result.get("overrides"),
            "top1_disagreements": result.get("top1_disagreements"),
            "b_worst_margin": UTILITY[landing],
            "b_mean_margin": UTILITY[landing],
            "b_success": UTILITY[landing] >= 9.0,
            "p_success": False,
            "d_success": False,
            "audit": audit,
            "marginal_kernel_calls": 3,
        }
        if "rollouts" in result:
            arms[name]["rollout_audits"] = [
                {"audit": r["audit"], "utility": r["utility"],
                 "trajectory": r["trajectory"]}
                for r in result["rollouts"]
            ]
    census = []
    for step, key in enumerate(arms["unconstrained_greedy"]["trajectory"]):
        _kept, stats = mask_successors(MOTIF.smarts, list(graph.get(key, [])))
        census.append({"step": step, "where": "unconstrained_greedy", **stats})
    return {
        "schema": "compose.pathwise.source_result",
        "status": "SMOKE_HELD_IN",
        "mask_leak": None,
        "index": index,
        "source": SOURCE,
        "start_key": SOURCE,
        "motif_smarts": MOTIF.smarts,
        "motif_atoms": MOTIF.atom_count,
        "motif_fraction": MOTIF.fraction,
        "horizon": 2,
        "goal": "B",
        "arm_run_order": list(STAGE_A),
        "b_worst_at_source": 0.0,
        "b_success_at_source": False,
        "arms": arms,
        "mask_census": census,
        "distinct_landings": len({a["landing"] for a in arms.values()}),
        "kernel_calls": 9,
        "seconds": 100.0,
    }


def run_analyser(tmp_path: Path, graph: dict, sources: int = 6) -> dict:
    shards = tmp_path / "shards"
    shards.mkdir(exist_ok=True)
    for index in range(sources):
        (shards / f"{index:03d}.json").write_text(
            json.dumps(build_shard(index, graph), indent=2))
    out = tmp_path / "analysis.json"
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "--shards", str(shards), "--out", str(out)],
        capture_output=True, text=True, cwd=str(REPO), check=False,
    )
    assert proc.returncode == 0, proc.stderr
    return json.loads(out.read_text())


def test_analyser_passes_the_vacuity_gate_when_the_constraint_bites(tmp_path):
    report = run_analyser(tmp_path, BITING_GRAPH)
    gate = report["gates"]["G1_constraint_non_vacuous"]
    assert gate["verdict"] == "PASS"
    # the unconstrained greedy arm walks SOURCE -> BREAK -> RESTORED
    assert gate["unconstrained_endpoint_valid_path_invalid"] == "6/6"
    assert report["gates"]["G0_mask_integrity"]["verdict"] == "PASS"


def test_analyser_FAILS_the_vacuity_gate_when_nothing_ever_violates(tmp_path):
    """The gate must be able to stop the workstream. If this returns PASS the
    instrument is broken, not the chemistry."""
    report = run_analyser(tmp_path, VACUOUS_GRAPH)
    gate = report["gates"]["G1_constraint_non_vacuous"]
    assert gate["verdict"] == "FAIL"
    assert gate["unconstrained_endpoint_valid_path_invalid"] == "0/6"
    assert gate["endpoint_valid_rollouts_that_are_path_invalid"].startswith("0/")


def test_mask_gate_fails_when_the_mask_removes_nothing(tmp_path):
    report = run_analyser(tmp_path, VACUOUS_GRAPH)
    gate = report["gates"]["G2_mask_leaves_room_to_act"]
    assert gate["verdict"] == "FAIL"
    assert gate["removed_fraction_along_unconstrained_states"]["mean"] == 0.0


def test_mask_gate_passes_when_the_mask_bites_but_leaves_room(tmp_path):
    report = run_analyser(tmp_path, BITING_GRAPH)
    gate = report["gates"]["G2_mask_leaves_room_to_act"]
    assert gate["verdict"] == "PASS"
    assert 0.0 < gate["removed_fraction_along_unconstrained_states"]["mean"] < 1.0


def test_price_of_the_guarantee_is_reported_with_a_free_sign(tmp_path):
    report = run_analyser(tmp_path, BITING_GRAPH)
    price = report["price_of_the_guarantee"]["paired_delta"]
    assert price["n"] == 6
    # masked arm cannot reach RESTORED, so the guarantee costs utility here
    assert price["mean"] < 0


def test_definitional_zero_is_recorded_but_never_a_gate(tmp_path):
    report = run_analyser(tmp_path, BITING_GRAPH)
    for name in ("pathwise_greedy", "pathwise_stochastic", "mask_only_sampling"):
        assert report["per_arm"][name]["any_violation_DEFINITIONAL_IF_MASKED"] == 0
    # and it must not appear as a gate verdict anywhere
    assert "violations" not in json.dumps(report["gates"]).lower().replace(
        "any_violation", "")


def test_analyser_refuses_an_all_void_run(tmp_path):
    shards = tmp_path / "shards"
    shards.mkdir()
    (shards / "000.json").write_text(json.dumps(
        {"status": "INVALID_INSTRUMENT", "error": "motif drift", "index": 0}))
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "--shards", str(shards),
         "--out", str(tmp_path / "o.json")],
        capture_output=True, text=True, cwd=str(REPO), check=False)
    assert proc.returncode != 0
    assert "INVALID_INSTRUMENT" in proc.stderr
