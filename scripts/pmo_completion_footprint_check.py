#!/usr/bin/env python
"""Does the completion law now reach the proposals it governs? ZERO ORACLE CALLS.

The first scored A/B spent 500 charged calls comparing a mechanism that executed
on roughly a fifth of the proposals it was designed to govern, because
``memory_channel_proposal`` -- the shallow lane's synthesis route whenever the
online memory is warm -- dropped the keyword.  Measured pre-fix footprint: 0.175.

This runs the REAL ``execute_task`` -- real controller, real memory, real
campaign, real allocator -- against a deterministic SYNTHETIC scorer, then
measures hop 1b of the funnel: of the eligible attempts that used
``segment_grow``/``segment_replace``, what fraction carried the completion law.

It is a GATE.  If the footprint does not clear the floor, the correct action is
to report rather than spend another 500 charged calls on a second throttle test.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rdkit import Chem, RDLogger  # noqa: E402

RDLogger.DisableLog("rdApp.*")

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "diagnostics/pmo_completion_repair_v1"
RUNS = OUT / "footprint_check"
BUDGET = int(os.environ.get("FOOTPRINT_BUDGET", "112"))

#: The treatment arm must reach at least this share, and the control must stay
#: at exactly zero. Declared here, before the check runs.
TREATMENT_FLOOR = 0.90
PRE_FIX_TREATMENT_FOOTPRINT = 0.175


def synthetic(smiles: str) -> float:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return 0.0
    return min(1.0, 0.02 * mol.GetNumAtoms() + 0.05 * mol.GetRingInfo().NumRings())


def main() -> None:
    import pmo_completion_funnel as funnel
    from pmo_completion_scored_ab import load_local_contract

    from compose_v4.experiments.pmo_population_v1 import execute_task

    contract = load_local_contract()
    if RUNS.exists():
        shutil.rmtree(RUNS)
    for arm, spec in contract["arms"].items():
        folder = RUNS / arm
        folder.mkdir(parents=True)
        result = execute_task(
            contract, ROOT, folder, "celecoxib_rediscovery",
            evaluate=synthetic,
            charged_calls_per_task=BUDGET,
            enable_online_memory=bool(spec["enable_online_memory"]),
            completion_law=spec["completion_law"],
        )
        print(f"  {arm}: synthetic calls {result['charged_oracle_calls']}", flush=True)

    funnel.configure(RUNS, tuple(contract["arms"]))
    arms = {arm: funnel.analyse(arm) for arm in contract["arms"]}
    control = arms["A_deployed_b"]
    treatment = arms["B_completion"]

    problems = []
    if treatment["hop_1b_eligible_using_a_completion_family"] < 20:
        problems.append(
            "too few completion-family attempts to measure a footprint: "
            f"{treatment['hop_1b_eligible_using_a_completion_family']}"
        )
    footprint = treatment["hop_1b_FOOTPRINT"]
    if footprint is None or footprint < TREATMENT_FLOOR:
        problems.append(
            f"treatment footprint {footprint} is below the {TREATMENT_FLOOR} floor; "
            "the law still does not reach the proposals it governs"
        )
    if control["hop_1b_FOOTPRINT"]:
        problems.append(
            f"the CONTROL carries the law (footprint {control['hop_1b_FOOTPRINT']}); "
            "the arms are not separated"
        )
    verdict = "FOOTPRINT_RESTORED" if not problems else "STILL_THROTTLED"

    report = {
        "schema_version": "pmo_completion_footprint_check_v1",
        "new_oracle_calls": 0,
        "scorer": "synthetic, deterministic, structure-dependent -- no oracle constructed",
        "synthetic_calls_per_arm": BUDGET,
        "floor": TREATMENT_FLOOR,
        "pre_fix_treatment_footprint_measured_on_the_scored_run": PRE_FIX_TREATMENT_FOOTPRINT,
        "arms": {
            arm: {
                k: arms[arm][k]
                for k in (
                    "rounds",
                    "hop_1_attempts_total",
                    "hop_1b_eligible_using_a_completion_family",
                    "hop_1b_of_those_carrying_the_law",
                    "hop_1b_FOOTPRINT",
                    "hop_2_eligible_attempts",
                    "hop_3_eligible_pool",
                    "hop_4_selected_by_credit_allocate",
                    "hop_4_selected_with_completion",
                    "allocator_discard_rate",
                    "allocator_discard_rate_for_completions",
                )
            }
            for arm in arms
        },
        "problems": problems,
        "verdict": verdict,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "footprint_check_v1.json").write_text(json.dumps(report, sort_keys=True, indent=1))
    print(f"\n  control   footprint {control['hop_1b_FOOTPRINT']} "
          f"({control['hop_1b_of_those_carrying_the_law']} of "
          f"{control['hop_1b_eligible_using_a_completion_family']})")
    print(f"  treatment footprint {footprint} "
          f"({treatment['hop_1b_of_those_carrying_the_law']} of "
          f"{treatment['hop_1b_eligible_using_a_completion_family']})"
          f"   pre-fix was {PRE_FIX_TREATMENT_FOOTPRINT}")
    print("\nVERDICT:", verdict)
    for problem in problems:
        print("  -", problem)
    sys.exit(0 if verdict == "FOOTPRINT_RESTORED" else 1)


main()
