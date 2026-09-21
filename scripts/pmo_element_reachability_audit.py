"""Which declared targets contain elements the proposal path cannot INSTALL?

Zero oracle calls. This is a gate that runs BEFORE any per-task transport test: a target
requiring an element the planner cannot install is unreachable BY CONSTRUCTION, and that
list is wanted up front rather than discovered one task at a time. Same class as the T4
charge-policy finding -- a vocabulary boundary masquerading as a search failure.

TWO SOURCES OF EVIDENCE, deliberately kept separate.

READ (structural): the growth modules in `dynamic_program_synthesis.py` pass explicit
element tuples to `_grow_actions` -- ("C", "N", "O") and ("C", "N", "O", "F"). Reading
this tells you what the code says.

EXECUTED (behavioural): from a source containing ONLY C/N/O/F, run the real PMO proposal
path and census the elements of every endpoint it produces. If an element never appears
across thousands of exactly-executed proposals from a source that does not already
contain it, insertion does not install it. This is the half that cannot be fooled by a
misread.

The two are reported side by side. A disagreement is a finding, not a formatting problem.
"""

from __future__ import annotations

import argparse
import collections
import json
from dataclasses import replace
from pathlib import Path

from rdkit import Chem, RDLogger

from compose_v4.control.adaptive_program_optimizer import ProgramSearchConfig
from compose_v4.control.dynamic_program_synthesis_v21 import (
    initial_dynamic_program_batch_v21,
)
from compose_v4.control.pmo_population_controller import PmoPopulationController
from compose_v4.experiments.editing_v2_evaluation_semantics import (
    production_state_from_smiles,
)

RDLogger.DisableLog("rdApp.*")
ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = json.loads(
    (ROOT / "diagnostics/pmo_joint_dependency_jump_gate_v1/attempt_2/checkpoints.json").read_text()
)["payload"]["checkpoints"]["shared_all_routes"]

#: What the growth modules declare, read from the source. Behaviour is measured below.
DECLARED_GROWTH_ELEMENTS = ("C", "N", "O", "F")

#: A CNOF-only starting molecule. Any non-CNOF element appearing in an endpoint therefore
#: had to be INSTALLED rather than inherited.
CNOF_SOURCE = "CC(=O)Nc1ccc(O)cc1"


def _eligibility(row):
    return {"oracle_eligible": Chem.MolFromSmiles(row["smiles"]) is not None}


def elements_of(smiles: str) -> set[str]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return set()
    return {atom.GetSymbol() for atom in mol.GetAtoms()}


def executed_element_census(*, seed: int, rounds: int) -> dict:
    """Elements appearing in endpoints produced from a CNOF-only source."""
    source = production_state_from_smiles(CNOF_SOURCE, max_atoms=48)
    config = replace(
        ProgramSearchConfig.program_only_recipe(seed=seed, score_direction="maximize"),
        attempts_per_batch=128,
        candidates_per_batch=16,
        wall_seconds=45.0,
        parent_allocation="niche_score",
    )
    batch = initial_dynamic_program_batch_v21(
        source, (), config,
        source_group="element-audit", oracle_protocol="free-no-oracle",
        eligibility=_eligibility,
    )
    controller = PmoPopulationController(
        config, source_group="element-audit", oracle_protocol="free-no-oracle",
        hierarchy=None, jump_checkpoint=CHECKPOINT, enable_online_memory=True,
    )
    for index, candidate in enumerate(batch["candidates"][:8]):
        controller.add_measured_program(candidate, receipt_id=f"seed{index}", score=0.5)

    seen = collections.Counter()
    endpoints = 0
    rules = collections.Counter()
    for round_index in range(rounds):
        proposed = controller.propose_batch(_eligibility)
        # Census the FULL pool, not the allocated subset: the question is what the
        # proposal path can BUILD, not what the allocator happened to fund.
        for candidate in proposed["eligible_pool"]["candidates"]:
            endpoints += 1
            for element in elements_of(candidate["endpoint"]):
                seen[element] += 1
            for action in candidate["trace"]["actions"]:
                rules[action["executor_rule"]] += 1
        selected = proposed["candidates"]
        ids = [row["candidate_id"] for row in selected]
        locked = controller.lock_query_subset(
            proposed["batch_id"], ids, {"policy": "element_audit", "selected_ids": ids}
        )
        controller.observe_batch(
            locked["batch_id"],
            [
                {"candidate_id": r["candidate_id"],
                 "receipt_id": f"r{round_index}-{r['candidate_id'][:10]}",
                 "score": 0.5, "oracle_protocol": controller.oracle_protocol}
                for r in locked["candidates"]
            ],
        )
    return {
        "source": CNOF_SOURCE,
        "source_elements": sorted(elements_of(CNOF_SOURCE)),
        "endpoints_censused": endpoints,
        "elements_observed": dict(sorted(seen.items())),
        "executor_rules": dict(rules.most_common()),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    audit = json.loads(
        (ROOT / "diagnostics/pmo_discovery_v1/goal_specification_audit_v1.json").read_text()
    )
    executed = executed_element_census(seed=args.seed, rounds=args.rounds)
    installed = set(executed["elements_observed"]) - set(executed["source_elements"])
    reachable = set(DECLARED_GROWTH_ELEMENTS) | installed

    rows = []
    for row in audit["rows"]:
        for declared in row.get("declared", []):
            if declared["kind"] != "smiles":
                continue
            present = elements_of(declared["value"])
            outside = sorted(present - reachable)
            rows.append({
                "task": row["task"],
                "goal_kind": row["goal_kind"],
                "target_elements": sorted(present),
                "elements_outside_growth_vocabulary": outside,
                "installable_by_growth": not outside,
            })
            break

    blocked = [r for r in rows if not r["installable_by_growth"]]
    print(f"EXECUTED census: {executed['endpoints_censused']} endpoints from a "
          f"{'/'.join(executed['source_elements'])} source")
    print(f"  elements observed: {sorted(executed['elements_observed'])}")
    print(f"  INSTALLED (not in source): {sorted(installed) or 'none'}")
    print(f"  declared growth vocabulary (READ): {list(DECLARED_GROWTH_ELEMENTS)}")
    print()
    for row in rows:
        flag = "OK " if row["installable_by_growth"] else "BLOCKED"
        print(f"  {flag} {row['task']:<30} {'/'.join(row['target_elements']):<14}"
              f" outside={row['elements_outside_growth_vocabulary']}")
    print(f"\n{len(blocked)} of {len(rows)} declared-structure targets need an element "
          f"the growth modules do not install.")
    report = {
        "schema_version": "pmo_element_reachability_audit_v1",
        "oracle_calls": 0,
        "declared_growth_elements_read_from_source": list(DECLARED_GROWTH_ELEMENTS),
        "executed_census": executed,
        "elements_installed_by_execution": sorted(installed),
        "targets": rows,
        "blocked_targets": [r["task"] for r in blocked],
        "note": (
            "A blocked target is not unreachable in principle: atom_restate_semantic can "
            "change an existing atom's element. It is unreachable by INSERTION, so the "
            "planner must emit restate-then-grow for those atoms rather than an insert."
        ),
    }
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + "\n")
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
