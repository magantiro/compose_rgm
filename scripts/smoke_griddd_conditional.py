#!/usr/bin/env python3
"""Run a bounded, no-training smoke of the GrIDDD execution contract."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path
import tempfile

from compose_v4.chem.molecular_graph import MolecularGraph, smiles_to_molecular_graph
from compose_v4.experiments.griddd_conditional import (
    CandidateGenerationResult,
    ConditionalArm,
    CorrectedBackboneContract,
    GridDDConditionalEvaluator,
    GridDDProtocol,
    LeadRecord,
    standard_conditional_arms,
)


LEAD_SMILES = "Cc1occc1C(=O)NNC(=O)Nc1ccc(F)cc1F"
SUCCESS_SMILES = "COc1cccc(C(=O)Nc2ccc(F)cc2F)c1"


def _atomic_json_write(payload: dict[str, object], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=output.parent,
        prefix=f".{output.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, output)


@dataclass
class _SmokeCandidateGenerator:
    lead: MolecularGraph
    success: MolecularGraph
    counters: dict[str, int] = field(default_factory=dict)

    def generate(
        self,
        *,
        lead_state,
        target_qed,
        arm: ConditionalArm,
        rng,
        oracle_scope,
    ) -> CandidateGenerationResult:
        del lead_state, target_qed, rng
        index = self.counters.get(arm.name, 0)
        self.counters[arm.name] = index + 1
        final = self.lead if arm.name == "direct" else self.success
        productive_calls = {
            "direct": index % 2,
            "controller": min(index + 1, oracle_scope.allowance),
            "combined": min(index + 2, oracle_scope.allowance),
        }[arm.name]
        for _ in range(productive_calls):
            oracle_scope.score(
                final,
                purpose="smoke_canonical_successor",
                influences_selection=arm.controller,
            )
        return CandidateGenerationResult(
            final_state=final,
            trajectory_states_audited=2,
            all_trajectory_states_valid=True,
            all_trajectory_states_connected=True,
            events=1,
            exhausted_event_budget=False,
            metadata={"scripted_smoke": True},
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260723)
    parser.add_argument("--candidates", type=int, default=2)
    parser.add_argument("--guidance-oracle-calls", type=int, default=4)
    args = parser.parse_args()
    if args.candidates <= 0 or args.guidance_oracle_calls < 4:
        raise ValueError("smoke requires positive candidates and at least four guidance calls")

    lead = smiles_to_molecular_graph(LEAD_SMILES)
    success = smiles_to_molecular_graph(SUCCESS_SMILES)
    contract = CorrectedBackboneContract(
        checkpoint_sha256=hashlib.sha256(b"unqualified-griddd-smoke").hexdigest(),
        weight_source="pancake_derived",
        canonical_successor_execution=True,
        molecular_self_transitions_virtualized=True,
        unconditional_qualification_passed=False,
    )
    protocol = GridDDProtocol(
        candidates_per_start=args.candidates,
        guidance_oracle_calls_per_candidate=args.guidance_oracle_calls,
        proposals_per_controlled_event=4,
        bootstrap_replicates=250,
        bootstrap_seed=args.seed + 1,
    )
    report = GridDDConditionalEvaluator(
        protocol=protocol,
        generator=_SmokeCandidateGenerator(lead, success),
        backbone_contract=contract,
        allow_unqualified_smoke=True,
    ).evaluate(
        (LeadRecord("cnof-griddd-fixture-0", lead),),
        seeds=(args.seed,),
        arms=standard_conditional_arms(beta=8.0),
    )
    arms = report["arms"]
    expected_calls = protocol.oracle_calls_per_start_seed_arm
    passed = bool(
        report["smoke_only"]
        and not report["training_launched"]
        and all(
            arms[name]["exact_oracle_budget_every_start_seed"]
            and arms[name]["total_oracle_calls"] == expected_calls
            and arms[name]["candidate_attempts"] == args.candidates
            and arms[name]["valid_candidates"] == args.candidates
            for name in ("direct", "controller", "combined")
        )
        and arms["direct"]["successful_start_seeds"] == 0
        and arms["controller"]["successful_start_seeds"] == 1
        and arms["combined"]["successful_start_seeds"] == 1
    )
    report["smoke_gate"] = {
        "passed": passed,
        "bounded": True,
        "scripted_candidates": True,
        "qualified_backbone_used": False,
        "large_training_launched": False,
        "expected_oracle_calls_per_arm": expected_calls,
    }
    if not passed:
        raise RuntimeError("GrIDDD execution-contract smoke failed")
    _atomic_json_write(report, args.output)
    print(json.dumps({"phase": "complete", "output": str(args.output), "passed": passed}))


if __name__ == "__main__":
    main()
