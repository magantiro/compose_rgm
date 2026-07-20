#!/usr/bin/env python3
"""Run, or fail-closed wait for, the qualified real-rewrite GrIDDD smoke."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile

from compose_v4.chem.molecular_graph import smiles_to_molecular_graph
from compose_v4.experiments.griddd_conditional import (
    CorrectedBackboneContract,
    FactorizedConditionalBackbone,
    GridDDConditionalEvaluator,
    GridDDProtocol,
    LeadRecord,
    RewriteLeadCandidateGenerator,
    file_sha256,
    load_unconditional_backbone_qualification,
    standard_conditional_arms,
)
from compose_v4.experiments.molecular_property_conditioning import (
    PropertyConditionNormalizer,
)
if __package__:
    from scripts.evaluate_tracelet_rollouts import load_factorized_rollout_checkpoint
else:
    from evaluate_tracelet_rollouts import load_factorized_rollout_checkpoint


LEAD_SMILES = "Cc1occc1C(=O)NNC(=O)Nc1ccc(F)cc1F"
DEFAULT_QUALIFICATION = Path(
    "diagnostics/canonical_successor_backbone_qualification.json"
)
DEFAULT_CONDITIONAL_CHECKPOINT = Path(
    "artifacts/griddd_qed_canonical_successor_smoke/checkpoint.pt"
)
DEFAULT_OUTPUT = Path("diagnostics/griddd_real_rewrite_smoke_status.json")


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


def _wait_payload(
    *,
    qualification_manifest: Path,
    conditional_checkpoint: Path,
    code: str,
    detail: str,
) -> dict[str, object]:
    return {
        "format": "compose_v4_griddd_real_rewrite_smoke_status_v1",
        "phase": "waiting",
        "launched": False,
        "completed": False,
        "large_benchmark_authorized": False,
        "training_launched": False,
        "qualification_manifest": str(qualification_manifest),
        "conditional_checkpoint": str(conditional_checkpoint),
        "blockers": [{"code": code, "detail": detail}],
        "next_action": (
            "rerun this exact command after the named artifact is present and passes; "
            "the 800x20 benchmark remains prohibited"
        ),
    }


def _normalizer(payload: dict[str, object]) -> PropertyConditionNormalizer:
    raw = payload.get("property_conditioning")
    if not isinstance(raw, dict):
        raise ValueError("conditional checkpoint lacks property-conditioning metadata")
    normalizer = PropertyConditionNormalizer(
        names=tuple(str(value) for value in raw["names"]),
        means=tuple(float(value) for value in raw["means"]),
        standard_deviations=tuple(
            float(value) for value in raw["standard_deviations"]
        ),
    )
    if normalizer.names != ("qed",):
        raise ValueError("real-rewrite smoke requires a QED-conditioned checkpoint")
    return normalizer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--qualification-manifest", type=Path, default=DEFAULT_QUALIFICATION
    )
    parser.add_argument(
        "--conditional-checkpoint", type=Path, default=DEFAULT_CONDITIONAL_CHECKPOINT
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=20260724)
    parser.add_argument("--candidates", type=int, default=1)
    parser.add_argument("--guidance-oracle-calls", type=int, default=4)
    parser.add_argument("--operational-horizon", type=float, default=0.4)
    parser.add_argument("--max-events", type=int, default=4)
    args = parser.parse_args()
    if args.candidates <= 0:
        raise ValueError("candidate count must be positive")
    if args.guidance_oracle_calls < 4:
        raise ValueError("real-rewrite smoke requires at least four guidance calls")
    if args.operational_horizon <= 0.0 or args.max_events <= 0:
        raise ValueError("rollout horizon and event limit must be positive")

    if not args.qualification_manifest.is_file():
        status = _wait_payload(
            qualification_manifest=args.qualification_manifest,
            conditional_checkpoint=args.conditional_checkpoint,
            code="qualification_manifest_missing",
            detail="the multi-state unconditional qualification manifest is absent",
        )
        _atomic_json_write(status, args.output)
        print(json.dumps(status, sort_keys=True))
        return
    try:
        qualification = load_unconditional_backbone_qualification(
            args.qualification_manifest
        )
    except (OSError, ValueError, json.JSONDecodeError) as error:
        status = _wait_payload(
            qualification_manifest=args.qualification_manifest,
            conditional_checkpoint=args.conditional_checkpoint,
            code="qualification_manifest_not_passed",
            detail=str(error),
        )
        _atomic_json_write(status, args.output)
        print(json.dumps(status, sort_keys=True))
        return

    if not args.conditional_checkpoint.is_file():
        status = _wait_payload(
            qualification_manifest=args.qualification_manifest,
            conditional_checkpoint=args.conditional_checkpoint,
            code="qualified_qed_child_checkpoint_missing",
            detail=(
                "a QED-conditioned child bound to the passed unconditional "
                "manifest is not present"
            ),
        )
        status["unconditional_qualification"] = qualification.to_dict()
        _atomic_json_write(status, args.output)
        print(json.dumps(status, sort_keys=True))
        return

    conditional_sha256 = file_sha256(args.conditional_checkpoint)
    model, checkpoint = load_factorized_rollout_checkpoint(
        args.conditional_checkpoint
    )
    contract = CorrectedBackboneContract.from_qualification_manifest(
        qualification,
        checkpoint,
        conditional_checkpoint_sha256=conditional_sha256,
    )
    # This flag is execution semantics, not a learned P1/P2 modifier.  It is
    # accepted only after the checkpoint payload and parent manifest both make
    # the same explicit assertion above.
    model.virtualize_legacy_self_grafts = bool(
        checkpoint["molecular_self_transitions_virtualized"]
    )
    backbone = FactorizedConditionalBackbone(
        model=model,
        contract=contract,
        qed_normalizer=_normalizer(checkpoint),
    )
    protocol = GridDDProtocol(
        candidates_per_start=args.candidates,
        guidance_oracle_calls_per_candidate=args.guidance_oracle_calls,
        proposals_per_controlled_event=4,
        bootstrap_replicates=250,
        bootstrap_seed=args.seed + 1,
    )
    generator = RewriteLeadCandidateGenerator(
        backbone=backbone,
        n_slots=40,
        operational_horizon=args.operational_horizon,
        time_step=0.1,
        max_events=args.max_events,
        proposals_per_event=4,
        minimum_similarity=protocol.minimum_tanimoto_similarity,
        fingerprint_radius=protocol.fingerprint_radius,
        fingerprint_bits=protocol.fingerprint_bits,
    )
    report = GridDDConditionalEvaluator(
        protocol=protocol,
        generator=generator,
        backbone_contract=contract,
    ).evaluate(
        (
            LeadRecord(
                "cnof-griddd-real-rewrite-smoke-0",
                smiles_to_molecular_graph(LEAD_SMILES),
            ),
        ),
        seeds=(args.seed,),
        arms=standard_conditional_arms(beta=8.0),
    )
    expected_calls = protocol.oracle_calls_per_start_seed_arm
    arms = report["arms"]
    passed = bool(
        all(
            arms[name]["exact_oracle_budget_every_start_seed"]
            and arms[name]["total_oracle_calls"] == expected_calls
            and arms[name]["candidate_attempts"] == args.candidates
            and arms[name]["valid_candidates"] == args.candidates
            and arms[name]["all_trajectory_states_valid"]
            and arms[name]["all_trajectory_states_connected"]
            for name in ("direct", "controller", "combined")
        )
    )
    output = {
        "format": "compose_v4_griddd_real_rewrite_smoke_status_v1",
        "phase": "complete" if passed else "failed",
        "launched": True,
        "completed": True,
        "passed": passed,
        "real_rewrite_generator": True,
        "efficacy_claim_authorized": False,
        "large_benchmark_authorized": False,
        "training_launched": False,
        "unconditional_qualification": qualification.to_dict(),
        "conditional_checkpoint": {
            "path": str(args.conditional_checkpoint),
            "sha256": conditional_sha256,
            "property_condition_dim": int(model.property_condition_dim),
        },
        "expected_oracle_calls_per_arm": expected_calls,
        "evaluation": report,
    }
    _atomic_json_write(output, args.output)
    print(
        json.dumps(
            {"phase": output["phase"], "output": str(args.output), "passed": passed}
        )
    )
    if not passed:
        raise RuntimeError("qualified real-rewrite conditional smoke failed")


if __name__ == "__main__":
    main()
