"""Generate T4 programs and apply reference-aware selection with zero docking calls."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch
from reference_guidance import source_identity, write_new_json

from compose_v4.control.fiber_control import ProgramValue, SearchState
from compose_v4.control.reference_guidance import GuidanceConfig
from compose_v4.control.reference_programs import (
    FrozenProgramReference,
    ProgramPanelGuidance,
    t4_program_input,
)
from compose_v4.control.structural_subgoal_realizer import RealizerConfig
from compose_v4.experiments.t4_fiber_campaign import Fiber, expand
from compose_v4.experiments.t4_integrated_route_fiber import attach_features, select_batch
from compose_v4.experiments.t4_route_proposals import (
    RouteProposalConfig,
    expand_route,
    load_route_expert,
)
from compose_v4.model.reference_checkpoint import load_frozen_reference


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lead", required=True)
    parser.add_argument("--delta", type=float, required=True)
    parser.add_argument("--draws", type=positive_int, help="shallow proposals, default 8")
    parser.add_argument("--horizon", type=int, choices=(1, 2, 3), help="shallow modules, default 3")
    parser.add_argument(
        "--proposal-lane", choices=("shallow", "route_complete_region"), default="shallow"
    )
    parser.add_argument("--program-template-prior", type=Path)
    parser.add_argument("--program-template-prior-sha256")
    parser.add_argument(
        "--route-expert", dest="program_template_prior", type=Path, help=argparse.SUPPRESS
    )
    parser.add_argument(
        "--route-expert-sha256", dest="program_template_prior_sha256", help=argparse.SUPPRESS
    )
    parser.add_argument("--route-pool-size", type=positive_int, default=64)
    parser.add_argument("--route-realizations", type=positive_int, default=32)
    parser.add_argument("--route-beam-width", type=positive_int, default=32)
    parser.add_argument("--route-expansion-width", type=positive_int, default=24)
    parser.add_argument("--route-bindings", type=positive_int, default=4)
    parser.add_argument("--proposal-seed", type=int, default=11)
    parser.add_argument("--selection-seed", type=int, default=17)
    parser.add_argument("--compiler-expansions", type=positive_int, default=64)
    parser.add_argument("--mode", choices=("off", "shadow", "active"), default="off")
    parser.add_argument("--strength", type=float, default=0)
    parser.add_argument("--log-weight-cap", type=float, default=1)
    parser.add_argument(
        "--coverage", choices=("baseline", "error", "preserve_mass"), default="baseline"
    )
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to replace output: {args.output}")
    if not np.isfinite(args.delta) or not 0 <= args.delta <= 1:
        parser.error("--delta must be finite and in [0, 1]")
    if args.proposal_seed < 0 or args.selection_seed < 0:
        parser.error("seeds must be nonnegative")
    template_prior = route_config = None
    if args.proposal_lane == "route_complete_region":
        if args.draws is not None or args.horizon is not None:
            parser.error("--draws and --horizon are shallow-lane settings")
        if args.program_template_prior is None or args.program_template_prior_sha256 is None:
            parser.error(
                "the route lane requires --program-template-prior and "
                "--program-template-prior-sha256"
            )
        template_prior = load_route_expert(
            args.program_template_prior,
            expected_sha256=args.program_template_prior_sha256,
        )
        route_config = RouteProposalConfig(
            pool_size=args.route_pool_size,
            realization_limit=args.route_realizations,
            beam_width=args.route_beam_width,
            expansion_width=args.route_expansion_width,
            max_bindings_per_template=args.route_bindings,
            maximum_expansions=args.compiler_expansions,
        )
    elif args.program_template_prior is not None or args.program_template_prior_sha256 is not None:
        parser.error("program template prior requires --proposal-lane route_complete_region")
    else:
        args.draws = 8 if args.draws is None else args.draws
        args.horizon = 3 if args.horizon is None else args.horizon
    config = GuidanceConfig(args.mode, args.strength, args.log_weight_cap, args.coverage)
    torch.set_num_threads(1)
    reference = None
    manifest_path = Path(__file__).resolve().parents[1] / "experiments/fragments/assets.json"
    manifest = json.loads(manifest_path.read_text())
    if args.mode != "off":
        if args.checkpoint is None:
            parser.error("--checkpoint is required in shadow and active modes")
        reference = FrozenProgramReference(
            load_frozen_reference(
                args.checkpoint,
                expected_sha256=manifest["assets"]["checkpoint"]["sha256"],
                expected_catalog_fingerprint=manifest["catalog_fingerprint"],
                catalog_path=manifest_path.parent.parent.parent
                / "local_assets/fragments"
                / manifest["assets"]["catalog"]["path"],
                expected_catalog_sha256=manifest["assets"]["catalog"]["sha256"],
            )
        )
    fiber = Fiber(args.lead, args.delta)
    generator = np.random.default_rng(args.proposal_seed)
    compiler = RealizerConfig(maximum_expansions=args.compiler_expansions)
    proposal_telemetry = None
    if template_prior is None:
        candidates = expand(
            args.lead,
            0.0,
            fiber,
            generator,
            draws=args.draws,
            horizon=args.horizon,
            include_realized_actions=True,
            realizer_config=compiler,
        )
    else:
        candidates, proposal_telemetry = expand_route(
            args.lead,
            0.0,
            fiber,
            template_prior,
            config=route_config,
            include_realized_actions=True,
        )
    print(f"Generated {len(candidates)} eligible candidates", flush=True)
    programs = tuple(t4_program_input(row, candidate_id=row["smiles"]) for row in candidates)
    guide = ProgramPanelGuidance(programs, reference, config, identity_field="smiles")
    state = SearchState()
    rows = attach_features(candidates, state, fiber)
    if len(rows) != len(candidates):
        raise RuntimeError("feature preparation changed the eligible panel")
    rng, baseline_rng = (
        np.random.default_rng(args.selection_seed),
        np.random.default_rng(args.selection_seed),
    )
    receipts = []
    settings = {"round_index": 1, "batch": 1, "exploration": 1, "expert_floor_rounds": 0}
    chosen = select_batch(
        rows,
        ProgramValue(),
        state,
        rng,
        **settings,
        reference_guide=guide,
        reference_receipts=receipts,
    )
    baseline = select_batch(rows, ProgramValue(), state, baseline_rng, **settings)
    identity = source_identity()
    identity["source_sha256"]["examples/t4_reference_panel.py"] = hashlib.sha256(
        Path(__file__).read_bytes()
    ).hexdigest()
    payload = {
        "schema": "compose.t4_reference_panel",
        "schema_version": 3,
        "purpose": "generated-program integration check without docking",
        "configuration": {
            "reference": asdict(config),
            "reference_progress": None if reference is None else reference.progress,
            "reference_batch_size": 16,
            "compiler": asdict(compiler),
            "lead": args.lead,
            "delta": args.delta,
            "endpoint_support": fiber.support,
            "draws": args.draws if template_prior is None else None,
            "horizon": args.horizon if template_prior is None else None,
            "proposal_lane": args.proposal_lane,
            "route_proposal": None if route_config is None else asdict(route_config),
            "multi_region": True,
            "proposal_seed": args.proposal_seed,
            "selection_seed": args.selection_seed,
            "seed_derivation": "literal CLI seeds",
            "selector": settings,
            "device": "cpu",
            "dtype": "float32",
            "threads": 1,
            "parent_score": 0.0,
            "parent_score_role": "synthetic fixture, not a docking score",
        },
        "candidates": candidates,
        "proposal_telemetry": proposal_telemetry,
        "program_template_prior": None
        if template_prior is None
        else {
            "path": str(args.program_template_prior.resolve()),
            "sha256": args.program_template_prior_sha256,
            "training_identity": template_prior.training_identity,
            "role": "controller input for program construction",
        },
        "programs": [
            {
                "candidate_id": p.candidate_id,
                "trace_sha256": p.trace_sha256,
                "trace": None if p.trace_json is None else json.loads(p.trace_json),
                "missing_reason": p.missing_reason,
            }
            for p in programs
        ],
        "selection": receipts,
        "chosen": [row["smiles"] for row in chosen],
        "baseline_chosen": [row["smiles"] for row in baseline],
        "rng": {
            "proposal": generator.bit_generator.state,
            "selection": rng.bit_generator.state,
            "baseline_selection": baseline_rng.bit_generator.state,
        },
        "candidate_count": len(candidates),
        "excluded_after_generation": 0,
        "oracle_calls": 0,
        "status": "complete" if candidates else "empty_panel",
        "asset_manifest": {
            "path": str(manifest_path),
            "sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        },
        "checkpoint_path": str(args.checkpoint.resolve()) if reference is not None else None,
        "implementation": identity,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            **{name: version(name) for name in ("torch", "numpy", "rdkit", "scipy")},
        },
    }
    write_new_json(args.output, payload)
    print(f"Saved panel and selection to {args.output}")


if __name__ == "__main__":
    main()
