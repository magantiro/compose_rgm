"""Compare selection modes on two executable toy programs, with no oracle calls.

Run from the source checkout with PYTHONPATH=src, or install the package first.
This is an integration example, not a benchmark or a proposed operating point.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import tempfile
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch

from compose_v4.chem.molecular_graph import ELEMENT_TO_IDX, smiles_to_molecular_graph
from compose_v4.chem.state import pad_molecular_graph
from compose_v4.control.reference_guidance import GuidanceConfig, guide_panel
from compose_v4.control.reference_programs import (
    FrozenProgramReference,
    ProgramInput,
    ProgramPanelGuidance,
)
from compose_v4.model.reference_checkpoint import load_frozen_reference
from compose_v4.rewrite.action_codec_v4 import encode_action
from compose_v4.rewrite.kernel import canonical_state_key, editing_v2_semantic_rewrite_system
from compose_v4.rewrite.operators import AtomInsert
from compose_v4.rewrite.trace_shard import encode_state

REFERENCE_MANIFEST = Path(__file__).resolve().parents[1] / "experiments/reference/model.json"


def example_programs() -> tuple[ProgramInput, ...]:
    """Two declared toy insertions from ethane; no fitted or task-selected inputs."""
    source = pad_molecular_graph(smiles_to_molecular_graph("CC"), 12)
    programs = []
    for element, hydrogens in (("O", 1), ("N", 2)):
        action = AtomInsert(2, ELEMENT_TO_IDX[element], 0, hydrogens, ((1, 1),))
        product = editing_v2_semantic_rewrite_system().apply(source, "atom_insert", action)
        trace = {
            "actions": [encode_action("atom_insert", action)],
            "states": [encode_state(source), encode_state(product)],
        }
        programs.append(ProgramInput.from_trace(element, canonical_state_key(product), trace))
    return tuple(programs)


def write_new_json(path: Path, payload: dict) -> None:
    """Complete, validated JSON published atomically without overwriting outputs."""
    data = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", dir=path.parent, prefix=".reference-", delete=False
    ) as stream:
        temporary = Path(stream.name)
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def source_identity() -> dict:
    root = Path(__file__).resolve().parents[1]
    revision = changes = None
    # A source export inside another checkout must not inherit that checkout's
    # revision. Linked worktrees have a .git file, ordinary clones a directory.
    if (root / ".git").exists():
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=False
        )
        changes = subprocess.run(
            ["git", "status", "--porcelain"], cwd=root, capture_output=True, text=True, check=False
        )
    # Source-tree hashes include new uncommitted implementation files; a Git
    # revision alone would misidentify a development invocation.
    paths = [Path(__file__), *sorted((root / "src/compose_v4").rglob("*.py"))]
    hashes = {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths
    }
    return {
        "git_commit": revision.stdout.strip()
        if revision is not None and revision.returncode == 0
        else None,
        "dirty": bool(changes.stdout.strip())
        if changes is not None and changes.returncode == 0
        else None,
        "source_sha256": hashes,
    }


def controller_draw(selector, programs, rng, *, guide=None, receipts=None):
    """Exercise an actual task allocator on fixed toy programs, without an oracle."""
    rows = [
        {
            "candidate_id": program.candidate_id,
            "smiles": program.endpoint,
            "features": [1.0, 1.0],
            "fingerprint": {index},
        }
        for index, program in enumerate(programs)
    ]
    if selector == "pmo":
        from compose_v4.control.pmo_reward_adaptive import RewardAdaptiveProgramController

        indices, _ = RewardAdaptiveProgramController().acquire(
            rows, 0.0, batch=1, rng=rng, reference_guide=guide, reference_receipts=receipts
        )
        return indices[0]
    if selector == "t4":
        from compose_v4.control.fiber_control import ProgramValue, SearchState
        from compose_v4.experiments.t4_integrated_route_fiber import select_batch

        selected = select_batch(
            rows,
            ProgramValue(),
            SearchState(),
            rng,
            round_index=1,
            batch=1,
            exploration=1,
            expert_floor_rounds=0,
            reference_guide=guide,
            reference_receipts=receipts,
        )
        return next(
            i for i, row in enumerate(rows) if row["candidate_id"] == selected[0]["candidate_id"]
        )
    raise ValueError(f"unknown controller selector: {selector!r}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("off", "shadow", "active"), default="off")
    parser.add_argument("--selector", choices=("toy", "pmo", "t4"), default="toy")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--strength", type=float, default=0.0)
    parser.add_argument("--log-weight-cap", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = GuidanceConfig(args.mode, strength=args.strength, log_weight_cap=args.log_weight_cap)
    if args.output.exists():
        raise FileExistsError(f"refusing to replace output: {args.output}")
    torch.set_num_threads(1)
    programs = example_programs()
    scorer = reference = None
    if config.mode != "off":
        if args.checkpoint is None:
            parser.error("shadow and active modes require --checkpoint")
        reference_manifest = json.loads(REFERENCE_MANIFEST.read_text())
        catalog = reference_manifest["catalog"]
        loaded = load_frozen_reference(
            args.checkpoint,
            expected_sha256=reference_manifest["checkpoint"]["sha256"],
            expected_catalog_fingerprint=catalog["fingerprint"],
            catalog_path=REFERENCE_MANIFEST.parent / catalog["path"],
            expected_catalog_sha256=catalog["sha256"],
        )
        reference = FrozenProgramReference(loaded)
        scorer = lambda: reference.score(programs)
    baseline_rng, selected_rng = np.random.default_rng(args.seed), np.random.default_rng(args.seed)
    if args.selector == "toy":
        baseline = (0.5, 0.5)
        result = guide_panel(
            tuple(p.candidate_id for p in programs), baseline, config=config, score=scorer
        )
        baseline_index = int(baseline_rng.choice(len(programs), p=baseline))
        selected_index = int(selected_rng.choice(len(programs), p=result.probabilities))
        selection = result.receipt()
    else:
        baseline_index = controller_draw(args.selector, programs, baseline_rng)
        receipts = []
        selected_index = controller_draw(
            args.selector,
            programs,
            selected_rng,
            guide=ProgramPanelGuidance(programs, reference, config),
            receipts=receipts,
        )
        selection = receipts[0]
    payload = {
        "schema": "compose.reference_guidance_example",
        "schema_version": 2,
        "purpose": "two-program, zero-oracle implementation smoke test; not benchmark evidence",
        "configuration": {
            **asdict(config),
            "selector": args.selector,
            "selector_context": "cold start, one slot, no reserved allocation floors",
            "seed": args.seed,
            "seed_derivation": "literal CLI seed",
            "reference_progress": None if reference is None else reference.progress,
            "batch_size": 16,
            "threads": 1,
            "device": "cpu",
            "dtype": "float32",
            "split": "synthetic unit fixtures",
        },
        "programs": [
            {
                "candidate_id": p.candidate_id,
                "endpoint": p.endpoint,
                "trace_sha256": p.trace_sha256,
                "trace": json.loads(p.trace_json),
            }
            for p in programs
        ],
        "selection": selection,
        "baseline_index": baseline_index,
        "selected_index": selected_index,
        "baseline_rng_state": baseline_rng.bit_generator.state,
        "selected_rng_state": selected_rng.bit_generator.state,
        "candidate_count": len(programs),
        "excluded_candidates": 0,
        "oracle_calls": 0,
        "checkpoint_path": str(args.checkpoint.resolve()) if config.mode != "off" else None,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            **{name: version(name) for name in ("torch", "numpy", "rdkit")},
        },
        "implementation": source_identity(),
    }
    write_new_json(args.output, payload)
    print(
        f"{args.selector}/{selection['outcome']}: "
        f"probabilities_changed={selection['probabilities_changed']}; output={args.output}"
    )


if __name__ == "__main__":
    main()
