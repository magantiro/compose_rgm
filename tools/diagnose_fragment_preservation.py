#!/usr/bin/env python3
"""Reproduce one (task, drug, seed) shard and show every non-preserving endpoint.

The sweep records preservation as a COUNT.  When that count is below the number
of committed endpoints, the interesting object is the molecule itself: it tells
you whether the pathwise region lock was violated (a correctness bug) or
whether the lock held at the graph level while the RDKit substructure query
stopped matching (a perception effect, and a limit on what may be CLAIMED).
The sweep's seeding is deterministic, so this re-runs the identical stream.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from rdkit import Chem

ROOT = Path(__file__).resolve().parents[1]
for extra in (ROOT / "tools", ROOT / "src", ROOT / "scripts"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from run_fragment_constrained_suite import (
    MANIFEST,
    audit_queries,
    contains_all_fragments,
    prompt_rng_seed,
)

from compose_v4.benchmark.fragment_conditioned_sampler import (
    RegionLock,
    SamplerConfig,
    SamplingReceipt,
    build_prompt_context,
    sample_completion,
)
from compose_v4.benchmark.fragment_constrained import (
    FragmentTask,
    load_genmol_prompts,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--task", required=True)
    parser.add_argument("--drug", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    from evaluate_tracelet_rollouts import load_factorized_rollout_checkpoint

    from compose_v4.rewrite.kernel import de_novo_rewrite_system

    model, _meta = load_factorized_rollout_checkpoint(args.checkpoint)
    system = de_novo_rewrite_system()
    task = FragmentTask(args.task)
    prompt = next(
        p
        for p in load_genmol_prompts(MANIFEST)
        if p.drug_name == args.drug and p.task is task
    )
    config = SamplerConfig()
    context = build_prompt_context(prompt, config=config)
    lock = RegionLock(context.start_state, context.locked_slots)
    queries = audit_queries(prompt)

    rng = np.random.default_rng(prompt_rng_seed(args.drug, task.value, args.seed))
    receipt = SamplingReceipt()
    for _ in range(args.samples):
        sample_completion(model, system, context, rng, config=config, receipt=receipt)

    offenders = [
        s for s in receipt.committed_endpoints if not contains_all_fragments(s, queries)
    ]
    core_smiles = Chem.MolToSmiles(queries[0], canonical=True) if queries else ""
    detail = []
    for smiles in offenders:
        mol = Chem.MolFromSmiles(smiles)
        kekule = Chem.MolFromSmiles(smiles)
        Chem.Kekulize(kekule, clearAromaticFlags=True)
        # Does the core match once aromaticity is taken out of the comparison?
        kekule_query = Chem.MolFromSmiles(core_smiles)
        Chem.Kekulize(kekule_query, clearAromaticFlags=True)
        detail.append(
            {
                "smiles": smiles,
                "heavy_atoms": mol.GetNumAtoms(),
                "matches_aromatic_query": mol.HasSubstructMatch(queries[0]),
                "matches_kekulized_query": kekule.HasSubstructMatch(kekule_query),
                "aromatic_rings": sum(
                    1
                    for ring in mol.GetRingInfo().BondRings()
                    if all(mol.GetBondWithIdx(b).GetIsAromatic() for b in ring)
                ),
            }
        )

    payload = {
        "schema": "compose_fragment_preservation_diagnosis_v1",
        "task": task.value,
        "drug": args.drug,
        "seed": args.seed,
        "rng_seed": prompt_rng_seed(args.drug, task.value, args.seed),
        "prompt_core": core_smiles,
        "attempts": args.samples,
        "committed_endpoints": len(receipt.committed_endpoints),
        "non_preserving": len(offenders),
        "region_lock_locked_slots": len(lock.locked_slots),
        "constraint_failures": receipt.constraint_failures,
        "offenders": detail,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2))
    print(json.dumps({k: v for k, v in payload.items() if k != "offenders"}, indent=2))
    for item in detail:
        print(item)


if __name__ == "__main__":
    raise SystemExit(main())
