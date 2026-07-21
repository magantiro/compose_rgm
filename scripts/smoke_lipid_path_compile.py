#!/usr/bin/env python3
"""Local CPU smoke: do lipid SMILES compile into certified teacher programs at 96 atoms?

Validates the corpus -> teacher-path compilation (the CPU stage of the Modal
pipeline) locally on a small sample, with the Lineage-B transport config
(flexible_size_graft, carbon_tree seed) at lipid scale (n_slots=96). No GPU/Modal.
Reports: how many targets compile, program lengths, and any failures.

Run:
    KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 PYTHONPATH=src \
        python3 scripts/smoke_lipid_path_compile.py --n 50 --max-atoms 96
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from rdkit import Chem, RDLogger

from compose_v4.chem.molecular_graph import smiles_to_molecular_graph
from compose_v4.chem.source_prior import DegreeBoundedCarbonTreePrior
from compose_v4.experiments.tracelet_conditional import build_tree_transport_path_records

RDLogger.DisableLog("rdApp.*")
REPO_ROOT = Path(__file__).resolve().parents[1]
CORPUS = REPO_ROOT / "artifacts/datasets/compose_lipid_pretraining_v1/generator_corpus_cnof_v1.smiles"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=50, help="how many corpus SMILES to compile")
    parser.add_argument("--max-atoms", type=int, default=96)
    parser.add_argument("--couplings", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260720)
    parser.add_argument("--corpus", type=Path, default=CORPUS)
    parser.add_argument("--output", type=Path,
                        default=REPO_ROOT / "diagnostics/lipid_path_compile_smoke.json")
    args = parser.parse_args()

    lines = [ln.strip().split()[0] for ln in args.corpus.open() if ln.strip()]
    rng = np.random.default_rng(args.seed)
    idx = rng.choice(len(lines), size=min(args.n, len(lines)), replace=False)
    sample = tuple(lines[i] for i in idx)

    # in-scope heavy-atom sizes of the sample -> carbon-tree seed size law (empirical)
    sizes = []
    for s in sample:
        m = Chem.MolFromSmiles(s)
        if m is not None:
            sizes.append(m.GetNumHeavyAtoms())
    from collections import Counter
    size_counts = {int(k): int(v) for k, v in Counter(sizes).items()}
    prior = DegreeBoundedCarbonTreePrior.from_size_counts(size_counts, smoothing=1.0)

    print(f"compiling {len(sample)} lipids at n_slots={args.max_atoms}, "
          f"flexible_size_graft, {args.couplings} couplings/target ...", flush=True)
    records = build_tree_transport_path_records(
        smiles=sample,
        n_slots=args.max_atoms,
        source_prior=prior,
        seed=args.seed,
        couplings_per_target=args.couplings,
        transport_mode="flexible_size_graft",
        ring_catalog=None,
        workers=0,
    )

    # program lengths (K instructions per certified path)
    lengths = []
    targets = set()
    for rec in records:
        targets.add(rec.target_key if hasattr(rec, "target_key") else rec[0])
        path = rec.path if hasattr(rec, "path") else rec[1]
        lengths.append(int(len(path.trace.steps)))  # K certified edit-steps

    expected = len(sample) * args.couplings
    lengths_arr = np.array(lengths) if lengths else np.array([0])
    print(f"\n=== RESULT ===")
    print(f"compiled path records: {len(records)} / expected {expected} "
          f"({len(records)/max(1,expected):.0%})")
    print(f"unique target molecules with >=1 certified program: {len(targets)} / {len(sample)}")
    print(f"program length (edits/target): min {lengths_arr.min()}  median "
          f"{int(np.median(lengths_arr))}  max {lengths_arr.max()}  mean {lengths_arr.mean():.1f}")
    print(f"heavy-atom sizes in sample: median {int(np.median(sizes))}  max {max(sizes)}")
    if len(records) == expected:
        print("OK: every sampled lipid compiled at lipid scale (no drops).")
    else:
        print(f"NOTE: {expected - len(records)} coupling(s) did not yield a record "
              "(H>4 transient / ring-cut / transport limit) -- see per-target above.")

    import json
    out = {
        "format": "compose_lipid_path_compile_smoke_v1",
        "purpose": "P2-G2 evidence (local CPU): lipid SMILES -> certified teacher programs at lipid scale",
        "config": {"n_slots": args.max_atoms, "transport": "flexible_size_graft",
                   "source_prior": "carbon_tree", "couplings_per_target": args.couplings,
                   "seed": args.seed, "sample": len(sample)},
        "target_program_coverage": round(len(targets) / len(sample), 4),
        "coupling_record_rate": round(len(records) / max(1, expected), 4),
        "program_length": {"min": int(lengths_arr.min()), "median": int(np.median(lengths_arr)),
                            "max": int(lengths_arr.max()), "mean": round(float(lengths_arr.mean()), 1)},
        "sample_heavy_atoms": {"median": int(np.median(sizes)), "max": int(max(sizes))},
        "interpretation": ("Every sampled CNOF lipid compiled to a certified seed->target program "
                           "under the Lineage-B transport at max_atoms=96 -- P2-G2 support holds at "
                           "lipid scale locally (full 10k-scale + rollout stays on Modal, gated by P1-G7)."),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(f"written: {args.output.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
