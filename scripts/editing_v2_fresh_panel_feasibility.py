"""Feasibility only: can we build a fresh held-out source-target panel?

Answers exactly four questions and stops:

  1. Can we produce >= 50 strict held-out real source->target transformations?
  2. Are they genuinely 4-6 step executable under the frozen executor?
  3. Are there meaningful numbers of REFERENCE_MONOTONE and REFERENCE_DIP?
  4. Is the leakage check clean?

It FREEZES NOTHING and selects no panel. Counting is deliberately separated from
selection so that a panel is never chosen on the same pass that discovered it.

WHY THIS ROUTE
--------------
The historical corpora are collections of supervised (state, successor)
examples, not trajectories -- 29,928 forked states in the training corpus alone
-- so traces cannot be reconstructed from them. Rather than excavate the trace
infrastructure, pairs are built FRESH, after the split, from molecules R_theta
never trained on.

Targets are nominated by a REAL shared-core relationship (one-cut MMP), never by
searching the rewrite graph for something that happens to connect. Both
endpoints are drawn from the matched reserve, whose source keys are disjoint
from the training-source universe (measured: 10,653 keys, zero overlap), so both
endpoints are held out by construction rather than by filtering.

LOCAL COUNTS ONLY
-----------------
`canonical_state_key` goes through RDKit, and this machine runs a different
RDKit from the Modal image that holds the authoritative kernel. These counts are
a feasibility estimate. The panel itself must be compiled on Modal so its
canonical keys match the kernel the controllers will run against; nothing here
is admissible as a frozen artifact.

NAMING
------
"verified executable 4-6 step transformation" -- NOT shortest path. No
minimality is proven or claimed. `REFERENCE_DIP` says THIS path is
non-monotone; it does not say no monotone path exists.
"""

from __future__ import annotations

import argparse
import collections
import gzip
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

MIN_STEPS = 4
MAX_STEPS = 6
SUFFICIENT = 50


def training_transitions(corpus: Path, training_ids: set[str]) -> set[tuple[str, str]]:
    """Every exact (source_key, target_key) R_theta was supervised on."""

    out: set[tuple[str, str]] = set()
    with gzip.open(corpus / "LIBRARY.jsonl.gz", "rt") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            if record["p50_entry_sha256"] not in training_ids:
                continue
            fiber = record["teacher_successor_fiber"]
            out.add((fiber["source_key"], fiber["target_key"]))
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reserve", required=True, type=Path)
    parser.add_argument("--corpus", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--sources", type=int, default=1500,
                        help="held-out molecules to mine pairs from")
    parser.add_argument("--max-pairs", type=int, default=4000,
                        help="cap on pairs compiled; a cap that BINDS is reported")
    parser.add_argument("--seed", type=int, default=20260811)
    parser.add_argument("--pool", choices=("reserve", "training"), default="reserve",
                        help="'training' mines HELD-IN molecules for h_phi teacher "
                             "data; both endpoints are then training sources and "
                             "cannot collide with the reserve-mined evaluation panel")
    args = parser.parse_args()

    from build_analogue_trace_pool import SYSTEM, compile_one_cut, mine_one_cut_pairs
    from compose_v4.chem.molecular_graph import smiles_to_molecular_graph
    from compose_v4.chem.state import pad_molecular_graph
    from compose_v4.drd2_oracle import tanimoto_to
    from compose_v4.rewrite.kernel import canonical_state_key
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")
    started = time.perf_counter()

    reserve = json.load(gzip.open(args.reserve, "rt"))
    training_sources = set(reserve["training_source_keys"])
    training_ids = set(reserve["training_entry_ids"])
    if args.pool == "training":
        # HELD-IN pool for h_phi teacher data. Both endpoints are training
        # sources, and the training-source and reserve key sets are disjoint,
        # so a pair mined here CANNOT collide with the reserve-mined evaluation
        # panel on either endpoint. No filtering is needed for that.
        reserve_sources = sorted(training_sources)
        leaked_sources = []
        print(f"HELD-IN pool: {len(reserve_sources):,} training molecules "
              f"(teacher data; disjoint from the evaluation panel by construction)")
    else:
        reserve_sources = list(reserve["reserve_source_keys"])
        leaked_sources = [s for s in reserve_sources if s in training_sources]
        print(f"held-out pool: {len(reserve_sources):,} reserve molecules; "
              f"{len(leaked_sources)} of them also in the training-source universe")
        if leaked_sources:
            reserve_sources = [s for s in reserve_sources if s not in training_sources]

    supervised = training_transitions(args.corpus, training_ids)
    print(f"[{time.perf_counter()-started:5.1f}s] {len(supervised):,} exact supervised "
          f"transitions to check against", flush=True)

    rng = random.Random(args.seed)
    pool = sorted(reserve_sources)
    rng.shuffle(pool)
    pool = pool[: args.sources]
    label = "HELD-IN" if args.pool == "training" else "held-out"
    print(f"mining one-cut pairs over {len(pool):,} {label} molecules "
          f"(seed {args.seed})", flush=True)

    pairs = mine_one_cut_pairs(pool)
    print(f"[{time.perf_counter()-started:5.1f}s] {len(pairs):,} shared-core pairs "
          f"nominated by a REAL relationship (no graph search)", flush=True)
    pair_cap_binds = len(pairs) > args.max_pairs
    if pair_cap_binds:
        rng.shuffle(pairs)
        pairs = pairs[: args.max_pairs]
        print(f"  cap BINDS: compiling {len(pairs):,} of them", flush=True)

    reasons: collections.Counter = collections.Counter()
    lengths: collections.Counter = collections.Counter()
    accepted: list[dict[str, Any]] = []
    leaked = 0

    for index, (a, b) in enumerate(pairs):
        try:
            mol_a, mol_b = Chem.MolFromSmiles(a), Chem.MolFromSmiles(b)
            if mol_a is None or mol_b is None:
                reasons["unparsable"] += 1
                continue
            slots = max(mol_a.GetNumHeavyAtoms(), mol_b.GetNumHeavyAtoms()) + 2
            steps, reason, meta = compile_one_cut(a, b, n_slots=slots)
        except Exception as error:  # noqa: BLE001 - one bad pair must not end the pass
            reasons[f"raised:{type(error).__name__}"] += 1
            continue
        reasons[reason] += 1
        if steps is None:
            continue
        lengths[len(steps)] += 1
        if not (MIN_STEPS <= len(steps) <= MAX_STEPS):
            continue

        # Replay the compiled path to canonical keys, so the leakage check is on
        # the exact transitions a controller would have to reproduce.
        try:
            state = pad_molecular_graph(smiles_to_molecular_graph(a), slots)
            keys = [canonical_state_key(state)]
            for step in steps:
                state = SYSTEM.apply(state, step.rule_name, step.action)
                keys.append(canonical_state_key(state))
        except Exception as error:  # noqa: BLE001 - counted, never fatal
            reasons[f"replay_failed:{type(error).__name__}"] += 1
            continue
        # The compiled path must actually land on the nominated target, or the
        # pair is not the transformation it claims to be.
        if keys[-1] != canonical_state_key(
                pad_molecular_graph(smiles_to_molecular_graph(b), slots)):
            reasons["endpoint_mismatch"] += 1
            continue

        transitions = list(zip(keys, keys[1:]))
        overlaps = any(t in supervised for t in transitions)
        if overlaps:
            leaked += 1
            # For the HELD-IN teacher pool, overlap with R_theta's supervised
            # transitions is irrelevant and must not exclude: h_phi learns
            # V_greedy, a property of the greedy policy, and is evaluated only
            # on the sealed panel. Counted, not rejected.
            if args.pool != "training":
                continue

        trajectory = [tanimoto_to(keys[-1], k) for k in keys]
        rising = all(y > x for x, y in zip(trajectory, trajectory[1:]))
        accepted.append({
            "source": a, "target": b, "steps": len(steps),
            "shape": "REFERENCE_MONOTONE" if rising else "REFERENCE_DIP",
            "similarity_trajectory": [round(v, 4) for v in trajectory],
            "heavy_atom_delta": mol_b.GetNumHeavyAtoms() - mol_a.GetNumHeavyAtoms(),
            # Emitted here so downstream Modal entrypoints -- which run in the
            # Modal CLI venv and have no RDKit -- never need chemistry.
            "source_heavy_atoms": mol_a.GetNumHeavyAtoms(),
            "target_heavy_atoms": mol_b.GetNumHeavyAtoms(),
            "slots": max(mol_a.GetNumHeavyAtoms(), mol_b.GetNumHeavyAtoms()) + 2,
            "ring_delta": (mol_b.GetRingInfo().NumRings()
                               - mol_a.GetRingInfo().NumRings()),
            "compiler_path_class": (meta or {}).get("compiler_path_class"),
            "intermediates_seen_as_training_source":
                sum(1 for k in keys[1:-1] if k in training_sources),
        })
        if index % 500 == 0:
            print(f"  [{time.perf_counter()-started:5.1f}s] {index:,}/{len(pairs):,} "
                  f"compiled, {len(accepted):,} accepted", flush=True)

    shapes = collections.Counter(r["shape"] for r in accepted)
    by_len = collections.Counter(r["steps"] for r in accepted)
    print(f"\n[{time.perf_counter()-started:5.1f}s] RESULT")
    print(f"  compiled path length distribution: {dict(sorted(lengths.items())[:12])}")
    print(f"  compile outcomes: {dict(reasons.most_common(8))}")
    print(f"  rejected for training leakage: {leaked:,}")
    kind = ("HELD-IN teacher transformations" if args.pool == "training"
            else "strict held-out 4-6 step transformations")
    print(f"  ACCEPTED {kind}: {len(accepted):,}")
    print(f"    REFERENCE_MONOTONE {shapes['REFERENCE_MONOTONE']:,} / "
          f"REFERENCE_DIP {shapes['REFERENCE_DIP']:,}")
    print(f"    steps 4/5/6: {by_len.get(4,0):,} / {by_len.get(5,0):,} / "
          f"{by_len.get(6,0):,}")

    verdict = ("SUFFICIENT" if len(accepted) >= SUFFICIENT else "THIN")
    print(f"\n  {verdict}: {len(accepted):,} against a {SUFFICIENT} threshold")

    args.out.write_text(json.dumps({
        "schema": "compose.editing_v2.fresh_panel_feasibility",
        "status": "FEASIBILITY_COUNTS_ONLY_LOCAL_RDKIT_FREEZES_NOTHING",
        "pool": args.pool,
        "pool_note": (
            "training = HELD-IN molecules for h_phi teacher data. Both endpoints "
            "are training sources, and the training-source and reserve key sets "
            "are disjoint, so these pairs cannot collide with the reserve-mined "
            "evaluation panel on either endpoint. Training-transition overlap is "
            "counted but NOT excluded here: h_phi learns V_greedy, a property of "
            "the greedy policy, and is evaluated only on the sealed panel."),
        "caveat": (
            "Counted with this machine's RDKit, which differs from the Modal "
            "image holding the authoritative kernel. Feasibility estimate only; "
            "the panel must be compiled on Modal before it is frozen."),
        "naming": (
            "verified executable 4-6 step transformation, NOT shortest path. "
            "REFERENCE_DIP means this path is non-monotone, not that no monotone "
            "path exists."),
        "held_out_pool": len(reserve_sources),
        "reserve_sources_in_training_universe": len(leaked_sources),
        "supervised_transitions_checked_against": len(supervised),
        "pairs_nominated": len(pairs),
        "pair_cap_binds": pair_cap_binds,
        "compile_outcomes": dict(reasons),
        "compiled_length_distribution": {str(k): v for k, v in sorted(lengths.items())},
        "rejected_for_leakage": leaked,
        "accepted": len(accepted),
        "shape": dict(shapes),
        "by_steps": {str(k): v for k, v in sorted(by_len.items())},
        "sufficiency_threshold": SUFFICIENT,
        "verdict": verdict,
        "transformations": accepted,
    }, indent=2) + "\n")
    print(f"  wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
