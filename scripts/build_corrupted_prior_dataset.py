#!/usr/bin/env python3
"""Verify the corrupted-molecule source prior before any GPU (native GM source-prior swap).

Runs the corruption over a real corpus and reports the realized distribution: the edit-OPERATOR
histogram (which skills the model will learn), the ring-change fraction (~0 -- micro edits are
ring-count-preserving), source->target Tanimoto (recognizable variant), source drug-likeness (QED),
edit-length spread, and both-direction yield. Gate before training: the operator mix covers
bioisostere/bond-order beyond bare insertion, NO bond_insert/bond_delete appears, and sources are
realistic. Serial by design (macOS fork deadlock;
at scale this generation runs on Modal inside the recipe).
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import QED, Descriptors, rdFingerprintGenerator

from compose_v4.chem.molecular_graph import (MolecularGraphError,
                                             molecular_graph_to_smiles,
                                             smiles_to_molecular_graph)
from compose_v4.chem.state import pad_molecular_graph
from compose_v4.rewrite.fiber import ActionFiberSpec
from compose_v4.rewrite.kernel import de_novo_rewrite_system
from compose_v4.rewrite.source_corruption import make_edit_pair

RDLogger.DisableLog("rdApp.*")
_ROOT = Path(__file__).resolve().parents[1]
CORPUS = _ROOT / "results/tree_fcd_transfer_stage1_factorized_v1/guacamol_heldout_val_5000_seed0.smiles"
OUT = _ROOT / "diagnostics" / "composition" / "corrupted_prior_verification.json"
_FP = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


def _tanimoto(a: str, b: str):
    ma = Chem.MolFromSmiles(a or ""); mb = Chem.MolFromSmiles(b or "")
    if ma is None or mb is None:
        return None
    return float(DataStructs.TanimotoSimilarity(_FP.GetFingerprint(ma), _FP.GetFingerprint(mb)))


def _rings(smi: str):
    m = Chem.MolFromSmiles(smi or "")
    return Descriptors.RingCount(m) if m is not None else None


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n", type=int, default=120)
    p.add_argument("--depth-max", type=int, default=5)
    p.add_argument("--n-slots", type=int, default=64)
    p.add_argument("--max-atoms", type=int, default=35)
    p.add_argument("--output", type=Path, default=None)
    args = p.parse_args()

    spec = ActionFiberSpec.neutral_cnof()
    system = de_novo_rewrite_system()
    rng = np.random.default_rng(0)
    smis = [line.strip() for line in open(CORPUS) if line.strip()]

    op_hist: Counter = Counter()
    lens, tanis, src_qed = [], [], []
    ring_changed, n_traces, processed, both_yield = 0, 0, 0, 0
    print(f"corrupted-prior verification ({args.n} molecules, depth ~U[1,{args.depth_max}]):", flush=True)
    for smi in smis:
        if processed >= args.n:
            break
        m = Chem.MolFromSmiles(smi)
        if m is None or m.GetNumAtoms() > args.max_atoms:
            continue
        try:
            graph = smiles_to_molecular_graph(smi)
        except MolecularGraphError:
            continue
        processed += 1
        target = pad_molecular_graph(graph, args.n_slots)
        depth = int(rng.integers(1, args.depth_max + 1))
        trim, grow = make_edit_pair(target, depth, spec=spec, system=system, rng=rng)
        if trim is not None and grow is not None:
            both_yield += 1
        for trace in (trim, grow):
            if trace is None:
                continue
            n_traces += 1
            lens.append(len(trace.steps))
            op_hist.update(step.rule_name for step in trace.steps)
            ssmi = molecular_graph_to_smiles(trace.source)
            tsmi = molecular_graph_to_smiles(trace.target)
            tani = _tanimoto(ssmi, tsmi)
            if tani is not None:
                tanis.append(tani)
            rs, rt = _rings(ssmi), _rings(tsmi)
            if rs is not None and rt is not None and rs != rt:
                ring_changed += 1
            sm = Chem.MolFromSmiles(ssmi or "")
            if sm is not None:
                src_qed.append(float(QED.qed(sm)))
        if processed % 40 == 0:
            print(f"  {processed} processed | {n_traces} traces", flush=True)

    summary = {
        "experiment": "corrupted_prior_verification", "processed": processed, "traces": n_traces,
        "both_direction_yield": both_yield / max(processed, 1),
        "operator_histogram": dict(op_hist.most_common()),
        "ring_change_fraction": ring_changed / max(n_traces, 1),
        "edit_len_mean": float(np.mean(lens)) if lens else 0.0,
        "edit_len_hist": {str(k): v for k, v in sorted(Counter(lens).items())},
        "src_tgt_tanimoto_mean": float(np.mean(tanis)) if tanis else 0.0,
        "src_tgt_tanimoto_p10_p90": [float(np.percentile(tanis, 10)),
                                     float(np.percentile(tanis, 90))] if tanis else [0.0, 0.0],
        "source_qed_mean": float(np.mean(src_qed)) if src_qed else 0.0,
        "has_forbidden_family": any(k in ("bond_insert", "bond_delete") for k in op_hist),
    }
    print(f"\ntraces {n_traces} | operators {summary['operator_histogram']}", flush=True)
    print(f"ring-change {summary['ring_change_fraction']:.0%} | edit-len mean {summary['edit_len_mean']:.1f} "
          f"{summary['edit_len_hist']} | src->tgt Tanimoto {summary['src_tgt_tanimoto_mean']:.2f} "
          f"(p10-p90 {summary['src_tgt_tanimoto_p10_p90'][0]:.2f}-{summary['src_tgt_tanimoto_p10_p90'][1]:.2f}) | "
          f"source QED {summary['source_qed_mean']:.2f} | forbidden-family {summary['has_forbidden_family']}", flush=True)
    out = args.output or OUT
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
