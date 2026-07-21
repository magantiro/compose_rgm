#!/usr/bin/env python3
"""Lipid unconditional-generator evaluation panel (P2-G3 / L1 contract).

Reuses the generic generator-quality core (validity / uniqueness / exact novelty /
NN-memorization via molecular_quality_report) and ADDS the lipid-specific reportable
marginals the paper requires: head/linker/tail architecture, tail count/length/
branching/unsaturation, degradable-linker rate + linker-type distribution,
protonation proxy (basic-N count), and lipid-scaffold (heteroatom-connector-core)
novelty. Compares a generated set against a reference set on every axis.

Until the generator trains (gated behind P1-G7), validate the panel by passing a
held-out corpus split as --generated and the training split as --reference/--train:
the generated marginals should match the reference (the corpus is its own target).
When rollouts exist, swap --generated for the generated SMILES.

Run:
    KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 PYTHONPATH=src \
        python3 scripts/analyze_lipid_generator_panel.py \
          --generated <smiles> --reference <smiles> --train <smiles>
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
from rdkit import Chem, RDLogger

from compose_v4.eval.molecular_quality import molecular_quality_report
from compose_v4.lipids.corpus_bias import heteroatom_connector_core, jensen_shannon, lipid_topology_features
from compose_v4.lipids.region_labels import REGION_NAMES, _LINKER_PATTS, lipid_region_labels
from compose_v4.oracles.head_domain import basic_nitrogens

RDLogger.DisableLog("rdApp.*")
REPO_ROOT = Path(__file__).resolve().parents[1]


def _load(path: Path, limit: int | None) -> list[str]:
    out = []
    for ln in path.open():
        s = ln.strip().split()[0] if ln.strip() else ""
        if s:
            out.append(s)
        if limit and len(out) >= limit:
            break
    return out


def _bin(v, edges, labels):
    for e, l in zip(edges, labels):
        if v <= e:
            return l
    return labels[-1]


def lipid_marginals(smiles: list[str]) -> dict:
    """The L1 reportable lipid marginals for one set of molecules."""
    n_tails, tail_len, branch, unsat, linker, basicN, region_share = (
        Counter(), Counter(), Counter(), Counter(), Counter(), Counter(), Counter())
    degradable = 0
    region_atoms_total = Counter()
    n = 0
    for s in smiles:
        m = Chem.MolFromSmiles(s)
        if m is None:
            continue
        n += 1
        f = lipid_topology_features(m)
        n_tails[f["long_tail_bin"]] += 1
        tail_len[_bin(f["max_aliphatic_tail_depth"], [7, 12, 17, 22], ["<=7", "8-12", "13-17", "18-22", ">22"])] += 1
        branch[_bin(f["carbon_branch_point_count"], [0, 1, 2], ["0", "1", "2", "3plus"])] += 1
        unsat[_bin(f["cc_unsaturation_count"], [0, 1, 2], ["0", "1", "2", "3plus"])] += 1
        basicN[_bin(len(basic_nitrogens(m)), [1, 2, 3], ["1", "2", "3", "4plus"])] += 1
        present = [name for name, patt in _LINKER_PATTS.items() if m.HasSubstructMatch(patt)]
        for name in present:
            linker[name] += 1
        # degradable = ester/amide/carbamate/carbonate/disulfide/acetal (hydrolyzable/cleavable)
        if any(x in present for x in ("ester", "amide", "carbamate", "carbonate", "disulfide", "acetal")):
            degradable += 1
        for code in lipid_region_labels(m):
            region_atoms_total[REGION_NAMES[code]] += 1

    def _dist(c):
        t = sum(c.values()) or 1
        return {k: round(v / t, 4) for k, v in c.most_common()}
    reg_tot = sum(region_atoms_total.values()) or 1
    return {
        "n": n,
        "architecture": {
            "n_tails": _dist(n_tails), "tail_length": _dist(tail_len),
            "branching": _dist(branch), "unsaturation": _dist(unsat),
            "basic_nitrogen_count": _dist(basicN),
            "region_atom_share": {k: round(region_atoms_total[k] / reg_tot, 4) for k in REGION_NAMES.values()},
        },
        "linker_type_distribution": _dist(linker),
        "degradable_linker_rate": round(degradable / max(1, n), 4),
    }


def _js_marginals(gen: dict, ref: dict) -> dict:
    """JS divergence generated-vs-reference for each architecture marginal (lower = closer)."""
    out = {}
    for axis in ("n_tails", "tail_length", "branching", "unsaturation", "basic_nitrogen_count"):
        g, r = gen["architecture"][axis], ref["architecture"][axis]
        cats = sorted(set(g) | set(r))
        def pv(d):
            v = np.array([d.get(c, 0.0) for c in cats])
            return v / max(v.sum(), 1e-9)
        out[axis] = round(jensen_shannon(pv(g), pv(r)), 4)
    return out


def lipid_scaffold_novelty(generated: list[str], train: list[str]) -> dict:
    """Fraction of generated whose heteroatom-connector-core is not seen in training."""
    train_cores = set()
    for s in train:
        m = Chem.MolFromSmiles(s)
        if m is not None:
            train_cores.add(heteroatom_connector_core(m))
    novel = seen = 0
    for s in generated:
        m = Chem.MolFromSmiles(s)
        if m is None:
            continue
        seen += 1
        if heteroatom_connector_core(m) not in train_cores:
            novel += 1
    return {"train_cores": len(train_cores), "evaluated": seen,
            "lipid_scaffold_novelty": round(novel / max(1, seen), 4)}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--generated", type=Path, required=True)
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument("--train", type=Path, required=True)
    p.add_argument("--generated-limit", type=int, default=1500)
    p.add_argument("--reference-limit", type=int, default=1500)
    p.add_argument("--train-limit", type=int, default=6000)
    p.add_argument("--label", type=str, default="lipid_corpus_selfcheck")
    p.add_argument("--output", type=Path, default=REPO_ROOT / "diagnostics/lipid_generator_panel.json")
    args = p.parse_args()

    generated = _load(args.generated, args.generated_limit)
    reference = _load(args.reference, args.reference_limit)
    train = _load(args.train, args.train_limit)

    quality = molecular_quality_report(tuple(generated), reference_smiles=tuple(reference),
                                       train_smiles=tuple(train), include_fcd=False)
    gen_marg = lipid_marginals(generated)
    ref_marg = lipid_marginals(reference)
    panel = {
        "format": "compose_lipid_generator_panel_v1",
        "label": args.label,
        "counts": {"generated": len(generated), "reference": len(reference), "train": len(train)},
        "generic_quality": quality,
        "lipid_marginals": {"generated": gen_marg, "reference": ref_marg,
                            "js_generated_vs_reference": _js_marginals(gen_marg, ref_marg)},
        "lipid_scaffold_novelty": lipid_scaffold_novelty(generated, train),
        "note": ("L1/P2-G3 panel. When run with a held-out corpus split as --generated, "
                 "generated marginals should match reference (low JS) -- a panel self-check. "
                 "Swap --generated for rollout SMILES once the generator trains."),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(panel, indent=2, sort_keys=True) + "\n")

    js = panel["lipid_marginals"]["js_generated_vs_reference"]
    print(f"=== {args.label}: {len(generated)} generated vs {len(reference)} reference ===")
    print(f"lipid-scaffold novelty vs train: {panel['lipid_scaffold_novelty']['lipid_scaffold_novelty']}")
    print(f"degradable-linker rate: gen {gen_marg['degradable_linker_rate']}  ref {ref_marg['degradable_linker_rate']}")
    print(f"region atom share (gen): {gen_marg['architecture']['region_atom_share']}")
    print(f"architecture JS gen-vs-ref: {js}")
    print(f"written: {args.output.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
