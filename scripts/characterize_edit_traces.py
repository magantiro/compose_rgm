#!/usr/bin/env python3
"""Characterize the B-edit training-trace layers (PAPER_MASTER_PLAN.md sec 0b).

The universal edit prior is trained on a THREE-LAYER mixture; this report measures the two layers
we can generate/inspect today and quantifies the failure modes the plan explicitly warns about:

  - LAYER 1 -- synthetic corruption (source_corruption.make_edit_pair): take a real GuacaMol molecule,
    walk it a few legal micro edits to a nearby valid source, and record BOTH directions (TRIM =
    real->corrupt, GROW = corrupt->real). Full operator support, controllable length, but at risk of
    IDENTITY BIAS if the traces are just "corrupt-and-reconstruct the identical molecule".
  - LAYER 2 -- MMP analogue traces (diagnostics/composition/analogue_trace_pool.jsonl): compiled
    one-cut matched-molecular-pair paths between DISTINCT real molecules, both directions (A->B / B->A).

For each layer AND the pooled overall set we report: trace count + forward/inverse balance; operator-
family mix (totals, per-trace mean, fractions); path-length distribution + data-driven empirical-tercile
edit-budget bins; source<->target Morgan(r=2, 2048) Tanimoto (incl. the [0.4, 0.9] "recognizable variant"
band); IDENTITY fraction (canonical source == canonical target); ring complexity + ring-change fraction;
heavy-atom size distribution; Bemis-Murcko scaffold concentration over the sources (unique / top-share /
Gini); transformation-signature concentration (operator histogram for corruption, source->target Murcko
for analogue); and the charged-species fraction. Safeguard flags fire for identity bias, frequent-scaffold
domination, transformation duplication, and rare/absent operator families.

Writes diagnostics/composition/edit_trace_characterization.json and prints a compact per-layer table.
Pure diagnostics -- reads the GuacaMol subset + the analogue pool, writes one JSON, edits nothing else.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold

from compose_v4.chem.molecular_graph import (ORGANIC_VOCABULARY, MolecularGraphError,
                                             molecular_graph_to_smiles, smiles_to_molecular_graph)
from compose_v4.chem.state import pad_molecular_graph
from compose_v4.rewrite.kernel import de_novo_rewrite_system
from compose_v4.rewrite.source_corruption import make_edit_pair

try:  # reuse the curriculum-stats tercile binner when the scripts dir is on PYTHONPATH
    from analogue_curriculum_stats import _quantile_bins
except Exception:  # noqa: BLE001 -- self-contained fallback (identical semantics)
    def _quantile_bins(lengths: list[int]) -> dict:
        """Empirical-tercile edit-budget bins over path length (data-driven, not preset)."""
        if not lengths:
            return {"axis": "path_length", "method": "empirical_terciles", "edges": [], "bins": []}
        low, high = (int(round(edge)) for edge in np.percentile(lengths, [100 / 3, 200 / 3]))
        high = max(high, low)
        bins = [
            {"name": "local", "predicate": f"length <= {low}",
             "count": sum(1 for n in lengths if n <= low)},
            {"name": "lead_opt", "predicate": f"{low} < length <= {high}",
             "count": sum(1 for n in lengths if low < n <= high)},
            {"name": "scaffold", "predicate": f"length > {high}",
             "count": sum(1 for n in lengths if n > high)},
        ]
        return {"axis": "path_length", "method": "empirical_terciles (data-driven)",
                "edges": [low, high], "bins": bins}


_ROOT = Path(__file__).resolve().parents[1]
GUACAMOL = _ROOT / "results" / "tree_fcd_transfer_stage1_factorized_v1" / \
    "guacamol_heldout_val_5000_seed0.smiles"
ANALOGUE_POOL = _ROOT / "diagnostics" / "composition" / "analogue_trace_pool.jsonl"
OUT_PATH = _ROOT / "diagnostics" / "composition" / "edit_trace_characterization.json"

# ---- Safeguard thresholds (report the raw number regardless; the flag is a coarse trip wire) ----
IDENTITY_BIAS_THRESHOLD = 0.05        # canonical(source) == canonical(target) fraction
SCAFFOLD_GINI_THRESHOLD = 0.60        # concentration of source Bemis-Murcko scaffolds
SCAFFOLD_TOP5_THRESHOLD = 0.25
TRANSFORMATION_TOP5_THRESHOLD = 0.50  # duplication of the transformation signature
RARE_FAMILY_FRACTION = 0.02           # operator families below this share are "near-zero support"
# Families the corruption vocabulary CAN emit (source_corruption docstring); absence is worth flagging.
EXPECTED_CORRUPTION_FAMILIES = (
    "atom_delete", "atom_insert", "atom_restate", "bond_reorder",
    "ring_system_restate", "bond_reroute", "ring_system_delete",
)


# ---- Small numeric helpers ----
def _hist(values) -> dict[str, int]:
    return {str(key): count for key, count in sorted(Counter(values).items())}


def _stats(values, extra_pcts=()) -> dict:
    if not values:
        return {}
    arr = np.asarray(values, dtype=float)
    out = {"min": float(arr.min()), "median": float(np.median(arr)),
           "mean": float(arr.mean()), "max": float(arr.max())}
    for pct in extra_pcts:
        out[f"p{int(pct)}"] = float(np.percentile(arr, pct))
    return out


def _gini(counts) -> float:
    """Gini coefficient over a frequency vector (0 = uniform, ->1 = one label dominates)."""
    arr = np.sort(np.asarray(list(counts), dtype=float))
    n = arr.size
    total = arr.sum()
    if n == 0 or total == 0:
        return 0.0
    index = np.arange(1, n + 1)
    return float((2.0 * np.sum(index * arr)) / (n * total) - (n + 1.0) / n)


def _concentration(labels, keep_examples: int = 5) -> dict:
    counts = Counter(labels)
    total = sum(counts.values())
    if total == 0:
        return {"n_items": 0, "n_unique": 0, "top1_share": 0.0, "top5_share": 0.0, "gini": 0.0,
                "top5": []}
    ordered = counts.most_common()
    top5 = sum(count for _, count in ordered[:5]) / total
    return {
        "n_items": total,
        "n_unique": len(counts),
        "top1_share": ordered[0][1] / total,
        "top5_share": top5,
        "gini": _gini(counts.values()),
        "top5": [[str(label)[:80], count] for label, count in ordered[:keep_examples]],
    }


def _morgan(mol):
    return AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)


def _murcko_from_mol(mol) -> str:
    try:
        return MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
    except Exception:  # noqa: BLE001
        return ""


def _murcko_from_smiles(smiles: str) -> str:
    try:
        return MurckoScaffold.MurckoScaffoldSmiles(smiles=smiles)
    except Exception:  # noqa: BLE001
        return ""


# ---- Layer 1: synthetic corruption traces ----
def build_corruption_records(guacamol_path: Path, n_molecules: int, seed: int) \
        -> tuple[list[dict], dict]:
    """Walk the first ``n_molecules`` parseable GuacaMol molecules and record both edit directions."""
    rng = np.random.default_rng(seed)
    system = de_novo_rewrite_system()
    records: list[dict] = []
    parse_skips = 0
    empty_pairs = 0
    used = 0
    with open(guacamol_path) as handle:
        for line in handle:
            token = line.split()
            if not token:
                continue
            smiles = token[0]
            try:
                base = smiles_to_molecular_graph(smiles)
            except MolecularGraphError:  # unsupported element / parse / H-count
                parse_skips += 1
                continue
            except Exception:  # noqa: BLE001 -- stay robust on any unexpected parse issue
                parse_skips += 1
                continue
            n_slots = int(base.n_real_atoms) + 8
            target = pad_molecular_graph(base, n_slots)
            depth = int(rng.integers(1, 6))
            try:
                trim, grow = make_edit_pair(
                    target, depth, system=system, rng=rng, vocabulary=ORGANIC_VOCABULARY)
            except Exception:  # noqa: BLE001 -- keep the sweep robust
                empty_pairs += 1
                used += 1
                if used >= n_molecules:
                    break
                continue
            got = False
            for trace, direction, orientation in (
                (trim, "real->corrupt", "forward"),
                (grow, "corrupt->real", "inverse"),
            ):
                if trace is None:
                    continue
                src = molecular_graph_to_smiles(trace.source)
                tgt = molecular_graph_to_smiles(trace.target)
                if not src or not tgt:
                    continue
                histogram = dict(Counter(step.rule_name for step in trace.steps))
                signature = "C|" + "|".join(f"{k}:{v}" for k, v in sorted(histogram.items()))
                records.append({
                    "layer": "corruption",
                    "direction": direction,
                    "orientation": orientation,
                    "source_smiles": src,
                    "target_smiles": tgt,
                    "operator_histogram": histogram,
                    "path_length": len(trace.steps),
                    "signature": signature,
                })
                got = True
            if not got:
                empty_pairs += 1
            used += 1
            if used >= n_molecules:
                break
    meta = {"molecules_used": used, "parse_skips": parse_skips, "empty_pairs": empty_pairs}
    return records, meta


# ---- Layer 2: MMP analogue traces ----
def build_analogue_records(pool_path: Path) -> tuple[list[dict], dict]:
    records: list[dict] = []
    with open(pool_path) as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            src = row["source_smiles"]
            tgt = row["target_smiles"]
            direction = row.get("direction", "A->B")
            signature = "A|" + f"{_murcko_from_smiles(src)}>>{_murcko_from_smiles(tgt)}"
            records.append({
                "layer": "analogue",
                "direction": direction,
                "orientation": "forward" if direction == "A->B" else "inverse",
                "source_smiles": src,
                "target_smiles": tgt,
                "operator_histogram": dict(row.get("operator_histogram", {})),
                "path_length": int(row["path_length"]),
                "signature": signature,
            })
    return records, {"pool_records": len(records)}


# ---- Metrics over a homogeneous record list ----
def characterize(records: list[dict]) -> dict:
    rows = []
    rdkit_skipped = 0
    for rec in records:
        source_mol = Chem.MolFromSmiles(rec["source_smiles"])
        target_mol = Chem.MolFromSmiles(rec["target_smiles"])
        if source_mol is None or target_mol is None:
            rdkit_skipped += 1
            continue
        rows.append((rec, source_mol, target_mol))
    if not rows:
        return {"trace_count": 0, "rdkit_skipped": rdkit_skipped}

    n = len(rows)
    # 1. counts + forward/inverse balance
    direction_counts = dict(Counter(rec["direction"] for rec, _, _ in rows))
    orientation_counts = dict(Counter(rec["orientation"] for rec, _, _ in rows))

    # 2. operator families
    op_totals: Counter = Counter()
    ops_per_trace = []
    for rec, _, _ in rows:
        op_totals.update(rec["operator_histogram"])
        ops_per_trace.append(sum(rec["operator_histogram"].values()))
    total_ops = sum(op_totals.values()) or 1
    families = sorted(op_totals)
    operator_family = {
        "totals": dict(op_totals.most_common()),
        "fractions": {fam: op_totals[fam] / total_ops for fam in families},
        "per_trace_mean": {
            fam: float(np.mean([rec["operator_histogram"].get(fam, 0) for rec, _, _ in rows]))
            for fam in families
        },
        "ops_per_trace_mean": float(np.mean(ops_per_trace)),
        "n_families": len(families),
    }

    # 3. path length
    lengths = [rec["path_length"] for rec, _, _ in rows]
    path_length = _stats(lengths, extra_pcts=(25, 75))
    path_length["hist"] = _hist(lengths)
    path_length["curriculum_bins"] = _quantile_bins(lengths)

    # 4. source<->target Tanimoto
    sims = [float(DataStructs.TanimotoSimilarity(_morgan(s), _morgan(t))) for _, s, t in rows]
    tanimoto = _stats(sims)
    tanimoto["frac_in_recognizable_band_0.4_0.9"] = float(
        np.mean([0.4 <= s <= 0.9 for s in sims]))

    # 5. identity fraction (IDENTITY-BIAS check)
    identical = sum(1 for _, s, t in rows if Chem.MolToSmiles(s) == Chem.MolToSmiles(t))
    identity_fraction = identical / n

    # 6. ring complexity + ring-change fraction
    src_rings = [Descriptors.RingCount(s) for _, s, _ in rows]
    tgt_rings = [Descriptors.RingCount(t) for _, _, t in rows]
    src_arom = [rdMolDescriptors.CalcNumAromaticRings(s) for _, s, _ in rows]
    tgt_arom = [rdMolDescriptors.CalcNumAromaticRings(t) for _, _, t in rows]
    ring_complexity = {
        "source_ring_count": _stats(src_rings),
        "target_ring_count": _stats(tgt_rings),
        "source_aromatic_rings": _stats(src_arom),
        "target_aromatic_rings": _stats(tgt_arom),
        "ring_change_fraction": float(np.mean([a != b for a, b in zip(src_rings, tgt_rings)])),
        "aromatic_ring_change_fraction": float(
            np.mean([a != b for a, b in zip(src_arom, tgt_arom)])),
    }

    # 7. size distribution
    src_heavy = [s.GetNumHeavyAtoms() for _, s, _ in rows]
    tgt_heavy = [t.GetNumHeavyAtoms() for _, _, t in rows]
    size_distribution = {"source_heavy_atoms": _stats(src_heavy),
                         "target_heavy_atoms": _stats(tgt_heavy)}

    # 8. scaffold concentration over SOURCE molecules (FREQUENT-SCAFFOLD-DOMINATION check)
    source_scaffolds = [_murcko_from_mol(s) for _, s, _ in rows]
    scaffold_concentration = _concentration(source_scaffolds)

    # 9. transformation concentration (TRANSFORMATION-DUPLICATION check)
    transformation_concentration = _concentration(rec["signature"] for rec, _, _ in rows)

    # 10. charged fraction (source OR target carries a nonzero formal charge)
    def _charged(mol) -> bool:
        return any(atom.GetFormalCharge() != 0 for atom in mol.GetAtoms())
    charged_fraction = float(np.mean([_charged(s) or _charged(t) for _, s, t in rows]))

    # ---- safeguards ----
    rare = {fam: operator_family["fractions"][fam]
            for fam in families if operator_family["fractions"][fam] < RARE_FAMILY_FRACTION}
    missing_expected = [fam for fam in EXPECTED_CORRUPTION_FAMILIES if fam not in op_totals] \
        if records and records[0]["layer"] in ("corruption", "overall") else []
    safeguards = {
        "identity_bias": {
            "triggered": identity_fraction > IDENTITY_BIAS_THRESHOLD,
            "identity_fraction": identity_fraction, "threshold": IDENTITY_BIAS_THRESHOLD},
        "scaffold_domination": {
            "triggered": (scaffold_concentration["gini"] > SCAFFOLD_GINI_THRESHOLD
                          or scaffold_concentration["top5_share"] > SCAFFOLD_TOP5_THRESHOLD),
            "gini": scaffold_concentration["gini"],
            "top1_share": scaffold_concentration["top1_share"],
            "top5_share": scaffold_concentration["top5_share"]},
        "transformation_duplication": {
            "triggered": transformation_concentration["top5_share"] > TRANSFORMATION_TOP5_THRESHOLD,
            "top5_share": transformation_concentration["top5_share"],
            "n_unique": transformation_concentration["n_unique"]},
        "rare_operator_families": {
            "triggered": bool(rare) or bool(missing_expected),
            "rare_fractions": rare, "missing_expected": missing_expected},
    }

    return {
        "trace_count": n,
        "rdkit_skipped": rdkit_skipped,
        "direction_counts": direction_counts,
        "orientation_counts": orientation_counts,
        "operator_family": operator_family,
        "path_length": path_length,
        "source_target_tanimoto": tanimoto,
        "identity_fraction": identity_fraction,
        "ring_complexity": ring_complexity,
        "size_distribution": size_distribution,
        "scaffold_concentration": scaffold_concentration,
        "transformation_concentration": transformation_concentration,
        "charged_fraction": charged_fraction,
        "safeguards": safeguards,
    }


# ---- Reporting ----
def _print_layer(name: str, report: dict) -> None:
    if report.get("trace_count", 0) == 0:
        print(f"\n=== {name}: NO TRACES (rdkit_skipped={report.get('rdkit_skipped', 0)}) ===",
              flush=True)
        return
    op = report["operator_family"]
    pl = report["path_length"]
    tan = report["source_target_tanimoto"]
    ring = report["ring_complexity"]
    scaf = report["scaffold_concentration"]
    trans = report["transformation_concentration"]
    print(f"\n=== {name} | {report['trace_count']} traces "
          f"(rdkit_skipped={report['rdkit_skipped']}) ===", flush=True)
    print(f"  balance        : {report['orientation_counts']} | dirs {report['direction_counts']}",
          flush=True)
    top_ops = sorted(op["fractions"].items(), key=lambda kv: -kv[1])[:6]
    print("  operator mix   : "
          + ", ".join(f"{fam} {100 * frac:.0f}%" for fam, frac in top_ops)
          + f"  ({op['ops_per_trace_mean']:.1f} ops/trace)", flush=True)
    print(f"  path length    : min={pl['min']:.0f} p25={pl['p25']:.0f} med={pl['median']:.0f} "
          f"mean={pl['mean']:.1f} p75={pl['p75']:.0f} max={pl['max']:.0f} | "
          f"tercile edges {pl['curriculum_bins']['edges']}", flush=True)
    print(f"  src<->tgt Tanimoto: med={tan['median']:.3f} mean={tan['mean']:.3f} "
          f"[min {tan['min']:.3f}, max {tan['max']:.3f}] | "
          f"in [0.4,0.9] band {100 * tan['frac_in_recognizable_band_0.4_0.9']:.0f}%", flush=True)
    print(f"  IDENTITY frac  : {100 * report['identity_fraction']:.1f}%   "
          f"charged frac {100 * report['charged_fraction']:.1f}%", flush=True)
    print(f"  ring change    : {100 * ring['ring_change_fraction']:.0f}% "
          f"(aromatic {100 * ring['aromatic_ring_change_fraction']:.0f}%) | "
          f"src rings mean {ring['source_ring_count']['mean']:.2f} "
          f"tgt {ring['target_ring_count']['mean']:.2f}", flush=True)
    print(f"  size (heavy)   : src med {report['size_distribution']['source_heavy_atoms']['median']:.0f} "
          f"tgt med {report['size_distribution']['target_heavy_atoms']['median']:.0f}", flush=True)
    print(f"  src scaffolds  : {scaf['n_unique']} unique / {scaf['n_items']} | "
          f"top1 {100 * scaf['top1_share']:.0f}% top5 {100 * scaf['top5_share']:.0f}% "
          f"Gini {scaf['gini']:.2f}", flush=True)
    print(f"  transformations: {trans['n_unique']} unique | top5 {100 * trans['top5_share']:.0f}%",
          flush=True)
    flags = [key for key, val in report["safeguards"].items() if val.get("triggered")]
    print(f"  SAFEGUARDS     : {'TRIGGERED -> ' + ', '.join(flags) if flags else 'none triggered'}",
          flush=True)
    rof = report["safeguards"]["rare_operator_families"]
    if rof["missing_expected"] or rof["rare_fractions"]:
        print(f"     rare/absent operators: missing={rof['missing_expected']} "
              f"rare={ {k: round(v, 4) for k, v in rof['rare_fractions'].items()} }", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-molecules", type=int, default=350,
                        help="number of parseable GuacaMol molecules to corrupt")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--guacamol", type=Path, default=GUACAMOL)
    parser.add_argument("--analogue-pool", type=Path, default=ANALOGUE_POOL)
    parser.add_argument("--out", type=Path, default=OUT_PATH)
    args = parser.parse_args()

    print(f"building LAYER 1 corruption traces from {args.n_molecules} GuacaMol molecules ...",
          flush=True)
    corruption_records, corruption_meta = build_corruption_records(
        args.guacamol, args.n_molecules, args.seed)
    print(f"building LAYER 2 analogue traces from {args.analogue_pool} ...", flush=True)
    analogue_records, analogue_meta = build_analogue_records(args.analogue_pool)
    overall_records = corruption_records + analogue_records
    # tag as "overall" so the pooled report still runs the corruption missing-family safeguard
    overall_tagged = [dict(rec, layer="overall") for rec in overall_records]

    report = {
        "corruption": characterize(corruption_records),
        "analogue": characterize(analogue_records),
        "overall": characterize(overall_tagged),
        "meta": {
            "corruption_traces": len(corruption_records),
            "analogue_traces": len(analogue_records),
            "guacamol_corpus": str(args.guacamol),
            "analogue_pool": str(args.analogue_pool),
            "n_guacamol_used": corruption_meta["molecules_used"],
            "guacamol_parse_skips": corruption_meta["parse_skips"],
            "corruption_empty_pairs": corruption_meta["empty_pairs"],
            "morgan_radius": 2, "morgan_nbits": 2048, "seed": args.seed,
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))

    _print_layer("LAYER 1 corruption", report["corruption"])
    _print_layer("LAYER 2 analogue (MMP)", report["analogue"])
    _print_layer("OVERALL (pooled)", report["overall"])
    print(f"\nwrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
