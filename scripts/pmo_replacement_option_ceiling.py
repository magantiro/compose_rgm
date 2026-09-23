"""What is the CEILING of the region-replacement option, before any lane delivers it?

Three region laws have now measured null end to end (``diagnostics/pmo_region_law_v1/``):
uncapped uniform, uncapped uniform with an uncapped replacement, and scale-balanced.  The
autopsy of that null is structural rather than chemical.  ``_weighted_module_order`` draws a
random permutation over the thirteen generic families and accepts the first that executes;
only ``segment_replace`` and ``substituent_delete`` consult a region law at all, and the
median program is one or two modules.  So a law governs a small minority of module draws and
cannot move a program-level statistic however well it is designed.

The proposed repair is to make the region replacement an OPTION -- an intent committed
BEFORE the family lottery -- rather than a law inside it.  This probe measures what such an
option could deliver if a lane selected it with certainty, which is a strict upper bound on
any lane that selects it sometimes.  If forcing the option does not reach the required
structural scale, no lane built to deliver it will either, and the build is not worth making.

ARM A  region_law=None       forced segment_replace, v1's cap of 8
ARM B  uncapped uniform      forced segment_replace, BridgeRegionLaw(maximum=None)
ARM C  scale-balanced        forced segment_replace, ScaleBalancedRegionLaw()

Matched RNG seeds across arms; only the law differs.  Every arm forces the same family, so
this isolates the law from the lottery that diluted it.  Zero oracle calls; no task
information, no similarity reference, no target of any kind enters any draw.
"""
from __future__ import annotations

import json
import sys
import time

import networkx as nx
import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFMCS

from compose_v4.chem.molecular_graph import smiles_to_molecular_graph
from compose_v4.chem.state import pad_molecular_graph
from compose_v4.control.bridge_region_law import BridgeRegionLaw
from compose_v4.control.dynamic_program_synthesis import synthesize_named_module_sequence
from compose_v4.control.scale_balanced_region_law import ScaleBalancedRegionLaw

RDLogger.DisableLog("rdApp.*")

SCHEMA = "pmo_replacement_option_ceiling_v1"
SEGMENTS = "/Users/rmaganti/compose_pmo_macro_data/segments.json"
OUT = "/Users/rmaganti/compose_pmo_macro_data/replacement_option_ceiling_v1.json"


def delta(a_smi, b_smi, timeout=5):
    """MCS-based structural delta; the same instrument the earlier gates used."""
    a_mol, b_mol = Chem.MolFromSmiles(a_smi), Chem.MolFromSmiles(b_smi)
    if a_mol is None or b_mol is None:
        return None
    res = rdFMCS.FindMCS([a_mol, b_mol], timeout=timeout, ringMatchesRingOnly=True)
    if res.canceled or res.numAtoms == 0:
        core, matched = 0, set()
    else:
        query = Chem.MolFromSmarts(res.smartsString)
        match = b_mol.GetSubstructMatch(query)
        core, matched = res.numAtoms, set(match) if match else set()
    changed = [a.GetIdx() for a in b_mol.GetAtoms() if a.GetIdx() not in matched]
    graph = nx.Graph()
    graph.add_nodes_from(changed)
    changed_set = set(changed)
    for bond in b_mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if i in changed_set and j in changed_set:
            graph.add_edge(i, j)
    comps = [len(c) for c in nx.connected_components(graph)] if changed else []
    return {
        "largest_changed_region": max(comps) if comps else 0,
        "n_changed_regions": len(comps),
        "retained_fraction_mcs": core / max(a_mol.GetNumHeavyAtoms(), 1),
        "d_heavy": b_mol.GetNumHeavyAtoms() - a_mol.GetNumHeavyAtoms(),
        "d_rings": b_mol.GetRingInfo().NumRings() - a_mol.GetRingInfo().NumRings(),
    }


def arm(src_smi, src, draws, law, seed):
    rng = np.random.default_rng(seed)
    rows, endpoints, failures = [], set(), 0
    for _ in range(draws):
        try:
            _s, program, _a, trace, _m = synthesize_named_module_sequence(
                src, rng, ("segment_replace",), region_law=law
            )
        except (ValueError, RuntimeError):
            failures += 1
            continue
        endpoint = trace.get("endpoint")
        if not endpoint:
            continue
        endpoints.add(endpoint)
        measured = delta(src_smi, endpoint)
        if measured:
            rows.append(
                {
                    **measured,
                    "k": trace.get("primitive_edits", len(program.marks)),
                    "endpoint": endpoint,
                }
            )
    return rows, len(endpoints), failures


def summarise(rows, unique, failures, draws, label):
    if not rows:
        return {"arm": label, "draws": draws, "executed": 0, "failures": failures}

    def col(key):
        return np.array([r[key] for r in rows], dtype=float)

    largest = col("largest_changed_region")
    return {
        "arm": label,
        "draws": draws,
        "executed": len(rows),
        "failures": failures,
        "executable_yield": len(rows) / draws,
        "distinct_endpoints": unique,
        "largest_changed_region_median": float(np.median(largest)),
        "largest_changed_region_p90": float(np.percentile(largest, 90)),
        "largest_changed_region_max": float(largest.max()),
        "fraction_largest_ge_12": float((largest >= 12).mean()),
        "fraction_largest_ge_15": float((largest >= 15).mean()),
        "retained_fraction_mcs_median": float(np.median(col("retained_fraction_mcs"))),
        "n_changed_regions_median": float(np.median(col("n_changed_regions"))),
        "d_heavy_median": float(np.median(col("d_heavy"))),
        "d_rings_median": float(np.median(col("d_rings"))),
        "k_median": float(np.median(col("k"))),
    }


def main():
    draws = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    with open(SEGMENTS) as handle:
        segments = json.load(handle)
    seen, sources = set(), []
    for group in segments:
        if group["source_smiles"] not in seen:
            seen.add(group["source_smiles"])
            sources.append(group)
    laws = (
        ("A_none", None),
        ("B_uncapped", BridgeRegionLaw(maximum=None)),
        ("C_scale_balanced", ScaleBalancedRegionLaw()),
    )
    out = {
        "schema_version": SCHEMA,
        "evidence_role": "zero_oracle_option_ceiling_measurement",
        "new_charged_oracle_calls": 0,
        "question": (
            "Forcing the region-replacement option on every draw, does the realized "
            "structural scale reach the band a productive macro occupies?  This is an "
            "upper bound on any lane that selects the option with probability < 1."
        ),
        "arms": {
            "A_none": "forced segment_replace, region_law=None (v1 cap of 8)",
            "B_uncapped": "forced segment_replace, BridgeRegionLaw(maximum=None)",
            "C_scale_balanced": "forced segment_replace, ScaleBalancedRegionLaw()",
        },
        "baseline_from_census": {"largest_changed_region": 7, "retained_fraction": 0.64},
        "band_from_witnesses": {"largest_changed_region": 19, "retained_fraction": 0.30},
        "band_is_reference_not_target": (
            "The witness numbers are answer-known and are quoted to say where a "
            "productive macro sits.  No constant from them appears in any law or in "
            "the sampler; they are never optimized against."
        ),
        "sources": [],
    }
    for index, group in enumerate(sources):
        src = pad_molecular_graph(smiles_to_molecular_graph(group["source_smiles"]), 48)
        assert len(src.atom_types) == 48, "PMO proposal states are 48 slots"
        record = {
            "route": group["route"],
            "seg": group["seg"],
            "source_smiles": group["source_smiles"],
            "source_heavy_atoms": int(src.n_real_atoms),
            "arms": [],
        }
        for label, law in laws:
            started = time.time()
            rows, unique, failures = arm(
                group["source_smiles"], src, draws, law, 20260923 + 1009 * index
            )
            summary = summarise(rows, unique, failures, draws, label)
            summary["seconds"] = round(time.time() - started, 1)
            summary["endpoints"] = [r["endpoint"] for r in rows[:40]]
            record["arms"].append(summary)
            print(
                f"{group['route'][:16]:16s} s{group['seg']} {label:17s} "
                f"| yield {summary.get('executable_yield', 0):.3f} "
                f"| largest med {summary.get('largest_changed_region_median', 0):5.1f} "
                f"p90 {summary.get('largest_changed_region_p90', 0):5.1f} "
                f"max {summary.get('largest_changed_region_max', 0):5.1f} "
                f"| >=12 {summary.get('fraction_largest_ge_12', 0):.3f} "
                f"| retained {summary.get('retained_fraction_mcs_median', 0):.3f} "
                f"| {summary['seconds']:.0f}s",
                flush=True,
            )
        out["sources"].append(record)
        with open(OUT, "w") as handle:
            json.dump(out, handle, indent=1)
    print("WROTE", OUT, flush=True)


if __name__ == "__main__":
    main()
