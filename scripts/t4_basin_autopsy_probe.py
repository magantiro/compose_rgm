"""Diagnostic 2: does q0 ever propose the transformation the strong basin needs?

Measured on the POOL (pre-selection), so it separates "never proposed" from
"proposed but not chosen". Zero oracle calls.
"""
import json, sys, time
from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")
from compose_v4.experiments.t4_fiber_expansion import expand_frontier

ROOT = "COC(=O)CC2Nc1ccccc1c3ccnc4[nH]cc2c34"
DRAWS = int(sys.argv[1]) if len(sys.argv) > 1 else 2400

FEATURES = {
    "ester broken (no C-O-C(=O)-CH2)": None,
    "amide linkage N-C(=O)-CH2":       "[NX3][CX3](=O)[CH2]",
    "ring N-C-N":                      "[NX3;R][CX4;R][NX3;R]",
    "ring N-C-C-N (piperazine)":       "[NX3;R][CX4;R][CX4;R][NX3;R]",
    "urea/carbamoyl on ring N":        "[NX3;R][CX3](=O)[NX3]",
    "BASIN = amide + diamine ring":    None,
}
ester = Chem.MolFromSmarts("[CX4][OX2][CX3](=O)[CH2]")
amide = Chem.MolFromSmarts("[NX3][CX3](=O)[CH2]")
ncn   = Chem.MolFromSmarts("[NX3;R][CX4;R][NX3;R]")
nccn  = Chem.MolFromSmarts("[NX3;R][CX4;R][CX4;R][NX3;R]")
urea  = Chem.MolFromSmarts("[NX3;R][CX3](=O)[NX3]")

t = time.time()
pool = expand_frontier([(ROOT, -8.1)], ROOT, 0.6, draws=DRAWS, workers=10, seed=4242)
print(f"free pool from the ROOT: {len(pool)} feasible endpoints "
      f"from {DRAWS} raw programs in {time.time()-t:.0f}s", flush=True)

counts = dict.fromkeys(
    ["ester_broken","amide","ncn","nccn","urea","basin"], 0)
for smi in pool:
    m = Chem.MolFromSmiles(smi)
    if m is None: continue
    has_ester = m.HasSubstructMatch(ester)
    has_amide = m.HasSubstructMatch(amide)
    has_ring  = m.HasSubstructMatch(ncn) or m.HasSubstructMatch(nccn)
    counts["ester_broken"] += (not has_ester)
    counts["amide"] += has_amide
    counts["ncn"]   += m.HasSubstructMatch(ncn)
    counts["nccn"]  += m.HasSubstructMatch(nccn)
    counts["urea"]  += m.HasSubstructMatch(urea)
    counts["basin"] += (has_amide and has_ring)
n = len(pool)
print(f"\n{'feature':<36}{'count':>8}{'rate':>9}   {'strong basin':>13}")
rows = [("ester broken", "ester_broken", 1.00), ("amide linkage", "amide", 1.00),
        ("ring N-C-N", "ncn", 1.00), ("ring N-C-C-N", "nccn", 0.86),
        ("urea/carbamoyl on ring N", "urea", 0.71),
        ("BASIN: amide + diamine ring", "basin", 1.00)]
for label, key, target in rows:
    print(f"{label:<36}{counts[key]:>8}{100*counts[key]/max(n,1):8.2f}%{100*target:12.0f}%")
json.dump({"pool": len(pool), "draws": DRAWS, "counts": counts,
           "pool_smiles": list(pool)},
          open("diagnostics/t4_fiber_control/diag2_root_proposal_rates.json","w"), indent=1)
