"""REFINE_RING provenance smoke: one shard per scaffold, run in parallel."""
import json, os, time, modal
SEEDS = ["FC(F)(F)c1cc(N2CC[NH2+]CC2)ccc1",   # 5HT1B witness
         "Oc1ccccc1",                          # phenol
         "CC(=O)Nc1ccccc1",                    # acetanilide
         "OCC1CCCCC1",                         # aliphatic, no aromatic scaffold
         "c1ccc2[nH]ccc2c1"]                   # indole
f = modal.Function.from_name("aryl-compile", "refine_provenance")
jobs = [dict(seeds=[s], cap=60, seed_rng=11 + 5 * i) for i, s in enumerate(SEEDS)]
t0 = time.time()
res = [r for r in f.map(jobs, order_outputs=False, return_exceptions=True)]
rows = [x for r in res if isinstance(r, dict) for x in r.get("results", [])]
print(f"done in {time.time()-t0:.0f}s   scaffolds={len(rows)}\n")
allok = True
for d in rows:
    if not d.get("built"):
        print(f"  {d['seed']:32s}  ring build failed (skipped)"); continue
    checks = {
        "partition exact":        d["partition_exact"],
        "admitted perturb ring":  d["admitted_perturb_ring"].split("/")[0] == d["admitted_perturb_ring"].split("/")[1],
        "zero off-ring leaks":    d["off_ring_leaks_into_R"].split("/")[0] == "0",
        "fiber support local":    d["fiber_support_subset_of_local"],
        "q sums to 1":            d["q_sums_to_one"],
        "eps0 == Rtheta^(1/T)":   d["eps0_matches_rtheta_pow_1_over_T"],
        "monotone in Rtheta":     d["monotone_in_rtheta"],
    }
    ok = all(checks.values()); allok = allok and ok
    print(f"  {d['seed']}")
    print(f"     ring R={d['R']}  restate actions={d['n_restate']}  ring-local={d['n_ring_local']}  off-ring={d['n_off_ring']}")
    print(f"     admitted perturb R: {d['admitted_perturb_ring']}    off-ring leaks into R: {d['off_ring_leaks_into_R']}")
    print("     " + "  ".join(f"{k}={'OK' if v else 'FAIL'}" for k, v in checks.items()))
    print(f"     ring-local fiber: {d['n_local_applied']} actions -> {d['n_local_canonical']} distinct canonical successors (max multiplicity {d['max_fiber_multiplicity']})")
    print(f"     refine2 -> {d['refine2_smiles']}   {d['refine2_trace']}")
    print()
json.dump(rows, open("diagnostics/refine_provenance.json", "w"), indent=2)
print("ALL CHECKS PASS" if allok else "SOME CHECKS FAILED")
print("wrote diagnostics/refine_provenance.json")
