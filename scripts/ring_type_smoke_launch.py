"""REFINE_RING contract smoke: 3 production ring types x 3 seeds.

CONTRACT UNDER TEST
    admit a  <=>  declared_locus(a) & R != {}         R = the exact constructed
                                                      cycle: fr.path + (u, v)

An admitted action MAY induce bond-order / aromaticity changes outside R within
the SAME fused system -- retyping an atom in a newly fused ring forces RDKit to
reshuffle Kekule bonds across the whole block. It may NOT retype an atom or
change connectivity outside R.

Parent-ring actions that are EXCLUDED but whose effects would reach R are
`coupled_omissions`: correct conservative behaviour, logged and quantified, not
scored as failures. A builder UNSAT is likewise not a REFINE_RING failure.
"""
import json, time, modal

SEEDS = ["FC(F)(F)c1cc(N2CC[NH2+]CC2)ccc1", "Oc1ccccc1", "CC(=O)Nc1ccccc1"]
SPECS = [dict(topology="pendant", size=6, state="aromatic"),
         dict(topology="fused",   size=6, state="aromatic"),
         dict(topology="fused",   size=6, state="saturated")]
jobs = [dict(seed=s, spec=sp) for sp in SPECS for s in SEEDS]

f = modal.Function.from_name("aryl-compile", "ring_type_smoke")
t0 = time.time()
res = [r for r in f.map(jobs, order_outputs=True, return_exceptions=True)]
rows = [r for r in res if isinstance(r, dict)]
crashes = len(res) - len(rows)
print(f"done in {time.time()-t0:.0f}s")
print(f"code versions that ran: {sorted({r.get('code_version') for r in rows})}")
print(f"crashes: {crashes}\n")

by_type = {}
for r in rows:
    by_type.setdefault(r["ring_type"], []).append(r)

fails, omissions_total = 0, 0
for rt, rs in by_type.items():
    print(f"  {rt}   builds OK on {sum(1 for r in rs if r['new0_status']=='OK')}/{len(rs)} seeds")
    for r in rs:
        print(f"     {r['seed']:34s} build={r['new0_status']:5s} refine0-parity="
              f"{'OK' if r['refine0_parity_exact'] else 'FAIL'}")
        if not r["refine0_parity_exact"]:
            fails += 1
        if r["new0_status"] != "OK":
            print("        builder UNSAT on this seed - pre-existing, not a REFINE_RING failure")
            continue
        print(f"        refine=2 -> {r['ref2_smiles']}")
        if not r.get("fiber_checked"):
            print(f"        fiber NOT checked: {r.get('why')}  -> FAIL"); fails += 1; continue
        adm = r["admitted_declared_meets_R"]
        chk = {
            "1 locus == constructed cycle == 6":
                r["expected_locus"] == r["actual_locus"] == 6 and not r["off_cycle_in_scope"],
            "2 every admitted declares locus in R":
                adm.split("/")[0] == adm.split("/")[1],
            "3 no admitted action disjoint from R":
                r["excluded_declared_touching_R"] == 0,
            "4a no retype outside R":
                not r["retype_outside_R"],
            "4b no connectivity change outside R":
                not r["connectivity_change_outside_R"],
            "4c no change outside ring system":
                not r["change_outside_ring_system"],
        }
        co = r["coupled_omissions"]; omissions_total += int(co.split("/")[0])
        print(f"        new {r['new_atoms']} + fusion {r['fusion_atoms']} = locus "
              f"{r['actual_locus']}/{r['expected_locus']}   ring system {r['ring_system']}")
        print(f"        admitted declared-in-R {adm}   fiber {r['n_ring_local']} -> "
              f"{r['n_local_canonical']} canonical (maxmult {r['max_fiber_multiplicity']})")
        print(f"        coupled_omissions {co} from parent atoms {r['coupled_omission_sources']}"
              f"  (correctly EXCLUDED, not failures)")
        bad = [k for k, v in chk.items() if not v]
        if bad:
            fails += 1
            for k in bad:
                print(f"        FAIL {k}")
        print(f"        -> {'OK' if not bad else 'FAIL'}")
    print()

covered = [rt for rt, rs in by_type.items() if any(r["new0_status"] == "OK" for r in rs)]
json.dump(rows, open("diagnostics/ring_type_smoke.json", "w"), indent=2)
print(f"ring types with >=1 OK build: {len(covered)}/3  {covered}")
print(f"total coupled_omissions logged (expected, not failures): {omissions_total}")
print(f"failures: {fails}   crashes: {crashes}")
print("PASS" if (fails == 0 and crashes == 0 and len(covered) == 3) else "FAIL")
