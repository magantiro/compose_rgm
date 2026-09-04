#!/usr/bin/env python3
"""Audit one T4 round: did Q(M|x,z) actually spread the oracle, and did the
large-scope bundles realize large structural change?

The question this answers is NOT "what did it score". It is whether the
implementation still behaves like the local-to-global controller:

  * bundle identity is the unit of outer-controller effort, so the 20 dockings
    should land on ~20 DIFFERENT (parent, region) bundles, not 19 siblings off
    one frontier;
  * a region with r_release 0.8 that yields an intermediate with r_coherent
    0.03 is a small realized change along a global-region trajectory, and must
    be reported as such rather than as an 80%-scale rewrite.

Usage: python3 tools/t4_audit.py [cellfile.json]
"""
import json, sys, glob
from collections import Counter

f = sys.argv[1] if len(sys.argv) > 1 else sorted(glob.glob('/tmp/t4a/*.json'))[0]
r = json.load(open(f))
print(f"cell={r['cell']}  n_dock={r['n_dock']}  best={r['best_ds']}")
for x in r.get('rounds', []):
    dk = x.get('docked', [])
    print(f"\n=== round {x['round']} ===")
    print(f"  bundles selected={x.get('n_bundles_selected')}  "
          f"bundles represented among the 20 docked={x.get('n_bundles_docked')}")
    print(f"  bundle scopes offered: {x.get('bundle_scopes')}")
    print(f"  candidates harvested={x['n_cand']}  docked={len(dk)}")
    if not dk:
        continue
    print(f"  distinct canonical molecules docked: {len({a['smi'] for a in dk})}")
    print(f"  INTENDED scope (r_release): {Counter(round(a['r_release'],1) for a in dk)}")
    rc = [a.get('r_coherent') or 0 for a in dk]
    print(f"  REALIZED coherent change:  min={min(rc):.2f} med="
          f"{sorted(rc)[len(rc)//2]:.2f} max={max(rc):.2f}")
    print(f"  ring-system deltas: {Counter(a.get('d_rings') for a in dk)}")
    print(f"  interfaces: {Counter(a['interface'] for a in dk)}")
    big = [a for a in dk if (a.get('r_coherent') or 0) >= 0.2]
    print(f"  candidates with >=0.20 realized coherent change: {len(big)}")
    for a in sorted(dk, key=lambda a: -(a.get('r_coherent') or 0))[:4]:
        print(f"    r_rel={a['r_release']:.2f} -> r_coh={(a.get('r_coherent') or 0):.2f} "
              f"d_rings={a.get('d_rings')} step={a.get('step')} ds={a['ds']}")
