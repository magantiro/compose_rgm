#!/usr/bin/env python3
"""Audit one T4 round of the three-level COMPOSE controller.

The question this answers is NOT "what did it score". It is whether the
implementation still behaves like the local-to-global controller:

  * bundle identity is the unit of outer-controller effort, so the 20 dockings
    should land on distinct (parent, region, option) bundles, not siblings from
    one frontier;
  * a region with r_release 0.8 that yields an intermediate with r_coherent
    0.03 is a small realized change along a global-region trajectory, and must
    be reported as such rather than as an 80%-scale rewrite.

  * constructive options must be audited by topology change, never inferred
    merely from their names or from docking score.

Usage: python3 tools/t4_audit.py [cellfile.json]
"""

import glob
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

f = sys.argv[1] if len(sys.argv) > 1 else min(glob.glob("/tmp/t4a/*.json"))
with Path(f).open() as handle:
    r = json.load(handle)
print(
    f"schema={r.get('schema_version')}  cell={r['cell']}  n_dock={r['n_dock']}  best={r['best_ds']}"
)
ctl = r.get("controller", {})
print(
    f"controller={ctl.get('factorization')}  region_prior={ctl.get('region_prior')}  "
    f"kappa={ctl.get('kappa')}  option_floor={ctl.get('epsilon_option')}"
)
for x in r.get("rounds", []):
    dk = x.get("docked", [])
    print(f"\n=== round {x['round']} ===")
    print(
        f"  parent batches={x.get('n_parent_batches')}  "
        f"region draws={x.get('n_region_draws')}  "
        f"bundles selected={x.get('n_bundles_selected')}  "
        f"bundles represented among the 20 docked={x.get('n_bundles_docked')}"
    )
    print(f"  bundle scopes offered: {x.get('bundle_scopes')}")
    print(f"  options selected: {x.get('options_selected')}")
    print(f"  options docked: {x.get('options_docked')}")
    print(f"  candidates by option: {x.get('candidates_by_option')}")
    print(f"  candidate structural outcomes: {x.get('candidate_option_diagnostics')}")
    print(f"  docked structural outcomes: {x.get('docked_option_diagnostics')}")
    print(
        f"  candidates harvested={x['n_cand']}  "
        f"unique={x.get('n_unique_candidates')}  docked={len(dk)}"
    )
    print(
        f"  candidate diversity={x.get('candidate_diversity')}  "
        f"docked diversity={x.get('docked_diversity')}"
    )
    print(f"  proposal time={x.get('t_propose')}s  docking time={x.get('t_dock')}s")

    bundles = x.get("bundles", [])
    compound = [b for b in bundles if b.get("option") == "build_ring_system"]
    if compound:
        print(
            f"  BUILD_RING_SYSTEM complete={sum(bool(b.get('program_complete')) for b in compound)}"
            f"/{len(compound)}; candidates={sum(int(b.get('n_candidates', 0)) for b in compound)}"
        )
        for bundle in compound:
            print(
                f"    lineage={bundle.get('parent_lineage_id')} "
                f"search_step={bundle.get('max_search_step')} "
                f"emitted={bundle.get('n_emitted')} halts={bundle.get('halt_counts')}"
            )
    if not dk:
        continue
    print(f"  distinct canonical molecules docked: {len({a['smi'] for a in dk})}")
    print(f"  INTENDED scope (r_release): {Counter(round(a['r_release'], 1) for a in dk)}")
    rc = [float(a.get("r_coherent") or 0) for a in dk]
    rr = [float(a.get("r_release") or 0) for a in dk]
    print(
        f"  REALIZED coherent change: min={min(rc):.2f} "
        f"med={statistics.median(rc):.2f} max={max(rc):.2f}"
    )
    if len(rc) > 1 and statistics.pstdev(rc) > 0 and statistics.pstdev(rr) > 0:
        import numpy as np

        print(f"  intended/realized correlation: {float(np.corrcoef(rr, rc)[0, 1]):.3f}")
    print(f"  ring-system deltas: {Counter(a.get('d_rings') for a in dk)}")
    print(f"  cycle-rank deltas: {Counter(a.get('d_cycle_rank') for a in dk)}")
    print(f"  heavy-atom deltas: {Counter(a.get('d_heavy') for a in dk)}")
    print(
        "  added decoration among docked: "
        f"terminal_halogen={sum(int(a.get('added_terminal_halogen') or 0) for a in dk)} "
        f"sulfur={sum(int(a.get('added_sulfur') or 0) for a in dk)} "
        f"backbone_CNO={sum(int(a.get('added_backbone_atoms') or 0) for a in dk)}"
    )
    print(f"  interfaces: {Counter(a['interface'] for a in dk)}")
    big = [a for a in dk if (a.get("r_coherent") or 0) >= 0.2]
    print(f"  candidates with >=0.20 realized coherent change: {len(big)}")
    constructive_names = {
        "grow",
        "cyclize",
        "rebuild",
        "annulate",
        "append_system",
        "build_ring_system",
    }
    constructive = [
        a
        for a in dk
        if a.get("option") in constructive_names
        and ((a.get("d_rings") or 0) > 0 or (a.get("d_cycle_rank") or 0) > 0)
    ]
    print(f"  constructive ring outcomes from constructive/program options: {len(constructive)}")
    for a in sorted(dk, key=lambda a: -(a.get("r_coherent") or 0))[:4]:
        print(
            f"    r_rel={a['r_release']:.2f} -> r_coh={(a.get('r_coherent') or 0):.2f} "
            f"option={a.get('option')} d_rings={a.get('d_rings')} "
            f"d_cycle={a.get('d_cycle_rank')} step={a.get('step')} ds={a['ds']}"
        )
