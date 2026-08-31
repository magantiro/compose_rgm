"""T4 scoreboard: COMPOSE arms vs GenMol and InVirtuoGen on the 30 published cells.

Both comparator tables are extracted artifacts, never hand-typed:
  docs/genmol_t4_targets.json       arXiv:2501.06158v3 Table 4
  docs/invirtuogen_t4_targets.json  InVirtuoGen_results README lead table

Docking noise from Gate 0 is 0.70 kcal/mol median, so a per-cell margin below that
is NOT a result. Cells-solved and dashes-filled are noise-immune; score margins are
reported with the noise band attached.
"""
import json, glob, sys, statistics as st

NOISE = 0.70

# D-VALIDITY GATE. The currently running arm D was launched BEFORE adapt_w and
# replay were added to the resume payload. A checkpoint written by that code does
# not contain the learned adapter, so resuming from it silently restarts the
# policy at pi_reset instead of pi_adapted -- mid-experiment, invisibly. Any D
# cell that shows a RESUMED event is therefore INVALID and must be rerun from
# call 0 under the corrected code, per cell. Later D runs resume normally.
def d_validity(cell_json):
    """Return (valid, reason). Applies to arm D cells from the pre-fix launch."""
    r = cell_json.get('resume') or {}
    resumed = cell_json.get('rounds', 0) > 0 and r.get('adapt_w') is None \
        and cell_json.get('arm_resumed_flag', False)
    if resumed:
        return False, "RESUMED under pre-fix code: adapter state was lost"
    return True, "no resume event"
def load(pat, tag):
    o = {}
    for f in glob.glob(pat, recursive=True):
        try: d = json.load(open(f))
        except Exception: continue
        if f"/{tag}_" in f or f"\\{tag}_" in f:
            o[(d['idx'], d['delta'])] = d
    return o

def main(root, tags):
    seeds = json.load(open('docs/GENMOL_T4_SEEDS.json'))
    G = {(r['target'], round(abs(r['seed_score']),1)): r
         for r in json.load(open('docs/genmol_t4_targets.json'))['rows']}
    try:
        I = {(r['target'], round(abs(r['seed_score']),1)): r
             for r in json.load(open('docs/invirtuogen_t4_targets.json'))['rows']}
    except Exception:
        I = {}
    arms = {t: load(f'{root}/**/*.json', t) for t in tags}
    print(f"{'cell':20s} " + " ".join(f"{t:>9s}" for t in tags) + f" {'GenMol':>7s} {'InVirtuo':>9s}")
    print("-" * (22 + 10*len(tags) + 18))
    keys = sorted({k for a in arms.values() for k in a})
    for k in keys:
        s = seeds[k[0]]; key = (s['target'], round(s['published_ds'],1))
        g = G.get(key, {}); i = I.get(key, {})
        gm = g.get('genmol_d04') if abs(k[1]-0.4) < 1e-6 else g.get('genmol_d06')
        iv = i.get('invirtuo_d04') if abs(k[1]-0.4) < 1e-6 else i.get('invirtuo_d06')
        cells = []
        for t in tags:
            d = arms[t].get(k)
            cells.append(f"{d['best_ds']:.1f}" if d and d.get('best_ds') is not None
                         else ("-" if d else ""))
        print(f"{s['target']+' s'+str(k[0])+' d'+str(k[1]):20s} "
              + " ".join(f"{c:>9s}" for c in cells)
              + f" {str(gm):>7s} {str(iv):>9s}")
    print()
    ng = sum(1 for kk in G.values() for v in (kk['genmol_d04'], kk['genmol_d06']) if v is not None)
    ni = sum(1 for kk in I.values() for v in (kk.get('invirtuo_d04'), kk.get('invirtuo_d06')) if v is not None)
    print(f"  SOLVED   GenMol {ng}/30   InVirtuoGen {ni}/30")
    for t in tags:
        a = arms[t]
        s_ = sum(1 for d in a.values() if d.get('best_ds') is not None)
        wg = sum(1 for k, d in a.items() if d.get('best_ds') is not None
                 and (lambda gm: gm is not None and d['best_ds'] < gm)(
                     (lambda kk: (G.get(kk, {}).get('genmol_d04') if abs(k[1]-0.4)<1e-6
                      else G.get(kk, {}).get('genmol_d06')))(
                      (seeds[k[0]]['target'], round(seeds[k[0]]['published_ds'],1)))))
        print(f"  {t:10s} solved {s_}/{len(a)}   beats GenMol on {wg}")
    print(f"\n  Gate-0 docking noise {NOISE} kcal/mol median: per-cell margins below")
    print(f"  that are inside noise. Cells-solved and dashes-filled are not.")

if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2].split(','))
