"""Is 'error falls with series depth' just 'deep-series molecules are smaller'?

identity_nll is -log P(y | x, F): a choice among the legal marks of family F on
molecule x. That candidate count grows with molecule size, so a smaller molecule
scores a lower NLL for free. Deep congeneric series could simply be smaller
scaffolds, which would make the depth curve an artifact of branching factor.

virtual_aliases is empty in the stored fiber -- the candidate set is built at
scoring time -- so heavy-atom count is used as the proxy that drives it.
"""
import gzip, json, statistics, collections, sys
from pathlib import Path
from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")
H = Path.home() / "compose_trainset_backup"
SP = Path(sys.argv[1])

rows = json.load(gzip.open(SP / "rows.json.gz", "rt"))
src = {}
with gzip.open(H / "consolidated_panel/LIBRARY.jsonl.gz", "rt") as h:
    for line in h:
        e = json.loads(line)
        src[str(e["p50_entry_sha256"])] = str(
            (e.get("teacher_successor_fiber") or {}).get("source_key", ""))

cache = {}
def heavy(smiles):
    if smiles not in cache:
        m = Chem.MolFromSmiles(smiles)
        cache[smiles] = m.GetNumHeavyAtoms() if m else None
    return cache[smiles]

ring = []
for r in rows:
    if r["panel_depth"] <= 0:
        continue
    n = heavy(src.get(r["entry_id"], ""))
    if n:
        r["heavy"] = n
        ring.append(r)
print(f"ring rows with a parsed source: {len(ring):,}")

DEPTH = ((1, 1), (2, 4), (5, 9), (10, 24), (25, 10**9))
def lab(v, edges=DEPTH):
    for lo, hi in edges:
        if lo <= v <= hi:
            return f"{lo}" if lo == hi else (f"{lo}-{hi}" if hi < 10**9 else f"{lo}+")
    return "?"
def order(s): return int(s.split("-")[0].rstrip("+"))

print("\n=== molecule size across the depth buckets ===")
print(f"{'depth':>8} {'n':>7} {'mean heavy':>11} {'median':>7} {'identity':>9}")
g = collections.defaultdict(list)
for r in ring: g[lab(r["panel_depth"])].append(r)
for k in sorted(g, key=order):
    c = g[k]
    print(f"{k:>8} {len(c):7,} {statistics.mean(x['heavy'] for x in c):11.1f} "
          f"{statistics.median(x['heavy'] for x in c):7.0f} "
          f"{statistics.mean(x['identity_nll'] for x in c):9.3f}")

xs = [r["panel_depth"] for r in ring]; ys = [r["heavy"] for r in ring]
mx, my = statistics.mean(xs), statistics.mean(ys)
num = sum((a-mx)*(b-my) for a, b in zip(xs, ys))
den = (sum((a-mx)**2 for a in xs)*sum((b-my)**2 for b in ys))**0.5
print(f"\nPearson(panel_depth, heavy_atoms) = {num/den:+.3f}")
zs = [r["identity_nll"] for r in ring]; mz = statistics.mean(zs)
num2 = sum((b-my)*(c-mz) for b, c in zip(ys, zs))
den2 = (sum((b-my)**2 for b in ys)*sum((c-mz)**2 for c in zs))**0.5
print(f"Pearson(heavy_atoms, identity_nll) = {num2/den2:+.3f}")

print("\n=== depth WITHIN a molecule-size band (size held, so branching held) ===")
SIZE = ((0, 24), (25, 34), (35, 44), (45, 10**9))
for lo, hi in SIZE:
    band = [r for r in ring if lo <= r["heavy"] <= hi]
    gb = collections.defaultdict(list)
    for r in band: gb[lab(r["panel_depth"])].append(r)
    usable = {k: v for k, v in gb.items() if len(v) >= 40}
    tag = f"{lo}-{hi}" if hi < 10**9 else f"{lo}+"
    if len(usable) < 2:
        print(f"  heavy {tag:>7}: {len(usable)} usable depth bucket(s), skipped")
        continue
    print(f"  heavy {tag:>7} (n={len(band):,}, mean heavy "
          f"{statistics.mean(r['heavy'] for r in band):.1f}):")
    for k in sorted(usable, key=order):
        c = usable[k]
        print(f"      depth {k:>6} n={len(c):6,} identity={statistics.mean(x['identity_nll'] for x in c):.3f} "
              f"heavy={statistics.mean(x['heavy'] for x in c):.1f}")
