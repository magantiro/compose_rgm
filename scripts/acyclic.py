"""Is 'unseen scaffold' really 'acyclic molecule with an empty Murcko scaffold'?"""
import json, gzip, collections
from pathlib import Path
from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold
RDLogger.DisableLog("rdApp.*")
H = Path.home() / "compose_trainset_backup"

cache = {}
def scaffold(smiles):
    if smiles not in cache:
        try:
            mol = Chem.MolFromSmiles(smiles)
            cache[smiles] = MurckoScaffold.MurckoScaffoldSmiles(mol=mol) if mol else None
        except Exception:
            cache[smiles] = None
    return cache[smiles]

def read(path):
    out = []
    with gzip.open(path, "rt") as handle:
        for line in handle:
            e = json.loads(line)
            f = e.get("teacher_successor_fiber") or {}
            out.append((str(e["p50_entry_sha256"]), str(f.get("source_key", ""))))
    return out

def classify(rows, label):
    srcs = {s for _i, s in rows}
    kinds = collections.Counter()
    for s in srcs:
        v = scaffold(s)
        kinds["unparseable" if v is None else ("acyclic (empty Murcko)" if v == "" else "has ring scaffold")] += 1
    ent = collections.Counter()
    for _i, s in rows:
        v = scaffold(s)
        ent["unparseable" if v is None else ("acyclic (empty Murcko)" if v == "" else "has ring scaffold")] += 1
    print(f"\n{label}: {len(srcs):,} distinct sources / {len(rows):,} entries")
    for k in ("has ring scaffold", "acyclic (empty Murcko)", "unparseable"):
        print(f"  {k:24} sources {kinds[k]:7,} ({kinds[k]/max(len(srcs),1):5.1%})"
              f"   entries {ent[k]:7,} ({ent[k]/max(len(rows),1):5.1%})")
    return srcs

train = read(H / "consolidated_train/LIBRARY.jsonl.gz")
panel = read(H / "consolidated_panel/LIBRARY.jsonl.gz")
tsrc = classify(train, "TRAIN LIBRARY")
psrc = classify(panel, "PANEL")

# Coverage counted correctly: treat acyclic as its own real scaffold class.
tscaf = collections.Counter()
for s in tsrc:
    v = scaffold(s)
    if v is not None: tscaf[v] += 1
cov_e = collections.Counter(); cov_s = collections.Counter()
for _i, s in panel:
    v = scaffold(s)
    cov_e["covered" if (v is not None and tscaf.get(v, 0) > 0) else "not covered"] += 1
for s in psrc:
    v = scaffold(s)
    cov_s["covered" if (v is not None and tscaf.get(v, 0) > 0) else "not covered"] += 1
print(f"\nCOVERAGE with acyclic treated as a real scaffold class:")
print(f"  panel entries  covered {cov_e['covered']:,} / {sum(cov_e.values()):,} = {cov_e['covered']/sum(cov_e.values()):.1%}")
print(f"  panel sources  covered {cov_s['covered']:,} / {sum(cov_s.values()):,} = {cov_s['covered']/sum(cov_s.values()):.1%}")
print(f"  train distinct scaffold classes {len(tscaf):,} (incl. acyclic={'yes' if '' in tscaf else 'no'}, "
      f"acyclic train sources={tscaf.get('',0):,})")
