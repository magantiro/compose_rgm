"""Build a Michael-skewed CNOF lipid corpus: aza-Michael dominant, all families kept."""
import csv, random, sys
from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")

CSV = "artifacts/datasets/compose_lipid_pretraining_v1/r1_reaction_grounded_corpus_v1.csv"
MICHAEL = "aza_michael_amine_acrylate"
# usage: build_michael_skewed_corpus.py [TARGET_TOTAL] [OUT_PATH]
# Michael unique caps at ~16k (loader dedups), so past ~42k Michael stays the #1
# family but its SHARE necessarily drops -- that is expected, not a bug.
TARGET_TOTAL = int(sys.argv[1]) if len(sys.argv) > 1 else 41500
OUT = sys.argv[2] if len(sys.argv) > 2 else \
    "artifacts/datasets/compose_lipid_pretraining_v1/generator_corpus_michael_v1.smiles"
CNOF = {1, 6, 7, 8, 9}
random.seed(21)

def cnof_ok(smi):
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return False
    return all(a.GetAtomicNum() in CNOF for a in m.GetAtoms())

# group CNOF-valid SMILES by family
fam = {}
with open(CSV) as f:
    for row in csv.DictReader(f):
        smi = row["canonical_smiles"].strip()
        if cnof_ok(smi):
            fam.setdefault(row["reaction_family"], []).append(smi)

michael = fam.pop(MICHAEL, [])
others = {k: v for k, v in fam.items() if v}  # S/P families are empty after CNOF filter
print(f"CNOF Michael available: {len(michael):,}")
print("CNOF other families:", {k: len(v) for k, v in sorted(others.items(), key=lambda x: -len(x[1]))})

# water-fill the remaining budget across the other families (small families take all, shortfall flows up)
need = TARGET_TOTAL - len(michael)
picked = {MICHAEL: list(michael)}
for i, k in enumerate(sorted(others, key=lambda k: len(others[k]))):
    share = need // (len(others) - i)
    take = min(share, len(others[k]))
    picked[k] = random.sample(others[k], take)
    need -= take

pool = [(s, k) for k, lst in picked.items() for s in lst]
random.shuffle(pool)
seen = set(); rows = []
for s, k in pool:
    if s not in seen:
        seen.add(s); rows.append(s)

with open(OUT, "w") as f:
    f.write("\n".join(rows) + "\n")

total = len(rows)
print(f"\nwrote {total:,} unique molecules -> {OUT}")
print("final family distribution:")
# recompute family per written row (dedup may have trimmed)
famset = {k: set(v) for k, v in picked.items()}
for k in sorted(picked, key=lambda k: -len(picked[k])):
    c = sum(1 for s in rows if s in famset[k])
    print(f"  {c:>7,}  {100*c/total:5.1f}%  {k}")
