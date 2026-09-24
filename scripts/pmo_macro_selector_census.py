"""Measure selector COVERAGE across many seeds. Zero oracle calls.

The 'winning class' here is a DIAGNOSTIC positive control only -- we know it because
thiothixene was already inspected by hand. It is never a production filter; the question is
whether a supposedly broad blind sampler systematically fails to represent a structural
class that is 8% of its own admissible pool.
"""
import sys, csv, importlib.util, collections
import numpy as np
sys.path.insert(0, "src")
from rdkit import Chem, RDLogger, DataStructs
from rdkit.Chem import AllChem
RDLogger.DisableLog("rdApp.*")
from compose_v4.control.contextual_region_replace import region_replacements, substituent_replacements, region_excisions
from compose_v4.control.attachment_compatibility import build_attachment_table, admissible

S = sys.argv[1]; TRIALS = int(sys.argv[2]); ARMS = int(sys.argv[3])
spec = importlib.util.spec_from_file_location("cen", "scripts/pmo_macro_rollout_census.py")
cen = importlib.util.module_from_spec(spec); spec.loader.exec_module(cen)
canon = {}
with open("diagnostics/pmo_prescreen_v1/zinc250k_canonical_v1.csv") as fh:
    for r in csv.DictReader(fh):
        canon[int(r["source_row_id"])] = r["canonical_smiles"]
ranked = cen._ranked_prescreen("thiothixene_rediscovery", f"{S}/all_scores", canon, 20000)
ones, twos = cen.mine_payloads(ranked, min_support=2)
lead = "CN1CCN(C(CCCN2C(=O)c3ccccc32)N2CCc3cc(S(=O)(=O)N(C)C)ccc32)CC1"
one_lib = [(s, {"library_mean": m, "library_n": n}) for s, m, n in ones[:300]]
two_lib = [(s, {"library_mean": m, "library_n": n}) for s, m, n in twos[:300]]
pool = (substituent_replacements(lead, one_lib, limit=8000)
        + region_replacements(lead, two_lib, limit=8000)
        + region_excisions(lead, min_removed=1, max_removed=24))
table = build_attachment_table([canon[k] for k in sorted(canon)[:60000]])
adm = [p for p in pool if admissible(lead, p.endpoint, table, min_support=2)]
alk = Chem.MolFromSmarts("[C;R]=[C;!R]")
def win(p):
    m = Chem.MolFromSmiles(p.endpoint)
    return m is not None and m.GetRingInfo().NumRings() >= 3 and m.HasSubstructMatch(alk)
flag = [win(p) for p in adm]
base = sum(flag) / len(adm)
print(f"admissible {len(adm)}, winning class {sum(flag)} = {100*base:.2f}%")
fam = collections.defaultdict(list)
for i, p in enumerate(adm):
    fam[p.family].append(i)
fps = {}
def fp(i):
    if i not in fps:
        fps[i] = AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(adm[i].endpoint), 2, 2048)
    return fps[i]

def sel_uniform(rng):
    return list(rng.choice(len(adm), ARMS, replace=False))

def sel_random_div(rng):
    out, keep = [], []
    for i in rng.permutation(len(adm)):
        f = fp(int(i))
        if keep and max(DataStructs.BulkTanimotoSimilarity(f, keep)) > 0.6:
            continue
        out.append(int(i)); keep.append(f)
        if len(out) >= ARMS: break
    return out

def sel_family_random(rng):
    out, keep = [], []
    present = [k for k in fam if fam[k]]
    quota = max(1, ARMS // len(present))
    for k in present:
        taken = 0
        for i in rng.permutation(len(fam[k])):
            if taken >= quota or len(out) >= ARMS: break
            idx = fam[k][int(i)]; f = fp(idx)
            if keep and max(DataStructs.BulkTanimotoSimilarity(f, keep)) > 0.6: continue
            out.append(idx); keep.append(f); taken += 1
    return out

def sel_libmean(rng):
    order = sorted(range(len(adm)),
                   key=lambda i: -(adm[i].detail.get("library_mean") or 0.0))
    return order[:ARMS]

def sel_hybrid(rng):
    half = sel_uniform(rng)[:ARMS // 2]
    rest = [i for i in sel_family_random(rng) if i not in half]
    return (half + rest)[:ARMS]

for name, fn in (("uniform", sel_uniform), ("random+diversity", sel_random_div),
                 ("family-random+div", sel_family_random),
                 ("hybrid 50/50", sel_hybrid), ("top-N library mean", sel_libmean)):
    trials = 1 if name == "top-N library mean" else TRIALS
    hits = []
    for t in range(trials):
        rng = np.random.default_rng(1000 + t)
        hits.append(sum(1 for i in fn(rng) if flag[i]))
    h = np.array(hits)
    print(f"  {name:20s} mean {h.mean():5.2f}  median {np.median(h):4.1f}  "
          f"p(zero) {np.mean(h == 0):.3f}  n_trials {trials}")
print(f"\nexpected under independent uniform: {ARMS*base:.2f}")
