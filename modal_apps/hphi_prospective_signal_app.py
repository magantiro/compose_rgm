"""Are eventual target-hitting runs distinguishable BEFORE they hit?

No new simulations. Mines the banked per-transition records, which exist only
for the H=24 cohort -- the archive app stores per-candidate summaries, so the
H=40 trajectories cannot be mined without re-running them.

THE FORK THIS DECIDES. If some observable separates future winners where h_phi
does not, that observable becomes the progress coordinate for the next
rare-event controller, with R_theta untouched -- a splitting method preserving
trajectories that crossed a level is NOT policy B, which greedily steered every
transition toward higher QED. If nothing separates them, controller variants
stop and the next move is a budget-extended h_phi.

Features per visited state, all computable from the stored SMILES:
    h_phi        as recorded, h_phi(y, b-1)
    qed          RDKit
    sim          Tanimoto to the ORIGINAL source
    slack        max(0, 0.90-qed) + max(0, 0.40-sim), distance to the region
    depth        step index

Compared at fixed LOOKAHEAD distances before the run ends, so a separation
cannot be an artifact of measuring nearer the hit. Reported as AUC: the
probability a random state from a hit run outranks a random state from a non-hit
run. 0.5 is no signal.
"""
from __future__ import annotations
import json
from collections import defaultdict
from pathlib import Path
from typing import Any
import modal
from modal_apps.run_process_v2_p50_app import ARTIFACT_ROOT, REMOTE_ROOT, artifact_volume
from modal_apps.run_process_v2_p50_app import image as _base_image

image = _base_image.env({"PYTHONPATH": f"{REMOTE_ROOT}/src:{REMOTE_ROOT}"})
app = modal.App("hphi-prospective")
RUN_ROOT = "/artifacts/editing_v2/r_theta_run"
LOOKAHEADS = (4, 8, 12)


def auc(pos: list[float], neg: list[float]) -> float:
    if not pos or not neg:
        return float("nan")
    import numpy as np
    a = np.concatenate([np.asarray(pos), np.asarray(neg)])
    order = a.argsort()
    ranks = np.empty(len(a), float)
    ranks[order] = np.arange(1, len(a) + 1)
    # average ranks for ties, else ties inflate or deflate the statistic
    _, inv, cnt = np.unique(a, return_inverse=True, return_counts=True)
    sums = np.zeros(len(cnt)); np.add.at(sums, inv, ranks)
    ranks = (sums / cnt)[inv]
    n1 = len(pos)
    return float((ranks[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * len(neg)))


@app.function(image=image, cpu=(2.0, 2.0), memory=4096, timeout=40 * 60,
              volumes={str(ARTIFACT_ROOT): artifact_volume})
def mine(strata: dict) -> dict[str, Any]:
    import numpy as np
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")
    from rdkit import Chem, DataStructs
    from rdkit.Chem import QED, rdFingerprintGenerator

    artifact_volume.reload()
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    d = Path(RUN_ROOT) / "hphi_smc_64" / "replicates"
    files = sorted(d.glob("*.json"))
    print(f"{len(files)} banked records with per-transition detail\n")

    feats = ("h_phi", "qed", "sim", "slack", "depth")
    pos: dict = {L: defaultdict(list) for L in LOOKAHEADS}
    neg: dict = {L: defaultdict(list) for L in LOOKAHEADS}
    n_hit = n_miss = 0

    for f in files:
        doc = json.loads(f.read_text())
        idx, rec = doc["index"], doc["record"]
        if strata.get(str(idx), "hard") == "reliable":
            continue                      # decision strata only
        tr = rec["transitions"]
        if not tr:
            continue
        hit_steps = [t["step"] for t in tr if t.get("h_y_bm1") == 1.0]
        is_hit = bool(hit_steps)
        end = min(hit_steps) if is_hit else max(t["step"] for t in tr)
        n_hit += int(is_hit); n_miss += int(not is_hit)

        src_mol = Chem.MolFromSmiles(rec["transitions"][0]["x"])
        if src_mol is None:
            continue
        src_fp = gen.GetFingerprint(src_mol)
        cache: dict = {}

        def props(smi):
            if smi not in cache:
                m = Chem.MolFromSmiles(smi)
                cache[smi] = (0.0, 0.0) if m is None else (
                    float(QED.qed(m)),
                    float(DataStructs.TanimotoSimilarity(
                        src_fp, gen.GetFingerprint(m))))
            return cache[smi]

        bucket = pos if is_hit else neg
        for L in LOOKAHEADS:
            s = end - L
            if s < 0:
                continue
            for t in tr:
                if t["step"] != s or t.get("h_y_bm1") is None:
                    continue
                q, sm = props(t["y"])
                bucket[L]["h_phi"].append(float(t["h_y_bm1"]))
                bucket[L]["qed"].append(q)
                bucket[L]["sim"].append(sm)
                bucket[L]["slack"].append(-(max(0.0, 0.90 - q)
                                            + max(0.0, 0.40 - sm)))
                bucket[L]["depth"].append(float(t["step"]))
        print(f"  {f.name} {'HIT' if is_hit else 'miss'}", flush=True)

    print(f"\nruns: {n_hit} hit, {n_miss} miss (marginal+hard only)")
    print(f"\n{'lookahead':<11}" + "".join(f"{k:>10}" for k in feats)
          + f"{'n pos':>9}{'n neg':>9}")
    out: dict[str, Any] = {"n_hit": n_hit, "n_miss": n_miss, "auc": {}}
    for L in LOOKAHEADS:
        row = {k: auc(pos[L][k], neg[L][k]) for k in feats}
        out["auc"][L] = row
        print(f"  {L:<9}" + "".join(f"{row[k]:>10.3f}" for k in feats)
              + f"{len(pos[L]['qed']):>9}{len(neg[L]['qed']):>9}")
    print("\n  AUC 0.5 is no prospective signal. A feature well above 0.5 several")
    print("  steps before the end separates eventual winners in advance, and is")
    print("  a candidate progress coordinate; h_phi is the incumbent.")
    return out


@app.local_entrypoint()
def main() -> None:
    root = Path(__file__).resolve().parents[1]
    g = json.loads((root / "docs/HPHI_SMC_64_GATE_BANKED.json").read_text())
    l = json.loads((root / "docs/HPHI_COVERAGE_LADDER_BANKED.json").read_text())
    r3 = json.loads((root / "docs/HPHI_LADDER_RUNG3.json").read_text())
    rel = set(g["solved_sources"])
    marg = set(l["rung2"]["new_sources"]) | set(r3["conversions"])
    strata = {str(i): ("reliable" if i in rel else
                       "marginal" if i in marg else "hard") for i in range(64)}
    o = mine.remote(strata)
    Path("docs/PROSPECTIVE_SIGNAL.json").write_text(json.dumps(o, indent=1))
    print("\nwrote docs/PROSPECTIVE_SIGNAL.json")
