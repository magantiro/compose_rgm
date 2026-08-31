"""Direct pose comparison: ours vs InVirtuoGen, same receptor, same box.

WHY THIS EXISTS. Every structural hypothesis derived from a DESCRIPTOR
difference has now failed under a matched intervention:

    7-membered rings   no-7-ring subset was WORSE (0/15 feasible)
    bridgeheads        rate 0.45 -> 0.07, feasible yield unchanged at 1.7%
    epsilon floor      eps 0 vs 0.15, dC 2.57 vs 2.65
    ring regularity    SA got worse (4.12 -> 4.90)
    aromatic intent    contract unsatisfiable at closure, arom stayed 2
    topology choice    close_disjoint and append_system both capped at -10.2
    lipophilicity      logP matched IVG (3.92 vs 3.7) and docking LOST 1.1

The last one is decisive: we hit their descriptor value almost exactly and got
a worse score. logP was correlated with IVG, not causal for binding. So no more
descriptor-driven hypotheses -- look at where the molecules actually sit in the
pocket and which contacts the better one makes that ours does not.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import modal

from modal_apps.genmol_t4_opt_app import ARTIFACT_ROOT, artifact_volume
from modal_apps.genmol_t4_opt_app import image as _opt_image

image = _opt_image
app = modal.App("pose-compare")


@app.function(image=image, cpu=(4.0, 4.0), memory=int(8 * 1024),
              timeout=60 * 60, retries=0,
              volumes={str(ARTIFACT_ROOT): artifact_volume})
def dock_with_pose(job: dict) -> list:
    """Dock each SMILES and KEEP the pose, then report receptor contacts."""
    import os, subprocess, collections, math
    from modal_apps.genmol_t4_opt_app import BOXES
    target = job["target"]
    (cx, cy, cz), (sx, sy, sz) = BOXES[target]
    rec = f"/opt/dock/receptors/{target}.pdbqt"

    # receptor heavy atoms with residue identity
    ratoms = []
    for line in open(rec):
        if line.startswith(("ATOM", "HETATM")):
            try:
                ratoms.append((line[17:20].strip(), line[22:26].strip(),
                               float(line[30:38]), float(line[38:46]),
                               float(line[46:54])))
            except Exception:
                pass

    out = []
    for k, (label, smi) in enumerate(job["molecules"]):
        d = f"/tmp/pose{k}"
        os.makedirs(d, exist_ok=True)
        mol, lig, o = f"{d}/l.mol", f"{d}/l.pdbqt", f"{d}/o.pdbqt"
        for p in (mol, lig, o):
            if os.path.exists(p):
                os.remove(p)
        try:
            subprocess.run(["obabel", f"-:{smi}", "--gen3D", "-O", mol],
                           capture_output=True, timeout=120, check=True)
            subprocess.run(["obabel", mol, "-O", lig],
                           capture_output=True, timeout=60, check=True)
            subprocess.run(
                ["/opt/dock/qvina02", "--receptor", rec, "--ligand", lig, "--out", o,
                 "--center_x", str(cx), "--center_y", str(cy), "--center_z", str(cz),
                 "--size_x", str(sx), "--size_y", str(sy), "--size_z", str(sz),
                 "--cpu", "4", "--num_modes", "10", "--exhaustiveness", "1"],
                capture_output=True, timeout=300, check=True)
        except Exception as exc:
            out.append(dict(label=label, smiles=smi, error=type(exc).__name__))
            continue

        score, lig_atoms, mode = None, [], 0
        for line in open(o):
            if line.startswith("MODEL"):
                mode += 1
            if mode > 1:
                break                       # best pose only
            if line.startswith("REMARK VINA RESULT") and score is None:
                score = float(line.split()[3])
            if line.startswith(("ATOM", "HETATM")):
                try:
                    lig_atoms.append((line[12:16].strip(), float(line[30:38]),
                                      float(line[38:46]), float(line[46:54])))
                except Exception:
                    pass
        # residues within 4.0 A of any ligand heavy atom
        contacts = collections.Counter()
        for _an, lx, ly, lz in lig_atoms:
            for rn, ri, rx, ry, rz in ratoms:
                if abs(rx - lx) > 4 or abs(ry - ly) > 4 or abs(rz - lz) > 4:
                    continue
                if math.dist((lx, ly, lz), (rx, ry, rz)) <= 4.0:
                    contacts[f"{rn}{ri}"] += 1
        cen = (sum(a[1] for a in lig_atoms) / max(len(lig_atoms), 1),
               sum(a[2] for a in lig_atoms) / max(len(lig_atoms), 1),
               sum(a[3] for a in lig_atoms) / max(len(lig_atoms), 1))
        out.append(dict(label=label, smiles=smi, ds=score, n_atoms=len(lig_atoms),
                        centroid=[round(c, 2) for c in cen],
                        contacts=dict(contacts.most_common()),
                        n_contact_residues=len(contacts)))
        print(f"{label}: ds={score} residues={len(contacts)}", flush=True)
    return out


@app.local_entrypoint()
def compare(cell: str = "parp1_s0_d0.4"):
    root = Path(__file__).resolve().parents[1]
    seeds = {f"{s['target']}_s{s['idx']}": s for s in
             json.loads((root / "docs/GENMOL_T4_SEEDS.json").read_text())}
    wins = json.loads((root / "diagnostics/ivg_winners.json").read_text())
    tgt, si, _dl = cell.rsplit("_", 2)
    sd = seeds[f"{tgt}_{si}"]
    mols = [("SEED", sd["smiles"])]
    for i, w in enumerate(wins[cell]["winners"][:3]):
        mols.append((f"IVG{i+1}_{w['ds']}", w["smiles"]))
    for i, s in enumerate([
        "CN(C(=O)c1ccc2c(c1)CNC(=O)c1cccn1-2)c1csc(=S)oc1=N",
        "COC1N(C(=O)c2ccc3c(c2)CNC(=O)c2cccn2-3)CC(=O)C1(C)C",
        "CCN(Cc1ccc2c(c1)CNC(=O)c1cccn1-2)C1CCN(C)CC1O"]):
        mols.append((f"OURS{i+1}", s))
    res = dock_with_pose.remote(dict(target=sd["target"], molecules=mols))
    (root / f"diagnostics/pose_compare_{cell}.json").write_text(json.dumps(res, indent=1))
    for r in res:
        if r.get("error"):
            print(f"{r['label']:16s} ERROR {r['error']}"); continue
        print(f"{r['label']:16s} ds={r['ds']:6.1f} atoms={r['n_atoms']:3d} "
              f"residues={r['n_contact_residues']:3d} centroid={r['centroid']}")
    print("\ncontact residues:")
    for r in res:
        if r.get("contacts"):
            top = list(r["contacts"])[:12]
            print(f"  {r['label']:16s} {top}")
