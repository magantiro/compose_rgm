"""Docking lane for the JAK2 contrastive mechanism experiment.

Docks a caller-supplied list of SMILES against one T4 receptor and streams each result as
it lands, so a run can be watched rather than waited on. It holds no model, no archive, no
proposal machinery and no selection rule: candidates are chosen offline by the controller
and this lane only measures binding.

That separation is the point. Every candidate handed here has already been shown to be a
legal COMPOSE program endpoint, exactly executable, and inside the T4 constraint gate
(QED, SA, similarity). The expensive oracle is therefore asked only the one question that
cannot be answered for free -- how well does this molecule bind -- rather than being spent
establishing feasibility.

Box geometry and the docking command are taken unchanged from `genmol_t4_opt_app` so
scores are comparable with every historical T4 number on this branch.
"""

from __future__ import annotations

import json
import time

import modal

from modal_apps.genmol_t4_opt_app import BOXES, image

app = modal.App("jak2-contrastive-dock")


@app.function(image=image, cpu=8, timeout=60 * 60, max_containers=1)
def dock_batch(payload: str) -> str:
    """Dock every candidate, printing each score the moment it is known."""
    import os
    import subprocess
    from concurrent.futures import ThreadPoolExecutor

    request = json.loads(payload)
    target, candidates = request["target"], request["candidates"]
    (cx, cy, cz), (sx, sy, sz) = BOXES[target]
    seed = int(request.get("seed", 20260917))

    def dock(item):
        index, smiles = item["index"], item["smiles"]
        tag = f"jak2ctr_{index}"
        directory = f"/tmp/{tag}"
        os.makedirs(directory, exist_ok=True)
        mol, ligand, out = f"{directory}/l.mol", f"{directory}/l.pdbqt", f"{directory}/o.pdbqt"
        for path in (mol, ligand, out):
            if os.path.exists(path):
                os.remove(path)
        try:
            subprocess.run(["obabel", f"-:{smiles}", "--gen3D", "-O", mol],
                           capture_output=True, timeout=120, check=True)
            subprocess.run(["obabel", mol, "-O", ligand],
                           capture_output=True, timeout=60, check=True)
        except (subprocess.SubprocessError, OSError) as error:
            return {**item, "score": None, "failure": f"prepare: {type(error).__name__}"}
        try:
            result = subprocess.run(
                ["/opt/dock/qvina02", "--receptor", f"/opt/dock/receptors/{target}.pdbqt",
                 "--ligand", ligand, "--out", out,
                 "--center_x", str(cx), "--center_y", str(cy), "--center_z", str(cz),
                 "--size_x", str(sx), "--size_y", str(sy), "--size_z", str(sz),
                 "--seed", str(seed), "--cpu", "1"],
                capture_output=True, timeout=600, check=True, text=True,
            )
        except (subprocess.SubprocessError, OSError) as error:
            return {**item, "score": None, "failure": f"dock: {type(error).__name__}"}
        score = None
        for line in result.stdout.splitlines():
            parts = line.split()
            if len(parts) >= 2 and parts[0] == "1":
                try:
                    score = float(parts[1])
                except ValueError:
                    score = None
                break
        return {**item, "score": score, "failure": None}

    started = time.time()
    print(f"[dock] {len(candidates)} candidates against {target}", flush=True)
    finished = []
    with ThreadPoolExecutor(max_workers=8) as pool:
        for record in pool.map(dock, candidates):
            finished.append(record)
            elapsed = time.time() - started
            score = record["score"]
            shown = f"{score:7.2f}" if score is not None else f"  FAIL {record['failure']}"
            print(f"[dock] {len(finished):>3}/{len(candidates)}  {elapsed:6.0f}s  "
                  f"bundle={record.get('bundle')} arm={record.get('value')}  {shown}  "
                  f"{record['smiles'][:60]}", flush=True)

    scored = [r["score"] for r in finished if r["score"] is not None]
    print(f"[dock] done in {time.time()-started:.0f}s; "
          f"{len(scored)}/{len(candidates)} scored; best {min(scored) if scored else 'none'}",
          flush=True)
    return json.dumps({"target": target, "seed": seed, "results": finished})
