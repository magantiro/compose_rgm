"""Executor audit: does every admitted edit change heavy-atom count by at most one?

Sec. 4.1 now asserts "Each admitted edit changes heavy-atom count by at most one",
which is a claim about the operator registry, not a consequence of anything else
in the section. It has to be verified against the executor rather than inferred
from the conceptual description of the families.

A partial check already exists and passed: 358 real one-step successors banked in
invgnn_v1/lookahead_truth_24.json show |delta n| <= 1 with all three cardinality
classes present (-1: 10.6%, 0: 57.0%, +1: 32.4%). But those are the successors
R_theta actually proposed, not the whole legal edit set, so a rare operator that
moves two heavy atoms could be absent from that sample.

This job enumerates the FULL A(x) -- every mark, no probability cap -- applies
each one, and reports the heavy-atom delta distribution BY OPERATOR FAMILY. A
single |delta n| > 1 falsifies the sentence, in which case the trans-dimensional
paragraph gets rewritten more generally rather than the implementation bent to
fit the prose.

CPU only. No oracle calls, no docking, no training.
"""

from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

import modal

from modal_apps.run_process_v2_p50_app import (
    ARTIFACT_ROOT, REMOTE_ROOT, ROOT, artifact_volume,
)
from modal_apps.run_process_v2_p50_app import image as _base_image

image = (_base_image
         .add_local_file(ROOT / "docs/GENMOL_T4_SEEDS.json",
                         str(REMOTE_ROOT / "docs/GENMOL_T4_SEEDS.json"), copy=True)
         .env({"PYTHONPATH": f"{REMOTE_ROOT}/src:{REMOTE_ROOT}", "OMP_NUM_THREADS": "1"}))

app = modal.App("compose-delta-n-audit")
RUN_ROOT = "/artifacts/editing_v2/r_theta_run"
TIME_POINT, CANONICAL_SLOTS = 0.5, 48

_RT: dict[str, Any] = {}


def _runtime():
    if "model" in _RT:
        return _RT
    import sys

    import torch
    sys.path.insert(0, str(REMOTE_ROOT / "src"))
    from compose_v4.experiments.editing_v2_process_v2_t1_panel import (
        open_process_v2_t1_source)
    from compose_v4.experiments.editing_v2_process_v2_t1_runtime import (
        build_process_v2_score_revised_scratch_runtime, load_materialized_scorer_state)
    from compose_v4.experiments.editing_v2_r_theta_corpus_training import (
        CHECKPOINT_FILENAME)
    from compose_v4.experiments.production_successor_kernel import (
        _default_rewrite_system)
    paths = json.loads((Path(RUN_ROOT) / "run_inputs" / "RUN_PATHS.json").read_text())
    src = open_process_v2_t1_source(
        Path(paths["active8_root"]),
        gate_zero_decision_path=Path(paths["gate_zero"]),
        artifact_root=Path(paths["artifact_root"]), repo_root=REMOTE_ROOT)
    bundle = load_materialized_scorer_state(Path(paths["materialized_scorer"]))
    runtime, _b, _c = build_process_v2_score_revised_scratch_runtime(
        src, materialized_state=bundle)
    model = runtime.model
    ck = torch.load(Path(RUN_ROOT) / "runs" / "run_v2_01" / CHECKPOINT_FILENAME,
                    map_location="cpu", weights_only=False)
    model.load_state_dict(ck["selected_model_state"], strict=True)
    model.eval()
    torch.set_grad_enabled(False)
    torch.set_num_threads(1)
    _RT.update({"model": model, "system": _default_rewrite_system(model)})
    return _RT


@app.function(image=image, cpu=(1.0, 1.0), memory=int(4.5 * 1024),
              timeout=4 * 60 * 60, max_containers=20,
              volumes={str(ARTIFACT_ROOT): artifact_volume})
def audit(task: dict[str, Any]) -> dict[str, Any]:
    import sys

    sys.path.insert(0, str(REMOTE_ROOT / "src"))
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")
    from compose_v4.chem.molecular_graph import smiles_to_molecular_graph
    from compose_v4.chem.state import pad_molecular_graph
    from compose_v4.experiments.production_successor_kernel import (
        canonical_state_key, enumerate_factorized_marked_law)

    rt = _runtime()
    model, system = rt["model"], rt["system"]
    smi = task["smiles"]
    t0 = time.perf_counter()
    h0 = Chem.MolFromSmiles(smi).GetNumHeavyAtoms()
    st = pad_molecular_graph(smiles_to_molecular_graph(smi), CANONICAL_SLOTS)
    law = enumerate_factorized_marked_law(model, st, float(TIME_POINT))

    by_fam: dict[str, Counter] = {}
    viol, n_ok, n_fail = [], 0, 0
    for mk in law.marks:                       # FULL fiber, no cap
        try:
            y = canonical_state_key(system.apply(
                st, mk.executor_rule_name, mk.action))
        except Exception:
            n_fail += 1
            continue
        m = Chem.MolFromSmiles(y) if y else None
        if m is None:
            n_fail += 1
            continue
        d = m.GetNumHeavyAtoms() - h0
        fam = str(mk.executor_rule_name)
        by_fam.setdefault(fam, Counter())[d] += 1
        n_ok += 1
        if abs(d) > 1 and len(viol) < 20:
            viol.append({"family": fam, "delta": d, "from": smi, "to": y})
    return {**task, "heavy": h0, "n_marks": len(law.marks), "n_applied": n_ok,
            "n_failed": n_fail,
            "by_family": {k: dict(v) for k, v in by_fam.items()},
            "violations": viol, "seconds": round(time.perf_counter() - t0, 1)}


@app.function(image=image, cpu=(1.0, 1.0), memory=2048, timeout=6 * 60 * 60,
              volumes={str(ARTIFACT_ROOT): artifact_volume})
def drive() -> dict[str, Any]:
    seeds = json.loads((REMOTE_ROOT / "docs/GENMOL_T4_SEEDS.json").read_text())
    tasks = [{"idx": i, "target": s["target"], "smiles": s["smiles"],
              "seed_heavy": s["heavy"]} for i, s in enumerate(seeds)]
    print(f"FULL fiber enumeration on {len(tasks)} molecules "
          f"(heavy {min(t['seed_heavy'] for t in tasks)}-"
          f"{max(t['seed_heavy'] for t in tasks)})\n", flush=True)
    out = [r for r in audit.map(tasks, order_outputs=True, return_exceptions=True,
                                wrap_returned_exceptions=False)
           if isinstance(r, dict)]

    agg: dict[str, Counter] = {}
    tot, viol = 0, []
    for r in out:
        tot += r["n_applied"]
        viol.extend(r["violations"])
        for fam, c in r["by_family"].items():
            agg.setdefault(fam, Counter()).update({int(k): v for k, v in c.items()})

    print(f"{'executor family':32s}{'-1':>8}{'0':>8}{'+1':>8}{'|d|>1':>8}{'total':>8}")
    for fam in sorted(agg):
        c = agg[fam]
        bad = sum(v for k, v in c.items() if abs(k) > 1)
        print(f"  {fam:30s}{c.get(-1,0):>8}{c.get(0,0):>8}{c.get(1,0):>8}"
              f"{bad:>8}{sum(c.values()):>8}")
    allc = Counter()
    for c in agg.values():
        allc.update(c)
    print(f"\n  successors applied: {tot:,} across {len(out)} molecules, "
          f"{len(agg)} executor families")
    print(f"  delta n observed  : {sorted(allc)}")
    print(f"  |delta n| > 1     : {sum(v for k,v in allc.items() if abs(k)>1)}")
    ok = all(abs(k) <= 1 for k in allc)
    print(f"\n  VERDICT: {'PASS -- the Sec 4.1 sentence holds on this evidence' if ok else 'FAIL -- rewrite the trans-dimensional paragraph'}")
    for v in viol[:5]:
        print(f"     {v['family']} delta={v['delta']}  {v['from'][:36]} -> {v['to'][:36]}")
    return {"n_molecules": len(out), "n_successors": tot,
            "families": {k: dict(v) for k, v in agg.items()},
            "delta_histogram": dict(allc), "violations": viol, "pass": ok}


@app.local_entrypoint()
def main() -> None:
    o = drive.remote()
    p = Path(__file__).resolve().parents[1] / "diagnostics/compose_delta_n_audit.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(o, indent=1))
    print(f"\nwrote {p}")
