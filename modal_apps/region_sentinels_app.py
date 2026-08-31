"""Three real-executor sentinels for region replacement.

  1 pendant            end-to-end plumbing: context frozen, every intermediate
                       valid and connected, finite conditional_path_logq,
                       nonzero lineage-changed fraction
  2 easy bridge        the first real algorithm test: an alternative connection
                       must be built BEFORE the old one is removed, and the
                       preserved components must never separate
  3 saturated bridge   the decisive primitive-grammar test. Both terminals carry
                       no free hydrogen, so a new bond cannot simply be added
                       and make-before-break needs reroute/cycle operations.
                       If the grammar cannot do it, this returns
                       UNSAT:no_connectivity_preserving_handoff. NO special-case
                       teleport or macro is added to force a pass -- a failure
                       here is an operator finding, not a controller finding.

Cases are SELECTED FROM THE DATA (region enumeration over dev seeds, filtered by
terminal free valence), not hand-chosen, so "saturated" means measured.

CPU only. No docking, no benchmark logic.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import modal

from modal_apps.run_process_v2_p50_app import (
    ARTIFACT_ROOT, REMOTE_ROOT, artifact_volume,
)
from modal_apps.run_process_v2_p50_app import image as _base_image

image = _base_image.env(
    {"PYTHONPATH": f"{REMOTE_ROOT}/src:{REMOTE_ROOT}", "OMP_NUM_THREADS": "1"})
app = modal.App("region-sentinels")
RUN_ROOT = "/artifacts/editing_v2/r_theta_run"
TIME_POINT, CANONICAL_SLOTS = 0.5, 48
MEM_MIB = int(3 * 1024)
_RT: dict[str, Any] = {}


def _runtime():
    if "model" in _RT:
        return _RT
    import sys
    import torch
    sys.path.insert(0, str(Path(REMOTE_ROOT) / "src"))
    from compose_v4.experiments.editing_v2_process_v2_t1_panel import (
        open_process_v2_t1_source)
    from compose_v4.experiments.editing_v2_process_v2_t1_runtime import (
        build_process_v2_score_revised_scratch_runtime,
        load_materialized_scorer_state)
    from compose_v4.experiments.editing_v2_r_theta_corpus_training import (
        CHECKPOINT_FILENAME)
    from compose_v4.experiments.production_successor_kernel import (
        _default_rewrite_system)
    paths = json.loads((Path(RUN_ROOT) / "run_inputs" / "RUN_PATHS.json").read_text())
    src = open_process_v2_t1_source(
        Path(paths["active8_root"]),
        gate_zero_decision_path=Path(paths["gate_zero"]),
        artifact_root=Path(paths["artifact_root"]),
        repo_root=Path(REMOTE_ROOT))
    bundle = load_materialized_scorer_state(Path(paths["materialized_scorer"]))
    runtime, _b, _c = build_process_v2_score_revised_scratch_runtime(
        src, materialized_state=bundle)
    model = runtime.model
    ck = torch.load(Path(RUN_ROOT) / "runs" / "run_v2_01" / CHECKPOINT_FILENAME,
                    map_location="cpu", weights_only=False)
    model.load_state_dict(ck["selected_model_state"], strict=True)
    model.eval(); torch.set_grad_enabled(False); torch.set_num_threads(1)
    _RT.update({"model": model, "system": _default_rewrite_system(model)})
    return _RT


def _pick_cases(smiles_list, max_region=8):
    """Choose one pendant, one free-valence bridge, one saturated bridge."""
    from compose_v4.chem.molecular_graph import smiles_to_molecular_graph
    from compose_v4.control.region import enumerate_regions
    out = {"pendant": None, "bridge_free": None, "bridge_saturated": None}
    for smi in smiles_list:
        try:
            g = smiles_to_molecular_graph(smi)
        except Exception:
            continue
        h = list(map(int, g.implicit_h_counts))
        for r in enumerate_regions(smi):
            if r.size > max_region or r.size < 1:
                continue
            terms = [int(t[1]) for t in r.boundary]      # context-side terminals
            free = all(h[t] > 0 for t in terms) if terms else False
            sat = all(h[t] == 0 for t in terms) if terms else False
            rec = {"smiles": smi, "atoms": sorted(int(a) for a in r.atoms),
                   "boundary": [[int(i), int(j), float(o)] for i, j, o in r.boundary],
                   "interface": r.interface, "size": r.size,
                   "released": r.released_fraction,
                   "terminal_h": [h[t] for t in terms]}
            if r.interface == "pendant" and out["pendant"] is None and r.size >= 1:
                out["pendant"] = rec
            if r.interface == "bridge" and free and out["bridge_free"] is None:
                out["bridge_free"] = rec
            if r.interface == "bridge" and sat and out["bridge_saturated"] is None:
                out["bridge_saturated"] = rec
        if all(out.values()):
            break
    return out


@app.function(image=image, cpu=(1.0, 1.0), memory=MEM_MIB, timeout=90 * 60,
              retries=0, volumes={ARTIFACT_ROOT: artifact_volume})
def sentinels(job: dict) -> dict:
    import sys
    sys.path.insert(0, str(Path(REMOTE_ROOT) / "src"))
    import numpy as np
    from compose_v4.chem.state import pad_molecular_graph
    from compose_v4.chem.molecular_graph import smiles_to_molecular_graph
    from compose_v4.experiments.production_successor_kernel import (
        canonical_state_key, enumerate_factorized_marked_law)
    from compose_v4.control.region import Region
    from compose_v4.control import region_rewrite as RR
    from compose_v4.control import graph_geometry as GG
    from compose_v4.gates.med_chem_gate import is_valid

    rt = _runtime(); model, system = rt["model"], rt["system"]
    cases = _pick_cases(job["smiles"])
    results = {}
    t0 = time.time()

    for name, rec in cases.items():
        if rec is None:
            results[name] = {"status": "NO_CASE_FOUND"}
            continue
        smi = rec["smiles"]
        st0 = pad_molecular_graph(smiles_to_molecular_graph(smi), CANONICAL_SLOTS)
        n_real = len(smiles_to_molecular_graph(smi).atom_types)
        region = Region(atoms=frozenset(rec["atoms"]),
                        boundary=tuple((i, j, o) for i, j, o in rec["boundary"]),
                        kind="", generator="sentinel", n_atoms_total=n_real,
                        n_context_components=(2 if "bridge" in name else 1),
                        interface=rec["interface"])
        ctx = RR.context_from_region(region)
        lin0 = RR.Lineage.initial(range(n_real))
        old_ids = frozenset(lin0.id_of[s] for s in rec["atoms"] if s in lin0.id_of)

        cache: dict = {}

        def enum_fn(st):
            k = canonical_state_key(st)
            if k not in cache:
                law = enumerate_factorized_marked_law(model, st, float(TIME_POINT))
                cache[k] = ([m.executor_rule_name for m in law.marks],
                            [m.action for m in law.marks],
                            np.array([m.probability for m in law.marks], float))
            return cache[k]

        def apply_fn(st, j):
            fams, acts, _ = enum_fn(st)
            try:
                return system.apply(st, fams[j], acts[j])
            except Exception:
                return None

        if name == "pendant":
            schedule = [("prune_old", max(1, rec["size"])), ("grow_new", 2)]
        else:
            schedule = [("grow_new", "until_handoff"),
                        ("prune_old", max(1, rec["size"]))]

        attempts, best = [], None
        for trial in range(int(job.get("trials", 8))):
            p = RR.propose(enum_fn, apply_fn, ctx, st0, schedule=schedule,
                           rng=np.random.default_rng(100 + trial),
                           lineage=lin0, original_region_ids=old_ids,
                           max_handoff_steps=int(job.get("max_handoff", 16)))
            attempts.append({"status": p.status, "stage": p.stage,
                             "n_steps": len(p.steps),
                             "logq": p.conditional_path_logq,
                             "families": [x["family"] for x in p.steps],
                             "rejections": dict(p.rejections),
                             "admissible_trace": p.n_admissible_seen[:12]})
            if p.status == "OK" and best is None:
                d = GG.structural_displacement(st0, p.endpoint, lin0, p.lineage)
                best = {"logq": p.conditional_path_logq,
                        "n_steps": len(p.steps),
                        "families": [s["family"] for s in p.steps],
                        "phases": [s["phase"] for s in p.steps],
                        "endpoint": canonical_state_key(p.endpoint),
                        "endpoint_valid": bool(is_valid(canonical_state_key(p.endpoint))),
                        "displacement": d}
        from collections import Counter
        results[name] = {
            "case": rec,
            "n_trials": len(attempts),
            "n_ok": sum(1 for a in attempts if a["status"] == "OK"),
            "failure_reasons": dict(Counter(a["stage"] for a in attempts
                                            if a["status"] != "OK")),
            "attempts": attempts,          # per-trial traces; without these the
                                           # artifact cannot say whether a
                                           # failure was missing MOVES or a
                                           # directionless PROPOSAL
            "best": best,
        }
    return {"results": results, "sec": time.time() - t0}


@app.local_entrypoint()
def main(trials: int = 8, n_seeds: int = 6):
    dv = json.loads(Path("docs/GENMOL_T4_DEV_SEEDS.json").read_text())["seeds"]
    smiles = [s["smiles"] for s in dv[:n_seeds]]
    r = sentinels.remote({"smiles": smiles, "trials": trials})
    Path("diagnostics").mkdir(exist_ok=True)
    Path("diagnostics/region_sentinels.json").write_text(json.dumps(r, indent=2))
    print(f"sec={r['sec']:.0f}\n")
    for name, v in r["results"].items():
        if v.get("status") == "NO_CASE_FOUND":
            print(f"{name:18s} NO CASE FOUND"); continue
        c = v["case"]
        print(f"{name:18s} {v['n_ok']}/{v['n_trials']} OK   region size {c['size']} "
              f"released {c['released']:.2f} terminal_H {c['terminal_h']}")
        if v["failure_reasons"]:
            print(f"{'':18s} failures: {v['failure_reasons']}")
        b = v["best"]
        if b:
            d = b["displacement"]
            print(f"{'':18s} steps {b['n_steps']} logq {b['logq']:.2f} "
                  f"changed {d['changed_fraction']:.2f} "
                  f"largest_region {d['largest_changed_region']} "
                  f"coherence {d['coherence']:.2f} dRank {d['d_cycle_rank']}")
            print(f"{'':18s} {b['endpoint']}")
