"""Does the H40-aware h_phi discriminate at long range where the H24 one collapses?

The banked comparison (docs/PROSPECTIVE_SIGNAL.json) reads, on marginal+hard
runs, h_phi AUC 0.796 at 4 steps -> 0.667 at 8 -> 0.556 at 12: by 12 steps out
the twist is near coin-flip while raw `slack` is at 0.81. That collapse is the
thing an H40-aware head is supposed to fix.

METHOD. Identical runs, identical positions, identical AUC code -- ONLY the
head changes. Each banked transition already carries `budget` and the h_phi
value the run actually used, so both heads are scored on exactly the states the
comparison was originally computed on. Nothing is re-simulated, so no
controller difference can leak into the measurement.

THE PARITY CHECK IS THE POINT. Re-scoring the OLD head must reproduce the
banked `h_y_bm1` it recorded during the run. If it does not, the feature build,
the normalisation or the budget clamp is wrong, and the NEW head's number would
be wrong in the same invisible way. So old-recomputed is carried as its own
column and compared against the banked column rather than assumed equal.

BOTH HEADS ARE CLEAN ON THIS PANEL. These are the dev-64 sources, which are
disjoint from hphi_train_1024 -- so neither the H24 head (trained on all 1,024)
nor the H40 head (trained on the first 256) has seen them.

CPU ONLY. This is inference and RDKit; there is nothing here a GPU would help.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import modal

from modal_apps.run_process_v2_p50_app import (
    ARTIFACT_ROOT,
    REMOTE_ROOT,
    ROOT,
    artifact_volume,
)
from modal_apps.run_process_v2_p50_app import image as _base_image

image = _base_image.env(
    {"PYTHONPATH": f"{REMOTE_ROOT}/src:{REMOTE_ROOT}", "OMP_NUM_THREADS": "1"}
)

app = modal.App("hphi-prospective-two-head")
RUN_ROOT = "/artifacts/editing_v2/r_theta_run"
REGION = (0.90, 0.40)
TIME_POINT, CANONICAL_SLOTS = 0.5, 48
LOOKAHEADS = (4, 8, 12, 16)

_RT: dict[str, Any] = {}


def auc(pos: list[float], neg: list[float]) -> float:
    """Mann-Whitney AUC with ties counted as half, 0.0 when undefined."""
    if not pos or not neg:
        return 0.0
    n = 0.0
    for p in pos:
        for q in neg:
            n += 1.0 if p > q else (0.5 if p == q else 0.0)
    return n / (len(pos) * len(neg))


def _runtime():
    if "model" in _RT:
        return _RT
    import sys

    import torch

    sys.path.insert(0, str(REMOTE_ROOT / "src"))
    from compose_v4.experiments.editing_v2_process_v2_t1_panel import (
        open_process_v2_t1_source,
    )
    from compose_v4.experiments.editing_v2_process_v2_t1_runtime import (
        build_process_v2_score_revised_scratch_runtime,
        load_materialized_scorer_state,
    )
    from compose_v4.experiments.editing_v2_r_theta_corpus_training import (
        CHECKPOINT_FILENAME,
    )

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
    model.eval(); torch.set_grad_enabled(False); torch.set_num_threads(1)
    _RT["model"] = model
    return _RT


@app.function(image=image, cpu=(2.0, 2.0), memory=16384, timeout=4 * 60 * 60,
              volumes={str(ARTIFACT_ROOT): artifact_volume})
def extract(strata: dict) -> dict[str, Any]:
    """Write the states this comparison needs, in the encode job's schema.

    Encoding in-process was the original design and it was wrong twice over: a
    single container encodes at ~4.5 s/state, which is ~20 h for this panel,
    and everything it built would die with the container. So states are
    enumerated here, encoded by the SEPARATE persisted parallel encode job,
    and only then scored. Same lesson as the corpus encode.
    """
    import gzip

    artifact_volume.reload()
    d = Path(RUN_ROOT) / "hphi_smc_64" / "replicates"
    files = sorted(d.glob("*.json"))
    results = []
    for f in files:
        doc = json.loads(f.read_text())
        idx, rec = doc["index"], doc["record"]
        if strata.get(str(idx), "hard") == "reliable":
            continue
        tr = rec["transitions"]
        if not tr:
            continue
        hit_steps = [t["step"] for t in tr if t.get("h_y_bm1") == 1.0]
        end = (min(hit_steps) if hit_steps
               else max(t["step"] for t in tr))
        source = tr[0]["x"]
        want = {source}                    # the source embedding is needed too
        for L in LOOKAHEADS:
            s_step = end - L
            if s_step < 0:
                continue
            for t in tr:
                if t["step"] == s_step and t.get("h_y_bm1") is not None:
                    want.add(t["y"])
        results.append({"index": len(results), "status": "OK",
                        "source": source,
                        "trajectories": [{"path": sorted(want)}]})
    uniq = {s for r in results for t in r["trajectories"] for s in t["path"]}
    out = Path(RUN_ROOT) / "hphi_rollout_corpus" / "banked_eval_states.json.gz"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(gzip.compress(json.dumps({"results": results}).encode()))
    artifact_volume.commit()
    print(f"{len(results)} runs -> {len(uniq):,} unique states -> {out.name}",
          flush=True)
    return {"n_runs": len(results), "n_states": len(uniq), "path": out.name}


@app.function(image=image, cpu=(4.0, 4.0), memory=16384, timeout=4 * 60 * 60,
              volumes={str(ARTIFACT_ROOT): artifact_volume})
def compare(strata: dict, new_dir: str, new_budget_max: int,
            emb_dirs: str) -> dict[str, Any]:
    import gzip
    import sys
    import time

    import numpy as np
    import torch

    sys.path.insert(0, str(REMOTE_ROOT / "src"))
    from rdkit import RDLogger

    RDLogger.DisableLog("rdApp.*")
    from rdkit import Chem, DataStructs
    from rdkit.Chem import QED, rdFingerprintGenerator

    from compose_v4.chem.molecular_graph import smiles_to_molecular_graph
    from compose_v4.chem.state import pad_molecular_graph
    from compose_v4.experiments.hphi_region_features import (
        build_features, in_region,
    )
    from compose_v4.experiments.production_successor_kernel import _one_state_batch

    artifact_volume.reload()

    def load_head(d: str):
        h = torch.jit.load(str(Path(RUN_ROOT) / d / "head.pt"), map_location="cpu")
        h.eval()
        n = json.loads((Path(RUN_ROOT) / d / "norm.json").read_text())
        return h, np.asarray(n["mu"]), np.asarray(n["sd"])

    old_head, old_mu, old_sd = load_head("hphi_v2")
    new_head, new_mu, new_sd = load_head(new_dir)
    print(f"old head width {old_mu.shape[0]}   new head width "
          f"{new_mu.shape[0]} ({new_dir}, budget_max={new_budget_max})",
          flush=True)

    _emb: dict[str, Any] = {}
    for dn in [p.strip() for p in emb_dirs.split(",") if p.strip()]:
        sd_ = Path(RUN_ROOT) / dn
        for fp in sorted(sd_.glob("shard_*.json.gz")):
            _emb.update(json.loads(gzip.decompress(fp.read_bytes()).decode()))
    print(f"{len(_emb):,} persisted embeddings loaded", flush=True)
    missing: set = set()

    def encode(smi: str):
        v = _emb.get(smi)
        if v is None:
            missing.add(smi)
            return None
        return np.asarray(v)

    def score(head, mu, sd, budget_max, e_y, e_src, q, s, budget) -> float:
        """The frozen twist, with only the one-hot width parameterised."""
        if in_region(q, s, REGION):
            return 1.0
        if budget <= 0:
            return 0.0
        f = build_features(e_y, e_src, q, s, REGION,
                           max(0, min(int(budget), int(budget_max))),
                           int(budget_max))
        x = torch.tensor(((f - mu) / sd).astype(np.float32)).unsqueeze(0)
        with torch.no_grad():
            return float(torch.sigmoid(head(x)).item())

    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    d = Path(RUN_ROOT) / "hphi_smc_64" / "replicates"
    files = sorted(d.glob("*.json"))
    print(f"{len(files)} banked records\n", flush=True)

    feats = ("h_phi_banked", "h_phi_old", "h_phi_new", "qed", "sim", "slack")
    pos: dict = {L: defaultdict(list) for L in LOOKAHEADS}
    neg: dict = {L: defaultdict(list) for L in LOOKAHEADS}
    n_hit = n_miss = 0
    parity: list[float] = []
    # RUN-LEVEL aggregates. The 32 particles scored at one step of one
    # trajectory are not 32 independent observations -- they share a lineage,
    # a source and a step. The trajectory is the independent unit, so each run
    # contributes ONE number per lookahead per feature.
    runlvl: dict = {L: [] for L in LOOKAHEADS}
    t0 = time.perf_counter()

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

        source = rec["transitions"][0]["x"]
        src_mol = Chem.MolFromSmiles(source)
        if src_mol is None:
            continue
        src_fp = gen.GetFingerprint(src_mol)
        e_src = encode(source)
        if e_src is None:
            continue
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
            s_step = end - L
            if s_step < 0:
                continue
            run_acc: dict = defaultdict(list)
            for t in tr:
                if t["step"] != s_step or t.get("h_y_bm1") is None:
                    continue
                q, sm = props(t["y"])
                b = int(t["budget"]) - 1          # h_y_bm1 is h(y, b-1)
                e_y = encode(t["y"])
                if e_y is None:
                    continue
                h_old = score(old_head, old_mu, old_sd, 24, e_y, e_src, q, sm, b)
                h_new = score(new_head, new_mu, new_sd, new_budget_max,
                              e_y, e_src, q, sm, b)
                parity.append(abs(h_old - float(t["h_y_bm1"])))
                bucket[L]["h_phi_banked"].append(float(t["h_y_bm1"]))
                bucket[L]["h_phi_old"].append(h_old)
                bucket[L]["h_phi_new"].append(h_new)
                bucket[L]["qed"].append(q)
                bucket[L]["sim"].append(sm)
                slack_v = -(max(0.0, 0.90 - q) + max(0.0, 0.40 - sm))
                bucket[L]["slack"].append(slack_v)
                run_acc["h_phi_old"].append(h_old)
                run_acc["h_phi_new"].append(h_new)
                run_acc["qed"].append(q); run_acc["sim"].append(sm)
                run_acc["slack"].append(slack_v)
            if run_acc:
                runlvl[L].append((
                    is_hit,
                    {k: float(np.mean(v)) for k, v in run_acc.items()},
                    {k: float(np.max(v)) for k, v in run_acc.items()}))
        print(f"  {f.name} {'HIT' if is_hit else 'miss'}  "
              f"{len(_emb):,} encoded  {time.perf_counter()-t0:.0f}s", flush=True)

    if missing:
        print(f"WARNING: {len(missing):,} states had no persisted embedding "
              f"and were skipped", flush=True)
    worst_parity = max(parity) if parity else float("nan")
    print(f"\nruns: {n_hit} hit, {n_miss} miss (marginal+hard only)")
    print(f"states encoded: {len(_emb):,}")
    print(f"PARITY old-recomputed vs banked h_y_bm1: worst |diff| "
          f"{worst_parity:.3e} over {len(parity):,} transitions")
    if not (worst_parity < 1e-4):
        print("  PARITY FAILED -- the re-scoring path does not reproduce the "
              "run's own values, so the NEW column is not trustworthy either.")

    print(f"\n{'lookahead':<11}" + "".join(f"{k:>14}" for k in feats))
    out: dict[str, Any] = {
        "n_hit": n_hit, "n_miss": n_miss, "auc": {},
        "worst_parity": worst_parity, "n_scored": len(parity),
        "new_dir": new_dir, "new_budget_max": new_budget_max,
        "n_states_available": len(_emb), "n_missing": len(missing),
    }
    for L in LOOKAHEADS:
        row = {k: auc(pos[L][k], neg[L][k]) for k in feats}
        row["n_pos"] = len(pos[L]["h_phi_new"])
        row["n_neg"] = len(neg[L]["h_phi_new"])
        out["auc"][L] = row
        print(f"{L:<11}" + "".join(f"{row[k]:>14.4f}" for k in feats)
              + f"   (n+ {row['n_pos']}, n- {row['n_neg']})")
    print("\nAUC is over TRANSITIONS at the fixed lookahead, pooled across "
          "runs, exactly as the banked baseline computed it.")

    # ---- RUN-LEVEL, the honest unit ---------------------------------------
    rng = np.random.default_rng(0)
    B = 4000
    out["run_level"] = {}
    for agg_i, agg_name in ((1, "mean"), (2, "max")):
        print(f"\n{'='*74}\nRUN-LEVEL ({agg_name} over particles) -- the "
              f"trajectory is the independent unit")
        print(f"{'lookahead':<10}{'n+':>4}{'n-':>5}{'old':>9}{'new':>9}"
              f"{'delta':>9}{'  95% CI on delta':>22}{'P(new>old)':>12}")
        out["run_level"][agg_name] = {}
        for L in LOOKAHEADS:
            rows = runlvl[L]
            P_ = [r[agg_i] for r in rows if r[0]]
            N_ = [r[agg_i] for r in rows if not r[0]]
            if not P_ or not N_:
                continue
            a_o = auc([d["h_phi_old"] for d in P_], [d["h_phi_old"] for d in N_])
            a_n = auc([d["h_phi_new"] for d in P_], [d["h_phi_new"] for d in N_])
            a_s = auc([d["slack"] for d in P_], [d["slack"] for d in N_])
            # PAIRED bootstrap: resample RUNS, recompute both AUCs on the same
            # resample, so the delta's CI is not inflated by shared run noise.
            deltas = []
            for _ in range(B):
                pi = rng.integers(0, len(P_), len(P_))
                ni = rng.integers(0, len(N_), len(N_))
                pp = [P_[i] for i in pi]; nn = [N_[i] for i in ni]
                deltas.append(
                    auc([d["h_phi_new"] for d in pp], [d["h_phi_new"] for d in nn])
                    - auc([d["h_phi_old"] for d in pp], [d["h_phi_old"] for d in nn]))
            deltas = np.asarray(deltas)
            lo, hi = np.percentile(deltas, [2.5, 97.5])
            pwin = float((deltas > 0).mean())
            out["run_level"][agg_name][L] = {
                "n_pos": len(P_), "n_neg": len(N_), "auc_old": a_o,
                "auc_new": a_n, "auc_slack": a_s, "delta": a_n - a_o,
                "delta_ci95": [float(lo), float(hi)], "p_new_gt_old": pwin}
            print(f"{L:<10}{len(P_):>4}{len(N_):>5}{a_o:>9.3f}{a_n:>9.3f}"
                  f"{a_n-a_o:>9.3f}   [{lo:>+6.3f}, {hi:>+6.3f}]{pwin:>12.3f}")
        print(f"  (slack, run level: " + ", ".join(
            f"L{L}={out['run_level'][agg_name][L]['auc_slack']:.3f}"
            for L in LOOKAHEADS if L in out["run_level"][agg_name]) + ")")
    print("\nCI is a PAIRED bootstrap over runs, 4,000 resamples. With only "
          "8 hit runs it will be wide; that width IS the result.")
    return out


@app.local_entrypoint()
def main(stage: str = "extract", new_dir: str = "hphi_v2_h40",
         new_budget_max: int = 40,
         emb_dirs: str = "hphi_v2/embeddings,hphi_v2/embeddings_h40,"
                         "hphi_v2/embeddings_banked") -> None:
    """stage: extract (list states) | score (needs the encode job to have run)."""
    # Strata built EXACTLY as the banked baseline built them, so the run set
    # being compared is the same one that produced 0.796 / 0.667 / 0.556.
    root = Path(__file__).resolve().parents[1]
    g = json.loads((root / "docs/HPHI_SMC_64_GATE_BANKED.json").read_text())
    l = json.loads((root / "docs/HPHI_COVERAGE_LADDER_BANKED.json").read_text())
    r3 = json.loads((root / "docs/HPHI_LADDER_RUNG3.json").read_text())
    rel = set(g["solved_sources"])
    marg = set(l["rung2"]["new_sources"]) | set(r3["conversions"])
    strata = {str(i): ("reliable" if i in rel else
                       "marginal" if i in marg else "hard") for i in range(64)}
    print(f"comparing hphi_v2 (H24) against {new_dir} "
          f"(budget_max={new_budget_max})")
    print(f"strata: {sum(v=='reliable' for v in strata.values())} reliable, "
          f"{sum(v=='marginal' for v in strata.values())} marginal, "
          f"{sum(v=='hard' for v in strata.values())} hard")
    if stage == "extract":
        o = extract.remote(strata)
        print(f"\n{o['n_states']:,} states -> {o['path']}")
        print("Now run hphi_encode_app on it, THEN this app with --stage score.")
        return
    out = compare.remote(strata, new_dir, new_budget_max, emb_dirs)
    Path("docs/PROSPECTIVE_TWO_HEAD.json").write_text(json.dumps(out, indent=1))
    print(f"\nworst parity {out['worst_parity']:.3e}")
