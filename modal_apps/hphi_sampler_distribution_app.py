"""Does the IMPLEMENTED sampler draw from the law it was proved to encode?

The law is verified analytically (1.8e-15 over 15,179 marks) and execution is
verified on 48 coordinates. Neither touches the SAMPLING PROCEDURE -- the family
draw, the empty-family rejection loop, the Gumbel weighted ordering, the
legality rejection. A bug there yields correct per-mark probabilities on paper
and a skewed empirical distribution in practice, which is exactly the failure
that would quietly corrupt science.

Two bugs already found only by running a real trajectory argue for this: a rare
family with no scorer that raised mid-run, and a fallback whose "no mark" was
read by the controller as a dead end and killed the particle. Both lived in the
sampling procedure, not in the law.

METHOD. Pull the exact eager logits, masks and family base logits for a few
fixed states, then run the REAL `sample_one_transition` many times against
stubs: a fake model whose `_encode_batch` returns the precomputed embeddings and
whose `_family_base_logits` returns the precomputed base; scorers that return
the precomputed per-table logits; and a legality predicate that is a lookup into
the exact eager mask. No neural forward, no RDKit, no chemistry -- so 100k draws
run in seconds and the ONLY thing under test is the control flow.

The rejection path is exercised for real: `atom_restate` and `cycle_insert` are
given their structural SUPERSET as the candidate set, so the sampler must reject
its way down to the exact mask, just as it does in production.

COMPARISON. Empirical frequencies against the analytic law, reported as total
variation distance and a chi-square p-value over the legal marks, plus the
family marginals. Sampling error is expected; structure is not.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import modal

from modal_apps.run_process_v2_p50_app import (
    ARTIFACT_ROOT,
    REMOTE_ROOT,
    artifact_volume,
)
from modal_apps.run_process_v2_p50_app import image as _base_image

image = _base_image.env({"PYTHONPATH": f"{REMOTE_ROOT}/src:{REMOTE_ROOT}"})
app = modal.App("hphi-sampler-distribution")
RUN_ROOT = "/artifacts/editing_v2/r_theta_run"
TIME_POINT, CANONICAL_SLOTS = 0.5, 48


@app.function(image=image, cpu=(2.0, 2.0), memory=6144, timeout=90 * 60,
              volumes={str(ARTIFACT_ROOT): artifact_volume})
def check(srcs: list[str], n_draws: int) -> dict[str, Any]:
    import sys

    import numpy as np
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
    from rdkit import RDLogger

    RDLogger.DisableLog("rdApp.*")
    from compose_v4.chem.molecular_graph import is_element, smiles_to_molecular_graph
    from compose_v4.chem.state import pad_molecular_graph
    from compose_v4.experiments import hphi_lazy_sampler as LS
    from compose_v4.experiments.production_successor_kernel import (
        MARK_RULE_NAMES, _one_state_batch,
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

    from compose_v4.experiments.hphi_lazy_sampler import _FAMILY_TABLE

    results = []
    for smi in srcs:
        st = pad_molecular_graph(smiles_to_molecular_graph(smi), CANONICAL_SLOTS)
        batch = _one_state_batch(model, st, float(TIME_POINT), prepared_batch=None)
        node, glob, pair = model._encode_batch(batch)
        masks, logits, action_log_z = model._action_tables(
            batch, node, glob, pair, require_exact_ring_support=False)
        base = model._family_base_logits(batch, glob)[0].double()

        # ---- ANALYTIC law over (table, coordinate) ----------------------
        enabled = torch.isfinite(action_log_z)[0]
        fam_lg = base.clone()
        fam_lg[~enabled] = float("-inf")
        fam_p = torch.softmax(fam_lg, dim=-1)
        analytic: dict[tuple, float] = {}
        for fi, fam in enumerate(MARK_RULE_NAMES):
            table = _FAMILY_TABLE.get(fam, fam)
            mk = masks.get(table)
            if mk is None or not bool(mk[0].any()):
                continue
            lg = logits[table][0].double()
            m = mk[0].bool()
            w = torch.exp(lg[m] - torch.logsumexp(lg[m], dim=0))
            coords = torch.nonzero(m, as_tuple=False)
            for c, wi in zip(coords, w):
                analytic[(table, tuple(int(v) for v in c))] = float(fam_p[fi] * wi)
        tot = sum(analytic.values())
        if tot <= 0:
            continue
        analytic = {k: v / tot for k, v in analytic.items()}

        # ---- STUBS: keep the control flow, remove the cost ---------------
        real = torch.from_numpy(np.asarray(is_element(st.atom_types), dtype=bool))
        n = real.shape[0]
        bonds = torch.from_numpy(np.asarray(st.bonds)).long()
        hyd = torch.from_numpy(np.asarray(st.implicit_h_counts)).long()
        upper = torch.triu(torch.ones(n, n, dtype=torch.bool), diagonal=1)

        supersets: dict[str, torch.Tensor] = {}
        for fam in MARK_RULE_NAMES:
            table = _FAMILY_TABLE.get(fam, fam)
            mk = masks.get(table)
            if mk is None:
                continue
            if table == "atom_restate":
                # SUPERSET, so the rejection path is genuinely exercised.
                supersets[table] = real.view(-1, 1).expand(
                    n, len(model.atom_vocabulary)).clone().unsqueeze(0)
            elif table == "cycle_insert":
                s = torch.zeros(n, n, 3, dtype=torch.bool)
                pr = real.view(-1, 1) & real.view(1, -1)
                for k, o in enumerate((1, 2, 3)):
                    s[:, :, k] = (upper & pr & (bonds == 0)
                                  & (hyd.view(-1, 1) >= o) & (hyd.view(1, -1) >= o))
                supersets[table] = s.unsqueeze(0)
            else:
                supersets[table] = mk[0].unsqueeze(0)

        class FakeModel:
            def _encode_batch(self, b):
                return node, glob, pair

            def _family_base_logits(self, b, g):
                return base.unsqueeze(0)

        stub_scorers = {
            t: (lambda m, nd, gl, pr, bt, _t=t: logits[_t])
            for t in logits
        }
        saved = dict(LS.__dict__.get("_stub_scorers", {}))  # noqa: F841

        import compose_v4.experiments.hphi_lazy_family_scores as SC
        real_scorers = dict(SC.LAZY_SCORERS)
        SC.LAZY_SCORERS.clear()
        SC.LAZY_SCORERS.update(stub_scorers)

        def legality_from_mask(table):
            m = masks[table][0].bool()
            return lambda _st, coord: bool(m[coord])

        helpers = {
            "build_batch": lambda _s, _t: None,
            "family_names": list(MARK_RULE_NAMES),
            "family_mask": (lambda _m, table, _s, _b, _p, _g, _n:
                            supersets.get(table)),
            "legality": {t: legality_from_mask(t)
                         for t in ("atom_restate", "cycle_insert")
                         if t in masks},
            "eager_fallback": lambda _s, _t, _r: (None, None),
        }

        rng = np.random.default_rng(4242)
        fake = FakeModel()
        counts = Counter()
        none_draws = 0
        for _ in range(n_draws):
            d = LS.sample_one_transition(fake, st, TIME_POINT, rng, helpers=helpers)
            if d.table is None:
                none_draws += 1
                continue
            counts[(d.table, d.coordinate)] += 1
        SC.LAZY_SCORERS.clear()
        SC.LAZY_SCORERS.update(real_scorers)

        n_ok = sum(counts.values())
        tv = 0.5 * sum(abs(counts.get(k, 0) / max(n_ok, 1) - p)
                       for k, p in analytic.items())
        tv += 0.5 * sum(v / max(n_ok, 1) for k, v in counts.items()
                        if k not in analytic)
        chi = sum((counts.get(k, 0) - n_ok * p) ** 2 / (n_ok * p)
                  for k, p in analytic.items() if n_ok * p >= 5)
        dof = sum(1 for p in analytic.values() if n_ok * p >= 5) - 1
        illegal = sum(v for k, v in counts.items() if k not in analytic)

        fam_emp = Counter()
        for (t, _c), v in counts.items():
            fam_emp[t] += v
        fam_an = Counter()
        for (t, _c), p in analytic.items():
            fam_an[t] += p

        print(f"\n{smi[:44]}")
        print(f"  legal marks {len(analytic)}   draws {n_ok}   "
              f"no-mark draws {none_draws}   ILLEGAL draws {illegal}")
        print(f"  total variation {tv:.5f}   chi2 {chi:.1f} on {dof} dof")
        print(f"  {'family':<22}{'analytic':>10}{'empirical':>11}")
        for t in sorted(fam_an, key=lambda k: -fam_an[k]):
            print(f"    {t:<20}{fam_an[t]:>10.4f}"
                  f"{fam_emp.get(t,0)/max(n_ok,1):>11.4f}")
        results.append({"smiles": smi, "n_legal": len(analytic), "draws": n_ok,
                        "total_variation": tv, "chi2": chi, "dof": dof,
                        "illegal_draws": illegal, "no_mark_draws": none_draws,
                        "family_analytic": dict(fam_an),
                        "family_empirical": {k: v / max(n_ok, 1)
                                             for k, v in fam_emp.items()}})

    worst_tv = max((r["total_variation"] for r in results), default=1.0)
    any_illegal = any(r["illegal_draws"] for r in results)
    print(f"\n{'='*66}")
    print(f"  worst total variation {worst_tv:.5f}")
    print(f"  illegal draws anywhere: {'YES' if any_illegal else 'NO'}")
    print("\n  A sampler drawing the right law shows TV shrinking like 1/sqrt(n)")
    print("  and NEVER returns a coordinate outside the exact legal support.")
    return {"results": results, "worst_total_variation": worst_tv,
            "any_illegal": any_illegal, "n_draws": n_draws}


@app.local_entrypoint()
def main(n_states: int = 3, n_draws: int = 120000) -> None:
    srcs = [s.strip() for s in
            (Path(__file__).resolve().parents[1]
             / "data/jin/dev_panel_qed_64.txt").read_text().split("\n")
            if s.strip()][:n_states]
    out = check.remote(srcs, n_draws)
    Path("docs/SAMPLER_DISTRIBUTION.json").write_text(json.dumps(out, indent=1))
    print(f"\nworst TV {out['worst_total_variation']:.5f}   "
          f"illegal draws: {'YES' if out['any_illegal'] else 'NO'}")
