"""Can the frozen law be sampled EXACTLY without building the full legality mask?

DIAGNOSTIC ONLY. Builds no sampler and changes nothing. Answers the two
questions that decide whether an exact rejection sampler is worth writing.

The law implemented by `enumerate_factorized_marked_law` is

    log p(f, a) = log_softmax(family_logits)[f] + s_f(a) - Z_f
    family_logits = base_f + Z_f   (superposed)  |  base_f  (hierarchical)
    Z_f = logsumexp over LEGAL coordinates of family f

Under `superposed` the Z_f cancels and the law is a single global masked softmax

    p(f, a)  proportional to  exp(base_f + s_f(a)) * 1[a legal]

for which sampling from the UNMASKED distribution and rejecting illegal draws is
exactly correct, with no normalizer and no full mask. Under `hierarchical` the
family marginal is defined independently of the coordinate scores, and family
probabilities are additionally restricted to families that HAVE a legal action,
which cannot be established by rejection alone -- proving a family empty needs an
exhaustive check. So the factorization is not a detail; it decides whether the
clean construction exists.

Question two is acceptance, and it must be measured by PROBABILITY MASS rather
than by count. Roughly 126 of ~1300 cycle-close coordinates are legal, which
would suggest ~10 checks per accepted sample -- but the model was trained with
these masks applied, so its logits on illegal coordinates were never trained
against anything and may carry arbitrary mass. If the raw law puts almost all of
its mass on illegal coordinates, rejection is useless despite a 10% legal count.

    alpha_f = sum over legal a of exp(s_f(a))  /  sum over ALL a of exp(s_f(a))
            = exp(Z_f_masked - Z_f_unmasked)

and 1/alpha_f is the expected number of expensive legality checks per sample.
Both quantities come straight out of the existing `_action_tables` output, so
this measures the real frozen model with no new machinery.

Also reports alpha under a cheap NECESSARY-CONDITION superset -- coordinates
whose atoms are real and, for cycle-close, that have no existing bond -- since
padding slots in a 48-slot table are trivially illegal and should not be counted
against the scheme.
"""

from __future__ import annotations

import json
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
app = modal.App("hphi-rejection-diag")
RUN_ROOT = "/artifacts/editing_v2/r_theta_run"
TIME_POINT, CANONICAL_SLOTS = 0.5, 48


@app.function(image=image, cpu=(1.0, 1.0), memory=6144, timeout=45 * 60,
              volumes={str(ARTIFACT_ROOT): artifact_volume})
def diagnose(srcs: list[str]) -> dict[str, Any]:
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
    from compose_v4.experiments.production_successor_kernel import _one_state_batch

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

    # ---- QUESTION 1: which factorization? ----------------------------------
    factorization = str(getattr(model, "rate_factorization", "?"))
    print(f"\nrate_factorization = {factorization!r}")
    if factorization == "superposed":
        print("  -> family_logits = base + Z_f, the Z_f cancels, and the law is")
        print("     a SINGLE GLOBAL MASKED SOFTMAX. Global rejection is exact.")
    else:
        print("  -> family marginal is NOT the global softmax. Family "
              "probabilities are\n     restricted to families having a legal "
              "action, which rejection alone\n     cannot establish; proving a "
              "family empty needs an exhaustive check.")

    # ---- QUESTION 2: acceptance by PROBABILITY MASS ------------------------
    per_family: dict[str, list[float]] = {}
    per_family_cheap: dict[str, list[float]] = {}
    counts: dict[str, list[tuple[int, int]]] = {}

    for smi in srcs:
        st = pad_molecular_graph(smiles_to_molecular_graph(smi), CANONICAL_SLOTS)
        batch = _one_state_batch(model, st, float(TIME_POINT), prepared_batch=None)
        node, glob, pair = model._encode_batch(batch)
        masks, logits, _z = model._action_tables(
            batch, node, glob, pair, require_exact_ring_support=False)

        real = torch.from_numpy(
            np.asarray(is_element(st.atom_types), dtype=bool))
        bonds = torch.from_numpy(np.asarray(st.bonds))

        for table_name, m in masks.items():
            lg = logits[table_name][0].double()
            mk = m[0].bool()
            n_legal = int(mk.sum())
            if n_legal == 0:
                continue
            # Full table, including padding slots.
            all_mass = torch.logsumexp(lg.reshape(-1), dim=0)
            legal_mass = torch.logsumexp(lg[mk], dim=0)
            alpha = float(torch.exp(legal_mass - all_mass))
            per_family.setdefault(table_name, []).append(alpha)
            counts.setdefault(table_name, []).append((n_legal, int(mk.numel())))

            # Cheap NECESSARY-condition superset: real atoms on every indexed
            # axis, and for a pairwise table no bond already present. Every
            # legal coordinate satisfies these, so restricting to them cannot
            # drop a legal action -- it only removes trivially-illegal padding.
            cheap = torch.ones_like(mk)
            nd = mk.dim()
            if nd >= 1 and mk.shape[0] == real.shape[0]:
                cheap &= real.reshape([-1] + [1] * (nd - 1))
            if nd >= 2 and mk.shape[1] == real.shape[0]:
                cheap &= real.reshape([1, -1] + [1] * (nd - 2))
                if mk.shape[0] == real.shape[0]:
                    nb = (bonds == 0)
                    cheap &= nb.reshape(list(nb.shape) + [1] * (nd - 2))
            if not bool((cheap & mk).sum() == n_legal):
                # A necessary condition that drops a legal action is NOT
                # necessary. Report rather than silently use it.
                per_family_cheap.setdefault(table_name, []).append(float("nan"))
                continue
            cheap_mass = torch.logsumexp(lg[cheap], dim=0)
            per_family_cheap.setdefault(table_name, []).append(
                float(torch.exp(legal_mass - cheap_mass)))
        print(f"  {smi[:38]:<38} done", flush=True)

    def pct(v, q):
        v = sorted(x for x in v if x == x)
        return v[min(len(v) - 1, int(q * len(v)))] if v else float("nan")

    print(f"\n{'table':<26}{'legal/total':>14}{'alpha':>9}{'1/alpha':>10}"
          f"{'cheap 1/a':>11}")
    out = {}
    for t, vals in sorted(per_family.items(), key=lambda kv: -len(kv[1])):
        c = counts[t]
        nl = sum(x for x, _ in c) / len(c)
        nt = sum(y for _, y in c) / len(c)
        med = pct(vals, 0.5)
        ch = per_family_cheap.get(t, [])
        medc = pct(ch, 0.5)
        out[t] = {"alpha_median": med, "n_legal_mean": nl, "n_total_mean": nt,
                  "alpha_p10": pct(vals, 0.1), "alpha_p90": pct(vals, 0.9),
                  "alpha_cheap_median": medc}
        print(f"  {t:<24}{nl:>7.0f}/{nt:<6.0f}{med:>9.4f}"
              f"{1/med if med else float('inf'):>10.1f}"
              f"{(1/medc if medc == medc and medc else float('nan')):>11.1f}")

    print("\n1/alpha is the EXPECTED NUMBER OF EXPENSIVE LEGALITY CHECKS per "
          "accepted sample.\nThe gate: under ~20 is worth building; near the "
          "~1300 we pay now is not.")
    return {"rate_factorization": factorization, "tables": out,
            "n_states": len(srcs)}


@app.local_entrypoint()
def main(n_states: int = 8) -> None:
    srcs = [s.strip() for s in
            (Path(__file__).resolve().parents[1]
             / "data/jin/dev_panel_qed_64.txt").read_text().split("\n")
            if s.strip()][:n_states]
    out = diagnose.remote(srcs)
    Path("docs/REJECTION_DIAGNOSTIC.json").write_text(json.dumps(out, indent=1))
    print("\nwrote docs/REJECTION_DIAGNOSTIC.json")
