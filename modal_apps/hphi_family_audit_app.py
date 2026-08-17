"""Operator-native candidate supersets, and how often a family is empty.

Two diagnostics that together decide whether an exact rejection sampler is worth
writing. Builds no sampler.

AUDIT A -- structural candidate space vs padded table.

`bond_reorder` reported 6 legal coordinates out of 6912, implying ~2015 draws per
legal action. But 6912 is 48*48*3: every ordered atom pair crossed with every
bond class, in a PADDED table. The operator means "take an EXISTING bond and
change its order", and whether a bond exists is an integer lookup in the bond
matrix, not chemistry. Starting rejection from the padded table is a statement
about the training representation, not about the operator.

So for each family this constructs the OPERATOR-NATIVE superset directly from
the pre-RDKit conditions in the corresponding `is_valid_*` function -- trivial
graph facts only, no semantic resolver, no RDKit -- and then:

  * VERIFIES containment: every currently legal coordinate must survive. A
    condition that drops a legal mark is not necessary, and using it would
    silently change the law. Reported as a failure, never quietly applied.
  * recomputes acceptance by PROBABILITY MASS over the superset, since that,
    not the count, is what a rejection sampler actually pays.

AUDIT B -- empty families under the hierarchical factorization.

`rate_factorization` is 'hierarchical', so p(f) is a softmax over families that
HAVE a legal action. Family-level rejection is exact -- draw f from the
unrestricted softmax, accept if enabled, redraw otherwise -- but deciding
"enabled" for a drawn family means finding a legal action (cheap when the family
is dense) or PROVING none exists (expensive). The cost therefore depends on how
often an empty family is drawn, which is exactly the raw family probability mass
sitting on empty families. That is measured here.
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
app = modal.App("hphi-family-audit")
RUN_ROOT = "/artifacts/editing_v2/r_theta_run"
TIME_POINT, CANONICAL_SLOTS = 0.5, 48


@app.function(image=image, cpu=(1.0, 1.0), memory=6144, timeout=60 * 60,
              volumes={str(ARTIFACT_ROOT): artifact_volume})
def audit(srcs: list[str]) -> dict[str, Any]:
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
    from compose_v4.chem.molecular_graph import (
        BOND_AROMATIC, is_element, smiles_to_molecular_graph,
    )
    from compose_v4.experiments.production_successor_kernel import (
        _TABLE_FAMILIES, MARK_RULE_TO_INDEX, _one_state_batch,
    )
    from compose_v4.chem.state import pad_molecular_graph

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

    # Use the model's OWN hydrogen-change table and limit rather than
    # reimplementing them; a divergence here would silently produce a superset
    # that is not a superset.
    from compose_v4.chem.molecular_graph import BOND_CLASS_TO_H_CHANGE, MAX_H_COUNT

    HCH = torch.tensor([int(BOND_CLASS_TO_H_CHANGE[i])
                        for i in range(len(BOND_CLASS_TO_H_CHANGE))])

    def supersets(st, shape_by_table):
        """Operator-native cheap supersets. Trivial graph facts ONLY -- no
        semantic resolver, no RDKit. Each mirrors the pre-RDKit conditions of
        the corresponding `is_valid_*` function."""
        b = torch.from_numpy(np.asarray(st.bonds)).long()
        h = torch.from_numpy(np.asarray(st.implicit_h_counts)).long()
        real = torch.from_numpy(np.asarray(is_element(st.atom_types), dtype=bool))
        n = b.shape[0]
        upper = torch.triu(torch.ones(n, n, dtype=torch.bool), diagonal=1)
        pair_real = real.view(-1, 1) & real.view(1, -1)
        h_old = HCH[b.clamp(min=0, max=len(HCH) - 1)]          # per-pair, from stored order
        out = {}

        # bond_reorder: an EXISTING, non-aromatic bond whose order CHANGES, with
        # both endpoints retaining a legal hydrogen count. This is the family
        # that looked hopeless at 6/6912 -- 6912 being 48*48*3, the padded table.
        if "bond_reorder" in shape_by_table:
            m = torch.zeros(n, n, 3, dtype=torch.bool)
            for k, new_order in enumerate((1, 2, 3)):
                delta = int(BOND_CLASS_TO_H_CHANGE[new_order]) - h_old
                nha = h.view(-1, 1) - delta
                nhb = h.view(1, -1) - delta
                m[:, :, k] = (upper & pair_real & (b != 0) & (b != BOND_AROMATIC)
                              & (b != new_order)
                              & (nha >= 0) & (nha <= MAX_H_COUNT)
                              & (nhb >= 0) & (nhb <= MAX_H_COUNT))
            out["bond_reorder"] = m

        # cycle_insert (cycle-close): NO existing bond, H capacity >= order.
        # These are exactly the conditions the frozen legacy path already uses
        # before the semantic admission mask replaces it.
        if "cycle_insert" in shape_by_table:
            m = torch.zeros(n, n, 3, dtype=torch.bool)
            for k, order in enumerate((1, 2, 3)):
                m[:, :, k] = (upper & pair_real & (b == 0)
                              & (h.view(-1, 1) >= order) & (h.view(1, -1) >= order))
            out["cycle_insert"] = m

        # cycle_attach (cycle-open): an EXISTING bond between real atoms.
        if "cycle_attach" in shape_by_table:
            out["cycle_attach"] = upper & pair_real & (b != 0)

        if "bond_reroute" in shape_by_table:
            out["bond_reroute"] = upper & pair_real

        for t in ("atom_restate", "atom_delete", "grow_connected", "grow_root"):
            if t in shape_by_table:
                shp = shape_by_table[t]
                if shp and shp[0] == n:
                    v = real.reshape([-1] + [1] * (len(shp) - 1))
                    out[t] = v.expand(shp).clone()
        return out

    tables: dict[str, dict[str, list]] = {}
    empty_events: dict[str, int] = {}
    empty_mass: list[float] = []
    n_states = 0

    for smi in srcs:
        st = pad_molecular_graph(smiles_to_molecular_graph(smi), CANONICAL_SLOTS)
        batch = _one_state_batch(model, st, float(TIME_POINT), prepared_batch=None)
        node, glob, pair = model._encode_batch(batch)
        masks, logits, action_log_z = model._action_tables(
            batch, node, glob, pair, require_exact_ring_support=False)
        n_states += 1

        shape_by_table = {t: tuple(m[0].shape) for t, m in masks.items()}
        sup = supersets(st, shape_by_table)

        for t, m in masks.items():
            mk = m[0].bool()
            lg = logits[t][0].double()
            n_legal = int(mk.sum())
            rec = tables.setdefault(t, {"legal": [], "padded": [], "sup": [],
                                        "alpha_pad": [], "alpha_sup": [],
                                        "contained": []})
            rec["legal"].append(n_legal)
            rec["padded"].append(int(mk.numel()))
            if n_legal == 0:
                empty_events[t] = empty_events.get(t, 0) + 1
                continue
            legal_mass = torch.logsumexp(lg[mk], dim=0)
            rec["alpha_pad"].append(
                float(torch.exp(legal_mass - torch.logsumexp(lg.reshape(-1), 0))))
            s = sup.get(t)
            if s is None:
                continue
            s = s.bool()
            contained = bool(((mk & ~s).sum()) == 0)
            rec["contained"].append(contained)
            rec["sup"].append(int(s.sum()))
            if contained and int(s.sum()) > 0:
                rec["alpha_sup"].append(
                    float(torch.exp(legal_mass - torch.logsumexp(lg[s], 0))))

        # --- AUDIT B: raw family mass sitting on EMPTY families -------------
        base = model._family_base_logits(batch, glob)[0].double()
        enabled = torch.isfinite(action_log_z)[0]
        p = torch.softmax(base, dim=-1)
        empty_mass.append(float(p[~enabled].sum()))
        print(f"  {smi[:36]:<36} empty-family raw mass "
              f"{empty_mass[-1]*100:5.1f}%", flush=True)

    def med(v):
        v = sorted(x for x in v if x == x)
        return v[len(v) // 2] if v else float("nan")

    print(f"\n{'table':<18}{'legal':>7}{'padded':>8}{'superset':>10}"
          f"{'1/a pad':>10}{'1/a super':>11}{'contains?':>11}{'empty':>7}")
    out = {}
    for t, rec in sorted(tables.items()):
        ap, asup = med(rec["alpha_pad"]), med(rec["alpha_sup"])
        cont = ("YES" if all(rec["contained"]) else "NO")
        if not rec["contained"]:
            cont = "-"
        out[t] = {
            "legal_median": med(rec["legal"]), "padded": med(rec["padded"]),
            "superset_median": med(rec["sup"]), "alpha_padded": ap,
            "alpha_superset": asup, "superset_contains_all_legal": cont,
            "empty_states": empty_events.get(t, 0)}
        print(f"  {t:<16}{med(rec['legal']):>7.0f}{med(rec['padded']):>8.0f}"
              f"{med(rec['sup']) if rec['sup'] else float('nan'):>10.0f}"
              f"{(1/ap if ap else float('nan')):>10.1f}"
              f"{(1/asup if asup == asup and asup else float('nan')):>11.1f}"
              f"{cont:>11}{empty_events.get(t,0):>7}")

    print(f"\nEMPTY-FAMILY RAW MASS over {n_states} states: "
          f"median {med(empty_mass)*100:.2f}%  max {max(empty_mass)*100:.2f}%")
    print("That is the probability a family-level rejection draw lands on a "
          "family with no\nlegal action, which is the only case that forces an "
          "exhaustive emptiness proof.")
    return {"tables": out, "empty_mass_median": med(empty_mass),
            "empty_mass_max": max(empty_mass) if empty_mass else None,
            "n_states": n_states}


@app.local_entrypoint()
def main(n_states: int = 12) -> None:
    srcs = [s.strip() for s in
            (Path(__file__).resolve().parents[1]
             / "data/jin/dev_panel_qed_64.txt").read_text().split("\n")
            if s.strip()][:n_states]
    out = audit.remote(srcs)
    Path("docs/FAMILY_AUDIT.json").write_text(json.dumps(out, indent=1))
    print("\nwrote docs/FAMILY_AUDIT.json")
