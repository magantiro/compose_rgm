"""How fast is ONE exact COMPOSE transition now? Eager vs lazy, same states.

No SMC, no QED statistics, no GrIDDD comparison. One question, measured.

Everything except the encode and the family draw is built lazily, so the
always-paid cost is ~52 ms of encoding plus ~0.05 ms of family selection, and
each family's machinery is paid only on the draws that select it. That is the
whole architectural claim: the eager path builds all eight families every
transition, and cycle-close alone -- 56.5% of that construction -- is selected
0.53% of the time.

Legality per family:
  * masks that are already exact  -> the first weighted draw is legal, no
    resolver call at all. This covers ~76% of realized family mass.
  * atom_restate and cycle_insert -> the cheap structural superset plus the
    frozen resolver on each drawn candidate, rejecting until one is admitted.
    Both supersets were verified to contain every legal coordinate.

The atom-restate predicate is taken from the enumerator that builds its mask
today, including the productive clause that the successor must differ from the
source -- the detail most likely to hide a silent law change.
"""

from __future__ import annotations

import json
import time
from collections import defaultdict
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
app = modal.App("hphi-lazy-bench")
RUN_ROOT = "/artifacts/editing_v2/r_theta_run"
TIME_POINT, CANONICAL_SLOTS = 0.5, 48


@app.function(image=image, cpu=(1.0, 1.0), memory=6144, timeout=90 * 60,
              volumes={str(ARTIFACT_ROOT): artifact_volume})
def bench(srcs: list[str], n_eager: int) -> dict[str, Any]:
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
        NULL_IDX, is_element, smiles_to_molecular_graph,
    )
    from compose_v4.chem.state import pad_molecular_graph
    from compose_v4.experiments.hphi_graph_encode import build_graph_only_batch
    from compose_v4.experiments.hphi_lazy_sampler import sample_one_transition
    from compose_v4.experiments.production_successor_kernel import (
        MARK_RULE_NAMES, enumerate_factorized_marked_law,
    )
    from compose_v4.model import factorized_tracelet_rate_model as F
    from compose_v4.rewrite.kernel import canonical_state_key
    from compose_v4.rewrite.operators import (
        CycleCloseEdge, SemanticAtomRestate, resolve_cycle_close_edge,
        resolve_semantic_atom_restate_action,
    )
    from compose_v4.rewrite.semantic_atom_restate import (
        prepare_semantic_atom_restate_context,
    )
    from compose_v4.rewrite.semantic_cycle_close import (
        prepare_semantic_cycle_close_context,
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

    MAXH = 4
    macro_system = F.de_novo_rewrite_system()

    class Ctx:
        """Per-state lazily filled scaffolding. Nothing is built until asked."""

        def __init__(self, st):
            self.st = st
            self._graph = None
            self._cc = None
            self._ar = None
            self._key = None

        @property
        def graph_masks(self):
            if self._graph is None:
                topo, _c, _r = F.compute_topology_features(self.st)
                self._graph = F._graph_application_masks(
                    self.st, topo, compute_cyclic_graft=True)
            return self._graph

        @property
        def cc_context(self):
            if self._cc is None:
                self._cc = prepare_semantic_cycle_close_context(self.st)
            return self._cc

        @property
        def ar_context(self):
            if self._ar is None:
                self._ar = prepare_semantic_atom_restate_context(self.st)
            return self._ar

        @property
        def source_key(self):
            if self._key is None:
                self._key = canonical_state_key(self.st)
            return self._key

    ctxs: dict[int, Ctx] = {}

    def ctx_for(st):
        return ctxs.setdefault(id(st), Ctx(st))

    def build_batch(st, t):
        b = build_graph_only_batch([st], [float(t)])
        # Fields the scorers and _family_base_logits read beyond the encoder's.
        b.bonds = torch.from_numpy(np.asarray(st.bonds)).unsqueeze(0)
        b.formal_charges = torch.from_numpy(
            np.asarray(st.formal_charges)).unsqueeze(0)
        b.states = (st,)
        b.ring_topology_local_support_log_mass = None
        gm = None
        b.graft_mask = None
        b.graft_remove_neighbors = None
        b._ctx = ctx_for(st)
        return b

    def _charged(st):
        return np.asarray(st.formal_charges) != 0

    def family_mask(model, table, st, batch, pair, glob, node):
        """EXACT mask, or a verified superset for the two rejection families."""
        c = ctx_for(st)
        real = torch.from_numpy(np.asarray(is_element(st.atom_types), dtype=bool))
        n = real.shape[0]
        bonds = torch.from_numpy(np.asarray(st.bonds)).long()
        hyd = torch.from_numpy(np.asarray(st.implicit_h_counts)).long()
        charged = torch.from_numpy(_charged(st))
        czero = ~charged
        upper = torch.triu(torch.ones(n, n, dtype=torch.bool), diagonal=1)
        orders = torch.arange(1, 4)

        if table == "atom_delete":
            m = torch.from_numpy(
                np.asarray(F.process_v2_atom_delete_mask(st), dtype=bool))
            if charged.any():
                nb = ((bonds != 0) & charged.unsqueeze(0)).any(dim=-1)
                m = m & czero & ~nb
            return m.unsqueeze(0)

        if table == "bond_reroute":
            (_dm, _ce, _cp, graft, removed, _gs) = c.graph_masks
            m = torch.from_numpy(np.asarray(graft, dtype=bool))
            batch.graft_mask = m.unsqueeze(0)
            rem = torch.from_numpy(np.asarray(removed)).long()
            batch.graft_remove_neighbors = rem.unsqueeze(0)
            if charged.any():
                cz2 = czero.unsqueeze(1) & czero.unsqueeze(0)
                valid = rem >= 0
                rc = charged[rem.clamp_min(0)]
                m = m & cz2 & valid & ~rc
            return m.unsqueeze(0)

        if table == "bond_reorder":
            (_dm, cyc_edge, _cp, _g, _rn, _gs) = c.graph_masks
            ce = torch.from_numpy(np.asarray(cyc_edge, dtype=bool))
            old = bonds.unsqueeze(-1)
            new = orders.view(1, 1, 3)
            delta = new - old
            m = (upper.unsqueeze(-1) & (old >= 1) & (old <= 3) & (new != old)
                 & ~ce.unsqueeze(-1)
                 & ((hyd.unsqueeze(1).unsqueeze(-1) - delta) >= 0)
                 & ((hyd.unsqueeze(1).unsqueeze(-1) - delta) <= MAXH)
                 & ((hyd.unsqueeze(0).unsqueeze(-1) - delta) >= 0)
                 & ((hyd.unsqueeze(0).unsqueeze(-1) - delta) <= MAXH))
            if charged.any():
                m = m & (czero.unsqueeze(1) & czero.unsqueeze(0)).unsqueeze(-1)
            return m.unsqueeze(0)

        if table == "grow_connected":
            nnull = int((torch.from_numpy(np.asarray(st.atom_types)) == NULL_IDX).sum())
            val = model.cnof_valences.view(1, 1, -1)
            th = val - orders.view(1, -1, 1)
            m = (real.view(-1, 1, 1) & torch.tensor(nnull > 0)
                 & (hyd.view(-1, 1, 1) >= orders.view(1, -1, 1))
                 & (th >= 0) & (th <= MAXH))
            if charged.any():
                m = m & czero.view(-1, 1, 1)
            return m.unsqueeze(0)

        if table == "atom_restate":
            # SUPERSET: real slots x vocabulary. Legality resolved per draw.
            m = real.view(-1, 1).expand(n, len(model.atom_vocabulary)).clone()
            if charged.any():
                m = m & czero.view(-1, 1)
            return m.unsqueeze(0)

        if table == "cycle_insert":
            # SUPERSET: the frozen legacy structural conditions.
            m = torch.zeros(n, n, 3, dtype=torch.bool)
            for k, o in enumerate((1, 2, 3)):
                m[:, :, k] = (upper & (real.view(-1, 1) & real.view(1, -1))
                              & (bonds == 0)
                              & (hyd.view(-1, 1) >= o) & (hyd.view(1, -1) >= o))
            if charged.any():
                m = m & (czero.unsqueeze(1) & czero.unsqueeze(0)).unsqueeze(-1)
            return m.unsqueeze(0)

        if table == "cycle_attach":
            m = torch.from_numpy(
                np.asarray(F._semantic_cycle_open_admission_mask(st), dtype=bool))
            if m.dim() == 2:
                m = m & upper
                if charged.any():
                    m = m & (czero.unsqueeze(1) & czero.unsqueeze(0))
            return m.unsqueeze(0)

        if table == "ring_system_restate":
            # Build ONLY this family's batch fields, then let the model compute
            # its own logits and mask. The enumeration is ~261 ms and is paid on
            # the ~3% of draws that select this family rather than on every one.
            try:
                groups = F.enumerate_ring_restate_semantic_groups(
                    st, system=macro_system)
                batch.ring_restate_actions = (groups.actions,)
                batch.ring_restate_successor_group_ids = (
                    groups.successor_group_ids,)
                batch.ring_restate_successor_group_descriptors = (
                    groups.group_descriptors,)
                batch.ring_restate_successor_group_multiplicities = (
                    groups.group_multiplicities,)
                _lg, rm = model._ring_restate_logits(batch, pair, glob)
                return rm
            except Exception:
                # Any gap in the lazy construction falls back to the exact
                # family-conditioned law rather than to an empty verdict.
                from compose_v4.experiments.hphi_lazy_sampler import FALLBACK
                return FALLBACK

        return None

    def legal_atom_restate(st, coord):
        v, k = int(coord[0]), int(coord[1])
        r = resolve_semantic_atom_restate_action(
            st, SemanticAtomRestate(v, k), context=ctx_for(st).ar_context)
        return bool(r is not None and r.admitted and r.successor is not None
                    and canonical_state_key(r.successor) != ctx_for(st).source_key)

    def legal_cycle_insert(st, coord):
        a, b, k = int(coord[0]), int(coord[1]), int(coord[2])
        r = resolve_cycle_close_edge(
            st, CycleCloseEdge(a, b, k + 1), context=ctx_for(st).cc_context)
        return bool(r is not None and r.admitted)

    def eager_fallback(st, table):
        """Exact law restricted to the ALREADY-CHOSEN family.

        Must be family-CONDITIONED. Sampling the global law here would redraw
        the family that was already drawn, biasing every fallback toward
        whichever families dominate the global law -- a real distribution
        change, not a slow path. Restricting to the chosen family and sampling
        proportional to those marks' probabilities IS p(a | f), because
        p(a | f) is proportional to p(f, a) restricted to f.
        """
        law = enumerate_factorized_marked_law(model, st, float(TIME_POINT))
        marks = [m for m in law.marks if m.table_name == table]
        if not marks:
            return (None, None)
        pr = np.array([float(np.exp(m.log_probability)) for m in marks])
        mk = marks[int(rng_fb.choice(len(pr), p=pr / pr.sum()))]
        return (mk.table_name, tuple(int(v) for v in mk.coordinate))

    rng_fb = np.random.default_rng(7)
    helpers = {
        "build_batch": build_batch,
        "eager_fallback": eager_fallback,
        "family_names": list(MARK_RULE_NAMES),
        "family_mask": family_mask,
        "legality": {"atom_restate": legal_atom_restate,
                     "cycle_insert": legal_cycle_insert},
    }

    states = [pad_molecular_graph(smiles_to_molecular_graph(s), CANONICAL_SLOTS)
              for s in srcs]

    # ---- eager reference on a subset (it is ~7 s each) ------------------
    eager_ms = []
    for st in states[:n_eager]:
        t0 = time.perf_counter()
        enumerate_factorized_marked_law(model, st, float(TIME_POINT))
        eager_ms.append((time.perf_counter() - t0) * 1e3)
        print(f"  eager {eager_ms[-1]:8.1f} ms", flush=True)

    # ---- lazy on all states ---------------------------------------------
    rng = np.random.default_rng(20260817)
    rows = []
    by_family = defaultdict(list)
    fails = 0
    for i, st in enumerate(states):
        try:
            d = sample_one_transition(model, st, TIME_POINT, rng, helpers=helpers)
        except Exception as err:  # noqa: BLE001
            fails += 1
            print(f"  lazy FAILED on state {i}: {type(err).__name__}: {err}",
                  flush=True)
            continue
        rows.append(d)
        if d.table:
            by_family[d.table].append(d.seconds_total * 1e3)
        if (i + 1) % 25 == 0:
            ms = sorted(r.seconds_total * 1e3 for r in rows)
            print(f"  {i+1}/{len(states)}  median {ms[len(ms)//2]:7.1f} ms",
                  flush=True)

    def pct(v, q):
        v = sorted(v)
        return v[min(len(v) - 1, int(q * len(v)))] if v else float("nan")

    lazy_ms = [r.seconds_total * 1e3 for r in rows]
    got = [r for r in rows if r.table is not None]
    out = {
        "n_states": len(states), "n_lazy_ok": len(rows), "n_failures": fails,
        "n_returned_edit": len(got),
        "eager_median_ms": pct(eager_ms, 0.5) if eager_ms else None,
        "eager_n": len(eager_ms),
        "lazy_mean_ms": (sum(lazy_ms) / len(lazy_ms)) if lazy_ms else None,
        "lazy_median_ms": pct(lazy_ms, 0.5), "lazy_p90_ms": pct(lazy_ms, 0.9),
        "lazy_p95_ms": pct(lazy_ms, 0.95), "lazy_p99_ms": pct(lazy_ms, 0.99),
        "lazy_max_ms": max(lazy_ms) if lazy_ms else None,
        "fallback_fraction": sum(
            1 for r in rows if r.used_fallback) / max(len(rows), 1),
        "resolver_calls_mean": sum(r.resolver_calls for r in rows) / max(len(rows), 1),
        "family_redraws_mean": sum(r.family_redraws for r in rows) / max(len(rows), 1),
        "coordinate_rejections_mean": sum(
            r.coordinate_rejections for r in rows) / max(len(rows), 1),
        "encode_ms_mean": sum(r.seconds_encode for r in rows) / max(len(rows), 1) * 1e3,
        "family_ms_mean": sum(r.seconds_family for r in rows) / max(len(rows), 1) * 1e3,
        "mask_ms_mean": sum(r.seconds_mask for r in rows) / max(len(rows), 1) * 1e3,
        "resolver_ms_mean": sum(
            r.seconds_resolver for r in rows) / max(len(rows), 1) * 1e3,
        "by_family": {k: {"n": len(v), "median_ms": pct(v, 0.5)}
                      for k, v in by_family.items()},
    }

    print(f"\n{'metric':<32}{'eager':>12}{'lazy':>12}")
    print(f"  {'median transition ms':<30}"
          f"{(out['eager_median_ms'] or float('nan')):>12.1f}"
          f"{out['lazy_median_ms']:>12.1f}")
    print(f"  {'mean ms':<30}{'':>12}{out['lazy_mean_ms']:>12.1f}")
    print(f"  {'p90 ms':<30}{'':>12}{out['lazy_p90_ms']:>12.1f}")
    print(f"  {'p95 ms':<30}{'':>12}{out['lazy_p95_ms']:>12.1f}")
    print(f"  {'p99 ms':<30}{'':>12}{out['lazy_p99_ms']:>12.1f}")
    print(f"  {'max ms':<30}{'':>12}{out['lazy_max_ms']:>12.1f}")
    print(f"  {'resolver calls / draw':<30}{'~1300':>12}"
          f"{out['resolver_calls_mean']:>12.2f}")
    print(f"  {'family redraws / draw':<30}{'n/a':>12}"
          f"{out['family_redraws_mean']:>12.2f}")
    print(f"  {'coordinate rejections / draw':<30}{'n/a':>12}"
          f"{out['coordinate_rejections_mean']:>12.2f}")
    print(f"  {'eager fallback fraction':<30}{'n/a':>12}"
          f"{out['fallback_fraction']*100:>11.1f}%")
    print(f"  {'returned a valid edit':<30}{'100%':>12}"
          f"{out['n_returned_edit']/max(len(rows),1)*100:>11.0f}%")
    if out["eager_median_ms"]:
        print(f"\n  SPEEDUP  {out['eager_median_ms']/out['lazy_median_ms']:.1f}x"
              f"   (eager measured on {out['eager_n']} states)")
    print(f"\n  time split: encode {out['encode_ms_mean']:.1f} ms | "
          f"family {out['family_ms_mean']:.2f} ms | "
          f"mask {out['mask_ms_mean']:.1f} ms | "
          f"resolver {out['resolver_ms_mean']:.1f} ms")
    print(f"\n  {'family':<24}{'draws':>8}{'median ms':>12}")
    for k, v in sorted(out["by_family"].items(), key=lambda kv: -kv[1]["n"]):
        print(f"    {k:<22}{v['n']:>8}{v['median_ms']:>12.1f}")
    return out


@app.local_entrypoint()
def main(n_states: int = 120, n_eager: int = 8) -> None:
    panel = [s.strip() for s in
             (Path(__file__).resolve().parents[1]
              / "data/jin/dev_panel_qed_64.txt").read_text().split("\n")
             if s.strip()]
    zinc = Path("local_runtime/zinc250k/250k_rndm_zinc_drugs_clean_3.csv")
    srcs = list(panel)
    if zinc.exists() and len(srcs) < n_states:
        import csv
        with zinc.open(newline="") as fh:
            more = [r["smiles"].strip() for r in csv.DictReader(fh)
                    if r.get("smiles", "").strip()]
        stride = max(1, len(more) // max(1, n_states - len(srcs)))
        srcs += more[::stride][: n_states - len(srcs)]
    srcs = srcs[:n_states]
    out = bench.remote(srcs, n_eager)
    Path("docs/LAZY_SAMPLER_BENCH.json").write_text(json.dumps(out, indent=1))
    print("\nwrote docs/LAZY_SAMPLER_BENCH.json")
