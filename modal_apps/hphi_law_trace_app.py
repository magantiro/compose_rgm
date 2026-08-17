"""What does a law call actually READ, and where inside the builder does the time go?

The profiler established that a 7.74 s law call spends 7.60 s (98%) in
`prepare_factorized_mark_batch` and 0.042 s in the neural encode plus action
tables. That says the builder is the target, but NOT which part of it, and a
"build only what is needed" change is only worth making if the fields we can
skip are the expensive ones.

Three questions, one launch, then exit:

  1. READ TRACE -- every attribute read on the real batch during a real law
     call, ATTRIBUTED TO THE READING FRAME. A first pass enumerated `batch.*`
     accesses inside `_encode_batch` and `_action_tables` by AST and concluded
     19 of 43 fields are unread. That is unsound twice over: those are not the
     only consumers (`_family_base_logits`, the family/table loop and the mark
     construction also see the batch), and `dataclasses.replace` plus `.to()`
     inside `_one_state_batch` touch every field, so an unattributed trace
     reports all 43 as live and hides the answer.

  2. PRODUCER TIMING -- wrap each expensive producer the builder calls and
     measure it directly. This is the decisive number. If the cost sits in
     `_graph_application_masks`, whose outputs ARE consumed, then skipping
     unread fields buys nothing and the branch dies here rather than after a
     day of implementation.

  3. CONFIGURATION -- the real capability flags and ring catalog, because they
     decide which branches execute at all. `ring_delete_candidates` is `None`
     with no catalog and the expensive macro filtering never runs; with one it
     always does.

Read-only. Computes nothing that is persisted and changes no frozen artifact.
"""

from __future__ import annotations

import json
import time
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
app = modal.App("hphi-law-trace")
RUN_ROOT = "/artifacts/editing_v2/r_theta_run"
TIME_POINT, CANONICAL_SLOTS = 0.5, 48

#: Frames that touch the batch as plumbing rather than as consumers. A read
#: made here says nothing about what the law needs, and counting them would
#: mark every field live.
PLUMBING = {"replace", "to", "__init__", "__repr__", "__eq__", "_asdict",
            "<module>", "_trace_getattr", "law_trace", "_timed"}

#: Producers called by `prepare_factorized_mark_batch`, by module. Wrapped with
#: a timer so the builder's internal breakdown is measured rather than guessed.
PRODUCERS = [
    "compute_topology_features",
    "_graph_application_masks",
    "process_v2_atom_delete_mask",
    "_semantic_cycle_close_admission_mask",
    "_semantic_atom_restate_admission_mask",
    "_semantic_cycle_open_admission_mask",
    "enumerate_clean_ring_system_deletes",
    "enumerate_structured_ring_system_deletes",
    "_charge_preserving_macro_actions",
    "resonance_invariant_bond_classes",
    "enumerate_ring_restate_semantic_groups",
    "enumerate_ring_system_restate_actions",
    "ring_system_template_local_support_mask",
    "molecular_state_cache_key",
    "structured_ring_system_templates",
    "structured_ring_system_template_aliases",
    "de_novo_rewrite_system",
]


def _load(threads: int = 1):
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
    from compose_v4.experiments.production_successor_kernel import (
        _default_rewrite_system,
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
    model.eval(); torch.set_grad_enabled(False)
    torch.set_num_threads(threads)
    return model, _default_rewrite_system(model)


@app.function(image=image, cpu=(1.0, 1.0), memory=6144, timeout=45 * 60,
              volumes={str(ARTIFACT_ROOT): artifact_volume})
def law_trace(srcs: list[str]) -> dict[str, Any]:
    import sys
    from collections import defaultdict

    import torch

    sys.path.insert(0, str(REMOTE_ROOT / "src"))
    from rdkit import RDLogger

    RDLogger.DisableLog("rdApp.*")
    from compose_v4.chem.molecular_graph import smiles_to_molecular_graph
    from compose_v4.chem.state import pad_molecular_graph
    from compose_v4.experiments.production_successor_kernel import (
        enumerate_factorized_marked_law,
    )
    from compose_v4.model import factorized_tracelet_rate_model as F
    from compose_v4.model.factorized_tracelet_rate_model import FactorizedMarkBatch

    model, _system = _load(1)

    # ---- 3. CONFIGURATION -------------------------------------------------
    caps = model.operator_capabilities
    config = {
        "ring_catalog_is_none": model.ring_catalog is None,
        "capabilities": {k: bool(getattr(caps, k))
                         for k in dir(caps) if k.startswith("compute_")},
        "editing_process_semantics": str(model.editing_process_semantics),
        "atom_restate_action_semantics": str(model.atom_restate_action_semantics),
        "ring_restate_scorer_mode": str(model.ring_restate_scorer_mode),
        "enable_heteroatom_scan": bool(model.enable_heteroatom_scan),
        "enable_cycle_ops": bool(model.enable_cycle_ops),
    }
    print("\nCONFIGURATION")
    for k, v in config.items():
        print(f"  {k:<36} {v}")

    # ---- 2. PRODUCER TIMING ------------------------------------------------
    timings: dict[str, list[float]] = defaultdict(list)
    originals = {}
    for name in PRODUCERS:
        fn = getattr(F, name, None)
        if fn is None or not callable(fn):
            continue
        originals[name] = fn

        def _timed(*a, __n=name, __f=fn, **k):
            t0 = time.perf_counter()
            try:
                return __f(*a, **k)
            finally:
                timings[__n].append(time.perf_counter() - t0)
        setattr(F, name, _timed)

    # ---- 1. READ TRACE -----------------------------------------------------
    reads: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    armed = {"on": False}
    base = FactorizedMarkBatch.__getattribute__

    def _trace_getattr(self, name):
        if armed["on"] and not name.startswith("__"):
            f = sys._getframe(1)
            fname = f.f_code.co_name
            depth = 0
            while fname in PLUMBING and depth < 10:
                f = f.f_back
                if f is None:
                    break
                fname = f.f_code.co_name
                depth += 1
            if fname not in PLUMBING:
                reads[name][fname] += 1
        return base(self, name)

    FactorizedMarkBatch.__getattribute__ = _trace_getattr

    fields = sorted(FactorizedMarkBatch.__dataclass_fields__)
    armed["on"] = True
    totals = []
    for smi in srcs:
        st = pad_molecular_graph(smiles_to_molecular_graph(smi), CANONICAL_SLOTS)
        t0 = time.perf_counter()
        law = enumerate_factorized_marked_law(model, st, float(TIME_POINT))
        dt = time.perf_counter() - t0
        totals.append(dt)
        print(f"  {smi[:40]:<40} {len(law.marks):>4} marks  {dt*1e3:8.1f} ms",
              flush=True)
    armed["on"] = False
    FactorizedMarkBatch.__getattribute__ = base
    for name, fn in originals.items():
        setattr(F, name, fn)

    read = set(reads)
    unread = [f for f in fields if f not in read]

    mean_law = sum(totals) / len(totals)
    print(f"\n{'='*76}\nPRODUCER TIMING  (mean per law call, over {len(srcs)} states)"
          f"\n{'='*76}")
    rows = sorted(((sum(v) / len(srcs), len(v), k) for k, v in timings.items()),
                  reverse=True)
    acc = 0.0
    for ms, n, k in rows:
        acc += ms
        print(f"  {k:<46} {ms*1e3:9.1f} ms  {ms/mean_law*100:5.1f}%  x{n}")
    print(f"  {'-'*46} {'-'*9}")
    print(f"  {'SUM OF TIMED PRODUCERS':<46} {acc*1e3:9.1f} ms  "
          f"{acc/mean_law*100:5.1f}%")
    print(f"  {'FULL LAW CALL':<46} {mean_law*1e3:9.1f} ms  100.0%")
    print(f"  {'UNATTRIBUTED (builder glue + neural + marks)':<46} "
          f"{(mean_law-acc)*1e3:9.1f} ms  {(mean_law-acc)/mean_law*100:5.1f}%")

    print(f"\n{'='*76}\nREAD TRACE   fields {len(fields)}   read {len(read)}   "
          f"never read {len(unread)}\n{'='*76}")
    for f in sorted(read):
        who = ", ".join(f"{k}x{v}" for k, v in
                        sorted(reads[f].items(), key=lambda kv: -kv[1])[:4])
        print(f"  {f:<44} {who}")
    print(f"\nNEVER READ ({len(unread)}):")
    for f in unread:
        print(f"  {f}")
    print("\n'never read' is NECESSARY but NOT SUFFICIENT for skipping a "
          "producer:\na producer can be load-bearing through cache mutation or "
          "a configuration guard.")

    return {"config": config,
            "mean_law_seconds": mean_law,
            "producers": {k: sum(v) / len(srcs) for k, v in timings.items()},
            "read": {k: dict(v) for k, v in reads.items()},
            "unread": unread,
            "n_fields": len(fields)}


@app.local_entrypoint()
def main(n_states: int = 5) -> None:
    # The image mounts only `src` and `configs`, so `data/` does NOT exist in
    # the container. The panel is read here and the SMILES travel as arguments,
    # which is how the SMC app does it too.
    srcs = [s.strip() for s in
            (Path(__file__).resolve().parents[1]
             / "data/jin/dev_panel_qed_64.txt").read_text().split("\n")
            if s.strip()][:n_states]
    out = law_trace.remote(srcs)
    Path("docs/HPHI_LAW_TRACE.json").write_text(json.dumps(out, indent=2))
    print("\nwrote docs/HPHI_LAW_TRACE.json")
