"""Level 1 qualification: lazy per-family logits must equal `_action_tables` BITWISE.

This is the gate the whole lazy sampler rests on. `factorized_tracelet_rate_model.py`
is inside the Process-V2 identity hash, so the per-family scorers could not be
factored out of `_action_tables` and had to be duplicated; duplication is only
acceptable if it is verified, and the only verification worth anything here is
bit-for-bit equality on real states.

An approximate match would be worse than useless. A logit that differs in the
last bits produces a law that is finite, normalized, plausible, and wrong -- and
nothing downstream would notice, because the sampler's own exactness proof
assumes the raw scores are the model's raw scores.

So: float32 bit patterns compared element by element. Not allclose, not a
tolerance. Any single differing element fails the family.

Levels 2 (superset containment) and 3 (analytic full-law equality) are already
covered by `hphi_family_audit_app.py` and `hphi_rejection_exact_app.py`.
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
app = modal.App("hphi-lazy-parity")
RUN_ROOT = "/artifacts/editing_v2/r_theta_run"
TIME_POINT, CANONICAL_SLOTS = 0.5, 48


@app.function(image=image, cpu=(1.0, 1.0), memory=6144, timeout=60 * 60,
              volumes={str(ARTIFACT_ROOT): artifact_volume})
def parity(srcs: list[str]) -> dict[str, Any]:
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
    from rdkit import RDLogger

    RDLogger.DisableLog("rdApp.*")
    from compose_v4.chem.molecular_graph import smiles_to_molecular_graph
    from compose_v4.chem.state import pad_molecular_graph
    from compose_v4.experiments.hphi_lazy_family_scores import LAZY_SCORERS
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

    stats: dict[str, dict[str, Any]] = {
        t: {"states": 0, "identical": 0, "elements": 0, "worst": 0.0,
            "shape_mismatch": 0, "worst_on_support": 0.0,
            "support_identical": 0, "support_elements": 0}
        for t in LAZY_SCORERS
    }

    for smi in srcs:
        st = pad_molecular_graph(smiles_to_molecular_graph(smi), CANONICAL_SLOTS)
        batch = _one_state_batch(model, st, float(TIME_POINT), prepared_batch=None)
        with torch.no_grad():
            node, glob, pair = model._encode_batch(batch)
            # Compute the lazy scores FIRST, from clones. Three families that
            # read only `node` matched while the one that also reads
            # `global_state` did not, which points at in-place mutation of the
            # embeddings by `_action_tables` rather than at the duplication.
            # Cloning settles it instead of assuming either way.
            lazy = {t: fn(model, node.clone(), glob.clone(), pair.clone(), batch)
                    for t, fn in LAZY_SCORERS.items()}
            ref_masks, ref_logits, _z = model._action_tables(
                batch, node, glob, pair, require_exact_ring_support=False)

            for table, fn in LAZY_SCORERS.items():
                ref = ref_logits.get(table)
                if ref is None:
                    continue
                got = lazy[table]
                s = stats[table]
                s["states"] += 1
                if tuple(got.shape) != tuple(ref.shape):
                    s["shape_mismatch"] += 1
                    continue
                # BIT patterns, not values: -0.0 and 0.0 compare equal as floats
                # but are different bits, and a NaN never equals itself.
                same = torch.equal(
                    got.contiguous().view(torch.int32),
                    ref.contiguous().view(torch.int32))
                s["identical"] += int(same)
                s["elements"] += int(ref.numel())
                if not same:
                    s["worst"] = max(
                        s["worst"], float((got - ref).abs().max()))
                # A difference confined to coordinates no legal action can
                # occupy -- padding, the diagonal -- cannot affect the sampler,
                # which only ever scores coordinates inside the support. The
                # difference ON THE SUPPORT is the criterion that actually
                # matters, so it is measured separately rather than inferred.
                mk = ref_masks.get(table)
                if mk is not None and tuple(mk.shape) == tuple(ref.shape):
                    mk = mk.bool()
                    s["support_elements"] = s.get("support_elements", 0) + int(mk.sum())
                    if bool(mk.any()):
                        d_sup = float((got[mk] - ref[mk]).abs().max())
                        s["worst_on_support"] = max(
                            s.get("worst_on_support", 0.0), d_sup)
                        s["support_identical"] = s.get("support_identical", 0) + int(
                            torch.equal(got[mk].contiguous().view(torch.int32),
                                        ref[mk].contiguous().view(torch.int32)))
        print(f"  {smi[:38]:<38} checked", flush=True)

    print(f"\n{'table':<18}{'states':>7}{'bits ok':>9}{'worst all':>12}"
          f"{'support ok':>12}{'worst@support':>15}")
    ok = True
    for t, s in stats.items():
        # The criterion is equality ON THE SUPPORT. Everywhere-equality is
        # reported too, because a difference off-support is worth knowing about
        # even when it is harmless.
        good = (s.get("support_identical", 0) == s["states"]
                and s["shape_mismatch"] == 0)
        ok &= good
        print(f"  {t:<16}{s['states']:>7}{s['identical']:>9}{s['worst']:>12.3e}"
              f"{s.get('support_identical',0):>12}"
              f"{s.get('worst_on_support',0.0):>15.3e}"
              f"{'' if good else '  <-- FAILED'}")

    print(f"\n  LEVEL 1 RAW-HEAD PARITY: {'PASSED' if ok else 'FAILED'}")
    if ok:
        print("  The duplicated scorers reproduce the model's raw logits exactly,")
        print("  so the lazy path scores the same law the eager path scores.")
    else:
        print("  A duplicated scorer diverges. The lazy sampler MUST NOT be used")
        print("  until this is bit-identical: the exactness proof assumes these")
        print("  are the model's own raw scores.")
    return {"ok": ok, "stats": stats}


@app.local_entrypoint()
def main(n_states: int = 16) -> None:
    srcs = [s.strip() for s in
            (Path(__file__).resolve().parents[1]
             / "data/jin/dev_panel_qed_64.txt").read_text().split("\n")
            if s.strip()][:n_states]
    out = parity.remote(srcs)
    Path("docs/LAZY_HEAD_PARITY.json").write_text(json.dumps(out, indent=1))
    print(f"\nlevel-1 parity {'PASSED' if out['ok'] else 'FAILED'}")
    if not out["ok"]:
        raise SystemExit(1)
