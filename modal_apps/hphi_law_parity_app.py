"""Law-level parity for the admission-mask optimisation. Bank, then check.

The mask-level test in `scripts/verification/admission_mask_parity.py` is the
tighter of the two -- it compares the admission masks and the ordered enumerator
output bit for bit -- but it is not the contract the controller actually depends
on. That contract is `enumerate_factorized_marked_law`: the ORDERED mark
identities and their normalized log probabilities. Masks feed `_action_tables`,
and other consumers reach the same resolvers by other routes, so law-level
equality is checked directly rather than inferred.

Run BEFORE the optimisation to bank, and AFTER to check:

    modal run modal_apps/hphi_law_parity_app.py --mode bank   --n-states 40
    modal run modal_apps/hphi_law_parity_app.py --mode check  --n-states 40

Probabilities are compared as exact float64 bit patterns, not with a tolerance.
The optimisation is memoization of pure functions, so it must reproduce the
arithmetic exactly; a tolerance would hide precisely the reordering bug worth
catching. Both runs execute the same graph on the same container class, so this
is not the cross-container float question that muddied the chemistry cache --
identical inputs must give identical bits here.
"""

from __future__ import annotations

import hashlib
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
app = modal.App("hphi-law-parity")
RUN_ROOT = "/artifacts/editing_v2/r_theta_run"
TIME_POINT, CANONICAL_SLOTS = 0.5, 48
BANK = Path("docs/LAW_PARITY_BASELINE.json")


def _load():
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
    return model


@app.function(image=image, cpu=(1.0, 1.0), memory=6144, timeout=60 * 60,
              volumes={str(ARTIFACT_ROOT): artifact_volume})
def law_fingerprints(srcs: list[str]) -> dict[str, Any]:
    import struct
    import sys

    sys.path.insert(0, str(REMOTE_ROOT / "src"))
    from rdkit import RDLogger

    RDLogger.DisableLog("rdApp.*")
    from compose_v4.chem.molecular_graph import smiles_to_molecular_graph
    from compose_v4.chem.state import pad_molecular_graph
    from compose_v4.experiments.production_successor_kernel import (
        enumerate_factorized_marked_law,
    )

    model = _load()
    out: dict[str, Any] = {}
    total = 0.0
    for i, smi in enumerate(srcs, 1):
        st = pad_molecular_graph(smiles_to_molecular_graph(smi), CANONICAL_SLOTS)
        t0 = time.perf_counter()
        law = enumerate_factorized_marked_law(model, st, float(TIME_POINT))
        dt = time.perf_counter() - t0
        total += dt

        # ORDERED. Position in the enumeration is part of the contract: the
        # sampler draws by index, so a permutation with identical content would
        # still change which molecule a given random draw returns.
        ident = hashlib.sha256()
        probs = hashlib.sha256()
        for mark in law.marks:
            ident.update(repr((mark.executor_rule_name, mark.action)).encode())
            # EXACT float64 bits, not a rounded decimal.
            probs.update(struct.pack("<d", float(mark.probability)))
        out[smi] = {"n_marks": len(law.marks),
                    "marks_sha256": ident.hexdigest(),
                    "probs_sha256": probs.hexdigest(),
                    "seconds": dt}
        print(f"  {i:>3}/{len(srcs)}  {len(law.marks):>4} marks  {dt*1e3:8.1f} ms  "
              f"{smi[:36]}", flush=True)
    return {"states": out, "mean_seconds": total / max(len(srcs), 1)}


@app.local_entrypoint()
def main(mode: str = "check", n_states: int = 40) -> None:
    srcs = [s.strip() for s in
            (Path(__file__).resolve().parents[1]
             / "data/jin/dev_panel_qed_64.txt").read_text().split("\n")
            if s.strip()][:n_states]
    got = law_fingerprints.remote(srcs)
    print(f"\nmean law call: {got['mean_seconds']*1e3:.1f} ms over {len(srcs)} states")

    if mode == "bank":
        BANK.write_text(json.dumps(got, indent=1))
        print(f"banked -> {BANK}")
        return

    base = json.loads(BANK.read_text())
    bad = [(s, k) for s in base["states"] if s in got["states"]
           for k in ("n_marks", "marks_sha256", "probs_sha256")
           if base["states"][s][k] != got["states"][s][k]]
    shared = set(base["states"]) & set(got["states"])
    print(f"compared {len(shared)} states x 3 exact digests")
    if bad:
        for s, k in bad[:10]:
            print(f"  MISMATCH {k:<14} {s[:56]}")
        print("\nLAW PARITY: FAILED")
        raise SystemExit(1)
    sp = base["mean_seconds"] / max(got["mean_seconds"], 1e-12)
    print(f"  law call {base['mean_seconds']*1e3:8.1f} -> "
          f"{got['mean_seconds']*1e3:8.1f} ms   {sp:5.2f}x")
    print(f"\nLAW PARITY: PASSED  ({len(shared)} states, ordered marks and "
          "float64 probability bits identical)")
