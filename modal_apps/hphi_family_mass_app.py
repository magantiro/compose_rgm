"""Which families actually carry probability mass? That sets the build scope.

The lazy sampler draws the family FIRST and then constructs only that family's
coordinate machinery. But `_action_tables` computes every family at once, so the
lazy path cannot use it -- each family on the fast path needs its head call
replicated exactly, and any divergence silently changes the logits.

That makes scope a real question. A family not on the fast path must fall back
to the full enumerative law at ~7 s, so expected cost is

    sum over families of  P(f) * cost(f)

and two fast families are worth little if they hold a fifth of the mass. This
measures P(f) under the frozen law -- both the raw base distribution the sampler
draws from, and the realised family distribution of the enumerated law -- so the
fast path can be built for the families that dominate, in order, and the
expected cost of any remaining fallback can be priced rather than assumed.

Reports cumulative mass by family so the answer to "how many families must be
implemented" is read off directly.

Read-only.
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
    artifact_volume,
)
from modal_apps.run_process_v2_p50_app import image as _base_image

image = _base_image.env({"PYTHONPATH": f"{REMOTE_ROOT}/src:{REMOTE_ROOT}"})
app = modal.App("hphi-family-mass")
RUN_ROOT = "/artifacts/editing_v2/r_theta_run"
TIME_POINT, CANONICAL_SLOTS = 0.5, 48


@app.function(image=image, cpu=(1.0, 1.0), memory=6144, timeout=60 * 60,
              volumes={str(ARTIFACT_ROOT): artifact_volume})
def measure(srcs: list[str]) -> dict[str, Any]:
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
    from compose_v4.experiments.production_successor_kernel import (
        MARK_RULE_NAMES, _one_state_batch, enumerate_factorized_marked_law,
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

    raw = defaultdict(list)      # softmax(base): what the sampler draws from
    realised = defaultdict(list)  # the enumerated law's family distribution
    mark_share = defaultdict(float)

    for smi in srcs:
        st = pad_molecular_graph(smiles_to_molecular_graph(smi), CANONICAL_SLOTS)
        batch = _one_state_batch(model, st, float(TIME_POINT), prepared_batch=None)
        node, glob, pair = model._encode_batch(batch)
        base = model._family_base_logits(batch, glob)[0].double()
        p_raw = torch.softmax(base, dim=-1)
        for i, name in enumerate(MARK_RULE_NAMES):
            raw[name].append(float(p_raw[i]))

        law = enumerate_factorized_marked_law(model, st, float(TIME_POINT))
        if not law.marks:
            continue
        for i, name in enumerate(MARK_RULE_NAMES):
            realised[name].append(float(
                torch.exp(torch.tensor(law.family_log_probabilities[i]))))
        for m in law.marks:
            mark_share[m.family_name] += float(torch.exp(
                torch.tensor(m.log_probability)))
        print(f"  {smi[:36]:<36} {len(law.marks):>4} marks", flush=True)

    n = max(len(v) for v in realised.values()) if realised else 1
    rows = []
    for name in raw:
        r = sum(raw[name]) / len(raw[name])
        z = sum(realised.get(name, [0.0])) / max(len(realised.get(name, [0.0])), 1)
        rows.append((z, r, name))
    rows.sort(reverse=True)

    print(f"\n{'family':<24}{'realised P(f)':>15}{'raw softmax':>14}"
          f"{'cumulative':>12}")
    cum = 0.0
    out = {}
    for z, r, name in rows:
        cum += z
        out[name] = {"realised": z, "raw": r, "cumulative": cum}
        print(f"  {name:<22}{z:>15.4f}{r:>14.4f}{cum:>12.4f}")

    print("\nThe fast path must cover enough of REALISED P(f) that the fallback "
          "is rare:\nexpected cost = sum P(f) * cost(f), and a fallback family "
          "costs the full ~7 s law.")
    return {"families": out, "n_states": n}


@app.local_entrypoint()
def main(n_states: int = 12) -> None:
    srcs = [s.strip() for s in
            (Path(__file__).resolve().parents[1]
             / "data/jin/dev_panel_qed_64.txt").read_text().split("\n")
            if s.strip()][:n_states]
    out = measure.remote(srcs)
    Path("docs/FAMILY_MASS.json").write_text(json.dumps(out, indent=1))
    print("\nwrote docs/FAMILY_MASS.json")
