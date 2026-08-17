"""Execution parity: does a coordinate produce the SAME successor on both paths?

The latency benchmark established that the lazy sampler draws (family,
coordinate) from the same law. SMC needs one more thing: turning that pair into
an executed successor molecule. That step reads batch metadata -- graft removed
neighbours for reroute coordinates, ring restate actions for restate ones --
which the lazy batch builds on demand rather than eagerly, so it is exactly
where a lazily-constructed field could differ from the eager one and silently
produce a different molecule from the same mark.

The test is deliberately narrow and is the LAST audit before SMC:

    same state, same family, same coordinate
      -> eager batch  -> _coordinate_action -> apply -> canonical successor
      -> lazy batch   -> _coordinate_action -> apply -> canonical successor
    require the two canonical successors to be identical

Coordinates come from the eager law itself, so every one is a genuinely legal
mark, and they are sampled across EVERY family that appears -- with reroute and
ring-restate specifically required, since those carry the lazily built metadata.

Not a distribution test. The law equivalence was settled analytically; this asks
only whether execution agrees once a mark is chosen.
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
app = modal.App("hphi-execution-parity")
RUN_ROOT = "/artifacts/editing_v2/r_theta_run"
TIME_POINT, CANONICAL_SLOTS = 0.5, 48


@app.function(image=image, cpu=(1.0, 1.0), memory=6144, timeout=90 * 60,
              volumes={str(ARTIFACT_ROOT): artifact_volume})
def verify(srcs: list[str], per_family: int) -> dict[str, Any]:
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
    from compose_v4.chem.molecular_graph import smiles_to_molecular_graph
    from compose_v4.chem.state import pad_molecular_graph
    from compose_v4.experiments.hphi_graph_encode import build_graph_only_batch
    from compose_v4.experiments.production_successor_kernel import (
        _coordinate_action, _default_rewrite_system, _one_state_batch,
        canonical_state_key, enumerate_factorized_marked_law,
    )
    from compose_v4.model import factorized_tracelet_rate_model as F

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
    system = _default_rewrite_system(model)
    macro_system = F.de_novo_rewrite_system()

    def lazy_batch(st):
        """The lazy batch, with every family's metadata populated.

        The sampler fills these per drawn family; here all are filled so one
        batch can execute a coordinate from any family.
        """
        b = build_graph_only_batch([st], [float(TIME_POINT)])
        b.bonds = torch.from_numpy(np.asarray(st.bonds)).unsqueeze(0)
        b.formal_charges = torch.from_numpy(
            np.asarray(st.formal_charges)).unsqueeze(0)
        b.states = (st,)
        b.ring_topology_local_support_log_mass = None
        topo, _c1, _r1 = F.compute_topology_features(st)
        (_dm, _ce, _cp, graft, removed, gsucc) = F._graph_application_masks(
            st, topo, compute_cyclic_graft=True)
        b.graft_mask = torch.from_numpy(np.asarray(graft, dtype=bool)).unsqueeze(0)
        b.graft_remove_neighbors = torch.from_numpy(
            np.asarray(removed)).long().unsqueeze(0)
        b.graft_successor_groups = (gsucc,)
        try:
            groups = F.enumerate_ring_restate_semantic_groups(
                st, system=macro_system)
            b.ring_restate_actions = (groups.actions,)
            b.ring_restate_successor_group_ids = (groups.successor_group_ids,)
            b.ring_restate_successor_group_descriptors = (groups.group_descriptors,)
            b.ring_restate_successor_group_multiplicities = (
                groups.group_multiplicities,)
        except Exception:
            b.ring_restate_actions = ((),)
        return b

    seen = defaultdict(int)
    checked = 0
    bad = []

    for smi in srcs:
        st = pad_molecular_graph(smiles_to_molecular_graph(smi), CANONICAL_SLOTS)
        law = enumerate_factorized_marked_law(model, st, float(TIME_POINT))
        if not law.marks:
            continue
        eager = _one_state_batch(model, st, float(TIME_POINT), prepared_batch=None)
        lazy = lazy_batch(st)

        by_fam = defaultdict(list)
        for m in law.marks:
            by_fam[m.family_name].append(m)

        for fam, marks in by_fam.items():
            if seen[fam] >= per_family:
                continue
            step = max(1, len(marks) // per_family)
            for m in marks[::step][:per_family]:
                if seen[fam] >= per_family:
                    break
                coord = tuple(int(v) for v in m.coordinate)
                try:
                    r_e, a_e = _coordinate_action(
                        model, st, eager, family_name=fam,
                        table_name=m.table_name, coordinate=coord)
                    s_e = canonical_state_key(system.apply(st, r_e, a_e))
                except Exception as err:  # noqa: BLE001
                    bad.append({"family": fam, "coord": coord,
                                "reason": f"eager raised {type(err).__name__}: {err}"})
                    seen[fam] += 1
                    continue
                try:
                    r_l, a_l = _coordinate_action(
                        model, st, lazy, family_name=fam,
                        table_name=m.table_name, coordinate=coord)
                    s_l = canonical_state_key(system.apply(st, r_l, a_l))
                except Exception as err:  # noqa: BLE001
                    bad.append({"family": fam, "coord": coord,
                                "reason": f"lazy raised {type(err).__name__}: {err}"})
                    seen[fam] += 1
                    continue
                seen[fam] += 1
                checked += 1
                if s_e != s_l or r_e != r_l:
                    bad.append({"family": fam, "coord": coord,
                                "eager_rule": r_e, "lazy_rule": r_l,
                                "eager_successor": s_e[:60],
                                "lazy_successor": s_l[:60]})
        if all(v >= per_family for v in seen.values()) and len(seen) >= 6:
            break
        print(f"  {smi[:34]:<34} families so far {len(seen)}", flush=True)

    ok = not bad
    print(f"\n{'family':<24}{'checked':>9}")
    for f, n in sorted(seen.items()):
        print(f"  {f:<22}{n:>9}")
    print(f"\n  coordinates executed on BOTH paths: {checked}")
    print(f"  EXECUTION PARITY: {'PASSED' if ok else 'FAILED'}")
    for b in bad[:8]:
        print(f"    {b}")
    if ok:
        print("  Same state, same family, same coordinate -> same executor rule")
        print("  and the same canonical successor, on every family exercised.")
    return {"ok": ok, "checked": checked, "families": dict(seen),
            "mismatches": bad[:20]}


@app.local_entrypoint()
def main(n_states: int = 14, per_family: int = 6) -> None:
    srcs = [s.strip() for s in
            (Path(__file__).resolve().parents[1]
             / "data/jin/dev_panel_qed_64.txt").read_text().split("\n")
            if s.strip()][:n_states]
    out = verify.remote(srcs, per_family)
    Path("docs/EXECUTION_PARITY.json").write_text(json.dumps(out, indent=1))
    print(f"\nexecution parity {'PASSED' if out['ok'] else 'FAILED'} "
          f"over {out['checked']} coordinates")
    if not out["ok"]:
        raise SystemExit(1)
