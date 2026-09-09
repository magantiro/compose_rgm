"""GenMol Table 4, Level 1 optimizer: local canonical-fiber sweep with
continuation from realized states.

    enumerate S(x)  ->  R_theta-weighted sweep  ->  select for docking
      ->  dock  ->  continue from realized winners

WHAT MAKES THIS COMPOSE-NATIVE RATHER THAN A GENETIC ALGORITHM. Every successor
in the fiber yields R_theta(y|x), QED, SA and similarity-to-seed BEFORE a single
docking call is spent. The complete legal local menu is free; the expensive
evaluation is spent only where the process chooses to spend it. And a docked
winner is itself a molecular state, so the next round continues FROM it rather
than regenerating from the seed. Edits already made are not discarded.

THE THREE-WAY CONSTRAINT SEPARATION (docs/AMENDMENT_GENMOL_T4.md, decision 3).

    hard at EVERY step   executability only: valence, connectivity, declared
                         state space, rewrite support. Enforced by the executor.
    GUIDANCE             QED, SA, similarity, docking. May rise AND FALL.
    RETURNED only        QED>=0.6, SA<=4, sim>=delta; among those, best docking.

v(x) is a NAVIGATION SIGNAL, NOT A MASK. Nine of the fifteen benchmark seeds have
v > 0, so masking on v would forbid leaving the seed at all. Trajectories may dip
below the similarity floor and climb back. Only the returned molecule is judged.

SELECTION RULES are a small predeclared family. One is chosen ONCE on disjoint
development molecules and frozen; it is never re-tuned per target or per delta.

    rtheta   sample proportional to R_theta(y|x).  Ignores v. The control that
             asks whether the learned law alone is already a useful proposal.
    tilt     sample proportional to R_theta(y|x) * exp(-v(y)/TAU_V).
    rank     take the best by v, ties broken by R_theta.

BUDGET. ROUNDS * PER_ROUND docking calls per cell, exactly. The published
protocol is 10 iterations x 100 generations = 1,000; development runs a reduced
budget so the ceiling is preserved for the official cells. The seed's own docking
score is NOT charged: it is published in actives.csv and was reproduced in Gate 0.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import modal

from modal_apps.run_process_v2_p50_app import (
    ARTIFACT_ROOT, REMOTE_ROOT, ROOT, artifact_volume,
)
from modal_apps.run_process_v2_p50_app import image as _base_image

MOOD = "https://raw.githubusercontent.com/SeulLee05/MOOD/main/scorer"

image = (
    _base_image
    .apt_install("openbabel", "curl", "ca-certificates")
    .run_commands(
        "mkdir -p /opt/dock/receptors",
        f"curl -sSL -o /opt/dock/qvina02 {MOOD}/qvina02",
        "chmod +x /opt/dock/qvina02",
        *[f"curl -sSL -o /opt/dock/receptors/{t}.pdbqt {MOOD}/receptors/{t}.pdbqt"
          for t in ("parp1", "fa7", "5ht1b", "braf", "jak2")],
    )
    .add_local_file(ROOT / "docs/GENMOL_T4_SEEDS.json",
                    str(REMOTE_ROOT / "docs/GENMOL_T4_SEEDS.json"), copy=True)
    .add_local_file(ROOT / "docs/GENMOL_T4_DEV_SEEDS.json",
                    str(REMOTE_ROOT / "docs/GENMOL_T4_DEV_SEEDS.json"), copy=True)
    .add_local_file(ROOT / "diagnostics/t4_task_search/value_check.json",
                    str(REMOTE_ROOT / "diagnostics/t4_task_search/value_check.json"), copy=True)
    .add_local_file(ROOT / "modal_apps/genmol_t4_opt_app.py",
                    str(REMOTE_ROOT / "modal_apps/genmol_t4_opt_app.py"), copy=True)
    .env({"PYTHONPATH": f"{REMOTE_ROOT}/src:{REMOTE_ROOT}", "OMP_NUM_THREADS": "1"})
)

app = modal.App("genmol-t4-opt")
RUN_ROOT = "/artifacts/editing_v2/r_theta_run"
TIME_POINT, CANONICAL_SLOTS, MAX_ACTIVE_ATOMS = 0.5, 48, 40

QED_MIN, SA_MAX, TAU_V = 0.6, 4.0, 0.10
APPLY_CAP = 300          # marks executed per parent, top-R_theta first
BOXES = {
    "fa7":   ((10.131, 41.879, 32.097), (20.673, 20.198, 21.362)),
    "parp1": ((26.413, 11.282, 27.238), (18.521, 17.479, 19.995)),
    "5ht1b": ((-26.602, 5.277, 17.898), (22.5, 22.5, 22.5)),
    "jak2":  ((114.758, 65.496, 11.345), (19.033, 17.929, 20.283)),
    "braf":  ((84.194, 6.949, -7.081), (22.032, 19.211, 14.106)),
}

_RT: dict[str, Any] = {}


@app.function(
    image=image,
    cpu=(1.0, 1.0),
    memory=6144,
    timeout=900,
    max_containers=1,
    retries=0,
    volumes={str(ARTIFACT_ROOT): artifact_volume},
)
def continuation_profile(task: dict[str, Any]) -> dict[str, Any]:
    """One capped, zero-oracle profile. Never calls a population/docking driver."""
    import hashlib
    import platform
    import threading
    import traceback
    from datetime import datetime, timezone

    import numpy as np
    import torch
    from rdkit import rdBase

    from compose_v4.experiments.continuation_profile import (
        canonical_bytes,
        initial_state,
        publish_json,
        run_profile,
        verify_file,
    )
    from compose_v4.experiments.production_successor_kernel import (
        enumerate_factorized_marked_law,
    )
    from modal_apps.run_process_v2_p50_app import _validate_remote_revision

    _validate_remote_revision(task["image_revision"])
    verify_file(REMOTE_ROOT / "modal_apps/genmol_t4_opt_app.py", task["app_sha256"])
    contract_path = REMOTE_ROOT / "configs/continuation_profile_v1.json"
    verify_file(contract_path, task["contract_sha256"])
    contract = json.loads(contract_path.read_text())
    if (
        task["run_id"]
        != hashlib.sha256(
            canonical_bytes(
                {
                    "contract_sha256": task["contract_sha256"],
                    "image_revision_sha256": task["image_revision"]["image_revision_sha256"],
                    "app_sha256": task["app_sha256"],
                }
            )
        ).hexdigest()
    ):
        raise ValueError("profile output identity does not match the exact launch")
    # This is an operational run namespace, not a cross-revision scientific cache key.
    output = ARTIFACT_ROOT / "continuation_profile" / task["run_id"]
    if (output / "result.json").exists():
        return json.loads((output / "result.json").read_text())
    if (output / "progress.json").exists():
        raise RuntimeError(
            "partial profile exists; preserve its completed rows and audit before resuming"
        )
    started = time.perf_counter()
    progress = {
        "schema_version": "continuation_profile_progress_v1",
        "phase": "inputs",
        "complete": False,
        "oracle_calls": 0,
        "run_id": task["run_id"],
        "code_revision": task["image_revision"]["commit"],
    }
    stop = threading.Event()

    def publish_progress():
        progress["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
        progress["elapsed_seconds"] = time.perf_counter() - started
        publish_json(output / "progress.json", dict(progress))
        artifact_volume.commit()

    def heartbeat():
        while not stop.wait(contract["compute"]["heartbeat_seconds"]):
            publish_progress()

    publish_json(output / "launch.json", task)
    publish_progress()
    thread = threading.Thread(target=heartbeat, daemon=True)
    thread.start()
    try:
        inputs = {
            "run_paths": verify_file(Path(contract["run_paths"]), contract["run_paths_sha256"]),
            "checkpoint": verify_file(Path(contract["checkpoint"]), contract["checkpoint_sha256"]),
            "source_manifest": verify_file(
                REMOTE_ROOT / contract["source_manifest"], contract["source_manifest_sha256"]
            ),
        }
        progress["phase"] = "frozen_runtime_initialization"
        init_start = time.perf_counter()
        runtime = _runtime()
        runtime_seconds = time.perf_counter() - init_start
        node = initial_state(
            contract, json.loads((REMOTE_ROOT / contract["source_manifest"]).read_text())
        )

        def enumerate_law(graph):
            law = enumerate_factorized_marked_law(runtime["model"], graph, contract["time_point"])
            return (
                tuple(m.executor_rule_name for m in law.marks),
                tuple(m.action for m in law.marks),
                tuple(m.probability for m in law.marks),
            )

        result = run_profile(
            node,
            enumerate_law,
            runtime["system"],
            contract,
            output,
            snapshot_id=task["run_id"],
            commit_volume=artifact_volume.commit,
            progress=progress,
        )
        result.update(
            {
                "complete": True,
                "code_revision": task["image_revision"]["commit"],
                "contract_sha256": task["contract_sha256"],
                "input_sha256": inputs,
                "configuration": contract,
                "runtime_seconds": runtime_seconds,
                "software": {
                    "python": platform.python_version(),
                    "numpy": np.__version__,
                    "torch": torch.__version__,
                    "rdkit": rdBase.rdkitVersion,
                },
                "hardware": {
                    "machine": platform.machine(),
                    "cpu": platform.processor(),
                    "torch_threads": torch.get_num_threads(),
                },
                "run_id": task["run_id"],
                "elapsed_seconds": time.perf_counter() - started,
                "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            }
        )
        publish_json(output / "result.json", result)
        progress.update(phase="complete", complete=True, decision_status=result["status"])
        return result
    except Exception as error:
        # Boundary receipt, then fail loudly. No input substitution or gate relaxation.
        progress.update(phase="failed", error_type=type(error).__name__, error=str(error))
        publish_json(output / "failure.json", {**progress, "traceback": traceback.format_exc()})
        raise
    finally:
        stop.set()
        thread.join(timeout=30)
        publish_progress()


@app.function(
    image=image,
    cpu=(1.0, 1.0),
    memory=6144,
    timeout=900,
    max_containers=1,
    retries=0,
    volumes={str(ARTIFACT_ROOT): artifact_volume},
)
def fused_reference_profile(task: dict[str, Any]) -> dict[str, Any]:
    """One authenticated frozen-model reference path; no T4 driver or docking."""
    from compose_v4.experiments.fused_reference_profile import run_remote_task
    from modal_apps.run_process_v2_p50_app import _validate_remote_revision

    return run_remote_task(
        task,
        repo_root=REMOTE_ROOT,
        artifact_root=ARTIFACT_ROOT,
        volume=artifact_volume,
        runtime_factory=_runtime,
        validate_revision=_validate_remote_revision,
    )


@app.function(
    image=image, cpu=(1.0, 1.0), memory=8192, timeout=3600,
    max_containers=1, retries=0, volumes={str(ARTIFACT_ROOT): artifact_volume},
)
def t4_matched_pilot(task: dict[str, Any]) -> dict[str, Any]:
    """One paired cold-start round; both candidate locks precede docking."""
    from compose_v4.experiments.t4_matched_pilot import run_remote
    from modal_apps.run_process_v2_p50_app import _validate_remote_revision

    return run_remote(
        task, REMOTE_ROOT, ARTIFACT_ROOT, artifact_volume, _runtime,
        _validate_remote_revision,
        lambda t, progress, cached: t4_population_cell.local(t, progress, cached),
        lambda smiles, arm: _dock_many(smiles, "parp1", f"{task['run_id']}_{arm}", workers=1),
    )


@app.function(
    image=image, cpu=(1.0, 1.0), memory=8192, timeout=3600,
    max_containers=1, retries=0, volumes={str(ARTIFACT_ROOT): artifact_volume},
)
def t4_warm_continuation(task: dict[str, Any]) -> dict[str, Any]:
    """Two guided warm rounds, 40 new calls, with exact-state archive checkpoints."""
    from compose_v4.experiments.t4_warm_continuation import run_remote
    from modal_apps.run_process_v2_p50_app import _validate_remote_revision

    return run_remote(
        task, REMOTE_ROOT, ARTIFACT_ROOT, artifact_volume, _runtime,
        _validate_remote_revision,
        lambda t, progress, cached, warm: t4_population_cell.local(t, progress, cached, warm),
        lambda smiles, rd: _dock_many(smiles, "parp1", f"{task['run_id']}_{rd}", workers=1),
    )


@app.function(
    image=image, cpu=(1.0, 1.0), memory=8192, timeout=3600,
    max_containers=1, retries=0, volumes={str(ARTIFACT_ROOT): artifact_volume},
)
def t4_feedback_round(task: dict[str, Any]) -> dict[str, Any]:
    """One twenty-call round from all 33 saved evaluations; no automatic next round."""
    from compose_v4.experiments.t4_feedback_round import run_remote
    from modal_apps.run_process_v2_p50_app import _validate_remote_revision

    return run_remote(
        task, REMOTE_ROOT, ARTIFACT_ROOT, artifact_volume, _runtime,
        _validate_remote_revision,
        lambda t, progress, cached, warm: t4_population_cell.local(t, progress, cached, warm),
        lambda smiles, rd: _dock_many(smiles, "parp1", f"{task['run_id']}_{rd}", workers=1),
    )


@app.function(
    image=image, cpu=(1.0, 1.0), memory=8192, timeout=3600,
    max_containers=1, retries=0, volumes={str(ARTIFACT_ROOT): artifact_volume},
)
def t4_ring_program_round(task: dict[str, Any]) -> dict[str, Any]:
    """One parameterized-option round from all 46 saved calls; at most 20 new dockings."""
    from compose_v4.experiments.t4_ring_program_round import run_remote
    from modal_apps.run_process_v2_p50_app import _validate_remote_revision

    return run_remote(
        task, REMOTE_ROOT, ARTIFACT_ROOT, artifact_volume, _runtime,
        _validate_remote_revision,
        lambda t, progress, cached, warm: t4_population_cell.local(t, progress, cached, warm),
        lambda smiles, rd: _dock_many(smiles, "parp1", f"{task['run_id']}_{rd}", workers=1),
    )


@app.function(
    image=image, cpu=(1.0, 1.0), memory=2048, timeout=3600,
    max_containers=1, retries=0, volumes={str(ARTIFACT_ROOT): artifact_volume},
)
def t4_partial_docking(task: dict[str, Any]) -> dict[str, Any]:
    """Dock only the approved saved prefix; no generator or automatic next round."""
    from compose_v4.experiments.t4_partial_docking import run_remote
    from modal_apps.run_process_v2_p50_app import _validate_remote_revision

    return run_remote(
        task, REMOTE_ROOT, ARTIFACT_ROOT, artifact_volume, _validate_remote_revision,
        lambda jobs: t4_partial_dock_one.map(jobs, order_outputs=False, return_exceptions=True),
        qed_min=QED_MIN, sa_max=SA_MAX,
    )


@app.function(
    image=image, cpu=(1.0, 1.0), memory=2048, timeout=600,
    max_containers=4, retries=0, volumes={str(ARTIFACT_ROOT): artifact_volume},
)
def t4_partial_dock_one(task: dict[str, Any]) -> dict[str, Any]:
    """One locked molecule per container, one oracle attempt, no retries."""
    from compose_v4.experiments.t4_partial_docking import dock_saved_row
    from modal_apps.run_process_v2_p50_app import _validate_remote_revision

    return dock_saved_row(
        task, REMOTE_ROOT, ARTIFACT_ROOT, artifact_volume, _validate_remote_revision,
        lambda smiles, tag: _dock(smiles, "parp1", tag, cpu=1),
    )


@app.function(
    image=image, cpu=(1.0, 1.0), memory=8192, timeout=3600,
    max_containers=1, retries=0, volumes={str(ARTIFACT_ROOT): artifact_volume},
)
def t4_task_search_audit(task: dict[str, Any]) -> dict[str, Any]:
    """One resumable zero-oracle audit of cross-option task planning."""
    from compose_v4.experiments.t4_task_search_audit import run_remote
    from modal_apps.run_process_v2_p50_app import _validate_remote_revision

    return run_remote(
        task, REMOTE_ROOT, ARTIFACT_ROOT, artifact_volume, _runtime,
        _validate_remote_revision,
        lambda t, progress, cached, warm: t4_population_cell.local(t, progress, cached, warm),
    )


@app.function(
    image=image, cpu=(1.0, 1.0), memory=8192, timeout=900,
    max_containers=1, retries=0, volumes={str(ARTIFACT_ROOT): artifact_volume},
)
def t4_lazy_reference_probe(task: dict[str, Any]) -> dict[str, Any]:
    """One saved parent, the unchanged planning allowance, and zero docking."""
    from compose_v4.experiments.t4_task_search_audit import run_remote
    from modal_apps.run_process_v2_p50_app import _validate_remote_revision

    return run_remote(
        task, REMOTE_ROOT, ARTIFACT_ROOT, artifact_volume, _runtime,
        _validate_remote_revision,
        lambda t, progress, cached, warm: t4_population_cell.local(t, progress, cached, warm),
        lazy_probe=True,
    )


@app.function(
    image=image, cpu=(1.0, 1.0), memory=8192, timeout=7200,
    max_containers=1, retries=0, volumes={str(ARTIFACT_ROOT): artifact_volume},
)
def t4_uncapped_lookahead_probe(task: dict[str, Any]) -> dict[str, Any]:
    """Same parent and horizon, metered search without executor/state cutoffs."""
    from compose_v4.experiments.t4_task_search_audit import run_remote
    from modal_apps.run_process_v2_p50_app import _validate_remote_revision

    return run_remote(
        task, REMOTE_ROOT, ARTIFACT_ROOT, artifact_volume, _runtime,
        _validate_remote_revision,
        lambda t, progress, cached, warm: t4_population_cell.local(t, progress, cached, warm),
        uncapped_probe=True,
    )


@app.function(
    image=image, cpu=(1.0, 1.0), memory=8192, timeout=4200,
    max_containers=1, retries=0, volumes={str(ARTIFACT_ROOT): artifact_volume},
)
def t4_target_recovery(task: dict[str, Any]) -> dict[str, Any]:
    """One answer-known controller recovery attempt, never docking or fitting."""
    from compose_v4.experiments.t4_target_recovery import run_remote
    from modal_apps.run_process_v2_p50_app import _validate_remote_revision

    return run_remote(
        task, REMOTE_ROOT, ARTIFACT_ROOT, artifact_volume, _runtime,
        _validate_remote_revision,
    )


def _runtime():
    if "model" in _RT:
        return _RT
    import sys

    import torch
    sys.path.insert(0, str(REMOTE_ROOT / "src"))
    from compose_v4.experiments.editing_v2_process_v2_t1_panel import (
        open_process_v2_t1_source)
    from compose_v4.experiments.editing_v2_process_v2_t1_runtime import (
        build_process_v2_score_revised_scratch_runtime, load_materialized_scorer_state)
    from compose_v4.experiments.editing_v2_r_theta_corpus_training import (
        CHECKPOINT_FILENAME)
    from compose_v4.experiments.production_successor_kernel import (
        _default_rewrite_system)
    paths = json.loads((Path(RUN_ROOT) / "run_inputs" / "RUN_PATHS.json").read_text())
    src = open_process_v2_t1_source(
        Path(paths["active8_root"]),
        gate_zero_decision_path=Path(paths["gate_zero"]),
        artifact_root=Path(paths["artifact_root"]), repo_root=REMOTE_ROOT)
    bundle = load_materialized_scorer_state(Path(paths["materialized_scorer"]))
    runtime, _b, _c = build_process_v2_score_revised_scratch_runtime(
        src, materialized_state=bundle)
    model = runtime.model
    model_checkpoint = Path(RUN_ROOT) / "runs" / "run_v2_01" / CHECKPOINT_FILENAME
    ck = torch.load(model_checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(ck["selected_model_state"], strict=True)
    model.eval()
    torch.set_grad_enabled(False)
    torch.set_num_threads(1)
    _RT.update({
        "model": model,
        "system": _default_rewrite_system(model),
        "model_checkpoint": str(model_checkpoint),
        "run_paths": str(Path(RUN_ROOT) / "run_inputs" / "RUN_PATHS.json"),
    })
    return _RT


def _dock(smiles: str, target: str, tag: str, cpu: int = 4) -> float | None:
    """`cpu` is QuickVina's own thread count. It defaults to 4, which is what
    Gate 0 used to establish docking parity, so the default path is unchanged.
    Batch docking passes cpu=1 because each concurrent docking gets one core."""

    import os
    import subprocess
    d = f"/tmp/{tag}"
    os.makedirs(d, exist_ok=True)
    mol, lig, out = f"{d}/l.mol", f"{d}/l.pdbqt", f"{d}/o.pdbqt"
    for p in (mol, lig, out):
        if os.path.exists(p):
            os.remove(p)
    try:
        subprocess.run(["obabel", f"-:{smiles}", "--gen3D", "-O", mol],
                       capture_output=True, timeout=120, check=True)
        subprocess.run(["obabel", mol, "-O", lig],
                       capture_output=True, timeout=60, check=True)
    except Exception:
        return None
    (cx, cy, cz), (sx, sy, sz) = BOXES[target]
    try:
        subprocess.run(
            ["/opt/dock/qvina02", "--receptor", f"/opt/dock/receptors/{target}.pdbqt",
             "--ligand", lig, "--out", out,
             "--center_x", str(cx), "--center_y", str(cy), "--center_z", str(cz),
             "--size_x", str(sx), "--size_y", str(sy), "--size_z", str(sz),
             "--cpu", str(int(cpu)), "--num_modes", "10", "--exhaustiveness", "1"],
            capture_output=True, timeout=300, check=True)
        for line in open(out):
            if line.startswith("REMARK VINA RESULT"):
                return float(line.split()[3])
    except Exception:
        return None
    return None


def _dock_one(args) -> tuple[int, float | None]:
    """(index, smiles, target, tag, cpu) -> (index, score). Index preserves order."""
    i, smiles, target, tag, cpu = args
    return i, _dock(smiles, target, tag, cpu=cpu)


def _dock_many(smiles_list, target: str, tag_prefix: str, workers: int = 8,
               cpu_per_dock: int = 1) -> list:
    """Dock a round's molecules concurrently. Same molecules, same scores, same count.

    The 20 dockings inside a round are conditionally independent -- the archive is
    only updated after all of them -- so running them serially wastes most of the
    round. QuickVina is an external process, so threads are the right tool: the
    GIL is released across subprocess.run.

    ORDER AND ACCOUNTING ARE PRESERVED EXACTLY. Results come back indexed and are
    reassembled in the caller's order, every input yields exactly one output
    (None included), and the caller still charges one budget unit per molecule.
    Each docking gets its OWN scratch directory; the serial code reused
    /tmp/{tag}, which concurrent dockings would clobber.

    This is execution parallelism only. No candidate selection, archive rule, or
    budget semantics changes.
    """

    from concurrent.futures import ThreadPoolExecutor
    jobs = [(i, s, target, f"{tag_prefix}_w{i}", cpu_per_dock)
            for i, s in enumerate(smiles_list)]
    out: list = [None] * len(jobs)
    if not jobs:
        return out
    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        for i, r in ex.map(_dock_one, jobs):
            out[i] = r
    return out


@app.function(image=image, cpu=(1.0, 1.0), memory=int(6 * 1024),
              timeout=10 * 60 * 60, max_containers=40,
              volumes={str(ARTIFACT_ROOT): artifact_volume})
def optimize(task: dict[str, Any]) -> dict[str, Any]:
    import os
    import sys

    import numpy as np
    sys.path.insert(0, str(REMOTE_ROOT / "src"))
    from rdkit import Chem, DataStructs, RDConfig, RDLogger
    from rdkit.Chem import QED, rdFingerprintGenerator
    RDLogger.DisableLog("rdApp.*")
    sys.path.append(os.path.join(RDConfig.RDContribDir, "SA_Score"))
    import sascorer
    from compose_v4.chem.molecular_graph import smiles_to_molecular_graph
    from compose_v4.chem.state import pad_molecular_graph
    from compose_v4.experiments.production_successor_kernel import (
        canonical_state_key, enumerate_factorized_marked_law)

    rt = _runtime()
    model, system = rt["model"], rt["system"]
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    seed, delta, target = task["smiles"], task["delta"], task["target"]
    rounds, per_round = task["rounds"], task["per_round"]
    parents_n, rule = task["parents"], task["rule"]
    seed_fp = gen.GetFingerprint(Chem.MolFromSmiles(seed))
    rng = np.random.default_rng(task["seed_rng"])
    T = {"fiber": 0.0, "props": 0.0, "dock": 0.0}

    def props(smi):
        m = Chem.MolFromSmiles(smi)
        if m is None:
            return None
        q = float(QED.qed(m)); s = float(sascorer.calculateScore(m))
        sim = float(DataStructs.TanimotoSimilarity(seed_fp, gen.GetFingerprint(m)))
        v = max(max(0.0, QED_MIN - q) / QED_MIN, max(0.0, s - SA_MAX) / SA_MAX,
                max(0.0, delta - sim) / delta)
        return {"qed": q, "sa": s, "sim": sim, "v": v}

    t0 = time.perf_counter()
    archive = [{"smiles": seed, **props(seed), "ds": None, "round": 0}]
    docked: dict[str, float] = {}
    n_dock = 0

    for rd in range(1, rounds + 1):
        # parents: prefer feasible-with-good-docking, else low v.  Realized
        # states only -- this is where continuation from winners happens.
        scored = sorted(
            archive,
            key=lambda a: (a["v"] > 0,
                           a["ds"] if (a["v"] == 0 and a["ds"] is not None) else 0.0,
                           a["v"]))
        parents = scored[:parents_n]

        tf = time.perf_counter()
        cand: dict[str, dict] = {}
        for p in parents:
            try:
                st = pad_molecular_graph(smiles_to_molecular_graph(p["smiles"]),
                                         CANONICAL_SLOTS)
                law = enumerate_factorized_marked_law(model, st, float(TIME_POINT))
            except Exception:
                continue
            if not law.marks:
                continue
            pr = np.array([m.probability for m in law.marks], float)
            for i in np.argsort(-pr)[:APPLY_CAP]:
                mk = law.marks[int(i)]
                try:
                    y = canonical_state_key(system.apply(
                        st, mk.executor_rule_name, mk.action))
                except Exception:
                    continue
                if not y or y == p["smiles"] or y in docked or y in cand:
                    continue
                cand[y] = {"rtheta": float(pr[int(i)])}
        T["fiber"] += time.perf_counter() - tf

        if not cand:
            break
        tp = time.perf_counter()
        for y in list(cand):
            pv = props(y)
            if pv is None:
                del cand[y]
            else:
                cand[y].update(pv)
        T["props"] += time.perf_counter() - tp
        if not cand:
            break

        keys = list(cand)
        k = min(per_round, len(keys))
        if rule == "rank":
            pick = sorted(keys, key=lambda y: (cand[y]["v"], -cand[y]["rtheta"]))[:k]
        else:
            w = np.array([cand[y]["rtheta"] for y in keys], float)
            if rule == "tilt":
                w = w * np.exp(-np.array([cand[y]["v"] for y in keys]) / TAU_V)
            w = np.clip(w, 1e-30, None); w /= w.sum()
            pick = [keys[i] for i in rng.choice(len(keys), size=k, replace=False, p=w)]

        td = time.perf_counter()
        for j, y in enumerate(pick):
            ds = _dock(y, target, f"c{task['idx']}_{rd}_{j}")
            n_dock += 1
            docked[y] = ds if ds is not None else 0.0
            if ds is not None:
                archive.append({"smiles": y, **cand[y], "ds": ds, "round": rd})
        T["dock"] += time.perf_counter() - td

    feas = [a for a in archive if a["v"] == 0.0 and a["ds"] is not None]
    best = min(feas, key=lambda a: a["ds"]) if feas else None
    return {**task, "n_docked": n_dock, "n_archive": len(archive),
            "n_feasible": len(feas),
            "best_ds": (best["ds"] if best else None),
            "best_smiles": (best["smiles"] if best else None),
            "best_round": (best["round"] if best else None),
            "timing": {k: round(v, 1) for k, v in T.items()},
            "seconds": round(time.perf_counter() - t0, 1)}


@app.function(image=image, cpu=(1.0, 1.0), memory=2048, timeout=12 * 60 * 60,
              volumes={str(ARTIFACT_ROOT): artifact_volume})
def drive(which: str, rounds: int, per_round: int, parents: int,
          rule: str, limit: int, deltas: str) -> dict[str, Any]:
    f = "GENMOL_T4_SEEDS.json" if which == "bench" else "GENMOL_T4_DEV_SEEDS.json"
    raw = json.loads((REMOTE_ROOT / f"docs/{f}").read_text())
    seeds = raw if isinstance(raw, list) else raw["seeds"]
    dl = [float(x) for x in deltas.split(",")]
    tasks = [{**s, "idx": i, "delta": d, "rounds": rounds, "per_round": per_round,
              "parents": parents, "rule": rule, "seed_rng": 20260820 + i}
             for i, (s, d) in enumerate((s, d) for s in seeds for d in dl)]
    if limit:
        tasks = tasks[:limit]
    print(f"{len(tasks)} cells   rule={rule}  rounds={rounds} x per_round="
          f"{per_round} = {rounds*per_round} dockings/cell   parents={parents}\n",
          flush=True)
    raw_out = list(optimize.map(tasks, order_outputs=True, return_exceptions=True,
                                wrap_returned_exceptions=False))
    out = [r for r in raw_out if isinstance(r, dict)]
    # NEVER silently drop failures. A transient Modal control-plane error once
    # produced a clean-looking "0 cells" result because the exceptions were
    # filtered out here without being counted.
    errs = [r for r in raw_out if not isinstance(r, dict)]
    if errs:
        from collections import Counter
        print(f"  !! {len(errs)}/{len(tasks)} cells FAILED", flush=True)
        for msg, k in Counter(f"{type(e).__name__}: {e}"[:180] for e in errs).most_common(5):
            print(f"     x{k}  {msg}", flush=True)
    if not out:
        raise RuntimeError(
            f"all {len(tasks)} cells failed; refusing to write an empty sweep")
    print(f"{'target':8s}{'delta':>6}{'seedDS':>8}{'bestDS':>8}{'gain':>7}"
          f"{'#feas':>7}{'#dock':>7}{'fiber':>8}{'props':>8}{'dock':>8}{'total':>8}",
          flush=True)
    for r in out:
        sd = -r.get("published_ds", 0.0) if "published_ds" in r else None
        bd = r["best_ds"]
        g = (sd - bd) if (sd is not None and bd is not None) else None
        print(f"  {r['target']:8s}{r['delta']:>6.1f}"
              f"{(f'{sd:.1f}' if sd else '-'):>8}{(f'{bd:.2f}' if bd else 'NONE'):>8}"
              f"{(f'{g:+.2f}' if g else '-'):>7}{r['n_feasible']:>7}{r['n_docked']:>7}"
              f"{r['timing']['fiber']:>8.0f}{r['timing']['props']:>8.0f}"
              f"{r['timing']['dock']:>8.0f}{r['seconds']:>8.0f}", flush=True)
    tot = sum(r["seconds"] for r in out)
    nd = sum(r["n_docked"] for r in out)
    print(f"\n  wall {tot:.0f} core-s over {len(out)} cells, {nd} dockings")
    if nd:
        print(f"  per docking: {sum(r['timing']['dock'] for r in out)/nd:.1f}s"
              f"   fiber+props share: "
              f"{sum(r['timing']['fiber']+r['timing']['props'] for r in out)/tot*100:.0f}%")
        full = tot / len(out) / (rounds * per_round) * 1000
        print(f"  EXTRAPOLATION: {full:.0f} core-s per cell at the official "
              f"1000-docking budget -> {full*30/3600:.1f} core-h for all 30 cells")
    return {"cells": out, "rule": rule, "rounds": rounds, "per_round": per_round,
            "parents": parents}


@app.local_entrypoint()
def main(which: str = "dev", rounds: int = 3, per_round: int = 8,
         parents: int = 3, rule: str = "tilt", limit: int = 2,
         deltas: str = "0.6", out: str = "diagnostics/genmol_t4_smoke.json") -> None:
    o = drive.remote(which, rounds, per_round, parents, rule, limit, deltas)
    p = Path(__file__).resolve().parents[1] / out
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(o, indent=1))
    print(f"\nwrote {p}")


# ---------------------------------------------------------------------------
# T4 with the local-to-global population search.
#
# Same task, same budget accounting, same feasibility contract as `optimize`
# above: QED >= 0.6, SA <= 4, sim >= delta on the RETURNED molecule, scored by
# the docking of the best feasible lead, with the seed's own score not charged.
#
# What changes is only the proposal. `optimize` samples the marked law and
# selects among realizations; this runs the qualified stack --
#
#     Q(M|x,z)              WHERE to rewrite, feasibility/cost-aware
#     Q(o|x,M,z)            WHAT to do, balanced applicability-aware prior
#     q_rewrite(w|x,M,o)    HOW, frozen committor + KL at kappa=1
#
# -- so it can choose the SCALE of each edit rather than only its content. The
# standing T4 conclusion located the failure exactly there: broad-C sampled the
# useful semantic 1.0% of the time against narrow-B's 33.3%, a proposal-
# probability bottleneck rather than a realization or ranking defect.
#
# Nothing below the region selector is retuned for T4.
# ---------------------------------------------------------------------------

POP_OUT = "/artifacts/t4_population"


@app.function(image=image, cpu=(8.0, 8.0), memory=int(24 * 1024),
              timeout=2 * 60 * 60, retries=0, max_containers=80,
              volumes={ARTIFACT_ROOT: artifact_volume})
def t4_population_cell(task: dict[str, Any], progress=None, parent_cache=None,
                       warm_start=None) -> dict[str, Any]:
    """Round-based population search, matching the prior T4 round contract.

    The prior official run was already a population method: 8 lineages, a large
    candidate pool per round, 20 dockings run CONCURRENTLY, and the archive
    updated only after the whole batch. An earlier version of this function
    replaced that with one parent -> one rewrite -> one docking -> wait, which
    destroyed both proposal and oracle parallelism and cost ~75s per docking
    call against the prior 1.3s. That was an implementation error, not a
    property of the controller.

    Only the OFFSPRING PROPOSAL changes here:

        old   sample the marked law, select among realizations
        new   M ~ Q(M|x,z), o ~ Q(o|x,M,z), then the frozen region rewrite
              q ∝ R_M,o * h_phi^eta with KL(q || R_M,o) <= kappa

    Everything else -- lineage count, parent selection, feasibility, batch size,
    budget accounting, archive update after the batch -- is kept, so a
    difference is attributable to the proposal.

    Within a round, candidate j+1 must not depend on the docking of candidate j;
    that is what re-serializes a population algorithm.
    """
    import hashlib
    import os
    import platform
    import sys
    from collections import Counter
    from concurrent.futures import ThreadPoolExecutor
    from datetime import datetime, timezone

    import numpy as np
    import torch
    sys.path.insert(0, str(REMOTE_ROOT / "src"))
    from rdkit import Chem, DataStructs, RDConfig, RDLogger
    from rdkit.Chem import rdFingerprintGenerator
    RDLogger.DisableLog("rdApp.*")
    sys.path.append(os.path.join(RDConfig.RDContribDir, "SA_Score"))
    import sascorer
    from compose_v4.chem.molecular_graph import (
        NULL_IDX,
        is_element,
        smiles_to_molecular_graph,
    )
    from compose_v4.chem.state import pad_molecular_graph
    from compose_v4.experiments.production_successor_kernel import (
        canonical_state_key, enumerate_factorized_marked_law)
    from compose_v4.control.region import enumerate_regions
    from compose_v4.control.fused_option import BUILD_FUSED_RING_OPTION, FusedProgress
    from compose_v4.control.ring_program import RingProgress, default_ring_options, ring_spec
    from compose_v4.control.option_continuation import (
        OptionContinuationKernel, OptionState, exact_graph_key,
    )
    from compose_v4.experiments.t4_matched_pilot import primitive_distribution
    from compose_v4.rewrite.trace_shard import decode_state, encode_state
    from compose_v4.experiments.t4_warm_continuation import exact_context, payload_hash
    from compose_v4.experiments.t4_endpoint_selection import (
        LEGACY_RANK_ALL, annotate_endpoints, calculate_properties, validate_policy,
    )
    from compose_v4.control import region_rewrite as RR
    from compose_v4.control import graph_geometry as GG
    from compose_v4.control.macro_engine import (
        BACKBONE_ELEMENTS,
        TERMINAL_ELEMENTS,
        state_contract_for,
    )
    from compose_v4.control.option_selector import (
        BUILD_RING_SYSTEM_OPTION,
        GENERIC_OPTION,
        OPTION_GROUPS,
        OPTIONS,
        applicable_options,
        bundle_identity,
        conditioned_action_distribution,
        option_horizon,
        primitive_option_at_step,
        retain_product_applicable_options,
        sample_option,
    )
    from compose_v4.control.region_selector import sample_region
    from compose_v4.control.task_value import TaskValue
    from compose_v4.gates.med_chem_gate import is_valid
    from compose_v4.rewrite.kernel import InvalidRewrite

    rt = _runtime(); model, system = rt["model"], rt["system"]
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    seed, delta, target = task["smiles"], task["delta"], task["target"]
    budget = int(task["budget"])
    per_round = int(task.get("per_round", 20))       # dockings per round
    n_lineages = int(task.get("lineages", 8))
    workers = int(task.get("workers", 8))
    parent_share = task.get("executor_calls_per_parent")
    if parent_share is not None and (
        workers != 1 or n_lineages != 8 or parent_share != 2500
        or task.get("max_executor_applications") != 20000
        or not task.get("prepare_only") or warm_start is None
    ):
        raise ValueError("parent shares require the approved single-worker 8 x 2500 feedback lane")
    arm = task.get("arm", "mu_exec")
    guidance = task.get("primitive_guidance", "committor")
    endpoint_policy = task.get("endpoint_selection_policy", LEGACY_RANK_ALL)
    validate_policy(endpoint_policy)
    if guidance not in ("committor", "reference"):
        raise ValueError(f"unknown primitive_guidance: {guidance!r}")
    include_fused = bool(task.get("include_fused", False))
    ring_options = tuple(task.get("ring_program_options", ()))
    if task.get("include_ring_programs", False):
        ring_options = ring_options or default_ring_options()
    if len(set(ring_options)) != len(ring_options) or any(ring_spec(o) is None for o in ring_options):
        raise ValueError("ring_program_options must be distinct parameterized construction options")
    tau = float(task.get("tau", 0.05))
    kappa = float(task.get("kappa", 1.0))
    if kappa != 1.0:
        raise ValueError(f"the qualified T4 controller freezes kappa=1.0, got {kappa}")
    eps_region = float(task.get("epsilon_region", 0.2))
    eps_option = float(task.get("epsilon_option", 0.1))
    macro_temperature = float(task.get("macro_temperature", 2.0))
    eps_macro = float(task.get("epsilon_macro", 0.15))
    rng = np.random.default_rng(int(task["seed_rng"]))
    if warm_start is not None:
        if payload_hash(warm_start) != task.get("warm_start_sha256"):
            raise ValueError("warm archive identity mismatch")
        if not task.get("prepare_only"):
            raise ValueError("exact-state warm continuation requires locked one-round preparation")
        rng.bit_generator.state = warm_start["rng_state"]
    exact_parents = {
        row["smiles"]: decode_state(row["state"])
        for row in warm_start["archive"]
    } if warm_start is not None else {}
    seed_fp = gen.GetFingerprint(Chem.MolFromSmiles(seed))
    tv = TaskValue.from_dict(task["value_table"]) if task.get("value_table") else None
    value_fn = tv.value_fn() if tv is not None else None

    def sha256_file(path):
        digest = hashlib.sha256()
        with open(path, "rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    artifact_volume.reload()
    committor_path = "/artifacts/region_committor/committor_bellman_v1.pt"
    seed_manifest_path = REMOTE_ROOT / "docs" / "GENMOL_T4_SEEDS.json"
    qvina_path = "/opt/dock/qvina02"
    receptor_path = f"/opt/dock/receptors/{target}.pdbqt"
    ck = torch.load(committor_path, map_location="cpu")
    input_hashes = {
        "committor": sha256_file(committor_path),
        "r_theta_checkpoint": sha256_file(rt["model_checkpoint"]),
        "r_theta_run_paths": sha256_file(rt["run_paths"]),
        "seed_manifest": sha256_file(seed_manifest_path),
        "qvina02": sha256_file(qvina_path),
        "receptor": sha256_file(receptor_path),
    }
    for name, expected in task.get("expected_input_sha256", {}).items():
        if input_hashes.get(name) != expected:
            raise ValueError(f"frozen input hash mismatch: {name}")
    expected_seed_hash = task.get("seed_manifest_sha256")
    if expected_seed_hash and expected_seed_hash != input_hashes["seed_manifest"]:
        raise ValueError(
            "seed manifest hash differs between the clean launch tree and deployed app: "
            f"{expected_seed_hash} != {input_hashes['seed_manifest']}"
        )
    started_at = datetime.now(timezone.utc).isoformat()
    if task.get("controller") == "hierarchical_task_v1":
        # Explicit prepare-only schema boundary. The legacy single-bundle
        # verifier/launcher must not silently consume cross-option paths.
        if (not task.get("prepare_only") or warm_start is None
                or target != "parp1" or delta != 0.4 or workers != 1):
            raise ValueError("hierarchical task search requires PARP1 d=0.4 warm preparation")
        from compose_v4.experiments.t4_matched_pilot import unseal
        from compose_v4.experiments.t4_task_search import PreparationConfig
        from compose_v4.experiments.t4_task_search import prepare as prepare_hierarchy
        from modal_apps.run_process_v2_p50_app import _validate_remote_revision

        _validate_remote_revision(task["image_revision"])
        archive_path = Path(task["warm_archive_path"])
        if sha256_file(archive_path) != task["warm_archive_file_sha256"]:
            raise ValueError("hierarchical warm archive physical identity mismatch")
        if payload_hash(unseal(archive_path)) != payload_hash(warm_start):
            raise ValueError("hierarchical warm archive payload mismatch")
        if task.get("expected_input_sha256") != input_hashes:
            raise ValueError("hierarchical preparation requires every frozen input identity")

        def hierarchy_law(graph):
            law = enumerate_factorized_marked_law(model, graph, float(TIME_POINT))
            return (
                tuple(mark.executor_rule_name for mark in law.marks),
                tuple(mark.action for mark in law.marks),
                tuple(mark.probability for mark in law.marks),
            )

        result = prepare_hierarchy(
            warm_start, source_sha256=task["warm_archive_file_sha256"],
            enumerate_law=hierarchy_law, system=system, input_sha256=input_hashes,
            progress=progress, parent_cache=parent_cache,
            config=PreparationConfig(**task["preparation"]),
        )
        result.update(started_at_utc=started_at, image_revision=task["image_revision"])
        return result
    net = torch.nn.Sequential(torch.nn.Linear(int(ck["n_features"]), 48),
                              torch.nn.ReLU(), torch.nn.Linear(48, 1))
    net.load_state_dict(ck["state_dict"]); net.eval()
    torch.set_grad_enabled(False)

    def props(smi):
        return calculate_properties(
            Chem.MolFromSmiles(smi), seed_fp=seed_fp, generator=gen,
            sa_scorer=sascorer.calculateScore, delta=delta, qed_min=QED_MIN, sa_max=SA_MAX,
        )

    def composition_delta(before_smi, after_smi):
        before_mol = Chem.MolFromSmiles(before_smi)
        after_mol = Chem.MolFromSmiles(after_smi)
        if before_mol is None or after_mol is None:
            return {}, 0, 0, 0
        before = Counter(atom.GetSymbol() for atom in before_mol.GetAtoms())
        after = Counter(atom.GetSymbol() for atom in after_mol.GetAtoms())
        delta_by_element = {
            element: int(after[element] - before[element])
            for element in sorted(set(before) | set(after))
            if after[element] != before[element]
        }
        added_terminal = sum(
            max(0, delta_by_element.get(element, 0))
            for element in TERMINAL_ELEMENTS
        )
        added_backbone = sum(
            max(0, delta_by_element.get(element, 0))
            for element in BACKBONE_ELEMENTS
        )
        added_sulfur = max(0, delta_by_element.get("S", 0))
        return delta_by_element, added_terminal, added_backbone, added_sulfur

    def batched_rewrite(parent, parent_lineage_id, regions_k, max_steps, rng_):
        """Search selected ``(parent, region, option)`` bundles concurrently.

        ``Q(M|x,z)`` allocates one region draw and ``Q(o|x,M,z)`` samples one
        applicable option for that draw.  Options never clone a region draw.
        Within a bundle, the existing macro machinery restricts primitive
        support before the frozen committor and KL tilt are applied.  Generic
        support is byte-for-byte the qualified raw region law.

        Every transition is still executed by the production executor and is
        checked for validity, connectivity, and frozen-context preservation.
        BUILD_RING_SYSTEM is an indivisible eleven-step proposal; its
        intermediates remain search states and only completed endpoints enter
        the candidate pool.
        """
        import hashlib
        from collections import Counter, defaultdict

        cands: dict = {}
        cache: dict = {}
        memo: dict = {}
        trace = []

        def slot_key(st):
            # Actions carry persistent-slot coordinates, so a canonical SMILES
            # key is scientifically invalid for marked-law reuse.
            return exact_graph_key(st)

        def enum_fn(st):
            k_ = slot_key(st)
            if k_ not in cache:
                law = enumerate_factorized_marked_law(model, st, float(TIME_POINT))
                cache[k_] = (
                    [m.executor_rule_name for m in law.marks],
                    [m.action for m in law.marks],
                    np.array([m.probability for m in law.marks], float),
                )
            return cache[k_]

        def apply_fn(st, jj):
            k_ = (slot_key(st), int(jj))
            if k_ not in memo:
                fams, acts, _ = enum_fn(st)
                try:
                    y = system.apply(st, fams[jj], acts[jj])
                except InvalidRewrite:  # executor rejects some enumerated marks
                    y = None
                if len(memo) < 20000:
                    memo[k_] = y
                else:
                    return y
            return memo[k_]

        def checked_successor(st, origin_st, ctx, action_index, contract):
            y = apply_fn(st, int(action_index))
            if y is None:
                return None
            if y.n_real_atoms > MAX_ACTIVE_ATOMS:
                return None
            try:
                key = canonical_state_key(y)
            except InvalidRewrite:
                return None
            if not key or not is_valid(key):
                return None
            if not RR.context_preserved(
                origin_st,
                y,
                ctx.frozen,
                ctx.terminal_context_slots,
            ):
                return None
            if not RR.graph_connected(y):
                return None
            if contract is not None and not contract(y):
                return None
            return y, key

        fused_kernel = OptionContinuationKernel(
            enum_fn, system,
            max_executor_applications=int(task.get("max_executor_applications", 20000)),
        ) if include_fused or ring_options else None

        def option_state(particle):
            return OptionState(
                particle["st"], particle["origin_st"], particle["ctx"],
                particle["lin"], particle["option"], int(particle["step"]),
                option_horizon(particle["option"], max_steps),
                particle["bundle_id"], particle.get("fused_progress"),
                ring_progress=particle.get("ring_progress"),
            )

        frontier = []
        receipts: dict[str, dict] = {}
        from compose_v4.experiments.t4_parent_budget import ParentExecutorShare

        share = ParentExecutorShare(task.get("executor_calls_per_parent"))
        with share.instrument():
            for reg, multiplicity, region_score in regions_k:
                if warm_start is None:
                    raw_ = smiles_to_molecular_graph(parent)
                    st_ = pad_molecular_graph(raw_, CANONICAL_SLOTS)
                    ctx_ = RR.context_from_region(reg)
                else:
                    st_ = exact_parents[parent]
                    ctx_ = exact_context(st_, parent, reg)
                n_ = st_.n_real_atoms
                lin_ = RR.Lineage.initial(np.flatnonzero(is_element(st_.atom_types)))
                old_ = frozenset(lin_.id_of[a] for a in ctx_.locus if a in lin_.id_of)
                fams, acts, _probs = enum_fn(st_)
                idx, _why = RR.admissible_indices(fams, acts, ctx_)
                n_free = min(
                    int(np.sum(np.asarray(st_.atom_types) == int(NULL_IDX))),
                    MAX_ACTIVE_ATOMS - n_,
                )
                support_applicable = applicable_options(
                    fams, idx, n_free_slots=n_free, include_fused=include_fused,
                    ring_options=ring_options,
                )

                def has_clean_product(
                    option,
                    fams_=fams,
                    probs_=_probs,
                    idx_=idx,
                    state_=st_,
                    context_=ctx_,
                    lineage_=lin_,
                    region_=reg,
                ):
                    if option == BUILD_FUSED_RING_OPTION or ring_spec(option) is not None:
                        node = OptionState(
                            state_, state_, context_, lineage_, option, 0,
                            option_horizon(option, max_steps),
                            bundle_identity(parent, parent_lineage_id, region_.key(), option),
                            FusedProgress() if option == BUILD_FUSED_RING_OPTION else None,
                            ring_progress=RingProgress() if ring_spec(option) is not None else None,
                        )
                        return bool(fused_kernel.row(node).successors)
                    conditioned = conditioned_action_distribution(
                        fams_,
                        probs_,
                        idx_,
                        option,
                        step=0,
                        temperature=macro_temperature,
                        exploration=eps_macro,
                    )
                    active = primitive_option_at_step(option, 0)
                    contract = state_contract_for(active, state_) if active else None
                    return any(
                        checked_successor(
                            state_, state_, context_, int(j), contract
                        )
                        is not None
                        for j in conditioned.indices
                    )

                options = retain_product_applicable_options(
                    support_applicable, has_clean_product
                )
                choice = sample_option(options, rng_, exploration=eps_option)
                bundle_id = bundle_identity(
                    parent, parent_lineage_id, reg.key(), choice.selected
                )
                receipt = receipts.setdefault(
                    bundle_id,
                    {
                        "bundle_id": bundle_id,
                        "parent": parent,
                        "parent_lineage_id": int(parent_lineage_id),
                        "region_atoms": sorted(int(x) for x in reg.atoms),
                        "region_slot_atoms": sorted(int(x) for x in ctx_.locus),
                        "region_id": hashlib.sha256(
                            json.dumps(
                                [list(reg.key()[0]), [list(x) for x in reg.key()[1]]],
                                separators=(",", ":"),
                            ).encode()
                        ).hexdigest()[:20],
                        "interface": reg.interface,
                        "kind": reg.kind,
                        "r_release": float(reg.released_fraction),
                        "option": choice.selected,
                        "q_option": float(choice.selected_probability),
                        "support_applicable_options": list(support_applicable),
                        "applicable_options": list(choice.applicable),
                        "option_probabilities": {
                            name: float(prob)
                            for name, prob in zip(choice.applicable, choice.probabilities)
                        },
                        "q_region": getattr(region_score, "selection_probability", None),
                        "q_region_base": getattr(region_score, "base_probability", None),
                        "q_region_floor": getattr(region_score, "floor_probability", None),
                        "mu_exec": getattr(region_score, "mu_exec", None),
                        "region_draws": 0,
                        "particles": 0,
                        "n_candidates": 0,
                        "n_emitted": 0,
                        "max_step": 0,
                        "max_search_step": 0,
                        "max_r_coherent": 0.0,
                        "ring_system_deltas": {},
                        "program_complete": False,
                        "halt_counts": {},
                    },
                )
                receipt["region_draws"] += 1
                receipt["particles"] += int(multiplicity)
                frontier.append(
                    {
                        "st": st_,
                        "lin": lin_,
                        "ctx": ctx_,
                        "old": old_,
                        "reg": reg,
                        "mult": int(multiplicity),
                        "step": 0,
                        "option": choice.selected,
                        "q_option": float(choice.selected_probability),
                        "bundle_id": bundle_id,
                        "region_id": receipt["region_id"],
                        "origin_st": st_,
                        "origin_lin": lin_,
                        "fused_progress": (
                            FusedProgress() if choice.selected == BUILD_FUSED_RING_OPTION else None
                        ),
                        "ring_progress": RingProgress() if ring_spec(choice.selected) is not None else None,
                    }
                )

            def note_halt(particle, reason):
                receipt = receipts[particle["bundle_id"]]
                counts = receipt["halt_counts"]
                counts[reason] = counts.get(reason, 0) + int(particle["mult"])

            def record_committed_candidate(particle, y, key, lineage, active_macro, step_, ring_progress=None):
                horizon = option_horizon(particle["option"], max_steps)
                program_complete = int(step_) >= horizon
                if (particle["option"] in (
                    BUILD_RING_SYSTEM_OPTION, BUILD_FUSED_RING_OPTION
                ) or ring_spec(particle["option"]) is not None) and not program_complete:
                    return
                candidate_key = (particle["bundle_id"], key)
                if candidate_key in cands:
                    return
                # Audit displacement is relative to the bundle parent, never merely
                # the latest primitive transition.
                dd = GG.structural_displacement(
                    particle["origin_st"], y, particle["origin_lin"], lineage
                )
                (
                    element_delta,
                    added_terminal,
                    added_backbone,
                    added_sulfur,
                ) = composition_delta(parent, key)
                cands[candidate_key] = {
                    "smiles": key,
                    "interface": particle["reg"].interface,
                    "kind": particle["reg"].kind,
                    "r_release": particle["reg"].released_fraction,
                    "step": int(step_),
                    "region_id": particle["region_id"],
                    "bundle_id": particle["bundle_id"],
                    "parent": parent,
                    "parent_lineage_id": int(parent_lineage_id),
                    "option": particle["option"],
                    "option_phase": active_macro or GENERIC_OPTION,
                    "q_option": particle["q_option"],
                    "program_complete": bool(program_complete),
                    "ring_progress": ring_progress.payload() if ring_progress else None,
                    "r_coherent": dd["largest_changed_fraction"],
                    "r_change": dd["changed_fraction"],
                    "d_ring_systems": dd["d_ring_systems"],
                    "d_cycle_rank": dd["d_cycle_rank"],
                    "d_heavy": dd["d_heavy"],
                    "element_delta": element_delta,
                    "added_terminal_halogen": added_terminal,
                    "added_backbone_atoms": added_backbone,
                    "added_sulfur": added_sulfur,
                }
                if warm_start is not None:
                    cands[candidate_key]["state"] = encode_state(y)

            for step in range(max_steps):
                if not frontier:
                    break
                # Only particles with identical slot state, context, lineage and
                # bundle may share arithmetic.  Canonically identical molecules can
                # carry different slot-addressed actions and must never be merged.
                groups: dict = {}
                for p_ in frontier:
                    gk = (
                        slot_key(p_["st"]),
                        p_["bundle_id"],
                        tuple(sorted(p_["ctx"].locus)),
                        tuple(sorted(p_["lin"].id_of.items())),
                        p_["lin"].next_id,
                        p_.get("fused_progress"),
                        p_.get("ring_progress"),
                    )
                    if gk in groups:
                        groups[gk]["mult"] += p_["mult"]
                    else:
                        groups[gk] = dict(p_)

                feats, meta = [], []
                for gk, p_ in groups.items():
                    horizon = option_horizon(p_["option"], max_steps)
                    if step >= horizon:
                        continue
                    if p_["option"] == BUILD_FUSED_RING_OPTION or p_.get("ring_progress") is not None:
                        row = fused_kernel.row(option_state(p_))
                        if not row.successors:
                            note_halt(p_, "no_clean_ring_program_product" if p_.get("ring_progress") is not None else "no_clean_fused_product")
                        for successor, base_prob in zip(row.successors, row.probabilities):
                            sh = RR.structural_features_shared(
                                successor.graph, successor.context, successor.lineage, p_["old"]
                            )
                            feats.append(RR.features_from_shared(sh, horizon - step, "establishment"))
                            meta.append((
                                gk, successor.graph, successor.lineage, successor.context,
                                float(base_prob), canonical_state_key(successor.graph), p_,
                                primitive_option_at_step(p_["option"], step), successor.ring_progress or successor.fused_progress,
                            ))
                        continue
                    fams, acts, probs = enum_fn(p_["st"])
                    idx, _why = RR.admissible_indices(fams, acts, p_["ctx"])
                    if not idx:
                        note_halt(p_, "no_region_admissible_action")
                        continue

                    # Avoid applying unrelated macro families.  The first pass is
                    # support-only; the second pass renormalizes after product-level
                    # validity, context and macro-contract checks.
                    pre = conditioned_action_distribution(
                        fams,
                        probs,
                        idx,
                        p_["option"],
                        step=step,
                        temperature=macro_temperature,
                        exploration=eps_macro,
                    )
                    if not pre.indices.size:
                        note_halt(p_, "no_option_support")
                        continue
                    active_macro = primitive_option_at_step(p_["option"], step)
                    contract = state_contract_for(active_macro, p_["st"]) if active_macro else None
                    clean = np.zeros(len(probs), dtype=bool)
                    successors = {}
                    for j in pre.indices:
                        j = int(j)
                        successor = checked_successor(
                            p_["st"], p_["origin_st"], p_["ctx"], j, contract
                        )
                        if successor is None:
                            continue
                        y, key = successor
                        clean[j] = True
                        successors[j] = (y, key)

                    conditioned = conditioned_action_distribution(
                        fams,
                        probs,
                        idx,
                        p_["option"],
                        step=step,
                        clean=clean,
                        temperature=macro_temperature,
                        exploration=eps_macro,
                    )
                    if not conditioned.indices.size:
                        note_halt(p_, "no_clean_option_product")
                        continue
                    budget_left = max(0, horizon - step)
                    for j, base_prob in zip(
                        conditioned.indices, conditioned.probabilities
                    ):
                        j = int(j)
                        y, key = successors[j]
                        l2 = p_["lin"].observe(fams[j], acts[j])
                        new_slot = RR.created_slot(acts[j])
                        ctx2 = (
                            p_["ctx"].with_locus(new_slot)
                            if new_slot is not None
                            else p_["ctx"]
                        )
                        sh = RR.structural_features_shared(y, ctx2, l2, p_["old"])
                        feats.append(
                            RR.features_from_shared(sh, budget_left, "establishment")
                        )
                        meta.append(
                            (
                                gk,
                                y,
                                l2,
                                ctx2,
                                float(base_prob),
                                key,
                                p_,
                                conditioned.active_macro,
                                None,
                            )
                        )
                if not feats:
                    break

                with torch.no_grad():
                    h_all = torch.sigmoid(
                        net(torch.tensor(feats, dtype=torch.float32)).squeeze(-1)
                    ).numpy() if guidance == "committor" else np.ones(len(feats))

                nxt = []
                for gk, p_ in groups.items():
                    sel = [i for i, m_ in enumerate(meta) if m_[0] == gk]
                    if not sel:
                        continue
                    w = np.array([meta[i][4] for i in sel], float)
                    if w.sum() <= 0:
                        continue
                    w = w / w.sum()
                    hs = np.array([h_all[i] for i in sel], float)
                    primitive_floor = float(task.get("epsilon", 0.1))
                    q = primitive_distribution(w, hs, guidance, kappa, primitive_floor)
                    draws = rng_.choice(len(sel), size=int(p_["mult"]), p=q)
                    for d_ in np.unique(draws):
                        i = sel[int(d_)]
                        _gk, y, l2, ctx2, _p, key, src, active_macro, progress = meta[i]
                        ring_progress = progress if ring_spec(src["option"]) is not None else None
                        fused_progress = None if ring_progress is not None else progress
                        if task.get("prepare_only", False):
                            trace.append({
                                "bundle_id": src["bundle_id"], "option": src["option"],
                                "step": step + 1, "source": encode_state(src["st"]),
                                "product": encode_state(y), "canonical_product": key,
                                "reference_probability": float(w[int(d_)]),
                                "sample_probability": float(q[int(d_)]),
                                "reference_support_size": len(w),
                                "h_min": float(hs.min()) if guidance == "committor" else None,
                                "h_max": float(hs.max()) if guidance == "committor" else None,
                                "kl": float(np.sum(q[q > 0] * np.log(q[q > 0] / w[q > 0]))),
                                "fused_progress": fused_progress.payload() if fused_progress else None,
                                "ring_progress": ring_progress.payload() if ring_progress else None,
                            })
                        nxt.append(
                            {
                                **src,
                                "st": y,
                                "lin": l2,
                                "ctx": ctx2,
                                "mult": int((draws == d_).sum()),
                                "step": step + 1,
                                "fused_progress": fused_progress,
                                "ring_progress": ring_progress,
                            }
                        )
                        record_committed_candidate(
                            src, y, key, l2, active_macro, step + 1, ring_progress
                        )

                for particle in nxt:
                    receipt = receipts[particle["bundle_id"]]
                    receipt["max_search_step"] = max(
                        int(receipt["max_search_step"]), int(particle["step"])
                    )

                # The measured width cap is per bundle.  Applying it across all
                # bundles would let whichever bundle appears first steal another
                # region-option draw's search allocation.
                cap = int(task.get("max_frontier", 8))
                by_bundle = defaultdict(list)
                for particle in nxt:
                    by_bundle[particle["bundle_id"]].append(particle)
                frontier = [
                    particle
                    for bundle_id in sorted(by_bundle)
                    for particle in by_bundle[bundle_id][:cap]
                ]

        if share.exhausted:
            for particle in frontier:
                if particle["step"] < option_horizon(particle["option"], max_steps):
                    counts = receipts[particle["bundle_id"]]["halt_counts"]
                    counts["parent_executor_share_exhausted"] = (
                        counts.get("parent_executor_share_exhausted", 0) + int(particle["mult"])
                    )

        # THE FRONTIER IS FOR SEARCHING; THE CANDIDATE POOL IS FOR SPENDING
        # ORACLE CALLS.  Emit a small representative set from each complete
        # (parent, region, option) bundle.
        n_emit = int(task.get("emit_per_bundle", 3))
        by_bundle = defaultdict(list)
        for candidate in cands.values():
            by_bundle[candidate["bundle_id"]].append(candidate)
        out = []
        for bundle_id in sorted(by_bundle):
            cs = by_bundle[bundle_id]
            reps, seen_ = [], set()

            def _add(candidate, seen=seen_, selected=reps):
                if candidate and candidate["smiles"] not in seen:
                    seen.add(candidate["smiles"])
                    selected.append(candidate)

            _add(max(cs, key=lambda candidate: candidate["step"]))
            with_properties = [
                (candidate, props(candidate["smiles"])) for candidate in cs
            ]
            with_properties = [item for item in with_properties if item[1] is not None]
            if with_properties:
                _add(
                    min(
                        with_properties,
                        key=lambda item: (item[1]["v"], -item[1]["qed"]),
                    )[0]
                )
            rings = [
                candidate
                for candidate in cs
                if (candidate.get("d_ring_systems") or 0) != 0
                or (candidate.get("d_cycle_rank") or 0) != 0
            ]
            if rings:
                _add(
                    max(
                        rings,
                        key=lambda candidate: (
                            abs(candidate.get("d_ring_systems") or 0),
                            abs(candidate.get("d_cycle_rank") or 0),
                        ),
                    )
                )
            _add(max(cs, key=lambda candidate: candidate.get("r_coherent") or 0.0))
            for candidate in sorted(
                cs, key=lambda candidate: -(candidate.get("r_coherent") or 0.0)
            ):
                if len(reps) >= n_emit:
                    break
                _add(candidate)
            emitted = reps[:n_emit]
            out.extend(emitted)

            receipt = receipts[bundle_id]
            receipt["n_candidates"] = len(cs)
            receipt["n_emitted"] = len(emitted)
            receipt["max_step"] = max(candidate["step"] for candidate in cs)
            receipt["max_r_coherent"] = max(
                float(candidate.get("r_coherent") or 0.0) for candidate in cs
            )
            receipt["ring_system_deltas"] = {
                str(k): int(v)
                for k, v in sorted(
                    Counter(candidate.get("d_ring_systems") for candidate in cs).items(),
                    key=lambda item: str(item[0]),
                )
            }
            receipt["program_complete"] = any(
                candidate.get("program_complete", False) for candidate in cs
            )
        return out, [receipts[key] for key in sorted(receipts)], {
            "parent_budget": share.receipt() if share.limit is not None else None,
            "unselected_region_draws": len(regions_k) - sum(
                int(receipt["region_draws"]) for receipt in receipts.values()
            ),
            "law_enumerations": len(cache), "memoized_option_products": len(memo),
            ("stateful_option_products" if ring_options else "fused_option_products"):
                fused_kernel.work.executor_applications if fused_kernel else 0,
            **({"ring_program_product_cache_hits": fused_kernel.work.product_cache_hits} if ring_options else {}),
            "sampled_transitions": trace,
            "rng_state": rng_.bit_generator.state,
        }

    # Oracle allocation needs an objective signal. Ranking by (v, -qed) fails
    # once molecules are feasible: v is identically 0 and QED ties break
    # arbitrarily, which is the documented defect that left parp1 seed0 at -8.5
    # with 175 feasible dockings. Same bootstrap-ridge Thompson surrogate the
    # prior T4 controller used, on Morgan fingerprints.
    ENSEMBLE, RIDGE, WARMUP = 8, 1.0, 16
    _sur_W = [None]

    def _fp(smi):
        m = Chem.MolFromSmiles(smi)
        if m is None:
            return None
        return np.asarray(gen.GetFingerprint(m), dtype=np.float32)

    def surrogate_fit(pairs):
        if len(pairs) < WARMUP:
            _sur_W[0] = None
            return
        X = np.stack([p_[0] for p_ in pairs]); y = np.array([p_[1] for p_ in pairs],
                                                            dtype=np.float32)
        n, d = X.shape
        Ws = []
        for _ in range(ENSEMBLE):
            idx = rng.integers(0, n, n)
            Xb, yb = X[idx], y[idx]
            A = Xb.T @ Xb + RIDGE * np.eye(d, dtype=np.float32)
            Ws.append(np.linalg.solve(A, Xb.T @ yb))
        _sur_W[0] = np.stack(Ws)

    def surrogate_score(X):
        """Thompson draw: lower is better, matching docking score."""
        if _sur_W[0] is None:
            return np.zeros(len(X), dtype=np.float32)
        return X @ _sur_W[0][rng.integers(0, len(_sur_W[0]))]

    def candidate_diversity(records):
        """Mean pairwise Morgan distance over distinct canonical molecules."""
        fps_ = []
        seen_ = set()
        for record in records:
            smi = record.get("smiles") or record.get("smi")
            if not smi or smi in seen_:
                continue
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue
            seen_.add(smi)
            fps_.append(gen.GetFingerprint(mol))
        if len(fps_) < 2:
            return 0.0
        distances = [
            1.0 - float(DataStructs.TanimotoSimilarity(fps_[i], fps_[j]))
            for i in range(len(fps_))
            for j in range(i + 1, len(fps_))
        ]
        return float(np.mean(distances))

    def option_diagnostics(records):
        """Structural outcomes by selected option, independent of docking."""
        grouped = {}
        for record in records:
            grouped.setdefault(record.get("option", "unknown"), []).append(record)
        out = {}
        for option in sorted(grouped):
            items = grouped[option]
            coherent = [float(item.get("r_coherent") or 0.0) for item in items]
            out[option] = {
                "n": len(items),
                "median_r_release": float(
                    np.median([float(item.get("r_release") or 0.0) for item in items])
                ),
                "median_r_coherent": float(np.median(coherent)),
                "max_r_coherent": max(coherent),
                "ring_system_increase": sum(
                    int((item.get("d_ring_systems") or 0) > 0) for item in items
                ),
                "cycle_rank_increase": sum(
                    int((item.get("d_cycle_rank") or 0) > 0) for item in items
                ),
                "any_topology_change": sum(
                    int(
                        (item.get("d_ring_systems") or 0) != 0
                        or (item.get("d_cycle_rank") or 0) != 0
                    )
                    for item in items
                ),
                "mean_d_heavy": float(
                    np.mean([float(item.get("d_heavy") or 0.0) for item in items])
                ),
                "added_backbone_atoms": sum(
                    int(item.get("added_backbone_atoms") or 0) for item in items
                ),
                "added_terminal_halogen": sum(
                    int(item.get("added_terminal_halogen") or 0) for item in items
                ),
                "added_sulfur": sum(int(item.get("added_sulfur") or 0) for item in items),
            }
        return out

    t0 = time.perf_counter()
    p0 = props(seed)
    archive = [{"smiles": seed, **p0, "ds": None}]
    sur_pairs: list = []
    docked: dict[str, float] = {}
    n_dock, rounds_log = 0, []
    rd = 0
    if warm_start is not None:
        archive = [dict(row) for row in warm_start["archive"]]
        archive[0].update(p0)
        docked = {canonical_state_key(decode_state(row["state"])): row["ds"]
                  for row in archive}
        sur_pairs = [(_fp(row["smiles"]), row["ds"]) for row in archive if row["ds"] is not None]
        rd = int(warm_start["round"])
        surrogate_fit(sur_pairs)

    while n_dock < budget:
        rd += 1
        # ---- parents: the prior rule, top lineages by feasible-then-docking
        scored = sorted(archive, key=lambda a: (
            0 if (a["v"] <= 0 and a["ds"] is not None) else 1,
            a["ds"] if a["ds"] is not None else 0.0, a["v"]))
        parents = [a["smiles"] for a in scored[:n_lineages]] or [seed]
        # The population contract is eight lineages from round one onward.  At
        # initialization those lineages share x0 but make independent Q(M),
        # Q(o) and primitive draws; returning only one parent silently reduced
        # the first round to three bundles instead of the declared twenty-four.
        if len(parents) < n_lineages:
            parents.extend([seed] * (n_lineages - len(parents)))

        # ---- PROPOSE: region bundles per lineage, one shared frontier each.
        # Regions come from Q(M|x,z); particles within a region share the early
        # frontier, and h_phi is scored for the whole frontier in one call.
        regions_per_lineage = int(task.get("regions_per_lineage", 3))
        particles_per_region = int(task.get("particles_per_region", 4))
        t_prop = time.perf_counter()
        bundles = []
        for parent_lineage_id, parent in enumerate(parents):
            regs = [x for x in enumerate_regions(parent) if 1 <= x.size <= 24]
            if not regs:
                continue
            picked = []
            for _ in range(regions_per_lineage):
                reg, region_score = sample_region(
                    regs,
                    rng,
                    epsilon=eps_region,
                    value_fn=value_fn,
                    tau=tau,
                )
                if reg is not None:
                    picked.append((reg, particles_per_region, region_score))
            if picked:
                bundles.append((parent, parent_lineage_id, picked))
        cands = []
        bundle_receipts = []
        work_receipts = []
        def prepare_parent(par, parent_lineage_id, picked, parent_rng):
            cached = (parent_cache or {}).get(parent_lineage_id)
            if cached is not None:
                return cached["candidates"], cached["bundles"], cached["work"]
            return batched_rewrite(
                par, parent_lineage_id, picked, int(task.get("max_handoff", 16)), parent_rng
            )
        jobs = [
            (par, parent_lineage_id, picked,
             np.random.default_rng(int(rng.integers(0, 10 ** 9))))
            for par, parent_lineage_id, picked in bundles
        ]

        def collect(result, cands=cands, bundle_receipts=bundle_receipts, work_receipts=work_receipts):
            candidates_, receipts_, work_ = result
            cands.extend(candidates_)
            bundle_receipts.extend(receipts_)
            work_receipts.append(work_)
            if progress is not None:
                progress({
                    "phase": "parent_complete", "parent_index": len(work_receipts) - 1,
                    "candidates": candidates_, "bundles": receipts_, "work": work_,
                })

        if workers == 1:
            # Complete and persist one restart unit before starting the next.
            for job in jobs:
                collect(prepare_parent(*job))
        else:
            with ThreadPoolExecutor(max_workers=workers) as ex:
                futs = [ex.submit(prepare_parent, *job) for job in jobs]
                for future in futs:
                    collect(future.result())
        t_prop = time.perf_counter() - t_prop

        # Repeated region draws may select the same option.  They are repeated
        # probability mass for one (parent, region, option) bundle, not fake
        # bundle diversity, so merge their receipts explicitly.
        merged_receipts = {}
        for receipt in bundle_receipts:
            bundle_id = receipt["bundle_id"]
            if bundle_id not in merged_receipts:
                merged_receipts[bundle_id] = dict(receipt)
                continue
            dst = merged_receipts[bundle_id]
            for field in (
                "region_draws",
                "particles",
                "n_candidates",
                "n_emitted",
            ):
                dst[field] += int(receipt[field])
            dst["max_step"] = max(int(dst["max_step"]), int(receipt["max_step"]))
            dst["max_search_step"] = max(
                int(dst["max_search_step"]), int(receipt["max_search_step"])
            )
            dst["max_r_coherent"] = max(
                float(dst["max_r_coherent"]), float(receipt["max_r_coherent"])
            )
            dst["program_complete"] = bool(
                dst["program_complete"] or receipt["program_complete"]
            )
            ring_counts = {
                key: int(value) for key, value in dst["ring_system_deltas"].items()
            }
            for key, value in receipt["ring_system_deltas"].items():
                ring_counts[key] = ring_counts.get(key, 0) + int(value)
            dst["ring_system_deltas"] = dict(sorted(ring_counts.items()))
            for reason, value in receipt["halt_counts"].items():
                dst["halt_counts"][reason] = dst["halt_counts"].get(reason, 0) + int(value)
        bundle_receipts = [merged_receipts[key] for key in sorted(merged_receipts)]

        # ---- cheap downselect BEFORE spending the oracle
        pool_by_smiles = {}
        for c in cands:
            if c["smiles"] in docked:
                continue
            pr = props(c["smiles"])
            if pr is None:
                continue
            # Global canonical deduplication is independent of the bundle
            # frontier.  One molecule earns at most one oracle call.
            existing = pool_by_smiles.get(c["smiles"])
            if existing is None:
                pool_by_smiles[c["smiles"]] = {
                    **c,
                    **pr,
                    "origin_bundle_ids": [c["bundle_id"]],
                    "origin_options": [c["option"]],
                }
            else:
                if c["bundle_id"] not in existing["origin_bundle_ids"]:
                    existing["origin_bundle_ids"].append(c["bundle_id"])
                if c["option"] not in existing["origin_options"]:
                    existing["origin_options"].append(c["option"])
        # Endpoint allocation only: rejected records remain in the locked pool.
        # Do not apply this mask to intermediate states or change bundle draws.
        pool = annotate_endpoints(list(pool_by_smiles.values()), endpoint_policy)
        # Round-robin across (parent, region, option) instead of a global sort.
        # A global sort by (v, -qed) put 19 of 20 dockings on ONE region: every
        # candidate is feasible with near-identical QED, so ties broke
        # arbitrarily and the oracle was spent on near-siblings from a single
        # frontier. Diversity of what gets DOCKED is the point of having 4816
        # candidates.
        from collections import defaultdict as _dd
        by_src = _dd(list)
        for a in pool:
            if a.get("oracle_eligible", True):
                by_src[a.get("bundle_id")].append(a)
        # rank WITHIN a bundle by the surrogate (falls back to QED before warmup)
        fps = {}
        for a in pool:
            f_ = _fp(a["smiles"])
            if f_ is not None:
                fps[a["smiles"]] = f_
        if fps and _sur_W[0] is not None:
            keys = list(fps)
            sc = surrogate_score(np.stack([fps[k] for k in keys]))
            pred = dict(zip(keys, sc))
        else:
            pred = {}
        if warm_start is not None:
            for candidate in pool:
                value = pred.get(candidate["smiles"])
                candidate["surrogate_score"] = float(value) if value is not None else None
        for v_ in by_src.values():
            v_.sort(key=lambda a: (a["v"], pred.get(a["smiles"], -a.get("qed", 0))))
        take = []
        srcs = [by_src[bundle_id] for bundle_id in sorted(by_src)]
        want = min(per_round, budget - n_dock)
        while len(take) < want and srcs:
            next_pass = []
            for group in srcs:
                if len(take) >= want:
                    break
                take.append(group.pop(0))
                if group:
                    next_pass.append(group)
            srcs = next_pass

        if task.get("prepare_only", False):
            return {
                "schema_version": "t4_candidate_lock_v1", "task": task,
                "input_sha256": input_hashes, "bundles": bundle_receipts,
                "pool": pool, "take": take, "round": rd,
                "n_candidates": len(cands), "work": work_receipts,
                "proposal_seconds": t_prop, "candidate_diversity": candidate_diversity(pool),
                "selected_diversity": candidate_diversity(take),
                "candidate_option_diagnostics": option_diagnostics(pool),
                "selected_option_diagnostics": option_diagnostics(take),
                "started_at_utc": started_at,
                "locked_at_utc": datetime.now(timezone.utc).isoformat(),
                "rng_state": rng.bit_generator.state,
                "oracle_calls": 0,
                "software": {
                    "python": platform.python_version(), "numpy": np.__version__,
                    "torch": torch.__version__, "rdkit": Chem.rdBase.rdkitVersion,
                },
            }

        # ---- BATCH docking, concurrent; archive updated only afterwards
        t_d = time.perf_counter()
        scores = _dock_many([a["smiles"] for a in take], target,
                            f"{task['cell']}_r{rd}", workers=workers)
        t_d = time.perf_counter() - t_d
        for a, ds in zip(take, scores):
            n_dock += 1
            docked[a["smiles"]] = ds if ds is not None else 0.0
            archive.append({**a, "ds": ds})
            f_ = _fp(a["smiles"])
            if f_ is not None and ds is not None:
                sur_pairs.append((f_, float(ds)))
        surrogate_fit(sur_pairs)

        feas = [a for a in archive if a["v"] <= 0 and a["ds"] is not None
                and a["smiles"] != seed]
        best = min(feas, key=lambda a: a["ds"]) if feas else None
        options_selected = Counter()
        for receipt in bundle_receipts:
            options_selected[receipt["option"]] += int(receipt["region_draws"])
        options_docked = Counter(a["option"] for a in take)
        candidates_by_option = Counter(a["option"] for a in pool)
        rounds_log.append({
            "round": rd,
            "n_parent_batches": len(bundles),
            "n_region_draws": sum(
                int(receipt["region_draws"]) for receipt in bundle_receipts
            ),
            "n_bundles": len(bundle_receipts),
            "n_cand": len(cands),
            "n_unique_candidates": len(pool),
            "n_pool": len(pool),
            "candidate_diversity": round(candidate_diversity(pool), 6),
            "docked_diversity": round(candidate_diversity(take), 6),
            "n_docked_total": n_dock,
            "n_docking_failures": sum(ds is None for ds in scores),
            "n_lineages": len(parents),
            "best_ds": best["ds"] if best else None,
            "t_propose": round(t_prop, 1),
            "t_dock": round(t_d, 1),
            "elapsed": round(time.perf_counter() - t0, 1),
            "scales": [round(c["r_release"], 2) for c in cands[:20]],
            "n_bundles_selected": len(bundle_receipts),
            "n_bundles_docked": len({a.get("bundle_id") for a in take}),
            "bundle_scopes": sorted(
                {round(receipt["r_release"], 2) for receipt in bundle_receipts}
            ),
            "options_selected": dict(sorted(options_selected.items())),
            "options_docked": dict(sorted(options_docked.items())),
            "candidates_by_option": dict(sorted(candidates_by_option.items())),
            "candidate_option_diagnostics": option_diagnostics(pool),
            "docked_option_diagnostics": option_diagnostics(take),
            "bundles": bundle_receipts,
            "docked": [
                {
                    "smi": a["smiles"],
                    "ds": ds,
                    "surrogate_score": pred.get(a["smiles"]),
                    "bundle_id": a.get("bundle_id"),
                    "origin_bundle_ids": a.get("origin_bundle_ids"),
                    "origin_options": a.get("origin_options"),
                    "parent_lineage_id": a.get("parent_lineage_id"),
                    "option": a.get("option"),
                    "option_phase": a.get("option_phase"),
                    "program_complete": a.get("program_complete"),
                    "r_release": a["r_release"],
                    "r_change": a.get("r_change"),
                    "r_coherent": a.get("r_coherent"),
                    "d_rings": a.get("d_ring_systems"),
                    "d_cycle_rank": a.get("d_cycle_rank"),
                    "d_heavy": a.get("d_heavy"),
                    "element_delta": a.get("element_delta"),
                    "added_backbone_atoms": a.get("added_backbone_atoms"),
                    "added_terminal_halogen": a.get("added_terminal_halogen"),
                    "added_sulfur": a.get("added_sulfur"),
                    "step": a.get("step"),
                    "interface": a["interface"],
                    "qed": a.get("qed"),
                    "sa": a.get("sa"),
                    "sim": a.get("sim"),
                }
                for a, ds in zip(take, scores)
            ],
        })
        res = {
            "schema_version": "t4_three_level_option_audit_v1",
            "session": task.get("session"),
            "cell": task["cell"],
            "arm": arm,
            "target": target,
            "delta": delta,
            "seed": seed,
            "seed_props": p0,
            "n_dock": n_dock,
            "n_feasible": len(feas),
            "n_docking_failures": sum(
                int(round_record.get("n_docking_failures", 0))
                for round_record in rounds_log
            ),
            "best_ds": best["ds"] if best else None,
            "best_smiles": best["smiles"] if best else None,
            "rounds": rounds_log,
            "complete": n_dock >= budget,
            "sec": time.perf_counter() - t0,
            "controller": {
                "factorization": "Q(M|x,z) -> Q(o|x,M,z) -> q(w|x,M,o)",
                "region_prior": "mu_exec" if value_fn is None else "mu_exec_exp_V",
                "region_value_identity": "none" if value_fn is None else "task_value",
                "kappa": kappa,
                "epsilon_primitive": float(task.get("epsilon", 0.1)),
                "epsilon_region": eps_region,
                "epsilon_option": eps_option,
                "macro_temperature": macro_temperature,
                "epsilon_macro": eps_macro,
                "options": list(OPTIONS) + ([BUILD_FUSED_RING_OPTION] if include_fused else []) + list(ring_options),
                "primitive_guidance": guidance,
                "option_groups": {
                    group: list(options) for group, options in OPTION_GROUPS.items()
                },
                "generic_permanently_active": True,
                "compound_program": BUILD_RING_SYSTEM_OPTION,
                "ordinary_macro_horizon": 1,
                "generic_horizon": int(task.get("max_handoff", 16)),
                "build_ring_system_horizon": 11,
                "max_active_atoms": MAX_ACTIVE_ATOMS,
                "persistent_slots": CANONICAL_SLOTS,
                "lineages": n_lineages,
                "per_round": per_round,
                "regions_per_lineage": regions_per_lineage,
                "particles_per_region": particles_per_region,
                "max_frontier_per_bundle": int(task.get("max_frontier", 8)),
                "emit_per_bundle": int(task.get("emit_per_bundle", 3)),
            },
            "provenance": {
                "code_revision": task.get("code_revision"),
                "seed_manifest_sha256": task.get("seed_manifest_sha256"),
                "input_sha256": input_hashes,
                "r_theta_run_root": RUN_ROOT,
                "committor_path": committor_path,
                "seed_rng": int(task["seed_rng"]),
                "started_at_utc": started_at,
                "updated_at_utc": datetime.now(timezone.utc).isoformat(),
                "python": platform.python_version(),
                "numpy": np.__version__,
                "torch": torch.__version__,
                "hardware": "Modal CPU",
                "precision": "float32 committor; float64 probability arrays",
            },
        }
        d = Path(POP_OUT); d.mkdir(parents=True, exist_ok=True)
        output_path = d / f"{task['cell']}.json"
        temporary_path = d / f".{task['cell']}.json.tmp"
        temporary_path.write_text(json.dumps(res, sort_keys=True))
        temporary_path.replace(output_path)
        artifact_volume.commit()
        print(f"[t4pop] {task['cell']} r{rd} dock={n_dock}/{budget} "
              f"cand={len(cands)} best={best['ds'] if best else None} "
              f"t_prop={t_prop:.0f}s t_dock={t_d:.0f}s "
              f"elapsed={time.perf_counter() - t0:.0f}s", flush=True)
        if not take:
            break
    return res


@app.local_entrypoint()
def t4_population(budget: int = 500, arms: str = "mu_exec,Q_taskvalue",
                  deltas: str = "0.4,0.6", n_seeds: int = 15,
                  tau: float = 0.05, lineages: int = 8, per_round: int = 20,
                  regions_per_lineage: int = 3, particles_per_region: int = 4,
                  workers: int = 8, dry_run: bool = True):
    """T4 with the local-to-global population search. DRY RUN BY DEFAULT.

    `dry_run=True` prints the full plan -- cells, arms, budget, total docking
    calls -- and launches nothing. Pass `--no-dry-run` to actually spend the
    oracle budget. The default is deliberate: this is the benchmark, and it
    should not be startable by a stray command.
    """
    import sys as _sys
    _sys.path.insert(0, "tools")
    from preflight import assert_synced
    assert_synced(strict=False)

    seeds = json.loads(Path("docs/GENMOL_T4_SEEDS.json").read_text())
    seeds = seeds[:n_seeds]
    table = None
    p = Path("diagnostics/task_value_qed.json")
    if p.exists():
        table = json.loads(p.read_text())
    arm_list = [a.strip() for a in arms.split(",")]
    d_list = [float(d) for d in deltas.split(",")]

    jobs = []
    for i, s in enumerate(seeds):
        for d in d_list:
            for a in arm_list:
                t = {"cell": f"{a}_{s['target']}_{i}_d{d}", "smiles": s["smiles"],
                     "target": s["target"], "delta": d, "budget": budget,
                     "arm": a, "tau": tau, "seed_rng": 1000 + i,
                     "kappa": 1.0, "epsilon_region": 0.2,
                     # prior round contract: 8 lineages, 20 docked per round
                     "lineages": lineages, "per_round": per_round,
                     "regions_per_lineage": regions_per_lineage,
                     "particles_per_region": particles_per_region,
                     "workers": workers}
                if a == "Q_taskvalue":
                    if table is None:
                        raise SystemExit("no task_value table; run the QED "
                                         "collect first or drop that arm")
                    t["value_table"] = table
                jobs.append(t)

    total = len(jobs) * budget
    print(f"\nT4 POPULATION SEARCH -- {'PLAN ONLY (dry run)' if dry_run else 'LAUNCHING'}")
    print(f"  seeds={len(seeds)}  deltas={d_list}  arms={arm_list}")
    print(f"  cells={len(jobs)}  budget={budget} docking calls/cell")
    print(f"  TOTAL DOCKING CALLS = {total:,}")
    print(f"  published protocol is 1,000/cell; this run uses {budget}")
    print(f"  frozen: R_theta, committor, kappa=1.0, mu_exec, "
          f"tau={tau}, epsilon_region=0.2")
    print(f"  feasibility: QED>={QED_MIN}, SA<={SA_MAX}, sim>=delta; "
          f"seed's own docking not charged")
    import statistics as st
    cmp_ = json.loads(Path("docs/genmol_t4_all_methods.json").read_text())["rows"]
    print("\n  comparators (read from docs/genmol_t4_all_methods.json):")
    for m, c4, c6 in (("GenMol", "genmol_d04", "genmol_d06"),
                      ("RetMol", "retmol_d04", "retmol_d06"),
                      ("GraphGA", "graphga_d04", "graphga_d06")):
        n = sum(1 for r in cmp_ for c in (c4, c6) if r.get(c) is not None)
        v4 = [r[c4] for r in cmp_ if r.get(c4) is not None]
        v6 = [r[c6] for r in cmp_ if r.get(c6) is not None]
        print(f"    {m:<8} solved {n}/30   mean d0.4 {st.mean(v4):.2f}   "
              f"mean d0.6 {st.mean(v6):.2f}")
    if dry_run:
        print("\n  DRY RUN -- nothing launched. Re-run with --no-dry-run to spend "
              "the oracle budget.")
        return
    out = [o for o in t4_population_cell.map(jobs) if o]
    Path("diagnostics").mkdir(exist_ok=True)
    Path("diagnostics/t4_population.json").write_text(json.dumps(out, default=str))
    t4_population_report(out)


def t4_population_report(out):
    import statistics as st
    print(f"\ncells={len(out)}")
    for arm in sorted({o["arm"] for o in out}):
        v = [o for o in out if o["arm"] == arm]
        for d in sorted({o["delta"] for o in v}):
            w = [o for o in v if o["delta"] == d]
            solved = [o for o in w if o["best_ds"] is not None]
            print(f"  {arm:<12} delta={d}  solved {len(solved)}/{len(w)}  "
                  f"mean best {st.mean([o['best_ds'] for o in solved]) if solved else float('nan'):.2f}")
    ev = [e for o in out for e in o["events"]]
    print(f"\nproposals={len(ev)} accepted={sum(1 for e in ev if e['ok'])}")
    print("scale usage (does it use different scales, question 3):")
    for lo, hi in ((0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.01)):
        b = [e for e in ev if e.get("r_release") is not None
             and lo <= e["r_release"] < hi]
        if b:
            print(f"  {lo}-{hi}: proposed {len(b)} accepted "
                  f"{sum(1 for e in b if e['ok'])}")


@app.local_entrypoint()
def t4_spawn(budget: int = 500, arms: str = "mu_exec,Q_taskvalue",
             deltas: str = "0.4,0.6", n_seeds: int = 15, tau: float = 0.05,
             lineages: int = 8, per_round: int = 20,
             regions_per_lineage: int = 3, particles_per_region: int = 4,
             workers: int = 8):
    """Fire the cells server-side and exit, leaving no client to lose.

    `modal run --detach` still cancels a .map() when the local client dies, and
    that killed three consecutive T4 attempts at roughly the two-hour mark --
    once before round 1, once at round 10 with 71 of 100 calls done. A
    multi-hour benchmark cannot depend on a laptop staying awake, so each cell
    is spawned instead: the call is queued server-side, this entrypoint returns
    immediately, and results are read from the volume afterwards.
    """
    import sys as _sys
    _sys.path.insert(0, "tools")
    from preflight import assert_synced
    assert_synced(strict=False)
    seeds = json.loads(Path("docs/GENMOL_T4_SEEDS.json").read_text())[:n_seeds]
    table = None
    p_ = Path("diagnostics/task_value_qed.json")
    if p_.exists():
        table = json.loads(p_.read_text())
    ids = []
    for i, s_ in enumerate(seeds):
        for d in [float(x) for x in deltas.split(",")]:
            for a in [x.strip() for x in arms.split(",")]:
                t = {"cell": f"{a}_{s_['target']}_{i}_d{d}", "smiles": s_["smiles"],
                     "target": s_["target"], "delta": d, "budget": budget,
                     "arm": a, "tau": tau, "seed_rng": 1000 + i, "kappa": 1.0,
                     "epsilon_region": 0.2, "lineages": lineages,
                     "per_round": per_round,
                     "regions_per_lineage": regions_per_lineage,
                     "particles_per_region": particles_per_region,
                     "workers": workers}
                if a == "Q_taskvalue":
                    if table is None:
                        raise SystemExit("no task_value table for the tilt arm")
                    t["value_table"] = table
                ids.append((t["cell"], t4_population_cell.spawn(t).object_id))
    Path("diagnostics").mkdir(exist_ok=True)
    Path("diagnostics/t4_spawned.json").write_text(json.dumps(
        {"budget": budget, "cells": [{"cell": c, "call_id": o} for c, o in ids]}))
    print(f"spawned {len(ids)} cells server-side; this client can now exit")
    for c, o in ids[:6]:
        print(f"  {c}  {o}")
    print("\nread progress with:  modal run modal_apps/genmol_t4_opt_app.py::t4_progress")


@app.function(image=image, cpu=(1.0, 1.0), memory=int(2 * 1024), timeout=15 * 60,
              volumes={ARTIFACT_ROOT: artifact_volume})
def _read_cells() -> list:
    artifact_volume.reload()
    d = Path(POP_OUT)
    return [json.loads(f.read_text()) for f in sorted(d.glob("*.json"))] if d.exists() else []


@app.local_entrypoint()
def t4_progress():
    """Read whatever the spawned cells have persisted so far."""
    out = _read_cells.remote()
    print(f"cells with data: {len(out)}")
    for r in sorted(out, key=lambda r: r["cell"]):
        rd = r.get("rounds", [])
        print(f"  {r['cell']:<28} dock={r.get('n_dock',0):>4}/{r.get('n_dock',0) and ''}"
              f"  best={str(r.get('best_ds')):>7}  rounds={len(rd):>3}  "
              f"complete={r.get('complete')}  {r.get('sec',0)/3600:.1f}h")


@app.local_entrypoint()
def t4_population_report_only():
    out = json.loads(Path("diagnostics/t4_population.json").read_text())
    t4_population_report(out)


@app.function(image=image, cpu=(2.0, 2.0), memory=int(8 * 1024), timeout=60 * 60,
              volumes={ARTIFACT_ROOT: artifact_volume})
def ring_admissibility_probe(task: dict[str, Any]) -> dict[str, Any]:
    """Why does region rewriting almost never form a ring?

    On this cell the winning molecule needed a NEW FUSED RING (+1 ring for +1
    heavy atom); our proposals instead decorate with halogens and sulfur and
    changed a ring system once in forty dockings.

    Ring formation generally has to bond region material ONTO context atoms, and
    `admissible_indices` refuses anything reaching non-terminal frozen context,
    because that is the invariant preserving C = x \\ M. So the hypothesis is
    that ring-forming families are proposed by R_theta and then systematically
    refused by region admissibility -- not that R_theta fails to propose them.

    This counts, per family: how often R_theta offers it, how much probability
    mass it carries, and how often it is admitted versus refused and why. No
    docking; nothing here can be tuned toward the benchmark.
    """
    import sys
    from collections import Counter, defaultdict
    sys.path.insert(0, str(REMOTE_ROOT / "src"))
    import numpy as np
    from compose_v4.chem.state import pad_molecular_graph
    from compose_v4.chem.molecular_graph import smiles_to_molecular_graph
    from compose_v4.experiments.production_successor_kernel import (
        enumerate_factorized_marked_law)
    from compose_v4.control.region import enumerate_regions
    from compose_v4.control import region_rewrite as RR

    rt = _runtime(); model = rt["model"]
    smi = task["smiles"]
    raw = smiles_to_molecular_graph(smi)
    st0 = pad_molecular_graph(raw, CANONICAL_SLOTS)
    law = enumerate_factorized_marked_law(model, st0, float(TIME_POINT))
    fams = [m.executor_rule_name for m in law.marks]
    acts = [m.action for m in law.marks]
    probs = np.array([m.probability for m in law.marks], float)

    offered = Counter(fams)
    mass = defaultdict(float)
    for f, p_ in zip(fams, probs):
        mass[f] += float(p_)

    regions = [r for r in enumerate_regions(smi) if 1 <= r.size <= 24]
    per_region = []
    admit = Counter(); reject = defaultdict(Counter)
    for reg in regions[: int(task.get("n_regions", 40))]:
        ctx = RR.context_from_region(reg)
        idx, _why = RR.admissible_indices(fams, acts, ctx)
        ok_f = Counter(fams[j] for j in idx)
        for f, c in ok_f.items():
            admit[f] += c
        # per-family refusal reason, recomputed so the funnel is attributable
        allowed = ctx.locus | ctx.terminal_context_slots
        for j, f in enumerate(fams):
            if j in set(idx):
                continue
            a = acts[j]
            reach = RR.touched_slots(a)
            new = RR.created_slot(a)
            if new is not None:
                reach = reach - {new}
            if reach & (ctx.frozen - ctx.terminal_context_slots):
                reject[f]["frozen_touched"] += 1
            elif not reach or not (reach <= allowed):
                reject[f]["not_in_locus"] += 1
            else:
                reject[f]["context_only"] += 1
        per_region.append({"size": reg.size, "interface": reg.interface,
                           "r_release": reg.released_fraction,
                           "n_admissible": len(idx),
                           "ring_admissible": sum(ok_f[f] for f in
                                                  ("cycle_insert", "cycle_attach",
                                                   "ring_system_grow",
                                                   "ring_system_restate"))})
    res = {"smiles": smi, "n_marks": len(fams),
           "offered": dict(offered), "mass": {k: round(v, 5) for k, v in mass.items()},
           "admitted_total": dict(admit),
           "rejected": {k: dict(v) for k, v in reject.items()},
           "regions_probed": len(per_region), "per_region": per_region}
    d = Path("/artifacts/t4_population"); d.mkdir(parents=True, exist_ok=True)
    (d / "ring_admissibility.json").write_text(json.dumps(res))
    artifact_volume.commit()
    return res


@app.local_entrypoint()
def ring_probe(smiles: str = "", n_regions: int = 40):
    if not smiles:
        smiles = json.loads(Path("docs/GENMOL_T4_SEEDS.json").read_text())[0]["smiles"]
    r = ring_admissibility_probe.remote({"smiles": smiles, "n_regions": n_regions})
    RING = ("cycle_insert", "cycle_attach", "ring_system_grow", "ring_system_restate")
    print(f"seed: {r['smiles']}\nmarks in the law: {r['n_marks']}, "
          f"regions probed: {r['regions_probed']}\n")
    print(f"{'family':<22} {'offered':>8} {'R_theta mass':>13} {'admitted':>9} {'frozen_rej':>11}")
    for f in sorted(r["offered"], key=lambda f: -r["offered"][f]):
        rej = r["rejected"].get(f, {})
        star = " <-- RING" if f in RING else ""
        print(f"{f:<22} {r['offered'][f]:>8} {r['mass'].get(f,0):>13.5f} "
              f"{r['admitted_total'].get(f,0):>9} {rej.get('frozen_touched',0):>11}{star}")
    ring_off = sum(r["offered"].get(f, 0) for f in RING)
    ring_adm = sum(r["admitted_total"].get(f, 0) for f in RING)
    ring_mass = sum(r["mass"].get(f, 0.0) for f in RING)
    print(f"\nRING-FORMING families: offered {ring_off} marks carrying "
          f"{ring_mass:.4f} of R_theta mass")
    print(f"  admitted across {r['regions_probed']} regions: {ring_adm}")
    nz = [p for p in r["per_region"] if p["ring_admissible"] > 0]
    print(f"  regions with ANY ring-forming action admissible: {len(nz)}/{r['regions_probed']}")
