"""Held-in calibration for the same-prefix goal intervention. CPU ONLY.

Decides FOUR things and stops. It is not an experiment and produces no claim.

  1. potency threshold -- which of 0.3 / 0.5 / 0.7 avoids floor and ceiling
  2. developability region base rate -- QED floor and cLogP interval
  3. horizon -- is there post-switch room at remaining budget 4, 3, 2, 1
  4. THE GATE on subclaim B: does the bounded developability region actually
     contain contrastive future-sensitive decisions?

(4) is the dynamic-retargeting analogue of C0, and it is the reason this run
exists. C0 asked the same question of monotone DRD2 maximisation and returned
STOP -- sacrificial actions paid off at chance. Asking it again of a BOUNDED
goal is the whole point: bounded feasibility is non-monotone, so a locally
attractive edit can consume the room needed to land inside the region later.

    top-1 disagreement   does the best remaining-budget action differ from
                         greedy's best immediate action?
    sacrifice-to-win     when greedy's action is NOT the lookahead's choice and
                         the lookahead's choice scores worse immediately, does
                         it end better? Null is 0.5.
    future regret        u_B(V_G of lookahead pick) - u_B(V_G of greedy pick)

DIFFERENCE FROM C0. C0 estimated future value by Monte Carlo and therefore had
to draw independent selection and evaluation samples to avoid selecting and
scoring on the same noise. Here the lookahead is the DETERMINISTIC greedy
continuation V_G -- the same object the sealed-67 controller commits. It carries
the policy-improvement guarantee, and there is no sampling noise to be optimistic
about, so no independent evaluation sample is needed.

ANTI-TUNING RULE, BINDING. This runs ONCE. If the gate comes back empty, that is
recorded and subclaim B leaves the paper. The interval is NOT retuned until the
measurement cooperates.

Every threshold is swept POST HOC from recorded trajectory properties wherever
the policy does not depend on it: the potency policy maximises DRD2 log-odds and
is threshold-free, so 0.3 / 0.5 / 0.7 are evaluated from one set of rollouts.
The developability policy does depend on its region, so that one is run at the
candidate parameters and its neighbours are reported as base rates only.
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
    ROOT,
    artifact_volume,
)
from modal_apps.run_process_v2_p50_app import image as _base_image

image = (
    _base_image
    .env({"PYTHONPATH": f"{REMOTE_ROOT}/src:{REMOTE_ROOT}"})
    .add_local_file(
        ROOT / "diagnostics/retarget_goal_language_normalizers.json",
        str(REMOTE_ROOT / "diagnostics/retarget_goal_language_normalizers.json"),
        copy=True)
    .add_local_file(
        ROOT / "diagnostics/retarget_calibration_cohort.json",
        str(REMOTE_ROOT / "diagnostics/retarget_calibration_cohort.json"),
        copy=True)
    .add_local_dir(ROOT / "artifacts/oracles/drd2_svm_v1",
                   str(REMOTE_ROOT / "artifacts/oracles/drd2_svm_v1"), copy=True)
)
app = modal.App("compose-v4-retarget-goal-calibration")

RUN_ROOT = "/artifacts/editing_v2/r_theta_run"
TIME_POINT = 0.5
CANONICAL_SLOTS = 48

#: Prefix under A, then the same budget again under B.
#: 4+4 CEILINGED -- greedy reached the developability region 29/30, leaving no
#: room to measure a planning advantage. The calibration mandate allowed exactly
#: one alternative, 3+3, and this is it. Budget 2 is NOT on the table: shortening
#: until greedy finally loses is result-shaping, not calibration.
PREFIX_STEPS = 3
POST_STEPS = 3

#: Candidate universe for a TARGET-FREE goal. Experiment C's top-similarity
#: stratum has no analogue without a target molecule, so the immediate-score
#: stratum takes its place. The random stratum is kept -- as in C0 -- because
#: this is a mechanism probe: a universe defined entirely by the greedy score
#: cannot show that a non-greedy action was worth taking.
CANDIDATES_TOP_IMMEDIATE = 4
CANDIDATES_TOP_REFERENCE = 2
CANDIDATES_RANDOM = 2

#: Candidate developability region. Set ONCE, from the local base-rate census.
#: Not retuned if the gate comes back empty -- see the anti-tuning rule.
QED_FLOOR = 0.6
LOGP_BOX = (1.0, 4.0)
POTENCY_GRID = (0.3, 0.5, 0.7)
#: Clip on normalised margins before the soft minimum. Without it the DRD2 term
#: -- 2.12 IQR below zero for a typical molecule -- pins the soft-min for the
#: whole trajectory and every developability move is invisible to the policy.
MARGIN_CLIP = 1.5
SOFTMIN_TAU = 0.25

#: Size band: reserve p10-p95 heavy atoms, an outcome-independent rule.
MIN_HEAVY, MAX_HEAVY = 18, 38


def _goal_helpers():
    """Margin/utility definitions, shared by selection and evaluation."""

    import numpy as np

    def softmin(margins: list[float]) -> float:
        clipped = [min(max(m, -MARGIN_CLIP), MARGIN_CLIP) for m in margins]
        return float(-SOFTMIN_TAU * np.log(
            sum(np.exp(-m / SOFTMIN_TAU) for m in clipped)))

    return softmin


@app.function(
    image=image, cpu=2.0, memory=12 * 1024, timeout=4 * 60 * 60,
    max_containers=30, retries=1,
    volumes={str(ARTIFACT_ROOT): artifact_volume},
)
def calibrate_source(task: dict[str, Any]) -> dict[str, Any]:
    import random
    import sys

    import numpy as np
    import torch

    sys.path.insert(0, str(REMOTE_ROOT / "src"))
    from rdkit import Chem, RDLogger
    from rdkit.Chem import Crippen, QED

    from compose_v4.chem.molecular_graph import smiles_to_molecular_graph
    from compose_v4.chem.state import pad_molecular_graph
    from compose_v4.drd2_oracle import load_default_oracle
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
        canonical_successor_result,
    )
    from compose_v4.rewrite.kernel import canonical_state_key

    RDLogger.DisableLog("rdApp.*")
    started = time.perf_counter()
    artifact_volume.reload()

    paths = json.loads((Path(RUN_ROOT) / "run_inputs" / "RUN_PATHS.json").read_text())
    source = open_process_v2_t1_source(
        Path(paths["active8_root"]),
        gate_zero_decision_path=Path(paths["gate_zero"]),
        artifact_root=Path(paths["artifact_root"]),
        repo_root=REMOTE_ROOT)
    bundle = load_materialized_scorer_state(Path(paths["materialized_scorer"]))
    runtime, _b, _c = build_process_v2_score_revised_scratch_runtime(
        source, materialized_state=bundle)
    model = runtime.model
    checkpoint = torch.load(Path(RUN_ROOT) / "runs" / "run_v2_01" / CHECKPOINT_FILENAME,
                            map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["selected_model_state"], strict=True)
    model.eval()

    norms = json.loads((REMOTE_ROOT /
                        "diagnostics/retarget_goal_language_normalizers.json"
                        ).read_text())["normalizers"]
    s_drd2 = norms["drd2_logodds"]["iqr"]
    s_qed = norms["qed"]["iqr"]
    s_logp = norms["clogp"]["iqr"]
    oracle = load_default_oracle(
        str(REMOTE_ROOT / "artifacts/oracles/drd2_svm_v1/drd2_oracle_manifest.json"))
    softmin = _goal_helpers()

    slots = int(task["slots"])
    start_key = canonical_state_key(
        pad_molecular_graph(smiles_to_molecular_graph(task["source"]), slots))

    property_cache: dict[str, tuple[float, float, float]] = {}
    enumeration: dict[str, list[tuple[str, float]]] = {}
    calls = 0

    def props(key: str) -> tuple[float, float, float]:
        """(drd2 log-odds, QED, cLogP). One oracle call per distinct molecule."""
        if key in property_cache:
            return property_cache[key]
        mol = Chem.MolFromSmiles(key)
        if mol is None:
            out = (float("nan"),) * 3
        else:
            margin = float(oracle.margin_many([key])[0])
            try:
                qed = float(QED.qed(mol))
            except Exception:  # noqa: BLE001
                qed = float("nan")
            out = (margin, qed, float(Crippen.MolLogP(mol)))
        property_cache[key] = out
        return out

    def potency_score(key: str) -> float:
        """Threshold-FREE: the policy maximises log-odds, so one set of
        rollouts serves every threshold in the grid."""
        return props(key)[0] / s_drd2

    def develop_margins(key: str) -> list[float]:
        _m, qed, logp = props(key)
        if not (np.isfinite(qed) and np.isfinite(logp)):
            return [-10.0, -10.0]
        return [(qed - QED_FLOOR) / s_qed,
                min(logp - LOGP_BOX[0], LOGP_BOX[1] - logp) / s_logp]

    def develop_score(key: str) -> float:
        return softmin(develop_margins(key))

    def develop_success(key: str) -> bool:
        return all(m >= 0.0 for m in develop_margins(key))

    def successors(key: str):
        nonlocal calls
        if key in enumeration:
            return enumeration[key]
        try:
            state = pad_molecular_graph(smiles_to_molecular_graph(key), slots)
        except Exception:  # noqa: BLE001
            enumeration[key] = []
            return []
        calls += 1
        with torch.no_grad():
            result = canonical_successor_result(model, state, float(TIME_POINT))
        enumeration[key] = [(s.key, float(s.probability))
                            for s in result.batch.successors]
        return enumeration[key]

    def greedy_step(key: str, score) -> str | None:
        rows = successors(key)
        if not rows:
            return None
        return max(rows, key=lambda row: (score(row[0]), row[0]))[0]

    def rollout(key: str, remaining: int, score) -> tuple[str, float]:
        """V_G under `score`: deterministic greedy continuation. Returns the
        landing state and its score -- the terminal value, not the best seen,
        because a bounded goal can be entered and then left again."""
        current = key
        for _ in range(remaining):
            nxt = greedy_step(current, score)
            if nxt is None:
                break
            current = nxt
        return current, score(current)

    # --- Phase 1: the A-prefix. Goal B is never consulted here. -------------
    prefix = [start_key]
    current = start_key
    for _ in range(PREFIX_STEPS):
        nxt = greedy_step(current, potency_score)
        if nxt is None:
            break
        current = nxt
        prefix.append(current)
    switch_key = prefix[-1]

    # --- Phase 2: threshold sweep, post hoc from recorded properties --------
    trajectory = [{"key": k, "drd2_logodds": props(k)[0], "qed": props(k)[1],
                   "clogp": props(k)[2]} for k in prefix]
    potency_reach = {}
    for threshold in POTENCY_GRID:
        need = float(np.log(threshold / (1 - threshold)))
        potency_reach[str(threshold)] = {
            "start_satisfies": bool(trajectory[0]["drd2_logodds"] >= need),
            "end_satisfies": bool(trajectory[-1]["drd2_logodds"] >= need),
            "best_logodds": float(max(t["drd2_logodds"] for t in trajectory)),
            "climb_achieved": float(trajectory[-1]["drd2_logodds"]
                                    - trajectory[0]["drd2_logodds"]),
        }

    # --- Phase 3: THE GATE. Contrastive decisions under the bounded goal ----
    rng = random.Random(int(task["index"]) * 7919 + 13)
    states = []
    current = switch_key
    for step in range(POST_STEPS):
        remaining = POST_STEPS - step
        rows = successors(current)
        if not rows:
            break
        keys = [r[0] for r in rows]
        reference = np.array([r[1] for r in rows], dtype=float)
        immediate = np.array([develop_score(k) for k in keys], dtype=float)

        greedy_index = max(range(len(keys)),
                           key=lambda i: (immediate[i], keys[i]))
        picked, seen = [greedy_index], {greedy_index}
        for stratum, count in ((-immediate, CANDIDATES_TOP_IMMEDIATE),
                               (-reference, CANDIDATES_TOP_REFERENCE)):
            for index in np.argsort(stratum, kind="stable")[:count]:
                if int(index) not in seen:
                    picked.append(int(index)); seen.add(int(index))
        rest = [i for i in range(len(keys)) if i not in seen]
        if rest and CANDIDATES_RANDOM > 0:
            for index in rng.sample(rest, min(CANDIDATES_RANDOM, len(rest))):
                picked.append(index); seen.add(index)

        # V_G for every candidate: commit it, then continue greedily.
        futures = {}
        for index in picked:
            _land, value = rollout(keys[index], remaining - 1, develop_score)
            futures[index] = value

        best_index = max(picked, key=lambda i: (futures[i], keys[i]))
        # STRICT IMPROVEMENT, as in the sealed-67 controller: override greedy
        # only when the lookahead is strictly better; ties keep greedy.
        chosen_index = (best_index if futures[best_index] > futures[greedy_index]
                        else greedy_index)
        disagree = best_index != greedy_index
        # A SACRIFICE is an action that looks worse right now. Only those can
        # demonstrate that future value changed the decision for the better.
        sacrificial = disagree and immediate[best_index] < immediate[greedy_index]
        regret = float(futures[best_index] - futures[greedy_index])
        states.append({
            "step": step, "remaining": remaining,
            "candidates": len(picked),
            "top1_disagreement": bool(disagree),
            "sacrificial": bool(sacrificial),
            "sacrifice_won": bool(sacrificial and regret > 0),
            "future_regret": regret,
            "greedy_immediate": float(immediate[greedy_index]),
            "chosen_immediate": float(immediate[best_index]),
            "greedy_future": float(futures[greedy_index]),
            "best_future": float(futures[best_index]),
            "greedy_success": bool(develop_success(keys[greedy_index])),
            "overrode_greedy": bool(chosen_index != greedy_index),
        })
        current = keys[chosen_index]

    # The two arms, now genuinely distinct: `current` followed the lookahead
    # under strict improvement, `greedy_land` is the pure myopic continuation.
    greedy_land, _g = rollout(switch_key, POST_STEPS, develop_score)
    verified_land = current
    payload = {
        "index": int(task["index"]),
        "source": task["source"],
        "switch_state": switch_key,
        "prefix_length": len(prefix) - 1,
        "trajectory": trajectory,
        "potency_reach": potency_reach,
        "develop_start_success": bool(develop_success(start_key)),
        "develop_switch_success": bool(develop_success(switch_key)),
        "develop_greedy_success": bool(develop_success(greedy_land)),
        "develop_verified_success": bool(develop_success(verified_land)),
        "develop_switch_score": float(develop_score(switch_key)),
        "develop_greedy_score": float(develop_score(greedy_land)),
        "develop_verified_score": float(develop_score(verified_land)),
        "decision_states": states,
        "kernel_calls": calls,
        "seconds": round(time.perf_counter() - started, 1),
    }
    out = Path(RUN_ROOT) / task.get("out_dir", "retarget_goal_calibration")
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{task['index']:03d}.json").write_text(json.dumps(payload, indent=2) + "\n")
    artifact_volume.commit()
    print(f"[{task['index']:03d}] {calls} calls  {payload['seconds']:.0f}s  "
          f"disagree={sum(s['top1_disagreement'] for s in states)}/{len(states)}  "
          f"sacrifice={sum(s['sacrificial'] for s in states)}",
          flush=True)
    return payload


@app.function(image=image, cpu=0.25, memory=2048, timeout=8 * 60 * 60,
              volumes={str(ARTIFACT_ROOT): artifact_volume})
def drive(tasks: list[dict[str, Any]]) -> int:
    """Runs the fan-out ON MODAL so a client disconnect cannot stall it."""
    done = 0
    for _ in calibrate_source.map(tasks, order_outputs=False, return_exceptions=True):
        done += 1
    return done


@app.local_entrypoint()
def main(sources: int = 10, smoke: bool = False) -> None:
    """The cohort is read from a COMMITTED artifact, frozen before this ran, so
    the selected set is auditable and identical across reruns. No molecule is
    chosen here."""

    cohort = json.loads(
        (ROOT / "diagnostics/retarget_calibration_cohort.json").read_text())
    chosen = [dict(row) for row in cohort["sources"]][: 2 if smoke else sources]
    for row in chosen:
        row["out_dir"] = ("retarget_goal_calibration_smoke" if smoke
                          else "retarget_goal_calibration_3plus3_fixed")

    print(f"{'SMOKE' if smoke else 'CALIBRATION'}: {len(chosen)} held-in sources "
          f"from cohort {cohort['cohort_sha256'][:16]}")
    print(f"prefix {PREFIX_STEPS} under potency, then {POST_STEPS} under "
          f"QED>={QED_FLOOR} and cLogP in {LOGP_BOX}")
    print("Decides thresholds, horizon and the subclaim-B gate. Runs ONCE.")
    done = drive.remote(chosen)
    print(f"completed {done}")
