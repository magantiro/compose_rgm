"""TEST C pre-registration: what productivity a blind seed of a given score buys.

Sealed BEFORE any blind result is read.  Its only input is Test B, which was
complete before this test began, so nothing here can be tuned to the answer.

Why a prediction at all: a null ("blind never entered a productive region") is
worthless from an under-powered probe.  Test B measured productivity at 33
seeds spanning the whole score range, and ``top_ten_new_mean`` is 87% explained
by the seed's own score.  That regression therefore says, per task and per
rung, WHAT SCORE a blind molecule must reach before the probe could possibly
report entry -- which turns "did it enter" into a question with a computable
expectation instead of a hope.

The regression is a POWER instrument, not the measurement.  Its residual is
large where it matters most (isomers_c7h8n2o2 near_anchor sits +0.40 above its
prediction), which is exactly why every reported rung is measured with the
objective and never predicted.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from compose_v4.experiments.pmo_atlas_routes import (
    DEVELOPMENT_INFORMED_LABEL,
    REGIME_STATEMENT,
    file_sha256,
    payload_sha256,
)


def fit(runs: list[dict[str, Any]]) -> dict[str, Any]:
    seeds = np.array([row["seed_score"] for row in runs], dtype=float)
    measured = np.array([row["top_ten_new_mean"] for row in runs], dtype=float)
    design = np.vstack([seeds, np.ones_like(seeds)]).T
    (slope, intercept), *_ = np.linalg.lstsq(design, measured, rcond=None)
    predicted = design @ np.array([slope, intercept])
    residual = measured - predicted
    return {
        "n": int(seeds.size),
        "slope": float(slope),
        "intercept": float(intercept),
        "r_squared": float(1.0 - residual.var() / measured.var()),
        "residual_sd": float(residual.std(ddof=2)),
        "max_abs_residual": float(np.abs(residual).max()),
        "largest_residual_run": max(
            ({"task": row["task"], "position": row["checkpoint_label"],
              "seed_score": row["seed_score"], "observed": row["top_ten_new_mean"],
              "predicted": float(value)}
             for row, value in zip(runs, predicted, strict=True)),
            key=lambda item: abs(item["observed"] - item["predicted"]),
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--test-b", default="diagnostics/pmo_atlas_v1/test_b_local_lift.json")
    parser.add_argument("--output", default="diagnostics/pmo_atlas_v1/test_c_prediction.json")
    args = parser.parse_args(argv)

    repo_root = Path(args.repo_root).resolve()
    test_b = json.loads((repo_root / args.test_b).read_text())["payload"]
    model = fit(test_b["runs"])

    per_task: dict[str, Any] = {}
    for run in test_b["runs"]:
        rung = {
            "position": run["checkpoint_label"],
            "teacher_seed_score": run["seed_score"],
            "teacher_top_ten_new_mean": run["top_ten_new_mean"],
            "blind_seed_score_required": round(
                (run["top_ten_new_mean"] - model["intercept"]) / model["slope"], 4
            ),
        }
        per_task.setdefault(run["task"], []).append(rung)
    for rungs in per_task.values():
        rungs.sort(key=lambda item: item["teacher_top_ten_new_mean"])

    payload = {
        "schema_version": "pmo_atlas_test_c_prediction_v1",
        "information_regime": DEVELOPMENT_INFORMED_LABEL,
        "information_regime_statement": REGIME_STATEMENT,
        "sealed_before": "any Test C blind result was read",
        "source": {"artifact": args.test_b, "file_sha256": file_sha256(repo_root / args.test_b)},
        "predictor": {
            "form": "top_ten_new_mean = slope * seed_score + intercept",
            "fitted_on": "the 33 Test B teacher-region runs, all complete before Test C began",
            "role": "power instrument; every reported rung is MEASURED, never predicted",
            **model,
        },
        "decision_rule": (
            "A blind molecule ENTERS a productive region when the frozen local controller, "
            "started there with 64 calls, reaches a top_ten_new_mean at or above a Test B "
            "rung for the same task. The rung named is the highest one cleared; a "
            "measurement below the lowest rung is reported as below_early, never rounded up."
        ),
        "falsifier": (
            "If blind search reaches the ANCHOR rung on tasks where Test A showed the "
            "destination is constructible, then discovery is not the gap and the remaining "
            "explanation is exploitation or allocation, both of which Test B already "
            "measured as adequate. If instead blind search tops out near the EARLY rung on "
            "the rediscovery tasks while clearing ANCHOR on the broad-objective tasks, the "
            "gap is task-shaped: narrow targets are not found, broad objectives are."
        ),
        "blind_seed_score_required_per_rung": per_task,
    }
    document = {"payload": payload, "payload_sha256": payload_sha256(payload)}
    output = repo_root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(document, indent=1, sort_keys=True))
    print(
        f"predictor: top10new = {model['slope']:.4f}*seed + {model['intercept']:.4f} "
        f"(n={model['n']}, R^2={model['r_squared']:.4f}, resid sd={model['residual_sd']:.4f})"
    )
    for task, rungs in sorted(per_task.items()):
        need = ", ".join(
            f"{r['position']}>={r['blind_seed_score_required']:.3f}" for r in rungs
        )
        print(f"  {task:26s} {need}")
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
