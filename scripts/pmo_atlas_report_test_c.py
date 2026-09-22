"""TEST C reduction: entry, exploitation, and whether structure predicted either.

Three questions are kept apart because they have different answers and
different evidence.

ENTRY     Did the blind trajectory contain a molecule from which the frozen
          local controller is as productive as it is from a teacher region?
          MEASURED by the 64-call probes, named against the Test B ladder.
EXPLOIT   Did the blind run itself realise that productivity?  Its own top ten
          over its whole budget is compared against the probe's top ten from
          its own best molecule.  A probe that beats the run it came from means
          the region was visited and not worked, which is an allocation
          finding, not a discovery finding.
STRUCTURE Did proximity to the teacher routes predict productivity?  Reported
          as a correlation across probes, never used to decide entry.

The predicted column is INFERRED from the sealed Test B regression and is
printed beside the measurement so an over- or under-shoot is visible rather
than absorbed.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

from compose_v4.experiments.pmo_atlas_routes import (
    DEVELOPMENT_INFORMED_LABEL,
    REGIME_STATEMENT,
    file_sha256,
    payload_sha256,
)


def _load(repo_root: Path, relative: str) -> dict[str, Any]:
    document = json.loads((repo_root / relative).read_text())
    return document.get("payload", document)


def _pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 3:
        return None
    mx, my = statistics.fmean(xs), statistics.fmean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=True))
    dx = sum((x - mx) ** 2 for x in xs) ** 0.5
    dy = sum((y - my) ** 2 for y in ys) ** 0.5
    return None if dx == 0 or dy == 0 else num / (dx * dy)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--blind", nargs="+", required=True)
    parser.add_argument("--probe", default="diagnostics/pmo_atlas_v1/test_c_entry.json")
    parser.add_argument("--prediction", default="diagnostics/pmo_atlas_v1/test_c_prediction.json")
    parser.add_argument("--output", default="diagnostics/pmo_atlas_v1/test_c_report.json")
    args = parser.parse_args(argv)

    repo_root = Path(args.repo_root).resolve()
    prediction = _load(repo_root, args.prediction)
    slope = prediction["predictor"]["slope"]
    intercept = prediction["predictor"]["intercept"]

    blind: dict[str, dict[str, Any]] = {}
    blind_calls = 0
    for relative in args.blind:
        payload = _load(repo_root, relative)
        blind_calls += payload["diagnostic_oracle_calls"]
        for run in payload["runs"]:
            blind[run["task"]] = run

    probes = _load(repo_root, args.probe)
    approaches = {row["task"]: row for row in probes["structural_approach"]}

    rows: list[dict[str, Any]] = []
    for probe in probes["probes"]:
        task = probe["task"]
        run = blind.get(task, {})
        measured = probe["top_ten_new_mean"]
        predicted = slope * (probe["seed_score"] or 0.0) + intercept
        rows.append(
            {
                "task": task,
                "selectors": probe["selectors"],
                "seed_charged_call": probe["seed_charged_call"],
                "seed_score": probe["seed_score"],
                "measured_top_ten_new_mean": measured,
                "predicted_top_ten_new_mean": round(predicted, 4),
                "prediction_residual": round((measured or 0.0) - predicted, 4),
                "rung_reached": probe["rung"]["rung_reached"],
                "fraction_of_anchor": probe["rung"]["fraction_of_anchor"],
                "anchor_productivity": probe["rung"]["anchor_productivity"],
                "similarity_to_atlas": probe["similarity_to_atlas"],
                "blind_run_top_ten_mean": run.get("top_ten_mean"),
                "blind_run_best_score": run.get("best_score"),
                "probe_beats_the_run_it_came_from": (
                    None
                    if measured is None or run.get("top_ten_mean") is None
                    else round(measured - run["top_ten_mean"], 4)
                ),
                "route_length": len(probe.get("route_to_seed", ())),
                "seed_planner_channel": probe.get("seed_planner_channel"),
            }
        )

    best_per_task: dict[str, dict[str, Any]] = {}
    for row in rows:
        if "blind_best_score" in row["selectors"]:
            best_per_task[row["task"]] = row

    reached = {}
    for task, row in best_per_task.items():
        reached[task] = row["rung_reached"]

    similarity = [row["similarity_to_atlas"] for row in rows if row["measured_top_ten_new_mean"]]
    productivity = [row["measured_top_ten_new_mean"] for row in rows if row["measured_top_ten_new_mean"]]

    payload = {
        "schema_version": "pmo_atlas_test_c_report_v1",
        "information_regime": DEVELOPMENT_INFORMED_LABEL,
        "information_regime_statement": REGIME_STATEMENT,
        "inputs": {
            "blind": [
                {"artifact": relative, "file_sha256": file_sha256(repo_root / relative)}
                for relative in args.blind
            ],
            "probe": {"artifact": args.probe, "file_sha256": file_sha256(repo_root / args.probe)},
            "prediction": {
                "artifact": args.prediction,
                "file_sha256": file_sha256(repo_root / args.prediction),
            },
        },
        "diagnostic_oracle_calls": {
            "blind_search": blind_calls,
            "entry_probes": probes["diagnostic_oracle_calls"],
            "total": blind_calls + probes["diagnostic_oracle_calls"],
        },
        "entry": {
            "rung_reached_from_the_blind_best_molecule": reached,
            "tasks_reaching_anchor": sorted(t for t, r in reached.items() if r == "anchor"),
            "tasks_reaching_near_anchor": sorted(
                t for t, r in reached.items() if r == "near_anchor"
            ),
            "tasks_reaching_early": sorted(t for t, r in reached.items() if r == "early"),
            "tasks_below_the_ladder": sorted(t for t, r in reached.items() if r.startswith("below")),
        },
        "exploitation": {
            "definition": (
                "probe top_ten_new_mean from the blind run's own best molecule, minus the "
                "blind run's own top ten over its whole budget"
            ),
            "per_task": {
                task: row["probe_beats_the_run_it_came_from"]
                for task, row in sorted(best_per_task.items())
            },
        },
        "structure_versus_productivity": {
            "pearson_r": (
                None
                if _pearson(similarity, productivity) is None
                else round(_pearson(similarity, productivity), 4)
            ),
            "n": len(similarity),
            "note": (
                "Structural proximity selects what to verify and never decides. A weak "
                "correlation here is the expected outcome, not a defect."
            ),
        },
        "nearest_approach": {
            task: {
                "max_similarity": row["max_similarity"],
                "at_charged_call": row["at_charged_call"],
                "nearest_blind_molecule": row["nearest_blind_molecule"],
            }
            for task, row in sorted(approaches.items())
        },
        "rows": rows,
    }
    document = {"payload": payload, "payload_sha256": payload_sha256(payload)}
    output = repo_root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(document, indent=1, sort_keys=True))

    header = (
        f"{'task':26s} {'selector':22s} {'seed':>6s} {'meas':>6s} {'pred':>6s} "
        f"{'rung':12s} {'frac':>5s} {'sim':>5s} {'run10':>6s} {'probe-run':>9s}"
    )
    print(header)
    for row in rows:
        print(
            f"{row['task']:26s} {row['selectors'][0]:22s} "
            f"{row['seed_score'] or 0:6.3f} {row['measured_top_ten_new_mean'] or 0:6.3f} "
            f"{row['predicted_top_ten_new_mean']:6.3f} {row['rung_reached']:12s} "
            f"{row['fraction_of_anchor'] or 0:5.2f} {row['similarity_to_atlas']:5.2f} "
            f"{row['blind_run_top_ten_mean'] or 0:6.3f} "
            f"{row['probe_beats_the_run_it_came_from'] or 0:+9.3f}"
        )
    print(f"\ndiagnostic oracle calls: {payload['diagnostic_oracle_calls']}")
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
