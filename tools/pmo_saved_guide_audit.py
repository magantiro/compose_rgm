"""Compare saved pre-outcome values with realized independent continuations.

Descriptive warm-development evidence only: one suffix per state, not calibrated
reference expectations, confidence intervals, new rollouts, or new oracle calls.
"""

import argparse
import importlib.metadata
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from compose_v4.experiments.continuation_profile import publish_json, sha256_file, verify_file
from compose_v4.experiments.inference_package import software


def analyze(path, receipt_path):
    receipt = json.loads(receipt_path.read_text())
    verify_file(path, receipt["input_paths_sha256"][str(path)])
    r = json.loads(path.read_text())
    if r["run_id"] != receipt["run_id"] or not receipt["particle_decisions_match_exactly"]:
        raise ValueError("saved guide audit requires the verified particle result")
    rounds = [row["arms"]["reference"] for row in r["rounds"]]
    n = r["configuration"]["particles"]
    if len(rounds) != r["configuration"]["boundaries"]:
        raise ValueError("continuation horizon mismatch")
    for index, row in enumerate(rounds):
        if (
            row["resampled"]
            or row["indices"] != list(range(n))
            or any(p is None for p in row["proposals"])
        ):
            raise ValueError("reference trajectories are not intact independent continuations")
        for slot, p in enumerate(row["proposals"]):
            if p["score"] != row["scores"][slot]:
                raise ValueError("saved candidate score mismatch")
            if index:
                previous = rounds[index - 1]["proposals"][slot]
                if p["chain"] != previous["chain"] + [p["id"]]:
                    raise ValueError("reference suffix ancestry changed")
    terminal = np.asarray(rounds[-1]["scores"], dtype=float)
    stages = []
    for index, row in enumerate(rounds[:-1]):
        current = np.asarray(row["scores"], dtype=float)
        guide = np.asarray(row["future_values"], dtype=float)
        observed_max = np.max(np.asarray([a["scores"] for a in rounds[index:]]), axis=0)
        remaining = [
            rounds[-1]["proposals"][i]["primitives"] - p["primitives"]
            for i, p in enumerate(row["proposals"])
        ]
        top_guide, top_current = int(np.argmax(guide)), int(np.argmax(current))
        stages.append(
            {
                "boundary": index + 1,
                "nominal_primitive_budget_supplied_to_head": (len(rounds) - index - 1) * 11,
                "actual_remaining_primitives": remaining,
                "immediate_scores": current.tolist(),
                "guide_predictions": guide.tolist(),
                "observed_suffix_maximum_including_current": observed_max.tolist(),
                "terminal_scores": terminal.tolist(),
                "guide_vs_suffix_spearman": float(spearmanr(guide, observed_max).statistic),
                "immediate_vs_suffix_spearman": float(spearmanr(current, observed_max).statistic),
                "guide_vs_terminal_spearman": float(spearmanr(guide, terminal).statistic),
                "immediate_vs_terminal_spearman": float(spearmanr(current, terminal).statistic),
                "top_guide_slot": top_guide,
                "top_immediate_slot": top_current,
                "observed_suffix_at_top_guide": float(observed_max[top_guide]),
                "observed_suffix_at_top_immediate": float(observed_max[top_current]),
                "guide_mean_minus_observed_suffix_mean": float(np.mean(guide - observed_max)),
            }
        )
    return {
        "schema_version": "saved_guide_audit_v1",
        "run_id": r["run_id"],
        "input_sha256": {str(p): sha256_file(p) for p in (path, receipt_path)},
        "analyzer_sha256": sha256_file(Path(__file__)),
        "analysis_software": {**software(), "scipy": importlib.metadata.version("scipy")},
        "executed_revision": r["image_revision"],
        "configuration": r["configuration"],
        "trajectories": n,
        "initial_distinct_molecules": len({p["smiles"] for p in r["initial_parents"]}),
        "nonterminal_boundaries": len(stages),
        "new_oracle_calls": 0,
        "new_model_calls": 0,
        "rows": stages,
        "claim_boundary": "correlated warm-development observations; one realized suffix per state is not an estimate of optimal reachability or a calibrated reference expectation",
        "next_decision": "do not treat this achieved-route head as qualified foresight; test compound proposal repair, then estimate continuation values under the actual option policy and horizon rather than teacher-route maxima at a primitive upper bound",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result", type=Path)
    parser.add_argument("--verified-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    r = analyze(args.result, args.verified_report)
    publish_json(args.output, r)
    for row in r["rows"]:
        print(json.dumps({k: v for k, v in row.items() if not isinstance(v, list)}))


if __name__ == "__main__":
    main()
