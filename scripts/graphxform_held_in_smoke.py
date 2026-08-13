"""ONE held-in smoke of the REAL upstream GraphXForm under all three counters.

Runs under the GraphXForm python-3.11 venv:

    cd /tmp/baseline_envs/graphxform-src && CUDA_VISIBLE_DEVICES="" \\
      PYTHONPATH=<repo>/src:/tmp/baseline_envs/graphxform-src \\
      /tmp/baseline_envs/graphxform-py311/bin/python \\
      <repo>/scripts/graphxform_held_in_smoke.py

Establishes: the upstream checkout is unmodified against its pinned hashes; the
frozen applicability domain partitions the panel and REPORTS what GraphXForm
cannot attempt; the official pretrained model extends supplied held-in sources
via the method's own `start_from_smiles`; and all three counters are wired.

Artifact status: `SMOKE_HELD_IN`. Not a GraphXForm quality measurement and not a
COMPOSE comparison.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from compose_v4.experiments.graphxform_adapter import (  # noqa: E402
    applicable_panel,
    run_graphxform_from_source,
    verify_upstream_unmodified,
)
from compose_v4.experiments.shared_evaluator import SharedEvaluator  # noqa: E402

COHORT = REPO / "diagnostics" / "retarget_calibration_cohort.json"
OUTPUT = REPO / "diagnostics" / "baselines" / "graphxform_held_in_smoke.json"
SOURCE_DIR = Path("/tmp/baseline_envs/graphxform-src")
CHECKPOINT = Path("/tmp/baseline_envs/graphxform-ckpt/weights.pt")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", type=int, default=3)
    parser.add_argument("--budget", type=int, default=200)
    parser.add_argument("--beam-width", type=int, default=8)
    parser.add_argument("--source-dir", type=Path, default=SOURCE_DIR)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()

    upstream = verify_upstream_unmodified(args.source_dir)

    cohort = json.loads(COHORT.read_text())
    if cohort["pool"] != "held-in training sources only":
        raise SystemExit(f"refusing a non-held-in cohort: {cohort['pool']!r}")
    panel = [row["source"] for row in cohort["sources"]]
    partition = applicable_panel(panel)
    chosen = partition["applicable_smiles"][: args.sources]

    started = time.perf_counter()
    runs = []
    # Every molecule crosses the boundary as raw SMILES and is canonicalized and
    # scored once, under the production pin, exactly as every other method.
    with SharedEvaluator(objective="developability") as evaluator:
        for source in chosen:
            result = run_graphxform_from_source(
                source=source,
                objective=evaluator.score,
                source_dir=args.source_dir,
                checkpoint=args.checkpoint,
                budget=args.budget,
                budget_counter="unique_valid_canonical_evaluations",
                beam_width=args.beam_width,
            )
            runs.append(
                {
                    "source": result.source,
                    "counters": result.counters,
                    "invariants": result.accountant_manifest["invariants"],
                    "demand_ratio_requests_over_unique": result.accountant_manifest[
                        "demand_ratio_requests_over_unique"
                    ],
                    "best_score": result.best_score,
                    "best_smiles": result.best_smiles,
                    "n_states": len(result.trajectory_states),
                    "trajectory_states": result.trajectory_states[:6],
                    "wall_seconds": result.wall_seconds,
                    "settings": result.settings,
                    "errors": [e for e in result.endpoints if "error" in e],
                }
            )
        evaluator_identity = evaluator.identity

    ok = [r for r in runs if not r["errors"]]
    checks = {
        "upstream_checkout_is_unmodified": upstream["unmodified"],
        "applicability_domain_applied_and_reported": (
            partition["n_applicable"] + partition["n_inapplicable"]
            == partition["n_total"]
        ),
        "model_ran_and_decided_for_every_source": len(ok) == len(chosen) and bool(chosen),
        "all_three_counters_wired": all(
            r["counters"]["oracle_requests"] > 0
            and r["counters"]["unique_valid_canonical_evaluations"] > 0
            and r["counters"]["evaluator_calls"] > 0
            for r in ok
        ),
        "counter_identity_holds": all(all(r["invariants"].values()) for r in runs),
        "scored_through_the_pinned_evaluator": (
            evaluator_identity.get("rdkit") == "2024.03.5"
        ),
    }

    report = {
        "schema": "compose.baselines.graphxform_held_in_smoke",
        "title": "GRAPHXFORM (real upstream) — HELD-IN ADAPTER SMOKE",
        "artifact_status": "SMOKE_HELD_IN",
        "GREEDY_PATH_ABANDONED": (
            "The hand-rolled greedy extension path in graphxform_adapter._greedy_action is "
            "WITHDRAWN as the route to a comparison. Routing through upstream's own "
            "masked_log_probs_for_current_action_level fixed the infeasible-action selection "
            "but then surfaced a logits/mask width mismatch (54 vs the network's padded "
            "output) on a larger source. Each fix exposing the next mismatch is the signal "
            "that hand-driving per-action decisions is the wrong construction: the adapter "
            "rule says use the method's own machinery, and GraphXForm's own machinery is its "
            "self-improvement fine-tuning loop plus TASAR/beam search, not greedy argmax. "
            "The remaining work is to drive upstream's search rather than to keep repairing "
            "a bespoke decision loop."
        ),
        "KNOWN_ADAPTER_DEFECT": (
            "VERDICT IS FAIL, and the defect is in OUR ADAPTER, not in GraphXForm. On 1 of 3 "
            "sources the run raised 'AssertionError: Trying to take action 1 on level 1, but "
            "it is set to infeasible'. Cause: _greedy_action in graphxform_adapter.py takes "
            "an argmax over raw logits with an ad-hoc mask lookup that does not exist on "
            "MoleculeDesign, so upstream's OWN masking is bypassed and an infeasible action "
            "can be selected at action level 1. The fix is to route the decision through "
            "upstream's masked_log_probs_for_current_action_level rather than a hand-rolled "
            "argmax -- i.e. to use the method's own machinery, which is what the adapter rule "
            "requires anyway. Recorded rather than worked around; the smoke does not pass "
            "until it is fixed."
        ),
        "pretrained_model_terminates_immediately": (
            "MEASURED, and it is CORRECT BEHAVIOUR rather than a defect. Greedy argmax from "
            "the un-fine-tuned pretrained checkpoint selects TERMINATE on the first decision "
            "for every held-in source, because the model is looking at an already-valid "
            "drug-like molecule (P(terminate) ~ 0.99 in an independent probe). GraphXForm's "
            "published protocol reaches extensions through per-objective SELF-IMPROVEMENT "
            "FINE-TUNING plus beam search, not greedy argmax on the pretrained weights. "
            "Masking the terminate action to force extension would ALTER THE PROPOSAL "
            "DISTRIBUTION, which the adapter rule bars, so it was not done. The consequence "
            "is that a meaningful GraphXForm comparison must include its fine-tuning loop, "
            "and the oracle calls that loop consumes must be counted -- which the fairness "
            "contract already requires."
        ),
        "what_this_is_not": (
            "A measurement of GraphXForm's optimization quality, and not a "
            "COMPOSE-versus-GraphXForm comparison. One greedy extension pass at a "
            "smoke-sized budget; the paper's configuration is beam width 512 with "
            "per-objective fine-tuning."
        ),
        "held_out_data_opened": False,
        "upstream": upstream,
        "applicability": {
            k: v for k, v in partition.items() if k != "applicable_smiles"
        },
        "shared_evaluator": evaluator_identity,
        "sources_used": chosen,
        "cohort_sha256": cohort["cohort_sha256"],
        "runs": runs,
        "total_wall_seconds": round(time.perf_counter() - started, 2),
        "environment": {
            "python": platform.python_version(),
            "note": "GraphXForm py3.11 venv; objective scored under pinned rdkit 2024.3.5",
        },
        "checks": checks,
        "verdict": "PASS" if all(checks.values()) else "FAIL",
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"wrote {args.output}")
    print(
        f"applicability: {partition['n_applicable']}/{partition['n_total']} applicable, "
        f"reasons {partition['inapplicable_reasons']}"
    )
    for name, passed in checks.items():
        print(f"  {'PASS' if passed else 'FAIL'}  {name}")
    print(f"verdict: {report['verdict']}")
    return 0 if report["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
