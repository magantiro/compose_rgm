"""Reconcile the matched, fresh-seed initial-family development pilot.

This is a small zero-oracle mechanism test, not an official benchmark row or a
selection gate. All 20 attempt records per prompt and arm must be present.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import rdkit
from fetch_official_fragment_evaluator import OFFICIAL_BLOBS, verify_only

ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT / "diagnostics/fragment_initial_family_dev_v1"
ARMS = ("baseline", "conditioned", "conditioned_strict")
DEV_DRUGS = ("BARICITINIB", "LOVASTATIN", "SPIRAPRIL")
ALL_DRUGS = (
    "BARICITINIB",
    "CYCLOTHIAZIDE",
    "ELIGLUSTAT",
    "ERLOTINIB",
    "FUTIBATINIB",
    "LESINURAD",
    "LIOTHYRONINE",
    "LOVASTATIN",
    "MARIBAVIR",
    "SPIRAPRIL",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, choices=(9, 10), default=9)
    args = parser.parse_args()
    drugs = DEV_DRUGS if args.seed == 9 else ALL_DRUGS
    suffix = f"seed{args.seed}_n20" + ("_all10" if args.seed == 10 else "")
    verified = verify_only()
    expected_evaluator = OFFICIAL_BLOBS["in_virtuo_gen/train_utils/metrics.py"][1]
    if expected_evaluator not in verified.values():
        raise RuntimeError("official evaluator hash not verified")
    sources = {}
    rows = []
    for arm in ARMS:
        path = FOLDER / f"{arm}_{suffix}.json"
        sources[str(path.relative_to(ROOT))] = sha256(path)
        payload = json.loads(path.read_text())
        control = payload["attachment_control"]["config"]
        if bool(control.get("condition_initial_locked_family", False)) != (arm != "baseline"):
            raise RuntimeError(f"initial-family arm wiring mismatch: {path}")
        if bool(control.get("hard_lock_effective_chemistry", False)) != (
            arm == "conditioned_strict"
        ):
            raise RuntimeError(f"strict-lock arm wiring mismatch: {path}")
        result = payload["results"]["superstructure_generation"]
        if result["build_failures"]:
            raise RuntimeError(f"prompt build failure: {path}")
        if set(result["per_drug"]) != set(drugs):
            raise RuntimeError(f"prompt census mismatch: {path}")
        for drug in drugs:
            [source] = result["per_drug"][drug]
            attempts = source["attempt_records"]
            if len(attempts) != 20 or source["attempts"] != 20:
                raise RuntimeError(f"attempts missing: {arm}/{drug}")
            committed = [a for a in attempts if a["committed_smiles"]]
            no_output = [a for a in attempts if not a["committed_smiles"]]
            zero_event = [a for a in no_output if a["events"] == 0]
            first_budget_exhausted = [
                a for a in zero_event if a["refusal_deltas"]["budget_exhausted"] > 0
            ]
            if len(committed) != source["committed_endpoints"]:
                raise RuntimeError(f"commit mismatch: {arm}/{drug}")
            if abs(source["official"]["validity"] - 5.0 * source["emitted_nonempty"]) > 1e-10:
                raise RuntimeError(f"official denominator mismatch: {arm}/{drug}")
            if source["committed_chemically_valid"] != len(committed):
                raise RuntimeError(f"invalid committed endpoint: {arm}/{drug}")
            rows.append(
                {
                    "arm": arm,
                    "drug": drug,
                    "attempts": 20,
                    "committed": len(committed),
                    "chemically_valid_committed": source["committed_chemically_valid"],
                    "fragment_preserving_committed": source["committed_fragment_preserving"],
                    "official_emitted": source["emitted_nonempty"],
                    "official_validity_pct": source["official"]["validity"],
                    "uniqueness_pct": source["official"]["uniqueness"],
                    "quality_pct": source["official"]["quality"],
                    "diversity": source["official"]["diversity"],
                    "no_output": len(no_output),
                    "no_output_zero_event": len(zero_event),
                    "no_output_zero_event_budget_exhausted": len(first_budget_exhausted),
                    "lock_rejections": source["lock_rejections"],
                    "conditioned_draws": source["initial_family_conditioned_draws"],
                    "conditioned_accepts": source["initial_family_conditioned_accepts"],
                    "seconds": source["seconds"],
                }
            )
    summary = {}
    for arm in ARMS:
        selected = [row for row in rows if row["arm"] == arm]
        keys = (
            "attempts",
            "committed",
            "chemically_valid_committed",
            "fragment_preserving_committed",
            "official_emitted",
            "no_output",
            "no_output_zero_event",
            "no_output_zero_event_budget_exhausted",
            "lock_rejections",
            "conditioned_draws",
            "conditioned_accepts",
        )
        summary[arm] = {key: sum(row[key] for row in selected) for key in keys}
        summary[arm]["official_validity_pct"] = (
            100.0 * summary[arm]["official_emitted"] / summary[arm]["attempts"]
        )
        summary[arm]["official_quality_pct_mean_of_prompts"] = sum(
            row["quality_pct"] for row in selected
        ) / len(selected)
        summary[arm]["official_uniqueness_pct_mean_of_prompts"] = sum(
            row["uniqueness_pct"] for row in selected
        ) / len(selected)
        summary[arm]["official_diversity_mean_of_prompts"] = sum(
            row["diversity"] for row in selected
        ) / len(selected)
        summary[arm]["seconds"] = sum(row["seconds"] for row in selected)
    result = {
        "schema": "fragment_initial_family_dev_audit_v1",
        "evidence_role": (
            "Matched all-ten development qualification; not a frozen independent "
            "official benchmark or promotion decision"
            if args.seed == 10
            else "Matched fresh-seed development pilot; too small for promotion or published comparison"
        ),
        "seed": args.seed,
        "drugs": list(drugs),
        "input_sha256": sources,
        "checkpoint_sha256": sha256(
            Path("/Users/rmaganti/compose_fragment_ckpt/ringcore_a7546e2_best.pt")
        ),
        "manifest_sha256": sha256(
            ROOT / "data/benchmarks/fragment_constrained/genmol_safe_drugs_fragments.csv"
        ),
        "official_evaluator_sha256": expected_evaluator,
        "implementation_sha256": {
            str(path.relative_to(ROOT)): sha256(path)
            for path in (
                ROOT / "src/compose_v4/model/factorized_tracelet_rate_model.py",
                ROOT / "src/compose_v4/benchmark/fragment_conditioned_sampler.py",
                ROOT / "src/compose_v4/benchmark/fragment_attachment_control.py",
                ROOT / "tools/run_fragment_constrained_suite.py",
            )
        },
        "auditor_sha256": sha256(Path(__file__)),
        "git_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "software": {"python": sys.version.split()[0], "rdkit": rdkit.__version__},
        "rows": rows,
        "summary": summary,
    }
    output = FOLDER / f"audit_{suffix}.json"
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=FOLDER, suffix=".tmp", delete=False
    ) as handle:
        temporary = Path(handle.name)
        json.dump(result, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    os.replace(temporary, output)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
