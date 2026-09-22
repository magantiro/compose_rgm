"""Instrument check: does the Test C probe path reproduce a Test B measurement?

The probe and the Test B driver are asserted to share a campaign geometry by a
kwargs-capture test, which is a structural guarantee.  This turns it into an
empirical one: run the TEST C probe from a TEST B teacher checkpoint and
compare the two numbers directly.  If they disagree, every rung Test C names is
measured on a different instrument from the ladder it is read against.

This is the only place a probe is allowed to start from an atlas molecule, and
it is labelled a control rather than an entry measurement.  Its calls are
counted and reported separately from the Test C budget.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path
from typing import Any

import pmo_atlas_entry_probe as probe

from compose_v4.experiments.pmo_atlas_discovery import TrajectoryRow
from compose_v4.experiments.pmo_atlas_objectives import (
    ReferenceExpectation,
    assert_reference_panel,
    build_oracle,
)
from compose_v4.experiments.pmo_atlas_routes import (
    DEVELOPMENT_INFORMED_LABEL,
    REGIME_STATEMENT,
    assert_not_scored_consumable,
    file_sha256,
    load_atlas,
    payload_sha256,
    route_checkpoints,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--tasks", nargs="+", required=True)
    parser.add_argument("--position", default="anchor")
    parser.add_argument("--probe-budget", type=int, default=64)
    parser.add_argument("--queries-per-round", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260921)
    parser.add_argument("--work-dir", default=None)
    parser.add_argument("--output", default="diagnostics/pmo_atlas_v1/test_c_probe_control.json")
    parser.add_argument("--test-b", default="diagnostics/pmo_atlas_v1/test_b_local_lift.json")
    parser.add_argument("--reference-panel", default="diagnostics/pmo_atlas_v1/reference_panel.json")
    args = parser.parse_args(argv)

    assert_not_scored_consumable("atlas_probe_instrument_control")
    repo_root = Path(args.repo_root).resolve()
    work = (
        Path(args.work_dir).expanduser()
        if args.work_dir
        else Path.home() / "compose_pmo_atlas_runs" / "test_c_probe_control"
    )
    work.mkdir(parents=True, exist_ok=True)

    dossier = load_atlas(repo_root)
    spines = {route.task: route for route in dossier.spines()}
    test_b = json.loads((repo_root / args.test_b).read_text())["payload"]
    panel_payload = json.loads((repo_root / args.reference_panel).read_text())["payload"]

    started = time.time()
    records: list[dict[str, Any]] = []
    for task_name in args.tasks:
        route = spines[task_name]
        positions = tuple(
            (label, fraction)
            for label, fraction in (("early", 0.25), ("near_anchor", 0.85), ("anchor", 1.0))
        )
        checkpoint = next(
            item
            for item in route_checkpoints(route, positions=positions)
            if item.label == args.position
        )
        oracle = build_oracle(task_name)
        assert_reference_panel(
            oracle,
            tuple(
                ReferenceExpectation(item["smiles"], item["value"], item["label"])
                for item in panel_payload["panels"][task_name]
            ),
        )
        row = TrajectoryRow(
            index=0,
            endpoint=checkpoint.smiles,
            score=0.0,
            role="teacher_checkpoint_control",
            state=checkpoint.state,
        )
        record = probe._probe(
            repo_root,
            work / f"{task_name}__{args.position}",
            task_name,
            row,
            [f"teacher_{args.position}_reproduction_control"],
            oracle,
            args.probe_budget,
            args.queries_per_round,
            args.seed,
        )
        reference = next(
            item
            for item in test_b["runs"]
            if item["task"] == task_name and item["checkpoint_label"] == args.position
        )
        record["test_b_seed_smiles"] = reference["seed_smiles"]
        record["seed_smiles_matches_test_b"] = reference["seed_smiles"] == checkpoint.smiles
        record["test_b_top_ten_new_mean"] = reference["top_ten_new_mean"]
        record["delta_vs_test_b"] = (
            None
            if record["top_ten_new_mean"] is None
            else round(record["top_ten_new_mean"] - reference["top_ten_new_mean"], 6)
        )
        records.append(record)
        print(
            f"{task_name:26s} {args.position:12s} testC={record['top_ten_new_mean']:.4f} "
            f"testB={reference['top_ten_new_mean']:.4f} delta={record['delta_vs_test_b']:+.6f} "
            f"same_seed={record['seed_smiles_matches_test_b']}",
            flush=True,
        )

    payload = {
        "schema_version": "pmo_atlas_test_c_probe_control_v1",
        "information_regime": DEVELOPMENT_INFORMED_LABEL,
        "information_regime_statement": REGIME_STATEMENT,
        "question": (
            "Run from a Test B teacher checkpoint, does the Test C probe path reproduce "
            "the Test B measurement? A structural kwargs guard says it must; this says "
            "whether it does."
        ),
        "role": "instrument reproduction control, not an entry measurement",
        "position": args.position,
        "diagnostic_oracle_calls": sum(item["charged_calls"] for item in records),
        "test_b_artifact": {
            "path": args.test_b,
            "file_sha256": file_sha256(repo_root / args.test_b),
        },
        "software": {
            "python": platform.python_version(),
            "rdkit": __import__("rdkit").__version__,
            "numpy": __import__("numpy").__version__,
            "pytdc": __import__("importlib.metadata", fromlist=["version"]).version("PyTDC"),
        },
        "elapsed_seconds": round(time.time() - started, 2),
        "controls": records,
    }
    document = {"payload": payload, "payload_sha256": payload_sha256(payload)}
    output = repo_root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(document, indent=1, sort_keys=True))
    print(f"\ndiagnostic oracle calls: {payload['diagnostic_oracle_calls']}")
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
