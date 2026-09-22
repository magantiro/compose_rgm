"""TEST C, arm 2: the atlas-aware EVALUATOR that watches a blind trajectory.

The optimizer has already run blind.  This driver never touches it.  It asks
two separate questions about what it produced, and keeps them separate because
they have different evidential standing.

1. STRUCTURAL, cheap, diagnostic only: how close did the blind trajectory get
   to the teacher routes, and at which charged call?  Reported as a running
   maximum so "how close and when" is one monotone curve.
2. PRODUCTIVE, measured with the objective: started at a blind molecule, does
   the SAME frozen local controller with the SAME 64-call budget fill the same
   quality of top ten that Test B measured from the teacher spine?  That
   comparison is the operational definition of entering a productive region,
   and it is the only thing allowed to decide the question.

Selectors are pre-declared so the probed set cannot be chosen after seeing the
answer: the blind run's best-scoring molecule, its structurally nearest
approach to the atlas, and its MEDIAN-scoring molecule as a floor control.  If
a typical blind molecule scored as productive, the ladder would be measuring
the controller rather than the region.

DEVELOPMENT_INFORMED_DIAGNOSTIC.  Every probe call is counted.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from pathlib import Path
from typing import Any

from compose_v4.control.docking_value import identity
from compose_v4.control.program_campaign import ProgramQueryLedger, run_program_campaign
from compose_v4.control.program_task import ProgramTask
from compose_v4.experiments.pmo_atlas_discovery import (
    TrajectoryRow,
    ancestry,
    atlas_molecules,
    classify_rung,
    load_ladder,
    local_lift_kwargs,
    nearest_approach,
    read_blind_trajectory,
    single_seed_initialization,
    summarize_lift,
)
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
)
from compose_v4.experiments.pmo_population_v1 import configuration

CHECKPOINTS = "diagnostics/pmo_joint_dependency_jump_gate_v1/attempt_2/checkpoints.json"

#: Pre-declared, in this order.  ``blind_median_score`` is the floor control.
SELECTORS: tuple[str, ...] = (
    "blind_best_score",
    "blind_nearest_atlas",
    "blind_median_score",
)


class ProbeSeedError(RuntimeError):
    """Raised when a probe seed did not come from the blind trajectory."""


def _jump_checkpoint(repo_root: Path) -> dict[str, Any]:
    envelope = json.loads((repo_root / CHECKPOINTS).read_text())
    payload = envelope["payload"]["checkpoints"]["shared_all_routes"]
    if payload.get("fit_scope") != "shared_all_routes":
        raise ValueError("the PMO runtime checkpoint must be shared and task blind")
    return payload


def assert_probe_seed_is_blind(
    row: TrajectoryRow,
    trajectory: tuple[TrajectoryRow, ...],
    atlas: frozenset[str],
) -> None:
    """A probe seed must be a molecule the BLIND run actually scored.

    Seeding a probe from the atlas would re-measure Test B while reporting it as
    Test C, so both halves are checked: the molecule is in the trajectory, and
    it is not an atlas molecule.
    """

    if row.endpoint not in {item.endpoint for item in trajectory}:
        raise ProbeSeedError(f"probe seed is not in the blind trajectory: {row.endpoint}")
    if row.endpoint in atlas:
        raise ProbeSeedError(f"probe seed is an answer-known atlas molecule: {row.endpoint}")
    if row.state is None:
        raise ProbeSeedError(
            f"probe seed carries no 48-slot state: {row.endpoint}; re-parsing its SMILES "
            "would hand the PMO path a tight graph it refuses"
        )


def _select(
    trajectory: tuple[TrajectoryRow, ...],
    similarity: list[float],
) -> dict[str, TrajectoryRow]:
    usable = [
        (row, value)
        for row, value in zip(trajectory, similarity, strict=True)
        if row.state is not None
    ]
    if not usable:
        return {}
    rows = [row for row, _ in usable]
    chosen: dict[str, TrajectoryRow] = {}
    chosen["blind_best_score"] = max(rows, key=lambda row: (row.score, -row.index))
    chosen["blind_nearest_atlas"] = max(usable, key=lambda pair: (pair[1], -pair[0].index))[0]
    ordered = sorted(rows, key=lambda row: (row.score, row.index))
    chosen["blind_median_score"] = ordered[len(ordered) // 2]
    return chosen


def _probe(
    repo_root: Path,
    folder: Path,
    task_name: str,
    row: TrajectoryRow,
    labels: list[str],
    oracle,
    budget: int,
    queries_per_round: int,
    seed: int,
    already_seen: frozenset[str] = frozenset(),
) -> dict[str, Any]:
    protocol = identity(
        {
            "role": "pmo_atlas_blind_entry_probe_diagnostic",
            "task": task_name,
            "seed_smiles": row.endpoint,
            "seed_charged_call": row.index,
            "information_regime": DEVELOPMENT_INFORMED_LABEL,
        }
    )
    task = ProgramTask(task_name, protocol, "pmo")
    ledger = ProgramQueryLedger(folder / "oracle", task, oracle, budget=budget)
    rounds = max(1, -(-budget // queries_per_round))
    began = time.time()
    run_program_campaign(
        output=folder / "campaign",
        task=task,
        ledger=ledger,
        initialization=single_seed_initialization(
            row.endpoint, row.state, f"blind_call_{row.index}"
        ),
        **local_lift_kwargs(
            repo_root=repo_root,
            config=configuration(seed),
            rounds=rounds,
            queries_per_round=queries_per_round,
            optimizer_kwargs={
                "jump_checkpoint": _jump_checkpoint(repo_root),
                "enable_online_memory": True,
            },
        ),
    )
    elapsed = time.time() - began
    rows = list(ledger.rows)
    summary = summarize_lift(rows, row.endpoint)
    # A probe re-charges molecules the blind run may already have scored. That is
    # correct for measuring what the region YIELDS, and it would overstate
    # NOVELTY, so the overlap is reported rather than left to be assumed away.
    produced = {record["endpoint"] for record in rows} - {row.endpoint}
    return {
        "already_scored_by_the_blind_run": len(produced & already_seen),
        "produced_distinct": len(produced),
        "task": task_name,
        "selectors": labels,
        "seed_smiles": row.endpoint,
        "seed_charged_call": row.index,
        "seed_role": row.role,
        "seed_heavy_atoms": row.heavy_atoms,
        "seed_planner_channel": row.planner_channel,
        "seed_parent_endpoint": row.parent_endpoint,
        "seed_parent_score": row.parent_score,
        "budget": budget,
        "seconds": round(elapsed, 2),
        **summary,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--tasks", nargs="*", default=None)
    parser.add_argument("--blind-budget", type=int, default=250)
    parser.add_argument("--probe-budget", type=int, default=64)
    parser.add_argument("--queries-per-round", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260921)
    parser.add_argument("--run-root", default=None)
    parser.add_argument("--work-dir", default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--test-b", default="diagnostics/pmo_atlas_v1/test_b_local_lift.json")
    parser.add_argument("--reference-panel", default="diagnostics/pmo_atlas_v1/reference_panel.json")
    args = parser.parse_args(argv)

    assert_not_scored_consumable("atlas_blind_entry_probe_diagnostic")
    repo_root = Path(args.repo_root).resolve()
    run_root = (
        Path(args.run_root).expanduser()
        if args.run_root
        else Path.home() / "compose_pmo_atlas_runs" / "test_c_blind"
    )
    work = (
        Path(args.work_dir).expanduser()
        if args.work_dir
        else Path.home() / "compose_pmo_atlas_runs" / "test_c_probe"
    )
    work.mkdir(parents=True, exist_ok=True)

    dossier = load_atlas(repo_root)
    ladder = load_ladder(json.loads((repo_root / args.test_b).read_text())["payload"])
    tasks = args.tasks or list(dossier.tasks)

    panel_document = json.loads((repo_root / args.reference_panel).read_text())
    panel_payload = panel_document.get("payload", panel_document)

    started = time.time()
    approaches: list[dict[str, Any]] = []
    probes: list[dict[str, Any]] = []
    controls: dict[str, Any] = {}

    def _payload(partial: bool) -> dict[str, Any]:
        return {
            "schema_version": "pmo_atlas_test_c_entry_probe_v1",
            "partial": partial,
            "information_regime": DEVELOPMENT_INFORMED_LABEL,
            "information_regime_statement": REGIME_STATEMENT,
            "question": (
                "Does a blind trajectory ever contain a molecule from which the frozen "
                "local controller is as productive as it is from a teacher region?"
            ),
            "productive_region_definition": (
                "A region is productive when the frozen local controller, started there "
                "with a 64-call budget, produces a top ten of DISTINCT new molecules at "
                "the level Test B measured from the teacher spine. Structural proximity "
                "selects what to verify and never decides."
            ),
            "selectors": list(SELECTORS),
            "blind_budget": args.blind_budget,
            "probe_budget": args.probe_budget,
            "seed": args.seed,
            "tasks_probed": tasks,
            "diagnostic_oracle_calls": sum(probe["charged_calls"] for probe in probes),
            "test_b_artifact": {
                "path": args.test_b,
                "file_sha256": file_sha256(repo_root / args.test_b),
            },
            "reference_panel": {
                "artifact": args.reference_panel,
                "file_sha256": file_sha256(repo_root / args.reference_panel),
                "controls": controls,
            },
            "software": {
                "python": platform.python_version(),
                "rdkit": __import__("rdkit").__version__,
                "numpy": __import__("numpy").__version__,
                "pytdc": __import__("importlib.metadata", fromlist=["version"]).version("PyTDC"),
            },
            "elapsed_seconds": round(time.time() - started, 2),
            "structural_approach": approaches,
            "probes": probes,
        }

    def _write(partial: bool) -> dict[str, Any]:
        payload = _payload(partial)
        document = {"payload": payload, "payload_sha256": payload_sha256(payload)}
        output = repo_root / args.output
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(document, indent=1, sort_keys=True))
        return payload

    for task_name in tasks:
        run_dir = run_root / f"{task_name}__blind_{args.blind_budget}"
        if not (run_dir / "oracle").is_dir():
            print(f"{task_name:26s} SKIP: no blind run at {run_dir}", flush=True)
            continue
        trajectory = read_blind_trajectory(run_dir)
        atlas = atlas_molecules(dossier, task=task_name)
        approach = nearest_approach(trajectory, sorted(atlas))
        similarity = approach.pop("per_row_similarity")
        scores = [row.score for row in trajectory]
        approaches.append(
            {
                "task": task_name,
                "charged_calls": len(trajectory),
                "blind_best_score": max(scores) if scores else None,
                "blind_top_ten_mean": (
                    sum(sorted(scores, reverse=True)[:10]) / min(10, len(scores))
                    if scores
                    else None
                ),
                "blind_median_score": statistics.median(scores) if scores else None,
                **approach,
            }
        )

        oracle = build_oracle(task_name)
        panel = tuple(
            ReferenceExpectation(item["smiles"], item["value"], item["label"])
            for item in panel_payload["panels"][task_name]
        )
        controls[task_name] = assert_reference_panel(oracle, panel)

        chosen = _select(trajectory, similarity)
        by_molecule: dict[str, list[str]] = {}
        for label in SELECTORS:
            row = chosen.get(label)
            if row is not None:
                by_molecule.setdefault(row.endpoint, []).append(label)
        rows_by_endpoint = {row.endpoint: row for row in chosen.values()}
        for endpoint, labels in by_molecule.items():
            row = rows_by_endpoint[endpoint]
            assert_probe_seed_is_blind(row, trajectory, atlas)
            folder = work / f"{task_name}__call{row.index:04d}"
            record = _probe(
                repo_root,
                folder,
                task_name,
                row,
                labels,
                oracle,
                args.probe_budget,
                args.queries_per_round,
                args.seed,
                frozenset(item.endpoint for item in trajectory),
            )
            record["similarity_to_atlas"] = round(similarity[row.index - 1], 4)
            record["route_to_seed"] = ancestry(trajectory, row.endpoint)
            record["rung"] = classify_rung(
                record["top_ten_new_mean"] or 0.0, ladder[task_name]
            )
            probes.append(record)
            _write(partial=True)
            print(
                f"{task_name:26s} {'+'.join(labels):46s} call={record['seed_charged_call']:4d} "
                f"seed={record['seed_score']:.4f} top10new={record['top_ten_new_mean']:.4f} "
                f"rung={record['rung']['rung_reached']:12s} sim={record['similarity_to_atlas']:.3f} "
                f"{record['seconds']:.0f}s",
                flush=True,
            )

    payload = _write(partial=False)
    print(f"\ndiagnostic oracle calls: {payload['diagnostic_oracle_calls']}")
    print(f"wrote {repo_root / args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
