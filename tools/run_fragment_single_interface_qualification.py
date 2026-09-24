"""Frozen held-prompt, zero-oracle qualification of the interface-count policy.

All three arms use the production fragment runner. Each unit is persisted
atomically and a resumed invocation refuses any changed code or asset hash.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import networkx
import pandas
import scipy
from fetch_official_fragment_evaluator import verify_only
from run_fragment_constrained_suite import (
    MANIFEST,
    _kernel_provenance,
    build_parser,
    control_from_args,
    frozen_attachment_identity,
    frozen_sampler_identity,
    run_task,
    sampler_config_from_args,
)
from run_fragment_interface_release_pilot import (
    CHECKPOINT,
    EXPECTED_SHA256,
    EXPECTED_SOFTWARE,
    ROOT,
    _atomic_json,
    _committed_metrics,
    _sha256,
)

from compose_v4.benchmark.fragment_constrained import FragmentTask, load_genmol_prompts
from compose_v4.rewrite.kernel import de_novo_rewrite_system

CONTRACT = Path("docs/FRAGMENT_SINGLE_INTERFACE_QUALIFICATION_2026-09-23.md")
OUTPUT = Path("diagnostics/fragment_single_interface_qualification_v1")
SAMPLES = 20
SEED = 0
HELD_MOTIF_DECORATION = (
    "CYCLOTHIAZIDE", "ELIGLUSTAT", "FUTIBATINIB", "LESINURAD",
    "LOVASTATIN", "SPIRAPRIL",
)
HELD_SUPERSTRUCTURE = (
    "CYCLOTHIAZIDE", "ELIGLUSTAT", "ERLOTINIB", "FUTIBATINIB",
    "LESINURAD", "LIOTHYRONINE",
)
TASK_DRUGS = (
    (FragmentTask.MOTIF_EXTENSION, HELD_MOTIF_DECORATION),
    (FragmentTask.SCAFFOLD_DECORATION, HELD_MOTIF_DECORATION),
    (FragmentTask.SUPERSTRUCTURE_GENERATION, HELD_SUPERSTRUCTURE),
)
ARMS = (
    ("permanent_restriction", ()),
    ("global_release", ("--allow-post-coverage-core-growth",)),
    ("single_interface_policy", ("--release-single-interface-after-coverage",)),
)
IMPLEMENTATION_FILES = (
    "src/compose_v4/benchmark/fragment_attachment_control.py",
    "src/compose_v4/benchmark/fragment_conditioned_sampler.py",
    "src/compose_v4/benchmark/fragment_constrained.py",
    "src/compose_v4/benchmark/fragment_official_metrics.py",
    "tools/run_fragment_constrained_suite.py",
    "tools/run_fragment_interface_release_pilot.py",
    "tools/run_fragment_single_interface_qualification.py",
    "tools/fetch_official_fragment_evaluator.py",
)


def _identity() -> dict:
    input_hashes = {}
    for path, expected in EXPECTED_SHA256.items():
        actual = _sha256(path)
        if actual != expected:
            raise RuntimeError(f"input hash mismatch: {path}: {actual} != {expected}")
        input_hashes[str(path)] = actual
    upstream = verify_only()
    software = _kernel_provenance()
    software.update(
        pandas=pandas.__version__,
        scipy=scipy.__version__,
        networkx=networkx.__version__,
    )
    for name, expected in EXPECTED_SOFTWARE.items():
        if software[name] != expected:
            raise RuntimeError(
                f"fragment kernel mismatch for {name}: {software[name]} != {expected}"
            )
    return {
        "schema": "fragment_single_interface_qualification_v1",
        "contract_sha256": _sha256(CONTRACT),
        "development_summary_sha256": _sha256(
            Path("diagnostics/fragment_interface_release_pilot_v1/summary.json")
        ),
        "base_git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "inputs_sha256": input_hashes,
        "implementation_sha256": {
            name: _sha256(ROOT / name) for name in IMPLEMENTATION_FILES
        },
        "official_ivg_evaluator_commit": (
            "b50bb3ae2bdcb9df581f0b219d79cf14b05d0fbb"
        ),
        "official_ivg_blobs_sha256": upstream,
        "software": software,
        "protocol": {
            "attempts_per_prompt": SAMPLES,
            "seed": SEED,
            "task_drugs": {task.value: list(drugs) for task, drugs in TASK_DRUGS},
            "arms": [arm for arm, _ in ARMS],
            "status": "held-prompt zero-oracle qualification; not full official benchmark",
        },
    }


def _configuration(extra: tuple[str, ...]):
    args = build_parser().parse_args(
        [
            "--checkpoint", str(CHECKPOINT), "--output", "unused.json",
            "--attachment-control", *extra,
        ]
    )
    return sampler_config_from_args(args), control_from_args(args)


def _unit_path(output: Path, arm: str, task: FragmentTask, drug: str) -> Path:
    return output / "units" / f"{arm}__{task.value}__{drug.lower()}.json"


def _rows(units: dict, arm: str, task: FragmentTask, drugs: tuple[str, ...]):
    return [
        units[(arm, task.value, drug)]["result"]["per_drug"][drug][0]
        for drug in drugs
    ]


def _summarize(units: dict) -> dict:
    summary = {}
    for arm, _ in ARMS:
        summary[arm] = {}
        for task, drugs in TASK_DRUGS:
            rows = _rows(units, arm, task, drugs)
            prompt_metrics = {
                drug: _committed_metrics([row])
                for drug, row in zip(drugs, rows, strict=True)
            }
            summary[arm][task.value] = {
                "attempts": sum(row["attempts"] for row in rows),
                "committed": sum(row["committed_endpoints"] for row in rows),
                "chemically_valid_committed": sum(
                    row["committed_chemically_valid"] for row in rows
                ),
                "fragment_preserving_committed": sum(
                    row["committed_fragment_preserving"] for row in rows
                ),
                "interfaces_covered_committed": sum(
                    row["committed_interfaces_covered"] for row in rows
                ),
                "full_task_successes": sum(row["emitted_nonempty"] for row in rows),
                "committed_official_formula": _committed_metrics(rows),
                "per_prompt_mean": {
                    key: sum(prompt_metrics[drug][key] for drug in drugs) / len(drugs)
                    for key in ("quality", "uniqueness", "diversity")
                },
                "per_prompt": {
                    drug: {
                        "attempts": row["attempts"],
                        "committed": row["committed_endpoints"],
                        "task_success": row["emitted_nonempty"],
                        "committed_official_formula": prompt_metrics[drug],
                        "emitted_official_formula": row["official"],
                    }
                    for drug, row in zip(drugs, rows, strict=True)
                },
            }
    for task, drugs in TASK_DRUGS:
        for drug in drugs:
            policy = _rows(units, "single_interface_policy", task, (drug,))[0]
            interface_count = len(policy["attachment_spec"]["interfaces"])
            expected = (
                "global_release" if interface_count == 1
                else "permanent_restriction"
            )
            control = _rows(units, expected, task, (drug,))[0]
            if interface_count and task is FragmentTask.SUPERSTRUCTURE_GENERATION:
                raise AssertionError("zero-interface control declares an interface")
            if policy["attempt_records"] != control["attempt_records"]:
                raise AssertionError(
                    f"interface-count routing differed from {expected}: {task.value}/{drug}"
                )
            if task is FragmentTask.SUPERSTRUCTURE_GENERATION:
                released = _rows(units, "global_release", task, (drug,))[0]
                if policy["attempt_records"] != released["attempt_records"]:
                    raise AssertionError(f"zero-interface no-op failed: {drug}")
    for arm, by_task in summary.items():
        for task, row in by_task.items():
            if row["committed"] != row["chemically_valid_committed"]:
                raise AssertionError(f"invalid committed endpoint: {arm}/{task}")
            if row["committed"] != row["fragment_preserving_committed"]:
                raise AssertionError(f"fragment loss: {arm}/{task}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    args = parser.parse_args()
    identity = _identity()
    from evaluate_tracelet_rollouts import load_factorized_rollout_checkpoint

    model, meta = load_factorized_rollout_checkpoint(CHECKPOINT)
    identity["checkpoint_metadata"] = {
        "completed_steps": meta.get("completed_steps"),
        "corpus_scope_hash": meta.get("corpus_scope_hash"),
    }
    system = de_novo_rewrite_system()
    prompts = load_genmol_prompts(MANIFEST)
    units = {}
    for arm, extra in ARMS:
        config, control = _configuration(extra)
        for task, drugs in TASK_DRUGS:
            for drug in drugs:
                key = (arm, task.value, drug)
                path = _unit_path(args.output_dir, arm, task, drug)
                if path.exists():
                    payload = json.loads(path.read_text())
                    if payload["identity"] != identity:
                        raise RuntimeError(f"stale or mismatched held unit: {path}")
                    units[key] = payload
                    print(f"reused {arm}/{task.value}/{drug}", flush=True)
                    continue
                result = run_task(
                    model, system, prompts, task,
                    seeds=1, samples=SAMPLES, config=config, control=control,
                    verbose=False, seed_list=[SEED], drugs=[drug],
                )
                if result["prompts_scored"] != 1:
                    raise RuntimeError(
                        f"prompt build failed: {arm}/{task.value}/{drug}: "
                        f"{result['build_failures']}"
                    )
                payload = {
                    "identity": identity,
                    "arm": arm,
                    "sampler": frozen_sampler_identity(config),
                    "attachment_control": frozen_attachment_identity(control),
                    "task": task.value,
                    "drug": drug,
                    "result": result,
                }
                _atomic_json(path, payload)
                units[key] = payload
                row = result["per_drug"][drug][0]
                print(
                    f"{arm}/{task.value}/{drug}: "
                    f"committed={row['committed_endpoints']}/{SAMPLES}, "
                    f"task_success={row['emitted_nonempty']}/{SAMPLES}",
                    flush=True,
                )
    summary = _summarize(units)
    output = args.output_dir / "summary.json"
    _atomic_json(
        output,
        {
            "identity": identity,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "units": [
                str(_unit_path(args.output_dir, arm, task, drug))
                for arm, _ in ARMS
                for task, drugs in TASK_DRUGS
                for drug in drugs
            ],
            "summary": summary,
        },
    )
    print(f"wrote {output}", flush=True)


if __name__ == "__main__":
    main()
