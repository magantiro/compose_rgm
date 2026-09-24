"""Run the frozen, small zero-oracle attachment-interface release ablation.

This is a development mechanism assay, not a 100-attempt/three-seed official
benchmark row. Every prompt-arm unit is written atomically and can be resumed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
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

from compose_v4.benchmark.fragment_constrained import FragmentTask, load_genmol_prompts
from compose_v4.benchmark.fragment_official_metrics import official_prompt_metrics
from compose_v4.rewrite.kernel import de_novo_rewrite_system

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = Path("/Users/rmaganti/compose_fragment_ckpt/ringcore_a7546e2_best.pt")
OUTPUT = Path("diagnostics/fragment_interface_release_pilot_v1")
CONTRACT = Path("docs/FRAGMENT_INTERFACE_RELEASE_PILOT_2026-09-23.md")
SAMPLES = 20
SEED = 0
MOTIF_DECORATION_DRUGS = (
    "BARICITINIB", "ERLOTINIB", "LIOTHYRONINE", "MARIBAVIR",
)
SUPERSTRUCTURE_DRUGS = ("BARICITINIB", "LOVASTATIN")
TASK_DRUGS = (
    (FragmentTask.MOTIF_EXTENSION, MOTIF_DECORATION_DRUGS),
    (FragmentTask.SCAFFOLD_DECORATION, MOTIF_DECORATION_DRUGS),
    (FragmentTask.SUPERSTRUCTURE_GENERATION, SUPERSTRUCTURE_DRUGS),
)
EXPECTED_SHA256 = {
    CHECKPOINT: "24117dfeaee91729bb4ebccb5eb5b993b4a6218605045e51917823d086b4c1e4",
    MANIFEST: "a4fb8357d0f1102cbdc8d79d802e15f66a59a9722c0b7125ce693fe7a29872a9",
}
IMPLEMENTATION_FILES = (
    "src/compose_v4/benchmark/fragment_attachment_control.py",
    "src/compose_v4/benchmark/fragment_conditioned_sampler.py",
    "src/compose_v4/benchmark/fragment_constrained.py",
    "src/compose_v4/benchmark/fragment_official_metrics.py",
    "tools/run_fragment_constrained_suite.py",
    "tools/run_fragment_interface_release_pilot.py",
    "tools/fetch_official_fragment_evaluator.py",
)
EXPECTED_SOFTWARE = {
    "python": "3.11.13",
    "rdkit": "2024.03.5",
    "numpy": "1.26.4",
    "torch": "2.4.0",
    "pandas": "2.2.3",
    "scipy": "1.13.1",
    "networkx": "3.3",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(_json_safe(payload), indent=2, sort_keys=True, allow_nan=False)
    )
    temporary.replace(path)


def _finite(value: float) -> float | None:
    return float(value) if math.isfinite(value) else None


def _json_safe(value):
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _identity() -> dict:
    inputs = {}
    for path, expected in EXPECTED_SHA256.items():
        actual = _sha256(path)
        if actual != expected:
            raise RuntimeError(f"input hash mismatch: {path}: {actual} != {expected}")
        inputs[str(path)] = actual
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
                f"fragment kernel mismatch for {name}: "
                f"{software[name]} != {expected}"
            )
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    return {
        "schema": "fragment_interface_release_pilot_v1",
        "contract_sha256": _sha256(CONTRACT),
        "base_git_commit": revision,
        "inputs_sha256": inputs,
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
            "task_drugs": {
                task.value: list(drugs) for task, drugs in TASK_DRUGS
            },
            "status": "small zero-oracle development pilot; not an official benchmark row",
        },
    }


def _runner_configuration(allow_growth: bool):
    # Consume the production CLI mapping, including its validation, rather than
    # independently reconstructing the field under test.
    argv = [
        "--checkpoint", str(CHECKPOINT), "--output", "unused.json",
        "--attachment-control",
    ]
    if allow_growth:
        argv.append("--allow-post-coverage-core-growth")
    args = build_parser().parse_args(argv)
    return sampler_config_from_args(args), control_from_args(args)


def _unit_path(output: Path, arm: str, task: str, drug: str) -> Path:
    return output / "units" / f"{arm}__{task}__{drug.lower()}.json"


def _committed_metrics(rows: list[dict]) -> dict:
    committed = [
        smiles
        for row in rows
        for smiles in row["committed_endpoint_smiles"]
    ]
    if not committed:
        return {"count": 0, "validity": 0.0, "uniqueness": 0.0,
                "quality": 0.0, "diversity": 0.0}
    metrics = official_prompt_metrics(committed, expected_samples=len(committed))
    return {"count": len(committed), **{k: _finite(v) for k, v in metrics.items()}}


def _summarize(units: dict[tuple[str, str, str], dict]) -> dict:
    summary = {}
    for arm in ("permanent_restriction", "release_after_coverage"):
        summary[arm] = {}
        for task, drugs in TASK_DRUGS:
            rows = [
                units[(arm, task.value, drug)]["result"]["per_drug"][drug][0]
                for drug in drugs
            ]
            attempts = sum(row["attempts"] for row in rows)
            committed = sum(row["committed_endpoints"] for row in rows)
            summary[arm][task.value] = {
                "attempts": attempts,
                "committed": committed,
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
                "redirections": sum(row["redirections"] for row in rows),
                "interface_rejections": sum(
                    row["interface_rejections"] for row in rows
                ),
                "staging_rejections": sum(
                    row["staging_rejections"] for row in rows
                ),
                "committed_official_formula": _committed_metrics(rows),
                "per_prompt": {
                    drug: {
                        "attempts": row["attempts"],
                        "committed": row["committed_endpoints"],
                        "task_success": row["emitted_nonempty"],
                        "committed_official_formula": _committed_metrics([row]),
                        "emitted_official_formula": row["official"],
                    }
                    for drug, row in zip(drugs, rows, strict=True)
                },
            }

    # A prompt with NO declared interfaces must leave the two sampler streams
    # identical. This fails the whole pilot rather than becoming a footnote.
    for drug in SUPERSTRUCTURE_DRUGS:
        left = units[("permanent_restriction", FragmentTask.SUPERSTRUCTURE_GENERATION.value, drug)]
        right = units[("release_after_coverage", FragmentTask.SUPERSTRUCTURE_GENERATION.value, drug)]
        lrow = left["result"]["per_drug"][drug][0]
        rrow = right["result"]["per_drug"][drug][0]
        if lrow["attempt_records"] != rrow["attempt_records"]:
            raise AssertionError(f"superstructure no-interface identity failed for {drug}")
    for arm, by_task in summary.items():
        for task, row in by_task.items():
            if row["committed"] != row["chemically_valid_committed"]:
                raise AssertionError(f"invalid committed endpoint in {arm}/{task}")
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
    for arm, allow_growth in (
        ("permanent_restriction", False),
        ("release_after_coverage", True),
    ):
        config, control = _runner_configuration(allow_growth)
        for task, drugs in TASK_DRUGS:
            for drug in drugs:
                key = arm, task.value, drug
                path = _unit_path(args.output_dir, *key)
                if path.exists():
                    payload = json.loads(path.read_text())
                    if payload["identity"] != identity:
                        raise RuntimeError(f"stale or mismatched pilot unit: {path}")
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
    _atomic_json(
        args.output_dir / "summary.json",
        {
            "identity": identity,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "units": [str(_unit_path(args.output_dir, *key)) for key in units],
            "summary": summary,
        },
    )
    print(f"wrote {args.output_dir / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
