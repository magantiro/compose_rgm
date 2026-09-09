"""Compile one declared answer-known whole-ring plan, with no model or docking."""

from __future__ import annotations

import argparse
import gzip
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import numpy as np
from rdkit import rdBase

from compose_v4.experiments.continuation_profile import ExecutorMeter, verify_file
from compose_v4.experiments.whole_ring_plan import first_winner_plan, verify_trace
from compose_v4.rewrite.kernel import InvalidRewrite
from tools.ivg_winner_paths import digest, implementation_closure, publish, sha

ROOT = Path(__file__).resolve().parents[1]
PAIR = "5705946f25f35dd88189b1204f01f294e96a43e78fd1c5030967c48cecceccc8"
AUDIT_SHA = "3b7fb79a541b94dd5421f30ffc0516c5cdc4d9316845cd40561d799ec517f922"


def run(args):
    if args.output.exists():
        raise ValueError(f"preserve previous result: {args.output}")
    if rdBase.rdkitVersion != "2024.03.5":
        raise ValueError("this diagnostic requires the pinned RDKit 2024.03.5 runtime")
    dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True)
    if dirty.strip():
        raise ValueError("run from a clean committed source tree; publish outside that tree")
    verify_file(args.audit, AUDIT_SHA)
    audit = json.loads(args.audit.read_text())
    row = next(r for r in audit["pairs"] if r["pair_id"] == PAIR)
    receipt_path = args.audit.parent / row["receipt"]
    verify_file(receipt_path, row["receipt_sha256"])
    receipt = json.loads(gzip.decompress(receipt_path.read_bytes()))
    if digest(receipt["payload"]) != receipt["payload_sha256"]:
        raise ValueError("winner receipt payload hash mismatch")
    path = receipt["payload"]["path"]
    if receipt["pair"]["pair_id"] != PAIR or path["status"] != "witness_found":
        raise ValueError("expected the saved first-winner witness")
    started = perf_counter()
    meter = ExecutorMeter(None)
    attempts = []
    with meter.instrument():
        for electronic in (False, True):
            before, start = meter.calls, perf_counter()
            meter.phase = f"compile_electronic_preparation_{electronic}"
            attempt = {"electronic_preparation": electronic}
            try:
                stages = first_winner_plan(path, electronic_preparation=electronic)
                compilation_calls, compilation_seconds = (
                    meter.calls - before,
                    perf_counter() - start,
                )
                meter.phase = "independent_saved_state_replay"
                replay_start = perf_counter()
                checks = verify_trace(path["source_state"], stages, path["target_2d"])
                attempt.update(
                    status="exact_winner",
                    stages=stages,
                    checks=checks,
                    compilation_executor_calls=compilation_calls,
                    compilation_seconds=compilation_seconds,
                    replay_seconds=perf_counter() - replay_start,
                )
            except (InvalidRewrite, ValueError) as error:
                attempt.update(
                    status="rejected", error_type=type(error).__name__, reason=str(error)
                )
            attempt.update(executor_calls=meter.calls - before, seconds=perf_counter() - start)
            attempts.append(attempt)
            if attempt["status"] == "exact_winner":
                break
    closure = implementation_closure(Path(__file__).resolve())
    closure["docs/T4_WHOLE_RING_PLAN.md"] = sha(ROOT / "docs/T4_WHOLE_RING_PLAN.md")
    closure["tools/ivg_winner_paths.py"] = sha(ROOT / "tools/ivg_winner_paths.py")
    report = {
        "schema_version": "t4_whole_ring_plan_v1",
        "evidence": "target-informed descriptor/executor witness; NOT autonomous recovery",
        "code_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "input_paths_and_sha256": {
            str(args.audit.resolve()): AUDIT_SHA,
            str(receipt_path.resolve()): row["receipt_sha256"],
        },
        "implementation_sha256": dict(sorted(closure.items())),
        "configuration": {
            "pair_id": PAIR,
            "structural_plan": "docs/T4_WHOLE_RING_PLAN.md",
            "core_variants": ["direct_four", "six_with_electronic_preparation"],
            "stop": "first exact endpoint, otherwise preserve both failures",
            "max_active_atoms": 40,
            "persistent_slots": 48,
            "seed": None,
            "seed_derivation": "deterministic descriptors; no RNG",
            "workers": 1,
            "law_enumerations": 0,
            "new_docking_calls": 0,
        },
        "split_identity": "inspected PARP1 seed0 d0.4 IVG development winner; answer-known",
        "access_basis": audit["access_basis_and_upstream"],
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "rdkit": rdBase.rdkitVersion,
        },
        "hardware": {
            "device": "CPU",
            "machine": platform.machine(),
            "workers": 1,
            "precision": "integer molecular states; no neural inference",
        },
        "source_state": path["source_state"],
        "target": path["target_2d"],
        "reference_witness_primitive_edits": path["witness_steps"],
        "attempts": attempts,
        "executor_attempts": meter.attempts,
        "summary": {
            "status": attempts[-1]["status"],
            "attempts": len(attempts),
            "executor_calls": meter.calls,
            "seconds": perf_counter() - started,
            "new_docking_calls": 0,
            "exclusions": [],
        },
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "limitations": [
            "one answer-known winner, not a shortest-path proof",
            "only ring stages match existing compound option contracts",
            "no Q(M), learned-fiber, or controlled probability qualification",
            "no measured docking or speed comparison against autonomous search",
            "local pinned chemistry runtime, not a Modal run",
        ],
    }
    publish(args.output, report)
    print(json.dumps(report["summary"], indent=2))
    for attempt in attempts:
        print(attempt["status"], attempt.get("reason", attempt.get("checks")))
        for stage in attempt.get("stages", []):
            print(stage["name"], stage["primitive_edits"], stage["endpoint"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())
