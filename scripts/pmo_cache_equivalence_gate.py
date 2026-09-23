"""Prove the serialization memo changes nothing the controller decides.

The memo is keyed on exact array bytes, so it SHOULD return what the uncached call returns.
"Should" is not evidence. This runs the same script twice -- identical code, identical seed,
identical initialization -- differing only in whether the cache context is active, and
requires seven equalities. Any difference disqualifies the optimized path.

Both arms are launched as SUBPROCESSES so neither can inherit the other's context-local
cache, module state or RNG.
"""
from __future__ import annotations

import json
import pathlib
import subprocess
import sys

CHECKS = (
    ("1_endpoint_sequence", "endpoint_sequence"),
    ("2_parent_probabilities", "parent_policy_history"),
    ("3_rebuild_option_probabilities", "option_policy"),
    ("4_selected_indices", "selected_endpoints_per_round"),
    ("5_oracle_query_order", "endpoint_sequence"),
    ("6_rewards_and_updates", "observations"),
    ("7_rng_progression", "rng_state"),
)


def _run(root: pathlib.Path, cache: int, budget: int, rounds: int) -> dict:
    out = root / f"cache_{cache}"
    subprocess.run(
        [
            sys.executable,
            "scripts/pmo_reward_adaptive_canary.py",
            str(budget),
            str(rounds),
            "16",
        ],
        check=True,
        env={
            **_environment(),
            "CANARY_OUT": str(out),
            "CANARY_SMILES_CACHE": str(cache),
        },
    )
    with open(out / "equivalence.json") as handle:
        return json.load(handle)


def _environment() -> dict:
    import os

    return {
        **os.environ,
        "KMP_DUPLICATE_LIB_OK": "TRUE",
        "OMP_NUM_THREADS": "1",
        "PYTHONPATH": "src:scripts",
    }


def main() -> int:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    budget = int(args[0]) if args else 64
    rounds = int(args[1]) if len(args) > 1 else 6
    root = pathlib.Path(
        args[2] if len(args) > 2 else "diagnostics/pmo_cache_equivalence_v1"
    )
    root.mkdir(parents=True, exist_ok=True)

    print(f"=== arm A: cache DISABLED (budget {budget}) ===", flush=True)
    plain = _run(root, 0, budget, rounds)
    print(f"=== arm B: cache 512 (budget {budget}) ===", flush=True)
    cached = _run(root, 512, budget, rounds)

    results, verdict = {}, True
    for name, key in CHECKS:
        same = plain.get(key) == cached.get(key)
        results[name] = {
            "identical": same,
            "field": key,
            "size": len(plain.get(key) or []) if isinstance(plain.get(key), list) else 1,
        }
        verdict &= same
        print(f"  {'PASS' if same else 'FAIL'}  {name}  ({key})", flush=True)

    # A comparison over empty evidence is not a pass. Every list-valued check must have
    # compared something, or the harness is asserting that nothing equals nothing.
    populated = all(
        (results[name]["size"] > 0) for name, key in CHECKS if isinstance(plain.get(key), list)
    )
    if not populated:
        verdict = False
        print("  FAIL  evidence is empty; a vacuous comparison is not a proof", flush=True)

    report = {
        "schema_version": "pmo_cache_equivalence_gate_v1",
        "evidence_role": "optimization_equivalence_proof",
        "budget": budget,
        "checks": results,
        "evidence_populated": populated,
        "verdict": "IDENTICAL" if verdict else "DIFFERS",
    }
    with open(root / "equivalence_gate.json", "w") as handle:
        json.dump(report, handle, indent=1)
    print(f"\nVERDICT: {report['verdict']}")
    print("WROTE", root / "equivalence_gate.json")
    return 0 if verdict else 1


if __name__ == "__main__":
    sys.exit(main())
