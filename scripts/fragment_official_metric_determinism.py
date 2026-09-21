#!/usr/bin/env python3
"""Is the official fragment evaluator bitwise reproducible across processes?

Two independent comparisons -- pinned kernel against laptop kernel, and the
attachment arm against the baseline arm on a task where the controller is
provably inert -- both showed every emitted molecule identical and
``official.diversity`` differing by about 2e-16.  A difference that survives
identical input and identical molecules is not chemistry, and attributing it to
the rdkit build would have been wrong.

This probe isolates the cause by holding the input fixed and varying only
``PYTHONHASHSEED``.  If diversity moves, the metric sums over a hash-ordered
collection and is not bitwise reproducible across processes -- which means byte
equality is the WRONG parity criterion for it, and any comparison that demands
it will report divergence that is not there.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

_CHILD = """
import json, sys
from compose_v4.benchmark.fragment_official_metrics import official_prompt_metrics
samples = json.loads(sys.stdin.read())
print(json.dumps(official_prompt_metrics(samples)))
"""


def _score_with_hash_seed(samples: list[str], seed: str, root: Path) -> dict:
    env = dict(os.environ)
    env["PYTHONHASHSEED"] = seed
    env["PYTHONPATH"] = f"{root}/src:{root}/scripts"
    env["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    env["OMP_NUM_THREADS"] = "1"
    proc = subprocess.run(
        [sys.executable, "-c", _CHILD],
        input=json.dumps(samples), capture_output=True, text=True, env=env, cwd=root,
        check=False,  # the return code is inspected below with the child's stderr
    )
    if proc.returncode != 0:
        raise SystemExit(f"scoring failed under PYTHONHASHSEED={seed}: {proc.stderr[-800:]}")
    return json.loads(proc.stdout.strip().splitlines()[-1])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--hash-seeds", default="0,1,2,12345,999")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    payload = json.loads(args.shard.read_text())
    samples = None
    for result in payload["results"].values():
        for entries in result.get("per_drug", {}).values():
            samples = entries[0]["emitted_samples"]
            break
        if samples:
            break
    if not samples:
        raise SystemExit(f"no emitted samples in {args.shard}")

    seeds = [s.strip() for s in args.hash_seeds.split(",") if s.strip()]
    scored = {seed: _score_with_hash_seed(samples, seed, root) for seed in seeds}

    metrics = sorted({m for row in scored.values() for m in row})
    spread = {
        metric: max(row[metric] for row in scored.values())
        - min(row[metric] for row in scored.values())
        for metric in metrics
    }
    unstable = sorted(m for m, width in spread.items() if width != 0.0)
    out = {
        "schema": "compose_fragment_official_metric_determinism_v1",
        "question": (
            "does the official evaluator return bitwise identical metrics for "
            "identical input across processes?"
        ),
        "shard": str(args.shard),
        "samples_scored": len(samples),
        "hash_seeds": seeds,
        "per_hash_seed": scored,
        "spread_by_metric": spread,
        "metrics_not_bitwise_reproducible": unstable,
        "verdict": (
            "HASH_ORDER_DEPENDENT" if unstable else "BITWISE_REPRODUCIBLE"
        ),
        "consequence": (
            "A metric listed in metrics_not_bitwise_reproducible cannot be "
            "compared by byte equality across processes. Compare it within a "
            "tolerance and report the observed spread; demanding exact equality "
            "manufactures divergence that is not chemistry."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2))
    print(json.dumps({k: v for k, v in out.items() if k != "per_hash_seed"}, indent=2))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
