"""Command form of the donor scored-path consumption gate. Zero oracle calls.

Run before any donor-arm launch:

    KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 PYTHONPATH=src python \
        scripts/pmo_donor_scored_gate.py --out diagnostics/pmo_discovery_v1/donor_scored_gate_v1.json

A PASS means the scored entry point's own `optimizer_kwargs` build a controller whose
production `propose_batch` reaches the donor cut law, and that `restore` accepts the same
kwargs. It does NOT mean the arm is authorized; arm D needs its own sealed payload.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

from compose_v4.experiments.pmo_donor_scored_gate import gate

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as scratch:
        report = gate(ROOT, Path(scratch))
    print(f"verdict: {report['verdict']}")
    print(f"  default cut law      {report['default_cut_law']}")
    print(f"  on-arm kwargs        {report['on_arm_optimizer_kwargs']}")
    print(f"  off-arm kwargs       {report['off_arm_optimizer_kwargs']}")
    print(f"  live channels        {report['live_channels']}")
    print(f"  law consumed at seed {report['law_consumed']['seed']}")
    print(f"  restore accepts      {report['restore_accepts_scored_kwargs']}")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
