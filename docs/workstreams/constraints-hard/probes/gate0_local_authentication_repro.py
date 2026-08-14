"""Reproduce the documented Gate-0 local authentication failure. NO BYPASS.

This exists so a reader can confirm in one command that the Stage-1 census
blocker is the already-diagnosed non-portable `source_index_sha256`
(`docs/PARETO_PARITY_ENVIRONMENT_STATUS.md`) and not something new.

It calls `open_process_v2_t1_source` exactly as the production apps do and
reports the failure. It does not patch, skip, or weaken any guard.

    python3 docs/workstreams/constraints-hard/probes/gate0_local_authentication_repro.py \
        --local-runtime /path/to/local_runtime

Repo root is derived from this file's location; override with COMPOSE_REPO_ROOT.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_ROOT = Path(
    os.environ.get(
        "COMPOSE_REPO_ROOT",
        Path(__file__).resolve().parents[4],
    )
)
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--local-runtime",
        type=Path,
        required=True,
        help="directory holding active8/, gate_zero_v6/, artifact_root/",
    )
    parser.add_argument("--gate-zero-dir", default="gate_zero_v6")
    args = parser.parse_args()

    from compose_v4.experiments.editing_v2_process_v2_t1_panel import (
        open_process_v2_t1_source,
    )

    runtime: Path = args.local_runtime
    active8_candidates = sorted((runtime / "active8").iterdir())
    if not active8_candidates:
        print(f"no Active8 tree under {runtime / 'active8'}")
        return 2
    active8_root = active8_candidates[0]
    decision = runtime / args.gate_zero_dir / "DECISION.json"

    print(f"active8 root : {active8_root}")
    print(f"gate-0       : {decision}")
    print(f"repo root    : {_ROOT}")

    try:
        source = open_process_v2_t1_source(
            active8_root,
            gate_zero_decision_path=decision,
            artifact_root=runtime / "artifact_root",
            repo_root=_ROOT,
        )
    except Exception as exc:  # noqa: BLE001
        print()
        print(f"BLOCKED (expected): {type(exc).__name__}")
        print(f"  {exc}")
        print()
        print("This is the documented non-portable `source_index_sha256`:")
        print("  editing_v2_process_v2_gate_zero.py:593 hashes a body whose first")
        print("  field is the absolute Active8 mount path, so identical content")
        print("  mounted locally authenticates against the wrong string.")
        print("  See docs/PARETO_PARITY_ENVIRONMENT_STATUS.md.")
        print()
        print("DO NOT patch the decision to make this pass. The chain is closed by")
        print("a canonical-JSON framing guard and a decision self-hash; three")
        print("integrity guards deep is a signal to stop.")
        return 1

    print(f"OPENED OK -> {type(source).__name__}")
    print("The blocker has been resolved upstream; re-cost the Stage-1 census.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
