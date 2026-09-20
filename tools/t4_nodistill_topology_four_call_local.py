#!/usr/bin/env python3
"""Seal, inspect, and eventually run the no-egress four-call binding."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from compose_v4.experiments.t4_nodistill_topology_four_call_local import (
    seal_local_execution_binding,
)
from compose_v4.experiments.t4_nodistill_topology_four_call_local_runtime import (
    main as runtime_main,
)

ROOT = Path(__file__).resolve().parents[1]


def main(argv: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments == ["seal-binding"]:
        result = seal_local_execution_binding(ROOT)
        json.dump(result, sys.stdout, sort_keys=True, separators=(",", ":"))
        sys.stdout.write("\n")
        return 0
    return runtime_main(arguments)


if __name__ == "__main__":
    raise SystemExit(main())
