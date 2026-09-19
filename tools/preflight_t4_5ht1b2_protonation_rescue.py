"""Print the fail-closed local preflight for the unlaunched scored rescue."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from compose_v4.experiments.t4_protonation_rescue_contract import (
    validate_rescue_preflight,
)

DEFAULT_CONTRACT = (
    ROOT / "configs/t4_shared_retained_fiber_5ht1b2_protonation_rescue_v1.json"
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--authorization-payload-sha256")
    parser.add_argument(
        "--allow-dirty-runtime", action="store_true", help=argparse.SUPPRESS
    )
    arguments = parser.parse_args()
    result = validate_rescue_preflight(
        ROOT,
        arguments.contract,
        authorization_payload_sha256=arguments.authorization_payload_sha256,
        require_clean_runtime=not arguments.allow_dirty_runtime,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
