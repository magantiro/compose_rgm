"""Create the delta=0.4 arms of the unified T4 controller.

WHAT CHANGES, AND WHAT MUST NOT
--------------------------------
`delta` is a BENCHMARK INPUT.  The T4 task is defined at two similarity
thresholds and supplies the threshold per instance, exactly as it supplies the
start molecule.  The absence of a delta=0.4 unified contract was a provenance
gap, not evidence that the unified policy does not cover delta=0.4.

So these arms are produced from the sealed delta=0.6 contracts by changing the
threshold and NOTHING ELSE that the controller reads.  Three other fields move,
and each is identity rather than policy:

    delta                   0.6 -> 0.4              the benchmark input
    runtime_inputs_sha256   the arm's own wrapper    per-arm identity
    claim_boundary /        prose naming the         prose
    experimental_setting    threshold

`tests/test_t4_delta04_contracts.py` enumerates every leaf of both payloads and
FAILS on any difference outside that allow-list, so "no other controller setting
changes" is enforced rather than asserted.  The test derives the leaf set from
the payloads themselves, so a controller field added later is covered without
anyone remembering to extend a list.

The arms are written UNAUTHORIZED (`modal_launch_authorized: false`).  The launch
guard in `t4_unified_controller_app._local_task` refuses to spawn while that is
false, so generating a contract cannot by itself start a scored run.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from compose_v4.control.docking_value import identity
from compose_v4.experiments.continuation_profile import sha256_file

TARGETS = ("5ht1b", "braf", "fa7", "jak2", "parp1")
SOURCE_DELTA = 0.6
TARGET_DELTA = 0.4

WRAPPER_TEMPLATE = '''"""Unified T4 controller launcher for {target} at delta={target_delta}.

The delta=0.4 sibling of `t4_unified_controller_{target}_app.py`.  It sets only the
arm's identity -- contract, route-expert checkpoint, volume, output namespace, app
name and receptor -- so the CONTROLLER is byte-identical to every other arm and both
the protein and the similarity threshold are benchmark INPUTS rather than
configuration.

This wrapper is PINNED in its contract's `runtime_inputs_sha256` and baked into the
image, because `_validate_task` re-hashes every pinned entry inside the container and
the wrapper is what selects the arm.
"""

from __future__ import annotations

import os

os.environ.setdefault(
    "COMPOSE_HELD_CONTRACT", "configs/t4_unified_controller_{target}_d04_v1.json"
)
os.environ.setdefault(
    "COMPOSE_HELD_CHECKPOINT",
    "{checkpoint}",
)
os.environ.setdefault("COMPOSE_HELD_VOLUME", "{volume}")
os.environ.setdefault("COMPOSE_HELD_OUTPUT", "{output}")
os.environ.setdefault("COMPOSE_HELD_APP", "{appname}")
os.environ.setdefault("COMPOSE_HELD_RECEPTOR_NAME", "{target}")
os.environ.setdefault(
    "COMPOSE_HELD_WRAPPER", "modal_apps/t4_unified_controller_{target}_d04_app.py"
)

from modal_apps.t4_unified_controller_app import app, main  # noqa: F401
'''


def _payload(path: Path) -> dict:
    return json.loads(path.read_text())["payload"]


def _wrapper_settings(target: str) -> dict:
    """Read the d06 wrapper's own environment so the d04 sibling cannot drift."""

    text = (ROOT / f"modal_apps/t4_unified_controller_{target}_app.py").read_text()
    settings = {}
    for key in (
        "COMPOSE_HELD_CHECKPOINT",
        "COMPOSE_HELD_VOLUME",
        "COMPOSE_HELD_OUTPUT",
        "COMPOSE_HELD_APP",
    ):
        marker = f'"{key}",'
        head = text.index(marker) + len(marker)
        value = text[head : text.index(")", head)].strip().strip('",\n ').strip('"')
        settings[key] = value
    return settings


def main() -> None:
    for target in TARGETS:
        settings = _wrapper_settings(target)
        wrapper_path = ROOT / f"modal_apps/t4_unified_controller_{target}_d04_app.py"
        wrapper_path.write_text(
            WRAPPER_TEMPLATE.format(
                target=target,
                target_delta=TARGET_DELTA,
                checkpoint=settings["COMPOSE_HELD_CHECKPOINT"],
                volume=settings["COMPOSE_HELD_VOLUME"],
                output=settings["COMPOSE_HELD_OUTPUT"],
                # A distinct app name so the two thresholds can run concurrently;
                # the VOLUME is shared, which is safe because every cell writes
                # under its own content-addressed run_id.
                appname=settings["COMPOSE_HELD_APP"].replace("-v1", "-d04-v1"),
            )
        )

        source_path = ROOT / f"configs/t4_unified_controller_{target}_d06_v1.json"
        payload = _payload(source_path)
        payload["delta"] = TARGET_DELTA

        payload["experimental_setting"] = payload["experimental_setting"].replace(
            f"delta={SOURCE_DELTA}", f"delta={TARGET_DELTA}"
        )
        payload["claim_boundary"] = payload["claim_boundary"].replace(
            f"delta {SOURCE_DELTA}", f"delta {TARGET_DELTA}"
        )
        payload["delta_provenance"] = {
            "derived_from": f"configs/t4_unified_controller_{target}_d06_v1.json",
            "derived_from_payload_sha256": identity(_payload(source_path)),
            "changed": ["delta", "runtime_inputs_sha256", "claim_boundary",
                        "experimental_setting", "delta_provenance"],
            "rule": (
                "delta is a benchmark input supplied per instance, like the start "
                "molecule. No controller setting changes between the two "
                "thresholds; tests/test_t4_delta04_contracts.py enforces that by "
                "enumerating every leaf of both payloads."
            ),
        }

        pins = dict(payload["runtime_inputs_sha256"])
        pins.pop(f"modal_apps/t4_unified_controller_{target}_app.py", None)
        pins[f"modal_apps/t4_unified_controller_{target}_d04_app.py"] = sha256_file(
            wrapper_path
        )
        # Re-hash EVERY pin from this tree rather than carrying values forward.
        payload["runtime_inputs_sha256"] = {
            relative: sha256_file(ROOT / relative) for relative in sorted(pins)
        }

        destination = ROOT / f"configs/t4_unified_controller_{target}_d04_v1.json"
        destination.write_text(
            json.dumps(
                {"payload": payload, "payload_sha256": identity(payload)},
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        print(
            f"{destination.name}  delta={payload['delta']}  "
            f"cells={len(payload['cells'])}  "
            f"authorized={payload['modal_launch_authorized']}  "
            f"payload_sha256={identity(payload)}"
        )


if __name__ == "__main__":
    main()
