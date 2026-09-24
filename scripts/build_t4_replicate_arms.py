"""The two additional stochastic replicates of the unified T4 controller.

WHAT A REPLICATE IS HERE
------------------------
Replicate 1 is the historical panel, read as the unified policy's semantics:
`diagnostics/t4_behavioral_equivalence/audit_v1.json` shows that for every
historical row that reached a fallback branch, the unified router selects that
same branch from the same molecular state.  Its seeds are already the seeds the
sealed unified arms carry -- verified 30/30, including the rescue rows -- so
replicate 1 is not re-run and is not re-seeded.

Replicates 2 and 3 change ONE thing: the stochastic replicate seed, under the
repository's existing derivation

    replicate_n = controller_seed + 8_000_000_029 * (n - 1)

which is the rule already recorded in the replication manifest, not a new one
invented here.

TERMINOLOGY, because the two are easy to conflate: the three molecules per
protein are STARTING MOLECULES; the number that changes between replicates is a
STOCHASTIC REPLICATE SEED.

BUDGET
------
Six cells per arm (3 starting molecules x replicates 2 and 3) at 250 charged
calls, ten arms: 15,000, which is the whole authorization for the two new
replicates.  The per-arm ceiling is DERIVED by summing the cells, never written
down, and `tests/test_t4_replicate_arms.py` recomputes it.

Seven historical rows carry a ceiling of 249 or 248 because the rescue contract
debited measured prior spend on that cell.  A fresh replicate has no such prior,
so these arms use the full 250 under the at-most-250 protocol.  That is a
deliberate, recorded choice rather than an inherited accident.
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
DELTA_TAGS = {"d04": 0.4, "d06": 0.6}
REPLICATES = (2, 3)
CHARGED_CALLS_PER_CELL = 250

#: The repository's existing replicate-seed derivation, recorded in
#: diagnostics/t4_replication_manifest/manifest_v1.json -> seed_derivation.
REPLICATE_STRIDE = 8_000_000_029

WRAPPER_TEMPLATE = '''"""Unified T4 controller launcher for {target} at delta={delta}, replicates 2 and 3.

Sets only the arm's identity. The CONTROLLER is byte-identical to every other arm;
the protein, the similarity threshold and the stochastic replicate seed are all
INPUTS rather than configuration.

Pinned in its contract's `runtime_inputs_sha256` and baked into the image, because
`_validate_task` re-hashes every pinned entry inside the container and the wrapper
is what selects the arm.
"""

from __future__ import annotations

import os

os.environ.setdefault(
    "COMPOSE_HELD_CONTRACT",
    "configs/t4_unified_controller_{target}_{tag}_r23_v1.json",
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
    "COMPOSE_HELD_WRAPPER",
    "modal_apps/t4_unified_controller_{target}_{tag}_r23_app.py",
)

from modal_apps.t4_unified_controller_app import app, main  # noqa: F401
'''


def replicate_seed(base: int, replicate: int) -> int:
    return base + REPLICATE_STRIDE * (replicate - 1)


def _payload(path: Path) -> dict:
    return json.loads(path.read_text())["payload"]


def _base_wrapper(target: str, tag: str) -> Path:
    suffix = "_d04_app.py" if tag == "d04" else "_app.py"
    return ROOT / f"modal_apps/t4_unified_controller_{target}{suffix}"


def _wrapper_settings(target: str, tag: str) -> dict:
    text = _base_wrapper(target, tag).read_text()
    settings = {}
    for key in (
        "COMPOSE_HELD_CHECKPOINT",
        "COMPOSE_HELD_VOLUME",
        "COMPOSE_HELD_OUTPUT",
        "COMPOSE_HELD_APP",
    ):
        marker = f'"{key}",'
        head = text.index(marker) + len(marker)
        settings[key] = text[head : text.index(")", head)].strip().strip('",\n ').strip('"')
    return settings


def main() -> None:
    total = 0
    for target in TARGETS:
        for tag, delta in DELTA_TAGS.items():
            base_contract = ROOT / f"configs/t4_unified_controller_{target}_{tag}_v1.json"
            base = _payload(base_contract)
            settings = _wrapper_settings(target, tag)

            wrapper = ROOT / f"modal_apps/t4_unified_controller_{target}_{tag}_r23_app.py"
            wrapper.write_text(
                WRAPPER_TEMPLATE.format(
                    target=target,
                    tag=tag,
                    delta=delta,
                    checkpoint=settings["COMPOSE_HELD_CHECKPOINT"],
                    volume=settings["COMPOSE_HELD_VOLUME"],
                    output=settings["COMPOSE_HELD_OUTPUT"],
                    appname=settings["COMPOSE_HELD_APP"].replace("-v1", "-r23-v1"),
                )
            )

            cells = []
            for row in base["cells"]:
                for replicate in REPLICATES:
                    cells.append(
                        {
                            "cell": f"{row['cell']}_r{replicate}",
                            "controller_seed": replicate_seed(
                                row["controller_seed"], replicate
                            ),
                            "replicate": replicate,
                            "replicate_1_controller_seed": row["controller_seed"],
                            "smiles": row["smiles"],
                            "source_cell": row["cell"],
                            "source_global_index": row["source_global_index"],
                        }
                    )

            payload = dict(base)
            payload["cells"] = cells
            payload["charged_calls_per_cell"] = CHARGED_CALLS_PER_CELL
            payload["total_charged_call_ceiling"] = sum(
                CHARGED_CALLS_PER_CELL for _ in cells
            )
            payload["schema_version"] = "t4_unified_controller_replicate_contract_v1"
            payload["scored_launch_authorized"] = False
            payload["modal_launch_authorized"] = False
            payload["status"] = "FROZEN_PENDING_OWNER_LAUNCH_AUTHORIZATION"
            payload["replicate_policy"] = {
                "replicates_in_this_arm": list(REPLICATES),
                "replicate_1_is_not_rerun": (
                    "Replicate 1 is the historical panel, admitted under the "
                    "branch-level equivalence audit at "
                    "diagnostics/t4_behavioral_equivalence/audit_v1.json. Its "
                    "seeds are already the seeds the sealed unified arms carry "
                    "(verified 30/30), so it is neither re-run nor re-seeded."
                ),
                "seed_rule": "controller_seed + 8_000_000_029 * (replicate - 1)",
                "seed_rule_source": (
                    "diagnostics/t4_replication_manifest/manifest_v1.json"
                    " -> seed_derivation"
                ),
                "varies_across_replicates": ["controller_seed"],
                "terminology": (
                    "the three molecules per protein are STARTING MOLECULES; the "
                    "value that changes between replicates is a STOCHASTIC "
                    "REPLICATE SEED"
                ),
                "budget_note": (
                    "Seven historical rows carry a 249 or 248 ceiling because the "
                    "rescue contract debited measured prior spend on that cell. A "
                    "fresh replicate has no such prior, so every cell here uses "
                    "the full 250 under the at-most-250 protocol."
                ),
            }
            payload["claim_boundary"] = (
                base["claim_boundary"]
                + " REPLICATES 2 AND 3: identical controller bytes and settings to "
                "replicate 1; only the stochastic replicate seed differs. Report "
                "the three-replicate MEAN, never the best of three."
            )

            pins = dict(base["runtime_inputs_sha256"])
            pins.pop(str(_base_wrapper(target, tag).relative_to(ROOT)), None)
            pins[str(wrapper.relative_to(ROOT))] = ""
            payload["runtime_inputs_sha256"] = {
                relative: sha256_file(ROOT / relative) for relative in sorted(pins)
            }

            destination = (
                ROOT / f"configs/t4_unified_controller_{target}_{tag}_r23_v1.json"
            )
            destination.write_text(
                json.dumps(
                    {"payload": payload, "payload_sha256": identity(payload)},
                    indent=2,
                    sort_keys=True,
                )
                + "\n"
            )
            total += payload["total_charged_call_ceiling"]
            print(
                f"{destination.name:48s} delta={delta} cells={len(cells)} "
                f"ceiling={payload['total_charged_call_ceiling']}"
            )
    print(f"TOTAL DERIVED CEILING FOR REPLICATES 2 AND 3 = {total}")


if __name__ == "__main__":
    main()
