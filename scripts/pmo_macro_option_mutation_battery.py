"""Mutation battery for the macro-option protection guards.

Each NEGATIVE mutation must turn a NAMED test red; the POSITIVE control changes bytes
and nothing else and must stay green.  Without a control that must pass, "everything
refused" is indistinguishable from a broken harness -- a battery in this repository once
read 20/20 refused because the temp tree had no `.git` and a cleanliness check fired
first.

A mutation that does not APPLY aborts the run instead of scoring as killed: a mutation
string that silently fails to match is a phantom survivor, and its verdict is fiction.

Usage:
  PYTHONPATH=src:scripts python scripts/pmo_macro_option_mutation_battery.py \
      --out diagnostics/pmo_macro_option_v1/mutation_battery_v1.json
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from time import perf_counter

LIBRARY = "src/compose_v4/control/pmo_macro_option.py"
CONTROLLER = "src/compose_v4/control/pmo_macro_option_controller.py"
ARMS = "src/compose_v4/experiments/pmo_macro_option_arms.py"
FAST = "tests/test_pmo_macro_option.py"
WIRING = "tests/test_pmo_macro_option_wiring.py"

# (name, file, old, new, test file, the named test the mutation must break)
NEGATIVES: list[tuple[str, str, str, str, str, str]] = [
    (
        "declare_a_macro_inside_the_ceiling",
        LIBRARY,
        "        if self.total_primitives <= self.realization_ceiling:",
        "        if False:",
        FAST,
        "test_declaration_refuses_a_macro_one_round_could_realize",
    ),
    (
        "accept_an_unbounded_window",
        LIBRARY,
        (
            "        if not isinstance(self.protection_rounds, int) or not (\n"
            "            1 <= self.protection_rounds <= PROTECTION_ROUNDS_CEILING\n"
            "        ):"
        ),
        "        if False:",
        FAST,
        "test_declaration_refuses_an_unbounded_or_absent_window",
    ),
    (
        "floor_never_applied",
        LIBRARY,
        "    if current >= floor or current >= 1.0:",
        "    if True:",
        FAST,
        "test_the_floor_lifts_a_starved_bridge_to_exactly_the_declared_share",
    ),
    (
        "floor_becomes_a_filter",
        LIBRARY,
        "    lifted = weights * ((1.0 - floor) / (1.0 - current))",
        "    lifted = weights * 0.0",
        FAST,
        "test_the_floor_is_a_reranking_and_never_a_filter",
    ),
    (
        "reservation_ignores_the_batch_limit",
        LIBRARY,
        "    reserved = reserved[:limit]",
        "    reserved = reserved[:]",
        FAST,
        "test_a_reservation_never_exceeds_the_authorized_batch",
    ),
    (
        "reservation_displaces_from_the_head",
        LIBRARY,
        "    merged = reserved + chosen[:keep]",
        "    merged = reserved + chosen[len(chosen) - keep :]",
        FAST,
        "test_a_reserved_continuation_displaces_from_the_tail_not_the_head",
    ),
    (
        "window_expires_one_round_early",
        LIBRARY,
        "            if horizon is None or round_index > horizon:",
        "            if horizon is None or round_index >= horizon:",
        FAST,
        "test_a_one_round_window_covers_exactly_the_round_after_the_bridge_is_charged",
    ),
    (
        "window_never_expires",
        LIBRARY,
        "            if horizon is None or round_index > horizon:",
        "            if False:",
        FAST,
        "test_a_one_round_window_covers_exactly_the_round_after_the_bridge_is_charged",
    ),
    (
        "restore_accepts_a_lengthened_window",
        LIBRARY,
        (
            "            if horizon is not None and (\n"
            "                opened is None or int(horizon) > int(opened) + option.protection_rounds\n"
            "            ):"
        ),
        "            if False:",
        FAST,
        "test_restore_refuses_a_window_lengthened_in_the_payload",
    ),
    (
        "frontier_moves_backwards",
        LIBRARY,
        (
            "            if nxt >= len(option.stages) or endpoint != option.stages[nxt].endpoint:\n"
            "                continue"
        ),
        (
            "            nxt = next(\n"
            "                (i for i, s in enumerate(option.stages) if s.endpoint == endpoint), -1\n"
            "            )\n"
            "            if nxt < 0:\n"
            "                continue"
        ),
        FAST,
        "test_the_frontier_only_moves_forward",
    ),
    (
        "protection_becomes_a_blanket_exemption",
        LIBRARY,
        "            if record.status == BRIDGE_CHARGED and record.frontier >= 0\n        }",
        "            if record.frontier >= 0\n        }",
        FAST,
        "test_a_chain_that_expires_mid_way_is_terminal",
    ),
    (
        "allocation_never_reserves",
        CONTROLLER,
        (
            "        merged, reservation = reserve_continuation_slots(\n"
            "            chosen, candidates, reserved, limit=limit\n"
            "        )"
        ),
        (
            "        merged, reservation = chosen, {\n"
            '            "reserved_available": 0,\n'
            '            "reserved_already_chosen": 0,\n'
            '            "reserved_added": 0,\n'
            '            "displaced": 0,\n'
            "        }"
        ),
        WIRING,
        "test_the_production_path_reaches_both_protection_hooks",
    ),
    (
        "selection_never_floors",
        CONTROLLER,
        "        lifted, _ = protected_parent_weights(",
        "        lifted, _ = (weights, None) if True else protected_parent_weights(",
        WIRING,
        "test_the_production_path_reaches_both_protection_hooks",
    ),
    (
        "legs_offered_without_a_drawn_parent",
        CONTROLLER,
        "            if stage.endpoint in archive_seen or stage.parent_endpoint not in drawn_parents:",
        "            if stage.endpoint in archive_seen:",
        WIRING,
        "test_a_continuation_is_not_offered_from_a_parent_the_round_did_not_draw",
    ),
    (
        "stored_leg_endpoint_drift_tolerated",
        CONTROLLER,
        '            raise ValueError("declared macro option leg no longer reproduces its endpoint")',
        "            pass",
        WIRING,
        "test_a_stored_leg_that_no_longer_reproduces_its_endpoint_is_refused",
    ),
    (
        "resume_silently_changes_the_arm",
        CONTROLLER,
        (
            "        if bool(state[\"enabled\"]) != result.enable_macro_options or bool(\n"
            "            state[\"protection_enabled\"]\n"
            "        ) != result.macro_option_protection:"
        ),
        "        if False:",
        WIRING,
        "test_a_resume_refuses_to_change_the_arm",
    ),
    (
        "channel_tag_not_written_at_synthesis",
        CONTROLLER,
        '        candidate["provenance"]["entry_channel"] = MACRO_OPTION_CHANNEL_TAG',
        '        candidate["provenance"]["entry_channel"] = "something_else"',
        WIRING,
        "test_every_offered_leg_carries_its_channel_tag_at_synthesis_time",
    ),
    (
        "harness_accepts_a_geometry_the_reservation_does_not_reach",
        ARMS,
        "    if int(queries_per_round) != int(config.candidates_per_batch):",
        "    if False:",
        WIRING,
        "test_the_harness_refuses_a_geometry_where_the_reservation_is_not_the_binding_gate",
    ),
    (
        "options_may_grow_instead_of_pruning_first",
        CONTROLLER,
        "                if prune_first and index == 0 and produced >= origin_atoms:",
        "                if False:",
        WIRING,
        "test_a_declared_option_prunes_before_it_installs",
    ),
]

# Changes bytes and nothing semantic. It MUST stay green, or the harness is broken and
# every refusal above is uninterpretable.
POSITIVE = (
    "cosmetic_report_key_order",
    LIBRARY,
    '            "schema_version": SCHEMA,\n            "round": self._round,',
    '            "round": self._round,\n            "schema_version": SCHEMA,',
    FAST,
)


# `diagnostics/` is 531 MB; copying it per mutation would be ~10 GB of I/O for two
# files. Only these two are read by the tests, so only these two are copied.
DIAGNOSTIC_FILES = (
    "diagnostics/parent_edit_cycles/prepared/init_20260921.json",
    "diagnostics/pmo_joint_dependency_jump_gate_v1/attempt_2/checkpoints.json",
)


def _tree(root: Path) -> Path:
    folder = Path(tempfile.mkdtemp(prefix="macro_mutation_"))
    for name in ("src", "tests", "pyproject.toml"):
        source = root / name
        if not source.exists():
            continue
        if source.is_dir():
            shutil.copytree(source, folder / name, symlinks=True)
        else:
            shutil.copy2(source, folder / name)
    for relative in DIAGNOSTIC_FILES:
        destination = folder / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(root / relative, destination)
    return folder


def _apply(tree: Path, relative: str, old: str, new: str) -> None:
    path = tree / relative
    text = path.read_text()
    if text.count(old) != 1:
        # A mutation that does not apply exactly once is a phantom: it would score as
        # killed or survived without ever having existed.
        raise RuntimeError(
            f"mutation does not apply exactly once to {relative} "
            f"(found {text.count(old)} matches)"
        )
    path.write_text(text.replace(old, new))


def _run(tree: Path, test_file: str, named: str | None) -> dict:
    target = f"{test_file}::{named}" if named else test_file
    began = perf_counter()
    completed = subprocess.run(
        [sys.executable, "-m", "pytest", target, "-q", "--no-header", "-x"],
        cwd=tree,
        capture_output=True,
        text=True,
        check=False,
        env={
            "PATH": "/usr/bin:/bin",
            "PYTHONPATH": "src",
            "KMP_DUPLICATE_LIB_OK": "TRUE",
            "OMP_NUM_THREADS": "1",
            "HOME": str(Path.home()),
        },
        timeout=1800,
    )
    return {
        "returncode": completed.returncode,
        "seconds": round(perf_counter() - began, 1),
        "tail": completed.stdout.strip().splitlines()[-3:],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--only", default=None, help="substring filter on mutation name")
    parser.add_argument(
        "--suite",
        choices=("all", "library", "wiring"),
        default="all",
        help="library mutations run in under a second; wiring mutations run a campaign",
    )
    arguments = parser.parse_args()
    root = Path.cwd()

    rows, killed, survived = [], 0, 0
    wanted = {"library": FAST, "wiring": WIRING}.get(arguments.suite)
    for name, relative, old, new, test_file, named in NEGATIVES:
        if arguments.only and arguments.only not in name:
            continue
        if wanted and test_file != wanted:
            continue
        tree = _tree(root)
        try:
            _apply(tree, relative, old, new)
            result = _run(tree, test_file, named)
        finally:
            shutil.rmtree(tree, ignore_errors=True)
        dead = result["returncode"] != 0
        killed += dead
        survived += not dead
        rows.append(
            {
                "mutation": name,
                "file": relative,
                "test": f"{test_file}::{named}",
                "killed": dead,
                **result,
            }
        )
        print(f"{'KILLED ' if dead else 'SURVIVED'} {name} ({result['seconds']}s)", flush=True)

    control = None
    if not arguments.only and arguments.suite in ("all", "library"):
        name, relative, old, new, test_file = POSITIVE
        tree = _tree(root)
        try:
            _apply(tree, relative, old, new)
            result = _run(tree, test_file, None)
        finally:
            shutil.rmtree(tree, ignore_errors=True)
        control = {"mutation": name, "file": relative, "passed": result["returncode"] == 0, **result}
        print(f"{'PASSED ' if control['passed'] else 'FAILED '} control {name}", flush=True)

    payload = {
        "schema_version": "pmo_macro_option_mutation_battery_v1",
        "suite": arguments.suite,
        "negatives": len(rows),
        "killed": killed,
        "survived": survived,
        "control": control,
        "harness_trustworthy": bool(control and control["passed"]) if control else None,
        "rows": rows,
    }
    arguments.out.parent.mkdir(parents=True, exist_ok=True)
    arguments.out.write_text(json.dumps(payload, indent=1, sort_keys=True))
    print(json.dumps({k: v for k, v in payload.items() if k != "rows"}, indent=1))
    return 0 if survived == 0 and (control is None or control["passed"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
