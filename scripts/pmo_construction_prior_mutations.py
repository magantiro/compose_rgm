"""Mutation battery for the construction-lane prior wiring.

Every hop that carries ``successor_prior`` is severed in turn and a NAMED test
must go red.  Three hops share one sink (``_grow_actions``), which is exactly the
case a single consultation test cannot catch, so each of them is severed
separately and the behavioural per-family tests are what kill them.

Two disciplines the repository has paid for:

* A mutation that does not APPLY must abort, never score as killed.  The runner
  diffs the file and refuses to emit a verdict when the text did not change.
* A CONTROL mutation that changes bytes and nothing semantic must still PASS.
  Without it, "everything refused" is indistinguishable from a broken harness.

Runs the focused wiring suite only (about three seconds), so the battery is
cheap enough to re-run on every change to the lane.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

TARGET_RELATIVE = "src/compose_v4/control/dynamic_program_synthesis.py"
SUITE_RELATIVE = "tests/test_construction_prior_wiring.py"

#: The second file the wiring spans: the controller supplies the attribute the
#: optimizer hop reads, and a break there is invisible to any test of the lane.
CONTROLLER_RELATIVE = "src/compose_v4/control/pmo_population_controller.py"

#: (name, file, old, new, expectation).  ``expectation`` is "red" for a severed
#: hop and "green" for a control that must survive.
MUTATIONS = [
    (
        "drop_prior_at_grow_sink",
        TARGET_RELATIVE,
        "    current, actions = source, []\n    anchors = [",
        "    successor_prior = None\n    current, actions = source, []\n    anchors = [",
        "red",
    ),
    (
        "drop_prior_at_segment_grow_call",
        TARGET_RELATIVE,
        'length=length,\n            elements=("C", "N", "O"),\n            successor_prior=successor_prior,',
        'length=length,\n            elements=("C", "N", "O"),',
        "red",
    ),
    (
        "drop_prior_at_functionalize_call",
        TARGET_RELATIVE,
        'length=1,\n            elements=("C", "N", "O", "F"),\n            successor_prior=successor_prior,',
        'length=1,\n            elements=("C", "N", "O", "F"),',
        "red",
    ),
    (
        "drop_prior_at_segment_replace_call",
        TARGET_RELATIVE,
        'elements=("C", "N", "O"),\n            anchor=anchor,\n            successor_prior=successor_prior,',
        'elements=("C", "N", "O"),\n            anchor=anchor,',
        "red",
    ),
    (
        "drop_prior_at_terminal_shrink_call",
        TARGET_RELATIVE,
        "requested_length=requested,\n            successor_prior=successor_prior,",
        "requested_length=requested,",
        "red",
    ),
    (
        "drop_prior_at_carbonyl_branch",
        TARGET_RELATIVE,
        "        if successor_prior is None:\n            action = carbonyls[int(rng.integers(len(anchors)))]",
        "        if True:\n            action = carbonyls[int(rng.integers(len(anchors)))]",
        "red",
    ),
    (
        "drop_prior_at_local_module_call",
        TARGET_RELATIVE,
        "        family=families[family],\n        label=family,\n        successor_prior=successor_prior,",
        "        family=families[family],\n        label=family,",
        "red",
    ),
    (
        "drop_prior_at_compile_generic_module_call",
        TARGET_RELATIVE,
        "                    region_law=region_law,\n                    successor_prior=successor_prior,",
        "                    region_law=region_law,",
        "red",
    ),
    (
        "drop_prior_at_named_sequence_call",
        TARGET_RELATIVE,
        "            region_law=region_law,\n            successor_prior=successor_prior,",
        "            region_law=region_law,",
        "red",
    ),
    (
        "drop_prior_at_optimizer_mutate",
        TARGET_RELATIVE,
        "            max_blocks=self.config.max_blocks,\n            successor_prior=self.construction_prior,",
        "            max_blocks=self.config.max_blocks,",
        "red",
    ),
    (
        "filter_the_growth_candidates_before_ranking",
        TARGET_RELATIVE,
        "        action = successor_prior.order(\n            current, rng, family=\"atom_insert\", actions=candidates\n        )[0]",
        "        candidates = candidates[: max(1, len(candidates) // 2)]\n        action = successor_prior.order(\n            current, rng, family=\"atom_insert\", actions=candidates\n        )[0]",
        "red",
    ),
    (
        "filter_the_shrink_candidates_before_ranking",
        TARGET_RELATIVE,
        "                    actions=[AtomDelete(int(v)) for v in candidates],",
        "                    actions=[AtomDelete(int(v)) for v in candidates[:1]],",
        "red",
    ),
    (
        "weight_the_family_choice_with_the_prior",
        TARGET_RELATIVE,
        "def _weighted_module_order(rng, *, near_capacity):",
        "def _weighted_module_order(rng, *, near_capacity, successor_prior=None):",
        "red",
    ),
    (
        "tag_every_module_regardless_of_the_law",
        TARGET_RELATIVE,
        "    if successor_prior is None:\n        return parameters\n    return {**parameters, \"construction_law\": CONSTRUCTION_LAW_TAG}",
        "    return {**parameters, \"construction_law\": CONSTRUCTION_LAW_TAG}",
        "red",
    ),
    (
        "break_the_off_rng_stream_in_growth",
        TARGET_RELATIVE,
        "        at = anchors[int(rng.integers(len(anchors)))]\n        for _ in range(length):",
        "        rng.integers(2)\n        at = anchors[int(rng.integers(len(anchors)))]\n        for _ in range(length):",
        "red",
    ),
    # -- CONTROL: bytes change, semantics do not.  This MUST stay green.
    (
        "controller_stops_setting_the_attribute_the_hop_reads",
        CONTROLLER_RELATIVE,
        "        self.construction_prior = construction_prior",
        "        self._construction_prior_unused = construction_prior",
        "red",
    ),
    (
        "controller_restore_drops_the_arm",
        CONTROLLER_RELATIVE,
        '                "construction_prior": construction_prior,\n',
        "",
        "red",
    ),
    (
        "CONTROL_cosmetic_comment",
        TARGET_RELATIVE,
        "SCHEMA = \"dynamic_generic_program_synthesis_v1\"",
        "# cosmetic control mutation: bytes move, behaviour does not\nSCHEMA = \"dynamic_generic_program_synthesis_v1\"",
        "green",
    ),
]


def _run_suite(python: str, root: Path) -> bool:
    result = subprocess.run(
        [python, "-m", "pytest", SUITE_RELATIVE, "-q", "-x", "--no-header"],
        check=False,
        capture_output=True,
        text=True,
        cwd=str(root),
        env={
            "PATH": "/usr/bin:/bin",
            "KMP_DUPLICATE_LIB_OK": "TRUE",
            "OMP_NUM_THREADS": "1",
            "PYTHONPATH": str(root / "src"),
            "HOME": str(Path.home()),
        },
    )
    return result.returncode == 0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument(
        "--root",
        type=Path,
        default=Path.cwd(),
        help=(
            "the tree to mutate.  Point it at a COPY: mutating the working tree "
            "while another job reads it is the load/write race this repository "
            "has already paid for twice."
        ),
    )
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    root = args.root.resolve()
    originals = {
        relative: (root / relative).read_text()
        for relative in (TARGET_RELATIVE, CONTROLLER_RELATIVE)
    }
    baseline = _run_suite(args.python, root)
    if not baseline:
        raise SystemExit("the unmutated suite is already red; fix that before mutating")

    results = []
    try:
        for name, relative, old, new, expectation in MUTATIONS:
            original = originals[relative]
            path = root / relative
            if old not in original:
                raise SystemExit(
                    f"mutation {name!r} does not apply: its anchor text is absent. "
                    "A mutation that silently fails to apply scores as killed and "
                    "proves nothing, so this aborts instead of reporting a verdict."
                )
            mutated = original.replace(old, new, 1)
            if mutated == original:
                raise SystemExit(f"mutation {name!r} changed no bytes")
            path.write_text(mutated)
            passed = _run_suite(args.python, root)
            path.write_text(original)
            observed = "green" if passed else "red"
            results.append(
                {
                    "mutation": name,
                    "file": relative,
                    "expected": expectation,
                    "observed": observed,
                    "verdict": "ok" if observed == expectation else "SURVIVED",
                }
            )
            print(f"{name}: expected {expectation}, observed {observed}")
    finally:
        for relative, text in originals.items():
            (root / relative).write_text(text)

    survivors = [row for row in results if row["verdict"] != "ok"]
    payload = {
        "schema_version": "pmo_construction_prior_mutations_v1",
        "oracle_calls_spent": 0,
        "targets": [TARGET_RELATIVE, CONTROLLER_RELATIVE],
        "suite": SUITE_RELATIVE,
        "root": str(root),
        "baseline_green": baseline,
        "mutations": results,
        "killed": sum(
            1 for r in results if r["expected"] == "red" and r["verdict"] == "ok"
        ),
        "controls_passed": sum(
            1 for r in results if r["expected"] == "green" and r["verdict"] == "ok"
        ),
        "survivors": survivors,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(json.dumps({k: payload[k] for k in
                      ("killed", "controls_passed", "survivors")}, indent=1))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
