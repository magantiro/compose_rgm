#!/usr/bin/env python
"""Prove the completion-repair guards are load-bearing.

Each NEGATIVE mutation must make a named test FAIL; each POSITIVE control must
leave the suite green.  The runner DIFFS the file after every substitution and
refuses to report a verdict when a mutation did not apply -- a mutation string
that silently fails to match reads exactly like a surviving mutant.

ZERO ORACLE CALLS.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PY_BIN = os.path.expanduser("~/compose_pmo_pinned_env/bin/python")
TEST = "tests/test_completion_component_law.py"

DPS = "src/compose_v4/control/dynamic_program_synthesis.py"
V21 = "src/compose_v4/control/dynamic_program_synthesis_v21.py"
LAW = "src/compose_v4/control/completion_component_law.py"
CONTRACT = "src/compose_v4/control/completion_law_contract.py"
PMO = "src/compose_v4/control/pmo_population_controller.py"

NEGATIVES = [
    ("optimizer_mutate_drops_the_law", DPS,
     "            max_blocks=self.config.max_blocks,\n"
     '            completion_law=getattr(self, "completion_law", None),\n        )',
     "            max_blocks=self.config.max_blocks,\n        )"),
    ("v21_channel_drops_the_law", V21,
     "                    panel_cache=self._v21_panel_cache,\n"
     '                    completion_law=getattr(self, "completion_law", None),\n'
     "                )",
     "                    panel_cache=self._v21_panel_cache,\n                )"),
    ("synthesis_drops_the_law_at_the_module", DPS,
     "                    region_law=region_law,\n"
     "                    completion_law=completion_law,\n                )",
     "                    region_law=region_law,\n                )"),
    ("segment_grow_ignores_the_law", DPS,
     "        if completion_law is not None:\n"
     "            actions, _product, parameters = completion_law.complete(source, rng)\n"
     "            return _execute_actions(source, family, actions, parameters)\n",
     "        if False:\n"
     "            actions, _product, parameters = completion_law.complete(source, rng)\n"
     "            return _execute_actions(source, family, actions, parameters)\n"),
    ("segment_replace_ignores_the_law", DPS,
     "        if completion_law is not None:\n"
     "            grow_actions, _product, completion = completion_law.complete(",
     "        if False:\n"
     "            grow_actions, _product, completion = completion_law.complete("),
    ("excision_bound_stays_at_the_v1_cap", DPS,
     "                MAX_SEGMENT_LENGTH if excision_law is None else source.n_real_atoms - 1",
     "                MAX_SEGMENT_LENGTH"),
    ("region_law_filters_instead_of_reranking", LAW,
     "        out[out <= 0.0] = 1.0 / max(1, len(regions))",
     "        out[sizes <= 2] = 0.0"),
    ("expanded_scale_falls_back_to_the_v1_bound", LAW,
     "        if self.maximum is None:\n            return scale_balanced_size(rng, capacity)",
     "        if self.maximum is None:\n"
     "            return bounded_uniform_size(rng, capacity, maximum=8)"),
    ("octave_bands_ignore_capacity", LAW,
     "        (low, min(high, capacity)) for low, high in OCTAVE_BANDS if low <= capacity",
     "        (low, high) for low, high in OCTAVE_BANDS if low <= capacity"),
    ("bank_order_ignores_capacity", LAW,
     "        usable = [size for size in self.sizes if size <= capacity]",
     "        usable = list(self.sizes)"),
    ("component_install_skips_ring_closure", LAW,
     "    for a, b, bond in spec.bonds:\n        if frozenset((a, b)) in tree_edges:\n            continue",
     "    for a, b, bond in spec.bonds:\n        if True:\n            continue"),
    ("no_hydrogen_is_reserved_at_insertion", LAW,
     "            spec.hydrogens[node] + reserved,",
     "            spec.hydrogens[node],"),
    ("bank_keeps_charged_components", LAW,
     '            if any(spec.charges):\n                census["charged_component"] += 1\n                continue',
     '            if False:\n                census["charged_component"] += 1\n                continue'),
    ("registered_arm_resolves_to_off", CONTRACT,
     "    if declared == SCALE_ONLY_V1:",
     "    if declared == SCALE_ONLY_V1 or True:\n        return None\n    if False:"),
    ("content_arm_loses_its_bank", CONTRACT,
     "            bank=load_bank(bank_path),\n            expand_excision=False,",
     "            bank=None,\n            expand_excision=False,"),
    ("restore_drops_the_arm", PMO,
     "        enable_online_memory: bool = False,\n        completion_law: Any = None,\n    ):",
     "        enable_online_memory: bool = False,\n    ):"),
    ("consumption_probe_raises_a_swallowed_exception", CONTRACT,
     'class _CompletionLawProbe(Exception):',
     'class _CompletionLawProbe(ValueError):'),
]

POSITIVES = [
    ("cosmetic_comment_only", LAW,
     "SCHEMA_VERSION = ", "# cosmetic, semantics unchanged\nSCHEMA_VERSION = "),
    ("cosmetic_docstring_whitespace", CONTRACT,
     "SCHEMA_VERSION = ", "\nSCHEMA_VERSION = "),
]


def _run(tree: str) -> tuple[bool, str]:
    env = {
        **os.environ,
        "KMP_DUPLICATE_LIB_OK": "TRUE",
        "OMP_NUM_THREADS": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    result = subprocess.run(
        [PY_BIN, "-m", "pytest", TEST, "-x", "-q", "--no-header", "-p", "no:cacheprovider"],
        cwd=tree, env=env, capture_output=True, text=True, timeout=1800,
    )
    return result.returncode == 0, (result.stdout + result.stderr)[-1500:]


def _failed_names(output: str) -> list[str]:
    return sorted(
        {
            line.split("::")[-1].split()[0]
            for line in output.splitlines()
            if line.startswith("FAILED") and "::" in line
        }
    )


def main() -> None:
    with tempfile.TemporaryDirectory() as workspace:
        tree = os.path.join(workspace, "tree")
        shutil.copytree(
            REPO, tree,
            ignore=shutil.ignore_patterns(
                ".git", "__pycache__", "*.pyc", ".venv", "paper*", "archive",
                "results", "upload", "third_party", "containers",
            ),
        )
        clean, output = _run(tree)
        if not clean:
            raise SystemExit(f"baseline suite is not green; refusing to run:\n{output}")
        print("baseline: GREEN")

        records = []
        for name, path, old, new in NEGATIVES + [
            (n, p, o, w) for n, p, o, w in POSITIVES
        ]:
            expect_fail = any(name == row[0] for row in NEGATIVES)
            full = os.path.join(tree, path)
            before = open(full).read()
            if old not in before:
                raise SystemExit(
                    f"MUTATION DID NOT APPLY: {name} -- pattern absent from {path}. "
                    "Refusing to report a verdict."
                )
            after = before.replace(old, new, 1)
            if after == before:
                raise SystemExit(f"MUTATION IS A NO-OP: {name}")
            open(full, "w").write(after)
            try:
                green, output = _run(tree)
            finally:
                open(full, "w").write(before)
            killed = (not green) if expect_fail else green
            records.append({
                "mutation": name, "file": path,
                "kind": "negative" if expect_fail else "positive_control",
                "suite_green": green,
                "verdict": "KILLED" if (expect_fail and not green)
                else ("SURVIVED" if expect_fail else ("PASSED" if green else "BROKE")),
                "failing_tests": _failed_names(output),
            })
            print(f"  {records[-1]['verdict']:9s} {name}"
                  f"  {records[-1]['failing_tests'][:3]}", flush=True)

    negatives = [r for r in records if r["kind"] == "negative"]
    positives = [r for r in records if r["kind"] == "positive_control"]
    survived = [r["mutation"] for r in negatives if r["verdict"] != "KILLED"]
    broke = [r["mutation"] for r in positives if r["verdict"] != "PASSED"]
    report = {
        "schema_version": "pmo_completion_mutation_battery_v1",
        "new_oracle_calls": 0,
        "negatives": len(negatives),
        "killed": sum(1 for r in negatives if r["verdict"] == "KILLED"),
        "survived": survived,
        "positive_controls": len(positives),
        "positive_controls_broken": broke,
        "records": records,
    }
    out = "diagnostics/pmo_completion_repair_v1/mutation_battery_v1.json"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(os.path.join(REPO, out), "w") as handle:
        json.dump(report, handle, sort_keys=True, indent=1)
    print(f"\n{report['killed']}/{report['negatives']} negatives killed; "
          f"{len(positives) - len(broke)}/{len(positives)} positive controls passed")
    if survived:
        print("SURVIVED:", survived)
    if broke:
        print("BROKEN CONTROLS:", broke)
    sys.exit(1 if (survived or broke) else 0)


main()
