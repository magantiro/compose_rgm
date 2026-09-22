"""Mutation battery for the unified-routing isolation guards.

WHY A RUNNER AND NOT JUST TESTS
-------------------------------
A guard is only tested where it BINDS.  Every assertion in
``tests/test_t4_unified_routing.py`` passes on the unmutated tree, which proves
nothing about whether it would catch a target identity or a reward leaking into the
routing decision.  This runner introduces that leak deliberately, one way at a time,
and requires the named test to go RED.

TWO THINGS THIS RUNNER DOES THAT A NAIVE ONE DOES NOT
------------------------------------------------------
* It DIFFS the file after substitution and ABORTS if the mutation did not apply.  A
  mutation written with the wrong quoting or indentation is a silent no-op, and the
  unmutated file then passes and reads as a weak guard.  That has happened in this
  repository more than once.
* It carries POSITIVE CONTROLS -- perturbations that change bytes and nothing
  semantic, which MUST stay green.  Without one, "everything refused" is
  indistinguishable from "the harness is broken".

It runs in a sandbox copy so a concurrent reader of the working tree never observes a
mutated module.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODULE = Path("src/compose_v4/control/t4_unified_routing.py")
TESTS = Path("tests/test_t4_unified_routing.py")
COPY = ("src", "tests", "configs", "diagnostics/t4_state_routing_v1")


@dataclass(frozen=True)
class Mutation:
    name: str
    target: Path
    old: str
    new: str
    expect: str          # "red" or "green"
    must_fail: str = ""  # a test name that must appear in the failure list
    why: str = ""


MUTATIONS = (
    Mutation(
        name="identity_branch_on_a_target_name",
        target=MODULE,
        old='    charged = int(applicability.net_formal_charge) != 0\n',
        new='    charged = int(applicability.net_formal_charge) != 0 or "5ht1b" == "5ht1b"\n',
        expect="red",
        must_fail="test_the_module_source_names_no_target_and_no_objective_value",
        why="a protein name reaching the mask must be caught by the source sweep",
    ),
    Mutation(
        name="reward_field_on_the_search_record",
        target=MODULE,
        old="    eligible_pool_size: int\n    consecutive_empty_rounds: int\n    rounds_completed: int\n",
        new="    eligible_pool_size: int\n    consecutive_empty_rounds: int\n    rounds_completed: int\n    parent_score: float = 0.0\n",
        expect="red",
        must_fail="test_the_routing_inputs_carry_no_identity_and_no_value",
        why="a docking value on the routing input must be caught by the pinned field set",
    ),
    Mutation(
        name="identity_argument_on_the_routing_function",
        target=MODULE,
        old="def kernel_weights(\n    applicability: MolecularApplicability, search: SearchProgress\n) -> dict[str, float]:",
        new="def kernel_weights(\n    applicability: MolecularApplicability, search: SearchProgress, cell: str = \"\"\n) -> dict[str, float]:",
        expect="red",
        must_fail="test_the_routing_functions_accept_only_those_two_records",
        why="a third argument is a route by which an identity could enter",
    ),
    Mutation(
        name="per_cell_discriminator_that_no_name_sweep_would_see",
        target=MODULE,
        old="    charged = int(applicability.net_formal_charge) != 0\n    return {\"local\": 1, \"region\": int(not charged), \"state_aware\": int(charged)}",
        new="    charged = int(applicability.net_formal_charge) != 0 and int(applicability.heavy_atoms) == 31\n    return {\"local\": 1, \"region\": int(not charged), \"state_aware\": int(charged)}",
        expect="red",
        must_fail="test_states_agreeing_on_the_declared_features_agree_on_the_decision",
        why=(
            "THE IMPORTANT ONE. Keying on heavy_atoms == 31 singles out 5ht1b_2 -- the "
            "only charged cell that historically exhausted -- using a legitimate state "
            "feature and no forbidden name at all. A textual sweep cannot see it; only "
            "the behavioural guard that charged cells must decide alike can."
        ),
    ),
    Mutation(
        name="ramp_fires_on_a_healthy_pool",
        target=MODULE,
        old="    if int(search.eligible_pool_size) > 0:\n        return 0.0\n    return 1.0",
        new="    if int(search.eligible_pool_size) > 4:\n        return 0.0\n    return 1.0",
        expect="red",
        must_fail="test_the_ramp_is_zero_for_any_non_empty_pool",
        why=(
            "an alternate kernel consulted on a healthy round breaks the non-trigger "
            "guarantee. NOTE, measured: the named guard is the PARAMETRIZED small-pool "
            "test, not `test_a_healthy_pool_consults_no_alternate_kernel` -- that one "
            "probes a pool of 8 and therefore survives a threshold mutated to 4. A "
            "guard is only tested where it binds, and here the binding cases are the "
            "pools of 1 and 2."
        ),
    ),
    Mutation(
        name="kernel_choice_reversed",
        target=MODULE,
        old='    return {"local": 1, "region": int(not charged), "state_aware": int(charged)}',
        new='    return {"local": 1, "region": int(charged), "state_aware": int(not charged)}',
        expect="red",
        must_fail="test_the_blind_routing_table_is_reproduced_cell_by_cell",
        why="the regression against the committed blind-routing evidence must bind",
    ),
    Mutation(
        name="graded_signal_wired_into_the_decision",
        target=MODULE,
        old="    charged = int(applicability.net_formal_charge) != 0\n    return {\"local\": 1,",
        new="    charged = charge_refused_region_share(applicability) > 0.4\n    return {\"local\": 1,",
        expect="red",
        must_fail="test_the_graded_signal_does_not_change_any_decision",
        why="swapping the decision rule after seeing a result is what the separation prevents",
    ),
    # ---- POSITIVE CONTROLS: bytes move, semantics do not. MUST stay green. ----
    Mutation(
        name="control_comment_only",
        target=MODULE,
        old="SCHEMA_VERSION = \"t4_unified_routing_v1\"",
        new="# a comment that changes bytes and nothing else\nSCHEMA_VERSION = \"t4_unified_routing_v1\"",
        expect="green",
        why="without a control that must pass, 'everything refused' could be a broken harness",
    ),
    Mutation(
        name="control_whitespace_only",
        target=MODULE,
        old="def support_ramp(search: SearchProgress) -> float:",
        new="def support_ramp(search:  SearchProgress) -> float:",
        expect="green",
        why="a cosmetic reformat must not move any verdict",
    ),
)


def _run(sandbox: Path) -> tuple[bool, set[str]]:
    done = subprocess.run(
        [sys.executable, "-m", "pytest", str(TESTS), "-q", "--no-header", "-p", "no:cacheprovider"],
        cwd=sandbox,
        capture_output=True,
        text=True,
        check=False,
        env={
            "PATH": "/usr/bin:/bin",
            "PYTHONPATH": f"{sandbox / 'src'}:{sandbox / 'scripts'}",
            "KMP_DUPLICATE_LIB_OK": "TRUE",
            "OMP_NUM_THREADS": "1",
            "HOME": str(Path.home()),
        },
    )
    failures = {
        line.split("::")[-1].split()[0]
        for line in done.stdout.splitlines()
        if line.startswith("FAILED") and "::" in line
    }
    return done.returncode == 0, failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "diagnostics/t4_routing_isolation_mutations_v1.json")
    args = parser.parse_args()

    results = []
    with tempfile.TemporaryDirectory(prefix="t4_routing_mut_") as scratch:
        base = Path(scratch) / "tree"
        base.mkdir()
        for relative in COPY:
            source = ROOT / relative
            if source.is_dir():
                shutil.copytree(source, base / relative, dirs_exist_ok=True,
                                ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
            else:
                (base / relative).parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, base / relative)
        pristine = {mutation.target: (base / mutation.target).read_text() for mutation in MUTATIONS}

        clean_green, clean_failures = _run(base)
        if not clean_green:
            raise SystemExit(
                f"the unmutated sandbox is already red ({sorted(clean_failures)}); "
                "the battery cannot distinguish a caught mutation from a broken harness"
            )

        for mutation in MUTATIONS:
            path = base / mutation.target
            before = pristine[mutation.target]
            after = before.replace(mutation.old, mutation.new, 1)
            if after == before:
                raise SystemExit(
                    f"MUTATION {mutation.name!r} DID NOT APPLY -- its `old` text is not in "
                    f"{mutation.target}. A no-op mutation scores as 'caught' and proves "
                    "nothing; refusing to emit a verdict."
                )
            path.write_text(after)
            green, failures = _run(base)
            path.write_text(before)
            observed = "green" if green else "red"
            row = {
                "mutation": mutation.name,
                "expected": mutation.expect,
                "observed": observed,
                "killed": observed == mutation.expect,
                "failing_tests": sorted(failures),
                # Prefix match: pytest reports a parametrized case as `name[1]`, and an
                # exact match silently scored a fired guard as not fired.
                "named_guard_fired": (
                    any(name.split("[")[0] == mutation.must_fail for name in failures)
                    if mutation.must_fail
                    else None
                ),
                "why": mutation.why,
            }
            results.append(row)
            print(
                f"{row['killed'] and 'OK  ' or 'MISS'} {mutation.name:52} "
                f"expected={mutation.expect:5} observed={observed:5} "
                f"guard={'fired' if row['named_guard_fired'] else row['named_guard_fired']}",
                flush=True,
            )

    payload = {
        "schema_version": "t4_routing_isolation_mutations_v1",
        "new_oracle_calls": 0,
        "unmutated_sandbox_green": True,
        "mutations": results,
        "killed": sum(1 for row in results if row["killed"]),
        "total": len(results),
        "named_guards_all_fired": all(
            row["named_guard_fired"] for row in results if row["named_guard_fired"] is not None
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: v for k, v in payload.items() if k != "mutations"}, indent=2))
    return 0 if payload["killed"] == payload["total"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
