"""Mutation battery for the PMO discovery allocator (arm C).

A passing suite is not evidence that its guards are load-bearing. This breaks the
production code one mutation at a time and requires a NAMED test to go red for each.

Three properties this runner has, each paid for by a defect recorded in
`.claude/context/learnings.md`:

1. IT VERIFIES THE MUTATION APPLIED. A mutation string that does not match its target
   is a silent no-op, and the unmutated file then passes and reads as a weak guard.
   Every mutation diffs the file and the run ABORTS rather than reporting a verdict if
   nothing changed.
2. A SURVIVED VERDICT ASSERTS AN EMPTY FAILING LIST. "Everything refused" is
   indistinguishable from "the harness is broken" unless the negative is recorded
   explicitly, so each row carries the tests that actually failed.
3. IT CARRIES A COSMETIC POSITIVE CONTROL. A mutation that changes bytes and nothing
   else MUST stay green. Without it, a harness that fails everything looks like a
   perfect battery.

Run:  KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 PYTHONPATH=src python \
        scripts/pmo_discovery_mutation_battery.py
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CREDIT = ROOT / "src/compose_v4/control/pmo_discovery.py"
CONTROLLER = ROOT / "src/compose_v4/control/pmo_population_controller.py"
SUITE = "tests/test_pmo_discovery.py"


@dataclass(frozen=True)
class Mutation:
    name: str
    path: Path
    old: str
    new: str
    #: Tests expected to go red. Empty means this is a positive control that must stay
    #: green -- the harness check, not a guard check.
    expect_red: tuple[str, ...]

    @property
    def is_control(self) -> bool:
        return not self.expect_red


MUTATIONS = (
    Mutation(
        name="floor_reverts_to_uniform_over_cells",
        path=CREDIT,
        old="""    explore = np.asarray(
        [1.0 / (n_basins * cells_per_basin[key.basin]) for key in keys], dtype=float
    )""",
        new="""    explore = np.full(len(keys), 1.0 / len(keys))""",
        expect_red=(
            "test_the_exploration_floor_is_captured_by_whoever_opens_more_cells",
            "test_every_basin_keeps_its_floor_under_an_arbitrarily_attractive_exploit",
        ),
    ),
    Mutation(
        name="floor_multiplied_into_credit_instead_of_added",
        path=CREDIT,
        old="    mixed = (1.0 - floor) * credit + floor * explore",
        new="    mixed = credit * (1.0 - floor + floor * explore)",
        expect_red=(
            "test_every_basin_keeps_its_floor_under_an_arbitrarily_attractive_exploit",
        ),
    ),
    Mutation(
        name="cell_value_drops_the_frontier_term",
        path=CREDIT,
        old="        return base + self.config.frontier_weight * frontier",
        new="        return base",
        expect_red=("test_frontier_evidence_sits_beside_parent_relative_and_raises_the_value",),
    ),
    Mutation(
        name="discovery_eligibility_counted_at_the_cell_not_the_basin",
        path=CREDIT,
        old="        return self.basin_trials().get(key.basin, 0) <= self.config.basin_trial_threshold",
        new="        return self.cell(key).trials <= self.config.basin_trial_threshold",
        expect_red=("test_a_basin_cannot_look_thin_by_opening_fresh_cells",),
    ),
    Mutation(
        name="quota_rounds_down_instead_of_up",
        path=CREDIT,
        old="    return min(room, max(1, math.ceil(fraction * room)))",
        new="    return min(room, int(fraction * room))",
        expect_red=("test_discovery_quota_rounds_up_and_never_exceeds_room",),
    ),
    Mutation(
        name="frontier_gain_accepts_a_negative",
        path=CREDIT,
        old="""        if value < 0.0:
            raise ValueError("a frontier gain is non-negative by construction")""",
        new="""        value = max(0.0, value)""",
        expect_red=("test_a_frontier_gain_must_be_finite_and_non_negative",),
    ),
    Mutation(
        name="restore_accepts_a_plain_credit_payload",
        path=CREDIT,
        old="""        if payload.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("discovery credit snapshot has an unexpected schema")""",
        new="""        pass""",
        expect_red=(
            "test_snapshot_round_trips_and_refuses_a_plain_credit_payload",
            "test_restore_refuses_to_change_the_arm_in_either_direction",
        ),
    ),
    # ---- Controller-side: the consumption guards ----
    Mutation(
        name="controller_drops_the_observe_frontier_call",
        path=CONTROLLER,
        old="""                    if self.enable_discovery and self.online_memory is not None:
                        self.credit.observe_frontier(""",
        new="""                    if False and self.online_memory is not None:
                        self.credit.observe_frontier(""",
        expect_red=("test_observe_batch_writes_frontier_credit_on_the_production_path",),
    ),
    Mutation(
        name="controller_never_reserves_a_discovery_slot",
        path=CONTROLLER,
        old="""        reserved = (
            discovery_quota(room, fraction=self.credit.config.discovery_fraction)
            if self.enable_discovery
            else 0
        )""",
        new="""        reserved = 0""",
        expect_red=("test_the_discovery_reservation_is_consumed_and_reaches_a_thin_basin",),
    ),
    Mutation(
        name="controller_ignores_the_reservation_when_drawing",
        path=CONTROLLER,
        old="""                if eligible:
                    pool, role = eligible, "discovery\"""",
        new="""                if False:
                    pool, role = eligible, "discovery\"""",
        expect_red=("test_the_discovery_reservation_is_consumed_and_reaches_a_thin_basin",),
    ),
    Mutation(
        name="controller_allows_discovery_without_the_memory",
        path=CONTROLLER,
        old="""        if enable_discovery and not enable_online_memory:""",
        new="""        if False:""",
        expect_red=("test_arm_c_requires_arm_b",),
    ),
    Mutation(
        name="controller_builds_a_plain_allocator_for_arm_c",
        path=CONTROLLER,
        old="            self.credit = DiscoveryCredit(config=config)",
        new="            self.credit = PopulationCredit()",
        expect_red=("test_the_controller_actually_constructs_the_discovery_allocator",),
    ),
    # ---- Positive control: bytes change, behaviour does not ----
    Mutation(
        name="POSITIVE_CONTROL_cosmetic_comment",
        path=CREDIT,
        old="DEFAULT_DISCOVERY_FRACTION = 0.25",
        new="DEFAULT_DISCOVERY_FRACTION = 0.25  # cosmetic mutation: behaviour unchanged",
        expect_red=(),
    ),
)

_FAILED = re.compile(r"^(?:FAILED|ERROR) [^:]+::([\w\[\]\-.]+)", re.MULTILINE)


def run_suite() -> tuple[int, set[str]]:
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", SUITE, "-q", "--no-header", "-p", "no:randomly"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    out = proc.stdout + proc.stderr
    # Strip a parametrize suffix so a mutation may name the test function alone.
    failed = {name.split("[")[0] for name in _FAILED.findall(out)}
    return proc.returncode, failed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    baseline_code, baseline_failed = run_suite()
    if baseline_code != 0:
        print(f"ABORT: the unmutated suite is not green ({sorted(baseline_failed)})")
        return 2
    print(f"baseline: {SUITE} green\n")

    rows, verdict_ok = [], True
    originals = {path: path.read_text() for path in {m.path for m in MUTATIONS}}
    try:
        for mutation in MUTATIONS:
            source = originals[mutation.path]
            occurrences = source.count(mutation.old)
            if occurrences != 1:
                print(f"ABORT: {mutation.name} matched {occurrences} sites, expected 1")
                return 2
            mutated = source.replace(mutation.old, mutation.new)
            # Property 1: the mutation must have changed the file, or the row that
            # follows is measuring the ORIGINAL code and means nothing.
            if mutated == source:
                print(f"ABORT: {mutation.name} did not change {mutation.path.name}")
                return 2
            mutation.path.write_text(mutated)
            try:
                code, failed = run_suite()
            finally:
                mutation.path.write_text(source)

            expected = set(mutation.expect_red)
            if mutation.is_control:
                killed = code == 0 and not failed
                status = "GREEN (correct)" if killed else "RED (HARNESS BROKEN)"
            else:
                # Property 2: a kill is a NON-EMPTY failing list that contains every
                # test named for it.
                killed = bool(failed) and expected.issubset(failed)
                status = "KILLED" if killed else "SURVIVED"
            verdict_ok &= killed
            rows.append(
                {
                    "mutation": mutation.name,
                    "file": mutation.path.name,
                    "is_positive_control": mutation.is_control,
                    "expected_red": sorted(expected),
                    "observed_red": sorted(failed),
                    "status": status,
                }
            )
            print(f"  {status:<22} {mutation.name}  (red: {len(failed)})")
    finally:
        for path, text in originals.items():
            path.write_text(text)

    guards = [r for r in rows if not r["is_positive_control"]]
    report = {
        "schema_version": "pmo_discovery_mutation_battery_v1",
        "suite": SUITE,
        "mutations": len(guards),
        "killed": sum(r["status"] == "KILLED" for r in guards),
        "positive_controls_green": all(
            r["status"].startswith("GREEN") for r in rows if r["is_positive_control"]
        ),
        "all_guards_load_bearing": verdict_ok,
        "rows": rows,
    }
    print(
        f"\n{report['killed']}/{report['mutations']} guard mutations killed; "
        f"positive control green: {report['positive_controls_green']}"
    )
    if args.out:
        args.out.write_text(json.dumps(report, indent=2) + "\n")
        print(f"wrote {args.out}")
    return 0 if verdict_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
