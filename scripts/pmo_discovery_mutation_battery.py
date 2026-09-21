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
V21 = ROOT / "src/compose_v4/control/dynamic_program_synthesis_v21.py"
SUITE = "tests/test_pmo_discovery.py"
IDENTITY_SUITE = "tests/test_pmo_proposal_scoring_identity.py"
TRANSPORT = ROOT / "src/compose_v4/control/pmo_transport_correspondence.py"
TRANSPORT_SUITE = "tests/test_pmo_transport_correspondence.py"


@dataclass(frozen=True)
class Mutation:
    name: str
    path: Path
    old: str
    new: str
    #: Tests expected to go red. Empty means this is a positive control that must stay
    #: green -- the harness check, not a guard check.
    expect_red: tuple[str, ...]
    #: The suite to run. The proposal/scoring identity guards drive the real proposal
    #: path and cost ~70 s a run, so only they pay that.
    suite: str = SUITE

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
    # ---- The T4 indirection, reproduced on the PMO path ----
    # `t4_fiber_campaign.expand` gates a molecule re-instantiated from an abstracted
    # goal rather than the one its program produced (measured recovered_fraction
    # 0.0000). PMO measures 1.0000. This mutation makes the candidate record carry a
    # molecule the program did not produce, which is exactly that defect class, and
    # requires the identity guards to catch it.
    Mutation(
        name="candidate_records_a_molecule_the_program_did_not_produce",
        path=V21,
        old="""        "assignment": list(binding),
        "endpoint": trace["endpoint"],""",
        new="""        "assignment": list(binding),
        "endpoint": "CCO",""",
        expect_red=(
            "test_every_proposal_endpoint_is_its_own_executed_program_endpoint",
            "test_the_scored_archive_records_the_produced_molecule",
        ),
        suite=IDENTITY_SUITE,
    ),
    # ---- Step 1: the transport correspondence ----
    Mutation(
        name="core_is_not_pruned_to_ring_complete",
        path=TRANSPORT,
        old="""        if not drop:
            return current
        current -= drop""",
        new="""        return current
        current -= drop""",
        expect_red=(
            "test_the_core_is_ring_complete_in_both_endpoints",
            "test_a_lone_ring_atom_is_pruned_out_of_the_core",
        ),
        suite=TRANSPORT_SUITE,
    ),
    Mutation(
        name="pruning_does_one_pass_instead_of_reaching_a_fixed_point",
        path=TRANSPORT,
        old="""        if not drop:
            return current
        current -= drop""",
        new="""        if not drop:
            return current
        return current - drop""",
        expect_red=("test_pruning_reaches_a_fixed_point_not_just_one_pass",),
        suite=TRANSPORT_SUITE,
    ),
    Mutation(
        name="scale_ignores_bond_order_changes",
        path=TRANSPORT,
        old="        return len(self.r_delete) + len(self.h_install) + len(self.core_bond_changes)",
        new="        return len(self.r_delete) + len(self.h_install)",
        expect_red=(
            "test_a_bond_order_only_difference_is_counted_as_transport",
            "test_zero_scale_is_reserved_for_genuinely_identical_molecules",
        ),
        suite=TRANSPORT_SUITE,
    ),
    Mutation(
        name="attachment_enumerates_only_the_source_side",
        path=TRANSPORT,
        old="""                        _boundary(source, kept_source, set(delete), "source")
                        + _boundary(target, kept_target, set(install), "target")""",
        new="""                        _boundary(source, kept_source, set(delete), "source")""",
        expect_red=("test_attachment_interfaces_are_enumerated_on_both_sides",),
        suite=TRANSPORT_SUITE,
    ),
    Mutation(
        # Reproduces the defect a tautological assertion had been hiding: an in-set
        # degree peel ignores how an atom connects to the retained core.
        name="deletion_order_reverts_to_an_in_set_degree_peel",
        path=TRANSPORT,
        old="""        ranked = sorted(
            remaining, key=lambda index: (len(adjacency[index] & surviving), index)
        )
        pick = next((index for index in ranked if connected_without(index)), ranked[0])""",
        new="""        ranked = sorted(
            remaining, key=lambda index: (len(adjacency[index] & remaining), index)
        )
        pick = ranked[0]""",
        expect_red=("test_deleting_in_the_recorded_order_keeps_every_intermediate_connected",),
        suite=TRANSPORT_SUITE,
    ),
    Mutation(
        # Reverts to the measured-wrong global ordering that ignores what already exists.
        name="install_order_ignores_what_already_exists",
        path=TRANSPORT,
        old="        pick = adjacent[0] if adjacent else min(pending)",
        new="        pick = min(pending)",
        expect_red=("test_install_order_grows_outward_from_what_already_exists",),
        suite=TRANSPORT_SUITE,
    ),
    # NOTE: a mutation taking one ring atom at a time instead of the whole system was
    # tried and SURVIVED, and that is a finding rather than a gap. Under anchored growth
    # the next adjacent atom is a ring neighbour anyway, so the two orders coincide and
    # the explicit grouping in `_growth_order` is redundant WITH IT. Ring integrity at a
    # STAGE BOUNDARY is a different guard, enforced by `_atomic_groups` in
    # `pmo_transport_staging` and exercised by the staging validation.
    Mutation(
        name="staging_splits_a_ring_across_a_stage_boundary",
        path=ROOT / "src/compose_v4/control/pmo_transport_staging.py",
        old="        if current and len(current) + len(unit) > max_primitives:",
        new="        if current and len(current) + 0 > max_primitives:",
        expect_red=("test_no_stage_exceeds_the_ceiling",),
        suite="tests/test_pmo_transport_staging.py",
    ),
    Mutation(
        name="unparseable_input_returns_empty_instead_of_raising",
        path=TRANSPORT,
        old="""    if source is None:
        raise ValueError(f"source does not parse: {source_smiles!r}")""",
        new="""    if source is None:
        return []""",
        expect_red=("test_unparseable_input_raises_rather_than_returning_no_core",),
        suite=TRANSPORT_SUITE,
    ),
    Mutation(
        name="disconnected_core_is_never_declared",
        path=TRANSPORT,
        old="        return seen != core",
        new="        return False",
        expect_red=("test_a_disconnected_core_is_declared_as_requiring_reattachment",),
        suite=TRANSPORT_SUITE,
    ),
    Mutation(
        # The predicate must DISCRIMINATE: one that always fires also passes the
        # positive case, so the negative control is what makes the guard real.
        name="every_core_is_declared_disconnected",
        path=TRANSPORT,
        old="        return seen != core",
        new="        return True",
        expect_red=("test_a_connected_core_is_not_flagged_for_reattachment",),
        suite=TRANSPORT_SUITE,
    ),
    Mutation(
        name="trajectory_omits_the_source_so_a_dip_is_invisible",
        path=ROOT / "src/compose_v4/control/pmo_transport_staging.py",
        old="    similarities = [source_similarity] if source_similarity is not None else []",
        new="    similarities = []",
        expect_red=("test_a_staged_transport_can_dip_below_its_own_starting_similarity",),
        suite="tests/test_pmo_transport_staging.py",
    ),
    Mutation(
        name="interleaving_is_ignored_and_always_prunes_first",
        path=ROOT / "src/compose_v4/control/pmo_transport_staging.py",
        old="    if interleave:",
        new="    if False:",
        expect_red=("test_interleaving_removes_the_dip_on_a_dipping_transport",),
        suite="tests/test_pmo_transport_staging.py",
    ),
    Mutation(
        name="dip_depth_is_collapsed_to_a_boolean",
        path=ROOT / "src/compose_v4/control/pmo_transport_staging.py",
        old="    depth = max(0.0, source_similarity - min(after))",
        new="    depth = 0.0",
        expect_red=("test_dip_depth_and_width_are_reported_not_just_a_boolean",),
        suite="tests/test_pmo_transport_staging.py",
    ),
    Mutation(
        name="selector_scores_inadmissible_orderings",
        path=ROOT / "src/compose_v4/control/pmo_transport_staging.py",
        old="    admissible = [c for c in candidates if c.admissible]",
        new="    admissible = list(candidates)",
        expect_red=("test_admissibility_is_a_hard_filter_not_a_term_in_the_score",),
        suite="tests/test_pmo_transport_staging.py",
    ),
    # THE SHAPE KEY'S TERMS ARE REDUNDANT ON REAL DATA, and two mutations established it
    # rather than a gap: a dip-free ordering has depth 0.0 AND width 0, so either term
    # alone prefers it, and no measured pair offers two DIPPING admissible candidates of
    # different depth. A separate "does it dip" boolean was therefore REMOVED as
    # decoration. What remains decidable is the tie-break ORDER between depth and width,
    # which this mutation targets against constructed candidates.
    Mutation(
        name="shape_key_ranks_width_before_depth",
        path=ROOT / "src/compose_v4/control/pmo_transport_staging.py",
        old="""        candidate.dip_depth_relative or 0.0,
        candidate.dip_width_stages or 0,""",
        new="""        candidate.dip_width_stages or 0,
        candidate.dip_depth_relative or 0.0,""",
        expect_red=(
            "test_among_dipping_orderings_the_SHALLOWER_is_preferred_over_the_narrower",
        ),
        suite="tests/test_pmo_transport_staging.py",
    ),
    Mutation(
        name="protected_rounds_is_an_open_allowance",
        path=ROOT / "src/compose_v4/control/pmo_transport_staging.py",
        old="            else int(chosen.dip_width_stages or 1)",
        new="            else 99",
        expect_red=("test_protected_rounds_is_the_trough_width_not_an_open_allowance",),
        suite="tests/test_pmo_transport_staging.py",
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


def run_suite(suite: str = SUITE) -> tuple[int, set[str]]:
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", suite, "-q", "--no-header", "-p", "no:randomly"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
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
                code, failed = run_suite(mutation.suite)
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
