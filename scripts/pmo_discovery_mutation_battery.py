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
DONOR = ROOT / "scripts/pmo_donor_transplant_feasibility.py"
DONOR_SUITE = "tests/test_pmo_donor_transplant_feasibility.py"
DONOR_CHANNEL_MODULE = ROOT / "src/compose_v4/control/pmo_donor_channel.py"
WIRING_SUITE = "tests/test_pmo_donor_channel_wiring.py"
SCORED_ENTRY = ROOT / "src/compose_v4/experiments/pmo_population_v1.py"
SCORED_SUITE = "tests/test_pmo_donor_scored_path.py"
WORKER_APP = ROOT / "modal_apps/pmo_population_v1_app.py"
MEMORY = ROOT / "src/compose_v4/control/pmo_online_memory.py"
BANK_SUITE = "tests/test_pmo_donor_scored_bank.py"


@dataclass(frozen=True)
class Mutation:
    name: str
    path: Path
    old: str
    new: str
    #: Tests expected to go red. Empty means this is a positive control that must stay
    #: green -- the harness check, not a guard check.
    expect_red: tuple[str, ...]
    #: Tests that must SURVIVE this mutation. This is how the two-hops-one-sink near-miss
    #: is recorded rather than remembered: when a law is consulted at two call sites, a
    #: probe that raises fires at the FIRST one, so dropping the SECOND leaves the
    #: consumption gate fully green while half the mechanism is gone. Naming the gate here
    #: asserts that blind spot exists, and so asserts that the behavioural test beside it
    #: is the thing actually doing the work.
    expect_green: tuple[str, ...] = ()
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
    # ---- Donor-transplant feasibility: the cut draw is the whole claim ----
    Mutation(
        name="retentive_arm_collapses_to_uniform",
        path=DONOR,
        old="    released = np.asarray([len(cut.component) / n_real for cut in cuts], dtype=float)\n"
            "    weights = np.maximum(0.05, np.exp(-released / 0.25))",
        new="    weights = np.ones(len(cuts), dtype=float)",
        expect_red=("test_retentive_prefers_small_releases_and_uniform_does_not",),
        suite=DONOR_SUITE,
    ),
    Mutation(
        name="retentive_floor_removed",
        path=DONOR,
        old="    weights = np.maximum(0.05, np.exp(-released / 0.25))",
        new="    weights = np.exp(-released / 0.25)",
        expect_red=("test_the_floor_actually_binds_on_a_large_release",),
        suite=DONOR_SUITE,
    ),
    Mutation(
        name="retentive_filters_instead_of_reranking",
        path=DONOR,
        old="    weights = np.maximum(0.05, np.exp(-released / 0.25))",
        new="    weights = np.where(released > 0.5, 0.0, np.exp(-released / 0.25))",
        expect_red=("test_retentive_shares_the_uniform_support_and_never_filters",),
        suite=DONOR_SUITE,
    ),
    Mutation(
        name="donor_pool_becomes_the_uncounted_prescreen_bank",
        path=DONOR,
        old='LEDGER = ROOT / "diagnostics/pmo_3x250_autopsy_v1.json"',
        new='LEDGER = ROOT / "diagnostics/pmo_banks_all.json"',
        expect_red=("test_the_ledger_is_a_charged_run_and_not_the_prescreen_bank",),
        suite=DONOR_SUITE,
    ),
    Mutation(
        name="score_key_stops_naming_the_charged_score",
        path=DONOR,
        old='SMILES_KEY, SCORE_KEY = "endpoint", "charged_score"',
        new='SMILES_KEY, SCORE_KEY = "endpoint", "score"',
        expect_red=("test_the_ledger_is_a_charged_run_and_not_the_prescreen_bank",),
        suite=DONOR_SUITE,
    ),
    Mutation(
        name="donor_pool_is_not_ranked_by_score",
        path=DONOR,
        old="    rows.sort(key=lambda r: (-r[SCORE_KEY], r[SMILES_KEY]))",
        new="    rows.sort(key=lambda r: r[SMILES_KEY])",
        expect_red=("test_ledger_rows_are_charged_scored_and_ranked",),
        suite=DONOR_SUITE,
    ),
    Mutation(
        name="retention_threshold_excludes_its_own_boundary",
        path=DONOR,
        old='"retained_at_least_half": sum(1 for v in retained if v >= 0.5),',
        new='"retained_at_least_half": sum(1 for v in retained if v > 0.5),',
        expect_red=("test_summarize_counts_retention_over_the_right_threshold",),
        suite=DONOR_SUITE,
    ),
    Mutation(
        name="POSITIVE_CONTROL_donor_cosmetic_comment",
        path=DONOR,
        old='LEDGER_PATH = ("celecoxib_distance", "all_rows")',
        new='LEDGER_PATH = ("celecoxib_distance", "all_rows")  # cosmetic: behaviour unchanged',
        expect_red=(),
        suite=DONOR_SUITE,
    ),
    # ---- Donor CHANNEL wiring: the mechanism must stay reachable ----
    Mutation(
        name="lane_stops_forwarding_the_law_to_the_draw",
        path=CONTROLLER,
        old="                    source, donors, self.donor_rng, law=self.donor_law",
        new="                    source, donors, self.donor_rng, law=None",
        expect_red=("test_the_donor_law_is_consumed_by_the_production_propose_batch",),
        suite=WIRING_SUITE,
    ),
    Mutation(
        name="propose_batch_never_calls_the_donor_lane",
        path=CONTROLLER,
        old="        if self.enable_donor_channel:\n"
            "            donor_attempts, donor_rows, donor_seconds = self._generate_donor_pool(",
        new="        if False:\n"
            "            donor_attempts, donor_rows, donor_seconds = self._generate_donor_pool(",
        expect_red=("test_the_donor_law_is_consumed_by_the_production_propose_batch",),
        suite=WIRING_SUITE,
    ),
    Mutation(
        name="enabled_lane_does_not_join_the_live_channel_set",
        path=CONTROLLER,
        old="        self.channels = CHANNELS + (DONOR_CHANNEL,) if self.enable_donor_channel else CHANNELS",
        new="        self.channels = CHANNELS",
        expect_red=("test_off_is_byte_identical_and_draws_no_extra_parents",),
        suite=WIRING_SUITE,
    ),
    Mutation(
        name="off_lane_leaks_into_the_channel_set",
        path=CONTROLLER,
        old="        self.channels = CHANNELS + (DONOR_CHANNEL,) if self.enable_donor_channel else CHANNELS",
        new="        self.channels = CHANNELS + (DONOR_CHANNEL,)",
        expect_red=("test_off_is_byte_identical_and_draws_no_extra_parents",),
        suite=WIRING_SUITE,
    ),
    Mutation(
        name="conversion_admits_a_multi_bond_bridge",
        path=DONOR_CHANNEL_MODULE,
        old="    if region.bond_order != 1:\n        return None",
        new="    if region.bond_order < 1:\n        return None",
        expect_red=("test_the_conversion_recovers_root_and_refuses_a_multi_bond_bridge",),
        suite=WIRING_SUITE,
    ),
    Mutation(
        name="retentive_margin_goes_flat",
        path=DONOR_CHANNEL_MODULE,
        old="        return (kept / total) - 1.0",
        new="        return 0.0",
        expect_red=("test_the_retentive_law_reranks_and_never_filters",),
        suite=WIRING_SUITE,
    ),
    Mutation(
        name="probe_exception_becomes_catchable_by_the_lane",
        path=DONOR_CHANNEL_MODULE,
        old="class DonorLawProbe(BaseException):",
        new="class DonorLawProbe(RuntimeError):",
        expect_red=("test_the_probe_exception_survives_the_lane_s_own_handler",),
        suite=WIRING_SUITE,
    ),
    Mutation(
        name="donor_lane_shares_the_jump_rng_stream",
        path=CONTROLLER,
        old="            np.random.default_rng(np.random.SeedSequence([self.config.seed, 311, 3]))",
        new="            np.random.default_rng(np.random.SeedSequence([self.config.seed, 311, 2]))",
        expect_red=("test_the_donor_lane_has_its_own_rng_stream",),
        suite=WIRING_SUITE,
    ),
    Mutation(
        name="POSITIVE_CONTROL_donor_channel_cosmetic",
        path=DONOR_CHANNEL_MODULE,
        old='DONOR_CHANNEL = "donor_transplant"',
        new='DONOR_CHANNEL = "donor_transplant"  # cosmetic: behaviour unchanged',
        expect_red=(),
        suite=WIRING_SUITE,
    ),
    # ---- The SCORED path: three hops, each able to fail on its own ----
    # Every inert mechanism this repository has shipped was inert at a hop ABOVE the
    # object that was tested, so each hop is broken separately here.
    Mutation(
        name="scored_entry_point_never_forwards_the_donor_arm",
        path=SCORED_ENTRY,
        old="""            **(
                {
                    "enable_donor_channel": True,
                    "donor_cut_law": str(donor_cut_law),
                }
                if enable_donor_channel
                else {}
            ),""",
        new="",
        expect_red=(
            "test_the_scored_entry_point_passes_the_donor_arm_to_the_campaign",
            "test_the_full_scored_chain_reaches_the_law_and_restores",
        ),
        suite=SCORED_SUITE,
    ),
    Mutation(
        name="scored_entry_point_forwards_the_donor_arm_unconditionally",
        path=SCORED_ENTRY,
        old="""                if enable_donor_channel
                else {}""",
        new="""                if True
                else {}""",
        expect_red=("test_the_off_arm_carries_no_donor_key_at_all",),
        suite=SCORED_SUITE,
    ),
    Mutation(
        name="restore_drops_the_donor_arm_before_the_constructor",
        path=CONTROLLER,
        old="""                "enable_donor_channel": enable_donor_channel,""",
        new="""                "enable_donor_channel": False,""",
        expect_red=("test_the_full_scored_chain_reaches_the_law_and_restores",),
        suite=SCORED_SUITE,
    ),
    Mutation(
        name="restore_signature_loses_the_donor_arm",
        path=CONTROLLER,
        old="""        enable_donor_channel: bool = False,
        donor_cut_law: str = DEFAULT_CUT_LAW,
    ):""",
        new="""        **_unused,
    ):""",
        expect_red=("test_restore_accepts_the_donor_arm_like_the_constructor",),
        suite=SCORED_SUITE,
    ),
    Mutation(
        name="snapshot_stops_recording_the_cut_law",
        path=CONTROLLER,
        old="""                    "donor_cut_law": self.donor_cut_law,""",
        new="",
        expect_red=("test_a_resume_cannot_swap_the_cut_law_under_a_running_arm",),
        suite=SCORED_SUITE,
    ),
    # ---- The named arms: the DEFAULT is the finding ----
    Mutation(
        name="production_default_reverts_to_the_shipped_uniform_cut",
        path=DONOR_CHANNEL_MODULE,
        old="DEFAULT_CUT_LAW = RETENTIVE_RELEASED_FRACTION",
        new="DEFAULT_CUT_LAW = UNIFORM_ORIENTED_SINGLE_BRIDGE_ARM",
        expect_red=("test_the_production_default_is_the_retentive_cut_law",),
        suite=SCORED_SUITE,
    ),
    Mutation(
        name="unknown_arm_falls_back_instead_of_raising",
        path=DONOR_CHANNEL_MODULE,
        old="""    if name not in CUT_LAWS:
        raise ValueError(
            f"unknown donor cut law {name!r}; the named arms are {sorted(CUT_LAWS)}"
        )
    return CUT_LAWS[name]""",
        new="    return CUT_LAWS.get(name, donor_region_law)",
        expect_red=(
            "test_an_unknown_cut_law_is_refused_rather_than_defaulted",
            "test_the_controller_refuses_an_unknown_arm_at_construction",
        ),
        suite=SCORED_SUITE,
    ),
    Mutation(
        name="uniform_arm_becomes_a_lookalike_that_consumes_rng_random",
        path=DONOR_CHANNEL_MODULE,
        old="    UNIFORM_ORIENTED_SINGLE_BRIDGE_ARM: None,",
        new="    UNIFORM_ORIENTED_SINGLE_BRIDGE_ARM: lambda _g: UNIFORM_ORIENTED_SINGLE_BRIDGE,",
        expect_red=(
            "test_the_uniform_arm_is_the_shipped_recipe_draw_by_name_and_by_behaviour",
        ),
        suite=SCORED_SUITE,
    ),
    # ---- THE SHARED SINK. Two call sites, one probe, and a green gate. ----
    # `donor_transplant_draw` consults the law once for the parent and once per donor. A
    # raising probe fires on the FIRST, so dropping the DONOR side leaves the consumption
    # gate green -- named in `expect_green` so that blind spot is ASSERTED rather than
    # assumed, and so the behavioural test beside it is shown to be the thing doing the
    # work. This is the exact near-miss the region-law wiring hit, where `segment_replace`
    # kept threading the law after `substituent_delete` stopped.
    Mutation(
        name="donor_side_cut_silently_reverts_to_uniform",
        path=DONOR_CHANNEL_MODULE,
        old="        donor_law = law(donor) if law is not None else None",
        new="        donor_law = None",
        expect_red=(
            "test_the_law_is_consulted_for_the_DONOR_as_well_as_the_source",
            "test_the_retentive_law_takes_a_smaller_graft_from_the_donor",
        ),
        expect_green=("test_the_full_scored_chain_reaches_the_law_and_restores",),
        suite=SCORED_SUITE,
    ),
    # ---- The launch hop: a sealed payload must be able to turn the lane on ----
    Mutation(
        name="worker_stops_selecting_the_donor_arm",
        path=WORKER_APP,
        old="""                if (contract_envelope["payload"].get("arm") or {}).get(
                    "enable_donor_channel", False
                )
                else {}""",
        new="""                if False
                else {}""",
        expect_red=("test_the_worker_selects_the_donor_arm_from_the_sealed_payload",),
        suite=SCORED_SUITE,
    ),
    Mutation(
        name="POSITIVE_CONTROL_scored_path_cosmetic",
        path=SCORED_ENTRY,
        old='SCHEMA = "pmo_population_controller_v1"',
        new='SCHEMA = "pmo_population_controller_v1"  # cosmetic: behaviour unchanged',
        expect_red=(),
        suite=SCORED_SUITE,
    ),
    # ---- Attribution: without the tag the B-vs-C ablation cannot see this channel ----
    Mutation(
        name="donor_candidate_loses_its_synthesis_time_tag",
        path=CONTROLLER,
        old="""            metadata = {
                DONOR_CHANNEL: proposal.payload(),
                **self._continuation_lineage(entry),
            }""",
        new="""            metadata = {**self._continuation_lineage(entry)}""",
        expect_red=(
            "test_a_donor_candidate_is_tagged_at_synthesis_and_survives_to_the_snapshot",
        ),
        suite=SCORED_SUITE,
    ),
    # ---- Positive control: bytes change, behaviour does not ----
    # ---- The stratified scored bank: selection, draw, update hops, join ----
    Mutation(
        name="elite_capacity_falls_below_the_superseded_pool",
        path=MEMORY,
        old="ELITE_CAPACITY = 100",
        new="ELITE_CAPACITY = 16",
        expect_red=("test_the_bank_is_a_widening_and_never_a_narrowing",),
        suite=BANK_SUITE,
    ),
    Mutation(
        name="the_strata_stop_being_disjoint",
        path=MEMORY,
        old="""        ][: room(ELITE)]
        taken.update(elite)""",
        new="""        ][: room(ELITE)]""",
        expect_red=("test_the_strata_are_disjoint_and_precedence_decides_ties",),
        suite=BANK_SUITE,
    ),
    Mutation(
        name="the_diverse_stratum_ignores_the_basins_already_occupied",
        path=MEMORY,
        old="""        occupied = {
            self.rows[endpoint]["basin"]
            for endpoint in taken
            if self.rows[endpoint]["basin"] is not None
        }""",
        new="""        occupied = set()""",
        expect_red=(
            "test_the_diverse_stratum_covers_a_basin_the_higher_strata_do_not",
        ),
        suite=BANK_SUITE,
    ),
    Mutation(
        name="promising_admits_a_parent_whose_child_did_not_improve",
        path=MEMORY,
        old='if endpoint not in taken and row["improvement"] > 0.0',
        new='if endpoint not in taken and row["improvement"] >= 0.0',
        expect_red=(
            "test_the_promising_stratum_is_evidence_about_a_molecule_as_a_SOURCE",
        ),
        suite=BANK_SUITE,
    ),
    Mutation(
        name="observe_lineage_invents_a_row_for_a_parent_the_ledger_never_charged",
        path=MEMORY,
        old="""        row = self.rows.get(parent_endpoint)
        if row is None:
            return False""",
        new="""        row = self.rows.get(parent_endpoint)
        if row is None:
            self.rows[parent_endpoint] = row = {
                "score": 0.0,
                "basin": None,
                "children": 0,
                "improvement": 0.0,
            }""",
        expect_red=("test_a_parent_the_ledger_never_charged_cannot_enter_the_bank",),
        suite=BANK_SUITE,
    ),
    Mutation(
        name="the_lineage_is_banked_BELOW_the_attribution_early_return",
        path=MEMORY,
        old="""        self.bank.observe_lineage(
            parent_endpoint=parent_endpoint,
            parent_score=parent_score,
            child_score=child_score,
        )
        delta_heavy = int(child_heavy) - int(parent_graph.n_real_atoms)""",
        new="""        delta_heavy = int(child_heavy) - int(parent_graph.n_real_atoms)""",
        expect_red=(
            "test_the_lineage_is_banked_even_when_the_edit_cannot_be_attributed",
        ),
        suite=BANK_SUITE,
    ),
    Mutation(
        name="restore_rebuilds_the_bank_from_the_frontier",
        path=MEMORY,
        old="""        self.bank.rows = {
            endpoint: dict(row)
            for endpoint, row in (payload.get("bank") or {}).get("rows", {}).items()
        }""",
        new="""        self.bank.rows = {
            endpoint: {
                "score": float(score),
                "basin": None,
                "children": 0,
                "improvement": 0.0,
            }
            for endpoint, score in self.frontier.scores.items()
        }""",
        expect_red=("test_the_bank_rides_the_resume_and_an_old_payload_restores_EMPTY",),
        suite=BANK_SUITE,
    ),
    Mutation(
        name="the_draw_flattens_the_strata_into_one_uniform_pool",
        path=DONOR_CHANNEL_MODULE,
        old="""    chosen = live[int(rng.choice(len(live), p=weights / total))]
    smiles, graph = chosen.members[int(rng.integers(len(chosen.members)))]
    return chosen.name, smiles, graph""",
        new="""    flat = [(s.name, *member) for s in live for member in s.members]
    return flat[int(rng.integers(len(flat)))]""",
        expect_red=(
            "test_the_draw_gives_each_stratum_its_DECLARED_mass_not_its_size_share",
        ),
        suite=BANK_SUITE,
    ),
    Mutation(
        name="the_donor_tag_stops_naming_the_stratum_its_donor_came_from",
        path=DONOR_CHANNEL_MODULE,
        old="""            "donor": self.donor,
            "donor_stratum": self.donor_stratum,""",
        new="""            "donor": self.donor,""",
        expect_red=("test_a_donor_candidate_names_the_stratum_its_donor_came_from",),
        suite=BANK_SUITE,
    ),
    Mutation(
        name="add_measured_program_stops_banking_the_bootstrap_molecule",
        path=CONTROLLER,
        old="""        self._bank_measured_program(record, score)
        return ProgramOptimizer.add_measured_program(""",
        new="""        return ProgramOptimizer.add_measured_program(""",
        expect_red=(
            "test_a_bootstrap_scored_molecule_reaches_the_bank_AND_NOTHING_ELSE",
        ),
        suite=BANK_SUITE,
    ),
    Mutation(
        name="add_measured_program_also_feeds_the_frontier_and_moves_arm_B",
        path=CONTROLLER,
        old="""            self.online_memory.bank.observe(
                endpoint=endpoint, score=float(score), basin=basin
            )""",
        new="""            self.online_memory.observe_scored_molecule(
                endpoint=endpoint, score=float(score), basin=basin
            )""",
        expect_red=(
            "test_a_bootstrap_scored_molecule_reaches_the_bank_AND_NOTHING_ELSE",
        ),
        suite=BANK_SUITE,
    ),
    Mutation(
        name="the_donor_pool_reverts_to_a_score_ranking",
        path=CONTROLLER,
        old="""        bank = self.online_memory.bank
        selection = bank.strata(capacity=DONOR_POOL_SIZE)
        weights = bank.weights(selection)""",
        new="""        bank = self.online_memory.bank
        ranked = sorted(bank.rows.items(), key=lambda kv: (-kv[1]["score"], kv[0]))
        selection = {
            STRATA[0]: [e for e, _ in ranked[:DONOR_POOL_SIZE]],
            STRATA[1]: [],
            STRATA[2]: [],
        }
        weights = {STRATA[0]: 1.0}""",
        expect_red=(
            "test_the_donor_pool_IS_the_bank_selection_and_not_a_score_ranking",
        ),
        suite=BANK_SUITE,
    ),
    Mutation(
        name="arm_D_stops_requiring_arm_B_at_the_controller",
        path=CONTROLLER,
        old="""        if enable_donor_channel and not enable_online_memory:
            raise ValueError(
                "the donor recombination lane requires the online memory: arm D is arm "
                "B plus donor recombination, and its donors are the memory's stratified "
                "scored bank"
            )""",
        new="""        if False:
            raise ValueError("unreachable")""",
        expect_red=("test_arm_D_without_arm_B_is_REFUSED_at_the_controller",),
        suite=BANK_SUITE,
    ),
    Mutation(
        name="arm_D_stops_requiring_arm_B_before_the_ledger_is_built",
        path=SCORED_ENTRY,
        old="""    if enable_donor_channel and not enable_online_memory:""",
        new="""    if False:""",
        expect_red=(
            "test_arm_D_without_arm_B_is_REFUSED_before_the_ledger_is_built",
        ),
        suite=BANK_SUITE,
    ),
    Mutation(
        name="the_parent_endpoint_is_read_straight_off_a_provenance_key_that_is_never_written",
        path=CONTROLLER,
        old="""        endpoint = provenance.get("parent_endpoint")
        if endpoint:
            return str(endpoint)
        entry = self.entries.get(provenance.get("entry_id"))
        return str((entry or {}).get("endpoint") or "")""",
        new="""        return str(provenance.get("parent_endpoint") or "")""",
        expect_red=(
            "test_the_promising_stratum_is_reachable_from_a_REAL_propose_observe_cycle",
        ),
        suite=BANK_SUITE,
    ),
    Mutation(
        name="the_donor_strata_are_rebuilt_per_attempt_instead_of_per_round",
        path=CONTROLLER,
        old="""        banked = self._donor_strata()""",
        new="""        banked = []  # rebuilt per attempt below""",
        expect_red=(
            "test_a_donor_candidate_names_the_stratum_its_donor_came_from",
        ),
        suite=BANK_SUITE,
    ),
    Mutation(
        name="POSITIVE_CONTROL_bank_cosmetic_comment",
        path=MEMORY,
        old="SCHEMA = \"pmo_online_memory_v1\"",
        new="SCHEMA = \"pmo_online_memory_v1\"  # cosmetic; must stay green",
        expect_red=(),
        suite=BANK_SUITE,
    ),
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
    parser.add_argument(
        "--only",
        default=None,
        help=(
            "run only mutations whose name contains this substring. A SUBSET run is a "
            "development convenience and its report says so: `mutations_are_a_subset` is "
            "True, so a partial verdict can never be read as the whole battery."
        ),
    )
    parser.add_argument(
        "--suite",
        default=None,
        help="run only mutations whose target suite path contains this substring",
    )
    args = parser.parse_args()
    selected = tuple(
        m
        for m in MUTATIONS
        if (args.only is None or args.only in m.name)
        and (args.suite is None or args.suite in m.suite)
    )
    if not selected:
        print("ABORT: the filters selected no mutations")
        return 2

    # Baseline EVERY suite the selected mutations will run. A mutation verdict is
    # meaningless against a suite that was already red, and with `--only` the suite under
    # test is often not the default one.
    for suite in sorted({m.suite for m in selected}):
        baseline_code, baseline_failed = run_suite(suite)
        if baseline_code != 0:
            print(f"ABORT: {suite} is not green unmutated ({sorted(baseline_failed)})")
            return 2
        print(f"baseline: {suite} green")
    print()

    rows, verdict_ok = [], True
    originals = {path: path.read_text() for path in {m.path for m in selected}}
    try:
        for mutation in selected:
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
                survived = set(mutation.expect_green)
                killed = (
                    bool(failed)
                    and expected.issubset(failed)
                    and survived.isdisjoint(failed)
                )
                status = "KILLED" if killed else "SURVIVED"
            verdict_ok &= killed
            rows.append(
                {
                    "mutation": mutation.name,
                    "file": mutation.path.name,
                    "is_positive_control": mutation.is_control,
                    "expected_red": sorted(expected),
                    "expected_green": sorted(mutation.expect_green),
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
        "mutations_are_a_subset": args.only is not None or args.suite is not None,
        "only_filter": args.only,
        "suite_filter": args.suite,
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
