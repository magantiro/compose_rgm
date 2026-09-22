"""Mutation battery for the jump-lane binding autopsy guards.

Each NEGATIVE mutation must turn a NAMED test red; if it does not, the guard is
decoration.  Each POSITIVE control perturbs something irrelevant and must still pass --
without one, "everything refused" is indistinguishable from a broken harness.

The runner ABORTS when a mutation did not apply (the file is diffed), because a
mutation that silently fails to substitute scores as "killed" and proves nothing.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

MODULE = Path("src/compose_v4/experiments/pmo_jump_binding_autopsy.py")
TESTS = "tests/test_pmo_jump_binding_autopsy.py"

# (name, old, new, test, expectation)
NEGATIVE = [
    (
        "recorder_decides_instead_of_observing",
        "        reason = original(prefix, step, plan, demand, spec)\n        log.append",
        "        reason = None\n        original(prefix, step, plan, demand, spec)\n        log.append",
        "test_the_recorder_never_changes_the_production_decision",
    ),
    (
        "recorder_discards_the_reason",
        'log.append({"step": int(step), "reason": reason})',
        'log.append({"step": int(step), "reason": None})',
        "test_root_and_depth_refusals_are_distinguished_on_real_pairs",
    ),
    (
        "recorder_does_not_restore_the_production_function",
        "    realization.propagate = wrapper\n    try:\n        yield log\n    finally:\n        realization.propagate = original",
        "    realization.propagate = wrapper\n    yield log\n    realization.propagate = original",
        "test_recording_propagate_restores_the_production_function_after_an_exception",
    ),
    (
        "budget_hit_reported_as_incompatibility",
        '    if row["outcome"] == realization.OUTCOME_EXHAUSTED:\n        return "search_budget_exhausted"',
        '    if row["outcome"] == realization.OUTCOME_EXHAUSTED:\n        return "no_role_consistent_successor"',
        "test_a_budget_hit_is_never_classified_as_incompatibility",
    ),
    (
        "invalid_plan_reported_as_incompatibility",
        '    if row["outcome"] == realization.OUTCOME_INVALID_PLAN:\n        return "invalid_plan"',
        '    if row["outcome"] == realization.OUTCOME_INVALID_PLAN:\n        return "no_role_consistent_successor"',
        "test_an_invalid_plan_is_never_classified_as_incompatibility",
    ),
    (
        "root_and_depth_refusals_collapsed",
        '        return f"{label}@root" if row["first_refusal_step"] == 0 else f"{label}@depth"',
        '        return f"{label}@root"',
        "test_root_and_depth_refusals_are_distinguished_on_real_pairs",
    ),
    (
        "role_view_relaxes_the_semantic_core",
        '    return RoleSpecification(name, frozenset(RELAXABLE_COMPONENTS))',
        '    return RoleSpecification(name, frozenset({*RELAXABLE_COMPONENTS, "atom_type"}))',
        "test_role_based_specification_keeps_the_semantic_core_and_the_dataflow_edge",
    ),
    (
        "role_view_relaxes_the_dataflow_edge",
        '    return RoleSpecification(name, frozenset(RELAXABLE_COMPONENTS))',
        '    return RoleSpecification(name, frozenset({*RELAXABLE_COMPONENTS, "created_ordinal"}))',
        "test_role_based_specification_keeps_the_semantic_core_and_the_dataflow_edge",
    ),
    (
        "projection_baseline_is_not_the_production_predicate",
        'SPEC_V1_FIELDS = RoleSpecification("v1_fields", frozenset())',
        'SPEC_V1_FIELDS = RoleSpecification("v1_fields", frozenset({"neighbor_element_histogram"}))',
        "test_production_identity_and_all_fields_specifications_agree_at_step_zero",
    ),
    (
        "projection_drops_a_field_from_the_sweep",
        "PROJECTABLE_FIELDS = (\n    *RELAXABLE_COMPONENTS,",
        "PROJECTABLE_FIELDS = (\n    *[c for c in RELAXABLE_COMPONENTS if c != 'creation_lag'],",
        "test_field_projection_sweeps_every_descriptor_component_production_defines",
    ),
    (
        "a_production_refusal_string_loses_its_mechanism_name",
        '    "insufficient_slot_capacity": "capacity",\n',
        "",
        "test_every_refusal_string_the_production_search_can_return_has_a_mechanism_name",
    ),
    (
        "root_refusal_returns_any_step",
        '    for row in log:\n        if row["step"] == 0:\n            return row["reason"]\n    return None',
        '    for row in log:\n        if row["reason"] is not None:\n            return row["reason"]\n    return None',
        "test_first_and_root_refusal_readers_agree_with_the_log",
    ),
    (
        "region_structure_collapsed_to_a_total_count",
        '        "largest_changed_region": max(components) if components else 0,',
        '        "largest_changed_region": sum(components) if components else 0,',
        "test_structural_delta_separates_one_large_region_from_scattered_small_ones",
    ),
    (
        "changed_atoms_never_joined_into_regions",
        "        if i in changed_set and j in changed_set:\n            graph.add_edge(i, j)",
        "        if i in changed_set and j in changed_set and False:\n            graph.add_edge(i, j)",
        "test_structural_delta_separates_one_large_region_from_scattered_small_ones",
    ),
    (
        "retention_measured_against_the_endpoint_not_the_source",
        '        "retained_fraction_mcs": core / max(before.GetNumHeavyAtoms(), 1),',
        '        "retained_fraction_mcs": core / max(after.GetNumHeavyAtoms(), 1),',
        "test_structural_delta_reports_the_region_shape_on_hand_checkable_pairs",
    ),
    (
        "unparseable_input_is_scored_instead_of_refused",
        "    if before is None or after is None:\n        return None",
        "    if before is None or after is None:\n        before = after = Chem.MolFromSmiles('C')",
        "test_structural_delta_reports_the_region_shape_on_hand_checkable_pairs",
    ),
]

# Irrelevant perturbations that MUST still pass.
POSITIVE = [
    (
        "cosmetic_comment",
        "SCHEMA = \"pmo_jump_binding_autopsy_v1\"",
        "# cosmetic control: this comment must change nothing\nSCHEMA = \"pmo_jump_binding_autopsy_v1\"",
        "test_fixture_covers_both_step_zero_verdicts_and_several_mechanisms",
    ),
    (
        "rename_a_local_variable",
        "    original = realization.propagate\n    log: list[dict[str, Any]] = []",
        "    production = realization.propagate\n    original = production\n    log: list[dict[str, Any]] = []",
        "test_recording_propagate_restores_the_production_function",
    ),
]


def run(test: str) -> bool:
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", f"{TESTS}::{test}", "-q", "--no-header", "-p", "no:cacheprovider"],
        capture_output=True,
        text=True,
        check=False,
    )
    return proc.returncode == 0


def main() -> None:
    baseline = MODULE.read_text()
    digest = hashlib.sha256(baseline.encode()).hexdigest()
    results = []
    try:
        for kind, cases in (("negative", NEGATIVE), ("positive", POSITIVE)):
            for name, old, new, test in cases:
                if old not in baseline:
                    raise SystemExit(
                        f"ABORT: mutation {name!r} does not apply -- its target text is "
                        f"absent from {MODULE}.  A mutation that cannot be applied "
                        f"proves nothing and must never be scored as killed."
                    )
                mutated = baseline.replace(old, new, 1)
                if mutated == baseline:
                    raise SystemExit(f"ABORT: mutation {name!r} produced no diff")
                MODULE.write_text(mutated)
                if MODULE.read_text() == baseline:
                    raise SystemExit(f"ABORT: mutation {name!r} did not reach the file")
                passed = run(test)
                results.append(
                    {
                        "kind": kind,
                        "mutation": name,
                        "test": test,
                        "test_passed": passed,
                        "verdict": (
                            ("KILLED" if not passed else "SURVIVED")
                            if kind == "negative"
                            else ("OK" if passed else "CONTROL_FAILED")
                        ),
                    }
                )
                print(f"  [{kind}] {name:52s} -> {results[-1]['verdict']}", flush=True)
                MODULE.write_text(baseline)
    finally:
        MODULE.write_text(baseline)
        restored = hashlib.sha256(MODULE.read_text().encode()).hexdigest()
        if restored != digest:
            raise SystemExit("ABORT: the module was not restored to its original bytes")

    survived = [r for r in results if r["verdict"] == "SURVIVED"]
    broken = [r for r in results if r["verdict"] == "CONTROL_FAILED"]
    summary = {
        "schema_version": "pmo_jump_autopsy_mutation_battery_v1",
        "negatives": len(NEGATIVE),
        "negatives_killed": sum(1 for r in results if r["verdict"] == "KILLED"),
        "survived": [r["mutation"] for r in survived],
        "positive_controls": len(POSITIVE),
        "positive_controls_passed": sum(1 for r in results if r["verdict"] == "OK"),
        "controls_failed": [r["mutation"] for r in broken],
        "results": results,
    }
    out = Path("diagnostics/pmo_jump_rebinding_v1/mutation_battery_v1.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(json.dumps({k: v for k, v in summary.items() if k != "results"}, indent=2))
    if survived or broken:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
