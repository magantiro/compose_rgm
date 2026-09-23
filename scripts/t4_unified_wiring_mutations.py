"""Mutation battery for the unified T4 controller's wiring guards.

WHY THIS EXISTS
---------------
A test that passes against deliberately broken production code is worthless, and
this repository has collected several ways for one to look load-bearing while being
inert: an expectation recomputed from the code under test, a floor asserted on a
fixture that never reaches a negative margin, a guard whose test another guard also
satisfies, and -- twice -- a mutation whose text substitution silently failed, so an
UNMUTATED file read as "the guard survived".

Every mutation here therefore DIFFS the file after substituting and ABORTS with a
traceback if nothing changed.  A mutation that cannot be applied is a broken
harness, not a surviving guard, and the two must never be reported the same way.

WHAT IT DOES
------------
For each mutation: copy the tree to a temp directory, apply every substitution,
print the unified diff, run the NAMED tests that must turn red, restore by throwing
the copy away, and record KILLED / SURVIVED.

POSITIVE CONTROLS
-----------------
Two mutations change bytes and nothing semantic -- a reformatted docstring and a
renamed local variable -- and MUST leave the suite green.  Without them, "every
mutation refused" is indistinguishable from "the harness is broken", which is
exactly how a battery that was silently failing to copy `.git` once read 20/20.

GIT
---
Nothing under test here shells out to git, so `.git` is deliberately NOT copied and
the temp tree is not a repository.  (A worktree's `.git` is a FILE pointing at the
parent repo, so copying one without detaching it misbehaves; the check that makes
that safe to skip is the absence of any `subprocess`/`git` call in the two test
files, asserted below.)

Charges nothing: CPU only, zero oracle calls, zero Modal.
"""

from __future__ import annotations

import argparse
import difflib
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

CONTROLLER = "src/compose_v4/experiments/t4_unified_controller.py"
PROPOSAL = "src/compose_v4/experiments/t4_unified_proposal.py"
ROUTING = "src/compose_v4/control/t4_unified_routing.py"

APP = "modal_apps/t4_unified_controller_app.py"
FA7_CONTRACT = "configs/t4_unified_controller_fa7_d06_v1.json"

CONTROLLER_TESTS = "tests/test_t4_unified_controller.py"
PROPOSAL_TESTS = "tests/test_t4_unified_proposal.py"
APP_TESTS = "tests/test_t4_unified_controller_app.py"


def controller(name: str) -> str:
    """A node id in the controller suite. One function builds them, so a renamed
    test cannot leave a mutation silently pointing at nothing."""

    return f"{CONTROLLER_TESTS}::{name}"


def proposal(name: str) -> str:
    return f"{PROPOSAL_TESTS}::{name}"


def appguard(name: str) -> str:
    return f"{APP_TESTS}::{name}"


def call_site(function: str, required: str) -> str:
    """The call-site row for one hop. Built here so a renamed parameter id cannot
    leave a mutation pointing at a test node that no longer exists."""

    return appguard(
        "test_the_load_bearing_call_is_made_by_the_function_that_must_make_it"
        f"[{function}-{required}]"
    )

#: Copied into the temp tree. `configs/` is needed because the behavioural tests
#: read the SHIPPED contracts rather than a fixture transcribed from them.
COPIED = ("src", "tests", "configs", "modal_apps", "pyproject.toml")


@dataclass
class Mutation:
    name: str
    path: str
    edits: list[tuple[str, str]]
    must_kill: list[str]
    positive_control: bool = False
    note: str = ""
    #: Tests that must STAY GREEN, so a mutation aimed at one hop is shown not to
    #: be killed incidentally by the other hop's guard.
    must_survive: list[str] = field(default_factory=list)


MUTATIONS: list[Mutation] = [
    Mutation(
        name="01_expansion_ignores_the_injected_router",
        path=CONTROLLER,
        edits=[
            (
                (
                    "    resolved_router = router if router is not None "
                    "else resolve_state_routing(contract)"
                ),
                "    resolved_router = resolve_state_routing(contract)",
            )
        ],
        must_kill=[
            controller("test_a_live_wiring_is_observed_on_the_first_attempt"),
            controller(
                "test_the_probe_survives_the_five_exception_types_the_draw_path_catches"
            ),
        ],
        must_survive=[
            controller("test_a_real_routed_expansion_is_accepted_as_terminal"),
            controller("test_an_injected_router_source_is_refused"),
        ],
        note="hop 2 only: the consumption probe can no longer be installed",
    ),
    Mutation(
        name="02_expansion_never_consults_the_router",
        path=CONTROLLER,
        edits=[
            (
                (
                    "        decision = resolved_router.decision(\n"
                    "            parent,\n"
                    "            eligible_pool_size=0,\n"
                    "            rounds_completed=int(rounds_completed),\n"
                    "        )"
                ),
                '        decision = {"activated_kernel": "region", "support_ramp": 1.0}',
            )
        ],
        must_kill=[
            controller(
                "test_5ht1b_2_routes_to_the_state_aware_kernel_and_its_rung_zero_lane"
            ),
            controller("test_every_parent_is_handed_to_rung_zero_with_its_own_route"),
        ],
        note=(
            "routes every parent to `region` regardless of its chemistry. NOT listed "
            "against the hop-2 probe: `_RouterProbe` raises from `lane_for` as well as "
            "from `decision`, so the consumption check still observes the injected "
            "router and it is the BEHAVIOURAL routing tests that refuse this"
        ),
    ),
    Mutation(
        name="03_probe_exception_becomes_a_swallowed_type",
        path=CONTROLLER,
        edits=[
            (
                "class _RouterProbeConsumed(Exception):",
                "class _RouterProbeConsumed(ValueError):",
            )
        ],
        must_kill=[
            controller(
                "test_the_probe_survives_the_five_exception_types_the_draw_path_catches"
            ),
            controller("test_the_probe_exception_is_none_of_the_swallowed_types"),
        ],
        note="`expand` catches ValueError per draw, so the probe would be swallowed",
    ),
    Mutation(
        name="04_terminal_check_drops_router_source",
        path=CONTROLLER,
        edits=[
            (
                '    for key in ("router_source", "ladder_source"):',
                '    for key in ("ladder_source",):',
            )
        ],
        must_kill=[
            controller("test_an_injected_router_source_is_refused"),
            controller("test_an_injected_router_or_ladder_is_recorded_as_injected"),
        ],
        must_survive=[
            controller("test_an_injected_ladder_source_is_refused"),
        ],
        note="a probe run could be published as a campaign result",
    ),
    Mutation(
        name="05_terminal_check_drops_the_ladder_hash",
        path=CONTROLLER,
        edits=[
            (
                (
                    "    if (\n"
                    '        ladder.get("policy") != FROZEN_LADDER_ID\n'
                    '        or ladder.get("policy_sha256") != FROZEN_LADDER_SHA256\n'
                    "    ):"
                ),
                "    if False:",
            )
        ],
        must_kill=[
            controller("test_a_ladder_that_is_not_the_frozen_one_is_refused"),
        ],
        must_survive=[
            controller("test_a_router_that_is_not_the_pinned_one_is_refused"),
        ],
        note="a per-target ladder could end a cell",
    ),
    Mutation(
        name="06_terminal_check_stops_delegating",
        path=CONTROLLER,
        edits=[
            (
                "    assert_support_expansion_is_consumed(outcome)",
                "    return  # the expansion's own stop-reason check is dropped",
            )
        ],
        must_kill=[
            controller(
                "test_the_expansions_own_stop_reason_check_is_delegated_not_duplicated"
            ),
        ],
        note="candidate exhaustion could be published from an event that never ran",
    ),
    Mutation(
        name="07_routing_block_hash_is_not_compared",
        path=CONTROLLER,
        edits=[
            (
                '    if block["policy_sha256"] != ROUTER_SHA256:',
                "    if False:",
            )
        ],
        must_kill=[controller("test_a_stale_policy_hash_is_refused")],
        must_survive=[controller("test_a_wrong_policy_name_is_refused")],
        note="a contract sealed against a different rule would validate",
    ),
    Mutation(
        name="08_declaration_ignores_per_cell_overrides",
        path=CONTROLLER,
        edits=[("    if offenders:", "    if False:")],
        must_kill=[
            controller("test_a_per_cell_proposal_override_is_refused"),
        ],
        note="the panel could become a family of controllers again",
    ),
    Mutation(
        name="09_declaration_accepts_any_region_law",
        path=CONTROLLER,
        edits=[("    if name != DECLARED_REGION_LAW:", "    if False:")],
        must_kill=[
            controller("test_a_different_region_law_is_refused"),
            controller("test_an_absent_region_law_field_is_refused"),
        ],
        note="a run under a different draw law would pass as the declared panel",
    ),
    Mutation(
        name="10_rung_zero_authorization_is_a_no_op",
        path=PROPOSAL,
        edits=[
            (
                (
                    "    from compose_v4.control.frozen_proposal_escalation import "
                    "resolve_frozen_escalation\n\n    ladder = "
                    "resolve_frozen_escalation(contract)"
                ),
                (
                    "    from compose_v4.control.frozen_proposal_escalation import "
                    "resolve_frozen_escalation\n\n    return\n    ladder = "
                    "resolve_frozen_escalation(contract)"
                ),
            )
        ],
        must_kill=[
            proposal(
                "test_a_rung_zero_lane_is_refused_without_a_ladder_that_authorizes_the_stage"
            ),
            proposal("test_a_rung_zero_lane_is_refused_without_a_resolvable_routing"),
            proposal("test_a_rung_zero_lane_is_refused_when_the_ladder_disables_the_stage"),
        ],
        note="a rung-0 lane could be invoked from an ordinary round by naming it",
    ),
    Mutation(
        name="11_rung_zero_record_is_labelled_with_its_own_stage",
        path=PROPOSAL,
        edits=[
            (
                ("                **record,\n" '                "proposal_lane": None,'),
                (
                    "                **record,\n"
                    '                "proposal_lane": "zero_support_fallback",'
                ),
            )
        ],
        must_kill=[
            proposal("test_a_rung_zero_record_carries_its_stage_and_no_proposal_lane"),
            proposal("test_a_real_rung_zero_record_survives_the_production_feature_path"),
        ],
        note="kills `attach_features` at the moment rung 0 first succeeds",
    ),
    Mutation(
        name="12_every_kernel_routes_to_the_region_lane",
        path=CONTROLLER,
        edits=[
            (
                "            return ROUTED_RUNG_ZERO_LANE[kernel]",
                '            return "zero_support_fallback"',
            )
        ],
        must_kill=[
            controller(
                "test_5ht1b_2_routes_to_the_state_aware_kernel_and_its_rung_zero_lane"
            ),
            controller("test_every_parent_is_handed_to_rung_zero_with_its_own_route"),
            controller("test_all_three_charged_sources_route_to_the_same_kernel"),
            controller(
                "test_the_router_refuses_a_kernel_with_no_registered_rung_zero_lane"
            ),
        ],
        note="a charged parent would be handed the lane its executor refuses",
    ),
    Mutation(
        name="13_router_hash_is_hardcoded_rather_than_derived",
        path=CONTROLLER,
        edits=[
            (
                "ROUTER_SHA256 = identity(ROUTER_DESCRIPTION)",
                'ROUTER_SHA256 = "b69cadcba1824ffc" + "0" * 48',
            )
        ],
        must_kill=[
            controller("test_the_pinned_hash_is_the_hash_of_the_declared_rule"),
        ],
        note="a stale constant would stop pinning the rule it claims to pin",
    ),
    Mutation(
        name="14_the_routing_reads_a_field_the_hash_does_not_pin",
        path=ROUTING,
        edits=[
            (
                "    heavy_atoms: int\n    growth_headroom: int",
                (
                    "    heavy_atoms: int\n    growth_headroom: int\n"
                    "    smuggled_field: int = 0"
                ),
            )
        ],
        must_kill=[
            controller(
                "test_the_declared_rule_pins_the_live_field_sets_of_both_routing_records"
            ),
        ],
        note=(
            "adding a field to MolecularApplicability is the only way to smuggle an "
            "identity into the routing; the pinned field set must move with it"
        ),
    ),
    Mutation(
        name="P1_positive_control_docstring_reflow",
        path=CONTROLLER,
        edits=[
            (
                '    """The production router: molecular state in, rung-0 kernel out.',
                '    """The production router -- molecular state in, rung-0 kernel out.',
            ),
            (
                '        """Measure the parent with the PRODUCTION excision path."""',
                (
                    '        """Measure the parent with the production excision '
                    'path.\n\n        Reflowed prose; the measurement is '
                    'unchanged.\n        """'
                ),
            ),
        ],
        must_kill=[],
        positive_control=True,
        note="bytes move, semantics do not: the suite must stay green",
    ),
    Mutation(
        name="P2_positive_control_local_variable_rename",
        path=CONTROLLER,
        edits=[
            ("    keys = frozenset(block)", "    present_keys = frozenset(block)"),
            (
                (
                    "    if keys != _REQUIRED_ROUTING_KEYS:\n"
                    "        extra = sorted(keys - _REQUIRED_ROUTING_KEYS)\n"
                    "        missing = sorted(_REQUIRED_ROUTING_KEYS - keys)"
                ),
                (
                    "    if present_keys != _REQUIRED_ROUTING_KEYS:\n"
                    "        extra = sorted(present_keys - _REQUIRED_ROUTING_KEYS)\n"
                    "        missing = sorted(_REQUIRED_ROUTING_KEYS - present_keys)"
                ),
            ),
        ],
        must_kill=[],
        positive_control=True,
        note="a rename cannot change behaviour; the suite must stay green",
    ),
    # ---- The APP hop. Five of these are call-site defects a static check can
    # see; A6 is the one it cannot, and it is the reason the by-execution guard
    # exists at all.
    Mutation(
        name="A1_engine_drives_the_stopping_rule_itself",
        path=APP,
        edits=[("    outcome, record = routed_support_expansion(",
                "    outcome, record = _bypassed_expansion(")],
        must_kill=[call_site("expand_support", "routed_support_expansion")],
        note="the empty-pool branch no longer goes through the routed expansion",
    ),
    Mutation(
        name="A2_terminal_guard_is_not_run_before_publishing_exhaustion",
        path=APP,
        edits=[("            assert_unified_expansion_is_consumed(expansion_record, expansion)",
                "            pass  # terminal guard removed")],
        must_kill=[
            appguard("test_candidate_exhaustion_is_published_only_behind_the_terminal_guard"),
            call_site("run_cell", "assert_unified_expansion_is_consumed"),
        ],
        note="hop 3: a round loop quietly returning to the bare hard stop",
    ),
    Mutation(
        name="A3_routing_consumption_is_never_proven",
        path=APP,
        edits=[("    routing_probe_attempts = assert_state_routing_is_consumed(_probe_route)",
                "    routing_probe_attempts = 0")],
        must_kill=[call_site("run_cell", "assert_state_routing_is_consumed")],
        note="hop 2: the app stops proving the declared routing is installed",
    ),
    Mutation(
        name="A4_proposal_worker_stops_using_the_shared_lane_unit",
        path=APP,
        edits=[("    answer = proposal_unit(", "    answer = _inlined_lane(")],
        must_kill=[call_site("proposal_worker", "proposal_unit")],
        note="the gate would then measure a different proposal law from the campaign",
    ),
    Mutation(
        name="A5_contract_policy_is_not_resolved_on_every_role",
        path=APP,
        edits=[("    resolve_unified_controller(contract)\n    return contract",
                "    return contract")],
        must_kill=[call_site("_validate_task", "resolve_unified_controller")],
        note="hop 1: a contract missing the routing or the ladder stops failing early",
    ),
    Mutation(
        name="A6_rung_zero_ignores_the_route_and_always_uses_the_region_lane",
        path=APP,
        edits=[
            (
                '        for item in assignments:\n            lane = item["lane"]',
                (
                    '        for item in assignments:\n'
                    '            lane = ROUTED_RUNG_ZERO_LANE["region"]'
                ),
            )
        ],
        must_kill=[
            appguard(
                "test_the_apps_own_expansion_routes_each_parent_and_dispatches_that_lane"
                "[5ht1b-5ht1b_2-state_aware]"
            )
        ],
        must_survive=[call_site("expand_support", "routed_support_expansion")],
        note=(
            "THE ONE A CALL-SITE CHECK CANNOT SEE: every required call is still made "
            "and the record still names the routed kernel, but the lane the fan-out "
            "is asked for no longer follows it. Only driving the app's own "
            "expand_support and reading the dispatched requests catches it."
        ),
    ),
    Mutation(
        name="A7_a_contract_authorizes_its_own_scored_launch",
        path=FA7_CONTRACT,
        edits=[('"scored_launch_authorized":false', '"scored_launch_authorized":true')],
        must_kill=[appguard("test_every_contract_refuses_to_launch_without_the_owner")],
        note="the 22,500-call decision is the owner's, and the suite says so",
    ),
    Mutation(
        name="P3_positive_control_app_comment_reflow",
        path=APP,
        edits=[("# ---- Resume ----", "# ---- Resume: restore a preempted cell ----")],
        must_kill=[],
        positive_control=True,
        note="bytes move, nothing semantic does",
    ),
]


def _stage(destination: Path) -> None:
    for name in COPIED:
        source = ROOT / name
        target = destination / name
        if source.is_dir():
            shutil.copytree(
                source,
                target,
                ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache"),
            )
        else:
            shutil.copy2(source, target)


def _apply(tree: Path, mutation: Mutation) -> str:
    """Apply every substitution, ABORTING if any of them did not change the file."""

    path = tree / mutation.path
    before = path.read_text()
    after = before
    for index, (old, new) in enumerate(mutation.edits):
        mutated = after.replace(old, new, 1)
        if mutated == after:
            raise AssertionError(
                f"mutation {mutation.name!r} edit {index} did not apply to "
                f"{mutation.path}: the anchor text is absent, so the file is "
                "UNMUTATED and any 'guard survived' verdict from it would be "
                f"fiction. Anchor was:\n{old!r}"
            )
        after = mutated
    path.write_text(after)
    return "".join(
        difflib.unified_diff(
            before.splitlines(keepends=True),
            after.splitlines(keepends=True),
            fromfile=f"a/{mutation.path}",
            tofile=f"b/{mutation.path}",
            n=1,
        )
    )


def _pytest(tree: Path, targets: list[str]) -> subprocess.CompletedProcess:
    environment = dict(os.environ)
    environment["PYTHONPATH"] = "src:scripts"
    environment["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    environment["OMP_NUM_THREADS"] = "1"
    return subprocess.run(
        [sys.executable, "-m", "pytest", *targets, "-q", "-p", "no:cacheprovider"],
        check=False,
        cwd=tree,
        env=environment,
        capture_output=True,
        text=True,
    )


def _assert_no_git_dependency() -> None:
    """The reason `.git` is not copied, checked rather than asserted in prose."""

    for relative in (CONTROLLER_TESTS, PROPOSAL_TESTS, APP_TESTS):
        text = (ROOT / relative).read_text()
        for token in ("subprocess", "git ", "check_output"):
            assert token not in text, (
                f"{relative} references {token!r}; if a test shells out to git the "
                "temp tree needs a detached repository and this harness does not "
                "build one"
            )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", default=None, help="run one mutation by name prefix")
    arguments = parser.parse_args()

    _assert_no_git_dependency()

    selected = [
        mutation
        for mutation in MUTATIONS
        if arguments.only is None or mutation.name.startswith(arguments.only)
    ]
    if not selected:
        raise SystemExit(f"no mutation matches {arguments.only!r}")

    print("=" * 78)
    print("BASELINE: the unmutated tree must be green before any verdict is read")
    print("=" * 78)
    baseline = _pytest(ROOT, [CONTROLLER_TESTS, PROPOSAL_TESTS, APP_TESTS])
    print(baseline.stdout.strip().splitlines()[-1] if baseline.stdout else "")
    if baseline.returncode != 0:
        print(baseline.stdout[-4000:])
        raise SystemExit("baseline is red; a mutation verdict would be meaningless")

    rows: list[tuple[str, str, str]] = []
    for mutation in selected:
        print("\n" + "=" * 78)
        kind = "POSITIVE CONTROL" if mutation.positive_control else "MUTATION"
        print(f"{kind}  {mutation.name}")
        print(f"  file: {mutation.path}")
        if mutation.note:
            print(f"  what it breaks: {mutation.note}")
        print("=" * 78)
        with tempfile.TemporaryDirectory(prefix="t4-unified-mutation-") as temporary:
            tree = Path(temporary) / "tree"
            tree.mkdir()
            _stage(tree)
            diff = _apply(tree, mutation)
            print(diff.rstrip())

            if mutation.positive_control:
                run = _pytest(tree, [CONTROLLER_TESTS, PROPOSAL_TESTS, APP_TESTS])
                tail = run.stdout.strip().splitlines()[-1] if run.stdout else ""
                verdict = "GREEN" if run.returncode == 0 else "RED"
                detail = tail
                if run.returncode != 0:
                    print(run.stdout[-3000:])
                rows.append((mutation.name, verdict, detail))
                print(f"\n  -> {verdict} (must be GREEN): {detail}")
                continue

            killed_by: list[str] = []
            for target in mutation.must_kill:
                run = _pytest(tree, [target])
                if run.returncode != 0:
                    killed_by.append(target.split("::")[-1])
                else:
                    print(f"  [survived] {target.split('::')[-1]}")
            survivors_held = True
            for target in mutation.must_survive:
                run = _pytest(tree, [target])
                if run.returncode != 0:
                    survivors_held = False
                    print(
                        f"  [collateral] {target.split('::')[-1]} also went red; this "
                        "mutation is not hop-specific"
                    )
            verdict = "KILLED" if killed_by else "SURVIVED"
            if killed_by and mutation.must_survive and not survivors_held:
                verdict = "KILLED (not hop-specific)"
            detail = ", ".join(killed_by) if killed_by else "no named test went red"
            rows.append((mutation.name, verdict, detail))
            print(f"\n  -> {verdict}: {detail}")

    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)
    width = max(len(name) for name, _, _ in rows)
    for name, verdict, detail in rows:
        print(f"{name:<{width}}  {verdict:<24}  {detail}")

    bad = [
        name
        for name, verdict, _ in rows
        if verdict not in ("KILLED", "GREEN") and not verdict.startswith("KILLED")
    ]
    print(f"\n{len(rows) - len(bad)}/{len(rows)} as required")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
