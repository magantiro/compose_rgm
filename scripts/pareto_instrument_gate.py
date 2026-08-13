"""Executable instrument gate for the Pareto-control lane.

THE RULE THIS ENFORCES
----------------------
Before reporting any statistic, ask what value it could take if the hypothesis
were false. If the answer is "none", it is not a measurement.

Five defects of exactly that shape have already been caught in this project,
each one a quantity whose sign was fixed before any data existed:

  1. an action selected by `argmax V_G` and then scored by `V_G`
  2. a "verified" arm that was secretly greedy, so headroom was 0 by definition
  3. a sign test against a null that policy improvement makes false
  4. a fabricated SHA-256 in a manifest
  5. a contrast that varied controller AND objective, inflating its effect ~11%

Prose reminders did not stop any of them; four were caught only after the run.
So the rule is executable here, and it fails the RUN rather than the reader.

Run with no arguments to self-test against this lane's declared contrasts --
which works at DESIGN_ONLY stage, before any result exists, and is how the gate
itself is demonstrated to work.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

REPO = Path(__file__).resolve().parents[1]

PARITY_DIMENSIONS = ("controller", "start", "budget", "objective")


class InstrumentGateError(AssertionError):
    """Raised when a statistic must not be reported."""


# ---------------------------------------------------------------------------
# Declarations
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Statistic:
    """A statistic may only be reported if it declares how it could be wrong."""

    name: str
    hypothesis: str
    #: What values it could take if the hypothesis were FALSE. Empty => not a
    #: measurement, and the gate refuses it.
    falsifying_range: str
    #: Names of the function used to SELECT and the function used to SCORE. If
    #: they are the same object, the statistic is circular (defect 1).
    selection_function: str
    scoring_function: str
    #: True when a theorem fixes the sign. Magnitude may be reported; a p-value
    #: or a sign test may not (defect 3).
    sign_is_guaranteed: bool = False
    reports_pvalue: bool = False


@dataclass(frozen=True)
class ArmSemantics:
    controller: str
    start: str
    budget: str
    objective: str

    def as_tuple(self) -> tuple[str, str, str, str]:
        return (self.controller, self.start, self.budget, self.objective)


@dataclass(frozen=True)
class Contrast:
    name: str
    arm: str
    base: str
    intended_dimension: str
    #: PRIMARY contrasts must isolate exactly one dimension and may carry a
    #: headline. CONTEXT_ONLY contrasts are reported with the dimensions they
    #: vary stated out loud, and may not. The main lane's audit handled its own
    #: confounded Q1 the same way rather than deleting it.
    status: str = "PRIMARY"
    #: Which COMPUTE axis this contrast claims parity on: "kernel", "native", or
    #: None. Only the claimed axis is enforced; the other is reported.
    #:
    #: Compute is NOT one of the four parity dimensions -- `budget` there means
    #: the EDIT budget, H=6, which every contrast holds. Two distinct reasons an
    #: axis goes unclaimed:
    #:
    #:   * P2 claims neither. A lookahead controller intrinsically spends more
    #:     compute than a myopic one, and throttling it to greedy's compute would
    #:     delete the mechanism under test.
    #:   * P3/P4 cannot claim BOTH, because one kernel call yields ~600
    #:     candidates: matching generate-then-rank on kernel calls starves it of
    #:     molecules, while matching it on native oracle calls hands it ~600x the
    #:     kernel budget. Demanding both would make the contrast unrunnable at
    #:     either end, so each instance claims one axis and the pair is reported
    #:     as a bracket.
    #:
    #: Equal endpoint counts are required regardless. That is the structural
    #: control on hypervolume inflation and it is never optional.
    budget_parity_axis: str | None = None


@dataclass
class GateReport:
    checks: list[dict[str, Any]] = field(default_factory=list)

    def add(self, check: str, passed: bool, detail: Any) -> None:
        self.checks.append({"check": check, "pass": bool(passed), "detail": detail})

    @property
    def failures(self) -> list[dict[str, Any]]:
        return [c for c in self.checks if not c["pass"]]

    @property
    def passed(self) -> bool:
        return not self.failures


# ---------------------------------------------------------------------------
# D1 -- selection and scoring must be different functions
# ---------------------------------------------------------------------------

def check_d1_selection_scoring_distinct(stats: Sequence[Statistic],
                                        report: GateReport) -> None:
    """Defect 1. `argmax V_G` scored by `V_G` gives `V_G(chosen) >= V_G(greedy)`
    by construction: of 55 disagreements, 50 higher, 5 tied, 0 lower. Zero
    losses was definitional. Determinism does NOT remove this bias -- the bias
    is selecting and scoring with the same function."""
    bad = [s.name for s in stats
           if s.selection_function == s.scoring_function
           and s.selection_function != "none"]
    report.add("D1_selection_scoring_distinct", not bad,
               {"circular_statistics": bad,
                "rule": "a statistic selected and scored by the same function has "
                        "no falsifying range; score on an independent continuation "
                        "or compare endpoints"})


def check_d1b_falsifying_range_declared(stats: Sequence[Statistic],
                                        report: GateReport) -> None:
    bad = [s.name for s in stats if not s.falsifying_range.strip()]
    report.add("D1b_falsifying_range_declared", not bad,
               {"statistics_without_a_falsifying_range": bad})


# ---------------------------------------------------------------------------
# D2 -- two arms that never act differently are one arm
# ---------------------------------------------------------------------------

def check_d2_arms_are_distinct(action_sequences: dict[str, dict[str, tuple]],
                               pairs: Sequence[tuple[str, str]],
                               report: GateReport) -> None:
    """Defect 2. `verified_land` was a plain greedy continuation while the
    decision loop committed greedy's action, so headroom was 0 by construction
    and two runs' CEILING verdicts were void. The tell was identical final
    utility on all 30 sources while the lookahead 'disagreed' at 38/90 states."""
    detail = {}
    ok = True
    for arm, base in pairs:
        a, b = action_sequences.get(arm, {}), action_sequences.get(base, {})
        shared = sorted(set(a) & set(b))
        if not shared:
            detail[f"{arm}_vs_{base}"] = "NO SHARED SOURCES -- cannot verify"
            ok = False
            continue
        differing = [s for s in shared if a[s] != b[s]]
        detail[f"{arm}_vs_{base}"] = {
            "sources": len(shared), "sources_with_different_actions": len(differing)}
        if not differing:
            ok = False
    report.add("D2_arms_are_distinct", ok,
               {"comparisons": detail,
                "rule": "identical committed action sequences on every source means "
                        "the two arms are the same arm; the run is INVALID_INSTRUMENT"})


# ---------------------------------------------------------------------------
# D3 -- no p-value against a null a theorem already makes false
# ---------------------------------------------------------------------------

def check_d3_no_false_null(stats: Sequence[Statistic], report: GateReport) -> None:
    """Defect 3. Policy improvement guarantees `verified >= greedy`, so a sign
    test against 0.5 tests something already known. Only the MAGNITUDE of a
    guaranteed-sign difference is admissible, read against the scale of the
    total movement."""
    bad = [s.name for s in stats if s.sign_is_guaranteed and s.reports_pvalue]
    report.add("D3_no_false_null", not bad,
               {"guaranteed_sign_statistics_reporting_a_pvalue": bad,
                "rule": "report magnitude only, and say the direction is guaranteed"})


# ---------------------------------------------------------------------------
# D4 -- every hash in the manifest is recomputed from disk
# ---------------------------------------------------------------------------

def check_d4_manifest_hashes(manifest: Path, repo: Path,
                             report: GateReport) -> None:
    """Defect 4. A SHA-256 that was never computed from the file it names is
    worse than no hash: it makes an unverifiable artifact look verified."""
    if not manifest.exists():
        report.add("D4_manifest_hashes", False, {"error": f"{manifest} missing"})
        return
    data = json.loads(manifest.read_text())
    entries = data.get("frozen_inputs", {})
    mismatches, missing, verified = {}, [], 0
    for path, declared in entries.items():
        target = repo / path
        if not target.exists():
            missing.append(path)
            continue
        actual = hashlib.sha256(target.read_bytes()).hexdigest()
        if actual != declared:
            mismatches[path] = {"declared": declared, "actual": actual}
        else:
            verified += 1
    report.add("D4_manifest_hashes", not mismatches and not missing,
               {"verified": verified, "mismatched": mismatches, "missing": missing})


# ---------------------------------------------------------------------------
# D5 -- every contrast varies exactly one dimension
# ---------------------------------------------------------------------------

def check_d5_parity(contrasts: Sequence[Contrast],
                    semantics: dict[str, ArmSemantics],
                    report: GateReport) -> None:
    """Defect 5. `verified_retarget` vs `continue_A` varied controller AND
    objective; the matched contrast is `greedy_retarget` vs `continue_A`. The
    confound inflated the effect by roughly 11%."""
    detail, ok = {}, True
    for c in contrasts:
        a, b = semantics.get(c.arm), semantics.get(c.base)
        if a is None or b is None:
            detail[c.name] = "UNDECLARED ARM"
            ok = False
            continue
        differs = [d for d, x, y in zip(PARITY_DIMENSIONS, a.as_tuple(), b.as_tuple())
                   if x != y]
        isolates = len(differs) == 1 and differs[0] == c.intended_dimension
        detail[c.name] = {"arm": c.arm, "base": c.base, "varies": differs,
                          "intended": c.intended_dimension, "status": c.status,
                          "isolates_one_mechanism": isolates}
        if c.status == "PRIMARY":
            ok &= isolates
        elif isolates:
            # A context contrast that turns out to be clean should be promoted,
            # not left understated -- otherwise the table hides a usable result.
            detail[c.name]["note"] = "declared CONTEXT_ONLY but isolates one " \
                                     "dimension; consider promoting to PRIMARY"
    report.add("D5_contrast_parity", ok,
               {"contrasts": detail,
                "rule": "a paired difference isolates the mechanism it is named "
                        "after only if the two arms differ in exactly that "
                        "mechanism. PRIMARY contrasts must; CONTEXT_ONLY contrasts "
                        "must state what else they vary and may not carry a "
                        "headline."})


# ---------------------------------------------------------------------------
# D6 -- hypervolume comparisons must be at matched budget
# ---------------------------------------------------------------------------

def check_d6_hv_budget_matched(contrasts: Sequence[Contrast],
                               costs: dict[str, dict[str, float]],
                               endpoint_counts: dict[str, int],
                               report: GateReport,
                               *, kernel_ratio_limit: float = 1.25,
                               native_tolerance: float = 0.10) -> None:
    """The hypervolume-specific version of the same trap: an arm that simply
    generates more molecules inflates HV without controlling anything better.

    Two structural controls, checked here rather than adjusted for afterwards:
    equal endpoint counts, and a bounded kernel-call ratio. A contrast that
    fails the ratio bound is flagged BUDGET_ASYMMETRIC and may not carry a
    headline.
    """
    detail, ok = {}, True
    for c in contrasts:
        ca, cb = costs.get(c.arm), costs.get(c.base)
        if ca is None or cb is None:
            continue  # not an HV contrast
        na, nb = endpoint_counts.get(c.arm), endpoint_counts.get(c.base)
        equal_points = na == nb
        native_a, native_b = ca["native_oracle_calls"], cb["native_oracle_calls"]
        denom = max(min(native_a, native_b), 1)
        native_gap = abs(native_a - native_b) / denom
        kernel_ratio = (max(ca["kernel_calls"], cb["kernel_calls"])
                        / max(min(ca["kernel_calls"], cb["kernel_calls"]), 1))
        entry = {"endpoint_counts": [na, nb], "equal_endpoint_counts": equal_points,
                 "native_call_gap": round(native_gap, 4),
                 "kernel_call_ratio": round(kernel_ratio, 3),
                 "budget_parity_axis": c.budget_parity_axis,
                 "flag": None}
        # Equal endpoint counts are enforced for EVERY hypervolume contrast.
        # That is the structural control on the inflation channel: an arm cannot
        # win by contributing more points, only by placing them better.
        if not equal_points:
            entry["flag"] = "UNEQUAL_ENDPOINTS"
            ok = False
        elif c.budget_parity_axis == "native" and native_gap > native_tolerance:
            entry["flag"] = "BUDGET_ASYMMETRIC_NATIVE"
            ok = False
        elif c.budget_parity_axis == "kernel" and kernel_ratio > kernel_ratio_limit:
            entry["flag"] = "BUDGET_ASYMMETRIC_KERNEL"
            ok = False
        elif kernel_ratio > kernel_ratio_limit or native_gap > native_tolerance:
            # Reported, not failed: on an UNCLAIMED axis the asymmetry is either
            # the mechanism under test (P2) or one declared end of a bracket
            # (P3/P4). Failing here would hide the experiment rather than a
            # confound; the ratio is what the reader needs instead.
            entry["flag"] = "COMPUTE_ASYMMETRIC_ON_AN_UNCLAIMED_AXIS_REPORT_THE_RATIO"
        detail[c.name] = entry
    # D6 is PER-CONTRAST by construction, unlike D1-D5 which are global
    # instrument properties. A budget failure on one contrast says nothing about
    # the others, so it INVALIDATES THAT CONTRAST rather than the whole run --
    # the same treatment P1 gets for its parity confound. Suppressing every
    # number because one comparison is unmatched would discard valid results and
    # would tempt the next person to loosen the tolerance to get them back.
    invalid = [name for name, e in detail.items()
               if e.get("flag") in ("UNEQUAL_ENDPOINTS", "BUDGET_ASYMMETRIC_NATIVE",
                                    "BUDGET_ASYMMETRIC_KERNEL")]
    report.add("D6_hv_budget_matched", ok,
               {"contrasts": detail,
                "invalidated_contrasts": invalid,
                "scope": ("per-contrast: these contrasts are barred, the rest of "
                          "the analysis stands"),
                "rule": "Equal endpoint counts are required everywhere -- an arm "
                        "must not win by contributing more points. Compute parity "
                        "is enforced only where a contrast claims it (the "
                        "generate-then-rank contrasts, where handing gen_rank the "
                        "control arm's budget is the whole point). Elsewhere the "
                        "compute ratio is REPORTED: throttling a lookahead "
                        "controller to a myopic one's compute would delete the "
                        "mechanism under test."})


# ---------------------------------------------------------------------------
# This lane's declarations
# ---------------------------------------------------------------------------

LANE_ARMS = {
    "unguided":          ArmSemantics("none", "x_0", "H=6", "none"),
    "gen_rank@greedy":   ArmSemantics("open_loop", "x_0", "H=6", "chebyshev_w"),
    "gen_rank@verified": ArmSemantics("open_loop", "x_0", "H=6", "chebyshev_w"),
    "greedy_pref":       ArmSemantics("greedy", "x_0", "H=6", "chebyshev_w"),
    "verified_pref":     ArmSemantics("verified", "x_0", "H=6", "chebyshev_w"),
    "greedy_pref@w=0.1": ArmSemantics("greedy", "x_0", "H=6", "chebyshev_w0.1"),
    "greedy_pref@w=0.9": ArmSemantics("greedy", "x_0", "H=6", "chebyshev_w0.9"),
    "branch@w_i":        ArmSemantics("greedy", "x_3", "H-3", "chebyshev_w_i"),
    "branch@w_j":        ArmSemantics("greedy", "x_3", "H-3", "chebyshev_w_j"),
}

LANE_CONTRASTS = (
    # P1 is CONTEXT_ONLY, and this gate is why. An arm with no controller cannot
    # have an objective either, so `greedy_pref` vs `unguided` unavoidably varies
    # BOTH -- exactly the shape of the confound the main lane's audit caught in
    # `verified_retarget` vs `continue_A`. It is kept as the floor, reported with
    # the confound stated, and it may not carry a headline. The parity-clean
    # forms of the same question are P3 (closed vs open loop under the same
    # objective) and P5 (objective only).
    Contrast("P1_unguided_floor", "greedy_pref", "unguided", "objective",
             status="CONTEXT_ONLY"),
    Contrast("P2_future_awareness", "verified_pref", "greedy_pref", "controller"),
    # The affordable end of the bracket: gen_rank matched on KERNEL calls. The
    # native-matched end needs ~2,600 unguided trajectories per source (~15,600
    # kernel calls, ~30 h) and is costed for main rather than run here.
    Contrast("P3_closed_vs_open_loop", "greedy_pref", "gen_rank@greedy", "controller",
             budget_parity_axis="kernel"),
    Contrast("P4_closed_vs_open_loop_verified", "verified_pref", "gen_rank@verified",
             "controller", budget_parity_axis="kernel"),
    Contrast("P5_preference_responsiveness", "greedy_pref@w=0.9", "greedy_pref@w=0.1",
             "objective"),
    Contrast("P6_same_prefix_branching", "branch@w_i", "branch@w_j", "objective"),
)

LANE_STATISTICS = (
    Statistic(
        name="normalized_hypervolume",
        hypothesis="preference control covers more of the achievable front",
        falsifying_range="[0, 1]; a worse controller scores lower",
        selection_function="chebyshev_argmin_per_step",
        scoring_function="hypervolume_over_committed_endpoints"),
    Statistic(
        name="hv_auc_native",
        hypothesis="control reaches good fronts sooner per distinct molecule scored",
        falsifying_range="[0, 1]",
        selection_function="chebyshev_argmin_per_step",
        scoring_function="hv_auc_vs_native_calls"),
    Statistic(
        name="hv_auc_raw",
        hypothesis="control reaches good fronts sooner per scoring invocation",
        falsifying_range="[0, 1]",
        selection_function="chebyshev_argmin_per_step",
        scoring_function="hv_auc_vs_raw_calls"),
    Statistic(
        name="preference_coverage",
        hypothesis="each preference is served by its own endpoint",
        falsifying_range="[0.0, 1.0]; 0 means all five preferences collapsed",
        selection_function="chebyshev_argmin_per_step",
        scoring_function="unique_argmin_within_endpoint_set"),
    Statistic(
        name="nondominated_set_size",
        hypothesis="the five endpoints span a front rather than a point",
        falsifying_range="[1, 5]; 1 means one endpoint dominates the rest",
        selection_function="chebyshev_argmin_per_step",
        scoring_function="pareto_front_indices"),
    Statistic(
        name="endpoint_structural_diversity",
        hypothesis="different preferences produce structurally different molecules",
        falsifying_range="[0, 1]; 0 means identical endpoints",
        selection_function="chebyshev_argmin_per_step",
        scoring_function="one_minus_mean_pairwise_tanimoto"),
    Statistic(
        name="top1_disagreement_rate",
        hypothesis="the lookahead would act differently from greedy",
        falsifying_range="[0, 1]; 0 means the lookahead never differs",
        selection_function="argmin_V_G",
        scoring_function="indicator_of_difference_from_greedy"),
    Statistic(
        name="verified_minus_greedy_endpoint_scalarized_loss",
        hypothesis="future-aware control lands better",
        falsifying_range="magnitude in [0, inf); the SIGN is guaranteed by policy "
                         "improvement and carries no information",
        selection_function="argmin_V_G",
        scoring_function="endpoint_scalarization_of_a_separately_run_greedy_arm",
        sign_is_guaranteed=True,
        reports_pvalue=False),
    Statistic(
        name="feasibility",
        hypothesis="arms complete the full budget",
        falsifying_range="[0, 1]",
        selection_function="none",
        scoring_function="fraction_of_complete_trajectories"),
)

#: Statistics this lane has WITHDRAWN. Kept declared so a future run cannot
#: quietly reintroduce them.
WITHDRAWN_STATISTICS = {
    "sacrifice_to_win": "circular: action chosen as argmax V_G, then scored by V_G",
    "verified_vs_greedy_sign_test": "policy improvement makes the null of 0.5 false",
    "best_candidate_reaches_pooled_p99": (
        "with a ~589-wide fiber this fires with probability 0.997 whatever the "
        "truth is; withdrawn from census gate G4 before any pair verdict was read"),
}


def run_gate(*, manifest: Path | None = None,
             action_sequences: dict[str, dict[str, tuple]] | None = None,
             costs: dict[str, dict[str, float]] | None = None,
             endpoint_counts: dict[str, int] | None = None) -> GateReport:
    report = GateReport()
    check_d1_selection_scoring_distinct(LANE_STATISTICS, report)
    check_d1b_falsifying_range_declared(LANE_STATISTICS, report)
    check_d3_no_false_null(LANE_STATISTICS, report)
    check_d5_parity(LANE_CONTRASTS, LANE_ARMS, report)
    if manifest is not None:
        check_d4_manifest_hashes(manifest, REPO, report)
    if action_sequences is not None:
        check_d2_arms_are_distinct(
            action_sequences,
            [("verified_pref", "greedy_pref"), ("greedy_pref", "unguided")], report)
    if costs is not None and endpoint_counts is not None:
        check_d6_hv_budget_matched(LANE_CONTRASTS, costs, endpoint_counts, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path,
                        default=REPO / "docs/workstreams/pareto-control/handoff.json")
    parser.add_argument("--results", type=Path, default=None,
                        help="run artifact carrying action sequences, costs and "
                             "endpoint counts; omit for the design-stage self-test")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    kwargs: dict[str, Any] = {}
    if args.manifest.exists():
        kwargs["manifest"] = args.manifest
    if args.results is not None:
        data = json.loads(args.results.read_text())
        kwargs["action_sequences"] = {
            arm: {src: tuple(seq) for src, seq in per_source.items()}
            for arm, per_source in data.get("action_sequences", {}).items()}
        kwargs["costs"] = data.get("costs")
        kwargs["endpoint_counts"] = data.get("endpoint_counts")

    report = run_gate(**kwargs)
    print("INSTRUMENT GATE" + ("" if args.results else "  (design-stage self-test)"))
    for check in report.checks:
        print(f"  {'PASS' if check['pass'] else 'FAIL'}  {check['check']}")
        if not check["pass"]:
            print(f"        {json.dumps(check['detail'])[:400]}")
    print(f"\nwithdrawn statistics that may not be reintroduced: "
          f"{', '.join(sorted(WITHDRAWN_STATISTICS))}")

    payload = {"schema": "compose.pareto.instrument_gate",
               "status": "DESIGN_ONLY" if not args.results else "SMOKE_HELD_IN",
               "passed": report.passed, "checks": report.checks,
               "withdrawn_statistics": WITHDRAWN_STATISTICS}
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"wrote {args.out}")

    if not report.passed:
        print("\nGATE FAILED -- these statistics may not be reported.")
        return 1
    print("\nGATE PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
