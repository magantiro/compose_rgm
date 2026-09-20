"""Consolidated per-source attribution of the T4 round-one zero-candidate failures.

DIAGNOSIS ONLY. Zero oracle calls, zero docking calls, nothing launched on Modal.
Reads only committed diagnostics; writes only `diagnostics/t4_failure_counterfactual_v1.json`.

Inputs, each produced by its own probe in this directory:

  t4_support_stage_audit_v1.json        the pinned funnel: what round one actually did
  t4_bridge_cut_census_v1.json          every single-non-ring-bond prune, uncapped, gated
  t4_prune_closure_probe_v1.json        the exact depth<=3 composed-prune support
  t4_region_law_counterfactual_v1.json  four region-law arms over that same support
  t4_single_edit_family_census_v1.json  the five `current_state_edits` families, gated
  t4_attachment_degeneracy_probe_v1.json  radius-1 address degeneracy of `assignments[0]`

Attribution is DERIVED from those measurements, not asserted: the binding gate comes
from the audit's own single-gate relaxation counts, the minimum depth and mass rank
from the closure, and the axis from whether any counterfactual arm moves the mass.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = "t4_failure_counterfactual_v1"

FAILED = ("braf_0", "braf_1", "fa7_0", "fa7_2", "5ht1b_2")
CONTROLS = ("braf_2", "fa7_1", "5ht1b_0", "5ht1b_1")
#: Each failed cell and the same-target cell that reached eligibility, which shares the
#: protein, the delta, the contract payload and the learned route prior byte for byte.
SAME_TARGET_CONTROL = {
    "braf_0": "braf_2",
    "braf_1": "braf_2",
    "fa7_0": "fa7_1",
    "fa7_2": "fa7_1",
    "5ht1b_2": "5ht1b_0",
}


def _load(name: str) -> dict:
    return json.loads((ROOT / "diagnostics" / name).read_text())


def _binding_gate(audit: dict) -> dict:
    """Which threshold actually blocks, read off the audit's conditional margins.

    Counting how many endpoints a relaxation unlocks cannot rank gates against each
    other, because "QED minus 0.20" and "similarity minus 0.20" are not the same
    quantity. The conditional margins answer it directly: they report, for the set that
    has already cleared the earlier gates, how far the BEST member still is from the
    next one.
    """
    margins = audit["conditional_margins"]
    qed_given_sim = (margins.get("delta_qed_given_similarity_pass") or {}).get("max")
    sim_given_qed = (margins.get("delta_sim_given_qed_pass") or {}).get("max")
    sa_given_both = (margins.get("delta_sa_given_sim_and_qed_pass") or {}).get("max")

    if audit["funnel"]["fully_eligible"]:
        binding = None
    elif sa_given_both is not None and sa_given_both < 0:
        binding = "sa"
    elif (qed_given_sim is not None and qed_given_sim < 0) and (
        sim_given_qed is not None and sim_given_qed < 0
    ):
        binding = "qed_similarity_jointly"
    elif qed_given_sim is not None and qed_given_sim < 0:
        binding = "qed"
    elif sim_given_qed is not None and sim_given_qed < 0:
        binding = "similarity"
    else:
        binding = None
    return {
        "binding_gate": binding,
        "best_qed_margin_among_similarity_passing": qed_given_sim,
        "best_similarity_margin_among_qed_passing": sim_given_qed,
        "best_sa_margin_among_similarity_and_qed_passing": sa_given_both,
        "single_gate_relaxation": audit["single_gate_relaxation"],
    }


#: Shallow-lane draws per cell, from every d06_250 contract's
#: proposal.shallow.draws. Identical across the five proteins.
SHALLOW_DRAWS = 480

#: dynamic_program_synthesis.CAPACITY_AWARE_THRESHOLD, mirrored so this reduction
#: does not import the proposal library.
CAPACITY_AWARE_THRESHOLD = 36


def expected_yield(closure: dict, mode_law: dict, draws: int = SHALLOW_DRAWS) -> dict:
    """Expected eligible endpoints per cell from the pure-prune route.

    E[hits] = draws * sum_k P(program is k bounded prunes) * P(land on an eligible
    state | k bounded prunes), the first factor MEASURED by sampling
    `synthesize_dynamic_program` and the second the exact closure mass at depth k.

    This deliberately covers only the prune route. Grow/replace/restate programs can
    also reach an eligible endpoint, so the number is a LOWER bound on the lane's true
    yield; it is reported because the prune route is the only one that reaches these
    cells' eligible sets at all.
    """
    by_depth: dict[int, float] = {}
    for row in closure["best_eligible"]:
        by_depth[row["depth"]] = by_depth.get(row["depth"], 0.0) + row["path_mass"]
    share = mode_law["all_prune_share_by_depth"]
    per_draw = sum(
        float(share.get(str(depth), 0.0)) * mass for depth, mass in by_depth.items()
    )
    return {
        "draws": draws,
        "eligible_closure_mass_by_depth": {
            str(k): round(v, 8) for k, v in sorted(by_depth.items())
        },
        "all_prune_share_by_depth": share,
        "per_draw_probability": round(per_draw, 8),
        "expected_eligible_endpoints": round(per_draw * draws, 4),
        "poisson_probability_of_zero": round(pow(2.718281828459045, -per_draw * draws), 4),
        "covers": "pure-prune programs only; a LOWER bound on the lane's yield",
    }


def attribute(name: str, sources: dict) -> dict:
    audit = sources["audit"]["cells"][name]
    closure = sources["closure"]["cells"][name]
    arms = sources["arms"]["cells"][name]["arms"]
    edits = sources["edits"]["cells"][name]
    anchors = sources["anchors"]["cells"][name]
    geometry = sources["audit"]["fiber_geometry_control"][name]

    gate = _binding_gate(audit)
    any_arm_has_mass = any(arm["eligible_states"] for arm in arms.values())
    witness = geometry["acyclic_truncation_beam"].get("eligible_witness") or geometry[
        "generic_single_edit_beam"
    ].get("eligible_witness")

    charge_axis_actions = sum(
        family["charge_changing_endpoints"] for family in edits["families"].values()
    )
    single_edit_eligible = edits["total_eligible"]

    reached = audit["funnel"]["fully_eligible"] > 0
    if reached:
        route = (
            "bounded-prune composition"
            if any_arm_has_mass
            else "single current-state edit"
        )
        axis = f"control: reached eligibility via {route}"
        explanation = (
            "This cell produced eligible candidates in round one. It is carried here as "
            "the same-target control, because it shares the protein, delta, contract "
            "payload and learned route prior with the failed cell byte for byte."
        )
    elif any_arm_has_mass:
        axis = "scale x retained-region-choice (R), compounded over modules"
        explanation = (
            "The witness IS inside v1's support. It requires removing one "
            "chemically coherent substituent larger than MAX_SEGMENT_LENGTH, which the "
            "vocabulary can only express as 2-3 independent bounded pendant cuts. The "
            "region law inside each cut is uniform and unconditioned, so the joint mass "
            "is the product of ~1/|cuts| per module."
        )
    elif (
        witness is not None
        and charge_axis_actions == 0
        and edits["net_formal_charge"] != 0
    ):
        axis = "topology / element vocabulary (H): net-charge coordinate absent"
        explanation = (
            "The witness changes the source's net formal charge. Across every action "
            "the five current_state_edits families enumerate, ZERO change net charge, "
            "and no bounded prune can reach the charged ring atom. The axis the witness "
            "moves along is not a proposal axis at all."
        )
    else:
        axis = "none: not a proposal failure"
        explanation = (
            "No arm and no single edit reaches an eligible endpoint, and the audit's own "
            "conditional margins show the similarity-passing set misses the binding gate "
            "by a margin no region law can close. The feasible region under this support "
            "is effectively empty."
        )

    return {
        "role": "failed" if name in FAILED else "control",
        "protein": closure["protein"],
        "smiles": closure["smiles"],
        "source_heavy_atoms": audit["source"]["heavy"],
        "source_qed": audit["source"]["qed"],
        "source_sa": audit["source"]["sa"],
        "observed_round_one": {
            "distinct_endpoints": audit["funnel"]["distinct_endpoints"],
            "similarity_pass": audit["funnel"]["similarity_pass"],
            "qed_pass": audit["funnel"]["qed_pass"],
            "similarity_and_qed_pass": audit["funnel"]["similarity_and_qed_pass"],
            "fully_eligible": audit["funnel"]["fully_eligible"],
            "selected": audit["round_one_decision"]["selected"],
        },
        **gate,
        "known_feasible_witness": witness,
        "proposal_support": {
            "closure_states": closure["closure_states"],
            "eligible_states": closure["eligible_states"],
            "minimum_depth_to_eligible": (
                min(row["depth"] for row in closure["best_eligible"])
                if closure["best_eligible"]
                else None
            ),
            "minimum_single_cut_size_for_eligibility": sources["cuts"]["cells"][name][
                "minimum_eligible_cut_size"
            ],
            "max_segment_length": sources["cuts"]["cells"][name]["max_segment_length"],
            "single_edit_actions_enumerated": edits["total_enumerated"],
            "single_edit_eligible": single_edit_eligible,
            "single_edit_charge_changing": charge_axis_actions,
        },
        "v1_rank_and_mass": {
            "best_eligible_rank": arms["v1"]["best_eligible_rank"],
            "of_states": arms["v1"]["closure_states"],
            "eligible_mass_share": arms["v1"]["eligible_mass_share"],
            "best_eligible_smiles": arms["v1"]["best_eligible_smiles"],
        },
        "counterfactual_arms": {
            arm: {
                "best_eligible_rank": data["best_eligible_rank"],
                "closure_states": data["closure_states"],
                "eligible_mass_share": data["eligible_mass_share"],
            }
            for arm, data in arms.items()
        },
        "attachment": {
            "radius1_degenerate_fraction": anchors["degenerate_fraction"],
            "largest_radius1_class": anchors["largest_class_size"],
            "eligible_cut_anchors": len(anchors["eligible_cut_anchors"]),
            "eligible_anchors_selected_by_assignments_zero": anchors[
                "eligible_anchors_that_are_representative"
            ],
        },
        "attributed_axis": axis,
        "explanation": explanation,
    }


def main() -> None:
    sources = {
        "audit": _load("t4_support_stage_audit_v1.json"),
        "cuts": _load("t4_bridge_cut_census_v1.json"),
        "closure": _load("t4_prune_closure_probe_v1.json"),
        "arms": _load("t4_region_law_counterfactual_v1.json"),
        "edits": _load("t4_single_edit_family_census_v1.json"),
        "anchors": _load("t4_attachment_degeneracy_probe_v1.json"),
    }
    mode_path = ROOT / "diagnostics/t4_proposal_mode_census_v1.json"
    mode = json.loads(mode_path.read_text())["cells"] if mode_path.exists() else {}

    cells = {}
    for name in sorted(set(FAILED) | set(CONTROLS)):
        cells[name] = attribute(name, sources)
        if name in mode:
            cells[name]["realized_mode_law"] = {
                "draws": mode[name]["draws"],
                "all_prune_share": mode[name]["all_prune_share"],
                "all_prune_share_by_depth": mode[name]["all_prune_share_by_depth"],
                "module_count_distribution": mode[name]["module_count_distribution"],
                "measured_on_this_cell": True,
            }

    # The mode law is NOT transferable across the capacity threshold. Below
    # CAPACITY_AWARE_THRESHOLD = 36 heavy atoms `_weighted_module_order` weights all
    # thirteen families equally, so P(first family = substituent_delete) is 1/13; at or
    # above it the near-capacity weights make it 6/18.5. A measured law is therefore
    # borrowed only by cells on the SAME side of that threshold, and cells on the other
    # side get no expected-yield figure rather than a wrong one.
    measured = {
        name: row["realized_mode_law"]
        for name, row in cells.items()
        if "realized_mode_law" in row
    }
    for name, row in cells.items():
        if not row["proposal_support"]["eligible_states"]:
            continue
        near = row["source_heavy_atoms"] >= CAPACITY_AWARE_THRESHOLD
        law = row.get("realized_mode_law")
        borrowed = None
        if law is None:
            donors = [
                donor
                for donor in sorted(measured)
                if (cells[donor]["source_heavy_atoms"] >= CAPACITY_AWARE_THRESHOLD)
                == near
            ]
            if not donors:
                row["expected_yield_from_prune_route"] = {
                    "unavailable": (
                        "no mode law measured on this side of"
                        " CAPACITY_AWARE_THRESHOLD; the family weighting differs"
                    ),
                    "near_capacity": near,
                }
                continue
            borrowed = donors[0]
            law = {**measured[borrowed], "measured_on_this_cell": False}
        row["expected_yield_from_prune_route"] = {
            "near_capacity": near,
            **expected_yield(sources["closure"]["cells"][name], law),
            "mode_law_measured_on_this_cell": law["measured_on_this_cell"],
            "mode_law_borrowed_from": borrowed,
        }

    comparisons = {}
    for failed, control in SAME_TARGET_CONTROL.items():
        left, right = cells[failed], cells[control]
        comparisons[f"{failed}_vs_{control}"] = {
            "failed": failed,
            "control": control,
            "binding_gate": [left["binding_gate"], right["binding_gate"]],
            "minimum_depth_to_eligible": [
                left["proposal_support"]["minimum_depth_to_eligible"],
                right["proposal_support"]["minimum_depth_to_eligible"],
            ],
            "v1_best_eligible_rank": [
                left["v1_rank_and_mass"]["best_eligible_rank"],
                right["v1_rank_and_mass"]["best_eligible_rank"],
            ],
            "v1_eligible_mass_share": [
                left["v1_rank_and_mass"]["eligible_mass_share"],
                right["v1_rank_and_mass"]["eligible_mass_share"],
            ],
            "radius1_degenerate_fraction": [
                left["attachment"]["radius1_degenerate_fraction"],
                right["attachment"]["radius1_degenerate_fraction"],
            ],
            "single_edit_eligible": [
                left["proposal_support"]["single_edit_eligible"],
                right["proposal_support"]["single_edit_eligible"],
            ],
        }

    payload = {
        "schema_version": SCHEMA_VERSION,
        "status": "DIAGNOSTIC_EVIDENCE_ONLY_ZERO_ORACLE_CALLS",
        "oracle_calls": 0,
        "docking_calls": 0,
        "modal_launches": 0,
        "seed_derivation": (
            "shallow lane proposal seed = cell.controller_seed + 1_000_003 * round_index"
            " + 10_007 * 0 + 101 * expert_index, with round_index=1 and expert_index=0"
            " because EXPERTS = ('shallow', 'anchored_replacement',"
            " 'route_complete_region'); round one is therefore deterministic given the"
            " cell."
        ),
        "inputs": {
            key: value.get("schema_version") for key, value in sources.items()
        },
        "failed_cells": list(FAILED),
        "control_cells": list(CONTROLS),
        "cells": cells,
        "failed_versus_same_target_control": comparisons,
    }
    destination = ROOT / "diagnostics/t4_failure_counterfactual_v1.json"
    destination.write_text(json.dumps(payload, indent=1, sort_keys=True))

    print(
        f"{'cell':10} {'role':8} {'gate':10} {'depth':6} {'rank':10} {'mass':8} axis"
    )
    for name, row in cells.items():
        support = row["proposal_support"]
        v1 = row["v1_rank_and_mass"]
        print(
            f"{name:10} {row['role']:8} {row['binding_gate']!s:10} "
            f"{support['minimum_depth_to_eligible']!s:6} "
            f"{str(v1['best_eligible_rank']) + '/' + str(v1['of_states']):10} "
            f"{v1['eligible_mass_share']:8.5f} {row['attributed_axis']}"
        )
    print(f"wrote {destination}")


if __name__ == "__main__":
    main()
