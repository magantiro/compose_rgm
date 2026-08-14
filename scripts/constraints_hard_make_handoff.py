"""Regenerate `docs/workstreams/constraints-hard/handoff.json`.

Every digest is recomputed from the file on disk; no hash is ever typed by hand.
`tests/test_hard_scaffold_constraint.py::test_handoff_manifest_digests_recompute`
fails if the manifest drifts from the tree, which is what keeps the provenance
machine-checkable rather than merely asserted.

    python3 scripts/constraints_hard_make_handoff.py

Repo root is derived from this file's location; override with COMPOSE_REPO_ROOT.
"""

import hashlib
import json
import os
import subprocess
from pathlib import Path

ROOT = Path(os.environ.get("COMPOSE_REPO_ROOT", Path(__file__).resolve().parents[1]))


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else "MISSING"


def git(*args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(ROOT), *args], capture_output=True, text=True
    ).stdout.strip()


ARTIFACTS = [
    "docs/workstreams/constraints-hard/STATUS.md",
    "docs/workstreams/constraints-hard/EXTERNAL_HARD_CONSTRAINT_AUDIT.md",
    "docs/workstreams/constraints-hard/CONSTRAINTS_SECTION_TRILEMMA.md",
    "docs/workstreams/constraints-hard/LANE2_FIVE_QUESTIONS.md",
    "docs/workstreams/constraints-hard/PROTOCOL.md",
    "docs/workstreams/constraints-hard/CONSTRAINT_SEMANTICS.md",
    "docs/workstreams/constraints-hard/SCAFFOLD_FEASIBILITY.md",
    "docs/workstreams/constraints-hard/BASELINE_TASK_MATRIX.md",
    "docs/workstreams/constraints-hard/SAME_LAB_LINEAGE.md",
    "docs/workstreams/constraints-hard/DECISION_LOG.md",
    "docs/workstreams/constraints-hard/HANDOFF.md",
    "docs/workstreams/constraints-hard/probes/identity_probe.py",
    "docs/workstreams/constraints-hard/probes/gate0_local_authentication_repro.py",
    "scripts/constraints_hard_scaffold_census.py",
    "scripts/constraints_hard_make_handoff.py",
    "src/compose_v4/experiments/constraints_hard_mask.py",
    "tests/test_hard_scaffold_constraint.py",
    "diagnostics/constraints_hard_scaffold_census.json",
]

FROZEN_INPUTS = {
    "held_in_pool": "diagnostics/editing_v2_matched_validation_reserve_ids.json.gz",
    "held_in_cohort": "diagnostics/retarget_calibration_cohort.json",
}

payload = {
    "schema": "compose.workstream.handoff",
    "schema_version": 1,
    "workstream": "constraints-hard",
    "lane": 6,
    "claim_id": "experiment_a_hard_structural_invariant",
    "branch": "codex/compose-constraints-hard",
    "base_commit": "f6146d7",
    "head_commit": git("rev-parse", "HEAD"),
    "head_commit_note": (
        "the commit containing every deliverable; handoff.json itself is "
        "committed in the immediately following commit, since it records this hash"
    ),
    "working_tree_clean_excluding_this_manifest": [
        line
        for line in git("status", "--porcelain").splitlines()
        if "handoff.json" not in line
    ]
    == [],
    "status": "DESIGN_ONLY",
    "lane_scope": "EXTERNAL_AUDIT_ONLY (narrowed 2026-08-13)",
    "governed_by": [
        "docs/COMPARATOR_ROLES_CANONICAL.md",
        "docs/AMENDMENT_PUBLISHED_NUMBER_FIRST.md",
    ],
    "comparator_roles_constraints_block": {
        "CDD": {"role": "FRAMEWORK_NEIGHBOR", "tier": 4, "evidence": "conceptual only"},
        "PRODIGY": {"role": "FRAMEWORK_NEIGHBOR", "tier": 4, "evidence": "conceptual only"},
        "ConStruct": {"role": "FRAMEWORK_NEIGHBOR", "tier": 4, "evidence": "conceptual only"},
        "posthoc_vs_soft_vs_exact_support": {"role": "MATCHED_CAUSAL_CONTROL", "owner": "Lane 2"},
        "GraphXForm": {"role": "TASK_COMPETENCE", "tier": 3},
        "Prompt-MolOpt^P": {"role": "TASK_COMPETENCE", "tier": 3},
        "MolEditRL": {"role": "TASK_COMPETENCE", "tier": 4},
        "InVirtuoGen": {"role": "TASK_COMPETENCE", "tier": 4, "blocker": "CC-BY-NC-SA"},
        "DDSBM": {"role": "FRAMEWORK_NEIGHBOR", "tier": 4, "scope": "general editing block; Lane 5 owns"},
        "GrIDDD": {"role": "FRAMEWORK_NEIGHBOR", "tier": "UNVERIFIED", "scope": "Lane 5 owns; not audited here"},
        "Edit Flows": {"role": "CONCEPTUAL_LINEAGE_ONLY", "tier": 4},
    },
    "no_tier_1_or_2_method_in_this_block": True,
    "constraints_section_cannot_be_carried_by_published_numbers": True,
    "cdd_task_reusable_verbatim": {
        "predicate": "SA(y) <= tau",
        "thresholds": [3.0, 3.5, 4.0, 4.5],
        "scorer": "rdkit.Contrib.SA_Score.sascorer via molecular_quality.py",
        "conditions": [
            "those four thresholds and no others",
            "pin our own RDKit version and state it - CDD pins none and SA is not version-portable",
            "no re-tuning of tau to suit our fiber",
        ],
        "why": "an externally fixed threshold cannot be selected on our outcome, which is the defect that killed the Bemis-Murcko branch",
        "note": "the task travels; the numbers do not - CDD stays contextual, non-head-to-head",
    },
    "lane2_open_decisions": {
        "owner": "Lane 2",
        "artifact": "docs/workstreams/constraints-hard/LANE2_FIVE_QUESTIONS.md",
        "handoff_self_sufficient": False,
        "O1_soft_guidance_arm_undefined": {
            "severity": "LOAD_BEARING - the experiment cannot be built without it",
            "needs": "functional form of the reweighting, registered as LAW_ONLY",
            "binding_constraint": "ONE pre-registered lambda from a stated requirement; no sweep-and-pick. A lambda chosen to make soft guidance LOSE is as invalid as one chosen to make it win",
        },
        "O2_predicate_choice": {
            "recommended": "CDD SA(y) <= tau, exogenous",
            "alternative": "Lane 2's own frozen corridor, endogenous",
            "reason": "an externally fixed threshold structurally cannot repeat the outcome-selection defect that killed Bemis-Murcko",
            "easy_to_miss": "pin our own RDKit version and state it; CDD pins none and SA is not version-portable",
        },
        "O3_gate_thresholds": "reuse Lane 2's frozen A2 criteria verbatim (V3>=20, V4a>=0.10, V4b<=0.05, V5a>1/3, V5b<=0.50); do not widen if a predicate fails",
        "O4_panel_horizon_budget": "entirely Lane 2's; Lane 6's ~170 s per arm-source anchor was measured for a different predicate and should be re-derived",
    },
    "open_questions_for_lead": [
        "confirm FRAMEWORK_NEIGHBOR vs CONCEPTUAL_LINEAGE_ONLY for CDD/PRODIGY/ConStruct under a strict reading of the four role definitions",
        "tier-1 audit for GraphXForm and Prompt-MolOpt^P not discharged - check whether COMPOSE can run under their published protocols before authorizing any rerun",
        "InVirtuoGen CC-BY-NC-SA is a licence decision, not this lane's to make",
    ],
    "artifact_statuses": {
        "diagnostics/constraints_hard_scaffold_census.json": "SMOKE_HELD_IN",
        "docs/workstreams/constraints-hard/PROTOCOL.md": "SUPERSEDED",
        "docs/workstreams/constraints-hard/EXTERNAL_HARD_CONSTRAINT_AUDIT.md": "DESIGN_ONLY",
        "_all_other_deliverables": "DESIGN_ONLY",
    },
    "internal_experiment_designed": False,
    "scaffold_stop_honored": True,
    "smaller_core_substituted": False,
    "external_audit": {
        "methods": ["CDD", "PRODIGY", "ConStruct"],
        "predicate_instantiable_verbatim": "SA(y) <= tau, via rdkit.Contrib.SA_Score.sascorer already in molecular_quality.py",
        "mechanism_instantiable": False,
        "head_to_head_comparable": False,
        "classification": "external hard-constraint competence benchmark - contextual, non-head-to-head",
        "decisive_finding": "CDD reports 0.0% violations at every tau, but over VALID MOLECULES ONLY, and validity collapses 895->353 (~60%) at tau=3.0. Thm 4.1 is a contraction bound assuming beta-prox-regular C with guarantees claimed only for convex C, so the 0% is empirical not structural. Gradients come from a GPT-2 (124M) surrogate, not sascorer.",
        "cdd_official_code_status": "PLACEHOLDER - repo jacobchristopher/CDD is a 59-byte README saying code will be added; two unofficial third-party reimplementations exist and must NOT be used as the published method",
        "cdd_protocol_unstated": ["sample count", "seeds", "variance/error bars", "train/val/test split", "canonicalization/dedup", "RDKit and sascorer version pin"],
        "cdd_companion_workshop_paper": "Constrained Molecular Generation with Discrete Diffusion for Drug Discovery (AI4D3 2025) adds a 3-membered-heterocycle ABSENCE constraint via RDKit substructure matching, CDD 0.0% violations - the closest any audited method comes to an arbitrary boolean structural predicate, achieved by a bespoke hand-written operator rather than the ALM projection",
        "prodigy_license": "NONE - no LICENSE file; same blocker class as DDSBM",
        "construct_nearest_analogue": "Appendix G.2 applies the projector at sampling time to an unconstrained QM9 model without constraint-specific training: 100.0 acyclicity at 99.8 validity. Appendix G.1 planarity is the authors own NEGATIVE result - too loose a constraint, slightly harming performance.",
        "cdd_venue": "NeurIPS 2025 - CONFIRMED from camera-ready footer, DOI 10.52202/085713-0415",
        "cdd_thresholds_confirmed": [3.0, 3.5, 4.0, 4.5],
        "cdd_official_code": None,
    },
    "held_out_opened": False,
    "held_out_note": (
        "held-in training_source_keys only; reserve_source_keys is deleted from "
        "the loaded payload before any sweep"
    ),
    "claim_bearing_compute_run": False,
    "modal_launched": False,
    "gpu_used": False,
    "external_dependencies_installed": [],
    "executor_semantics_verdict": "LABELED_SUBGRAPH_PRESENCE_INVARIANT",
    "identity_invariant_provable": False,
    "protected_object_rule": "atom_and_bond_labeled_bemis_murcko_scaffold_v1",
    "thresholds_reused_verbatim_from": {
        "lane": "codex/compose-pathwise-constraints",
        "tip": "bcc4a40",
        "file": "src/compose_v4/experiments/pathwise_constraints.py",
        "constants": {
            "MIN_CORE_ATOMS": 6,
            "CORE_FRACTION_BAND": [0.20, 0.70],
            "MIN_FREE_ATOMS": 8,
            "HEAVY_ATOM_BAND": [18, 38],
        },
        "viability_criteria": {
            "V3_min_events": 20,
            "V4a_pooled_median_support_retention_floor": 0.10,
            "V4b_mask_empty_ceiling": 0.05,
            "V5a_source_spread_floor": 1 / 3,
            "V5b_single_source_share_ceiling": 0.50,
        },
        "note": "reused, not minted; applied to a strictly larger protected object, which is the conservative direction",
    },
    "results": {
        "held_in_pool_applicability": {
            "sources": 96094,
            "eligible": 21166,
            "eligible_fraction": 0.2203,
            "core_fraction_median": 0.7895,
            "free_atoms_median": 5.0,
        },
        "cohort_applicability": {"sources": 30, "eligible": 6},
        "fiber_quantities_measured": None,
    },
    "blockers": [
        {
            "id": "gate0_local_authentication",
            "blocks": "Stage 1b fiber census and all Stage 2 runs",
            "cause": "source_index_sha256 embeds the absolute Active8 mount path",
            "reference": "docs/PARETO_PARITY_ENVIRONMENT_STATUS.md",
            "code": "src/compose_v4/data/editing_v2_process_v2_gate_zero.py:593",
            "reproduced_by": "docs/workstreams/constraints-hard/probes/gate0_local_authentication_repro.py",
            "bypassed": False,
        }
    ],
    "unmet_prerequisites": [
        {
            "what": "pathwise_constraints.fragment_smarts / preserves_motif",
            "why": "the predicate must reuse Lane 2's exact-label matcher, not reimplement it",
            "where": "branch codex/compose-pathwise-constraints (unmerged)",
        },
        {
            "what": "Lane 3 comparator registry + oracle_accounting + shared_evaluator",
            "why": "reuse mandated; this lane rebuilt none of it",
            "where": "branch codex/compose-baseline-qualification (unmerged)",
        },
    ],
    "baseline_status": {
        "GraphXForm": "MUST_RUN as source-conditioned editor; N/A for the hard-constraint cell (no user-specified preservation, add-only action space)",
        "Prompt-MolOpt^P": "MUST_RUN - the one direct structure-preserving editor",
        "MolEditRL": "OFFICIAL_CODE_UNAVAILABLE",
        "InVirtuoGen": "CONDITIONAL - token-level clamp, CC-BY-NC-SA",
        "ConStruct": "CONTEXT_ONLY - cite, do not port",
        "DDSBM": "CONTEXT_ONLY - no LICENSE file, no checkpoints",
    },
    "costed_stage_2": {
        "basis": "measured from diagnostics/pathwise_constraints_smoke.json cost block",
        "per_arm_source_seconds": 170,
        "arms": 4,
        "sources": 24,
        "serial_cpu_hours": 4.5,
        "plus_stage_1b_fiber_census_cpu_hours": 1.1,
        "total_cpu_hours": 5.6,
        "wall_minutes_at_24_way_fanout": "12-25",
        "uncertainty_band": "0.7x-2.5x",
        "authorization_required": True,
    },
    "tests": {
        "tests/test_hard_scaffold_constraint.py": {"passed": 13, "failed": 0},
    },
    "reproduction_commands": [
        "PYTHONPATH=src python3 docs/workstreams/constraints-hard/probes/identity_probe.py",
        "PYTHONPATH=src python3 -m pytest tests/test_hard_scaffold_constraint.py -q",
        "python3 scripts/constraints_hard_scaffold_census.py --cohort diagnostics/retarget_calibration_cohort.json --pool diagnostics/editing_v2_matched_validation_reserve_ids.json.gz --out diagnostics/constraints_hard_scaffold_census.json",
        "python3 docs/workstreams/constraints-hard/probes/gate0_local_authentication_repro.py --local-runtime <local_runtime>",
    ],
    "rdkit_version_used": "2025.09.6",
    "rdkit_production_pin": "2024.03.5",
    "rdkit_pin_matches_production": False,
    "artifact_sha256": {p: sha(ROOT / p) for p in ARTIFACTS},
    "frozen_input_sha256": {k: sha(ROOT / v) for k, v in FROZEN_INPUTS.items()},
    "frozen_input_paths": FROZEN_INPUTS,
    "files_modified_belonging_to_other_lanes": [],
    "recommended_next_action": (
        "Decide whether the related-work paragraph in "
        "EXTERNAL_HARD_CONSTRAINT_AUDIT.md section 5 goes into the paper as "
        "drafted, and whether to cite ConStruct Appendix G.2 as the nearest "
        "published analogue to COMPOSE's setup."
    ),
    "self_corrections": [
        "earlier report said CDD's venue was unverified/possibly ICML 2025; it is NeurIPS 2025",
        "earlier report said CDD reports 21.3%/63.9% satisfaction; those numbers are not in the paper - CDD reports 0.0% violations on a valid-only denominator",
    ],
    "superseded": {
        "docs/workstreams/constraints-hard/PROTOCOL.md": (
            "yield contrast folded into Lane 2 (this lane's own accepted "
            "recommendation); protected object failed its feasibility census"
        )
    },
    "binding_prohibitions": [
        "no smaller protected core may be substituted for the Bemis-Murcko scaffold",
        "no new constraint may be invented from our own failed scaffold result",
        "COMPOSE's 100% constraint satisfaction is a construction check, never an empirical win",
        "no CDD number may be presented as a rerun; cite as reported or not at all",
        "no CDD violation rate may be quoted without its valid-only denominator",
        "the 203.4% CFG/CBG headline is a cross-paper comparison copied from Schiff et al.; do not propagate it",
    ],
}

out = ROOT / "docs/workstreams/constraints-hard/handoff.json"
out.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n")
print(f"wrote {out}")
print(f"head_commit={payload['head_commit']}")
print(
    "clean_excluding_this_manifest="
    f"{payload['working_tree_clean_excluding_this_manifest']}"
)
