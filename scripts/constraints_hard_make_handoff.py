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
        "decisive_finding": "CDD enforces a differentiable ML surrogate of SA, not RDKit sascorer; measured satisfaction 21.3% at tau=3.0, 63.9% at tau=4.5 - it is not a hard constraint",
        "cdd_venue": "UNVERIFIED - charter said NeurIPS 2025; arXiv:2503.09790 comment suggests ICML 2025 submission; DO NOT CITE A VENUE YET",
        "cdd_thresholds_confirmed": [3.0, 4.5],
        "cdd_thresholds_unverified": [3.5, 4.0],
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
        "Check CDD's venue and full threshold set against the proceedings before "
        "any citation. The charter said NeurIPS 2025; the arXiv comment suggests "
        "an ICML 2025 submission. Then decide whether the related-work paragraph "
        "in EXTERNAL_HARD_CONSTRAINT_AUDIT.md section 5 goes into the paper."
    ),
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
