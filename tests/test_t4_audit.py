"""Smoke test for the human-readable three-level T4 audit."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_t4_audit_reports_option_and_decoration_diagnostics(tmp_path: Path) -> None:
    payload = {
        "schema_version": "t4_three_level_option_audit_v1",
        "cell": "macro_prior_parp1_0_d0.4",
        "n_dock": 1,
        "best_ds": -8.0,
        "controller": {
            "factorization": "Q(M|x,z) -> Q(o|x,M,z) -> q(w|x,M,o)",
            "region_prior": "mu_exec",
            "kappa": 1.0,
            "epsilon_option": 0.1,
        },
        "rounds": [
            {
                "round": 1,
                "n_parent_batches": 8,
                "n_region_draws": 24,
                "n_bundles_selected": 24,
                "n_bundles_docked": 1,
                "bundle_scopes": [0.4],
                "options_selected": {"cyclize": 1},
                "options_docked": {"cyclize": 1},
                "candidates_by_option": {"cyclize": 1},
                "candidate_option_diagnostics": {"cyclize": {"n": 1}},
                "docked_option_diagnostics": {"cyclize": {"n": 1}},
                "n_cand": 1,
                "n_unique_candidates": 1,
                "candidate_diversity": 0.0,
                "docked_diversity": 0.0,
                "t_propose": 1.0,
                "t_dock": 2.0,
                "bundles": [],
                "docked": [
                    {
                        "smi": "C1CC1",
                        "ds": -8.0,
                        "option": "cyclize",
                        "r_release": 0.4,
                        "r_coherent": 0.2,
                        "d_rings": 1,
                        "d_cycle_rank": 1,
                        "d_heavy": 0,
                        "added_terminal_halogen": 0,
                        "added_backbone_atoms": 0,
                        "added_sulfur": 0,
                        "interface": "segment",
                        "step": 1,
                    }
                ],
            }
        ],
    }
    path = tmp_path / "audit.json"
    path.write_text(json.dumps(payload))
    root = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [sys.executable, str(root / "tools" / "t4_audit.py"), str(path)],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "options selected: {'cyclize': 1}" in completed.stdout
    assert "terminal_halogen=0 sulfur=0 backbone_CNO=0" in completed.stdout
    assert "constructive ring outcomes" in completed.stdout
