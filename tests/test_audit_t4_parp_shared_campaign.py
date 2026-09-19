from __future__ import annotations

import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "tools/audit_t4_parp_shared_campaign.py"
SPEC = importlib.util.spec_from_file_location("audit_t4_parp_shared_campaign", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)


def test_lineage_uses_the_missing_parent_score_at_each_hop() -> None:
    root = {"smiles": "C", "score": -1.0}
    docked = [
        {
            "smiles": "CC",
            "score": -2.0,
            "parent": "C",
            "parent_score": -1.0,
            "proposal_lane": "route_complete_region",
        },
        {
            "smiles": "CCC",
            "score": -3.0,
            "parent": "CC",
            "parent_score": -2.0,
            "proposal_lane": "shallow",
        },
    ]
    best = {
        "smiles": "CCCC",
        "score": -4.0,
        "parent": "CCC",
        "parent_score": -3.0,
        "proposal_lane": "shallow",
    }

    lineage = AUDIT._lineage(best, docked, root)

    assert [row["score"] for row in lineage] == [-1.0, -2.0, -3.0, -4.0]
    assert [row["proposal_lane"] for row in lineage] == [
        "root",
        "route_complete_region",
        "shallow",
        "shallow",
    ]


def test_lineage_records_the_immediate_missing_parent_score() -> None:
    root = {"smiles": "C", "score": -1.0}
    best = {
        "smiles": "CCCC",
        "score": -4.0,
        "parent": "CCC",
        "parent_score": -3.0,
        "proposal_lane": "shallow",
    }

    lineage = AUDIT._lineage(best, [], root)

    assert lineage[0] == {
        "score": -3.0,
        "smiles": "CCC",
        "proposal_lane": "migrated_archive_parent",
    }
    assert lineage[1]["score"] == -4.0
