"""How much of the real PMO proposal stream routes through the prior's call site?

A prior wired into one call site can only matter in proportion to how often that
site runs.  This counts, from the COMPLETED run's own archived entries, how many
proposals carry a `current_state_edit` record -- the detail dict that
`current_state_program` emits and `ProgramOptimizer._mutate` files under
`metadata`.  Zero oracle calls; it reads artifacts only.
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

ROOT = Path("/Users/rmaganti/compose_pmo_replay_data")
TASKS = ("gsk3b", "celecoxib_rediscovery", "perindopril_mpo")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    entries = 0
    site = 0
    families: collections.Counter = collections.Counter()
    per_task: dict[str, dict] = {}
    other_lanes: collections.Counter = collections.Counter()
    for task in TASKS:
        t_entries = t_site = 0
        for round_dir in sorted((ROOT / task / "campaign").glob("round_*")):
            path = round_dir / "complete.json"
            if not path.exists():
                continue
            snapshot = json.loads(path.read_text())["snapshot"]
            for value in snapshot["entries"].values():
                entries += 1
                t_entries += 1
                metadata = (value.get("provenance") or {}).get("metadata") or {}
                if not isinstance(metadata, dict):
                    continue
                for lane in ("dynamic_generic_composition", "dynamic_v1_structured_program",
                             "mutations", "proposal_kind"):
                    if lane in metadata:
                        other_lanes[lane] += 1
                record = metadata.get("current_state_edit")
                if isinstance(record, dict):
                    site += 1
                    t_site += 1
                    families[record.get("family", "?")] += 1
        per_task[task] = {
            "entries": t_entries,
            "current_state_edits": t_site,
            "pct": round(100.0 * t_site / t_entries, 2) if t_entries else 0.0,
        }

    signal_families = ("atom_restate_semantic", "cycle_close")
    with_signal = sum(families[f] for f in signal_families)
    payload = {
        "schema_version": "pmo_call_site_reach_v1",
        "oracle_calls_spent": 0,
        "run_id": "35fc7bcd4bf5c964294698014c1ed40da483ad013f6b92c6201b01dee2d3d306",
        "method": (
            "Count archived entries whose provenance.metadata carries the "
            "`current_state_edit` detail dict that current_state_program emits."
        ),
        "archived_entries": entries,
        "entries_through_the_prior_call_site": site,
        "call_site_reach_pct": round(100.0 * site / entries, 2) if entries else 0.0,
        "families_at_the_call_site": dict(families.most_common()),
        "families_where_the_prior_has_measured_signal": list(signal_families),
        "pct_of_call_site_draws_with_signal": (
            round(100.0 * with_signal / site, 2) if site else 0.0
        ),
        "effective_reach_pct": (
            round(100.0 * with_signal / entries, 2) if entries else 0.0
        ),
        "other_proposal_lanes_seen": dict(other_lanes.most_common()),
        "reading": (
            "Wiring this one site is necessary and not sufficient. The prior can "
            "influence roughly `effective_reach_pct` of all proposals; the larger "
            "generic-composition and structured lanes draw their successors "
            "elsewhere and would each need their own call site."
        ),
        "per_task": per_task,
    }
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(json.dumps({k: payload[k] for k in (
        "archived_entries", "entries_through_the_prior_call_site",
        "call_site_reach_pct", "pct_of_call_site_draws_with_signal",
        "effective_reach_pct", "families_at_the_call_site")}, indent=2))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
