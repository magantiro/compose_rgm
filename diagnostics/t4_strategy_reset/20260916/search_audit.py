"""Allocation, proposal funnels, and secondary historical-data inventory."""

import json
from collections import Counter, defaultdict

from audit import CODE, INPUTS, OUT, identity, read, stats


def main():
    audit = read(OUT / "audit.json")
    rows = [json.loads(s) for s in (OUT / "scored_rows.jsonl").read_text().splitlines()]
    by_run = defaultdict(list)
    for r in rows:
        by_run[r["run"]].append(r)
    allocation = []
    for unit in audit["units"]:
        rs = by_run[unit["run"]]
        if not unit["query_mapping_verified"] or len(rs) < 20:
            continue
        rs = sorted(rs, key=lambda r: r["query"])
        best = float("inf")
        paid_near = paid_far = gains_parent = gains_best = 0
        channels = defaultdict(Counter)
        improvement_rows = []
        for r in rs:
            if r["parent_score"] is not None:
                gap = r["parent_score"] - best
                paid_near += int(gap <= 1)
                paid_far += int(gap > 1)
                parent_gain = r["score"] < r["parent_score"]
                gains_parent += int(parent_gain)
            else:
                parent_gain = None
                gap = None
            best_gain = r["score"] < best
            if best_gain and best != float("inf"):
                gains_best += 1
                improvement_rows.append(
                    {
                        "query": r["query"],
                        "score": r["score"],
                        "prior_best": best,
                        "parent_score": r["parent_score"],
                        "parent_gap_to_best": gap,
                        "planner_channel": r["planner_channel"],
                        "channel": r["channel"],
                    }
                )
            c = channels[r["planner_channel"] or r["channel"]]
            c["scored"] += 1
            c["parent_improvements"] += int(parent_gain is True)
            c["best_improvements_excluding_first"] += int(best_gain and best != float("inf"))
            best = min(best, r["score"])
        reasons = Counter()
        bootstrap_unique = set()
        first_prefixes = []
        for p in sorted((OUT / "raw").glob(unit["run"] + "_rounds_*_batch.json.gz")):
            saved = read(p)
            attempts = saved["batch"]["attempts"]
            for a in attempts:
                if a.get("status") == "ineligible":
                    reasons.update(a.get("properties", {}).get("endpoint_exclusion_reasons", []))
            if saved["bootstrap"]:
                first_prefixes.append([identity(a) for a in attempts])
                bootstrap_unique.update(identity(a) for a in attempts)
        allocation.append(
            {
                "run": unit["run"],
                "scored": len(rs),
                "paid_parent_within_one_of_incumbent": paid_near,
                "paid_parent_more_than_one_worse": paid_far,
                "parent_improvements": gains_parent,
                "best_improvements_excluding_first": gains_best,
                "channel_efficiency": {k: dict(v) for k, v in channels.items()},
                "improving_calls": improvement_rows,
                "ineligible_reasons": dict(reasons),
                "unique_bootstrap_attempts": len(bootstrap_unique),
            }
        )
    empty = []
    for unit in audit["units"]:
        if unit["query_count"]:
            continue
        attempts = []
        by_round = []
        reasons = Counter()
        for p in sorted((OUT / "raw").glob(unit["run"] + "_rounds_*_batch.json.gz")):
            s = read(p)
            a = s["batch"]["attempts"]
            attempts.extend(a)
            by_round.append([identity(v) for v in a])
            for v in a:
                reasons.update(v.get("properties", {}).get("endpoint_exclusion_reasons", []))
        smallest = min(map(len, by_round), default=0)
        empty.append(
            {
                "run": unit["run"],
                "logged_attempts": len(attempts),
                "unique_attempts": len({identity(a) for a in attempts}),
                "same_common_prefix": bool(by_round)
                and all(r[:smallest] == by_round[0][:smallest] for r in by_round),
                "ineligible_reasons": dict(reasons),
            }
        )

    secondary = []
    named = {
        "program_curriculum": "diagnostics/t4_program_curriculum/attempt_1/remote_result.json",
        "program_pool": "diagnostics/t4_program_pool/attempt_1/remote_result.json",
        "program_replay": "diagnostics/t4_program_pool/attempt_1/scored_program_replay.json",
        "winner_refinement": "diagnostics/t4_winner_refinement/attempt_1/resumed_result.json",
        "matched_reference": "diagnostics/t4_matched_repaired/attempt_1/reference/docking.json",
        "matched_committor": "diagnostics/t4_matched_repaired/attempt_1/committor/docking.json",
    }
    paths = [(name, CODE / path) for name, path in named.items()]
    paths += [
        ("parent_edit_cycle", p)
        for p in sorted((CODE / "diagnostics/parent_edit_cycles/t4_units").glob("*.json"))
    ]
    paths += [
        ("utility_acquisition", p)
        for p in sorted(
            (CODE / "diagnostics/t4_utility_data_acquisition_launch/attempt_1/requests").glob(
                "*/result.json"
            )
        )
    ]
    for family, path in paths:
        obj = read(path)
        records = obj.get("rows", obj.get("docked", obj.get("records", [])))
        if family == "utility_acquisition" and isinstance(obj.get("score"), (int, float)):
            records = [obj]
        if family == "program_replay":
            records = [
                {**r, "docking_score": r.get("oracle_label", {}).get("docking_score")}
                for r in records
            ]
        if not isinstance(records, list):
            secondary.append({"family": family, "path": str(path), "status": "nonlist_unadmitted"})
            continue
        scorekey = next(
            (
                k
                for k in ("score", "ds", "docking_score")
                if any(isinstance(r.get(k), (int, float)) for r in records)
            ),
            None,
        )
        observed = [r for r in records if scorekey and isinstance(r.get(scorekey), (int, float))]
        secondary.append(
            {
                "family": family,
                "path": str(path),
                "rows": len(records),
                "numeric_score_rows": len(observed),
                "score_field": scorekey,
                "with_protocol": sum(bool(r.get("oracle_protocol")) for r in observed),
                "with_receipt_id": sum(bool(r.get("receipt_id")) for r in observed),
                "with_trace": sum(bool(r.get("trace")) for r in observed),
                "with_call_or_index": sum(
                    r.get("index", r.get("query_index")) is not None for r in observed
                ),
                "snapshot_entries": len(obj.get("snapshot", {}).get("entries", {})),
                "status": "separate_inventory_not_added_to_unique_call_total",
            }
        )
    repeated = audit["repeated_identical_endpoints"]
    result = {
        "schema": "t4_strategy_reset_search_audit_v1",
        "new_oracle_calls": 0,
        "allocation": allocation,
        "zero_yield": empty,
        "secondary_inventory": secondary,
        "repeated_endpoint_ranges": stats([r["range"] for r in repeated]),
        "repeated_all_negative_ranges": stats(
            [r["range"] for r in repeated if max(r["scores"]) < 0]
        ),
        "repeated_all_better_than_minus8_ranges": stats(
            [r["range"] for r in repeated if max(r["scores"]) <= -8]
        ),
        "inputs_sha256": INPUTS,
        "limitations": [
            "Parent quality counted against observed best before each call, not a causal policy counterfactual.",
            "Historical duplicate score variation is an observation, not attributed conclusively to a specific pipeline component.",
            "Secondary artifacts can overlap primary calls or replay old labels; deliberately not added to the independent-call total.",
        ],
    }
    (OUT / "search_audit.json").write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")
    print(
        json.dumps(
            {
                "allocation_runs": len(allocation),
                "zero_yield": empty,
                "secondary": secondary,
                "repeat_ranges": result["repeated_endpoint_ranges"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
