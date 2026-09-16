"""Same-input historical Dynamic neighborhoods of all 77 teacher witnesses."""

import json
import time
from collections import defaultdict

import numpy as np
from audit import CODE, INPUTS, OUT, molinfo, read, stats
from probes import region_description
from rdkit import DataStructs

from compose_v4.control.edit_program_graph import changed_input_sites
from compose_v4.rewrite.kernel import canonical_state_key
from compose_v4.rewrite.trace_shard import decode_state


def source_span(source, slots):
    """Max shortest-path distance between changed original atoms, not radius."""
    if not slots:
        return None
    n = source.n_atoms
    distance = np.where(source.bonds != 0, 1.0, np.inf)
    np.fill_diagonal(distance, 0)
    for k in range(n):
        distance = np.minimum(distance, distance[:, k, None] + distance[k, None, :])
    value = float(distance[np.ix_(slots, slots)].max())
    return value if np.isfinite(value) else None


def main():
    began = time.monotonic()
    rows_path = OUT / "scored_rows.jsonl"
    import hashlib

    INPUTS[str(rows_path)] = hashlib.sha256(rows_path.read_bytes()).hexdigest()
    rows = [json.loads(line) for line in rows_path.read_text().splitlines()]
    histories = defaultdict(list)
    for row in rows:
        if row["arm"] in {"v0", "v1", "v21"}:
            histories[(row["target"], row["input_smiles"], row["arm"])].append(row)
    old = read(CODE / "diagnostics/t4_known_good_transformation_forensics/attempt_1/result.json")
    result_rows = []
    for record in old["routes"]:
        witness = read(CODE / record["teacher_receipt"])["path"]
        source, product = map(decode_state, (witness["states"][0], witness["states"][-1]))
        src, endpoint = map(canonical_state_key, (source, product))
        change = changed_input_sites(source, product, witness["actions"])
        sfp, sscaf, _, srings = molinfo(src)
        tfp, tscaf, _, trings = molinfo(endpoint)
        comparisons = {}
        for arm in ("v0", "v1", "v21"):
            candidates = histories[(record["target"], src, arm)]
            if not candidates:
                comparisons[arm] = {"observations": 0, "status": "no_same_input_scored_history"}
                continue
            similarities = DataStructs.BulkTanimotoSimilarity(
                tfp, [molinfo(r["endpoint"])[0] for r in candidates]
            )
            best = max(
                range(len(candidates)), key=lambda i: (similarities[i], -candidates[i]["score"])
            )
            closest = candidates[best]
            comparisons[arm] = {
                "observations": len(candidates),
                "exact_teacher_endpoints": sum(r["endpoint"] == endpoint for r in candidates),
                "closest_endpoint_similarity": similarities[best],
                "closest_score": closest["score"],
                "closest_receipt": closest["receipt_id"],
                "closest_run": closest["run"],
                "closest_query": closest["query"],
                "closest_primitive_count": closest["primitive_count"],
                "closest_changed_atoms": closest["changed_slot_count"],
                "closest_net_created": closest["net_created"],
                "closest_net_deleted": closest["net_deleted"],
                "closest_block_labels": closest["block_labels"],
            }
        result_rows.append(
            {
                "probe_id": record["probe_id"],
                "cell": record["cell"],
                "source": src,
                "endpoint": endpoint,
                "known_labels": record["known_labels"],
                "primitives": len(witness["actions"]),
                **region_description(witness),
                "changed_original_atoms": len(change["changed_original_slots"]),
                "changed_source_span_bonds": source_span(source, change["changed_original_slots"]),
                "net_deleted": change["deleted_original_atoms"],
                "net_created": change["surviving_new_atoms"],
                "source_endpoint_similarity": DataStructs.TanimotoSimilarity(sfp, tfp),
                "ring_count_change": trings - srings,
                "same_scaffold": sscaf == tscaf,
                "historical_dynamic_same_input": comparisons,
            }
        )
    summary = {
        field: stats([r[field] for r in result_rows if r[field] is not None])
        for field in (
            "primitives",
            "dependency_regions",
            "changed_original_atoms",
            "changed_source_span_bonds",
            "net_deleted",
            "net_created",
            "ring_count_change",
        )
    }
    by_arm = {}
    for arm in ("v0", "v1", "v21"):
        present = [
            r["historical_dynamic_same_input"][arm]
            for r in result_rows
            if r["historical_dynamic_same_input"][arm]["observations"]
        ]
        by_arm[arm] = {
            "teachers_with_same_input_history": len(present),
            "teachers_with_exact_endpoint_in_history": sum(
                bool(r["exact_teacher_endpoints"]) for r in present
            ),
            "closest_endpoint_similarity": stats(
                [r["closest_endpoint_similarity"] for r in present]
            ),
        }
    result = {
        "schema": "t4_strategy_reset_route_comparison_v1",
        "new_oracle_calls": 0,
        "routes": result_rows,
        "summary": summary,
        "by_arm": by_arm,
        "same_scaffold_routes": sum(r["same_scaffold"] for r in result_rows),
        "inputs_sha256": INPUTS,
        "seconds": time.monotonic() - began,
        "limitations": [
            "Same-input comparison is conditional on an observed, scored proposal. It is not all attempted proposals or a likelihood estimate.",
            "Missing exact endpoint is not proof of zero autonomous support.",
            "External teacher labels remain reported public labels, not newly measured COMPOSE scores.",
            "Changed-source span is a graph-distance descriptor, not the radius of a single connected patch.",
            "Similarity is diagnostic only; no utility is inferred for unscored neighbors.",
        ],
    }
    (OUT / "route_comparison.json").write_text(
        json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n"
    )
    print(json.dumps({"routes": len(result_rows), "summary": summary, "by_arm": by_arm}))


if __name__ == "__main__":
    main()
