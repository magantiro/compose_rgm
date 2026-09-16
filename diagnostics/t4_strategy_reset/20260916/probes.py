"""Answer-known probes through existing Dynamic interfaces; zero new scoring."""

from __future__ import annotations

import json
import time
from collections import Counter
from dataclasses import asdict, replace

import numpy as np
from audit import CODE, INPUTS, OUT, descriptor, identity, read
from compose_v4.control.dependency_region_program import trace_structure

from compose_v4.control import dynamic_program_synthesis_v1 as v1
from compose_v4.control.adaptive_program_optimizer import ProgramSearchConfig
from compose_v4.control.dynamic_program_synthesis import DynamicProgramOptimizer
from compose_v4.experiments.t4_frozen_program_benchmark import strict_endpoint_scorer
from compose_v4.rewrite.kernel import canonical_state_key
from compose_v4.rewrite.trace_shard import decode_state, encode_state
from tools.t4_dynamic_v1 import _jak2_requests


def region_description(trace):
    """Same lifetime-connectivity definition as the sealed region analysis, no replay."""
    structure = trace_structure(tuple(trace["states"]), tuple(trace["actions"]))
    from compose_v4.control.dependency_region_program import _connected_tokens, _UnionFind

    footprints = structure["footprints"]
    union = _UnionFind(len(footprints))
    for a in range(len(footprints)):
        for b in range(a + 1, len(footprints)):
            if _connected_tokens(
                footprints[a],
                footprints[b],
                structure["lifetime_bonds"],
                join_lifetime_neighbors=False,
            ):
                union.union(a, b)
    for edge in structure["dependency_edges"]:
        union.union(edge["producer"], edge["consumer"])
    groups = Counter(union.find(i) for i in range(len(footprints)))
    return {
        "dependency_regions": len(groups),
        "region_primitives": sorted(groups.values()),
        "created_handles": len(structure["created_handles"]),
        "reused_created_handles": sum(bool(h["consumers"]) for h in structure["created_handles"]),
        "dependency_edges": len(structure["dependency_edges"]),
    }


def panel_probability(candidates, endpoint):
    n = len(candidates)
    hits = [i for i, c in enumerate(candidates) if c.stage["endpoint"] == endpoint]
    if not n:
        return {"candidates": 0, "target_ranks": [], "probability": 0.0}
    weights = np.exp(-np.arange(n) / max(1.0, n / 4))
    weights = 0.2 / n + 0.8 * weights / weights.sum()
    return {
        "candidates": n,
        "target_ranks": [i + 1 for i in hits],
        "probability": float(sum(weights[i] for i in hits)),
    }


def main():
    began = time.monotonic()
    output = {
        "schema": "t4_strategy_reset_known_answer_probes_v1",
        "new_oracle_calls": 0,
        "injected_for_diagnostics_only": True,
    }
    jac = read(OUT / "raw/full146_jak2_1_r0_result.json")
    source, requests = _jak2_requests(jac["champion"]["candidate"])
    current = source
    preferred = frozenset()
    stages = []
    for index, request in enumerate(requests):
        if index == 0:
            product, stage = v1.compile_pendant_delete(current, request)
            panel = v1.enumerate_pendant_deletions(
                current, preferred_anchors=preferred, max_candidates=16
            )
            evidence = panel_probability(panel, stage["endpoint"])
        elif index == 1:
            product, stage = v1.compile_substituted_ring(current, request)
            panels = []
            for panel_id in range(4):
                key = (
                    "construct_substituted_ring",
                    identity(encode_state(current)),
                    tuple(sorted(preferred)),
                    panel_id,
                )
                rng = np.random.default_rng(int(identity(key)[:16], 16))
                panel = v1.enumerate_substituted_rings(
                    current, rng, preferred_anchors=preferred, max_candidates=16, max_trials=32
                )
                panels.append({"panel_id": panel_id, **panel_probability(panel, stage["endpoint"])})
            evidence = {"panels": panels, "probability": sum(p["probability"] for p in panels) / 4}
        else:
            product, stage = v1.compile_ring_path_remodel(current, request)
            panel = v1.enumerate_ring_path_remodels(
                current, preferred_slots=preferred, max_candidates=16
            )
            evidence = panel_probability(panel, stage["endpoint"])
        stages.append(
            {
                "module": type(request).__name__,
                "request": asdict(request),
                "forced_endpoint": stage["endpoint"],
                "actual_parameter_panel": evidence,
            }
        )
        current = product
        preferred = v1._following_context(current, stage)
    output["jak2_actual_v1_panels"] = {
        "source": canonical_state_key(source),
        "stages": stages,
        "forced_exact_final": canonical_state_key(current) == jac["champion"]["endpoint"],
        "scope": "Conditional on this exact known-answer prefix/context, not an impossibility proof for every alternative route.",
    }

    # Test actual archive boundary and unchanged v0 proposal code with genuine old scores.
    archive_probes = []
    for name in (
        "full146_jak2_1_r0",
        "v0_braf_1_r0",
        "v1_5ht1b_0_r0",
        "full146_parp1_0_r0",
        "full146_fa7_0_r0",
    ):
        result = read(OUT / "raw" / (name + "_result.json"))
        candidate = result["champion"]["candidate"]
        conf = read(OUT / "raw/v0_jak2_1_r0_checkpoint.json.gz")["search"]["configuration"]
        conf["channel_probabilities"] = tuple(conf["channel_probabilities"])
        config = replace(
            ProgramSearchConfig(**conf),
            attempts_per_batch=16,
            candidates_per_batch=4,
            wall_seconds=10.0,
            seed=2026091601,
        )
        optimizer = DynamicProgramOptimizer(
            config,
            source_group=candidate["source_group"],
            oracle_protocol=candidate["oracle_protocol"],
        )
        row = {
            "run": name,
            "old_score": result["champion"]["score"],
            "seed": config.seed,
            "configuration": asdict(config),
        }
        try:
            optimizer.add_measured_program(
                candidate,
                receipt_id=result["champion"]["receipt_id"],
                score=result["champion"]["score"],
            )
            row["archive_exact_admission"] = True
            batch = optimizer.propose_batch(
                strict_endpoint_scorer(result["unit"]["original_seed"], delta=0.4)
            )
            row.update(
                attempts=len(batch["attempts"]),
                status_counts=dict(Counter(a["status"] for a in batch["attempts"])),
                novel_eligible=len(batch["candidates"]),
                proposals=[
                    {
                        "endpoint": c["endpoint"],
                        "program": c["program"],
                        "assignment": c["assignment"],
                        "source_state": c["source_state"],
                        "trace": c["trace"],
                        "provenance": c["provenance"],
                    }
                    for c in batch["candidates"]
                ],
                seconds=batch["proposal_seconds"],
            )
        except ValueError as exc:
            row.update(status="runtime_value_error", error=str(exc))
        archive_probes.append(row)
        print(
            json.dumps({k: v for k, v in row.items() if k not in ("proposals", "configuration")}),
            flush=True,
        )
    output["archive_injection_probes"] = archive_probes
    (OUT / "probes_partial.json").write_text(json.dumps(output, sort_keys=True, indent=2) + "\n")

    # Describe ALL 77 bound witness routes, without new compiler work or scoring.
    old = read(CODE / "diagnostics/t4_known_good_transformation_forensics/attempt_1/result.json")
    route_rows = []
    for record in old["routes"]:
        witness = read(CODE / record["teacher_receipt"])["path"]
        d = region_description(witness)
        if d["dependency_regions"] != record["dependency_regions"]:
            raise ValueError("Region measurement disagrees with frozen audit")
        trace = witness
        initial = decode_state(trace["states"][0])
        seed = canonical_state_key(initial)
        evaluate = strict_endpoint_scorer(seed, delta=0.4)
        gates = [
            evaluate({"smiles": canonical_state_key(decode_state(s))}) for s in trace["states"]
        ]
        route_rows.append(
            {
                "id": record["probe_id"],
                "cell": record["cell"],
                "receipt": record["teacher_receipt"],
                "primitives": len(trace["actions"]),
                "ineligible_internal_prefixes": sum(not g["oracle_eligible"] for g in gates[1:-1]),
                "endpoint_eligible_under_current_rdkit": gates[-1]["oracle_eligible"],
                "v0_exact_autonomous_support": "not_exhaustively_decided",
                "corrected_region_runtime_supported": True,
                **d,
            }
        )
    output["teacher_route_descriptors"] = route_rows
    # Main actual champions and call-one evidence, recovered rather than fabricated.
    exemplars = []
    for name in (
        "full146_jak2_1_r0",
        "v0_jak2_1_r0",
        "v0_braf_1_r0",
        "v1_5ht1b_0_r0",
        "v21_jak2_1_r0",
    ):
        result = read(OUT / "raw" / (name + "_result.json"))
        batch = read(OUT / "raw" / (name + "_rounds_round_0000_batch.json.gz"))["batch"]
        for kind, candidate in (
            ("first_candidate", batch["candidates"][0]),
            ("champion", result["champion"]["candidate"]),
        ):
            desc = descriptor(candidate)
            evaluate = strict_endpoint_scorer(result["unit"]["original_seed"], delta=0.4)
            gates = [
                evaluate({"smiles": canonical_state_key(decode_state(s))})
                for s in candidate["trace"]["states"]
            ]
            exemplars.append(
                {
                    "run": name,
                    "role": kind,
                    "endpoint": candidate["endpoint"],
                    "score": result["curve"][0]["best_score"]
                    if kind == "first_candidate"
                    else result["champion"]["score"],
                    "ineligible_internal_prefixes": sum(
                        not g["oracle_eligible"] for g in gates[1:-1]
                    ),
                    **desc,
                    **region_description(candidate["trace"]),
                }
            )
    output["historical_examples"] = exemplars
    output["inputs_sha256"] = INPUTS
    output["seconds"] = time.monotonic() - began
    output["limitations"] = [
        "Runtime probes used the local RDKit version reported by audit.json, not a replacement of the pinned compiler gate.",
        "No descendant utility inferred from validity or eligibility.",
        "Known-answer injection is diagnostic, never autonomous recovery.",
        "Supplied witness length is not a shortest-path proof; unresolved endpoint support remains unknown.",
    ]
    (OUT / "probes.json").write_text(json.dumps(output, sort_keys=True, indent=2) + "\n")
    print(json.dumps({"teacher_routes": len(route_rows), "seconds": output["seconds"]}), flush=True)


if __name__ == "__main__":
    main()
