"""Focused integrity verification of the retrospective audit, not a release gate."""

import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

from audit import CODE, OUT, read


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    reports = {
        name: read(OUT / name)
        for name in ("audit.json", "probes.json", "route_comparison.json", "search_audit.json")
    }
    inputs = {}
    for report in reports.values():
        assert report["new_oracle_calls"] == 0
        for path, expected in report["inputs_sha256"].items():
            actual = digest(Path(path))
            if actual != expected:
                raise ValueError(f"Material input changed: {path}")
            inputs[path] = actual
    rows = [json.loads(line) for line in (OUT / "scored_rows.jsonl").read_text().splitlines()]
    assert (
        len(rows)
        == 35895
        == sum(v["observations"] for v in reports["audit.json"]["by_arm"].values())
    )
    assert len({r["receipt_id"] for r in rows}) == len(rows)
    assert sum(r["query"] is not None for r in rows) == 14085
    assert sum(r["parent_score"] is not None for r in rows) == 35328
    assert all(r["query_verified"] or r["query"] is None for r in rows)
    assert len(reports["audit.json"]["units"]) == 71
    probes = reports["probes.json"]
    assert len(probes["teacher_route_descriptors"]) == 77
    assert sum(r["dependency_regions"] for r in probes["teacher_route_descriptors"]) == 147
    assert all(r["archive_exact_admission"] for r in probes["archive_injection_probes"])
    assert probes["jak2_actual_v1_panels"]["forced_exact_final"]
    assert (
        probes["jak2_actual_v1_panels"]["stages"][1]["actual_parameter_panel"]["probability"] == 0
    )
    assert len(reports["route_comparison.json"]["routes"]) == 77

    # Verify existing complete-region evidence, without rerunning the compiler.
    compiler_path = CODE / "diagnostics/t4_complete_region_runtime/attempt_1/result.json"
    compiler = read(compiler_path)
    inputs[str(compiler_path)] = digest(compiler_path)
    assert compiler["census"] == {"regions": 147, "routes": 77, "status_counts": {"committed": 77}}
    for route in compiler["route_results"]:
        path = CODE / route["path"]
        if digest(path) != route["sha256"]:
            raise ValueError(f"Compiler receipt changed: {path}")
        read(path)  # Checks the self-hashed payload as well.
        inputs[str(path)] = digest(path)
        assert route["exact_endpoint_reconstruction"]
        assert route["primitive_teacher_actions_used"] == 0
    code_paths = [
        CODE / "src/compose_v4/control" / name
        for name in (
            "adaptive_program_optimizer.py",
            "dynamic_program_synthesis.py",
            "dynamic_program_synthesis_v1.py",
            "dynamic_program_synthesis_v21.py",
            "dependency_region_program.py",
            "edit_program_graph.py",
        )
    ]
    code_paths.append(CODE / "src/compose_v4/experiments/t4_frozen_program_benchmark.py")
    report_hashes = {name: digest(OUT / name) for name in reports}
    report_hashes["scored_rows.jsonl"] = digest(OUT / "scored_rows.jsonl")
    result = {
        "schema": "t4_strategy_reset_verification_v1",
        "status": "passed",
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        "hardware": platform.platform(),
        "precision": "CPU float64 descriptor statistics",
        "new_oracle_calls": 0,
        "new_model_fits": 0,
        "input_sha256": inputs,
        "output_sha256": report_hashes,
        "analysis_code_sha256": {p.name: digest(p) for p in sorted(OUT.glob("*.py"))},
        "runtime_code_sha256": {str(p): digest(p) for p in code_paths},
        "recovered_observations": len(rows),
        "ambiguous_endpoint_to_entry_rows": sum(r["entry_alternatives"] != 1 for r in rows),
        "existing_compiler_receipts_verified": 77,
        "scope": "Audit integrity only. No new scientific launch, compiler gate, benchmark or controller release.",
    }
    (OUT / "verification.json").write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")
    print(
        json.dumps(
            {
                k: result[k]
                for k in (
                    "status",
                    "recovered_observations",
                    "ambiguous_endpoint_to_entry_rows",
                    "existing_compiler_receipts_verified",
                    "new_oracle_calls",
                )
            }
        )
    )


if __name__ == "__main__":
    main()
