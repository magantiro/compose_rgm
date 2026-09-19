"""Run the bounded PARP1-0 production route-timeout support gate.

The prior known-answer diagnostic is read only to obtain the three expected
canonical endpoint keys.  Route generation itself receives only the fresh
source from the rescue contract and the task-independent shared checkpoint.
No oracle, docking adapter, live run, or teacher route is accessed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing
import os
import platform
import subprocess
import sys
from pathlib import Path
from time import perf_counter
from typing import Any

from rdkit import rdBase

from compose_v4.chem.molecular_graph import smiles_to_molecular_graph
from compose_v4.chem.state import pad_molecular_graph
from compose_v4.control.docking_value import identity
from compose_v4.control.route_distilled_goal_expert import (
    RouteDistilledGoalExpert,
    propose_route_expert_candidates,
)
from compose_v4.rewrite.kernel import canonical_state_key

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "t4_parp_route_timeout_support_gate_v1"
CONTRACT = ROOT / "configs/t4_shared_retained_fiber_parp1_p0_rescue_v1.json"
CHECKPOINT = ROOT / "diagnostics/t4_shared_retained_rewrite_v1/checkpoint.json"
KNOWN_ANSWER_AUDIT = (
    ROOT
    / "diagnostics/t4_shared_retained_fiber_parp1_v2/audit_20260919/"
    "parp1_0_route_support.json"
)
DEFAULT_OUTPUT_DIR = (
    ROOT / "diagnostics/t4_shared_retained_fiber_parp1_p0_rescue_v1/support_gate"
)

EXPECTED_CONFIGURATION = {
    "pool_size": 192,
    "realization_limit": 96,
    "beam_width": 48,
    "expansion_width": 48,
    "max_bindings_per_template": 4,
    "maximum_expansions": 4_000,
    "scale_balanced": True,
    "per_candidate_timeout_seconds": 10.0,
}

IMPLEMENTATION_INPUTS = (
    "src/compose_v4/control/route_distilled_goal_expert.py",
    "src/compose_v4/control/structural_subgoal_policy.py",
    "src/compose_v4/control/complete_region_program.py",
    "src/compose_v4/control/structural_subgoal_realizer.py",
    "tools/t4_parp_route_timeout_support_gate.py",
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_sealed(path: Path) -> dict[str, Any]:
    envelope = json.loads(path.read_text())
    payload = envelope.get("payload") if isinstance(envelope, dict) else None
    if not isinstance(payload, dict) or envelope.get("payload_sha256") != identity(
        payload
    ):
        raise ValueError(f"input is not a valid self-hashed envelope: {path}")
    return payload


def _revision() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def _matches_head(relative: str) -> bool:
    tracked = subprocess.run(
        ["git", "ls-files", "--error-unmatch", relative],
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    unchanged = subprocess.run(
        ["git", "diff", "--quiet", "HEAD", "--", relative],
        cwd=ROOT,
        check=False,
    )
    return tracked.returncode == 0 and unchanged.returncode == 0


def _publish_json(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    if path.exists():
        raise ValueError(f"refusing to overwrite support-gate artifact: {path}")
    envelope = {"payload": payload, "payload_sha256": identity(payload)}
    encoded = json.dumps(envelope, sort_keys=True, separators=(",", ":")) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(encoded)
    temporary.replace(path)
    return envelope


def _publish_report(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise ValueError(f"refusing to overwrite support-gate report: {path}")
    statuses = payload["result"]["realization_status_counts"]
    bands = payload["result"]["realized_primitive_band_counts"]
    gate = payload["gate"]
    lines = [
        "# PARP1-0 route-timeout support gate",
        "",
        f"**Decision: {'PASS' if gate['passed'] else 'FAIL'}**",
        "",
        (
            "The actual production route expert returned after independently bounding "
            "each candidate realization at 10 seconds. This was a local zero-oracle "
            "run. It made no docking call, launched no Modal job, and accessed no live "
            "campaign."
        ),
        "",
        "## Measured result",
        "",
        f"- Elapsed wall time: {payload['result']['elapsed_seconds']:.6f} seconds",
        f"- Returned records: {payload['result']['returned_record_count']}",
        f"- Realization statuses: `{json.dumps(statuses, sort_keys=True)}`",
        f"- Realized primitive bands: `{json.dumps(bands, sort_keys=True)}`",
        (
            "- Exact realization precision: "
            f"{payload['result']['exact_realization_precision_numerator']}/"
            f"{payload['result']['exact_realization_precision_denominator']}"
        ),
        (
            "- Expected endpoint recovery: "
            f"{payload['result']['expected_endpoint_recovery_count']}/"
            f"{payload['result']['expected_endpoint_count']}"
        ),
        "",
        "## Gate checks",
        "",
    ]
    lines.extend(
        f"- {'PASS' if value else 'FAIL'}: `{key}`"
        for key, value in gate.items()
        if key != "passed"
    )
    lines.extend(
        [
            "",
            "## Claim boundary",
            "",
            (
                "This gate establishes production-path runtime support and exact recovery "
                "of three previously identified structural endpoints. It does not measure "
                "docking utility, autonomous novelty, or prospective optimization "
                "performance."
            ),
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("\n".join(lines))
    temporary.replace(path)


def _expected_endpoint_keys(audit: dict[str, Any]) -> list[str]:
    if (
        audit.get("schema_version")
        != "t4_parp_route_known_answer_support_probe_v1"
        or audit.get("cell") != "parp1_0"
        or audit.get("evidence_status")
        != "computed_zero_oracle_known_answer_diagnostic"
    ):
        raise ValueError("known-answer audit identity changed")
    rows = audit.get("teacher_routes")
    if not isinstance(rows, list) or len(rows) != 3:
        raise ValueError("PARP1-0 expected-endpoint census changed")
    keys = sorted(str(row["target_endpoint_key"]) for row in rows)
    if len(set(keys)) != 3 or any(not key for key in keys):
        raise ValueError("PARP1-0 expected endpoint keys are invalid")
    return keys


def run(output_dir: Path) -> dict[str, Any]:
    contract = _load_sealed(CONTRACT)
    checkpoint = _load_sealed(CHECKPOINT)
    known_answer_audit = _load_sealed(KNOWN_ANSWER_AUDIT)

    if contract.get("schema_version") != (
        "t4_shared_retained_fiber_parp1_p0_rescue_contract_v1"
    ):
        raise ValueError("PARP1-0 rescue contract schema changed")
    cells = contract.get("cells")
    if (
        not isinstance(cells, list)
        or len(cells) != 1
        or cells[0].get("cell") != "parp1_0"
    ):
        raise ValueError("PARP1-0 rescue source census changed")
    route = contract["proposal"]["route_complete_region"]
    configuration = {key: route[key] for key in EXPECTED_CONFIGURATION}
    if configuration != EXPECTED_CONFIGURATION:
        raise ValueError("production route settings disagree with the rescue contract")
    checkpoint_relative = str(CHECKPOINT.relative_to(ROOT))
    checkpoint_sha256 = _sha256_file(CHECKPOINT)
    if contract["runtime_inputs_sha256"].get(checkpoint_relative) != checkpoint_sha256:
        raise ValueError("shared runtime checkpoint disagrees with the rescue contract")
    route_code_relative = "src/compose_v4/control/route_distilled_goal_expert.py"
    if contract["runtime_inputs_sha256"].get(route_code_relative) != _sha256_file(
        ROOT / route_code_relative
    ):
        raise ValueError("production route expert disagrees with the rescue contract")

    material_relative = (
        str(CONTRACT.relative_to(ROOT)),
        str(CHECKPOINT.relative_to(ROOT)),
        str(KNOWN_ANSWER_AUDIT.relative_to(ROOT)),
        route_code_relative,
    )
    material_inputs_match_code_revision = all(
        _matches_head(path) for path in material_relative
    )
    if not material_inputs_match_code_revision:
        raise ValueError("a material runtime input differs from the committed revision")

    expected_keys = _expected_endpoint_keys(known_answer_audit)
    expert = RouteDistilledGoalExpert.from_checkpoint(checkpoint["expert"])
    source = pad_molecular_graph(smiles_to_molecular_graph(cells[0]["smiles"]), 48)

    started = perf_counter()
    records, telemetry = propose_route_expert_candidates(
        source,
        expert,
        pool_size=configuration["pool_size"],
        realization_limit=configuration["realization_limit"],
        beam_width=configuration["beam_width"],
        expansion_width=configuration["expansion_width"],
        max_bindings_per_template=configuration["max_bindings_per_template"],
        maximum_expansions=configuration["maximum_expansions"],
        scale_balanced=configuration["scale_balanced"],
        per_candidate_timeout_seconds=configuration[
            "per_candidate_timeout_seconds"
        ],
    )
    elapsed_seconds = perf_counter() - started

    returned_keys = [
        canonical_state_key(
            pad_molecular_graph(smiles_to_molecular_graph(row["smiles"]), 48)
        )
        for row in records
    ]
    returned_key_set = set(returned_keys)
    recovered_expected_keys = sorted(set(expected_keys) & returned_key_set)
    missing_expected_keys = sorted(set(expected_keys) - returned_key_set)
    statuses = {
        str(key): int(value)
        for key, value in telemetry["realization_status_counts"].items()
    }
    bands = {
        str(key): int(value)
        for key, value in telemetry["realized_primitive_band_counts"].items()
    }
    timeout_count = statuses.get("realizer_candidate_timeout", 0)
    attempted_realizations = sum(statuses.values())
    precision_numerator = int(telemetry["exact_realization_precision_numerator"])
    precision_denominator = int(telemetry["exact_realization_precision_denominator"])

    material_inputs_sha256 = {
        path: _sha256_file(ROOT / path) for path in material_relative
    }
    implementation_inputs_sha256 = {
        path: _sha256_file(ROOT / path) for path in IMPLEMENTATION_INPUTS
    }
    gate = {
        "material_inputs_match_code_revision": material_inputs_match_code_revision,
        "exact_rescue_configuration_used": configuration == EXPECTED_CONFIGURATION,
        "full_realization_prefix_accounted": attempted_realizations
        == configuration["realization_limit"],
        "candidate_timeout_path_exercised": timeout_count > 0,
        "returned_after_candidate_timeouts": timeout_count > 0 and bool(records),
        "exact_realization_precision_one": precision_denominator > 0
        and precision_numerator == precision_denominator == len(records),
        "small_medium_large_programs_committed": all(
            bands.get(band, 0) > 0 for band in ("small", "medium", "large")
        ),
        "all_three_expected_endpoint_keys_recovered": len(expected_keys) == 3
        and not missing_expected_keys,
        "zero_oracle_execution": True,
    }
    gate["passed"] = all(gate.values())

    payload = {
        "schema_version": SCHEMA,
        "evidence_status": "computed_zero_oracle_production_runtime_gate",
        "scientific_problem": (
            "verify that independently time-bounded route realization prevents one "
            "pathological candidate from blocking the PARP1-0 production pool"
        ),
        "primary_model_output": (
            "complete exactly realized route-expert candidate records from one fresh "
            "PARP1-0 source graph"
        ),
        "central_claim_under_test": (
            "the committed per-candidate timeout lets production proposal continue past "
            "slow realizations without losing exact small, medium, large, or known "
            "teacher-scale support"
        ),
        "experimental_setting": (
            "local single-CPU zero-oracle call to the production route proposer using "
            "the frozen PARP1-0 rescue configuration"
        ),
        "primary_baseline": (
            "the prior known-answer diagnostic that stopped at production rank 34 under "
            "a monolithic realization horizon"
        ),
        "declared_support": (
            "task-independent shared retained-subgraph checkpoint, 48 persistent slots, "
            "at most 32 realized primitives per complete supported program"
        ),
        "claim_boundary": (
            "known answers supply only three expected canonical endpoint keys; this gate "
            "measures runtime support and exact execution, not docking utility or "
            "autonomous discovery"
        ),
        "code_revision": _revision(),
        "material_inputs_sha256": material_inputs_sha256,
        "implementation_inputs_sha256": implementation_inputs_sha256,
        "contract_payload_sha256": identity(contract),
        "checkpoint_payload_sha256": identity(checkpoint),
        "known_answer_audit_payload_sha256": identity(known_answer_audit),
        "configuration": configuration,
        "determinism": {
            "proposal_rng": "none",
            "proposal_order": "deterministic score and canonical-key ordering",
            "serialization": "sorted-key compact JSON with SHA-256 payload identity",
            "elapsed_seconds_is_operational_telemetry": True,
        },
        "environment": {
            "python": platform.python_version(),
            "python_implementation": platform.python_implementation(),
            "rdkit": rdBase.rdkitVersion,
            "platform_system": platform.system(),
            "platform_release": platform.release(),
            "machine": platform.machine(),
            "cpu_count": os.cpu_count(),
            "precision": "native CPU graph execution; no learned numeric inference",
            "multiprocessing_start_methods": sorted(
                multiprocessing.get_all_start_methods()
            ),
        },
        "result": {
            "elapsed_seconds": elapsed_seconds,
            "returned_record_count": len(records),
            "unique_returned_endpoint_count": len(returned_key_set),
            "attempted_realizations": attempted_realizations,
            "realization_status_counts": dict(sorted(statuses.items())),
            "candidate_timeout_count": timeout_count,
            "other_noncommitted_status_counts": {
                key: value
                for key, value in sorted(statuses.items())
                if key not in {"committed", "realizer_candidate_timeout"}
            },
            "exact_realization_precision_numerator": precision_numerator,
            "exact_realization_precision_denominator": precision_denominator,
            "exact_realization_precision": (
                precision_numerator / precision_denominator
                if precision_denominator
                else None
            ),
            "realized_primitive_band_counts": dict(sorted(bands.items())),
            "expected_endpoint_count": len(expected_keys),
            "expected_endpoint_keys": expected_keys,
            "recovered_expected_endpoint_keys": recovered_expected_keys,
            "missing_expected_endpoint_keys": missing_expected_keys,
            "expected_endpoint_recovery_count": len(recovered_expected_keys),
        },
        "gate": gate,
        "costs": {
            "oracle_calls": 0,
            "docking_calls": 0,
            "modal_launches": 0,
            "gpu_seconds": 0,
        },
    }

    output_json = output_dir / "result.json"
    output_md = output_dir / "REPORT.md"
    envelope = _publish_json(output_json, payload)
    _publish_report(output_md, payload)
    return envelope


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    envelope = run(args.output_dir.resolve())
    payload = envelope["payload"]
    print(
        json.dumps(
            {
                "payload_sha256": envelope["payload_sha256"],
                "gate": payload["gate"],
                "elapsed_seconds": payload["result"]["elapsed_seconds"],
                "returned_record_count": payload["result"][
                    "returned_record_count"
                ],
                "realization_status_counts": payload["result"][
                    "realization_status_counts"
                ],
                "realized_primitive_band_counts": payload["result"][
                    "realized_primitive_band_counts"
                ],
                "expected_endpoint_recovery_count": payload["result"][
                    "expected_endpoint_recovery_count"
                ],
            },
            sort_keys=True,
        )
    )
    if not payload["gate"]["passed"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
