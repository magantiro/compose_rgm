"""Export locked PMO routes as generic, zero-oracle primitive supervision."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import platform
import shutil
import subprocess
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from rdkit import rdBase

from compose_v4.control.docking_value import identity
from compose_v4.control.option_demonstrations import descriptor_menu, recognize_trace
from compose_v4.control.pmo_action_roles import (
    ACTION_ROLE_SCHEMA,
    action_role_supervision,
)
from compose_v4.control.route_distilled_program_policy import SCHEMA as POLICY_SCHEMA
from compose_v4.control.route_distilled_program_policy import (
    STAGE_DESCRIPTOR_NAMES,
    stage_descriptor,
)
from compose_v4.control.trajectory_value import molecule_features
from compose_v4.rewrite.kernel import canonical_state_key
from compose_v4.rewrite.trace_shard import decode_state

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = Path("configs/pmo_route_distillation_export_v1.json")
DATASET_SCHEMA = "pmo_route_distillation_training_dataset_v1"
RESULT_SCHEMA = "pmo_route_distillation_export_result_v1"

FORBIDDEN_GENERIC_KEYS = {
    "assignment",
    "endpoint",
    "program",
    "route_id",
    "slot",
    "smiles",
    "source_atom",
    "source_graph",
    "target",
    "task",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for key, child in value.items():
            yield str(key)
            yield from _strings(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            yield from _strings(child)


def _keys(value):
    if isinstance(value, dict):
        for key, child in value.items():
            yield str(key)
            yield from _keys(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            yield from _keys(child)


def load_contract(root: Path = ROOT) -> dict:
    path = root / CONTRACT
    envelope = json.loads(path.read_text())
    payload = envelope.get("payload")
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != "pmo_route_distillation_export_contract_v1"
        or envelope.get("contract_sha256") != identity(payload)
    ):
        raise ValueError(f"PMO route-distillation contract is invalid: {path}")
    expected = payload.get("expected_counts", {})
    if expected != {
        "curriculum_programs": 181,
        "witness_routes": 5,
        "route_instances": 186,
        "primitive_transitions_with_multiplicity": 6212,
        "unique_endpoints_in_curricula": 178,
        "unique_exact_traces": 184,
        "deduplicated_decisions": 6143,
        "lineages": 18,
        "runtime_supported_route_instances": 107,
        "long_route_local_only_instances": 79,
        "runtime_supported_unique_traces": 106,
        "long_route_local_only_unique_traces": 78,
        "recognized_compound_stages": 0,
    }:
        raise ValueError("PMO route-distillation expected census changed")
    if payload.get("outputs", {}).get("runtime_checkpoint") is not None:
        raise ValueError("this revision may not produce a runtime checkpoint")
    return payload


def _verified_json(root: Path, relative: str, expected_sha256: str) -> dict:
    path = root / relative
    observed = sha256_file(path)
    if observed != expected_sha256:
        raise ValueError(
            f"source hash mismatch for {relative}: expected {expected_sha256}, got {observed}"
        )
    return json.loads(path.read_text())


def _verified_envelope(root: Path, relative: str, expected_sha256: str) -> dict:
    envelope = _verified_json(root, relative, expected_sha256)
    if not isinstance(envelope.get("payload"), dict) or envelope.get("payload_sha256") != identity(
        envelope["payload"]
    ):
        raise ValueError(f"source envelope self-hash mismatch: {relative}")
    return envelope["payload"]


@dataclass(frozen=True)
class RouteInstance:
    collection: str
    task: str
    task_family: str
    member_id: str
    source_path: str
    states: tuple[dict, ...]
    actions: tuple[dict, ...]
    endpoint: str
    base_steps: int
    lineage_identity: str
    evidence_role: str

    @property
    def trace_identity(self) -> str:
        return identity({"states": self.states, "actions": self.actions})

    @property
    def instance_identity(self) -> str:
        return identity(
            {
                "collection": self.collection,
                "task": self.task,
                "member_id": self.member_id,
                "source_path": self.source_path,
            }
        )


def _program_member_id(program: dict, index: int) -> str:
    for key in ("candidate_id", "variant_id"):
        if isinstance(program.get(key), str) and program[key]:
            return program[key]
    return f"program_{index:04d}"


def collect_routes(contract: dict, root: Path = ROOT) -> tuple[list[RouteInstance], dict]:
    routes: list[RouteInstance] = []
    source_checks = {}
    endpoint_identities = set()
    for source in contract["curricula"]:
        curriculum = _verified_envelope(root, source["path"], source["sha256"])
        result = _verified_envelope(root, source["result_path"], source["result_sha256"])
        if result.get("curriculum_sha256") != source["sha256"]:
            raise ValueError(f"result does not bind its curriculum bytes: {source['result_path']}")
        source_checks[source["path"]] = {
            "sha256": source["sha256"],
            "payload_sha256_verified": True,
            "result_path": source["result_path"],
            "result_sha256": source["result_sha256"],
        }
        count = 0
        if source["collection"] == "winner_program_curriculum":
            task_items = (("perindopril_mpo", curriculum["programs"], None),)
        else:
            task_items = tuple(
                (task, row["programs"], row["base_route"])
                for task, row in sorted(curriculum["tasks"].items())
            )
        for task, programs, base_route in task_items:
            if task not in contract["task_families"]:
                raise ValueError(f"task has no frozen family: {task}")
            for index, program in enumerate(programs):
                receipt = program.get("receipt")
                if not isinstance(receipt, dict) or receipt.get("complete") is not True:
                    raise ValueError(f"incomplete curriculum receipt: {task}/{index}")
                actions = tuple(receipt.get("actions", ()))
                states = tuple(receipt.get("states", ()))
                if base_route is None:
                    blocks = program.get("program", {}).get("blocks", ())
                    if not blocks or blocks[0].get("label") != "public_endpoint_recovery":
                        raise ValueError("Perindopril program lost its common recovery prefix")
                    base_steps = int(blocks[0]["stop"])
                    base_marks = program["program"]["marks"][:base_steps]
                    lineage_identity = identity(
                        {
                            "input_atoms": program["program"]["input_atoms"],
                            "input_bonds": program["program"]["input_bonds"],
                            "base_marks": base_marks,
                        }
                    )
                else:
                    base_steps = len(base_route["actions"])
                    lineage_identity = identity(
                        {
                            "source_state": base_route["states"][0],
                            "base_actions": base_route["actions"],
                        }
                    )
                endpoint = str(receipt["endpoint"])
                endpoint_identities.add(identity(endpoint))
                routes.append(
                    RouteInstance(
                        collection=source["collection"],
                        task=task,
                        task_family=contract["task_families"][task],
                        member_id=_program_member_id(program, index),
                        source_path=source["path"],
                        states=states,
                        actions=actions,
                        endpoint=endpoint,
                        base_steps=base_steps,
                        lineage_identity=lineage_identity,
                        evidence_role="answer_known_complete_program",
                    )
                )
                count += 1
        if count != source["programs"]:
            raise ValueError(f"program census mismatch for {source['collection']}: {count}")

    witness = contract["witnesses"]
    audit = _verified_json(root, witness["audit_path"], witness["audit_sha256"])
    result = _verified_json(root, witness["result_path"], witness["result_sha256"])
    if audit.get("route_count") != 5 or result.get("source_count") != 5:
        raise ValueError("Perindopril witness census changed")
    result_paths = {row["path"]: row["sha256"] for row in result["paths"]}
    if result_paths != witness["paths"]:
        raise ValueError("Perindopril result path manifest changed")
    for relative, digest in sorted(witness["paths"].items()):
        payload = _verified_json(root, relative, digest)
        route = payload.get("result", {})
        if route.get("status") != "witness_found":
            raise ValueError(f"saved witness is not complete: {relative}")
        actions = tuple(route["actions"])
        states = tuple(route["states"])
        routes.append(
            RouteInstance(
                collection="perindopril_public_witness",
                task=witness["task"],
                task_family=contract["task_families"][witness["task"]],
                member_id=str(payload["source_id"]),
                source_path=relative,
                states=states,
                actions=actions,
                endpoint=canonical_state_key(decode_state(states[-1])),
                base_steps=len(actions),
                lineage_identity=identity({"source_state": states[0], "base_actions": actions}),
                evidence_role="answer_known_exact_witness",
            )
        )
    source_checks[witness["audit_path"]] = {
        "sha256": witness["audit_sha256"],
        "route_count": audit["route_count"],
        "new_oracle_calls": audit["new_oracle_calls"],
    }
    for relative, row in sorted(contract["excluded"].items()):
        observed = sha256_file(root / relative)
        if observed != row["sha256"]:
            raise ValueError(f"excluded input changed: {relative}")
    return routes, {
        "source_checks": source_checks,
        "unique_curriculum_endpoint_identities": len(endpoint_identities),
        "excluded": contract["excluded"],
    }


def _validate_and_recognize(route: RouteInstance) -> tuple:
    if len(route.states) != len(route.actions) + 1 or not route.actions:
        raise ValueError(f"noncontiguous trace: {route.instance_identity}")
    graphs = [decode_state(state) for state in route.states]
    if any(graph.n_atoms != 48 or not 1 <= graph.n_real_atoms <= 40 for graph in graphs):
        raise ValueError(f"trace exceeds frozen support: {route.instance_identity}")
    if canonical_state_key(graphs[-1]) != route.endpoint:
        raise ValueError(f"trace endpoint mismatch: {route.instance_identity}")
    segments = recognize_trace(list(route.states), list(route.actions))
    if any(segment.start + 1 != segment.stop or segment.compound for segment in segments):
        raise ValueError("compound PMO stages require a separately authorized schema revision")
    if len(segments) != len(route.actions):
        raise RuntimeError("primitive recognizer lost an action")
    return graphs, segments


def _deduplicated_routes(routes: list[RouteInstance]):
    grouped = defaultdict(list)
    for route in routes:
        grouped[route.trace_identity].append(route)
    result = []
    for trace_identity, members in sorted(grouped.items()):
        members.sort(key=lambda row: row.instance_identity)
        if (
            len({row.task_family for row in members}) != 1
            or len({row.lineage_identity for row in members}) != 1
        ):
            raise ValueError("an exact duplicate trace crosses frozen balance groups")
        result.append((trace_identity, members[0], tuple(members)))
    return result


def build_dataset(
    contract: dict, routes: list[RouteInstance], source_audit: dict
) -> tuple[dict, dict]:
    expected = contract["expected_counts"]
    curriculum_count = sum(
        route.evidence_role == "answer_known_complete_program" for route in routes
    )
    witness_count = len(routes) - curriculum_count
    transition_count = sum(len(route.actions) for route in routes)
    if (
        curriculum_count != expected["curriculum_programs"]
        or witness_count != expected["witness_routes"]
        or len(routes) != expected["route_instances"]
        or transition_count != expected["primitive_transitions_with_multiplicity"]
        or source_audit["unique_curriculum_endpoint_identities"]
        != expected["unique_endpoints_in_curricula"]
    ):
        raise ValueError("admitted PMO route census differs from the frozen contract")
    unique_routes = _deduplicated_routes(routes)
    lineage_decisions = Counter()
    lineage_family = {}
    recognized = {}
    for trace_identity, route, _ in unique_routes:
        graphs, segments = _validate_and_recognize(route)
        recognized[trace_identity] = (graphs, segments)
        lineage_decisions[route.lineage_identity] += len(segments)
        previous = lineage_family.setdefault(route.lineage_identity, route.task_family)
        if previous != route.task_family:
            raise ValueError("one lineage crosses task families")
    families = sorted(set(lineage_family.values()))
    family_lineages = {
        family: sorted(
            lineage for lineage, observed in lineage_family.items() if observed == family
        )
        for family in families
    }
    generic_rows, provenance, route_manifest = [], [], []
    family_counts, rule_counts = Counter(), Counter()
    runtime_supported_instances = sum(len(route.actions) <= 32 for route in routes)
    for trace_identity, route, members in unique_routes:
        graphs, segments = recognized[trace_identity]
        family = route.task_family
        denominator = (
            len(families) * len(family_lineages[family]) * lineage_decisions[route.lineage_identity]
        )
        weight = 1.0 / denominator
        created: dict[int, tuple[int, int]] = {}
        next_ordinal = 0
        member_rows = [
            {
                "instance_identity": member.instance_identity,
                "collection": member.collection,
                "task": member.task,
                "task_family": member.task_family,
                "member_id": member.member_id,
                "source_path": member.source_path,
                "evidence_role": member.evidence_role,
            }
            for member in members
        ]
        route_manifest.append(
            {
                "trace_identity": trace_identity,
                "lineage_identity": route.lineage_identity,
                "members": member_rows,
                "primitive_transitions": len(route.actions),
                "runtime_complete_route_supported": len(route.actions) <= 32,
                "endpoint_identity": identity(route.endpoint),
            }
        )
        for step, (graph, segment, action) in enumerate(
            zip(graphs[:-1], segments, route.actions, strict=True)
        ):
            action_supervision, next_ordinal = action_role_supervision(
                graph, action, created, step, next_ordinal
            )
            descriptor = stage_descriptor(graph, [action])
            row_index = len(generic_rows)
            generic_rows.append(
                {
                    "row_index": row_index,
                    "state_features": molecule_features(canonical_state_key(graph)).tolist(),
                    "option": segment.option,
                    "stage_descriptor": descriptor.tolist(),
                    "action_supervision": action_supervision,
                    "runtime_complete_route_supported": len(route.actions) <= 32,
                    "balance": {
                        "domain_count": 1,
                        "family_count": len(families),
                        "lineage_count_within_family": len(family_lineages[family]),
                        "decision_count_within_lineage": lineage_decisions[route.lineage_identity],
                        "weight": weight,
                    },
                }
            )
            provenance.append(
                {
                    "row_index": row_index,
                    "trace_identity": trace_identity,
                    "lineage_identity": route.lineage_identity,
                    "primitive_index": step,
                    "members": member_rows,
                }
            )
            family_counts[family] += 1
            rule_counts[action["executor_rule"]] += 1
    total_weight = float(sum(row["balance"]["weight"] for row in generic_rows))
    family_weight = {
        family: float(
            sum(
                generic_rows[row["row_index"]]["balance"]["weight"]
                for row in provenance
                if row["members"][0]["task_family"] == family
            )
        )
        for family in families
    }
    if not np.isclose(total_weight, 1.0) or any(
        not np.isclose(value, 1 / len(families)) for value in family_weight.values()
    ):
        raise RuntimeError("hierarchical supervision weights are not normalized")
    forbidden_keys = sorted(FORBIDDEN_GENERIC_KEYS & set(_keys(generic_rows)))
    teacher_values = set(contract["task_families"])
    for route in routes:
        teacher_values.update(
            (
                route.endpoint,
                route.member_id,
                route.source_path,
                route.instance_identity,
                route.trace_identity,
            )
        )
    forbidden_values = sorted(teacher_values & set(_strings(generic_rows)))
    if forbidden_keys or forbidden_values:
        raise RuntimeError(
            f"generic supervision leaked teacher content: keys={forbidden_keys}, values={forbidden_values}"
        )
    dataset = {
        "schema_version": DATASET_SCHEMA,
        "evidence": "answer_known_pmo_route_supervision",
        "artifact_role": "training_only_not_a_runtime_checkpoint",
        "policy_schema": POLICY_SCHEMA,
        "action_role_schema": ACTION_ROLE_SCHEMA,
        "state_feature_dim": len(generic_rows[0]["state_features"]),
        "stage_descriptor_names": list(STAGE_DESCRIPTOR_NAMES),
        "option_menu": list(descriptor_menu()),
        "generic_rows": generic_rows,
        "training_provenance": provenance,
        "route_manifest": route_manifest,
        "exclusion_manifest": source_audit["excluded"],
        "actor_training_performed": False,
        "module_count_targets_emitted": False,
        "binding_prototype_targets_emitted": False,
        "new_oracle_calls": 0,
    }
    summary = {
        "route_instances": len(routes),
        "curriculum_programs": curriculum_count,
        "witness_routes": witness_count,
        "unique_exact_traces": len(unique_routes),
        "duplicate_route_instances": len(routes) - len(unique_routes),
        "primitive_transitions_with_multiplicity": transition_count,
        "emitted_deduplicated_decisions": len(generic_rows),
        "recognized_compound_stages": 0,
        "primitive_fallback_decisions": len(generic_rows),
        "runtime_supported_route_instances": runtime_supported_instances,
        "long_route_local_only_instances": len(routes) - runtime_supported_instances,
        "runtime_supported_unique_traces": sum(
            len(route.actions) <= 32 for _, route, _ in unique_routes
        ),
        "long_route_local_only_unique_traces": sum(
            len(route.actions) > 32 for _, route, _ in unique_routes
        ),
        "task_families": len(families),
        "task_route_census": {
            task: {
                "route_instances": sum(route.task == task for route in routes),
                "primitive_transitions_with_multiplicity": sum(
                    len(route.actions) for route in routes if route.task == task
                ),
                "runtime_supported_route_instances": sum(
                    route.task == task and len(route.actions) <= 32 for route in routes
                ),
                "long_route_local_only_instances": sum(
                    route.task == task and len(route.actions) > 32 for route in routes
                ),
            }
            for task in sorted(contract["task_families"])
        },
        "lineages": len(lineage_decisions),
        "family_decision_counts": dict(sorted(family_counts.items())),
        "executor_rule_counts": dict(sorted(rule_counts.items())),
        "total_weight": total_weight,
        "family_weights": family_weight,
        "generic_forbidden_key_hits": forbidden_keys,
        "generic_forbidden_value_hits": forbidden_values,
    }
    for key in (
        "unique_exact_traces",
        "deduplicated_decisions",
        "lineages",
        "runtime_supported_route_instances",
        "long_route_local_only_instances",
        "runtime_supported_unique_traces",
        "long_route_local_only_unique_traces",
    ):
        observed_key = "emitted_deduplicated_decisions" if key == "deduplicated_decisions" else key
        if summary[observed_key] != expected[key]:
            raise ValueError(
                f"frozen PMO route statistic changed for {key}: "
                f"expected {expected[key]}, got {summary[observed_key]}"
            )
    return dataset, summary


def _publish(path: Path, payload: dict, *, compressed: bool = False) -> None:
    envelope = {"payload": payload, "payload_sha256": identity(payload)}
    raw = json.dumps(envelope, sort_keys=True, separators=(",", ":")) + "\n"
    if compressed:
        path.write_bytes(gzip.compress(raw.encode(), mtime=0))
    else:
        path.write_text(raw)


def run(output: Path, root: Path = ROOT) -> dict:
    if output.exists():
        raise ValueError(f"refusing to overwrite PMO route-distillation output: {output}")
    contract = load_contract(root)
    routes, source_audit = collect_routes(contract, root)
    dataset, summary = build_dataset(contract, routes, source_audit)
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    dirty = bool(
        subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True).strip()
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        dataset_path = temporary / contract["outputs"]["dataset"]
        _publish(dataset_path, dataset, compressed=True)
        report = {
            "schema_version": RESULT_SCHEMA,
            "decision": "zero_oracle_primitive_supervision_export_passed_actor_training_prohibited",
            "contract_path": str(CONTRACT),
            "contract_sha256": sha256_file(root / CONTRACT),
            "source_checks": source_audit["source_checks"],
            "summary": summary,
            "outputs": {contract["outputs"]["dataset"]: sha256_file(dataset_path)},
            "gates": {
                "input_manifest": True,
                "exact_trace_replay": True,
                "generic_descriptors": True,
                "runtime_support_separated": True,
                "hierarchical_balance": True,
                "generic_row_leakage_scan": True,
                "zero_compound_negative_result_preserved": True,
                "actor_training_absent": True,
            },
            "implementation": {
                "revision": revision,
                "working_tree_dirty": dirty,
                "script_sha256": sha256_file(Path(__file__)),
                "policy_sha256": sha256_file(
                    root / "src/compose_v4/control/route_distilled_program_policy.py"
                ),
                "python": platform.python_version(),
                "numpy": np.__version__,
                "rdkit": rdBase.rdkitVersion,
                "device": "CPU",
                "randomness": "none",
            },
            "costs": {"new_oracle_calls": 0, "new_docking_calls": 0},
            "limitations": [
                "All PMO supervision is answer-known, panel-informed or winner-informed development evidence.",
                "No compound PMO stage was recognized; every emitted decision is a primitive fallback.",
                "Long routes are local-decision supervision, not runtime-supported complete routes.",
                "Terminal PMO scores are not pathwise value labels and do not weight this dataset.",
                "No actor, runtime checkpoint, module-count target or binding-prototype target was produced.",
            ],
        }
        _publish(temporary / contract["outputs"]["audit"], report)
        temporary.replace(output)
    except Exception:
        shutil.rmtree(temporary)
        raise
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "diagnostics/pmo_route_distillation/attempt_1",
    )
    result = run(parser.parse_args().output)
    print(json.dumps({"decision": result["decision"], "summary": result["summary"]}, indent=2))
