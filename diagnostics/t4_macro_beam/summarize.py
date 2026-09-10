"""Reduce locked macro-beam evidence without generation, fitting or docking."""

from __future__ import annotations

import ast
import hashlib
import json
import platform
import statistics
import subprocess
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

from rdkit import Chem, rdBase

from compose_v4.gates.med_chem_gate import validity_reasons


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def distribution(values):
    values = list(values)
    return (
        {
            "n": len(values),
            "min": min(values),
            "median": statistics.median(values),
            "max": max(values),
        }
        if values
        else {"n": 0, "min": None, "median": None, "max": None}
    )


def summarize(root):
    hashes = {}

    def read(relative, sealed=False):
        path = root / relative
        raw = path.read_bytes()
        hashes[relative] = hashlib.sha256(raw).hexdigest()
        value = json.loads(raw)
        if sealed:
            if digest(value["payload"]) != value["payload_sha256"]:
                raise ValueError(f"corrupt sealed artifact: {path}")
            return value["payload"]
        return value

    launches = [read(f"attempt_{attempt}/launch.json") for attempt in (1, 2)]
    contract_path = "../../configs/t4_macro_beam.json"
    contract = read(contract_path)
    assert rdBase.rdkitVersion == contract["required_rdkit"]
    archive_path = "../t4_ring_program_round/attempt_1/run/round_4/archive.json"
    archive = read(archive_path, True)
    assert hashes[archive_path] == contract["archive"]["sha256"]
    canonical_archive = {
        Chem.MolToSmiles(Chem.MolFromSmiles(r["smiles"])) for r in archive["archive"]
    }
    assert all(hashes[contract_path] == launch["task"]["contract_sha256"] for launch in launches)
    assert (
        digest({k: v for k, v in contract.items() if k != "contract_sha256"})
        == contract["contract_sha256"]
    )
    cases, snapshots, paired_locks = [], [], {}
    for case in range(4):
        attempt = 2 if case % 2 == 0 else 1
        launch = launches[attempt - 1]
        prefix = f"attempt_{attempt}/case_{case}/"
        result = read(prefix + "result.json")
        lock = read(prefix + "generation_lock.json", True)
        scored = read(prefix + "scored_lock.json", True)
        cache = read(prefix + "law_cache_inventory.json", True)
        gate = read(prefix + "runtime_gate.json")
        heartbeat = read(prefix + "heartbeat.json")
        assert result["case_index"] == case
        assert result["code_revision"] == launch["task"]["image_revision"]["commit"]
        assert result["configuration"] == contract
        assert result["runtime_gate"] == gate
        assert gate["input_sha256"] == result["configuration"]["expected_input_sha256"]
        assert heartbeat["phase"] == "complete"
        assert result["oracle_calls"] == lock["oracle_calls"] == launch["oracle_calls"] == 0
        assert lock["winner_used"] is False and result["replay_verified"] is True
        assert result["config"] == lock["config"]
        assert scored["candidates"] == result["candidates"]
        snapshots.append(result["value_snapshot_sha256"])
        assert scored["value_snapshot_sha256"] == snapshots[-1]
        attempts = lock["attempts"]
        assert len(attempts) == result["attempts"] <= 30
        assert all(a["status"] in ("complete", "support_dead_end") for a in attempts)
        complete = [a for a in attempts if a["status"] == "complete"]
        assert len(complete) == result["completed"] == len(result["candidates"])
        assert all(a["node"]["stage"] == "where" for a in complete)
        assert all(
            a["replayed_primitives"] == sum("mark" in e for e in a["events"]) for a in complete
        )
        by_attempt = {a["attempt_id"]: a for a in attempts}
        products = {p["attempt_id"]: p for p in result["candidates"]}
        unique = {p["smiles"]: p for p in products.values()}
        medchem = {s: validity_reasons(s) for s in unique}
        assert all(
            p["in_prior_archive"] == (p["smiles"] in canonical_archive) for p in products.values()
        )
        assert result["source"] in archive["archive"]
        eligible = {s: p for s, p in unique.items() if p["oracle_eligible"]}
        new = {s: p for s, p in eligible.items() if not p["in_prior_archive"]}
        assert len(unique) == result["unique_completed"]
        assert len(new) == result["unique_new_eligible"]
        levels = []
        for level in lock["levels"]:
            decision, beam = level["decision"], level["beam"]
            assert len({p["smiles"] for p in beam}) == len(beam) <= 3
            if result["config"]["arm"] == "post_hoc":
                assert decision.get("first_slot_values") is None
            if "first_slot_probabilities" in decision:
                q, p = decision["first_slot_probabilities"], decision["first_slot_reference"]
                assert abs(sum(q) - 1) < 1e-10 and all(a >= 0.1 * b - 1e-12 for a, b in zip(q, p))
                assert decision["first_slot_kl_against_empirical_pool"] <= 1 + 1e-10
            levels.append(
                {
                    "depth": level["depth"] + 1,
                    "attempts": len(level["attempts"]),
                    "complete": sum(a["status"] == "complete" for a in level["attempts"]),
                    "retained": [products[r["attempt_id"]]["smiles"] for r in beam],
                    "first_slot_kl": decision.get("first_slot_kl_against_empirical_pool"),
                    "desirability_distribution": distribution(
                        decision.get("first_slot_values") or []
                    ),
                    "zero_desirabilities": sum(
                        v == 0 for v in (decision.get("first_slot_values") or [])
                    ),
                }
            )
        chains = []
        for chain in result["final_chains"]:
            endpoint = products[chain[-1]]
            chains.append(
                {
                    "options": [by_attempt[a]["bundle"]["option"] for a in chain],
                    "primitive_steps": sum(by_attempt[a]["replayed_primitives"] for a in chain),
                    "endpoint": endpoint["smiles"],
                    "eligible": endpoint["oracle_eligible"],
                    "predicted_docking": endpoint["predicted_docking"],
                    "cumulative_change": endpoint["cumulative_change"],
                }
            )
        best = min(new.values(), key=lambda p: (p["predicted_docking"], p["smiles"]), default=None)
        cases.append(
            {
                "case_index": case,
                "generation_revision": result["code_revision"],
                "run_id": launch["task"]["run_id"],
                "config": result["config"],
                "source": result["source"]["smiles"],
                "source_observed_docking": result["source"]["ds"],
                "source_predicted_docking": result["source_prediction"]["predicted_docking"],
                "source_properties": result["source_prediction"],
                "attempts": len(attempts),
                "complete": len(complete),
                "support_dead_ends": len(attempts) - len(complete),
                "support_dead_end_details": [
                    {
                        "attempt_id": a["attempt_id"],
                        "option": a["bundle"]["option"] if a["bundle"] else None,
                        "committed_primitives": sum("mark" in e for e in a["events"]),
                    }
                    for a in attempts
                    if a["status"] == "support_dead_end"
                ],
                "unique_complete": len(unique),
                "unique_eligible": len(eligible),
                "unique_new_eligible": len(new),
                "unique_passing_existing_medchem_screen": sum(
                    not reasons for reasons in medchem.values()
                ),
                "medchem_failure_reasons": {
                    s: reasons for s, reasons in medchem.items() if reasons
                },
                "similarity_complete": distribution(p["sim"] for p in unique.values()),
                "eligibility_failure_counts_unique_nonexclusive": {
                    "qed_below_0.6": sum(p["qed"] < 0.6 for p in unique.values()),
                    "sa_above_4": sum(p["sa"] > 4 for p in unique.values()),
                    "similarity_below_0.4": sum(p["sim"] < 0.4 for p in unique.values()),
                    "medchem_only": sum(
                        p["v"] == 0 and not p["oracle_eligible"] for p in unique.values()
                    ),
                },
                "completed_option_added_cycles": sum(
                    p["structural_change"]["d_cycle_rank"] > 0 for p in products.values()
                ),
                "completed_option_added_ring_systems": sum(
                    p["structural_change"]["d_ring_systems"] > 0 for p in products.values()
                ),
                "complete_cycle_rank_deltas_from_root": dict(
                    sorted(
                        Counter(
                            p["cumulative_change"]["d_cycle_rank"] for p in unique.values()
                        ).items()
                    )
                ),
                "complete_ring_system_deltas_from_root": dict(
                    sorted(
                        Counter(
                            p["cumulative_change"]["d_ring_systems"] for p in unique.values()
                        ).items()
                    )
                ),
                "option_attempts": dict(
                    sorted(Counter(a["bundle"]["option"] for a in attempts if a["bundle"]).items())
                ),
                "unassigned_attempts": sum(a["bundle"] is None for a in attempts),
                "primitive_steps_replayed": sum(a["replayed_primitives"] for a in complete),
                "intended_release_complete": distribution(
                    p["bundle"]["r_release"] for p in products.values()
                ),
                "realized_coherent_per_option_complete": distribution(
                    p["structural_change"]["largest_changed_fraction"] for p in products.values()
                ),
                "realized_coherent_from_root_new_eligible": distribution(
                    p["cumulative_change"]["largest_changed_fraction"] for p in new.values()
                ),
                "new_eligible_cycle_rank_deltas_from_root": dict(
                    sorted(
                        Counter(
                            p["cumulative_change"]["d_cycle_rank"] for p in new.values()
                        ).items()
                    )
                ),
                "new_eligible_ring_system_deltas_from_root": dict(
                    sorted(
                        Counter(
                            p["cumulative_change"]["d_ring_systems"] for p in new.values()
                        ).items()
                    )
                ),
                "new_eligible_with_added_cycles": sum(
                    p["cumulative_change"]["d_cycle_rank"] > 0 for p in new.values()
                ),
                "best_new_eligible": None
                if best is None
                else {
                    k: best[k]
                    for k in (
                        "smiles",
                        "predicted_docking",
                        "desirability",
                        "qed",
                        "sa",
                        "sim",
                        "chain",
                        "cumulative_change",
                    )
                },
                "final_chains": chains,
                "levels": levels,
                "proposal_seconds": result["proposal_seconds_this_invocation"],
                "initialization_seconds": gate["initialization_seconds"],
                "total_seconds": result["elapsed_seconds"],
                "law_work": result["law_work"],
                "cache_inventory": cache,
                "public_executor_calls": result["public_executor_calls_this_invocation"],
                "peak_rss_kib_linux": result["peak_rss_native_units"],
            }
        )
        paired_locks[case] = lock
    assert len(set(snapshots)) == 1
    pairs = []
    # Before the first task-informed retention, paired generation must coincide.
    for case in (0, 1):
        assert (
            paired_locks[case]["levels"][0]["attempts"]
            == paired_locks[case + 2]["levels"][0]["attempts"]
        )
        left = {
            a["candidate"]["smiles"]
            for a in paired_locks[case]["attempts"]
            if a["status"] == "complete"
        }
        right = {
            a["candidate"]["smiles"]
            for a in paired_locks[case + 2]["attempts"]
            if a["status"] == "complete"
        }
        pairs.append(
            {
                "root_index": case,
                "canonical_intersection": len(left & right),
                "canonical_union": len(left | right),
                "identical_complete_attempts": paired_locks[case]["attempts"]
                == paired_locks[case + 2]["attempts"],
            }
        )
    module = "src/compose_v4/experiments/t4_macro_beam.py"
    revisions = [launch["task"]["image_revision"] for launch in launches]
    assert {p: h for p, h in revisions[0]["serialized_sources"].items() if p != module} == {
        p: h for p, h in revisions[1]["serialized_sources"].items() if p != module
    }

    class RemoveDocumentedInputRepair(ast.NodeTransformer):
        def visit_FunctionDef(self, node):
            if node.name in ("canonical_smiles", "exact_archive_graph"):
                return None
            if node.name == "runner":
                node.body = [
                    statement
                    for statement in node.body
                    if not (
                        isinstance(statement, ast.Assign)
                        and any(
                            isinstance(t, ast.Name) and t.id in ("graph", "old")
                            for t in statement.targets
                        )
                    )
                    and not (
                        isinstance(statement, ast.If)
                        and "beam root exact state differs from canonical metadata"
                        in ast.unparse(statement)
                    )
                ]
            return self.generic_visit(node)

    source_dumps = []
    for revision in revisions:
        source = subprocess.check_output(
            ["git", "show", f"{revision['commit']}:{module}"], cwd=root, text=True
        )
        source_dumps.append(ast.dump(RemoveDocumentedInputRepair().visit(ast.parse(source))))
    assert source_dumps[0] == source_dumps[1]
    gate_source = "src/compose_v4/gates/med_chem_gate.py"
    gate_hash = hashlib.sha256((root.parents[1] / gate_source).read_bytes()).hexdigest()
    assert all(revision["serialized_sources"][gate_source] == gate_hash for revision in revisions)
    hashes["../../" + gate_source] = gate_hash
    failures = []
    for case in (0, 2):
        failure = read(f"attempt_1/case_{case}/failure.json")
        heartbeat = read(f"attempt_1/case_{case}/heartbeat.json")
        gate = read(f"attempt_1/case_{case}/runtime_gate.json")
        assert failure["message"] == "beam root exact state differs from canonical metadata"
        assert failure["phase"] == "initialization"
        failures.append(
            {
                "case_index": case,
                "failure": failure,
                "elapsed_seconds": heartbeat["elapsed_seconds"],
                "initialization_seconds": gate["initialization_seconds"],
            }
        )
    return {
        "schema_version": "t4_macro_beam_summary_v1",
        "cases": cases,
        "paired_generation": pairs,
        "generation_revisions": [r["commit"] for r in revisions],
        "run_ids": [launch["task"]["run_id"] for launch in launches],
        "preserved_failed_attempts": failures,
        "repair_containment": "serialized dependencies equal; driver AST equal outside root-identity validation and post-lock archive identity canonicalization",
        "oracle_calls": 0,
        "value_snapshot_sha256": snapshots[0],
        "input_sha256": dict(sorted(hashes.items())),
        "claim": "paired winner-blind development search; surrogate predictions are not docking outcomes",
    }


if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    suite = ET.parse(root / "focused_tests.xml").getroot()
    assert sum(int(s.get("tests", "0")) for s in suite.iter("testsuite")) == 8
    assert all(
        int(s.get(k, "0")) == 0
        for s in suite.iter("testsuite")
        for k in ("failures", "errors", "skipped")
    )
    repair_suite = ET.parse(root / "repair_focused_tests.xml").getroot()
    assert sum(int(s.get("tests", "0")) for s in repair_suite.iter("testsuite")) == 9
    assert all(
        int(s.get(k, "0")) == 0
        for s in repair_suite.iter("testsuite")
        for k in ("failures", "errors", "skipped")
    )
    result = summarize(root)
    result["analysis"] = {
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "verification_sha256": hashlib.sha256(
            (root / "focused_tests.xml").read_bytes()
        ).hexdigest(),
        "repair_verification_sha256": hashlib.sha256(
            (root / "repair_focused_tests.xml").read_bytes()
        ).hexdigest(),
        "revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip(),
        "python": platform.python_version(),
        "rdkit": rdBase.rdkitVersion,
        "platform": platform.platform(),
        "command": "python3 diagnostics/t4_macro_beam/summarize.py",
        "randomness": "none",
    }
    path = root / "summary.json"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)
    print(json.dumps({"cases": result["cases"], "oracle_calls": 0}, sort_keys=True, indent=2))
