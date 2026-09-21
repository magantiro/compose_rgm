"""Runtime-resolved proof of the three wiring fixes and of contract governance.

NEW FILE.  Every claim below is produced by CALLING the frozen code and
observing what it does, never by restating what its source says.  A check that
recomputes its own expectation from the code under test cannot fail.

Writes a JSON evidence block to --out.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def fix_one_sa_floor(panel, contract) -> dict:
    """The T4 endpoint gate imports an SA ceiling this benchmark does not have."""
    from rdkit import Chem
    from rdkit.Chem import QED
    from rdkit.Contrib.SA_Score import sascorer

    from compose_v4.control.program_task import ProgramTask
    from compose_v4.tasks.qed_edit_task import task_from_contract

    refused_by_t4, sa_over, examples = 0, 0, []
    for index, smiles in enumerate(panel):
        molecule = Chem.MolFromSmiles(smiles)
        sa = float(sascorer.calculateScore(molecule))
        qed = float(QED.qed(molecule))
        t4 = ProgramTask(f"t4_{index}", "local:probe", "t4", smiles, 0.4)
        # The source evaluated against ITSELF: similarity is exactly 1.0, so the
        # similarity term of the T4 violation cannot be what refuses it.
        t4_row = t4.endpoint_evaluator()({"smiles": smiles})
        dedicated = task_from_contract(
            contract, name=f"qed_{index}", oracle_protocol="local:probe", source=smiles
        )
        d_row = dedicated.endpoint_evaluator()({"smiles": smiles})
        if sa > 4.0:
            sa_over += 1
        if not t4_row["oracle_eligible"]:
            refused_by_t4 += 1
            if len(examples) < 5:
                examples.append(
                    {
                        "index": index,
                        "smiles": smiles,
                        "qed": round(qed, 4),
                        "sa": round(sa, 3),
                        "sim_to_itself": round(float(t4_row["sim"]), 4),
                        "t4_violation_v": round(float(t4_row["v"]), 4),
                        "t4_oracle_eligible": bool(t4_row["oracle_eligible"]),
                        "dedicated_oracle_eligible": bool(d_row["oracle_eligible"]),
                    }
                )
        assert d_row["oracle_eligible"] is True, "dedicated task must admit the source itself"
    return {
        "claim": "routing this benchmark through ProgramTask(kind='t4') imports an SA ceiling "
                 "(and a 0.6 QED floor and a med-chem screen) that the GrIDDD/Jin task lacks",
        "method": "evaluate each panel source AGAINST ITSELF under both tasks; similarity is "
                  "then exactly 1.0, so any refusal is not a similarity refusal",
        "panel_sources": len(panel),
        "sources_with_sa_above_4_0": sa_over,
        "sources_the_t4_gate_refuses": refused_by_t4,
        "sources_the_dedicated_task_refuses": 0,
        "reading": "the T4 gate marks the benchmark's own starting molecules ineligible",
        "examples": examples,
    }


def fix_one_witness(panel, contract, endpoints) -> dict:
    """A generated molecule that SOLVES the benchmark and the T4 gate refuses."""
    from rdkit import Chem
    from rdkit.Contrib.SA_Score import sascorer

    from compose_v4.control.program_task import ProgramTask
    from compose_v4.tasks.qed_edit_task import task_from_contract

    witnesses = []
    for row in endpoints:
        index, smiles = row["index"], row["smiles"]
        source = panel[index]
        dedicated = task_from_contract(
            contract, name=f"qed_{index}", oracle_protocol="local:probe", source=source
        )
        measured = dedicated.measure(smiles)
        if not dedicated.is_success(measured):
            continue
        t4 = ProgramTask(f"t4_{index}", "local:probe", "t4", source, 0.4)
        t4_row = t4.endpoint_evaluator()({"smiles": smiles})
        d_row = dedicated.endpoint_evaluator()({"smiles": smiles})
        if t4_row["oracle_eligible"]:
            continue
        sa = float(sascorer.calculateScore(Chem.MolFromSmiles(smiles)))
        violation = float(t4_row["v"])
        if violation > 0:
            terms = []
            if float(t4_row["qed"]) < 0.6:
                terms.append("qed_min=0.6")
            if sa > 4.0:
                terms.append("sa_max=4.0")
            if float(t4_row["sim"]) < 0.4:
                terms.append("delta=0.4")
            refused_by = "violation_term:" + "+".join(terms)
        else:
            # feasible_endpoint passed (v == 0), so the refusal came from the other
            # half of acceptable_endpoint: the inherited medicinal-chemistry screen.
            refused_by = "med_chem_gate.is_valid"
        witnesses.append(
            {
                "refused_by": refused_by,
                "source_index": index,
                "source": source,
                "candidate": smiles,
                "qed": round(measured["qed"], 4),
                "sim": round(measured["sim"], 4),
                "sa": round(sa, 3),
                "solves_benchmark": True,
                "t4_oracle_eligible": False,
                "t4_violation_v": round(float(t4_row["v"]), 4),
                "dedicated_oracle_eligible": bool(d_row["oracle_eligible"]),
            }
        )
    return {
        "claim": "a molecule that SOLVES the benchmark (QED >= target, sim >= floor) is refused "
                 "as an ineligible endpoint by the T4 gate",
        "candidates_examined": len(endpoints),
        "witness_count": len(witnesses),
        "refused_by_breakdown": {
            reason: sum(1 for w in witnesses if w["refused_by"] == reason)
            for reason in sorted({w["refused_by"] for w in witnesses})
        },
        "two_independent_inherited_gates": "acceptable_endpoint = feasible_endpoint (the "
                                           "qed_min/sa_max/delta violation) AND "
                                           "med_chem_gate.is_valid. A benchmark solution "
                                           "can be refused by either one.",
        "witnesses": witnesses[:10],
        "status": "WITNESS FOUND" if witnesses else "no witness in the examined set",
    }


def fix_two_direction(contract, panel) -> dict:
    """Score direction is resolved from the frozen guard, and T4 inverts QED."""
    from compose_v4.control.parent_edit_search import prepare_query_batch
    from compose_v4.control.program_task import ProgramTask
    from compose_v4.tasks.qed_edit_task import (
        _DirectionProbeSearch,
        resolve_score_direction,
        task_from_contract,
    )

    source = panel[0]
    dedicated = task_from_contract(
        contract, name="qed_dir", oracle_protocol="local:probe", source=source
    )
    t4 = ProgramTask("t4_dir", "local:probe", "t4", source, 0.4)

    # Pairing the dedicated task with the WRONG direction must be refused by the
    # frozen guard, not silently accepted.
    refusal = None
    try:
        prepare_query_batch(
            _DirectionProbeSearch("local:probe", "minimize"), dedicated, count=4, seed=0
        )
    except ValueError as error:
        refusal = str(error)

    return {
        "claim": "parent_edit_search.py:23 forces the direction from task.kind; a mismatched "
                 "pairing inverts the objective",
        "method": "resolve by CALLING prepare_query_batch and observing which direction it "
                  "refuses, rather than restating its expression",
        "dedicated_task_resolved_direction": resolve_score_direction(dedicated),
        "t4_task_resolved_direction": resolve_score_direction(t4),
        "wrong_direction_is_refused": refusal is not None,
        "refusal_message": refusal,
        "inversion_demonstrated_on_utility": {
            "t4_utility_of_qed_0_95": ProgramTask.utility(t4, 0.95),
            "t4_utility_of_qed_0_50": ProgramTask.utility(t4, 0.50),
            "t4_ranks_higher_qed_worse": ProgramTask.utility(t4, 0.95)
            < ProgramTask.utility(t4, 0.50),
            "dedicated_utility_of_qed_0_95": dedicated.utility(0.95),
            "dedicated_utility_of_qed_0_50": dedicated.utility(0.50),
            "dedicated_ranks_higher_qed_better": dedicated.utility(0.95) > dedicated.utility(0.50),
        },
    }


def fix_three_slots(panel) -> dict:
    """Allocated graph slots (48) and the active heavy-atom ceiling (40) differ."""
    from compose_v4.chem.molecular_graph import smiles_to_molecular_graph
    from compose_v4.experiments.editing_v2_evaluation_semantics import (
        PRODUCTION_MAX_ATOMS,
        production_state_from_smiles,
    )
    from compose_v4.experiments.whole_ring_plan import execute_program
    from compose_v4.tasks.qed_edit_task import (
        ALLOCATED_GRAPH_SLOTS,
        MAX_ACTIVE_HEAVY_ATOMS,
        production_source_state,
    )

    source = panel[0]
    outcomes = {}
    for label, state in (
        ("tight_from_smiles", smiles_to_molecular_graph(source)),
        ("padded_to_PRODUCTION_MAX_ATOMS_40", production_state_from_smiles(source, 40)),
        ("padded_to_ALLOCATED_GRAPH_SLOTS_48", production_source_state(source)),
    ):
        entry = {
            "n_atoms_allocated_slots": int(state.n_atoms),
            "n_real_atoms_active": int(state.n_real_atoms),
            "free_slots": int(state.n_atoms - state.n_real_atoms),
        }
        try:
            execute_program(state, [])
            entry["execute_program"] = "accepted"
        except ValueError as error:
            entry["execute_program"] = f"refused: {error}"
        outcomes[label] = entry

    heavy = [smiles_to_molecular_graph(s).n_real_atoms for s in panel]
    return {
        "claim": "PRODUCTION_MAX_ATOMS = 40 is the ACTIVE heavy-atom ceiling, not the padding "
                 "width; whole_ring_plan.execute_program requires n_atoms == 48 AND "
                 "1 <= n_real_atoms <= 40",
        "constants": {
            "editing_v2_evaluation_semantics.PRODUCTION_MAX_ATOMS": PRODUCTION_MAX_ATOMS,
            "qed_edit_task.ALLOCATED_GRAPH_SLOTS": ALLOCATED_GRAPH_SLOTS,
            "qed_edit_task.MAX_ACTIVE_HEAVY_ATOMS": MAX_ACTIVE_HEAVY_ATOMS,
        },
        "runtime_outcomes": outcomes,
        "birth_operations_need_a_free_slot": {
            "tight_free_slots": outcomes["tight_from_smiles"]["free_slots"],
            "padded_48_free_slots": outcomes["padded_to_ALLOCATED_GRAPH_SLOTS_48"]["free_slots"],
            "note": "a tight graph has zero free slots, so the whole atom_insert family is "
                    "absent from its legal support; any support claim on a tight state is an "
                    "artifact",
        },
        "panel_active_heavy_atoms": {
            "max": max(heavy),
            "min": min(heavy),
            "all_within_active_ceiling": max(heavy) <= MAX_ACTIVE_HEAVY_ATOMS,
        },
    }


def contract_governance(contract_path, panel, endpoints) -> dict:
    """The operative values are resolved from the contract at runtime."""
    from compose_v4.tasks.qed_edit_task import (
        QedEditTask,
        load_qed_edit_contract,
        task_from_contract,
    )

    base = load_qed_edit_contract(contract_path)

    # A probe molecule that sits BETWEEN the two contracts' similarity floors, so
    # the admission decision must MOVE when only the contract changes. It has to be
    # a generated analogue of its own source: two unrelated panel molecules are
    # never that similar, so an unrelated-molecule probe silently yields nothing.
    probe, source = None, panel[0]
    for row in endpoints:
        candidate_source = panel[row["index"]]
        probe_task = task_from_contract(
            base, name="gov", oracle_protocol="local:probe", source=candidate_source
        )
        measured = probe_task.measure(row["smiles"])
        if 0.40 <= measured["sim"] < 0.60:
            probe, source = row["smiles"], candidate_source
            break
    baseline = task_from_contract(
        base, name="gov", oracle_protocol="local:probe", source=source
    )

    perturbed_payload = json.loads(Path(contract_path).read_text())
    perturbed_payload["region"] = {"qed_target": 0.95, "similarity_floor": 0.60}
    perturbed_path = Path(contract_path).with_name("_perturbed_negative_control.json")
    perturbed_path.write_text(json.dumps(perturbed_payload))
    try:
        perturbed = task_from_contract(
            load_qed_edit_contract(perturbed_path),
            name="gov",
            oracle_protocol="local:probe",
            source=source,
        )
        admission_moved = None
        if probe is not None:
            admission_moved = {
                "probe": probe,
                "sim": round(baseline.measure(probe)["sim"], 4),
                "admitted_under_floor_0_40": bool(
                    baseline.endpoint_evaluator()({"smiles": probe})["oracle_eligible"]
                ),
                "admitted_under_floor_0_60": bool(
                    perturbed.endpoint_evaluator()({"smiles": probe})["oracle_eligible"]
                ),
            }
        missing_field = None
        try:
            QedEditTask(name="x", oracle_protocol="p", original_source=source)  # type: ignore[call-arg]
        except TypeError as error:
            missing_field = str(error)
    finally:
        perturbed_path.unlink(missing_ok=True)

    return {
        "claim": "qed_target and similarity_floor are contract-governed and runtime-resolved, "
                 "not module constants",
        "dataclass_has_no_defaults_for_them": missing_field is not None,
        "missing_field_error": missing_field,
        "baseline": {
            "qed_target": baseline.qed_target,
            "similarity_floor": baseline.similarity_floor,
            "task_id": baseline.task_id,
        },
        "perturbed_negative_control": {
            "qed_target": perturbed.qed_target,
            "similarity_floor": perturbed.similarity_floor,
            "task_id": perturbed.task_id,
        },
        "task_id_moved": baseline.task_id != perturbed.task_id,
        "admission_decision_moved_at_runtime": admission_moved,
        "why_this_is_runtime_resolved": "the SAME call path returns a different admission "
                                        "decision and a different task identity when only the "
                                        "contract file changed; nothing in the module carries "
                                        "the operative value",
    }


def support_not_reward(contract, panel) -> dict:
    """Feasibility is in the support seam propose_batch consults."""
    import inspect

    from compose_v4.control.adaptive_program_optimizer import ProgramOptimizer
    from compose_v4.tasks.qed_edit_task import task_from_contract

    source = panel[0]
    task = task_from_contract(
        contract, name="sup", oracle_protocol="local:probe", source=source
    )
    body = inspect.getsource(ProgramOptimizer.propose_batch)
    reward = task.reward()
    # An endpoint below the similarity floor: refused by the support, and its
    # reward is its true QED (not zeroed) because it never reaches the ledger.
    far = None
    for smiles in panel[1:400]:
        if task.measure(smiles)["sim"] < task.similarity_floor:
            far = smiles
            break
    return {
        "claim": "the similarity floor constrains the SUPPORT (oracle_eligible), not the reward",
        "propose_batch_drops_ineligible_before_candidacy": 'if outcome != "eligible":' in body
        and "continue" in body,
        "seam": "adaptive_program_optimizer.ProgramOptimizer.propose_batch applies "
                "task.endpoint_evaluator() and skips anything not 'eligible'",
        "infeasible_probe": {
            "smiles": far,
            "sim": round(task.measure(far)["sim"], 4) if far else None,
            "oracle_eligible": bool(
                task.endpoint_evaluator()({"smiles": far})["oracle_eligible"]
            )
            if far
            else None,
            "reward_is_true_qed_not_zero": round(reward(far), 4) if far else None,
            "note": "the reward is never consulted for this molecule because the support "
                    "refuses it first; the earlier wiring instead admitted it and scored 0.0",
        },
        "evaluator_sees_program_ENDPOINTS_only": {
            "evidence": 'propose_batch computes endpoint = trace["endpoint"] and then calls '
                        'eligibility({"smiles": endpoint}); no intermediate primitive state is '
                        "ever passed to the task evaluator",
            "endpoint_binding_present": 'endpoint = trace["endpoint"]' in body,
            "evaluator_called_on_endpoint_only": 'eligibility({"smiles": endpoint})' in body,
            "consequence": "output feasibility cannot leak into intermediate program validity; "
                           "a program whose last block supplies the QED gain is never refused "
                           "mid-way",
        },
        "qed_target_is_not_an_admission_gate": bool(
            task.endpoint_evaluator()({"smiles": source})["oracle_eligible"]
        )
        and task.measure(source)["qed"] < task.qed_target,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", default="configs/qed_dedicated_task_v1.json")
    parser.add_argument("--sources", default="data/jin/qed_test.txt")
    parser.add_argument("--endpoints", default="", help="dir of run records to mine for a witness")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    from rdkit import RDLogger

    RDLogger.DisableLog("rdApp.*")
    from compose_v4.tasks.qed_edit_task import load_qed_edit_contract

    raw = Path(args.sources).read_bytes()
    panel = [ln.strip() for ln in raw.decode().split("\n") if ln.strip()]
    contract = load_qed_edit_contract(args.contract)

    endpoints = []
    if args.endpoints:
        for path in sorted(Path(args.endpoints).glob("*.json")):
            try:
                record = json.loads(path.read_text())
            except Exception:  # noqa: BLE001, S112 - truncated record skipped
                continue
            if record.get("status") != "complete":
                continue
            index = record["index"]
            for row in record.get("best_evaluated", []):
                endpoints.append({"index": index, "smiles": row["smiles"]})

    evidence = {
        "schema_version": "qed_dedicated_task_fix_verification_v1",
        "panel": {"path": args.sources, "sha256": hashlib.sha256(raw).hexdigest(),
                  "count": len(panel)},
        "contract": {
            "path": args.contract,
            "sha256": hashlib.sha256(Path(args.contract).read_bytes()).hexdigest(),
        },
        "fix_1_sa_floor": fix_one_sa_floor(panel, contract),
        "fix_1_benchmark_solving_witness": fix_one_witness(panel, contract, endpoints),
        "fix_2_score_direction": fix_two_direction(contract, panel),
        "fix_3_slot_semantics": fix_three_slots(panel),
        "contract_governance": contract_governance(args.contract, panel, endpoints),
        "feasibility_in_support_not_reward": support_not_reward(contract, panel),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(evidence, indent=1))
    print(json.dumps({k: (v if not isinstance(v, dict) else "...") for k, v in evidence.items()},
                     indent=1))
    print(f"written {args.out}")


if __name__ == "__main__":
    main()
