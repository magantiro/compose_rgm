"""One answer-known hierarchy diagnostic; no witness path or docking interface."""

from __future__ import annotations

import json
import platform
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
from rdkit import Chem, rdBase
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Contrib.SA_Score import sascorer

from compose_v4.control.continuation import ContinuationBudgetExceeded
from compose_v4.control.docking_value import graph_kernel, identity, molecular_features
from compose_v4.control.graph_geometry import structural_displacement, topology
from compose_v4.control.molecular_task_search import MolecularHierarchy, MolecularSearchState
from compose_v4.control.option_continuation import (
    EXECUTABLE_PRODUCT_GATE,
    OptionContinuationKernel,
    exact_graph_key,
    product_gate_accepts,
)
from compose_v4.control.region_rewrite import context_preserved
from compose_v4.control.task_search import TaskSearch
from compose_v4.data.charge_policy import charge_policy_preserved
from compose_v4.experiments.continuation_profile import (
    ExecutorMeter,
    encode_action,
    sha256_file,
    state_payload,
    verify_file,
)
from compose_v4.experiments.t4_endpoint_selection import acceptable_endpoint, calculate_properties
from compose_v4.experiments.t4_matched_pilot import run_remote as common_remote
from compose_v4.experiments.t4_matched_pilot import seal, unseal
from compose_v4.gates.med_chem_gate import validity_reasons
from compose_v4.rewrite.kernel import canonical_state_key
from compose_v4.rewrite.trace_shard import decode_state, encode_state

KIND = "t4_target_recovery"


@dataclass(frozen=True)
class RecoveryConfig:
    primitive_budget: int = 32
    max_rollouts: int = 8
    initial_rollouts: int = 2
    seed: int = 1000
    stop_seconds: int = 3600

    def __post_init__(self):
        for name, value in asdict(self).items():
            if type(value) is not int or value < (0 if name == "seed" else 1):
                raise ValueError(f"invalid target-recovery field {name}: {value!r}")
        if self.primitive_budget > 32 or self.max_rollouts > 8:
            raise ValueError("target recovery exceeds the authorized diagnostic allocation")
        if self.initial_rollouts > self.max_rollouts:
            raise ValueError("initial rollouts exceed total rollouts")


class TargetObjective:
    """Existing graph features, target-only reward; equality is checked separately."""

    def __init__(self, target_smiles: str):
        self.target, self.features = molecular_features(target_smiles)
        self.cache = {}
        self.snapshot = identity(
            {"kind": "known_target_mean_morgan_atompair_v1", "target": self.target}
        )

    def evaluate(self, graph):
        key = canonical_state_key(graph)
        if key not in self.cache:
            _, features = molecular_features(key)
            exact = key == self.target
            similarity = float(graph_kernel([features], [self.features])[0, 0])
            self.cache[key] = {
                "smiles": key,
                "similarity": similarity,
                # Fingerprint collisions never create an exact-hit reward.
                "value": 1.0 if exact else min(similarity, 1.0 - 1e-8),
                "exact_hit": exact,
                "state": encode_state(graph),
            }
        return {**self.cache[key], "state": encode_state(graph)}


def verify_transition(node, following, mark, system):
    """Replay only the selected mark with the existing production executor."""
    rule, action = mark
    replay = system.apply(node.graph, rule, action)
    if (
        exact_graph_key(replay) != exact_graph_key(following.graph)
        or not charge_policy_preserved(node.graph, replay)
        or not context_preserved(
            node.active.origin,
            replay,
            node.active.context.frozen,
            node.active.context.terminal_context_slots,
        )
        or not product_gate_accepts(canonical_state_key(replay), EXECUTABLE_PRODUCT_GATE)
    ):
        raise ValueError("target-recovery committed edit failed exact/pathwise verification")


def run_recovery(
    source,
    target_smiles,
    enumerate_law,
    system,
    *,
    config=None,
    save=lambda name, payload: None,
    progress=None,
):
    """Search receives a source and destination, never a witness or its actions."""
    config = config or RecoveryConfig()
    objective = TargetObjective(target_smiles)
    started, laws, path, law_seconds = perf_counter(), {}, [], 0.0
    progress = {} if progress is None else progress
    root = MolecularSearchState.start(
        source, budget=config.primitive_budget, root_id=objective.snapshot
    )
    node = root
    source_score = objective.evaluate(source)
    status = "primitive_budget_complete"
    seeds = np.random.SeedSequence(config.seed).spawn(2)
    acting_rng = np.random.default_rng(seeds[1])
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    seed_fp = generator.GetFingerprint(Chem.MolFromSmiles(canonical_state_key(source)))

    def describe(graph, lineage):
        row = dict(objective.evaluate(graph))
        properties = calculate_properties(
            Chem.MolFromSmiles(row["smiles"]),
            seed_fp=seed_fp,
            generator=generator,
            sa_scorer=sascorer.calculateScore,
            delta=0.4,
            qed_min=0.6,
            sa_max=4.0,
        )
        row.update(properties)
        row.update(
            acceptable=acceptable_endpoint(row),
            medchem_reasons=list(validity_reasons(row["smiles"])),
            topology=topology(graph),
            displacement=structural_displacement(source, graph, root.lineage, lineage),
        )
        return row

    with ExecutorMeter(None).instrument() as meter:

        def cached_law(graph):
            nonlocal law_seconds
            key = exact_graph_key(graph)
            if key not in laws:
                if perf_counter() - started >= config.stop_seconds:
                    raise ContinuationBudgetExceeded("administrative_elapsed_stop")
                progress.update(
                    phase=meter.phase,
                    laws_completed=len(laws),
                    executor_calls=meter.calls,
                    rollouts_completed=planner.work.rollouts_completed,
                    rollouts_started=planner.work.rollouts_started,
                    committed_edits=sum(e["stage"] == "how" for e in path),
                )
                before = perf_counter()
                laws[key] = enumerate_law(graph)
                elapsed = perf_counter() - before
                law_seconds += elapsed
                families, actions, probabilities = laws[key]
                save(
                    f"laws/{identity(encode_state(graph))}",
                    {
                        "source": encode_state(graph),
                        "marks": [
                            encode_action(f, a) for f, a in zip(families, actions, strict=True)
                        ],
                        "probabilities": list(probabilities),
                        "seconds": elapsed,
                    },
                )
            return laws[key]

        def terminal(n):
            value = objective.evaluate(n.graph)
            return value["value"] if n.budget == 0 or value["exact_hit"] else None

        kernel = OptionContinuationKernel(
            cached_law, system, max_executor_applications=None, product_gate=EXECUTABLE_PRODUCT_GATE
        )
        hierarchy = MolecularHierarchy(kernel, lazy_applicability=True)
        planner = TaskSearch(
            hierarchy.row,
            terminal,
            MolecularSearchState.key,
            snapshot_id=objective.snapshot,
            seed=int(seeds[0].generate_state(1)[0]),
            max_rows=None,
            max_terminals=256,
            max_rollouts=config.max_rollouts,
            max_path_steps=3 * config.primitive_budget,
            reference_draw=hierarchy.sample_reference,
            endpoint=lambda n: objective.evaluate(n.graph)["value"] if n.stage == "where" else None,
        )
        try:
            for event_index in range(3 * config.primitive_budget):
                if objective.evaluate(node.graph)["exact_hit"]:
                    status = "target_recovered"
                    break
                if node.budget == 0:
                    break
                meter.phase = "planning"
                planner.plan(node, config.initial_rollouts if not path else 1)
                meter.phase = "committed_decision"
                decision, row = planner.decision(node), planner.row(node)
                if not row.successors:
                    status = "no_admissible_action"
                    break
                index = int(acting_rng.choice(len(row.successors), p=decision["probabilities"]))
                following = row.successors[index]
                event = {
                    "event": event_index,
                    "stage": node.stage,
                    "budget_before": node.budget,
                    "source": encode_state(node.graph),
                    "product": encode_state(following.graph),
                    "selected_index": index,
                    "selected_label": row.labels[index],
                    "decision": decision,
                }
                if node.stage == "where":
                    event.update(
                        region_atoms=sorted(following.region.atoms),
                        r_release=following.region.released_fraction,
                        interface=following.region.interface,
                    )
                elif node.stage == "what":
                    event["option_initial"] = state_payload(following.active)
                else:
                    mark = kernel.marks(node.active)[index]
                    meter.phase = "selected_replay"
                    verify_transition(node, following, mark, system)
                    event.update(
                        mark=encode_action(*mark),
                        option_source=state_payload(node.active),
                        option_product=state_payload(kernel.row(node.active).successors[index]),
                        option_completed=following.stage == "where",
                        endpoint=describe(following.graph, following.lineage),
                    )
                path.append(event)
                node = following
                save(
                    f"decisions/{event_index:03d}",
                    {
                        **event,
                        "acting_rng_state": acting_rng.bit_generator.state,
                        "planner": planner.receipt(),
                    },
                )
                progress.update(
                    phase="decision_saved",
                    last_event=event_index,
                    target_similarity=objective.evaluate(node.graph)["similarity"],
                )
                print(
                    f"target_recovery event={event_index} stage={event['stage']} "
                    f"edits={config.primitive_budget - node.budget} "
                    f"similarity={progress['target_similarity']:.4f} "
                    f"rollouts={planner.work.rollouts_completed}/{planner.work.rollouts_started}",
                    flush=True,
                )
        except ContinuationBudgetExceeded as error:
            status = str(error)
        if objective.evaluate(node.graph)["exact_hit"]:
            status = "target_recovered"
        committed = [e["endpoint"] for e in path if e["stage"] == "how"]
        result = {
            "schema_version": "t4_target_recovery_v1",
            "status": status,
            "target": objective.target,
            "objective_snapshot": objective.snapshot,
            "configuration": asdict(config),
            "source": source_score,
            "final": describe(node.graph, node.lineage),
            "best_committed": max([source_score, *committed], key=lambda r: r["value"]),
            "committed_edits": len(committed),
            "exact_replays": len(committed),
            "path": path,
            "option_counts": dict(
                sorted(Counter(e["selected_label"] for e in path if e["stage"] == "what").items())
            ),
            "target_evaluations": [objective.cache[k] for k in sorted(objective.cache)],
            "planning_or_scoring_target_encounter": any(
                r["exact_hit"] for r in objective.cache.values()
            ),
            "planner": planner.receipt(),
            "kernel_work": asdict(kernel.work),
            "law_enumerations": len(laws),
            "law_seconds": law_seconds,
            "executor_calls": meter.calls,
            "executor_attempts": meter.attempts,
            "new_oracle_calls": 0,
            "automatic_docking": False,
            "seconds": perf_counter() - started,
            "software": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "rdkit": rdBase.rdkitVersion,
            },
            "scoring_asset_sha256": {
                name: sha256_file(Path(sascorer.__file__).parent / name)
                for name in ("sascorer.py", "fpscores.pkl.gz")
            },
        }
        save("search", result)
    return result


def run_remote(task, repo_root, artifact_root, volume, runtime_factory, validate_revision):
    from compose_v4.experiments.production_successor_kernel import enumerate_factorized_marked_law

    contract_path = f"configs/{KIND}.json"
    contract = json.loads((repo_root / contract_path).read_text())

    def runner(actual_task, _prepare, _dock, output, *, commit, progress):
        if contract["compute"]["oracle_call_limit"] != 0 or rdBase.rdkitVersion != "2024.03.5":
            raise ValueError(
                "target recovery requires zero docking and the pinned chemistry runtime"
            )
        if (output / "search.json").exists():
            return unseal(output / "search.json")
        if (output / "started.json").exists():
            raise ValueError("interrupted target recovery requires review, not automatic replay")
        source_path = artifact_root / contract["source"]["path"]
        verify_file(source_path, contract["source"]["sha256"])
        warm = unseal(source_path)
        source = decode_state(warm["archive"][0]["state"])
        if (
            warm["oracle_attempts"] != 51
            or canonical_state_key(source) != contract["source"]["smiles"]
        ):
            raise ValueError("target recovery source/archive identity mismatch")
        runtime = runtime_factory()

        def law(graph):
            row = enumerate_factorized_marked_law(runtime["model"], graph, 0.5)
            return (
                tuple(m.executor_rule_name for m in row.marks),
                tuple(m.action for m in row.marks),
                tuple(m.probability for m in row.marks),
            )

        def save(name, payload):
            seal(output / f"{name}.json", payload)
            commit()

        save("started", {"task": actual_task, "contract_sha256": task["contract_sha256"]})
        return run_recovery(
            source,
            contract["target"]["smiles"],
            law,
            runtime["system"],
            config=RecoveryConfig(**contract["recovery"]),
            save=save,
            progress=progress,
        )

    volume.reload()
    return common_remote(
        task,
        repo_root,
        artifact_root,
        volume,
        runtime_factory,
        validate_revision,
        None,
        None,
        contract_path=contract_path,
        run_kind=KIND,
        runner=runner,
    )
