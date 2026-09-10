"""Winner-blind search over completed executable options; no docking interface."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from functools import lru_cache
from time import perf_counter

import numpy as np
from rdkit import Chem, rdBase
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Contrib.SA_Score import sascorer

from compose_v4.control.continuation import _tilted
from compose_v4.control.docking_value import DockingValue, identity, molecular_features
from compose_v4.control.graph_geometry import structural_displacement, topology
from compose_v4.control.molecular_search_codec import decode_search_state, encode_search_state
from compose_v4.control.molecular_task_search import MolecularHierarchy, MolecularSearchState
from compose_v4.control.option_continuation import EXECUTABLE_PRODUCT_GATE, OptionContinuationKernel
from compose_v4.experiments.continuation_profile import ExecutorMeter, verify_file
from compose_v4.experiments.t4_endpoint_selection import acceptable_endpoint, calculate_properties
from compose_v4.experiments.t4_matched_pilot import run_remote as common_remote
from compose_v4.experiments.t4_matched_pilot import seal, unseal
from compose_v4.rewrite import action_codec, action_codec_v4
from compose_v4.rewrite.kernel import canonical_state_key
from compose_v4.rewrite.trace_shard import decode_state, encode_state

KIND = "t4_macro_beam"
CONTRACT_PATH = f"configs/{KIND}.json"
ARMS = ("post_hoc", "guided")


@dataclass(frozen=True)
class BeamConfig:
    arm: str = "guided"
    root_index: int = 0
    depth: int = 4
    width: int = 3
    branches: int = 3
    primitive_budget: int = 44
    seed: int = 2000

    def __post_init__(self):
        if self.arm not in ARMS or self.root_index not in (0, 1):
            raise ValueError("beam requires a declared arm and root index")
        for name in ("depth", "width", "branches", "primitive_budget", "seed"):
            value = getattr(self, name)
            if type(value) is not int or value < (0 if name == "seed" else 1):
                raise ValueError(f"invalid beam {name}: {value!r}")
        if self.depth > 4 or self.width > 3 or self.branches > 3:
            raise ValueError("beam exceeds the bounded zero-oracle diagnostic")
        if self.primitive_budget < self.depth * 11:
            raise ValueError("bookkeeping budget must not truncate a registered option")


@lru_cache(maxsize=4096)
def fingerprint(smiles):
    return frozenset(molecular_features(smiles)[1][0])


def distance(a, b):
    x, y = fingerprint(a), fingerprint(b)
    return 1 - len(x & y) / len(x | y) if x or y else 0.0


def retain(candidates, config, rng, score):
    """Finite-pool retention, not a full-reference molecular transition law."""
    unique = {}
    for row in sorted(candidates, key=lambda r: r["attempt_id"]):
        unique.setdefault(row["smiles"], row)
    pool = list(unique.values())
    if len(pool) <= config.width:
        return pool, {"policy": "retain_all", "unique_pool": len(pool)}
    p = np.full(len(pool), 1 / len(pool))
    values = None
    q, eta = p, 0.0
    if config.arm == "guided":
        values = np.asarray([score(r["smiles"])["desirability"] for r in pool])
        core, eta, _ = _tilted(p, (1 + values) / 2, kappa=1.0, exploration=0.0)
        q = 0.9 * core + 0.1 * p
    kl = float(np.sum(q * np.log(q / p)))
    if kl > 1 + 1e-10 or not np.isclose(q.sum(), 1) or np.any(q < 0.1 * p - 1e-12):
        raise ValueError("empirical retention KL/floor invariant failed")
    chosen = [int(rng.choice(len(pool), p=q))]
    remaining = [i for i in range(len(pool)) if i not in chosen]
    if config.width >= 2:
        novel = max(
            remaining,
            key=lambda i: (
                min(distance(pool[i]["smiles"], pool[j]["smiles"]) for j in chosen),
                pool[i]["smiles"],
            ),
        )
        chosen.append(novel)
        remaining.remove(novel)
    while len(chosen) < config.width:
        index = int(rng.choice(remaining))
        chosen.append(index)
        remaining.remove(index)
    return [pool[i] for i in chosen], {
        "policy": "task_tilt_diversity_uniform"
        if config.arm == "guided"
        else "uniform_diversity_uniform",
        "pool": [r["attempt_id"] for r in pool],
        "first_slot_reference": p.tolist(),
        "first_slot_probabilities": q.tolist(),
        "first_slot_values": None if values is None else values.tolist(),
        "first_slot_kl_against_empirical_pool": kl,
        "eta": eta,
        "selected_indices": chosen,
    }


class WitnessIndex:
    def __init__(self, meter):
        self.meter, self.cursor, self.marks = meter, 0, {}

    def find(self, source, product):
        for row in self.meter.attempts[self.cursor :]:
            if row["status"] == "executed":
                self.marks.setdefault(
                    (identity(row["source"]), identity(row["product"])), row["mark"]
                )
        self.cursor = len(self.meter.attempts)
        key = (identity(encode_state(source)), identity(encode_state(product)))
        if key not in self.marks:
            raise ValueError("sampled molecular transition has no executed primitive witness")
        return self.marks[key]


def replay(events, system):
    count = 0
    for event in events:
        if "mark" not in event:
            continue
        mark = event["mark"]
        codec = action_codec_v4 if mark["schema_version"] == 4 else action_codec
        family, action = codec.decode_action(mark)
        product = system.apply(decode_state(event["source"]["graph"]), family, action)
        if encode_state(product) != event["product"]["graph"]:
            raise ValueError("completed-option primitive replay differs from exact product")
        count += 1
    return count


def run_search(root, hierarchy, *, config, score, save, read, meter, progress):
    started = perf_counter()
    signature = {"root": encode_search_state(root), "config": asdict(config)}
    existing = read("search_identity")
    if existing is not None and existing != signature:
        raise ValueError("beam resume changed source or configuration")
    if existing is None:
        save("search_identity", signature)
    locked = read("generation_lock")
    if locked is not None:
        return locked
    witnesses = WitnessIndex(meter)
    selection_rows = {}
    beam = [{"node": encode_search_state(root), "chain": []}]
    levels, all_attempts = [], []
    for depth in range(config.depth):
        stage = f"levels/{depth:02}"
        previous = read(f"{stage}/selection")
        if previous is not None:
            beam = previous["beam"]
            levels.append(previous)
            all_attempts.extend(previous["attempts"])
            continue
        candidates, attempts = [], []
        for parent_index, parent in enumerate(beam):
            origin = decode_search_state(parent["node"])
            for branch in range(config.branches):
                name = f"{stage}/attempt_{parent_index:02}_{branch:02}"
                progress.update(
                    phase="option_proposal", depth=depth, parent=parent_index, branch=branch
                )
                rng = np.random.default_rng(
                    np.random.SeedSequence(
                        [config.seed, config.root_index, depth, parent_index, branch]
                    )
                )
                record = read(name)
                if record is not None and record["status"] != "running":
                    if record["source"] != parent["node"]:
                        raise ValueError("resumed complete option has a different parent")
                    if record["status"] == "complete":
                        candidates.append(record["candidate"])
                    attempts.append(record)
                    continue
                if record is None:
                    node, events, bundle = origin, [], None
                else:
                    if record["source"] != parent["node"]:
                        raise ValueError("resumed option has a different parent")
                    node, events, bundle = (
                        decode_search_state(record["node"]),
                        record["events"],
                        record["bundle"],
                    )
                    rng.bit_generator.state = record["rng_state"]
                status = "running" if record is None else record["status"]
                while status == "running":
                    if node.stage == "where" and events:
                        status = "complete"
                        break
                    progress.update(stage=node.stage, primitive_depth=root.budget - node.budget)
                    event = {"source": encode_search_state(node)}
                    if node.stage in ("where", "what"):
                        if node.key() not in selection_rows:
                            selection_rows[node.key()] = hierarchy.row(node)
                        row = selection_rows[node.key()]
                        if not row.successors:
                            status = "support_dead_end"
                            break
                        selected = int(rng.choice(len(row.successors), p=row.reference))
                        following = row.successors[selected]
                        event.update(
                            labels=list(row.labels),
                            reference=row.reference.tolist(),
                            selected=selected,
                        )
                        if node.stage == "what":
                            bundle = {
                                "option": following.active.option,
                                "bundle_id": following.active.bundle_id,
                                "r_release": node.region.released_fraction,
                                "region": encode_search_state(node)["region"],
                            }
                    else:
                        following = hierarchy.sample_reference(node, rng)
                        if following is None:
                            status = "support_dead_end"
                            break
                        event["mark"] = witnesses.find(node.graph, following.graph)
                    event["product"] = encode_search_state(following)
                    events.append(event)
                    node = following
                    record = {
                        "attempt_id": name,
                        "source": parent["node"],
                        "node": encode_search_state(node),
                        "events": events,
                        "bundle": bundle,
                        "status": "running",
                        "rng_state": rng.bit_generator.state,
                    }
                    save(name, record)
                record = {
                    "attempt_id": name,
                    "source": parent["node"],
                    "node": encode_search_state(node),
                    "events": events,
                    "bundle": bundle,
                    "status": status,
                    "rng_state": rng.bit_generator.state,
                }
                if status == "complete":
                    if not events or node.stage != "where" or bundle is None:
                        raise ValueError("incomplete option reached beam retention")
                    record["replayed_primitives"] = replay(events, hierarchy.kernel.system)
                    candidate = {
                        "attempt_id": name,
                        "node": record["node"],
                        "chain": parent["chain"] + [name],
                        "smiles": canonical_state_key(node.graph),
                        "bundle": bundle,
                        "structural_change": structural_displacement(
                            origin.graph, node.graph, origin.lineage, node.lineage
                        ),
                        "cumulative_change": structural_displacement(
                            root.graph, node.graph, root.lineage, node.lineage
                        ),
                        "topology": topology(node.graph),
                    }
                    record["candidate"] = candidate
                    candidates.append(candidate)
                save(name, record)
                attempts.append(record)
        # All programs finish/abstain and are saved before any retention score.
        progress.update(phase="beam_retention", depth=depth, completed=len(candidates))
        rng = np.random.default_rng(
            np.random.SeedSequence([config.seed, config.root_index, depth, 991])
        )
        beam, decision = retain(candidates, config, rng, score)
        selection = {
            "depth": depth,
            "beam": beam,
            "attempts": attempts,
            "decision": decision,
            "rng_state": rng.bit_generator.state,
        }
        save(f"{stage}/selection", selection)
        levels.append(selection)
        all_attempts.extend(attempts)
        if not beam:
            break
    lock = {
        "schema_version": "t4_macro_beam_lock_v1",
        "config": asdict(config),
        "root": encode_search_state(root),
        "levels": levels,
        "attempts": all_attempts,
        "final_beam": beam,
        "oracle_calls": 0,
        "winner_used": False,
        "proposal_seconds_this_invocation": perf_counter() - started,
    }
    lock = json.loads(json.dumps(lock, allow_nan=False))
    save("generation_lock", lock)
    return lock


def run_remote(task, repo_root, artifact_root, volume, runtime_factory, validate_revision):
    contract = json.loads((repo_root / CONTRACT_PATH).read_text())
    case = task["case_index"]
    if type(case) is not int or not 0 <= case < 4 or contract["oracle_calls"] != 0:
        raise ValueError("macro beam permits only the four approved zero-oracle cases")
    if rdBase.rdkitVersion != contract["required_rdkit"]:
        raise ValueError("macro beam requires pinned RDKit")

    def runner(actual_task, _prepare, _dock, output, *, commit, progress):
        from compose_v4.experiments.saved_marked_law import SavedMarkedLaw

        def save(name, payload):
            seal(output / f"{name}.json", payload)
            commit()

        def read(name):
            path = output / f"{name}.json"
            return unseal(path) if path.exists() else None

        if read("diagnosis") is not None:
            return read("diagnosis")
        for name in ("archive", "value_snapshot"):
            verify_file(artifact_root / contract[name]["path"], contract[name]["sha256"])
        warm = unseal(artifact_root / contract["archive"]["path"])
        model = DockingValue.from_payload(
            unseal(artifact_root / contract["value_snapshot"]["path"])
        )
        if (
            model.payload["source_sha256"] != contract["archive"]["sha256"]
            or warm["oracle_attempts"] != 51
        ):
            raise ValueError(
                "beam requires the complete frozen 51-call archive and its value snapshot"
            )
        generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
        seed_fp = generator.GetFingerprint(Chem.MolFromSmiles(warm["archive"][0]["smiles"]))

        @lru_cache(maxsize=4096)
        def score(smiles):
            props = calculate_properties(
                Chem.MolFromSmiles(smiles),
                seed_fp=seed_fp,
                generator=generator,
                sa_scorer=sascorer.calculateScore,
                delta=0.4,
                qed_min=0.6,
                sa_max=4.0,
            )
            eligible = acceptable_endpoint({"smiles": smiles, **props})
            return {
                "smiles": smiles,
                **props,
                "oracle_eligible": eligible,
                "predicted_docking": float(model.predict([smiles])[0]),
                "desirability": model.desirability(smiles, eligible),
            }

        # Root selection uses observed task labels, not predictions or winners.
        eligible = [r for r in warm["archive"] if r["ds"] is not None and acceptable_endpoint(r)]
        roots = [warm["archive"][0], min(eligible, key=lambda r: (r["ds"], r["smiles"]))]
        config = BeamConfig(arm=ARMS[case // 2], root_index=case % 2, **contract["search"])
        root_row = roots[config.root_index]
        graph = decode_state(root_row["state"])
        if canonical_state_key(graph) != root_row["smiles"]:
            raise ValueError("beam root exact state differs from canonical metadata")
        root = MolecularSearchState.start(
            graph, budget=config.primitive_budget, root_id=f"beam-root-{config.root_index}"
        )
        runtime = runtime_factory()
        law = SavedMarkedLaw(
            runtime["model"],
            output,
            save,
            read,
            repo_root=repo_root,
            artifact_root=artifact_root,
            contract=contract,
            progress=progress,
        )
        meter = ExecutorMeter(None)
        with meter.instrument():
            kernel = OptionContinuationKernel(
                law,
                runtime["system"],
                max_executor_applications=None,
                product_gate=EXECUTABLE_PRODUCT_GATE,
            )
            hierarchy = MolecularHierarchy(
                kernel, lazy_applicability=True, include_carbonyl_options=True
            )
            lock = run_search(
                root,
                hierarchy,
                config=config,
                score=score,
                save=save,
                read=read,
                meter=meter,
                progress=progress,
            )
            save("executor_attempts", meter.attempts)
        progress.update(phase="post_lock_scoring")
        old = {r["smiles"] for r in warm["archive"]}
        candidates = [
            {
                **a["candidate"],
                **score(a["candidate"]["smiles"]),
                "in_prior_archive": a["candidate"]["smiles"] in old,
            }
            for a in lock["attempts"]
            if a["status"] == "complete"
        ]
        save(
            "scored_lock",
            {"candidates": candidates, "value_snapshot_sha256": model.payload["snapshot_sha256"]},
        )
        result = {
            "schema_version": "t4_macro_beam_diagnosis_v1",
            "case_index": case,
            "config": asdict(config),
            "source": root_row,
            "source_prediction": score(root_row["smiles"]),
            "value_snapshot_sha256": model.payload["snapshot_sha256"],
            "candidates": candidates,
            "final_chains": [r["chain"] for r in lock["final_beam"]],
            "attempts": len(lock["attempts"]),
            "completed": len(candidates),
            "unique_completed": len({r["smiles"] for r in candidates}),
            "unique_new_eligible": len(
                {
                    r["smiles"]
                    for r in candidates
                    if r["oracle_eligible"] and not r["in_prior_archive"]
                }
            ),
            "proposal_seconds_this_invocation": lock["proposal_seconds_this_invocation"],
            "law_work": law.counts,
            "public_executor_calls_this_invocation": meter.calls,
            "kernel_work": asdict(kernel.work),
            "replay_verified": True,
            "oracle_calls": 0,
            "software": {"numpy": np.__version__, "rdkit": rdBase.rdkitVersion},
            "claim": "winner-blind development search; no observed docking improvement or generalization claim",
        }
        save("diagnosis", result)
        return result

    return common_remote(
        task,
        repo_root,
        artifact_root,
        volume,
        runtime_factory,
        validate_revision,
        None,
        None,
        contract_path=CONTRACT_PATH,
        run_kind=f"{KIND}/case_{case}",
        runner=runner,
    )
