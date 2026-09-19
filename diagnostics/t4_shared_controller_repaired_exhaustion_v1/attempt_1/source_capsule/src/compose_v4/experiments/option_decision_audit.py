"""Small reference proposal locks followed by separate answer-known diagnosis."""

from __future__ import annotations

import json
from dataclasses import asdict
from time import perf_counter

import numpy as np
from rdkit import Chem, rdBase
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Contrib.SA_Score import sascorer

from compose_v4.chem.molecular_graph import NULL_IDX, is_element
from compose_v4.control.carbonyl_option import INSERT_RING_CARBONYL_OPTION, CarbonylProgress
from compose_v4.control.docking_value import DockingValue, identity
from compose_v4.control.fused_option import BUILD_FUSED_RING_OPTION, FusedProgress
from compose_v4.control.graph_geometry import topology
from compose_v4.control.option_continuation import (
    EXECUTABLE_PRODUCT_GATE,
    OptionContinuationKernel,
    OptionState,
    exact_graph_key,
)
from compose_v4.control.option_selector import (
    applicable_options,
    balanced_option_prior,
    option_horizon,
    retain_product_applicable_options,
)
from compose_v4.control.region_rewrite import Lineage, RewriteContext
from compose_v4.control.ring_program import RingProgress, default_ring_options, ring_spec
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
from compose_v4.rewrite import action_codec, action_codec_v4
from compose_v4.rewrite.kernel import canonical_state_key
from compose_v4.rewrite.trace_shard import decode_state, encode_state

CONTRACT_PATH = "configs/t4_option_decision_audit.json"
KIND = "t4_option_decision_audit"
FOCAL_OPTIONS = (
    "construct:pendant:6:6,0,0:aromatic:0",
    "construct:fused:6:6,0,0:nonaromatic:0",
    "add_carbonyl",
    INSERT_RING_CARBONYL_OPTION,
)


def initial(graph, option, bundle):
    real = frozenset(int(i) for i in np.flatnonzero(is_element(graph.atom_types)))
    return OptionState(
        graph,
        graph,
        RewriteContext(frozenset(), real, (), "multi", 0),
        Lineage.initial(real),
        option,
        0,
        option_horizon(option, 3),
        bundle,
        fused_progress=FusedProgress() if option == BUILD_FUSED_RING_OPTION else None,
        ring_progress=RingProgress() if ring_spec(option) is not None else None,
        carbonyl_progress=CarbonylProgress() if option == INSERT_RING_CARBONYL_OPTION else None,
    )


def generate_lock(graph, focal, kernel, *, seed, save, read, progress):
    """No target, saved continuation, or task-value input is accepted here."""
    existing = read("proposal_lock")
    if existing is not None:
        return existing
    families, actions, _ = kernel.enumerate_law(graph)
    options = applicable_options(
        families,
        range(len(actions)),
        include_fused=True,
        include_carbonyl=True,
        n_free_slots=min(40 - graph.n_real_atoms, int(np.sum(graph.atom_types == NULL_IDX))),
        ring_options=default_ring_options(),
    )
    states = {o: initial(graph, o, f"{seed}:{o}") for o in options}
    options = retain_product_applicable_options(
        options, lambda o: kernel.lazy_row(states[o]).has_product()
    )
    prior = balanced_option_prior(options)
    option_rng = np.random.default_rng(np.random.SeedSequence([seed, 0]))
    selected = [focal, focal] + [
        options[int(option_rng.choice(len(options), p=prior))] for _ in range(2)
    ]
    attempts = []
    from compose_v4.control.molecular_search_codec import decode_option

    for index, option in enumerate(selected):
        progress.update(phase="generate", attempt=index, option=option)
        name = f"attempts/{index:02}"
        saved = read(name)
        if saved is not None and saved["status"] != "running":
            attempts.append(saved)
            continue
        rng = np.random.default_rng(np.random.SeedSequence([seed, index + 1]))
        node = initial(graph, option, f"{seed}:{index}:{option}")
        trajectory = []
        if saved is not None:
            node = decode_option(saved["node"])
            trajectory = saved["trajectory"]
            rng.bit_generator.state = saved["rng_state"]
        status = "running"
        while node.remaining:
            progress["primitive_step"] = node.step
            child = kernel.lazy_row(node).sample(rng)
            if child is None:
                status = "support_dead_end"
                break
            node = child
            trajectory.append(state_payload(node))
            save(
                name,
                {
                    "option": option,
                    "status": status,
                    "node": state_payload(node),
                    "trajectory": trajectory,
                    "rng_state": rng.bit_generator.state,
                },
            )
        if not node.remaining:
            status = "complete"
        result = {
            "option": option,
            "stratum": "focal" if index < 2 else "balanced_prior",
            "status": status,
            "node": state_payload(node),
            "trajectory": trajectory,
            "rng_state": rng.bit_generator.state,
        }
        save(name, result)
        attempts.append(result)
    lock = {
        "schema_version": "option_decision_proposal_lock_v1",
        "source": encode_state(graph),
        "seed": seed,
        "focal_option": focal,
        "options": list(options),
        "option_prior": prior.tolist(),
        "attempts": attempts,
        "oracle_calls": 0,
        "winner_or_task_value_used": False,
    }
    lock = json.loads(json.dumps(lock, allow_nan=False))
    save("proposal_lock", lock)
    return lock


def witness_support(graph, option, states, kernel):
    """Aggregate descriptor branches of a supplied exact-state prefix, post-lock."""
    if len(states) != option_horizon(option, 3) + 1 or exact_graph_key(
        decode_state(states[0])
    ) != exact_graph_key(graph):
        raise ValueError("known prefix must start at the exact source and match the option horizon")
    frontier = [(initial(graph, option, "witness"), 1.0)]
    steps = []
    for step, payload in enumerate(states[1:]):
        expected = exact_graph_key(decode_state(payload))
        canonical_expected = canonical_state_key(decode_state(payload))
        row_successors, canonical_matches = 0, 0
        next_nodes = {}
        for node, prefix in frontier:
            row = kernel.row(node)
            row_successors += len(row.successors)
            for child, mass in zip(row.successors, row.probabilities, strict=True):
                canonical_matches += canonical_state_key(child.graph) == canonical_expected
                if exact_graph_key(child.graph) == expected:
                    key = child.key()
                    next_nodes[key] = (child, next_nodes.get(key, (None, 0.0))[1] + prefix * mass)
        frontier = list(next_nodes.values())
        steps.append(
            {
                "step": step + 1,
                "branches": len(frontier),
                "exact_prefix_probability": sum(p for _, p in frontier),
                "row_successors": row_successors,
                "canonical_matches_before_exact_slot_match": canonical_matches,
            }
        )
        if not frontier:
            break
    return {
        "steps": steps,
        "supported": bool(frontier) and all(not n.remaining for n, _ in frontier),
        "exact_path_probability": sum(p for _, p in frontier),
        "probability_scope": "specified exact-state prefix conditional on option and fully mutable region",
    }


def scorer(warm, source_sha256):
    model = DockingValue.fit(
        warm["archive"], before_round=warm["round"] + 1, source_sha256=source_sha256
    )
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    seed_fp = generator.GetFingerprint(Chem.MolFromSmiles(warm["archive"][0]["smiles"]))

    def describe(state):
        smiles = canonical_state_key(state)
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
            "topology": topology(state),
        }

    return model, describe


def run_remote(task, repo_root, artifact_root, volume, runtime_factory, validate_revision):
    contract = json.loads((repo_root / CONTRACT_PATH).read_text())
    case = task["case_index"]
    if type(case) is not int or not 0 <= case < 4 or contract["oracle_calls"] != 0:
        raise ValueError("only the four approved zero-oracle cases are permitted")
    if rdBase.rdkitVersion != "2024.03.5":
        raise ValueError("option diagnosis requires the pinned chemistry runtime")

    def runner(actual_task, _prepare, _dock, output, *, commit, progress):
        from compose_v4.experiments.production_successor_kernel import (
            enumerate_factorized_marked_law,
        )

        def save(name, payload):
            seal(output / f"{name}.json", payload)
            commit()

        def read(name):
            path = output / f"{name}.json"
            return unseal(path) if path.exists() else None

        if read("diagnosis") is not None:
            return read("diagnosis")
        paths = {
            "plan": repo_root / contract["plan"]["path"],
            "archive": artifact_root / contract["archive"]["path"],
        }
        for name, path in paths.items():
            verify_file(path, contract[name]["sha256"])
        plan = json.loads(paths["plan"].read_text())
        stage = plan["attempts"][0]["stages"][case + 1]
        source = decode_state(stage["states"][0])
        warm = unseal(paths["archive"])
        model, describe = scorer(warm, contract["archive"]["sha256"])
        save("task_snapshot", model.payload)
        runtime = runtime_factory()
        memory = {}
        counts = {"fresh_laws": 0, "cached_laws": 0, "law_seconds": 0.0}

        # Conservative containment check: all model/chemistry/executor/data
        # files plus evaluator dependencies must match the old serialized tree.
        old_root = (
            artifact_root
            / "t4_target_recovery/70611576f82d337385717acac425591ed6dd76fc21b017ab8ad59a6ac4949936"
        )
        old_launch = old_root / "launch.json"
        reuse = {"available": False, "matched_files": 0, "mismatched_files": []}
        if old_launch.exists():
            verify_file(
                old_launch, "de04263ee32d964b863649c492694272ca038b6ff254e6e16d1596a6935d9924"
            )
            sources = json.loads(old_launch.read_text())["image_revision"]["serialized_sources"]
            dependencies = {
                p: h
                for p, h in sources.items()
                if p.startswith(
                    tuple(f"src/compose_v4/{d}/" for d in ("chem", "model", "rewrite", "data"))
                )
                or p
                in (
                    "src/compose_v4/experiments/production_successor_kernel.py",
                    "src/compose_v4/experiments/factorized_mark_conditional.py",
                    "src/compose_v4/experiments/tracelet_conditional.py",
                    "src/compose_v4/experiments/successor_kernel.py",
                )
            }
            mismatched = [p for p, h in dependencies.items() if sha256_file(repo_root / p) != h]
            old_gate = json.loads((old_root / "runtime_gate.json").read_text())
            reuse = {
                "available": bool(dependencies)
                and not mismatched
                and old_gate["input_sha256"] == contract["expected_input_sha256"],
                "matched_files": len(dependencies) - len(mismatched),
                "mismatched_files": mismatched,
                "launch_sha256": sha256_file(old_launch),
                "used_law_sha256": {},
            }
        save("cache_inventory", reuse)

        def law(graph):
            key = identity(encode_state(graph))
            if key in memory:
                counts["cached_laws"] += 1
                return memory[key]
            name = f"laws/{key}"
            payload = read(name)
            old_path = old_root / f"laws/{key}.json"
            if payload is None and reuse["available"] and old_path.exists():
                payload = unseal(old_path)
                reuse["used_law_sha256"][str(old_path)] = sha256_file(old_path)
                save("cache_inventory", reuse)
                save(name, payload)
            if payload is None:
                progress.update(phase="law_enumeration", latest_law=key, **counts)
                started = perf_counter()
                row = enumerate_factorized_marked_law(runtime["model"], graph, 0.5)
                payload = {
                    "source": encode_state(graph),
                    "marks": [encode_action(m.executor_rule_name, m.action) for m in row.marks],
                    "probabilities": [m.probability for m in row.marks],
                }
                counts["fresh_laws"] += 1
                counts["law_seconds"] += perf_counter() - started
                save(name, payload)
            else:
                counts["cached_laws"] += 1
            if payload["source"] != encode_state(graph):
                raise ValueError("saved law differs from exact slot state")
            pairs = [
                (action_codec_v4 if m["schema_version"] == 4 else action_codec).decode_action(m)
                for m in payload["marks"]
            ]
            result = (
                tuple(f for f, _ in pairs),
                tuple(a for _, a in pairs),
                tuple(payload["probabilities"]),
            )
            memory[key] = result
            return result

        meter = ExecutorMeter(None)
        with meter.instrument():
            kernel = OptionContinuationKernel(
                law,
                runtime["system"],
                max_executor_applications=None,
                product_gate=EXECUTABLE_PRODUCT_GATE,
            )
            lock = generate_lock(
                source,
                FOCAL_OPTIONS[case],
                kernel,
                seed=1000 + case,
                save=save,
                read=read,
                progress=progress,
            )
            save("generation_executor_attempts", meter.attempts)
            progress.update(phase="post_lock_scoring")
            products = [
                {"attempt": i, "option": a["option"], **describe(decode_state(a["node"]["graph"]))}
                for i, a in enumerate(lock["attempts"])
                if a["status"] == "complete"
            ]
            save(
                "scored_lock", {"products": products, "snapshot": model.payload["snapshot_sha256"]}
            )
            progress.update(phase="post_lock_known_path_support")
            support = witness_support(source, FOCAL_OPTIONS[case], stage["states"], kernel)
            known = describe(decode_state(stage["states"][-1]))
            result = {
                "schema_version": "option_decision_diagnosis_v1",
                "case_index": case,
                "option": FOCAL_OPTIONS[case],
                "source": describe(source),
                "products": products,
                "known_continuation": known,
                "known_path_support": support,
                "known_exact_matches_in_samples": sum(
                    p["smiles"] == known["smiles"] for p in products
                ),
                "known_score_rank_in_augmented_pool": 1
                + sum(p["desirability"] > known["desirability"] for p in products),
                "known_score_ties_in_sample_pool": sum(
                    p["desirability"] == known["desirability"] for p in products
                ),
                "known_raw_prediction_rank_in_augmented_pool": 1
                + sum(p["predicted_docking"] < known["predicted_docking"] for p in products),
                "proposal_attempts": len(lock["attempts"]),
                "completed_attempts": len(products),
                "distinct_completed_molecules": len({p["smiles"] for p in products}),
                "kernel_work": asdict(kernel.work),
                "law_work": counts,
                "public_executor_calls": meter.calls,
                "oracle_calls": 0,
                "input_sha256": {name: sha256_file(path) for name, path in paths.items()},
                "limitations": [
                    "answer-known development starting states",
                    "fully mutable conditional region; WHERE not evaluated",
                    "two focal and two balanced-prior attempts per case; low statistical precision",
                    "surrogate rank is not observed docking rank",
                    "known route never used during generation",
                ],
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
