"""Bounded winner-initialized refinement with existing broad executable proposals."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import numpy as np
from rdkit import Chem, DataStructs, rdBase
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Contrib.SA_Score import sascorer

from compose_v4.control.docking_value import identity
from compose_v4.control.graph_geometry import structural_displacement, topology
from compose_v4.control.molecular_task_search import MolecularHierarchy, MolecularSearchState
from compose_v4.control.option_continuation import EXECUTABLE_PRODUCT_GATE, OptionContinuationKernel
from compose_v4.experiments.continuation_profile import publish_json, sha256_file, verify_file
from compose_v4.experiments.saved_marked_law import SavedMarkedLaw
from compose_v4.experiments.t4_endpoint_selection import acceptable_endpoint, calculate_properties
from compose_v4.experiments.t4_matched_pilot import _stamp, seal, unseal
from compose_v4.experiments.t4_repair_neighbors import enumerate_products
from compose_v4.experiments.whole_ring_plan import execute_program
from compose_v4.rewrite import action_codec, action_codec_v4
from compose_v4.rewrite.kernel import canonical_state_key
from compose_v4.rewrite.trace_shard import decode_state, encode_state

KIND = "t4_winner_refinement"
CONTRACT_PATH = f"configs/{KIND}.json"


def contract_at(root):
    contract = json.loads((root / CONTRACT_PATH).read_text())
    if (
        identity({k: v for k, v in contract.items() if k != "contract_sha256"})
        != contract["contract_sha256"]
    ):
        raise ValueError("winner-refinement contract hash mismatch")
    if (
        contract["compute"]["oracle_call_limit"] != 19
        or contract["search"]["candidate_limit"] != 14
    ):
        raise ValueError("winner-refinement authorization is limited to nineteen calls")
    return contract


def winner_at(root, contract):
    path = root / contract["winner"]["path"]
    verify_file(path, contract["winner"]["sha256"])
    report = json.loads(path.read_text())
    matches = [r for r in report["attempts"] if r["status"] == "exact_winner"]
    if len(matches) != 1:
        raise ValueError("expected one exact saved winner route")
    graph = decode_state(matches[0]["stages"][-1]["states"][-1])
    if canonical_state_key(graph) != report["target"]:
        raise ValueError("winner persistent state differs from canonical endpoint")
    return graph


def property_scorer(seed_smiles):
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    seed_fp = generator.GetFingerprint(Chem.MolFromSmiles(seed_smiles))

    def score(row):
        result = {
            **row,
            **calculate_properties(
                Chem.MolFromSmiles(row["smiles"]),
                seed_fp=seed_fp,
                generator=generator,
                sa_scorer=sascorer.calculateScore,
                delta=0.4,
                qed_min=0.6,
                sa_max=4.0,
            ),
        }
        return {**result, "oracle_eligible": acceptable_endpoint(result)}

    return score


def select_candidates(rows, winner_smiles, *, limit=14, multi_quota=7):
    """Canonical diversity within balanced groups; never read task predictions."""
    unique = {}
    for row in sorted(rows, key=lambda r: (r["smiles"], -r["primitive_count"], r["group"])):
        if row["oracle_eligible"] and row["smiles"] != winner_smiles:
            unique.setdefault(row["smiles"], row)
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    fps = {s: generator.GetFingerprint(Chem.MolFromSmiles(s)) for s in unique}
    winner_fp = generator.GetFingerprint(Chem.MolFromSmiles(winner_smiles))
    selected, used = [], set()

    def take(pool, count):
        groups = defaultdict(list)
        for row in pool:
            if row["smiles"] not in used:
                groups[row["group"]].append(row)
        while groups and count:
            for key in sorted(groups):
                references = [winner_fp, *(fps[r["smiles"]] for r in selected)]
                row = min(
                    groups[key],
                    key=lambda r: (
                        max(
                            DataStructs.TanimotoSimilarity(fps[r["smiles"]], fp)
                            for fp in references
                        ),
                        identity([20260912, r["smiles"]]),
                    ),
                )
                selected.append(row)
                used.add(row["smiles"])
                groups[key].remove(row)
                if not groups[key]:
                    del groups[key]
                count -= 1
                if not count:
                    break

    take([r for r in unique.values() if r["primitive_count"] > 1], min(multi_quota, limit))
    take(list(unique.values()), limit - len(selected))
    return selected


def sample_stream(graph, hierarchy, *, seed, options, budget, progress):
    """Record complete existing hierarchy draws, including exact primitive marks."""
    rng = np.random.default_rng(seed)
    origin = node = MolecularSearchState.start(graph, budget=budget, root_id=f"winner/{seed}")
    actions, states, bundles, products = [], [encode_state(graph)], [], []
    status = "complete"
    for boundary in range(options):
        if node.budget == 0:
            break
        for expected in ("where", "what"):
            if node.stage != expected:
                raise ValueError("option boundary hierarchy is inconsistent")
            before = node
            row = hierarchy.row(before)
            available = [
                i
                for i, label in enumerate(row.labels)
                if expected != "what" or label != "build_ring_system"
            ]
            if not available:
                return products, {"status": "selection_dead_end", "bundles": bundles}
            probabilities = row.reference[available]
            at = int(rng.choice(available, p=probabilities / probabilities.sum()))
            node = row.successors[at]
            if expected == "where":
                region = {
                    "key": repr(node.region.key()),
                    "r_release": node.region.released_fraction,
                    "probability": float(row.reference[at]),
                }
            else:
                bundles.append(
                    {
                        "boundary": boundary,
                        "region": region,
                        "option": node.active.option,
                        "probability": float(row.reference[at] / probabilities.sum()),
                        "disabled_program": "build_ring_system",
                    }
                )
        while node.stage == "how":
            progress.update(
                phase="option_execution",
                primitive=len(actions),
                boundary=boundary,
                option=node.active.option,
            )
            old = node
            lazy = hierarchy.kernel.lazy_row(old.active)
            active = lazy.sample(rng)
            if active is None:
                status = "primitive_dead_end"
                break
            indices = [
                i
                for branch in lazy.branches.values()
                for i, product in branch.products.items()
                if product is active
            ]
            if not indices:
                raise RuntimeError("sampled product lacks its production marked-law witness")
            families, marks, _ = hierarchy.kernel.enumerate_law(old.graph)
            from compose_v4.experiments.continuation_profile import encode_action

            actions.append(encode_action(families[indices[0]], marks[indices[0]]))
            states.append(encode_state(active.graph))
            node = hierarchy._successor(old, active)
        if status != "complete":
            break
        replayed, replay = execute_program(graph, actions)
        if replay["states"] != states or encode_state(replayed) != encode_state(node.graph):
            raise RuntimeError("sampled complete options failed exact persistent-slot replay")
        products.append(
            {
                "smiles": canonical_state_key(node.graph),
                "state": encode_state(node.graph),
                "group": "options/" + bundles[-1]["option"],
                "channel": "option_reference",
                "primitive_count": len(actions),
                "bundles": list(bundles),
                "trace": replay,
                "structural_change": structural_displacement(
                    graph, node.graph, origin.lineage, node.lineage
                ),
                "topology": topology(node.graph),
            }
        )
    return products, {"status": status, "bundles": bundles, "primitive_count": len(actions)}


def proposal_remote(task, root, artifacts, volume, runtime_factory, validate_revision):
    validate_revision(task["image_revision"])
    verify_file(root / "modal_apps/genmol_t4_opt_app.py", task["app_sha256"])
    verify_file(root / CONTRACT_PATH, task["contract_sha256"])
    contract = contract_at(root)
    index = task["unit"]
    if type(index) is not int or not 0 <= index <= contract["search"]["streams"]:
        raise ValueError("proposal unit outside the locked task census")
    if rdBase.rdkitVersion != contract["required_rdkit"]:
        raise ValueError("proposal chemistry version differs")
    volume.reload()
    output = artifacts / KIND / task["run_id"] / "proposals" / f"{index:02}"
    if (output / "result.json").exists():
        return unseal(output / "result.json")
    started = perf_counter()
    progress = {"unit": index, "phase": "initialization"}

    def save(name, payload):
        seal(output / f"{name}.json", payload)
        publish_json(output / "progress.json", {**progress, "updated_at_utc": _stamp()})
        volume.commit()

    def read(name):
        path = output / f"{name}.json"
        if not path.exists() and index:
            path = output.parent / "00" / f"{name}.json"
        return unseal(path) if path.exists() else None

    save("task", task)
    runtime = runtime_factory()
    for name, path in (
        ("r_theta_checkpoint", runtime["model_checkpoint"]),
        ("r_theta_run_paths", runtime["run_paths"]),
    ):
        verify_file(Path(path), contract["expected_input_sha256"][name])
    graph = winner_at(root, contract)
    law = SavedMarkedLaw(
        runtime["model"],
        output,
        save,
        read,
        repo_root=root,
        artifact_root=artifacts,
        contract=contract,
        progress=progress,
    )
    if index == 0:
        census = enumerate_products(graph, law, runtime["system"])
        products = []
        origin = MolecularSearchState.start(graph, budget=14, root_id="winner")
        for row in census["products"]:
            record = row["witnesses"][0]
            codec = action_codec_v4 if record["schema_version"] == 4 else action_codec
            family, action = codec.decode_action(record)
            product, trace = execute_program(graph, [record])
            if encode_state(product) != row["state"]:
                raise RuntimeError("census witness disagrees with persistent state")
            products.append(
                {
                    **row,
                    "trace": trace,
                    "channel": "primitive_census",
                    "group": "primitive/" + family,
                    "primitive_count": 1,
                    "structural_change": structural_displacement(
                        graph, product, origin.lineage, origin.lineage.observe(family, action)
                    ),
                }
            )
        detail = {"status": "complete", "census": census["counts"]}
    else:
        kernel = OptionContinuationKernel(
            law,
            runtime["system"],
            max_executor_applications=None,
            product_gate=EXECUTABLE_PRODUCT_GATE,
        )
        hierarchy = MolecularHierarchy(
            kernel, lazy_applicability=True, include_carbonyl_options=True
        )
        seed = int(np.random.SeedSequence([contract["search"]["seed"], index]).generate_state(1)[0])
        products, detail = sample_stream(
            graph,
            hierarchy,
            seed=seed,
            options=contract["search"]["options_per_stream"],
            budget=contract["search"]["primitive_budget"],
            progress=progress,
        )
        detail.update(seed=seed, kernel_work=asdict(kernel.work))
    seed_row = json.loads((root / "docs/GENMOL_T4_SEEDS.json").read_text())[0]
    score = property_scorer(seed_row["smiles"])
    result = {
        "schema_version": "t4_winner_refinement_proposal_v1",
        "unit": index,
        "products": [score(r) for r in products],
        "detail": detail,
        "law_work": law.counts,
        "seconds": perf_counter() - started,
        "oracle_calls": 0,
        "source": encode_state(graph),
        "task": task,
    }
    save("result", result)
    return result


def replicate_summary(winner_rows, candidate_rows):
    if (
        len(winner_rows) != 3
        or len(candidate_rows) != 3
        or any(r["ds"] is None for r in [*winner_rows, *candidate_rows])
    ):
        return {"status": "inconclusive_oracle_failure"}
    winner = np.asarray([r["ds"] for r in winner_rows])
    candidate = np.asarray([r["ds"] for r in candidate_rows])
    return {
        "status": "complete_development",
        "winner_scores": winner.tolist(),
        "candidate_scores": candidate.tolist(),
        "mean_difference": float((candidate - winner).mean()),
        "repeat_only_mean_difference": float((candidate[1:] - winner[1:]).mean()),
        "individual_differences": (candidate - winner).tolist(),
        "winner_range": [float(winner.min()), float(winner.max())],
        "candidate_range": [float(candidate.min()), float(candidate.max())],
        "interpretation": "three-repeat winner-initialized refinement; first replicate selected the candidate; not benchmark superiority",
    }


def driver_remote(task, root, artifacts, volume, validate_revision, propose, dock_many):
    from compose_v4.experiments.t4_matched_pilot import run_remote

    def runner(actual_task, _prepare, _dock, output, *, commit, progress):
        contract = contract_at(root)
        if (output / "completed.json").exists():
            return unseal(output / "completed.json")
        graph = winner_at(root, contract)
        score = property_scorer(actual_task["smiles"])
        winner = score(
            {
                "smiles": canonical_state_key(graph),
                "state": encode_state(graph),
                "role": "winner",
                "topology": topology(graph),
            }
        )
        if not winner["oracle_eligible"]:
            raise ValueError("published winner fails the original benchmark endpoint gate")

        def batch(stage, candidates):
            folder = artifacts / KIND / stage / task["run_id"]
            lock = {
                "schema_version": "t4_winner_refinement_lock_v1",
                "task": actual_task,
                "required_rdkit": contract["required_rdkit"],
                "take": candidates,
            }
            path = folder / "candidate_lock.json"
            if path.exists() and unseal(path) != lock:
                raise ValueError("winner-refinement lock changed after publication")
            seal(path, lock)
            digest = sha256_file(path)
            publish_json(
                folder / "docking_started.json",
                {"candidate_lock_sha256": digest, "maximum_attempts": len(candidates)},
            )
            commit()
            rows = list(
                dock_many(
                    [
                        {**task, "stage": stage, "index": i, "candidate_lock_sha256": digest}
                        for i in range(len(candidates))
                    ]
                )
            )
            rows.sort(key=lambda r: r["index"])
            if [r["index"] for r in rows] != list(range(len(candidates))):
                raise ValueError("missing or duplicated locked docking receipt")
            result = [{**candidate, **row} for candidate, row in zip(candidates, rows, strict=True)]
            seal(folder / "docked.json", result)
            commit()
            return result

        winner_rows = batch(
            "winner", [{**winner, "docking_seed": s} for s in contract["docking"]["seeds"]]
        )
        progress.update(phase="proposal_census", winner_scores=[r["ds"] for r in winner_rows])
        print(f"winner controls: {[r['ds'] for r in winner_rows]}", flush=True)
        census = list(propose([{**task, "unit": 0}]))
        progress.update(phase="option_proposals")
        streams = list(
            propose([{**task, "unit": i} for i in range(1, contract["search"]["streams"] + 1)])
        )
        preparations = sorted(census + streams, key=lambda r: r["unit"])
        pool = [r for unit in preparations for r in unit["products"]]
        selected = select_candidates(pool, winner["smiles"])
        seal(
            output / "selection.json",
            {
                "take": selected,
                "pool_count": len(pool),
                "eligible": sum(r["oracle_eligible"] for r in pool),
                "proposal_summaries": [
                    {k: v for k, v in p.items() if k != "products"} for p in preparations
                ],
            },
        )
        commit()
        print(
            f"locked {len(selected)} from {len(pool)} products; groups={dict(Counter(r['group'] for r in selected))}",
            flush=True,
        )
        progress.update(phase="candidate_docking", candidates=len(selected))
        docked = (
            batch(
                "discovery",
                [{**r, "docking_seed": contract["docking"]["seeds"][0]} for r in selected],
            )
            if selected
            else []
        )
        successful = [r for r in docked if r["ds"] is not None]
        best = min(successful, key=lambda r: (r["ds"], r["smiles"]), default=None)
        repeated = []
        if best is not None:
            progress.update(phase="repeat_best", best_score=best["ds"])
            repeats = [
                {**next(r for r in selected if r["smiles"] == best["smiles"]), "docking_seed": s}
                for s in contract["docking"]["seeds"][1:]
            ]
            repeated = batch("confirmation", repeats)
        result = {
            "schema_version": "t4_winner_refinement_result_v1",
            "status": "complete" if best is not None else "no_scored_candidates",
            "winner": winner_rows,
            "docked": docked,
            "best": best,
            "confirmation": repeated,
            "new_oracle_attempts": len(winner_rows) + len(docked) + len(repeated),
            "comparison": replicate_summary(winner_rows, [best, *repeated])
            if best
            else {"status": "no_scored_candidates"},
            "proposal_seconds": [p["seconds"] for p in preparations],
            "proposal_counts": {
                "all": len(pool),
                "unique": len({r["smiles"] for r in pool}),
                "selected": len(selected),
            },
            "historical_route_calls": 6,
            "benchmark_claim": False,
            "training_performed": False,
        }
        if result["new_oracle_attempts"] > contract["compute"]["oracle_call_limit"]:
            raise RuntimeError("oracle budget exceeded")
        seal(output / "completed.json", result)
        commit()
        return result

    volume.reload()
    return run_remote(
        task,
        root,
        artifacts,
        volume,
        None,
        validate_revision,
        None,
        None,
        contract_path=CONTRACT_PATH,
        run_kind=KIND,
        runner=runner,
    )


def dock_remote(task, root, artifacts, volume, validate_revision, dock):
    import shutil

    from compose_v4.experiments.t4_partial_docking import dock_saved_row

    stage = task["stage"]
    limits = {"winner": 3, "discovery": 14, "confirmation": 2}
    if stage not in limits:
        raise ValueError("unknown winner-refinement docking stage")
    contract = contract_at(root)
    verify_file(root / CONTRACT_PATH, task["contract_sha256"])
    volume.reload()
    folder = artifacts / KIND / stage / task["run_id"]
    verify_file(folder / "candidate_lock.json", task["candidate_lock_sha256"])
    lock = unseal(folder / "candidate_lock.json")
    index = task["index"]
    if type(index) is not int or not 0 <= index < len(lock["take"]) <= limits[stage]:
        raise ValueError("winner-refinement docking index outside stage allowance")
    candidate = lock["take"][index]
    expected_seed = contract["docking"]["seeds"][
        0 if stage == "discovery" else index + (stage == "confirmation")
    ]
    if candidate["docking_seed"] != expected_seed:
        raise ValueError("docking replicate seed differs from frozen schedule")

    def preserving_dock(smiles, tag):
        tag = f"{tag}_{stage}"
        value = dock(smiles, tag, expected_seed)
        destination = folder / "poses" / f"{index:02}"
        destination.mkdir(parents=True, exist_ok=True)
        hashes = {}
        for name in ("l.mol", "l.pdbqt", "o.pdbqt"):
            path = Path("/tmp") / tag / name
            if path.exists():
                shutil.copyfile(path, destination / name)
                hashes[name] = sha256_file(destination / name)
        publish_json(
            destination / "manifest.json",
            {"smiles": smiles, "docking_seed": expected_seed, "sha256": hashes},
        )
        return value

    return dock_saved_row(
        task,
        root,
        artifacts,
        volume,
        validate_revision,
        preserving_dock,
        run_kind=f"{KIND}/{stage}",
        batch_limit=limits[stage],
    )
