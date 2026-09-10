"""Bounded one-edit repair availability, not an optimizer or a value estimator."""

from __future__ import annotations

import json
import math
from collections import Counter
from functools import lru_cache
from time import perf_counter

import numpy as np
from rdkit import Chem, rdBase
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Contrib.SA_Score import sascorer

from compose_v4.control.docking_value import DockingValue
from compose_v4.control.graph_geometry import topology
from compose_v4.control.molecular_search_codec import decode_search_state
from compose_v4.experiments.continuation_profile import ExecutorMeter, encode_action, verify_file
from compose_v4.experiments.saved_marked_law import SavedMarkedLaw
from compose_v4.experiments.t4_endpoint_selection import acceptable_endpoint, calculate_properties
from compose_v4.experiments.t4_macro_beam import canonical_smiles
from compose_v4.experiments.t4_matched_pilot import run_remote as common_remote
from compose_v4.experiments.t4_matched_pilot import seal, unseal
from compose_v4.rewrite import action_codec, action_codec_v4
from compose_v4.rewrite.kernel import canonical_state_key
from compose_v4.rewrite.trace_shard import encode_state

KIND = "t4_repair_neighbors"
CONTRACT_PATH = f"configs/{KIND}.json"


def select_panel(candidates, count):
    """Select by existing violation, never by a successor result or winner."""
    if type(count) is not int or not 1 <= count <= 3:
        raise ValueError("repair diagnostic requires one to three roots")
    unique = {}
    for row in sorted(candidates, key=lambda r: (r["v"], r["smiles"], r["attempt_id"])):
        if not math.isfinite(row["v"]) or row["v"] < 0:
            raise ValueError("repair panel has an invalid constraint violation")
        if row["v"] > 0:
            unique.setdefault(row["smiles"], row)
    if len(unique) < count:
        raise ValueError("insufficient distinct ineligible roots for declared panel")
    return list(unique.values())[:count]


def enumerate_products(graph, law, system):
    """Execute all positive-mass marks. Do not compute a second successor law.

    The production marked-law provider owns admissibility and probabilities.
    Canonical deduplication here is a support census only: no molecular
    probabilities, guidance, family weighting or top-k are implemented here.
    """
    families, actions, probabilities = law(graph)
    if not len(families) == len(actions) == len(probabilities):
        raise ValueError("marked-law lengths differ")
    if len(probabilities) and (
        not all(math.isfinite(p) and p >= 0 for p in probabilities)
        or not np.isclose(sum(probabilities), 1.0, atol=1e-12)
    ):
        raise ValueError("invalid production marked-law probabilities")
    products, counts = {}, Counter()
    source_key = canonical_state_key(graph)
    for family, action, probability in zip(families, actions, probabilities, strict=True):
        if probability == 0:
            counts["zero_mass_marks"] += 1
            continue
        product = system.apply(graph, family, action)
        counts["positive_mass_marks"] += 1
        key = canonical_state_key(product)
        if key == source_key:
            counts["canonical_self_marks"] += 1
            continue
        row = products.setdefault(
            key,
            {
                "smiles": key,
                "state": encode_state(product),
                "witnesses": [],
                "topology": topology(product),
            },
        )
        row["witnesses"].append(encode_action(family, action))
    return {
        "source": encode_state(graph),
        "source_smiles": source_key,
        "counts": dict(sorted(counts.items())),
        "products": [products[k] for k in sorted(products)],
        "oracle_calls": 0,
        "scope": "complete positive-mass production mark support; no region or option restriction",
    }


def recovery_counts(rows, archive_smiles, incumbent):
    eligible = {r["smiles"] for r in rows if r["oracle_eligible"]}
    return {
        "unique_products": len({r["smiles"] for r in rows}),
        "benchmark_feasible": sum(r["v"] == 0 for r in rows),
        "eligible": len(eligible),
        "new_eligible": len(eligible - archive_smiles),
        "known_eligible": len(eligible & archive_smiles),
        "returns_to_incumbent": int(incumbent in eligible),
        "benchmark_failures": sum(r["v"] > 0 for r in rows),
        "medchem_only_failures": sum(r["v"] == 0 and not r["oracle_eligible"] for r in rows),
    }


def run_remote(task, repo_root, artifact_root, volume, runtime_factory, validate_revision):
    contract = json.loads((repo_root / CONTRACT_PATH).read_text())
    if contract["oracle_calls"] != 0 or rdBase.rdkitVersion != contract["required_rdkit"]:
        raise ValueError("repair support census requires zero docking and pinned RDKit")

    def runner(actual_task, _prepare, _dock, output, *, commit, progress):
        def save(name, payload):
            seal(output / f"{name}.json", payload)
            commit()

        def read(name):
            path = output / f"{name}.json"
            return unseal(path) if path.exists() else None

        if read("diagnosis") is not None:
            return read("diagnosis")
        for name in ("archive", "value_snapshot", "panel"):
            verify_file(artifact_root / contract[name]["path"], contract[name]["sha256"])
        archive = unseal(artifact_root / contract["archive"]["path"])
        snapshot = unseal(artifact_root / contract["value_snapshot"]["path"])
        source_lock = unseal(artifact_root / contract["panel"]["path"])
        model = DockingValue.from_payload(snapshot)
        if (
            archive["oracle_attempts"] != 51
            or model.payload["source_sha256"] != contract["archive"]["sha256"]
            or source_lock["value_snapshot_sha256"] != model.payload["snapshot_sha256"]
        ):
            raise ValueError("repair inputs disagree with the frozen 51-call archive")
        panel = select_panel(source_lock["candidates"], contract["panel"]["count"])
        save("panel_lock", {"selection": contract["panel"], "roots": panel})
        generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
        seed_fp = generator.GetFingerprint(Chem.MolFromSmiles(actual_task["smiles"]))

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
            return {**props, "oracle_eligible": acceptable_endpoint({"smiles": smiles, **props})}

        old = {canonical_smiles(r["smiles"]) for r in archive["archive"]}
        incumbent = min(
            (r for r in archive["archive"] if r["ds"] is not None and acceptable_endpoint(r)),
            key=lambda r: (r["ds"], r["smiles"]),
        )["smiles"]
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
        summaries = []
        started = perf_counter()
        with meter.instrument():
            for index, root in enumerate(panel):
                node = decode_search_state(root["node"])
                if node.stage != "where" or canonical_state_key(node.graph) != root["smiles"]:
                    raise ValueError("repair root is not the locked completed-option state")
                if score(root["smiles"]) != {k: root[k] for k in score(root["smiles"])}:
                    raise ValueError("locked root properties disagree with frozen evaluator")
                progress.update(phase="repair_enumeration", root_index=index, roots=len(panel))
                name = f"roots/{index:02d}"
                generation = read(f"{name}/generation_lock")
                if generation is None:
                    generation = enumerate_products(node.graph, law, runtime["system"])
                    save(f"{name}/generation_lock", generation)
                if generation["source"] != encode_state(node.graph):
                    raise ValueError("saved repair census has a different exact root")
                rows = read(f"{name}/scored_lock")
                if rows is None:
                    progress.update(phase="repair_post_lock_scoring", root_index=index)
                    rows = [{**r, **score(r["smiles"])} for r in generation["products"]]
                    predictions = model.predict([r["smiles"] for r in rows]) if rows else []
                    for row, prediction in zip(rows, predictions, strict=True):
                        row["predicted_docking"] = float(prediction)
                        row["in_prior_archive"] = row["smiles"] in old
                        # Exact replay of every representative eligible edge.
                        if row["oracle_eligible"]:
                            witness = row["witnesses"][0]
                            codec = (
                                action_codec_v4 if witness["schema_version"] == 4 else action_codec
                            )
                            family, action = codec.decode_action(witness)
                            replayed = runtime["system"].apply(node.graph, family, action)
                            if encode_state(replayed) != row["state"]:
                                raise ValueError("repair witness failed exact-state replay")
                    save(f"{name}/scored_lock", rows)
                summary = {
                    "root_index": index,
                    "root": root,
                    **recovery_counts(rows, old, incumbent),
                    "eligible_products": [r for r in rows if r["oracle_eligible"]],
                    "generation_counts": generation["counts"],
                }
                save(f"{name}/summary", summary)
                summaries.append(summary)
                progress.update(phase="root_complete", roots_complete=index + 1)
        save("executor_attempts", meter.attempts)
        result = {
            "schema_version": "t4_repair_neighbors_result_v1",
            "roots": summaries,
            "oracle_calls": 0,
            "winner_used": False,
            "proposal_seconds_this_invocation": perf_counter() - started,
            "law_work": law.counts,
            "executor_calls_this_invocation": meter.calls,
            "software": {"numpy": np.__version__, "rdkit": rdBase.rdkitVersion},
            "claim": "one-edit support availability only; not controller recovery, future-value calibration or observed docking improvement",
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
        run_kind=KIND,
        runner=runner,
    )
