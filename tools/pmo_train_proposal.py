"""Prepare reusable noise-contrastive decisions, then fit a frozen proposal policy."""

import argparse
import json
import math
import subprocess
from collections import Counter
from pathlib import Path
from time import perf_counter

import numpy as np

from compose_v4.control.docking_value import identity
from compose_v4.control.learned_proposal import RECIPE, features, fit
from compose_v4.control.molecular_search_codec import decode_search_state
from compose_v4.control.option_continuation import EXECUTABLE_PRODUCT_GATE, OptionContinuationKernel
from compose_v4.experiments.continuation_profile import publish_json, sha256_file, verify_file
from compose_v4.experiments.inference_package import software
from compose_v4.experiments.t4_macro_beam import replay
from compose_v4.experiments.t4_matched_pilot import unseal
from compose_v4.rewrite import action_codec, action_codec_v4
from compose_v4.rewrite.kernel import canonical_state_key, editing_v2_semantic_rewrite_system
from compose_v4.rewrite.trace_shard import encode_state

ROOT = Path(__file__).resolve().parents[1]


def prepare(manifest_path, output):
    manifest = json.loads(manifest_path.read_text())
    cache, started = manifest_path.parent, perf_counter()
    records, census, used_inputs = [], [], {str(manifest_path): sha256_file(manifest_path)}
    expected = None
    for run in manifest["runs"]:
        inputs = run["configuration"]["expected_input_sha256"]
        if expected is not None and expected != inputs:
            raise ValueError("training runs use different scientific inputs")
        expected = inputs
        # Reuse production laws, not a new neural evaluator in the local runtime.
        for path, digest in run["image_revision"]["serialized_sources"].items():
            if path.startswith(("src/compose_v4/chem/", "src/compose_v4/rewrite/")):
                verify_file(ROOT / path, digest)
        result_path = Path(run["result_path"])
        verify_file(result_path, manifest["inputs"][str(result_path)])
        result = json.loads(result_path.read_text())
        prepared_path = ROOT / run["configuration"]["prepared"]["path"]
        verify_file(prepared_path, run["configuration"]["prepared"]["sha256"])
        old = json.loads(prepared_path.read_text())
        known = dict(old["observed"])
        for row in result["oracle_rows"]:
            if row["smiles"] in known and known[row["smiles"]] != row["score"]:
                raise ValueError("conflicting historical labels")
            known[row["smiles"]] = row["score"]
        for proposal in run["proposals"]:
            for key, label in (("smiles", "score"), ("parent_smiles", "parent_score")):
                if known[proposal[key]] != proposal[label]:
                    raise ValueError("proposal reward lacks charged oracle evidence")
            path = cache / run["prefix"] / f"{proposal['id']}.json"
            verify_file(path, manifest["files"][str(path.relative_to(cache))]["cached_sha256"])
            record = unseal(path)
            if record["status"] != "complete" or record["candidate"]["node"] != proposal["node"]:
                raise ValueError("training trace is not the scored exact endpoint")
            records.append((run, proposal, record, path))
        census.append(
            {
                "run": run["prefix"],
                "scored_options": len(run["proposals"]),
                "new_oracle_calls_in_original_run": run["new_oracle_calls"],
                "failed_options": sum(
                    a["status"] != "complete" for w in result["workers"] for a in w["attempts"]
                ),
                "historical_controller_differences": {
                    p: {"historical": h, "current": sha256_file(ROOT / p)}
                    for p, h in run["image_revision"]["serialized_sources"].items()
                    if p.startswith("src/compose_v4/control/")
                    and (ROOT / p).exists()
                    and sha256_file(ROOT / p) != h
                },
            }
        )
    parents = Counter(identity(r[2]["source"]["graph"]) for r in records)
    examples, options = [], Counter()
    system = editing_v2_semantic_rewrite_system()
    for at, (run, proposal, record, path) in enumerate(records):
        primitives = replay(record["events"], system)
        if primitives != proposal["primitive_count"]:
            raise ValueError("scored option primitive count disagrees with exact replay")
        worker = path.parent.parent
        laws = {}

        def law(graph, laws=laws, worker=worker):
            key = identity(encode_state(graph))
            if key not in laws:
                law_path = worker / f"laws/{key}.json"
                rel = str(law_path.relative_to(cache))
                verify_file(law_path, manifest["files"][rel]["cached_sha256"])
                saved = unseal(law_path)
                if saved["source"] != encode_state(graph):
                    raise ValueError("law does not bind exact primitive source")
                pairs = [
                    (action_codec_v4 if m["schema_version"] == 4 else action_codec).decode_action(m)
                    for m in saved["marks"]
                ]
                p = np.asarray(saved["probabilities"])
                if (
                    len(p) != len(pairs)
                    or not np.isfinite(p).all()
                    or np.any(p < 0)
                    or not np.isclose(p.sum(), 1)
                ):
                    raise ValueError("invalid saved reference measure")
                laws[key] = tuple(f for f, _ in pairs), tuple(a for _, a in pairs), tuple(p)
            return laws[key]

        kernel = OptionContinuationKernel(
            law, system, max_executor_applications=None, product_gate=EXECUTABLE_PRODUCT_GATE
        )
        parent = identity(record["source"]["graph"])
        weight = (
            math.exp(RECIPE["return_beta"] * (proposal["score"] - proposal["parent_score"]))
            / parents[parent]
        )
        option = proposal["bundle"]["option"]
        options[option] += 1
        for step, event in enumerate(record["events"]):
            node, following = (
                decode_search_state(event["source"]),
                decode_search_state(event["product"]),
            )
            if node.stage == "where":
                continue
            rng = np.random.default_rng(np.random.SeedSequence([RECIPE["seed"], at, step]))
            rows = [features(node, option, following.graph)]
            noise = []
            if node.stage == "what":
                reference = event["allocation"]["reference"]
                if (
                    reference != event["allocation"]["probabilities"]
                    or event["labels"][event["selected"]] != option
                ):
                    raise ValueError("WHAT training decision was not drawn from the declared base")
                for index in rng.choice(len(reference), size=RECIPE["noise_samples"], p=reference):
                    label = event["labels"][int(index)]
                    rows.append(features(node, label, node.graph))
                    noise.append(label)
            else:
                lazy = kernel.lazy_row(node.active)
                for _ in range(RECIPE["noise_samples"]):
                    child = lazy.sample(rng)
                    if child is None:
                        raise ValueError("observed legal primitive has an empty replayed proposal")
                    rows.append(features(node, option, child.graph))
                    noise.append(canonical_state_key(child.graph))
            examples.append(
                {
                    "trajectory": f"{run['prefix']}/{proposal['id']}",
                    "parent": parent,
                    "step": step,
                    "stage": node.stage,
                    "option": option,
                    "endpoint_score": proposal["score"],
                    "parent_score": proposal["parent_score"],
                    "weight": weight / 2 / (primitives if node.stage == "how" else 1),
                    "features": rows,
                    "noise_identities": noise,
                }
            )
        if (at + 1) % 10 == 0:
            print(
                f"prepared {at + 1}/{len(records)} scored options, {len(examples)} decisions",
                flush=True,
            )
    data = {
        "schema_version": "proposal_decisions_v1",
        "recipe": RECIPE,
        "examples": examples,
        "input_hashes": used_inputs,
        "census": census,
        "option_counts": dict(options),
        "parents": len(parents),
        "scored_options": len(records),
        "new_oracle_calls": 0,
        "new_neural_evaluations": 0,
        "seconds": perf_counter() - started,
        "software": software(),
        "producer_sha256": sha256_file(Path(__file__)),
        "feature_source_sha256": sha256_file(ROOT / "src/compose_v4/control/learned_proposal.py"),
        "current_noise_kernel_sha256": {
            str(p.relative_to(ROOT)): sha256_file(p)
            for p in sorted((ROOT / "src/compose_v4/control").glob("*.py"))
        },
        "code_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "split": "all listed historical options are training; future comparison labels excluded by chronological lock",
        "hardware": {"device": "cpu", "threads": 1, "dtype": "float64"},
    }
    publish_json(output, data)
    return {k: v for k, v in data.items() if k != "examples"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "fit"))
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.mode == "prepare":
        print(json.dumps(prepare(args.input, args.output)))
    else:
        started = perf_counter()
        data = json.loads(args.input.read_text())
        if data["recipe"] != RECIPE:
            raise ValueError("prepared policy recipe mismatch")
        model = fit(data["examples"], data_sha256=sha256_file(args.input))
        publish_json(args.output, model.payload)
        publish_json(
            args.output.with_suffix(".fit.json"),
            {
                "schema_version": "proposal_fit_receipt_v1",
                "model_sha256": sha256_file(args.output),
                "data_sha256": sha256_file(args.input),
                "seconds": perf_counter() - started,
                "recipe": RECIPE,
                "fit": model.payload["fit"],
                "software": software(),
                "training_only": True,
                "new_oracle_calls": 0,
            },
        )
        print(json.dumps(model.payload["fit"]))


if __name__ == "__main__":
    main()
