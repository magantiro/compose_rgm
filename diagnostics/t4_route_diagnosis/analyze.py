"""Score a saved answer-known route; never generate, fit, replay, or dock."""

from __future__ import annotations

import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import numpy as np
from rdkit import Chem, rdBase
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Contrib.SA_Score import sascorer

from compose_v4.control.docking_value import DockingValue, identity, molecular_features
from compose_v4.experiments import t4_macro_beam
from compose_v4.experiments.continuation_profile import sha256_file, verify_file
from compose_v4.experiments.t4_endpoint_selection import acceptable_endpoint, calculate_properties
from compose_v4.rewrite.kernel import canonical_state_key
from compose_v4.rewrite.trace_shard import decode_state
from tools.ivg_winner_paths import ROOT as SOURCE_ROOT
from tools.ivg_winner_paths import implementation_closure, publish

ROOT = Path(__file__).resolve().parents[2]
INPUTS = {
    "diagnostics/t4_whole_ring_plan/result.json": "252dccee4785d3a8e972df30c4a075e7e98bdf02e120300990bcb1de9d21cee9",
    "diagnostics/t4_option_decision_audit/summary.json": "50e96c58a311e682ee9508368d7db0fa39e08216daa055237740bb88ab8d8d04",
    "diagnostics/t4_recovery_lookahead/attempt_2/value_snapshot.json": "5359cabfd241b427f5da6dd10ad3f5103be706e098e0f0c9b27a8bbb467c400e",
    "diagnostics/t4_no_similarity_penalty/attempt_1/generation_lock.json": "68c80f669e3e1f3f26991555a3ef3930ca00c11bf71ada1228e3bb45042727c7",
}


def flatten(source, stages):
    """Preserve exact coordinates and every primitive; do not infer missing states."""
    states, boundaries = [source], []
    for stage in stages:
        if stage["states"][0] != states[-1]:
            raise ValueError(f"{stage['name']}: broken exact-state stage continuity")
        count = stage["primitive_edits"]
        if count != len(stage["actions"]) or len(stage["states"]) != count + 1:
            raise ValueError(f"{stage['name']}: inconsistent state/action count")
        states.extend(stage["states"][1:])
        boundaries.append({"name": stage["name"], "step": len(states) - 1, "edits": count})
    return states, boundaries


def run():
    if rdBase.rdkitVersion != "2024.03.5":
        raise ValueError("route diagnosis requires RDKit 2024.03.5")
    started = perf_counter()
    inputs = dict(INPUTS)

    def read(path, *, sealed=False):
        if path not in inputs:
            raise ValueError(f"unbound input: {path}")
        verify_file(ROOT / path, inputs[path])
        value = json.loads((ROOT / path).read_text())
        if sealed:
            if identity(value["payload"]) != value["payload_sha256"]:
                raise ValueError(f"{path}: corrupt envelope")
            return value["payload"]
        return value

    plan, option_summary, snapshot, search = [
        read(path, sealed=index >= 2) for index, path in enumerate(INPUTS)
    ]
    model = DockingValue.from_payload(snapshot)
    for row, features in zip(snapshot["training_rows"], snapshot["features"], strict=True):
        if identity(molecular_features(row["smiles"])[1]) != identity(features):
            raise ValueError("frozen training features do not reproduce")
    attempts = [a for a in plan["attempts"] if a["status"] == "exact_winner"]
    if len(attempts) != 1:
        raise ValueError("expected one successful declared route")
    stages = attempts[0]["stages"]
    states, boundaries = flatten(plan["source_state"], stages)
    if len(states) != 22 or len(boundaries) != 5:
        raise ValueError("route census changed")
    smiles = [canonical_state_key(decode_state(s)) for s in states]
    if smiles[-1] != plan["target"]:
        raise ValueError("saved terminal state differs from known target")
    for stage, boundary in zip(stages, boundaries, strict=True):
        if smiles[boundary["step"]] != stage["endpoint"]:
            raise ValueError("stage endpoint metadata differs from exact state")
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    seed_fp = generator.GetFingerprint(Chem.MolFromSmiles(smiles[0]))
    config = json.loads((ROOT / "configs/t4_no_similarity_penalty.json").read_text())
    inputs["configs/t4_no_similarity_penalty.json"] = sha256_file(
        ROOT / "configs/t4_no_similarity_penalty.json"
    )
    policy = config["no_similarity_guidance"]

    def score(s):
        props = calculate_properties(
            Chem.MolFromSmiles(s),
            seed_fp=seed_fp,
            generator=generator,
            sa_scorer=sascorer.calculateScore,
            delta=0.4,
            qed_min=0.6,
            sa_max=4.0,
        )
        eligible = acceptable_endpoint({"smiles": s, **props})
        prediction = float(model.predict([s])[0])
        return {
            "smiles": s,
            **props,
            "oracle_eligible": eligible,
            "predicted_docking": prediction,
            "terminal_desirability": model.desirability(s, eligible),
            "graded_desirability": t4_macro_beam.recovery_desirability(
                props["v"], prediction, snapshot, policy["scale"]
            ),
            "no_similarity_desirability": t4_macro_beam.no_similarity_desirability(
                props, prediction, snapshot, policy
            ),
        }

    scored = [
        dict(step=i, state_sha256=identity(s), **score(smiles[i])) for i, s in enumerate(states)
    ]
    incumbent = score(canonical_state_key(decode_state(search["root"]["graph"])))
    cases = []
    for relative, expected in option_summary["input_sha256"].items():
        path = "diagnostics/t4_option_decision_audit/attempt_1/" + relative
        inputs[path] = expected
        verify_file(ROOT / path, expected)
    for i, case in enumerate(option_summary["cases"]):
        raw = read(f"diagnostics/t4_option_decision_audit/attempt_1/case_{i}/result.json")
        for point in [raw["source"], raw["known_continuation"], *raw["products"]]:
            if (
                abs(score(point["smiles"])["predicted_docking"] - point["predicted_docking"])
                > 1e-12
            ):
                raise ValueError("saved option prediction does not match current frozen guide")
        cases.append(
            {
                key: case[key]
                for key in (
                    "case_index",
                    "option",
                    "focal_option_balanced_prior",
                    "focal_attempts",
                    "focal_known_canonical_matches",
                    "known_rank_among_unique_eligible",
                    "known_path_support",
                )
            }
        )
    result = {
        "schema_version": "t4_route_diagnosis_v1",
        "input_sha256": inputs,
        "analysis_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "analysis_sha256": sha256_file(Path(__file__)),
        "source_root": str(SOURCE_ROOT),
        "source_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=SOURCE_ROOT, text=True
        ).strip(),
        "source_closure": implementation_closure(Path(t4_macro_beam.__file__)),
        "publication_helper_sha256": sha256_file(SOURCE_ROOT / "tools/ivg_winner_paths.py"),
        "scoring_assets": {
            n: sha256_file(Path(sascorer.__file__).parent / n)
            for n in ("sascorer.py", "fpscores.pkl.gz")
        },
        "configuration": {
            "similarity_floor": 0.4,
            "qed_min": 0.6,
            "sa_max": 4.0,
            "soft_policy": policy,
            "seed": None,
            "randomness": "none",
        },
        "software": {
            "python": platform.python_version(),
            "rdkit": rdBase.rdkitVersion,
            "numpy": np.__version__,
        },
        "hardware": {
            "platform": platform.platform(),
            "workers": 1,
            "accelerator": None,
            "precision": "float64 scoring",
        },
        "split_identity": "answer-known inspected development route; no fitting or held-out claim",
        "route": scored,
        "boundaries": boundaries,
        "incumbent": incumbent,
        "same_source_as_current_search": smiles[0] == incumbent["smiles"],
        "winner_prediction_minus_incumbent": scored[-1]["predicted_docking"]
        - incumbent["predicted_docking"],
        "ineligible_route_steps": [s["step"] for s in scored if not s["oracle_eligible"]],
        "conditional_option_evidence": cases,
        "new_oracle_calls": 0,
        "learned_law_calls": 0,
        "new_executor_replays": 0,
        "exclusions": [],
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "seconds_including_input_checks": perf_counter() - started,
        "limitations": [
            "21 edits is a known witness, not shortest-path proof",
            "conditional focal tests do not measure end-to-end route probability",
            "surrogate predictions are not observed docking or calibrated future values",
            "original-seed route is not a mapped route from the current warm root",
            "no actual selection probabilities or ranks computed for unsampled route states",
        ],
    }
    return result


if __name__ == "__main__":
    destination = Path(__file__).with_name("result.json")
    if destination.exists():
        raise ValueError("refusing to overwrite a banked diagnosis")
    result = run()
    publish(destination, result)
    print(
        json.dumps(
            {
                "boundaries": [
                    result["route"][0],
                    *[result["route"][b["step"]] for b in result["boundaries"]],
                ],
                "incumbent": result["incumbent"],
                "ineligible_route_steps": result["ineligible_route_steps"],
                "seconds": result["seconds_including_input_checks"],
            },
            indent=2,
        )
    )
