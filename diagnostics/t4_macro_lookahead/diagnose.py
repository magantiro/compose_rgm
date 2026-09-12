"""Read-only scientific reduction of the completed episode, no molecular search.

Winner structure is used only in retrospective queries, never as a run input.
Run against the clean source recorded in the episode and pinned chemistry.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import numpy as np
from rdkit import Chem, RDLogger, rdBase

from compose_v4.control.docking_value import DockingValue, identity
from compose_v4.experiments.t4_macro_lookahead import planning_scorer
from compose_v4.experiments.t4_matched_pilot import unseal
from compose_v4.gates.med_chem_gate import validity_reasons

RUN = "8776743bbb813429663f339f60cbce587e59d427248dfaee80020014b11a6b28"
ROOT = Path(__file__).resolve().parents[2]
RUN_ROOT = Path(__file__).resolve().parent / RUN
SOURCE = Path(sys.modules[DockingValue.__module__].__file__).resolve().parents[3]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    started = perf_counter()
    RDLogger.DisableLog("rdApp.warning")
    if rdBase.rdkitVersion != "2024.03.5":
        raise ValueError("requires episode-matched RDKit 2024.03.5")
    inputs = {}

    def read(path, sealed=False):
        inputs[str(path.relative_to(ROOT))] = sha(path)
        return unseal(path) if sealed else json.loads(path.read_text())

    receipt = read(RUN_ROOT / "collections/20260910T150234112228Z.json")
    for name, digest in receipt["sha256"].items():
        assert sha(RUN_ROOT / name) == digest, name
    launch = read(RUN_ROOT / "launch.json")
    result = read(RUN_ROOT / "result.json")
    initial = read(RUN_ROOT / "initial.json", True)
    rounds = [read(RUN_ROOT / f"rounds/{i:02}/after.json", True) for i in range(4)]
    generated = [read(RUN_ROOT / f"rounds/{i:02}/generated.json", True) for i in range(4)]
    models = [
        DockingValue.from_payload(read(RUN_ROOT / f"rounds/{i:02}/prior/value.json", True))
        for i in range(4)
    ]
    final_model = DockingValue.from_payload(read(RUN_ROOT / "rounds/03/posterior/value.json", True))
    seed = initial["archive"][0]["smiles"]
    scores = [planning_scorer(model, seed) for model in models]
    final_score = planning_scorer(final_model, seed)
    final = rounds[-1]["state"]
    old_keys = {c["smiles"] for c in initial["pool"]}
    labels = {r["smiles"]: r["ds"] for r in final["archive"]}

    def properties(smiles):
        p = final_score(smiles)
        return {
            **{k: p[k] for k in ("qed", "sa", "sim", "v", "oracle_eligible")},
            "failures": [
                k
                for k, failed in (
                    ("qed", p["qed"] < 0.6),
                    ("sa", p["sa"] > 4),
                    ("similarity", p["sim"] < 0.4),
                    ("medchem", bool(validity_reasons(smiles))),
                )
                if failed
            ],
        }

    proposals, parents, decisions, predictions = [], [], [], []
    groups = defaultdict(Counter)
    by_id = {}
    for i, g in enumerate(generated):
        state = initial if i == 0 else rounds[i - 1]["state"]
        roles = (
            result["initial_parent_selection"]
            if i == 0
            else rounds[i - 1]["summary"]["parent_selection"]
        )
        assert models[i].payload["snapshot_sha256"] == rounds[i]["summary"]["prior_value_sha256"]
        for w in g["workers"]:
            index = w["worker_index"]
            parent = state["parents"][index // 2]
            assert identity(parent) == w["parent_sha256"]
            parents.append(
                {
                    "round": i + 1,
                    "worker": index,
                    "role": roles[index // 2]["role"],
                    "smiles": parent["smiles"],
                    **properties(parent["smiles"]),
                }
            )
            for option, count in w["options"].items():
                group = (
                    ":".join(option.split(":")[:2]) if option.startswith("construct:") else option
                )
                groups[group]["attempts"] += count
            local = {c["attempt_id"]: c for c in w["candidates"]}
            decision = w["branch_decision"]
            if decision["status"] == "selected":
                field = decision["score_field"]
                expected = [
                    max(scores[i](local[name]["smiles"])[field] for name in branch)
                    for branch in decision["branches"]
                ]
                assert np.allclose(expected, decision["values"], rtol=0, atol=1e-12)
                decisions.append({"round": i + 1, "worker": index, **decision})
            for c in w["candidates"]:
                assert c["attempt_id"] not in by_id
                by_id[c["attempt_id"]] = c
                option = c["bundle"]["option"]
                group = (
                    ":".join(option.split(":")[:2]) if option.startswith("construct:") else option
                )
                p = properties(c["smiles"])
                groups[group].update(completed=1, eligible=int(p["oracle_eligible"]))
                groups[group].update(p["failures"])
                proposals.append(
                    {
                        "round": i + 1,
                        "worker": index,
                        "role": roles[index // 2]["role"],
                        "root_smiles": parent["smiles"],
                        **{
                            k: c[k]
                            for k in (
                                "attempt_id",
                                "smiles",
                                "parent_smiles",
                                "chain",
                                "structural_change",
                            )
                        },
                        "option": option,
                        "remaining_budget": c["node"]["budget"],
                        "is_new_identity": c["smiles"] not in old_keys,
                        **p,
                        "observed_ds_if_any": labels.get(c["smiles"]),
                    }
                )
        for row in rounds[i]["summary"]["docked"]:
            p = scores[i](row["smiles"])
            for field in ("qed", "sa", "sim", "v", "predicted_docking"):
                assert abs(p[field] - row[field]) < 1e-10, field
            predictions.append(
                {
                    "round": i + 1,
                    **{
                        k: row[k]
                        for k in (
                            "smiles",
                            "attempt_id",
                            "ds",
                            "predicted_docking",
                            "oracle_selection_role",
                        )
                    },
                    "prior_mean": models[i].payload["mean"],
                }
            )
    assert len(predictions) == result["new_oracle_attempts"] == 40
    assert len(proposals) == sum(r["summary"]["completed"] for r in rounds)
    assert len(final["pool"]) == len({c["smiles"] for c in final["pool"]})
    novel = [c for c in final["pool"] if c["smiles"] not in old_keys]
    route = read(ROOT / "diagnostics/t4_route_diagnosis/result.json")
    witness = read(ROOT / "diagnostics/t4_whole_ring_plan/result.json")
    keys = {c["smiles"] for c in final["pool"]}
    stages = []
    for boundary in route["boundaries"]:
        row = next(r for r in route["route"] if r["step"] == boundary["step"])
        smiles = Chem.MolToSmiles(Chem.MolFromSmiles(row["smiles"]))
        stages.append(
            {
                **boundary,
                "smiles": smiles,
                **properties(smiles),
                "in_final_pool": smiles in keys,
                "final_predicted_docking": final_score(smiles)["predicted_docking"],
            }
        )
    assert stages[-1]["smiles"] == Chem.MolToSmiles(Chem.MolFromSmiles(witness["target"]))
    motif_queries = {
        "peripheral_ketone_fused_system": "O=C1CCCc2ccccc21",
        "modified_core": "O=C1NCc2ccccc2C(=O)n2cccc21",
        "uncarbonylated_tetralin": "c1ccc2c(c1)CCCC2",
    }
    motifs = {}
    for name, query_smiles in motif_queries.items():
        query = Chem.MolFromSmiles(query_smiles)
        assert query is not None
        matches = [
            c for c in final["pool"] if Chem.MolFromSmiles(c["smiles"]).HasSubstructMatch(query)
        ]
        motifs[name] = {
            "query_smiles": query_smiles,
            "matches": [
                {"smiles": c["smiles"], "attempt_id": c["attempt_id"], **properties(c["smiles"])}
                for c in matches
            ],
        }
    best = result["best"]
    best_descendants = [c for c in proposals if best["attempt_id"] in c["chain"][:-1]]
    best_workers = [p for p in parents if p["smiles"] == best["smiles"]]
    graph_comparison = {}
    for label, smiles in (("best", best["smiles"]), ("diagnostic_target", witness["target"])):
        molecule = Chem.MolFromSmiles(smiles)
        graph_comparison[label] = {
            "smiles": smiles,
            "heavy_atoms": molecule.GetNumHeavyAtoms(),
            "cycle_rank": molecule.GetNumBonds() - molecule.GetNumAtoms() + 1,
            "elements": dict(Counter(atom.GetSymbol() for atom in molecule.GetAtoms())),
            **properties(smiles),
        }
    summaries = {
        "attempts": sum(g["attempts"] for g in generated),
        "completed": len(proposals),
        "unique_new_molecules": len(novel),
        "eligible_new_molecules": sum(properties(c["smiles"])["oracle_eligible"] for c in novel),
        "exclusions_overlap": dict(
            Counter(k for c in novel for k in properties(c["smiles"])["failures"])
        ),
        "pending_eligible": [
            c["smiles"]
            for c in final["pool"]
            if c["smiles"] not in labels and properties(c["smiles"])["oracle_eligible"]
        ],
        "by_option": dict(groups),
        "best": {
            k: best[k]
            for k in (
                "smiles",
                "ds",
                "qed",
                "sa",
                "sim",
                "attempt_id",
                "bundle",
                "structural_change",
            )
        },
        "best_parent_worker_count": len(best_workers),
        "best_descendants": best_descendants,
        "graph_comparison": graph_comparison,
        "branch_score_field_counts": dict(Counter(d["score_field"] for d in decisions)),
        "selected_continuation_stage": dict(
            Counter(
                "lookahead" if "/lookahead/" in d["selected_attempt"] else "first"
                for d in decisions
            )
        ),
        "exploratory_state_selections": sum(d["exploratory_state"] for d in decisions),
        "new_dockings_by_generation_stage": dict(
            Counter(
                "guided_followup"
                if "/guided_followup/" in r["attempt_id"]
                else "lookahead"
                if "/lookahead/" in r["attempt_id"]
                else "first"
                for r in predictions
            )
        ),
        "guided_followup_best_observed": min(
            (r["ds"] for r in predictions if "/guided_followup/" in r["attempt_id"]), default=None
        ),
        "prior_only_prediction_mae": float(
            np.mean([abs(r["ds"] - r["predicted_docking"]) for r in predictions])
        ),
        "prior_mean_prediction_mae": float(
            np.mean([abs(r["ds"] - r["prior_mean"]) for r in predictions])
        ),
        "remaining_budget_min": min(c["remaining_budget"] for c in proposals),
        "route_stages": stages,
        "target_motifs": motifs,
        "final_prediction_of_best": final_score(best["smiles"])["predicted_docking"],
        "final_planning_value_of_best": final_score(best["smiles"])["planning_docking"],
        "seconds": perf_counter() - started,
    }
    source_hashes = {}
    for module in tuple(sys.modules.values()):
        file = getattr(module, "__file__", None)
        if file is None:
            continue
        path = Path(file).resolve()
        if path.is_relative_to(SOURCE / "src/compose_v4") and path.suffix == ".py":
            rel = str(path.relative_to(SOURCE))
            source_hashes[rel] = sha(path)
            assert source_hashes[rel] == launch["image_revision"]["serialized_sources"][rel], rel
    report = {
        "schema_version": "t4_macro_lookahead_retrospective_v1",
        "summary": summaries,
        "proposals": proposals,
        "parents": parents,
        "branch_decisions": decisions,
        "docking_predictions": predictions,
        "input_sha256": inputs,
        "source_closure": source_hashes,
        "run_revision": result["code_revision"],
        "analysis_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "analysis_sha256": sha(Path(__file__)),
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "rdkit": rdBase.rdkitVersion,
        },
        "configuration": {"delta": 0.4, "qed_min": 0.6, "sa_max": 4, "randomness": "none"},
        "hardware": {"machine": platform.machine(), "precision": "float64", "threads": 1},
        "split_identity": "inspected PARP1 seed0 development; target queries post-hoc only",
        "new_oracle_calls": 0,
        "new_law_calls": 0,
        "new_training": False,
        "checks": [
            "collection physical hashes",
            "sealed hashes",
            "production source closure",
            "parent identities",
            "branch values",
            "prior docking predictions and endpoint properties",
            "canonical pool uniqueness",
        ],
        "limitations": [
            "no matched causal comparison",
            "absence from sampled pool is not unreachability",
            "winner score not verified here",
            "no docking labels for undocked route stages",
            "unequal adaptive exposure, not calibrated guide performance",
        ],
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    target = RUN_ROOT / "diagnosis.json"
    temp = target.with_suffix(".json.tmp")
    temp.write_text(json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n")
    temp.replace(target)
    print(json.dumps(summaries, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
