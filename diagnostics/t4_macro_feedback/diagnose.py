"""Retrospective ledger analysis only: no generation, fitting, replay, or docking.

Run with the completed episode's clean source tree and pinned chemistry on PYTHONPATH.
The published winner is read only for post-run structural coverage diagnosis.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from functools import cache
from pathlib import Path
from time import perf_counter

import numpy as np
from rdkit import Chem, RDLogger, rdBase
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Contrib.SA_Score import sascorer

from compose_v4.control.docking_value import DockingValue, identity
from compose_v4.experiments.t4_endpoint_selection import acceptable_endpoint, calculate_properties
from compose_v4.experiments.t4_matched_pilot import unseal
from compose_v4.gates.med_chem_gate import validity_reasons

RUN = "3e8730df1962c80fcc8cfad050010c76a2e533f6191cd19b4bb25206ffc57c11"
ROOT = Path(__file__).resolve().parents[2]
RUN_ROOT = Path(__file__).resolve().parent / RUN
SOURCE_ROOT = Path(sys.modules[DockingValue.__module__].__file__).resolve().parents[3]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    started = perf_counter()
    RDLogger.DisableLog("rdApp.warning")
    if rdBase.rdkitVersion != "2024.03.5":
        raise ValueError("diagnosis requires run-matched RDKit 2024.03.5")
    inputs = {}

    def read(path, sealed=False):
        inputs[str(path.relative_to(ROOT))] = sha(path)
        return unseal(path) if sealed else json.loads(path.read_text())

    launch = read(RUN_ROOT / "launch.json")
    result = read(RUN_ROOT / "result.json")
    initial = read(RUN_ROOT / "initial.json", True)
    rounds = [read(RUN_ROOT / f"rounds/{i:02}/after.json", True) for i in range(10)]
    generated = [read(RUN_ROOT / f"rounds/{i:02}/generated.json", True) for i in range(10)]
    prior = [
        DockingValue.from_payload(read(RUN_ROOT / f"rounds/{i:02}/prior/value.json", True))
        for i in range(10)
    ]
    posterior = [
        DockingValue.from_payload(read(RUN_ROOT / f"rounds/{i:02}/posterior/value.json", True))
        for i in range(10)
    ]
    final = rounds[-1]["state"]
    seed = initial["pool"][0]["smiles"]
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    seed_fp = generator.GetFingerprint(Chem.MolFromSmiles(seed))

    @cache
    def props(smiles):
        value = calculate_properties(
            Chem.MolFromSmiles(smiles),
            seed_fp=seed_fp,
            generator=generator,
            sa_scorer=sascorer.calculateScore,
            delta=0.4,
            qed_min=0.6,
            sa_max=4.0,
        )
        reasons = [
            name
            for name, fails in (
                ("qed", value["qed"] < 0.6),
                ("sa", value["sa"] > 4),
                ("similarity", value["sim"] < 0.4),
                ("medchem", bool(validity_reasons(smiles))),
            )
            if fails
        ]
        return {
            **value,
            "failures": reasons,
            "oracle_eligible": acceptable_endpoint({"smiles": smiles, **value}),
        }

    attempts = [c for g in generated for c in g["candidates"]]
    by_id = {c["attempt_id"]: c for c in attempts}
    labels = {r["smiles"]: r["ds"] for r in final["archive"]}
    old = {r["smiles"] for r in initial["archive"]}
    docked = [r for a in rounds for r in a["summary"]["docked"]]
    for r in docked:
        assert all(abs(props(r["smiles"])[k] - r[k]) < 1e-10 for k in ("qed", "sa", "sim", "v"))
    assert len(docked) == result["new_oracle_attempts"] == 27
    unique = [c for c in final["pool"] if c["chain"]]
    assert len(unique) == len({c["smiles"] for c in unique})
    eligible = [c for c in unique if props(c["smiles"])["oracle_eligible"]]
    groups, roles = defaultdict(Counter), defaultdict(Counter)
    parent_rows, proposal_rows, guide_rows = [], [], []
    for i, g in enumerate(generated):
        state = initial if i == 0 else rounds[i - 1]["state"]
        selections = None if i == 0 else rounds[i - 1]["summary"]["parent_selection"]
        assert prior[i].payload["snapshot_sha256"] == rounds[i]["summary"]["prior_value_sha256"]
        for worker in g["workers"]:
            index = worker["worker_index"]
            parent = state["parents"][index]
            assert identity(parent) == worker["parent_sha256"]
            role = (
                ("initial_seed" if index < 4 else "initial_observed")
                if i == 0
                else selections[index]["role"]
            )
            prop = props(parent["smiles"])
            children = worker["candidates"]
            eventual_eligible = [
                c
                for c in attempts
                if parent["chain"]
                and parent["attempt_id"] in c["chain"][:-1]
                and props(c["smiles"])["oracle_eligible"]
            ]
            counts = roles[role]
            counts.update(
                slots=1,
                eligible_parent=int(prop["oracle_eligible"]),
                completed=len(children),
                eligible_children=sum(props(c["smiles"])["oracle_eligible"] for c in children),
            )
            if not prop["oracle_eligible"]:
                counts.update(
                    ineligible_parent_slots=1,
                    children_of_ineligible_parent=len(children),
                    eligible_children_of_ineligible_parent=sum(
                        props(c["smiles"])["oracle_eligible"] for c in children
                    ),
                    ineligible_parent_with_later_eligible_descendant=int(bool(eventual_eligible)),
                )
            parent_rows.append(
                {
                    "round": i + 1,
                    "worker": index,
                    "role": role,
                    "attempt_id": parent["attempt_id"],
                    "smiles": parent["smiles"],
                    **prop,
                    "children": [c["attempt_id"] for c in children],
                    "eventual_eligible_descendants": [c["attempt_id"] for c in eventual_eligible],
                }
            )
            if role == "guided":
                d = selections[index]["decision"]
                q, values = (
                    np.asarray(d["first_slot_probabilities"]),
                    np.asarray(d["first_slot_values"]),
                )
                guide_rows.append(
                    {
                        "round": i + 1,
                        "pool_size": len(q),
                        "eta": d["eta"],
                        "kl": d["first_slot_kl_against_empirical_pool"],
                        "max_probability": float(q.max()),
                        "effective_size": float(1 / (q @ q)),
                        "value_min": float(values.min()),
                        "value_max": float(values.max()),
                        "selected_eligible": prop["oracle_eligible"],
                    }
                )
            for option, n in worker["options"].items():
                group = (
                    ":".join(option.split(":")[:2]) if option.startswith("construct:") else option
                )
                groups[group]["attempts"] += n
            for c in children:
                option = c["bundle"]["option"]
                group = (
                    ":".join(option.split(":")[:2]) if option.startswith("construct:") else option
                )
                cp = props(c["smiles"])
                groups[group].update(
                    completed=1,
                    eligible=int(cp["oracle_eligible"]),
                    from_eligible_parent=int(prop["oracle_eligible"]),
                )
                groups[group].update(cp["failures"])
                proposal_rows.append(
                    {
                        "round": i + 1,
                        "role": role,
                        "attempt_id": c["attempt_id"],
                        "smiles": c["smiles"],
                        "option": option,
                        **cp,
                        "parent_eligible": prop["oracle_eligible"],
                        "parent_smiles": parent["smiles"],
                        "remaining_budget": c["node"]["budget"],
                        "chain": c["chain"],
                        "r_release": c["bundle"]["r_release"],
                        "structural_change": c["structural_change"],
                        "observed_ds_if_any": labels.get(c["smiles"]),
                    }
                )

    scoring = []
    pairs, concordant = 0, 0.0
    for i, a in enumerate(rounds):
        rows = a["summary"]["docked"]
        predictions = prior[i].predict([r["smiles"] for r in rows])
        for r, prediction in zip(rows, predictions, strict=True):
            assert abs(r["predicted_docking"] - prediction) < 1e-10
            assert r["ds"] is not None
            scoring.append(
                {
                    "round": i + 1,
                    "smiles": r["smiles"],
                    "actual": r["ds"],
                    "predicted": float(prediction),
                    "baseline": prior[i].payload["mean"],
                    "selection_role": r["oracle_selection_role"],
                }
            )
        for j, arow in enumerate(rows):
            for brow in rows[j + 1 :]:
                if arow["ds"] == brow["ds"]:
                    continue
                pairs += 1
                product = (arow["ds"] - brow["ds"]) * (
                    arow["predicted_docking"] - brow["predicted_docking"]
                )
                concordant += 1 if product > 0 else 0.5 if product == 0 else 0

    route = read(ROOT / "diagnostics/t4_route_diagnosis/result.json")
    winner = read(ROOT / "diagnostics/t4_whole_ring_plan/result.json")["target"]
    # Post-hoc graph queries from the existing target witness, never controller inputs.
    queries = {
        "peripheral_ketone_fused_system": "O=C1CCCc2ccccc21",
        "modified_core": "O=C1NCc2ccccc2C(=O)n2cccc21",
        "uncarbonylated_tetralin": "c1ccc2c(c1)CCCC2",
    }
    motifs = {}
    for name, smiles in queries.items():
        query = Chem.MolFromSmiles(smiles)
        assert (
            Chem.MolFromSmiles(winner).HasSubstructMatch(query) or name == "uncarbonylated_tetralin"
        )
        motifs[name] = {
            "query_smiles": smiles,
            "matches": [
                {"smiles": c["smiles"], "attempt_id": c["attempt_id"], **props(c["smiles"])}
                for c in unique
                if Chem.MolFromSmiles(c["smiles"]).HasSubstructMatch(query)
            ],
        }
    keys = {c["smiles"] for c in unique}
    stages = []
    for b in route["boundaries"]:
        row = next(r for r in route["route"] if r["step"] == b["step"])
        smiles = Chem.MolToSmiles(Chem.MolFromSmiles(row["smiles"]))
        stages.append(
            {
                **b,
                "smiles": smiles,
                "exact_match_in_generated_pool": smiles in keys,
                **props(smiles),
                "final_guide_prediction": float(posterior[-1].predict([smiles])[0]),
            }
        )
    best_chain = [
        {
            "attempt_id": name,
            "option": by_id[name]["bundle"]["option"],
            "smiles": by_id[name]["smiles"],
            "ds": labels.get(by_id[name]["smiles"]),
            "expanded_parent_slots": sum(p["smiles"] == by_id[name]["smiles"] for p in parent_rows),
        }
        for name in result["best"]["chain"]
    ]
    source_hashes = {}
    for module in tuple(sys.modules.values()):
        file = getattr(module, "__file__", None)
        if file is None:
            continue
        path = Path(file).resolve()
        if path.is_relative_to(SOURCE_ROOT / "src/compose_v4") and path.suffix == ".py":
            rel = str(path.relative_to(SOURCE_ROOT))
            source_hashes[rel] = sha(path)
            assert source_hashes[rel] == launch["image_revision"]["serialized_sources"][rel], rel
    summary = {
        "attempts": sum(g["attempts"] for g in generated),
        "complete": len(attempts),
        "unique_generated": len(unique),
        "eligible_unique_generated": len(eligible),
        "eligible_already_in_source_archive": [c["smiles"] for c in eligible if c["smiles"] in old],
        "final_undocked_eligible": [c["smiles"] for c in eligible if c["smiles"] not in labels],
        "exclusion_counts_overlap": dict(
            Counter(r for c in unique for r in props(c["smiles"])["failures"])
        ),
        "exclusion_combinations": dict(
            Counter(",".join(props(c["smiles"])["failures"]) or "eligible" for c in unique)
        ),
        "by_option": dict(groups),
        "parent_roles": dict(roles),
        "remaining_budget_min": min(c["node"]["budget"] for c in attempts),
        "max_option_depth": max(len(c["chain"]) for c in attempts),
        "proposal_wall_seconds": sum(g["proposal_seconds"] for g in generated),
        "worker_proposal_seconds": sum(
            w["proposal_seconds"] for g in generated for w in g["workers"]
        ),
        "initial_oracle_attempts": result["initial_oracle_attempts"],
        "new_oracle_attempts": len(docked),
        "best_ds": result["best"]["ds"],
        "best_chain": best_chain,
        "prior_only_guide": {
            "n": len(scoring),
            "within_round_pairs": pairs,
            "within_round_concordance": concordant / pairs,
            "mae": float(np.mean([abs(r["actual"] - r["predicted"]) for r in scoring])),
            "prior_mean_mae": float(np.mean([abs(r["actual"] - r["baseline"]) for r in scoring])),
        },
        "target_motifs_posthoc": motifs,
        "target_stages_posthoc": stages,
        "final_guide_prediction_of_observed_best": float(
            posterior[-1].predict([result["best"]["smiles"]])[0]
        ),
        "notable_continuations": [
            {
                "candidate": c,
                "later_parent_slots": [
                    {k: p[k] for k in ("round", "role")}
                    for p in parent_rows
                    if p["smiles"] == c["smiles"]
                ],
                "descendants": [
                    p["attempt_id"] for p in proposal_rows if c["attempt_id"] in p["chain"][:-1]
                ],
            }
            for c in proposal_rows
            if c["option"] == "insert_ring_carbonyl"
            or (c["option"].startswith("construct:fused") and c["oracle_eligible"])
        ],
    }
    report = {
        "schema_version": "t4_macro_feedback_retrospective_v1",
        "summary": summary,
        "proposals": proposal_rows,
        "parent_slots": parent_rows,
        "guide_decisions": guide_rows,
        "docking_predictions": scoring,
        "input_sha256": inputs,
        "source_closure": source_hashes,
        "run_revision": launch["image_revision"]["commit"],
        "analysis_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "analysis_sha256": sha(Path(__file__)),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "rdkit": rdBase.rdkitVersion,
        },
        "hardware": {"machine": platform.machine(), "precision": "float64", "threads": 1},
        "configuration": {
            "delta": 0.4,
            "qed_min": 0.6,
            "sa_max": 4,
            "randomness": "none; deterministic retrospective reduction",
        },
        "split_identity": "inspected PARP1 seed0 development cell; prior-only predictions, target post-hoc only",
        "new_oracle_calls": 0,
        "new_law_calls": 0,
        "new_training": False,
        "checks": [
            "sealed payload hashes",
            "source closure equals remote image",
            "parent identities",
            "docked properties and frozen predictions reproduced",
            "unique canonical pool",
        ],
        "limitations": [
            "one adaptive development episode, not a causal ablation or independent benchmark",
            "substructure absence is sampled coverage, not proof of unreachability",
            "endpoint failure is not proof that an intermediate cannot recover",
            "docking concordance is descriptive on adaptively selected candidates",
        ],
        "seconds": perf_counter() - started,
    }
    target = RUN_ROOT / "diagnosis.json"
    temp = target.with_suffix(".json.tmp")
    temp.write_text(json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n")
    temp.replace(target)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
